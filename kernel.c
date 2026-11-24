/*
 * =============================================================================
 *  KRONOS-X v5.0 :: Single-File x86_64 Bare-Metal-Grade CPU Microkernel &
 *                   Dynamic Silicon Execution Engine (C11 + x86_64 Assembly)
 * =============================================================================
 *
 *  Build (Linux x86_64, GCC or Clang — zero -mavx2/-mfma/-mavx512f/-lm flags):
 *      gcc -O3 -Wall -Wextra -Wshadow -pthread kronos_kernel.c -o kronos
 *      ./kronos [--cores N] [--gemm N] [--quantum-us Q]
 *               [--gemm-kernel auto|avx512|avx512-9x48|avx2|scalar]
 *               [--no-color] [--json | --csv]
 *
 *  Subsystems (v5.0 — All 37 Deep-Audit Fixes, Optimizations & Overhauls):
 *    [1] Silicon, Full-Active-Core Hybrid ISA-Intersect (CPUID 0x1A + Sysfs),
 *        NUMA + L3-CCD Topology Ordering, Exact Active-Set L2 Sharing,
 *        Bracketed TSC Calibration, Boot Hardware FMA Peak Micro-Probe,
 *        Hybrid-Aware Grouped Per-TID PMU w/ Zero-Run Multiplexing Guard
 *    [2] Raw Linux Syscalls (`futex`/`FUTEX_WAIT_BITSET`, `membarrier`,
 *        `tgkill`, `arch_prctl`), Fair Ticket Locks & Contention-Aware TTAS,
 *        CSE-Proof Per-Fiber `KRONOS_ERRNO`, `asm goto` Single-Insn `%fs:`
 *        Preemption Word
 *    [3] Bonwick Two-Magazine O(1) Slab Allocator (Branchless Sentinel-Mag
 *        Fast Path, Exact Modular Cache-Line Coloring, Zero-Strand Multi-Chunk
 *        Carver, Fair Depot Locks, Optional `-DKRONOS_SLAB_DEBUG` Poisoning)
 *    [4] Hand-Written x86_64 Assembly (.hidden + Exact DWARF CFI + CET IBT):
 *        4-Way Switcher (`to_sched`, `from_sched`, `fninit`-Hardened `exit`,
 *        1-Hop `switch_direct`), Stack-Safe `XSAVE64`/`XRSTOR64` Trampoline,
 *        Lossless CF+DF+x87+MXCSR+ZMM/k+128B Red-Zone Canaries, Hardware FMA
 *        Peak Probes & AVX-512 VNNI INT8 Dot-Product Kernel
 *    [5] Lock-Free M:N Work-Stealing Kernel (Non-Stealable Core-Pinned NUMA
 *        Queues, NUMA/L3-Biased Steal-Half w/ Chain Wake, Per-Worker Unlocked-
 *        Resize Timer Min-Heaps, Bug-Free Absolute-Time 0 Hz Tickless LAPIC,
 *        Writer-Locked Seqlock Code Ranges + Static-Link Guard, Bounded Batch
 *        Stack Pool w/ Lazy Reaping)
 *    [6] Linearizable `KronosChan` (Adaptive Spin/Yield Before `membarrier`,
 *        Tickless External Futex, Bit-63 Atomic Close), 4-State UAF-Proof
 *        `KronosLatch` w/ Yield Escape, Branchless $O(\log P)$ Dissemination
 *        Barrier w/ Wrap-Safe Epochs & Oversubscription Adaptation
 *    [7] Sandboxed K-ISA Verifier & Optimizing W^X JIT (Fixed-Point Multi-Pass
 *        DCE, Hardware Fuel-Metered `KBPF_CAP_BOUNDED_LOOP`, Byte-Bounded
 *        `R3 >> shift` + `lfence`, `AND+CMP0` Flag Fusion, BMI1 `TZCNT`,
 *        `LZCNT`, BMI2 `RORX`/`SHLX`/`SHRX`/`SARX`, `PROT_NONE` Page Recycler)
 *    [8] Silicon-Calibrated Double-Buffered Persistent-Fiber BLIS GEMM Engine
 *        (Pinned NUMA/L3 Workers, AVX-512 `9x48` 2.25-FMA/Load & `14x32` w/
 *        Staggered `prefetchw` + Zero-Copy Opmask Edge Stores, Aligned-Redzone
 *        AVX `pack_A`, Analytical L1d Sizing, Balanced `nc_step`+`kc_step`,
 *        Full `TransA`/`TransB` NN/NT/TN/TT Support)
 * =============================================================================
 */

#define _GNU_SOURCE

#if !defined(__x86_64__) || !defined(__linux__)
#error "KRONOS-X requires Linux x86_64 (System V AMD64 ABI + Linux futex/ucontext)."
#endif

#if defined(__PIC__) && !defined(__PIE__)
#error "KRONOS-X uses local-exec TLS (@tpoff); compile as an executable (default or -fPIE), not -fPIC shared lib."
#endif

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <link.h>
#include <linux/futex.h>
#include <linux/perf_event.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#ifdef __NR_membarrier
#include <linux/membarrier.h>
#endif

/* ============================================================================
 *  0. CONFIGURATION, CACHE-LINE ALIGNMENT, HELPERS & OUTPUT MODES
 * ============================================================================ */

#define KRONOS_MAX_CORES         128
#define KRONOS_DEQUE_CAP         4096
#define KRONOS_DEQUE_MASK        (KRONOS_DEQUE_CAP - 1)
#define KRONOS_INBOX_CAP         256
#define KRONOS_INBOX_MASK        (KRONOS_INBOX_CAP - 1)
#define KRONOS_GLOBAL_RQ_CAP     4096
#define KRONOS_GLOBAL_RQ_MASK    (KRONOS_GLOBAL_RQ_CAP - 1)
#define KRONOS_FIBER_STACK_SZ    (128 * 1024)
#define KRONOS_PAGE_SZ           4096
#define KRONOS_HUGEPAGE_SZ       (2ULL * 1024ULL * 1024ULL)
#define KRONOS_STACK_BATCH       8
#define KRONOS_FIBER_POOL_MAX    256
#define KRONOS_LOCAL_POOL_MAX    16
#define KRONOS_MAX_CODE_RANGES   64
#define KRONOS_INIT_TIMER_CAP    64
#define KRONOS_JIT_CACHE_MAX     8

#define KRONOS_SLAB_CLASSES      16
#define KRONOS_SLAB_MAG_CAP_SM   64
#define KRONOS_SLAB_MAG_CAP_LG   16
#define KRONOS_SLAB_CHUNK_SZ     (16ULL * 1024ULL * 1024ULL)

#define KRONOS_NEED_RESCHED_BIT  0x80000000u
#define KRONOS_PREEMPT_MASK      0x7FFFFFFFu

#define KRONOS_WORKER_IN_PREEMPTIBLE_OFF 128

#define KSTR1(x)                 #x
#define KSTR(x)                  KSTR1(x)

#define CACHELINE_ALIGN          _Alignas(64)
#define LIKELY(x)                __builtin_expect(!!(x), 1)
#define UNLIKELY(x)              __builtin_expect(!!(x), 0)

#if defined(__GNUC__) && !defined(__clang__)
#define KRONOS_ASM_VISIBLE __attribute__((used, visibility("hidden"), externally_visible))
#else
#define KRONOS_ASM_VISIBLE __attribute__((used, visibility("hidden")))
#endif

_Static_assert((KRONOS_DEQUE_CAP & KRONOS_DEQUE_MASK) == 0,
               "KRONOS_DEQUE_CAP must be a power of two");
_Static_assert((KRONOS_INBOX_CAP & KRONOS_INBOX_MASK) == 0,
               "KRONOS_INBOX_CAP must be a power of two");
_Static_assert((KRONOS_GLOBAL_RQ_CAP & KRONOS_GLOBAL_RQ_MASK) == 0,
               "KRONOS_GLOBAL_RQ_CAP must be a power of two");
_Static_assert(sizeof(_Atomic uint32_t) == sizeof(uint32_t) &&
               _Alignof(_Atomic uint32_t) == _Alignof(uint32_t),
               "_Atomic uint32_t must match Linux futex 32-bit word layout");

typedef enum {
    OUT_PRETTY = 0,
    OUT_JSON,
    OUT_CSV
} KronosOutMode;

static int           g_use_color    = 0;
static KronosOutMode g_out_mode     = OUT_PRETTY;
static _Atomic int   g_failures     = 0;
static double        g_pause_cycles = 40.0;
static int           g_has_membar   = 0;

#define C_RST   (g_use_color ? "\033[0m"        : "")
#define C_BOLD  (g_use_color ? "\033[1m"        : "")
#define C_DIM   (g_use_color ? "\033[2m"        : "")
#define C_CYAN  (g_use_color ? "\033[38;5;51m"  : "")
#define C_BLUE  (g_use_color ? "\033[38;5;39m"  : "")
#define C_GRN   (g_use_color ? "\033[38;5;82m"  : "")
#define C_YEL   (g_use_color ? "\033[38;5;220m" : "")
#define C_MAG   (g_use_color ? "\033[38;5;207m" : "")
#define C_RED   (g_use_color ? "\033[38;5;196m" : "")

#define KPRINTF(...) do { if (g_out_mode == OUT_PRETTY) printf(__VA_ARGS__); } while (0)

typedef struct {
    double   slab_cy_pair;
    double   slab_mops;
    double   slab_hit_pct;
    uint32_t slab_unique_colors;
    double   ctx_runnext_cy;
    double   ctx_mux_ns;
    double   mpmc_mmsgs;
    uint32_t canary_preempts;
    double   probe_avg_us;
    double   probe_max_us;
    double   jit_speedup;
    double   jit_par_mops;
    double   jit_bw_gbs;
    uint32_t jit_preempts;
    const char *gemm_kernel;
    int      gemm_dim;
    double   gemm_peak_gflops;
    double   gemm_med_gflops;
    double   gemm_fpc;
    double   gemm_eff_pct;
    double   gemm_ipc;
    double   gemm_turbo_ghz;
    uint64_t tot_fibers;
    uint64_t tot_switches;
    uint64_t tot_direct;
    uint64_t tot_steals;
    uint64_t tot_preempts;
    uint64_t tot_ipis;
    uint64_t tot_futex;
    uint64_t timer_wakeups;
    double   timer_jitter_avg_us;
    double   timer_jitter_max_us;
} KronosTelemetry;

static KronosTelemetry g_tel;

/* Globals read directly from file-scope assembly (hidden + LTO/ThinLTO-safe) */
int      g_cpu_has_avx_asm       KRONOS_ASM_VISIBLE = 0;
int      g_cpu_use_xsave_asm     KRONOS_ASM_VISIBLE = 0;
int      g_cpu_has_prefetchw_asm KRONOS_ASM_VISIBLE = 0;
uint32_t g_cpu_xsave_mask_lo_asm KRONOS_ASM_VISIBLE = 0;
uint64_t g_cpu_xsave_size_asm    KRONOS_ASM_VISIBLE = 512;
uint32_t g_default_mxcsr_asm     KRONOS_ASM_VISIBLE = 0x1F80u;
uint16_t g_default_fcw_asm       KRONOS_ASM_VISIBLE = 0x037Fu;

static inline float make_qnan_f32(void) {
    union { uint32_t u; float f; } v = { 0x7FC00000u };
    return v.f;
}

static inline int is_finite_f32(float x) {
    union { float f; uint32_t u; } v = { x };
    return ((v.u & 0x7F800000u) != 0x7F800000u);
}

static inline int is_finite_f64(double x) {
    union { double d; uint64_t u; } v = { x };
    return ((v.u & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL);
}

static void *xaligned_alloc(size_t align, size_t size) {
    if (align < sizeof(void *) || (align & (align - 1u)) != 0) align = 64;
    if (size == 0) size = align;
    if (UNLIKELY(size > SIZE_MAX - align)) {
        fprintf(stderr, "[KRONOS] Fatal: aligned_alloc size overflow (%zu)\n", size);
        exit(1);
    }
    size_t rounded = (size + align - 1u) & ~(align - 1u);
    void *p = aligned_alloc(align, rounded);
    if (UNLIKELY(!p)) {
        fprintf(stderr, "[KRONOS] Fatal: aligned_alloc(%zu, %zu) failed\n", align, rounded);
        exit(1);
    }
    return p;
}

/* Overflow-hardened 2 MiB-aligned THP mmap (requires full 2*HUGEPAGE + PAGE headroom) */
static void *xmmap_huge_aligned(size_t bytes, size_t *out_mapped_len) {
    if (UNLIKELY(bytes > SIZE_MAX - 2ULL * KRONOS_HUGEPAGE_SZ - KRONOS_PAGE_SZ)) {
        fprintf(stderr, "[KRONOS] Fatal: xmmap_huge_aligned size overflow (%zu)\n", bytes);
        exit(1);
    }
    size_t len = (bytes + KRONOS_PAGE_SZ - 1u) & ~(size_t)(KRONOS_PAGE_SZ - 1u);
    if (len == 0) len = KRONOS_PAGE_SZ;
    if (len < (512ULL * 1024ULL)) {
        void *p = mmap(NULL, len, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { perror("mmap"); exit(1); }
        *out_mapped_len = len;
        return p;
    }
    size_t thp_len = (len + KRONOS_HUGEPAGE_SZ - 1u) & ~(size_t)(KRONOS_HUGEPAGE_SZ - 1u);
    size_t reserve = thp_len + KRONOS_HUGEPAGE_SZ;
    void *raw = mmap(NULL, reserve, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) { perror("mmap 2M"); exit(1); }
    uintptr_t raw_addr     = (uintptr_t)raw;
    uintptr_t aligned_addr = (raw_addr + KRONOS_HUGEPAGE_SZ - 1u) & ~(uintptr_t)(KRONOS_HUGEPAGE_SZ - 1u);
    size_t prefix = (size_t)(aligned_addr - raw_addr);
    size_t suffix = reserve - prefix - thp_len;
    if (prefix > 0) (void)munmap(raw, prefix);
    if (suffix > 0) (void)munmap((void *)(aligned_addr + thp_len), suffix);
#ifdef MADV_HUGEPAGE
    (void)madvise((void *)aligned_addr, thp_len, MADV_HUGEPAGE);
#endif
    *out_mapped_len = thp_len;
    return (void *)aligned_addr;
}

static inline uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Overflow-safe absolute CLOCK_MONOTONIC sleep immune to repeated EINTR */
static void sleep_ns_eintr(uint64_t ns) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    uint64_t max_target = ((uint64_t)INT64_MAX / 2ULL);
    uint64_t target = (ns > max_target - now) ? max_target : (now + ns);
    ts.tv_sec  = (time_t)(target / 1000000000ULL);
    ts.tv_nsec = (long)(target % 1000000000ULL);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
}

static inline uint64_t safe_delta_ns(uint64_t t1, uint64_t t0) {
    return (t1 > t0) ? (t1 - t0) : 1ULL;
}

static inline void stat_inc(_Atomic uint64_t *p) {
    uint64_t v = atomic_load_explicit(p, memory_order_relaxed);
    atomic_store_explicit(p, v + 1ULL, memory_order_relaxed);
}

static inline uint64_t stat_load(const _Atomic uint64_t *p) {
    return atomic_load_explicit(p, memory_order_relaxed);
}

static inline uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    if (UNLIKELY(x == 0)) x = 0x9E3779B97F4A7C15ULL;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

/* ============================================================================
 *  1. SILICON, FULL HYBRID ISA-INTERSECT, NUMA/L3-CCD TOPOLOGY & HYBRID PMU
 * ============================================================================ */

typedef struct {
    char     vendor[16];
    char     brand[64];
    int      is_intel;
    uint32_t max_leaf;
    uint32_t max_ext_leaf;
    int      online_cpus;
    int      avail_cpus;
    int      phys_cores;
    int      p_cores;
    int      num_packages;
    int      num_numa_nodes;
    int      num_l3_domains;
    int      ordered_cpus[KRONOS_MAX_CORES];
    uint8_t  ordered_is_ecore[KRONOS_MAX_CORES];
    int      ordered_pkg[KRONOS_MAX_CORES];
    int      ordered_numa[KRONOS_MAX_CORES];
    int      ordered_l3[KRONOS_MAX_CORES];
    int      has_sse42;
    int      has_popcnt;
    int      has_aes;
    int      has_f16c;
    int      has_osxsave;
    int      has_xsaveopt;
    int      has_avx;
    int      has_avx2;
    int      has_fma;
    int      has_bmi1;
    int      has_bmi2;
    int      has_lzcnt;
    int      has_prefetchw;
    int      has_avx512f;
    int      has_avx512dq;
    int      has_avx512bw;
    int      has_avx512vl;
    int      has_avx512vnni;
    int      has_rdtscp;
    int      has_inv_tsc;
    int      tsc_clocksource_ok;
    int      has_hybrid;
    int      cet_shstk_active;
    int      cet_shstk_disabled;
    uint64_t xcr0;
    uint32_t xsave_mask_lo;
    uint64_t xsave_size;
    size_t   altstack_sz;
    uint32_t cache_line_bytes;
    uint64_t l1d_bytes;
    uint32_t l1d_ways;
    uint32_t l1d_shared_by;
    uint64_t l1i_bytes;
    uint64_t l2_bytes;
    uint64_t l2_eff_per_core;
    uint32_t l2_ways;
    uint32_t l2_shared_by;
    uint64_t l3_bytes;
    uint32_t l3_ways;
    uint32_t l3_shared_by;
    double   tsc_ghz;
    double   ns_per_cycle;
    const char *tsc_src;
    int      pmu_available;
    int      pmu_type_pcore;
    int      pmu_type_ecore;
    double   peak_fma_fpc_avx512;
    double   peak_fma_fpc_avx2;
    double   peak_fma_fpc_sse2;
} KronosCpuInfo;

static KronosCpuInfo g_cpu;

/* Forward declarations for boot FMA peak micro-probes (defined in Section 4 asm) */
extern void kronos_fma_peak_avx512(uint64_t iters);
extern void kronos_fma_peak_avx2(uint64_t iters);
extern void kronos_fma_peak_sse2(uint64_t iters);

static inline void cpuid_ex(uint32_t leaf, uint32_t subleaf,
                            uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d) {
    __asm__ volatile("cpuid"
        : "=a"(*a), "=b"(*b), "=c"(*c), "=d"(*d)
        : "a"(leaf), "c"(subleaf));
}

static inline uint64_t xgetbv0(void) {
    uint32_t lo, hi;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtsc_raw(void) {
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtsc_begin(void) {
    uint32_t lo, hi;
    __asm__ volatile("lfence\n\trdtsc\n\tlfence" : "=a"(lo), "=d"(hi) :: "memory");
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtsc_end(void) {
    uint32_t lo, hi;
    if (LIKELY(g_cpu.has_rdtscp)) {
        uint32_t aux;
        __asm__ volatile("rdtscp\n\tlfence" : "=a"(lo), "=d"(hi), "=c"(aux) :: "memory");
        (void)aux;
    } else {
        __asm__ volatile("lfence\n\trdtsc\n\tlfence" : "=a"(lo), "=d"(hi) :: "memory");
    }
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t kronos_quantum_clock_now(void) {
    if (LIKELY(g_cpu.tsc_clocksource_ok)) return rdtsc_raw();
    return monotonic_ns();
}

static long sys_arch_prctl_raw(int code, unsigned long addr) {
    long ret = (long)SYS_arch_prctl;
    __asm__ volatile(
        "syscall"
        : "+a"(ret)
        : "D"((long)code), "S"((long)addr)
        : "rcx", "r11", "cc", "memory"
    );
    return ret;
}

static int probe_linux_tsc_clocksource(void) {
    FILE *fp = fopen("/sys/devices/system/clocksource/clocksource0/current_clocksource", "r");
    if (!fp) return 1;
    char buf[64] = {0};
    int ok = 0;
    if (fgets(buf, sizeof(buf), fp)) {
        if (strncmp(buf, "tsc", 3) == 0 && (buf[3] == '\0' || buf[3] == '\n' || buf[3] == ' '))
            ok = 1;
    }
    fclose(fp);
    return ok;
}

static int read_sysfs_first_int(const char *path, int fallback) {
    FILE *fp = fopen(path, "r");
    if (!fp) return fallback;
    int val = fallback;
    if (fscanf(fp, "%d", &val) != 1) val = fallback;
    fclose(fp);
    return val;
}

/* Overflow-proof, zero-truncation sysfs cpulist parser ("0-15,24-27,...") */
static int parse_sysfs_cpulist(const char *path, uint8_t *bmap, int max_cpu) {
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    int found = 0;
    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        if (ch < '0' || ch > '9') continue;
        uint64_t a = (uint64_t)(ch - '0');
        while ((ch = fgetc(fp)) != EOF && ch >= '0' && ch <= '9') {
            if (a < 100000000ULL) a = a * 10ULL + (uint64_t)(ch - '0');
        }
        uint64_t b = a;
        if (ch == '-') {
            int ch2 = fgetc(fp);
            if (ch2 >= '0' && ch2 <= '9') {
                b = (uint64_t)(ch2 - '0');
                while ((ch = fgetc(fp)) != EOF && ch >= '0' && ch <= '9') {
                    if (b < 100000000ULL) b = b * 10ULL + (uint64_t)(ch - '0');
                }
            }
        }
        if (a >= (uint64_t)max_cpu) continue;
        if (b >= (uint64_t)max_cpu) b = (uint64_t)(max_cpu - 1);
        for (uint64_t c = a; c <= b; c++) { bmap[c] = 1u; found = 1; }
    }
    fclose(fp);
    return found;
}

typedef struct {
    int      has_sse42, has_popcnt, has_aes, has_f16c, has_osxsave, has_xsaveopt;
    int      has_avx, has_avx2, has_fma, has_bmi1, has_bmi2, has_lzcnt, has_prefetchw;
    int      has_avx512f, has_avx512dq, has_avx512bw, has_avx512vl, has_avx512vnni;
    int      has_rdtscp, has_inv_tsc, has_hybrid;
    uint8_t  hw_core_type; /* CPUID 0x1A: 0x20=Atom/E-core, 0x40=Core/P-core, 0=unknown */
    uint32_t max_leaf, max_ext_leaf;
    uint64_t xcr0;
    uint32_t xsave_mask_lo;
    uint64_t xsave_size;
    uint32_t line_sz;
    uint64_t l1d_bytes, l1i_bytes, l2_bytes, l3_bytes;
    uint32_t l1d_ways, l1d_shared, l2_ways, l2_shared, l3_ways, l3_shared;
} CpuCoreSample;

static void probe_cache_det_into(uint32_t leaf, CpuCoreSample *s) {
    for (uint32_t sub = 0; sub < 32; sub++) {
        uint32_t a, b, c, d;
        cpuid_ex(leaf, sub, &a, &b, &c, &d);
        uint32_t type = a & 0x1Fu;
        if (type == 0) break;
        uint32_t level      = (a >> 5) & 0x7u;
        uint32_t shared_by  = ((a >> 14) & 0xFFFu) + 1u;
        uint32_t line_sz    = (b & 0xFFFu) + 1u;
        uint32_t partitions = ((b >> 12) & 0x3FFu) + 1u;
        uint32_t ways       = ((b >> 22) & 0x3FFu) + 1u;
        uint64_t sets       = (uint64_t)c + 1ULL;
        uint64_t size       = (uint64_t)ways * (uint64_t)partitions * (uint64_t)line_sz * sets;
        if (line_sz >= 32 && line_sz <= 256 && line_sz > s->line_sz) s->line_sz = line_sz;
        if (level == 1 && type == 1 && size >= 8192 && size <= 1048576) {
            s->l1d_bytes = size; s->l1d_ways = ways; s->l1d_shared = shared_by;
        }
        if (level == 1 && type == 2 && size >= 8192 && size <= 1048576) {
            s->l1i_bytes = size;
        }
        if (level == 2 && (type == 1 || type == 3) && size >= 65536 && size <= 67108864ULL) {
            s->l2_bytes = size; s->l2_ways = ways; s->l2_shared = shared_by;
        }
        if (level == 3 && (type == 1 || type == 3) && size >= 524288 && size <= 8589934592ULL) {
            s->l3_bytes = size; s->l3_ways = ways; s->l3_shared = shared_by;
        }
    }
}

static void probe_xsave_sample(CpuCoreSample *s) {
    s->xsave_size    = 512;
    s->xsave_mask_lo = 0;
    if (!s->has_osxsave || s->max_leaf < 0x0Du) return;
    uint32_t mask = (uint32_t)(s->xcr0 & 0xE7ULL);
    if ((mask & 0x3u) != 0x3u) return;

    uint32_t a = 0, b = 0, c = 0, d = 0;
    cpuid_ex(0x0Du, 0, &a, &b, &c, &d);
    uint64_t max_end = 576;
    static const uint32_t comps[4] = { 2, 5, 6, 7 };
    for (int i = 0; i < 4; i++) {
        uint32_t comp = comps[i];
        if (mask & (1u << comp)) {
            uint32_t csz = 0, coff = 0, cc = 0, cd = 0;
            cpuid_ex(0x0Du, comp, &csz, &coff, &cc, &cd);
            if (csz == 0 || coff < 576) {
                if (b >= 576 && b <= 4096) {
                    if ((uint64_t)b > max_end) max_end = (uint64_t)b;
                } else {
                    return;
                }
            } else {
                uint64_t end = (uint64_t)coff + (uint64_t)csz;
                if (end > max_end) max_end = end;
            }
        }
    }
    max_end = (max_end + 63ULL) & ~63ULL;
    if (max_end >= 576 && max_end <= 4096) {
        s->xsave_mask_lo = mask;
        s->xsave_size    = max_end;
    }
}

static void probe_current_core_sample(CpuCoreSample *s) {
    memset(s, 0, sizeof(*s));
    uint32_t a, b, c, d;
    cpuid_ex(0, 0, &a, &b, &c, &d);
    s->max_leaf = a;

    if (s->max_leaf >= 1) {
        cpuid_ex(1, 0, &a, &b, &c, &d);
        s->has_sse42   = (int)((c >> 20) & 1u);
        s->has_popcnt  = (int)((c >> 23) & 1u);
        s->has_aes     = (int)((c >> 25) & 1u);
        s->has_osxsave = (int)((c >> 27) & 1u);
        s->has_f16c    = (int)((c >> 29) & 1u);
        int hw_avx     = (int)((c >> 28) & 1u);
        int hw_fma     = (int)((c >> 12) & 1u);
        int os_ymm_ok  = 0;
        int os_zmm_ok  = 0;
        if (s->has_osxsave) {
            s->xcr0   = xgetbv0();
            os_ymm_ok = ((s->xcr0 & 0x06ULL) == 0x06ULL);
            os_zmm_ok = ((s->xcr0 & 0xE6ULL) == 0xE6ULL);
        }
        s->has_avx = hw_avx && os_ymm_ok;
        s->has_fma = hw_fma && hw_avx && os_ymm_ok;

        if (s->max_leaf >= 7) {
            cpuid_ex(7, 0, &a, &b, &c, &d);
            s->has_bmi1       = (int)((b >> 3)  & 1u);
            s->has_avx2       = (int)((b >> 5)  & 1u) && hw_avx && os_ymm_ok;
            s->has_bmi2       = (int)((b >> 8)  & 1u);
            s->has_avx512f    = (int)((b >> 16) & 1u) && hw_avx && os_zmm_ok;
            s->has_avx512dq   = (int)((b >> 17) & 1u) && s->has_avx512f;
            s->has_avx512bw   = (int)((b >> 30) & 1u) && s->has_avx512f;
            s->has_avx512vl   = (int)((b >> 31) & 1u) && s->has_avx512f;
            s->has_avx512vnni = (int)((c >> 11) & 1u) && s->has_avx512f;
            s->has_hybrid     = (int)((d >> 15) & 1u);
        }
        if (s->has_osxsave && s->max_leaf >= 0x0Du) {
            cpuid_ex(0x0Du, 1, &a, &b, &c, &d);
            s->has_xsaveopt = (int)(a & 1u);
        }
        if (s->has_hybrid && s->max_leaf >= 0x1Au) {
            cpuid_ex(0x1Au, 0, &a, &b, &c, &d);
            s->hw_core_type = (uint8_t)((a >> 24) & 0xFFu);
        }
    }

    probe_xsave_sample(s);

    cpuid_ex(0x80000000u, 0, &a, &b, &c, &d);
    s->max_ext_leaf = a;
    if (s->max_ext_leaf >= 0x80000001u) {
        cpuid_ex(0x80000001u, 0, &a, &b, &c, &d);
        s->has_lzcnt     = (int)((c >> 5)  & 1u);
        s->has_prefetchw = (int)((c >> 8)  & 1u);
        s->has_rdtscp    = (int)((d >> 27) & 1u);
        if (((c >> 22) & 1u) && s->max_ext_leaf >= 0x8000001Du) {
            probe_cache_det_into(0x8000001Du, s);
        }
    }
    if (s->l1d_bytes == 0 && s->max_leaf >= 4) {
        probe_cache_det_into(4, s);
    }
    if (s->l1d_bytes == 0 && s->max_ext_leaf >= 0x80000006u) {
        cpuid_ex(0x80000005u, 0, &a, &b, &c, &d);
        s->l1d_bytes = (uint64_t)((c >> 24) & 0xFFu) * 1024ULL;
        s->l1i_bytes = (uint64_t)((d >> 24) & 0xFFu) * 1024ULL;
        s->line_sz   = c & 0xFFu;
        cpuid_ex(0x80000006u, 0, &a, &b, &c, &d);
        s->l2_bytes = (uint64_t)((c >> 16) & 0xFFFFu) * 1024ULL;
        s->l3_bytes = (uint64_t)((d >> 18) & 0x3FFFu) * 512ULL * 1024ULL;
    }
    if (s->max_ext_leaf >= 0x80000007u) {
        cpuid_ex(0x80000007u, 0, &a, &b, &c, &d);
        s->has_inv_tsc = (int)((d >> 8) & 1u);
    }
    if (s->line_sz == 0)    s->line_sz    = 64;
    if (s->l1d_bytes == 0)  s->l1d_bytes  = 32ULL * 1024ULL;
    if (s->l2_bytes == 0)   s->l2_bytes   = 256ULL * 1024ULL;
    if (s->l3_bytes == 0)   s->l3_bytes   = 8ULL * 1024ULL * 1024ULL;
    if (s->l1d_ways == 0)   s->l1d_ways   = 8;
    if (s->l1d_shared == 0) s->l1d_shared = 1;
    if (s->l2_shared == 0)  s->l2_shared  = 1;
    if (s->l3_shared == 0)  s->l3_shared  = 1;
}

static double calibrate_tsc_ghz(const char **out_src) {
    if (g_cpu.max_leaf >= 0x15u) {
        uint32_t denom = 0, numer = 0, crystal_hz = 0, d = 0;
        cpuid_ex(0x15u, 0, &denom, &numer, &crystal_hz, &d);
        if (denom != 0 && numer != 0 && crystal_hz != 0) {
            double ghz = ((double)crystal_hz * ((double)numer / (double)denom)) / 1e9;
            if (ghz >= 0.5 && ghz <= 10.0) {
                *out_src = "CPUID-0x15";
                return ghz;
            }
        }
    }

    /* Fast bracket-filtered 5x 2ms calibration (~10 ms total, rejects jittered brackets) */
    sleep_ns_eintr(500000ULL);
    double samples[5];
    int n_good = 0;
    for (int trial = 0; trial < 9 && n_good < 5; trial++) {
        uint64_t c0a = rdtsc_begin();
        uint64_t ns0 = monotonic_ns();
        uint64_t c0b = rdtsc_end();
        sleep_ns_eintr(2000000ULL);
        uint64_t c1a = rdtsc_begin();
        uint64_t ns1 = monotonic_ns();
        uint64_t c1b = rdtsc_end();
        uint64_t br0 = (c0b >= c0a) ? (c0b - c0a) : 0ULL;
        uint64_t br1 = (c1b >= c1a) ? (c1b - c1a) : 0ULL;
        if (trial < 4 && (br0 > 40000ULL || br1 > 40000ULL)) continue;
        uint64_t t0  = c0a + (br0 >> 1);
        uint64_t t1  = c1a + (br1 >> 1);
        double dt_ns = (double)safe_delta_ns(ns1, ns0);
        double dt_cy = (t1 > t0) ? (double)(t1 - t0) : 3.0 * dt_ns;
        samples[n_good++] = dt_cy / dt_ns;
    }
    if (n_good == 0) {
        if (g_cpu.max_leaf >= 0x16u) {
            uint32_t mhz = 0, b = 0, c = 0, d = 0;
            cpuid_ex(0x16u, 0, &mhz, &b, &c, &d);
            double ghz = (double)(mhz & 0xFFFFu) / 1000.0;
            if (ghz >= 0.5 && ghz <= 10.0) { *out_src = "CPUID-0x16"; return ghz; }
        }
        *out_src = "fallback-3GHz";
        return 3.0;
    }
    for (int i = 1; i < n_good; i++) {
        double key = samples[i];
        int j = i - 1;
        while (j >= 0 && samples[j] > key) { samples[j + 1] = samples[j]; j--; }
        samples[j + 1] = key;
    }
    double med = samples[n_good / 2];
    if (med < 0.1 || med > 15.0) med = 3.0;
    *out_src = "bracketed-med5";
    return med;
}

static double calibrate_pause_cycles(void) {
    uint64_t best = UINT64_MAX;
    for (int trial = 0; trial < 7; trial++) {
        uint64_t c0 = rdtsc_begin();
        for (int i = 0; i < 256; i++) {
            __asm__ volatile("pause\n\tpause\n\tpause\n\tpause" ::: "memory");
        }
        uint64_t c1 = rdtsc_end();
        if (c1 > c0 && (c1 - c0) < best) best = c1 - c0;
    }
    double cy = (best == UINT64_MAX) ? 40.0 : (double)best / 1024.0;
    if (cy < 4.0)   cy = 4.0;
    if (cy > 500.0) cy = 500.0;
    return cy;
}

typedef struct {
    int     cpu;
    int     numa;
    int     pkg;
    int     l3_dom;
    uint8_t is_smt;
    uint8_t is_ecore;
} TopoCpuEntry;

static int cmp_topo_entry(const void *pa, const void *pb) {
    const TopoCpuEntry *a = (const TopoCpuEntry *)pa;
    const TopoCpuEntry *b = (const TopoCpuEntry *)pb;
    /* Priority: physical before SMT -> NUMA node -> package -> L3 CCD -> P-core -> cpu */
    if (a->is_smt   != b->is_smt)   return (int)a->is_smt   - (int)b->is_smt;
    if (a->numa     != b->numa)     return a->numa          - b->numa;
    if (a->pkg      != b->pkg)      return a->pkg           - b->pkg;
    if (a->l3_dom   != b->l3_dom)   return a->l3_dom        - b->l3_dom;
    if (a->is_ecore != b->is_ecore) return (int)a->is_ecore - (int)b->is_ecore;
    return a->cpu - b->cpu;
}

static int count_distinct_ints(const int *vals, int n) {
    if (n <= 0) return 0;
    int *tmp = (int *)malloc((size_t)n * sizeof(int));
    if (!tmp) return 1;
    memcpy(tmp, vals, (size_t)n * sizeof(int));
    for (int i = 1; i < n; i++) {
        int key = tmp[i], j = i - 1;
        while (j >= 0 && tmp[j] > key) { tmp[j + 1] = tmp[j]; j--; }
        tmp[j + 1] = key;
    }
    int cnt = 1;
    for (int i = 1; i < n; i++) if (tmp[i] != tmp[i - 1]) cnt++;
    free(tmp);
    return cnt;
}

static void probe_cpu_topology(void) {
    long nconf = sysconf(_SC_NPROCESSORS_CONF);
    long nonln = sysconf(_SC_NPROCESSORS_ONLN);
    g_cpu.online_cpus = (nonln > 0) ? (int)nonln : 4;

    int max_probe = (nconf > 1024) ? (int)nconf : 1024;
    if (max_probe > 65536) max_probe = 65536;

    TopoCpuEntry *entries = (TopoCpuEntry *)malloc((size_t)max_probe * sizeof(TopoCpuEntry));
    uint8_t *seen_owner   = (uint8_t *)calloc((size_t)max_probe, 1);
    uint8_t *is_pcore     = (uint8_t *)calloc((size_t)max_probe, 1);
    int     *cpu_to_numa  = (int *)malloc((size_t)max_probe * sizeof(int));
    uint8_t *node_bmap    = (uint8_t *)calloc((size_t)max_probe, 1);
    if (!entries || !seen_owner || !is_pcore || !cpu_to_numa || !node_bmap) {
        perror("alloc topo"); exit(1);
    }
    for (int i = 0; i < max_probe; i++) cpu_to_numa[i] = -1;

    /* Discover NUMA nodes via /sys/devices/system/node/node%d/cpulist */
    for (int node = 0; node < 64; node++) {
        char npath[128];
        snprintf(npath, sizeof(npath), "/sys/devices/system/node/node%d/cpulist", node);
        memset(node_bmap, 0, (size_t)max_probe);
        if (parse_sysfs_cpulist(npath, node_bmap, max_probe)) {
            for (int c = 0; c < max_probe; c++) {
                if (node_bmap[c] && cpu_to_numa[c] < 0) cpu_to_numa[c] = node;
            }
        }
    }

    int n_allowed = 0;
    cpu_set_t *dyn_mask = CPU_ALLOC((size_t)max_probe);
    size_t dyn_sz = CPU_ALLOC_SIZE((size_t)max_probe);
    if (dyn_mask) {
        CPU_ZERO_S(dyn_sz, dyn_mask);
        if (sched_getaffinity(0, dyn_sz, dyn_mask) == 0) {
            for (int cpu = 0; cpu < max_probe; cpu++) {
                if (CPU_ISSET_S((size_t)cpu, dyn_sz, dyn_mask)) {
                    entries[n_allowed++].cpu = cpu;
                }
            }
        }
        CPU_FREE(dyn_mask);
    }
    if (n_allowed == 0) {
        n_allowed = (g_cpu.online_cpus < max_probe) ? g_cpu.online_cpus : max_probe;
        for (int i = 0; i < n_allowed; i++) entries[i].cpu = i;
    }

    int has_pcore_map = parse_sysfs_cpulist("/sys/devices/cpu_core/cpus", is_pcore, max_probe);

    /* Fallback hardware hybrid detection via CPUID leaf 0x1A if sysfs cpu_core is absent */
    if (!has_pcore_map && n_allowed <= 256) {
        cpu_set_t *orig_aff = CPU_ALLOC((size_t)max_probe);
        cpu_set_t *tmp_aff  = CPU_ALLOC((size_t)max_probe);
        if (orig_aff && tmp_aff && sched_getaffinity(0, dyn_sz, orig_aff) == 0) {
            uint32_t a, b, c, d;
            cpuid_ex(0, 0, &a, &b, &c, &d);
            uint32_t ml = a;
            int is_hyb = 0;
            if (ml >= 7) { cpuid_ex(7, 0, &a, &b, &c, &d); is_hyb = (int)((d >> 15) & 1u); }
            if (is_hyb && ml >= 0x1Au) {
                int saw_core = 0, saw_atom = 0;
                for (int i = 0; i < n_allowed; i++) {
                    CPU_ZERO_S(dyn_sz, tmp_aff);
                    CPU_SET_S((size_t)entries[i].cpu, dyn_sz, tmp_aff);
                    if (sched_setaffinity(0, dyn_sz, tmp_aff) == 0) {
                        cpuid_ex(0x1Au, 0, &a, &b, &c, &d);
                        uint8_t ctype = (uint8_t)((a >> 24) & 0xFFu);
                        if (ctype == 0x40u) { is_pcore[entries[i].cpu] = 1u; saw_core = 1; }
                        else if (ctype == 0x20u) { saw_atom = 1; }
                    }
                }
                if (saw_core && saw_atom) has_pcore_map = 1;
                (void)sched_setaffinity(0, dyn_sz, orig_aff);
            }
        }
        if (orig_aff) CPU_FREE(orig_aff);
        if (tmp_aff)  CPU_FREE(tmp_aff);
    }

    int total_phys = 0, total_p = 0;
    for (int i = 0; i < n_allowed; i++) {
        int cpu = entries[i].cpu;
        char path[160];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        int owner = read_sysfs_first_int(path, cpu);
        if (owner < 0 || owner >= max_probe) owner = cpu;

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        int pkg = read_sysfs_first_int(path, 0);
        if (pkg < 0) pkg = 0;

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", cpu);
        int l3_dom = read_sysfs_first_int(path, pkg);
        if (l3_dom < 0) l3_dom = pkg;

        int numa = (cpu_to_numa[cpu] >= 0) ? cpu_to_numa[cpu] : pkg;

        uint8_t is_smt   = (seen_owner[owner] != 0) ? 1u : 0u;
        uint8_t is_ecore = (has_pcore_map && !is_pcore[cpu]) ? 1u : 0u;
        if (!is_smt) {
            seen_owner[owner] = 1;
            total_phys++;
            if (!is_ecore) total_p++;
        }
        entries[i].numa     = numa;
        entries[i].pkg      = pkg;
        entries[i].l3_dom   = l3_dom;
        entries[i].is_smt   = is_smt;
        entries[i].is_ecore = is_ecore;
    }

    qsort(entries, (size_t)n_allowed, sizeof(TopoCpuEntry), cmp_topo_entry);

    int out = (n_allowed < KRONOS_MAX_CORES) ? n_allowed : KRONOS_MAX_CORES;
    int pkgs[KRONOS_MAX_CORES], numas[KRONOS_MAX_CORES], l3s[KRONOS_MAX_CORES];
    for (int i = 0; i < out; i++) {
        g_cpu.ordered_cpus[i]     = entries[i].cpu;
        g_cpu.ordered_is_ecore[i] = entries[i].is_ecore;
        g_cpu.ordered_pkg[i]      = entries[i].pkg;
        g_cpu.ordered_numa[i]     = entries[i].numa;
        g_cpu.ordered_l3[i]       = entries[i].l3_dom;
        pkgs[i]  = entries[i].pkg;
        numas[i] = entries[i].numa;
        l3s[i]   = entries[i].l3_dom;
    }
    g_cpu.avail_cpus     = out;
    g_cpu.phys_cores     = (total_phys > 0) ? total_phys : out;
    g_cpu.p_cores        = has_pcore_map ? total_p : g_cpu.phys_cores;
    g_cpu.num_packages   = count_distinct_ints(pkgs, out);
    g_cpu.num_numa_nodes = count_distinct_ints(numas, out);
    g_cpu.num_l3_domains = count_distinct_ints(l3s, out);

    free(entries);
    free(seen_owner);
    free(is_pcore);
    free(cpu_to_numa);
    free(node_bmap);
}

/* Hybrid-Aware Synchronized Grouped Per-TID PMU w/ Zero-Run Multiplexing Guard */
typedef struct {
    int      num_fds;
    int      fd_cycles[KRONOS_MAX_CORES];
    int      fd_insns[KRONOS_MAX_CORES];
    int      active;
    int      multiplexed;
    int      invalid;
    uint64_t cycles;
    uint64_t insns;
} KronosPmu;

typedef struct {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t values[2];
} PmuGroupRead;

static int pmu_open_hw_tid_ex(uint64_t base_config, pid_t tid, int is_ecore,
                              int group_fd, uint64_t read_fmt) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type           = PERF_TYPE_HARDWARE;
    pe.size           = sizeof(pe);
    pe.disabled       = (unsigned)(group_fd < 0);
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;
    pe.read_format    = read_fmt;

    /* On Intel hybrid CPUs, try extended PMU type in config[63:32] first */
    int pmu_t = is_ecore ? g_cpu.pmu_type_ecore : g_cpu.pmu_type_pcore;
    if (g_cpu.has_hybrid && pmu_t > 0) {
        pe.config = (((uint64_t)(uint32_t)pmu_t) << 32) | base_config;
        int fd = (int)syscall(SYS_perf_event_open, &pe, tid, -1, group_fd, 0);
        if (fd >= 0) return fd;
    }
    pe.config = base_config;
    return (int)syscall(SYS_perf_event_open, &pe, tid, -1, group_fd, 0);
}

static void pmu_close_all(KronosPmu *p, int upto) {
    for (int j = 0; j < upto; j++) {
        if (p->fd_insns[j]  >= 0) { close(p->fd_insns[j]);  p->fd_insns[j]  = -1; }
        if (p->fd_cycles[j] >= 0) { close(p->fd_cycles[j]); p->fd_cycles[j] = -1; }
    }
}

static void pmu_begin_tids(KronosPmu *p, const pid_t *tids, const uint8_t *is_ecore, int count) {
    memset(p, 0, sizeof(*p));
    for (int i = 0; i < KRONOS_MAX_CORES; i++) {
        p->fd_cycles[i] = -1;
        p->fd_insns[i]  = -1;
    }
    if (count <= 0 || count > KRONOS_MAX_CORES) return;

    const uint64_t rfmt = PERF_FORMAT_TOTAL_TIME_ENABLED |
                          PERF_FORMAT_TOTAL_TIME_RUNNING |
                          PERF_FORMAT_GROUP;
    for (int i = 0; i < count; i++) {
        int ec = is_ecore ? (int)is_ecore[i] : 0;
        int fc = pmu_open_hw_tid_ex(PERF_COUNT_HW_CPU_CYCLES, tids[i], ec, -1, rfmt);
        if (fc < 0) { pmu_close_all(p, i); return; }
        p->fd_cycles[i] = fc;
        int fi = pmu_open_hw_tid_ex(PERF_COUNT_HW_INSTRUCTIONS, tids[i], ec, fc, rfmt);
        if (fi < 0) { pmu_close_all(p, i + 1); return; }
        p->fd_insns[i] = fi;
    }
    p->num_fds = count;
    for (int i = 0; i < count; i++) {
        (void)ioctl(p->fd_cycles[i], PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP);
        (void)ioctl(p->fd_cycles[i], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }
    p->active = 1;
}

static void pmu_end(KronosPmu *p) {
    if (!p->active) return;
    double sum_c = 0.0, sum_i = 0.0;
    for (int i = 0; i < p->num_fds; i++) {
        (void)ioctl(p->fd_cycles[i], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    }
    for (int i = 0; i < p->num_fds; i++) {
        PmuGroupRead rd;
        memset(&rd, 0, sizeof(rd));
        if (read(p->fd_cycles[i], &rd, sizeof(rd)) == (ssize_t)sizeof(rd) && rd.nr == 2) {
            if (rd.time_enabled > 0 && rd.time_running == 0) {
                /* Core was 100% multiplexed out; mark sample invalid instead of under-reporting! */
                p->invalid = 1;
            } else {
                double scale = 1.0;
                if (rd.time_running > 0 && rd.time_running < rd.time_enabled) {
                    scale = (double)rd.time_enabled / (double)rd.time_running;
                    p->multiplexed = 1;
                }
                sum_c += (double)rd.values[0] * scale;
                sum_i += (double)rd.values[1] * scale;
            }
        } else {
            p->invalid = 1;
        }
    }
    pmu_close_all(p, p->num_fds);
    p->cycles = p->invalid ? 0ULL : (uint64_t)sum_c;
    p->insns  = p->invalid ? 0ULL : (uint64_t)sum_i;
    p->active = 0;
}

/* 1 ms Boot Hardware FMA Peak Micro-Probe (auto-detects 32 vs 64 FLOP/cy AVX-512, etc.) */
static double probe_single_fma_tier(void (*fn)(uint64_t), double flops_per_iter,
                                    double nominal_fallback, int is_ecore) {
    const uint64_t iters = 120000ULL;
    fn(20000ULL);

    double best_fpc = 0.0;
    pid_t self_tid  = (pid_t)syscall(SYS_gettid);
    uint8_t ec_u8   = (uint8_t)is_ecore;

    for (int r = 0; r < 3; r++) {
        KronosPmu pmu;
        pmu_begin_tids(&pmu, &self_tid, &ec_u8, 1);
        uint64_t c0 = rdtsc_begin();
        fn(iters);
        uint64_t c1 = rdtsc_end();
        pmu_end(&pmu);

        double cyc = (!pmu.invalid && pmu.cycles > 10000ULL)
                     ? (double)pmu.cycles
                     : (double)((c1 > c0) ? (c1 - c0) : 1ULL);
        double fpc = (flops_per_iter * (double)iters) / cyc;
        if (fpc > best_fpc) best_fpc = fpc;
    }
    if (best_fpc < 1.0 || best_fpc > 256.0) return nominal_fallback;
    return best_fpc;
}

static void kronos_probe_silicon(int req_cores) {
    memset(&g_cpu, 0, sizeof(g_cpu));
    probe_cpu_topology();

    g_cpu.pmu_type_pcore = read_sysfs_first_int("/sys/devices/cpu_core/type", 0);
    g_cpu.pmu_type_ecore = read_sysfs_first_int("/sys/devices/cpu_atom/type", 0);

    int eff_cores = (req_cores > 0 && req_cores <= g_cpu.avail_cpus) ? req_cores : g_cpu.avail_cpus;

    int max_cpu_id = 1024;
    for (int i = 0; i < eff_cores; i++) {
        if (g_cpu.ordered_cpus[i] + 1 > max_cpu_id) max_cpu_id = g_cpu.ordered_cpus[i] + 1;
    }
    uint8_t *active_bmap = (uint8_t *)calloc((size_t)max_cpu_id, 1);
    uint8_t *l2_bmap     = (uint8_t *)calloc((size_t)max_cpu_id, 1);
    if (!active_bmap || !l2_bmap) { perror("calloc bmap"); exit(1); }
    for (int i = 0; i < eff_cores; i++) active_bmap[g_cpu.ordered_cpus[i]] = 1u;

    cpu_set_t *orig_aff = CPU_ALLOC((size_t)max_cpu_id);
    cpu_set_t *pin_aff  = CPU_ALLOC((size_t)max_cpu_id);
    size_t aff_sz       = CPU_ALLOC_SIZE((size_t)max_cpu_id);
    int saved_aff       = (orig_aff && pin_aff && sched_getaffinity(0, aff_sz, orig_aff) == 0);

    if (saved_aff) {
        CPU_ZERO_S(aff_sz, pin_aff);
        CPU_SET_S((size_t)g_cpu.ordered_cpus[0], aff_sz, pin_aff);
        (void)sched_setaffinity(0, aff_sz, pin_aff);
    }

    uint32_t a, b, c, d;
    cpuid_ex(0, 0, &a, &b, &c, &d);
    memcpy(g_cpu.vendor + 0, &b, 4);
    memcpy(g_cpu.vendor + 4, &d, 4);
    memcpy(g_cpu.vendor + 8, &c, 4);
    g_cpu.vendor[12] = '\0';
    g_cpu.is_intel = (strcmp(g_cpu.vendor, "GenuineIntel") == 0);

    CpuCoreSample base;
    probe_current_core_sample(&base);

    /* Compute exact active-set L2 sharing for CPU 0 */
    uint64_t min_l2_eff = base.l2_bytes;
    {
        char l2p[160];
        snprintf(l2p, sizeof(l2p),
                 "/sys/devices/system/cpu/cpu%d/cache/index2/shared_cpu_list", g_cpu.ordered_cpus[0]);
        memset(l2_bmap, 0, (size_t)max_cpu_id);
        int active_sharers = 0;
        if (parse_sysfs_cpulist(l2p, l2_bmap, max_cpu_id)) {
            for (int k = 0; k < eff_cores; k++) {
                if (l2_bmap[g_cpu.ordered_cpus[k]]) active_sharers++;
            }
        }
        if (active_sharers > 1) min_l2_eff = base.l2_bytes / (uint64_t)active_sharers;
    }

    /* Probe EVERY active worker core (1..eff_cores-1) for airtight ISA & cache intersection! */
    for (int i = 1; i < eff_cores && saved_aff; i++) {
        CPU_ZERO_S(aff_sz, pin_aff);
        CPU_SET_S((size_t)g_cpu.ordered_cpus[i], aff_sz, pin_aff);
        if (sched_setaffinity(0, aff_sz, pin_aff) != 0) continue;

        CpuCoreSample s;
        probe_current_core_sample(&s);

        if (s.max_leaf     < base.max_leaf)     base.max_leaf     = s.max_leaf;
        if (s.max_ext_leaf < base.max_ext_leaf) base.max_ext_leaf = s.max_ext_leaf;
        base.has_sse42      &= s.has_sse42;
        base.has_popcnt     &= s.has_popcnt;
        base.has_aes        &= s.has_aes;
        base.has_f16c       &= s.has_f16c;
        base.has_osxsave    &= s.has_osxsave;
        base.has_xsaveopt   &= s.has_xsaveopt;
        base.has_avx        &= s.has_avx;
        base.has_avx2       &= s.has_avx2;
        base.has_fma        &= s.has_fma;
        base.has_bmi1       &= s.has_bmi1;
        base.has_bmi2       &= s.has_bmi2;
        base.has_lzcnt      &= s.has_lzcnt;
        base.has_prefetchw  &= s.has_prefetchw;
        base.has_avx512f    &= s.has_avx512f;
        base.has_avx512dq   &= s.has_avx512dq;
        base.has_avx512bw   &= s.has_avx512bw;
        base.has_avx512vl   &= s.has_avx512vl;
        base.has_avx512vnni &= s.has_avx512vnni;
        base.has_rdtscp     &= s.has_rdtscp;
        base.has_inv_tsc    &= s.has_inv_tsc;
        base.has_hybrid     |= s.has_hybrid;
        base.xcr0           &= s.xcr0;
        base.xsave_mask_lo  &= s.xsave_mask_lo;
        if (s.xsave_size > base.xsave_size) base.xsave_size = s.xsave_size;
        if (s.hw_core_type == 0x20u) g_cpu.ordered_is_ecore[i] = 1u;

        if (s.l1d_bytes < base.l1d_bytes) {
            base.l1d_bytes = s.l1d_bytes; base.l1d_ways = s.l1d_ways; base.l1d_shared = s.l1d_shared;
        }
        if (s.l1i_bytes < base.l1i_bytes) base.l1i_bytes = s.l1i_bytes;
        if (s.l2_bytes < base.l2_bytes) {
            base.l2_bytes = s.l2_bytes; base.l2_ways = s.l2_ways; base.l2_shared = s.l2_shared;
        }
        if (s.l3_bytes < base.l3_bytes) {
            base.l3_bytes = s.l3_bytes; base.l3_ways = s.l3_ways; base.l3_shared = s.l3_shared;
        }
        if (s.line_sz > base.line_sz) base.line_sz = s.line_sz;

        char l2p[160];
        snprintf(l2p, sizeof(l2p),
                 "/sys/devices/system/cpu/cpu%d/cache/index2/shared_cpu_list", g_cpu.ordered_cpus[i]);
        memset(l2_bmap, 0, (size_t)max_cpu_id);
        int active_sharers = 0;
        if (parse_sysfs_cpulist(l2p, l2_bmap, max_cpu_id)) {
            for (int k = 0; k < eff_cores; k++) {
                if (l2_bmap[g_cpu.ordered_cpus[k]]) active_sharers++;
            }
        }
        uint64_t l2_e = (active_sharers > 1) ? (s.l2_bytes / (uint64_t)active_sharers) : s.l2_bytes;
        if (l2_e < min_l2_eff) min_l2_eff = l2_e;
    }

    if (saved_aff) {
        CPU_ZERO_S(aff_sz, pin_aff);
        CPU_SET_S((size_t)g_cpu.ordered_cpus[0], aff_sz, pin_aff);
        (void)sched_setaffinity(0, aff_sz, pin_aff);
    }

    g_cpu.max_leaf         = base.max_leaf;
    g_cpu.max_ext_leaf     = base.max_ext_leaf;
    g_cpu.has_sse42        = base.has_sse42;
    g_cpu.has_popcnt       = base.has_popcnt;
    g_cpu.has_aes          = base.has_aes;
    g_cpu.has_f16c         = base.has_f16c;
    g_cpu.has_osxsave      = base.has_osxsave;
    g_cpu.has_xsaveopt     = base.has_xsaveopt;
    g_cpu.has_avx          = base.has_avx;
    g_cpu.has_avx2         = base.has_avx2;
    g_cpu.has_fma          = base.has_fma;
    g_cpu.has_bmi1         = base.has_bmi1;
    g_cpu.has_bmi2         = base.has_bmi2;
    g_cpu.has_lzcnt        = base.has_lzcnt;
    g_cpu.has_prefetchw    = base.has_prefetchw;
    g_cpu.has_avx512f      = base.has_avx512f;
    g_cpu.has_avx512dq     = base.has_avx512dq;
    g_cpu.has_avx512bw     = base.has_avx512bw;
    g_cpu.has_avx512vl     = base.has_avx512vl;
    g_cpu.has_avx512vnni   = base.has_avx512vnni;
    g_cpu.has_rdtscp       = base.has_rdtscp;
    g_cpu.has_inv_tsc      = base.has_inv_tsc;
    g_cpu.has_hybrid       = base.has_hybrid;
    g_cpu.xcr0             = base.xcr0;
    g_cpu.cache_line_bytes = base.line_sz;
    g_cpu.l1d_bytes        = base.l1d_bytes;
    g_cpu.l1d_ways         = base.l1d_ways;
    g_cpu.l1d_shared_by    = base.l1d_shared;
    g_cpu.l1i_bytes        = base.l1i_bytes;
    g_cpu.l2_bytes         = base.l2_bytes;
    g_cpu.l2_eff_per_core  = min_l2_eff;
    g_cpu.l2_ways          = base.l2_ways;
    g_cpu.l2_shared_by     = base.l2_shared;
    g_cpu.l3_bytes         = base.l3_bytes;
    g_cpu.l3_ways          = base.l3_ways;
    g_cpu.l3_shared_by     = base.l3_shared;

    g_cpu_has_avx_asm       = g_cpu.has_avx;
    g_cpu_has_prefetchw_asm = g_cpu.has_prefetchw;
    if (base.xsave_mask_lo >= 0x3u && base.xsave_size >= 576 && base.xsave_size <= 4096) {
        g_cpu.xsave_mask_lo     = base.xsave_mask_lo;
        g_cpu.xsave_size        = base.xsave_size;
        g_cpu_xsave_mask_lo_asm = base.xsave_mask_lo;
        g_cpu_xsave_size_asm    = base.xsave_size;
        g_cpu_use_xsave_asm     = 1;
    } else {
        g_cpu.xsave_mask_lo     = 0;
        g_cpu.xsave_size        = 512;
        g_cpu_xsave_mask_lo_asm = 0;
        g_cpu_xsave_size_asm    = 512;
        g_cpu_use_xsave_asm     = 0;
    }

    if (g_cpu.max_ext_leaf >= 0x80000004u) {
        char raw[49];
        for (uint32_t i = 0; i < 3; i++) {
            cpuid_ex(0x80000002u + i, 0, &a, &b, &c, &d);
            memcpy(raw + i * 16 + 0,  &a, 4);
            memcpy(raw + i * 16 + 4,  &b, 4);
            memcpy(raw + i * 16 + 8,  &c, 4);
            memcpy(raw + i * 16 + 12, &d, 4);
        }
        raw[48] = '\0';
        char *p = raw;
        while (*p == ' ') p++;
        size_t len = strlen(p);
        while (len > 0 && p[len - 1] == ' ') p[--len] = '\0';
        snprintf(g_cpu.brand, sizeof(g_cpu.brand), "%s", p);
    } else {
        snprintf(g_cpu.brand, sizeof(g_cpu.brand), "%s x86_64", g_cpu.vendor);
    }

    g_cpu.tsc_clocksource_ok = g_cpu.has_inv_tsc && probe_linux_tsc_clocksource();

    size_t min_sig = 0;
#ifdef AT_MINSIGSTKSZ
    min_sig = (size_t)getauxval(AT_MINSIGSTKSZ);
#endif
    if (min_sig == 0) {
        long sc = sysconf(_SC_SIGSTKSZ);
        min_sig = (sc > 0) ? (size_t)sc : 16384u;
    }
    size_t want_alt = min_sig + 16384u;
    if (want_alt < 32768u) want_alt = 32768u;
    g_cpu.altstack_sz = (want_alt + KRONOS_PAGE_SZ - 1u) & ~(size_t)(KRONOS_PAGE_SZ - 1u);

    /* CET Shadow Stack probe: attempt clean disable if active & unlocked */
    uint64_t shstk_feats = 0;
    if (sys_arch_prctl_raw(0x5005, (unsigned long)&shstk_feats) == 0 && (shstk_feats & 1ULL)) {
        if (sys_arch_prctl_raw(0x5002, 1UL) == 0) {
            shstk_feats = 0;
            if (sys_arch_prctl_raw(0x5005, (unsigned long)&shstk_feats) == 0 && !(shstk_feats & 1ULL)) {
                g_cpu.cet_shstk_disabled = 1;
            } else {
                g_cpu.cet_shstk_active = 1;
            }
        } else {
            g_cpu.cet_shstk_active = 1;
        }
    }

#if defined(__NR_membarrier) && defined(MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED)
    if (syscall(__NR_membarrier, MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED, 0, 0) == 0) {
        g_has_membar = 1;
    }
#endif

    int pmu_fc = pmu_open_hw_tid_ex(PERF_COUNT_HW_CPU_CYCLES, 0, g_cpu.ordered_is_ecore[0], -1,
                                    PERF_FORMAT_TOTAL_TIME_ENABLED |
                                    PERF_FORMAT_TOTAL_TIME_RUNNING |
                                    PERF_FORMAT_GROUP);
    if (pmu_fc >= 0) {
        int pmu_fi = pmu_open_hw_tid_ex(PERF_COUNT_HW_INSTRUCTIONS, 0, g_cpu.ordered_is_ecore[0], pmu_fc,
                                        PERF_FORMAT_TOTAL_TIME_ENABLED |
                                        PERF_FORMAT_TOTAL_TIME_RUNNING |
                                        PERF_FORMAT_GROUP);
        if (pmu_fi >= 0) { g_cpu.pmu_available = 1; close(pmu_fi); }
        close(pmu_fc);
    }

    g_cpu.tsc_ghz      = calibrate_tsc_ghz(&g_cpu.tsc_src);
    g_cpu.ns_per_cycle = 1.0 / g_cpu.tsc_ghz;
    g_pause_cycles     = calibrate_pause_cycles();

    /* Boot hardware FMA peak micro-probe (exact true FLOP/core-cycle per ISA tier) */
    g_cpu.peak_fma_fpc_sse2 = probe_single_fma_tier(kronos_fma_peak_sse2, 96.0, 8.0, g_cpu.ordered_is_ecore[0]);
    if (g_cpu.has_avx2 && g_cpu.has_fma) {
        g_cpu.peak_fma_fpc_avx2 = probe_single_fma_tier(kronos_fma_peak_avx2, 384.0, 32.0, g_cpu.ordered_is_ecore[0]);
    }
    if (g_cpu.has_avx512f) {
        g_cpu.peak_fma_fpc_avx512 = probe_single_fma_tier(kronos_fma_peak_avx512, 768.0, 64.0, g_cpu.ordered_is_ecore[0]);
    }

    if (saved_aff) (void)sched_setaffinity(0, aff_sz, orig_aff);
    if (orig_aff) CPU_FREE(orig_aff);
    if (pin_aff)  CPU_FREE(pin_aff);
    free(active_bmap);
    free(l2_bmap);
}

/* ============================================================================
 *  2. RAW SYSCALLS, TICKET/TTAS LOCKS, CSE-PROOF ERRNO & `asm goto` PREEMPT
 * ============================================================================ */

static inline void cpu_pause(void) {
    __asm__ volatile("pause" ::: "memory");
}

static inline void cpu_pause_ns(uint64_t target_ns) {
    double target_cy = (double)target_ns * g_cpu.tsc_ghz;
    int n = (int)(target_cy / g_pause_cycles);
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    for (int i = 0; i < n; i++) cpu_pause();
}

static inline uint32_t kronos_mxcsr_get(void) {
    uint32_t v;
    __asm__ volatile("stmxcsr %0" : "=m"(v));
    return v;
}

static inline void kronos_mxcsr_set(uint32_t v) {
    __asm__ volatile("ldmxcsr %0" :: "m"(v) : "memory");
}

static inline uint16_t kronos_fcw_get(void) {
    uint16_t v;
    __asm__ volatile("fnstcw %0" : "=m"(v));
    return v;
}

static inline void kronos_fcw_set(uint16_t v) {
    __asm__ volatile("fldcw %0" :: "m"(v) : "memory");
}

/* CSE-proof per-fiber errno accessor (defeats glibc __errno_location __const__ hoisting!) */
static __attribute__((noinline)) int *kronos_errno_loc(void) {
    int *(*volatile fn)(void) = __errno_location;
    int *p = fn();
    __asm__ volatile("" : "+r"(p) :: "memory");
    return p;
}

#define KRONOS_ERRNO (*kronos_errno_loc())

static inline long sys_futex_ex(void *uaddr, int op, uint32_t val,
                                const struct timespec *timeout, uint32_t val3) {
    long ret = (long)SYS_futex;
    register long r10 __asm__("r10") = (long)timeout;
    register long r8  __asm__("r8")  = 0;
    register long r9  __asm__("r9")  = (long)(uint64_t)val3;
    __asm__ volatile(
        "syscall"
        : "+a"(ret)
        : "D"(uaddr), "S"((long)op), "d"((long)val), "r"(r10), "r"(r8), "r"(r9)
        : "rcx", "r11", "cc", "memory"
    );
    return ret;
}

static inline long sys_futex(void *uaddr, int op, uint32_t val,
                             const struct timespec *timeout) {
    return sys_futex_ex(uaddr, op, val, timeout, 0u);
}

static inline long sys_futex_wait_abs_ns(void *uaddr, uint32_t val, uint64_t abs_mono_ns) {
    if (abs_mono_ns == UINT64_MAX) {
        return sys_futex(uaddr, FUTEX_WAIT_PRIVATE, val, NULL);
    }
#if defined(FUTEX_WAIT_BITSET_PRIVATE) && defined(FUTEX_BITSET_MATCH_ANY)
    struct timespec ts = { (time_t)(abs_mono_ns / 1000000000ULL),
                           (long)(abs_mono_ns % 1000000000ULL) };
    return sys_futex_ex(uaddr, FUTEX_WAIT_BITSET_PRIVATE, val, &ts, FUTEX_BITSET_MATCH_ANY);
#else
    uint64_t now = monotonic_ns();
    uint64_t rem = (abs_mono_ns > now) ? (abs_mono_ns - now) : 1ULL;
    struct timespec ts = { (time_t)(rem / 1000000000ULL), (long)(rem % 1000000000ULL) };
    return sys_futex(uaddr, FUTEX_WAIT_PRIVATE, val, &ts);
#endif
}

static inline long sys_tgkill_raw(pid_t tgid, pid_t tid, int sig) {
    long ret = (long)SYS_tgkill;
    __asm__ volatile(
        "syscall"
        : "+a"(ret)
        : "D"((long)tgid), "S"((long)tid), "d"((long)sig)
        : "rcx", "r11", "cc", "memory"
    );
    return ret;
}

static inline void sys_membarrier_expedited(void) {
#if defined(__NR_membarrier) && defined(MEMBARRIER_CMD_PRIVATE_EXPEDITED)
    if (LIKELY(g_has_membar)) {
        long ret = (long)__NR_membarrier;
        __asm__ volatile(
            "syscall"
            : "+a"(ret)
            : "D"((long)MEMBARRIER_CMD_PRIVATE_EXPEDITED), "S"(0L), "d"(0L)
            : "rcx", "r11", "cc", "memory"
        );
        return;
    }
#endif
    atomic_thread_fence(memory_order_seq_cst);
}

/* Compact 4B TTAS lock for embedded per-object structures */
typedef struct {
    _Atomic int locked;
} KronosCompactSpin;

static inline void cspin_init(KronosCompactSpin *s) {
    atomic_init(&s->locked, 0);
}

static inline void cspin_lock(KronosCompactSpin *s) {
    if (LIKELY(atomic_load_explicit(&s->locked, memory_order_relaxed) == 0 &&
               atomic_exchange_explicit(&s->locked, 1, memory_order_acquire) == 0)) {
        return;
    }
    int max_pauses = (int)(1200.0 / g_pause_cycles);
    if (max_pauses < 2)  max_pauses = 2;
    if (max_pauses > 32) max_pauses = 32;

    int backoff = 1;
    int rounds  = 0;
    for (;;) {
        while (atomic_load_explicit(&s->locked, memory_order_relaxed) != 0) {
            if (UNLIKELY(++rounds > 256)) {
                sched_yield();
                rounds = 0;
            } else {
                for (int i = 0; i < backoff; i++) cpu_pause();
                if (backoff < max_pauses) backoff <<= 1;
            }
        }
        if (LIKELY(atomic_exchange_explicit(&s->locked, 1, memory_order_acquire) == 0))
            return;
    }
}

static inline void cspin_unlock(KronosCompactSpin *s) {
    atomic_store_explicit(&s->locked, 0, memory_order_release);
}

/* FIFO Fair Ticket Lock for global contended structures (starvation-free at 128 cores) */
typedef struct {
    CACHELINE_ALIGN _Atomic uint32_t next_ticket;
    _Atomic uint32_t                 now_serving;
} KronosTicketLock;

static inline void tlock_init(KronosTicketLock *l) {
    atomic_init(&l->next_ticket, 0u);
    atomic_init(&l->now_serving, 0u);
}

static inline void tlock_lock(KronosTicketLock *l) {
    uint32_t my = atomic_fetch_add_explicit(&l->next_ticket, 1u, memory_order_relaxed);
    int rounds  = 0;
    for (;;) {
        uint32_t cur = atomic_load_explicit(&l->now_serving, memory_order_acquire);
        if (LIKELY(cur == my)) return;
        uint32_t dist = my - cur;
        if (UNLIKELY(++rounds > 256)) {
            sched_yield();
            rounds = 0;
        } else {
            uint32_t pauses = (dist > 8u) ? 8u : dist;
            for (uint32_t i = 0; i < pauses; i++) cpu_pause();
        }
    }
}

static inline void tlock_unlock(KronosTicketLock *l) {
    uint32_t cur = atomic_load_explicit(&l->now_serving, memory_order_relaxed);
    atomic_store_explicit(&l->now_serving, cur + 1u, memory_order_release);
}

/* Forward declarations */
typedef struct KronosFiber   KronosFiber;
typedef struct KronosWorker  KronosWorker;
typedef struct SlabCoreCache SlabCoreCache;

__thread KronosWorker *g_tls_worker
    KRONOS_ASM_VISIBLE __attribute__((tls_model("local-exec"))) = NULL;
__thread SlabCoreCache *g_tls_slab
    KRONOS_ASM_VISIBLE __attribute__((tls_model("local-exec"))) = NULL;
__thread _Atomic uint32_t g_tls_preempt_word
    KRONOS_ASM_VISIBLE __attribute__((tls_model("local-exec"))) = 0;

void kronos_deferred_preempt_c(void) __attribute__((noinline, cold));

static inline KronosWorker *kronos_self(void) {
    KronosWorker *w;
    __asm__ volatile("movq %%fs:g_tls_worker@tpoff, %0" : "=r"(w));
    return w;
}

static inline void kronos_set_self(KronosWorker *w) {
    __asm__ volatile("movq %0, %%fs:g_tls_worker@tpoff" :: "r"(w) : "memory");
}

static inline SlabCoreCache *kronos_self_slab(void) {
    SlabCoreCache *sc;
    __asm__ volatile("movq %%fs:g_tls_slab@tpoff, %0" : "=r"(sc));
    return sc;
}

static inline void kronos_set_self_slab(SlabCoreCache *sc) {
    __asm__ volatile("movq %0, %%fs:g_tls_slab@tpoff" :: "r"(sc) : "memory");
}

static inline uint32_t kronos_preempt_read_word(void) {
    uint32_t w;
    __asm__ volatile("movl %%fs:g_tls_preempt_word@tpoff, %0" : "=r"(w));
    return w;
}

static inline void kronos_preempt_set_word(uint32_t w) {
    __asm__ volatile("movl %0, %%fs:g_tls_preempt_word@tpoff" :: "r"(w) : "memory");
}

static inline void kronos_preempt_latch_resched(void) {
    __asm__ volatile("orl $0x80000000, %%fs:g_tls_preempt_word@tpoff" ::: "cc", "memory");
}

static inline void kronos_preempt_disable(void) {
    __asm__ volatile("addl $1, %%fs:g_tls_preempt_word@tpoff" ::: "cc", "memory");
}

static inline void kronos_preempt_enable_noresched(void) {
    __asm__ volatile("subl $1, %%fs:g_tls_preempt_word@tpoff" ::: "cc", "memory");
}

static inline void kronos_preempt_enable(void) {
    __asm__ volatile goto(
        "subl $1, %%fs:g_tls_preempt_word@tpoff\n\t"
        "js %l0"
        :
        :
        : "cc", "memory"
        : slow
    );
    return;
slow:
    kronos_deferred_preempt_c();
}

/* ============================================================================
 *  3. BONWICK TWO-MAGAZINE O(1) SLAB (SENTINEL FAST PATH + EXACT COLORING)
 * ============================================================================ */

static const size_t k_slab_sizes[KRONOS_SLAB_CLASSES] = {
    32, 48, 64, 80, 96, 128, 160, 192,
    256, 320, 384, 512, 768, 1024, 2048, 4096
};

static uint8_t g_slab_lut[257];

typedef struct SlabMagazine {
    struct SlabMagazine *next;
    uint32_t             count;
    uint32_t             cap;
    void                *objs[];
} SlabMagazine;

#define KRONOS_MAG_STRIDE_SM  576u
#define KRONOS_MAG_STRIDE_LG  192u

_Static_assert(offsetof(SlabMagazine, objs) == 16, "SlabMagazine header must be 16B");

/* Read-only sentinel eliminates m != NULL branch on BOTH alloc and free fast paths! */
static SlabMagazine g_empty_mag = { NULL, 0u, 0u };

typedef struct {
    SlabMagazine *loaded;
    SlabMagazine *prev;
} SlabCoreClass;

struct SlabCoreCache {
    CACHELINE_ALIGN SlabCoreClass classes[KRONOS_SLAB_CLASSES];
    _Atomic uint64_t fast_allocs;
    _Atomic uint64_t fast_frees;
    _Atomic uint64_t mag_swaps;
    _Atomic uint64_t depot_refills;
    _Atomic uint64_t depot_drains;
};

typedef struct {
    CACHELINE_ALIGN KronosTicketLock lock;
    uint32_t      mag_cap;
    SlabMagazine *full_mags;
    SlabMagazine *empty_mags;
    SlabMagazine *depot_mag;
    size_t        obj_size;
    CACHELINE_ALIGN _Atomic uint32_t color_cursor;
} SlabDepot;

typedef struct SlabArenaChunk {
    struct SlabArenaChunk *next;
    size_t                 size;
    size_t                 used;
    uint64_t               _pad[5];
} SlabArenaChunk;

_Static_assert(sizeof(SlabArenaChunk) == 64, "SlabArenaChunk header must be 64B aligned");

typedef struct {
    CACHELINE_ALIGN KronosTicketLock arena_lock;
    SlabArenaChunk                   *chunks;
    SlabMagazine                     *free_sm_mags;
    SlabMagazine                     *free_lg_mags;
    _Atomic size_t                    total_mapped_bytes;
    SlabDepot                         depots[KRONOS_SLAB_CLASSES];
    SlabCoreCache                     cores[KRONOS_MAX_CORES];
} KronosSlab;

static KronosSlab g_slab;

static inline int slab_class_for(size_t sz) {
    if (UNLIKELY(sz > 4096u)) return -1;
    return (int)g_slab_lut[(sz + 15u) >> 4];
}

/* Exact modular cache-colored zero-strand multi-chunk arena carver */
static uint8_t *slab_arena_carve_colored(size_t bytes, size_t align, int use_color, size_t want_mod) {
    if (align < 64u) align = 64u;
    bytes = (bytes + 63u) & ~(size_t)63u;
    for (;;) {
        tlock_lock(&g_slab.arena_lock);
        for (SlabArenaChunk *ch = g_slab.chunks; ch != NULL; ch = ch->next) {
            size_t off = (ch->used + align - 1u) & ~(align - 1u);
            if (use_color) {
                size_t cur_mod = off & (KRONOS_PAGE_SZ - 1u);
                size_t delta   = (want_mod - cur_mod) & (KRONOS_PAGE_SZ - 1u);
                off += delta;
            }
            if (off <= ch->size && ch->size - off >= bytes) {
                uint8_t *ptr = (uint8_t *)ch + off;
                ch->used = off + bytes;
                tlock_unlock(&g_slab.arena_lock);
                return ptr;
            }
        }
        tlock_unlock(&g_slab.arena_lock);

        size_t need     = bytes + align + KRONOS_PAGE_SZ * 2u;
        size_t alloc_sz = (need > KRONOS_SLAB_CHUNK_SZ)
                          ? ((need + KRONOS_PAGE_SZ - 1u) & ~(size_t)(KRONOS_PAGE_SZ - 1u))
                          : KRONOS_SLAB_CHUNK_SZ;
        size_t mapped_len = 0;
        void *p = xmmap_huge_aligned(alloc_sz, &mapped_len);
        SlabArenaChunk *nch = (SlabArenaChunk *)p;
        nch->size = mapped_len;
        nch->used = KRONOS_PAGE_SZ;

        tlock_lock(&g_slab.arena_lock);
        for (SlabArenaChunk *ch = g_slab.chunks; ch != NULL; ch = ch->next) {
            size_t off = (ch->used + align - 1u) & ~(align - 1u);
            if (use_color) {
                size_t cur_mod = off & (KRONOS_PAGE_SZ - 1u);
                size_t delta   = (want_mod - cur_mod) & (KRONOS_PAGE_SZ - 1u);
                off += delta;
            }
            if (off <= ch->size && ch->size - off >= bytes) {
                uint8_t *ptr = (uint8_t *)ch + off;
                ch->used = off + bytes;
                tlock_unlock(&g_slab.arena_lock);
                (void)munmap(p, mapped_len);
                return ptr;
            }
        }
        nch->next     = g_slab.chunks;
        g_slab.chunks = nch;
        atomic_fetch_add_explicit(&g_slab.total_mapped_bytes, mapped_len, memory_order_relaxed);
        size_t off = (nch->used + align - 1u) & ~(align - 1u);
        if (use_color) {
            size_t cur_mod = off & (KRONOS_PAGE_SZ - 1u);
            size_t delta   = (want_mod - cur_mod) & (KRONOS_PAGE_SZ - 1u);
            off += delta;
        }
        uint8_t *ptr = (uint8_t *)nch + off;
        nch->used    = off + bytes;
        tlock_unlock(&g_slab.arena_lock);
        return ptr;
    }
}

static SlabMagazine *slab_alloc_mag_batch(size_t count, uint32_t cap) {
    int is_lg        = (cap <= KRONOS_SLAB_MAG_CAP_LG);
    size_t stride    = is_lg ? KRONOS_MAG_STRIDE_LG : KRONOS_MAG_STRIDE_SM;
    SlabMagazine **pool_ptr = is_lg ? &g_slab.free_lg_mags : &g_slab.free_sm_mags;

    SlabMagazine *head = NULL;
    size_t got = 0;
    while (got < count) {
        tlock_lock(&g_slab.arena_lock);
        while (got < count && *pool_ptr != NULL) {
            SlabMagazine *m = *pool_ptr;
            *pool_ptr = m->next;
            m->next   = head;
            m->count  = 0;
            m->cap    = cap;
            head      = m;
            got++;
        }
        tlock_unlock(&g_slab.arena_lock);
        if (got < count) {
            size_t block  = is_lg ? 16384u : 65536u;
            uint8_t *base = slab_arena_carve_colored(block, 64u, 0, 0u);
            size_t nmags  = block / stride;
            SlabMagazine *new_free = NULL;
            for (size_t i = nmags; i > 0; i--) {
                SlabMagazine *m = (SlabMagazine *)(base + (i - 1u) * stride);
                m->next  = new_free;
                new_free = m;
            }
            SlabMagazine *tail = (SlabMagazine *)(base + (nmags - 1u) * stride);
            tlock_lock(&g_slab.arena_lock);
            tail->next = *pool_ptr;
            *pool_ptr  = new_free;
            tlock_unlock(&g_slab.arena_lock);
        }
    }
    return head;
}

static void kronos_slab_init(void) {
    memset(&g_slab, 0, sizeof(g_slab));
    tlock_init(&g_slab.arena_lock);
    atomic_init(&g_slab.total_mapped_bytes, 0);

    for (size_t idx = 0; idx <= 256; idx++) {
        size_t sz = idx << 4;
        uint8_t cls = KRONOS_SLAB_CLASSES - 1;
        for (int c = 0; c < KRONOS_SLAB_CLASSES; c++) {
            if (sz <= k_slab_sizes[c]) { cls = (uint8_t)c; break; }
        }
        g_slab_lut[idx] = cls;
    }

    for (int c = 0; c < KRONOS_SLAB_CLASSES; c++) {
        tlock_init(&g_slab.depots[c].lock);
        g_slab.depots[c].obj_size = k_slab_sizes[c];
        g_slab.depots[c].mag_cap  = (k_slab_sizes[c] <= 512u) ? KRONOS_SLAB_MAG_CAP_SM
                                                              : KRONOS_SLAB_MAG_CAP_LG;
        atomic_init(&g_slab.depots[c].color_cursor, 0u);
    }

    for (int core = 0; core < KRONOS_MAX_CORES; core++) {
        for (int c = 0; c < KRONOS_SLAB_CLASSES; c++) {
            g_slab.cores[core].classes[c].loaded = &g_empty_mag;
            g_slab.cores[core].classes[c].prev   = &g_empty_mag;
        }
        atomic_init(&g_slab.cores[core].fast_allocs,   0ULL);
        atomic_init(&g_slab.cores[core].fast_frees,    0ULL);
        atomic_init(&g_slab.cores[core].mag_swaps,     0ULL);
        atomic_init(&g_slab.cores[core].depot_refills, 0ULL);
        atomic_init(&g_slab.cores[core].depot_drains,  0ULL);
    }

    size_t mapped_len = 0;
    void *p = xmmap_huge_aligned(KRONOS_SLAB_CHUNK_SZ, &mapped_len);
    SlabArenaChunk *ch = (SlabArenaChunk *)p;
    ch->next = NULL;
    ch->size = mapped_len;
    ch->used = KRONOS_PAGE_SZ;
    g_slab.chunks = ch;
    atomic_store_explicit(&g_slab.total_mapped_bytes, mapped_len, memory_order_relaxed);
}

static void kronos_slab_destroy(void) {
    SlabArenaChunk *ch = g_slab.chunks;
    while (ch) {
        SlabArenaChunk *next = ch->next;
        size_t sz = ch->size;
        (void)munmap(ch, sz);
        ch = next;
    }
    memset(&g_slab, 0, sizeof(g_slab));
}

/* Exact modular cache-line coloring (uniform across ALL ncolors even under mixed classes!) */
static SlabMagazine *slab_build_mags_unlocked(int cls, SlabMagazine **out_tail) {
    SlabDepot *d     = &g_slab.depots[cls];
    size_t obj_sz    = d->obj_size;
    uint32_t mcap    = d->mag_cap;
    size_t mag_bytes = obj_sz * (size_t)mcap;

    size_t nmags = (65536u + mag_bytes - 1u) / mag_bytes;
    if (nmags < 1) nmags = 1;
    size_t raw_bytes  = nmags * mag_bytes;
    size_t nat_align  = obj_sz & (~obj_sz + 1u);
    size_t color_step = (nat_align >= 64u) ? nat_align : 64u;

    uint8_t *base;
    if (obj_sz >= KRONOS_PAGE_SZ) {
        base = slab_arena_carve_colored(raw_bytes, KRONOS_PAGE_SZ, 0, 0u);
    } else {
        uint32_t cur    = atomic_fetch_add_explicit(&d->color_cursor, 1u, memory_order_relaxed);
        size_t ncolors  = KRONOS_PAGE_SZ / color_step;
        if (ncolors < 1u) ncolors = 1u;
        size_t want_mod = (cur % ncolors) * color_step;
        base = slab_arena_carve_colored(raw_bytes, color_step, 1, want_mod);
    }

    SlabMagazine *mags = slab_alloc_mag_batch(nmags, mcap);
    SlabMagazine *cur  = mags;
    SlabMagazine *tail = NULL;
    for (size_t mi = 0; mi < nmags; mi++) {
        uint8_t *mbase = base + mi * mag_bytes;
        for (uint32_t k = 0; k < mcap; k++) {
            cur->objs[mcap - 1u - k] = mbase + (size_t)k * obj_sz;
        }
        cur->count = mcap;
        tail = cur;
        cur  = cur->next;
    }
    *out_tail = tail;
    return mags;
}

static void *slab_alloc_depot(int cls) {
    SlabDepot *d = &g_slab.depots[cls];
    for (;;) {
        tlock_lock(&d->lock);
        SlabMagazine *m = d->depot_mag;
        if (LIKELY(m != NULL && m->count > 0)) {
            void *res = m->objs[--m->count];
            tlock_unlock(&d->lock);
            return res;
        }
        if (m) {
            m->next       = d->empty_mags;
            d->empty_mags = m;
            d->depot_mag  = NULL;
        }
        if (LIKELY(d->full_mags != NULL)) {
            m            = d->full_mags;
            d->full_mags = m->next;
            m->next      = NULL;
            d->depot_mag = m;
            void *res    = m->objs[--m->count];
            tlock_unlock(&d->lock);
            return res;
        }
        tlock_unlock(&d->lock);

        SlabMagazine *tail  = NULL;
        SlabMagazine *batch = slab_build_mags_unlocked(cls, &tail);
        tlock_lock(&d->lock);
        tail->next   = d->full_mags;
        d->full_mags = batch;
        tlock_unlock(&d->lock);
    }
}

static void slab_free_depot(int cls, void *ptr) {
    SlabDepot *d = &g_slab.depots[cls];
    for (;;) {
        tlock_lock(&d->lock);
        SlabMagazine *m = d->depot_mag;
        if (LIKELY(m != NULL && m->count < m->cap)) {
            m->objs[m->count++] = ptr;
            tlock_unlock(&d->lock);
            return;
        }
        if (m) {
            m->next      = d->full_mags;
            d->full_mags = m;
            d->depot_mag = NULL;
        }
        if (LIKELY(d->empty_mags != NULL)) {
            m             = d->empty_mags;
            d->empty_mags = m->next;
            m->next       = NULL;
            d->depot_mag  = m;
            m->objs[m->count++] = ptr;
            tlock_unlock(&d->lock);
            return;
        }
        tlock_unlock(&d->lock);

        SlabMagazine *nm = slab_alloc_mag_batch(1, d->mag_cap);
        tlock_lock(&d->lock);
        nm->next      = d->empty_mags;
        d->empty_mags = nm;
        tlock_unlock(&d->lock);
    }
}

static void *slab_alloc_slow(SlabCoreCache *cc, int cls) __attribute__((noinline, cold));
static void *slab_alloc_slow(SlabCoreCache *cc, int cls) {
    SlabCoreClass *cl = &cc->classes[cls];
    SlabMagazine  *m  = cl->loaded;
    if (cl->prev->count > 0) {
        cl->loaded = cl->prev;
        cl->prev   = m;
        m          = cl->loaded;
        stat_inc(&cc->mag_swaps);
        void *res = m->objs[--m->count];
        stat_inc(&cc->fast_allocs);
        return res;
    }

    SlabMagazine *to_ret = cl->prev;
    cl->prev   = m;
    cl->loaded = &g_empty_mag;

    SlabDepot *d = &g_slab.depots[cls];
    tlock_lock(&d->lock);
    if (to_ret != &g_empty_mag) {
        to_ret->next  = d->empty_mags;
        d->empty_mags = to_ret;
    }
    SlabMagazine *nm = d->full_mags;
    if (LIKELY(nm != NULL)) {
        d->full_mags = nm->next;
        tlock_unlock(&d->lock);
        nm->next = NULL;
    } else {
        tlock_unlock(&d->lock);
        SlabMagazine *tail  = NULL;
        SlabMagazine *batch = slab_build_mags_unlocked(cls, &tail);
        nm = batch;
        SlabMagazine *rest = batch->next;
        nm->next = NULL;
        if (rest != NULL) {
            tlock_lock(&d->lock);
            tail->next   = d->full_mags;
            d->full_mags = rest;
            tlock_unlock(&d->lock);
        }
    }

    cl->loaded = nm;
    stat_inc(&cc->depot_refills);
    void *res = nm->objs[--nm->count];
    stat_inc(&cc->fast_allocs);
    return res;
}

static void slab_free_slow(SlabCoreCache *cc, int cls, void *ptr) __attribute__((noinline, cold));
static void slab_free_slow(SlabCoreCache *cc, int cls, void *ptr) {
    SlabCoreClass *cl = &cc->classes[cls];
    SlabMagazine  *m  = cl->loaded;
    if (cl->prev->count < cl->prev->cap) {
        cl->loaded = cl->prev;
        cl->prev   = m;
        m          = cl->loaded;
        stat_inc(&cc->mag_swaps);
        m->objs[m->count++] = ptr;
        stat_inc(&cc->fast_frees);
        return;
    }

    SlabMagazine *to_ret = cl->prev;
    cl->prev   = m;
    cl->loaded = &g_empty_mag;

    SlabDepot *d = &g_slab.depots[cls];
    tlock_lock(&d->lock);
    if (to_ret != &g_empty_mag) {
        to_ret->next = d->full_mags;
        d->full_mags = to_ret;
    }
    SlabMagazine *nm = d->empty_mags;
    if (LIKELY(nm != NULL)) {
        d->empty_mags = nm->next;
        tlock_unlock(&d->lock);
        nm->next = NULL;
    } else {
        tlock_unlock(&d->lock);
        nm = slab_alloc_mag_batch(1, d->mag_cap);
    }

    cl->loaded = nm;
    stat_inc(&cc->depot_drains);
    nm->objs[nm->count++] = ptr;
    stat_inc(&cc->fast_frees);
}

static void *kronos_alloc(size_t sz) {
    int cls = slab_class_for(sz);
    if (UNLIKELY(cls < 0)) {
        kronos_preempt_disable();
        void *big = xaligned_alloc(64, sz);
        kronos_preempt_enable();
        return big;
    }

    kronos_preempt_disable();
    SlabCoreCache *cc = kronos_self_slab();
    if (UNLIKELY(!cc)) {
        kronos_preempt_enable();
        void *res = slab_alloc_depot(cls);
#ifdef KRONOS_SLAB_DEBUG
        memset(res, 0xAA, k_slab_sizes[cls]);
#endif
        return res;
    }

    SlabMagazine *m = cc->classes[cls].loaded;
    if (LIKELY(m->count > 0)) {
        void *res = m->objs[--m->count];
        stat_inc(&cc->fast_allocs);
        kronos_preempt_enable();
#ifdef KRONOS_SLAB_DEBUG
        memset(res, 0xAA, k_slab_sizes[cls]);
#endif
        return res;
    }

    void *res = slab_alloc_slow(cc, cls);
    kronos_preempt_enable();
#ifdef KRONOS_SLAB_DEBUG
    memset(res, 0xAA, k_slab_sizes[cls]);
#endif
    return res;
}

static void kronos_free(void *ptr, size_t sz) {
    if (!ptr) return;
    int cls = slab_class_for(sz);
    if (UNLIKELY(cls < 0)) {
        kronos_preempt_disable();
        free(ptr);
        kronos_preempt_enable();
        return;
    }

    size_t obj_sz    = k_slab_sizes[cls];
    size_t nat_align = obj_sz & (~obj_sz + 1u);
    if (UNLIKELY(((uintptr_t)ptr & (nat_align - 1u)) != 0)) {
        fprintf(stderr, "[KRONOS] Fatal: kronos_free(%p, %zu) violates class-%d align %zu\n",
                ptr, sz, cls, nat_align);
        abort();
    }
#ifdef KRONOS_SLAB_DEBUG
    memset(ptr, 0xDE, obj_sz);
#endif

    kronos_preempt_disable();
    SlabCoreCache *cc = kronos_self_slab();
    if (UNLIKELY(!cc)) {
        kronos_preempt_enable();
        slab_free_depot(cls, ptr);
        return;
    }

    SlabMagazine *m = cc->classes[cls].loaded;
    if (LIKELY(m->count < m->cap)) {
        m->objs[m->count++] = ptr;
        stat_inc(&cc->fast_frees);
        kronos_preempt_enable();
        return;
    }

    slab_free_slow(cc, cls, ptr);
    kronos_preempt_enable();
}

/* ============================================================================
 *  4. HAND-WRITTEN x86_64 ASSEMBLY SWITCHER, XSAVE64, CANARIES & PEAK PROBES
 * ============================================================================ */

typedef struct {
    CACHELINE_ALIGN uint64_t rsp;   /* +0  */
    uint64_t rbp;                   /* +8  */
    uint64_t rbx;                   /* +16 */
    uint64_t r12;                   /* +24 */
    uint64_t r13;                   /* +32 */
    uint64_t r14;                   /* +40 */
    uint64_t r15;                   /* +48 */
    uint32_t mxcsr;                 /* +56 */
    uint16_t fcw;                   /* +60 */
    uint16_t _pad;                  /* +62 */
} KronosCtx;

_Static_assert(sizeof(KronosCtx) == 64,         "KronosCtx must fit in 1 cache line (64B)");
_Static_assert(offsetof(KronosCtx, rsp)   == 0,  "KronosCtx.rsp mismatch");
_Static_assert(offsetof(KronosCtx, rbp)   == 8,  "KronosCtx.rbp mismatch");
_Static_assert(offsetof(KronosCtx, rbx)   == 16, "KronosCtx.rbx mismatch");
_Static_assert(offsetof(KronosCtx, r12)   == 24, "KronosCtx.r12 mismatch");
_Static_assert(offsetof(KronosCtx, r13)   == 32, "KronosCtx.r13 mismatch");
_Static_assert(offsetof(KronosCtx, r14)   == 40, "KronosCtx.r14 mismatch");
_Static_assert(offsetof(KronosCtx, r15)   == 48, "KronosCtx.r15 mismatch");
_Static_assert(offsetof(KronosCtx, mxcsr) == 56, "KronosCtx.mxcsr mismatch");
_Static_assert(offsetof(KronosCtx, fcw)   == 60, "KronosCtx.fcw mismatch");

extern void kronos_ctx_to_sched(KronosCtx *fiber, const KronosCtx *sched);
extern char kronos_ctx_to_sched_end[];
extern void kronos_ctx_from_sched(KronosCtx *sched, const KronosCtx *fiber);
extern char kronos_ctx_from_sched_end[];
extern void kronos_ctx_exit_to_sched(const KronosCtx *sched) __attribute__((noreturn));
extern char kronos_ctx_exit_to_sched_end[];
extern void kronos_ctx_switch_direct(KronosCtx *from, const KronosCtx *to);
extern char kronos_ctx_switch_direct_end[];
extern void kronos_fiber_bootstrap(void);
extern void kronos_async_preempt_trampoline(void);
extern char kronos_async_preempt_trampoline_end[];

extern uint64_t kronos_preempt_canary_avx512(uint64_t seed, uint64_t iters);
extern uint64_t kronos_preempt_canary_avx2(uint64_t seed, uint64_t iters);
extern uint64_t kronos_preempt_canary_sse2(uint64_t seed, uint64_t iters);
extern int64_t  kronos_vnni_dot_u8s8(int64_t n64, const uint8_t *a, const int8_t *b);

void kronos_fiber_entry_c(KronosFiber *f) KRONOS_ASM_VISIBLE __attribute__((noreturn));
void kronos_preempt_yield_c(void)         KRONOS_ASM_VISIBLE;

#define KRONOS_CANARY_PROLOGUE()                               \
    "pushq %rbx\n\t" ".cfi_adjust_cfa_offset 8\n\t" ".cfi_rel_offset rbx, 0\n\t" \
    "pushq %rbp\n\t" ".cfi_adjust_cfa_offset 8\n\t" ".cfi_rel_offset rbp, 0\n\t" \
    "pushq %r12\n\t" ".cfi_adjust_cfa_offset 8\n\t" ".cfi_rel_offset r12, 0\n\t" \
    "pushq %r13\n\t" ".cfi_adjust_cfa_offset 8\n\t" ".cfi_rel_offset r13, 0\n\t" \
    "pushq %r14\n\t" ".cfi_adjust_cfa_offset 8\n\t" ".cfi_rel_offset r14, 0\n\t" \
    "pushq %r15\n\t" ".cfi_adjust_cfa_offset 8\n\t" ".cfi_rel_offset r15, 0\n\t" \
    "subq $24, %rsp\n\t" ".cfi_adjust_cfa_offset 24\n\t"       \
    "stmxcsr 0(%rsp)\n\t"                                      \
    "movl 0(%rsp), %eax\n\t"                                   \
    "andl $0xFFFF9FFF, %eax\n\t"                               \
    "orl  $0x2000, %eax\n\t"                                   \
    "movl %eax, 4(%rsp)\n\t"                                   \
    "ldmxcsr 4(%rsp)\n\t"                                      \
    "movq %rdi, %rax\n\t"                                      \
    "movabsq $0x6A09E667F3BCC908, %rcx\n\t"                    \
    "xorq %rcx, %rax\n\t"                                      \
    "movq %rax, 8(%rsp)\n\t"                                   \
    "movabsq $0xBB67AE8584CAA73B, %rcx\n\t"                    \
    "addq %rdi, %rcx\n\t"                                      \
    "movq %rcx, 16(%rsp)\n\t"                                  \
    "fildq 8(%rsp)\n\t"                                        \
    "fildq 16(%rsp)\n\t"

#define KRONOS_CANARY_EPILOGUE()                               \
    "stmxcsr 16(%rsp)\n\t"                                     \
    "movl 16(%rsp), %ecx\n\t"                                  \
    "xorl 4(%rsp), %ecx\n\t"                                   \
    "andl $0xFFC0, %ecx\n\t"                                   \
    "shlq $32, %rcx\n\t"                                       \
    "xorq %rcx, %rax\n\t"                                      \
    "ldmxcsr 0(%rsp)\n\t"                                      \
    "fistpq 16(%rsp)\n\t"                                      \
    "fistpq 8(%rsp)\n\t"                                       \
    "xorq 16(%rsp), %rax\n\t"                                  \
    "rolq $17, %rax\n\t"                                       \
    "addq 8(%rsp), %rax\n\t"                                   \
    "addq $24, %rsp\n\t" ".cfi_adjust_cfa_offset -24\n\t"      \
    "popq %r15\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r15\n\t" \
    "popq %r14\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r14\n\t" \
    "popq %r13\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r13\n\t" \
    "popq %r12\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r12\n\t" \
    "popq %rbp\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore rbp\n\t" \
    "popq %rbx\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore rbx\n\t" \
    "ret\n\t"

#define KRONOS_CANARY_GPR_CF_DF_STEP()         \
    "xorq -8(%rsp), %rdx\n\t"                  \
    "std\n\t"                                  \
    "addq %r8,  %rax\n\t"                      \
    "adcq %r9,  %rbx\n\t"                      \
    "adcq %r10, %rcx\n\t"                      \
    "adcq %r11, %rdx\n\t"                      \
    "adcq %r12, %rbp\n\t"                      \
    "adcq %r13, %r14\n\t"                      \
    "adcq %r15, %rdi\n\t"                      \
    "pushfq\n\t" ".cfi_adjust_cfa_offset 8\n\t"\
    "cld\n\t"                                  \
    "testq $0x400, (%rsp)\n\t"                 \
    "jnz 1f\n\t"                               \
    "notq %rax\n\t"                            \
    "1:\n\t"                                   \
    "addq $8, %rsp\n\t" ".cfi_adjust_cfa_offset -8\n\t" \
    "rolq $7,   %r12\n\t"                      \
    "rorq $11,  %r13\n\t"                      \
    "xorq %rax, %r15\n\t"                      \
    "xorq -128(%rsp), %rax\n\t" "movq %rax, -128(%rsp)\n\t" \
    "addq  -64(%rsp), %rbx\n\t" "movq %rbx,  -64(%rsp)\n\t" \
    "xorq  -16(%rsp), %rcx\n\t" "movq %rcx,  -16(%rsp)\n\t" \
    "movq %rdx, -8(%rsp)\n\t"

#define KRONOS_CANARY_FOLD_REDZONE_AND_ALL_GPRS()    \
    "xorq -128(%rsp), %rax\n\t" "rolq $5,  %rax\n\t" \
    "addq -120(%rsp), %rax\n\t" "rolq $7,  %rax\n\t" \
    "xorq -112(%rsp), %rax\n\t" "rolq $11, %rax\n\t" \
    "addq -104(%rsp), %rax\n\t" "rolq $13, %rax\n\t" \
    "xorq  -96(%rsp), %rax\n\t" "rolq $17, %rax\n\t" \
    "addq  -88(%rsp), %rax\n\t" "rolq $19, %rax\n\t" \
    "xorq  -80(%rsp), %rax\n\t" "rolq $23, %rax\n\t" \
    "addq  -72(%rsp), %rax\n\t" "rolq $29, %rax\n\t" \
    "xorq  -64(%rsp), %rax\n\t" "rolq $31, %rax\n\t" \
    "addq  -56(%rsp), %rax\n\t" "rolq $5,  %rax\n\t" \
    "xorq  -48(%rsp), %rax\n\t" "rolq $7,  %rax\n\t" \
    "addq  -40(%rsp), %rax\n\t" "rolq $11, %rax\n\t" \
    "xorq  -32(%rsp), %rax\n\t" "rolq $13, %rax\n\t" \
    "addq  -24(%rsp), %rax\n\t" "rolq $17, %rax\n\t" \
    "xorq  -16(%rsp), %rax\n\t" "rolq $19, %rax\n\t" \
    "addq   -8(%rsp), %rax\n\t"                      \
    "xorq %rbx, %rax\n\t" "addq %rcx, %rax\n\t" "xorq %rdx, %rax\n\t" \
    "addq %rbp, %rax\n\t" "xorq %rdi, %rax\n\t" "addq %r8,  %rax\n\t" \
    "xorq %r9,  %rax\n\t" "addq %r10, %rax\n\t" "xorq %r11, %rax\n\t" \
    "addq %r12, %rax\n\t" "xorq %r13, %rax\n\t" "addq %r14, %rax\n\t" \
    "xorq %r15, %rax\n\t"

__asm__(
    ".pushsection .text, \"ax\", @progbits\n\t"

    /* ------------------------------------------------------------------------
     *  void kronos_ctx_to_sched(KronosCtx *fiber [rdi], const KronosCtx *sched [rsi])
     * ------------------------------------------------------------------------ */
    ".globl  kronos_ctx_to_sched\n\t"
    ".hidden kronos_ctx_to_sched\n\t"
    ".type   kronos_ctx_to_sched, @function\n\t"
    ".p2align 5\n\t"
    "kronos_ctx_to_sched:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "stmxcsr     56(%rdi)\n\t"
    "fnstcw      60(%rdi)\n\t"
    "movq %rsp,   0(%rdi)\n\t"
    "movq %rbp,   8(%rdi)\n\t"
    "movq %rbx,  16(%rdi)\n\t"
    "movq %r12,  24(%rdi)\n\t"
    "movq %r13,  32(%rdi)\n\t"
    "movq %r14,  40(%rdi)\n\t"
    "movq %r15,  48(%rdi)\n\t"

    "movl   56(%rdi), %eax\n\t"
    "movzwl 60(%rdi), %ecx\n\t"
    "xorl   $0x1F80,  %eax\n\t"
    "cmpl   $0x037F,  %ecx\n\t"
    "jne .Lkts_slow_fp\n\t"
    "testl  $0xFFC0,  %eax\n\t"
    "jnz .Lkts_load_mxcsr\n\t"

    ".Lkts_rest:\n\t"
    "movq  0(%rsi), %rsp\n\t"
    "movq  8(%rsi), %rbp\n\t"
    "movq 16(%rsi), %rbx\n\t"
    "movq 24(%rsi), %r12\n\t"
    "movq 32(%rsi), %r13\n\t"
    "movq 40(%rsi), %r14\n\t"
    "movq 48(%rsi), %r15\n\t"
    "ret\n\t"

    ".Lkts_slow_fp:\n\t"
    "fldcw g_default_fcw_asm(%rip)\n\t"
    "testl $0xFFC0, %eax\n\t"
    "jz .Lkts_rest\n\t"
    ".Lkts_load_mxcsr:\n\t"
    "ldmxcsr g_default_mxcsr_asm(%rip)\n\t"
    "jmp .Lkts_rest\n\t"
    ".cfi_endproc\n\t"
    ".globl  kronos_ctx_to_sched_end\n\t"
    ".hidden kronos_ctx_to_sched_end\n\t"
    "kronos_ctx_to_sched_end:\n\t"
    ".size kronos_ctx_to_sched, .-kronos_ctx_to_sched\n\t"

    /* ------------------------------------------------------------------------
     *  void kronos_ctx_exit_to_sched(const KronosCtx *sched [rdi])
     *  Interleaves rbp..r15 loads; uses fninit on dirty exit to wipe leaked x87 stack.
     * ------------------------------------------------------------------------ */
    ".globl  kronos_ctx_exit_to_sched\n\t"
    ".hidden kronos_ctx_exit_to_sched\n\t"
    ".type   kronos_ctx_exit_to_sched, @function\n\t"
    ".p2align 5\n\t"
    "kronos_ctx_exit_to_sched:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "stmxcsr -8(%rsp)\n\t"
    "fnstcw  -4(%rsp)\n\t"
    "movq  8(%rdi), %rbp\n\t"
    "movq 16(%rdi), %rbx\n\t"
    "movq 24(%rdi), %r12\n\t"
    "movq 32(%rdi), %r13\n\t"
    "movq 40(%rdi), %r14\n\t"
    "movq 48(%rdi), %r15\n\t"
    "movl   -8(%rsp), %eax\n\t"
    "movzwl -4(%rsp), %ecx\n\t"
    "xorl   $0x1F80,  %eax\n\t"
    "cmpl   $0x037F,  %ecx\n\t"
    "jne .Lkes_slow_fp\n\t"
    "testl  $0xFFC0,  %eax\n\t"
    "jnz .Lkes_load_mxcsr\n\t"

    ".Lkes_rest:\n\t"
    "movq  0(%rdi), %rsp\n\t"
    "ret\n\t"

    ".Lkes_slow_fp:\n\t"
    "fninit\n\t"
    "testl $0xFFC0, %eax\n\t"
    "jz .Lkes_rest\n\t"
    ".Lkes_load_mxcsr:\n\t"
    "ldmxcsr g_default_mxcsr_asm(%rip)\n\t"
    "jmp .Lkes_rest\n\t"
    ".cfi_endproc\n\t"
    ".globl  kronos_ctx_exit_to_sched_end\n\t"
    ".hidden kronos_ctx_exit_to_sched_end\n\t"
    "kronos_ctx_exit_to_sched_end:\n\t"
    ".size kronos_ctx_exit_to_sched, .-kronos_ctx_exit_to_sched\n\t"

    /* ------------------------------------------------------------------------
     *  void kronos_ctx_from_sched(KronosCtx *sched [rdi], const KronosCtx *fiber [rsi])
     * ------------------------------------------------------------------------ */
    ".globl  kronos_ctx_from_sched\n\t"
    ".hidden kronos_ctx_from_sched\n\t"
    ".type   kronos_ctx_from_sched, @function\n\t"
    ".p2align 5\n\t"
    "kronos_ctx_from_sched:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "movq %rsp,   0(%rdi)\n\t"
    "movq %rbp,   8(%rdi)\n\t"
    "movq %rbx,  16(%rdi)\n\t"
    "movq %r12,  24(%rdi)\n\t"
    "movq %r13,  32(%rdi)\n\t"
    "movq %r14,  40(%rdi)\n\t"
    "movq %r15,  48(%rdi)\n\t"

    "movl   56(%rsi), %eax\n\t"
    "movzwl 60(%rsi), %ecx\n\t"
    "xorl   $0x1F80,  %eax\n\t"
    "cmpl   $0x037F,  %ecx\n\t"
    "jne .Lkfs_slow_fp\n\t"
    "testl  $0xFFC0,  %eax\n\t"
    "jnz .Lkfs_load_mxcsr\n\t"

    ".Lkfs_rest:\n\t"
    "movq  0(%rsi), %rsp\n\t"
    "movq  8(%rsi), %rbp\n\t"
    "movq 16(%rsi), %rbx\n\t"
    "movq 24(%rsi), %r12\n\t"
    "movq 32(%rsi), %r13\n\t"
    "movq 40(%rsi), %r14\n\t"
    "movq 48(%rsi), %r15\n\t"
    "ret\n\t"

    ".Lkfs_slow_fp:\n\t"
    "fldcw 60(%rsi)\n\t"
    "testl $0xFFC0, %eax\n\t"
    "jz .Lkfs_rest\n\t"
    ".Lkfs_load_mxcsr:\n\t"
    "ldmxcsr 56(%rsi)\n\t"
    "jmp .Lkfs_rest\n\t"
    ".cfi_endproc\n\t"
    ".globl  kronos_ctx_from_sched_end\n\t"
    ".hidden kronos_ctx_from_sched_end\n\t"
    "kronos_ctx_from_sched_end:\n\t"
    ".size kronos_ctx_from_sched, .-kronos_ctx_from_sched\n\t"

    /* ------------------------------------------------------------------------
     *  void kronos_ctx_switch_direct(KronosCtx *from [rdi], const KronosCtx *to [rsi])
     * ------------------------------------------------------------------------ */
    ".globl  kronos_ctx_switch_direct\n\t"
    ".hidden kronos_ctx_switch_direct\n\t"
    ".type   kronos_ctx_switch_direct, @function\n\t"
    ".p2align 5\n\t"
    "kronos_ctx_switch_direct:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "stmxcsr     56(%rdi)\n\t"
    "fnstcw      60(%rdi)\n\t"
    "movq %rsp,   0(%rdi)\n\t"
    "movq %rbp,   8(%rdi)\n\t"
    "movq %rbx,  16(%rdi)\n\t"
    "movq %r12,  24(%rdi)\n\t"
    "movq %r13,  32(%rdi)\n\t"
    "movq %r14,  40(%rdi)\n\t"
    "movq %r15,  48(%rdi)\n\t"

    "movl   56(%rdi), %eax\n\t"
    "movzwl 60(%rdi), %ecx\n\t"
    "xorl   56(%rsi), %eax\n\t"
    "cmpw   60(%rsi), %cx\n\t"
    "jne .Lksd_slow_fp\n\t"
    "testl  $0xFFC0,  %eax\n\t"
    "jnz .Lksd_load_mxcsr\n\t"

    ".Lksd_rest:\n\t"
    "movq  0(%rsi), %rsp\n\t"
    "movq  8(%rsi), %rbp\n\t"
    "movq 16(%rsi), %rbx\n\t"
    "movq 24(%rsi), %r12\n\t"
    "movq 32(%rsi), %r13\n\t"
    "movq 40(%rsi), %r14\n\t"
    "movq 48(%rsi), %r15\n\t"
    "ret\n\t"

    ".Lksd_slow_fp:\n\t"
    "fldcw 60(%rsi)\n\t"
    "testl $0xFFC0, %eax\n\t"
    "jz .Lksd_rest\n\t"
    ".Lksd_load_mxcsr:\n\t"
    "ldmxcsr 56(%rsi)\n\t"
    "jmp .Lksd_rest\n\t"
    ".cfi_endproc\n\t"
    ".globl  kronos_ctx_switch_direct_end\n\t"
    ".hidden kronos_ctx_switch_direct_end\n\t"
    "kronos_ctx_switch_direct_end:\n\t"
    ".size kronos_ctx_switch_direct, .-kronos_ctx_switch_direct\n\t"

    /* ------------------------------------------------------------------------
     *  kronos_fiber_bootstrap
     * ------------------------------------------------------------------------ */
    ".globl  kronos_fiber_bootstrap\n\t"
    ".hidden kronos_fiber_bootstrap\n\t"
    ".type   kronos_fiber_bootstrap, @function\n\t"
    ".p2align 4\n\t"
    "kronos_fiber_bootstrap:\n\t"
    ".cfi_startproc\n\t"
    ".cfi_undefined rip\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "xorq %rbp, %rbp\n\t"
    "movq %r14, %rdi\n\t"
    "call kronos_fiber_entry_c\n\t"
    "ud2\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_fiber_bootstrap, .-kronos_fiber_bootstrap\n\t"

    /* ------------------------------------------------------------------------
     *  kronos_async_preempt_trampoline (stack-safe XSAVE64/XRSTOR64 + exact CFI)
     * ------------------------------------------------------------------------ */
    ".globl  kronos_async_preempt_trampoline\n\t"
    ".hidden kronos_async_preempt_trampoline\n\t"
    ".type   kronos_async_preempt_trampoline, @function\n\t"
    ".p2align 5\n\t"
    "kronos_async_preempt_trampoline:\n\t"
    ".cfi_startproc\n\t"
    ".cfi_signal_frame\n\t"
    ".cfi_def_cfa rsp, 136\n\t"
    ".cfi_offset rip, -136\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "pushfq\n\t"     ".cfi_adjust_cfa_offset 8\n\t"
    "pushq %rax\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset rax, 0\n\t"
    "pushq %rcx\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset rcx, 0\n\t"
    "pushq %rdx\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset rdx, 0\n\t"
    "pushq %rbx\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset rbx, 0\n\t"
    "pushq %rbp\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset rbp, 0\n\t"
    "pushq %rsi\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset rsi, 0\n\t"
    "pushq %rdi\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset rdi, 0\n\t"
    "pushq %r8\n\t"  ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset r8, 0\n\t"
    "pushq %r9\n\t"  ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset r9, 0\n\t"
    "pushq %r10\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset r10, 0\n\t"
    "pushq %r11\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset r11, 0\n\t"
    "pushq %r12\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset r12, 0\n\t"
    "pushq %r13\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset r13, 0\n\t"
    "pushq %r14\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset r14, 0\n\t"
    "pushq %r15\n\t" ".cfi_adjust_cfa_offset 8\n\t"  ".cfi_rel_offset r15, 0\n\t"
    "movq %rsp, %rbp\n\t"
    ".cfi_def_cfa_register rbp\n\t"
    "andq $-64, %rsp\n\t"
    "subq g_cpu_xsave_size_asm(%rip), %rsp\n\t"
    "cmpl $0, g_cpu_use_xsave_asm(%rip)\n\t"
    "je .Lkp_fxsave\n\t"
    "xorq %rax, %rax\n\t"
    "movq %rax, 512(%rsp)\n\t"
    "movq %rax, 520(%rsp)\n\t"
    "movq %rax, 528(%rsp)\n\t"
    "movq %rax, 536(%rsp)\n\t"
    "movq %rax, 544(%rsp)\n\t"
    "movq %rax, 552(%rsp)\n\t"
    "movq %rax, 560(%rsp)\n\t"
    "movq %rax, 568(%rsp)\n\t"
    "movl g_cpu_xsave_mask_lo_asm(%rip), %eax\n\t"
    "xorl %edx, %edx\n\t"
    "xsave64 (%rsp)\n\t"
    "cmpl $0, g_cpu_has_avx_asm(%rip)\n\t"
    "je .Lkp_call_yield\n\t"
    "vzeroupper\n\t"
    "jmp .Lkp_call_yield\n\t"
    ".Lkp_fxsave:\n\t"
    "fxsave64 (%rsp)\n\t"
    ".Lkp_call_yield:\n\t"
    "fninit\n\t"
    "ldmxcsr g_default_mxcsr_asm(%rip)\n\t"
    "cld\n\t"
    "call kronos_preempt_yield_c\n\t"
    "cmpl $0, g_cpu_use_xsave_asm(%rip)\n\t"
    "je .Lkp_fxrstor\n\t"
    "movl g_cpu_xsave_mask_lo_asm(%rip), %eax\n\t"
    "xorl %edx, %edx\n\t"
    "xrstor64 (%rsp)\n\t"
    "jmp .Lkp_rest_gprs\n\t"
    ".Lkp_fxrstor:\n\t"
    "fxrstor64 (%rsp)\n\t"
    ".Lkp_rest_gprs:\n\t"
    "movq %rbp, %rsp\n\t"
    ".cfi_def_cfa_register rsp\n\t"
    "popq %r15\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r15\n\t"
    "popq %r14\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r14\n\t"
    "popq %r13\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r13\n\t"
    "popq %r12\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r12\n\t"
    "popq %r11\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r11\n\t"
    "popq %r10\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r10\n\t"
    "popq %r9\n\t"  ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r9\n\t"
    "popq %r8\n\t"  ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore r8\n\t"
    "popq %rdi\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore rdi\n\t"
    "popq %rsi\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore rsi\n\t"
    "popq %rbp\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore rbp\n\t"
    "popq %rbx\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore rbx\n\t"
    "popq %rdx\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore rdx\n\t"
    "popq %rcx\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore rcx\n\t"
    "movl $0, %fs:g_tls_preempt_word@tpoff\n\t"
    "movq %fs:g_tls_worker@tpoff, %rax\n\t"
    "movl $1, " KSTR(KRONOS_WORKER_IN_PREEMPTIBLE_OFF) "(%rax)\n\t"
    "popq %rax\n\t" ".cfi_adjust_cfa_offset -8\n\t" ".cfi_restore rax\n\t"
    "popfq\n\t"     ".cfi_adjust_cfa_offset -8\n\t"
    "ret $128\n\t"
    ".cfi_endproc\n\t"
    ".globl  kronos_async_preempt_trampoline_end\n\t"
    ".hidden kronos_async_preempt_trampoline_end\n\t"
    "kronos_async_preempt_trampoline_end:\n\t"
    ".size kronos_async_preempt_trampoline, .-kronos_async_preempt_trampoline\n\t"

    /* ------------------------------------------------------------------------
     *  Boot Hardware FMA Peak Probes (12 independent accumulators, pure reg-reg)
     * ------------------------------------------------------------------------ */
    ".globl  kronos_fma_peak_avx512\n\t"
    ".hidden kronos_fma_peak_avx512\n\t"
    ".type   kronos_fma_peak_avx512, @function\n\t"
    ".p2align 5\n\t"
    "kronos_fma_peak_avx512:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "vpxord %zmm0,  %zmm0,  %zmm0\n\t"  "vpxord %zmm1,  %zmm1,  %zmm1\n\t"
    "vpxord %zmm2,  %zmm2,  %zmm2\n\t"  "vpxord %zmm3,  %zmm3,  %zmm3\n\t"
    "vpxord %zmm4,  %zmm4,  %zmm4\n\t"  "vpxord %zmm5,  %zmm5,  %zmm5\n\t"
    "vpxord %zmm6,  %zmm6,  %zmm6\n\t"  "vpxord %zmm7,  %zmm7,  %zmm7\n\t"
    "vpxord %zmm8,  %zmm8,  %zmm8\n\t"  "vpxord %zmm9,  %zmm9,  %zmm9\n\t"
    "vpxord %zmm10, %zmm10, %zmm10\n\t" "vpxord %zmm11, %zmm11, %zmm11\n\t"
    "vpxord %zmm14, %zmm14, %zmm14\n\t" "vpxord %zmm15, %zmm15, %zmm15\n\t"
    "testq %rdi, %rdi\n\t"
    "jz .Lfp512_done\n\t"
    ".p2align 5\n\t"
    ".Lfp512_loop:\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm0\n\t"  "vfmadd231ps %zmm14, %zmm15, %zmm1\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm2\n\t"  "vfmadd231ps %zmm14, %zmm15, %zmm3\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm4\n\t"  "vfmadd231ps %zmm14, %zmm15, %zmm5\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm6\n\t"  "vfmadd231ps %zmm14, %zmm15, %zmm7\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm8\n\t"  "vfmadd231ps %zmm14, %zmm15, %zmm9\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm10\n\t" "vfmadd231ps %zmm14, %zmm15, %zmm11\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm0\n\t"  "vfmadd231ps %zmm14, %zmm15, %zmm1\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm2\n\t"  "vfmadd231ps %zmm14, %zmm15, %zmm3\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm4\n\t"  "vfmadd231ps %zmm14, %zmm15, %zmm5\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm6\n\t"  "vfmadd231ps %zmm14, %zmm15, %zmm7\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm8\n\t"  "vfmadd231ps %zmm14, %zmm15, %zmm9\n\t"
    "vfmadd231ps %zmm14, %zmm15, %zmm10\n\t" "vfmadd231ps %zmm14, %zmm15, %zmm11\n\t"
    "decq %rdi\n\t"
    "jnz .Lfp512_loop\n\t"
    ".Lfp512_done:\n\t"
    "vzeroupper\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_fma_peak_avx512, .-kronos_fma_peak_avx512\n\t"

    ".globl  kronos_fma_peak_avx2\n\t"
    ".hidden kronos_fma_peak_avx2\n\t"
    ".type   kronos_fma_peak_avx2, @function\n\t"
    ".p2align 5\n\t"
    "kronos_fma_peak_avx2:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "vxorps %ymm0,  %ymm0,  %ymm0\n\t"  "vxorps %ymm1,  %ymm1,  %ymm1\n\t"
    "vxorps %ymm2,  %ymm2,  %ymm2\n\t"  "vxorps %ymm3,  %ymm3,  %ymm3\n\t"
    "vxorps %ymm4,  %ymm4,  %ymm4\n\t"  "vxorps %ymm5,  %ymm5,  %ymm5\n\t"
    "vxorps %ymm6,  %ymm6,  %ymm6\n\t"  "vxorps %ymm7,  %ymm7,  %ymm7\n\t"
    "vxorps %ymm8,  %ymm8,  %ymm8\n\t"  "vxorps %ymm9,  %ymm9,  %ymm9\n\t"
    "vxorps %ymm10, %ymm10, %ymm10\n\t" "vxorps %ymm11, %ymm11, %ymm11\n\t"
    "vxorps %ymm14, %ymm14, %ymm14\n\t" "vxorps %ymm15, %ymm15, %ymm15\n\t"
    "testq %rdi, %rdi\n\t"
    "jz .Lfp2_done\n\t"
    ".p2align 5\n\t"
    ".Lfp2_loop:\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm0\n\t"  "vfmadd231ps %ymm14, %ymm15, %ymm1\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm2\n\t"  "vfmadd231ps %ymm14, %ymm15, %ymm3\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm4\n\t"  "vfmadd231ps %ymm14, %ymm15, %ymm5\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm6\n\t"  "vfmadd231ps %ymm14, %ymm15, %ymm7\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm8\n\t"  "vfmadd231ps %ymm14, %ymm15, %ymm9\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm10\n\t" "vfmadd231ps %ymm14, %ymm15, %ymm11\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm0\n\t"  "vfmadd231ps %ymm14, %ymm15, %ymm1\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm2\n\t"  "vfmadd231ps %ymm14, %ymm15, %ymm3\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm4\n\t"  "vfmadd231ps %ymm14, %ymm15, %ymm5\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm6\n\t"  "vfmadd231ps %ymm14, %ymm15, %ymm7\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm8\n\t"  "vfmadd231ps %ymm14, %ymm15, %ymm9\n\t"
    "vfmadd231ps %ymm14, %ymm15, %ymm10\n\t" "vfmadd231ps %ymm14, %ymm15, %ymm11\n\t"
    "decq %rdi\n\t"
    "jnz .Lfp2_loop\n\t"
    ".Lfp2_done:\n\t"
    "vzeroupper\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_fma_peak_avx2, .-kronos_fma_peak_avx2\n\t"

    ".globl  kronos_fma_peak_sse2\n\t"
    ".hidden kronos_fma_peak_sse2\n\t"
    ".type   kronos_fma_peak_sse2, @function\n\t"
    ".p2align 5\n\t"
    "kronos_fma_peak_sse2:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "xorps %xmm0,  %xmm0\n\t"  "xorps %xmm1,  %xmm1\n\t"
    "xorps %xmm2,  %xmm2\n\t"  "xorps %xmm3,  %xmm3\n\t"
    "xorps %xmm4,  %xmm4\n\t"  "xorps %xmm5,  %xmm5\n\t"
    "xorps %xmm6,  %xmm6\n\t"  "xorps %xmm7,  %xmm7\n\t"
    "xorps %xmm8,  %xmm8\n\t"  "xorps %xmm9,  %xmm9\n\t"
    "xorps %xmm10, %xmm10\n\t" "xorps %xmm11, %xmm11\n\t"
    "xorps %xmm14, %xmm14\n\t" "xorps %xmm15, %xmm15\n\t"
    "testq %rdi, %rdi\n\t"
    "jz .Lfpsse_done\n\t"
    ".p2align 5\n\t"
    ".Lfpsse_loop:\n\t"
    "mulps %xmm14, %xmm0\n\t" "addps %xmm15, %xmm6\n\t"
    "mulps %xmm14, %xmm1\n\t" "addps %xmm15, %xmm7\n\t"
    "mulps %xmm14, %xmm2\n\t" "addps %xmm15, %xmm8\n\t"
    "mulps %xmm14, %xmm3\n\t" "addps %xmm15, %xmm9\n\t"
    "mulps %xmm14, %xmm4\n\t" "addps %xmm15, %xmm10\n\t"
    "mulps %xmm14, %xmm5\n\t" "addps %xmm15, %xmm11\n\t"
    "mulps %xmm14, %xmm0\n\t" "addps %xmm15, %xmm6\n\t"
    "mulps %xmm14, %xmm1\n\t" "addps %xmm15, %xmm7\n\t"
    "mulps %xmm14, %xmm2\n\t" "addps %xmm15, %xmm8\n\t"
    "mulps %xmm14, %xmm3\n\t" "addps %xmm15, %xmm9\n\t"
    "mulps %xmm14, %xmm4\n\t" "addps %xmm15, %xmm10\n\t"
    "mulps %xmm14, %xmm5\n\t" "addps %xmm15, %xmm11\n\t"
    "decq %rdi\n\t"
    "jnz .Lfpsse_loop\n\t"
    ".Lfpsse_done:\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_fma_peak_sse2, .-kronos_fma_peak_sse2\n\t"

    /* ------------------------------------------------------------------------
     *  int64_t kronos_vnni_dot_u8s8(int64_t n64 [rdi], const uint8_t *a [rsi], const int8_t *b [rdx])
     *  Uses raw EVEX bytes for vpdpbusd %zmm1, %zmm0, %zmm2 (works on all binutils)
     * ------------------------------------------------------------------------ */
    ".globl  kronos_vnni_dot_u8s8\n\t"
    ".hidden kronos_vnni_dot_u8s8\n\t"
    ".type   kronos_vnni_dot_u8s8, @function\n\t"
    ".p2align 5\n\t"
    "kronos_vnni_dot_u8s8:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "vpxord %zmm2, %zmm2, %zmm2\n\t"
    "testq %rdi, %rdi\n\t"
    "jle .Lvnni_done\n\t"
    ".p2align 4\n\t"
    ".Lvnni_loop:\n\t"
    "vmovdqu64 (%rsi), %zmm0\n\t"
    "vmovdqu64 (%rdx), %zmm1\n\t"
    ".byte 0x62, 0xf2, 0x7d, 0x48, 0x50, 0xd1\n\t" /* vpdpbusd %zmm1, %zmm0, %zmm2 */
    "addq $64, %rsi\n\t"
    "addq $64, %rdx\n\t"
    "decq %rdi\n\t"
    "jnz .Lvnni_loop\n\t"
    ".Lvnni_done:\n\t"
    "vmovdqu64 %zmm2, -64(%rsp)\n\t"
    "vzeroupper\n\t"
    "xorq %rax, %rax\n\t"
    "movl $16, %ecx\n\t"
    "leaq -64(%rsp), %r8\n\t"
    ".Lvnni_sum:\n\t"
    "movslq (%r8), %r9\n\t"
    "addq %r9, %rax\n\t"
    "addq $4, %r8\n\t"
    "decl %ecx\n\t"
    "jnz .Lvnni_sum\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_vnni_dot_u8s8, .-kronos_vnni_dot_u8s8\n\t"

    /* ------------------------------------------------------------------------
     *  kronos_preempt_canary_avx512 (32 ZMM + k0..7 + x87/MXCSR + CF+DF + 128B RZ)
     * ------------------------------------------------------------------------ */
    ".globl  kronos_preempt_canary_avx512\n\t"
    ".hidden kronos_preempt_canary_avx512\n\t"
    ".type   kronos_preempt_canary_avx512, @function\n\t"
    ".p2align 5\n\t"
    "kronos_preempt_canary_avx512:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    KRONOS_CANARY_PROLOGUE()

    "movq %rdi, %rax\n\t"
    "movabsq $0x9E3779B97F4A7C15, %r8\n\t"
    "movabsq $0xBF58476D1CE4E5B9, %r9\n\t"
    "movabsq $0x94D049BB133111EB, %r10\n\t"
    "movabsq $0xD6E8FEB86659FD93, %r11\n\t"
    "movq %rax, %rbx\n\t" "xorq %r8,  %rbx\n\t"
    "movq %rax, %rcx\n\t" "addq %r9,  %rcx\n\t"
    "movq %rax, %rdx\n\t" "xorq %r10, %rdx\n\t"
    "movq %rax, %rbp\n\t" "addq %r11, %rbp\n\t"
    "movq %rbx, %r12\n\t" "addq %rcx, %r12\n\t"
    "movq %rcx, %r13\n\t" "xorq %rdx, %r13\n\t"
    "movq %rdx, %r14\n\t" "addq %rbp, %r14\n\t"
    "movq %rbp, %r15\n\t" "xorq %rax, %r15\n\t"

    "movq %rax, -128(%rsp)\n\t" "movq %rbx, -120(%rsp)\n\t"
    "movq %rcx, -112(%rsp)\n\t" "movq %rdx, -104(%rsp)\n\t"
    "movq %rbp,  -96(%rsp)\n\t" "movq %r8,   -88(%rsp)\n\t"
    "movq %r9,   -80(%rsp)\n\t" "movq %r10,  -72(%rsp)\n\t"
    "movq %r11,  -64(%rsp)\n\t" "movq %r12,  -56(%rsp)\n\t"
    "movq %r13,  -48(%rsp)\n\t" "movq %r14,  -40(%rsp)\n\t"
    "movq %r15,  -32(%rsp)\n\t" "movq %rdi,  -24(%rsp)\n\t"
    "movq %r8,   -16(%rsp)\n\t" "movq %r9,    -8(%rsp)\n\t"

    "kmovw %eax, %k1\n\t" "kmovw %ebx, %k2\n\t" "kmovw %ecx, %k3\n\t"
    "kmovw %edx, %k4\n\t" "kmovw %r8d, %k5\n\t" "kmovw %r9d, %k6\n\t"
    "kmovw %r10d, %k7\n\t"

    "vmovdqu64 -128(%rsp), %zmm0\n\t"
    "vmovdqu64  -64(%rsp), %zmm1\n\t"
    "vpaddq %zmm0,  %zmm1,  %zmm2\n\t"  "vpxorq %zmm1,  %zmm2,  %zmm3\n\t"
    "vpaddq %zmm2,  %zmm3,  %zmm4\n\t"  "vpxorq %zmm3,  %zmm4,  %zmm5\n\t"
    "vpaddq %zmm4,  %zmm5,  %zmm6\n\t"  "vpxorq %zmm5,  %zmm6,  %zmm7\n\t"
    "vpaddq %zmm6,  %zmm7,  %zmm8\n\t"  "vpxorq %zmm7,  %zmm8,  %zmm9\n\t"
    "vpaddq %zmm8,  %zmm9,  %zmm10\n\t" "vpxorq %zmm9,  %zmm10, %zmm11\n\t"
    "vpaddq %zmm10, %zmm11, %zmm12\n\t" "vpxorq %zmm11, %zmm12, %zmm13\n\t"
    "vpaddq %zmm12, %zmm13, %zmm14\n\t" "vpxorq %zmm13, %zmm14, %zmm15\n\t"
    "vpaddq %zmm14, %zmm15, %zmm16\n\t" "vpxorq %zmm15, %zmm16, %zmm17\n\t"
    "vpaddq %zmm16, %zmm17, %zmm18\n\t" "vpxorq %zmm17, %zmm18, %zmm19\n\t"
    "vpaddq %zmm18, %zmm19, %zmm20\n\t" "vpxorq %zmm19, %zmm20, %zmm21\n\t"
    "vpaddq %zmm20, %zmm21, %zmm22\n\t" "vpxorq %zmm21, %zmm22, %zmm23\n\t"
    "vpaddq %zmm22, %zmm23, %zmm24\n\t" "vpxorq %zmm23, %zmm24, %zmm25\n\t"
    "vpaddq %zmm24, %zmm25, %zmm26\n\t" "vpxorq %zmm25, %zmm26, %zmm27\n\t"
    "vpaddq %zmm26, %zmm27, %zmm28\n\t" "vpxorq %zmm27, %zmm28, %zmm29\n\t"
    "vpaddq %zmm28, %zmm29, %zmm30\n\t" "vpxorq %zmm29, %zmm30, %zmm31\n\t"

    "testq %rsi, %rsi\n\t"
    "jz .Lca512_done\n\t"

    ".p2align 5\n\t"
    ".Lca512_loop:\n\t"
    KRONOS_CANARY_GPR_CF_DF_STEP()
    "kshiftlw $3,  %k1, %k0\n\t"
    "kshiftrw $13, %k1, %k1\n\t"
    "korw  %k0, %k1, %k1\n\t"
    "kxorw %k1, %k2, %k2\n\t" "kxorw %k2, %k3, %k3\n\t"
    "kxorw %k3, %k4, %k4\n\t" "kxorw %k4, %k5, %k5\n\t"
    "kxorw %k5, %k6, %k6\n\t" "kxorw %k6, %k7, %k7\n\t"
    "kxorw %k7, %k1, %k1\n\t" "knotw %k7, %k7\n\t"
    "vshufi64x2 $0x39, %zmm0, %zmm0, %zmm0\n\t"
    "vpaddq %zmm0,  %zmm1,  %zmm1{%k1}\n\t" "vpxorq %zmm1,  %zmm2,  %zmm2\n\t"
    "vpaddq %zmm2,  %zmm3,  %zmm3\n\t"      "vpxorq %zmm3,  %zmm4,  %zmm4\n\t"
    "vpaddq %zmm4,  %zmm5,  %zmm5\n\t"      "vpxorq %zmm5,  %zmm6,  %zmm6\n\t"
    "vpaddq %zmm6,  %zmm7,  %zmm7\n\t"      "vpxorq %zmm7,  %zmm8,  %zmm8\n\t"
    "vpaddq %zmm8,  %zmm9,  %zmm9\n\t"      "vpxorq %zmm9,  %zmm10, %zmm10\n\t"
    "vpaddq %zmm10, %zmm11, %zmm11\n\t"     "vpxorq %zmm11, %zmm12, %zmm12\n\t"
    "vpaddq %zmm12, %zmm13, %zmm13\n\t"     "vpxorq %zmm13, %zmm14, %zmm14\n\t"
    "vpaddq %zmm14, %zmm15, %zmm15\n\t"     "vpxorq %zmm15, %zmm16, %zmm16\n\t"
    "vpaddq %zmm16, %zmm17, %zmm17\n\t"     "vpxorq %zmm17, %zmm18, %zmm18\n\t"
    "vpaddq %zmm18, %zmm19, %zmm19\n\t"     "vpxorq %zmm19, %zmm20, %zmm20\n\t"
    "vpaddq %zmm20, %zmm21, %zmm21\n\t"     "vpxorq %zmm21, %zmm22, %zmm22\n\t"
    "vpaddq %zmm22, %zmm23, %zmm23\n\t"     "vpxorq %zmm23, %zmm24, %zmm24\n\t"
    "vpaddq %zmm24, %zmm25, %zmm25\n\t"     "vpxorq %zmm25, %zmm26, %zmm26\n\t"
    "vpaddq %zmm26, %zmm27, %zmm27\n\t"     "vpxorq %zmm27, %zmm28, %zmm28\n\t"
    "vpaddq %zmm28, %zmm29, %zmm29\n\t"     "vpxorq %zmm29, %zmm30, %zmm30\n\t"
    "vpaddq %zmm30, %zmm31, %zmm31\n\t"     "vpaddq %zmm31, %zmm0,  %zmm0{%k2}\n\t"
    "decq %rsi\n\t"
    "jnz .Lca512_loop\n\t"

    ".Lca512_done:\n\t"
    KRONOS_CANARY_FOLD_REDZONE_AND_ALL_GPRS()
    "vpxorq %zmm1,  %zmm0, %zmm0\n\t" "vpaddq %zmm2,  %zmm0, %zmm0\n\t"
    "vpxorq %zmm3,  %zmm0, %zmm0\n\t" "vpaddq %zmm4,  %zmm0, %zmm0\n\t"
    "vpxorq %zmm5,  %zmm0, %zmm0\n\t" "vpaddq %zmm6,  %zmm0, %zmm0\n\t"
    "vpxorq %zmm7,  %zmm0, %zmm0\n\t" "vpaddq %zmm8,  %zmm0, %zmm0\n\t"
    "vpxorq %zmm9,  %zmm0, %zmm0\n\t" "vpaddq %zmm10, %zmm0, %zmm0\n\t"
    "vpxorq %zmm11, %zmm0, %zmm0\n\t" "vpaddq %zmm12, %zmm0, %zmm0\n\t"
    "vpxorq %zmm13, %zmm0, %zmm0\n\t" "vpaddq %zmm14, %zmm0, %zmm0\n\t"
    "vpxorq %zmm15, %zmm0, %zmm0\n\t" "vpaddq %zmm16, %zmm0, %zmm0\n\t"
    "vpxorq %zmm17, %zmm0, %zmm0\n\t" "vpaddq %zmm18, %zmm0, %zmm0\n\t"
    "vpxorq %zmm19, %zmm0, %zmm0\n\t" "vpaddq %zmm20, %zmm0, %zmm0\n\t"
    "vpxorq %zmm21, %zmm0, %zmm0\n\t" "vpaddq %zmm22, %zmm0, %zmm0\n\t"
    "vpxorq %zmm23, %zmm0, %zmm0\n\t" "vpaddq %zmm24, %zmm0, %zmm0\n\t"
    "vpxorq %zmm25, %zmm0, %zmm0\n\t" "vpaddq %zmm26, %zmm0, %zmm0\n\t"
    "vpxorq %zmm27, %zmm0, %zmm0\n\t" "vpaddq %zmm28, %zmm0, %zmm0\n\t"
    "vpxorq %zmm29, %zmm0, %zmm0\n\t" "vpaddq %zmm30, %zmm0, %zmm0\n\t"
    "vpxorq %zmm31, %zmm0, %zmm0\n\t"
    "vmovdqu64 %zmm0, -128(%rsp)\n\t"
    "vzeroupper\n\t"

    "xorq -128(%rsp), %rax\n\t" "rolq $11, %rax\n\t"
    "addq -120(%rsp), %rax\n\t" "rolq $13, %rax\n\t"
    "xorq -112(%rsp), %rax\n\t" "rolq $17, %rax\n\t"
    "addq -104(%rsp), %rax\n\t" "rolq $19, %rax\n\t"
    "xorq  -96(%rsp), %rax\n\t" "rolq $23, %rax\n\t"
    "addq  -88(%rsp), %rax\n\t" "rolq $29, %rax\n\t"
    "xorq  -80(%rsp), %rax\n\t" "rolq $31, %rax\n\t"
    "addq  -72(%rsp), %rax\n\t"
    "kmovw %k1, %r8d\n\t" "xorq %r8, %rax\n\t" "rolq $5, %rax\n\t"
    "kmovw %k2, %r8d\n\t" "addq %r8, %rax\n\t" "rolq $5, %rax\n\t"
    "kmovw %k3, %r8d\n\t" "xorq %r8, %rax\n\t" "rolq $5, %rax\n\t"
    "kmovw %k4, %r8d\n\t" "addq %r8, %rax\n\t" "rolq $5, %rax\n\t"
    "kmovw %k5, %r8d\n\t" "xorq %r8, %rax\n\t" "rolq $5, %rax\n\t"
    "kmovw %k6, %r8d\n\t" "addq %r8, %rax\n\t" "rolq $5, %rax\n\t"
    "kmovw %k7, %r8d\n\t" "xorq %r8, %rax\n\t"
    KRONOS_CANARY_EPILOGUE()
    ".cfi_endproc\n\t"
    ".size kronos_preempt_canary_avx512, .-kronos_preempt_canary_avx512\n\t"

    /* ------------------------------------------------------------------------
     *  kronos_preempt_canary_avx2 (16 YMM + x87/MXCSR + CF+DF + 128B RZ)
     * ------------------------------------------------------------------------ */
    ".globl  kronos_preempt_canary_avx2\n\t"
    ".hidden kronos_preempt_canary_avx2\n\t"
    ".type   kronos_preempt_canary_avx2, @function\n\t"
    ".p2align 5\n\t"
    "kronos_preempt_canary_avx2:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    KRONOS_CANARY_PROLOGUE()

    "movq %rdi, %rax\n\t"
    "movabsq $0x9E3779B97F4A7C15, %r8\n\t"
    "movabsq $0xBF58476D1CE4E5B9, %r9\n\t"
    "movabsq $0x94D049BB133111EB, %r10\n\t"
    "movabsq $0xD6E8FEB86659FD93, %r11\n\t"
    "movq %rax, %rbx\n\t" "xorq %r8,  %rbx\n\t"
    "movq %rax, %rcx\n\t" "addq %r9,  %rcx\n\t"
    "movq %rax, %rdx\n\t" "xorq %r10, %rdx\n\t"
    "movq %rax, %rbp\n\t" "addq %r11, %rbp\n\t"
    "movq %rbx, %r12\n\t" "addq %rcx, %r12\n\t"
    "movq %rcx, %r13\n\t" "xorq %rdx, %r13\n\t"
    "movq %rdx, %r14\n\t" "addq %rbp, %r14\n\t"
    "movq %rbp, %r15\n\t" "xorq %rax, %r15\n\t"

    "movq %rax, -128(%rsp)\n\t" "movq %rbx, -120(%rsp)\n\t"
    "movq %rcx, -112(%rsp)\n\t" "movq %rdx, -104(%rsp)\n\t"
    "movq %rbp,  -96(%rsp)\n\t" "movq %r8,   -88(%rsp)\n\t"
    "movq %r9,   -80(%rsp)\n\t" "movq %r10,  -72(%rsp)\n\t"
    "movq %r11,  -64(%rsp)\n\t" "movq %r12,  -56(%rsp)\n\t"
    "movq %r13,  -48(%rsp)\n\t" "movq %r14,  -40(%rsp)\n\t"
    "movq %r15,  -32(%rsp)\n\t" "movq %rdi,  -24(%rsp)\n\t"
    "movq %r8,   -16(%rsp)\n\t" "movq %r9,    -8(%rsp)\n\t"

    "vmovdqu -128(%rsp), %ymm0\n\t"
    "vmovdqu  -96(%rsp), %ymm1\n\t"
    "vmovdqu  -64(%rsp), %ymm2\n\t"
    "vmovdqu  -32(%rsp), %ymm3\n\t"
    "vpaddq %ymm0,  %ymm1,  %ymm4\n\t"  "vpxor %ymm1,  %ymm2,  %ymm5\n\t"
    "vpaddq %ymm2,  %ymm3,  %ymm6\n\t"  "vpxor %ymm3,  %ymm4,  %ymm7\n\t"
    "vpaddq %ymm4,  %ymm5,  %ymm8\n\t"  "vpxor %ymm5,  %ymm6,  %ymm9\n\t"
    "vpaddq %ymm6,  %ymm7,  %ymm10\n\t" "vpxor %ymm7,  %ymm8,  %ymm11\n\t"
    "vpaddq %ymm8,  %ymm9,  %ymm12\n\t" "vpxor %ymm9,  %ymm10, %ymm13\n\t"
    "vpaddq %ymm10, %ymm11, %ymm14\n\t" "vpxor %ymm11, %ymm12, %ymm15\n\t"

    "testq %rsi, %rsi\n\t"
    "jz .Lca2_done\n\t"

    ".p2align 5\n\t"
    ".Lca2_loop:\n\t"
    KRONOS_CANARY_GPR_CF_DF_STEP()
    "vpermq $0x39, %ymm0, %ymm0\n\t"
    "vpaddq %ymm0,  %ymm1,  %ymm1\n\t"  "vpxor  %ymm1,  %ymm2,  %ymm2\n\t"
    "vpaddq %ymm2,  %ymm3,  %ymm3\n\t"  "vpxor  %ymm3,  %ymm4,  %ymm4\n\t"
    "vpaddq %ymm4,  %ymm5,  %ymm5\n\t"  "vpxor  %ymm5,  %ymm6,  %ymm6\n\t"
    "vpaddq %ymm6,  %ymm7,  %ymm7\n\t"  "vpxor  %ymm7,  %ymm8,  %ymm8\n\t"
    "vpaddq %ymm8,  %ymm9,  %ymm9\n\t"  "vpxor  %ymm9,  %ymm10, %ymm10\n\t"
    "vpaddq %ymm10, %ymm11, %ymm11\n\t" "vpxor  %ymm11, %ymm12, %ymm12\n\t"
    "vpaddq %ymm12, %ymm13, %ymm13\n\t" "vpxor  %ymm13, %ymm14, %ymm14\n\t"
    "vpaddq %ymm14, %ymm15, %ymm15\n\t" "vpaddq %ymm15, %ymm0,  %ymm0\n\t"
    "decq %rsi\n\t"
    "jnz .Lca2_loop\n\t"

    ".Lca2_done:\n\t"
    KRONOS_CANARY_FOLD_REDZONE_AND_ALL_GPRS()
    "vpxor %ymm1,  %ymm0, %ymm0\n\t" "vpaddq %ymm2,  %ymm0, %ymm0\n\t"
    "vpxor %ymm3,  %ymm0, %ymm0\n\t" "vpaddq %ymm4,  %ymm0, %ymm0\n\t"
    "vpxor %ymm5,  %ymm0, %ymm0\n\t" "vpaddq %ymm6,  %ymm0, %ymm0\n\t"
    "vpxor %ymm7,  %ymm0, %ymm0\n\t" "vpaddq %ymm8,  %ymm0, %ymm0\n\t"
    "vpxor %ymm9,  %ymm0, %ymm0\n\t" "vpaddq %ymm10, %ymm0, %ymm0\n\t"
    "vpxor %ymm11, %ymm0, %ymm0\n\t" "vpaddq %ymm12, %ymm0, %ymm0\n\t"
    "vpxor %ymm13, %ymm0, %ymm0\n\t" "vpaddq %ymm14, %ymm0, %ymm0\n\t"
    "vpxor %ymm15, %ymm0, %ymm0\n\t"
    "vmovdqu %ymm0, -96(%rsp)\n\t"
    "vzeroupper\n\t"

    "xorq  -96(%rsp), %rax\n\t" "rolq $13, %rax\n\t"
    "addq  -88(%rsp), %rax\n\t" "rolq $17, %rax\n\t"
    "xorq  -80(%rsp), %rax\n\t" "rolq $23, %rax\n\t"
    "addq  -72(%rsp), %rax\n\t"
    KRONOS_CANARY_EPILOGUE()
    ".cfi_endproc\n\t"
    ".size kronos_preempt_canary_avx2, .-kronos_preempt_canary_avx2\n\t"

    /* ------------------------------------------------------------------------
     *  kronos_preempt_canary_sse2 (16 XMM + x87/MXCSR + CF+DF + 128B RZ)
     * ------------------------------------------------------------------------ */
    ".globl  kronos_preempt_canary_sse2\n\t"
    ".hidden kronos_preempt_canary_sse2\n\t"
    ".type   kronos_preempt_canary_sse2, @function\n\t"
    ".p2align 5\n\t"
    "kronos_preempt_canary_sse2:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    KRONOS_CANARY_PROLOGUE()

    "movq %rdi, %rax\n\t"
    "movabsq $0x9E3779B97F4A7C15, %r8\n\t"
    "movabsq $0xBF58476D1CE4E5B9, %r9\n\t"
    "movabsq $0x94D049BB133111EB, %r10\n\t"
    "movabsq $0xD6E8FEB86659FD93, %r11\n\t"
    "movq %rax, %rbx\n\t" "xorq %r8,  %rbx\n\t"
    "movq %rax, %rcx\n\t" "addq %r9,  %rcx\n\t"
    "movq %rax, %rdx\n\t" "xorq %r10, %rdx\n\t"
    "movq %rax, %rbp\n\t" "addq %r11, %rbp\n\t"
    "movq %rbx, %r12\n\t" "addq %rcx, %r12\n\t"
    "movq %rcx, %r13\n\t" "xorq %rdx, %r13\n\t"
    "movq %rdx, %r14\n\t" "addq %rbp, %r14\n\t"
    "movq %rbp, %r15\n\t" "xorq %rax, %r15\n\t"

    "movq %rax, -128(%rsp)\n\t" "movq %rbx, -120(%rsp)\n\t"
    "movq %rcx, -112(%rsp)\n\t" "movq %rdx, -104(%rsp)\n\t"
    "movq %rbp,  -96(%rsp)\n\t" "movq %r8,   -88(%rsp)\n\t"
    "movq %r9,   -80(%rsp)\n\t" "movq %r10,  -72(%rsp)\n\t"
    "movq %r11,  -64(%rsp)\n\t" "movq %r12,  -56(%rsp)\n\t"
    "movq %r13,  -48(%rsp)\n\t" "movq %r14,  -40(%rsp)\n\t"
    "movq %r15,  -32(%rsp)\n\t" "movq %rdi,  -24(%rsp)\n\t"
    "movq %r8,   -16(%rsp)\n\t" "movq %r9,    -8(%rsp)\n\t"

    "movdqu -128(%rsp), %xmm0\n\t"  "movdqu -112(%rsp), %xmm1\n\t"
    "movdqu  -96(%rsp), %xmm2\n\t"  "movdqu  -80(%rsp), %xmm3\n\t"
    "movdqu  -64(%rsp), %xmm4\n\t"  "movdqu  -48(%rsp), %xmm5\n\t"
    "movdqu  -32(%rsp), %xmm6\n\t"  "movdqu  -16(%rsp), %xmm7\n\t"
    "movdqa %xmm0, %xmm8\n\t"  "paddq %xmm1, %xmm8\n\t"
    "movdqa %xmm2, %xmm9\n\t"  "pxor  %xmm3, %xmm9\n\t"
    "movdqa %xmm4, %xmm10\n\t" "paddq %xmm5, %xmm10\n\t"
    "movdqa %xmm6, %xmm11\n\t" "pxor  %xmm7, %xmm11\n\t"
    "movdqa %xmm8, %xmm12\n\t" "paddq %xmm9, %xmm12\n\t"
    "movdqa %xmm10, %xmm13\n\t" "pxor %xmm11, %xmm13\n\t"
    "movdqa %xmm12, %xmm14\n\t" "paddq %xmm13, %xmm14\n\t"
    "movdqa %xmm14, %xmm15\n\t" "pxor %xmm0, %xmm15\n\t"

    "testq %rsi, %rsi\n\t"
    "jz .Lcs2_done\n\t"

    ".p2align 5\n\t"
    ".Lcs2_loop:\n\t"
    KRONOS_CANARY_GPR_CF_DF_STEP()
    "pshufd $0x4E, %xmm0, %xmm0\n\t"
    "paddq %xmm0,  %xmm1\n\t"  "pxor  %xmm1,  %xmm2\n\t"
    "paddq %xmm2,  %xmm3\n\t"  "pxor  %xmm3,  %xmm4\n\t"
    "paddq %xmm4,  %xmm5\n\t"  "pxor  %xmm5,  %xmm6\n\t"
    "paddq %xmm6,  %xmm7\n\t"  "pxor  %xmm7,  %xmm8\n\t"
    "paddq %xmm8,  %xmm9\n\t"  "pxor  %xmm9,  %xmm10\n\t"
    "paddq %xmm10, %xmm11\n\t" "pxor  %xmm11, %xmm12\n\t"
    "paddq %xmm12, %xmm13\n\t" "pxor  %xmm13, %xmm14\n\t"
    "paddq %xmm14, %xmm15\n\t" "paddq %xmm15, %xmm0\n\t"
    "decq %rsi\n\t"
    "jnz .Lcs2_loop\n\t"

    ".Lcs2_done:\n\t"
    KRONOS_CANARY_FOLD_REDZONE_AND_ALL_GPRS()
    "pxor %xmm1,  %xmm0\n\t" "paddq %xmm2,  %xmm0\n\t"
    "pxor %xmm3,  %xmm0\n\t" "paddq %xmm4,  %xmm0\n\t"
    "pxor %xmm5,  %xmm0\n\t" "paddq %xmm6,  %xmm0\n\t"
    "pxor %xmm7,  %xmm0\n\t" "paddq %xmm8,  %xmm0\n\t"
    "pxor %xmm9,  %xmm0\n\t" "paddq %xmm10, %xmm0\n\t"
    "pxor %xmm11, %xmm0\n\t" "paddq %xmm12, %xmm0\n\t"
    "pxor %xmm13, %xmm0\n\t" "paddq %xmm14, %xmm0\n\t"
    "pxor %xmm15, %xmm0\n\t"
    "movdqu %xmm0, -96(%rsp)\n\t"

    "xorq  -96(%rsp), %rax\n\t" "rolq $13, %rax\n\t"
    "addq  -88(%rsp), %rax\n\t"
    KRONOS_CANARY_EPILOGUE()
    ".cfi_endproc\n\t"
    ".size kronos_preempt_canary_sse2, .-kronos_preempt_canary_sse2\n\t"
    ".popsection\n\t"
    ".pushsection .note.GNU-stack,\"\",@progbits\n\t"
    ".popsection\n\t"
);

/* ============================================================================
 *  5. M:N WORK-STEALING KERNEL, PINNED NUMA QUEUES, PER-WORKER TIMERS & LAPIC
 * ============================================================================ */

typedef void (*KronosFiberFn)(void *arg);

typedef enum {
    POST_NONE = 0,
    POST_REQUEUE,
    POST_DIRECT_REQUEUE,
    POST_PREEMPT_REQUEUE,
    POST_UNLOCK_CSPIN,
    POST_UNLOCK_CSPIN_WAKE_LAPIC,
    POST_FIBER_DONE
} PostSwitchAction;

/* Exactly 128B (2 cache lines -> Slab Class 5) with pinned_core, errno & preempt_count */
struct KronosFiber {
    KronosCtx      ctx;                /* +0..63  */
    void          *stack_base;         /* +64..71 */
    KronosFiberFn  fn;                 /* +72..79 */
    void          *arg;                /* +80..87 */
    uint64_t       id;                 /* +88..95 */
    KronosFiber   *next_waiter;        /* +96..103 */
    uint32_t       stack_size;         /* +104..107 */
    uint32_t       preempt_hits;       /* +108..111 */
    int            saved_errno;        /* +112..115 */
    uint32_t       saved_preempt;      /* +116..119 */
    int16_t        pinned_core;        /* +120..121 (-1 = unpinned, >=0 = strictly pinned) */
    uint8_t        preemptible;        /* +122 */
    uint8_t        from_batch;         /* +123 */
    uint8_t        uses_since_trim;    /* +124 */
    uint8_t        _pad[3];            /* +125..127 */
};

_Static_assert(sizeof(KronosFiber) == 128, "KronosFiber must fit exactly in 128B slab class");

typedef struct FiberStackChunk {
    struct FiberStackChunk *next;
    void                   *base;
    size_t                  total_bytes;
} FiberStackChunk;

/* Strict C11 Chase-Lev Work-Stealing Deque */
typedef struct {
    CACHELINE_ALIGN _Atomic int64_t top;
    CACHELINE_ALIGN _Atomic int64_t bottom;
    CACHELINE_ALIGN _Atomic(KronosFiber *) buf[KRONOS_DEQUE_CAP];
} ChaseLevDeque;

typedef enum {
    STEAL_EMPTY = 0,
    STEAL_OK,
    STEAL_ABORT
} StealStatus;

static void deque_init(ChaseLevDeque *d) {
    atomic_init(&d->top, 0);
    atomic_init(&d->bottom, 0);
}

static inline int64_t deque_approx_size(const ChaseLevDeque *d) {
    int64_t t = atomic_load_explicit(&d->top, memory_order_acquire);
    int64_t b = atomic_load_explicit(&d->bottom, memory_order_acquire);
    return (b > t) ? (b - t) : 0;
}

static bool deque_push(ChaseLevDeque *d, KronosFiber *f) {
    int64_t b = atomic_load_explicit(&d->bottom, memory_order_relaxed);
    int64_t t = atomic_load_explicit(&d->top, memory_order_acquire);
    if (UNLIKELY(b - t >= KRONOS_DEQUE_CAP - 1)) return false;
    atomic_store_explicit(&d->buf[b & KRONOS_DEQUE_MASK], f, memory_order_relaxed);
    atomic_store_explicit(&d->bottom, b + 1, memory_order_release);
    return true;
}

static KronosFiber *deque_pop(ChaseLevDeque *d) {
    int64_t b0 = atomic_load_explicit(&d->bottom, memory_order_relaxed);
    int64_t t0 = atomic_load_explicit(&d->top, memory_order_relaxed);
    if (b0 <= t0) return NULL;

    int64_t b = b0 - 1;
    atomic_store_explicit(&d->bottom, b, memory_order_relaxed);
    atomic_thread_fence(memory_order_seq_cst);
    int64_t t = atomic_load_explicit(&d->top, memory_order_relaxed);
    KronosFiber *f = NULL;
    if (t <= b) {
        f = atomic_load_explicit(&d->buf[b & KRONOS_DEQUE_MASK], memory_order_relaxed);
        if (t == b) {
            if (!atomic_compare_exchange_strong_explicit(
                    &d->top, &t, t + 1,
                    memory_order_seq_cst, memory_order_relaxed)) {
                f = NULL;
            }
            atomic_store_explicit(&d->bottom, b + 1, memory_order_relaxed);
        }
    } else {
        atomic_store_explicit(&d->bottom, b + 1, memory_order_relaxed);
    }
    return f;
}

static StealStatus deque_steal_ex(ChaseLevDeque *d, KronosFiber **out) {
    int64_t t = atomic_load_explicit(&d->top, memory_order_acquire);
    atomic_thread_fence(memory_order_seq_cst);
    int64_t b = atomic_load_explicit(&d->bottom, memory_order_acquire);
    if (t < b) {
        KronosFiber *f = atomic_load_explicit(&d->buf[t & KRONOS_DEQUE_MASK], memory_order_relaxed);
        if (atomic_compare_exchange_strong_explicit(
                &d->top, &t, t + 1,
                memory_order_seq_cst, memory_order_relaxed)) {
            *out = f;
            return STEAL_OK;
        }
        return STEAL_ABORT;
    }
    return STEAL_EMPTY;
}

typedef struct {
    _Atomic uint64_t seq;
    KronosFiber     *item;
} FiberSlot;

/* Cache-line-strided stealable MPMC WorkerInbox */
typedef struct {
    CACHELINE_ALIGN _Atomic uint64_t tail;
    CACHELINE_ALIGN _Atomic uint64_t head;
    CACHELINE_ALIGN FiberSlot        slots[KRONOS_INBOX_CAP];
} WorkerInbox;

static inline uint64_t inbox_slot_idx(uint64_t pos) {
    uint64_t p = pos & KRONOS_INBOX_MASK;
    return ((p & 63ULL) << 2) | ((p >> 6) & 3ULL);
}

static void inbox_init(WorkerInbox *in) {
    atomic_init(&in->tail, 0);
    atomic_init(&in->head, 0);
    for (uint64_t i = 0; i < KRONOS_INBOX_CAP; i++) {
        uint64_t idx = inbox_slot_idx(i);
        atomic_init(&in->slots[idx].seq, i);
        in->slots[idx].item = NULL;
    }
}

static bool inbox_push(WorkerInbox *in, KronosFiber *f) {
    kronos_preempt_disable();
    uint64_t pos = atomic_load_explicit(&in->tail, memory_order_relaxed);
    for (;;) {
        FiberSlot *slot = &in->slots[inbox_slot_idx(pos)];
        uint64_t seq    = atomic_load_explicit(&slot->seq, memory_order_acquire);
        intptr_t diff   = (intptr_t)seq - (intptr_t)pos;
        if (diff == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &in->tail, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                slot->item = f;
                atomic_store_explicit(&slot->seq, pos + 1, memory_order_release);
                kronos_preempt_enable();
                return true;
            }
        } else if (diff < 0) {
            kronos_preempt_enable();
            return false;
        } else {
            pos = atomic_load_explicit(&in->tail, memory_order_relaxed);
        }
    }
}

static KronosFiber *inbox_pop_nopreempt(WorkerInbox *in) {
    uint64_t pos = atomic_load_explicit(&in->head, memory_order_relaxed);
    for (;;) {
        FiberSlot *slot = &in->slots[inbox_slot_idx(pos)];
        uint64_t seq    = atomic_load_explicit(&slot->seq, memory_order_acquire);
        intptr_t diff   = (intptr_t)seq - (intptr_t)(pos + 1);
        if (diff == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &in->head, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                KronosFiber *f = slot->item;
                atomic_store_explicit(&slot->seq, pos + KRONOS_INBOX_CAP, memory_order_release);
                return f;
            }
        } else if (diff < 0) {
            return NULL;
        } else {
            pos = atomic_load_explicit(&in->head, memory_order_relaxed);
        }
    }
}

static KronosFiber *inbox_pop(WorkerInbox *in) {
    kronos_preempt_disable();
    KronosFiber *f = inbox_pop_nopreempt(in);
    kronos_preempt_enable();
    return f;
}

static inline bool inbox_has_item(const WorkerInbox *in) {
    uint64_t h = atomic_load_explicit(&in->head, memory_order_acquire);
    uint64_t t = atomic_load_explicit(&in->tail, memory_order_acquire);
    return (t > h);
}

/* Non-stealable per-worker pinned MPSC queue (guarantees 100% NUMA/L3-CCD affinity) */
typedef struct {
    CACHELINE_ALIGN KronosCompactSpin lock;
    _Atomic uint32_t                  count;
    KronosFiber                      *head;
    KronosFiber                      *tail;
} PinnedQueue;

static void pinned_q_init(PinnedQueue *q) {
    cspin_init(&q->lock);
    atomic_init(&q->count, 0u);
    q->head = q->tail = NULL;
}

static void pinned_q_push_nopreempt(PinnedQueue *q, KronosFiber *f) {
    f->next_waiter = NULL;
    cspin_lock(&q->lock);
    if (q->tail) { q->tail->next_waiter = f; q->tail = f; }
    else         { q->head = q->tail = f; }
    atomic_fetch_add_explicit(&q->count, 1u, memory_order_release);
    cspin_unlock(&q->lock);
}

static void pinned_q_push(PinnedQueue *q, KronosFiber *f) {
    kronos_preempt_disable();
    pinned_q_push_nopreempt(q, f);
    kronos_preempt_enable();
}

static KronosFiber *pinned_q_pop_nopreempt(PinnedQueue *q) {
    if (LIKELY(atomic_load_explicit(&q->count, memory_order_acquire) == 0u)) return NULL;
    cspin_lock(&q->lock);
    KronosFiber *f = q->head;
    if (f) {
        q->head = f->next_waiter;
        if (!q->head) q->tail = NULL;
        f->next_waiter = NULL;
        atomic_fetch_sub_explicit(&q->count, 1u, memory_order_relaxed);
    }
    cspin_unlock(&q->lock);
    return f;
}

static inline bool pinned_q_has_item(const PinnedQueue *q) {
    return atomic_load_explicit(&q->count, memory_order_acquire) > 0u;
}

/* Underflow-Free Lock-Free Strided Vyukov MPMC GlobalRunQ */
typedef struct {
    CACHELINE_ALIGN _Atomic uint64_t tail;
    CACHELINE_ALIGN _Atomic uint64_t head;
    CACHELINE_ALIGN FiberSlot        slots[KRONOS_GLOBAL_RQ_CAP];
    CACHELINE_ALIGN KronosCompactSpin ovf_lock;
    _Atomic uint32_t                  ovf_count;
    KronosFiber                      *ovf_head;
    KronosFiber                      *ovf_tail;
} GlobalRunQ;

static inline uint64_t grq_slot_idx(uint64_t pos) {
    uint64_t p = pos & KRONOS_GLOBAL_RQ_MASK;
    return ((p & 1023ULL) << 2) | ((p >> 10) & 3ULL);
}

static void grq_init(GlobalRunQ *q) {
    atomic_init(&q->tail, 0);
    atomic_init(&q->head, 0);
    for (uint64_t i = 0; i < KRONOS_GLOBAL_RQ_CAP; i++) {
        uint64_t idx = grq_slot_idx(i);
        atomic_init(&q->slots[idx].seq, i);
        q->slots[idx].item = NULL;
    }
    cspin_init(&q->ovf_lock);
    atomic_init(&q->ovf_count, 0);
    q->ovf_head = q->ovf_tail = NULL;
}

static inline bool grq_has_item(const GlobalRunQ *q) {
    uint64_t h = atomic_load_explicit(&q->head, memory_order_acquire);
    uint64_t t = atomic_load_explicit(&q->tail, memory_order_acquire);
    if (t > h) return true;
    return atomic_load_explicit(&q->ovf_count, memory_order_acquire) > 0u;
}

static uint32_t grq_push_nopreempt(GlobalRunQ *q, KronosFiber *f) {
    uint64_t pos = atomic_load_explicit(&q->tail, memory_order_relaxed);
    for (;;) {
        FiberSlot *slot = &q->slots[grq_slot_idx(pos)];
        uint64_t seq    = atomic_load_explicit(&slot->seq, memory_order_acquire);
        intptr_t diff   = (intptr_t)seq - (intptr_t)pos;
        if (diff == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &q->tail, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                slot->item = f;
                atomic_store_explicit(&slot->seq, pos + 1, memory_order_release);
                uint64_t h = atomic_load_explicit(&q->head, memory_order_relaxed);
                return (pos > h) ? (uint32_t)(pos - h) : 0u;
            }
        } else if (diff < 0) {
            break;
        } else {
            pos = atomic_load_explicit(&q->tail, memory_order_relaxed);
        }
    }

    f->next_waiter = NULL;
    cspin_lock(&q->ovf_lock);
    if (q->ovf_tail) { q->ovf_tail->next_waiter = f; q->ovf_tail = f; }
    else             { q->ovf_head = q->ovf_tail = f; }
    uint32_t prev = atomic_fetch_add_explicit(&q->ovf_count, 1u, memory_order_release);
    cspin_unlock(&q->ovf_lock);
    return prev + KRONOS_GLOBAL_RQ_CAP;
}

static uint32_t grq_push(GlobalRunQ *q, KronosFiber *f) {
    kronos_preempt_disable();
    uint32_t r = grq_push_nopreempt(q, f);
    kronos_preempt_enable();
    return r;
}

static KronosFiber *grq_pop_nopreempt(GlobalRunQ *q) {
    if (!grq_has_item(q)) return NULL;

    uint64_t pos = atomic_load_explicit(&q->head, memory_order_relaxed);
    for (;;) {
        FiberSlot *slot = &q->slots[grq_slot_idx(pos)];
        uint64_t seq    = atomic_load_explicit(&slot->seq, memory_order_acquire);
        intptr_t diff   = (intptr_t)seq - (intptr_t)(pos + 1);
        if (diff == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &q->head, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                KronosFiber *f = slot->item;
                atomic_store_explicit(&slot->seq, pos + KRONOS_GLOBAL_RQ_CAP, memory_order_release);
                return f;
            }
        } else if (diff < 0) {
            break;
        } else {
            pos = atomic_load_explicit(&q->head, memory_order_relaxed);
        }
    }

    if (atomic_load_explicit(&q->ovf_count, memory_order_acquire) > 0u) {
        cspin_lock(&q->ovf_lock);
        KronosFiber *f = q->ovf_head;
        if (f) {
            q->ovf_head = f->next_waiter;
            if (!q->ovf_head) q->ovf_tail = NULL;
            f->next_waiter = NULL;
            atomic_fetch_sub_explicit(&q->ovf_count, 1u, memory_order_relaxed);
        }
        cspin_unlock(&q->ovf_lock);
        return f;
    }
    return NULL;
}

static KronosFiber *grq_pop(GlobalRunQ *q) {
    kronos_preempt_disable();
    KronosFiber *f = grq_pop_nopreempt(q);
    kronos_preempt_enable();
    return f;
}

typedef struct {
    uint64_t     deadline_ns;
    KronosFiber *fiber;
} KronosTimerEntry;

/* Per-worker local min-heap timer queue (zero global timer_lock contention at 128 cores!) */
typedef struct {
    CACHELINE_ALIGN KronosCompactSpin lock;
    _Atomic uint32_t                  count;
    _Atomic uint64_t                  earliest_ns;
    uint32_t                          cap;
    KronosTimerEntry                 *entries;
} WorkerTimerHeap;

static void worker_timer_init(WorkerTimerHeap *th) {
    cspin_init(&th->lock);
    atomic_init(&th->count, 0u);
    atomic_init(&th->earliest_ns, UINT64_MAX);
    th->cap     = KRONOS_INIT_TIMER_CAP;
    th->entries = (KronosTimerEntry *)calloc(th->cap, sizeof(KronosTimerEntry));
    if (!th->entries) { perror("calloc worker timers"); exit(1); }
}

struct KronosWorker {
    CACHELINE_ALIGN KronosCtx    sched_ctx;        /* +0..63 */
    int                          id;               /* +64 */
    int                          hw_cpu;           /* +68 */
    int                          numa_node;        /* +72 */
    int                          l3_dom;           /* +76 */
    pid_t                        linux_tid;        /* +80 */
    int                          running_preempt;  /* +84 */
    pthread_t                    thread;           /* +88..95 */

    /* Isolated LAPIC-polled cache line at exact offset +128! */
    CACHELINE_ALIGN _Atomic int  in_preemptible;   /* +128 */
    _Atomic int                  ready;            /* +132 */
    _Atomic uint32_t            *preempt_word_ptr; /* +136 */
    _Atomic uint64_t             slice_start_tick; /* +144 */
    uint8_t                      is_ecore;         /* +152 */
    uint8_t                      _lapic_pad[39];

    /* Per-worker targeted futex park word (0=RUN, 1=SLEEP, 2=NOTIFIED) */
    CACHELINE_ALIGN _Atomic uint32_t park_state;

    /* Worker-private hot execution state */
    CACHELINE_ALIGN _Atomic(KronosFiber *) current_fiber;
    PostSwitchAction             post_action;
    KronosFiber                 *post_fiber;
    KronosCompactSpin           *post_cspin;
    KronosFiber                 *local_pool_head;
    int                          local_pool_count;
    uint64_t                     rng_state;

    _Atomic uint64_t             stat_executed;
    _Atomic uint64_t             stat_ctx_switches;
    _Atomic uint64_t             stat_direct_switches;
    _Atomic uint64_t             stat_steals;
    _Atomic uint64_t             stat_preemptions;
    _Atomic uint64_t             stat_deferred_preempts;
    _Atomic uint64_t             stat_futex_sleeps;

    PinnedQueue                  pinned;
    WorkerTimerHeap              timers;
    WorkerInbox                  inbox;
    ChaseLevDeque                deque;
};

_Static_assert(offsetof(KronosWorker, in_preemptible) == KRONOS_WORKER_IN_PREEMPTIBLE_OFF,
               "KronosWorker.in_preemptible offset must match asm trampoline constant");

/* Cache-line padded seqlock range entry (writers serialized by code_range_lock) */
typedef struct {
    CACHELINE_ALIGN _Atomic uint32_t  seq;
    _Atomic int                       active;
    _Atomic uintptr_t                 lo;
    _Atomic uintptr_t                 hi;
} KronosCodeRange;

typedef struct {
    CACHELINE_ALIGN int  num_cores;
    int                  preempt_supported;
    int                  is_static_binary;
    pid_t                host_pid;
    int                  static_code_slots;
    uint64_t             quantum_us;
    uint64_t             quantum_ticks;
    KronosWorker        *workers;
    const char          *code_src;
    pthread_t            lapic_thread;
    struct sigaction     old_sigusr1;
    sigset_t             old_main_sigmask;

    /* Hot LAPIC / lifecycle control isolated from read-only constants */
    CACHELINE_ALIGN _Atomic int      lapic_stop;
    _Atomic int                      lapic_asleep;
    _Atomic uint32_t                 lapic_wake_epoch;
    _Atomic uint64_t                 lapic_sleep_until_ns;
    _Atomic int                      active_preemptible_fibers;
    _Atomic uint32_t                 active_timers;
    _Atomic int                      shutdown;
    _Atomic int                      code_range_hwm;
    _Atomic int                      altstack_failures;

    KronosTicketLock code_range_lock;
    KronosCodeRange  code_ranges[KRONOS_MAX_CODE_RANGES];
    GlobalRunQ       global_rq;

    CACHELINE_ALIGN KronosTicketLock pool_lock;
    KronosFiber                     *pool_head;
    _Atomic int                      pool_count;
    _Atomic int                      batch_stacks_allocated;
    FiberStackChunk                 *stack_chunks;

    /* Strict W^X PROT_NONE JIT 4 KiB page recycler */
    CACHELINE_ALIGN KronosTicketLock jit_cache_lock;
    void                            *jit_page_cache[KRONOS_JIT_CACHE_MAX];
    int                              jit_page_count;

    CACHELINE_ALIGN _Atomic int      sleeping_workers;
    CACHELINE_ALIGN _Atomic int      spinning_workers;
    CACHELINE_ALIGN _Atomic uint64_t next_fiber_id;
    CACHELINE_ALIGN _Atomic uint64_t rr_cursor;
    CACHELINE_ALIGN _Atomic uint64_t total_preempt_ipis;
    CACHELINE_ALIGN _Atomic uint64_t timer_wakeups;
    CACHELINE_ALIGN _Atomic uint64_t timer_jitter_sum_ns;
    CACHELINE_ALIGN _Atomic uint64_t timer_jitter_max_ns;
    CACHELINE_ALIGN _Atomic int64_t  in_flight_fibers;
} KronosKernel;

static KronosKernel g_kern;

static inline void kronos_poke_lapic(void) {
    atomic_fetch_add_explicit(&g_kern.lapic_wake_epoch, 1u, memory_order_seq_cst);
    sys_futex(&g_kern.lapic_wake_epoch, FUTEX_WAKE_PRIVATE, 1u, NULL);
}

static inline void worker_set_preempt_accounting(KronosWorker *w, int is_p) {
    if (UNLIKELY(w->running_preempt != is_p)) {
        w->running_preempt = is_p;
        if (is_p) {
            atomic_fetch_add_explicit(&g_kern.active_preemptible_fibers, 1, memory_order_seq_cst);
            if (UNLIKELY(atomic_load_explicit(&g_kern.lapic_asleep, memory_order_acquire) != 0)) {
                kronos_poke_lapic();
            }
        } else {
            atomic_fetch_sub_explicit(&g_kern.active_preemptible_fibers, 1, memory_order_release);
        }
    }
}

static void code_range_update_hwm(int slot) {
    int want = slot + 1;
    int cur  = atomic_load_explicit(&g_kern.code_range_hwm, memory_order_relaxed);
    while (want > cur) {
        if (atomic_compare_exchange_weak_explicit(
                &g_kern.code_range_hwm, &cur, want,
                memory_order_release, memory_order_relaxed)) {
            break;
        }
    }
}

static int kronos_register_code_range_at(int slot, uintptr_t lo, uintptr_t hi) {
    if (slot < 0 || slot >= KRONOS_MAX_CODE_RANGES || lo >= hi) return -1;
    KronosCodeRange *r = &g_kern.code_ranges[slot];
    uint32_t s = atomic_load_explicit(&r->seq, memory_order_relaxed);
    atomic_store_explicit(&r->seq, s + 1u, memory_order_release);
    atomic_thread_fence(memory_order_acq_rel);
    atomic_store_explicit(&r->lo, lo, memory_order_relaxed);
    atomic_store_explicit(&r->hi, hi, memory_order_relaxed);
    atomic_store_explicit(&r->active, 1, memory_order_relaxed);
    atomic_store_explicit(&r->seq, s + 2u, memory_order_release);
    code_range_update_hwm(slot);
    return slot;
}

/* Writer-locked seqlock registration (eliminates writer-writer seq corruption!) */
static int kronos_register_code_range(uintptr_t lo, uintptr_t hi) {
    if (lo >= hi) return -1;
    int start = (g_kern.static_code_slots > 0) ? g_kern.static_code_slots : 1;
    kronos_preempt_disable();
    tlock_lock(&g_kern.code_range_lock);
    int found = -1;
    for (int i = start; i < KRONOS_MAX_CODE_RANGES; i++) {
        KronosCodeRange *r = &g_kern.code_ranges[i];
        if (atomic_load_explicit(&r->active, memory_order_relaxed) == 0) {
            uint32_t s = atomic_load_explicit(&r->seq, memory_order_relaxed);
            atomic_store_explicit(&r->seq, s + 1u, memory_order_release);
            atomic_thread_fence(memory_order_acq_rel);
            atomic_store_explicit(&r->lo, lo, memory_order_relaxed);
            atomic_store_explicit(&r->hi, hi, memory_order_relaxed);
            atomic_store_explicit(&r->active, 1, memory_order_relaxed);
            atomic_store_explicit(&r->seq, s + 2u, memory_order_release);
            code_range_update_hwm(i);
            found = i;
            break;
        }
    }
    tlock_unlock(&g_kern.code_range_lock);
    kronos_preempt_enable();
    return found;
}

static void kronos_unregister_code_range(int slot) {
    int start = (g_kern.static_code_slots > 0) ? g_kern.static_code_slots : 1;
    if (slot < start || slot >= KRONOS_MAX_CODE_RANGES) return;
    kronos_preempt_disable();
    tlock_lock(&g_kern.code_range_lock);
    KronosCodeRange *r = &g_kern.code_ranges[slot];
    if (atomic_load_explicit(&r->active, memory_order_relaxed) != 0) {
        uint32_t s = atomic_load_explicit(&r->seq, memory_order_relaxed);
        atomic_store_explicit(&r->seq, s + 1u, memory_order_release);
        atomic_thread_fence(memory_order_acq_rel);
        atomic_store_explicit(&r->lo, 0, memory_order_relaxed);
        atomic_store_explicit(&r->hi, 0, memory_order_relaxed);
        atomic_store_explicit(&r->active, 0, memory_order_relaxed);
        atomic_store_explicit(&r->seq, s + 2u, memory_order_release);
    }
    tlock_unlock(&g_kern.code_range_lock);
    kronos_preempt_enable();
}

typedef struct {
    uintptr_t probe;
    int       saw_shared_libc;
} DlPhdrCtx;

static int dl_phdr_code_cb(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    DlPhdrCtx *ctx = (DlPhdrCtx *)data;
    if (info->dlpi_name && strstr(info->dlpi_name, "libc") != NULL) {
        ctx->saw_shared_libc = 1;
    }

    int contains_probe = 0;
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
        if (ph->p_type == PT_LOAD && (ph->p_flags & PF_X)) {
            uintptr_t lo = (uintptr_t)info->dlpi_addr + (uintptr_t)ph->p_vaddr;
            uintptr_t hi = lo + (uintptr_t)ph->p_memsz;
            if (ctx->probe >= lo && ctx->probe < hi) { contains_probe = 1; break; }
        }
    }
    if (!contains_probe) return 0;

    for (int i = 0; i < info->dlpi_phnum && g_kern.static_code_slots < 16; i++) {
        const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
        if (ph->p_type == PT_LOAD && (ph->p_flags & PF_X) && ph->p_memsz > 0) {
            uintptr_t lo = (uintptr_t)info->dlpi_addr + (uintptr_t)ph->p_vaddr;
            uintptr_t hi = lo + (uintptr_t)ph->p_memsz;
            kronos_register_code_range_at(g_kern.static_code_slots++, lo, hi);
        }
    }
    g_kern.code_src = "ELF-PT_LOAD";
    return 0; /* Keep iterating so we also detect shared libc.so! */
}

static void kronos_discover_code_bounds(void) {
    g_kern.code_src          = "none";
    g_kern.static_code_slots = 0;
    g_kern.is_static_binary  = 0;

    DlPhdrCtx ctx = { (uintptr_t)&kronos_ctx_to_sched, 0 };
    (void)dl_iterate_phdr(dl_phdr_code_cb, &ctx);
    if (g_kern.static_code_slots > 0) {
        if (!ctx.saw_shared_libc) g_kern.is_static_binary = 1;
        return;
    }

    FILE *fp = fopen("/proc/self/maps", "r");
    if (!fp) return;
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long lo = 0, hi = 0;
        char perms[8] = {0};
        if (sscanf(line, "%llx-%llx %7s", &lo, &hi, perms) == 3) {
            if (perms[2] == 'x' && ctx.probe >= (uintptr_t)lo && ctx.probe < (uintptr_t)hi) {
                kronos_register_code_range_at(g_kern.static_code_slots++, (uintptr_t)lo, (uintptr_t)hi);
                g_kern.code_src = "/proc/self/maps";
            }
            if (strstr(line, "libc") != NULL) ctx.saw_shared_libc = 1;
        }
        if (strchr(line, '\n') == NULL) {
            int ch;
            while ((ch = fgetc(fp)) != '\n' && ch != EOF) {}
        }
    }
    fclose(fp);
    if (!ctx.saw_shared_libc) g_kern.is_static_binary = 1;
}

static inline bool kronos_wake_worker(int target) {
    if (LIKELY(g_has_membar)) atomic_signal_fence(memory_order_seq_cst);
    else                      atomic_thread_fence(memory_order_seq_cst);

    KronosWorker *w = &g_kern.workers[target];
    if (UNLIKELY(atomic_load_explicit(&w->park_state, memory_order_acquire) == 1u)) {
        if (atomic_exchange_explicit(&w->park_state, 2u, memory_order_acq_rel) == 1u) {
            sys_futex(&w->park_state, FUTEX_WAKE_PRIVATE, 1u, NULL);
            return true;
        }
    }
    return false;
}

static void kronos_wake_any(int n) {
    if (n <= 0) return;
    if (LIKELY(g_has_membar)) atomic_signal_fence(memory_order_seq_cst);
    else                      atomic_thread_fence(memory_order_seq_cst);

    if (UNLIKELY(atomic_load_explicit(&g_kern.sleeping_workers, memory_order_acquire) > 0)) {
        int P = g_kern.num_cores;
        int start = (int)(atomic_fetch_add_explicit(&g_kern.rr_cursor, 1, memory_order_relaxed) % (uint64_t)P);
        int woken = 0;
        for (int k = 0; k < P && woken < n; k++) {
            KronosWorker *w = &g_kern.workers[(start + k) % P];
            if (atomic_load_explicit(&w->park_state, memory_order_acquire) == 1u) {
                if (atomic_exchange_explicit(&w->park_state, 2u, memory_order_acq_rel) == 1u) {
                    sys_futex(&w->park_state, FUTEX_WAKE_PRIVATE, 1u, NULL);
                    woken++;
                }
            }
        }
    }
}

static inline void kronos_wake_one(void) {
    kronos_wake_any(1);
}

static void fiber_destroy_raw(KronosFiber *f) {
    if (!f->from_batch && f->stack_base) {
        (void)munmap((uint8_t *)f->stack_base - KRONOS_PAGE_SZ,
                     KRONOS_PAGE_SZ + (size_t)f->stack_size);
    }
    kronos_free(f, sizeof(KronosFiber));
}

/* Capped batch stack allocator (prevents 128-core boot over-allocation!) */
static KronosFiber *fiber_alloc_stack_batch(void) {
    size_t unit  = KRONOS_PAGE_SZ + KRONOS_FIBER_STACK_SZ;
    size_t total = unit * KRONOS_STACK_BATCH;
    void *base   = mmap(NULL, total, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) { perror("mmap fiber batch"); exit(1); }

    for (int i = 0; i < KRONOS_STACK_BATCH; i++) {
        uint8_t *slot = (uint8_t *)base + (size_t)i * unit;
        if (mprotect(slot, KRONOS_PAGE_SZ, PROT_NONE) != 0) {
            perror("mprotect guard page");
            exit(1);
        }
    }

    FiberStackChunk *sc = (FiberStackChunk *)kronos_alloc(sizeof(FiberStackChunk));
    sc->base        = base;
    sc->total_bytes = total;

    KronosFiber *first = NULL;
    KronosFiber *extra_head = NULL;
    for (int i = 0; i < KRONOS_STACK_BATCH; i++) {
        uint8_t *slot  = (uint8_t *)base + (size_t)i * unit;
        KronosFiber *f = (KronosFiber *)kronos_alloc(sizeof(KronosFiber));
        memset(f, 0, sizeof(*f));
        f->stack_base  = slot + KRONOS_PAGE_SZ;
        f->stack_size  = KRONOS_FIBER_STACK_SZ;
        f->from_batch  = 1;
        f->pinned_core = -1;
        if (i == 0) {
            first = f;
        } else {
            f->next_waiter = extra_head;
            extra_head = f;
        }
    }

    kronos_preempt_disable();
    tlock_lock(&g_kern.pool_lock);
    sc->next            = g_kern.stack_chunks;
    g_kern.stack_chunks = sc;
    while (extra_head) {
        KronosFiber *next = extra_head->next_waiter;
        extra_head->next_waiter = g_kern.pool_head;
        g_kern.pool_head = extra_head;
        atomic_fetch_add_explicit(&g_kern.pool_count, 1, memory_order_relaxed);
        extra_head = next;
    }
    tlock_unlock(&g_kern.pool_lock);
    kronos_preempt_enable();

    return first;
}

static KronosFiber *fiber_alloc_single_stack(void) {
    size_t total = KRONOS_PAGE_SZ + KRONOS_FIBER_STACK_SZ;
    void *base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) { perror("mmap fiber stack"); exit(1); }
    if (mprotect(base, KRONOS_PAGE_SZ, PROT_NONE) != 0) {
        perror("mprotect single guard page");
        exit(1);
    }
    KronosFiber *f = (KronosFiber *)kronos_alloc(sizeof(KronosFiber));
    memset(f, 0, sizeof(*f));
    f->stack_base  = (uint8_t *)base + KRONOS_PAGE_SZ;
    f->stack_size  = KRONOS_FIBER_STACK_SZ;
    f->from_batch  = 0;
    f->pinned_core = -1;
    return f;
}

static KronosFiber *fiber_acquire(void) {
    KronosFiber *f = NULL;
    kronos_preempt_disable();
    KronosWorker *w = kronos_self();
    if (LIKELY(w != NULL && w->local_pool_head != NULL)) {
        f = w->local_pool_head;
        w->local_pool_head = f->next_waiter;
        w->local_pool_count--;
        kronos_preempt_enable();
        f->next_waiter  = NULL;
        f->preempt_hits = 0;
        return f;
    }

    tlock_lock(&g_kern.pool_lock);
    if (g_kern.pool_head) {
        f = g_kern.pool_head;
        g_kern.pool_head = f->next_waiter;
        atomic_fetch_sub_explicit(&g_kern.pool_count, 1, memory_order_relaxed);
        if (w) {
            int want = KRONOS_LOCAL_POOL_MAX / 2;
            while (w->local_pool_count < want && g_kern.pool_head) {
                KronosFiber *extra = g_kern.pool_head;
                g_kern.pool_head = extra->next_waiter;
                atomic_fetch_sub_explicit(&g_kern.pool_count, 1, memory_order_relaxed);
                extra->next_waiter = w->local_pool_head;
                w->local_pool_head = extra;
                w->local_pool_count++;
            }
        }
    }
    tlock_unlock(&g_kern.pool_lock);
    kronos_preempt_enable();

    if (f) {
        f->next_waiter  = NULL;
        f->preempt_hits = 0;
        return f;
    }

    int cur_batch = atomic_load_explicit(&g_kern.batch_stacks_allocated, memory_order_relaxed);
    while (cur_batch + KRONOS_STACK_BATCH <= KRONOS_FIBER_POOL_MAX) {
        if (atomic_compare_exchange_weak_explicit(
                &g_kern.batch_stacks_allocated, &cur_batch, cur_batch + KRONOS_STACK_BATCH,
                memory_order_acq_rel, memory_order_relaxed)) {
            return fiber_alloc_stack_batch();
        }
    }
    return fiber_alloc_single_stack();
}

static void fiber_release(KronosWorker *w, KronosFiber *f) {
    if (UNLIKELY(!f->from_batch &&
                 atomic_load_explicit(&g_kern.pool_count, memory_order_relaxed) >= KRONOS_FIBER_POOL_MAX / 2)) {
        fiber_destroy_raw(f);
        return;
    }

    int local_cap = (g_kern.num_cores <= 16) ? KRONOS_LOCAL_POOL_MAX : 8;
    if (LIKELY(w != NULL && w->local_pool_count < local_cap)) {
        f->next_waiter = w->local_pool_head;
        w->local_pool_head = f;
        w->local_pool_count++;
        return;
    }

    KronosFiber *spill_head = f;
    KronosFiber *spill_tail = f;
    if (w) {
        int keep = local_cap / 2;
        while (w->local_pool_count > keep && w->local_pool_head) {
            KronosFiber *x = w->local_pool_head;
            w->local_pool_head = x->next_waiter;
            w->local_pool_count--;
            spill_tail->next_waiter = x;
            spill_tail = x;
        }
    }
    spill_tail->next_waiter = NULL;

    /* Split spill list cleanly into pool vs destroy; lazy MADV_DONTNEED ONLY on batch stacks! */
    int pcnt = atomic_load_explicit(&g_kern.pool_count, memory_order_relaxed);
    KronosFiber *to_pool    = NULL;
    KronosFiber *to_destroy = NULL;
    while (spill_head) {
        KronosFiber *next = spill_head->next_waiter;
        if (!spill_head->from_batch && pcnt >= KRONOS_FIBER_POOL_MAX) {
            spill_head->next_waiter = to_destroy;
            to_destroy = spill_head;
        } else {
            if (spill_head->from_batch && pcnt > (3 * KRONOS_FIBER_POOL_MAX) / 4 &&
                spill_head->uses_since_trim >= 4u) {
                (void)madvise(spill_head->stack_base,
                              (size_t)spill_head->stack_size - 16384u, MADV_DONTNEED);
                spill_head->uses_since_trim = 0u;
            }
            spill_head->next_waiter = to_pool;
            to_pool = spill_head;
            pcnt++;
        }
        spill_head = next;
    }

    if (to_pool) {
        tlock_lock(&g_kern.pool_lock);
        while (to_pool) {
            KronosFiber *next = to_pool->next_waiter;
            to_pool->next_waiter = g_kern.pool_head;
            g_kern.pool_head = to_pool;
            atomic_fetch_add_explicit(&g_kern.pool_count, 1, memory_order_relaxed);
            to_pool = next;
        }
        tlock_unlock(&g_kern.pool_lock);
    }

    while (to_destroy) {
        KronosFiber *next = to_destroy->next_waiter;
        fiber_destroy_raw(to_destroy);
        to_destroy = next;
    }
}

/* Affinity-preserving wakeup/enqueue helper (respects f->pinned_core everywhere!) */
static void kronos_requeue_fiber(KronosFiber *f) {
    if (UNLIKELY(f->pinned_core >= 0)) {
        int dst = (int)f->pinned_core % g_kern.num_cores;
        pinned_q_push(&g_kern.workers[dst].pinned, f);
        (void)kronos_wake_worker(dst);
        return;
    }
    (void)grq_push(&g_kern.global_rq, f);
    kronos_wake_one();
}

static void kronos_enqueue_fiber(KronosFiber *f) {
    if (UNLIKELY(f->pinned_core >= 0)) {
        int dst = (int)f->pinned_core % g_kern.num_cores;
        pinned_q_push(&g_kern.workers[dst].pinned, f);
        (void)kronos_wake_worker(dst);
        return;
    }

    kronos_preempt_disable();
    KronosWorker *w = kronos_self();
    if (w) {
        if (!deque_push(&w->deque, f)) {
            (void)grq_push_nopreempt(&g_kern.global_rq, f);
        }
        kronos_preempt_enable();
        kronos_wake_one();
        return;
    }
    kronos_preempt_enable();

    uint64_t idx = atomic_fetch_add_explicit(&g_kern.rr_cursor, 1, memory_order_relaxed);
    int target   = (int)(idx % (uint64_t)g_kern.num_cores);
    if (inbox_push(&g_kern.workers[target].inbox, f)) {
        if (!kronos_wake_worker(target)) {
            kronos_wake_one();
        }
    } else {
        (void)grq_push(&g_kern.global_rq, f);
        kronos_wake_one();
    }
}

static KronosFiber *kronos_prepare_fiber(KronosFiberFn fn, void *arg,
                                         int preemptible, int pinned_core) {
    KronosFiber *f = fiber_acquire();
    f->fn            = fn;
    f->arg           = arg;
    f->id            = atomic_fetch_add_explicit(&g_kern.next_fiber_id, 1, memory_order_relaxed);
    f->preemptible   = (preemptible && g_kern.preempt_supported) ? 1u : 0u;
    f->preempt_hits  = 0;
    f->saved_errno   = 0;
    f->saved_preempt = 1u;
    f->pinned_core   = (int16_t)((pinned_core >= 0 && pinned_core < g_kern.num_cores) ? pinned_core : -1);
    if (f->uses_since_trim < 255u) f->uses_since_trim++;
    f->next_waiter   = NULL;

    uintptr_t sp = (((uintptr_t)f->stack_base + (uintptr_t)f->stack_size) & ~15ULL) - 16ULL;
    sp -= 8ULL;
    uint64_t boot = (uint64_t)(uintptr_t)&kronos_fiber_bootstrap;
    memcpy((void *)sp, &boot, sizeof(boot));

    memset(&f->ctx, 0, sizeof(f->ctx));
    f->ctx.rsp   = (uint64_t)sp;
    f->ctx.r14   = (uint64_t)(uintptr_t)f;
    f->ctx.mxcsr = 0x1F80u;
    f->ctx.fcw   = 0x037Fu;

    atomic_fetch_add_explicit(&g_kern.in_flight_fibers, 1, memory_order_relaxed);
    return f;
}

static void kronos_spawn_ex(KronosFiberFn fn, void *arg, int preemptible) {
    KronosFiber *f = kronos_prepare_fiber(fn, arg, preemptible, -1);
    kronos_enqueue_fiber(f);
}

static void kronos_spawn(KronosFiberFn fn, void *arg) {
    kronos_spawn_ex(fn, arg, 1);
}

static void kronos_spawn_batch(KronosFiberFn fn, void *args_base, size_t arg_stride,
                               int count, int preemptible) {
    if (count <= 0) return;
    int P = g_kern.num_cores;
    int used_grq = 0;
    for (int i = 0; i < count; i++) {
        void *a = (uint8_t *)args_base + (size_t)i * arg_stride;
        KronosFiber *f = kronos_prepare_fiber(fn, a, preemptible, -1);
        int target = i % P;
        if (!inbox_push(&g_kern.workers[target].inbox, f)) {
            (void)grq_push(&g_kern.global_rq, f);
            used_grq = 1;
        }
    }
    int wake_lim = (count < P) ? count : P;
    int missed   = 0;
    for (int i = 0; i < wake_lim; i++) {
        if (!kronos_wake_worker(i)) missed++;
    }
    if (used_grq || missed > 0) kronos_wake_any(wake_lim);
}

/* Strictly core-pinned batch spawn (zero cross-NUMA/CCD stealing for GEMM & first-touch!) */
static void kronos_spawn_batch_pinned(KronosFiberFn fn, void *args_base, size_t arg_stride,
                                      int count, int preemptible) {
    if (count <= 0) return;
    int P = g_kern.num_cores;
    for (int i = 0; i < count; i++) {
        void *a = (uint8_t *)args_base + (size_t)i * arg_stride;
        int target = i % P;
        KronosFiber *f = kronos_prepare_fiber(fn, a, preemptible, target);
        pinned_q_push(&g_kern.workers[target].pinned, f);
    }
    int wake_lim = (count < P) ? count : P;
    for (int i = 0; i < wake_lim; i++) {
        (void)kronos_wake_worker(i);
    }
}

static uint32_t kronos_current_fiber_preempt_hits(void) {
    kronos_preempt_disable();
    KronosWorker *w = kronos_self();
    KronosFiber *f = w ? atomic_load_explicit(&w->current_fiber, memory_order_relaxed) : NULL;
    uint32_t hits = f ? f->preempt_hits : 0;
    kronos_preempt_enable();
    return hits;
}

static int kronos_current_core_id(void) {
    kronos_preempt_disable();
    KronosWorker *w = kronos_self();
    int id = w ? w->id : -1;
    kronos_preempt_enable();
    return id;
}

/* Runs immediately upon resuming any fiber to finish deferred requeue from 1-hop switch */
static inline void kronos_finish_prev_switch(KronosWorker *w) {
    if (UNLIKELY(w->post_action == POST_DIRECT_REQUEUE)) {
        KronosFiber *prev = w->post_fiber;
        w->post_action = POST_NONE;
        w->post_fiber  = NULL;
        if (UNLIKELY(prev->pinned_core >= 0)) {
            int dst = (int)prev->pinned_core % g_kern.num_cores;
            pinned_q_push_nopreempt(&g_kern.workers[dst].pinned, prev);
            (void)kronos_wake_worker(dst);
        } else {
            (void)grq_push_nopreempt(&g_kern.global_rq, prev);
            kronos_wake_one();
        }
    }
}

static inline void kronos_resume_fiber_epilogue(KronosFiber *f) {
    KronosWorker *w = kronos_self();
    kronos_finish_prev_switch(w);
    KRONOS_ERRNO = f->saved_errno;
    if (f->preemptible) atomic_store_explicit(&w->in_preemptible, 1, memory_order_release);
    kronos_preempt_enable();
}

static void kronos_yield(void) {
    kronos_preempt_disable();
    KronosWorker *w = kronos_self();
    KronosFiber *f = w ? atomic_load_explicit(&w->current_fiber, memory_order_relaxed) : NULL;
    if (UNLIKELY(!f)) {
        kronos_preempt_enable();
        sched_yield();
        return;
    }

    uint32_t saved_p = kronos_preempt_read_word() & KRONOS_PREEMPT_MASK;
    atomic_store_explicit(&w->in_preemptible, 0, memory_order_relaxed);
    /* Clear NEED_RESCHED_BIT once committed to yield (prevents spurious deferred_preempt) */
    kronos_preempt_set_word(saved_p);
    f->saved_errno   = KRONOS_ERRNO;
    f->saved_preempt = saved_p;

    /* Zero-redundant-RMW pops for fast 1-hop direct fiber-to-fiber handoff */
    KronosFiber *next = pinned_q_pop_nopreempt(&w->pinned);
    if (!next) next = grq_pop_nopreempt(&g_kern.global_rq);
    if (!next) next = inbox_pop_nopreempt(&w->inbox);
    if (!next) next = deque_pop(&w->deque);

    if (next != NULL) {
        w->post_action = POST_DIRECT_REQUEUE;
        w->post_fiber  = f;
        atomic_store_explicit(&w->current_fiber, next, memory_order_relaxed);
        worker_set_preempt_accounting(w, next->preemptible);
        kronos_preempt_set_word(next->saved_preempt);
        if (next->preemptible) {
            atomic_store_explicit(&w->slice_start_tick, kronos_quantum_clock_now(), memory_order_relaxed);
        }
        stat_inc(&w->stat_ctx_switches);
        stat_inc(&w->stat_direct_switches);
        kronos_ctx_switch_direct(&f->ctx, &next->ctx);
    } else {
        w->post_action = POST_REQUEUE;
        w->post_fiber  = f;
        kronos_ctx_to_sched(&f->ctx, &w->sched_ctx);
    }

    kronos_resume_fiber_epilogue(f);
}

/* Per-worker unlocked-resize binary min-heap timer push (zero global timer_lock!) */
static bool worker_timer_push_unlocked_resize(WorkerTimerHeap *th, uint64_t deadline, KronosFiber *f) {
    for (;;) {
        cspin_lock(&th->lock);
        uint32_t cnt = atomic_load_explicit(&th->count, memory_order_relaxed);
        if (LIKELY(cnt < th->cap)) {
            uint32_t idx = cnt;
            while (idx > 0) {
                uint32_t parent = (idx - 1u) >> 1;
                if (th->entries[parent].deadline_ns <= deadline) break;
                th->entries[idx] = th->entries[parent];
                idx = parent;
            }
            th->entries[idx].deadline_ns = deadline;
            th->entries[idx].fiber       = f;
            atomic_store_explicit(&th->count, cnt + 1u, memory_order_release);
            atomic_store_explicit(&th->earliest_ns, th->entries[0].deadline_ns, memory_order_release);
            uint32_t prev_glob = atomic_fetch_add_explicit(&g_kern.active_timers, 1u, memory_order_seq_cst);
            uint64_t lapic_until = atomic_load_explicit(&g_kern.lapic_sleep_until_ns, memory_order_acquire);
            /* Returns WITH th->lock HELD; scheduler unlocks it first and then pokes LAPIC! */
            return (prev_glob == 0u || deadline < lapic_until);
        }
        uint32_t ncap = (th->cap == 0) ? KRONOS_INIT_TIMER_CAP : (th->cap * 2u);
        cspin_unlock(&th->lock);

        KronosTimerEntry *nt = (KronosTimerEntry *)malloc((size_t)ncap * sizeof(KronosTimerEntry));
        if (UNLIKELY(!nt)) { perror("malloc worker timers"); exit(1); }

        KronosTimerEntry *old = NULL;
        cspin_lock(&th->lock);
        if (th->cap < ncap) {
            uint32_t cur_cnt = atomic_load_explicit(&th->count, memory_order_relaxed);
            if (cur_cnt > 0 && th->entries) {
                memcpy(nt, th->entries, (size_t)cur_cnt * sizeof(KronosTimerEntry));
            }
            old         = th->entries;
            th->entries = nt;
            th->cap     = ncap;
            nt          = NULL;
        }
        cspin_unlock(&th->lock);
        if (old) free(old);
        if (nt)  free(nt);
    }
}

static KronosFiber *worker_timer_pop_locked(WorkerTimerHeap *th, uint64_t *out_deadline) {
    uint32_t cnt = atomic_load_explicit(&th->count, memory_order_relaxed);
    if (cnt == 0) return NULL;
    if (out_deadline) *out_deadline = th->entries[0].deadline_ns;
    KronosFiber *res = th->entries[0].fiber;
    cnt--;
    if (cnt > 0) {
        KronosTimerEntry last = th->entries[cnt];
        uint32_t idx = 0;
        for (;;) {
            uint32_t l = idx * 2u + 1u;
            if (l >= cnt) break;
            uint32_t r  = l + 1u;
            uint32_t sm = (r < cnt && th->entries[r].deadline_ns < th->entries[l].deadline_ns) ? r : l;
            if (last.deadline_ns <= th->entries[sm].deadline_ns) break;
            th->entries[idx] = th->entries[sm];
            idx = sm;
        }
        th->entries[idx] = last;
        atomic_store_explicit(&th->earliest_ns, th->entries[0].deadline_ns, memory_order_release);
    } else {
        atomic_store_explicit(&th->earliest_ns, UINT64_MAX, memory_order_release);
    }
    atomic_store_explicit(&th->count, cnt, memory_order_release);
    atomic_fetch_sub_explicit(&g_kern.active_timers, 1u, memory_order_release);
    return res;
}

static void kronos_sleep_ns(uint64_t ns) {
    if (ns == 0) { kronos_yield(); return; }
    kronos_preempt_disable();
    KronosWorker *w = kronos_self();
    KronosFiber *f = w ? atomic_load_explicit(&w->current_fiber, memory_order_relaxed) : NULL;
    if (UNLIKELY(!f)) {
        kronos_preempt_enable();
        sleep_ns_eintr(ns);
        return;
    }

    uint64_t now      = monotonic_ns();
    uint64_t max_dl   = UINT64_MAX - 1ULL; /* Never collide with UINT64_MAX sentinel! */
    uint64_t deadline = (ns > max_dl - now) ? max_dl : (now + ns);

    f->saved_errno   = KRONOS_ERRNO;
    f->saved_preempt = kronos_preempt_read_word() & KRONOS_PREEMPT_MASK;

    bool poke_lapic = worker_timer_push_unlocked_resize(&w->timers, deadline, f);
    atomic_store_explicit(&w->in_preemptible, 0, memory_order_relaxed);
    w->post_action = poke_lapic ? POST_UNLOCK_CSPIN_WAKE_LAPIC : POST_UNLOCK_CSPIN;
    w->post_cspin  = &w->timers.lock;
    kronos_ctx_to_sched(&f->ctx, &w->sched_ctx);

    kronos_resume_fiber_epilogue(f);
}

/* Double-yield-free & underflow-safe deferred preemption */
void kronos_deferred_preempt_c(void) {
    kronos_preempt_disable();
    uint32_t pw = kronos_preempt_read_word();
    if (UNLIKELY(pw == 0u)) {
        kronos_preempt_set_word(0u);
        return;
    }
    if (pw != (KRONOS_NEED_RESCHED_BIT | 1u)) {
        kronos_preempt_enable_noresched();
        return;
    }

    for (;;) {
        kronos_preempt_set_word(1u);
        KronosWorker *w = kronos_self();
        KronosFiber *f = w ? atomic_load_explicit(&w->current_fiber, memory_order_relaxed) : NULL;
        if (f && f->preemptible) {
            f->saved_errno   = KRONOS_ERRNO;
            f->saved_preempt = 1u;
            atomic_store_explicit(&w->in_preemptible, 0, memory_order_relaxed);
            stat_inc(&w->stat_preemptions);
            stat_inc(&w->stat_deferred_preempts);
            f->preempt_hits++;
            w->post_action = POST_PREEMPT_REQUEUE;
            w->post_fiber  = f;
            kronos_ctx_to_sched(&f->ctx, &w->sched_ctx);

            w = kronos_self();
            kronos_finish_prev_switch(w);
            KRONOS_ERRNO = f->saved_errno;
            atomic_store_explicit(&w->in_preemptible, 1, memory_order_release);
        }

        uint8_t is_neg;
        __asm__ volatile(
            "subl $1, %%fs:g_tls_preempt_word@tpoff\n\t"
            "sets %0"
            : "=q"(is_neg) :: "cc", "memory"
        );
        if (LIKELY(!is_neg)) break;

        kronos_preempt_disable();
        pw = kronos_preempt_read_word();
        if (pw == 0u) {
            kronos_preempt_set_word(0u);
            break;
        }
        if (pw != (KRONOS_NEED_RESCHED_BIT | 1u)) {
            kronos_preempt_enable_noresched();
            break;
        }
    }
}

void kronos_fiber_entry_c(KronosFiber *f) {
    KronosWorker *w = kronos_self();
    kronos_finish_prev_switch(w);
    KRONOS_ERRNO = 0;
    if (f->preemptible) atomic_store_explicit(&w->in_preemptible, 1, memory_order_release);
    kronos_preempt_enable();

    f->fn(f->arg);

    kronos_preempt_disable();
    w = kronos_self();
    atomic_store_explicit(&w->in_preemptible, 0, memory_order_relaxed);
    w->post_action = POST_FIBER_DONE;
    w->post_fiber  = f;
    kronos_ctx_exit_to_sched(&w->sched_ctx);
}

void kronos_preempt_yield_c(void) {
    KronosWorker *w = kronos_self();
    KronosFiber *f = atomic_load_explicit(&w->current_fiber, memory_order_relaxed);
    f->saved_errno   = KRONOS_ERRNO;
    f->saved_preempt = 1u;
    atomic_store_explicit(&w->in_preemptible, 0, memory_order_relaxed);
    stat_inc(&w->stat_preemptions);
    f->preempt_hits++;
    w->post_action = POST_PREEMPT_REQUEUE;
    w->post_fiber  = f;
    kronos_ctx_to_sched(&f->ctx, &w->sched_ctx);

    w = kronos_self();
    kronos_finish_prev_switch(w);
    KRONOS_ERRNO = f->saved_errno;
}

static inline int rip_in_registered_code(uintptr_t rip) {
    int lim = atomic_load_explicit(&g_kern.code_range_hwm, memory_order_acquire);
    if (lim > KRONOS_MAX_CODE_RANGES) lim = KRONOS_MAX_CODE_RANGES;
    for (int i = 0; i < lim; i++) {
        const KronosCodeRange *r = &g_kern.code_ranges[i];
        uint32_t s1 = atomic_load_explicit(&r->seq, memory_order_acquire);
        if (UNLIKELY(s1 & 1u)) continue;
        int act      = atomic_load_explicit(&r->active, memory_order_relaxed);
        uintptr_t lo = atomic_load_explicit(&r->lo, memory_order_relaxed);
        uintptr_t hi = atomic_load_explicit(&r->hi, memory_order_relaxed);
        atomic_thread_fence(memory_order_acquire);
        uint32_t s2  = atomic_load_explicit(&r->seq, memory_order_relaxed);
        if (LIKELY(s1 == s2) && act == 1 && rip >= lo && rip < hi) {
            return 1;
        }
    }
    return 0;
}

static void kronos_preempt_sig_handler(int sig, siginfo_t *si, void *uctx) {
    (void)sig;
    (void)si;
    int saved_errno = errno;

    KronosWorker *w = kronos_self();
    if (!w) goto out;
    if (!atomic_load_explicit(&w->in_preemptible, memory_order_relaxed)) goto out;

    KronosFiber *f = atomic_load_explicit(&w->current_fiber, memory_order_relaxed);
    if (!f || !f->preemptible) goto out;

    uint64_t now   = kronos_quantum_clock_now();
    uint64_t start = atomic_load_explicit(&w->slice_start_tick, memory_order_relaxed);
    if ((int64_t)(now - start) < (int64_t)g_kern.quantum_ticks) goto out;

    ucontext_t *uc    = (ucontext_t *)uctx;
    uintptr_t old_rip = (uintptr_t)uc->uc_mcontext.gregs[REG_RIP];
    uintptr_t old_rsp = (uintptr_t)uc->uc_mcontext.gregs[REG_RSP];

    /* Guard 2: Never preempt inside assembly switchers or trampoline */
    if (old_rip >= (uintptr_t)&kronos_async_preempt_trampoline &&
        old_rip <  (uintptr_t)&kronos_async_preempt_trampoline_end) goto out;
    if (old_rip >= (uintptr_t)&kronos_ctx_to_sched &&
        old_rip <  (uintptr_t)&kronos_ctx_to_sched_end) goto out;
    if (old_rip >= (uintptr_t)&kronos_ctx_from_sched &&
        old_rip <  (uintptr_t)&kronos_ctx_from_sched_end) goto out;
    if (old_rip >= (uintptr_t)&kronos_ctx_exit_to_sched &&
        old_rip <  (uintptr_t)&kronos_ctx_exit_to_sched_end) goto out;
    if (old_rip >= (uintptr_t)&kronos_ctx_switch_direct &&
        old_rip <  (uintptr_t)&kronos_ctx_switch_direct_end) goto out;

    uint32_t pw = kronos_preempt_read_word();
    if ((pw & KRONOS_PREEMPT_MASK) != 0u) {
        kronos_preempt_latch_resched();
        goto out;
    }

    /* Guard 1: RIP must lie inside seqlock-verified ELF .text or active JIT RX range */
    if (!rip_in_registered_code(old_rip)) goto out;

    /* Guard 3: RSP must lie inside active fiber stack with >= 16 KiB headroom */
    uintptr_t slo = (uintptr_t)f->stack_base + 16384u;
    uintptr_t shi = (uintptr_t)f->stack_base + (uintptr_t)f->stack_size;
    if (old_rsp < slo || old_rsp > shi) goto out;

    kronos_preempt_set_word(1u);
    atomic_store_explicit(&w->in_preemptible, 0, memory_order_relaxed);

    uintptr_t new_rsp = old_rsp - 128u - 8u;
    uint64_t rip_copy = (uint64_t)old_rip;
    memcpy((void *)new_rsp, &rip_copy, sizeof(rip_copy));
    uc->uc_mcontext.gregs[REG_RSP] = (greg_t)new_rsp;
    uc->uc_mcontext.gregs[REG_RIP] = (greg_t)(uintptr_t)&kronos_async_preempt_trampoline;

out:
    errno = saved_errno;
}

/* Lock-free fast-check + per-worker min-heap timer servicing (returns absolute next deadline) */
static uint64_t lapic_service_timers(void) {
    if (atomic_load_explicit(&g_kern.active_timers, memory_order_acquire) == 0u) {
        return UINT64_MAX;
    }

    uint64_t now_ns      = monotonic_ns();
    uint64_t min_dl      = UINT64_MAX;
    KronosFiber *ready_h = NULL;
    int woken_unpinned   = 0;

    for (int i = 0; i < g_kern.num_cores; i++) {
        WorkerTimerHeap *th = &g_kern.workers[i].timers;
        uint64_t e = atomic_load_explicit(&th->earliest_ns, memory_order_acquire);
        if (e == UINT64_MAX) continue;
        if (e <= now_ns) {
            cspin_lock(&th->lock);
            while (atomic_load_explicit(&th->count, memory_order_relaxed) > 0 &&
                   th->entries[0].deadline_ns <= now_ns) {
                uint64_t dl = 0;
                KronosFiber *f = worker_timer_pop_locked(th, &dl);
                uint64_t jit = (now_ns > dl) ? (now_ns - dl) : 0ULL;
                atomic_fetch_add_explicit(&g_kern.timer_wakeups, 1ULL, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_kern.timer_jitter_sum_ns, jit, memory_order_relaxed);
                uint64_t cur_max = atomic_load_explicit(&g_kern.timer_jitter_max_ns, memory_order_relaxed);
                while (jit > cur_max) {
                    if (atomic_compare_exchange_weak_explicit(
                            &g_kern.timer_jitter_max_ns, &cur_max, jit,
                            memory_order_relaxed, memory_order_relaxed)) break;
                }
                f->next_waiter = ready_h;
                ready_h        = f;
            }
            e = atomic_load_explicit(&th->earliest_ns, memory_order_relaxed);
            cspin_unlock(&th->lock);
        }
        if (e < min_dl) min_dl = e;
    }

    while (ready_h) {
        KronosFiber *next = ready_h->next_waiter;
        ready_h->next_waiter = NULL;
        if (UNLIKELY(ready_h->pinned_core >= 0)) {
            int dst = (int)ready_h->pinned_core % g_kern.num_cores;
            pinned_q_push(&g_kern.workers[dst].pinned, ready_h);
            (void)kronos_wake_worker(dst);
        } else {
            (void)grq_push(&g_kern.global_rq, ready_h);
            woken_unpinned++;
        }
        ready_h = next;
    }
    if (woken_unpinned > 0) kronos_wake_any(woken_unpinned);
    return min_dl;
}

static bool lapic_any_timer_earlier_than(uint64_t bound_ns) {
    if (atomic_load_explicit(&g_kern.active_timers, memory_order_acquire) == 0u) return false;
    for (int i = 0; i < g_kern.num_cores; i++) {
        if (atomic_load_explicit(&g_kern.workers[i].timers.earliest_ns, memory_order_acquire) < bound_ns) {
            return true;
        }
    }
    return false;
}

/* Bug-Free, Drift-Free Absolute-Time 0 Hz Tickless LAPIC + Per-Worker Timer Controller */
static void *kronos_lapic_main(void *arg) {
    (void)arg;
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGUSR1);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);

#ifdef PR_SET_TIMERSLACK
    (void)prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
#endif

    uint64_t base_tick_us = g_kern.quantum_us / 4ULL;
    if (base_tick_us < 50ULL)  base_tick_us = 50ULL;
    if (base_tick_us > 250ULL) base_tick_us = 250ULL;
    uint64_t base_tick_ns = base_tick_us * 1000ULL;

    while (!atomic_load_explicit(&g_kern.lapic_stop, memory_order_acquire)) {
        uint32_t ep = atomic_load_explicit(&g_kern.lapic_wake_epoch, memory_order_seq_cst);
        uint64_t next_dl_ns = lapic_service_timers();

        uint64_t now_q = kronos_quantum_clock_now();
        int active_preempt = atomic_load_explicit(&g_kern.active_preemptible_fibers, memory_order_acquire);
        if (LIKELY(g_kern.preempt_supported && active_preempt > 0)) {
            for (int i = 0; i < g_kern.num_cores; i++) {
                KronosWorker *w = &g_kern.workers[i];
                if (!atomic_load_explicit(&w->ready, memory_order_acquire)) continue;
                if (!atomic_load_explicit(&w->in_preemptible, memory_order_acquire)) continue;
                uint32_t pw = atomic_load_explicit(w->preempt_word_ptr, memory_order_relaxed);
                if ((pw & KRONOS_NEED_RESCHED_BIT) != 0u) continue;
                uint64_t start = atomic_load_explicit(&w->slice_start_tick, memory_order_relaxed);
                if ((int64_t)(now_q - start) >= (int64_t)g_kern.quantum_ticks) {
                    (void)sys_tgkill_raw(g_kern.host_pid, w->linux_tid, SIGUSR1);
                    atomic_fetch_add_explicit(&g_kern.total_preempt_ipis, 1, memory_order_relaxed);
                }
            }
        }

        if (atomic_load_explicit(&g_kern.lapic_stop, memory_order_acquire)) break;

        uint64_t now_ns = monotonic_ns();
        uint64_t sleep_until_ns;
        if (active_preempt > 0) {
            uint64_t q_dl = now_ns + base_tick_ns;
            sleep_until_ns = (next_dl_ns < q_dl) ? next_dl_ns : q_dl;
        } else {
            sleep_until_ns = next_dl_ns; /* UINT64_MAX -> 0 Hz infinite sleep! */
        }

        if (sleep_until_ns != UINT64_MAX && sleep_until_ns <= now_ns) continue;

        uint64_t wait_ns = (sleep_until_ns == UINT64_MAX) ? UINT64_MAX : (sleep_until_ns - now_ns);
        if (wait_ns < 20000ULL) {
            cpu_pause_ns(wait_ns);
            continue;
        }

        /*
         * CRITICAL FIX: Announce lapic_sleep_until_ns AND lapic_asleep (whenever active_preempt==0,
         * BOTH for infinite idle AND for long timer sleeps!) with seq_cst re-verification!
         */
        atomic_store_explicit(&g_kern.lapic_sleep_until_ns, sleep_until_ns, memory_order_seq_cst);
        if (active_preempt == 0) {
            atomic_store_explicit(&g_kern.lapic_asleep, 1, memory_order_seq_cst);
        }

        bool abort_sleep =
            (atomic_load_explicit(&g_kern.lapic_stop, memory_order_acquire) != 0) ||
            (active_preempt == 0 &&
             atomic_load_explicit(&g_kern.active_preemptible_fibers, memory_order_acquire) > 0) ||
            lapic_any_timer_earlier_than(sleep_until_ns);

        if (!abort_sleep) {
            (void)sys_futex_wait_abs_ns(&g_kern.lapic_wake_epoch, ep, sleep_until_ns);
        }

        if (active_preempt == 0) {
            atomic_store_explicit(&g_kern.lapic_asleep, 0, memory_order_release);
        }
        atomic_store_explicit(&g_kern.lapic_sleep_until_ns, 0ULL, memory_order_release);
    }
    return NULL;
}

static bool kronos_any_work_visible(int self_id) {
    if (pinned_q_has_item(&g_kern.workers[self_id].pinned)) return true;
    if (inbox_has_item(&g_kern.workers[self_id].inbox)) return true;
    if (deque_approx_size(&g_kern.workers[self_id].deque) > 0) return true;
    if (grq_has_item(&g_kern.global_rq)) return true;
    for (int i = 0; i < g_kern.num_cores; i++) {
        if (i == self_id) continue;
        if (deque_approx_size(&g_kern.workers[i].deque) > 0) return true;
        if (inbox_has_item(&g_kern.workers[i].inbox)) return true;
    }
    return false;
}

/* NUMA/L3-CCD-biased steal-half burst fanout w/ chain wake (never steals pinned queues!) */
static KronosFiber *worker_try_steal(KronosWorker *w, int max_probes) {
    int P = g_kern.num_cores;
    if (P <= 1) return NULL;

    /* Pass 0: local L3-CCD / same NUMA peers; Pass 1: all remaining peers */
    int tried = 0;
    int start = (int)(xorshift64(&w->rng_state) % (uint64_t)P);

    for (int pass = 0; pass < 2 && tried < max_probes; pass++) {
        for (int k = 0; k < P && tried < max_probes; k++) {
            int vid = (start + k) % P;
            if (vid == w->id) continue;
            int is_local = (g_kern.workers[vid].l3_dom == w->l3_dom ||
                            g_kern.workers[vid].numa_node == w->numa_node);
            if ((pass == 0 && !is_local) || (pass == 1 && is_local)) continue;
            tried++;

            if (UNLIKELY(inbox_has_item(&g_kern.workers[vid].inbox))) {
                KronosFiber *f = inbox_pop_nopreempt(&g_kern.workers[vid].inbox);
                if (f) {
                    stat_inc(&w->stat_steals);
                    return f;
                }
            }

            ChaseLevDeque *vd = &g_kern.workers[vid].deque;
            int64_t avail = deque_approx_size(vd);
            if (avail <= 0) continue;

            KronosFiber *first = NULL;
            for (int tries = 0; tries < 2; tries++) {
                StealStatus st = deque_steal_ex(vd, &first);
                if (st == STEAL_OK) {
                    stat_inc(&w->stat_steals);
                    int extra_target = (int)(avail / 2) - 1;
                    if (extra_target > 16) extra_target = 16;
                    if (extra_target < 0)  extra_target = 0;
                    int stolen_extra = 0;
                    for (int e = 0; e < extra_target; e++) {
                        KronosFiber *extra = NULL;
                        if (deque_steal_ex(vd, &extra) == STEAL_OK) {
                            stat_inc(&w->stat_steals);
                            stolen_extra++;
                            if (!deque_push(&w->deque, extra)) {
                                (void)grq_push_nopreempt(&g_kern.global_rq, extra);
                            }
                        } else {
                            break;
                        }
                    }
                    if (stolen_extra > 0) kronos_wake_one();
                    return first;
                }
                if (st == STEAL_EMPTY) break;
                cpu_pause();
            }
        }
    }
    return NULL;
}

static bool try_acquire_spin_slot(int max_spin) {
    int cur = atomic_load_explicit(&g_kern.spinning_workers, memory_order_relaxed);
    while (cur < max_spin) {
        if (atomic_compare_exchange_weak_explicit(
                &g_kern.spinning_workers, &cur, cur + 1,
                memory_order_acquire, memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static void *kronos_worker_main(void *arg) {
    KronosWorker *w = (KronosWorker *)arg;

    int set_cpus = (w->hw_cpu + 1 > 1024) ? (w->hw_cpu + 1) : 1024;
    cpu_set_t *cpuset = CPU_ALLOC((size_t)set_cpus);
    if (cpuset) {
        size_t sz = CPU_ALLOC_SIZE((size_t)set_cpus);
        CPU_ZERO_S(sz, cpuset);
        CPU_SET_S((size_t)w->hw_cpu, sz, cpuset);
        (void)pthread_setaffinity_np(pthread_self(), sz, cpuset);
        CPU_FREE(cpuset);
    }

    kronos_mxcsr_set(g_default_mxcsr_asm);
    kronos_fcw_set(g_default_fcw_asm);

    void *alt_mem = xaligned_alloc(64, g_cpu.altstack_sz);
    stack_t ss;
    ss.ss_sp    = alt_mem;
    ss.ss_size  = g_cpu.altstack_sz;
    ss.ss_flags = 0;
    if (sigaltstack(&ss, NULL) != 0) {
        atomic_fetch_add_explicit(&g_kern.altstack_failures, 1, memory_order_release);
    }

    kronos_set_self(w);
    kronos_set_self_slab(&g_slab.cores[w->id]);
    kronos_preempt_set_word(1u);
    w->linux_tid        = (pid_t)syscall(SYS_gettid);
    w->preempt_word_ptr = &g_tls_preempt_word;
    w->running_preempt  = 0;
    atomic_store_explicit(&w->in_preemptible, 0, memory_order_relaxed);
    atomic_store_explicit(&w->park_state, 0u, memory_order_relaxed);
    w->rng_state = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)(w->id + 1) * 0xBF58476D1CE4E5B9ULL);

    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGUSR1);
    pthread_sigmask(SIG_UNBLOCK, &mask, NULL);

    atomic_store_explicit(&w->ready, 1, memory_order_release);

    KronosFiber *next_fiber = NULL;
    uint64_t sched_tick  = 0;
    uint32_t next_streak = 0;
    int idle_spins       = 0;
    int holds_spin_slot  = 0;
    int prefer_global    = 0;
    int max_spin         = (g_kern.num_cores > 1) ? ((g_kern.num_cores / 2 > 4) ? 4 : (g_kern.num_cores / 2)) : 1;

    while (!atomic_load_explicit(&g_kern.shutdown, memory_order_acquire)) {
        KronosFiber *f = NULL;
        sched_tick++;

        /* Strictly pinned fibers always take highest priority on their home core */
        if (UNLIKELY(pinned_q_has_item(&w->pinned))) {
            f = pinned_q_pop_nopreempt(&w->pinned);
            if (f) {
                if (next_fiber) {
                    if (next_fiber->pinned_core >= 0) pinned_q_push_nopreempt(&w->pinned, next_fiber);
                    else if (!deque_push(&w->deque, next_fiber)) (void)grq_push_nopreempt(&g_kern.global_rq, next_fiber);
                    next_fiber = NULL;
                }
                goto got_fiber;
            }
        }

        if (next_fiber != NULL) {
            if (LIKELY(!grq_has_item(&g_kern.global_rq) &&
                       !inbox_has_item(&w->inbox) && (next_streak & 31u) != 31u)) {
                f = next_fiber;
                next_fiber = NULL;
                next_streak++;
                goto got_fiber;
            }
            next_streak = 0;
            f = grq_pop_nopreempt(&g_kern.global_rq);
            if (!f) f = inbox_pop_nopreempt(&w->inbox);
            if (!f) f = worker_try_steal(w, 4);
            if (f != NULL) {
                if (next_fiber->pinned_core >= 0) pinned_q_push_nopreempt(&w->pinned, next_fiber);
                else if (!deque_push(&w->deque, next_fiber)) (void)grq_push_nopreempt(&g_kern.global_rq, next_fiber);
                next_fiber = NULL;
                goto got_fiber;
            }
            f = next_fiber;
            next_fiber = NULL;
            goto got_fiber;
        } else {
            next_streak = 0;
        }

        if (prefer_global || (sched_tick & 15u) == 0) {
            prefer_global = 0;
            f = grq_pop_nopreempt(&g_kern.global_rq);
        }
        if (!f) f = inbox_pop_nopreempt(&w->inbox);
        if (!f) f = deque_pop(&w->deque);
        if (!f) f = grq_pop_nopreempt(&g_kern.global_rq);
        if (!f && (idle_spins < 4 || (idle_spins & 3) == 0)) {
            int probe_budget = (idle_spins == 0) ? g_kern.num_cores : 4;
            f = worker_try_steal(w, probe_budget);
        }

    got_fiber:
        if (LIKELY(f != NULL)) {
            if (holds_spin_slot) {
                atomic_fetch_sub_explicit(&g_kern.spinning_workers, 1, memory_order_release);
                holds_spin_slot = 0;
                if (grq_has_item(&g_kern.global_rq)) kronos_wake_one();
            }
            idle_spins = 0;

            atomic_store_explicit(&w->current_fiber, f, memory_order_relaxed);
            w->post_action = POST_NONE;
            w->post_fiber  = NULL;
            w->post_cspin  = NULL;
            kronos_preempt_set_word(f->saved_preempt);
            worker_set_preempt_accounting(w, f->preemptible);
            if (f->preemptible) {
                atomic_store_explicit(&w->slice_start_tick, kronos_quantum_clock_now(), memory_order_relaxed);
            }
            stat_inc(&w->stat_ctx_switches);

            kronos_ctx_from_sched(&w->sched_ctx, &f->ctx);

            worker_set_preempt_accounting(w, 0);
            atomic_store_explicit(&w->current_fiber, NULL, memory_order_relaxed);
            atomic_store_explicit(&w->in_preemptible, 0, memory_order_relaxed);
            kronos_preempt_set_word(1u);

            switch (w->post_action) {
            case POST_REQUEUE:
                if (LIKELY(deque_approx_size(&w->deque) == 0 &&
                           !pinned_q_has_item(&w->pinned) &&
                           !inbox_has_item(&w->inbox) &&
                           !grq_has_item(&g_kern.global_rq))) {
                    next_fiber = w->post_fiber;
                } else if (UNLIKELY(w->post_fiber->pinned_core >= 0)) {
                    int dst = (int)w->post_fiber->pinned_core % g_kern.num_cores;
                    pinned_q_push_nopreempt(&g_kern.workers[dst].pinned, w->post_fiber);
                    (void)kronos_wake_worker(dst);
                } else {
                    uint32_t prev = grq_push_nopreempt(&g_kern.global_rq, w->post_fiber);
                    if (prev > 0) prefer_global = 1;
                    kronos_wake_one();
                }
                break;
            case POST_DIRECT_REQUEUE:
            case POST_PREEMPT_REQUEUE:
                if (UNLIKELY(w->post_fiber->pinned_core >= 0)) {
                    int dst = (int)w->post_fiber->pinned_core % g_kern.num_cores;
                    pinned_q_push_nopreempt(&g_kern.workers[dst].pinned, w->post_fiber);
                    (void)kronos_wake_worker(dst);
                } else {
                    uint32_t prev = grq_push_nopreempt(&g_kern.global_rq, w->post_fiber);
                    if (prev > 0) prefer_global = 1;
                    kronos_wake_one();
                }
                break;
            case POST_UNLOCK_CSPIN:
                cspin_unlock(w->post_cspin);
                break;
            case POST_UNLOCK_CSPIN_WAKE_LAPIC:
                cspin_unlock(w->post_cspin);
                kronos_poke_lapic();
                break;
            case POST_FIBER_DONE:
                stat_inc(&w->stat_executed);
                fiber_release(w, w->post_fiber);
                atomic_fetch_sub_explicit(&g_kern.in_flight_fibers, 1, memory_order_release);
                break;
            case POST_NONE:
                break;
            }
        } else {
            if (!holds_spin_slot && idle_spins == 0) {
                holds_spin_slot = try_acquire_spin_slot(max_spin) ? 1 : 0;
            }
            if (holds_spin_slot && idle_spins < 32) {
                idle_spins++;
                cpu_pause_ns((idle_spins < 10) ? 200ULL : 800ULL);
            } else {
                if (holds_spin_slot) {
                    atomic_fetch_sub_explicit(&g_kern.spinning_workers, 1, memory_order_release);
                    holds_spin_slot = 0;
                }
                atomic_store_explicit(&w->park_state, 1u, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_kern.sleeping_workers, 1, memory_order_relaxed);
                sys_membarrier_expedited();
                if (!atomic_load_explicit(&g_kern.shutdown, memory_order_acquire) &&
                    atomic_load_explicit(&w->park_state, memory_order_acquire) == 1u &&
                    !kronos_any_work_visible(w->id)) {
                    long fret = sys_futex(&w->park_state, FUTEX_WAIT_PRIVATE, 1u, NULL);
                    if (fret != -EAGAIN) stat_inc(&w->stat_futex_sleeps);
                }
                atomic_store_explicit(&w->park_state, 0u, memory_order_release);
                atomic_fetch_sub_explicit(&g_kern.sleeping_workers, 1, memory_order_release);
                idle_spins = 0;
            }
        }
    }

    if (holds_spin_slot) {
        atomic_fetch_sub_explicit(&g_kern.spinning_workers, 1, memory_order_release);
    }
    if (next_fiber) {
        fiber_release(w, next_fiber);
        next_fiber = NULL;
    }

    atomic_store_explicit(&w->ready, 0, memory_order_release);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    kronos_set_self(NULL);
    kronos_set_self_slab(NULL);
    ss.ss_flags = SS_DISABLE;
    ss.ss_sp    = NULL;
    ss.ss_size  = 0;
    (void)sigaltstack(&ss, NULL);
    free(alt_mem);
    return NULL;
}

static void kronos_init(int num_cores, uint64_t quantum_us) {
    memset(&g_kern, 0, sizeof(g_kern));
    if (num_cores <= 0) num_cores = g_cpu.avail_cpus;
    if (num_cores > KRONOS_MAX_CORES) num_cores = KRONOS_MAX_CORES;
    if (quantum_us < 200ULL)    quantum_us = 200ULL;
    if (quantum_us > 100000ULL) quantum_us = 100000ULL;

    g_kern.num_cores  = num_cores;
    g_kern.host_pid   = getpid();
    g_kern.quantum_us = quantum_us;
    g_kern.quantum_ticks = g_cpu.tsc_clocksource_ok
        ? (uint64_t)((double)g_kern.quantum_us * 1000.0 * g_cpu.tsc_ghz)
        : (g_kern.quantum_us * 1000ULL);

    (void)kronos_quantum_clock_now();

    grq_init(&g_kern.global_rq);
    tlock_init(&g_kern.code_range_lock);
    tlock_init(&g_kern.pool_lock);
    tlock_init(&g_kern.jit_cache_lock);
    atomic_init(&g_kern.pool_count, 0);
    atomic_init(&g_kern.batch_stacks_allocated, 0);
    atomic_init(&g_kern.lapic_wake_epoch, 0u);
    atomic_init(&g_kern.lapic_asleep, 0);
    atomic_init(&g_kern.lapic_sleep_until_ns, 0ULL);
    atomic_init(&g_kern.active_preemptible_fibers, 0);
    atomic_init(&g_kern.active_timers, 0u);
    atomic_init(&g_kern.code_range_hwm, 0);
    atomic_init(&g_kern.altstack_failures, 0);
    atomic_init(&g_kern.sleeping_workers, 0);
    atomic_init(&g_kern.spinning_workers, 0);
    atomic_init(&g_kern.next_fiber_id, 1ULL);
    atomic_init(&g_kern.rr_cursor, 0ULL);
    atomic_init(&g_kern.total_preempt_ipis, 0ULL);
    atomic_init(&g_kern.timer_wakeups, 0ULL);
    atomic_init(&g_kern.timer_jitter_sum_ns, 0ULL);
    atomic_init(&g_kern.timer_jitter_max_ns, 0ULL);
    atomic_init(&g_kern.in_flight_fibers, 0);
    atomic_init(&g_kern.lapic_stop, 0);
    atomic_init(&g_kern.shutdown, 0);

    for (int i = 0; i < KRONOS_MAX_CODE_RANGES; i++) {
        atomic_init(&g_kern.code_ranges[i].seq, 0u);
        atomic_init(&g_kern.code_ranges[i].lo, 0);
        atomic_init(&g_kern.code_ranges[i].hi, 0);
        atomic_init(&g_kern.code_ranges[i].active, 0);
    }

    kronos_discover_code_bounds();
    g_kern.preempt_supported = (g_kern.static_code_slots > 0) &&
                               (!g_kern.is_static_binary) &&
                               (!g_cpu.has_avx || g_cpu_use_xsave_asm);

    kronos_slab_init();

    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGUSR1);
    pthread_sigmask(SIG_BLOCK, &mask, &g_kern.old_main_sigmask);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = kronos_preempt_sig_handler;
    sa.sa_flags     = SA_SIGINFO | SA_RESTART | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGUSR1, &sa, &g_kern.old_sigusr1) != 0) {
        perror("sigaction SIGUSR1");
        exit(1);
    }

    g_kern.workers = (KronosWorker *)xaligned_alloc(64, (size_t)g_kern.num_cores * sizeof(KronosWorker));
    memset(g_kern.workers, 0, (size_t)g_kern.num_cores * sizeof(KronosWorker));

    for (int i = 0; i < g_kern.num_cores; i++) {
        int slot = i % g_cpu.avail_cpus;
        KronosWorker *w = &g_kern.workers[i];
        w->id        = i;
        w->hw_cpu    = g_cpu.ordered_cpus[slot];
        w->numa_node = g_cpu.ordered_numa[slot];
        w->l3_dom    = g_cpu.ordered_l3[slot];
        w->is_ecore  = g_cpu.ordered_is_ecore[slot];
        pinned_q_init(&w->pinned);
        worker_timer_init(&w->timers);
        inbox_init(&w->inbox);
        deque_init(&w->deque);
        atomic_init(&w->in_preemptible, 0);
        atomic_init(&w->ready, 0);
        atomic_init(&w->park_state, 0u);
        atomic_init(&w->slice_start_tick, 0ULL);
        atomic_init(&w->current_fiber, NULL);
        atomic_init(&w->stat_executed, 0ULL);
        atomic_init(&w->stat_ctx_switches, 0ULL);
        atomic_init(&w->stat_direct_switches, 0ULL);
        atomic_init(&w->stat_steals, 0ULL);
        atomic_init(&w->stat_preemptions, 0ULL);
        atomic_init(&w->stat_deferred_preempts, 0ULL);
        atomic_init(&w->stat_futex_sleeps, 0ULL);
        if (pthread_create(&w->thread, NULL, kronos_worker_main, w) != 0) {
            perror("pthread_create worker");
            exit(1);
        }
    }

    for (int i = 0; i < g_kern.num_cores; i++) {
        while (!atomic_load_explicit(&g_kern.workers[i].ready, memory_order_acquire)) {
            cpu_pause();
        }
    }

    if (atomic_load_explicit(&g_kern.altstack_failures, memory_order_acquire) > 0) {
        g_kern.preempt_supported = 0;
    }

    if (pthread_create(&g_kern.lapic_thread, NULL, kronos_lapic_main, NULL) != 0) {
        perror("pthread_create lapic");
        exit(1);
    }
}

static void kronos_quiesce_in_flight(void) {
    uint64_t t0 = monotonic_ns();
    int spins = 0;
    while (atomic_load_explicit(&g_kern.in_flight_fibers, memory_order_acquire) > 0) {
        if (monotonic_ns() - t0 > 5000000000ULL) break;
        if (++spins > 64) { sched_yield(); spins = 0; }
        else              { cpu_pause(); }
    }
}

static void kronos_shutdown(void) {
    kronos_quiesce_in_flight();

    atomic_store_explicit(&g_kern.lapic_stop, 1, memory_order_release);
    atomic_fetch_add_explicit(&g_kern.lapic_wake_epoch, 1u, memory_order_seq_cst);
    sys_futex(&g_kern.lapic_wake_epoch, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL);
    pthread_join(g_kern.lapic_thread, NULL);

    atomic_store_explicit(&g_kern.shutdown, 1, memory_order_release);
    for (int i = 0; i < g_kern.num_cores; i++) {
        atomic_store_explicit(&g_kern.workers[i].park_state, 2u, memory_order_seq_cst);
        sys_futex(&g_kern.workers[i].park_state, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL);
    }

    for (int i = 0; i < g_kern.num_cores; i++) {
        pthread_join(g_kern.workers[i].thread, NULL);
    }

    (void)sigaction(SIGUSR1, &g_kern.old_sigusr1, NULL);
    (void)pthread_sigmask(SIG_SETMASK, &g_kern.old_main_sigmask, NULL);

    for (int i = 0; i < g_kern.num_cores; i++) {
        KronosWorker *w = &g_kern.workers[i];
        KronosFiber *f;
        while ((f = pinned_q_pop_nopreempt(&w->pinned)) != NULL) fiber_destroy_raw(f);
        while ((f = inbox_pop(&w->inbox)) != NULL) fiber_destroy_raw(f);
        while ((f = deque_pop(&w->deque)) != NULL) fiber_destroy_raw(f);
        uint32_t tcnt = atomic_load_explicit(&w->timers.count, memory_order_relaxed);
        for (uint32_t j = 0; j < tcnt; j++) {
            if (w->timers.entries[j].fiber) fiber_destroy_raw(w->timers.entries[j].fiber);
        }
        free(w->timers.entries);
        w->timers.entries = NULL;

        f = w->local_pool_head;
        while (f) {
            KronosFiber *next = f->next_waiter;
            fiber_destroy_raw(f);
            f = next;
        }
        w->local_pool_head  = NULL;
        w->local_pool_count = 0;
    }

    KronosFiber *f;
    while ((f = grq_pop(&g_kern.global_rq)) != NULL) fiber_destroy_raw(f);

    f = g_kern.pool_head;
    while (f) {
        KronosFiber *next = f->next_waiter;
        fiber_destroy_raw(f);
        f = next;
    }
    g_kern.pool_head = NULL;

    FiberStackChunk *sc = g_kern.stack_chunks;
    while (sc) {
        FiberStackChunk *next = sc->next;
        (void)munmap(sc->base, sc->total_bytes);
        kronos_free(sc, sizeof(FiberStackChunk));
        sc = next;
    }
    g_kern.stack_chunks = NULL;

    for (int i = 0; i < g_kern.jit_page_count; i++) {
        if (g_kern.jit_page_cache[i]) (void)munmap(g_kern.jit_page_cache[i], KRONOS_PAGE_SZ);
    }
    g_kern.jit_page_count = 0;

    free(g_kern.workers);
    g_kern.workers = NULL;
    kronos_slab_destroy();
}

/* ============================================================================
 *  6. 4-STATE UAF-PROOF LATCH, BRANCHLESS DISSEM BARRIER & TICKLESS CHANNEL
 * ============================================================================ */

typedef struct {
    CACHELINE_ALIGN _Atomic uint32_t remaining;
    CACHELINE_ALIGN _Atomic uint32_t state;
    KronosCompactSpin                lock;
    KronosFiber                     *waiters;
} KronosLatch;

static void latch_init(KronosLatch *l, uint32_t count) {
    atomic_init(&l->remaining, count);
    atomic_init(&l->state, (count == 0) ? 3u : 0u);
    cspin_init(&l->lock);
    l->waiters = NULL;
}

static void latch_done(KronosLatch *l) {
    uint32_t prev = atomic_fetch_sub_explicit(&l->remaining, 1u, memory_order_acq_rel);
    if (UNLIKELY(prev == 0u)) {
        atomic_fetch_add_explicit(&l->remaining, 1u, memory_order_relaxed);
        return;
    }
    if (prev == 1u) {
        kronos_preempt_disable();
        cspin_lock(&l->lock);
        KronosFiber *list = l->waiters;
        l->waiters = NULL;
        uint32_t old = atomic_exchange_explicit(&l->state, 2u, memory_order_acq_rel);
        cspin_unlock(&l->lock);

        if (old == 1u) {
            sys_futex(&l->state, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL);
        }
        atomic_store_explicit(&l->state, 3u, memory_order_release);
        kronos_preempt_enable();

        int woken_unpinned = 0;
        while (list) {
            KronosFiber *next = list->next_waiter;
            list->next_waiter = NULL;
            if (UNLIKELY(list->pinned_core >= 0)) {
                int dst = (int)list->pinned_core % g_kern.num_cores;
                pinned_q_push(&g_kern.workers[dst].pinned, list);
                (void)kronos_wake_worker(dst);
            } else {
                (void)grq_push(&g_kern.global_rq, list);
                woken_unpinned++;
            }
            list = next;
        }
        if (woken_unpinned > 0) kronos_wake_any(woken_unpinned);
    }
}

static void latch_wait(KronosLatch *l) {
    if (LIKELY(kronos_self() == NULL)) {
        int spin_cap = (int)(3000.0 / g_pause_cycles);
        if (spin_cap < 16) spin_cap = 16;
        if (spin_cap > 96) spin_cap = 96;
        for (int sp = 0; sp < spin_cap; sp++) {
            if (LIKELY(atomic_load_explicit(&l->state, memory_order_acquire) == 3u)) return;
            cpu_pause();
        }
        int s2_spins = 0;
        for (;;) {
            uint32_t st = atomic_load_explicit(&l->state, memory_order_acquire);
            if (LIKELY(st == 3u)) return;
            if (st == 2u) {
                if (++s2_spins > 256) { sched_yield(); s2_spins = 0; }
                else                  { cpu_pause(); }
                continue;
            }
            if (st == 0u) {
                if (!atomic_compare_exchange_strong_explicit(
                        &l->state, &st, 1u,
                        memory_order_acq_rel, memory_order_acquire)) {
                    continue;
                }
            }
            sys_futex(&l->state, FUTEX_WAIT_PRIVATE, 1u, NULL);
        }
    }

    int s2_spins = 0;
    for (;;) {
        uint32_t st = atomic_load_explicit(&l->state, memory_order_acquire);
        if (LIKELY(st == 3u)) return;
        if (st == 2u) {
            if (++s2_spins > 256) { sched_yield(); s2_spins = 0; }
            else                  { cpu_pause(); }
            continue;
        }

        kronos_preempt_disable();
        KronosWorker *w = kronos_self();
        KronosFiber *f  = atomic_load_explicit(&w->current_fiber, memory_order_relaxed);
        cspin_lock(&l->lock);
        if (atomic_load_explicit(&l->state, memory_order_acquire) >= 2u) {
            cspin_unlock(&l->lock);
            kronos_preempt_enable();
            continue;
        }
        f->saved_errno   = KRONOS_ERRNO;
        f->saved_preempt = kronos_preempt_read_word() & KRONOS_PREEMPT_MASK;
        f->next_waiter   = l->waiters;
        l->waiters       = f;
        atomic_store_explicit(&w->in_preemptible, 0, memory_order_relaxed);
        w->post_action = POST_UNLOCK_CSPIN;
        w->post_cspin  = &l->lock;
        kronos_ctx_to_sched(&f->ctx, &w->sched_ctx);

        kronos_resume_fiber_epilogue(f);
        return;
    }
}

/* Branchless O(log P) 2-Bank Dissemination Barrier (zero IDIV, wrap-safe epochs) */
#define KRONOS_DISSEM_MAX_ROUNDS 7

typedef struct {
    CACHELINE_ALIGN _Atomic uint32_t flag[2][KRONOS_DISSEM_MAX_ROUNDS];
    uint32_t _pad[2];
} DissemNode;

_Static_assert(sizeof(DissemNode) == 64, "DissemNode must be exactly 1 cache line");

typedef struct {
    DissemNode nodes[KRONOS_MAX_CORES];
} KronosDissemBarrier;

static void dissem_init(KronosDissemBarrier *b, int max_participants) {
    if (max_participants > KRONOS_MAX_CORES) max_participants = KRONOS_MAX_CORES;
    for (int i = 0; i < max_participants; i++) {
        for (int bank = 0; bank < 2; bank++) {
            for (int r = 0; r < KRONOS_DISSEM_MAX_ROUNDS; r++) {
                atomic_init(&b->nodes[i].flag[bank][r], 0u);
            }
        }
    }
}

static inline void dissem_wait_group(KronosDissemBarrier *b, int group_base,
                                     int local_id, int group_size, uint32_t *epoch_io) {
    if (group_size <= 1) return;
    /* Advance epoch monotonically; skip 0 AND preserve bank parity across UINT32_MAX wrap!
     * bank = (ep & 1) alternates 0/1 every wait; skipping 0 from 0xFFFFFFFF -> 2 preserves
     * parity (0xFFFFFFFF is odd -> bank 1, 2 is even -> bank 0) so fast peers cannot alias
     * bank 1 of epoch 0xFFFFFFFF with bank 1 of epoch 1! */
    uint32_t ep = (*epoch_io) + 1u;
    if (UNLIKELY(ep == 0u)) ep = 2u;
    *epoch_io = ep;
    uint32_t bank = ep & 1u;
    int self_idx  = group_base + local_id;

    int yield_after;
    if (UNLIKELY(g_kern.num_cores > g_cpu.avail_cpus)) {
        yield_after = 64; /* Oversubscribed: yield quickly instead of burning cores */
    } else {
        yield_after = (int)(300000.0 / g_pause_cycles);
        if (yield_after < 512)  yield_after = 512;
        if (yield_after > 4096) yield_after = 4096;
    }

    for (int r = 0, step = 1; step < group_size; r++, step <<= 1) {
        /* Branchless 1-cycle wrap (eliminates 35-cycle IDIV modulo!) */
        int dst_local = local_id + step;
        if (dst_local >= group_size) dst_local -= group_size;
        int dst = group_base + dst_local;
        atomic_store_explicit(&b->nodes[dst].flag[bank][r], ep, memory_order_release);

        int spins = 0;
        while (atomic_load_explicit(&b->nodes[self_idx].flag[bank][r], memory_order_acquire) != ep) {
            if (++spins >= yield_after) {
                spins = 0;
                kronos_preempt_disable();
                KronosWorker *w = kronos_self();
                bool has_other = (w != NULL) &&
                    (pinned_q_has_item(&w->pinned) || grq_has_item(&g_kern.global_rq) ||
                     inbox_has_item(&w->inbox) || deque_approx_size(&w->deque) > 0);
                kronos_preempt_enable();
                if (has_other) kronos_yield();
                else           sched_yield();
            } else {
                cpu_pause();
            }
        }
    }
}

/* Linearizable, Concurrent-Close-Safe, Tickless-External Vyukov MPMC Channel */
#define KRONOS_CHAN_CAP        256
#define KRONOS_CHAN_MASK       (KRONOS_CHAN_CAP - 1u)
#define KRONOS_CHAN_CLOSED_BIT (1ULL << 63)

typedef struct {
    _Atomic uint64_t seq;
    uint64_t         val;
} ChanSlot;

typedef struct {
    CACHELINE_ALIGN _Atomic uint64_t enq_pos;
    CACHELINE_ALIGN _Atomic uint64_t deq_pos;
    CACHELINE_ALIGN ChanSlot         slots[KRONOS_CHAN_CAP];
    CACHELINE_ALIGN _Atomic uint32_t recv_waiters;
    CACHELINE_ALIGN _Atomic uint32_t send_waiters;
    CACHELINE_ALIGN KronosCompactSpin wait_lock;
    _Atomic uint32_t                 closed;
    _Atomic uint32_t                 send_ext_waiters;
    _Atomic uint32_t                 recv_ext_waiters;
    _Atomic uint32_t                 send_ext_epoch;
    _Atomic uint32_t                 recv_ext_epoch;
    KronosFiber                     *send_q_head, *send_q_tail;
    KronosFiber                     *recv_q_head, *recv_q_tail;
} KronosChan;

static inline uint64_t chan_slot_idx(uint64_t pos) {
    uint64_t p = pos & KRONOS_CHAN_MASK;
    return ((p & 63ULL) << 2) | ((p >> 6) & 3ULL);
}

static void chan_init(KronosChan *ch) {
    memset(ch, 0, sizeof(*ch));
    atomic_init(&ch->enq_pos, 0);
    atomic_init(&ch->deq_pos, 0);
    for (uint64_t i = 0; i < KRONOS_CHAN_CAP; i++) {
        atomic_init(&ch->slots[chan_slot_idx(i)].seq, i);
    }
    atomic_init(&ch->recv_waiters, 0u);
    atomic_init(&ch->send_waiters, 0u);
    atomic_init(&ch->closed, 0u);
    atomic_init(&ch->send_ext_waiters, 0u);
    atomic_init(&ch->recv_ext_waiters, 0u);
    atomic_init(&ch->send_ext_epoch, 0u);
    atomic_init(&ch->recv_ext_epoch, 0u);
    cspin_init(&ch->wait_lock);
}

static inline void waiter_push(KronosFiber **h, KronosFiber **t, KronosFiber *f) {
    f->next_waiter = NULL;
    if (*t) { (*t)->next_waiter = f; *t = f; }
    else    { *h = *t = f; }
}

static inline KronosFiber *waiter_pop(KronosFiber **h, KronosFiber **t) {
    KronosFiber *f = *h;
    if (f) {
        *h = f->next_waiter;
        if (!*h) *t = NULL;
        f->next_waiter = NULL;
    }
    return f;
}

static void chan_wake_one_waiter(KronosChan *ch, int wake_recv) {
    kronos_preempt_disable();
    cspin_lock(&ch->wait_lock);
    KronosFiber *w = wake_recv
        ? waiter_pop(&ch->recv_q_head, &ch->recv_q_tail)
        : waiter_pop(&ch->send_q_head, &ch->send_q_tail);
    if (w) {
        atomic_fetch_sub_explicit(wake_recv ? &ch->recv_waiters : &ch->send_waiters,
                                  1u, memory_order_relaxed);
    }
    int wake_ext = (w == NULL &&
        atomic_load_explicit(wake_recv ? &ch->recv_ext_waiters : &ch->send_ext_waiters,
                             memory_order_relaxed) > 0);
    _Atomic uint32_t *ep_ptr = wake_recv ? &ch->recv_ext_epoch : &ch->send_ext_epoch;
    if (wake_ext) atomic_fetch_add_explicit(ep_ptr, 1u, memory_order_release);
    cspin_unlock(&ch->wait_lock);
    kronos_preempt_enable();

    if (w) {
        kronos_requeue_fiber(w);
    } else if (wake_ext) {
        sys_futex(ep_ptr, FUTEX_WAKE_PRIVATE, 1u, NULL);
    }
}

static int chan_try_send_core(KronosChan *ch, uint64_t val, int *out_need_wake) {
    *out_need_wake = 0;
    kronos_preempt_disable();
    uint64_t pos = atomic_load_explicit(&ch->enq_pos, memory_order_relaxed);
    for (;;) {
        if (UNLIKELY((pos & KRONOS_CHAN_CLOSED_BIT) != 0ULL)) {
            kronos_preempt_enable();
            return -1;
        }
        ChanSlot *slot = &ch->slots[chan_slot_idx(pos)];
        uint64_t seq   = atomic_load_explicit(&slot->seq, memory_order_acquire);
        intptr_t diff  = (intptr_t)seq - (intptr_t)pos;
        if (diff == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &ch->enq_pos, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                slot->val = val;
                atomic_store_explicit(&slot->seq, pos + 1, memory_order_release);
                if (LIKELY(g_has_membar)) atomic_signal_fence(memory_order_seq_cst);
                else                      atomic_thread_fence(memory_order_seq_cst);
                if (UNLIKELY(atomic_load_explicit(&ch->recv_waiters, memory_order_acquire) > 0)) {
                    *out_need_wake = 1;
                }
                kronos_preempt_enable();
                return 1;
            }
        } else if (diff < 0) {
            kronos_preempt_enable();
            return 0;
        } else {
            pos = atomic_load_explicit(&ch->enq_pos, memory_order_relaxed);
        }
    }
}

static int chan_try_recv_core(KronosChan *ch, uint64_t *out, int *out_need_wake) {
    *out_need_wake = 0;
    kronos_preempt_disable();
    uint64_t pos = atomic_load_explicit(&ch->deq_pos, memory_order_relaxed);
    int close_spins = 0;
    for (;;) {
        ChanSlot *slot = &ch->slots[chan_slot_idx(pos)];
        uint64_t seq   = atomic_load_explicit(&slot->seq, memory_order_acquire);
        intptr_t diff  = (intptr_t)seq - (intptr_t)(pos + 1);
        if (diff == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &ch->deq_pos, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                *out = slot->val;
                atomic_store_explicit(&slot->seq, pos + KRONOS_CHAN_CAP, memory_order_release);
                if (LIKELY(g_has_membar)) atomic_signal_fence(memory_order_seq_cst);
                else                      atomic_thread_fence(memory_order_seq_cst);
                if (UNLIKELY(atomic_load_explicit(&ch->send_waiters, memory_order_acquire) > 0)) {
                    *out_need_wake = 1;
                }
                kronos_preempt_enable();
                return 1;
            }
        } else if (diff < 0) {
            uint64_t ep = atomic_load_explicit(&ch->enq_pos, memory_order_acquire);
            if (UNLIKELY((ep & KRONOS_CHAN_CLOSED_BIT) != 0ULL)) {
                uint64_t cpos = ep & ~KRONOS_CHAN_CLOSED_BIT;
                if (pos >= cpos) {
                    kronos_preempt_enable();
                    return -1;
                }
                if (++close_spins >= 64) {
                    kronos_preempt_enable();
                    return -2;
                }
                cpu_pause();
                pos = atomic_load_explicit(&ch->deq_pos, memory_order_relaxed);
                continue;
            }
            kronos_preempt_enable();
            return 0;
        } else {
            pos = atomic_load_explicit(&ch->deq_pos, memory_order_relaxed);
        }
    }
}

static int chan_try_send(KronosChan *ch, uint64_t val) {
    int wake = 0;
    int r = chan_try_send_core(ch, val, &wake);
    if (UNLIKELY(wake)) chan_wake_one_waiter(ch, 1);
    return r;
}

static int chan_try_recv(KronosChan *ch, uint64_t *out) {
    int wake = 0;
    int r = chan_try_recv_core(ch, out, &wake);
    if (UNLIKELY(wake)) chan_wake_one_waiter(ch, 0);
    return r;
}

static bool chan_send(KronosChan *ch, uint64_t val) {
    for (;;) {
        int r = chan_try_send(ch, val);
        if (LIKELY(r == 1)) return true;
        if (UNLIKELY(r < 0)) return false;

        /* Spin 64 pauses + 1 cooperative yield before triggering cross-core membarrier IPI! */
        for (int s = 0; s < 64; s++) {
            cpu_pause();
            r = chan_try_send(ch, val);
            if (r == 1) return true;
            if (r < 0)  return false;
        }
        if (kronos_self() != NULL) {
            kronos_yield();
            r = chan_try_send(ch, val);
            if (r == 1) return true;
            if (r < 0)  return false;
        }

        atomic_fetch_add_explicit(&ch->send_waiters, 1u, memory_order_relaxed);
        sys_membarrier_expedited();

        r = chan_try_send(ch, val);
        if (r != 0) {
            atomic_fetch_sub_explicit(&ch->send_waiters, 1u, memory_order_release);
            return (r == 1);
        }

        kronos_preempt_disable();
        KronosWorker *w = kronos_self();
        KronosFiber *f  = w ? atomic_load_explicit(&w->current_fiber, memory_order_relaxed) : NULL;
        cspin_lock(&ch->wait_lock);
        if (UNLIKELY(!f)) atomic_fetch_add_explicit(&ch->send_ext_waiters, 1u, memory_order_relaxed);
        uint32_t ep = atomic_load_explicit(&ch->send_ext_epoch, memory_order_acquire);

        int wake = 0;
        r = chan_try_send_core(ch, val, &wake);
        if (r != 0) {
            if (UNLIKELY(!f)) atomic_fetch_sub_explicit(&ch->send_ext_waiters, 1u, memory_order_relaxed);
            cspin_unlock(&ch->wait_lock);
            atomic_fetch_sub_explicit(&ch->send_waiters, 1u, memory_order_release);
            kronos_preempt_enable();
            if (UNLIKELY(wake)) chan_wake_one_waiter(ch, 1);
            return (r == 1);
        }
        if (UNLIKELY(!f)) {
            cspin_unlock(&ch->wait_lock);
            kronos_preempt_enable();
            sys_futex(&ch->send_ext_epoch, FUTEX_WAIT_PRIVATE, ep, NULL); /* 0 Hz tickless */
            atomic_fetch_sub_explicit(&ch->send_ext_waiters, 1u, memory_order_release);
            atomic_fetch_sub_explicit(&ch->send_waiters, 1u, memory_order_release);
            continue;
        }
        f->saved_errno   = KRONOS_ERRNO;
        f->saved_preempt = kronos_preempt_read_word() & KRONOS_PREEMPT_MASK;
        waiter_push(&ch->send_q_head, &ch->send_q_tail, f);
        atomic_store_explicit(&w->in_preemptible, 0, memory_order_relaxed);
        w->post_action = POST_UNLOCK_CSPIN;
        w->post_cspin  = &ch->wait_lock;
        kronos_ctx_to_sched(&f->ctx, &w->sched_ctx);

        kronos_resume_fiber_epilogue(f);
    }
}

static bool chan_recv_ex(KronosChan *ch, uint64_t *out) {
    for (;;) {
        int r = chan_try_recv(ch, out);
        if (LIKELY(r == 1)) return true;
        if (UNLIKELY(r == -1)) return false;
        if (UNLIKELY(r == -2)) { sched_yield(); continue; }

        for (int s = 0; s < 64; s++) {
            cpu_pause();
            r = chan_try_recv(ch, out);
            if (r == 1)  return true;
            if (r == -1) return false;
            if (r == -2) break;
        }
        if (UNLIKELY(r == -2)) { sched_yield(); continue; }
        if (kronos_self() != NULL) {
            kronos_yield();
            r = chan_try_recv(ch, out);
            if (r == 1)  return true;
            if (r == -1) return false;
            if (r == -2) { sched_yield(); continue; }
        }

        atomic_fetch_add_explicit(&ch->recv_waiters, 1u, memory_order_relaxed);
        sys_membarrier_expedited();

        r = chan_try_recv(ch, out);
        if (r != 0) {
            atomic_fetch_sub_explicit(&ch->recv_waiters, 1u, memory_order_release);
            if (r == -2) { sched_yield(); continue; }
            return (r == 1);
        }

        kronos_preempt_disable();
        KronosWorker *w = kronos_self();
        KronosFiber *f  = w ? atomic_load_explicit(&w->current_fiber, memory_order_relaxed) : NULL;
        cspin_lock(&ch->wait_lock);
        if (UNLIKELY(!f)) atomic_fetch_add_explicit(&ch->recv_ext_waiters, 1u, memory_order_relaxed);
        uint32_t ep = atomic_load_explicit(&ch->recv_ext_epoch, memory_order_acquire);

        int wake = 0;
        r = chan_try_recv_core(ch, out, &wake);
        if (r != 0) {
            if (UNLIKELY(!f)) atomic_fetch_sub_explicit(&ch->recv_ext_waiters, 1u, memory_order_relaxed);
            cspin_unlock(&ch->wait_lock);
            atomic_fetch_sub_explicit(&ch->recv_waiters, 1u, memory_order_release);
            kronos_preempt_enable();
            if (UNLIKELY(wake)) chan_wake_one_waiter(ch, 0);
            if (UNLIKELY(r == -2)) { sched_yield(); continue; }
            return (r == 1);
        }
        if (UNLIKELY(!f)) {
            cspin_unlock(&ch->wait_lock);
            kronos_preempt_enable();
            sys_futex(&ch->recv_ext_epoch, FUTEX_WAIT_PRIVATE, ep, NULL); /* 0 Hz tickless */
            atomic_fetch_sub_explicit(&ch->recv_ext_waiters, 1u, memory_order_release);
            atomic_fetch_sub_explicit(&ch->recv_waiters, 1u, memory_order_release);
            continue;
        }
        f->saved_errno   = KRONOS_ERRNO;
        f->saved_preempt = kronos_preempt_read_word() & KRONOS_PREEMPT_MASK;
        waiter_push(&ch->recv_q_head, &ch->recv_q_tail, f);
        atomic_store_explicit(&w->in_preemptible, 0, memory_order_relaxed);
        w->post_action = POST_UNLOCK_CSPIN;
        w->post_cspin  = &ch->wait_lock;
        kronos_ctx_to_sched(&f->ctx, &w->sched_ctx);

        kronos_resume_fiber_epilogue(f);
    }
}

static uint64_t chan_recv(KronosChan *ch) {
    uint64_t v = 0;
    (void)chan_recv_ex(ch, &v);
    return v;
}

static void chan_close(KronosChan *ch) {
    (void)atomic_fetch_or_explicit(&ch->enq_pos, KRONOS_CHAN_CLOSED_BIT, memory_order_seq_cst);

    kronos_preempt_disable();
    cspin_lock(&ch->wait_lock);
    atomic_store_explicit(&ch->closed, 1u, memory_order_seq_cst);
    KronosFiber *wlist = NULL;
    KronosFiber *f;
    while ((f = waiter_pop(&ch->recv_q_head, &ch->recv_q_tail)) != NULL) {
        atomic_fetch_sub_explicit(&ch->recv_waiters, 1u, memory_order_relaxed);
        f->next_waiter = wlist;
        wlist = f;
    }
    while ((f = waiter_pop(&ch->send_q_head, &ch->send_q_tail)) != NULL) {
        atomic_fetch_sub_explicit(&ch->send_waiters, 1u, memory_order_relaxed);
        f->next_waiter = wlist;
        wlist = f;
    }
    atomic_fetch_add_explicit(&ch->recv_ext_epoch, 1u, memory_order_release);
    atomic_fetch_add_explicit(&ch->send_ext_epoch, 1u, memory_order_release);
    cspin_unlock(&ch->wait_lock);
    kronos_preempt_enable();

    sys_futex(&ch->recv_ext_epoch, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL);
    sys_futex(&ch->send_ext_epoch, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL);
    int woken = 0;
    while (wlist) {
        KronosFiber *next = wlist->next_waiter;
        wlist->next_waiter = NULL;
        if (UNLIKELY(wlist->pinned_core >= 0)) {
            int dst = (int)wlist->pinned_core % g_kern.num_cores;
            pinned_q_push(&g_kern.workers[dst].pinned, wlist);
            (void)kronos_wake_worker(dst);
        } else {
            (void)grq_push(&g_kern.global_rq, wlist);
            woken++;
        }
        wlist = next;
    }
    if (woken > 0) kronos_wake_any(woken);
}

/* ============================================================================
 *  7. SANDBOXED K-ISA VERIFIER, FIXED-POINT DCE, FUEL SANDBOX & W^X JIT
 * ============================================================================ */

#define KBPF_MAX_INSNS         256
#define KBPF_CAP_NONE          0x0u
#define KBPF_CAP_MEM_RD        0x1u
#define KBPF_CAP_MEM_WR        0x2u
#define KBPF_CAP_MEM_BOUNDED   0x4u
#define KBPF_CAP_BOUNDED_LOOP  0x8u /* Hardware-enforced backedge fuel sandbox */

#define KBPF_LOOP_FUEL_INIT    200000u
#define KBPF_FUEL_EXHAUSTED    0xDEAD0000FEED0000ULL

typedef enum {
    KOP_MOV_REG = 1,
    KOP_MOV_IMM,
    KOP_MOV_IMM64,
    KOP_ADD_REG,
    KOP_ADD_IMM,
    KOP_SUB_REG,
    KOP_SUB_IMM,
    KOP_MUL_REG,
    KOP_MUL_IMM,
    KOP_UDIV_REG,
    KOP_UMOD_REG,
    KOP_NEG,
    KOP_BSWAP64,
    KOP_POPCNT,
    KOP_LZCNT,
    KOP_TZCNT,
    KOP_AND_REG,
    KOP_AND_IMM,
    KOP_OR_REG,
    KOP_OR_IMM,
    KOP_XOR_REG,
    KOP_XOR_IMM,
    KOP_SHL_IMM,
    KOP_SHR_IMM,
    KOP_SAR_IMM,
    KOP_ROR_IMM,
    KOP_SHL_REG,
    KOP_SHR_REG,
    KOP_SAR_REG,
    KOP_LOAD64,
    KOP_STORE64,
    KOP_LOAD32,
    KOP_STORE32,
    KOP_CMP_REG,
    KOP_CMP_IMM,
    KOP_CMOVE,
    KOP_CMOVNE,
    KOP_CMOVB,
    KOP_CMOVAE,
    KOP_CMOVBE,
    KOP_CMOVA,
    KOP_CMOVL,
    KOP_CMOVGE,
    KOP_CMOVLE,
    KOP_CMOVG,
    KOP_JMP,
    KOP_JE,
    KOP_JNE,
    KOP_JB,
    KOP_JAE,
    KOP_JBE,
    KOP_JA,
    KOP_JL,
    KOP_JGE,
    KOP_JLE,
    KOP_JG,
    KOP_RET
} KBpfOp;

typedef struct {
    uint8_t  op;
    uint8_t  dst;
    uint8_t  src;
    uint8_t  idx;
    int32_t  imm32;
    uint64_t imm64;
} KBpfInsn;

static const uint8_t k_hw_reg[8] = { 0, 7, 6, 2, 8, 9, 10, 11 };

#define K_HW_REG_OK(v) ((((v) & 7u) != 4u) && (((v) & 7u) != 5u))
_Static_assert(K_HW_REG_OK(0) && K_HW_REG_OK(7) && K_HW_REG_OK(6) && K_HW_REG_OK(2) &&
               K_HW_REG_OK(8) && K_HW_REG_OK(9) && K_HW_REG_OK(10) && K_HW_REG_OK(11),
               "k_hw_reg must never contain RSP/RBP/R12/R13 SIB/disp32 escape encodings");

typedef uint64_t (*KBpfJitFn)(uint64_t a0, uint64_t a1, uint64_t a2);

typedef struct {
    void     *code;
    size_t    alloc_size;
    size_t    code_len;
    int       code_range_slot;
    int       short_branches;
    int       long_branches;
    int       peephole_opts;
    int       dce_elided;
    int       pruned_prologue_zeros;
    int       jcc_erratum_pads;
    KBpfJitFn fn;
} KBpfCompiled;

typedef struct {
    uint8_t reachable[KBPF_MAX_INSNS];
    uint8_t dead[KBPF_MAX_INSNS];
    uint8_t flags_live_after[KBPF_MAX_INSNS];
    uint8_t regs_live_after[KBPF_MAX_INSNS];
    uint8_t is_jump_target[KBPF_MAX_INSNS];
    uint8_t is_loop_hdr[KBPF_MAX_INSNS];
    uint8_t prologue_zero_mask;
    int     backedges;
} KBpfAnalysis;

static inline bool kbpf_is_jcc(uint8_t op)    { return (op >= KOP_JE && op <= KOP_JG); }
static inline bool kbpf_is_cmov(uint8_t op)   { return (op >= KOP_CMOVE && op <= KOP_CMOVG); }
static inline bool kbpf_is_branch(uint8_t op) { return (op >= KOP_JMP && op <= KOP_JG); }

static uint8_t kbpf_insn_read_mask(const KBpfInsn *in, uint32_t caps) {
    switch (in->op) {
    case KOP_MOV_REG:
    case KOP_POPCNT:
    case KOP_LZCNT:
    case KOP_TZCNT:
        return (uint8_t)(1u << in->src);
    case KOP_MOV_IMM:
    case KOP_MOV_IMM64:
        return 0u;
    case KOP_SUB_REG:
    case KOP_XOR_REG:
        return (in->dst == in->src) ? 0u : (uint8_t)((1u << in->dst) | (1u << in->src));
    case KOP_ADD_REG: case KOP_MUL_REG: case KOP_UDIV_REG: case KOP_UMOD_REG:
    case KOP_AND_REG: case KOP_OR_REG:  case KOP_CMP_REG:
    case KOP_SHL_REG: case KOP_SHR_REG: case KOP_SAR_REG:
    case KOP_CMOVE:   case KOP_CMOVNE:  case KOP_CMOVB:    case KOP_CMOVAE:
    case KOP_CMOVBE:  case KOP_CMOVA:   case KOP_CMOVL:    case KOP_CMOVGE:
    case KOP_CMOVLE:  case KOP_CMOVG:
        return (uint8_t)((1u << in->dst) | (1u << in->src));
    case KOP_MUL_IMM:
    case KOP_AND_IMM:
        return (in->imm32 == 0) ? 0u : (uint8_t)(1u << in->dst);
    case KOP_ADD_IMM: case KOP_SUB_IMM: case KOP_OR_IMM:
    case KOP_XOR_IMM: case KOP_CMP_IMM: case KOP_NEG:     case KOP_BSWAP64:
    case KOP_SHL_IMM: case KOP_SHR_IMM: case KOP_SAR_IMM: case KOP_ROR_IMM:
        return (uint8_t)(1u << in->dst);
    case KOP_LOAD64:
    case KOP_LOAD32:
        return (uint8_t)((1u << in->src) | (1u << in->idx) |
                         ((caps & KBPF_CAP_MEM_BOUNDED) ? (1u << 3) : 0u));
    case KOP_STORE64:
    case KOP_STORE32:
        return (uint8_t)((1u << in->dst) | (1u << in->src) | (1u << in->idx) |
                         ((caps & KBPF_CAP_MEM_BOUNDED) ? (1u << 3) : 0u));
    case KOP_RET:
        return 1u;
    default:
        return 0u;
    }
}

static uint8_t kbpf_insn_kill_uninit_mask(const KBpfInsn *in) {
    switch (in->op) {
    case KOP_MOV_REG:
    case KOP_MOV_IMM:
    case KOP_MOV_IMM64:
    case KOP_POPCNT:
    case KOP_LZCNT:
    case KOP_TZCNT:
    case KOP_LOAD64:
    case KOP_LOAD32:
        return (uint8_t)(1u << in->dst);
    case KOP_SUB_REG:
    case KOP_XOR_REG:
        return (in->dst == in->src) ? (uint8_t)(1u << in->dst) : 0u;
    case KOP_MUL_IMM:
    case KOP_AND_IMM:
        return (in->imm32 == 0) ? (uint8_t)(1u << in->dst) : 0u;
    default:
        return 0u;
    }
}

static uint8_t kbpf_insn_write_mask(const KBpfInsn *in) {
    switch (in->op) {
    case KOP_STORE64: case KOP_STORE32:
    case KOP_CMP_REG: case KOP_CMP_IMM:
    case KOP_RET:
        return 0u;
    default:
        if (kbpf_is_branch(in->op)) return 0u;
        return (uint8_t)(1u << in->dst);
    }
}

static inline bool kbpf_is_pure_dce_eligible(uint8_t op) {
    switch (op) {
    case KOP_STORE64: case KOP_STORE32:
    case KOP_LOAD64:  case KOP_LOAD32:
    case KOP_CMP_REG: case KOP_CMP_IMM:
    case KOP_RET:
        return false;
    default:
        return !kbpf_is_branch(op);
    }
}

static void kbpf_disasm_insn(int pc, const KBpfInsn *in, char *buf, size_t sz) {
    switch (in->op) {
    case KOP_MOV_REG:   snprintf(buf, sz, "%02d: mov     r%u, r%u", pc, in->dst, in->src); break;
    case KOP_MOV_IMM:   snprintf(buf, sz, "%02d: mov     r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_MOV_IMM64: snprintf(buf, sz, "%02d: mov64   r%u, 0x%016" PRIX64, pc, in->dst, in->imm64); break;
    case KOP_ADD_REG:   snprintf(buf, sz, "%02d: add     r%u, r%u", pc, in->dst, in->src); break;
    case KOP_ADD_IMM:   snprintf(buf, sz, "%02d: add     r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_SUB_REG:   snprintf(buf, sz, "%02d: sub     r%u, r%u", pc, in->dst, in->src); break;
    case KOP_SUB_IMM:   snprintf(buf, sz, "%02d: sub     r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_MUL_REG:   snprintf(buf, sz, "%02d: mul     r%u, r%u", pc, in->dst, in->src); break;
    case KOP_MUL_IMM:   snprintf(buf, sz, "%02d: mul     r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_UDIV_REG:  snprintf(buf, sz, "%02d: udiv    r%u, r%u", pc, in->dst, in->src); break;
    case KOP_UMOD_REG:  snprintf(buf, sz, "%02d: umod    r%u, r%u", pc, in->dst, in->src); break;
    case KOP_NEG:       snprintf(buf, sz, "%02d: neg     r%u", pc, in->dst); break;
    case KOP_BSWAP64:   snprintf(buf, sz, "%02d: bswap64 r%u", pc, in->dst); break;
    case KOP_POPCNT:    snprintf(buf, sz, "%02d: popcnt  r%u, r%u", pc, in->dst, in->src); break;
    case KOP_LZCNT:     snprintf(buf, sz, "%02d: lzcnt   r%u, r%u", pc, in->dst, in->src); break;
    case KOP_TZCNT:     snprintf(buf, sz, "%02d: tzcnt   r%u, r%u", pc, in->dst, in->src); break;
    case KOP_AND_REG:   snprintf(buf, sz, "%02d: and     r%u, r%u", pc, in->dst, in->src); break;
    case KOP_AND_IMM:   snprintf(buf, sz, "%02d: and     r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_OR_REG:    snprintf(buf, sz, "%02d: or      r%u, r%u", pc, in->dst, in->src); break;
    case KOP_OR_IMM:    snprintf(buf, sz, "%02d: or      r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_XOR_REG:   snprintf(buf, sz, "%02d: xor     r%u, r%u", pc, in->dst, in->src); break;
    case KOP_XOR_IMM:   snprintf(buf, sz, "%02d: xor     r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_SHL_IMM:   snprintf(buf, sz, "%02d: shl     r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_SHR_IMM:   snprintf(buf, sz, "%02d: shr     r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_SAR_IMM:   snprintf(buf, sz, "%02d: sar     r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_ROR_IMM:   snprintf(buf, sz, "%02d: ror     r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_SHL_REG:   snprintf(buf, sz, "%02d: shl     r%u, r%u", pc, in->dst, in->src); break;
    case KOP_SHR_REG:   snprintf(buf, sz, "%02d: shr     r%u, r%u", pc, in->dst, in->src); break;
    case KOP_SAR_REG:   snprintf(buf, sz, "%02d: sar     r%u, r%u", pc, in->dst, in->src); break;
    case KOP_LOAD64:    snprintf(buf, sz, "%02d: ld64    r%u, [r%u + r%u*8]", pc, in->dst, in->src, in->idx); break;
    case KOP_STORE64:   snprintf(buf, sz, "%02d: st64    [r%u + r%u*8], r%u", pc, in->src, in->idx, in->dst); break;
    case KOP_LOAD32:    snprintf(buf, sz, "%02d: ld32    r%u, [r%u + r%u*4]", pc, in->dst, in->src, in->idx); break;
    case KOP_STORE32:   snprintf(buf, sz, "%02d: st32    [r%u + r%u*4], r%u", pc, in->src, in->idx, in->dst); break;
    case KOP_CMP_REG:   snprintf(buf, sz, "%02d: cmp     r%u, r%u", pc, in->dst, in->src); break;
    case KOP_CMP_IMM:   snprintf(buf, sz, "%02d: cmp     r%u, %d", pc, in->dst, in->imm32); break;
    case KOP_CMOVE:     snprintf(buf, sz, "%02d: cmove   r%u, r%u", pc, in->dst, in->src); break;
    case KOP_CMOVNE:    snprintf(buf, sz, "%02d: cmovne  r%u, r%u", pc, in->dst, in->src); break;
    case KOP_CMOVB:     snprintf(buf, sz, "%02d: cmovb   r%u, r%u", pc, in->dst, in->src); break;
    case KOP_CMOVAE:    snprintf(buf, sz, "%02d: cmovae  r%u, r%u", pc, in->dst, in->src); break;
    case KOP_CMOVBE:    snprintf(buf, sz, "%02d: cmovbe  r%u, r%u", pc, in->dst, in->src); break;
    case KOP_CMOVA:     snprintf(buf, sz, "%02d: cmova   r%u, r%u", pc, in->dst, in->src); break;
    case KOP_CMOVL:     snprintf(buf, sz, "%02d: cmovl   r%u, r%u", pc, in->dst, in->src); break;
    case KOP_CMOVGE:    snprintf(buf, sz, "%02d: cmovge  r%u, r%u", pc, in->dst, in->src); break;
    case KOP_CMOVLE:    snprintf(buf, sz, "%02d: cmovle  r%u, r%u", pc, in->dst, in->src); break;
    case KOP_CMOVG:     snprintf(buf, sz, "%02d: cmovg   r%u, r%u", pc, in->dst, in->src); break;
    case KOP_JMP:       snprintf(buf, sz, "%02d: jmp     L%02d", pc, in->imm32); break;
    case KOP_JE:        snprintf(buf, sz, "%02d: je      L%02d", pc, in->imm32); break;
    case KOP_JNE:       snprintf(buf, sz, "%02d: jne     L%02d", pc, in->imm32); break;
    case KOP_JB:        snprintf(buf, sz, "%02d: jb      L%02d", pc, in->imm32); break;
    case KOP_JAE:       snprintf(buf, sz, "%02d: jae     L%02d", pc, in->imm32); break;
    case KOP_JBE:       snprintf(buf, sz, "%02d: jbe     L%02d", pc, in->imm32); break;
    case KOP_JA:        snprintf(buf, sz, "%02d: ja      L%02d", pc, in->imm32); break;
    case KOP_JL:        snprintf(buf, sz, "%02d: jl      L%02d", pc, in->imm32); break;
    case KOP_JGE:       snprintf(buf, sz, "%02d: jge     L%02d", pc, in->imm32); break;
    case KOP_JLE:       snprintf(buf, sz, "%02d: jle     L%02d", pc, in->imm32); break;
    case KOP_JG:        snprintf(buf, sz, "%02d: jg      L%02d", pc, in->imm32); break;
    case KOP_RET:       snprintf(buf, sz, "%02d: ret     r0", pc); break;
    default:            snprintf(buf, sz, "%02d: <invalid op %u>", pc, in->op); break;
    }
}

static void kbpf_disasm_print(const KBpfInsn *prog, int n) {
    if (g_out_mode != OUT_PRETTY) return;
    char line[96];
    for (int i = 0; i < n; i += 2) {
        kbpf_disasm_insn(i, &prog[i], line, sizeof(line));
        if (i + 1 < n) {
            char line2[96];
            kbpf_disasm_insn(i + 1, &prog[i + 1], line2, sizeof(line2));
            printf("       %s%-34s  |  %s%s\n", C_DIM, line, line2, C_RST);
        } else {
            printf("       %s%s%s\n", C_DIM, line, C_RST);
        }
    }
}

static int kbpf_verify(const KBpfInsn *prog, int n, uint32_t caps,
                       char *err_buf, size_t err_sz, KBpfAnalysis *out_an) {
    if (n <= 0 || n > KBPF_MAX_INSNS) {
        snprintf(err_buf, err_sz, "program length %d outside [1, %d]", n, KBPF_MAX_INSNS);
        return -1;
    }

    int bounded_mem  = ((caps & KBPF_CAP_MEM_BOUNDED) != 0u);
    int bounded_loop = ((caps & KBPF_CAP_BOUNDED_LOOP) != 0u);
    int backedges    = 0;

    for (int pc = 0; pc < n; pc++) {
        const KBpfInsn *in = &prog[pc];
        if (in->dst >= 8 || in->src >= 8 || in->idx >= 8) {
            snprintf(err_buf, err_sz, "pc %d: register index out of [0,7]", pc);
            return -1;
        }
        if (bounded_mem) {
            uint8_t wm = kbpf_insn_write_mask(in);
            if (wm & ((1u << 1) | (1u << 3))) {
                snprintf(err_buf, err_sz,
                         "pc %d: CAP_MEM_BOUNDED forbids mutating base/byte-len regs R1/R3", pc);
                return -1;
            }
        }
        switch (in->op) {
        case KOP_MOV_REG: case KOP_MOV_IMM: case KOP_MOV_IMM64:
        case KOP_ADD_REG: case KOP_ADD_IMM: case KOP_SUB_REG: case KOP_SUB_IMM:
        case KOP_MUL_REG: case KOP_MUL_IMM: case KOP_UDIV_REG: case KOP_UMOD_REG:
        case KOP_NEG:     case KOP_BSWAP64:
        case KOP_AND_REG: case KOP_AND_IMM: case KOP_OR_REG:  case KOP_OR_IMM:
        case KOP_XOR_REG: case KOP_XOR_IMM:
        case KOP_SHL_REG: case KOP_SHR_REG: case KOP_SAR_REG:
        case KOP_CMP_REG: case KOP_CMP_IMM:
        case KOP_CMOVE:   case KOP_CMOVNE:  case KOP_CMOVB:   case KOP_CMOVAE:
        case KOP_CMOVBE:  case KOP_CMOVA:   case KOP_CMOVL:   case KOP_CMOVGE:
        case KOP_CMOVLE:  case KOP_CMOVG:
        case KOP_RET:
            break;
        case KOP_POPCNT:
            if (!g_cpu.has_popcnt) {
                snprintf(err_buf, err_sz, "pc %d: POPCNT unsupported on host silicon", pc);
                return -1;
            }
            break;
        case KOP_LZCNT:
            if (!g_cpu.has_lzcnt) {
                snprintf(err_buf, err_sz, "pc %d: LZCNT unsupported on host silicon", pc);
                return -1;
            }
            break;
        case KOP_TZCNT:
            if (!g_cpu.has_bmi1) {
                snprintf(err_buf, err_sz, "pc %d: TZCNT (BMI1) unsupported on host silicon", pc);
                return -1;
            }
            break;
        case KOP_SHL_IMM: case KOP_SHR_IMM: case KOP_SAR_IMM: case KOP_ROR_IMM:
            if (in->imm32 < 0 || in->imm32 > 63) {
                snprintf(err_buf, err_sz, "pc %d: shift count %d out of [0,63]", pc, in->imm32);
                return -1;
            }
            break;
        case KOP_LOAD64:
        case KOP_LOAD32:
            if (!(caps & (KBPF_CAP_MEM_RD | KBPF_CAP_MEM_BOUNDED))) {
                snprintf(err_buf, err_sz, "pc %d: LOAD denied (missing CAP_MEM_RD/BOUNDED)", pc);
                return -1;
            }
            if (bounded_mem && in->src != 1) {
                snprintf(err_buf, err_sz, "pc %d: CAP_MEM_BOUNDED requires base reg R1", pc);
                return -1;
            }
            break;
        case KOP_STORE64:
        case KOP_STORE32:
            if (!(caps & KBPF_CAP_MEM_WR)) {
                snprintf(err_buf, err_sz, "pc %d: STORE denied (missing CAP_MEM_WR)", pc);
                return -1;
            }
            if (bounded_mem && in->src != 1) {
                snprintf(err_buf, err_sz, "pc %d: CAP_MEM_BOUNDED requires base reg R1", pc);
                return -1;
            }
            break;
        case KOP_JMP:
        case KOP_JE: case KOP_JNE: case KOP_JB: case KOP_JAE:
        case KOP_JBE: case KOP_JA: case KOP_JL: case KOP_JGE:
        case KOP_JLE: case KOP_JG:
            if (in->imm32 < 0 || in->imm32 >= n) {
                snprintf(err_buf, err_sz, "pc %d: branch target %d out of [0,%d)", pc, in->imm32, n);
                return -1;
            }
            if (in->imm32 == pc) {
                snprintf(err_buf, err_sz, "pc %d: illegal trivial infinite self-loop", pc);
                return -1;
            }
            if (in->imm32 < pc) backedges++;
            break;
        default:
            snprintf(err_buf, err_sz, "pc %d: unknown opcode %u", pc, in->op);
            return -1;
        }
    }

    uint8_t seen[KBPF_MAX_INSNS];
    memset(seen, 0, sizeof(seen));
    struct { int pc; uint8_t fv; } q[KBPF_MAX_INSNS * 2];
    int qh = 0, qt = 0;
    int reachable_ret = 0;

    seen[0] = 1u;
    q[qt].pc = 0; q[qt].fv = 0; qt++;

    while (qh < qt) {
        int pc     = q[qh].pc;
        uint8_t fv = q[qh].fv;
        qh++;
        const KBpfInsn *in = &prog[pc];

        int next1 = -1, next2 = -1;
        uint8_t fv1 = fv, fv2 = fv;

        if (in->op == KOP_RET) {
            reachable_ret = 1;
        } else if (in->op == KOP_JMP) {
            next1 = in->imm32;
            fv1   = (bounded_loop && in->imm32 < pc) ? 0u : fv;
        } else if (kbpf_is_jcc(in->op)) {
            if (!fv) {
                snprintf(err_buf, err_sz, "pc %d: conditional branch with undefined/clobbered flags", pc);
                return -1;
            }
            if (pc + 1 >= n) {
                snprintf(err_buf, err_sz, "pc %d: conditional branch fallthrough exits program", pc);
                return -1;
            }
            next1 = in->imm32;
            fv1   = (bounded_loop && in->imm32 < pc) ? 0u : fv;
            next2 = pc + 1;
            fv2   = fv;
        } else if (kbpf_is_cmov(in->op)) {
            if (!fv) {
                snprintf(err_buf, err_sz, "pc %d: CMOV with undefined/clobbered flags", pc);
                return -1;
            }
            if (pc + 1 >= n) {
                snprintf(err_buf, err_sz, "pc %d: control flow falls off end of program", pc);
                return -1;
            }
            next1 = pc + 1; fv1 = fv;
        } else if (in->op == KOP_CMP_REG || in->op == KOP_CMP_IMM) {
            if (pc + 1 >= n) {
                snprintf(err_buf, err_sz, "pc %d: control flow falls off end of program", pc);
                return -1;
            }
            next1 = pc + 1; fv1 = 1;
        } else if (in->op == KOP_MOV_REG || in->op == KOP_MOV_IMM ||
                   in->op == KOP_MOV_IMM64 || in->op == KOP_BSWAP64 ||
                   ((in->op == KOP_LOAD64 || in->op == KOP_STORE64 ||
                     in->op == KOP_LOAD32 || in->op == KOP_STORE32) && !bounded_mem)) {
            if (pc + 1 >= n) {
                snprintf(err_buf, err_sz, "pc %d: control flow falls off end of program", pc);
                return -1;
            }
            next1 = pc + 1; fv1 = fv;
        } else {
            if (pc + 1 >= n) {
                snprintf(err_buf, err_sz, "pc %d: control flow falls off end of program", pc);
                return -1;
            }
            next1 = pc + 1; fv1 = 0;
        }

        if (next1 >= 0 && !(seen[next1] & (1u << fv1))) {
            seen[next1] |= (uint8_t)(1u << fv1);
            q[qt].pc = next1; q[qt].fv = fv1; qt++;
        }
        if (next2 >= 0 && !(seen[next2] & (1u << fv2))) {
            seen[next2] |= (uint8_t)(1u << fv2);
            q[qt].pc = next2; q[qt].fv = fv2; qt++;
        }
    }

    if (!reachable_ret) {
        snprintf(err_buf, err_sz, "program has no reachable RET terminator");
        return -1;
    }

    if (out_an) {
        memset(out_an, 0, sizeof(*out_an));
        uint8_t fl_in[KBPF_MAX_INSNS], fl_out[KBPF_MAX_INSNS];
        uint8_t rl_in[KBPF_MAX_INSNS], rl_out[KBPF_MAX_INSNS];
        uint8_t dead[KBPF_MAX_INSNS];
        memset(dead, 0, sizeof(dead));

        /* Iterative fixed-point backward liveness + transitive dead-code elimination! */
        int outer_changed = 1;
        while (outer_changed) {
            outer_changed = 0;
            memset(fl_in,  0, sizeof(fl_in));
            memset(fl_out, 0, sizeof(fl_out));
            memset(rl_in,  0, sizeof(rl_in));
            memset(rl_out, 0, sizeof(rl_out));

            int changed = 1;
            while (changed) {
                changed = 0;
                for (int pc = n - 1; pc >= 0; pc--) {
                    if (!seen[pc]) continue;
                    const KBpfInsn *in = &prog[pc];
                    uint8_t fo = 0, ro = 0;
                    if (in->op == KOP_RET) {
                        fo = 0; ro = 0;
                    } else if (in->op == KOP_JMP) {
                        fo = (bounded_loop && in->imm32 < pc) ? 0u : fl_in[in->imm32];
                        ro = rl_in[in->imm32];
                    } else if (kbpf_is_jcc(in->op)) {
                        uint8_t ft = (bounded_loop && in->imm32 < pc) ? 0u : fl_in[in->imm32];
                        fo = (uint8_t)(ft | fl_in[pc + 1]);
                        ro = (uint8_t)(rl_in[in->imm32] | rl_in[pc + 1]);
                    } else {
                        fo = fl_in[pc + 1];
                        ro = rl_in[pc + 1];
                    }

                    uint8_t fi, ri;
                    if (dead[pc]) {
                        fi = fo;
                        ri = ro;
                    } else {
                        if (kbpf_is_jcc(in->op) || kbpf_is_cmov(in->op)) {
                            fi = 1;
                        } else if (in->op == KOP_CMP_REG || in->op == KOP_CMP_IMM || in->op == KOP_RET) {
                            fi = 0;
                        } else if (in->op == KOP_MOV_REG || in->op == KOP_MOV_IMM ||
                                   in->op == KOP_MOV_IMM64 || in->op == KOP_BSWAP64 || in->op == KOP_JMP ||
                                   ((in->op == KOP_LOAD64 || in->op == KOP_STORE64 ||
                                     in->op == KOP_LOAD32 || in->op == KOP_STORE32) && !bounded_mem)) {
                            fi = fo;
                        } else {
                            fi = 0;
                        }
                        uint8_t rd = kbpf_insn_read_mask(in, caps);
                        uint8_t kl = kbpf_insn_kill_uninit_mask(in);
                        ri = (uint8_t)((ro & ~kl) | rd);
                    }

                    if (fo != fl_out[pc] || fi != fl_in[pc] || ro != rl_out[pc] || ri != rl_in[pc]) {
                        fl_out[pc] = fo; fl_in[pc] = fi;
                        rl_out[pc] = ro; rl_in[pc] = ri;
                        changed = 1;
                    }
                }
            }

            for (int pc = 0; pc < n; pc++) {
                if (!seen[pc] || dead[pc]) continue;
                const KBpfInsn *in = &prog[pc];
                if (kbpf_is_pure_dce_eligible(in->op) && !(rl_out[pc] & (1u << in->dst))) {
                    dead[pc] = 1u;
                    outer_changed = 1;
                }
            }
        }

        uint8_t uninit_in[KBPF_MAX_INSNS];
        memset(uninit_in, 0, sizeof(uninit_in));
        uninit_in[0] = 0xF1u;
        uint8_t needs_zero = 0;

        int changed = 1;
        while (changed) {
            changed = 0;
            for (int pc = 0; pc < n; pc++) {
                if (!seen[pc]) continue;
                const KBpfInsn *in = &prog[pc];
                uint8_t cur = uninit_in[pc];
                uint8_t out_u = cur;
                if (!dead[pc]) {
                    needs_zero |= (uint8_t)(cur & kbpf_insn_read_mask(in, caps));
                    out_u = (uint8_t)(cur & ~kbpf_insn_kill_uninit_mask(in));
                }

                if (in->op == KOP_RET) continue;
                if (in->op == KOP_JMP) {
                    int t = in->imm32;
                    if ((uninit_in[t] | out_u) != uninit_in[t]) { uninit_in[t] |= out_u; changed = 1; }
                } else if (kbpf_is_jcc(in->op)) {
                    int t1 = in->imm32, t2 = pc + 1;
                    if ((uninit_in[t1] | out_u) != uninit_in[t1]) { uninit_in[t1] |= out_u; changed = 1; }
                    if ((uninit_in[t2] | out_u) != uninit_in[t2]) { uninit_in[t2] |= out_u; changed = 1; }
                } else {
                    int t = pc + 1;
                    if ((uninit_in[t] | out_u) != uninit_in[t]) { uninit_in[t] |= out_u; changed = 1; }
                }
            }
        }

        for (int pc = 0; pc < n; pc++) {
            out_an->reachable[pc]        = (seen[pc] != 0) ? 1u : 0u;
            out_an->dead[pc]             = dead[pc];
            out_an->flags_live_after[pc] = fl_out[pc];
            out_an->regs_live_after[pc]  = rl_out[pc];
            if (seen[pc] && kbpf_is_branch(prog[pc].op)) {
                out_an->is_jump_target[prog[pc].imm32] = 1u;
                if (prog[pc].imm32 < pc) out_an->is_loop_hdr[prog[pc].imm32] = 1u;
            }
        }
        out_an->prologue_zero_mask = (uint8_t)(needs_zero & 0xF1u);
        out_an->backedges          = backedges;
    }
    return 0;
}

static inline uint8_t jit_rex(int w, uint8_t r, uint8_t x, uint8_t b) {
    return (uint8_t)(0x40u | ((w & 1) << 3) | (((r >> 3) & 1u) << 2)
                           | (((x >> 3) & 1u) << 1) | ((b >> 3) & 1u));
}

static inline uint8_t jit_modrm(uint8_t mod, uint8_t reg, uint8_t rm) {
    return (uint8_t)(((mod & 3u) << 6) | ((reg & 7u) << 3) | (rm & 7u));
}

typedef struct {
    uint8_t *buf;
    size_t   len;
    size_t   cap;
    int      overflow;
} JitBuf;

static inline void emit8(JitBuf *j, uint8_t v) {
    if (LIKELY(j->len + 1 <= j->cap)) { if (j->buf) j->buf[j->len] = v; j->len++; }
    else j->overflow = 1;
}

static inline void emit32(JitBuf *j, uint32_t v) {
    if (LIKELY(j->len + 4 <= j->cap)) { if (j->buf) memcpy(j->buf + j->len, &v, 4); j->len += 4; }
    else j->overflow = 1;
}

static inline void emit64(JitBuf *j, uint64_t v) {
    if (LIKELY(j->len + 8 <= j->cap)) { if (j->buf) memcpy(j->buf + j->len, &v, 8); j->len += 8; }
    else j->overflow = 1;
}

static void emit_nops(JitBuf *j, size_t rem) {
    static const uint8_t nop1[] = { 0x90 };
    static const uint8_t nop2[] = { 0x66, 0x90 };
    static const uint8_t nop3[] = { 0x0F, 0x1F, 0x00 };
    static const uint8_t nop4[] = { 0x0F, 0x1F, 0x40, 0x00 };
    static const uint8_t nop5[] = { 0x0F, 0x1F, 0x44, 0x00, 0x00 };
    static const uint8_t nop6[] = { 0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00 };
    static const uint8_t nop7[] = { 0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00 };
    static const uint8_t nop8[] = { 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00 };
    static const uint8_t nop9[] = { 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00 };
    static const uint8_t *const tbl[10] = {
        NULL, nop1, nop2, nop3, nop4, nop5, nop6, nop7, nop8, nop9
    };
    while (rem > 0) {
        size_t chunk = (rem > 9u) ? 9u : rem;
        for (size_t i = 0; i < chunk; i++) emit8(j, tbl[chunk][i]);
        rem -= chunk;
    }
}

static void emit_align16_nops(JitBuf *j) {
    emit_nops(j, (16u - (j->len & 15u)) & 15u);
}

static void emit_xor_zero32(JitBuf *j, uint8_t hr) {
    if (hr >= 8) emit8(j, jit_rex(0, hr, 0, hr));
    emit8(j, 0x31);
    emit8(j, jit_modrm(3, hr, hr));
}

static void emit_lea_scale_pair(JitBuf *j, uint8_t hd, uint8_t hbase, uint8_t hidx, uint8_t scale) {
    emit8(j, jit_rex(1, hd, hidx, hbase));
    emit8(j, 0x8D);
    emit8(j, jit_modrm(0, hd, 4));
    emit8(j, (uint8_t)((scale << 6) | ((hidx & 7u) << 3) | (hbase & 7u)));
}

static void emit_lea_disp(JitBuf *j, uint8_t hd, uint8_t hs, int32_t disp) {
    emit8(j, jit_rex(1, hd, 0, hs));
    emit8(j, 0x8D);
    if (disp >= -128 && disp <= 127) {
        emit8(j, jit_modrm(1, hd, hs));
        emit8(j, (uint8_t)(int8_t)disp);
    } else {
        emit8(j, jit_modrm(2, hd, hs));
        emit32(j, (uint32_t)disp);
    }
}

/* BMI2 RORX r64, r64, imm8: VEX.LZ.F2.0F3A.W1 F0 /r ib */
static void emit_bmi2_rorx(JitBuf *j, uint8_t hd, uint8_t hs, uint8_t imm8) {
    uint8_t v1 = (uint8_t)((((~(hd >> 3)) & 1u) << 7) | 0x40u |
                           ((((~(hs >> 3)) & 1u) << 5)) | 0x03u);
    emit8(j, 0xC4);
    emit8(j, v1);
    emit8(j, 0xFB);
    emit8(j, 0xF0);
    emit8(j, jit_modrm(3, hd, hs));
    emit8(j, imm8);
}

/* BMI2 SHLX/SARX/SHRX r64, r64, r64: VEX.LZ.{66|F3|F2}.0F38.W1 F7 /r */
static void emit_bmi2_shiftx(JitBuf *j, uint8_t hd, uint8_t hcnt, uint8_t pp) {
    uint8_t rb = (uint8_t)((~(hd >> 3)) & 1u);
    uint8_t v1 = (uint8_t)((rb << 7) | 0x40u | (rb << 5) | 0x02u);
    uint8_t v2 = (uint8_t)(0x80u | (((~hcnt) & 0x0Fu) << 3) | (pp & 3u));
    emit8(j, 0xC4);
    emit8(j, v1);
    emit8(j, v2);
    emit8(j, 0xF7);
    emit8(j, jit_modrm(3, hd, hd));
}

static void emit_alu_imm64(JitBuf *j, uint8_t ext, uint8_t hd, int32_t imm32) {
    static const uint8_t rax_op[8] = { 0x05, 0x0D, 0x15, 0x1D, 0x25, 0x2D, 0x35, 0x3D };
    emit8(j, jit_rex(1, 0, 0, hd));
    if (imm32 >= -128 && imm32 <= 127) {
        emit8(j, 0x83); emit8(j, jit_modrm(3, ext, hd)); emit8(j, (uint8_t)(int8_t)imm32);
    } else if (hd == 0) {
        emit8(j, rax_op[ext & 7u]); emit32(j, (uint32_t)imm32);
    } else {
        emit8(j, 0x81); emit8(j, jit_modrm(3, ext, hd)); emit32(j, (uint32_t)imm32);
    }
}

typedef struct {
    size_t patch_off;
    int    src_pc;
    int    target_pc; /* -1 = fuel_abort stub at end */
    int    is_short;
} JitPatch;

static inline uint8_t kbpf_cc_nibble(uint8_t op) {
    switch (op) {
    case KOP_JE:  case KOP_CMOVE:  return 0x4u;
    case KOP_JNE: case KOP_CMOVNE: return 0x5u;
    case KOP_JB:  case KOP_CMOVB:  return 0x2u;
    case KOP_JAE: case KOP_CMOVAE: return 0x3u;
    case KOP_JBE: case KOP_CMOVBE: return 0x6u;
    case KOP_JA:  case KOP_CMOVA:  return 0x7u;
    case KOP_JL:  case KOP_CMOVL:  return 0xCu;
    case KOP_JGE: case KOP_CMOVGE: return 0xDu;
    case KOP_JLE: case KOP_CMOVLE: return 0xEu;
    case KOP_JG:  case KOP_CMOVG:  return 0xFu;
    default:                       return 0x4u;
    }
}

static int kbpf_next_live_pc(const KBpfAnalysis *an, int pc, int n) {
    for (int k = pc + 1; k < n; k++) {
        if (an->reachable[k] && !an->dead[k]) return k;
    }
    return -1;
}

static size_t kbpf_cmp_byte_len(const KBpfInsn *in) {
    if (in->op == KOP_CMP_REG) return 3;
    if (in->imm32 == 0) return 3;
    if (in->imm32 >= -128 && in->imm32 <= 127) return 4;
    return (k_hw_reg[in->dst] == 0) ? 6 : 7;
}

static inline void emit_jcc_erratum_guard(JitBuf *jb, size_t insn_bytes, int *pads) {
    if (!g_cpu.is_intel) return;
    size_t start = jb->len;
    size_t end   = start + insn_bytes;
    if ((end & 31u) == 0u || (start >> 5) != ((end - 1u) >> 5)) {
        size_t pad = 32u - (start & 31u);
        if (pad < 32u) { emit_nops(jb, pad); (*pads)++; }
    }
}

static void kbpf_emit_pass(const KBpfInsn *prog, int n, uint32_t caps,
                           const KBpfAnalysis *an,
                           const uint8_t *is_short,
                           JitBuf *jb,
                           size_t *pc_off,
                           size_t *out_fuel_abort_off,
                           JitPatch *patches,
                           int *out_npatches,
                           int *out_peephole,
                           int *out_dce_elided,
                           int *out_pruned_zeros,
                           int *out_jcc_pads) {
    jb->len      = 0;
    jb->overflow = 0;
    int npatches   = 0;
    int peephole   = 0;
    int dce_elided = 0;
    int pruned     = 0;
    int jcc_pads   = 0;
    int bounded_mem  = ((caps & KBPF_CAP_MEM_BOUNDED) != 0u);
    int bounded_loop = ((caps & KBPF_CAP_BOUNDED_LOOP) != 0u) && (an->backedges > 0);
    int prev_fused_jcc = 0;

    emit8(jb, 0xF3); emit8(jb, 0x0F); emit8(jb, 0x1E); emit8(jb, 0xFA); /* endbr64 */
    static const uint8_t non_arg_regs[5] = { 0, 4, 5, 6, 7 };
    for (int i = 0; i < 5; i++) {
        uint8_t kr = non_arg_regs[i];
        if (an->prologue_zero_mask & (1u << kr)) emit_xor_zero32(jb, k_hw_reg[kr]);
        else pruned++;
    }
    if (bounded_loop) {
        /* movl $KBPF_LOOP_FUEL_INIT, -24(%rsp) (below UDIV push slots, inside 128B red zone) */
        emit8(jb, 0xC7); emit8(jb, 0x44); emit8(jb, 0x24); emit8(jb, 0xE8);
        emit32(jb, KBPF_LOOP_FUEL_INIT);
    }

    for (int pc = 0; pc < n; pc++) {
        if (!an->reachable[pc] || an->dead[pc]) {
            pc_off[pc] = jb->len;
            if (an->reachable[pc] && an->dead[pc]) dce_elided++;
            else peephole++;
            prev_fused_jcc = 0;
            continue;
        }
        if (an->is_loop_hdr[pc]) emit_align16_nops(jb);

        const KBpfInsn *in = &prog[pc];

        if ((in->op == KOP_CMP_REG || in->op == KOP_CMP_IMM) &&
            pc + 1 < n && an->reachable[pc + 1] && !an->dead[pc + 1] &&
            !an->is_jump_target[pc + 1] && kbpf_is_jcc(prog[pc + 1].op)) {
            int j_back = bounded_loop && (prog[pc + 1].imm32 < pc + 1);
            size_t jlen = j_back ? 2u : (is_short[pc + 1] ? 2u : 6u);
            emit_jcc_erratum_guard(jb, kbpf_cmp_byte_len(in) + jlen, &jcc_pads);
            prev_fused_jcc = 1;
        } else {
            if (!prev_fused_jcc) {
                if (kbpf_is_jcc(in->op)) {
                    int j_back = bounded_loop && (in->imm32 < pc);
                    emit_jcc_erratum_guard(jb, j_back ? 2u : (is_short[pc] ? 2u : 6u), &jcc_pads);
                } else if (in->op == KOP_JMP && in->imm32 != kbpf_next_live_pc(an, pc, n)) {
                    if (!(bounded_loop && in->imm32 < pc)) {
                        emit_jcc_erratum_guard(jb, is_short[pc] ? 2u : 5u, &jcc_pads);
                    }
                } else if (in->op == KOP_RET) {
                    emit_jcc_erratum_guard(jb, 1u, &jcc_pads);
                }
            }
            prev_fused_jcc = 0;
        }

        pc_off[pc] = jb->len;

        uint8_t hd = k_hw_reg[in->dst];
        uint8_t hs = k_hw_reg[in->src];
        uint8_t hx = k_hw_reg[in->idx];

        /* Expanded 2-instruction fusions starting from MOV dst, src */
        if (in->op == KOP_MOV_REG && pc + 1 < n && an->reachable[pc + 1] && !an->dead[pc + 1] &&
            !an->is_jump_target[pc + 1] && prog[pc + 1].dst == in->dst) {
            const KBpfInsn *nx = &prog[pc + 1];
            if (nx->op == KOP_ADD_IMM && nx->imm32 != 0 && !an->flags_live_after[pc + 1]) {
                emit_lea_disp(jb, hd, hs, nx->imm32);
                pc_off[pc + 1] = jb->len; pc++; peephole++; continue;
            }
            if (nx->op == KOP_SUB_IMM && nx->imm32 != 0 && nx->imm32 != INT32_MIN &&
                !an->flags_live_after[pc + 1]) {
                emit_lea_disp(jb, hd, hs, -nx->imm32);
                pc_off[pc + 1] = jb->len; pc++; peephole++; continue;
            }
            if (nx->op == KOP_ADD_REG && !an->flags_live_after[pc + 1]) {
                uint8_t h2 = (nx->src == in->dst) ? hs : k_hw_reg[nx->src];
                emit_lea_scale_pair(jb, hd, hs, h2, 0u);
                pc_off[pc + 1] = jb->len; pc++; peephole++; continue;
            }
            if (nx->op == KOP_MUL_IMM && (nx->imm32 == 3 || nx->imm32 == 5 || nx->imm32 == 9) &&
                !an->flags_live_after[pc + 1]) {
                uint8_t sc = (nx->imm32 == 3) ? 1u : ((nx->imm32 == 5) ? 2u : 3u);
                emit_lea_scale_pair(jb, hd, hs, hs, sc);
                pc_off[pc + 1] = jb->len; pc++; peephole++; continue;
            }
            if (nx->op == KOP_ROR_IMM && g_cpu.has_bmi2 && nx->imm32 != 0) {
                emit_bmi2_rorx(jb, hd, hs, (uint8_t)(nx->imm32 & 63));
                pc_off[pc + 1] = jb->len; pc++; peephole++; continue;
            }
        }

        switch (in->op) {
        case KOP_MOV_REG:
            if (hd == hs) { peephole++; break; }
            emit8(jb, jit_rex(1, hs, 0, hd));
            emit8(jb, 0x89);
            emit8(jb, jit_modrm(3, hs, hd));
            break;
        case KOP_MOV_IMM:
            if (in->imm32 == 0 && !an->flags_live_after[pc]) {
                emit_xor_zero32(jb, hd); peephole++;
            } else if (in->imm32 >= 0) {
                if (hd >= 8) emit8(jb, jit_rex(0, 0, 0, hd));
                emit8(jb, (uint8_t)(0xB8u | (hd & 7u)));
                emit32(jb, (uint32_t)in->imm32);
            } else {
                emit8(jb, jit_rex(1, 0, 0, hd)); emit8(jb, 0xC7);
                emit8(jb, jit_modrm(3, 0, hd)); emit32(jb, (uint32_t)in->imm32);
            }
            break;
        case KOP_MOV_IMM64:
            if (in->imm64 == 0 && !an->flags_live_after[pc]) {
                emit_xor_zero32(jb, hd); peephole++;
            } else if (in->imm64 <= 0xFFFFFFFFULL) {
                if (hd >= 8) emit8(jb, jit_rex(0, 0, 0, hd));
                emit8(jb, (uint8_t)(0xB8u | (hd & 7u)));
                emit32(jb, (uint32_t)in->imm64);
                peephole++;
            } else if ((int64_t)in->imm64 >= INT32_MIN && (int64_t)in->imm64 < 0) {
                emit8(jb, jit_rex(1, 0, 0, hd)); emit8(jb, 0xC7);
                emit8(jb, jit_modrm(3, 0, hd)); emit32(jb, (uint32_t)(int32_t)(int64_t)in->imm64);
                peephole++;
            } else {
                emit8(jb, jit_rex(1, 0, 0, hd));
                emit8(jb, (uint8_t)(0xB8u | (hd & 7u)));
                emit64(jb, in->imm64);
            }
            break;
        case KOP_ADD_REG:
            emit8(jb, jit_rex(1, hs, 0, hd)); emit8(jb, 0x01); emit8(jb, jit_modrm(3, hs, hd));
            break;
        case KOP_ADD_IMM:
            if (in->imm32 == 0) { peephole++; break; }
            emit_alu_imm64(jb, 0, hd, in->imm32);
            break;
        case KOP_SUB_REG:
            if (hd == hs) { emit_xor_zero32(jb, hd); peephole++; break; }
            emit8(jb, jit_rex(1, hs, 0, hd)); emit8(jb, 0x29); emit8(jb, jit_modrm(3, hs, hd));
            break;
        case KOP_SUB_IMM:
            if (in->imm32 == 0) { peephole++; break; }
            emit_alu_imm64(jb, 5, hd, in->imm32);
            break;
        case KOP_MUL_REG:
            emit8(jb, jit_rex(1, hd, 0, hs)); emit8(jb, 0x0F); emit8(jb, 0xAF);
            emit8(jb, jit_modrm(3, hd, hs));
            break;
        case KOP_MUL_IMM:
            if (in->imm32 == 1) { peephole++; break; }
            if (in->imm32 == 0) { emit_xor_zero32(jb, hd); peephole++; break; }
            if (in->imm32 == -1) {
                emit8(jb, jit_rex(1, 0, 0, hd)); emit8(jb, 0xF7); emit8(jb, jit_modrm(3, 3, hd));
                peephole++; break;
            }
            if (in->imm32 == 2) {
                emit8(jb, jit_rex(1, hd, 0, hd)); emit8(jb, 0x01); emit8(jb, jit_modrm(3, hd, hd));
                peephole++; break;
            }
            if (in->imm32 == 3 || in->imm32 == 5 || in->imm32 == 9) {
                uint8_t sc = (in->imm32 == 3) ? 1u : ((in->imm32 == 5) ? 2u : 3u);
                emit_lea_scale_pair(jb, hd, hd, hd, sc); peephole++; break;
            }
            if (in->imm32 == 6 || in->imm32 == 10 || in->imm32 == 12) {
                uint8_t sc = (in->imm32 == 10) ? 2u : 1u;
                emit_lea_scale_pair(jb, hd, hd, hd, sc);
                if (in->imm32 == 12) {
                    emit8(jb, jit_rex(1, 0, 0, hd)); emit8(jb, 0xC1);
                    emit8(jb, jit_modrm(3, 4, hd)); emit8(jb, 2);
                } else {
                    emit8(jb, jit_rex(1, hd, 0, hd)); emit8(jb, 0x01); emit8(jb, jit_modrm(3, hd, hd));
                }
                peephole++; break;
            }
            if (in->imm32 > 2 && (((uint32_t)in->imm32 & ((uint32_t)in->imm32 - 1u)) == 0u)) {
                uint8_t sh = (uint8_t)__builtin_ctz((unsigned)in->imm32);
                emit8(jb, jit_rex(1, 0, 0, hd)); emit8(jb, 0xC1);
                emit8(jb, jit_modrm(3, 4, hd)); emit8(jb, sh);
                peephole++; break;
            }
            emit8(jb, jit_rex(1, hd, 0, hd));
            if (in->imm32 >= -128 && in->imm32 <= 127) {
                emit8(jb, 0x6B); emit8(jb, jit_modrm(3, hd, hd)); emit8(jb, (uint8_t)(int8_t)in->imm32);
            } else {
                emit8(jb, 0x69); emit8(jb, jit_modrm(3, hd, hd)); emit32(jb, (uint32_t)in->imm32);
            }
            break;
        case KOP_UDIV_REG:
        case KOP_UMOD_REG: {
            int save_rax = (hd != 0) && ((an->regs_live_after[pc] & (1u << 0)) != 0u);
            int save_rdx = (hd != 2) && ((an->regs_live_after[pc] & (1u << 3)) != 0u);
            if (!save_rax && hd != 0) peephole++;
            if (!save_rdx && hd != 2) peephole++;

            emit8(jb, jit_rex(1, hs, 0, 1)); emit8(jb, 0x89); emit8(jb, jit_modrm(3, hs, 1));
            if (save_rax) emit8(jb, 0x50);
            if (save_rdx) emit8(jb, 0x52);
            emit_jcc_erratum_guard(jb, 5u, &jcc_pads);
            emit8(jb, 0x48); emit8(jb, 0x85); emit8(jb, 0xC9);
            emit8(jb, 0x74); size_t jz_off = jb->len; emit8(jb, 0);
            if (hd != 0) {
                emit8(jb, jit_rex(1, hd, 0, 0)); emit8(jb, 0x89); emit8(jb, jit_modrm(3, hd, 0));
            }
            emit8(jb, 0x31); emit8(jb, 0xD2);
            emit8(jb, 0x48); emit8(jb, 0xF7); emit8(jb, 0xF1);
            uint8_t res_reg = (in->op == KOP_UDIV_REG) ? 0u : 2u;
            emit8(jb, jit_rex(1, res_reg, 0, 1)); emit8(jb, 0x89); emit8(jb, jit_modrm(3, res_reg, 1));
            emit_jcc_erratum_guard(jb, 2u, &jcc_pads);
            emit8(jb, 0xEB); size_t jmp_off = jb->len; emit8(jb, 0);
            size_t div0_off = jb->len;
            emit8(jb, 0x31); emit8(jb, 0xC9);
            size_t done_off = jb->len;
            if (jb->buf) {
                jb->buf[jz_off]  = (uint8_t)(int8_t)(div0_off - (jz_off + 1));
                jb->buf[jmp_off] = (uint8_t)(int8_t)(done_off - (jmp_off + 1));
            }
            if (save_rdx) emit8(jb, 0x5A);
            if (save_rax) emit8(jb, 0x58);
            emit8(jb, jit_rex(1, 1, 0, hd)); emit8(jb, 0x89); emit8(jb, jit_modrm(3, 1, hd));
            break;
        }
        case KOP_NEG:
            emit8(jb, jit_rex(1, 0, 0, hd)); emit8(jb, 0xF7); emit8(jb, jit_modrm(3, 3, hd));
            break;
        case KOP_BSWAP64:
            emit8(jb, jit_rex(1, 0, 0, hd)); emit8(jb, 0x0F); emit8(jb, (uint8_t)(0xC8u | (hd & 7u)));
            break;
        case KOP_POPCNT:
            if (hd != hs) emit_xor_zero32(jb, hd);
            emit8(jb, 0xF3); emit8(jb, jit_rex(1, hd, 0, hs));
            emit8(jb, 0x0F); emit8(jb, 0xB8); emit8(jb, jit_modrm(3, hd, hs));
            break;
        case KOP_LZCNT:
            if (hd != hs) emit_xor_zero32(jb, hd);
            emit8(jb, 0xF3); emit8(jb, jit_rex(1, hd, 0, hs));
            emit8(jb, 0x0F); emit8(jb, 0xBD); emit8(jb, jit_modrm(3, hd, hs));
            break;
        case KOP_TZCNT:
            if (hd != hs) emit_xor_zero32(jb, hd);
            emit8(jb, 0xF3); emit8(jb, jit_rex(1, hd, 0, hs));
            emit8(jb, 0x0F); emit8(jb, 0xBC); emit8(jb, jit_modrm(3, hd, hs));
            break;
        case KOP_AND_REG:
            emit8(jb, jit_rex(1, hs, 0, hd)); emit8(jb, 0x21); emit8(jb, jit_modrm(3, hs, hd));
            /* Fuse AND_REG hd, hs + CMP_IMM hd, 0 (AND already sets exact ZF/SF/OF=0/CF=0!) */
            if (pc + 1 < n && an->reachable[pc + 1] && !an->dead[pc + 1] && !an->is_jump_target[pc + 1] &&
                prog[pc + 1].op == KOP_CMP_IMM && prog[pc + 1].dst == in->dst &&
                prog[pc + 1].imm32 == 0) {
                pc_off[pc + 1] = jb->len; pc++; peephole++;
            }
            break;
        case KOP_AND_IMM: {
            int emitted_flags = 1;
            if (in->imm32 == -1) {
                peephole++;
                emitted_flags = 0; /* No instruction emitted -> flags NOT set for dst! */
            } else if (in->imm32 == 0) {
                emit_xor_zero32(jb, hd); peephole++;
            } else if (in->imm32 > 0) {
                if (hd >= 8) emit8(jb, jit_rex(0, 0, 0, hd));
                if (in->imm32 <= 127) {
                    emit8(jb, 0x83); emit8(jb, jit_modrm(3, 4, hd)); emit8(jb, (uint8_t)in->imm32);
                } else if (hd == 0) {
                    emit8(jb, 0x25); emit32(jb, (uint32_t)in->imm32);
                } else {
                    emit8(jb, 0x81); emit8(jb, jit_modrm(3, 4, hd)); emit32(jb, (uint32_t)in->imm32);
                }
                peephole++;
            } else {
                emit_alu_imm64(jb, 4, hd, in->imm32);
            }
            if (emitted_flags && pc + 1 < n && an->reachable[pc + 1] && !an->dead[pc + 1] &&
                !an->is_jump_target[pc + 1] && prog[pc + 1].op == KOP_CMP_IMM &&
                prog[pc + 1].dst == in->dst && prog[pc + 1].imm32 == 0) {
                pc_off[pc + 1] = jb->len; pc++; peephole++;
            }
            break;
        }
        case KOP_OR_REG:
            emit8(jb, jit_rex(1, hs, 0, hd)); emit8(jb, 0x09); emit8(jb, jit_modrm(3, hs, hd));
            break;
        case KOP_OR_IMM:
            if (in->imm32 == 0) { peephole++; break; }
            emit_alu_imm64(jb, 1, hd, in->imm32);
            break;
        case KOP_XOR_REG:
            if (hd == hs) { emit_xor_zero32(jb, hd); peephole++; break; }
            emit8(jb, jit_rex(1, hs, 0, hd)); emit8(jb, 0x31); emit8(jb, jit_modrm(3, hs, hd));
            break;
        case KOP_XOR_IMM:
            if (in->imm32 == 0) { peephole++; break; }
            emit_alu_imm64(jb, 6, hd, in->imm32);
            break;
        case KOP_SHL_IMM:
            if (in->imm32 == 0) { peephole++; break; }
            emit8(jb, jit_rex(1, 0, 0, hd));
            if (in->imm32 == 1) { emit8(jb, 0xD1); emit8(jb, jit_modrm(3, 4, hd)); peephole++; }
            else { emit8(jb, 0xC1); emit8(jb, jit_modrm(3, 4, hd)); emit8(jb, (uint8_t)(in->imm32 & 63)); }
            break;
        case KOP_SHR_IMM:
            if (in->imm32 == 0) { peephole++; break; }
            emit8(jb, jit_rex(1, 0, 0, hd));
            if (in->imm32 == 1) { emit8(jb, 0xD1); emit8(jb, jit_modrm(3, 5, hd)); peephole++; }
            else { emit8(jb, 0xC1); emit8(jb, jit_modrm(3, 5, hd)); emit8(jb, (uint8_t)(in->imm32 & 63)); }
            break;
        case KOP_SAR_IMM:
            if (in->imm32 == 0) { peephole++; break; }
            emit8(jb, jit_rex(1, 0, 0, hd));
            if (in->imm32 == 1) { emit8(jb, 0xD1); emit8(jb, jit_modrm(3, 7, hd)); peephole++; }
            else { emit8(jb, 0xC1); emit8(jb, jit_modrm(3, 7, hd)); emit8(jb, (uint8_t)(in->imm32 & 63)); }
            break;
        case KOP_ROR_IMM:
            if (in->imm32 == 0) { peephole++; break; }
            if (g_cpu.has_bmi2) {
                emit_bmi2_rorx(jb, hd, hd, (uint8_t)(in->imm32 & 63));
                peephole++;
            } else {
                emit8(jb, jit_rex(1, 0, 0, hd));
                if (in->imm32 == 1) { emit8(jb, 0xD1); emit8(jb, jit_modrm(3, 1, hd)); peephole++; }
                else { emit8(jb, 0xC1); emit8(jb, jit_modrm(3, 1, hd)); emit8(jb, (uint8_t)(in->imm32 & 63)); }
            }
            break;
        case KOP_SHL_REG:
        case KOP_SHR_REG:
        case KOP_SAR_REG:
            if (g_cpu.has_bmi2) {
                uint8_t pp = (in->op == KOP_SHL_REG) ? 1u : ((in->op == KOP_SAR_REG) ? 2u : 3u);
                emit_bmi2_shiftx(jb, hd, hs, pp);
                peephole++;
            } else {
                uint8_t ext = (in->op == KOP_SHL_REG) ? 4u : ((in->op == KOP_SHR_REG) ? 5u : 7u);
                emit8(jb, jit_rex(1, hs, 0, 1)); emit8(jb, 0x89); emit8(jb, jit_modrm(3, hs, 1));
                emit8(jb, jit_rex(1, 0, 0, hd)); emit8(jb, 0xD3); emit8(jb, jit_modrm(3, ext, hd));
            }
            break;
        case KOP_LOAD64:
        case KOP_LOAD32: {
            int is64   = (in->op == KOP_LOAD64);
            uint8_t sc = is64 ? 3u : 2u;
            if (bounded_mem) {
                emit8(jb, 0x48); emit8(jb, 0x89); emit8(jb, 0xD1);
                emit8(jb, 0x48); emit8(jb, 0xC1); emit8(jb, 0xE9); emit8(jb, sc);
                emit_jcc_erratum_guard(jb, 5u, &jcc_pads);
                emit8(jb, jit_rex(1, 1, 0, hx)); emit8(jb, 0x39); emit8(jb, jit_modrm(3, 1, hx));
                emit8(jb, 0x73); size_t jae_off = jb->len; emit8(jb, 0);
                emit8(jb, 0x0F); emit8(jb, 0xAE); emit8(jb, 0xE8);
                if (is64 || hd >= 8 || hx >= 8 || hs >= 8) emit8(jb, jit_rex(is64, hd, hx, hs));
                emit8(jb, 0x8B); emit8(jb, jit_modrm(0, hd, 4));
                emit8(jb, (uint8_t)((sc << 6) | ((hx & 7u) << 3) | (hs & 7u)));
                emit_jcc_erratum_guard(jb, 2u, &jcc_pads);
                emit8(jb, 0xEB); size_t jmp_off = jb->len; emit8(jb, 0);
                size_t oob_off = jb->len;
                emit_xor_zero32(jb, hd);
                size_t ok_off = jb->len;
                if (jb->buf) {
                    jb->buf[jae_off] = (uint8_t)(int8_t)(oob_off - (jae_off + 1));
                    jb->buf[jmp_off] = (uint8_t)(int8_t)(ok_off - (jmp_off + 1));
                }
            } else {
                if (is64 || hd >= 8 || hx >= 8 || hs >= 8) emit8(jb, jit_rex(is64, hd, hx, hs));
                emit8(jb, 0x8B); emit8(jb, jit_modrm(0, hd, 4));
                emit8(jb, (uint8_t)((sc << 6) | ((hx & 7u) << 3) | (hs & 7u)));
            }
            break;
        }
        case KOP_STORE64:
        case KOP_STORE32: {
            int is64   = (in->op == KOP_STORE64);
            uint8_t sc = is64 ? 3u : 2u;
            if (bounded_mem) {
                emit8(jb, 0x48); emit8(jb, 0x89); emit8(jb, 0xD1);
                emit8(jb, 0x48); emit8(jb, 0xC1); emit8(jb, 0xE9); emit8(jb, sc);
                emit_jcc_erratum_guard(jb, 5u, &jcc_pads);
                emit8(jb, jit_rex(1, 1, 0, hx)); emit8(jb, 0x39); emit8(jb, jit_modrm(3, 1, hx));
                emit8(jb, 0x73); size_t jae_off = jb->len; emit8(jb, 0);
                emit8(jb, 0x0F); emit8(jb, 0xAE); emit8(jb, 0xE8);
                if (is64 || hd >= 8 || hx >= 8 || hs >= 8) emit8(jb, jit_rex(is64, hd, hx, hs));
                emit8(jb, 0x89); emit8(jb, jit_modrm(0, hd, 4));
                emit8(jb, (uint8_t)((sc << 6) | ((hx & 7u) << 3) | (hs & 7u)));
                size_t ok_off = jb->len;
                if (jb->buf) jb->buf[jae_off] = (uint8_t)(int8_t)(ok_off - (jae_off + 1));
            } else {
                if (is64 || hd >= 8 || hx >= 8 || hs >= 8) emit8(jb, jit_rex(is64, hd, hx, hs));
                emit8(jb, 0x89); emit8(jb, jit_modrm(0, hd, 4));
                emit8(jb, (uint8_t)((sc << 6) | ((hx & 7u) << 3) | (hs & 7u)));
            }
            break;
        }
        case KOP_CMP_REG:
            emit8(jb, jit_rex(1, hs, 0, hd)); emit8(jb, 0x39); emit8(jb, jit_modrm(3, hs, hd));
            break;
        case KOP_CMP_IMM:
            if (in->imm32 == 0) {
                emit8(jb, jit_rex(1, hd, 0, hd)); emit8(jb, 0x85); emit8(jb, jit_modrm(3, hd, hd));
                peephole++;
            } else {
                emit_alu_imm64(jb, 7, hd, in->imm32);
            }
            break;
        case KOP_CMOVE:  case KOP_CMOVNE: case KOP_CMOVB:  case KOP_CMOVAE:
        case KOP_CMOVBE: case KOP_CMOVA:  case KOP_CMOVL:  case KOP_CMOVGE:
        case KOP_CMOVLE: case KOP_CMOVG:
            if (hd == hs) { peephole++; break; }
            emit8(jb, jit_rex(1, hd, 0, hs));
            emit8(jb, 0x0F); emit8(jb, (uint8_t)(0x40u | kbpf_cc_nibble(in->op)));
            emit8(jb, jit_modrm(3, hd, hs));
            break;
        case KOP_JMP:
            if (in->imm32 == kbpf_next_live_pc(an, pc, n)) { peephole++; break; }
            if (bounded_loop && in->imm32 < pc) {
                /* subl $1, -24(%rsp); jz rel32 fuel_abort */
                emit_jcc_erratum_guard(jb, 11u, &jcc_pads);
                emit8(jb, 0x83); emit8(jb, 0x6C); emit8(jb, 0x24); emit8(jb, 0xE8); emit8(jb, 0x01);
                emit8(jb, 0x0F); emit8(jb, 0x84);
                patches[npatches++] = (JitPatch){ jb->len, pc, -1, 0 };
                emit32(jb, 0);
                emit_jcc_erratum_guard(jb, is_short[pc] ? 2u : 5u, &jcc_pads);
            }
            if (is_short[pc]) {
                emit8(jb, 0xEB);
                patches[npatches++] = (JitPatch){ jb->len, pc, in->imm32, 1 };
                emit8(jb, 0);
            } else {
                emit8(jb, 0xE9);
                patches[npatches++] = (JitPatch){ jb->len, pc, in->imm32, 0 };
                emit32(jb, 0);
            }
            break;
        case KOP_JE:  case KOP_JNE: case KOP_JB:  case KOP_JAE:
        case KOP_JBE: case KOP_JA:  case KOP_JL:  case KOP_JGE:
        case KOP_JLE: case KOP_JG: {
            uint8_t cc_lo = kbpf_cc_nibble(in->op);
            if (bounded_loop && in->imm32 < pc) {
                /* Conditional backedge: J<inv_cc> skip; subl $1,-24(%rsp); jz fuel_abort; jmp target */
                emit8(jb, (uint8_t)(0x70u | (cc_lo ^ 1u)));
                size_t skip_off = jb->len; emit8(jb, 0);
                emit_jcc_erratum_guard(jb, 11u, &jcc_pads);
                emit8(jb, 0x83); emit8(jb, 0x6C); emit8(jb, 0x24); emit8(jb, 0xE8); emit8(jb, 0x01);
                emit8(jb, 0x0F); emit8(jb, 0x84);
                patches[npatches++] = (JitPatch){ jb->len, pc, -1, 0 };
                emit32(jb, 0);
                emit_jcc_erratum_guard(jb, is_short[pc] ? 2u : 5u, &jcc_pads);
                if (is_short[pc]) {
                    emit8(jb, 0xEB);
                    patches[npatches++] = (JitPatch){ jb->len, pc, in->imm32, 1 };
                    emit8(jb, 0);
                } else {
                    emit8(jb, 0xE9);
                    patches[npatches++] = (JitPatch){ jb->len, pc, in->imm32, 0 };
                    emit32(jb, 0);
                }
                if (jb->buf) jb->buf[skip_off] = (uint8_t)(int8_t)(jb->len - (skip_off + 1));
            } else if (is_short[pc]) {
                emit8(jb, (uint8_t)(0x70u | cc_lo));
                patches[npatches++] = (JitPatch){ jb->len, pc, in->imm32, 1 };
                emit8(jb, 0);
            } else {
                emit8(jb, 0x0F); emit8(jb, (uint8_t)(0x80u | cc_lo));
                patches[npatches++] = (JitPatch){ jb->len, pc, in->imm32, 0 };
                emit32(jb, 0);
            }
            break;
        }
        case KOP_RET:
            emit8(jb, 0xC3);
            break;
        }
    }

    *out_fuel_abort_off = jb->len;
    if (bounded_loop) {
        /* fuel_abort: movabsq $KBPF_FUEL_EXHAUSTED, %rax; ret */
        emit8(jb, 0x48); emit8(jb, 0xB8); emit64(jb, KBPF_FUEL_EXHAUSTED);
        emit8(jb, 0xC3);
    }

    *out_npatches     = npatches;
    *out_peephole     = peephole;
    *out_dce_elided   = dce_elided;
    *out_pruned_zeros = pruned;
    *out_jcc_pads     = jcc_pads;
}

/* Strict W^X PROT_NONE page recycler (zero dangling RX gadgets while cached!) */
static void *jit_acquire_rw_pages(size_t alloc_sz) {
    if (alloc_sz == KRONOS_PAGE_SZ) {
        kronos_preempt_disable();
        tlock_lock(&g_kern.jit_cache_lock);
        if (g_kern.jit_page_count > 0) {
            void *p = g_kern.jit_page_cache[--g_kern.jit_page_count];
            tlock_unlock(&g_kern.jit_cache_lock);
            kronos_preempt_enable();
            if (mprotect(p, KRONOS_PAGE_SZ, PROT_READ | PROT_WRITE) == 0) return p;
            (void)munmap(p, KRONOS_PAGE_SZ);
        } else {
            tlock_unlock(&g_kern.jit_cache_lock);
            kronos_preempt_enable();
        }
    }
    void *mem = mmap(NULL, alloc_sz, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (mem == MAP_FAILED) ? NULL : mem;
}

static void jit_release_pages(void *mem, size_t alloc_sz) {
    if (!mem) return;
    if (alloc_sz == KRONOS_PAGE_SZ) {
        if (mprotect(mem, KRONOS_PAGE_SZ, PROT_READ | PROT_WRITE) == 0) {
            memset(mem, 0xCC, KRONOS_PAGE_SZ);
            if (mprotect(mem, KRONOS_PAGE_SZ, PROT_NONE) == 0) {
                kronos_preempt_disable();
                tlock_lock(&g_kern.jit_cache_lock);
                if (g_kern.jit_page_count < KRONOS_JIT_CACHE_MAX) {
                    g_kern.jit_page_cache[g_kern.jit_page_count++] = mem;
                    tlock_unlock(&g_kern.jit_cache_lock);
                    kronos_preempt_enable();
                    return;
                }
                tlock_unlock(&g_kern.jit_cache_lock);
                kronos_preempt_enable();
            }
        }
    }
    (void)munmap(mem, alloc_sz);
}

static int kbpf_jit_compile(const KBpfInsn *prog, int n, uint32_t caps,
                            KBpfCompiled *out, char *err_buf, size_t err_sz) {
    memset(out, 0, sizeof(*out));
    out->code_range_slot = -1;
    KBpfAnalysis an;
    if (kbpf_verify(prog, n, caps, err_buf, err_sz, &an) != 0) return -1;

    uint8_t is_short[KBPF_MAX_INSNS];
    for (int pc = 0; pc < n; pc++) is_short[pc] = 1;

    size_t pc_off[KBPF_MAX_INSNS];
    size_t fuel_abort_off = 0;
    JitPatch patches[KBPF_MAX_INSNS * 2];
    int npatches = 0, peephole = 0, dce_elided = 0, pruned = 0, jcc_pads = 0;

    JitBuf dry = { NULL, 0, 16368, 0 };
    for (int iter = 0; iter <= n + 1; iter++) {
        kbpf_emit_pass(prog, n, caps, &an, is_short, &dry,
                       pc_off, &fuel_abort_off, patches, &npatches,
                       &peephole, &dce_elided, &pruned, &jcc_pads);
        if (dry.overflow) {
            snprintf(err_buf, err_sz, "JIT code buffer overflow (> %zu bytes)", dry.cap);
            return -1;
        }
        int relaxed_any = 0;
        for (int i = 0; i < npatches; i++) {
            if (patches[i].is_short) {
                size_t t_off = (patches[i].target_pc < 0) ? fuel_abort_off
                                                          : pc_off[patches[i].target_pc];
                intptr_t rel = (intptr_t)t_off - (intptr_t)(patches[i].patch_off + 1);
                if (rel < -128 || rel > 127) {
                    is_short[patches[i].src_pc] = 0;
                    relaxed_any = 1;
                }
            }
        }
        if (!relaxed_any) break;
    }

    size_t alloc_sz = (dry.len + 64u + KRONOS_PAGE_SZ - 1u) & ~(size_t)(KRONOS_PAGE_SZ - 1u);
    void *mem = jit_acquire_rw_pages(alloc_sz);
    if (!mem) {
        snprintf(err_buf, err_sz, "mmap RW code page failed");
        return -1;
    }
    memset(mem, 0xCC, alloc_sz);

    JitBuf jb = { (uint8_t *)mem, 0, alloc_sz, 0 };
    kbpf_emit_pass(prog, n, caps, &an, is_short, &jb,
                   pc_off, &fuel_abort_off, patches, &npatches,
                   &peephole, &dce_elided, &pruned, &jcc_pads);
    if (jb.overflow) {
        jit_release_pages(mem, alloc_sz);
        snprintf(err_buf, err_sz, "JIT final emit overflow");
        return -1;
    }

    int n_short = 0, n_long = 0;
    for (int i = 0; i < npatches; i++) {
        size_t p_off = patches[i].patch_off;
        size_t t_off = (patches[i].target_pc < 0) ? fuel_abort_off : pc_off[patches[i].target_pc];
        if (patches[i].is_short) {
            intptr_t rel = (intptr_t)t_off - (intptr_t)(p_off + 1);
            if (UNLIKELY(rel < -128 || rel > 127)) {
                jit_release_pages(mem, alloc_sz);
                snprintf(err_buf, err_sz, "internal rel8 relaxation invariant error");
                return -1;
            }
            jb.buf[p_off] = (uint8_t)(int8_t)rel;
            n_short++;
        } else {
            int32_t rel = (int32_t)((intptr_t)t_off - (intptr_t)(p_off + 4));
            memcpy(jb.buf + p_off, &rel, 4);
            n_long++;
        }
    }

    if (mprotect(mem, alloc_sz, PROT_READ | PROT_EXEC) != 0) {
        (void)munmap(mem, alloc_sz);
        snprintf(err_buf, err_sz, "mprotect RX W^X seal failed");
        return -1;
    }

    int slot = kronos_register_code_range((uintptr_t)mem, (uintptr_t)mem + jb.len);
    if (UNLIKELY(slot < 0 && g_kern.preempt_supported)) {
        jit_release_pages(mem, alloc_sz);
        snprintf(err_buf, err_sz, "preemptible code_ranges table full (%d max)", KRONOS_MAX_CODE_RANGES);
        return -1;
    }

    out->code_range_slot       = slot;
    out->code                  = mem;
    out->alloc_size            = alloc_sz;
    out->code_len              = jb.len;
    out->short_branches        = n_short;
    out->long_branches         = n_long;
    out->peephole_opts         = peephole;
    out->dce_elided            = dce_elided;
    out->pruned_prologue_zeros = pruned;
    out->jcc_erratum_pads      = jcc_pads;
    memcpy(&out->fn, &mem, sizeof(out->fn));
    return 0;
}

#if defined(__clang__)
#define KRONOS_NO_UBSAN_FUNC __attribute__((no_sanitize("function")))
#else
#define KRONOS_NO_UBSAN_FUNC
#endif

static inline KRONOS_NO_UBSAN_FUNC uint64_t kbpf_call_jit(KBpfJitFn fn,
                                                          uint64_t a0, uint64_t a1, uint64_t a2) {
    return fn(a0, a1, a2);
}

static void kbpf_jit_free(KBpfCompiled *c) {
    if (c && c->code) {
        kronos_unregister_code_range(c->code_range_slot);
        c->code_range_slot = -1;
        jit_release_pages(c->code, c->alloc_size);
        c->code = NULL;
        c->fn   = NULL;
    }
}

static uint64_t kbpf_exec_interp_unchecked(const KBpfInsn *prog, int n, uint32_t caps,
                                           uint64_t a0, uint64_t a1, uint64_t a2) {
    int bounded_mem  = ((caps & KBPF_CAP_MEM_BOUNDED) != 0u);
    int bounded_loop = ((caps & KBPF_CAP_BOUNDED_LOOP) != 0u);
    uint32_t fuel    = KBPF_LOOP_FUEL_INIT;
    uint64_t r[8] = { 0, a0, a1, a2, 0, 0, 0, 0 };
    uint64_t cmp_a = 0, cmp_b = 0;
    int pc = 0;

    while (pc >= 0 && pc < n) {
        int cur_pc = pc;
        const KBpfInsn *in = &prog[pc++];
        switch (in->op) {
        case KOP_MOV_REG:   r[in->dst] = r[in->src]; break;
        case KOP_MOV_IMM:   r[in->dst] = (uint64_t)(int64_t)in->imm32; break;
        case KOP_MOV_IMM64: r[in->dst] = in->imm64; break;
        case KOP_ADD_REG:   r[in->dst] += r[in->src]; break;
        case KOP_ADD_IMM:   r[in->dst] += (uint64_t)(int64_t)in->imm32; break;
        case KOP_SUB_REG:   r[in->dst] -= r[in->src]; break;
        case KOP_SUB_IMM:   r[in->dst] -= (uint64_t)(int64_t)in->imm32; break;
        case KOP_MUL_REG:   r[in->dst] *= r[in->src]; break;
        case KOP_MUL_IMM:   r[in->dst] *= (uint64_t)(int64_t)in->imm32; break;
        case KOP_UDIV_REG:  r[in->dst] = (r[in->src] == 0) ? 0ULL : (r[in->dst] / r[in->src]); break;
        case KOP_UMOD_REG:  r[in->dst] = (r[in->src] == 0) ? 0ULL : (r[in->dst] % r[in->src]); break;
        case KOP_NEG:       r[in->dst] = 0ULL - r[in->dst]; break;
        case KOP_BSWAP64:   r[in->dst] = __builtin_bswap64(r[in->dst]); break;
        case KOP_POPCNT:    r[in->dst] = (uint64_t)__builtin_popcountll(r[in->src]); break;
        case KOP_LZCNT:     r[in->dst] = (r[in->src] == 0) ? 64ULL : (uint64_t)__builtin_clzll(r[in->src]); break;
        case KOP_TZCNT:     r[in->dst] = (r[in->src] == 0) ? 64ULL : (uint64_t)__builtin_ctzll(r[in->src]); break;
        case KOP_AND_REG:   r[in->dst] &= r[in->src]; break;
        case KOP_AND_IMM:   r[in->dst] &= (uint64_t)(int64_t)in->imm32; break;
        case KOP_OR_REG:    r[in->dst] |= r[in->src]; break;
        case KOP_OR_IMM:    r[in->dst] |= (uint64_t)(int64_t)in->imm32; break;
        case KOP_XOR_REG:   r[in->dst] ^= r[in->src]; break;
        case KOP_XOR_IMM:   r[in->dst] ^= (uint64_t)(int64_t)in->imm32; break;
        case KOP_SHL_IMM:   r[in->dst] <<= (unsigned)(in->imm32 & 63); break;
        case KOP_SHR_IMM:   r[in->dst] >>= (unsigned)(in->imm32 & 63); break;
        case KOP_SAR_IMM: {
            unsigned s = (unsigned)(in->imm32 & 63);
            uint64_t v = r[in->dst];
            r[in->dst] = (s == 0) ? v : ((v >> s) | ((v & (1ULL << 63)) ? (~0ULL << (64u - s)) : 0ULL));
            break;
        }
        case KOP_ROR_IMM: {
            unsigned s = (unsigned)(in->imm32 & 63);
            r[in->dst] = (s == 0) ? r[in->dst] : ((r[in->dst] >> s) | (r[in->dst] << (64u - s)));
            break;
        }
        case KOP_SHL_REG:   r[in->dst] <<= (unsigned)(r[in->src] & 63ULL); break;
        case KOP_SHR_REG:   r[in->dst] >>= (unsigned)(r[in->src] & 63ULL); break;
        case KOP_SAR_REG: {
            unsigned s = (unsigned)(r[in->src] & 63ULL);
            uint64_t v = r[in->dst];
            r[in->dst] = (s == 0) ? v : ((v >> s) | ((v & (1ULL << 63)) ? (~0ULL << (64u - s)) : 0ULL));
            break;
        }
        case KOP_LOAD64:
            if (bounded_mem && r[in->idx] >= (r[3] >> 3)) { r[in->dst] = 0ULL; }
            else {
                uint64_t tmp = 0;
                memcpy(&tmp, (const uint8_t *)(uintptr_t)(r[in->src] + r[in->idx] * 8ULL), 8);
                r[in->dst] = tmp;
            }
            break;
        case KOP_STORE64:
            if (!bounded_mem || r[in->idx] < (r[3] >> 3)) {
                uint64_t tmp = r[in->dst];
                memcpy((uint8_t *)(uintptr_t)(r[in->src] + r[in->idx] * 8ULL), &tmp, 8);
            }
            break;
        case KOP_LOAD32:
            if (bounded_mem && r[in->idx] >= (r[3] >> 2)) { r[in->dst] = 0ULL; }
            else {
                uint32_t tmp = 0;
                memcpy(&tmp, (const uint8_t *)(uintptr_t)(r[in->src] + r[in->idx] * 4ULL), 4);
                r[in->dst] = (uint64_t)tmp;
            }
            break;
        case KOP_STORE32:
            if (!bounded_mem || r[in->idx] < (r[3] >> 2)) {
                uint32_t tmp = (uint32_t)r[in->dst];
                memcpy((uint8_t *)(uintptr_t)(r[in->src] + r[in->idx] * 4ULL), &tmp, 4);
            }
            break;
        case KOP_CMP_REG:   cmp_a = r[in->dst]; cmp_b = r[in->src]; break;
        case KOP_CMP_IMM:   cmp_a = r[in->dst]; cmp_b = (uint64_t)(int64_t)in->imm32; break;
        case KOP_CMOVE:     if (cmp_a == cmp_b) r[in->dst] = r[in->src]; break;
        case KOP_CMOVNE:    if (cmp_a != cmp_b) r[in->dst] = r[in->src]; break;
        case KOP_CMOVB:     if (cmp_a <  cmp_b) r[in->dst] = r[in->src]; break;
        case KOP_CMOVAE:    if (cmp_a >= cmp_b) r[in->dst] = r[in->src]; break;
        case KOP_CMOVBE:    if (cmp_a <= cmp_b) r[in->dst] = r[in->src]; break;
        case KOP_CMOVA:     if (cmp_a >  cmp_b) r[in->dst] = r[in->src]; break;
        case KOP_CMOVL:     if ((int64_t)cmp_a <  (int64_t)cmp_b) r[in->dst] = r[in->src]; break;
        case KOP_CMOVGE:    if ((int64_t)cmp_a >= (int64_t)cmp_b) r[in->dst] = r[in->src]; break;
        case KOP_CMOVLE:    if ((int64_t)cmp_a <= (int64_t)cmp_b) r[in->dst] = r[in->src]; break;
        case KOP_CMOVG:     if ((int64_t)cmp_a >  (int64_t)cmp_b) r[in->dst] = r[in->src]; break;
        case KOP_JMP:
        case KOP_JE:  case KOP_JNE: case KOP_JB:  case KOP_JAE:
        case KOP_JBE: case KOP_JA:  case KOP_JL:  case KOP_JGE:
        case KOP_JLE: case KOP_JG: {
            int taken = 0;
            switch (in->op) {
            case KOP_JMP: taken = 1; break;
            case KOP_JE:  taken = (cmp_a == cmp_b); break;
            case KOP_JNE: taken = (cmp_a != cmp_b); break;
            case KOP_JB:  taken = (cmp_a <  cmp_b); break;
            case KOP_JAE: taken = (cmp_a >= cmp_b); break;
            case KOP_JBE: taken = (cmp_a <= cmp_b); break;
            case KOP_JA:  taken = (cmp_a >  cmp_b); break;
            case KOP_JL:  taken = ((int64_t)cmp_a <  (int64_t)cmp_b); break;
            case KOP_JGE: taken = ((int64_t)cmp_a >= (int64_t)cmp_b); break;
            case KOP_JLE: taken = ((int64_t)cmp_a <= (int64_t)cmp_b); break;
            case KOP_JG:  taken = ((int64_t)cmp_a >  (int64_t)cmp_b); break;
            }
            if (taken) {
                if (bounded_loop && in->imm32 < cur_pc) {
                    if (--fuel == 0u) return KBPF_FUEL_EXHAUSTED;
                }
                pc = in->imm32;
            }
            break;
        }
        case KOP_RET:       return r[0];
        }
    }
    return r[0];
}

static uint64_t kbpf_exec_interp_ex(const KBpfInsn *prog, int n, uint32_t caps,
                                    uint64_t a0, uint64_t a1, uint64_t a2) {
    char err[64];
    if (UNLIKELY(kbpf_verify(prog, n, caps, err, sizeof(err), NULL) != 0)) return 0;
    return kbpf_exec_interp_unchecked(prog, n, caps, a0, a1, a2);
}

/* ============================================================================
 *  8. SILICON-CALIBRATED DOUBLE-BUF BLIS GEMM (9x48 + 14x32 + 6x16 + TRANS A/B)
 * ============================================================================ */

#define GEMM_MAX_MR 14
#define GEMM_MAX_NR 48

typedef void (*GemmUkernelFn)(int64_t kc, const float *Ap, const float *Bp,
                              float *C, int64_t ldc_bytes, int64_t beta_mode, float beta);
typedef void (*GemmEdgeUkernelFn)(int64_t kc, const float *Ap, const float *Bp,
                                  float *C, int64_t ldc_bytes, uint64_t ctrl, float beta);

extern void kronos_gemm_14x32_avx512(int64_t kc, const float *Ap, const float *Bp,
                                     float *C, int64_t ldc_bytes, int64_t beta_mode, float beta);
extern void kronos_gemm_14x32_avx512_edge(int64_t kc, const float *Ap, const float *Bp,
                                          float *C, int64_t ldc_bytes, uint64_t ctrl, float beta);
extern void kronos_gemm_9x48_avx512(int64_t kc, const float *Ap, const float *Bp,
                                    float *C, int64_t ldc_bytes, int64_t beta_mode, float beta);
extern void kronos_gemm_9x48_avx512_edge(int64_t kc, const float *Ap, const float *Bp,
                                         float *C, int64_t ldc_bytes, uint64_t ctrl, float beta);
extern void kronos_gemm_6x16_fma(int64_t kc, const float *Ap, const float *Bp,
                                 float *C, int64_t ldc_bytes, int64_t beta_mode, float beta);
extern void kronos_pack_b_nr48_avx512(int64_t kc, const float *B, int64_t ldb_bytes, float *Bp);
extern void kronos_pack_b_nr32_avx512(int64_t kc, const float *B, int64_t ldb_bytes, float *Bp);
extern void kronos_pack_b_nr16_avx2(int64_t kc, const float *B, int64_t ldb_bytes, float *Bp);
extern void kronos_pack_a_mr6_avx2_8k(int64_t kc_mul8, const float *A, int64_t lda_bytes,
                                      float *Ap, int64_t scale_alpha, float alpha);
extern void kronos_pack_a_mr14_avx512_16k(int64_t kc_mul16, const float *A, int64_t lda_bytes,
                                          float *Ap, int64_t scale_alpha, float alpha);
extern void kronos_pack_a_mr9_avx512_16k(int64_t kc_mul16, const float *A, int64_t lda_bytes,
                                         float *Ap, int64_t scale_alpha, float alpha);

#define KRONOS_FMA6X16_RANK1(a0,a1,a2,a3,a4,a5, b0,b1) \
    "vmovaps " #b0 "(%rdx), %ymm0\n\t"                 \
    "vmovaps " #b1 "(%rdx), %ymm1\n\t"                 \
    "vbroadcastss " #a0 "(%rsi), %ymm2\n\t"            \
    "vbroadcastss " #a1 "(%rsi), %ymm3\n\t"            \
    "vfmadd231ps %ymm0, %ymm2, %ymm4\n\t"              \
    "vfmadd231ps %ymm1, %ymm2, %ymm5\n\t"              \
    "vbroadcastss " #a2 "(%rsi), %ymm2\n\t"            \
    "vfmadd231ps %ymm0, %ymm3, %ymm6\n\t"              \
    "vfmadd231ps %ymm1, %ymm3, %ymm7\n\t"              \
    "vbroadcastss " #a3 "(%rsi), %ymm3\n\t"            \
    "vfmadd231ps %ymm0, %ymm2, %ymm8\n\t"              \
    "vfmadd231ps %ymm1, %ymm2, %ymm9\n\t"              \
    "vbroadcastss " #a4 "(%rsi), %ymm2\n\t"            \
    "vfmadd231ps %ymm0, %ymm3, %ymm10\n\t"             \
    "vfmadd231ps %ymm1, %ymm3, %ymm11\n\t"             \
    "vbroadcastss " #a5 "(%rsi), %ymm3\n\t"            \
    "vfmadd231ps %ymm0, %ymm2, %ymm12\n\t"             \
    "vfmadd231ps %ymm1, %ymm2, %ymm13\n\t"             \
    "vfmadd231ps %ymm0, %ymm3, %ymm14\n\t"             \
    "vfmadd231ps %ymm1, %ymm3, %ymm15\n\t"

#define KRONOS_AVX2_ADD_ROW_PTR(ptr, y_lo, y_hi)       \
    "vaddps  0(" ptr "), " y_lo ", " y_lo "\n\t"       \
    "vaddps 32(" ptr "), " y_hi ", " y_hi "\n\t"       \
    "vmovups " y_lo ",  0(" ptr ")\n\t"                \
    "vmovups " y_hi ", 32(" ptr ")\n\t"

#define KRONOS_AVX2_SET_ROW_PTR(ptr, y_lo, y_hi)       \
    "vmovups " y_lo ",  0(" ptr ")\n\t"                \
    "vmovups " y_hi ", 32(" ptr ")\n\t"

#define KRONOS_AVX2_GEN_ROW_PTR(ptr, y_lo, y_hi)       \
    "vfmadd231ps  0(" ptr "), %ymm0, " y_lo "\n\t"     \
    "vfmadd231ps 32(" ptr "), %ymm0, " y_hi "\n\t"     \
    "vmovups " y_lo ",  0(" ptr ")\n\t"                \
    "vmovups " y_hi ", 32(" ptr ")\n\t"

#define KRONOS_AVX512_14X32_RANK1(abase, bbase)        \
    "vmovaps " #bbase "(%rdx), %zmm0\n\t"              \
    "vmovaps " #bbase "+64(%rdx), %zmm1\n\t"           \
    "vbroadcastss " #abase "+0(%rsi),  %zmm2\n\t"      \
    "vbroadcastss " #abase "+4(%rsi),  %zmm3\n\t"      \
    "vfmadd231ps %zmm0, %zmm2, %zmm4\n\t"              \
    "vfmadd231ps %zmm1, %zmm2, %zmm5\n\t"              \
    "vbroadcastss " #abase "+8(%rsi),  %zmm2\n\t"      \
    "vfmadd231ps %zmm0, %zmm3, %zmm6\n\t"              \
    "vfmadd231ps %zmm1, %zmm3, %zmm7\n\t"              \
    "vbroadcastss " #abase "+12(%rsi), %zmm3\n\t"      \
    "vfmadd231ps %zmm0, %zmm2, %zmm8\n\t"              \
    "vfmadd231ps %zmm1, %zmm2, %zmm9\n\t"              \
    "vbroadcastss " #abase "+16(%rsi), %zmm2\n\t"      \
    "vfmadd231ps %zmm0, %zmm3, %zmm10\n\t"             \
    "vfmadd231ps %zmm1, %zmm3, %zmm11\n\t"             \
    "vbroadcastss " #abase "+20(%rsi), %zmm3\n\t"      \
    "vfmadd231ps %zmm0, %zmm2, %zmm12\n\t"             \
    "vfmadd231ps %zmm1, %zmm2, %zmm13\n\t"             \
    "vbroadcastss " #abase "+24(%rsi), %zmm2\n\t"      \
    "vfmadd231ps %zmm0, %zmm3, %zmm14\n\t"             \
    "vfmadd231ps %zmm1, %zmm3, %zmm15\n\t"             \
    "vbroadcastss " #abase "+28(%rsi), %zmm3\n\t"      \
    "vfmadd231ps %zmm0, %zmm2, %zmm16\n\t"             \
    "vfmadd231ps %zmm1, %zmm2, %zmm17\n\t"             \
    "vbroadcastss " #abase "+32(%rsi), %zmm2\n\t"      \
    "vfmadd231ps %zmm0, %zmm3, %zmm18\n\t"             \
    "vfmadd231ps %zmm1, %zmm3, %zmm19\n\t"             \
    "vbroadcastss " #abase "+36(%rsi), %zmm3\n\t"      \
    "vfmadd231ps %zmm0, %zmm2, %zmm20\n\t"             \
    "vfmadd231ps %zmm1, %zmm2, %zmm21\n\t"             \
    "vbroadcastss " #abase "+40(%rsi), %zmm2\n\t"      \
    "vfmadd231ps %zmm0, %zmm3, %zmm22\n\t"             \
    "vfmadd231ps %zmm1, %zmm3, %zmm23\n\t"             \
    "vbroadcastss " #abase "+44(%rsi), %zmm3\n\t"      \
    "vfmadd231ps %zmm0, %zmm2, %zmm24\n\t"             \
    "vfmadd231ps %zmm1, %zmm2, %zmm25\n\t"             \
    "vbroadcastss " #abase "+48(%rsi), %zmm2\n\t"      \
    "vfmadd231ps %zmm0, %zmm3, %zmm26\n\t"             \
    "vfmadd231ps %zmm1, %zmm3, %zmm27\n\t"             \
    "vbroadcastss " #abase "+52(%rsi), %zmm3\n\t"      \
    "vfmadd231ps %zmm0, %zmm2, %zmm28\n\t"             \
    "vfmadd231ps %zmm1, %zmm2, %zmm29\n\t"             \
    "vfmadd231ps %zmm0, %zmm3, %zmm30\n\t"             \
    "vfmadd231ps %zmm1, %zmm3, %zmm31\n\t"

/* 9x48 AVX-512F Rank-1 (27 accs, 3 B loads, 2 A bcasts -> 2.25 FMAs/load!) */
#define KRONOS_AVX512_9X48_RANK1(abase, bbase)         \
    "vmovaps " #bbase "+0(%rdx),   %zmm0\n\t"          \
    "vmovaps " #bbase "+64(%rdx),  %zmm1\n\t"          \
    "vmovaps " #bbase "+128(%rdx), %zmm2\n\t"          \
    "vbroadcastss " #abase "+0(%rsi),  %zmm3\n\t"      \
    "vbroadcastss " #abase "+4(%rsi),  %zmm4\n\t"      \
    "vfmadd231ps %zmm0, %zmm3, %zmm5\n\t"              \
    "vfmadd231ps %zmm1, %zmm3, %zmm6\n\t"              \
    "vfmadd231ps %zmm2, %zmm3, %zmm7\n\t"              \
    "vbroadcastss " #abase "+8(%rsi),  %zmm3\n\t"      \
    "vfmadd231ps %zmm0, %zmm4, %zmm8\n\t"              \
    "vfmadd231ps %zmm1, %zmm4, %zmm9\n\t"              \
    "vfmadd231ps %zmm2, %zmm4, %zmm10\n\t"             \
    "vbroadcastss " #abase "+12(%rsi), %zmm4\n\t"      \
    "vfmadd231ps %zmm0, %zmm3, %zmm11\n\t"             \
    "vfmadd231ps %zmm1, %zmm3, %zmm12\n\t"             \
    "vfmadd231ps %zmm2, %zmm3, %zmm13\n\t"             \
    "vbroadcastss " #abase "+16(%rsi), %zmm3\n\t"      \
    "vfmadd231ps %zmm0, %zmm4, %zmm14\n\t"             \
    "vfmadd231ps %zmm1, %zmm4, %zmm15\n\t"             \
    "vfmadd231ps %zmm2, %zmm4, %zmm16\n\t"             \
    "vbroadcastss " #abase "+20(%rsi), %zmm4\n\t"      \
    "vfmadd231ps %zmm0, %zmm3, %zmm17\n\t"             \
    "vfmadd231ps %zmm1, %zmm3, %zmm18\n\t"             \
    "vfmadd231ps %zmm2, %zmm3, %zmm19\n\t"             \
    "vbroadcastss " #abase "+24(%rsi), %zmm3\n\t"      \
    "vfmadd231ps %zmm0, %zmm4, %zmm20\n\t"             \
    "vfmadd231ps %zmm1, %zmm4, %zmm21\n\t"             \
    "vfmadd231ps %zmm2, %zmm4, %zmm22\n\t"             \
    "vbroadcastss " #abase "+28(%rsi), %zmm4\n\t"      \
    "vfmadd231ps %zmm0, %zmm3, %zmm23\n\t"             \
    "vfmadd231ps %zmm1, %zmm3, %zmm24\n\t"             \
    "vfmadd231ps %zmm2, %zmm3, %zmm25\n\t"             \
    "vbroadcastss " #abase "+32(%rsi), %zmm3\n\t"      \
    "vfmadd231ps %zmm0, %zmm4, %zmm26\n\t"             \
    "vfmadd231ps %zmm1, %zmm4, %zmm27\n\t"             \
    "vfmadd231ps %zmm2, %zmm4, %zmm28\n\t"             \
    "vfmadd231ps %zmm0, %zmm3, %zmm29\n\t"             \
    "vfmadd231ps %zmm1, %zmm3, %zmm30\n\t"             \
    "vfmadd231ps %zmm2, %zmm3, %zmm31\n\t"

#define KRONOS_AVX512_ADD_ROW_PTR(ptr, z_lo, z_hi)     \
    "vaddps  0(" ptr "), " z_lo ", " z_lo "\n\t"       \
    "vaddps 64(" ptr "), " z_hi ", " z_hi "\n\t"       \
    "vmovups " z_lo ",  0(" ptr ")\n\t"                \
    "vmovups " z_hi ", 64(" ptr ")\n\t"

#define KRONOS_AVX512_SET_ROW_PTR(ptr, z_lo, z_hi)     \
    "vmovups " z_lo ",  0(" ptr ")\n\t"                \
    "vmovups " z_hi ", 64(" ptr ")\n\t"

#define KRONOS_AVX512_GEN_ROW_PTR(ptr, z_lo, z_hi)     \
    "vfmadd231ps  0(" ptr "), %zmm0, " z_lo "\n\t"     \
    "vfmadd231ps 64(" ptr "), %zmm0, " z_hi "\n\t"     \
    "vmovups " z_lo ",  0(" ptr ")\n\t"                \
    "vmovups " z_hi ", 64(" ptr ")\n\t"

#define KRONOS_AVX512_ADD_ROW3(ptr, z0, z1, z2)        \
    "vaddps   0(" ptr "), " z0 ", " z0 "\n\t"          \
    "vaddps  64(" ptr "), " z1 ", " z1 "\n\t"          \
    "vaddps 128(" ptr "), " z2 ", " z2 "\n\t"          \
    "vmovups " z0 ",   0(" ptr ")\n\t"                 \
    "vmovups " z1 ",  64(" ptr ")\n\t"                 \
    "vmovups " z2 ", 128(" ptr ")\n\t"

#define KRONOS_AVX512_SET_ROW3(ptr, z0, z1, z2)        \
    "vmovups " z0 ",   0(" ptr ")\n\t"                 \
    "vmovups " z1 ",  64(" ptr ")\n\t"                 \
    "vmovups " z2 ", 128(" ptr ")\n\t"

#define KRONOS_AVX512_GEN_ROW3(ptr, z0, z1, z2)        \
    "vfmadd231ps   0(" ptr "), %zmm0, " z0 "\n\t"      \
    "vfmadd231ps  64(" ptr "), %zmm0, " z1 "\n\t"      \
    "vfmadd231ps 128(" ptr "), %zmm0, " z2 "\n\t"      \
    "vmovups " z0 ",   0(" ptr ")\n\t"                 \
    "vmovups " z1 ",  64(" ptr ")\n\t"                 \
    "vmovups " z2 ", 128(" ptr ")\n\t"

/* Zero-copy opmask edge row helpers (full AVX-512F fault suppression!) */
#define KRONOS_EDGE_14X32_ROW_SET(z_lo, z_hi, done_lbl) \
    "vmovups " z_lo ",  0(%rcx){%k1}\n\t"               \
    "vmovups " z_hi ", 64(%rcx){%k2}\n\t"               \
    "addq %r8, %rcx\n\t"                                \
    "decq %r10\n\t"                                     \
    "jz " done_lbl "\n\t"

#define KRONOS_EDGE_14X32_ROW_ADD(z_lo, z_hi, done_lbl) \
    "vmovups  0(%rcx), %zmm1{%k1}{z}\n\t"               \
    "vmovups 64(%rcx), %zmm2{%k2}{z}\n\t"               \
    "vaddps %zmm1, " z_lo ", " z_lo "\n\t"              \
    "vaddps %zmm2, " z_hi ", " z_hi "\n\t"              \
    "vmovups " z_lo ",  0(%rcx){%k1}\n\t"               \
    "vmovups " z_hi ", 64(%rcx){%k2}\n\t"               \
    "addq %r8, %rcx\n\t"                                \
    "decq %r10\n\t"                                     \
    "jz " done_lbl "\n\t"

#define KRONOS_EDGE_14X32_ROW_GEN(z_lo, z_hi, done_lbl) \
    "vmovups  0(%rcx), %zmm1{%k1}{z}\n\t"               \
    "vmovups 64(%rcx), %zmm2{%k2}{z}\n\t"               \
    "vfmadd231ps %zmm0, %zmm1, " z_lo "\n\t"            \
    "vfmadd231ps %zmm0, %zmm2, " z_hi "\n\t"            \
    "vmovups " z_lo ",  0(%rcx){%k1}\n\t"               \
    "vmovups " z_hi ", 64(%rcx){%k2}\n\t"               \
    "addq %r8, %rcx\n\t"                                \
    "decq %r10\n\t"                                     \
    "jz " done_lbl "\n\t"

#define KRONOS_EDGE_9X48_ROW_SET(z0, z1, z2, done_lbl)  \
    "vmovups " z0 ",   0(%rcx){%k1}\n\t"                \
    "vmovups " z1 ",  64(%rcx){%k2}\n\t"                \
    "vmovups " z2 ", 128(%rcx){%k3}\n\t"                \
    "addq %r8, %rcx\n\t"                                \
    "decq %r10\n\t"                                     \
    "jz " done_lbl "\n\t"

#define KRONOS_EDGE_9X48_ROW_ADD(z0, z1, z2, done_lbl)  \
    "vmovups   0(%rcx), %zmm1{%k1}{z}\n\t"              \
    "vmovups  64(%rcx), %zmm2{%k2}{z}\n\t"              \
    "vmovups 128(%rcx), %zmm3{%k3}{z}\n\t"              \
    "vaddps %zmm1, " z0 ", " z0 "\n\t"                  \
    "vaddps %zmm2, " z1 ", " z1 "\n\t"                  \
    "vaddps %zmm3, " z2 ", " z2 "\n\t"                  \
    "vmovups " z0 ",   0(%rcx){%k1}\n\t"                \
    "vmovups " z1 ",  64(%rcx){%k2}\n\t"                \
    "vmovups " z2 ", 128(%rcx){%k3}\n\t"                \
    "addq %r8, %rcx\n\t"                                \
    "decq %r10\n\t"                                     \
    "jz " done_lbl "\n\t"

#define KRONOS_EDGE_9X48_ROW_GEN(z0, z1, z2, done_lbl)  \
    "vmovups   0(%rcx), %zmm1{%k1}{z}\n\t"              \
    "vmovups  64(%rcx), %zmm2{%k2}{z}\n\t"              \
    "vmovups 128(%rcx), %zmm3{%k3}{z}\n\t"              \
    "vfmadd231ps %zmm0, %zmm1, " z0 "\n\t"              \
    "vfmadd231ps %zmm0, %zmm2, " z1 "\n\t"              \
    "vfmadd231ps %zmm0, %zmm3, " z2 "\n\t"              \
    "vmovups " z0 ",   0(%rcx){%k1}\n\t"                \
    "vmovups " z1 ",  64(%rcx){%k2}\n\t"                \
    "vmovups " z2 ", 128(%rcx){%k3}\n\t"                \
    "addq %r8, %rcx\n\t"                                \
    "decq %r10\n\t"                                     \
    "jz " done_lbl "\n\t"

#define KRONOS_PACK_A_MR6_BODY(SCALE_INSNS)            \
    "movq %rsi, %rax\n\t"                              \
    "prefetcht0 96(%rax)\n\t"                          \
    "prefetcht0 96(%rax, %r8)\n\t"                     \
    "prefetcht0 96(%rax, %r9)\n\t"                     \
    "prefetcht0 96(%rax, %r10)\n\t"                    \
    "vmovups (%rax),       %ymm0\n\t"                  \
    "vmovups (%rax, %r8),  %ymm1\n\t"                  \
    "vmovups (%rax, %r9),  %ymm2\n\t"                  \
    "vmovups (%rax, %r10), %ymm3\n\t"                  \
    "leaq (%rax, %r8, 4),  %rax\n\t"                   \
    "prefetcht0 96(%rax)\n\t"                          \
    "prefetcht0 96(%rax, %r8)\n\t"                     \
    "vmovups (%rax),       %ymm4\n\t"                  \
    "vmovups (%rax, %r8),  %ymm5\n\t"                  \
    SCALE_INSNS                                        \
    "vunpcklps %ymm1, %ymm0, %ymm6\n\t"                \
    "vunpckhps %ymm1, %ymm0, %ymm7\n\t"                \
    "vunpcklps %ymm3, %ymm2, %ymm8\n\t"                \
    "vunpckhps %ymm3, %ymm2, %ymm9\n\t"                \
    "vshufps $0x44, %ymm8, %ymm6, %ymm10\n\t"          \
    "vshufps $0xEE, %ymm8, %ymm6, %ymm11\n\t"          \
    "vshufps $0x44, %ymm9, %ymm7, %ymm12\n\t"          \
    "vshufps $0xEE, %ymm9, %ymm7, %ymm13\n\t"          \
    "vmovups %xmm10,   0(%rcx)\n\t"                    \
    "vmovups %xmm11,  24(%rcx)\n\t"                    \
    "vmovups %xmm12,  48(%rcx)\n\t"                    \
    "vmovups %xmm13,  72(%rcx)\n\t"                    \
    "vextractf128 $1, %ymm10, %xmm10\n\t"              \
    "vextractf128 $1, %ymm11, %xmm11\n\t"              \
    "vextractf128 $1, %ymm12, %xmm12\n\t"              \
    "vextractf128 $1, %ymm13, %xmm13\n\t"              \
    "vmovups %xmm10,  96(%rcx)\n\t"                    \
    "vmovups %xmm11, 120(%rcx)\n\t"                    \
    "vmovups %xmm12, 144(%rcx)\n\t"                    \
    "vmovups %xmm13, 168(%rcx)\n\t"                    \
    "vunpcklps %ymm5, %ymm4, %ymm14\n\t"               \
    "vunpckhps %ymm5, %ymm4, %ymm15\n\t"               \
    "vmovlps %xmm14,  16(%rcx)\n\t"                    \
    "vmovhps %xmm14,  40(%rcx)\n\t"                    \
    "vmovlps %xmm15,  64(%rcx)\n\t"                    \
    "vmovhps %xmm15,  88(%rcx)\n\t"                    \
    "vextractf128 $1, %ymm14, %xmm14\n\t"              \
    "vextractf128 $1, %ymm15, %xmm15\n\t"              \
    "vmovlps %xmm14, 112(%rcx)\n\t"                    \
    "vmovhps %xmm14, 136(%rcx)\n\t"                    \
    "vmovlps %xmm15, 160(%rcx)\n\t"                    \
    "vmovhps %xmm15, 184(%rcx)\n\t"                    \
    "addq $32,  %rsi\n\t"                              \
    "addq $192, %rcx\n\t"

#define KRONOS_AVX512_TRANSPOSE_4X16_STORE(row_off, stride, SCALE_INSNS) \
    "prefetcht0 128(%rax)\n\t"                                           \
    "prefetcht0 128(%rax, %r8)\n\t"                                      \
    "prefetcht0 128(%rax, %r9)\n\t"                                      \
    "prefetcht0 128(%rax, %r10)\n\t"                                     \
    "vmovups (%rax),       %zmm0\n\t"                                    \
    "vmovups (%rax, %r8),  %zmm1\n\t"                                    \
    "vmovups (%rax, %r9),  %zmm2\n\t"                                    \
    "vmovups (%rax, %r10), %zmm3\n\t"                                    \
    SCALE_INSNS                                                          \
    "vunpcklps %zmm1, %zmm0, %zmm6\n\t"                                  \
    "vunpckhps %zmm1, %zmm0, %zmm7\n\t"                                  \
    "vunpcklps %zmm3, %zmm2, %zmm8\n\t"                                  \
    "vunpckhps %zmm3, %zmm2, %zmm9\n\t"                                  \
    "vshufps $0x44, %zmm8, %zmm6, %zmm10\n\t"                            \
    "vshufps $0xEE, %zmm8, %zmm6, %zmm11\n\t"                            \
    "vshufps $0x44, %zmm9, %zmm7, %zmm12\n\t"                            \
    "vshufps $0xEE, %zmm9, %zmm7, %zmm13\n\t"                            \
    "vmovups %xmm10, " #row_off "+0*" #stride "(%rcx)\n\t"               \
    "vmovups %xmm11, " #row_off "+1*" #stride "(%rcx)\n\t"               \
    "vmovups %xmm12, " #row_off "+2*" #stride "(%rcx)\n\t"               \
    "vmovups %xmm13, " #row_off "+3*" #stride "(%rcx)\n\t"               \
    "vextractf32x4 $1, %zmm10, %xmm14\n\t"                               \
    "vextractf32x4 $1, %zmm11, %xmm15\n\t"                               \
    "vextractf32x4 $1, %zmm12, %xmm16\n\t"                               \
    "vextractf32x4 $1, %zmm13, %xmm17\n\t"                               \
    "vmovups %xmm14, " #row_off "+4*" #stride "(%rcx)\n\t"               \
    "vmovups %xmm15, " #row_off "+5*" #stride "(%rcx)\n\t"               \
    "vmovups %xmm16, " #row_off "+6*" #stride "(%rcx)\n\t"               \
    "vmovups %xmm17, " #row_off "+7*" #stride "(%rcx)\n\t"               \
    "vextractf32x4 $2, %zmm10, %xmm14\n\t"                               \
    "vextractf32x4 $2, %zmm11, %xmm15\n\t"                               \
    "vextractf32x4 $2, %zmm12, %xmm16\n\t"                               \
    "vextractf32x4 $2, %zmm13, %xmm17\n\t"                               \
    "vmovups %xmm14, " #row_off "+8*"  #stride "(%rcx)\n\t"              \
    "vmovups %xmm15, " #row_off "+9*"  #stride "(%rcx)\n\t"              \
    "vmovups %xmm16, " #row_off "+10*" #stride "(%rcx)\n\t"              \
    "vmovups %xmm17, " #row_off "+11*" #stride "(%rcx)\n\t"              \
    "vextractf32x4 $3, %zmm10, %xmm14\n\t"                               \
    "vextractf32x4 $3, %zmm11, %xmm15\n\t"                               \
    "vextractf32x4 $3, %zmm12, %xmm16\n\t"                               \
    "vextractf32x4 $3, %zmm13, %xmm17\n\t"                               \
    "vmovups %xmm14, " #row_off "+12*" #stride "(%rcx)\n\t"              \
    "vmovups %xmm15, " #row_off "+13*" #stride "(%rcx)\n\t"              \
    "vmovups %xmm16, " #row_off "+14*" #stride "(%rcx)\n\t"              \
    "vmovups %xmm17, " #row_off "+15*" #stride "(%rcx)\n\t"

#define KRONOS_PACK_A_MR14_BODY(SCALE4, SCALE2)                          \
    "movq %rsi, %rax\n\t"                                                \
    KRONOS_AVX512_TRANSPOSE_4X16_STORE(0,  56, SCALE4)                   \
    "leaq (%rax, %r8, 4), %rax\n\t"                                      \
    KRONOS_AVX512_TRANSPOSE_4X16_STORE(16, 56, SCALE4)                   \
    "leaq (%rax, %r8, 4), %rax\n\t"                                      \
    KRONOS_AVX512_TRANSPOSE_4X16_STORE(32, 56, SCALE4)                   \
    "leaq (%rax, %r8, 4), %rax\n\t"                                      \
    "prefetcht0 128(%rax)\n\t"                                           \
    "prefetcht0 128(%rax, %r8)\n\t"                                      \
    "vmovups (%rax),      %zmm4\n\t"                                     \
    "vmovups (%rax, %r8), %zmm5\n\t"                                     \
    SCALE2                                                               \
    "vunpcklps %zmm5, %zmm4, %zmm14\n\t"                                 \
    "vunpckhps %zmm5, %zmm4, %zmm15\n\t"                                 \
    "vmovlps %xmm14,  48+0*56(%rcx)\n\t"  "vmovhps %xmm14,  48+1*56(%rcx)\n\t" \
    "vmovlps %xmm15,  48+2*56(%rcx)\n\t"  "vmovhps %xmm15,  48+3*56(%rcx)\n\t" \
    "vextractf32x4 $1, %zmm14, %xmm16\n\t" "vextractf32x4 $1, %zmm15, %xmm17\n\t" \
    "vmovlps %xmm16,  48+4*56(%rcx)\n\t"  "vmovhps %xmm16,  48+5*56(%rcx)\n\t" \
    "vmovlps %xmm17,  48+6*56(%rcx)\n\t"  "vmovhps %xmm17,  48+7*56(%rcx)\n\t" \
    "vextractf32x4 $2, %zmm14, %xmm16\n\t" "vextractf32x4 $2, %zmm15, %xmm17\n\t" \
    "vmovlps %xmm16,  48+8*56(%rcx)\n\t"  "vmovhps %xmm16,  48+9*56(%rcx)\n\t" \
    "vmovlps %xmm17, 48+10*56(%rcx)\n\t"  "vmovhps %xmm17, 48+11*56(%rcx)\n\t" \
    "vextractf32x4 $3, %zmm14, %xmm16\n\t" "vextractf32x4 $3, %zmm15, %xmm17\n\t" \
    "vmovlps %xmm16, 48+12*56(%rcx)\n\t"  "vmovhps %xmm16, 48+13*56(%rcx)\n\t" \
    "vmovlps %xmm17, 48+14*56(%rcx)\n\t"  "vmovhps %xmm17, 48+15*56(%rcx)\n\t" \
    "addq $64,  %rsi\n\t"                                                \
    "addq $896, %rcx\n\t"

#define KRONOS_PACK_A_MR9_BODY(SCALE4, SCALE1)                           \
    "movq %rsi, %rax\n\t"                                                \
    KRONOS_AVX512_TRANSPOSE_4X16_STORE(0,  36, SCALE4)                   \
    "leaq (%rax, %r8, 4), %rax\n\t"                                      \
    KRONOS_AVX512_TRANSPOSE_4X16_STORE(16, 36, SCALE4)                   \
    "leaq (%rax, %r8, 4), %rax\n\t"                                      \
    "prefetcht0 128(%rax)\n\t"                                           \
    "vmovups (%rax), %zmm4\n\t"                                          \
    SCALE1                                                               \
    "vmovups %zmm4, -64(%rsp)\n\t"                                       \
    "movl -64(%rsp), %r11d\n\t" "movl %r11d, 32+0*36(%rcx)\n\t"          \
    "movl -60(%rsp), %r11d\n\t" "movl %r11d, 32+1*36(%rcx)\n\t"          \
    "movl -56(%rsp), %r11d\n\t" "movl %r11d, 32+2*36(%rcx)\n\t"          \
    "movl -52(%rsp), %r11d\n\t" "movl %r11d, 32+3*36(%rcx)\n\t"          \
    "movl -48(%rsp), %r11d\n\t" "movl %r11d, 32+4*36(%rcx)\n\t"          \
    "movl -44(%rsp), %r11d\n\t" "movl %r11d, 32+5*36(%rcx)\n\t"          \
    "movl -40(%rsp), %r11d\n\t" "movl %r11d, 32+6*36(%rcx)\n\t"          \
    "movl -36(%rsp), %r11d\n\t" "movl %r11d, 32+7*36(%rcx)\n\t"          \
    "movl -32(%rsp), %r11d\n\t" "movl %r11d, 32+8*36(%rcx)\n\t"          \
    "movl -28(%rsp), %r11d\n\t" "movl %r11d, 32+9*36(%rcx)\n\t"          \
    "movl -24(%rsp), %r11d\n\t" "movl %r11d, 32+10*36(%rcx)\n\t"         \
    "movl -20(%rsp), %r11d\n\t" "movl %r11d, 32+11*36(%rcx)\n\t"         \
    "movl -16(%rsp), %r11d\n\t" "movl %r11d, 32+12*36(%rcx)\n\t"         \
    "movl -12(%rsp), %r11d\n\t" "movl %r11d, 32+13*36(%rcx)\n\t"         \
    "movl  -8(%rsp), %r11d\n\t" "movl %r11d, 32+14*36(%rcx)\n\t"         \
    "movl  -4(%rsp), %r11d\n\t" "movl %r11d, 32+15*36(%rcx)\n\t"         \
    "addq $64,  %rsi\n\t"                                                \
    "addq $576, %rcx\n\t"

__asm__(
    ".pushsection .text, \"ax\", @progbits\n\t"

    /* ========================================================================
     *  kronos_gemm_6x16_fma (staggered LFB-safe prefetchw + disp8 B bias)
     * ======================================================================== */
    ".globl  kronos_gemm_6x16_fma\n\t"
    ".hidden kronos_gemm_6x16_fma\n\t"
    ".type   kronos_gemm_6x16_fma, @function\n\t"
    ".p2align 5\n\t"
    "kronos_gemm_6x16_fma:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "vmovss %xmm0, -4(%rsp)\n\t"
    "vxorps %ymm4,  %ymm4,  %ymm4\n\t"  "vxorps %ymm5,  %ymm5,  %ymm5\n\t"
    "vxorps %ymm6,  %ymm6,  %ymm6\n\t"  "vxorps %ymm7,  %ymm7,  %ymm7\n\t"
    "vxorps %ymm8,  %ymm8,  %ymm8\n\t"  "vxorps %ymm9,  %ymm9,  %ymm9\n\t"
    "vxorps %ymm10, %ymm10, %ymm10\n\t" "vxorps %ymm11, %ymm11, %ymm11\n\t"
    "vxorps %ymm12, %ymm12, %ymm12\n\t" "vxorps %ymm13, %ymm13, %ymm13\n\t"
    "vxorps %ymm14, %ymm14, %ymm14\n\t" "vxorps %ymm15, %ymm15, %ymm15\n\t"

    "testq %rdi, %rdi\n\t"
    "jle .Lg6x16_kc_zero\n\t"

    "movq %rdi, %rax\n\t"
    "shrq $2, %rax\n\t"
    "cmpq $1, %rax\n\t"
    "jbe .Lg6x16_tail_prep\n\t"
    "decq %rax\n\t"

    "addq $128, %rdx\n\t"
    ".p2align 5\n\t"
    ".Lg6x16_loop4:\n\t"
    "prefetcht0 384(%rdx)\n\t"
    "prefetcht0 256(%rsi)\n\t"
    KRONOS_FMA6X16_RANK1( 0,  4,  8, 12, 16, 20, -128, -96)
    "prefetcht0 448(%rdx)\n\t"
    KRONOS_FMA6X16_RANK1(24, 28, 32, 36, 40, 44,  -64, -32)
    "prefetcht0 512(%rdx)\n\t"
    "prefetcht0 320(%rsi)\n\t"
    KRONOS_FMA6X16_RANK1(48, 52, 56, 60, 64, 68,    0,  32)
    "prefetcht0 576(%rdx)\n\t"
    KRONOS_FMA6X16_RANK1(72, 76, 80, 84, 88, 92,   64,  96)
    "addq $96,  %rsi\n\t"
    "addq $256, %rdx\n\t"
    "decq %rax\n\t"
    "jnz .Lg6x16_loop4\n\t"
    "subq $128, %rdx\n\t"
    "andq $3, %rdi\n\t"
    "addq $4, %rdi\n\t"

    /* Staggered prefetchw: 2 rows per k-step across first 3 tail steps (LFB-safe!) */
    ".Lg6x16_tail_prep:\n\t"
    "movq %rcx, %r10\n\t"
    "xorl %r11d, %r11d\n\t"
    "cmpl $0, g_cpu_has_prefetchw_asm(%rip)\n\t"
    "je .Lg6x16_tail_loop\n\t"
    "movl $3, %r11d\n\t"

    ".p2align 4\n\t"
    ".Lg6x16_tail_loop:\n\t"
    "testl %r11d, %r11d\n\t"
    "jz .Lg6x16_no_pw\n\t"
    "decl %r11d\n\t"
    "prefetchw (%r10)\n\t" "prefetchw 60(%r10)\n\t" "addq %r8, %r10\n\t"
    "prefetchw (%r10)\n\t" "prefetchw 60(%r10)\n\t" "addq %r8, %r10\n\t"
    ".Lg6x16_no_pw:\n\t"
    KRONOS_FMA6X16_RANK1(0, 4, 8, 12, 16, 20, 0, 32)
    "addq $24, %rsi\n\t"
    "addq $64, %rdx\n\t"
    "decq %rdi\n\t"
    "jnz .Lg6x16_tail_loop\n\t"

    ".Lg6x16_store:\n\t"
    "leaq (%rcx, %r8),    %r10\n\t"
    "leaq (%rcx, %r8, 2), %r11\n\t"
    "leaq (%r10, %r8, 2), %rax\n\t"
    "testq %r9, %r9\n\t"
    "jz .Lg6x16_store_zero\n\t"
    "cmpq $1, %r9\n\t"
    "jne .Lg6x16_store_gen\n\t"

    KRONOS_AVX2_ADD_ROW_PTR("%rcx", "%ymm4",  "%ymm5")
    KRONOS_AVX2_ADD_ROW_PTR("%r10", "%ymm6",  "%ymm7")
    KRONOS_AVX2_ADD_ROW_PTR("%r11", "%ymm8",  "%ymm9")
    KRONOS_AVX2_ADD_ROW_PTR("%rax", "%ymm10", "%ymm11")
    "leaq (%r11, %r8, 2), %rcx\n\t"
    "leaq (%rax, %r8, 2), %r10\n\t"
    KRONOS_AVX2_ADD_ROW_PTR("%rcx", "%ymm12", "%ymm13")
    KRONOS_AVX2_ADD_ROW_PTR("%r10", "%ymm14", "%ymm15")
    "vzeroupper\n\t"
    "ret\n\t"

    ".Lg6x16_kc_zero:\n\t"
    "cmpq $1, %r9\n\t"
    "jne .Lg6x16_store\n\t"
    "vzeroupper\n\t"
    "ret\n\t"

    ".Lg6x16_store_zero:\n\t"
    KRONOS_AVX2_SET_ROW_PTR("%rcx", "%ymm4",  "%ymm5")
    KRONOS_AVX2_SET_ROW_PTR("%r10", "%ymm6",  "%ymm7")
    KRONOS_AVX2_SET_ROW_PTR("%r11", "%ymm8",  "%ymm9")
    KRONOS_AVX2_SET_ROW_PTR("%rax", "%ymm10", "%ymm11")
    "leaq (%r11, %r8, 2), %rcx\n\t"
    "leaq (%rax, %r8, 2), %r10\n\t"
    KRONOS_AVX2_SET_ROW_PTR("%rcx", "%ymm12", "%ymm13")
    KRONOS_AVX2_SET_ROW_PTR("%r10", "%ymm14", "%ymm15")
    "vzeroupper\n\t"
    "ret\n\t"

    ".Lg6x16_store_gen:\n\t"
    "vbroadcastss -4(%rsp), %ymm0\n\t"
    KRONOS_AVX2_GEN_ROW_PTR("%rcx", "%ymm4",  "%ymm5")
    KRONOS_AVX2_GEN_ROW_PTR("%r10", "%ymm6",  "%ymm7")
    KRONOS_AVX2_GEN_ROW_PTR("%r11", "%ymm8",  "%ymm9")
    KRONOS_AVX2_GEN_ROW_PTR("%rax", "%ymm10", "%ymm11")
    "leaq (%r11, %r8, 2), %rcx\n\t"
    "leaq (%rax, %r8, 2), %r10\n\t"
    KRONOS_AVX2_GEN_ROW_PTR("%rcx", "%ymm12", "%ymm13")
    KRONOS_AVX2_GEN_ROW_PTR("%r10", "%ymm14", "%ymm15")
    "vzeroupper\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_gemm_6x16_fma, .-kronos_gemm_6x16_fma\n\t"

    /* ========================================================================
     *  kronos_gemm_14x32_avx512 (staggered 2-rows/k LFB-safe prefetchw!)
     * ======================================================================== */
    ".globl  kronos_gemm_14x32_avx512\n\t"
    ".hidden kronos_gemm_14x32_avx512\n\t"
    ".type   kronos_gemm_14x32_avx512, @function\n\t"
    ".p2align 5\n\t"
    "kronos_gemm_14x32_avx512:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "vmovss %xmm0, -4(%rsp)\n\t"
    "vpxord %zmm4,  %zmm4,  %zmm4\n\t"  "vpxord %zmm5,  %zmm5,  %zmm5\n\t"
    "vpxord %zmm6,  %zmm6,  %zmm6\n\t"  "vpxord %zmm7,  %zmm7,  %zmm7\n\t"
    "vpxord %zmm8,  %zmm8,  %zmm8\n\t"  "vpxord %zmm9,  %zmm9,  %zmm9\n\t"
    "vpxord %zmm10, %zmm10, %zmm10\n\t" "vpxord %zmm11, %zmm11, %zmm11\n\t"
    "vpxord %zmm12, %zmm12, %zmm12\n\t" "vpxord %zmm13, %zmm13, %zmm13\n\t"
    "vpxord %zmm14, %zmm14, %zmm14\n\t" "vpxord %zmm15, %zmm15, %zmm15\n\t"
    "vpxord %zmm16, %zmm16, %zmm16\n\t" "vpxord %zmm17, %zmm17, %zmm17\n\t"
    "vpxord %zmm18, %zmm18, %zmm18\n\t" "vpxord %zmm19, %zmm19, %zmm19\n\t"
    "vpxord %zmm20, %zmm20, %zmm20\n\t" "vpxord %zmm21, %zmm21, %zmm21\n\t"
    "vpxord %zmm22, %zmm22, %zmm22\n\t" "vpxord %zmm23, %zmm23, %zmm23\n\t"
    "vpxord %zmm24, %zmm24, %zmm24\n\t" "vpxord %zmm25, %zmm25, %zmm25\n\t"
    "vpxord %zmm26, %zmm26, %zmm26\n\t" "vpxord %zmm27, %zmm27, %zmm27\n\t"
    "vpxord %zmm28, %zmm28, %zmm28\n\t" "vpxord %zmm29, %zmm29, %zmm29\n\t"
    "vpxord %zmm30, %zmm30, %zmm30\n\t" "vpxord %zmm31, %zmm31, %zmm31\n\t"

    "testq %rdi, %rdi\n\t"
    "jle .Lg14x32_kc_zero\n\t"

    /* Keep 8..11 tail steps so all 14 rows prefetch at 2 rows/k (4-6 LFBs, zero overflow!) */
    "movq %rdi, %rax\n\t"
    "shrq $2, %rax\n\t"
    "cmpq $2, %rax\n\t"
    "jbe .Lg14x32_tail_prep\n\t"
    "subq $2, %rax\n\t"

    ".p2align 5\n\t"
    ".Lg14x32_loop4:\n\t"
    "prefetcht0 768(%rdx)\n\t"  "prefetcht0 832(%rdx)\n\t"  "prefetcht0 448(%rsi)\n\t"
    KRONOS_AVX512_14X32_RANK1(  0,   0)
    "prefetcht0 896(%rdx)\n\t"  "prefetcht0 960(%rdx)\n\t"  "prefetcht0 512(%rsi)\n\t"
    KRONOS_AVX512_14X32_RANK1( 56, 128)
    "prefetcht0 1024(%rdx)\n\t" "prefetcht0 1088(%rdx)\n\t" "prefetcht0 576(%rsi)\n\t"
    KRONOS_AVX512_14X32_RANK1(112, 256)
    "prefetcht0 1152(%rdx)\n\t" "prefetcht0 1216(%rdx)\n\t" "prefetcht0 640(%rsi)\n\t"
    KRONOS_AVX512_14X32_RANK1(168, 384)
    "addq $224, %rsi\n\t"
    "addq $512, %rdx\n\t"
    "decq %rax\n\t"
    "jnz .Lg14x32_loop4\n\t"
    "andq $3, %rdi\n\t"
    "addq $8, %rdi\n\t"

    ".Lg14x32_tail_prep:\n\t"
    "movq %rcx, %r10\n\t"
    "xorl %r11d, %r11d\n\t"
    "cmpl $0, g_cpu_has_prefetchw_asm(%rip)\n\t"
    "je .Lg14x32_tail_loop\n\t"
    "movl $7, %r11d\n\t"

    ".p2align 4\n\t"
    ".Lg14x32_tail_loop:\n\t"
    "testl %r11d, %r11d\n\t"
    "jz .Lg14x32_no_pw\n\t"
    "decl %r11d\n\t"
    "prefetchw (%r10)\n\t" "prefetchw 64(%r10)\n\t" "prefetchw 124(%r10)\n\t" "addq %r8, %r10\n\t"
    "prefetchw (%r10)\n\t" "prefetchw 64(%r10)\n\t" "prefetchw 124(%r10)\n\t" "addq %r8, %r10\n\t"
    ".Lg14x32_no_pw:\n\t"
    KRONOS_AVX512_14X32_RANK1(0, 0)
    "addq $56,  %rsi\n\t"
    "addq $128, %rdx\n\t"
    "decq %rdi\n\t"
    "jnz .Lg14x32_tail_loop\n\t"

    ".Lg14x32_store:\n\t"
    "leaq (%rcx, %r8),    %r10\n\t"
    "leaq (%rcx, %r8, 2), %r11\n\t"
    "leaq (%r10, %r8, 2), %rax\n\t"
    "leaq (,%r8, 4),      %rdx\n\t"
    "testq %r9, %r9\n\t"
    "jz .Lg14x32_store_zero\n\t"
    "cmpq $1, %r9\n\t"
    "jne .Lg14x32_store_gen\n\t"

    KRONOS_AVX512_ADD_ROW_PTR("%rcx", "%zmm4",  "%zmm5")
    KRONOS_AVX512_ADD_ROW_PTR("%r10", "%zmm6",  "%zmm7")
    KRONOS_AVX512_ADD_ROW_PTR("%r11", "%zmm8",  "%zmm9")
    KRONOS_AVX512_ADD_ROW_PTR("%rax", "%zmm10", "%zmm11")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t" "addq %rdx, %r11\n\t" "addq %rdx, %rax\n\t"
    KRONOS_AVX512_ADD_ROW_PTR("%rcx", "%zmm12", "%zmm13")
    KRONOS_AVX512_ADD_ROW_PTR("%r10", "%zmm14", "%zmm15")
    KRONOS_AVX512_ADD_ROW_PTR("%r11", "%zmm16", "%zmm17")
    KRONOS_AVX512_ADD_ROW_PTR("%rax", "%zmm18", "%zmm19")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t" "addq %rdx, %r11\n\t" "addq %rdx, %rax\n\t"
    KRONOS_AVX512_ADD_ROW_PTR("%rcx", "%zmm20", "%zmm21")
    KRONOS_AVX512_ADD_ROW_PTR("%r10", "%zmm22", "%zmm23")
    KRONOS_AVX512_ADD_ROW_PTR("%r11", "%zmm24", "%zmm25")
    KRONOS_AVX512_ADD_ROW_PTR("%rax", "%zmm26", "%zmm27")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t"
    KRONOS_AVX512_ADD_ROW_PTR("%rcx", "%zmm28", "%zmm29")
    KRONOS_AVX512_ADD_ROW_PTR("%r10", "%zmm30", "%zmm31")
    "vzeroupper\n\t"
    "ret\n\t"

    ".Lg14x32_kc_zero:\n\t"
    "cmpq $1, %r9\n\t"
    "jne .Lg14x32_store\n\t"
    "vzeroupper\n\t"
    "ret\n\t"

    ".Lg14x32_store_zero:\n\t"
    KRONOS_AVX512_SET_ROW_PTR("%rcx", "%zmm4",  "%zmm5")
    KRONOS_AVX512_SET_ROW_PTR("%r10", "%zmm6",  "%zmm7")
    KRONOS_AVX512_SET_ROW_PTR("%r11", "%zmm8",  "%zmm9")
    KRONOS_AVX512_SET_ROW_PTR("%rax", "%zmm10", "%zmm11")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t" "addq %rdx, %r11\n\t" "addq %rdx, %rax\n\t"
    KRONOS_AVX512_SET_ROW_PTR("%rcx", "%zmm12", "%zmm13")
    KRONOS_AVX512_SET_ROW_PTR("%r10", "%zmm14", "%zmm15")
    KRONOS_AVX512_SET_ROW_PTR("%r11", "%zmm16", "%zmm17")
    KRONOS_AVX512_SET_ROW_PTR("%rax", "%zmm18", "%zmm19")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t" "addq %rdx, %r11\n\t" "addq %rdx, %rax\n\t"
    KRONOS_AVX512_SET_ROW_PTR("%rcx", "%zmm20", "%zmm21")
    KRONOS_AVX512_SET_ROW_PTR("%r10", "%zmm22", "%zmm23")
    KRONOS_AVX512_SET_ROW_PTR("%r11", "%zmm24", "%zmm25")
    KRONOS_AVX512_SET_ROW_PTR("%rax", "%zmm26", "%zmm27")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t"
    KRONOS_AVX512_SET_ROW_PTR("%rcx", "%zmm28", "%zmm29")
    KRONOS_AVX512_SET_ROW_PTR("%r10", "%zmm30", "%zmm31")
    "vzeroupper\n\t"
    "ret\n\t"

    ".Lg14x32_store_gen:\n\t"
    "vbroadcastss -4(%rsp), %zmm0\n\t"
    KRONOS_AVX512_GEN_ROW_PTR("%rcx", "%zmm4",  "%zmm5")
    KRONOS_AVX512_GEN_ROW_PTR("%r10", "%zmm6",  "%zmm7")
    KRONOS_AVX512_GEN_ROW_PTR("%r11", "%zmm8",  "%zmm9")
    KRONOS_AVX512_GEN_ROW_PTR("%rax", "%zmm10", "%zmm11")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t" "addq %rdx, %r11\n\t" "addq %rdx, %rax\n\t"
    KRONOS_AVX512_GEN_ROW_PTR("%rcx", "%zmm12", "%zmm13")
    KRONOS_AVX512_GEN_ROW_PTR("%r10", "%zmm14", "%zmm15")
    KRONOS_AVX512_GEN_ROW_PTR("%r11", "%zmm16", "%zmm17")
    KRONOS_AVX512_GEN_ROW_PTR("%rax", "%zmm18", "%zmm19")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t" "addq %rdx, %r11\n\t" "addq %rdx, %rax\n\t"
    KRONOS_AVX512_GEN_ROW_PTR("%rcx", "%zmm20", "%zmm21")
    KRONOS_AVX512_GEN_ROW_PTR("%r10", "%zmm22", "%zmm23")
    KRONOS_AVX512_GEN_ROW_PTR("%r11", "%zmm24", "%zmm25")
    KRONOS_AVX512_GEN_ROW_PTR("%rax", "%zmm26", "%zmm27")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t"
    KRONOS_AVX512_GEN_ROW_PTR("%rcx", "%zmm28", "%zmm29")
    KRONOS_AVX512_GEN_ROW_PTR("%r10", "%zmm30", "%zmm31")
    "vzeroupper\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_gemm_14x32_avx512, .-kronos_gemm_14x32_avx512\n\t"

    /* ========================================================================
     *  kronos_gemm_14x32_avx512_edge (zero-copy AVX-512 opmask edge kernel!)
     *  ctrl [r9]: bits 0..1=beta_mode, bits 2..5=mr (1..14), bits 16..47=nr mask
     * ======================================================================== */
    ".globl  kronos_gemm_14x32_avx512_edge\n\t"
    ".hidden kronos_gemm_14x32_avx512_edge\n\t"
    ".type   kronos_gemm_14x32_avx512_edge, @function\n\t"
    ".p2align 5\n\t"
    "kronos_gemm_14x32_avx512_edge:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "vmovss %xmm0, -4(%rsp)\n\t"
    "vpxord %zmm4,  %zmm4,  %zmm4\n\t"  "vpxord %zmm5,  %zmm5,  %zmm5\n\t"
    "vpxord %zmm6,  %zmm6,  %zmm6\n\t"  "vpxord %zmm7,  %zmm7,  %zmm7\n\t"
    "vpxord %zmm8,  %zmm8,  %zmm8\n\t"  "vpxord %zmm9,  %zmm9,  %zmm9\n\t"
    "vpxord %zmm10, %zmm10, %zmm10\n\t" "vpxord %zmm11, %zmm11, %zmm11\n\t"
    "vpxord %zmm12, %zmm12, %zmm12\n\t" "vpxord %zmm13, %zmm13, %zmm13\n\t"
    "vpxord %zmm14, %zmm14, %zmm14\n\t" "vpxord %zmm15, %zmm15, %zmm15\n\t"
    "vpxord %zmm16, %zmm16, %zmm16\n\t" "vpxord %zmm17, %zmm17, %zmm17\n\t"
    "vpxord %zmm18, %zmm18, %zmm18\n\t" "vpxord %zmm19, %zmm19, %zmm19\n\t"
    "vpxord %zmm20, %zmm20, %zmm20\n\t" "vpxord %zmm21, %zmm21, %zmm21\n\t"
    "vpxord %zmm22, %zmm22, %zmm22\n\t" "vpxord %zmm23, %zmm23, %zmm23\n\t"
    "vpxord %zmm24, %zmm24, %zmm24\n\t" "vpxord %zmm25, %zmm25, %zmm25\n\t"
    "vpxord %zmm26, %zmm26, %zmm26\n\t" "vpxord %zmm27, %zmm27, %zmm27\n\t"
    "vpxord %zmm28, %zmm28, %zmm28\n\t" "vpxord %zmm29, %zmm29, %zmm29\n\t"
    "vpxord %zmm30, %zmm30, %zmm30\n\t" "vpxord %zmm31, %zmm31, %zmm31\n\t"

    "testq %rdi, %rdi\n\t"
    "jle .Le14_store\n\t"
    "movq %rdi, %rax\n\t"
    "shrq $2, %rax\n\t"
    "jz .Le14_tail\n\t"
    ".p2align 5\n\t"
    ".Le14_loop4:\n\t"
    KRONOS_AVX512_14X32_RANK1(  0,   0)
    KRONOS_AVX512_14X32_RANK1( 56, 128)
    KRONOS_AVX512_14X32_RANK1(112, 256)
    KRONOS_AVX512_14X32_RANK1(168, 384)
    "addq $224, %rsi\n\t"
    "addq $512, %rdx\n\t"
    "decq %rax\n\t"
    "jnz .Le14_loop4\n\t"
    "andq $3, %rdi\n\t"
    "jz .Le14_store\n\t"
    ".Le14_tail:\n\t"
    KRONOS_AVX512_14X32_RANK1(0, 0)
    "addq $56,  %rsi\n\t"
    "addq $128, %rdx\n\t"
    "decq %rdi\n\t"
    "jnz .Le14_tail\n\t"

    ".Le14_store:\n\t"
    "movq %r9, %r10\n\t"
    "shrq $2, %r10\n\t"
    "andq $15, %r10\n\t"   /* r10 = mr (1..14) */
    "jz .Le14_done\n\t"
    "movq %r9, %r11\n\t"
    "shrq $16, %r11\n\t"
    "kmovw %r11d, %k1\n\t"
    "shrq $16, %r11\n\t"
    "kmovw %r11d, %k2\n\t"
    "andq $3, %r9\n\t"
    "jz .Le14_set\n\t"
    "cmpq $1, %r9\n\t"
    "jne .Le14_gen\n\t"

    KRONOS_EDGE_14X32_ROW_ADD("%zmm4",  "%zmm5",  ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm6",  "%zmm7",  ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm8",  "%zmm9",  ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm10", "%zmm11", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm12", "%zmm13", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm14", "%zmm15", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm16", "%zmm17", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm18", "%zmm19", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm20", "%zmm21", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm22", "%zmm23", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm24", "%zmm25", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm26", "%zmm27", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm28", "%zmm29", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_ADD("%zmm30", "%zmm31", ".Le14_done")
    "jmp .Le14_done\n\t"

    ".Le14_set:\n\t"
    KRONOS_EDGE_14X32_ROW_SET("%zmm4",  "%zmm5",  ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm6",  "%zmm7",  ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm8",  "%zmm9",  ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm10", "%zmm11", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm12", "%zmm13", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm14", "%zmm15", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm16", "%zmm17", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm18", "%zmm19", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm20", "%zmm21", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm22", "%zmm23", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm24", "%zmm25", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm26", "%zmm27", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm28", "%zmm29", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_SET("%zmm30", "%zmm31", ".Le14_done")
    "jmp .Le14_done\n\t"

    ".Le14_gen:\n\t"
    "vbroadcastss -4(%rsp), %zmm0\n\t"
    KRONOS_EDGE_14X32_ROW_GEN("%zmm4",  "%zmm5",  ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm6",  "%zmm7",  ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm8",  "%zmm9",  ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm10", "%zmm11", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm12", "%zmm13", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm14", "%zmm15", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm16", "%zmm17", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm18", "%zmm19", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm20", "%zmm21", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm22", "%zmm23", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm24", "%zmm25", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm26", "%zmm27", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm28", "%zmm29", ".Le14_done")
    KRONOS_EDGE_14X32_ROW_GEN("%zmm30", "%zmm31", ".Le14_done")

    ".Le14_done:\n\t"
    "vzeroupper\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_gemm_14x32_avx512_edge, .-kronos_gemm_14x32_avx512_edge\n\t"

    /* ========================================================================
     *  kronos_gemm_9x48_avx512 (32/32 ZMM, 2.25 FMAs/load, staggered prefetchw)
     * ======================================================================== */
    ".globl  kronos_gemm_9x48_avx512\n\t"
    ".hidden kronos_gemm_9x48_avx512\n\t"
    ".type   kronos_gemm_9x48_avx512, @function\n\t"
    ".p2align 5\n\t"
    "kronos_gemm_9x48_avx512:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "vmovss %xmm0, -4(%rsp)\n\t"
    "vpxord %zmm5,  %zmm5,  %zmm5\n\t"  "vpxord %zmm6,  %zmm6,  %zmm6\n\t"  "vpxord %zmm7,  %zmm7,  %zmm7\n\t"
    "vpxord %zmm8,  %zmm8,  %zmm8\n\t"  "vpxord %zmm9,  %zmm9,  %zmm9\n\t"  "vpxord %zmm10, %zmm10, %zmm10\n\t"
    "vpxord %zmm11, %zmm11, %zmm11\n\t" "vpxord %zmm12, %zmm12, %zmm12\n\t" "vpxord %zmm13, %zmm13, %zmm13\n\t"
    "vpxord %zmm14, %zmm14, %zmm14\n\t" "vpxord %zmm15, %zmm15, %zmm15\n\t" "vpxord %zmm16, %zmm16, %zmm16\n\t"
    "vpxord %zmm17, %zmm17, %zmm17\n\t" "vpxord %zmm18, %zmm18, %zmm18\n\t" "vpxord %zmm19, %zmm19, %zmm19\n\t"
    "vpxord %zmm20, %zmm20, %zmm20\n\t" "vpxord %zmm21, %zmm21, %zmm21\n\t" "vpxord %zmm22, %zmm22, %zmm22\n\t"
    "vpxord %zmm23, %zmm23, %zmm23\n\t" "vpxord %zmm24, %zmm24, %zmm24\n\t" "vpxord %zmm25, %zmm25, %zmm25\n\t"
    "vpxord %zmm26, %zmm26, %zmm26\n\t" "vpxord %zmm27, %zmm27, %zmm27\n\t" "vpxord %zmm28, %zmm28, %zmm28\n\t"
    "vpxord %zmm29, %zmm29, %zmm29\n\t" "vpxord %zmm30, %zmm30, %zmm30\n\t" "vpxord %zmm31, %zmm31, %zmm31\n\t"

    "testq %rdi, %rdi\n\t"
    "jle .Lg9x48_kc_zero\n\t"

    "movq %rdi, %rax\n\t"
    "shrq $2, %rax\n\t"
    "cmpq $2, %rax\n\t"
    "jbe .Lg9x48_tail_prep\n\t"
    "subq $2, %rax\n\t"

    ".p2align 5\n\t"
    ".Lg9x48_loop4:\n\t"
    "prefetcht0  960(%rdx)\n\t" "prefetcht0 1024(%rdx)\n\t" "prefetcht0 1088(%rdx)\n\t" "prefetcht0 384(%rsi)\n\t"
    KRONOS_AVX512_9X48_RANK1(  0,   0)
    "prefetcht0 1152(%rdx)\n\t" "prefetcht0 1216(%rdx)\n\t" "prefetcht0 1280(%rdx)\n\t"
    KRONOS_AVX512_9X48_RANK1( 36, 192)
    "prefetcht0 1344(%rdx)\n\t" "prefetcht0 1408(%rdx)\n\t" "prefetcht0 1472(%rdx)\n\t" "prefetcht0 448(%rsi)\n\t"
    KRONOS_AVX512_9X48_RANK1( 72, 384)
    "prefetcht0 1536(%rdx)\n\t" "prefetcht0 1600(%rdx)\n\t" "prefetcht0 1664(%rdx)\n\t"
    KRONOS_AVX512_9X48_RANK1(108, 576)
    "addq $144, %rsi\n\t"
    "addq $768, %rdx\n\t"
    "decq %rax\n\t"
    "jnz .Lg9x48_loop4\n\t"
    "andq $3, %rdi\n\t"
    "addq $8, %rdi\n\t"

    ".Lg9x48_tail_prep:\n\t"
    "movq %rcx, %r10\n\t"
    "xorl %r11d, %r11d\n\t"
    "cmpl $0, g_cpu_has_prefetchw_asm(%rip)\n\t"
    "je .Lg9x48_tail_loop\n\t"
    "movl $5, %r11d\n\t" /* 2 rows/step for 4 steps + 1 row on step 5 = all 9 rows! */

    ".p2align 4\n\t"
    ".Lg9x48_tail_loop:\n\t"
    "testl %r11d, %r11d\n\t"
    "jz .Lg9x48_no_pw\n\t"
    "decl %r11d\n\t"
    "prefetchw (%r10)\n\t" "prefetchw 64(%r10)\n\t" "prefetchw 128(%r10)\n\t" "prefetchw 188(%r10)\n\t"
    "addq %r8, %r10\n\t"
    "testl %r11d, %r11d\n\t"
    "jz .Lg9x48_no_pw\n\t"
    "prefetchw (%r10)\n\t" "prefetchw 64(%r10)\n\t" "prefetchw 128(%r10)\n\t" "prefetchw 188(%r10)\n\t"
    "addq %r8, %r10\n\t"
    ".Lg9x48_no_pw:\n\t"
    KRONOS_AVX512_9X48_RANK1(0, 0)
    "addq $36,  %rsi\n\t"
    "addq $192, %rdx\n\t"
    "decq %rdi\n\t"
    "jnz .Lg9x48_tail_loop\n\t"

    ".Lg9x48_store:\n\t"
    "leaq (%rcx, %r8),    %r10\n\t"
    "leaq (%rcx, %r8, 2), %r11\n\t"
    "leaq (%r10, %r8, 2), %rax\n\t"
    "leaq (,%r8, 4),      %rdx\n\t"
    "testq %r9, %r9\n\t"
    "jz .Lg9x48_store_zero\n\t"
    "cmpq $1, %r9\n\t"
    "jne .Lg9x48_store_gen\n\t"

    KRONOS_AVX512_ADD_ROW3("%rcx", "%zmm5",  "%zmm6",  "%zmm7")
    KRONOS_AVX512_ADD_ROW3("%r10", "%zmm8",  "%zmm9",  "%zmm10")
    KRONOS_AVX512_ADD_ROW3("%r11", "%zmm11", "%zmm12", "%zmm13")
    KRONOS_AVX512_ADD_ROW3("%rax", "%zmm14", "%zmm15", "%zmm16")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t" "addq %rdx, %r11\n\t" "addq %rdx, %rax\n\t"
    KRONOS_AVX512_ADD_ROW3("%rcx", "%zmm17", "%zmm18", "%zmm19")
    KRONOS_AVX512_ADD_ROW3("%r10", "%zmm20", "%zmm21", "%zmm22")
    KRONOS_AVX512_ADD_ROW3("%r11", "%zmm23", "%zmm24", "%zmm25")
    KRONOS_AVX512_ADD_ROW3("%rax", "%zmm26", "%zmm27", "%zmm28")
    "addq %rdx, %rcx\n\t"
    KRONOS_AVX512_ADD_ROW3("%rcx", "%zmm29", "%zmm30", "%zmm31")
    "vzeroupper\n\t"
    "ret\n\t"

    ".Lg9x48_kc_zero:\n\t"
    "cmpq $1, %r9\n\t"
    "jne .Lg9x48_store\n\t"
    "vzeroupper\n\t"
    "ret\n\t"

    ".Lg9x48_store_zero:\n\t"
    KRONOS_AVX512_SET_ROW3("%rcx", "%zmm5",  "%zmm6",  "%zmm7")
    KRONOS_AVX512_SET_ROW3("%r10", "%zmm8",  "%zmm9",  "%zmm10")
    KRONOS_AVX512_SET_ROW3("%r11", "%zmm11", "%zmm12", "%zmm13")
    KRONOS_AVX512_SET_ROW3("%rax", "%zmm14", "%zmm15", "%zmm16")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t" "addq %rdx, %r11\n\t" "addq %rdx, %rax\n\t"
    KRONOS_AVX512_SET_ROW3("%rcx", "%zmm17", "%zmm18", "%zmm19")
    KRONOS_AVX512_SET_ROW3("%r10", "%zmm20", "%zmm21", "%zmm22")
    KRONOS_AVX512_SET_ROW3("%r11", "%zmm23", "%zmm24", "%zmm25")
    KRONOS_AVX512_SET_ROW3("%rax", "%zmm26", "%zmm27", "%zmm28")
    "addq %rdx, %rcx\n\t"
    KRONOS_AVX512_SET_ROW3("%rcx", "%zmm29", "%zmm30", "%zmm31")
    "vzeroupper\n\t"
    "ret\n\t"

    ".Lg9x48_store_gen:\n\t"
    "vbroadcastss -4(%rsp), %zmm0\n\t"
    KRONOS_AVX512_GEN_ROW3("%rcx", "%zmm5",  "%zmm6",  "%zmm7")
    KRONOS_AVX512_GEN_ROW3("%r10", "%zmm8",  "%zmm9",  "%zmm10")
    KRONOS_AVX512_GEN_ROW3("%r11", "%zmm11", "%zmm12", "%zmm13")
    KRONOS_AVX512_GEN_ROW3("%rax", "%zmm14", "%zmm15", "%zmm16")
    "addq %rdx, %rcx\n\t" "addq %rdx, %r10\n\t" "addq %rdx, %r11\n\t" "addq %rdx, %rax\n\t"
    KRONOS_AVX512_GEN_ROW3("%rcx", "%zmm17", "%zmm18", "%zmm19")
    KRONOS_AVX512_GEN_ROW3("%r10", "%zmm20", "%zmm21", "%zmm22")
    KRONOS_AVX512_GEN_ROW3("%r11", "%zmm23", "%zmm24", "%zmm25")
    KRONOS_AVX512_GEN_ROW3("%rax", "%zmm26", "%zmm27", "%zmm28")
    "addq %rdx, %rcx\n\t"
    KRONOS_AVX512_GEN_ROW3("%rcx", "%zmm29", "%zmm30", "%zmm31")
    "vzeroupper\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_gemm_9x48_avx512, .-kronos_gemm_9x48_avx512\n\t"

    /* ========================================================================
     *  kronos_gemm_9x48_avx512_edge (zero-copy 3-opmask k1/k2/k3 edge kernel!)
     * ======================================================================== */
    ".globl  kronos_gemm_9x48_avx512_edge\n\t"
    ".hidden kronos_gemm_9x48_avx512_edge\n\t"
    ".type   kronos_gemm_9x48_avx512_edge, @function\n\t"
    ".p2align 5\n\t"
    "kronos_gemm_9x48_avx512_edge:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "vmovss %xmm0, -4(%rsp)\n\t"
    "vpxord %zmm5,  %zmm5,  %zmm5\n\t"  "vpxord %zmm6,  %zmm6,  %zmm6\n\t"  "vpxord %zmm7,  %zmm7,  %zmm7\n\t"
    "vpxord %zmm8,  %zmm8,  %zmm8\n\t"  "vpxord %zmm9,  %zmm9,  %zmm9\n\t"  "vpxord %zmm10, %zmm10, %zmm10\n\t"
    "vpxord %zmm11, %zmm11, %zmm11\n\t" "vpxord %zmm12, %zmm12, %zmm12\n\t" "vpxord %zmm13, %zmm13, %zmm13\n\t"
    "vpxord %zmm14, %zmm14, %zmm14\n\t" "vpxord %zmm15, %zmm15, %zmm15\n\t" "vpxord %zmm16, %zmm16, %zmm16\n\t"
    "vpxord %zmm17, %zmm17, %zmm17\n\t" "vpxord %zmm18, %zmm18, %zmm18\n\t" "vpxord %zmm19, %zmm19, %zmm19\n\t"
    "vpxord %zmm20, %zmm20, %zmm20\n\t" "vpxord %zmm21, %zmm21, %zmm21\n\t" "vpxord %zmm22, %zmm22, %zmm22\n\t"
    "vpxord %zmm23, %zmm23, %zmm23\n\t" "vpxord %zmm24, %zmm24, %zmm24\n\t" "vpxord %zmm25, %zmm25, %zmm25\n\t"
    "vpxord %zmm26, %zmm26, %zmm26\n\t" "vpxord %zmm27, %zmm27, %zmm27\n\t" "vpxord %zmm28, %zmm28, %zmm28\n\t"
    "vpxord %zmm29, %zmm29, %zmm29\n\t" "vpxord %zmm30, %zmm30, %zmm30\n\t" "vpxord %zmm31, %zmm31, %zmm31\n\t"

    "testq %rdi, %rdi\n\t"
    "jle .Le9_store\n\t"
    "movq %rdi, %rax\n\t"
    "shrq $2, %rax\n\t"
    "jz .Le9_tail\n\t"
    ".p2align 5\n\t"
    ".Le9_loop4:\n\t"
    KRONOS_AVX512_9X48_RANK1(  0,   0)
    KRONOS_AVX512_9X48_RANK1( 36, 192)
    KRONOS_AVX512_9X48_RANK1( 72, 384)
    KRONOS_AVX512_9X48_RANK1(108, 576)
    "addq $144, %rsi\n\t"
    "addq $768, %rdx\n\t"
    "decq %rax\n\t"
    "jnz .Le9_loop4\n\t"
    "andq $3, %rdi\n\t"
    "jz .Le9_store\n\t"
    ".Le9_tail:\n\t"
    KRONOS_AVX512_9X48_RANK1(0, 0)
    "addq $36,  %rsi\n\t"
    "addq $192, %rdx\n\t"
    "decq %rdi\n\t"
    "jnz .Le9_tail\n\t"

    ".Le9_store:\n\t"
    "movq %r9, %r10\n\t"
    "shrq $2, %r10\n\t"
    "andq $15, %r10\n\t"
    "jz .Le9_done\n\t"
    "movq %r9, %r11\n\t"
    "shrq $16, %r11\n\t" "kmovw %r11d, %k1\n\t"
    "shrq $16, %r11\n\t" "kmovw %r11d, %k2\n\t"
    "shrq $16, %r11\n\t" "kmovw %r11d, %k3\n\t"
    "andq $3, %r9\n\t"
    "jz .Le9_set\n\t"
    "cmpq $1, %r9\n\t"
    "jne .Le9_gen\n\t"

    KRONOS_EDGE_9X48_ROW_ADD("%zmm5",  "%zmm6",  "%zmm7",  ".Le9_done")
    KRONOS_EDGE_9X48_ROW_ADD("%zmm8",  "%zmm9",  "%zmm10", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_ADD("%zmm11", "%zmm12", "%zmm13", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_ADD("%zmm14", "%zmm15", "%zmm16", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_ADD("%zmm17", "%zmm18", "%zmm19", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_ADD("%zmm20", "%zmm21", "%zmm22", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_ADD("%zmm23", "%zmm24", "%zmm25", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_ADD("%zmm26", "%zmm27", "%zmm28", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_ADD("%zmm29", "%zmm30", "%zmm31", ".Le9_done")
    "jmp .Le9_done\n\t"

    ".Le9_set:\n\t"
    KRONOS_EDGE_9X48_ROW_SET("%zmm5",  "%zmm6",  "%zmm7",  ".Le9_done")
    KRONOS_EDGE_9X48_ROW_SET("%zmm8",  "%zmm9",  "%zmm10", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_SET("%zmm11", "%zmm12", "%zmm13", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_SET("%zmm14", "%zmm15", "%zmm16", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_SET("%zmm17", "%zmm18", "%zmm19", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_SET("%zmm20", "%zmm21", "%zmm22", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_SET("%zmm23", "%zmm24", "%zmm25", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_SET("%zmm26", "%zmm27", "%zmm28", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_SET("%zmm29", "%zmm30", "%zmm31", ".Le9_done")
    "jmp .Le9_done\n\t"

    ".Le9_gen:\n\t"
    "vbroadcastss -4(%rsp), %zmm0\n\t"
    KRONOS_EDGE_9X48_ROW_GEN("%zmm5",  "%zmm6",  "%zmm7",  ".Le9_done")
    KRONOS_EDGE_9X48_ROW_GEN("%zmm8",  "%zmm9",  "%zmm10", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_GEN("%zmm11", "%zmm12", "%zmm13", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_GEN("%zmm14", "%zmm15", "%zmm16", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_GEN("%zmm17", "%zmm18", "%zmm19", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_GEN("%zmm20", "%zmm21", "%zmm22", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_GEN("%zmm23", "%zmm24", "%zmm25", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_GEN("%zmm26", "%zmm27", "%zmm28", ".Le9_done")
    KRONOS_EDGE_9X48_ROW_GEN("%zmm29", "%zmm30", "%zmm31", ".Le9_done")

    ".Le9_done:\n\t"
    "vzeroupper\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_gemm_9x48_avx512_edge, .-kronos_gemm_9x48_avx512_edge\n\t"

    /* ========================================================================
     *  SIMD Assembly B-Packers (NR=48 AVX-512F, NR=32 AVX-512F, NR=16 AVX)
     * ======================================================================== */
    ".globl  kronos_pack_b_nr48_avx512\n\t"
    ".hidden kronos_pack_b_nr48_avx512\n\t"
    ".type   kronos_pack_b_nr48_avx512, @function\n\t"
    ".p2align 5\n\t"
    "kronos_pack_b_nr48_avx512:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "testq %rdi, %rdi\n\t"
    "jle .Lpb48_done\n\t"
    "leaq (,%rdx,4), %r8\n\t"
    ".p2align 4\n\t"
    ".Lpb48_loop:\n\t"
    "prefetcht0   0(%rsi, %r8)\n\t"
    "prefetcht0  64(%rsi, %r8)\n\t"
    "prefetcht0 128(%rsi, %r8)\n\t"
    "vmovups   0(%rsi), %zmm0\n\t"
    "vmovups  64(%rsi), %zmm1\n\t"
    "vmovups 128(%rsi), %zmm2\n\t"
    "vmovaps %zmm0,   0(%rcx)\n\t"
    "vmovaps %zmm1,  64(%rcx)\n\t"
    "vmovaps %zmm2, 128(%rcx)\n\t"
    "addq %rdx, %rsi\n\t"
    "addq $192, %rcx\n\t"
    "decq %rdi\n\t"
    "jnz .Lpb48_loop\n\t"
    "vzeroupper\n\t"
    ".Lpb48_done:\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_pack_b_nr48_avx512, .-kronos_pack_b_nr48_avx512\n\t"

    ".globl  kronos_pack_b_nr32_avx512\n\t"
    ".hidden kronos_pack_b_nr32_avx512\n\t"
    ".type   kronos_pack_b_nr32_avx512, @function\n\t"
    ".p2align 5\n\t"
    "kronos_pack_b_nr32_avx512:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "testq %rdi, %rdi\n\t"
    "jle .Lpb32_done\n\t"
    "leaq (,%rdx,4), %r8\n\t"
    ".p2align 4\n\t"
    ".Lpb32_loop:\n\t"
    "prefetcht0 (%rsi, %r8)\n\t"
    "prefetcht0 64(%rsi, %r8)\n\t"
    "vmovups  0(%rsi), %zmm0\n\t"
    "vmovups 64(%rsi), %zmm1\n\t"
    "vmovaps %zmm0,  0(%rcx)\n\t"
    "vmovaps %zmm1, 64(%rcx)\n\t"
    "addq %rdx, %rsi\n\t"
    "addq $128, %rcx\n\t"
    "decq %rdi\n\t"
    "jnz .Lpb32_loop\n\t"
    "vzeroupper\n\t"
    ".Lpb32_done:\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_pack_b_nr32_avx512, .-kronos_pack_b_nr32_avx512\n\t"

    ".globl  kronos_pack_b_nr16_avx2\n\t"
    ".hidden kronos_pack_b_nr16_avx2\n\t"
    ".type   kronos_pack_b_nr16_avx2, @function\n\t"
    ".p2align 5\n\t"
    "kronos_pack_b_nr16_avx2:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "testq %rdi, %rdi\n\t"
    "jle .Lpb16_done\n\t"
    "leaq (,%rdx,4), %r8\n\t"
    ".p2align 4\n\t"
    ".Lpb16_loop:\n\t"
    "prefetcht0 (%rsi, %r8)\n\t"
    "vmovups  0(%rsi), %ymm0\n\t"
    "vmovups 32(%rsi), %ymm1\n\t"
    "vmovaps %ymm0,  0(%rcx)\n\t"
    "vmovaps %ymm1, 32(%rcx)\n\t"
    "addq %rdx, %rsi\n\t"
    "addq $64,  %rcx\n\t"
    "decq %rdi\n\t"
    "jnz .Lpb16_loop\n\t"
    "vzeroupper\n\t"
    ".Lpb16_done:\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_pack_b_nr16_avx2, .-kronos_pack_b_nr16_avx2\n\t"

    /* ========================================================================
     *  Branch-Free Hoisted-Alpha SIMD A-Packers (32B-aligned red-zone spill!)
     * ======================================================================== */
    ".globl  kronos_pack_a_mr6_avx2_8k\n\t"
    ".hidden kronos_pack_a_mr6_avx2_8k\n\t"
    ".type   kronos_pack_a_mr6_avx2_8k, @function\n\t"
    ".p2align 5\n\t"
    "kronos_pack_a_mr6_avx2_8k:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "shrq $3, %rdi\n\t"
    "jz .Lpa6_done\n\t"
    "movq %r8,  %r11\n\t"
    "movq %rdx, %r8\n\t"
    "leaq (,%rdx,2), %r9\n\t"
    "leaq (%rdx,%rdx,2), %r10\n\t"
    "testq %r11, %r11\n\t"
    "jnz .Lpa6_scale_prep\n\t"
    ".p2align 5\n\t"
    ".Lpa6_loop_plain:\n\t"
    KRONOS_PACK_A_MR6_BODY("")
    "decq %rdi\n\t"
    "jnz .Lpa6_loop_plain\n\t"
    "vzeroupper\n\t"
    ".Lpa6_done:\n\t"
    "ret\n\t"
    ".Lpa6_scale_prep:\n\t"
    /* Strictly 32-byte aligned inside [-64..-32](%rsp) red zone -> zero cache-line splits! */
    "leaq -32(%rsp), %r11\n\t"
    "andq $-32, %r11\n\t"
    "vshufps $0, %xmm0, %xmm0, %xmm0\n\t"
    "vinsertf128 $1, %xmm0, %ymm0, %ymm0\n\t"
    "vmovaps %ymm0, (%r11)\n\t"
    ".p2align 5\n\t"
    ".Lpa6_loop_scale:\n\t"
    KRONOS_PACK_A_MR6_BODY(
        "vmulps (%r11), %ymm0, %ymm0\n\t"
        "vmulps (%r11), %ymm1, %ymm1\n\t"
        "vmulps (%r11), %ymm2, %ymm2\n\t"
        "vmulps (%r11), %ymm3, %ymm3\n\t"
        "vmulps (%r11), %ymm4, %ymm4\n\t"
        "vmulps (%r11), %ymm5, %ymm5\n\t")
    "decq %rdi\n\t"
    "jnz .Lpa6_loop_scale\n\t"
    "vzeroupper\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_pack_a_mr6_avx2_8k, .-kronos_pack_a_mr6_avx2_8k\n\t"

    ".globl  kronos_pack_a_mr14_avx512_16k\n\t"
    ".hidden kronos_pack_a_mr14_avx512_16k\n\t"
    ".type   kronos_pack_a_mr14_avx512_16k, @function\n\t"
    ".p2align 5\n\t"
    "kronos_pack_a_mr14_avx512_16k:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "shrq $4, %rdi\n\t"
    "jz .Lpa14z_done\n\t"
    "movq %r8,  %r11\n\t"
    "movq %rdx, %r8\n\t"
    "leaq (,%rdx,2), %r9\n\t"
    "leaq (%rdx,%rdx,2), %r10\n\t"
    "testq %r11, %r11\n\t"
    "jnz .Lpa14z_scale_prep\n\t"
    ".p2align 5\n\t"
    ".Lpa14z_loop_plain:\n\t"
    KRONOS_PACK_A_MR14_BODY("", "")
    "decq %rdi\n\t"
    "jnz .Lpa14z_loop_plain\n\t"
    "vzeroupper\n\t"
    ".Lpa14z_done:\n\t"
    "ret\n\t"
    ".Lpa14z_scale_prep:\n\t"
    "vbroadcastss %xmm0, %zmm31\n\t"
    ".p2align 5\n\t"
    ".Lpa14z_loop_scale:\n\t"
    KRONOS_PACK_A_MR14_BODY(
        "vmulps %zmm31, %zmm0, %zmm0\n\t"
        "vmulps %zmm31, %zmm1, %zmm1\n\t"
        "vmulps %zmm31, %zmm2, %zmm2\n\t"
        "vmulps %zmm31, %zmm3, %zmm3\n\t",
        "vmulps %zmm31, %zmm4, %zmm4\n\t"
        "vmulps %zmm31, %zmm5, %zmm5\n\t")
    "decq %rdi\n\t"
    "jnz .Lpa14z_loop_scale\n\t"
    "vzeroupper\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_pack_a_mr14_avx512_16k, .-kronos_pack_a_mr14_avx512_16k\n\t"

    ".globl  kronos_pack_a_mr9_avx512_16k\n\t"
    ".hidden kronos_pack_a_mr9_avx512_16k\n\t"
    ".type   kronos_pack_a_mr9_avx512_16k, @function\n\t"
    ".p2align 5\n\t"
    "kronos_pack_a_mr9_avx512_16k:\n\t"
    ".cfi_startproc\n\t"
    ".byte 0xf3, 0x0f, 0x1e, 0xfa\n\t"
    "shrq $4, %rdi\n\t"
    "jz .Lpa9z_done\n\t"
    "movq %r8,  %r11\n\t"
    "movq %rdx, %r8\n\t"
    "leaq (,%rdx,2), %r9\n\t"
    "leaq (%rdx,%rdx,2), %r10\n\t"
    "testq %r11, %r11\n\t"
    "jnz .Lpa9z_scale_prep\n\t"
    ".p2align 5\n\t"
    ".Lpa9z_loop_plain:\n\t"
    KRONOS_PACK_A_MR9_BODY("", "")
    "decq %rdi\n\t"
    "jnz .Lpa9z_loop_plain\n\t"
    "vzeroupper\n\t"
    ".Lpa9z_done:\n\t"
    "ret\n\t"
    ".Lpa9z_scale_prep:\n\t"
    "vbroadcastss %xmm0, %zmm31\n\t"
    ".p2align 5\n\t"
    ".Lpa9z_loop_scale:\n\t"
    KRONOS_PACK_A_MR9_BODY(
        "vmulps %zmm31, %zmm0, %zmm0\n\t"
        "vmulps %zmm31, %zmm1, %zmm1\n\t"
        "vmulps %zmm31, %zmm2, %zmm2\n\t"
        "vmulps %zmm31, %zmm3, %zmm3\n\t",
        "vmulps %zmm31, %zmm4, %zmm4\n\t")
    "decq %rdi\n\t"
    "jnz .Lpa9z_loop_scale\n\t"
    "vzeroupper\n\t"
    "ret\n\t"
    ".cfi_endproc\n\t"
    ".size kronos_pack_a_mr9_avx512_16k, .-kronos_pack_a_mr9_avx512_16k\n\t"
    ".popsection\n\t"
    ".pushsection .note.GNU-stack,\"\",@progbits\n\t"
    ".popsection\n\t"
);

static void kronos_gemm_6x16_fallback(int64_t kc, const float *Ap, const float *Bp,
                                      float *C, int64_t ldc_bytes, int64_t beta_mode, float beta) {
    float acc[6][16];
    memset(acc, 0, sizeof(acc));
    for (int64_t p = 0; p < kc; p++) {
        const float *a = Ap + p * 6;
        const float *b = Bp + p * 16;
        for (int r = 0; r < 6; r++) {
            float av = a[r];
            for (int c = 0; c < 16; c++) acc[r][c] += av * b[c];
        }
    }
    int64_t ldc = ldc_bytes / (int64_t)sizeof(float);
    if (beta_mode == 0) {
        for (int r = 0; r < 6; r++) memcpy(C + r * ldc, acc[r], 16 * sizeof(float));
    } else if (beta_mode == 1) {
        for (int r = 0; r < 6; r++) {
            float *crow = C + r * ldc;
            for (int c = 0; c < 16; c++) crow[c] += acc[r][c];
        }
    } else {
        for (int r = 0; r < 6; r++) {
            float *crow = C + r * ldc;
            for (int c = 0; c < 16; c++) crow[c] = beta * crow[c] + acc[r][c];
        }
    }
}

/* Transposed A packer (A is K x M row-major -> element (i, p) is A[p * lda + i] — contiguous!) */
static void pack_A_trans_mr(int MR, int mc, int kc, const float *A, int lda, float *Ap, float alpha) {
    int scale_alpha = (alpha != 1.0f);
    for (int i = 0; i < mc; i += MR) {
        int mr = (i + MR <= mc) ? MR : (mc - i);
        for (int p = 0; p < kc; p++) {
            const float *src = A + (size_t)p * (size_t)lda + (size_t)i;
            if (!scale_alpha) {
                for (int r = 0; r < mr; r++) Ap[r] = src[r];
            } else {
                for (int r = 0; r < mr; r++) Ap[r] = src[r] * alpha;
            }
            for (int r = mr; r < MR; r++) Ap[r] = 0.0f;
            Ap += MR;
        }
    }
}

static void pack_A_mr6(int transA, int mc, int kc, const float *A, int lda, float *Ap, float alpha) {
    if (UNLIKELY(transA)) { pack_A_trans_mr(6, mc, kc, A, lda, Ap, alpha); return; }
    int scale_alpha = (alpha != 1.0f) ? 1 : 0;
    int64_t lda_bytes = (int64_t)lda * (int64_t)sizeof(float);
    for (int i = 0; i < mc; i += 6) {
        int mr = (i + 6 <= mc) ? 6 : (mc - i);
        if (LIKELY(mr == 6)) {
            int p = 0;
            if (LIKELY(g_cpu.has_avx && kc >= 8)) {
                int kc8 = kc & ~7;
                kronos_pack_a_mr6_avx2_8k(kc8, A + (size_t)i * (size_t)lda, lda_bytes,
                                          Ap, scale_alpha, alpha);
                Ap += (size_t)kc8 * 6u;
                p   = kc8;
            }
            const float *rows[6];
            for (int r = 0; r < 6; r++) rows[r] = A + (size_t)(i + r) * (size_t)lda;
            if (!scale_alpha) {
                for (; p < kc; p++) { for (int r = 0; r < 6; r++) Ap[r] = rows[r][p]; Ap += 6; }
            } else {
                for (; p < kc; p++) { for (int r = 0; r < 6; r++) Ap[r] = rows[r][p] * alpha; Ap += 6; }
            }
        } else {
            for (int p = 0; p < kc; p++) {
                for (int r = 0; r < mr; r++)
                    Ap[r] = A[(size_t)(i + r) * (size_t)lda + (size_t)p] * alpha;
                for (int r = mr; r < 6; r++) Ap[r] = 0.0f;
                Ap += 6;
            }
        }
    }
}

static void pack_A_mr14(int transA, int mc, int kc, const float *A, int lda, float *Ap, float alpha) {
    if (UNLIKELY(transA)) { pack_A_trans_mr(14, mc, kc, A, lda, Ap, alpha); return; }
    int scale_alpha = (alpha != 1.0f) ? 1 : 0;
    int64_t lda_bytes = (int64_t)lda * (int64_t)sizeof(float);
    for (int i = 0; i < mc; i += 14) {
        int mr = (i + 14 <= mc) ? 14 : (mc - i);
        if (LIKELY(mr == 14)) {
            int p = 0;
            if (LIKELY(kc >= 16)) {
                int kc16 = kc & ~15;
                kronos_pack_a_mr14_avx512_16k(kc16, A + (size_t)i * (size_t)lda, lda_bytes,
                                              Ap, scale_alpha, alpha);
                Ap += (size_t)kc16 * 14u;
                p   = kc16;
            }
            const float *rows[14];
            for (int r = 0; r < 14; r++) rows[r] = A + (size_t)(i + r) * (size_t)lda;
            if (!scale_alpha) {
                for (; p < kc; p++) { for (int r = 0; r < 14; r++) Ap[r] = rows[r][p]; Ap += 14; }
            } else {
                for (; p < kc; p++) { for (int r = 0; r < 14; r++) Ap[r] = rows[r][p] * alpha; Ap += 14; }
            }
        } else {
            for (int p = 0; p < kc; p++) {
                for (int r = 0; r < mr; r++)
                    Ap[r] = A[(size_t)(i + r) * (size_t)lda + (size_t)p] * alpha;
                for (int r = mr; r < 14; r++) Ap[r] = 0.0f;
                Ap += 14;
            }
        }
    }
}

static void pack_A_mr9(int transA, int mc, int kc, const float *A, int lda, float *Ap, float alpha) {
    if (UNLIKELY(transA)) { pack_A_trans_mr(9, mc, kc, A, lda, Ap, alpha); return; }
    int scale_alpha = (alpha != 1.0f) ? 1 : 0;
    int64_t lda_bytes = (int64_t)lda * (int64_t)sizeof(float);
    for (int i = 0; i < mc; i += 9) {
        int mr = (i + 9 <= mc) ? 9 : (mc - i);
        if (LIKELY(mr == 9)) {
            int p = 0;
            if (LIKELY(kc >= 16)) {
                int kc16 = kc & ~15;
                kronos_pack_a_mr9_avx512_16k(kc16, A + (size_t)i * (size_t)lda, lda_bytes,
                                             Ap, scale_alpha, alpha);
                Ap += (size_t)kc16 * 9u;
                p   = kc16;
            }
            const float *rows[9];
            for (int r = 0; r < 9; r++) rows[r] = A + (size_t)(i + r) * (size_t)lda;
            if (!scale_alpha) {
                for (; p < kc; p++) {
                    for (int r = 0; r < 9; r++) Ap[r] = rows[r][p];
                    Ap += 9;
                }
            } else {
                for (; p < kc; p++) {
                    for (int r = 0; r < 9; r++) Ap[r] = rows[r][p] * alpha;
                    Ap += 9;
                }
            }
        } else {
            for (int p = 0; p < kc; p++) {
                for (int r = 0; r < mr; r++)
                    Ap[r] = A[(size_t)(i + r) * (size_t)lda + (size_t)p] * alpha;
                for (int r = mr; r < 9; r++) Ap[r] = 0.0f;
                Ap += 9;
            }
        }
    }
}

static void pack_B_single_trans(int NR, int k_len, int nr, const float *B, int ldb, float *Bp) {
    /* Transposed B: B is N x K row-major -> element (p, c) is B[c * ldb + p] */
    for (int p = 0; p < k_len; p++) {
        float *dst = Bp + (size_t)p * (size_t)NR;
        for (int c = 0; c < nr; c++) dst[c] = B[(size_t)c * (size_t)ldb + (size_t)p];
        for (int c = nr; c < NR; c++) dst[c] = 0.0f;
    }
}

static void pack_B_single_nr16(int transB, int k_len, int nr, const float *B, int ldb, float *Bp) {
    if (UNLIKELY(transB)) { pack_B_single_trans(16, k_len, nr, B, ldb, Bp); return; }
    int64_t ldb_bytes = (int64_t)ldb * (int64_t)sizeof(float);
    if (LIKELY(nr == 16)) {
        if (LIKELY(g_cpu.has_avx)) kronos_pack_b_nr16_avx2(k_len, B, ldb_bytes, Bp);
        else {
            for (int p = 0; p < k_len; p++)
                memcpy(Bp + (size_t)p * 16u, B + (size_t)p * (size_t)ldb, 16 * sizeof(float));
        }
    } else {
        for (int p = 0; p < k_len; p++) {
            memcpy(Bp + (size_t)p * 16u, B + (size_t)p * (size_t)ldb, (size_t)nr * sizeof(float));
            memset(Bp + (size_t)p * 16u + nr, 0, (size_t)(16 - nr) * sizeof(float));
        }
    }
}

static void pack_B_single_nr32(int transB, int k_len, int nr, const float *B, int ldb, float *Bp) {
    if (UNLIKELY(transB)) { pack_B_single_trans(32, k_len, nr, B, ldb, Bp); return; }
    int64_t ldb_bytes = (int64_t)ldb * (int64_t)sizeof(float);
    if (LIKELY(nr == 32)) {
        kronos_pack_b_nr32_avx512(k_len, B, ldb_bytes, Bp);
    } else {
        for (int p = 0; p < k_len; p++) {
            memcpy(Bp + (size_t)p * 32u, B + (size_t)p * (size_t)ldb, (size_t)nr * sizeof(float));
            memset(Bp + (size_t)p * 32u + nr, 0, (size_t)(32 - nr) * sizeof(float));
        }
    }
}

static void pack_B_single_nr48(int transB, int k_len, int nr, const float *B, int ldb, float *Bp) {
    if (UNLIKELY(transB)) { pack_B_single_trans(48, k_len, nr, B, ldb, Bp); return; }
    int64_t ldb_bytes = (int64_t)ldb * (int64_t)sizeof(float);
    if (LIKELY(nr == 48)) {
        kronos_pack_b_nr48_avx512(k_len, B, ldb_bytes, Bp);
    } else {
        for (int p = 0; p < k_len; p++) {
            memcpy(Bp + (size_t)p * 48u, B + (size_t)p * (size_t)ldb, (size_t)nr * sizeof(float));
            memset(Bp + (size_t)p * 48u + nr, 0, (size_t)(48 - nr) * sizeof(float));
        }
    }
}

static void pack_B_nr16(int transB, int kc, int nc, const float *B, int ldb, float *Bp) {
    for (int j = 0; j < nc; j += 16) {
        int nr = (j + 16 <= nc) ? 16 : (nc - j);
        const float *bsrc = transB ? (B + (size_t)j * (size_t)ldb) : (B + j);
        pack_B_single_nr16(transB, kc, nr, bsrc, ldb, Bp + (size_t)j * (size_t)kc);
    }
}

static void pack_B_nr32(int transB, int kc, int nc, const float *B, int ldb, float *Bp) {
    for (int j = 0; j < nc; j += 32) {
        int nr = (j + 32 <= nc) ? 32 : (nc - j);
        const float *bsrc = transB ? (B + (size_t)j * (size_t)ldb) : (B + j);
        pack_B_single_nr32(transB, kc, nr, bsrc, ldb, Bp + (size_t)j * (size_t)kc);
    }
}

static void pack_B_nr48(int transB, int kc, int nc, const float *B, int ldb, float *Bp) {
    for (int j = 0; j < nc; j += 48) {
        int nr = (j + 48 <= nc) ? 48 : (nc - j);
        const float *bsrc = transB ? (B + (size_t)j * (size_t)ldb) : (B + j);
        pack_B_single_nr48(transB, kc, nr, bsrc, ldb, Bp + (size_t)j * (size_t)kc);
    }
}

typedef struct {
    int               mr;
    int               nr;
    int               isa_tier; /* 2=avx512, 1=avx2, 0=scalar */
    const char       *short_name;
    const char       *name;
    GemmUkernelFn     ukernel;
    GemmEdgeUkernelFn edge_ukernel; /* Non-NULL for zero-copy AVX-512 opmask edge path */
    void (*pack_a)(int transA, int mc, int kc, const float *A, int lda, float *Ap, float alpha);
    void (*pack_b)(int transB, int kc, int nc, const float *B, int ldb, float *Bp);
    void (*pack_b_one)(int transB, int k_len, int nr, const float *B, int ldb, float *Bp);
} GemmKernelDesc;

static const GemmKernelDesc g_gemm_avx512 = {
    14, 32, 2, "avx512",
    "kronos_gemm_14x32_avx512 (32 ZMM, staggered prefetchw, opmask zero-copy edge)",
    kronos_gemm_14x32_avx512, kronos_gemm_14x32_avx512_edge,
    pack_A_mr14, pack_B_nr32, pack_B_single_nr32
};

static const GemmKernelDesc g_gemm_avx512_9x48 = {
    9, 48, 2, "avx512-9x48",
    "kronos_gemm_9x48_avx512 (32 ZMM, 2.25 FMAs/load, opmask zero-copy edge)",
    kronos_gemm_9x48_avx512, kronos_gemm_9x48_avx512_edge,
    pack_A_mr9, pack_B_nr48, pack_B_single_nr48
};

static const GemmKernelDesc g_gemm_avx2 = {
    6, 16, 1, "avx2",
    "kronos_gemm_6x16_fma (16 YMM disp8, staggered prefetchw, aligned-RZ pack_A)",
    kronos_gemm_6x16_fma, NULL,
    pack_A_mr6, pack_B_nr16, pack_B_single_nr16
};

static const GemmKernelDesc g_gemm_scalar = {
    6, 16, 0, "scalar",
    "kronos_gemm_6x16_fallback (portable C, double-buf BLIS)",
    kronos_gemm_6x16_fallback, NULL,
    pack_A_mr6, pack_B_nr16, pack_B_single_nr16
};

static double gemm_calibrated_peak_fpc(const GemmKernelDesc *kd) {
    if (kd->isa_tier == 2) return (g_cpu.peak_fma_fpc_avx512 > 1.0) ? g_cpu.peak_fma_fpc_avx512 : 64.0;
    if (kd->isa_tier == 1) return (g_cpu.peak_fma_fpc_avx2   > 1.0) ? g_cpu.peak_fma_fpc_avx2   : 32.0;
    return (g_cpu.peak_fma_fpc_sse2 > 1.0) ? g_cpu.peak_fma_fpc_sse2 : 8.0;
}

typedef struct {
    int MC, KC, NC;
} GemmTuning;

/* Analytical BLIS L1d sizing (~45-58% using L1d associativity ways) + active-set L2 */
static GemmTuning kronos_autotune_gemm(const GemmKernelDesc *kd) {
    GemmTuning t;
    uint64_t eff_l2 = (g_cpu.l2_eff_per_core > 0) ? g_cpu.l2_eff_per_core : g_cpu.l2_bytes;

    /* Reserve 2 ways of L1d for streaming Ap + C rows (Low et al. BLIS analytical model) */
    uint32_t ways = (g_cpu.l1d_ways >= 4) ? g_cpu.l1d_ways : 8u;
    double l1_frac = (double)(ways - 2u) / (double)ways;
    if (l1_frac > 0.58) l1_frac = 0.58;
    if (l1_frac < 0.45) l1_frac = 0.45;

    int kc = (int)(((double)g_cpu.l1d_bytes * l1_frac) / (double)((size_t)kd->nr * sizeof(float)));
    kc = (kc / 16) * 16;
    if (kc < 128) kc = 128;
    if (kc > 384) kc = 384;

    int mc = (int)((eff_l2 * 55ULL / 100ULL) / ((size_t)kc * sizeof(float)));
    mc = (mc / kd->mr) * kd->mr;
    if (mc < kd->mr * 10) mc = kd->mr * 10;
    if (mc > kd->mr * 48) mc = kd->mr * 48;

    int nc = (int)((g_cpu.l3_bytes * 35ULL / 100ULL) / ((size_t)kc * sizeof(float)));
    nc = (nc / kd->nr) * kd->nr;
    if (nc < 288)  nc = 288;
    if (nc > 2304) nc = 2304;
    nc = (nc / kd->nr) * kd->nr;

    t.MC = mc; t.KC = kc; t.NC = nc;
    return t;
}

typedef struct {
    uint8_t             *buf;
    size_t               cap;
    KronosDissemBarrier *global_bar;
    KronosDissemBarrier *group_bar;
} GemmWorkspace;

static void gemm_ws_ensure(GemmWorkspace *ws, size_t total_bytes) {
    if (total_bytes > ws->cap) {
        if (ws->buf) (void)munmap(ws->buf, ws->cap);
        ws->buf = (uint8_t *)xmmap_huge_aligned(total_bytes, &ws->cap);
    }
    if (!ws->global_bar) {
        ws->global_bar = (KronosDissemBarrier *)xaligned_alloc(64, sizeof(KronosDissemBarrier));
        ws->group_bar  = (KronosDissemBarrier *)xaligned_alloc(64, sizeof(KronosDissemBarrier));
    }
}

static void gemm_ws_destroy(GemmWorkspace *ws) {
    if (ws->buf)        (void)munmap(ws->buf, ws->cap);
    if (ws->global_bar) free(ws->global_bar);
    if (ws->group_bar)  free(ws->group_bar);
    memset(ws, 0, sizeof(*ws));
}

typedef struct {
    int                   transA, transB;
    int                   M, N, K;
    int                   lda, ldb, ldc;
    int                   P, p_m, p_n;
    int                   kc_step;
    int                   nc_step;
    int                   panels_per_mc;
    int                   total_mr_panels;
    float                 alpha, beta;
    GemmTuning            tun;
    const float          *A;
    const float          *B;
    float                *C;
    uint8_t              *bp_base;
    size_t                bp_stride_bytes;
    uint8_t              *ap_base;
    size_t                ap_group_bytes;
    const GemmKernelDesc *kd;
    KronosDissemBarrier  *global_bar;
    KronosDissemBarrier  *group_bar;
    KronosLatch          *done_latch;
} GemmSharedCtx;

typedef struct {
    int                  tid;
    const GemmSharedCtx *s;
} GemmWorkerArg;

static inline uint64_t gemm_pack_edge_ctrl(int64_t beta_mode, int mr, int nr) {
    uint64_t mask = (nr >= 48) ? 0xFFFFFFFFFFFFULL : ((1ULL << nr) - 1ULL);
    return (uint64_t)(beta_mode & 3) | ((uint64_t)(mr & 15) << 2) | (mask << 16);
}

static void gemm_persistent_worker_fiber(void *arg) {
    const GemmWorkerArg *wa  = (const GemmWorkerArg *)arg;
    const GemmSharedCtx *s   = wa->s;
    const GemmKernelDesc *kd = s->kd;
    const int tid       = wa->tid;
    const int P         = s->P;
    const int MR        = kd->mr;
    const int NR        = kd->nr;
    const int my_mi     = tid / s->p_n;
    const int my_nj     = tid % s->p_n;
    const int grp_base  = my_mi * s->p_n;

    uint32_t g_ep   = 0;
    uint32_t grp_ep = 0;

    if (UNLIKELY(s->alpha == 0.0f || s->K <= 0)) {
        int r0 = (int)(((int64_t)s->M * tid) / P);
        int r1 = (int)(((int64_t)s->M * (tid + 1)) / P);
        for (int r = r0; r < r1; r++) {
            float *crow = s->C + (size_t)r * (size_t)s->ldc;
            if (s->beta == 0.0f) {
                memset(crow, 0, (size_t)s->N * sizeof(float));
            } else if (s->beta != 1.0f) {
                for (int c = 0; c < s->N; c++) crow[c] *= s->beta;
            }
        }
        latch_done(s->done_latch);
        return;
    }

    int64_t ldc_bytes = (int64_t)s->ldc * (int64_t)sizeof(float);
    int wave_mr_step  = s->p_m * s->panels_per_mc;
    if (wave_mr_step < 1) wave_mr_step = 1;

    int step_b = 0;
    for (int jc = 0; jc < s->N; jc += s->nc_step) {
        int nc = (jc + s->nc_step <= s->N) ? s->nc_step : (s->N - jc);
        int total_nr_panels = (nc + NR - 1) / NR;

        for (int pc = 0; pc < s->K; pc += s->kc_step) {
            int kc = (pc + s->kc_step <= s->K) ? s->kc_step : (s->K - pc);
            int64_t beta_mode = (pc > 0) ? 1 : ((s->beta == 0.0f) ? 0 : ((s->beta == 1.0f) ? 1 : 2));

            float *Bp_cur = (float *)(s->bp_base + (size_t)(step_b & 1) * s->bp_stride_bytes);
            step_b++;

            if (total_nr_panels >= P) {
                int base_pb = total_nr_panels / P;
                int rem_pb  = total_nr_panels % P;
                int pb_cnt  = base_pb + ((tid < rem_pb) ? 1 : 0);
                int pb_p0   = tid * base_pb + ((tid < rem_pb) ? tid : rem_pb);
                if (pb_cnt > 0) {
                    int j_off = pb_p0 * NR;
                    int j_len = pb_cnt * NR;
                    if (j_off + j_len > nc) j_len = nc - j_off;
                    const float *bsrc = s->transB
                        ? (s->B + (size_t)(jc + j_off) * (size_t)s->ldb + (size_t)pc)
                        : (s->B + (size_t)pc * (size_t)s->ldb + (size_t)(jc + j_off));
                    kd->pack_b(s->transB, kc, j_len, bsrc, s->ldb,
                               Bp_cur + (size_t)j_off * (size_t)kc);
                }
            } else {
                int k0 = (kc * tid) / P;
                int k1 = (kc * (tid + 1)) / P;
                int kl = k1 - k0;
                if (kl > 0) {
                    for (int pn = 0; pn < total_nr_panels; pn++) {
                        int j_off = pn * NR;
                        int nr    = (j_off + NR <= nc) ? NR : (nc - j_off);
                        const float *bsrc = s->transB
                            ? (s->B + (size_t)(jc + j_off) * (size_t)s->ldb + (size_t)(pc + k0))
                            : (s->B + (size_t)(pc + k0) * (size_t)s->ldb + (size_t)(jc + j_off));
                        kd->pack_b_one(s->transB, kl, nr, bsrc, s->ldb,
                                       Bp_cur + (size_t)j_off * (size_t)kc + (size_t)k0 * (size_t)NR);
                    }
                }
            }
            dissem_wait_group(s->global_bar, 0, tid, P, &g_ep);

            int base_n    = total_nr_panels / s->p_n;
            int rem_n     = total_nr_panels % s->p_n;
            int n_cnt     = base_n + ((my_nj < rem_n) ? 1 : 0);
            int n_p0      = my_nj * base_n + ((my_nj < rem_n) ? my_nj : rem_n);
            int j_sub0    = n_p0 * NR;
            int j_sub_len = n_cnt * NR;
            if (j_sub0 + j_sub_len > nc) j_sub_len = (j_sub0 < nc) ? (nc - j_sub0) : 0;

            int wave_idx = 0;
            for (int wave_p0 = 0; wave_p0 < s->total_mr_panels; wave_p0 += wave_mr_step, wave_idx++) {
                int wave_panels = s->total_mr_panels - wave_p0;
                if (wave_panels > wave_mr_step) wave_panels = wave_mr_step;

                int base_m = wave_panels / s->p_m;
                int rem_m  = wave_panels % s->p_m;
                int g_cnt  = base_m + ((my_mi < rem_m) ? 1 : 0);
                int g_p0   = wave_p0 + my_mi * base_m + ((my_mi < rem_m) ? my_mi : rem_m);
                int ic     = g_p0 * MR;
                int mc     = g_cnt * MR;
                if (ic + mc > s->M) mc = (ic < s->M) ? (s->M - ic) : 0;

                size_t bank_off = (s->p_n > 1) ? ((size_t)(wave_idx & 1) * (size_t)s->p_m * s->ap_group_bytes) : 0u;
                float *Ap_grp   = (float *)(s->ap_base + bank_off + (size_t)my_mi * s->ap_group_bytes);

                if (s->p_n == 1) {
                    if (mc > 0) {
                        const float *asrc = s->transA
                            ? (s->A + (size_t)pc * (size_t)s->lda + (size_t)ic)
                            : (s->A + (size_t)ic * (size_t)s->lda + (size_t)pc);
                        kd->pack_a(s->transA, mc, kc, asrc, s->lda, Ap_grp, s->alpha);
                    }
                } else {
                    int pa_base = g_cnt / s->p_n;
                    int pa_rem  = g_cnt % s->p_n;
                    int pa_cnt  = pa_base + ((my_nj < pa_rem) ? 1 : 0);
                    int pa_off  = my_nj * pa_base + ((my_nj < pa_rem) ? my_nj : pa_rem);
                    if (pa_cnt > 0) {
                        int sub_ic = (g_p0 + pa_off) * MR;
                        int sub_mc = pa_cnt * MR;
                        if (sub_ic + sub_mc > s->M) sub_mc = s->M - sub_ic;
                        const float *asrc = s->transA
                            ? (s->A + (size_t)pc * (size_t)s->lda + (size_t)sub_ic)
                            : (s->A + (size_t)sub_ic * (size_t)s->lda + (size_t)pc);
                        kd->pack_a(s->transA, sub_mc, kc, asrc, s->lda,
                                   Ap_grp + (size_t)(pa_off * MR) * (size_t)kc, s->alpha);
                    }
                    dissem_wait_group(s->group_bar, grp_base, my_nj, s->p_n, &grp_ep);
                }

                if (mc > 0 && j_sub_len > 0) {
                    const float *Bp_sub = Bp_cur + (size_t)j_sub0 * (size_t)kc;
                    float *C_tile       = s->C + (size_t)ic * (size_t)s->ldc + (size_t)(jc + j_sub0);
                    int n_mp            = (mc + MR - 1) / MR;
                    int j_panel         = 0;

                    for (int j = 0; j < j_sub_len; j += NR, j_panel++) {
                        int nr = (j + NR <= j_sub_len) ? NR : (j_sub_len - j);
                        const float *Bp_j = Bp_sub + (size_t)j * (size_t)kc;

                        for (int ii = 0; ii < n_mp; ii++) {
                            int pi = (j_panel & 1) ? (n_mp - 1 - ii) : ii;
                            int i  = pi * MR;
                            int mr = (i + MR <= mc) ? MR : (mc - i);
                            const float *Ap_i = Ap_grp + (size_t)i * (size_t)kc;
                            float *C_ij = C_tile + (size_t)i * (size_t)s->ldc + (size_t)j;

                            if (LIKELY(mr == MR && nr == NR)) {
                                kd->ukernel(kc, Ap_i, Bp_j, C_ij, ldc_bytes, beta_mode, s->beta);
                            } else if (kd->edge_ukernel != NULL) {
                                /* Zero-copy AVX-512 opmask edge path (no stack c_edge bounce!) */
                                uint64_t ctrl = gemm_pack_edge_ctrl(beta_mode, mr, nr);
                                kd->edge_ukernel(kc, Ap_i, Bp_j, C_ij, ldc_bytes, ctrl, s->beta);
                            } else {
                                _Alignas(64) float c_edge[GEMM_MAX_MR * GEMM_MAX_NR];
                                kd->ukernel(kc, Ap_i, Bp_j, c_edge,
                                            (int64_t)NR * (int64_t)sizeof(float), 0, 0.0f);
                                if (beta_mode == 0) {
                                    for (int r = 0; r < mr; r++) {
                                        for (int c = 0; c < nr; c++)
                                            C_ij[(size_t)r * (size_t)s->ldc + (size_t)c] = c_edge[r * NR + c];
                                    }
                                } else if (beta_mode == 1) {
                                    for (int r = 0; r < mr; r++) {
                                        for (int c = 0; c < nr; c++)
                                            C_ij[(size_t)r * (size_t)s->ldc + (size_t)c] += c_edge[r * NR + c];
                                    }
                                } else {
                                    for (int r = 0; r < mr; r++) {
                                        for (int c = 0; c < nr; c++) {
                                            float *dst = &C_ij[(size_t)r * (size_t)s->ldc + (size_t)c];
                                            *dst = s->beta * (*dst) + c_edge[r * NR + c];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    latch_done(s->done_latch);
}

static void kronos_sgemm_tn(int transA, int transB, int M, int N, int K, float alpha,
                            const float *A, int lda,
                            const float *B, int ldb,
                            float beta, float *C, int ldc,
                            GemmTuning tun, const GemmKernelDesc *kd, GemmWorkspace *ws) {
    if (M <= 0 || N <= 0) return;
    int req_lda = transA ? M : K;
    int req_ldb = transB ? K : N;
    if (UNLIKELY((K > 0 && (lda < req_lda || ldb < req_ldb)) || ldc < N)) {
        fprintf(stderr, "[KRONOS] Invalid GEMM strides: lda=%d ldb=%d ldc=%d (M=%d N=%d K=%d)\n",
                lda, ldb, ldc, M, N, K);
        return;
    }

    const int P  = g_kern.num_cores;
    const int MR = kd->mr;
    const int NR = kd->nr;
    const int total_mr_panels = (M + MR - 1) / MR;

    /* Balanced K-split AND balanced N-split (eliminates both skinny K-tails and N-tails!) */
    int kc_step = tun.KC;
    if (K > 0) {
        int k_blocks = (K + tun.KC - 1) / tun.KC;
        kc_step = (k_blocks <= 1) ? K : (((K + k_blocks - 1) / k_blocks + 3) & ~3);
    } else {
        kc_step = 1;
    }

    int nc_step = tun.NC;
    {
        int n_blocks = (N + tun.NC - 1) / tun.NC;
        if (n_blocks <= 1) {
            nc_step = N;
        } else {
            int raw_nc = (N + n_blocks - 1) / n_blocks;
            nc_step = ((raw_nc + NR - 1) / NR) * NR;
        }
    }
    const int max_nr_panels = (nc_step + NR - 1) / NR;

    int p_m = P;
    int p_n = 1;
    if (P > 1 && max_nr_panels >= 2) {
        double best_cost = 1e30;
        for (int d = 1; d <= P; d++) {
            if (P % d != 0) continue;
            int cand_n = d;
            int cand_m = P / d;
            if (cand_n > max_nr_panels && cand_n > 1) continue;
            int max_mp = (total_mr_panels + cand_m - 1) / cand_m;
            int max_np = (max_nr_panels   + cand_n - 1) / cand_n;
            double cost = (double)(max_mp * max_np);
            if (cand_n > 1) cost *= (1.0 + 0.025 * (double)(cand_n - 1));
            if (max_mp * MR > tun.MC) cost *= 1.03;
            if (cost < best_cost) {
                best_cost = cost;
                p_m       = cand_m;
                p_n       = cand_n;
            }
        }
    }

    int panels_per_mc = tun.MC / MR;
    if (panels_per_mc < 1) panels_per_mc = 1;
    int max_group_panels = panels_per_mc;
    if (total_mr_panels < p_m * panels_per_mc) {
        int ceil_div = (total_mr_panels + p_m - 1) / p_m;
        if (ceil_div < max_group_panels) max_group_panels = ceil_div;
    }
    if (max_group_panels < 1) max_group_panels = 1;

    size_t bp_stride      = ((size_t)kc_step * (size_t)max_nr_panels * (size_t)NR * sizeof(float) + 63u) & ~63u;
    if ((bp_stride & 4095u) == 0u) bp_stride += 64u;
    size_t ap_group_bytes = ((size_t)max_group_panels * (size_t)MR * (size_t)kc_step * sizeof(float) + 63u) & ~63u;
    if ((ap_group_bytes & 4095u) == 0u) ap_group_bytes += 64u;

    size_t bp_total = bp_stride * 2u;
    size_t ap_banks = (p_n > 1) ? 2u : 1u;
    size_t ap_total = ap_group_bytes * (size_t)p_m * ap_banks;
    size_t ws_total = ((bp_total + 4095u) & ~4095u) + ap_total;

    gemm_ws_ensure(ws, ws_total);

    dissem_init(ws->global_bar, P);
    dissem_init(ws->group_bar,  P);
    KronosLatch done;
    latch_init(&done, (uint32_t)P);

    GemmSharedCtx s = {
        .transA          = transA,
        .transB          = transB,
        .M               = M,
        .N               = N,
        .K               = K,
        .lda             = lda,
        .ldb             = ldb,
        .ldc             = ldc,
        .P               = P,
        .p_m             = p_m,
        .p_n             = p_n,
        .kc_step         = kc_step,
        .nc_step         = nc_step,
        .panels_per_mc   = panels_per_mc,
        .total_mr_panels = total_mr_panels,
        .alpha           = alpha,
        .beta            = beta,
        .tun             = tun,
        .A               = A,
        .B               = B,
        .C               = C,
        .bp_base         = ws->buf,
        .bp_stride_bytes = bp_stride,
        .ap_base         = ws->buf + ((bp_total + 4095u) & ~4095u),
        .ap_group_bytes  = ap_group_bytes,
        .kd              = kd,
        .global_bar      = ws->global_bar,
        .group_bar       = ws->group_bar,
        .done_latch      = &done
    };

    GemmWorkerArg wargs[KRONOS_MAX_CORES];
    for (int i = 0; i < P; i++) {
        wargs[i].tid = i;
        wargs[i].s   = &s;
    }
    /* Strictly core-pinned spawn guarantees tid == w->id & 100% NUMA/L3-CCD locality! */
    kronos_spawn_batch_pinned(gemm_persistent_worker_fiber, wargs, sizeof(GemmWorkerArg), P, 0);
    latch_wait(&done);
}

static void kronos_sgemm_ex(int M, int N, int K, float alpha,
                            const float *A, const float *B, float beta, float *C,
                            GemmTuning tun, const GemmKernelDesc *kd, GemmWorkspace *ws) {
    kronos_sgemm_tn(0, 0, M, N, K, alpha, A, K, B, N, beta, C, N, tun, kd, ws);
}

/* ============================================================================
 *  9. 5-STAGE EXHAUSTIVE KERNEL VERIFICATION, ADVERSARIAL STRESS & TELEMETRY
 * ============================================================================ */

static inline float f_abs(float x) { return x < 0.0f ? -x : x; }

static int verify_gemm_tn(int transA, int transB, int M, int N, int K,
                          int pad_a, int pad_b, int pad_c,
                          float alpha, float beta,
                          const GemmKernelDesc *kd, GemmWorkspace *ws) {
    int a_rows = transA ? K : M, a_cols = transA ? M : K;
    int b_rows = transB ? N : K, b_cols = transB ? K : N;
    int lda = a_cols + pad_a, ldb = b_cols + pad_b, ldc = N + pad_c;
    GemmTuning tun = kronos_autotune_gemm(kd);
    size_t a_elems = (K > 0) ? (size_t)a_rows * (size_t)lda : 1u;
    size_t b_elems = (K > 0) ? (size_t)b_rows * (size_t)ldb : 1u;
    float *A  = (float *)xaligned_alloc(64, a_elems * sizeof(float));
    float *B  = (float *)xaligned_alloc(64, b_elems * sizeof(float));
    float *C  = (float *)xaligned_alloc(64, (size_t)M * (size_t)ldc * sizeof(float));
    float *C0 = (float *)xaligned_alloc(64, (size_t)M * (size_t)ldc * sizeof(float));

    uint64_t rng = 0x123456789ABCDEF0ULL ^ ((uint64_t)M << 32) ^ ((uint64_t)N << 16) ^
                   (uint64_t)K ^ ((uint64_t)transA << 1) ^ (uint64_t)transB;
    for (size_t i = 0; i < a_elems; i++)
        A[i] = ((float)(int)(xorshift64(&rng) & 255u) - 128.0f) * 0.01f;
    for (size_t i = 0; i < b_elems; i++)
        B[i] = ((float)(int)(xorshift64(&rng) & 255u) - 128.0f) * 0.01f;

    float qnan = make_qnan_f32();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < ldc; j++) {
            float v = ((float)(int)(xorshift64(&rng) & 255u) - 128.0f) * 0.01f;
            C0[(size_t)i * (size_t)ldc + (size_t)j] = v;
            C[(size_t)i * (size_t)ldc + (size_t)j]  = (j < N && beta == 0.0f) ? qnan : v;
        }
    }

    kronos_sgemm_tn(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, tun, kd, ws);

    int ok = 1;
    float max_rel = 0.0f;
    for (int i = 0; i < M && ok; i++) {
        for (int j = N; j < ldc; j++) {
            if (C[(size_t)i * (size_t)ldc + (size_t)j] != C0[(size_t)i * (size_t)ldc + (size_t)j]) {
                ok = 0; break;
            }
        }
        for (int j = 0; j < N && ok; j++) {
            float got = C[(size_t)i * (size_t)ldc + (size_t)j];
            if (!is_finite_f32(got)) { ok = 0; break; }
            double ref = (beta == 0.0f) ? 0.0 : (double)beta * (double)C0[(size_t)i * (size_t)ldc + (size_t)j];
            double dot = 0.0;
            for (int k = 0; k < K; k++) {
                float av = transA ? A[(size_t)k * (size_t)lda + (size_t)i]
                                  : A[(size_t)i * (size_t)lda + (size_t)k];
                float bv = transB ? B[(size_t)j * (size_t)ldb + (size_t)k]
                                  : B[(size_t)k * (size_t)ldb + (size_t)j];
                dot += (double)av * (double)bv;
            }
            ref += (double)alpha * dot;
            float diff  = f_abs(got - (float)ref);
            float scale = f_abs((float)ref) + 1.0f;
            float rel   = diff / scale;
            if (!(rel <= max_rel)) max_rel = rel;
        }
    }
    free(A); free(B); free(C); free(C0);
    return ok && is_finite_f32(max_rel) && (max_rel < 1e-4f);
}

#ifndef KRONOS_NO_MAIN
static void print_banner(void) {
    KPRINTF("\n%s%s", C_CYAN, C_BOLD);
    KPRINTF("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    KPRINTF("║  KRONOS-X v5.0 :: x86_64 PREEMPTIVE CPU MICROKERNEL & SILICON ENGINE         ║\n");
    KPRINTF("║  Pinned NUMA M:N | Per-Core Timers | Fuel-Safe JIT | 9x48 & 14x32 BLIS GEMM  ║\n");
    KPRINTF("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    KPRINTF("%s", C_RST);
}

static void print_stage(int idx, const char *title) {
    KPRINTF("\n%s%s[STAGE %d]%s %s%s%s\n", C_MAG, C_BOLD, idx, C_RST, C_BOLD, title, C_RST);
    KPRINTF("%s──────────────────────────────────────────────────────────────────────────────%s\n", C_DIM, C_RST);
}

/* --- Stage 1: Sentinel-Mag Slab + Exact Multi-Color Verification --- */
typedef struct {
    int          fiber_idx;
    int          iters;
    KronosLatch *latch;
    uint64_t     cycles;
    uint64_t     corruptions;
} SlabBenchArg;

static void slab_bench_fiber(void *arg) {
    SlabBenchArg *a = (SlabBenchArg *)arg;
    void    *ptrs[32];
    uint64_t tags[32];
    size_t   sizes[32];
    uint64_t errs = 0;

    uint64_t c0 = rdtsc_begin();
    for (int i = 0; i < a->iters; i++) {
        int slot = i & 31;
        if (i >= 32) {
            if (((volatile uint64_t *)ptrs[slot])[0] != tags[slot]) errs++;
            kronos_free(ptrs[slot], sizes[slot]);
        }
        size_t sz = k_slab_sizes[i & 15];
        uint64_t tag = ((uint64_t)(a->fiber_idx + 1) << 40) ^ ((uint64_t)i * 0x9E3779B97F4A7C15ULL);
        void *p = kronos_alloc(sz);
        size_t nat_align = sz & (~sz + 1u);
        if (((uintptr_t)p & (nat_align - 1u)) != 0) errs++;
        ((volatile uint64_t *)p)[0] = tag;
        ptrs[slot]  = p;
        tags[slot]  = tag;
        sizes[slot] = sz;
    }
    for (int s = 0; s < 32; s++) {
        int idx = a->iters - 32 + s;
        if (idx >= 0) {
            int slot = idx & 31;
            if (((volatile uint64_t *)ptrs[slot])[0] != tags[slot]) errs++;
            kronos_free(ptrs[slot], sizes[slot]);
        }
    }
    uint64_t c1 = rdtsc_end();
    a->cycles      = (c1 > c0) ? (c1 - c0) : 1ULL;
    a->corruptions = errs;
    latch_done(a->latch);
}

typedef struct {
    KronosChan  *ch;
    int          count;
    int          core_id;
    uint64_t     errs;
    KronosLatch *latch;
} SlabXferArg;

static void slab_xfer_producer(void *arg) {
    SlabXferArg *a = (SlabXferArg *)arg;
    a->core_id = kronos_current_core_id();
    for (int i = 0; i < a->count; i++) {
        size_t sz = (i & 1) ? 128u : 1024u;
        uint64_t *blk = (uint64_t *)kronos_alloc(sz);
        blk[0] = 0xA5A5000000000000ULL ^ ((uint64_t)i * 0x9E3779B97F4A7C15ULL);
        blk[1] = (uint64_t)i;
        blk[2] = (uint64_t)sz;
        (void)chan_send(a->ch, (uint64_t)(uintptr_t)blk);
    }
    latch_done(a->latch);
}

static void slab_xfer_consumer(void *arg) {
    SlabXferArg *a = (SlabXferArg *)arg;
    a->core_id = kronos_current_core_id();
    uint64_t errs = 0;
    for (int i = 0; i < a->count; i++) {
        uint64_t *blk = (uint64_t *)(uintptr_t)chan_recv(a->ch);
        uint64_t idx  = blk[1];
        size_t   sz   = (size_t)blk[2];
        uint64_t exp  = 0xA5A5000000000000ULL ^ (idx * 0x9E3779B97F4A7C15ULL);
        if ((((uintptr_t)blk & (sz - 1u)) != 0) || blk[0] != exp) errs++;
        kronos_free(blk, sz);
    }
    a->errs = errs;
    latch_done(a->latch);
}

static void run_stage1_slab(void) {
    print_stage(1, "16-CLASS BONWICK SLAB (SENTINEL FAST PATH + EXACT MODULAR COLORING)");

    int lut_ok = (slab_class_for(0) == 0 && slab_class_for(32) == 0 && slab_class_for(33) == 1 &&
                  slab_class_for(48) == 1 && slab_class_for(49) == 2 && slab_class_for(64) == 2 &&
                  slab_class_for(512) == 11 && slab_class_for(768) == 12 &&
                  slab_class_for(4096) == 15 && slab_class_for(4097) == -1);
    void *p4k  = kronos_alloc(4096);
    void *pbig = kronos_alloc(12288);
    int align_ok = (((uintptr_t)p4k & 4095u) == 0) && (((uintptr_t)pbig & 63u) == 0);
    kronos_free(p4k, 4096);
    kronos_free(pbig, 12288);

    /* Direct mixed-class cache-color spread verification (verifies all 4 colors for 1024B!) */
    uint8_t seen_colors_1k[4] = {0};
    void *c1k_ptrs[320];
    void *mix_ptrs[320];
    for (int i = 0; i < 320; i++) {
        c1k_ptrs[i] = kronos_alloc(1024);
        mix_ptrs[i] = kronos_alloc(64); /* Interleave another class to defeat naive coloring! */
        size_t col = ((uintptr_t)c1k_ptrs[i] & 4095u) >> 10;
        seen_colors_1k[col] = 1u;
    }
    for (int i = 0; i < 320; i++) {
        kronos_free(c1k_ptrs[i], 1024);
        kronos_free(mix_ptrs[i], 64);
    }
    uint32_t unique_1k_colors = (uint32_t)(seen_colors_1k[0] + seen_colors_1k[1] +
                                           seen_colors_1k[2] + seen_colors_1k[3]);

    int nf    = g_kern.num_cores;
    int iters = 200000;
    SlabBenchArg *args = (SlabBenchArg *)calloc((size_t)nf, sizeof(SlabBenchArg));
    if (!args) { perror("calloc SlabBenchArg"); exit(1); }
    KronosLatch latch;
    latch_init(&latch, (uint32_t)nf);

    uint64_t allocs_before = 0, refills_before = 0, swaps_before = 0;
    for (int c = 0; c < g_kern.num_cores; c++) {
        allocs_before  += stat_load(&g_slab.cores[c].fast_allocs);
        refills_before += stat_load(&g_slab.cores[c].depot_refills);
        swaps_before   += stat_load(&g_slab.cores[c].mag_swaps);
    }

    for (int i = 0; i < nf; i++) args[i] = (SlabBenchArg){ i, iters, &latch, 0, 0 };
    uint64_t t0 = monotonic_ns();
    kronos_spawn_batch_pinned(slab_bench_fiber, args, sizeof(SlabBenchArg), nf, 0);
    latch_wait(&latch);
    uint64_t t1 = monotonic_ns();
    uint64_t dt_ns = safe_delta_ns(t1, t0);

    uint64_t fast_allocs = 0, mag_swaps = 0, depot_refills = 0;
    for (int c = 0; c < g_kern.num_cores; c++) {
        fast_allocs   += stat_load(&g_slab.cores[c].fast_allocs);
        mag_swaps     += stat_load(&g_slab.cores[c].mag_swaps);
        depot_refills += stat_load(&g_slab.cores[c].depot_refills);
    }
    fast_allocs   -= allocs_before;
    mag_swaps     -= swaps_before;
    depot_refills -= refills_before;

    KronosChan xch;
    chan_init(&xch);
    KronosLatch xlatch;
    latch_init(&xlatch, 2);
    SlabXferArg xprod = { &xch, 16384, -1, 0, &xlatch };
    SlabXferArg xcons = { &xch, 16384, -1, 0, &xlatch };
    kronos_spawn(slab_xfer_producer, &xprod);
    kronos_spawn(slab_xfer_consumer, &xcons);
    latch_wait(&xlatch);

    uint64_t total_ops = (uint64_t)nf * (uint64_t)iters * 2ULL;
    uint64_t sum_cy = 0, total_errs = xcons.errs;
    for (int i = 0; i < nf; i++) {
        sum_cy     += args[i].cycles;
        total_errs += args[i].corruptions;
    }

    int pass = lut_ok && align_ok && (unique_1k_colors == 4u) && (total_errs == 0);
    if (!pass) atomic_fetch_add(&g_failures, 1);

    double cy_per_pair = (double)sum_cy / ((double)nf * (double)iters);
    double ns_per_pair = cy_per_pair * g_cpu.ns_per_cycle;
    double mops        = (double)total_ops / ((double)dt_ns / 1e3);
    double hit_pct     = (fast_allocs > 0)
        ? 100.0 * (double)(fast_allocs - depot_refills) / (double)fast_allocs : 100.0;

    g_tel.slab_cy_pair       = cy_per_pair;
    g_tel.slab_mops          = mops;
    g_tel.slab_hit_pct       = hit_pct;
    g_tel.slab_unique_colors = unique_1k_colors;

    KPRINTF("  Classes: 16 (32B..4KiB + >4KiB) | Fibers: %-3d | Ops: %" PRIu64 " | Arena: %zu MiB (2M THP)\n",
            nf, total_ops, atomic_load(&g_slab.total_mapped_bytes) / (1024u * 1024u));
    KPRINTF("  Latency: %s%.1f cycles%s (%.1f ns) per alloc+free | Aggregate: %s%.1f M ops/s%s (%.2f ms)\n",
            C_GRN, cy_per_pair, C_RST, ns_per_pair, C_CYAN, mops, C_RST, (double)dt_ns / 1e6);
    KPRINTF("  Local Hit: %.3f%% (%" PRIu64 " swaps, %" PRIu64 " refills) | 1K Colors: %u/4 | XCore(v%d->v%d): 16K  [%s%s%s]\n",
            hit_pct, mag_swaps, depot_refills, unique_1k_colors, xprod.core_id, xcons.core_id,
            pass ? C_GRN : C_RED, pass ? "PASS" : "FAIL", C_RST);
    free(args);
}

/* --- Stage 2: Direct Switch, Joined 2xClose, Per-Worker Timers, Long-Timer Preempt, Canary, FP/errno, CodeRange Stress --- */
typedef struct {
    int          switches;
    KronosLatch *latch;
    uint64_t     total_cycles;
} CtxBenchArg;

static void ctx_bench_fiber(void *arg) {
    CtxBenchArg *a = (CtxBenchArg *)arg;
    uint64_t c0 = rdtsc_begin();
    for (int i = 0; i < a->switches; i++) {
        kronos_yield();
    }
    uint64_t c1 = rdtsc_end();
    a->total_cycles = (c1 > c0) ? (c1 - c0) : 1ULL;
    latch_done(a->latch);
}

typedef struct {
    KronosChan  *ch;
    int          worker_idx;
    int          count;
    uint64_t     sent_ok;
    uint64_t     xor_sum;
    uint64_t     add_sum;
    KronosLatch *latch;
} MpmcArg;

static void mpmc_producer(void *arg) {
    MpmcArg *a = (MpmcArg *)arg;
    uint64_t base = (uint64_t)a->worker_idx * (uint64_t)a->count;
    uint64_t x = 0, s = 0, ok = 0;
    for (int i = 1; i <= a->count; i++) {
        uint64_t v = (base + (uint64_t)i) * 0x9E3779B97F4A7C15ULL;
        uint64_t m = v ^ (v >> 19);
        if (!chan_send(a->ch, m)) break;
        ok++; x ^= m; s += m;
    }
    a->sent_ok = ok;
    a->xor_sum = x;
    a->add_sum = s;
    latch_done(a->latch);
}

static void mpmc_consumer(void *arg) {
    MpmcArg *a = (MpmcArg *)arg;
    uint64_t x = 0, s = 0, cnt = 0, m = 0;
    while (chan_recv_ex(a->ch, &m)) {
        cnt++; x ^= m; s += m;
    }
    a->sent_ok = cnt;
    a->xor_sum = x;
    a->add_sum = s;
    latch_done(a->latch);
}

typedef struct {
    KronosChan  *ch;
    KronosLatch *latch;
} CloserArg;

static void chan_closer_fiber(void *arg) {
    CloserArg *ca = (CloserArg *)arg;
    kronos_sleep_ns(1500000ULL);
    chan_close(ca->ch);
    latch_done(ca->latch);
}

typedef struct {
    uint64_t     sleep_ns;
    uint64_t     actual_ns;
    uint64_t     wake_ts;
    KronosLatch *latch;
} SleepArg;

static void sleep_test_fiber(void *arg) {
    SleepArg *a = (SleepArg *)arg;
    uint64_t t0 = monotonic_ns();
    kronos_sleep_ns(a->sleep_ns);
    a->wake_ts   = monotonic_ns();
    a->actual_ns = safe_delta_ns(a->wake_ts, t0);
    latch_done(a->latch);
}

typedef uint64_t (*CanaryFn)(uint64_t seed, uint64_t iters);

typedef struct {
    uint64_t     seed;
    uint64_t     iters;
    uint64_t     result;
    uint32_t     preempts_observed;
    CanaryFn     fn;
    _Atomic int *active_rogues;
    KronosLatch *latch;
} RogueArg;

static void rogue_hog_fiber(void *arg) {
    RogueArg *a = (RogueArg *)arg;
    a->result            = a->fn(a->seed, a->iters);
    a->preempts_observed = kronos_current_fiber_preempt_hits();
    atomic_fetch_sub_explicit(a->active_rogues, 1, memory_order_release);
    latch_done(a->latch);
}

typedef struct {
    uint64_t     spawn_ns;
    uint64_t     start_ns;
    int          was_saturated;
    _Atomic int *active_rogues;
    KronosLatch *latch;
} ProbeArg;

static void latency_probe_fiber(void *arg) {
    ProbeArg *p = (ProbeArg *)arg;
    p->start_ns      = monotonic_ns();
    p->was_saturated = (atomic_load_explicit(p->active_rogues, memory_order_acquire) >= g_kern.num_cores);
    latch_done(p->latch);
}

typedef struct {
    uint32_t     target_mxcsr_rc;
    uint16_t     target_fcw_rc;
    int          target_errno;
    int          iters;
    int          dirty_exit;
    int          spin_preempt;
    int          mismatches;
    int          migrations;
    KronosLatch *latch;
} FpCtrlArg;

static void fp_ctrl_iso_fiber(void *arg) {
    FpCtrlArg *a = (FpCtrlArg *)arg;
    uint32_t orig_mx = kronos_mxcsr_get();
    uint16_t orig_fc = kronos_fcw_get();
    uint32_t want_mx = (orig_mx & ~0x6000u) | a->target_mxcsr_rc;
    uint16_t want_fc = (uint16_t)((orig_fc & ~0x0C00u) | a->target_fcw_rc);
    kronos_mxcsr_set(want_mx);
    kronos_fcw_set(want_fc);
    KRONOS_ERRNO = a->target_errno;

    int bad = 0, mig = 0;
    int prev_core = kronos_current_core_id();
    volatile uint64_t sink = 0;

    for (int i = 0; i < a->iters; i++) {
        if (a->spin_preempt && (i & 63) == 0) {
            /* Busy compute slice to trigger real async SIGUSR1 preemption while FP/errno dirty! */
            uint64_t x = (uint64_t)i ^ 0x9E3779B97F4A7C15ULL;
            for (int s = 0; s < 4000; s++) x = x * 6364136223846793005ULL + 1ULL;
            sink ^= x;
        }
        kronos_yield();
        int cur_core = kronos_current_core_id();
        if (cur_core != prev_core) { mig++; prev_core = cur_core; }
        if ((kronos_mxcsr_get() & 0xFFC0u) != (want_mx & 0xFFC0u)) bad++;
        if (kronos_fcw_get() != want_fc) bad++;
        if (KRONOS_ERRNO != a->target_errno) bad++;
    }
    (void)sink;
    if (!a->dirty_exit) {
        kronos_mxcsr_set(orig_mx);
        kronos_fcw_set(orig_fc);
    }
    a->mismatches = bad;
    a->migrations = mig;
    latch_done(a->latch);
}

static void fp_clean_check_fiber(void *arg) {
    FpCtrlArg *a = (FpCtrlArg *)arg;
    int bad = 0;
    if ((kronos_mxcsr_get() & 0xFFC0u) != 0x1F80u) bad++;
    if (kronos_fcw_get() != 0x037Fu) bad++;
    if (KRONOS_ERRNO != 0) bad++;
    a->mismatches = bad;
    latch_done(a->latch);
}

typedef struct {
    int          id;
    int          iters;
    int          errs;
    KronosLatch *latch;
} CodeRangeStressArg;

static void code_range_stress_fiber(void *arg) {
    CodeRangeStressArg *a = (CodeRangeStressArg *)arg;
    int errs = 0;
    for (int i = 0; i < a->iters; i++) {
        uintptr_t base = 0x7F0000000000ULL + ((uintptr_t)a->id << 24) + ((uintptr_t)i << 12);
        int slot = kronos_register_code_range(base, base + 4096u);
        if (slot >= 0) {
            if (!rip_in_registered_code(base + 128u)) errs++;
            kronos_unregister_code_range(slot);
        }
    }
    a->errs = errs;
    latch_done(a->latch);
}

static void run_stage2_scheduler_and_preemption(void) {
    print_stage(2, "DIRECT M:N KERNEL, JOINED 2xCLOSE MPMC, PER-CORE TIMERS & XSAVE64 CANARY");

    const int k_sw = 150000;
    CtxBenchArg cba = { .switches = k_sw, .latch = NULL, .total_cycles = 0 };
    KronosLatch l_ctx;
    latch_init(&l_ctx, 1);
    cba.latch = &l_ctx;
    kronos_spawn_ex(ctx_bench_fiber, &cba, 0);
    latch_wait(&l_ctx);

    double cy_runnext = (double)cba.total_cycles / (double)(k_sw * 2);
    double ns_runnext = cy_runnext * g_cpu.ns_per_cycle;

    int n_mux  = g_kern.num_cores * 2;
    int mux_sw = 25000;
    CtxBenchArg *margs = (CtxBenchArg *)calloc((size_t)n_mux, sizeof(CtxBenchArg));
    if (!margs) { perror("calloc CtxBenchArg"); exit(1); }
    KronosLatch l_mux;
    latch_init(&l_mux, (uint32_t)n_mux);
    for (int i = 0; i < n_mux; i++) margs[i] = (CtxBenchArg){ mux_sw, &l_mux, 0 };
    uint64_t mt0 = monotonic_ns();
    kronos_spawn_batch(ctx_bench_fiber, margs, sizeof(CtxBenchArg), n_mux, 0);
    latch_wait(&l_mux);
    uint64_t mt1 = monotonic_ns();
    double mux_ns_sw = (double)safe_delta_ns(mt1, mt0) * (double)g_kern.num_cores /
                       (double)((uint64_t)n_mux * (uint64_t)mux_sw);
    free(margs);

    g_tel.ctx_runnext_cy = cy_runnext;
    g_tel.ctx_mux_ns     = mux_ns_sw;

    KPRINTF("  [2A] 4-Way Context Switch (1-hop `switch_direct` + zero-RMW queue pops):\n");
    KPRINTF("       1-Fiber runnext: %s%.1f cy/sw%s (%s%.1f ns%s) | %d-Fiber 1-Hop Direct Mux: %.1f ns/sw/core\n",
            C_GRN, cy_runnext, C_RST, C_CYAN, ns_runnext, C_RST, n_mux, mux_ns_sw);

    /* 2B-1: Full-throughput MPMC + clean close */
    int n_pairs = g_kern.num_cores;
    if (n_pairs < 2) n_pairs = 2;
    if (n_pairs > 8) n_pairs = 8;
    const int per_worker = 25000;
    const int total_msgs = n_pairs * per_worker;

    KronosChan ch;
    chan_init(&ch);
    KronosLatch l_prod, l_cons;
    latch_init(&l_prod, (uint32_t)n_pairs);
    latch_init(&l_cons, (uint32_t)n_pairs);
    MpmcArg prods[8], cons[8];

    uint64_t pt0 = monotonic_ns();
    for (int i = 0; i < n_pairs; i++) {
        prods[i] = (MpmcArg){ &ch, i, per_worker, 0, 0, 0, &l_prod };
        cons[i]  = (MpmcArg){ &ch, i, 0,          0, 0, 0, &l_cons };
        kronos_spawn(mpmc_producer, &prods[i]);
        kronos_spawn(mpmc_consumer, &cons[i]);
    }
    latch_wait(&l_prod);
    chan_close(&ch);
    latch_wait(&l_cons);
    uint64_t pt1    = monotonic_ns();
    uint64_t pdt_ns = safe_delta_ns(pt1, pt0);

    uint64_t got_xor = 0, got_sum = 0, exp_xor = 0, exp_sum = 0;
    for (int i = 0; i < n_pairs; i++) {
        got_xor ^= cons[i].xor_sum;  got_sum += cons[i].add_sum;
        exp_xor ^= prods[i].xor_sum; exp_sum += prods[i].add_sum;
    }

    /* 2B-2: Adversarial CONCURRENT send + double-close WITH joined closer latch! */
    KronosChan rch;
    chan_init(&rch);
    KronosLatch l_rp, l_rc, l_closers;
    latch_init(&l_rp, 4);
    latch_init(&l_rc, 4);
    latch_init(&l_closers, 2);
    MpmcArg rprods[4], rcons[4];
    CloserArg cargs_close[2] = { { &rch, &l_closers }, { &rch, &l_closers } };
    for (int i = 0; i < 4; i++) {
        rprods[i] = (MpmcArg){ &rch, i, 200000, 0, 0, 0, &l_rp };
        rcons[i]  = (MpmcArg){ &rch, i, 0,      0, 0, 0, &l_rc };
        kronos_spawn(mpmc_producer, &rprods[i]);
        kronos_spawn(mpmc_consumer, &rcons[i]);
    }
    kronos_spawn(chan_closer_fiber, &cargs_close[0]);
    kronos_spawn(chan_closer_fiber, &cargs_close[1]);
    latch_wait(&l_rp);
    latch_wait(&l_rc);
    latch_wait(&l_closers); /* Strictly join both closers before rch leaves scope! */
    uint64_t rp_cnt = 0, rc_cnt = 0, rp_xor = 0, rc_xor = 0;
    for (int i = 0; i < 4; i++) {
        rp_cnt += rprods[i].sent_ok; rp_xor ^= rprods[i].xor_sum;
        rc_cnt += rcons[i].sent_ok;  rc_xor ^= rcons[i].xor_sum;
    }
    int race_close_ok = (rp_cnt == rc_cnt) && (rp_xor == rc_xor);

    /* 2B-3: Out-of-Order Per-Worker Timers + Long-Timer Preemption Regression Check! */
    KronosLatch l_slp;
    latch_init(&l_slp, 4);
    SleepArg sargs[4] = {
        { 12000000ULL, 0, 0, &l_slp },
        {  2000000ULL, 0, 0, &l_slp },
        {  6000000ULL, 0, 0, &l_slp },
        {  3000000ULL, 0, 0, &l_slp }
    };
    uint64_t st0 = monotonic_ns();
    kronos_spawn_ex(sleep_test_fiber, &sargs[0], 0);
    sleep_ns_eintr(500000ULL);
    kronos_spawn_ex(sleep_test_fiber, &sargs[1], 0);
    kronos_spawn_ex(sleep_test_fiber, &sargs[2], 0);
    kronos_spawn_ex(sleep_test_fiber, &sargs[3], 0);
    latch_wait(&l_slp);
    uint64_t st_wall = safe_delta_ns(monotonic_ns(), st0);
    int slp_ok = (st_wall < 250000000ULL) &&
                 (sargs[0].actual_ns >= 12000000ULL) && (sargs[1].actual_ns >= 2000000ULL) &&
                 (sargs[2].actual_ns >=  6000000ULL) && (sargs[3].actual_ns >= 3000000ULL) &&
                 (sargs[1].wake_ts < sargs[0].wake_ts);

    /* Regression test for Audit #2: Arm a 400ms timer while idle, THEN spawn a preemptible hog! */
    int long_timer_preempt_ok = 1;
    if (g_kern.preempt_supported) {
        KronosLatch l_long_slp, l_quick_hog;
        latch_init(&l_long_slp, 1);
        latch_init(&l_quick_hog, 1);
        SleepArg long_sa = { 40000000ULL, 0, 0, &l_long_slp }; /* 40ms timer > quantum */
        kronos_spawn_ex(sleep_test_fiber, &long_sa, 0);
        sleep_ns_eintr(2000000ULL); /* Let LAPIC enter timer sleep with active_preempt == 0 */
        _Atomic int dummy_r = 1;
        RogueArg quick_ra = { 0xCAFE1234ULL, 3000000ULL, 0, 0, kronos_preempt_canary_sse2, &dummy_r, &l_quick_hog };
        kronos_spawn_ex(rogue_hog_fiber, &quick_ra, 1);
        latch_wait(&l_quick_hog);
        latch_wait(&l_long_slp);
        uint64_t qref = kronos_preempt_canary_sse2(0xCAFE1234ULL, 3000000ULL);
        long_timer_preempt_ok = (quick_ra.result == qref) && (quick_ra.preempts_observed > 0);
    }

    int ipc_ok = (got_xor == exp_xor) && (got_sum == exp_sum) &&
                 race_close_ok && slp_ok && long_timer_preempt_ok;
    if (!ipc_ok) atomic_fetch_add(&g_failures, 1);
    g_tel.mpmc_mmsgs = (double)total_msgs / ((double)pdt_ns / 1e3);

    KPRINTF("  [2B] Deadlock-Free MPMC (%dP/%dC) + Joined 2xClose + Per-Core OOO Timers:\n",
            n_pairs, n_pairs);
    KPRINTF("       MPMC: %s%.2f M msg/s%s | 2xClose(%" PRIu64 "): %s | OOO(%.1f<%.1fms) | TimerWakePreempt: %s [%s%s%s]\n",
            C_CYAN, g_tel.mpmc_mmsgs, C_RST,
            rc_cnt, race_close_ok ? "EXACT" : "ERR",
            (double)sargs[1].actual_ns / 1e6, (double)sargs[0].actual_ns / 1e6,
            long_timer_preempt_ok ? "OK" : "FAIL",
            ipc_ok ? C_GRN : C_RED, ipc_ok ? "PASS" : "FAIL", C_RST);

    /* 2C: Adversarial Dual-Seed XSAVE64 Hardware Canary */
    CanaryFn canary_fn = kronos_preempt_canary_sse2;
    const char *canary_desc = "16x XMM + x87/MXCSR + 14 GPRs + CF/DF + 128B RedZone";
    if (g_cpu.has_avx512f && g_cpu_use_xsave_asm && ((g_cpu_xsave_mask_lo_asm & 0xE7u) == 0xE7u)) {
        canary_fn   = kronos_preempt_canary_avx512;
        canary_desc = "32x ZMM + k0..7 + x87/MXCSR + 14 GPRs + CF/DF + 128B RedZone";
    } else if (g_cpu.has_avx2 && g_cpu_use_xsave_asm && ((g_cpu_xsave_mask_lo_asm & 0x07u) == 0x07u)) {
        canary_fn   = kronos_preempt_canary_avx2;
        canary_desc = "16x YMM + x87/MXCSR + 14 GPRs + CF/DF + 128B RedZone";
    }

    const uint64_t seed_a = 0x1337C0DECAFEBABEULL;
    const uint64_t seed_b = 0xDEADBEEF8BADF00DULL;

    (void)canary_fn(seed_a, 100000ULL);
    uint64_t cal_t0 = monotonic_ns();
    (void)canary_fn(seed_a, 250000ULL);
    uint64_t cal_ns = safe_delta_ns(monotonic_ns(), cal_t0);
    if (cal_ns < 100000ULL) cal_ns = 100000ULL;

    double target_ns = 60000000.0;
    double q_target  = (double)g_kern.quantum_us * 8000.0;
    if (q_target > target_ns) target_ns = q_target;
    if (target_ns > 500000000.0) target_ns = 500000000.0;

    uint64_t rogue_iters = (uint64_t)((target_ns / (double)cal_ns) * 250000.0);
    if (rogue_iters < 1000000ULL)  rogue_iters = 1000000ULL;
    if (rogue_iters > 80000000ULL) rogue_iters = 80000000ULL;

    uint64_t ref_a = canary_fn(seed_a, rogue_iters);
    uint64_t ref_b = canary_fn(seed_b, rogue_iters);

    int n_rogues = (g_kern.num_cores <= 16) ? (g_kern.num_cores * 2) : (g_kern.num_cores + 16);
    RogueArg *rogues = (RogueArg *)calloc((size_t)n_rogues, sizeof(RogueArg));
    if (!rogues) { perror("calloc RogueArg"); exit(1); }
    KronosLatch l_rogue;
    latch_init(&l_rogue, (uint32_t)n_rogues);
    _Atomic int active_rogues = n_rogues;

    uint64_t pre_before = atomic_load(&g_kern.total_preempt_ipis);
    uint64_t rt0 = monotonic_ns();
    for (int i = 0; i < n_rogues; i++) {
        rogues[i] = (RogueArg){
            (i & 1) ? seed_b : seed_a, rogue_iters, 0, 0, canary_fn, &active_rogues, &l_rogue
        };
    }
    kronos_spawn_batch(rogue_hog_fiber, rogues, sizeof(RogueArg), n_rogues, 1);

    const int n_probes = 12;
    ProbeArg probes[12];
    KronosLatch l_probe;
    latch_init(&l_probe, (uint32_t)n_probes);
    uint64_t probe_gap_ns = g_kern.quantum_us * 1000ULL;
    if (probe_gap_ns < 2000000ULL) probe_gap_ns = 2000000ULL;
    if (probe_gap_ns > 8000000ULL) probe_gap_ns = 8000000ULL;
    for (int i = 0; i < n_probes; i++) {
        sleep_ns_eintr(probe_gap_ns);
        probes[i] = (ProbeArg){ monotonic_ns(), 0, 0, &active_rogues, &l_probe };
        kronos_spawn_ex(latency_probe_fiber, &probes[i], 0);
    }
    latch_wait(&l_probe);
    latch_wait(&l_rogue);
    uint64_t rt1 = monotonic_ns();
    uint64_t pre_after = atomic_load(&g_kern.total_preempt_ipis);

    int all_bit_exact = 1;
    uint32_t total_hits = 0;
    for (int i = 0; i < n_rogues; i++) {
        uint64_t exp = (i & 1) ? ref_b : ref_a;
        if (rogues[i].result != exp) all_bit_exact = 0;
        total_hits += rogues[i].preempts_observed;
    }
    int preempt_ok = all_bit_exact && (!g_kern.preempt_supported || total_hits > 0);
    if (!preempt_ok) atomic_fetch_add(&g_failures, 1);

    double avg_probe_us = 0.0, max_probe_us = 0.0;
    int sat_probes = 0;
    for (int i = 0; i < n_probes; i++) {
        double us = (double)safe_delta_ns(probes[i].start_ns, probes[i].spawn_ns) / 1e3;
        avg_probe_us += us;
        if (us > max_probe_us) max_probe_us = us;
        if (probes[i].was_saturated) sat_probes++;
    }
    avg_probe_us /= (double)n_probes;

    g_tel.canary_preempts = total_hits;
    g_tel.probe_avg_us    = avg_probe_us;
    g_tel.probe_max_us    = max_probe_us;

    const char *save_eng = g_cpu_use_xsave_asm ? "XSAVE64/XRSTOR64 (stack-safe)" : "FXSAVE64/FXRSTOR64";

    KPRINTF("  [2C] Adversarial Dual-Seed CF+DF Hardware Canary (%d hogs, %d cores, Q=%" PRIu64 " us):\n",
            n_rogues, g_kern.num_cores, g_kern.quantum_us);
    KPRINTF("       Engine: %s (%" PRIu64 " B) | Canary: %s\n",
            save_eng, g_cpu_xsave_size_asm, canary_desc);
    KPRINTF("       IPIs: %" PRIu64 " | Preemptions: %s%" PRIu32 "%s | Probes(%d/%d sat): avg %.1f us, max %.1f us (%.1f ms)\n",
            (pre_after - pre_before), C_YEL, total_hits, C_RST, sat_probes, n_probes,
            avg_probe_us, max_probe_us, (double)safe_delta_ns(rt1, rt0) / 1e6);
    KPRINTF("       Dual Digests: 0x%016" PRIX64 " / 0x%016" PRIX64 "  [%s%s%s]\n",
            ref_a, ref_b,
            !g_kern.preempt_supported ? C_YEL : (preempt_ok ? C_GRN : C_RED),
            !g_kern.preempt_supported ? "SKIPPED" : (preempt_ok ? "BIT-EXACT ZERO CORRUPTION" : "FAILURE"),
            C_RST);
    free(rogues);

    /* 2D: Cross-Core Migration MXCSR + x87 FCW + CSE-Proof KRONOS_ERRNO (Coop + Preempt) + CodeRange Race Audit */
    int n_fp = g_kern.num_cores * 4;
    FpCtrlArg *fargs = (FpCtrlArg *)calloc((size_t)n_fp, sizeof(FpCtrlArg));
    if (!fargs) { perror("calloc FpCtrlArg"); exit(1); }
    KronosLatch l_fp;
    latch_init(&l_fp, (uint32_t)n_fp);
    static const uint32_t mx_modes[4] = { 0x0000u, 0x2000u, 0x4000u, 0x6000u };
    static const uint16_t fc_modes[4] = { 0x0000u, 0x0400u, 0x0800u, 0x0C00u };
    for (int i = 0; i < n_fp; i++) {
        fargs[i] = (FpCtrlArg){ mx_modes[i & 3], fc_modes[i & 3], 100 + i, 1500, (i & 1), 1, 0, 0, &l_fp };
    }
    kronos_spawn_batch(fp_ctrl_iso_fiber, fargs, sizeof(FpCtrlArg), n_fp, 1);
    latch_wait(&l_fp);
    int fp_bad = 0, fp_migs = 0;
    for (int i = 0; i < n_fp; i++) { fp_bad += fargs[i].mismatches; fp_migs += fargs[i].migrations; }
    free(fargs);

    FpCtrlArg *cargs = (FpCtrlArg *)calloc((size_t)g_kern.num_cores, sizeof(FpCtrlArg));
    if (!cargs) { perror("calloc clean check"); exit(1); }
    KronosLatch l_cln;
    latch_init(&l_cln, (uint32_t)g_kern.num_cores);
    for (int i = 0; i < g_kern.num_cores; i++) cargs[i].latch = &l_cln;
    kronos_spawn_batch_pinned(fp_clean_check_fiber, cargs, sizeof(FpCtrlArg), g_kern.num_cores, 0);
    latch_wait(&l_cln);
    for (int i = 0; i < g_kern.num_cores; i++) fp_bad += cargs[i].mismatches;
    free(cargs);

    /* Concurrent register/unregister seqlock stress test (Audit #4 regression check) */
    int n_cr = (g_kern.num_cores > 8) ? 8 : g_kern.num_cores;
    CodeRangeStressArg cr_args[8];
    KronosLatch l_cr;
    latch_init(&l_cr, (uint32_t)n_cr);
    for (int i = 0; i < n_cr; i++) cr_args[i] = (CodeRangeStressArg){ i, 2000, 0, &l_cr };
    kronos_spawn_batch(code_range_stress_fiber, cr_args, sizeof(CodeRangeStressArg), n_cr, 1);
    latch_wait(&l_cr);
    int cr_errs = 0;
    for (int i = 0; i < n_cr; i++) cr_errs += cr_args[i].errs;
    /* Verify zero odd (stuck) seqlock slots remain! */
    for (int s = 0; s < KRONOS_MAX_CODE_RANGES; s++) {
        if (atomic_load_explicit(&g_kern.code_ranges[s].seq, memory_order_acquire) & 1u) cr_errs++;
    }

    int iso_ok = (fp_bad == 0) && (cr_errs == 0);
    if (!iso_ok) atomic_fetch_add(&g_failures, 1);
    KPRINTF("  [2D] Coop+Preempt MXCSR/FCW/CSE-Proof errno (%d migs) + Seqlock CodeRange Stress:\n", fp_migs);
    KPRINTF("       FP/errno Leaks: %d | CodeRange Seqlock Errs (%dx2K ops): %d  [%s%s%s]\n",
            fp_bad, n_cr, cr_errs, iso_ok ? C_GRN : C_RED, iso_ok ? "ISOLATED" : "FAILED", C_RST);
}

/* --- Stage 3: Fixed-Point DCE, Fuel-Metered Loops, BMI1/2/LZCNT & Preemptible W^X JIT --- */
static void print_hex_machine_code(const uint8_t *code, size_t len) {
    if (g_out_mode != OUT_PRETTY) return;
    printf("       x86_64 machine code (%zu bytes, W^X RX + 0xCC trap sled):\n       %s", len, C_DIM);
    for (size_t i = 0; i < len; i++) {
        printf("%02X ", code[i]);
        if ((i + 1) % 24 == 0 && i + 1 < len) printf("\n       ");
    }
    printf("%s\n", C_RST);
}

typedef struct {
    KBpfJitFn    fn;
    uint64_t     seed;
    uint64_t     iters;
    uint64_t     result;
    uint32_t     preempt_hits;
    KronosLatch *latch;
} JitFiberArg;

static void jit_parallel_fiber(void *arg) {
    JitFiberArg *a = (JitFiberArg *)arg;
    a->result       = kbpf_call_jit(a->fn, a->seed, a->iters, 0);
    a->preempt_hits = kronos_current_fiber_preempt_hits();
    latch_done(a->latch);
}

static void run_stage3_jit(void) {
    print_stage(3, "K-ISA VERIFIER, FIXED-POINT DCE, FUEL SANDBOX, BMI1/2 & PREEMPTIBLE JIT");

    char err[128];

    const KBpfInsn bad_cap[]    = { { KOP_LOAD64, 0, 1, 2, 0, 0 }, { KOP_RET, 0, 0, 0, 0, 0 } };
    const KBpfInsn bad_flags[]  = {
        { KOP_CMP_REG, 1, 2, 0, 0, 0 }, { KOP_ADD_IMM, 0, 0, 0, 1, 0 },
        { KOP_JE, 0, 0, 0, 3, 0 },      { KOP_RET, 0, 0, 0, 0, 0 }
    };
    const KBpfInsn bad_cmov[]   = { { KOP_CMOVE, 0, 1, 0, 0, 0 }, { KOP_RET, 0, 0, 0, 0, 0 } };
    const KBpfInsn bad_loop[]   = { { KOP_JMP, 0, 0, 0, 0, 0 },   { KOP_RET, 0, 0, 0, 0, 0 } };
    const KBpfInsn bad_fall[]   = { { KOP_MOV_IMM, 0, 0, 0, 42, 0 } };
    const KBpfInsn bad_shift[]  = { { KOP_SHL_IMM, 0, 0, 0, 64, 0 }, { KOP_RET, 0, 0, 0, 0, 0 } };
    const KBpfInsn bad_bound[]  = {
        { KOP_MOV_IMM64, 3, 0, 0, 0, UINT64_MAX },
        { KOP_LOAD64,    0, 1, 2, 0, 0 },
        { KOP_RET,       0, 0, 0, 0, 0 }
    };

    struct { const char *label; const KBpfInsn *p; int n; uint32_t caps; } probes[7] = {
        { "untrusted LOAD64    ", bad_cap,   2, KBPF_CAP_NONE        },
        { "clobbered Jcc flags ", bad_flags, 4, KBPF_CAP_NONE        },
        { "undefined CMOV flags", bad_cmov,  2, KBPF_CAP_NONE        },
        { "self-loop DoS       ", bad_loop,  2, KBPF_CAP_NONE        },
        { "missing RET falloff ", bad_fall,  1, KBPF_CAP_NONE        },
        { "invalid shift >= 64 ", bad_shift, 2, KBPF_CAP_NONE        },
        { "BOUNDED R3 tamper   ", bad_bound, 3, KBPF_CAP_MEM_BOUNDED }
    };
    int blocked = 0;
    KPRINTF("  [3A] Sandboxed CFG + RFLAGS + Byte-Bound + Fuel Verifier (7 probes + fuel DoS):\n");
    for (int i = 0; i < 7; i++) {
        err[0] = '\0';
        int rc = kbpf_verify(probes[i].p, probes[i].n, probes[i].caps, err, sizeof(err), NULL);
        if (rc != 0) blocked++;
        KPRINTF("       Probe #%d (%s): %s%s%s (%s)\n", i + 1, probes[i].label,
                rc != 0 ? C_YEL : C_RED, rc != 0 ? "REJECTED" : "BYPASSED!", C_RST,
                rc != 0 ? err : "UNEXPECTED ACCEPT");
    }

    /* Probe #8: Hardware fuel-metered infinite-loop DoS containment (must abort deterministically!) */
    const KBpfInsn inf_prog[] = {
        { KOP_CMP_IMM, 1, 0, 0, 0, 0 },
        { KOP_JE,      0, 0, 0, 4, 0 }, /* Reachable RET when R1==0 */
        { KOP_ADD_IMM, 0, 0, 0, 1, 0 },
        { KOP_JMP,     0, 0, 0, 2, 0 }, /* Infinite loop when R1!=0! */
        { KOP_RET,     0, 0, 0, 0, 0 }
    };
    KBpfCompiled fuel_jit;
    int fuel_ok = 0;
    if (kbpf_jit_compile(inf_prog, 5, KBPF_CAP_BOUNDED_LOOP, &fuel_jit, err, sizeof(err)) == 0) {
        uint64_t r_safe_j = kbpf_call_jit(fuel_jit.fn, 0, 0, 0);
        uint64_t r_inf_j  = kbpf_call_jit(fuel_jit.fn, 1, 0, 0);
        uint64_t r_inf_i  = kbpf_exec_interp_unchecked(inf_prog, 5, KBPF_CAP_BOUNDED_LOOP, 1, 0, 0);
        fuel_ok = (r_safe_j == 0ULL) && (r_inf_j == KBPF_FUEL_EXHAUSTED) && (r_inf_i == KBPF_FUEL_EXHAUSTED);
        kbpf_jit_free(&fuel_jit);
    }
    KPRINTF("       Probe #8 (CAP_BOUNDED_LOOP DoS): %s%s%s (aborted @ %u iters -> 0x%" PRIX64 ")\n",
            fuel_ok ? C_YEL : C_RED, fuel_ok ? "CONTAINED" : "HUNG/FAIL", C_RST,
            KBPF_LOOP_FUEL_INIT, (uint64_t)KBPF_FUEL_EXHAUSTED);
    if (blocked != 7 || !fuel_ok) atomic_fetch_add(&g_failures, 1);

    /* 3B: Kernel #1 — 64-bit ARX/SplitMix */
    const KBpfInsn arx_prog[] = {
        { KOP_MOV_REG,   0, 1, 0, 0,  0 },
        { KOP_MOV_IMM,   4, 0, 0, 0,  0 },
        { KOP_MOV_IMM64, 5, 0, 0, 0,  0x9E3779B97F4A7C15ULL },
        { KOP_MOV_IMM64, 6, 0, 0, 0,  0xBF58476D1CE4E5B9ULL },
        { KOP_CMP_REG,   4, 2, 0, 0,  0 },
        { KOP_JAE,       0, 0, 0, 14, 0 },
        { KOP_ADD_REG,   0, 5, 0, 0,  0 },
        { KOP_MOV_REG,   7, 0, 0, 0,  0 },
        { KOP_SHR_IMM,   7, 0, 0, 30, 0 },
        { KOP_XOR_REG,   0, 7, 0, 0,  0 },
        { KOP_MUL_REG,   0, 6, 0, 0,  0 },
        { KOP_ROR_IMM,   0, 0, 0, 27, 0 },
        { KOP_ADD_IMM,   4, 0, 0, 1,  0 },
        { KOP_JMP,       0, 0, 0, 4,  0 },
        { KOP_RET,       0, 0, 0, 0,  0 },
    };
    const int n_arx = (int)(sizeof(arx_prog) / sizeof(arx_prog[0]));
    KBpfCompiled arx_jit;
    if (kbpf_jit_compile(arx_prog, n_arx, KBPF_CAP_NONE, &arx_jit, err, sizeof(err)) != 0) {
        KPRINTF("  [ERROR] ARX JIT compile failed: %s\n", err);
        atomic_fetch_add(&g_failures, 1);
        return;
    }

    const uint64_t seed  = 0xDEADBEEFCAFEBABEULL;
    const uint64_t iters = 2000000ULL;
    (void)kbpf_call_jit(arx_jit.fn, seed, 10000ULL, 0);

    uint64_t ti0 = monotonic_ns();
    uint64_t res_interp = kbpf_exec_interp_unchecked(arx_prog, n_arx, KBPF_CAP_NONE, seed, iters, 0);
    uint64_t ti1 = monotonic_ns();

    uint64_t tj0 = monotonic_ns();
    uint64_t res_jit = kbpf_call_jit(arx_jit.fn, seed, iters, 0);
    uint64_t tj1 = monotonic_ns();

    int njf = g_kern.num_cores * 2;
    uint64_t p_iters = iters * 4ULL;
    uint64_t ref_p   = kbpf_call_jit(arx_jit.fn, seed, p_iters, 0);
    JitFiberArg *jf_args = (JitFiberArg *)calloc((size_t)njf, sizeof(JitFiberArg));
    if (!jf_args) { perror("calloc JitFiberArg"); exit(1); }
    KronosLatch l_jit;
    latch_init(&l_jit, (uint32_t)njf);
    for (int i = 0; i < njf; i++) jf_args[i] = (JitFiberArg){ arx_jit.fn, seed, p_iters, 0, 0, &l_jit };
    uint64_t tp0 = monotonic_ns();
    kronos_spawn_batch(jit_parallel_fiber, jf_args, sizeof(JitFiberArg), njf, 1);
    latch_wait(&l_jit);
    uint64_t tp1 = monotonic_ns();

    int arx_ok = (res_jit == res_interp);
    uint32_t jit_preempts = 0;
    for (int i = 0; i < njf; i++) {
        if (jf_args[i].result != ref_p) arx_ok = 0;
        jit_preempts += jf_args[i].preempt_hits;
    }
    if (!arx_ok) atomic_fetch_add(&g_failures, 1);
    free(jf_args);

    double interp_ms = (double)safe_delta_ns(ti1, ti0) / 1e6;
    double jit_ms    = (double)safe_delta_ns(tj1, tj0) / 1e6;
    double par_ms    = (double)safe_delta_ns(tp1, tp0) / 1e6;
    g_tel.jit_speedup  = interp_ms / jit_ms;
    g_tel.jit_par_mops = ((double)p_iters * (double)njf) / (par_ms * 1e3);
    g_tel.jit_preempts = jit_preempts;

    KPRINTF("  [3B] JIT Kernel #1 — 64-bit ARX (%d rel8, %d opt, %d/5 pruned, %d JCC-pad, %zu B, BMI2=%s):\n",
            arx_jit.short_branches, arx_jit.peephole_opts, arx_jit.pruned_prologue_zeros,
            arx_jit.jcc_erratum_pads, arx_jit.code_len, g_cpu.has_bmi2 ? "RORX" : "ROR");
    kbpf_disasm_print(arx_prog, n_arx);
    print_hex_machine_code((const uint8_t *)arx_jit.code, arx_jit.code_len);
    KPRINTF("       Interp: %.2f ms | Native JIT: %s%.2f ms%s (%s%.1fx%s) | %d Preemptible Fibers: %s%.1f M/s%s (%" PRIu32 " JIT preempts) [%s%s%s]\n",
            interp_ms, C_GRN, jit_ms, C_RST, C_CYAN, g_tel.jit_speedup, C_RST,
            njf, C_CYAN, g_tel.jit_par_mops, C_RST,
            jit_preempts, arx_ok ? C_GRN : C_RED, arx_ok ? "BIT-EXACT" : "MISMATCH", C_RST);
    kbpf_jit_free(&arx_jit);

    /* 3C: SIB 64-bit Dot+Store + Byte-Bounded 32/64 + R3=0/7/11 Unaligned OOB + PROT_NONE Recycler */
    const KBpfInsn rw_prog[] = {
        { KOP_MOV_IMM, 0, 0, 0, 0,  0 }, { KOP_MOV_IMM, 4, 0, 0, 0,  0 },
        { KOP_CMP_REG, 4, 3, 0, 0,  0 }, { KOP_JAE,     0, 0, 0, 12, 0 },
        { KOP_LOAD64,  5, 1, 4, 0,  0 }, { KOP_LOAD64,  6, 2, 4, 0,  0 },
        { KOP_MUL_REG, 5, 6, 0, 0,  0 }, { KOP_ADD_REG, 0, 5, 0, 0,  0 },
        { KOP_XOR_REG, 5, 0, 0, 0,  0 }, { KOP_STORE64, 5, 2, 4, 0,  0 },
        { KOP_ADD_IMM, 4, 0, 0, 1,  0 }, { KOP_JMP,     0, 0, 0, 2,  0 },
        { KOP_RET,     0, 0, 0, 0,  0 },
    };
    const int n_rw = (int)(sizeof(rw_prog) / sizeof(rw_prog[0]));
    KBpfCompiled rw_jit;
    if (kbpf_jit_compile(rw_prog, n_rw, KBPF_CAP_MEM_RD | KBPF_CAP_MEM_WR, &rw_jit, err, sizeof(err)) != 0) {
        KPRINTF("  [ERROR] RW JIT compile failed: %s\n", err);
        atomic_fetch_add(&g_failures, 1);
        return;
    }

    const size_t n_vec = 500000;
    uint64_t *va  = (uint64_t *)xaligned_alloc(64, n_vec * sizeof(uint64_t));
    uint64_t *vb1 = (uint64_t *)xaligned_alloc(64, n_vec * sizeof(uint64_t));
    uint64_t *vb2 = (uint64_t *)xaligned_alloc(64, n_vec * sizeof(uint64_t));
    for (size_t i = 0; i < n_vec; i++) {
        va[i]  = i * 3ULL + 7ULL;
        vb1[i] = (n_vec - i) * 5ULL + 11ULL;
        vb2[i] = vb1[i];
    }

    uint64_t di0 = monotonic_ns();
    uint64_t res_i = kbpf_exec_interp_ex(rw_prog, n_rw, KBPF_CAP_MEM_RD | KBPF_CAP_MEM_WR,
                                         (uintptr_t)va, (uintptr_t)vb1, n_vec);
    uint64_t di1 = monotonic_ns();
    uint64_t dj0 = monotonic_ns();
    uint64_t res_j = kbpf_call_jit(rw_jit.fn, (uintptr_t)va, (uintptr_t)vb2, n_vec);
    uint64_t dj1 = monotonic_ns();

    const KBpfInsn bnd32_prog[] = {
        { KOP_LOAD32,  4, 1, 2, 0, 0 },
        { KOP_XOR_IMM, 4, 0, 0, 0x5A5A5A5A, 0 },
        { KOP_STORE32, 4, 1, 2, 0, 0 },
        { KOP_LOAD32,  0, 1, 2, 0, 0 },
        { KOP_RET,     0, 0, 0, 0, 0 }
    };
    const KBpfInsn bnd64_prog[] = {
        { KOP_LOAD64, 0, 1, 2, 0, 0 },
        { KOP_RET,    0, 0, 0, 0, 0 }
    };
    KBpfCompiled bnd32_jit, bnd64_jit;
    int bnd_ok = (kbpf_jit_compile(bnd32_prog, 5, KBPF_CAP_MEM_BOUNDED | KBPF_CAP_MEM_WR,
                                   &bnd32_jit, err, sizeof(err)) == 0) &&
                 (kbpf_jit_compile(bnd64_prog, 2, KBPF_CAP_MEM_BOUNDED,
                                   &bnd64_jit, err, sizeof(err)) == 0);
    if (bnd_ok) {
        uint32_t arr32[10] = { 10, 20, 30, 40, 50, 60, 70, 80, 0xCAFEBABEu, 0xDEADBEEFu };
        const uint64_t byte_len = 8u * sizeof(uint32_t);
        uint64_t v_in     = kbpf_call_jit(bnd32_jit.fn, (uintptr_t)arr32, 3, byte_len);
        uint64_t v_oob    = kbpf_call_jit(bnd32_jit.fn, (uintptr_t)arr32, 8, byte_len);
        uint64_t v_zero   = kbpf_call_jit(bnd32_jit.fn, 0ULL, 0, 0ULL);          /* NULL base + R3=0 -> safe 0 */
        uint64_t v_un7_ok = kbpf_call_jit(bnd32_jit.fn, (uintptr_t)arr32, 0, 7); /* 7 bytes = 1 u32 -> idx 0 OK */
        uint64_t v_un7_no = kbpf_call_jit(bnd32_jit.fn, (uintptr_t)arr32, 1, 7); /* 7 bytes = 1 u32 -> idx 1 OOB */
        uint64_t v64_un7  = kbpf_call_jit(bnd64_jit.fn, (uintptr_t)arr32, 0, 7); /* 7 bytes = 0 u64 -> idx 0 OOB */
        uint64_t v64_un11 = kbpf_call_jit(bnd64_jit.fn, (uintptr_t)arr32, 0, 11);/* 11 bytes = 1 u64 -> idx 0 OK */
        uint64_t v64_oob  = kbpf_call_jit(bnd64_jit.fn, (uintptr_t)arr32, 1, 11);/* 11 bytes = 1 u64 -> idx 1 OOB */
        uint64_t v64_exp0 = ((uint64_t)arr32[1] << 32) | (uint64_t)arr32[0];
        bnd_ok = (v_in == (40u ^ 0x5A5A5A5Au)) && (arr32[3] == (40u ^ 0x5A5A5A5Au)) &&
                 (arr32[8] == 0xCAFEBABEu) && (arr32[9] == 0xDEADBEEFu) &&
                 (v_oob == 0ULL) && (v_zero == 0ULL) &&
                 (v_un7_ok == (10u ^ 0x5A5A5A5Au)) && (v_un7_no == 0ULL) &&
                 (v64_un7 == 0ULL) && (v64_un11 == v64_exp0) && (v64_oob == 0ULL);
        kbpf_jit_free(&bnd32_jit);
        kbpf_jit_free(&bnd64_jit);
    }

    int mem_exact = (res_i == res_j) && (memcmp(vb1, vb2, n_vec * sizeof(uint64_t)) == 0) && bnd_ok;
    if (!mem_exact) atomic_fetch_add(&g_failures, 1);
    double di_ms  = (double)safe_delta_ns(di1, di0) / 1e6;
    double dj_ms  = (double)safe_delta_ns(dj1, dj0) / 1e6;
    double bw_gbs = ((double)(n_vec * 3 * sizeof(uint64_t)) / (dj_ms / 1e3)) / 1e9;
    g_tel.jit_bw_gbs = bw_gbs;
    KPRINTF("  [3C] JIT Kernel #2 — SIB Dot+Store (%zu elems) + Byte-Bounded (0/7/11/32B) + lfence:\n", n_vec);
    KPRINTF("       Interp: %.2f ms | JIT: %s%.2f ms%s (%.1fx, %.1f GB/s) | OOB/Unalign/PROT_NONE: %s  [%s%s%s]\n",
            di_ms, C_GRN, dj_ms, C_RST, di_ms / dj_ms, bw_gbs,
            bnd_ok ? "SAFE" : "FAIL",
            mem_exact ? C_GRN : C_RED, mem_exact ? "BIT-EXACT" : "MISMATCH", C_RST);
    free(va); free(vb1); free(vb2);
    kbpf_jit_free(&rw_jit);

    /* 3D: Exhaustive Audit across ALL 57 Opcodes, Transitive Multi-Pass DCE, AND+CMP0 Fusion, BMI1/2/LZCNT, rel32 */
    const KBpfInsn opt_prog[] = {
        { KOP_MOV_IMM,   5, 0, 0, 111,        0 }, /* Transitive dead chain #1 -> fixed-point DCE! */
        { KOP_ADD_IMM,   5, 0, 0, 222,        0 }, /* Transitive dead chain #2 -> fixed-point DCE! */
        { KOP_MOV_REG,   7, 5, 0, 0,          0 }, /* Transitive dead chain #3 -> fixed-point DCE! */
        { KOP_MOV_REG,   0, 1, 0, 0,          0 },
        { KOP_ADD_IMM,   0, 0, 0, 19,         0 }, /* Fuses MOV+ADD -> LEA */
        { KOP_MOV_REG,   7, 0, 0, 0,          0 },
        { KOP_SUB_IMM,   7, 0, 0, 7,          0 }, /* Fuses MOV+SUB -> LEA */
        { KOP_MOV_REG,   5, 7, 0, 0,          0 },
        { KOP_ADD_REG,   5, 1, 0, 0,          0 }, /* Fuses MOV+ADD_REG -> 3-op LEA */
        { KOP_MOV_REG,   6, 5, 0, 0,          0 },
        { KOP_MUL_IMM,   6, 0, 0, 5,          0 }, /* Fuses MOV+MUL_5 -> LEA */
        { KOP_MOV_REG,   0, 6, 0, 0,          0 },
        { KOP_ROR_IMM,   0, 0, 0, 13,         0 }, /* Fuses MOV+ROR -> BMI2 RORX */
        { KOP_SHL_REG,   0, 2, 0, 0,          0 }, /* BMI2 SHLX or SHL cl */
        { KOP_SHR_REG,   0, 2, 0, 0,          0 }, /* BMI2 SHRX or SHR cl */
        { KOP_SAR_REG,   7, 2, 0, 0,          0 }, /* BMI2 SARX or SAR cl */
        { KOP_XOR_REG,   0, 7, 0, 0,          0 },
        { KOP_AND_IMM,   6, 0, 0, 0xFF,       0 }, /* Fuses AND_IMM + CMP_IMM 0 -> zero TEST! */
        { KOP_CMP_IMM,   6, 0, 0, 0,          0 },
        { KOP_JE,        0, 0, 0, 22,         0 },
        { KOP_ADD_IMM,   0, 0, 0, INT32_MIN,  0 },
        { KOP_SUB_IMM,   0, 0, 0, INT32_MIN,  0 },
        { KOP_SAR_IMM,   0, 0, 0, 3,          0 },
        { KOP_CMP_IMM,   0, 0, 0, 100000,     0 },
        { KOP_MOV_IMM,   4, 0, 0, 0,          0 }, /* Preserves live flags! */
        { KOP_JLE,       0, 0, 0, 33,         0 },
        { KOP_ADD_IMM,   4, 0, 0, 500,        0 },
        { KOP_MUL_IMM,   4, 0, 0, 3,          0 },
        { KOP_MUL_IMM,   0, 0, 0, 5,          0 },
        { KOP_MUL_IMM,   4, 0, 0, 6,          0 },
        { KOP_MUL_IMM,   4, 0, 0, 10,         0 },
        { KOP_MUL_IMM,   4, 0, 0, 12,         0 },
        { KOP_JMP,       0, 0, 0, 39,         0 },
        { KOP_MOV_IMM64, 4, 0, 0, 0,          0xFFFFFFFF80000000ULL },
        { KOP_MUL_IMM,   4, 0, 0, -1,         0 },
        { KOP_MUL_IMM,   4, 0, 0, 16,         0 },
        { KOP_MUL_IMM,   4, 0, 0, -3,         0 },
        { KOP_MUL_IMM,   4, 0, 0, 100003,     0 },
        { KOP_MUL_IMM,   4, 0, 0, INT32_MIN,  0 },
        { KOP_AND_IMM,   4, 0, 0, 0x1234567,  0 },
        { KOP_AND_IMM,   0, 0, 0, 0x7FFFFFFF, 0 },
        { KOP_AND_IMM,   4, 0, 0, INT32_MIN,  0 },
        { KOP_OR_IMM,    4, 0, 0, INT32_MIN,  0 },
        { KOP_XOR_IMM,   4, 0, 0, INT32_MIN,  0 },
        { KOP_NEG,       4, 0, 0, 0,          0 },
        { KOP_BSWAP64,   4, 0, 0, 0,          0 },
        { KOP_MOV_REG,   5, 4, 0, 0,          0 },
        { KOP_UDIV_REG,  5, 2, 0, 0,          0 },
        { KOP_MOV_REG,   6, 4, 0, 0,          0 },
        { KOP_UMOD_REG,  6, 2, 0, 0,          0 },
        { KOP_XOR_REG,   0, 5, 0, 0,          0 },
        { KOP_ADD_REG,   0, 6, 0, 0,          0 },
        { KOP_AND_REG,   4, 1, 0, 0,          0 },
        { KOP_CMP_IMM,   4, 0, 0, 0,          0 }, /* Fuses AND_REG + CMP_IMM 0! */
        { KOP_CMOVE,     0, 1, 0, 0,          0 },
        { KOP_CMP_REG,   1, 2, 0, 0,          0 },
        { KOP_CMOVNE,    0, 4, 0, 0,          0 },
        { KOP_CMOVB,     0, 5, 0, 0,          0 },
        { KOP_CMOVAE,    0, 6, 0, 0,          0 },
        { KOP_CMOVBE,    0, 1, 0, 0,          0 },
        { KOP_CMOVA,     0, 4, 0, 0,          0 },
        { KOP_CMOVL,     0, 5, 0, 0,          0 },
        { KOP_CMOVGE,    0, 6, 0, 0,          0 },
        { KOP_CMOVLE,    0, 1, 0, 0,          0 },
        { KOP_CMOVG,     0, 4, 0, 0,          0 },
        { KOP_RET,       0, 0, 0, 0,          0 },
    };
    const int n_opt = (int)(sizeof(opt_prog) / sizeof(opt_prog[0]));
    KBpfCompiled opt_jit;
    int opt_dce = 0;
    int opt_ok = (kbpf_jit_compile(opt_prog, n_opt, KBPF_CAP_NONE, &opt_jit, err, sizeof(err)) == 0);
    if (opt_ok) {
        opt_dce = opt_jit.dce_elided;
        if (opt_dce < 3) opt_ok = 0; /* Must eliminate ALL 3 transitive dead insns! */
        const uint64_t cases[8] = {
            0ULL, 1ULL, 2ULL, 16ULL, 0x7FFFFFFFFFFFFFFFULL,
            0x8000000000000000ULL, (uint64_t)-64LL, (uint64_t)-1LL
        };
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                uint64_t ri = kbpf_exec_interp_unchecked(opt_prog, n_opt, KBPF_CAP_NONE, cases[i], cases[j], 0);
                uint64_t rj = kbpf_call_jit(opt_jit.fn, cases[i], cases[j], 0);
                if (ri != rj) opt_ok = 0;
            }
        }
        kbpf_jit_free(&opt_jit);
    }

    int ext_ok = 1;
    if (g_cpu.has_popcnt && g_cpu.has_lzcnt && g_cpu.has_bmi1) {
        const KBpfInsn bit_prog[] = {
            { KOP_POPCNT,  4, 1, 0, 0, 0 },
            { KOP_LZCNT,   5, 1, 0, 0, 0 },
            { KOP_TZCNT,   6, 1, 0, 0, 0 },
            { KOP_ADD_REG, 4, 5, 0, 0, 0 },
            { KOP_SHL_IMM, 4, 0, 0, 8, 0 },
            { KOP_OR_REG,  4, 6, 0, 0, 0 },
            { KOP_MOV_REG, 0, 4, 0, 0, 0 },
            { KOP_RET,     0, 0, 0, 0, 0 }
        };
        KBpfCompiled bjit;
        if (kbpf_jit_compile(bit_prog, 8, KBPF_CAP_NONE, &bjit, err, sizeof(err)) == 0) {
            const uint64_t btests[4] = { 0ULL, 1ULL, 0x8000000000000000ULL, 0x000DCAFEBABE1230ULL };
            for (int t = 0; t < 4; t++) {
                uint64_t ri = kbpf_exec_interp_unchecked(bit_prog, 8, KBPF_CAP_NONE, btests[t], 0, 0);
                uint64_t rj = kbpf_call_jit(bjit.fn, btests[t], 0, 0);
                if (ri != rj) ext_ok = 0;
            }
            kbpf_jit_free(&bjit);
        } else {
            ext_ok = 0;
        }
    }

    KBpfInsn long_prog[48];
    int lp = 0;
    long_prog[lp++] = (KBpfInsn){ KOP_MOV_REG, 0, 1, 0, 0, 0 };
    long_prog[lp++] = (KBpfInsn){ KOP_MOV_IMM, 4, 0, 0, 0, 0 };
    int loop_pc = lp;
    long_prog[lp++] = (KBpfInsn){ KOP_CMP_REG, 4, 2, 0, 0, 0 };
    int exit_jmp_pc = lp++;
    for (int k = 0; k < 16; k++) {
        uint64_t c64 = 0x100000001B3ULL + (uint64_t)k * 0x9E3779B97F4A7C15ULL;
        long_prog[lp++] = (KBpfInsn){ KOP_MOV_IMM64, 5, 0, 0, 0, c64 };
        long_prog[lp++] = (KBpfInsn){ KOP_XOR_REG,   0, 5, 0, 0, 0 };
    }
    long_prog[lp++] = (KBpfInsn){ KOP_ADD_IMM, 4, 0, 0, 1, 0 };
    long_prog[lp++] = (KBpfInsn){ KOP_JMP,     0, 0, 0, loop_pc, 0 };
    int ret_pc = lp;
    long_prog[lp++] = (KBpfInsn){ KOP_RET,     0, 0, 0, 0, 0 };
    long_prog[exit_jmp_pc] = (KBpfInsn){ KOP_JAE, 0, 0, 0, ret_pc, 0 };

    KBpfCompiled long_jit;
    int rel32_ok = (kbpf_jit_compile(long_prog, lp, KBPF_CAP_BOUNDED_LOOP, &long_jit, err, sizeof(err)) == 0);
    int n_long_br = 0;
    if (rel32_ok) {
        n_long_br = long_jit.long_branches;
        uint64_t ri = kbpf_exec_interp_unchecked(long_prog, lp, KBPF_CAP_BOUNDED_LOOP, 0xCAFEBABE12345678ULL, 100ULL, 0);
        uint64_t rj = kbpf_call_jit(long_jit.fn, 0xCAFEBABE12345678ULL, 100ULL, 0);
        rel32_ok = (ri == rj) && (n_long_br >= 2);
        kbpf_jit_free(&long_jit);
    }

    int audit_ok = opt_ok && ext_ok && rel32_ok;
    if (!audit_ok) atomic_fetch_add(&g_failures, 1);
    KPRINTF("  [3D] Full 57-Opcode + Transitive DCE(%d elided) + AND+CMP0 + BMI1/2/LZCNT + rel32:\n", opt_dce);
    KPRINTF("       64 boundary pairs (INT32_MIN/INT64_MIN/div0) + %d rel32 vs interp  [%s%s%s]\n",
            n_long_br, audit_ok ? C_GRN : C_RED, audit_ok ? "ALL BIT-EXACT" : "DIVERGED", C_RST);
}

/* --- Stage 4: Silicon-Calibrated Double-Buf BLIS GEMM + TransA/B + VNNI + Hybrid PMU --- */
typedef struct {
    int          tid, P, M, N, K;
    float       *A, *B, *C;
    KronosLatch *latch;
} GemmInitArg;

static void gemm_numa_init_fiber(void *arg) {
    GemmInitArg *a = (GemmInitArg *)arg;
    uint64_t rng = 0xCAFEBABE12345678ULL ^ ((uint64_t)(a->tid + 1) * 0x9E3779B97F4A7C15ULL);

    size_t a_tot = (size_t)a->M * (size_t)a->K;
    size_t a0 = (a_tot * (size_t)a->tid) / (size_t)a->P;
    size_t a1 = (a_tot * (size_t)(a->tid + 1)) / (size_t)a->P;
    for (size_t i = a0; i < a1; i++)
        a->A[i] = ((float)(int)(xorshift64(&rng) & 1023u) - 512.0f) * 0.002f;

    size_t b_tot = (size_t)a->K * (size_t)a->N;
    size_t b0 = (b_tot * (size_t)a->tid) / (size_t)a->P;
    size_t b1 = (b_tot * (size_t)(a->tid + 1)) / (size_t)a->P;
    for (size_t i = b0; i < b1; i++)
        a->B[i] = ((float)(int)(xorshift64(&rng) & 1023u) - 512.0f) * 0.002f;

    size_t c_tot = (size_t)a->M * (size_t)a->N;
    size_t c0 = (c_tot * (size_t)a->tid) / (size_t)a->P;
    size_t c1 = (c_tot * (size_t)(a->tid + 1)) / (size_t)a->P;
    float qnan = make_qnan_f32();
    for (size_t i = c0; i < c1; i++) a->C[i] = qnan;

    latch_done(a->latch);
}

static void run_stage4_gemm(int dim, const char *req_kernel) {
    print_stage(4, "SILICON-CALIBRATED BLIS GEMM (PINNED NUMA + 9x48/14x32 + TRANS + PMU)");

    int M = dim, N = dim, K = dim;
    const GemmKernelDesc *kd = &g_gemm_scalar;
    if (strcmp(req_kernel, "avx512-9x48") == 0) {
        if (g_cpu.has_avx512f) kd = &g_gemm_avx512_9x48;
        else KPRINTF("  %s[WARN] --gemm-kernel avx512-9x48 unavailable; falling back.%s\n", C_YEL, C_RST);
    } else if (strcmp(req_kernel, "avx512") == 0) {
        if (g_cpu.has_avx512f) kd = &g_gemm_avx512;
        else KPRINTF("  %s[WARN] --gemm-kernel avx512 unavailable; falling back.%s\n", C_YEL, C_RST);
    } else if (strcmp(req_kernel, "avx2") == 0) {
        if (g_cpu.has_avx2 && g_cpu.has_fma) kd = &g_gemm_avx2;
        else KPRINTF("  %s[WARN] --gemm-kernel avx2 unavailable; falling back.%s\n", C_YEL, C_RST);
    } else if (strcmp(req_kernel, "scalar") == 0) {
        kd = &g_gemm_scalar;
    } else {
        if (g_cpu.has_avx512f)                    kd = &g_gemm_avx512;
        else if (g_cpu.has_avx2 && g_cpu.has_fma) kd = &g_gemm_avx2;
    }

    GemmWorkspace ws;
    memset(&ws, 0, sizeof(ws));

    /* Auto-select bake-off between 14x32 and 9x48 on AVX-512 silicon! */
    if (strcmp(req_kernel, "auto") == 0 && g_cpu.has_avx512f && dim >= 384) {
        int bdim = 384;
        float *ba = (float *)xaligned_alloc(64, (size_t)bdim * (size_t)bdim * sizeof(float));
        float *bb = (float *)xaligned_alloc(64, (size_t)bdim * (size_t)bdim * sizeof(float));
        float *bc = (float *)xaligned_alloc(64, (size_t)bdim * (size_t)bdim * sizeof(float));
        memset(ba, 0, (size_t)bdim * (size_t)bdim * sizeof(float));
        memset(bb, 0, (size_t)bdim * (size_t)bdim * sizeof(float));
        GemmTuning t14 = kronos_autotune_gemm(&g_gemm_avx512);
        GemmTuning t9  = kronos_autotune_gemm(&g_gemm_avx512_9x48);
        kronos_sgemm_ex(bdim, bdim, bdim, 1.0f, ba, bb, 0.0f, bc, t14, &g_gemm_avx512, &ws);
        uint64_t a0 = monotonic_ns();
        kronos_sgemm_ex(bdim, bdim, bdim, 1.0f, ba, bb, 0.0f, bc, t14, &g_gemm_avx512, &ws);
        uint64_t dt14 = safe_delta_ns(monotonic_ns(), a0);
        kronos_sgemm_ex(bdim, bdim, bdim, 1.0f, ba, bb, 0.0f, bc, t9, &g_gemm_avx512_9x48, &ws);
        uint64_t b0 = monotonic_ns();
        kronos_sgemm_ex(bdim, bdim, bdim, 1.0f, ba, bb, 0.0f, bc, t9, &g_gemm_avx512_9x48, &ws);
        uint64_t dt9 = safe_delta_ns(monotonic_ns(), b0);
        if (dt9 < dt14) kd = &g_gemm_avx512_9x48;
        free(ba); free(bb); free(bc);
    }

    GemmTuning tun = kronos_autotune_gemm(kd);
    double peak_fpc = gemm_calibrated_peak_fpc(kd);

    KPRINTF("  Dispatch: %s%s%s  | Matrix: %dx%dx%d (FP32)\n", C_CYAN, kd->name, C_RST, M, N, K);
    KPRINTF("  Auto-Tuned Blocking (Analytical L1d / Active-L2): MC=%d  KC=%d  NC=%d  MR=%d  NR=%d  Peak=%.1f FLOP/cy\n",
            tun.MC, tun.KC, tun.NC, kd->mr, kd->nr, peak_fpc);

    /* Exhaustive edge suite: odd, skinny-N, 1x1x1, K=0, beta=0/1/gen, alpha!=1, strided, NN/NT/TN/TT! */
    int odd_ok = verify_gemm_tn(0, 0, 137, 149, 403, 0,  0,  0,  1.0f,   0.0f,  kd, &ws) &&
                 verify_gemm_tn(0, 0, 7,   13,  5,   0,  0,  0,  1.0f,   0.0f,  kd, &ws) &&
                 verify_gemm_tn(0, 0, 19,  257, 65,  0,  0,  0,  1.0f,   1.0f,  kd, &ws) &&
                 verify_gemm_tn(0, 0, 45,  17,  211, 0,  0,  0,  1.0f,   0.0f,  kd, &ws) &&
                 verify_gemm_tn(0, 0, 1,   1,   1,   0,  0,  0,  1.0f,   0.0f,  kd, &ws) &&
                 verify_gemm_tn(0, 0, 53,  67,  41,  7,  11, 13, 1.5f,   0.5f,  kd, &ws) &&
                 verify_gemm_tn(0, 0, 97,  113, 89,  3,  5,  7, -1.25f, -0.5f,  kd, &ws) &&
                 verify_gemm_tn(0, 0, 29,  33,  17,  3,  5,  7,  0.0f,   0.25f, kd, &ws) &&
                 verify_gemm_tn(0, 0, 23,  19,  0,   0,  0,  5,  2.0f,   0.0f,  kd, &ws) &&
                 verify_gemm_tn(0, 0, 23,  19,  0,   0,  0,  5,  2.0f,   0.5f,  kd, &ws) &&
                 verify_gemm_tn(1, 0, 59,  71,  47,  3,  5,  7,  1.25f,  0.5f,  kd, &ws) && /* TN */
                 verify_gemm_tn(0, 1, 59,  71,  47,  3,  5,  7, -0.75f,  1.0f,  kd, &ws) && /* NT */
                 verify_gemm_tn(1, 1, 59,  71,  47,  5,  3,  7,  1.5f,   0.0f,  kd, &ws);   /* TT */
    if (g_cpu.has_avx512f) {
        const GemmKernelDesc *other512 = (kd == &g_gemm_avx512) ? &g_gemm_avx512_9x48 : &g_gemm_avx512;
        odd_ok = odd_ok &&
                 verify_gemm_tn(0, 0, 73, 101, 67, 5, 7, 3, 1.5f, 0.5f, other512, &ws) &&
                 verify_gemm_tn(1, 1, 41,  53, 37, 3, 5, 7, -1.0f, 1.0f, other512, &ws);
    }
    if (kd->isa_tier == 2 && g_cpu.has_avx2 && g_cpu.has_fma) {
        odd_ok = odd_ok &&
                 verify_gemm_tn(0, 0, 137, 149, 403, 0, 0, 0,  1.0f,  0.0f, &g_gemm_avx2, &ws) &&
                 verify_gemm_tn(0, 0,  61,  79,  64, 7, 9, 5, -1.5f,  0.5f, &g_gemm_avx2, &ws);
    }

    int vnni_ok = 1;
    if (g_cpu.has_avx512vnni) {
        uint8_t *ua = (uint8_t *)xaligned_alloc(64, 4096);
        int8_t  *sb = (int8_t  *)xaligned_alloc(64, 4096);
        int64_t ref_v = 0;
        for (int i = 0; i < 4096; i++) {
            ua[i] = (uint8_t)((i * 37 + 13) & 0xFF);
            sb[i] = (int8_t)(((i * 53 + 7) & 0xFF) - 128);
            ref_v += (int64_t)ua[i] * (int64_t)sb[i];
        }
        int64_t got_v = kronos_vnni_dot_u8s8(4096 / 64, ua, sb);
        vnni_ok = (got_v == ref_v);
        free(ua); free(sb);
    }

    size_t a_map = 0, b_map = 0, c_map = 0;
    float *A = (float *)xmmap_huge_aligned((size_t)M * (size_t)K * sizeof(float), &a_map);
    float *B = (float *)xmmap_huge_aligned((size_t)K * (size_t)N * sizeof(float), &b_map);
    float *C = (float *)xmmap_huge_aligned((size_t)M * (size_t)N * sizeof(float), &c_map);

    int P = g_kern.num_cores;
    pid_t   worker_tids[KRONOS_MAX_CORES];
    uint8_t worker_ecore[KRONOS_MAX_CORES];
    for (int i = 0; i < P; i++) {
        worker_tids[i]  = g_kern.workers[i].linux_tid;
        worker_ecore[i] = g_kern.workers[i].is_ecore;
    }

    /* Pinned NUMA-local first-touch initialization! */
    KronosLatch l_init;
    latch_init(&l_init, (uint32_t)P);
    GemmInitArg iargs[KRONOS_MAX_CORES];
    for (int i = 0; i < P; i++) iargs[i] = (GemmInitArg){ i, P, M, N, K, A, B, C, &l_init };
    kronos_spawn_batch_pinned(gemm_numa_init_fiber, iargs, sizeof(GemmInitArg), P, 0);
    latch_wait(&l_init);

    kronos_sgemm_ex(M, N, K, 1.0f, A, B, 0.0f, C, tun, kd, &ws);

    double times_s[3];
    uint64_t best_pmu_cyc = 0, best_pmu_ins = 0;
    int best_pmu_mux = 0;
    double best_s = 1e9;
    for (int r = 0; r < 3; r++) {
        KronosPmu pmu;
        pmu_begin_tids(&pmu, worker_tids, worker_ecore, P);
        uint64_t t0 = monotonic_ns();
        kronos_sgemm_ex(M, N, K, 1.0f, A, B, 0.0f, C, tun, kd, &ws);
        uint64_t t1 = monotonic_ns();
        pmu_end(&pmu);
        times_s[r] = (double)safe_delta_ns(t1, t0) / 1e9;
        if (times_s[r] < best_s) {
            best_s       = times_s[r];
            best_pmu_cyc = pmu.cycles;
            best_pmu_ins = pmu.insns;
            best_pmu_mux = pmu.multiplexed;
        }
    }
    if (times_s[0] > times_s[1]) { double t = times_s[0]; times_s[0] = times_s[1]; times_s[1] = t; }
    if (times_s[1] > times_s[2]) { double t = times_s[1]; times_s[1] = times_s[2]; times_s[2] = t; }
    if (times_s[0] > times_s[1]) { double t = times_s[0]; times_s[0] = times_s[1]; times_s[1] = t; }
    double med_s = times_s[1];

    double flops       = 2.0 * (double)M * (double)N * (double)K;
    double gflops_best = (flops / best_s) / 1e9;
    double gflops_med  = (flops / med_s)  / 1e9;
    double gflops_core = gflops_best / (double)P;

    int main_finite = 1;
    float max_rel_dot = 0.0f;
    int test_r[8] = { 0, 1, 5, 13, 14, 31, M / 2, M - 1 };
    int test_c[8] = { 0, 1, 5, 15, 16, 31, N / 2, N - 1 };
    for (int ri = 0; ri < 8 && main_finite; ri++) {
        int i = test_r[ri] % M;
        for (int cj = 0; cj < 8; cj++) {
            int j = test_c[cj] % N;
            float got = C[(size_t)i * (size_t)N + (size_t)j];
            if (!is_finite_f32(got)) { main_finite = 0; break; }
            double ref = 0.0;
            for (int k = 0; k < K; k++)
                ref += (double)A[(size_t)i * (size_t)K + (size_t)k] *
                       (double)B[(size_t)k * (size_t)N + (size_t)j];
            float err   = f_abs(got - (float)ref);
            float scale = f_abs((float)ref) + 1.0f;
            float rel   = err / scale;
            if (!is_finite_f32(rel)) { main_finite = 0; break; }
            if (!(rel <= max_rel_dot)) max_rel_dot = rel;
        }
    }

    uint64_t rng = 0x5555AAAA12345678ULL;
    double *v  = (double *)malloc((size_t)N * sizeof(double));
    double *bv = (double *)malloc((size_t)K * sizeof(double));
    if (!v || !bv) { perror("malloc Freivalds"); exit(1); }
    float freiv_max_rel = 0.0f;
    for (int rd = 0; rd < 4 && main_finite; rd++) {
        for (int j = 0; j < N; j++) v[j] = (xorshift64(&rng) & 1u) ? 1.0 : -1.0;
        for (int k = 0; k < K; k++) {
            double s = 0.0;
            for (int j = 0; j < N; j++) s += (double)B[(size_t)k * (size_t)N + (size_t)j] * v[j];
            bv[k] = s;
        }
        for (int i = 0; i < M; i++) {
            double s1 = 0.0, s2 = 0.0;
            for (int j = 0; j < N; j++) s1 += (double)C[(size_t)i * (size_t)N + (size_t)j] * v[j];
            for (int k = 0; k < K; k++) s2 += (double)A[(size_t)i * (size_t)K + (size_t)k] * bv[k];
            if (!is_finite_f64(s1) || !is_finite_f64(s2)) { main_finite = 0; break; }
            float e     = f_abs((float)(s1 - s2));
            float scale = f_abs((float)s2) + 1.0f;
            float rel   = e / scale;
            if (!is_finite_f32(rel)) { main_finite = 0; break; }
            if (!(rel <= freiv_max_rel)) freiv_max_rel = rel;
        }
    }
    int verified = odd_ok && vnni_ok && main_finite && (max_rel_dot < 1e-4f) && (freiv_max_rel < 1e-4f);
    if (!verified) atomic_fetch_add(&g_failures, 1);

    g_tel.gemm_kernel      = kd->short_name;
    g_tel.gemm_dim         = dim;
    g_tel.gemm_peak_gflops = gflops_best;
    g_tel.gemm_med_gflops  = gflops_med;

    KPRINTF("  Peak Compute: %s%s%.2f GFLOP/s%s (med %.2f GF/s, %.2f ms, %.2f GF/core)\n",
            C_GRN, C_BOLD, gflops_best, C_RST, gflops_med, best_s * 1e3, gflops_core);
    if (best_pmu_cyc > 0) {
        double real_fpc = flops / (double)best_pmu_cyc;
        double eff_pct  = 100.0 * real_fpc / peak_fpc;
        double ipc      = (best_pmu_ins > 0) ? ((double)best_pmu_ins / (double)best_pmu_cyc) : 0.0;
        double eff_ghz  = ((double)best_pmu_cyc / (double)P) / (best_s * 1e9);
        g_tel.gemm_fpc       = real_fpc;
        g_tel.gemm_eff_pct   = eff_pct;
        g_tel.gemm_ipc       = ipc;
        g_tel.gemm_turbo_ghz = eff_ghz;
        KPRINTF("  Grouped PMU : %.2f FLOP/core-cyc (%.1f%% of %.1f-FLOP/cy calibrated peak) | IPC=%.2f | Turbo=%.2f GHz%s\n",
                real_fpc, eff_pct, peak_fpc, ipc, eff_ghz,
                best_pmu_mux ? " [mux-scaled]" : "");
    } else {
        double nom_fpc = gflops_core / g_cpu.tsc_ghz;
        g_tel.gemm_fpc     = nom_fpc;
        g_tel.gemm_eff_pct = 100.0 * nom_fpc / peak_fpc;
        KPRINTF("  Nominal Ratio: %.1f FLOP/TSC-cyc/core (%.1f%% of %.1f calibrated peak; base TSC %.2f GHz)\n",
                nom_fpc, g_tel.gemm_eff_pct, peak_fpc, g_cpu.tsc_ghz);
    }
    KPRINTF("  Accuracy Audit (13+ NN/NT/TN/TT/edge/K=0/alpha!=1 + qNaN + 4x Freivalds%s):\n",
            g_cpu.has_avx512vnni ? " + VNNI" : "");
    KPRINTF("       edges+TN=%s%s | finite=%s | rel_dot=%.2e | rel_freivalds=%.2e  [%s%s%s]\n",
            odd_ok ? "OK" : "FAIL",
            g_cpu.has_avx512vnni ? (vnni_ok ? " | vnni=OK" : " | vnni=FAIL") : "",
            main_finite ? "OK" : "NaN!", max_rel_dot, freiv_max_rel,
            verified ? C_GRN : C_RED, verified ? "VERIFIED" : "ERROR", C_RST);

    free(v); free(bv);
    (void)munmap(A, a_map); (void)munmap(B, b_map); (void)munmap(C, c_map);
    gemm_ws_destroy(&ws);
}

/* --- Stage 5: Quiesced Per-Core NUMA/L3 Telemetry HUD + JSON/CSV Export --- */
static void run_stage5_telemetry_hud(void) {
    kronos_quiesce_in_flight();
    print_stage(5, "KRONOS-X v5.0 PER-CORE NUMA/L3 WORK-STEALING, TIMERS & PREEMPTION HUD");

    uint64_t tot_exec = 0, tot_sw = 0, tot_dir = 0, tot_steal = 0, tot_pre = 0, tot_def = 0, tot_sleep = 0;
    uint64_t max_sw = 1;
    uint64_t sw_snap[KRONOS_MAX_CORES];
    for (int i = 0; i < g_kern.num_cores; i++) {
        KronosWorker *w = &g_kern.workers[i];
        uint64_t sw = stat_load(&w->stat_ctx_switches);
        sw_snap[i] = sw;
        tot_exec  += stat_load(&w->stat_executed);
        tot_sw    += sw;
        tot_dir   += stat_load(&w->stat_direct_switches);
        tot_steal += stat_load(&w->stat_steals);
        tot_pre   += stat_load(&w->stat_preemptions);
        tot_def   += stat_load(&w->stat_deferred_preempts);
        tot_sleep += stat_load(&w->stat_futex_sleeps);
        if (sw > max_sw) max_sw = sw;
    }

    uint64_t tw      = atomic_load_explicit(&g_kern.timer_wakeups, memory_order_relaxed);
    uint64_t tj_sum  = atomic_load_explicit(&g_kern.timer_jitter_sum_ns, memory_order_relaxed);
    uint64_t tj_max  = atomic_load_explicit(&g_kern.timer_jitter_max_ns, memory_order_relaxed);
    uint64_t tot_ipi = atomic_load_explicit(&g_kern.total_preempt_ipis, memory_order_relaxed);

    g_tel.tot_fibers          = tot_exec;
    g_tel.tot_switches        = tot_sw;
    g_tel.tot_direct          = tot_dir;
    g_tel.tot_steals          = tot_steal;
    g_tel.tot_preempts        = tot_pre;
    g_tel.tot_ipis            = tot_ipi;
    g_tel.tot_futex           = tot_sleep;
    g_tel.timer_wakeups       = tw;
    g_tel.timer_jitter_avg_us = (tw > 0) ? ((double)tj_sum / (double)tw / 1e3) : 0.0;
    g_tel.timer_jitter_max_us = (double)tj_max / 1e3;

    KPRINTF("  %-10s %-7s %7s %10s %9s %7s %7s %6s  %s\n",
            "vCore/CPU", "NUMA/L3", "Fibers", "CtxSwitch", "1HopDir", "Stolen", "Preempt", "Futex", "Activity Load Bar");
    KPRINTF("  %s────────── ─────── ─────── ────────── ───────── ─────── ─────── ──────  ────────────────────%s\n",
            C_DIM, C_RST);

    for (int i = 0; i < g_kern.num_cores; i++) {
        KronosWorker *w = &g_kern.workers[i];
        uint64_t sw = sw_snap[i];
        int bar = (int)((sw * 20ULL + max_sw / 2) / max_sw);
        if (bar > 20) bar = 20;
        KPRINTF("  %sv%02d->c%-3d%s n%-1d/L%-3d %7" PRIu64 " %10" PRIu64 " %9" PRIu64 " %7" PRIu64 " %7" PRIu64 " %6" PRIu64 "  %s",
                C_BLUE, i, w->hw_cpu, C_RST, w->numa_node, w->l3_dom,
                stat_load(&w->stat_executed), sw,
                stat_load(&w->stat_direct_switches),
                stat_load(&w->stat_steals), stat_load(&w->stat_preemptions),
                stat_load(&w->stat_futex_sleeps), C_CYAN);
        if (g_out_mode == OUT_PRETTY) {
            for (int b = 0; b < bar; b++) printf("█");
            printf("%s", C_DIM);
            for (int b = bar; b < 20; b++) printf("░");
            printf("%s\n", C_RST);
        }
    }

    KPRINTF("  %s────────── ─────── ─────── ────────── ───────── ─────── ─────── ──────  ────────────────────%s\n", C_DIM, C_RST);
    KPRINTF("  %sTOTAL%s              %7" PRIu64 " %10" PRIu64 " %9" PRIu64 " %7" PRIu64 " %7" PRIu64 " %6" PRIu64 "\n",
            C_BOLD, C_RST, tot_exec, tot_sw, tot_dir, tot_steal, tot_pre, tot_sleep);
    KPRINTF("  IPIs: %" PRIu64 " | Deferred Resched: %" PRIu64 " | Timers: %" PRIu64 " (avg jitter %.1f us, max %.1f us)\n",
            tot_ipi, tot_def, tw, g_tel.timer_jitter_avg_us, g_tel.timer_jitter_max_us);
}

static void emit_structured_telemetry(int fails) {
    if (g_out_mode == OUT_JSON) {
        printf("{\n"
               "  \"version\": \"5.0\",\n"
               "  \"cpu_brand\": \"%s\",\n"
               "  \"vendor\": \"%s\",\n"
               "  \"cores\": %d,\n"
               "  \"numa_nodes\": %d,\n"
               "  \"l3_domains\": %d,\n"
               "  \"tsc_ghz\": %.4f,\n"
               "  \"slab_cy_pair\": %.2f,\n"
               "  \"slab_mops\": %.2f,\n"
               "  \"slab_hit_pct\": %.3f,\n"
               "  \"slab_unique_1k_colors\": %u,\n"
               "  \"ctx_runnext_cy\": %.2f,\n"
               "  \"ctx_mux_ns\": %.2f,\n"
               "  \"mpmc_mmsgs\": %.3f,\n"
               "  \"canary_preempts\": %" PRIu32 ",\n"
               "  \"probe_avg_us\": %.2f,\n"
               "  \"probe_max_us\": %.2f,\n"
               "  \"jit_speedup\": %.2f,\n"
               "  \"jit_par_mops\": %.2f,\n"
               "  \"jit_bw_gbs\": %.2f,\n"
               "  \"jit_preempts\": %" PRIu32 ",\n"
               "  \"gemm_kernel\": \"%s\",\n"
               "  \"gemm_dim\": %d,\n"
               "  \"gemm_peak_gflops\": %.2f,\n"
               "  \"gemm_med_gflops\": %.2f,\n"
               "  \"gemm_fpc\": %.2f,\n"
               "  \"gemm_eff_pct\": %.2f,\n"
               "  \"gemm_ipc\": %.3f,\n"
               "  \"gemm_turbo_ghz\": %.3f,\n"
               "  \"tot_fibers\": %" PRIu64 ",\n"
               "  \"tot_switches\": %" PRIu64 ",\n"
               "  \"tot_direct_switches\": %" PRIu64 ",\n"
               "  \"tot_steals\": %" PRIu64 ",\n"
               "  \"tot_preempts\": %" PRIu64 ",\n"
               "  \"tot_ipis\": %" PRIu64 ",\n"
               "  \"tot_futex_sleeps\": %" PRIu64 ",\n"
               "  \"timer_wakeups\": %" PRIu64 ",\n"
               "  \"timer_jitter_avg_us\": %.2f,\n"
               "  \"timer_jitter_max_us\": %.2f,\n"
               "  \"failures\": %d\n"
               "}\n",
               g_cpu.brand, g_cpu.vendor, g_kern.num_cores, g_cpu.num_numa_nodes,
               g_cpu.num_l3_domains, g_cpu.tsc_ghz,
               g_tel.slab_cy_pair, g_tel.slab_mops, g_tel.slab_hit_pct, g_tel.slab_unique_colors,
               g_tel.ctx_runnext_cy, g_tel.ctx_mux_ns, g_tel.mpmc_mmsgs,
               g_tel.canary_preempts, g_tel.probe_avg_us, g_tel.probe_max_us,
               g_tel.jit_speedup, g_tel.jit_par_mops, g_tel.jit_bw_gbs, g_tel.jit_preempts,
               g_tel.gemm_kernel ? g_tel.gemm_kernel : "none", g_tel.gemm_dim,
               g_tel.gemm_peak_gflops, g_tel.gemm_med_gflops, g_tel.gemm_fpc,
               g_tel.gemm_eff_pct, g_tel.gemm_ipc, g_tel.gemm_turbo_ghz,
               g_tel.tot_fibers, g_tel.tot_switches, g_tel.tot_direct, g_tel.tot_steals,
               g_tel.tot_preempts, g_tel.tot_ipis, g_tel.tot_futex,
               g_tel.timer_wakeups, g_tel.timer_jitter_avg_us, g_tel.timer_jitter_max_us,
               fails);
    } else if (g_out_mode == OUT_CSV) {
        printf("version,cores,numa_nodes,l3_domains,tsc_ghz,slab_cy_pair,slab_mops,ctx_runnext_cy,"
               "mpmc_mmsgs,canary_preempts,jit_speedup,gemm_kernel,gemm_dim,gemm_peak_gflops,"
               "gemm_eff_pct,timer_jitter_avg_us,failures\n");
        printf("5.0,%d,%d,%d,%.4f,%.2f,%.2f,%.2f,%.3f,%" PRIu32 ",%.2f,%s,%d,%.2f,%.2f,%.2f,%d\n",
               g_kern.num_cores, g_cpu.num_numa_nodes, g_cpu.num_l3_domains, g_cpu.tsc_ghz,
               g_tel.slab_cy_pair, g_tel.slab_mops, g_tel.ctx_runnext_cy, g_tel.mpmc_mmsgs,
               g_tel.canary_preempts, g_tel.jit_speedup,
               g_tel.gemm_kernel ? g_tel.gemm_kernel : "none", g_tel.gemm_dim,
               g_tel.gemm_peak_gflops, g_tel.gemm_eff_pct, g_tel.timer_jitter_avg_us, fails);
    }
}

static int parse_u64_arg(const char *flag, const char *s, uint64_t *out) {
    if (!s || *s < '0' || *s > '9') {
        fprintf(stderr, "Invalid numeric value for %s: '%s'\n", flag, s ? s : "");
        return -1;
    }
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        fprintf(stderr, "Invalid numeric value for %s: '%s'\n", flag, s);
        return -1;
    }
    *out = (uint64_t)v;
    return 0;
}

int main(int argc, char **argv) {
    g_use_color = isatty(STDOUT_FILENO) && (getenv("NO_COLOR") == NULL);

    int req_cores = 0;
    int gemm_dim  = 1024;
    uint64_t quantum_us = 1000;
    const char *req_gemm_kernel = "auto";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cores") == 0) {
            uint64_t v = 0;
            if (i + 1 >= argc || parse_u64_arg("--cores", argv[++i], &v) != 0 || v == 0 || v > KRONOS_MAX_CORES) {
                fprintf(stderr, "--cores requires integer in [1, %d]\n", KRONOS_MAX_CORES);
                return 1;
            }
            req_cores = (int)v;
        } else if (strcmp(argv[i], "--gemm") == 0) {
            uint64_t v = 0;
            if (i + 1 >= argc || parse_u64_arg("--gemm", argv[++i], &v) != 0 || v < 64 || v > 8192) {
                fprintf(stderr, "--gemm requires integer in [64, 8192]\n");
                return 1;
            }
            gemm_dim = (int)v;
        } else if (strcmp(argv[i], "--quantum-us") == 0) {
            uint64_t v = 0;
            if (i + 1 >= argc || parse_u64_arg("--quantum-us", argv[++i], &v) != 0 || v < 200 || v > 100000) {
                fprintf(stderr, "--quantum-us requires integer in [200, 100000]\n");
                return 1;
            }
            quantum_us = v;
        } else if (strcmp(argv[i], "--gemm-kernel") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--gemm-kernel requires auto|avx512|avx512-9x48|avx2|scalar\n");
                return 1;
            }
            req_gemm_kernel = argv[++i];
            if (strcmp(req_gemm_kernel, "auto") != 0 && strcmp(req_gemm_kernel, "avx512") != 0 &&
                strcmp(req_gemm_kernel, "avx512-9x48") != 0 && strcmp(req_gemm_kernel, "avx2") != 0 &&
                strcmp(req_gemm_kernel, "scalar") != 0) {
                fprintf(stderr, "Invalid --gemm-kernel '%s' (expected auto|avx512|avx512-9x48|avx2|scalar)\n",
                        req_gemm_kernel);
                return 1;
            }
        } else if (strcmp(argv[i], "--no-color") == 0) {
            g_use_color = 0;
        } else if (strcmp(argv[i], "--json") == 0) {
            g_out_mode  = OUT_JSON;
            g_use_color = 0;
        } else if (strcmp(argv[i], "--csv") == 0) {
            g_out_mode  = OUT_CSV;
            g_use_color = 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--cores N (1..%d)] [--gemm N (64..8192)] [--quantum-us Q (200..100000)]\n"
                   "       [--gemm-kernel auto|avx512|avx512-9x48|avx2|scalar] [--no-color] [--json|--csv]\n",
                   argv[0], KRONOS_MAX_CORES);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s (try --help)\n", argv[i]);
            return 1;
        }
    }

    print_banner();

    print_stage(0, "SILICON, FULL-CORE HYBRID ISA-INTERSECT, NUMA/L3 TOPOLOGY & PEAK PROBE");
    kronos_probe_silicon(req_cores);
    if (req_cores <= 0) req_cores = g_cpu.avail_cpus;

    KPRINTF("  CPU Model   : %s%s%s (%s%s)\n", C_BOLD, g_cpu.brand, C_RST,
            g_cpu.vendor, g_cpu.has_hybrid ? ", Hybrid P/E" : "");
    KPRINTF("  Topology    : %d online / %d allowed (%d phys, %d P-core, %d pkg, %d NUMA, %d L3) -> %s%d vCores%s\n",
            g_cpu.online_cpus, g_cpu.avail_cpus, g_cpu.phys_cores, g_cpu.p_cores,
            g_cpu.num_packages, g_cpu.num_numa_nodes, g_cpu.num_l3_domains,
            C_CYAN, req_cores, C_RST);
    if (req_cores > g_cpu.avail_cpus) {
        KPRINTF("  %s[WARN] Oversubscribed: %d vCores on %d allowed CPUs (barrier yield throttling active).%s\n",
                C_YEL, req_cores, g_cpu.avail_cpus, C_RST);
    }
    KPRINTF("  TSC & Pause : %s%.3f GHz%s (%s, %.3f ns/cy) | PAUSE=%.1f cy | RDTSCP=%s | TSC-CS=%s\n",
            C_GRN, g_cpu.tsc_ghz, C_RST, g_cpu.tsc_src, g_cpu.ns_per_cycle, g_pause_cycles,
            g_cpu.has_rdtscp ? "YES" : "NO", g_cpu.tsc_clocksource_ok ? "VERIFIED" : "MONOTONIC");
    KPRINTF("  Caches (Min): L1d=%" PRIu64 "KiB(%uw/%ut) | L1i=%" PRIu64 "KiB | L2=%" PRIu64 "KiB(act-eff %" PRIu64 "KiB) | L3=%" PRIu64 "KiB | Line=%uB\n",
            (uint64_t)(g_cpu.l1d_bytes / 1024ULL), g_cpu.l1d_ways, g_cpu.l1d_shared_by,
            (uint64_t)(g_cpu.l1i_bytes / 1024ULL),
            (uint64_t)(g_cpu.l2_bytes / 1024ULL), (uint64_t)(g_cpu.l2_eff_per_core / 1024ULL),
            (uint64_t)(g_cpu.l3_bytes / 1024ULL), g_cpu.cache_line_bytes);
    KPRINTF("  ISA (Common): AVX2=%d FMA=%d BMI1/2=%d/%d LZCNT=%d POPCNT=%d PREFETCHW=%d AVX-512F/DQ/BW/VL/VNNI=%d/%d/%d/%d/%d\n",
            g_cpu.has_avx2, g_cpu.has_fma, g_cpu.has_bmi1, g_cpu.has_bmi2, g_cpu.has_lzcnt,
            g_cpu.has_popcnt, g_cpu.has_prefetchw,
            g_cpu.has_avx512f, g_cpu.has_avx512dq, g_cpu.has_avx512bw, g_cpu.has_avx512vl, g_cpu.has_avx512vnni);
    KPRINTF("  Calib Peak  : SSE2=%.1f | AVX2+FMA=%.1f | AVX-512F=%.1f FLOP/core-cy | XSAVE64=%" PRIu64 "B (0x%" PRIX32 ")\n",
            g_cpu.peak_fma_fpc_sse2, g_cpu.peak_fma_fpc_avx2, g_cpu.peak_fma_fpc_avx512,
            g_cpu.xsave_size, g_cpu.xsave_mask_lo);

    if (g_cpu.cet_shstk_active) {
        fprintf(stderr, "[KRONOS] Hardware CET Shadow Stack (SHSTK) is active & locked; cannot switch stacks.\n");
        return 2;
    }

    kronos_init(req_cores, quantum_us);
    KPRINTF("  OS Services : membarrier=%s | Grouped PMU=%s | AltStack=%zuKiB | CET-SHSTK=%s | Link=%s\n",
            g_has_membar ? "EXPEDITED(raw-sys)" : "fallback",
            g_cpu.pmu_available ? (g_cpu.has_hybrid ? "HYBRID-ACTIVE" : "ACTIVE") : "restricted",
            g_cpu.altstack_sz / 1024u,
            g_cpu.cet_shstk_disabled ? "disabled" : "off",
            g_kern.is_static_binary ? "STATIC(preempt-off)" : "dynamic");

    uintptr_t c_lo = atomic_load(&g_kern.code_ranges[0].lo);
    uintptr_t c_hi = atomic_load(&g_kern.code_ranges[0].hi);
    KPRINTF("  Kernel Boot : .text=[0x%" PRIxPTR "..0x%" PRIxPTR "] (%s x%d + %d JIT slots) Q=%" PRIu64 "us (%s)  [%sONLINE%s]\n",
            c_lo, c_hi, g_kern.code_src, g_kern.static_code_slots,
            KRONOS_MAX_CODE_RANGES - g_kern.static_code_slots,
            g_kern.quantum_us, g_cpu.tsc_clocksource_ok ? "InvTSC" : "MONOTONIC",
            C_GRN, C_RST);
    if (!g_kern.preempt_supported) {
        KPRINTF("  %s[WARN] Async preemption safely disabled (static binary, .text unknown, altstack failed, or AVX w/o XSAVE).%s\n",
                C_YEL, C_RST);
    }

    run_stage1_slab();
    run_stage2_scheduler_and_preemption();
    run_stage3_jit();
    run_stage4_gemm(gemm_dim, req_gemm_kernel);
    run_stage5_telemetry_hud();

    int fails = atomic_load(&g_failures);
    emit_structured_telemetry(fails);

    kronos_shutdown();

    if (fails == 0) {
        KPRINTF("\n%s%s[KRONOS-X v5.0] All 5 kernel stages verified with 0 failures. Clean shutdown complete.%s\n\n",
                C_GRN, C_BOLD, C_RST);
        return 0;
    } else {
        KPRINTF("\n%s%s[KRONOS-X v5.0] %d verification failure(s) detected!%s\n\n",
                C_RED, C_BOLD, fails, C_RST);
        return 1;
    }
}
#endif