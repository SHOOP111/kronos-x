# KRONOS-X v5.0

Single-file Linux `x86_64` bare-metal-grade preemptive CPU microkernel and dynamic silicon execution engine written in C11 + hand-written `x86_64` assembly (`kernel.c`).

Zero external dependencies beyond POSIX/Linux (`-pthread`). Zero compile-time ISA flags (`-mavx2`, `-mfma`, `-mavx512f`, `-lm` are **not** required) — all SIMD kernels, hardware canaries, and JIT codegen paths are dispatched dynamically from runtime CPUID + XCR0 + per-core ISA intersection.

## Architecture & Subsystems

1. **Silicon & Topology Probe (`Section 1`)**
   - Full active-core ISA intersection (CPUID leaf `0x1A` + `sysfs`) for heterogeneous P/E-core CPUs.
   - NUMA node + package + L3 CCD topology ordering and active-set L2 sharing calculation.
   - Bracketed median-of-5 TSC frequency calibration, `PAUSE` latency calibration, and boot-time hardware FMA peak micro-probes (SSE2, AVX2+FMA, AVX-512F).
   - Hybrid-aware grouped per-TID Linux `perf_event_open` PMU reader with multiplexing guard.
2. **Raw Linux Syscalls & Primitives (`Section 2`)**
   - Direct inline `syscall` wrappers for `futex`/`FUTEX_WAIT_BITSET_PRIVATE`, `membarrier` (`MEMBARRIER_CMD_PRIVATE_EXPEDITED`), `tgkill`, and `arch_prctl`.
   - FIFO fair ticket locks (`KronosTicketLock`), contention-aware TTAS compact spinlocks (`KronosCompactSpin`), CSE-proof per-fiber `KRONOS_ERRNO`, and single-instruction `%fs:` `asm goto` preemption count.
3. **Bonwick Two-Magazine O(1) Slab Allocator (`Section 3`)**
   - 16 size classes (`32 B` .. `4 KiB`, plus `>4 KiB` fallback), branchless sentinel-magazine fast path, exact modular cache-line coloring, zero-strand multi-chunk 2 MiB THP arena carver, optional `-DKRONOS_SLAB_DEBUG` poisoning.
4. **Hand-Written `x86_64` Assembly Switcher & Canaries (`Section 4`)**
   - `.hidden` visibility, exact DWARF CFI `.eh_frame` unwind metadata, CET IBT `endbr64` landing pads, non-executable `.note.GNU-stack`.
   - 4-way context switcher (`to_sched`, `from_sched`, `fninit`-hardened `exit`, 1-hop `switch_direct`).
   - Stack-safe `XSAVE64`/`XRSTOR64` async preemption trampoline preserving GPRs, flags (including `CF`/`DF`), x87, MXCSR, XMM/YMM/ZMM, opmasks `k0..k7`, and the 128 B System V red zone.
   - Hardware AVX-512 VNNI `vpdpbusd` `u8*s8` dot-product kernel.
5. **Lock-Free M:N Work-Stealing Scheduler & Tickless LAPIC (`Section 5`)**
   - Strict C11 Chase-Lev work-stealing deques, cache-line-strided MPMC per-worker inboxes, overflow-safe lock-free global run queue, and non-stealable per-core pinned queues for strict NUMA/L3 locality.
   - NUMA/L3-CCD-biased steal-half fanout, per-worker unlocked-resize binary min-heap timers, 0 Hz tickless absolute-time LAPIC controller, writer-locked seqlock executable code ranges, and static-link guard.
6. **Synchronization & IPC (`Section 6`)**
   - 4-state UAF-proof `KronosLatch`, bank-parity-preserving wrap-safe $O(\log P)$ dissemination barrier (`KronosDissemBarrier`), and linearizable Vyukov MPMC channel (`KronosChan`) with bit-63 atomic close and tickless external futex waiters.
7. **Sandboxed K-ISA Verifier & Optimizing W^X JIT (`Section 7`)**
   - 57-opcode register ISA with static CFG/RFLAGS/capability verification, iterative fixed-point dead-code elimination, hardware backedge fuel metering (`KBPF_CAP_BOUNDED_LOOP`), byte-bounded Spectre-v1 `lfence` memory sandboxing (`KBPF_CAP_MEM_BOUNDED`), Intel JCC-erratum alignment guards, BMI1/BMI2/LZCNT/POPCNT emission, rel8/rel32 relaxation, and `PROT_NONE` W^X page recycling.
8. **Double-Buffered Persistent-Fiber BLIS SGEMM Engine (`Section 8`)**
   - Analytical L1d/active-L2 blocking, balanced `kc_step`/`nc_step`, strictly core-pinned NUMA/L3 workers, full `NN`/`NT`/`TN`/`TT` transpose support, and dynamic kernel dispatch:
     - AVX-512F `14x32` and `9x48` (2.25 FMAs/load) with staggered LFB-safe `prefetchw`, zero-copy AVX-512 opmask edge kernels, and SIMD `pack_A`/`pack_B` (`MR=14` and `MR=9`).
     - AVX2+FMA `6x16` with aligned-redzone AVX `pack_A`.
     - Portable scalar fallback.

## Building & Running (Linux x86_64)

```bash
gcc -std=c11 -O3 -Wall -Wextra -Wshadow -pthread kernel.c -o kronos
./kronos [--cores N] [--gemm N] [--quantum-us Q] \
         [--gemm-kernel auto|avx512|avx512-9x48|avx2|scalar] \
         [--no-color] [--json | --csv]
```

## White-Box Adversarial Test Suite

```bash
gcc -std=c11 -O3 -Wall -Wextra -Wshadow -Wno-unused-function -Wno-unused-variable -pthread \
    tests/test_suite.c -o test_suite
./test_suite
./test_suite --cores 1
./test_suite --cores 8
```

## CI Matrix (GitHub Actions)

Every push runs on GitHub Actions across:
- **GCC `-O3` strict** (`-Wall -Wextra -Wshadow -Wpedantic -Wconversion`), **Clang `-O3` strict**, **GCC `-O3 -flto -fPIE -pie`**, **GCC `-O3 -no-pie -DKRONOS_SLAB_DEBUG`**, **GCC `-O0 -g`**, **Clang `-O2 -fsanitize=undefined`**, and **GCC `-O2 -static`**.
- **Strict `-Werror` audit** (GCC + Clang), ELF `GNU_STACK` non-exec check, DWARF `.eh_frame` FDE check, and `cppcheck`.
- **QEMU-x86_64 cross-microarchitecture verification** across `Nehalem` (SSE4.2 only), `Haswell` (AVX2+FMA+BMI1/2+LZCNT), `Skylake-Server` (AVX-512F `14x32` & `9x48`), and `Cascadelake-Server,+avx512vnni` (AVX-512 VNNI).
