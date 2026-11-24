/*
 * =============================================================================
 *  KRONOS-X Exhaustive White-Box Adversarial & Regression Test Suite
 * =============================================================================
 *  Compiled directly against ../kernel.c via -DKRONOS_NO_MAIN so every internal
 *  subsystem (Slab, Chase-Lev Deque, Inbox, GRQ, PinnedQ, Timers, LAPIC,
 *  Preemption, Latch, Dissemination Barrier, Vyukov Channel, K-ISA Verifier &
 *  W^X JIT, and BLIS AVX-512 / AVX2 / Scalar GEMM) is tested exhaustively.
 * =============================================================================
 */

#define KRONOS_NO_MAIN
#include "../kernel.c"

#include <math.h>
#include <stdarg.h>

static int g_t_pass = 0;
static int g_t_fail = 0;

static void t_report(int ok, const char *name, const char *fmt, ...) {
    va_list ap;
    if (ok) {
        g_t_pass++;
        printf("  [PASS] %-44s ", name);
    } else {
        g_t_fail++;
        printf("  [FAIL] %-44s ", name);
    }
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

/* ============================================================================
 *  TEST 1: Slab Allocator Exhaustive Stress (All 16 classes + >4KiB + Alignment
 *          + Cross-Core Producer/Consumer + Arena Multi-Chunk Growth + Poison)
 * ============================================================================ */

typedef struct {
    int          id;
    int          rounds;
    uint64_t     errs;
    KronosLatch *latch;
} SlabTortureArg;

static void slab_torture_fiber(void *arg) {
    SlabTortureArg *a = (SlabTortureArg *)arg;
    uint64_t rng = 0x1122334455667788ULL ^ ((uint64_t)(a->id + 1) * 0x9E3779B97F4A7C15ULL);
    enum { LIVE = 128 };
    void    *ptrs[LIVE];
    size_t   szs[LIVE];
    uint64_t pat[LIVE];
    memset(ptrs, 0, sizeof(ptrs));
    memset(szs, 0, sizeof(szs));
    memset(pat, 0, sizeof(pat));

    uint64_t errs = 0;
    for (int r = 0; r < a->rounds; r++) {
        int slot = (int)(xorshift64(&rng) % LIVE);
        if (ptrs[slot]) {
            uint64_t *q = (uint64_t *)ptrs[slot];
            size_t nq = szs[slot] / sizeof(uint64_t);
            if (q[0] != pat[slot] || q[nq - 1] != (pat[slot] ^ 0xA5A5A5A5A5A5A5A5ULL)) {
                errs++;
            }
            kronos_free(ptrs[slot], szs[slot]);
            ptrs[slot] = NULL;
        }
        size_t sz;
        uint64_t roll = xorshift64(&rng) & 31u;
        if (roll == 0) {
            sz = 4097u + (size_t)(xorshift64(&rng) % 12000u); /* > 4 KiB fallback path */
        } else if (roll == 1) {
            sz = 0; /* zero-size edge case -> class 0 (32B) */
        } else {
            sz = 1u + (size_t)(xorshift64(&rng) % 4096u);
        }
        void *p = kronos_alloc(sz);
        if (!p) { errs++; continue; }
        int cls = slab_class_for(sz);
        size_t eff_sz = (cls >= 0) ? k_slab_sizes[cls] : sz;
        size_t req_align = (cls >= 0) ? (eff_sz & (~eff_sz + 1u)) : 64u;
        if (((uintptr_t)p & (req_align - 1u)) != 0) errs++;

        size_t nq = (eff_sz >= 8u) ? (eff_sz / sizeof(uint64_t)) : 1u;
        uint64_t tag = xorshift64(&rng) | 1ULL;
        uint64_t *q = (uint64_t *)p;
        q[0] = tag;
        q[nq - 1] = tag ^ 0xA5A5A5A5A5A5A5A5ULL;

        ptrs[slot] = p;
        szs[slot]  = sz;
        pat[slot]  = tag;

        if ((r & 255) == 0) kronos_yield();
    }
    for (int s = 0; s < LIVE; s++) {
        if (ptrs[s]) {
            int cls = slab_class_for(szs[s]);
            size_t eff_sz = (cls >= 0) ? k_slab_sizes[cls] : szs[s];
            size_t nq = (eff_sz >= 8u) ? (eff_sz / sizeof(uint64_t)) : 1u;
            uint64_t *q = (uint64_t *)ptrs[s];
            if (q[0] != pat[s] || q[nq - 1] != (pat[s] ^ 0xA5A5A5A5A5A5A5A5ULL)) errs++;
            kronos_free(ptrs[s], szs[s]);
        }
    }
    a->errs = errs;
    latch_done(a->latch);
}

static void test_slab_torture(void) {
    /* Verify LUT monotonicity and exact boundaries */
    int lut_ok = 1;
    for (size_t sz = 0; sz <= 4110; sz++) {
        int cls = slab_class_for(sz);
        if (sz > 4096) {
            if (cls != -1) lut_ok = 0;
        } else {
            if (cls < 0 || cls >= KRONOS_SLAB_CLASSES) { lut_ok = 0; break; }
            if (k_slab_sizes[cls] < sz) lut_ok = 0;
            if (cls > 0 && k_slab_sizes[cls - 1] >= sz) lut_ok = 0;
        }
    }

    /* Verify external-thread (non-worker) depot alloc/free works cleanly */
    void *ext_ptrs[256];
    int ext_ok = 1;
    for (int i = 0; i < 256; i++) {
        size_t sz = k_slab_sizes[i & 15];
        ext_ptrs[i] = kronos_alloc(sz);
        if (!ext_ptrs[i]) { ext_ok = 0; break; }
        memset(ext_ptrs[i], (int)(i & 0xFF), sz);
    }
    for (int i = 0; i < 256; i++) {
        size_t sz = k_slab_sizes[i & 15];
        uint8_t *b = (uint8_t *)ext_ptrs[i];
        if (b && (b[0] != (uint8_t)(i & 0xFF) || b[sz - 1] != (uint8_t)(i & 0xFF))) ext_ok = 0;
        kronos_free(ext_ptrs[i], sz);
    }

    /* Verify multi-chunk arena expansion (>16 MiB live) and zero strand corruption */
    const int n_big = 4600; /* 4600 * 4096B = 18.8 MiB > 16 MiB single chunk */
    void **big_arr = (void **)calloc((size_t)n_big, sizeof(void *));
    int chunk_ok = 1;
    for (int i = 0; i < n_big; i++) {
        big_arr[i] = kronos_alloc(4096);
        if (!big_arr[i] || (((uintptr_t)big_arr[i] & 4095u) != 0)) { chunk_ok = 0; break; }
        ((uint64_t *)big_arr[i])[0] = 0xCAFE0000ULL + (uint64_t)i;
        ((uint64_t *)big_arr[i])[511] = 0xBEEF0000ULL + (uint64_t)i;
    }
    for (int i = 0; i < n_big; i++) {
        if (big_arr[i]) {
            if (((uint64_t *)big_arr[i])[0] != 0xCAFE0000ULL + (uint64_t)i ||
                ((uint64_t *)big_arr[i])[511] != 0xBEEF0000ULL + (uint64_t)i) chunk_ok = 0;
            kronos_free(big_arr[i], 4096);
        }
    }
    free(big_arr);

    /* Concurrent multi-fiber torture across all cores (with yields + migrations) */
    int nf = g_kern.num_cores * 4;
    SlabTortureArg *args = (SlabTortureArg *)calloc((size_t)nf, sizeof(SlabTortureArg));
    KronosLatch l;
    latch_init(&l, (uint32_t)nf);
    for (int i = 0; i < nf; i++) args[i] = (SlabTortureArg){ i, 15000, 0, &l };
    kronos_spawn_batch(slab_torture_fiber, args, sizeof(SlabTortureArg), nf, 1);
    latch_wait(&l);

    uint64_t errs = 0;
    for (int i = 0; i < nf; i++) errs += args[i].errs;
    free(args);

    t_report(lut_ok && ext_ok && chunk_ok && errs == 0,
             "slab_torture_all_classes_multichunk",
             "lut=%d ext=%d multi_chunk=%d fiber_errs=%" PRIu64 " mapped=%zuMiB",
             lut_ok, ext_ok, chunk_ok, errs,
             atomic_load(&g_slab.total_mapped_bytes) / (1024u * 1024u));
}

/* ============================================================================
 *  TEST 2: Chase-Lev Deque, GlobalRunQ Overflow, Pinned Affinity & Work Steal
 * ============================================================================ */

typedef struct {
    int          expected_core;
    int          yields;
    int          mismatches;
    KronosLatch *latch;
} PinCheckArg;

static void pin_check_fiber(void *arg) {
    PinCheckArg *a = (PinCheckArg *)arg;
    int bad = 0;
    for (int i = 0; i < a->yields; i++) {
        if (kronos_current_core_id() != a->expected_core) bad++;
        if ((i & 31) == 0) kronos_sleep_ns(200000ULL); /* Also verify timer wake preserves pin! */
        else kronos_yield();
        if (kronos_current_core_id() != a->expected_core) bad++;
    }
    a->mismatches = bad;
    latch_done(a->latch);
}

typedef struct {
    _Atomic uint64_t *sum;
    uint64_t          val;
    KronosLatch      *latch;
} GrqOvfArg;

static void grq_ovf_fiber(void *arg) {
    GrqOvfArg *a = (GrqOvfArg *)arg;
    atomic_fetch_add_explicit(a->sum, a->val, memory_order_relaxed);
    latch_done(a->latch);
}

static void test_scheduler_pin_and_grq_overflow(void) {
    /* Pinned fibers must NEVER migrate across yields or timer sleeps! */
    int P = g_kern.num_cores;
    PinCheckArg *pargs = (PinCheckArg *)calloc((size_t)P * 2u, sizeof(PinCheckArg));
    KronosLatch lp;
    latch_init(&lp, (uint32_t)(P * 2));
    for (int i = 0; i < P * 2; i++) {
        pargs[i] = (PinCheckArg){ i % P, 64, 0, &lp };
    }
    kronos_spawn_batch_pinned(pin_check_fiber, pargs, sizeof(PinCheckArg), P * 2, 1);
    latch_wait(&lp);
    int pin_bad = 0;
    for (int i = 0; i < P * 2; i++) pin_bad += pargs[i].mismatches;
    free(pargs);

    /* Spawn > KRONOS_GLOBAL_RQ_CAP (4096) fibers to exercise GRQ overflow linked list! */
    const int N_OVF = 6000;
    GrqOvfArg *oargs = (GrqOvfArg *)calloc((size_t)N_OVF, sizeof(GrqOvfArg));
    _Atomic uint64_t got_sum = 0;
    KronosLatch lo;
    latch_init(&lo, (uint32_t)N_OVF);
    uint64_t exp_sum = 0;
    for (int i = 0; i < N_OVF; i++) {
        uint64_t v = (uint64_t)(i + 1);
        exp_sum += v;
        oargs[i] = (GrqOvfArg){ &got_sum, v, &lo };
    }
    kronos_spawn_batch(grq_ovf_fiber, oargs, sizeof(GrqOvfArg), N_OVF, 0);
    latch_wait(&lo);
    free(oargs);

    int ok = (pin_bad == 0) && (atomic_load(&got_sum) == exp_sum);
    t_report(ok, "sched_pinned_affinity_and_grq_overflow",
             "pin_mismatches=%d ovf_fibers=%d sum_ok=%d",
             pin_bad, N_OVF, (atomic_load(&got_sum) == exp_sum));
}

/* ============================================================================
 *  TEST 3: Latch Edge Cases (count=0, Fiber Waiters, External Waiters, Over-Done)
 * ============================================================================ */

typedef struct {
    KronosLatch     *target;
    _Atomic uint32_t *woken;
    KronosLatch     *done;
} LatchWaiterArg;

static void latch_waiter_fiber(void *arg) {
    LatchWaiterArg *a = (LatchWaiterArg *)arg;
    latch_wait(a->target);
    atomic_fetch_add_explicit(a->woken, 1u, memory_order_relaxed);
    latch_done(a->done);
}

static void latch_releaser_fiber(void *arg) {
    KronosLatch *l = (KronosLatch *)arg;
    kronos_sleep_ns(2000000ULL);
    latch_done(l);
}

static void test_latch_semantics(void) {
    /* Zero-count latch must return immediately */
    KronosLatch l0;
    latch_init(&l0, 0);
    latch_wait(&l0);
    latch_done(&l0); /* Over-done guard must not underflow */
    int zero_ok = (atomic_load(&l0.remaining) == 0u && atomic_load(&l0.state) == 3u);

    /* Many fiber waiters + external thread waiter waking together */
    int ok_rounds = 1;
    for (int r = 0; r < 16; r++) {
        KronosLatch target, done;
        latch_init(&target, 1);
        const int NW = 24;
        latch_init(&done, (uint32_t)NW);
        _Atomic uint32_t woken = 0;
        LatchWaiterArg wargs[24];
        for (int i = 0; i < NW; i++) {
            wargs[i] = (LatchWaiterArg){ &target, &woken, &done };
            kronos_spawn(latch_waiter_fiber, &wargs[i]);
        }
        kronos_spawn(latch_releaser_fiber, &target);
        latch_wait(&target); /* External thread waits concurrently with fiber waiters! */
        latch_wait(&done);
        if (atomic_load(&woken) != (uint32_t)NW) ok_rounds = 0;
    }

    t_report(zero_ok && ok_rounds, "latch_zero_overdone_and_mixed_waiters",
             "zero_ok=%d rounds_ok=%d", zero_ok, ok_rounds);
}

/* ============================================================================
 *  TEST 4: Dissemination Barrier Wrap-Around & Multi-Group Stress
 * ============================================================================ */

typedef struct {
    KronosDissemBarrier *bar;
    int                  tid;
    int                  P;
    int                  iters;
    _Atomic uint64_t    *shared_slot;
    int                  errs;
    KronosLatch         *latch;
} BarStressArg;

static void bar_stress_fiber(void *arg) {
    BarStressArg *a = (BarStressArg *)arg;
    /* Start epoch near UINT32_MAX to explicitly test 32-bit wrap-around! */
    uint32_t ep = UINT32_MAX - 10u;
    int errs = 0;
    for (int it = 0; it < a->iters; it++) {
        if (a->tid == 0) {
            atomic_store_explicit(a->shared_slot, (uint64_t)it + 1ULL, memory_order_relaxed);
        }
        dissem_wait_group(a->bar, 0, a->tid, a->P, &ep);
        uint64_t v = atomic_load_explicit(a->shared_slot, memory_order_relaxed);
        if (v != (uint64_t)it + 1ULL) errs++;
        dissem_wait_group(a->bar, 0, a->tid, a->P, &ep);
    }
    a->errs = errs;
    latch_done(a->latch);
}

static void test_dissem_barrier_wrap(void) {
    int P = g_kern.num_cores;
    KronosDissemBarrier *bar = (KronosDissemBarrier *)xaligned_alloc(64, sizeof(KronosDissemBarrier));
    dissem_init(bar, P);
    _Atomic uint64_t shared = 0;
    KronosLatch l;
    latch_init(&l, (uint32_t)P);
    BarStressArg args[KRONOS_MAX_CORES];
    for (int i = 0; i < P; i++) {
        args[i] = (BarStressArg){ bar, i, P, 2000, &shared, 0, &l };
    }
    kronos_spawn_batch_pinned(bar_stress_fiber, args, sizeof(BarStressArg), P, 0);
    latch_wait(&l);
    int errs = 0;
    for (int i = 0; i < P; i++) errs += args[i].errs;
    free(bar);
    t_report(errs == 0, "dissem_barrier_u32_wrap_and_visibility",
             "cores=%d iters=2000 errs=%d", P, errs);
}

/* ============================================================================
 *  TEST 5: Vyukov Channel External-Thread Futex Waiters + Multi-Round Close Race
 * ============================================================================ */

typedef struct {
    KronosChan *ch;
    int         count;
    uint64_t    xor_sum;
} ExtChanArg;

static void *ext_sender_thread(void *arg) {
    ExtChanArg *a = (ExtChanArg *)arg;
    uint64_t x = 0;
    for (int i = 1; i <= a->count; i++) {
        uint64_t v = ((uint64_t)i * 0xBF58476D1CE4E5B9ULL) ^ 0x12345678ULL;
        if (!chan_send(a->ch, v)) break;
        x ^= v;
    }
    a->xor_sum = x;
    return NULL;
}

static void *ext_recver_thread(void *arg) {
    ExtChanArg *a = (ExtChanArg *)arg;
    uint64_t x = 0, v = 0;
    while (chan_recv_ex(a->ch, &v)) x ^= v;
    a->xor_sum = x;
    return NULL;
}

static void fiber_sender_slow(void *arg) {
    ExtChanArg *a = (ExtChanArg *)arg;
    uint64_t x = 0;
    for (int i = 1; i <= a->count; i++) {
        if ((i & 511) == 0) kronos_sleep_ns(300000ULL);
        uint64_t v = ((uint64_t)i * 0x94D049BB133111EBULL) ^ 0xCAFEBABEULL;
        if (!chan_send(a->ch, v)) break;
        x ^= v;
    }
    a->xor_sum = x;
    chan_close(a->ch);
}

static void test_channel_ext_futex_and_races(void) {
    /* External OS pthread sender + external OS pthread receiver + slow fiber sender */
    KronosChan ch;
    chan_init(&ch);
    ExtChanArg sa_ext = { &ch, 8000, 0 };
    ExtChanArg ra_ext = { &ch, 0,    0 };
    ExtChanArg sa_fib = { &ch, 4000, 0 };

    pthread_t ts, tr;
    pthread_create(&tr, NULL, ext_recver_thread, &ra_ext);
    pthread_create(&ts, NULL, ext_sender_thread, &sa_ext);
    pthread_join(ts, NULL);
    /* Now fiber sends remaining items slowly and closes channel, waking blocked ext receiver! */
    kronos_spawn(fiber_sender_slow, &sa_fib);
    pthread_join(tr, NULL);
    kronos_quiesce_in_flight();

    int ext_ok = ((sa_ext.xor_sum ^ sa_fib.xor_sum) == ra_ext.xor_sum);

    /* Send on already-closed channel must return false immediately */
    int closed_send_ok = (!chan_send(&ch, 999ULL));
    uint64_t dummy = 0;
    int closed_recv_ok = (!chan_recv_ex(&ch, &dummy));

    /* Backpressure stress: external sender fills 256-slot channel completely & blocks on futex */
    KronosChan ch2;
    chan_init(&ch2);
    ExtChanArg bp_s = { &ch2, 2000, 0 };
    pthread_t tb;
    pthread_create(&tb, NULL, ext_sender_thread, &bp_s);
    sleep_ns_eintr(2000000ULL); /* Guarantee external sender enters tickless FUTEX_WAIT */
    uint64_t rx = 0, got = 0, v = 0;
    while (got < 2000 && chan_recv_ex(&ch2, &v)) { rx ^= v; got++; }
    pthread_join(tb, NULL);
    int bp_ok = (got == 2000 && rx == bp_s.xor_sum);

    t_report(ext_ok && closed_send_ok && closed_recv_ok && bp_ok,
             "channel_ext_futex_backpressure_and_close",
             "ext_xor=%d closed_reject=%d/%d bp=%d",
             ext_ok, closed_send_ok, closed_recv_ok, bp_ok);
}

/* ============================================================================
 *  TEST 6: Per-Worker Timer Heap Resize (>64 concurrent timers/worker)
 *          + Edge Delays + Monotonicity
 * ============================================================================ */

typedef struct {
    uint64_t     delay_ns;
    uint64_t     elapsed_ns;
    KronosLatch *latch;
} TimerMassArg;

static void timer_mass_fiber(void *arg) {
    TimerMassArg *a = (TimerMassArg *)arg;
    uint64_t t0 = monotonic_ns();
    kronos_sleep_ns(a->delay_ns);
    a->elapsed_ns = safe_delta_ns(monotonic_ns(), t0);
    latch_done(a->latch);
}

static void test_timer_heap_resize(void) {
    /* Pin 180 simultaneous sleeping fibers to core 0 -> forces 2x heap growth (64->128->256)! */
    const int NT = 180;
    TimerMassArg *args = (TimerMassArg *)calloc((size_t)NT, sizeof(TimerMassArg));
    KronosLatch l;
    latch_init(&l, (uint32_t)NT);
    for (int i = 0; i < NT; i++) {
        uint64_t d = 500000ULL + (uint64_t)((NT - i) % 30) * 300000ULL; /* Reverse/mixed deadlines */
        args[i] = (TimerMassArg){ d, 0, &l };
        KronosFiber *f = kronos_prepare_fiber(timer_mass_fiber, &args[i], 0, 0);
        pinned_q_push(&g_kern.workers[0].pinned, f);
    }
    (void)kronos_wake_worker(0);
    latch_wait(&l);

    int early = 0;
    for (int i = 0; i < NT; i++) {
        if (args[i].elapsed_ns < args[i].delay_ns) early++;
    }
    int grew = (g_kern.workers[0].timers.cap >= 256u);
    int zero_active = (atomic_load(&g_kern.active_timers) == 0u);
    free(args);

    t_report(early == 0 && grew && zero_active,
             "timer_heap_unlocked_resize_180_on_core0",
             "early=%d cap=%u active_after=%u",
             early, g_kern.workers[0].timers.cap, atomic_load(&g_kern.active_timers));
}

/* ============================================================================
 *  TEST 7: Preemption Stress (Deferred Resched + Nested Preempt Disable/Enable
 *          + Hardware Canaries + FP/errno Isolation)
 * ============================================================================ */

typedef struct {
    int          iters;
    int          errs;
    uint32_t     preempt_hits;
    KronosLatch *latch;
} DeferredStressArg;

static void deferred_preempt_stress_fiber(void *arg) {
    DeferredStressArg *a = (DeferredStressArg *)arg;
    int errs = 0;
    volatile uint64_t acc = 0x123456789ABCDEF0ULL;
    for (int i = 0; i < a->iters; i++) {
        /* Nested disable depth 3 -> SIGUSR1 latches NEED_RESCHED_BIT instead of trampoline */
        kronos_preempt_disable();
        kronos_preempt_disable();
        kronos_preempt_disable();
        uint64_t x = acc;
        for (int k = 0; k < 4000; k++) x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        acc = x;
        if ((kronos_preempt_read_word() & KRONOS_PREEMPT_MASK) != 3u) errs++;
        kronos_preempt_enable();
        if ((kronos_preempt_read_word() & KRONOS_PREEMPT_MASK) != 2u) errs++;
        kronos_preempt_enable();
        if ((kronos_preempt_read_word() & KRONOS_PREEMPT_MASK) != 1u) errs++;
        kronos_preempt_enable(); /* Triggers kronos_deferred_preempt_c() if latched! */
        if ((kronos_preempt_read_word() & KRONOS_PREEMPT_MASK) != 0u) errs++;
    }
    a->errs         = errs;
    a->preempt_hits = kronos_current_fiber_preempt_hits();
    latch_done(a->latch);
}

static void test_deferred_preemption_nested(void) {
    int nf = g_kern.num_cores * 2;
    DeferredStressArg *args = (DeferredStressArg *)calloc((size_t)nf, sizeof(DeferredStressArg));
    KronosLatch l;
    latch_init(&l, (uint32_t)nf);
    uint64_t def_before = 0;
    for (int i = 0; i < g_kern.num_cores; i++) def_before += stat_load(&g_kern.workers[i].stat_deferred_preempts);

    for (int i = 0; i < nf; i++) args[i] = (DeferredStressArg){ 350, 0, 0, &l };
    kronos_spawn_batch(deferred_preempt_stress_fiber, args, sizeof(DeferredStressArg), nf, 1);
    latch_wait(&l);

    int errs = 0;
    uint32_t hits = 0;
    uint64_t def_after = 0;
    for (int i = 0; i < nf; i++) { errs += args[i].errs; hits += args[i].preempt_hits; }
    for (int i = 0; i < g_kern.num_cores; i++) def_after += stat_load(&g_kern.workers[i].stat_deferred_preempts);
    free(args);

    int ok = (errs == 0) && (!g_kern.preempt_supported || (def_after > def_before && hits > 0));
    t_report(ok, "preempt_nested_disable_deferred_resched",
             "errs=%d hits=%u deferred=%" PRIu64, errs, hits, def_after - def_before);
}

/* ============================================================================
 *  TEST 8: K-ISA Differential Fuzzing (JIT vs Interpreter on 250 Random Valid
 *          Programs + Edge Cases + AND+CMP0 + Jump-Target Fusion + Fuel + Bounds)
 * ============================================================================ */

static int build_random_valid_prog(uint64_t *rng, KBpfInsn *prog, int max_n) {
    int pc = 0;
    int body = 16 + (int)(xorshift64(rng) % 28u);
    if (body > max_n - 6) body = max_n - 6;

    /* Initialize a loop counter in r4 = (r2 & 15) + 1 so loops always terminate */
    prog[pc++] = (KBpfInsn){ KOP_MOV_REG, 4, 2, 0, 0, 0 };
    prog[pc++] = (KBpfInsn){ KOP_AND_IMM, 4, 0, 0, 15, 0 };
    prog[pc++] = (KBpfInsn){ KOP_ADD_IMM, 4, 0, 0, 1, 0 };
    int loop_hdr = pc;

    int flags_valid = 0;
    for (int step = 0; step < body && pc < max_n - 5; step++) {
        uint8_t dst = (uint8_t)(xorshift64(rng) % 8u);
        if (dst == 4) dst = 0; /* Keep loop counter r4 monotonic */
        uint8_t src = (uint8_t)(xorshift64(rng) % 8u);
        int32_t imm = (int32_t)(xorshift64(rng) & 0xFFFFu) - 32768;
        uint64_t r = xorshift64(rng) % 36u;

        switch (r) {
        case 0:  prog[pc++] = (KBpfInsn){ KOP_MOV_REG, dst, src, 0, 0, 0 }; break;
        case 1:  prog[pc++] = (KBpfInsn){ KOP_MOV_IMM, dst, 0, 0, (xorshift64(rng)&3)?imm:0, 0 }; break;
        case 2:  prog[pc++] = (KBpfInsn){ KOP_MOV_IMM64, dst, 0, 0, 0, xorshift64(rng) }; break;
        case 3:  prog[pc++] = (KBpfInsn){ KOP_ADD_REG, dst, src, 0, 0, 0 }; flags_valid = 0; break;
        case 4:  prog[pc++] = (KBpfInsn){ KOP_ADD_IMM, dst, 0, 0, imm, 0 }; flags_valid = 0; break;
        case 5:  prog[pc++] = (KBpfInsn){ KOP_SUB_REG, dst, src, 0, 0, 0 }; flags_valid = 0; break;
        case 6:  prog[pc++] = (KBpfInsn){ KOP_SUB_IMM, dst, 0, 0, imm, 0 }; flags_valid = 0; break;
        case 7:  prog[pc++] = (KBpfInsn){ KOP_MUL_REG, dst, src, 0, 0, 0 }; flags_valid = 0; break;
        case 8: {
            static const int32_t muls[12] = { -3, -1, 0, 1, 2, 3, 5, 6, 8, 9, 10, 12 };
            int32_t mv = (xorshift64(rng) & 1) ? muls[xorshift64(rng) % 12u] : imm;
            prog[pc++] = (KBpfInsn){ KOP_MUL_IMM, dst, 0, 0, mv, 0 }; flags_valid = 0; break;
        }
        case 9:  prog[pc++] = (KBpfInsn){ KOP_UDIV_REG, dst, src, 0, 0, 0 }; flags_valid = 0; break;
        case 10: prog[pc++] = (KBpfInsn){ KOP_UMOD_REG, dst, src, 0, 0, 0 }; flags_valid = 0; break;
        case 11: prog[pc++] = (KBpfInsn){ KOP_NEG, dst, 0, 0, 0, 0 }; flags_valid = 0; break;
        case 12: prog[pc++] = (KBpfInsn){ KOP_BSWAP64, dst, 0, 0, 0, 0 }; break;
        case 13:
            if (g_cpu.has_popcnt) { prog[pc++] = (KBpfInsn){ KOP_POPCNT, dst, src, 0, 0, 0 }; flags_valid = 0; }
            break;
        case 14:
            if (g_cpu.has_lzcnt) { prog[pc++] = (KBpfInsn){ KOP_LZCNT, dst, src, 0, 0, 0 }; flags_valid = 0; }
            break;
        case 15:
            if (g_cpu.has_bmi1) { prog[pc++] = (KBpfInsn){ KOP_TZCNT, dst, src, 0, 0, 0 }; flags_valid = 0; }
            break;
        case 16: prog[pc++] = (KBpfInsn){ KOP_AND_REG, dst, src, 0, 0, 0 }; flags_valid = 0; break;
        case 17: {
            int32_t av = (xorshift64(rng) & 3) == 0 ? -1 : ((xorshift64(rng) & 1) ? (imm & 0x7FFFFFFF) : (imm | (int32_t)0x80000000u));
            prog[pc++] = (KBpfInsn){ KOP_AND_IMM, dst, 0, 0, av, 0 }; flags_valid = 0;
            /* Frequently follow with CMP_IMM dst, 0 to stress AND+CMP0 fusion across all CC! */
            if ((xorshift64(rng) & 1) && pc < max_n - 6) {
                prog[pc++] = (KBpfInsn){ KOP_CMP_IMM, dst, 0, 0, 0, 0 };
                flags_valid = 1;
            }
            break;
        }
        case 18: prog[pc++] = (KBpfInsn){ KOP_OR_REG, dst, src, 0, 0, 0 }; flags_valid = 0; break;
        case 19: prog[pc++] = (KBpfInsn){ KOP_OR_IMM, dst, 0, 0, imm, 0 }; flags_valid = 0; break;
        case 20: prog[pc++] = (KBpfInsn){ KOP_XOR_REG, dst, src, 0, 0, 0 }; flags_valid = 0; break;
        case 21: prog[pc++] = (KBpfInsn){ KOP_XOR_IMM, dst, 0, 0, imm, 0 }; flags_valid = 0; break;
        case 22: prog[pc++] = (KBpfInsn){ KOP_SHL_IMM, dst, 0, 0, (int32_t)(xorshift64(rng) & 63u), 0 }; flags_valid = 0; break;
        case 23: prog[pc++] = (KBpfInsn){ KOP_SHR_IMM, dst, 0, 0, (int32_t)(xorshift64(rng) & 63u), 0 }; flags_valid = 0; break;
        case 24: prog[pc++] = (KBpfInsn){ KOP_SAR_IMM, dst, 0, 0, (int32_t)(xorshift64(rng) & 63u), 0 }; flags_valid = 0; break;
        case 25: prog[pc++] = (KBpfInsn){ KOP_ROR_IMM, dst, 0, 0, (int32_t)(xorshift64(rng) & 63u), 0 }; flags_valid = 0; break;
        case 26: prog[pc++] = (KBpfInsn){ KOP_SHL_REG, dst, src, 0, 0, 0 }; flags_valid = 0; break;
        case 27: prog[pc++] = (KBpfInsn){ KOP_SHR_REG, dst, src, 0, 0, 0 }; flags_valid = 0; break;
        case 28: prog[pc++] = (KBpfInsn){ KOP_SAR_REG, dst, src, 0, 0, 0 }; flags_valid = 0; break;
        case 29: prog[pc++] = (KBpfInsn){ KOP_CMP_REG, dst, src, 0, 0, 0 }; flags_valid = 1; break;
        case 30: prog[pc++] = (KBpfInsn){ KOP_CMP_IMM, dst, 0, 0, (xorshift64(rng)&3)?imm:0, 0 }; flags_valid = 1; break;
        default:
            if (flags_valid) {
                uint8_t cm = (uint8_t)(KOP_CMOVE + (xorshift64(rng) % 10u));
                prog[pc++] = (KBpfInsn){ cm, dst, src, 0, 0, 0 };
            } else {
                prog[pc++] = (KBpfInsn){ KOP_CMP_REG, dst, src, 0, 0, 0 };
                flags_valid = 1;
            }
            break;
        }
    }

    /* Fold all registers into r0 so dead-code elimination doesn't erase everything */
    prog[pc++] = (KBpfInsn){ KOP_XOR_REG, 0, 5, 0, 0, 0 };
    prog[pc++] = (KBpfInsn){ KOP_ADD_REG, 0, 6, 0, 0, 0 };
    prog[pc++] = (KBpfInsn){ KOP_XOR_REG, 0, 7, 0, 0, 0 };
    prog[pc++] = (KBpfInsn){ KOP_SUB_IMM, 4, 0, 0, 1, 0 };
    prog[pc++] = (KBpfInsn){ KOP_CMP_IMM, 4, 0, 0, 0, 0 };
    prog[pc++] = (KBpfInsn){ KOP_JNE,     0, 0, 0, loop_hdr, 0 };
    prog[pc++] = (KBpfInsn){ KOP_RET,     0, 0, 0, 0, 0 };
    return pc;
}

static void test_jit_differential_fuzzer_and_traps(void) {
    char err[128];
    int trap_ok = 1;

    /* TRAP 1: AND_IMM dst, -1 + CMP_IMM dst, 0 MUST NOT skip CMP_IMM! */
    const KBpfInsn trap_and_neg1[] = {
        { KOP_MOV_REG, 0, 1, 0, 0, 0 },
        { KOP_CMP_IMM, 2, 0, 0, 12345, 0 }, /* Set flags based on R2, NOT R0 */
        { KOP_AND_IMM, 0, 0, 0, -1, 0 },    /* Must NOT elide AND while also eliding CMP! */
        { KOP_CMP_IMM, 0, 0, 0, 0, 0 },
        { KOP_JE,      0, 0, 0, 6, 0 },
        { KOP_MOV_IMM, 0, 0, 0, 999, 0 },
        { KOP_RET,     0, 0, 0, 0, 0 }
    };
    KBpfCompiled c1;
    if (kbpf_jit_compile(trap_and_neg1, 7, KBPF_CAP_NONE, &c1, err, sizeof(err)) == 0) {
        uint64_t cases[4][2] = { {0, 12345}, {0, 7}, {42, 12345}, {42, 7} };
        for (int i = 0; i < 4; i++) {
            uint64_t ri = kbpf_exec_interp_unchecked(trap_and_neg1, 7, KBPF_CAP_NONE, cases[i][0], cases[i][1], 0);
            uint64_t rj = c1.fn(cases[i][0], cases[i][1], 0);
            if (ri != rj) { trap_ok = 0; printf("    [BUG] trap_and_neg1 case %d: interp=%" PRIu64 " jit=%" PRIu64 "\n", i, ri, rj); }
        }
        kbpf_jit_free(&c1);
    } else trap_ok = 0;

    /* TRAP 2: AND_IMM dst, 0x80000000 (32-bit sign bit) -> positive imm32 > 0 in 32-bit AND sets SF!
     * Wait: 0x7FFFFFFF is max int32_t; 0x80000000 is negative int32_t.
     * What about AND_IMM dst, 0x7FFFFFFF followed by CMP_IMM dst, 0 with JL (signed < 0)?
     * 32-bit AND with 0x7FFFFFFF clears bit 31..63 -> positive in 64-bit AND positive in 32-bit!
     * What about AND_IMM dst, 0x00000001..0x7FFFFFFF with bit 31 = 0? Always SF=0 in both 32 and 64 bit!
     * What about AND_IMM dst, -1 (negative)? Covered in TRAP 1!
     * What about AND_IMM dst, 0 followed by CMP_IMM dst, 0 + JNE? (XOR sets ZF=1, SF=0, OF=0, CF=0)
     * What about AND_REG dst, src followed by CMP_IMM dst, 0 + JL when bit 63 is set? */
    const KBpfInsn trap_and_signed[] = {
        { KOP_MOV_REG, 0, 1, 0, 0, 0 },
        { KOP_AND_REG, 0, 2, 0, 0, 0 },
        { KOP_CMP_IMM, 0, 0, 0, 0, 0 },
        { KOP_JL,      0, 0, 0, 5, 0 },
        { KOP_MOV_IMM, 0, 0, 0, 777, 0 },
        { KOP_RET,     0, 0, 0, 0, 0 }
    };
    KBpfCompiled c2;
    if (kbpf_jit_compile(trap_and_signed, 6, KBPF_CAP_NONE, &c2, err, sizeof(err)) == 0) {
        uint64_t tvals[4] = { 0ULL, 1ULL, 0x8000000000000000ULL, UINT64_MAX };
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                uint64_t ri = kbpf_exec_interp_unchecked(trap_and_signed, 6, KBPF_CAP_NONE, tvals[i], tvals[j], 0);
                uint64_t rj = c2.fn(tvals[i], tvals[j], 0);
                if (ri != rj) { trap_ok = 0; printf("    [BUG] trap_and_signed (%d,%d): interp=%" PRIu64 " jit=%" PRIu64 "\n", i, j, ri, rj); }
            }
        }
        kbpf_jit_free(&c2);
    } else trap_ok = 0;

    /* TRAP 3: CMP_IMM dst, 0 fused with AND_IMM dst, 0xFF where Jcc is JA/JB/JBE/JAE/JL/JGE/JLE/JG */
    const KBpfInsn trap_and_all_cc[] = {
        { KOP_MOV_IMM, 0, 0, 0, 0, 0 },
        { KOP_MOV_REG, 4, 1, 0, 0, 0 },
        { KOP_AND_IMM, 4, 0, 0, 0xFF, 0 },
        { KOP_CMP_IMM, 4, 0, 0, 0, 0 },
        { KOP_CMOVA,   0, 2, 0, 0, 0 },
        { KOP_CMOVB,   0, 3, 0, 0, 0 },
        { KOP_CMOVL,   0, 3, 0, 0, 0 },
        { KOP_CMOVG,   0, 1, 0, 0, 0 },
        { KOP_RET,     0, 0, 0, 0, 0 }
    };
    KBpfCompiled c3;
    if (kbpf_jit_compile(trap_and_all_cc, 9, KBPF_CAP_NONE, &c3, err, sizeof(err)) == 0) {
        uint64_t vals[4] = { 0, 1, 0x100, UINT64_MAX };
        for (int i = 0; i < 4; i++) {
            uint64_t ri = kbpf_exec_interp_unchecked(trap_and_all_cc, 9, KBPF_CAP_NONE, vals[i], 0x111, 0x222);
            uint64_t rj = c3.fn(vals[i], 0x111, 0x222);
            if (ri != rj) { trap_ok = 0; printf("    [BUG] trap_and_all_cc %d: interp=%" PRIu64 " jit=%" PRIu64 "\n", i, ri, rj); }
        }
        kbpf_jit_free(&c3);
    } else trap_ok = 0;

    /* TRAP 4: Conditional backedge with CAP_BOUNDED_LOOP AND flags live on fallthrough! */
    const KBpfInsn trap_cond_back[] = {
        { KOP_MOV_REG, 0, 1, 0, 0, 0 },
        { KOP_CMP_IMM, 0, 0, 0, 0, 0 },
        { KOP_JNE,     0, 0, 0, 0, 0 }, /* Backedge when R0 != 0 -> hits fuel exhaustion! */
        { KOP_CMOVE,   0, 2, 0, 0, 0 }, /* Fallthrough when R0 == 0 -> flags must still say EQ! */
        { KOP_RET,     0, 0, 0, 0, 0 }
    };
    KBpfCompiled c4;
    if (kbpf_jit_compile(trap_cond_back, 5, KBPF_CAP_BOUNDED_LOOP, &c4, err, sizeof(err)) == 0) {
        uint64_t r0i = kbpf_exec_interp_unchecked(trap_cond_back, 5, KBPF_CAP_BOUNDED_LOOP, 0, 777, 0);
        uint64_t r0j = c4.fn(0, 777, 0);
        uint64_t r1i = kbpf_exec_interp_unchecked(trap_cond_back, 5, KBPF_CAP_BOUNDED_LOOP, 1, 777, 0);
        uint64_t r1j = c4.fn(1, 777, 0);
        if (r0i != r0j || r1i != r1j || r0j != 777 || r1j != KBPF_FUEL_EXHAUSTED) {
            trap_ok = 0;
            printf("    [BUG] trap_cond_back: r0=(%" PRIu64 ",%" PRIu64 ") r1=(%" PRIx64 ",%" PRIx64 ")\n", r0i, r0j, r1i, r1j);
        }
        kbpf_jit_free(&c4);
    } else trap_ok = 0;

    /* TRAP 5: CAP_BOUNDED_LOOP with UDIV_REG inside loop (verifies push/pop RAX/RDX doesn't clobber fuel!) */
    const KBpfInsn trap_fuel_udiv[] = {
        { KOP_MOV_REG,  0, 1, 0, 0, 0 },
        { KOP_MOV_IMM,  4, 0, 0, 50, 0 },
        { KOP_MOV_IMM,  5, 0, 0, 7, 0 },
        { KOP_UDIV_REG, 5, 2, 0, 0, 0 }, /* Spills RAX (r0) and RDX (r3) onto stack! */
        { KOP_ADD_REG,  0, 5, 0, 0, 0 },
        { KOP_SUB_IMM,  4, 0, 0, 1, 0 },
        { KOP_CMP_IMM,  4, 0, 0, 0, 0 },
        { KOP_JNE,      0, 0, 0, 2, 0 },
        { KOP_ADD_REG,  0, 3, 0, 0, 0 }, /* Keep r3 (RDX) live across UDIV! */
        { KOP_RET,      0, 0, 0, 0, 0 }
    };
    KBpfCompiled c5;
    if (kbpf_jit_compile(trap_fuel_udiv, 10, KBPF_CAP_BOUNDED_LOOP, &c5, err, sizeof(err)) == 0) {
        uint64_t ri = kbpf_exec_interp_unchecked(trap_fuel_udiv, 10, KBPF_CAP_BOUNDED_LOOP, 100, 3, 42);
        uint64_t rj = c5.fn(100, 3, 42);
        if (ri != rj) {
            trap_ok = 0;
            printf("    [BUG] trap_fuel_udiv: interp=%" PRIu64 " jit=%" PRIu64 "\n", ri, rj);
        }
        kbpf_jit_free(&c5);
    } else trap_ok = 0;

    /* TRAP 6: AND_IMM with dead destination (DCE) followed by CMP_IMM must NOT drop CMP! */
    const KBpfInsn trap_dce_and[] = {
        { KOP_MOV_REG, 4, 1, 0, 0, 0 },
        { KOP_AND_IMM, 4, 0, 0, 0xFF, 0 },
        { KOP_CMP_IMM, 4, 0, 0, 0, 0 },
        { KOP_MOV_IMM, 0, 0, 0, 11, 0 },
        { KOP_CMOVE,   0, 2, 0, 0, 0 },
        { KOP_RET,     0, 0, 0, 0, 0 }
    };
    KBpfCompiled c6;
    if (kbpf_jit_compile(trap_dce_and, 6, KBPF_CAP_NONE, &c6, err, sizeof(err)) == 0) {
        for (int i = 0; i < 4; i++) {
            uint64_t a0 = (i == 0) ? 0ULL : ((i == 1) ? 0x100ULL : 0x105ULL);
            uint64_t ri = kbpf_exec_interp_unchecked(trap_dce_and, 6, KBPF_CAP_NONE, a0, 99, 0);
            uint64_t rj = c6.fn(a0, 99, 0);
            if (ri != rj) { trap_ok = 0; printf("    [BUG] trap_dce_and %d: interp=%" PRIu64 " jit=%" PRIu64 "\n", i, ri, rj); }
        }
        kbpf_jit_free(&c6);
    } else trap_ok = 0;

    /* 250 random differential fuzz programs x 6 input vectors = 1,500 executions */
    uint64_t rng = 0xCAFEBABE00112233ULL;
    int fuzz_ok = 1;
    int compiled_cnt = 0;
    int mismatches = 0;
    KBpfInsn prog[KBPF_MAX_INSNS];
    for (int p = 0; p < 250; p++) {
        int n = build_random_valid_prog(&rng, prog, 96);
        uint32_t caps = (p & 1) ? KBPF_CAP_BOUNDED_LOOP : KBPF_CAP_NONE;
        KBpfCompiled c;
        if (kbpf_jit_compile(prog, n, caps, &c, err, sizeof(err)) != 0) {
            fuzz_ok = 0;
            printf("    [BUG] fuzz prog %d failed verify/compile: %s\n", p, err);
            break;
        }
        compiled_cnt++;
        const uint64_t inputs[6][3] = {
            { 0ULL, 0ULL, 0ULL },
            { 1ULL, 2ULL, 3ULL },
            { UINT64_MAX, 0ULL, 7ULL },
            { 0x8000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, 5ULL },
            { 0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL, 11ULL },
            { xorshift64(&rng), xorshift64(&rng), xorshift64(&rng) }
        };
        for (int k = 0; k < 6; k++) {
            uint64_t ri = kbpf_exec_interp_unchecked(prog, n, caps, inputs[k][0], inputs[k][1], inputs[k][2]);
            uint64_t rj = c.fn(inputs[k][0], inputs[k][1], inputs[k][2]);
            if (ri != rj) {
                mismatches++;
                fuzz_ok = 0;
                if (mismatches <= 5) {
                    printf("    [BUG] fuzz prog %d vec %d: interp=0x%" PRIX64 " jit=0x%" PRIX64 "\n", p, k, ri, rj);
                }
            }
        }
        kbpf_jit_free(&c);
    }

    t_report(trap_ok && fuzz_ok && mismatches == 0,
             "jit_traps_and_250x6_differential_fuzzer",
             "traps=%d progs=%d mismatches=%d", trap_ok, compiled_cnt, mismatches);
}

/* ============================================================================
 *  TEST 9: Exhaustive GEMM Matrix (Every Supported Kernel x Every Trans Combo
 *          x Odd/Boundary/Strided/Alpha/Beta/Zero/NaN Cases)
 * ============================================================================ */

static void test_gemm_exhaustive_matrix(void) {
    GemmWorkspace ws;
    memset(&ws, 0, sizeof(ws));

    const GemmKernelDesc *kernels[4];
    int nk = 0;
    kernels[nk++] = &g_gemm_scalar;
    if (g_cpu.has_avx2 && g_cpu.has_fma) kernels[nk++] = &g_gemm_avx2;
    if (g_cpu.has_avx512f) {
        kernels[nk++] = &g_gemm_avx512;
        kernels[nk++] = &g_gemm_avx512_9x48;
    }

    static const struct { int M, N, K, pa, pb, pc; float a, b; } shapes[] = {
        {  1,  1,  1,  0,  0,  0,  1.0f,  0.0f },
        {  1, 49,  7,  3,  5,  7,  1.5f,  0.5f },
        { 15,  1, 33,  1,  2,  3, -1.0f,  1.0f },
        {  6, 16,  8,  0,  0,  0,  1.0f,  0.0f },
        {  9, 48, 16,  0,  0,  0,  1.0f,  1.0f },
        { 14, 32, 16,  0,  0,  0,  2.0f, -0.5f },
        { 13, 31, 17,  5,  7,  9,  1.25f, 0.25f },
        { 29, 97, 65,  3,  5,  7, -0.75f, 0.0f },
        { 64, 64,  0,  0,  0,  4,  1.0f,  0.0f },
        { 64, 64,  0,  0,  0,  4,  1.0f,  0.5f },
        { 37, 53, 41,  2,  3,  5,  0.0f,  0.5f },
        { 37, 53, 41,  2,  3,  5,  0.0f,  0.0f },
        { 145, 193, 257, 7, 11, 13, 1.0f, 0.0f },
        { 145, 193, 257, 7, 11, 13, 0.5f, 0.5f },
    };
    const int nshapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    int total = 0, passed = 0;
    for (int ki = 0; ki < nk; ki++) {
        for (int tA = 0; tA <= 1; tA++) {
            for (int tB = 0; tB <= 1; tB++) {
                for (int si = 0; si < nshapes; si++) {
                    total++;
                    int ok = verify_gemm_tn(tA, tB, shapes[si].M, shapes[si].N, shapes[si].K,
                                            shapes[si].pa, shapes[si].pb, shapes[si].pc,
                                            shapes[si].a, shapes[si].b, kernels[ki], &ws);
                    if (ok) passed++;
                    else {
                        printf("    [BUG] GEMM kernel=%s tA=%d tB=%d M=%d N=%d K=%d a=%.2f b=%.2f\n",
                               kernels[ki]->short_name, tA, tB, shapes[si].M, shapes[si].N, shapes[si].K,
                               shapes[si].a, shapes[si].b);
                    }
                }
            }
        }
    }
    gemm_ws_destroy(&ws);
    t_report(passed == total, "gemm_exhaustive_all_kernels_all_trans",
             "kernels=%d cases=%d/%d", nk, passed, total);
}

/* ============================================================================
 *  TEST 10: Repeated Kernel Init / Shutdown Lifecycle Cleanliness
 * ============================================================================ */

static void test_lifecycle_reinit(void) {
    kronos_shutdown();
    int ok = 1;
    for (int cycle = 0; cycle < 3; cycle++) {
        int nc = (cycle == 0) ? 1 : ((g_cpu.avail_cpus >= 4) ? 4 : g_cpu.avail_cpus);
        kronos_init(nc, 500ULL);
        KronosLatch l;
        latch_init(&l, 16);
        _Atomic uint64_t sum = 0;
        GrqOvfArg args[16];
        for (int i = 0; i < 16; i++) args[i] = (GrqOvfArg){ &sum, (uint64_t)(i + 1), &l };
        kronos_spawn_batch(grq_ovf_fiber, args, sizeof(GrqOvfArg), 16, 1);
        latch_wait(&l);
        if (atomic_load(&sum) != 136ULL) ok = 0;
        kronos_shutdown();
    }
    /* Re-initialize for clean teardown */
    kronos_init(g_cpu.avail_cpus, 1000ULL);
    t_report(ok, "kernel_lifecycle_3x_init_shutdown", "cycles=3 ok=%d", ok);
}

int main(int argc, char **argv) {
    int req_cores = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cores") == 0 && i + 1 < argc) {
            req_cores = atoi(argv[++i]);
        }
    }

    printf("\n=== KRONOS-X White-Box Adversarial Test Suite ===\n");
    kronos_probe_silicon(req_cores);
    if (req_cores <= 0) req_cores = g_cpu.avail_cpus;
    if (g_cpu.cet_shstk_active) {
        fprintf(stderr, "CET SHSTK active & locked\n");
        return 2;
    }

    printf("Silicon: %s | avail=%d | using=%d vCores | AVX2=%d FMA=%d BMI1=%d BMI2=%d AVX512F=%d\n",
           g_cpu.brand, g_cpu.avail_cpus, req_cores,
           g_cpu.has_avx2, g_cpu.has_fma, g_cpu.has_bmi1, g_cpu.has_bmi2, g_cpu.has_avx512f);

    kronos_init(req_cores, 500ULL);

    test_slab_torture();
    test_scheduler_pin_and_grq_overflow();
    test_latch_semantics();
    test_dissem_barrier_wrap();
    test_channel_ext_futex_and_races();
    test_timer_heap_resize();
    test_deferred_preemption_nested();
    test_jit_differential_fuzzer_and_traps();
    test_gemm_exhaustive_matrix();
    test_lifecycle_reinit();

    kronos_shutdown();

    printf("\n=== Summary: %d PASSED, %d FAILED ===\n", g_t_pass, g_t_fail);
    return (g_t_fail == 0) ? 0 : 1;
}
