
import math
import multiprocessing as mp

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 190103371
P_LOG = mp.cpu_count()

N_STRONG = 1_200_000
N0_WEAK = 200_000

REPEATS = 7
WARMUP = 2


def stage_a(n):
    """Serial Stage A: first 5% of elements."""
    total = 0.0
    count = max(1, int(0.05 * n))

    for i in range(count):
        x = i * 0.00001 + 1.0

        for j in range(5):
            total += math.sin(x + j) ** 2
            total += math.sqrt(abs(x + j) + 1.0)

    return total


def compute_chunk(args):
    """Parallel Stage B: 30 math operations per element."""
    start, end = args
    total = 0.0

    for i in range(start, end):
        x = i * 0.00001 + 1.0

        for j in range(15):
            x = x * 1.0000001 + math.sin(x)
            x = x * 0.9999999 + math.cos(x)

        total += x

    return total


def make_chunks(n, p):
    chunk_size = (n + p - 1) // p
    chunks = []

    for worker in range(p):
        start = worker * chunk_size
        end = min(n, start + chunk_size)

        if start < n:
            chunks.append((start, end))

    return chunks


def run_pipeline(pool, n, p):
    # Stage A executes in the parent process
    serial_result = stage_a(n)

    # Stage B executes in the persistent process pool
    chunks = make_chunks(n, p)
    results = pool.map(compute_chunk, chunks)

    return serial_result + sum(results)


def benchmark_pipeline(n, p):
    with mp.Pool(processes=p) as pool:

        stats = run_benchmark(
            lambda: run_pipeline(pool, n, p),
            warmup=WARMUP,
            runs=REPEATS
        )

    return stats


def strong_scaling():
    print("\nPHASE 2A: STRONG SCALING")
    print("=" * 75)

    results = {}

    for p in range(1, P_LOG + 1):
        print(f"Testing strong scaling with p={p}...")

        stats = benchmark_pipeline(N_STRONG, p)
        results[p] = stats

        print(f"Median: {stats.median_ms:.4f} ms")

    t1 = results[1].median_ms

    print("\nSTRONG SCALING RESULTS")
    print("-" * 75)
    print(f"{'p':<8}{'Time ms':<16}{'Speedup':<16}{'Efficiency':<16}")

    for p in range(1, P_LOG + 1):
        tp = results[p].median_ms
        speedup = t1 / tp
        efficiency = speedup / p

        print(
            f"{p:<8}{tp:<16.4f}"
            f"{speedup:<16.4f}{efficiency:<16.4f}"
        )

    # Estimate serial fraction from Stage A relative to total p=1 time
    stage_a_time = run_benchmark(
        lambda: stage_a(N_STRONG),
        warmup=WARMUP,
        runs=REPEATS
    ).median_ms

    serial_fraction = stage_a_time / t1
    serial_fraction = min(max(serial_fraction, 0.0), 1.0)

    amdahl_cap = (
        float("inf")
        if serial_fraction == 0
        else 1.0 / serial_fraction
    )

    print(f"\nStage A time: {stage_a_time:.4f} ms")
    print(f"Estimated serial fraction: {serial_fraction:.6f}")
    print(f"Amdahl theoretical cap: {amdahl_cap:.4f}")

    return results


def weak_scaling():
    print("\nPHASE 2B: WEAK SCALING")
    print("=" * 75)

    results = {}
    stage_a_times = {}

    for p in range(1, P_LOG + 1):
        n = p * N0_WEAK

        print(f"Testing weak scaling: p={p}, N={n:,}")

        stats = benchmark_pipeline(n, p)
        results[p] = stats

        stage_stats = run_benchmark(
            lambda: stage_a(n),
            warmup=WARMUP,
            runs=REPEATS
        )

        stage_a_times[p] = stage_stats.median_ms

        print(f"Median: {stats.median_ms:.4f} ms")

    t1 = results[1].median_ms

    print("\nWEAK SCALING RESULTS")
    print("-" * 85)
    print(
        f"{'p':<8}{'N':<14}{'Time ms':<16}"
        f"{'Efficiency':<16}{'Gustafson':<16}"
    )

    gustafson_values = {}

    for p in range(1, P_LOG + 1):
        tp = results[p].median_ms
        efficiency = t1 / tp

        alpha = stage_a_times[p] / tp
        gustafson = p - alpha * (p - 1)

        gustafson_values[p] = gustafson

        print(
            f"{p:<8}{p * N0_WEAK:<14}"
            f"{tp:<16.4f}{efficiency:<16.4f}"
            f"{gustafson:<16.4f}"
        )

    return results, gustafson_values


def main():
    print("PARALLEL COMPUTING LAB - PHASE 2")
    print("--------------------------------")
    print(f"Logical CPU threads: {P_LOG}")

    strong = strong_scaling()
    weak, gustafson = weak_scaling()

    # The assignment asks for strong speedup at P_LOG
    t1_strong = strong[1].median_ms
    tlog_strong = strong[P_LOG].median_ms
    strong_speedup_log = t1_strong / tlog_strong

    gustafson_log = gustafson[P_LOG]

    print("\nPHASE 2 VERIFICATION")
    print("--------------------")

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            strong_speedup_log,
            gustafson_log
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
