
import multiprocessing as mp
import math
import time
import statistics

from harness import emit_verify_tag


STUDENT_ID = 190103371
P_LOG = 12
N_STRONG = 1_600_000
N0 = 250_000
TIMED_RUNS = 3


def serial_stage(n):
    result = 0.0
    count = int(n * 0.05)

    for i in range(count):
        x = (i % 1000) * 0.001 + 1.0

        for _ in range(6):
            result += math.sin(x)
            result += math.sqrt(abs(x) + 1.0)
            x += 0.00001

    return result


def parallel_worker(bounds):
    start, end = bounds
    result = 0.0

    for i in range(start, end):
        x = (i % 1000) * 0.001 + 1.0

        for _ in range(35):
            x = x * 1.0000001 + 0.000001
            x += math.sin(x) * 0.000001
            x += math.sqrt(abs(x) + 1.0) * 0.000001

        result += x

    return result


def make_ranges(n, p):
    ranges = []

    for worker in range(p):
        start = worker * n // p
        end = (worker + 1) * n // p
        ranges.append((start, end))

    return ranges


def run_trial(pool, n, p):
    start = time.perf_counter()

    stage_start = time.perf_counter()
    serial_result = serial_stage(n)
    stage_a_time = time.perf_counter() - stage_start

    ranges = make_ranges(n, p)
    results = pool.map(parallel_worker, ranges)

    total_result = serial_result + sum(results)
    total_time = time.perf_counter() - start

    return total_time * 1000, stage_a_time * 1000, total_result


def benchmark(pool, n, p):
    # One warm-up
    run_trial(pool, n, p)

    total_times = []
    stage_times = []

    for _ in range(TIMED_RUNS):
        total, stage_a, _ = run_trial(pool, n, p)
        total_times.append(total)
        stage_times.append(stage_a)

    return (
        statistics.median(total_times),
        statistics.median(stage_times)
    )


def main():
    mp.freeze_support()

    print("=" * 80)
    print("PHASE 2: STRONG AND WEAK SCALING")
    print("=" * 80)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical cores: {mp.cpu_count()}")
    print(f"Strong workload: {N_STRONG:,}")
    print(f"Weak workload per process: {N0:,}")

    processes = list(range(1, P_LOG + 1))

    # STRONG SCALING

    print("\nEXPERIMENT 2A: STRONG SCALING")

    strong = {}

    for p in processes:
        print(f"Testing strong scaling p={p}...")

        with mp.Pool(processes=p) as pool:
            total, stage_a = benchmark(pool, N_STRONG, p)

        strong[p] = (total, stage_a)

    t1_strong = strong[1][0]
    serial_fraction = strong[1][1] / t1_strong

    print("\nSTRONG SCALING RESULTS")
    print("-" * 75)
    print(f"{'p':<8}{'Time ms':<18}{'Speedup':<18}{'Efficiency':<18}")

    for p in processes:
        total = strong[p][0]
        speedup = t1_strong / total
        efficiency = speedup / p

        print(
            f"{p:<8}{total:<18.4f}"
            f"{speedup:<18.4f}{efficiency:<18.4f}"
        )

    theoretical_max = 1 / serial_fraction

    amdahl_speedup = 1 / (
            serial_fraction +
            (1 - serial_fraction) / P_LOG
    )

    observed_speedup = t1_strong / strong[P_LOG][0]

    print("\nAMDAHL ANALYSIS")
    print(f"Serial fraction: {serial_fraction:.6f}")
    print(f"Theoretical maximum speedup: {theoretical_max:.4f}")
    print(f"Theoretical speedup at p=12: {amdahl_speedup:.4f}")
    print(f"Observed speedup at p=12: {observed_speedup:.4f}")

    # WEAK SCALING

    print("\nEXPERIMENT 2B: WEAK SCALING")

    weak = {}

    for p in processes:
        n = p * N0
        print(f"Testing weak scaling p={p}, N={n:,}...")

        with mp.Pool(processes=p) as pool:
            total, stage_a = benchmark(pool, n, p)

        weak[p] = (total, stage_a)

    t1_weak = weak[1][0]

    print("\nWEAK SCALING RESULTS")
    print("-" * 90)
    print(
        f"{'p':<8}{'N':<15}{'Time ms':<16}"
        f"{'Efficiency':<16}{'Gustafson':<16}"
    )

    gustafson_results = {}

    for p in processes:
        total, stage_a = weak[p]
        n = p * N0

        efficiency = t1_weak / total
        alpha = stage_a / total

        gustafson = p - alpha * (p - 1)
        gustafson_results[p] = gustafson

        print(
            f"{p:<8}{n:<15}{total:<16.4f}"
            f"{efficiency:<16.4f}{gustafson:<16.4f}"
        )

    print("\nPHASE 2 VERIFICATION")

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            observed_speedup,
            gustafson_results[P_LOG]
        )
    )

    print("\nPHASE 2 COMPLETED")


if __name__ == "__main__":
    main()
