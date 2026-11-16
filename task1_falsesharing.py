
import ctypes
import multiprocessing as mp
import time

from multiprocessing import shared_memory
from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 190103371
M = 24_000_000
PHYSICAL_CORES = 8

STRIDES = [1, 2, 4, 8, 16, 32]
PROCESS_COUNTS = [1, 2, 4, 8, 12]


def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005

    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc

    del arr
    shm.close()


def run_false_sharing(p, stride):

    shm = shared_memory.SharedMemory(create=True, size=4096)

    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

        for i in range(512):
            arr[i] = 0

        del arr

        processes = []

        iterations = M // p

        for t in range(p):
            slot_idx = t * stride

            process = mp.Process(
                target=fs_worker,
                args=(shm.name, slot_idx, iterations)
            )

            processes.append(process)

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {process.exitcode}"
                )

    finally:
        shm.close()
        shm.unlink()


def benchmark_configuration(p, stride):

    return run_benchmark(
        lambda: run_false_sharing(p, stride),
        warmup_runs=2,
        timed_runs=7
    )


def print_stride_results(results):

    print("\nTABLE 1A: STRIDE SENSITIVITY")
    print("-" * 75)

    print(
        f"{'Stride':<10}"
        f"{'Bytes':<12}"
        f"{'Median ms':<15}"
        f"{'IQR ms':<15}"
        f"{'Mops/s':<15}"
    )

    for stride, stats in results.items():

        byte_distance = stride * 8
        throughput = M / (stats.median_ms * 1000)

        print(
            f"{stride:<10}"
            f"{byte_distance:<12}"
            f"{stats.median_ms:<15.4f}"
            f"{stats.iqr_ms:<15.4f}"
            f"{throughput:<15.4f}"
        )


def print_scaling_results(results, stride):

    print(f"\nTABLE 1B: SCALING, STRIDE = {stride}")
    print("-" * 85)

    t1 = results[1].median_ms

    print(
        f"{'Processes':<12}"
        f"{'Median ms':<15}"
        f"{'Speedup':<15}"
        f"{'Efficiency':<15}"
        f"{'Karp-Flatt':<15}"
    )

    for p in results:

        tp = results[p].median_ms
        speedup = t1 / tp
        efficiency = speedup / p

        if p >= 2:
            karp_flatt = (
                    (1 / speedup - 1 / p) /
                    (1 - 1 / p)
            )
            kf_text = f"{karp_flatt:.6f}"
        else:
            kf_text = "N/A"

        print(
            f"{p:<12}"
            f"{tp:<15.4f}"
            f"{speedup:<15.4f}"
            f"{efficiency:<15.4f}"
            f"{kf_text:<15}"
        )


def main():

    mp.freeze_support()

    print("=" * 85)
    print("PHASE 1: FALSE SHARING & CACHE-LINE CONTENTION")
    print("=" * 85)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores: {PHYSICAL_CORES}")
    print(f"Logical cores: {mp.cpu_count()}")
    print(f"Total operations: {M:,}")

    # EXPERIMENT 1: STRIDE SWEEP

    stride_results = {}

    print("\nStarting stride sensitivity experiment...")

    for stride in STRIDES:

        print(f"Testing stride={stride} ({stride * 8} bytes)...")

        stats = benchmark_configuration(
            PHYSICAL_CORES,
            stride
        )

        stride_results[stride] = stats

    print_stride_results(stride_results)

    # EXPERIMENT 2: PROCESS SCALING

    all_scaling_results = {}

    for stride in [1, 16]:

        print(
            f"\nStarting process scaling experiment: "
            f"stride={stride}"
        )

        scaling_results = {}

        for p in PROCESS_COUNTS:

            print(f"Testing processes={p}...")

            stats = benchmark_configuration(p, stride)

            scaling_results[p] = stats

        all_scaling_results[stride] = scaling_results

        print_scaling_results(
            scaling_results,
            stride
        )

    # VERIFICATION HASH

    median_stride1 = stride_results[1].median_ms
    median_stride16 = stride_results[16].median_ms

    print("\n" + "=" * 85)
    print("PHASE 1 VERIFICATION")
    print("=" * 85)

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            median_stride1,
            median_stride16
        )
    )

    print("\nPHASE 1 COMPLETED")


if __name__ == "__main__":
    main()
