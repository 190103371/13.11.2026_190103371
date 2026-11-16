
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 190103371

M = 16_000_000
P_PHYS = 8
P_LOG = mp.cpu_count()

STRIDES = [1, 2, 4, 8, 16, 32]


def fs_worker(shm_name: str, slot_idx: int, iters: int):
    """
    Worker performs read-modify-write operations
    on its assigned uint64 slot in shared memory.
    """
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


def run_false_sharing(p: int, stride: int) -> None:
    """
    Create shared memory and run p worker processes.
    """
    shm = shared_memory.SharedMemory(create=True, size=4096)

    processes = []

    try:
        # Initialize shared memory slots
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

        for i in range(512):
            arr[i] = 0

        del arr

        iters = M // p

        for t in range(p):
            slot_idx = t * stride

            process = mp.Process(
                target=fs_worker,
                args=(shm.name, slot_idx, iters)
            )

            processes.append(process)

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker process failed: exit code {process.exitcode}"
                )

    finally:
        shm.close()
        shm.unlink()


def print_stride_sweep():
    print("\nPHASE 1A: FALSE SHARING STRIDE SWEEP")
    print("=" * 65)

    results = {}

    for stride in STRIDES:
        print(f"\nTesting stride={stride} ({stride * 8} bytes)...")

        stats = run_benchmark(
            lambda: run_false_sharing(P_PHYS, stride),
            warmup=2,
            runs=7
        )

        throughput = M / (stats.median_ms * 1000)

        results[stride] = stats

        print(f"Median:     {stats.median_ms:.4f} ms")
        print(f"IQR:        {stats.iqr_ms:.4f} ms")
        print(f"Throughput: {throughput:.4f} Mops/s")

    print("\nSTRIDE SUMMARY")
    print("-" * 65)
    print(
        f"{'Stride':<10}"
        f"{'Bytes':<10}"
        f"{'Median ms':<16}"
        f"{'IQR ms':<14}"
        f"{'Mops/s':<12}"
    )

    for stride, stats in results.items():
        throughput = M / (stats.median_ms * 1000)

        print(
            f"{stride:<10}"
            f"{stride * 8:<10}"
            f"{stats.median_ms:<16.4f}"
            f"{stats.iqr_ms:<14.4f}"
            f"{throughput:<12.4f}"
        )

    return results


def process_scaling(stride: int):
    print(f"\nPROCESS SCALING: STRIDE={stride}")
    print("=" * 75)

    # Include physical cores and all logical threads
    process_counts = [1, 2, 4, 8, P_LOG]
    process_counts = sorted(set(p for p in process_counts if p <= P_LOG))

    results = {}

    for p in process_counts:
        print(f"\nTesting p={p} processes...")

        stats = run_benchmark(
            lambda: run_false_sharing(p, stride),
            warmup=2,
            runs=7
        )

        results[p] = stats

        print(
            f"Median: {stats.median_ms:.4f} ms | "
            f"IQR: {stats.iqr_ms:.4f} ms"
        )

    t1 = results[1].median_ms

    print("\nSCALING RESULTS")
    print("-" * 75)
    print(
        f"{'p':<8}"
        f"{'Time ms':<16}"
        f"{'Speedup':<14}"
        f"{'Efficiency':<14}"
        f"{'Karp-Flatt':<14}"
    )

    for p in process_counts:
        tp = results[p].median_ms
        speedup = t1 / tp
        efficiency = speedup / p

        if p >= 2:
            karp_flatt = (
                    (1 / speedup - 1 / p) /
                    (1 - 1 / p)
            )
            karp_text = f"{karp_flatt:.6f}"
        else:
            karp_text = "N/A"

        print(
            f"{p:<8}"
            f"{tp:<16.4f}"
            f"{speedup:<14.4f}"
            f"{efficiency:<14.4f}"
            f"{karp_text:<14}"
        )

    return results


def main():
    print("PARALLEL COMPUTING LAB - PHASE 1")
    print("--------------------------------")
    print(f"Physical cores: {P_PHYS}")
    print(f"Logical threads: {P_LOG}")
    print(f"Total operations: {M:,}")

    # Part A: stride sweep at physical core count
    stride_results = print_stride_sweep()

    # Part B: process scaling for stride 1
    scaling_stride_1 = process_scaling(stride=1)

    # Part C: process scaling for stride 16
    scaling_stride_16 = process_scaling(stride=16)

    # Get measurements at p = P_PHYS
    # These are the values required for the verification tag.
    time_stride_1 = scaling_stride_1[P_PHYS].median_ms
    time_stride_16 = scaling_stride_16[P_PHYS].median_ms

    print("\nPHASE 1 VERIFICATION")
    print("--------------------")

    tag = emit_verify_tag(
        "PHASE1",
        STUDENT_ID,
        time_stride_1,
        time_stride_16
    )

    print(tag)


if __name__ == "__main__":
    mp.freeze_support()
    main()
