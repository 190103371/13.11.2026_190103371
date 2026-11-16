
import math
import multiprocessing as mp
import time
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 190103371
P_PHYS = 8

BUFFER_SIZE = 64 * 1024 * 1024
N = 256
BLOCK_SIZES = [8, 16, 32, 64, 96, 128]


# --------------------------------------------------
# PART A: DRAM BANDWIDTH
# --------------------------------------------------

def copy_worker(args):
    name_a, name_b, start, end = args

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)

    a = shm_a.buf
    b = shm_b.buf

    a[start:end] = b[start:end]

    del a
    del b

    shm_a.close()
    shm_b.close()


def run_memory_copy():
    shm_a = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )
    shm_b = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )

    try:
        shm_b.buf[:] = b"\x01" * BUFFER_SIZE

        chunk = BUFFER_SIZE // P_PHYS
        tasks = []

        for worker in range(P_PHYS):
            start = worker * chunk
            end = (
                BUFFER_SIZE
                if worker == P_PHYS - 1
                else start + chunk
            )

            tasks.append(
                (shm_a.name, shm_b.name, start, end)
            )

        with mp.Pool(P_PHYS) as pool:
            pool.map(copy_worker, tasks)

    finally:
        shm_a.close()
        shm_b.close()
        shm_a.unlink()
        shm_b.unlink()


def measure_bandwidth():
    print("\nPHASE 3A: DRAM BANDWIDTH")
    print("=" * 60)

    stats = run_benchmark(
        run_memory_copy,
        warmup=1,
        runs=5
    )

    bandwidth = (
            2 * BUFFER_SIZE /
            (stats.median_ms * 1_000_000)
    )

    print(f"Median copy time: {stats.median_ms:.4f} ms")
    print(f"Estimated bandwidth: {bandwidth:.4f} GB/s")

    return bandwidth


# --------------------------------------------------
# PART B: COMPUTE CEILING
# --------------------------------------------------

def compute_worker(iterations):
    a0 = 1.001
    a1 = 1.002
    a2 = 1.003
    a3 = 1.004
    a4 = 1.005
    a5 = 1.006
    a6 = 1.007
    a7 = 1.008
    a8 = 1.009
    a9 = 1.010
    a10 = 1.011
    a11 = 1.012
    a12 = 1.013
    a13 = 1.014
    a14 = 1.015
    a15 = 1.016

    for _ in range(iterations):
        a0 = a0 * 1.0000001 + 0.000001
        a1 = a1 * 1.0000001 + 0.000001
        a2 = a2 * 1.0000001 + 0.000001
        a3 = a3 * 1.0000001 + 0.000001
        a4 = a4 * 1.0000001 + 0.000001
        a5 = a5 * 1.0000001 + 0.000001
        a6 = a6 * 1.0000001 + 0.000001
        a7 = a7 * 1.0000001 + 0.000001
        a8 = a8 * 1.0000001 + 0.000001
        a9 = a9 * 1.0000001 + 0.000001
        a10 = a10 * 1.0000001 + 0.000001
        a11 = a11 * 1.0000001 + 0.000001
        a12 = a12 * 1.0000001 + 0.000001
        a13 = a13 * 1.0000001 + 0.000001
        a14 = a14 * 1.0000001 + 0.000001
        a15 = a15 * 1.0000001 + 0.000001

    return (
            a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
            + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15
    )


def run_compute_ceiling():
    iterations = 1_875_000

    with mp.Pool(P_PHYS) as pool:
        pool.map(compute_worker, [iterations] * P_PHYS)


def measure_compute():
    print("\nPHASE 3A: COMPUTE CEILING")
    print("=" * 60)

    stats = run_benchmark(
        run_compute_ceiling,
        warmup=1,
        runs=5
    )

    total_flops = 60_000_000 * P_PHYS
    gflops = total_flops / (stats.median_ms * 1_000_000)

    print(f"Median compute time: {stats.median_ms:.4f} ms")
    print(f"Estimated compute rate: {gflops:.4f} GFLOP/s")

    return gflops


# --------------------------------------------------
# PART C: GEMM
# --------------------------------------------------

def create_matrices():
    a = [
        ((i * 7 + 3) % 100) / 100.0
        for i in range(N * N)
    ]

    b = [
        ((i * 11 + 5) % 100) / 100.0
        for i in range(N * N)
    ]

    return a, b


def parallel_rows_worker(args):
    a, b, start_row, end_row, variant, bs = args

    row_count = end_row - start_row
    output = [0.0] * (row_count * N)

    if variant == "naive":
        for i in range(start_row, end_row):
            local_row = (i - start_row) * N

            for j in range(N):
                total = 0.0

                for k in range(N):
                    total += a[i * N + k] * b[k * N + j]

                output[local_row + j] = total

    elif variant == "reordered":
        for i in range(start_row, end_row):
            local_row = (i - start_row) * N

            for k in range(N):
                aik = a[i * N + k]
                b_row = k * N

                for j in range(N):
                    output[local_row + j] += aik * b[b_row + j]

    elif variant == "blocked":
        for ii in range(start_row, end_row, bs):
            i_end = min(ii + bs, end_row)

            for kk in range(0, N, bs):
                k_end = min(kk + bs, N)

                for jj in range(0, N, bs):
                    j_end = min(jj + bs, N)

                    for i in range(ii, i_end):
                        local_row = (i - start_row) * N

                        for k in range(kk, k_end):
                            aik = a[i * N + k]
                            b_row = k * N

                            for j in range(jj, j_end):
                                output[local_row + j] += (
                                        aik * b[b_row + j]
                                )

    return output


def parallel_gemm(a, b, variant, bs=32):
    rows_per_worker = (N + P_PHYS - 1) // P_PHYS
    tasks = []

    for worker in range(P_PHYS):
        start = worker * rows_per_worker
        end = min(N, start + rows_per_worker)

        if start < N:
            tasks.append(
                (a, b, start, end, variant, bs)
            )

    with mp.Pool(P_PHYS) as pool:
        parts = pool.map(parallel_rows_worker, tasks)

    result = []

    for part in parts:
        result.extend(part)

    return result


def measure_gemm(a, b, variant, bs=32):
    stats = run_benchmark(
        lambda: parallel_gemm(a, b, variant, bs),
        warmup=1,
        runs=5
    )

    gflops = (2 * N ** 3) / (
            stats.median_ms * 1_000_000
    )

    return stats, gflops


def gemm_tests():
    print("\nPHASE 3B: GEMM TESTS")
    print("=" * 70)

    a, b = create_matrices()

    stats1, gflops1 = measure_gemm(a, b, "naive")

    print(
        f"Variant 1 (naive): "
        f"{stats1.median_ms:.4f} ms, "
        f"{gflops1:.4f} GFLOP/s"
    )

    stats2, gflops2 = measure_gemm(a, b, "reordered")

    print(
        f"Variant 2 (reordered): "
        f"{stats2.median_ms:.4f} ms, "
        f"{gflops2:.4f} GFLOP/s"
    )

    reference = parallel_gemm(a, b, "reordered")

    blocked_results = {}

    for bs in BLOCK_SIZES:
        print(f"\nTesting block size {bs}...")

        stats, gflops = measure_gemm(
            a, b, "blocked", bs
        )

        blocked_results[bs] = (stats, gflops)

        print(
            f"Median: {stats.median_ms:.4f} ms | "
            f"GFLOP/s: {gflops:.4f}"
        )

    best_bs = max(
        blocked_results,
        key=lambda size: blocked_results[size][1]
    )

    blocked_matrix = parallel_gemm(
        a, b, "blocked", best_bs
    )

    max_error = max(
        abs(x - y)
        for x, y in zip(reference, blocked_matrix)
    )

    print(f"\nBest block size: {best_bs}")
    print(f"Maximum numerical error: {max_error:.12e}")

    if max_error >= 1e-9:
        print("WARNING: Numerical verification failed.")

    best_gflops = blocked_results[best_bs][1]

    return gflops1, best_gflops


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("PARALLEL COMPUTING LAB - PHASE 3")
    print("--------------------------------")

    bandwidth = measure_bandwidth()
    compute_rate = measure_compute()

    ridge_point = compute_rate / bandwidth

    print(f"\nRidge point: {ridge_point:.4f} FLOP/byte")

    gflops1, gflops3 = gemm_tests()

    print("\nPHASE 3 VERIFICATION")
    print("--------------------")

    print(
        emit_verify_tag(
            "PHASE3",
            STUDENT_ID,
            gflops1,
            gflops3
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
