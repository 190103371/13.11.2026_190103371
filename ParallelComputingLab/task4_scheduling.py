
import math
import multiprocessing as mp
import time

from harness import emit_verify_tag


STUDENT_ID = 190103371

N = 8000
P_PHYS = 8
CHUNK_SIZES = [1, 8, 64, 512]


def item_work(i):
    """Triangular workload from the lab specification."""
    k_count = 1 + int((i / 40) ** 2)

    value = 0.0

    for j in range(k_count):
        x = i * 0.001 + j * 0.0001 + 1.0

        value += math.sin(x)
        value += math.cos(x)
        value += math.sqrt(abs(x) + 1.0)

    return value


def worker_static(worker_id, p, strategy, times, results):
    start_time = time.perf_counter()

    if strategy == "block":
        start = worker_id * N // p
        end = (worker_id + 1) * N // p
        indices = range(start, end)

    elif strategy == "cyclic":
        indices = range(worker_id, N, p)

    else:
        raise ValueError("Unknown static strategy")

    total = 0.0

    for i in indices:
        total += item_work(i)

    results[worker_id] = total
    times[worker_id] = time.perf_counter() - start_time


def dynamic_worker(
        worker_id,
        p,
        chunk_size,
        counter,
        lock,
        times,
        results
):
    start_time = time.perf_counter()
    total = 0.0

    while True:
        # Claim the next chunk safely.
        with lock:
            start = counter.value
            counter.value += chunk_size

        if start >= N:
            break

        end = min(start + chunk_size, N)

        for i in range(start, end):
            total += item_work(i)

    results[worker_id] = total
    times[worker_id] = time.perf_counter() - start_time


def run_static(strategy):
    times = mp.Array("d", P_PHYS)
    results = mp.Array("d", P_PHYS)

    processes = []

    start_wall = time.perf_counter()

    for worker in range(P_PHYS):
        process = mp.Process(
            target=worker_static,
            args=(worker, P_PHYS, strategy, times, results)
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

        if process.exitcode != 0:
            raise RuntimeError(
                f"Worker failed with exit code {process.exitcode}"
            )

    wall_time = time.perf_counter() - start_wall

    worker_times = list(times)
    average_time = sum(worker_times) / P_PHYS

    imbalance = (
        max(worker_times) / average_time
        if average_time > 0
        else 0.0
    )

    return wall_time, imbalance, worker_times


def run_dynamic(chunk_size):
    times = mp.Array("d", P_PHYS)
    results = mp.Array("d", P_PHYS)

    counter = mp.Value("i", 0)
    lock = mp.Lock()

    processes = []

    start_wall = time.perf_counter()

    for worker in range(P_PHYS):
        process = mp.Process(
            target=dynamic_worker,
            args=(
                worker,
                P_PHYS,
                chunk_size,
                counter,
                lock,
                times,
                results
            )
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

        if process.exitcode != 0:
            raise RuntimeError(
                f"Worker failed with exit code {process.exitcode}"
            )

    wall_time = time.perf_counter() - start_wall

    worker_times = list(times)
    average_time = sum(worker_times) / P_PHYS

    imbalance = (
        max(worker_times) / average_time
        if average_time > 0
        else 0.0
    )

    return wall_time, imbalance, worker_times


def print_result(name, result):
    wall, imbalance, worker_times = result

    print(f"\n{name}")
    print(f"Wall time: {wall:.4f} seconds")
    print(f"Load imbalance lambda: {imbalance:.4f}")

    for worker, elapsed in enumerate(worker_times):
        print(f"Worker {worker}: {elapsed:.4f} seconds")


def main():
    print("PARALLEL COMPUTING LAB - PHASE 4")
    print("--------------------------------")
    print(f"Items: {N}")
    print(f"Physical workers: {P_PHYS}")

    print("\nSTRATEGY 1: CONTIGUOUS BLOCK STATIC")
    block_result = run_static("block")
    print_result("Block Static", block_result)

    print("\nSTRATEGY 2: CYCLIC ROUND-ROBIN STATIC")
    cyclic_result = run_static("cyclic")
    print_result("Cyclic Static", cyclic_result)

    print("\nSTRATEGY 3: DYNAMIC CHUNK QUEUE")

    dynamic_results = {}

    for chunk_size in CHUNK_SIZES:
        print(f"\nTesting chunk size C={chunk_size}")

        result = run_dynamic(chunk_size)
        dynamic_results[chunk_size] = result

        print_result(f"Dynamic C={chunk_size}", result)

    # Choose the dynamic chunk size with the lowest wall time.
    best_chunk = min(
        dynamic_results,
        key=lambda c: dynamic_results[c][0]
    )

    best_dynamic = dynamic_results[best_chunk]

    print("\nSUMMARY")
    print("=" * 60)
    print(f"Block Static time: {block_result[0]:.4f} s")
    print(f"Cyclic Static time: {cyclic_result[0]:.4f} s")

    for chunk_size, result in dynamic_results.items():
        print(
            f"Dynamic C={chunk_size}: "
            f"{result[0]:.4f} s, "
            f"lambda={result[1]:.4f}"
        )

    print(f"\nBest dynamic chunk size: {best_chunk}")

    print("\nPHASE 4 VERIFICATION")
    print("--------------------")

    print(
        emit_verify_tag(
            "PHASE4",
            STUDENT_ID,
            block_result[0] * 1000,
            best_dynamic[0] * 1000
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
