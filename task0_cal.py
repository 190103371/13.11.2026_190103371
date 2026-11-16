
import multiprocessing as mp
import time

from harness import emit_verify_tag


def main():
    student_id = 190103371
    reads = 2_000_000

    logical_cores = mp.cpu_count()

    start = time.perf_counter_ns()

    for _ in range(reads):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_time_ns = end - start
    total_time_ms = total_time_ns / 1e6
    average_latency = total_time_ns / reads

    print("=" * 60)
    print("PHASE 0: HARDWARE CALIBRATION")
    print("=" * 60)

    print(f"Student ID: {student_id}")
    print(f"Logical CPU cores: {logical_cores}")
    print(f"Clock reads: {reads}")
    print(f"Total time: {total_time_ms:.4f} ms")
    print(f"Average clock latency: {average_latency:.4f} ns")

    print("-" * 60)

    # The PDF does not define the two Phase 0 hash metrics.
    # Convention: primary = average latency (ns),
    # secondary = total elapsed time (ms).
    print(
        emit_verify_tag(
            "PHASE0",
            student_id,
            average_latency,
            total_time_ms
        )
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
