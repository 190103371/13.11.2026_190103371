import multiprocessing as mp
import time

from harness import emit_verify_tag


STUDENT_ID = 190103371
ITERATIONS = 1_000_000


def main():
    # Number of logical CPU threads
    p_log = mp.cpu_count()

    # Measure timer call latency
    start = time.perf_counter_ns()

    for _ in range(ITERATIONS):
        time.perf_counter_ns()

    elapsed = time.perf_counter_ns() - start
    avg_latency = elapsed / ITERATIONS

    print("PHASE 0: HARDWARE TELEMETRY")
    print("---------------------------")
    print(f"Logical CPU threads: {p_log}")
    print(f"Timer average latency: {avg_latency:.4f} ns")

    # Generate verification tag
    result = emit_verify_tag(
        "PHASE0",
        STUDENT_ID,
        avg_latency,
        float(p_log)
    )

    print(result)


if __name__ == "__main__":
    main()