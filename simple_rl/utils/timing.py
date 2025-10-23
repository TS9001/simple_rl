import time
from collections import OrderedDict, defaultdict
from typing import Dict, Any, Optional


class TimingManager:
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timings = {}
        self.timing_enabled = True

    def start_timer(self, operation_name: str) -> None:
        if self.timing_enabled:
            self.current_timings[operation_name] = time.perf_counter()

    def end_timer(self, operation_name: str) -> float:
        if not self.timing_enabled or operation_name not in self.current_timings:
            return 0.0

        elapsed = time.perf_counter() - self.current_timings[operation_name]
        self.timings[operation_name].append(elapsed)
        del self.current_timings[operation_name]
        return elapsed

    def reset_timings(self) -> None:
        self.timings.clear()
        self.current_timings.clear()

    def print_timing_summary(self, title: str = "Operation Timings") -> None:
        if not self.timings:
            return

        print(f"\n[TIMING] {title}")
        print("-" * 120)

        timing_stats = OrderedDict()
        total_time = 0
        total_calls = 0

        for op_name, times in self.timings.items():
            if times:
                mean_time = sum(times) / len(times)
                total_time += sum(times)
                total_calls += len(times)
                timing_stats[op_name] = {
                    "mean": mean_time,
                    "median": sorted(times)[len(times) // 2] if times else 0,
                    "min": min(times),
                    "max": max(times),
                    "std": (sum((x - mean_time) ** 2 for x in times) / len(times)) ** 0.5 if times else 0,
                    "var": sum((x - mean_time) ** 2 for x in times) / len(times) if times else 0,
                    "count": len(times),
                    "total": sum(times),
                    "p95": sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else max(times),
                    "p99": sorted(times)[int(len(times) * 0.99)] if len(times) > 1 else max(times),
                }

        timing_stats = OrderedDict(
            sorted(timing_stats.items(), key=lambda x: x[1]["total"], reverse=True)
        )

        print(
            f"{'Operation':<25} {'Total(s)':<9} {'Mean(s)':<9} {'Median(s)':<10} {'Min(s)':<8} {'Max(s)':<8} "
            f"{'StdDev(s)':<10} {'Var(s)':<9} {'P95(s)':<8} {'P99(s)':<8} {'Count':<6} {'%Total':<7}"
        )
        print("-" * 120)

        for op_name, stats in timing_stats.items():
            percentage = (stats["total"] / total_time * 100) if total_time > 0 else 0

            print(
                f"{op_name:<25} {stats['total']:<9.6f} {stats['mean']:<9.6f} {stats['median']:<10.6f} "
                f"{stats['min']:<8.6f} {stats['max']:<8.6f} {stats['std']:<10.6f} {stats['var']:<9.6f} "
                f"{stats['p95']:<8.6f} {stats['p99']:<8.6f} {stats['count']:<6} {percentage:<7.2f}"
            )

        print("-" * 120)
        print(
            f"SUMMARY: Total execution time: {total_time:.6f}s | Total operations: {total_calls} | Operations tracked: {len(timing_stats)}"
        )

        if len(timing_stats) > 0:
            times_per_op = [stats["mean"] for stats in timing_stats.values()]
            print(
                f"STATS: Mean operation time: {sum(times_per_op) / len(times_per_op):.6f}s | "
                f"Operation time stddev: {(sum((x - sum(times_per_op) / len(times_per_op)) ** 2 for x in times_per_op) / len(times_per_op)) ** 0.5:.6f}s | "
                f"Slowest operation: {max(timing_stats.keys(), key=lambda x: timing_stats[x]['total'])}"
            )
        print()

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        timing_stats = {}

        for op_name, times in self.timings.items():
            if times:
                mean_time = sum(times) / len(times)
                timing_stats[op_name] = {
                    "mean": mean_time,
                    "median": sorted(times)[len(times) // 2] if times else 0,
                    "min": min(times),
                    "max": max(times),
                    "std": (sum((x - mean_time) ** 2 for x in times) / len(times)) ** 0.5 if times else 0,
                    "var": sum((x - mean_time) ** 2 for x in times) / len(times) if times else 0,
                    "count": len(times),
                    "total": sum(times),
                    "p95": sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else max(times),
                    "p99": sorted(times)[int(len(times) * 0.99)] if len(times) > 1 else max(times),
                }

        return timing_stats
