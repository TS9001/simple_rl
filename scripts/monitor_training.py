#!/usr/bin/env python3
"""
Training Progress Monitor
=========================

Monitors training progress by reading logs/progress.json and displaying
a formatted status update. Useful for remote monitoring via SSH.

Usage:
    # Single check:
    python scripts/monitor_training.py

    # Continuous monitoring (updates every 5 seconds):
    watch -n 5 python scripts/monitor_training.py

    # Or use the built-in watch mode:
    python scripts/monitor_training.py --watch
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta


def format_time(seconds):
    """Format seconds into human-readable time."""
    return str(timedelta(seconds=int(seconds)))


def display_progress(progress_file="logs/progress.json"):
    """Display current training progress."""
    progress_path = Path(progress_file)

    if not progress_path.exists():
        print("❌ No progress file found at:", progress_path)
        print("   Training may not have started yet.")
        return False

    try:
        with open(progress_path) as f:
            progress = json.load(f)
    except json.JSONDecodeError:
        print("⚠️  Error reading progress file (may be being written)")
        return False

    # Clear screen for watch mode
    print("\033[H\033[J", end="")

    # Header
    print("=" * 70)
    print("🚀 GRPO TRAINING PROGRESS MONITOR")
    print("=" * 70)
    print()

    # Status
    status = progress.get("status", "unknown")
    status_emoji = {
        "initializing": "🔄",
        "training": "🏃",
        "completed": "✅",
        "error": "❌"
    }.get(status, "❓")

    print(f"Status: {status_emoji} {status.upper()}")
    print(f"Phase:  {progress.get('phase', 'N/A')}")
    print()

    # Time information
    start_time = datetime.fromisoformat(progress.get("start_time", datetime.now().isoformat()))
    elapsed = progress.get("elapsed_time_seconds", 0)
    last_update = datetime.fromisoformat(progress.get("last_update", datetime.now().isoformat()))
    time_since_update = (datetime.now() - last_update).total_seconds()

    print(f"Started:      {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed:      {format_time(elapsed)}")
    print(f"Last update:  {int(time_since_update)}s ago")
    print()

    # Progress
    current = progress.get("current_episode", 0)
    total = progress.get("total_episodes", 0)
    percent = progress.get("progress_percent", 0.0)

    if total > 0:
        bar_width = 40
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"Progress: [{bar}] {percent:.1f}%")
        print(f"Episode:  {current} / {total}")

        # ETA calculation
        if current > 0 and elapsed > 0:
            time_per_episode = elapsed / current
            remaining_episodes = total - current
            eta_seconds = time_per_episode * remaining_episodes
            print(f"ETA:      {format_time(eta_seconds)}")
    print()

    # Latest metrics
    metrics = progress.get("latest_metrics", {})
    if metrics:
        print("Latest Metrics:")
        print(f"  • Loss:            {metrics.get('pg_loss', 0):.4f}")
        print(f"  • KL Divergence:   {metrics.get('kl_divergence', 0):.4f}")
        print(f"  • Reward (mean):   {metrics.get('reward_mean', 0):.3f}")
        print(f"  • Format Reward:   {metrics.get('format_reward_mean', 0):.3f}")
        print(f"  • Correct Reward:  {metrics.get('correctness_reward_mean', 0):.3f}")
        print(f"  • Gradient Norm:   {metrics.get('grad_norm', 0):.3f}")
        print(f"  • Learning Rate:   {metrics.get('lr', 0):.2e}")
        print()

    # Checkpoints
    checkpoints = progress.get("checkpoints", [])
    if checkpoints:
        print(f"Checkpoints: {len(checkpoints)} saved")
        if len(checkpoints) > 0:
            latest = checkpoints[-1]
            print(f"  Latest: {latest.get('path', 'N/A')}")
            print(f"  At episode: {latest.get('episode', 'N/A')}")
        print()

    # Error handling
    error = progress.get("error")
    if error:
        print("❌ ERROR:")
        print(f"  {error}")
        print()

    print("=" * 70)
    print(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(description="Monitor GRPO training progress")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor (refresh every 5 seconds)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--file",
        default="logs/progress.json",
        help="Path to progress.json file"
    )

    args = parser.parse_args()

    if args.watch:
        print("Starting continuous monitoring (Ctrl+C to stop)...")
        print()
        try:
            while True:
                display_progress(args.file)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        display_progress(args.file)


if __name__ == "__main__":
    main()
