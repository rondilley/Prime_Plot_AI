#!/usr/bin/env python
"""List and manage experiment runs.

Usage:
    python list_runs.py                    # List recent runs
    python list_runs.py --type discovery   # Filter by type
    python list_runs.py --limit 50         # Show more runs
    python list_runs.py --cleanup --keep 5 # Clean old runs (dry-run)
    python list_runs.py --cleanup --keep 5 --force  # Actually delete
"""

import argparse
from prime_plot.utils.run_manager import get_run_manager


def main():
    parser = argparse.ArgumentParser(description="List and manage experiment runs")
    parser.add_argument(
        "--type",
        type=str,
        help="Filter by run type (discovery, evolution, training, evaluation)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum runs to show (default: 20)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up old runs"
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=10,
        help="Number of runs to keep when cleaning (default: 10)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually delete (default is dry-run)"
    )

    args = parser.parse_args()

    manager = get_run_manager()

    if args.cleanup:
        print(f"Cleaning up runs, keeping {args.keep} most recent...")
        deleted = manager.cleanup_old_runs(
            keep_count=args.keep,
            run_type=args.type,
            dry_run=not args.force,
        )
        if deleted:
            action = "Deleted" if args.force else "Would delete"
            print(f"{action} {len(deleted)} runs:")
            for run_id in deleted:
                print(f"  - {run_id}")
        else:
            print("No runs to clean up")
        return

    # List runs
    runs = manager.list_runs(run_type=args.type, limit=args.limit)

    if not runs:
        print("No runs found")
        if args.type:
            print(f"(filtered by type: {args.type})")
        return

    print(f"{'Run ID':<55} {'Type':<12} {'Status':<10}")
    print("-" * 80)

    for run in runs:
        # Extract date from run ID for readability
        parts = run.metadata.run_id.split("_")
        if len(parts) >= 3:
            date_str = parts[0]
            time_str = parts[1]
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}"
        else:
            formatted_date = ""

        summary = ""
        if run.metadata.summary:
            if "best_fitness" in run.metadata.summary:
                summary = f"fitness={run.metadata.summary['best_fitness']:.4f}"
            elif "result" in run.metadata.summary:
                summary = run.metadata.summary["result"]

        print(f"{run.metadata.run_id:<55} {run.metadata.run_type:<12} {run.metadata.status:<10}")
        if summary:
            print(f"  -> {summary}")

    print()
    print(f"Total: {len(runs)} runs shown")
    if args.type:
        print(f"(filtered by type: {args.type})")


if __name__ == "__main__":
    main()
