"""
Load flagged accounts from the daily job's anomalies/day_XXXX.json files into the alert queue.

The daily job does this automatically for each new day. Use this to backfill days that were
generated before the queue existed:

    python -m ml.cases.ingest
"""

import argparse
import glob
import json
import os

from ml.cases.store import CaseStore

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(ROOT, "ml", "data_store")


def ingest_directory(data_dir: str) -> int:
    """Queue serious flags from every day file. Idempotent. Returns the number of new alerts."""
    store = CaseStore(os.path.join(data_dir, "cases.db"))
    created = 0
    for path in sorted(glob.glob(os.path.join(data_dir, "anomalies", "day_*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except ValueError:
            continue  # a half-written or corrupt file is skipped, not fatal
        created += store.ingest(data.get("records", []), job_id=data.get("job_id"))
    return created


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args(argv)
    n = ingest_directory(args.data_dir)
    print(f"{n} new alert{'s' if n != 1 else ''} queued from {args.data_dir}")


if __name__ == "__main__":
    main()
