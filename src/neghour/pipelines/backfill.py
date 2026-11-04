"""
Backfill: run the feature pipeline over a range of delivery days.

It uses the same code as the daily run and writes idempotent upserts, so
running it twice changes nothing.

Invoked via:  python -m neghour.pipelines.backfill --first-day 2024-01-01
Triggered by: .github/workflows/backfill.yml
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

from neghour.pipelines.feature_pipeline import run, write_step_summary
from neghour.timeutils import berlin_today

logger = logging.getLogger(__name__)

DEFAULT_FIRST_DAY = date(2024, 1, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-day", type=date.fromisoformat, default=DEFAULT_FIRST_DAY
    )
    parser.add_argument(
        "--last-day",
        type=date.fromisoformat,
        default=berlin_today() + timedelta(days=1),
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    args = parse_args()
    summary = run(args.first_day, args.last_day)
    logger.info("Backfill finished: %s", summary)
    write_step_summary(summary)


if __name__ == "__main__":
    main()
