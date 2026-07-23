from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_dc.catalog import discover_data_sources
from football_dc.quality import check_match_file, reports_to_frame


def main() -> None:
    reports = [check_match_file(source.path) for source in discover_data_sources()]
    if not reports:
        print("No data sources found.")
        return

    frame = reports_to_frame(reports)
    print(frame.to_string(index=False))

    error_count = int(frame["has_errors"].sum())
    if error_count:
        raise SystemExit(f"Data quality check found errors in {error_count} source(s).")


if __name__ == "__main__":
    main()
