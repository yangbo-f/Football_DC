from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_dc.backtesting import evaluate_predictions, walk_forward_dixon_coles, write_backtest_report
from football_dc.data import load_matches


def main() -> None:
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data/worldcup/finals_2026.csv"
    competition = sys.argv[2] if len(sys.argv) > 2 else "WorldCup"
    min_train_matches = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    matches = load_matches(data_path)
    predictions = walk_forward_dixon_coles(matches, competition, min_train_matches=min_train_matches)
    metrics = evaluate_predictions(predictions, "DixonColes")
    frame = predictions.copy()
    report_path = PROJECT_ROOT / "reports" / f"backtest_{data_path.stem}_{competition}.csv"
    write_backtest_report(predictions, report_path)

    print(metrics.to_dict())
    print(f"predictions={len(frame)}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
