from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.modeling.stage_e_classical_baselines import qlike_loss


def _latest_complete_run(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "validation_predictions.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no completed Stage E run found under {root}")
    return candidates[-1]


def _model_frame(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    frame = predictions[predictions["model"].eq(model)].copy()
    if frame.empty:
        raise ValueError(f"model not found in predictions: {model}")
    return frame.sort_values("sample_id").reset_index(drop=True)


def paired_episode_bootstrap(
    predictions: pd.DataFrame,
    *,
    challenger: str = "sequence_ridge",
    reference: str = "har_ridge",
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    challenger_frame = _model_frame(predictions, challenger)
    reference_frame = _model_frame(predictions, reference)
    id_columns = ["sample_id", "episode_id", "label", "lead", "split"]
    if not challenger_frame[id_columns].equals(reference_frame[id_columns]):
        raise ValueError("challenger and reference predictions are not sample-aligned")

    actual_columns = [f"actual_h{h}" for h in range(1, 11)]
    predicted_columns = [f"predicted_h{h}" for h in range(1, 11)]
    actual = challenger_frame[actual_columns].to_numpy(dtype=float)
    challenger_pred = challenger_frame[predicted_columns].to_numpy(dtype=float)
    reference_pred = reference_frame[predicted_columns].to_numpy(dtype=float)
    sample_delta = qlike_loss(actual, challenger_pred).mean(axis=1) - qlike_loss(
        actual, reference_pred
    ).mean(axis=1)

    frame = challenger_frame[id_columns].copy()
    frame["sample_delta_qlike"] = sample_delta
    episode_delta = frame.groupby("episode_id", as_index=False).agg(
        label=("label", "first"),
        lead=("lead", "first"),
        delta_qlike=("sample_delta_qlike", "mean"),
    )

    groups: list[tuple[str, str, pd.Series]] = [
        ("pooled", "all", pd.Series(True, index=episode_delta.index)),
        ("label", "positive", episode_delta["label"].eq(1)),
        ("label", "control", episode_delta["label"].eq(0)),
    ]
    for lead in sorted(episode_delta["lead"].unique()):
        groups.append(("lead", str(int(lead)), episode_delta["lead"].eq(lead)))
        groups.append(
            (
                "lead_label",
                f"L{int(lead)}_positive",
                episode_delta["lead"].eq(lead) & episode_delta["label"].eq(1),
            )
        )
        groups.append(
            (
                "lead_label",
                f"L{int(lead)}_control",
                episode_delta["lead"].eq(lead) & episode_delta["label"].eq(0),
            )
        )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for group_type, group_value, mask in groups:
        values = episode_delta.loc[mask, "delta_qlike"].to_numpy(dtype=float)
        if values.size == 0:
            continue
        draws = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(axis=1)
        rows.append(
            {
                "challenger": challenger,
                "reference": reference,
                "group_type": group_type,
                "group_value": group_value,
                "n_episodes": int(values.size),
                "mean_delta_qlike": float(values.mean()),
                "ci_2_5": float(np.quantile(draws, 0.025)),
                "ci_97_5": float(np.quantile(draws, 0.975)),
                "probability_challenger_better": float(np.mean(draws < 0.0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired episode bootstrap for Stage E validation QLIKE differences."
    )
    parser.add_argument(
        "--stage-e-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_classical_sanity"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--challenger", default="sequence_ridge")
    parser.add_argument("--reference", default="har_ridge")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_complete_run(args.stage_e_root)
    predictions = pd.read_csv(run_dir / "validation_predictions.csv")
    result = paired_episode_bootstrap(
        predictions,
        challenger=args.challenger,
        reference=args.reference,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    output_path = run_dir / f"bootstrap_{args.challenger}_vs_{args.reference}.csv"
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False))
    print(json.dumps({"output": str(output_path), "test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
