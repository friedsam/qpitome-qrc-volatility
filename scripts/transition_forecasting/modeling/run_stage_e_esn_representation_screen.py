from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.esn_representation import INPUT_REPRESENTATIONS, POOLINGS, pool_trajectory, reservoir_trajectory, transform_input
from baselines.numpy_esn import make_esn_weights
from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import HAR_FEATURES, TARGET_COLUMNS, qlike_loss

ROLLING_SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_rolling_origin.py")
SPEC = importlib.util.spec_from_file_location("stage_e_rolling_origin", ROLLING_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ROLLING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROLLING)

ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
WASHOUTS = (0, 5, 10)
CONFIG = {"n": 300, "sr": 0.9, "inp": 0.3, "leak": 0.3}


def _scale_channels(X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = X[train_mask].reshape(-1, X.shape[-1])
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (X - mean[None, None, :]) / scale[None, None, :]


def _fit_har(manifest: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x = manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=100.0)
    model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
    return model.predict(scaler.transform(x))


def _score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    return (
        float(qlike_loss(y[mask], prediction[mask]).mean()),
        float(np.sqrt(np.mean((y[mask] - prediction[mask]) ** 2))),
    )


def evaluate_fold(
    manifest: pd.DataFrame,
    sequences: np.ndarray,
    *,
    fold: int,
    seeds: tuple[int, ...] = (1, 2, 3),
    alphas: tuple[float, ...] = ALPHAS,
    washouts: tuple[int, ...] = WASHOUTS,
    representations: tuple[str, ...] = INPUT_REPRESENTATIONS,
    poolings: tuple[str, ...] = POOLINGS,
    n_reservoir: int = 300,
) -> pd.DataFrame:
    train_mask = manifest["fold_split"].eq("train").to_numpy()
    val_mask = manifest["fold_split"].eq("val").to_numpy()
    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = _fit_har(manifest, y, train_mask)
    residual = y - har
    rows: list[dict[str, object]] = []

    for representation in representations:
        inputs = _scale_channels(transform_input(sequences, representation), train_mask)
        for seed in seeds:
            w_in, w = make_esn_weights(
                n_inputs=inputs.shape[-1],
                n_reservoir=n_reservoir,
                spectral_radius=float(CONFIG["sr"]),
                input_scale=float(CONFIG["inp"]),
                seed=int(seed),
            )
            ordered_states = reservoir_trajectory(inputs, w_in, w, float(CONFIG["leak"]))
            rng = np.random.default_rng(int(seed) + 20000)
            shuffled_inputs = inputs.copy()
            for sample in range(len(shuffled_inputs)):
                shuffled_inputs[sample] = shuffled_inputs[sample, rng.permutation(shuffled_inputs.shape[1]), :]
            shuffled_states = reservoir_trajectory(shuffled_inputs, w_in, w, float(CONFIG["leak"]))

            for washout in washouts:
                for pooling in poolings:
                    for order_name, states in (("ordered", ordered_states), ("shuffled", shuffled_states)):
                        features = pool_trajectory(states, pooling, washout)
                        scaler = StandardScaler()
                        train_features = scaler.fit_transform(features[train_mask])
                        all_features = scaler.transform(features)
                        for alpha in alphas:
                            model = Ridge(alpha=float(alpha))
                            model.fit(train_features, residual[train_mask])
                            prediction = har + model.predict(all_features)
                            qlike, rmse = _score(y, prediction, val_mask)
                            rows.append({
                                "fold": fold,
                                "representation": representation,
                                "pooling": pooling,
                                "washout": int(washout),
                                "order": order_name,
                                "seed": int(seed),
                                "alpha": float(alpha),
                                "val_qlike": qlike,
                                "val_rmse": rmse,
                            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory ESN input/pooling/washout representation screen.")
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument("--rolling-run", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n-reservoir", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("results/transition_forecasting/modeling/stage_e_esn_representation_screen"))
    parser.add_argument("--run-id")
    args = parser.parse_args()

    _, sequences = ROLLING.load_stage_d(args.stage_d_run)
    assignments = pd.read_csv(args.rolling_run / "rolling_fold_manifest.csv")
    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)

    frames = []
    for fold in sorted(assignments["fold"].unique()):
        fold_manifest = assignments[assignments["fold"].eq(fold)].reset_index(drop=True)
        frames.append(evaluate_fold(
            fold_manifest,
            sequences,
            fold=int(fold),
            seeds=tuple(args.seeds),
            n_reservoir=args.n_reservoir,
        ))
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(run_dir / "representation_results_by_seed.csv", index=False)

    summary = results.groupby(
        ["fold", "representation", "pooling", "washout", "order", "alpha"],
        as_index=False,
    ).agg(
        mean_val_qlike=("val_qlike", "mean"),
        std_seed_qlike=("val_qlike", "std"),
        mean_val_rmse=("val_rmse", "mean"),
    )
    summary.to_csv(run_dir / "representation_summary_by_fold.csv", index=False)

    best_by_fold = (
        summary.sort_values(["fold", "order", "mean_val_qlike", "mean_val_rmse"])
        .groupby(["fold", "order"], as_index=False)
        .first()
    )
    best_by_fold.to_csv(run_dir / "representation_best_by_fold.csv", index=False)

    aggregate = summary.groupby(
        ["representation", "pooling", "washout", "order", "alpha"],
        as_index=False,
    ).agg(
        mean_val_qlike=("mean_val_qlike", "mean"),
        std_across_folds=("mean_val_qlike", "std"),
        mean_val_rmse=("mean_val_rmse", "mean"),
    )
    aggregate = aggregate.sort_values("mean_val_qlike")
    aggregate.to_csv(run_dir / "representation_aggregate.csv", index=False)

    best_ordered = aggregate[aggregate["order"].eq("ordered")].iloc[0].to_dict()
    best_shuffled = aggregate[aggregate["order"].eq("shuffled")].iloc[0].to_dict()
    payload = {
        "stage_d_run": str(args.stage_d_run),
        "rolling_run": str(args.rolling_run),
        "test_evaluated": False,
        "screening_only": True,
        "fixed_reservoir": {**CONFIG, "n": args.n_reservoir},
        "best_ordered": best_ordered,
        "best_shuffled": best_shuffled,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
