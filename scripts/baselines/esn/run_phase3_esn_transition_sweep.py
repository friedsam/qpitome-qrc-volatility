#!/usr/bin/env python3
"""Controlled ESN sweep for rv_innovation_20d."""
from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from qpitome_qrc.baselines.numpy_esn import make_esn_weights
from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.data.targets import RV_INNOVATION_TARGET, RV_LEVEL_TARGET, RV_REFERENCE_COLUMN, add_rv_innovation_target, reconstruct_future_rv
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.evaluation.transition import evaluate_transition_forecast
from qpitome_qrc.evaluation.walkforward import make_purged_walkforward_folds, slice_fold_frames

SPLITS = ("train", "val", "test")
HAR = ("rv_5d", "rv_10d", "rv_20d", "rv_60d")


def csvnums(text, cast):
    return [cast(x) for x in text.split(",") if x.strip()]


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("results/canonical/transition_v1/esn"))
    p.add_argument("--tag", default="esn_transition_v1")
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--representation-grid", default="6,10,16,26")
    p.add_argument("--alphas", default="10,30,100,300,1000")
    p.add_argument("--search-seed", type=int, default=42)
    p.add_argument("--robust-seeds", default="7,42,123")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--only-folds", nargs="*", type=int)
    return p.parse_args()


def grid():
    out = []
    for n, srs, inps, leaks in (
        (300, (.3, .7, .9, 1.1), (.1, .3), (.3, .5, 1.0)),
        (500, (.7, .9), (.1, .2), (.3, .5)),
    ):
        for sr in srs:
            for inp in inps:
                for leak in leaks:
                    out.append(dict(n=n, sr=sr, inp=inp, leak=leak))
    return out


def windows(a, L):
    return np.transpose(
        np.lib.stride_tricks.sliding_window_view(a, L, axis=0),
        (0, 2, 1),
    ).copy()


def states(X, Win, W, leak):
    h = np.zeros((len(X), W.shape[0]))
    A, B = Win.T, W.T
    for t in range(X.shape[1]):
        h = (1 - leak) * h + leak * np.tanh(X[:, t, :] @ A + h @ B)
    return np.concatenate([h, X[:, -1, :]], axis=1)


def ridge(H, y, alpha):
    sc = StandardScaler().fit(H["train"])
    m = Ridge(alpha=alpha).fit(sc.transform(H["train"]), y["train"])
    return {s: m.predict(sc.transform(H[s])) for s in SPLITS}


def select_alpha(H, y, alphas):
    best = None
    for a in alphas:
        sc = ridge(H, y, a)
        rm = evaluate_transition_forecast(y["val"], sc["val"])["innovation_rmse"]
        if best is None or rm < best[0]:
            best = (rm, a, sc)
    return best


def rep(frames, d):
    sc = StandardScaler().fit(frames["train"][FEATURE_COLUMNS])
    z = {s: sc.transform(frames[s][FEATURE_COLUMNS]) for s in SPLITS}
    if d == len(FEATURE_COLUMNS):
        return z
    p = PCA(n_components=d, random_state=42).fit(z["train"])
    return {s: p.transform(z[s]) for s in SPLITS}


def row(name, fid, y, sc, refs, levels, extra):
    r = dict(fold=fid, model=name, **extra)
    for s in ("val", "test"):
        r.update({
            f"{s}_{k}": v
            for k, v in evaluate_transition_forecast(y[s], sc[s]).items()
        })
        rec = reconstruct_future_rv(refs[s], sc[s])
        r.update({
            f"{s}_reconstructed_{k}": v
            for k, v in asdict(
                evaluate_volatility_forecast(levels[s], rec)
            ).items()
        })
    return r


def main():
    a = args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dims = csvnums(a.representation_grid, int)
    alphas = csvnums(a.alphas, float)
    seeds = csvnums(a.robust_seeds, int)
    cfgs = grid()

    if any(d < 1 or d > len(FEATURE_COLUMNS) for d in dims):
        raise ValueError("invalid representation dimension")

    df = add_rv_innovation_target(
        pd.read_csv(a.data).sort_values("date").reset_index(drop=True)
    )
    folds = make_purged_walkforward_folds(
        len(df),
        n_folds=a.n_folds,
        min_train=a.min_train,
        val_size=a.val_size,
        purge=a.purge,
    )
    if a.only_folds:
        folds = [f for f in folds if int(f["fold"]) in set(a.only_folds)]

    mp = a.out_dir / f"esn_transition_metrics_{a.tag}.csv"
    gp = a.out_dir / f"esn_transition_grid_{a.tag}.csv"
    pp = a.out_dir / f"esn_transition_predictions_{a.tag}.csv"

    M = pd.read_csv(mp).to_dict("records") if mp.exists() else []
    G = pd.read_csv(gp).to_dict("records") if gp.exists() else []
    P = pd.read_csv(pp).to_dict("records") if pp.exists() else []
    done = {
        int(r["fold"])
        for r in M
        if r.get("model") == "esn_selected"
    }

    for f in folds:
        fid = int(f["fold"])
        if fid in done:
            print(f"SKIP completed ESN fold {fid}")
            continue

        fr = slice_fold_frames(df, f)
        al = {
            s: fr[s].iloc[a.lookback - 1:].reset_index(drop=True)
            for s in SPLITS
        }
        y = {s: al[s][RV_INNOVATION_TARGET].to_numpy(float) for s in SPLITS}
        refs = {s: al[s][RV_REFERENCE_COLUMN].to_numpy(float) for s in SPLITS}
        levels = {s: al[s][RV_LEVEL_TARGET].to_numpy(float) for s in SPLITS}
        dates = {s: al[s]["date"].to_numpy() for s in SPLITS}

        zero = {s: np.zeros_like(y[s]) for s in SPLITS}
        M.append(row("zero_change", fid, y, zero, refs, levels, {}))

        for name, cols in (
            ("har_rv_linear", HAR),
            ("full_linear", FEATURE_COLUMNS),
        ):
            H = {s: al[s][list(cols)].to_numpy(float) for s in SPLITS}
            _, aa, sc = select_alpha(H, y, alphas)
            M.append(row(
                name, fid, y, sc, refs, levels,
                {"selected_alpha": aa},
            ))

        br = None
        be = None

        for d in dims:
            z = rep(fr, d)
            X = {s: windows(z[s], a.lookback) for s in SPLITS}
            last = {s: z[s][a.lookback - 1:] for s in SPLITS}

            rm, aa, sc = select_alpha(last, y, alphas)
            if br is None or rm < br[0]:
                br = (rm, d, aa, sc)

            for c in cfgs:
                Win, W = make_esn_weights(
                    z["train"].shape[1],
                    c["n"],
                    c["sr"],
                    c["inp"],
                    a.search_seed,
                )
                H = {s: states(X[s], Win, W, c["leak"]) for s in SPLITS}

                for aa in alphas:
                    sc = ridge(H, y, aa)
                    vm = evaluate_transition_forecast(y["val"], sc["val"])
                    G.append(dict(
                        fold=fid,
                        representation_dim=d,
                        **c,
                        alpha=aa,
                        seed=a.search_seed,
                        **{f"val_{k}": v for k, v in vm.items()},
                    ))
                    if be is None or vm["innovation_rmse"] < be[0]:
                        be = (
                            vm["innovation_rmse"],
                            d,
                            c,
                            aa,
                            sc,
                        )

        _, d, aa, sc = br
        M.append(row(
            "representation_linear_selected",
            fid,
            y,
            sc,
            refs,
            levels,
            {"selected_dim": d, "selected_alpha": aa},
        ))

        _, d, c, aa, sc = be
        z = rep(fr, d)
        X = {s: windows(z[s], a.lookback) for s in SPLITS}
        seedr = {}

        for seed in seeds:
            Win, W = make_esn_weights(
                z["train"].shape[1],
                c["n"],
                c["sr"],
                c["inp"],
                seed,
            )
            H = {s: states(X[s], Win, W, c["leak"]) for s in SPLITS}
            ss = ridge(H, y, aa)
            seedr[str(seed)] = evaluate_transition_forecast(
                y["test"], ss["test"]
            )["innovation_r2"]
            if seed == a.search_seed:
                sc = ss

        M.append(row(
            "esn_selected",
            fid,
            y,
            sc,
            refs,
            levels,
            {
                "selected_dim": d,
                "selected_alpha": aa,
                "selected_seed": a.search_seed,
                "seed_test_r2": json.dumps(seedr, sort_keys=True),
                **c,
            },
        ))

        for s in ("val", "test"):
            rec = reconstruct_future_rv(refs[s], sc[s])
            for dt, yt, yp, lt, lp in zip(
                dates[s], y[s], sc[s], levels[s], rec, strict=True
            ):
                P.append(dict(
                    fold=fid,
                    split=s,
                    date=dt,
                    model="esn_selected",
                    innovation_true=yt,
                    innovation_pred=yp,
                    future_rv_true=lt,
                    predicted_future_rv=lp,
                ))

        pd.DataFrame(M).to_csv(mp, index=False)
        pd.DataFrame(G).to_csv(gp, index=False)
        pd.DataFrame(P).to_csv(pp, index=False)

    metrics = pd.DataFrame(M)
    summary = dict(
        tag=a.tag,
        target=RV_INNOVATION_TARGET,
        selection="validation innovation RMSE only",
        representation_grid=dims,
        alphas=alphas,
        reservoir_grid_size=len(cfgs),
        median_test_metrics=metrics.groupby("model")[
            [
                "test_innovation_rmse",
                "test_innovation_r2",
                "test_reconstructed_rmse",
            ]
        ].median().to_dict(orient="index"),
    )
    (
        a.out_dir / f"esn_transition_summary_{a.tag}.json"
    ).write_text(json.dumps(summary, indent=2, default=str))

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
