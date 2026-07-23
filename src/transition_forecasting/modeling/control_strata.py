from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from transition_forecasting.catalogue.transition_events import (
    HORIZON,
    K,
    M,
    MATCH_FEATURES,
    PRIOR_MAX,
    PRIOR_WIN,
)
from transition_forecasting.modeling.chronological_control_matching import (
    MatchConfig,
    rematch_controls_within_partition,
)

ControlStratum = Literal["calm", "hard_negative", "persistent_excluded"]

CALM: Final[str] = "calm"
HARD_NEGATIVE: Final[str] = "hard_negative"
PERSISTENT_EXCLUDED: Final[str] = "persistent_excluded"
TRANSITION: Final[str] = "transition"


@dataclass(frozen=True)
class ControlStrataPolicy:
    """Frozen two-stratum negative-control policy.

    Outcome observations assign the supervised negative stratum. Candidate matching
    and deterministic tie-breaking use MATCH_FEATURES, computed entirely from the
    40-day pre-origin input.
    """

    forecast_horizon: int = HORIZON
    persistence_required: int = K
    persistence_window: int = M
    prior_window: int = PRIOR_WIN
    prior_max_crossings: int = PRIOR_MAX
    calm_max_horizon_crossings: int = 0
    calm_controls_per_positive: int = 1
    hard_controls_per_positive: int = 2

    def validate(self) -> None:
        if self.forecast_horizon < 1:
            raise ValueError("forecast_horizon must be positive")
        if self.persistence_window < 1:
            raise ValueError("persistence_window must be positive")
        if not 1 <= self.persistence_required <= self.persistence_window:
            raise ValueError("persistence_required must lie inside persistence_window")
        if self.prior_window < 1:
            raise ValueError("prior_window must be positive")
        if not 0 <= self.prior_max_crossings < self.prior_window:
            raise ValueError("prior_max_crossings must lie below prior_window")
        if not 0 <= self.calm_max_horizon_crossings < self.forecast_horizon:
            raise ValueError(
                "calm_max_horizon_crossings must lie below forecast_horizon"
            )
        if self.calm_controls_per_positive < 1:
            raise ValueError("at least one calm control is required per positive")
        if self.hard_controls_per_positive < 1:
            raise ValueError("at least one hard negative is required per positive")

    @property
    def future_assessment_rows(self) -> int:
        return self.forecast_horizon + self.persistence_window - 1

    @property
    def controls_per_positive(self) -> int:
        return self.calm_controls_per_positive + self.hard_controls_per_positive

    @property
    def requested_counts(self) -> dict[str, int]:
        return {
            CALM: self.calm_controls_per_positive,
            HARD_NEGATIVE: self.hard_controls_per_positive,
        }

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["future_assessment_rows"] = self.future_assessment_rows
        payload["controls_per_positive"] = self.controls_per_positive
        payload["matching_features"] = list(MATCH_FEATURES)
        payload["outcome_values_used_for"] = "negative_stratum_label_only"
        payload["matching_information"] = "pre_origin_only"
        payload["persistent_rule"] = (
            "catalogue onset rule: onset begins within forecast_horizon, prior_window "
            "has at most prior_max_crossings, and the following persistence_window "
            "has at least persistence_required threshold days"
        )
        return payload

    @classmethod
    def from_total_controls(cls, controls_per_positive: int) -> "ControlStrataPolicy":
        """Preserve the historical total while guaranteeing both strata.

        Odd totals assign the additional archived control to the hard-negative stratum,
        which is more prevalent in the provisional audit. Final evaluation selects one
        calm and one hard control per transition episode.
        """

        if controls_per_positive < 2:
            raise ValueError(
                "deterministic two-stratum controls require at least two controls per positive"
            )
        calm = controls_per_positive // 2
        hard = controls_per_positive - calm
        policy = cls(
            calm_controls_per_positive=calm,
            hard_controls_per_positive=hard,
        )
        policy.validate()
        return policy


def _persistent_onset_offsets(
    prior_history: np.ndarray,
    future: np.ndarray,
    *,
    threshold: float,
    policy: ControlStrataPolicy,
) -> list[int]:
    prior = prior_history[-policy.prior_window :]
    combined = np.concatenate([prior, future])
    high = combined >= float(threshold)
    future_start = len(prior)
    offsets: list[int] = []
    for start in range(policy.forecast_horizon):
        onset = future_start + start
        prior_window = high[onset - policy.prior_window : onset]
        persistence_window = high[onset : onset + policy.persistence_window]
        if (
            bool(high[onset])
            and int(prior_window.sum()) <= policy.prior_max_crossings
            and int(persistence_window.sum()) >= policy.persistence_required
        ):
            offsets.append(start + 1)
    return offsets


def classify_control_future(
    future: np.ndarray,
    *,
    prior_history: np.ndarray,
    threshold: float,
    policy: ControlStrataPolicy = ControlStrataPolicy(),
) -> dict[str, object]:
    """Classify a negative with the catalogue's exact local onset conditions.

    A persistent candidate has a quiet-enough prior window, an onset at h1...hH,
    and satisfies the positive K-of-M rule. Calm controls have no threshold crossing
    inside h1...hH. Hard negatives cross during the forecast horizon but never form
    such an onset. Outcome values never enter matching or model inputs.
    """

    policy.validate()
    values = np.asarray(future, dtype=float).reshape(-1)
    prior = np.asarray(prior_history, dtype=float).reshape(-1)
    required = policy.future_assessment_rows
    if len(values) < required:
        raise ValueError(
            f"future assessment requires {required} values; got {len(values)}"
        )
    if len(prior) < policy.prior_window:
        raise ValueError(
            f"prior assessment requires {policy.prior_window} values; got {len(prior)}"
        )
    values = values[:required]
    prior = prior[-policy.prior_window :]
    if (
        not np.isfinite(values).all()
        or not np.isfinite(prior).all()
        or not np.isfinite(float(threshold))
    ):
        raise ValueError("prior, future, and threshold must be finite")

    horizon_values = values[: policy.forecast_horizon]
    horizon_crossings = int(
        np.count_nonzero(horizon_values >= float(threshold))
    )
    persistent_offsets = _persistent_onset_offsets(
        prior,
        values,
        threshold=float(threshold),
        policy=policy,
    )
    persistent = bool(persistent_offsets)

    if persistent:
        stratum: ControlStratum = PERSISTENT_EXCLUDED
    elif horizon_crossings <= policy.calm_max_horizon_crossings:
        stratum = CALM
    else:
        stratum = HARD_NEGATIVE

    return {
        "control_stratum": stratum,
        "future_threshold_crossings": horizon_crossings,
        "future_threshold_max": float(horizon_values.max()),
        "future_threshold_max_excess": float(
            horizon_values.max() - float(threshold)
        ),
        "future_assessment_max": float(values.max()),
        "future_persistent": persistent,
        "future_persistent_onset_count": int(len(persistent_offsets)),
        "future_first_persistent_offset": (
            int(persistent_offsets[0]) if persistent_offsets else -1
        ),
        "future_assessment_rows": int(required),
        "prior_threshold_crossings_at_origin": int(
            np.count_nonzero(prior >= float(threshold))
        ),
    }


def rematch_control_strata_within_partition(
    positives: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    partition_columns: tuple[str, ...] = ("fold", "fold_split", "index"),
    candidate_filter_columns: tuple[str, ...] = ("lead",),
    policy: ControlStrataPolicy = ControlStrataPolicy(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match calm and hard negatives separately with deterministic no replacement."""

    policy.validate()
    if "control_stratum" not in candidates.columns:
        raise ValueError("candidates are missing required column: control_stratum")
    unknown = set(candidates["control_stratum"].dropna().astype(str).unique()).difference(
        {CALM, HARD_NEGATIVE}
    )
    if unknown:
        raise ValueError(f"unexpected eligible control strata: {sorted(unknown)}")

    matched_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    prefixes = {CALM: "CALM", HARD_NEGATIVE: "HARD"}

    for stratum, requested in policy.requested_counts.items():
        local_candidates = candidates.loc[
            candidates["control_stratum"].astype(str).eq(stratum)
        ].copy()
        matched, audit = rematch_controls_within_partition(
            positives,
            local_candidates,
            partition_columns=partition_columns,
            candidate_filter_columns=candidate_filter_columns,
            config=MatchConfig(controls_per_positive=requested),
        )
        audit["control_stratum"] = stratum
        audit["controls_requested"] = int(requested)

        if not matched.empty:
            matched["control_stratum"] = stratum
            matched = matched.sort_values(
                ["matched_positive_id", "match_distance", "origin_pos", "origin_date"],
                kind="mergesort",
            ).reset_index(drop=True)
            ranks = matched.groupby("matched_positive_id", sort=False).cumcount() + 1
            matched["sample_id"] = [
                f"N_{prefixes[stratum]}_{positive_id}_{int(rank)}"
                for positive_id, rank in zip(
                    matched["matched_positive_id"].astype(str),
                    ranks,
                    strict=True,
                )
            ]
        matched_parts.append(matched)
        audit_parts.append(audit)

    matched_all = pd.concat(matched_parts, ignore_index=True, sort=False)
    audit_all = pd.concat(audit_parts, ignore_index=True, sort=False)

    required_strata = set(policy.requested_counts)
    completion = (
        audit_all.groupby("positive_sample_id", sort=False)
        .agg(
            strata_seen=("control_stratum", lambda values: set(values.astype(str))),
            all_complete=("complete_match", "all"),
        )
    )
    complete_ids = {
        str(sample_id)
        for sample_id, row in completion.iterrows()
        if bool(row["all_complete"]) and row["strata_seen"] == required_strata
    }
    audit_all["complete_all_strata"] = audit_all["positive_sample_id"].astype(str).isin(
        complete_ids
    )

    uniqueness_key = [*partition_columns, "origin_date"]
    if not matched_all.empty and matched_all.duplicated(uniqueness_key).any():
        raise ValueError(
            "control origin reuse detected across calm/hard strata or forecast leads"
        )
    if not matched_all.empty and matched_all["sample_id"].duplicated().any():
        raise ValueError("stratified control sample IDs are not unique")

    return matched_all, audit_all
