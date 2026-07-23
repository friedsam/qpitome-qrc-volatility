from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from transition_forecasting.catalogue.transition_events import K, M, MATCH_FEATURES
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

    Future observations are used only to assign the supervised negative stratum.
    Candidate matching and deterministic tie-breaking use MATCH_FEATURES, which are
    computed entirely from the 40-day pre-origin history.
    """

    persistence_required: int = K
    persistence_window: int = M
    calm_max_crossings: int = 0
    calm_controls_per_positive: int = 2
    hard_controls_per_positive: int = 1

    def validate(self) -> None:
        if self.persistence_window < 1:
            raise ValueError("persistence_window must be positive")
        if not 1 <= self.persistence_required <= self.persistence_window:
            raise ValueError("persistence_required must lie inside persistence_window")
        if not 0 <= self.calm_max_crossings < self.persistence_required:
            raise ValueError("calm_max_crossings must be below persistence_required")
        if self.calm_controls_per_positive < 1:
            raise ValueError("at least one calm control is required per positive")
        if self.hard_controls_per_positive < 1:
            raise ValueError("at least one hard negative is required per positive")

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
        payload["controls_per_positive"] = self.controls_per_positive
        payload["matching_features"] = list(MATCH_FEATURES)
        payload["future_values_used_for"] = "negative_stratum_label_only"
        payload["matching_information"] = "pre_origin_only"
        return payload

    @classmethod
    def from_total_controls(cls, controls_per_positive: int) -> "ControlStrataPolicy":
        """Preserve the historical total while guaranteeing both strata.

        Odd totals assign the additional control to the calm stratum because calm
        false-upward corrections are the primary safety diagnostic. The hard-negative
        stratum always remains represented.
        """

        if controls_per_positive < 2:
            raise ValueError(
                "deterministic two-stratum controls require at least two controls per positive"
            )
        hard = controls_per_positive // 2
        calm = controls_per_positive - hard
        policy = cls(
            calm_controls_per_positive=calm,
            hard_controls_per_positive=hard,
        )
        policy.validate()
        return policy


def classify_control_future(
    future: np.ndarray,
    *,
    threshold: float,
    policy: ControlStrataPolicy = ControlStrataPolicy(),
) -> dict[str, object]:
    """Classify one eligible negative using the positive persistence rule.

    Calm controls have no threshold crossing in the assessment window. Hard
    negatives cross the threshold transiently but fail the required persistence.
    Candidates satisfying the positive persistence count are excluded rather than
    relabeled as controls.
    """

    policy.validate()
    values = np.asarray(future, dtype=float).reshape(-1)
    if len(values) < policy.persistence_window:
        raise ValueError(
            f"future assessment requires {policy.persistence_window} values; got {len(values)}"
        )
    values = values[: policy.persistence_window]
    if not np.isfinite(values).all() or not np.isfinite(float(threshold)):
        raise ValueError("future assessment and threshold must be finite")

    crossings = int(np.count_nonzero(values >= float(threshold)))
    if crossings >= policy.persistence_required:
        stratum: ControlStratum = PERSISTENT_EXCLUDED
    elif crossings <= policy.calm_max_crossings:
        stratum = CALM
    else:
        stratum = HARD_NEGATIVE

    return {
        "control_stratum": stratum,
        "future_threshold_crossings": crossings,
        "future_threshold_max": float(values.max()),
        "future_threshold_max_excess": float(values.max() - float(threshold)),
        "future_persistent": bool(crossings >= policy.persistence_required),
        "future_assessment_rows": int(policy.persistence_window),
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
        raise ValueError("control origin reuse detected across calm/hard strata or forecast leads")
    if not matched_all.empty and matched_all["sample_id"].duplicated().any():
        raise ValueError("stratified control sample IDs are not unique")

    return matched_all, audit_all
