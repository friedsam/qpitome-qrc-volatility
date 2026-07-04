# Frequency Structure Audit Report

## Primary conclusion

At least one series/band passed the cross-period stability screen. This supports reproducible low-frequency structure, not a fixed deterministic cycle.

## Passing rows

```text
series,band_lo_days,band_hi_days,n_nonoverlap_periods_significant,significant_periods,full_sample_significant,crisis_period_only,passes_stability_screen
rv_ratio_20_60,60,125,2,1993_2004;2005_2014,True,False,True
rv_ratio_20_60,125,250,2,1993_2004;2005_2014,True,False,True
state__vol_level_x_acceleration_grid__is_6,20,60,2,1993_2004;2005_2014,False,False,True

```

## Interpretation

Large ACF timescales indicate persistence, not periodicity. Slow structure is interesting only when broad-band power exceeds block-surrogate nulls in at least two non-overlapping periods.