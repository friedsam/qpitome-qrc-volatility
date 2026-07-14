from __future__ import annotations

from pathlib import Path

from data.download_external_datasets import DatasetSpec, acquire_external_datasets

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

SPECS = [
    DatasetSpec(
        name="fred_tb3ms",
        target_path=Path("data/raw/macro_fred_monthly/fred_TB3MS_three_month_tbill.csv"),
        source_url=f"{FRED_BASE}TB3MS",
    ),
    DatasetSpec(
        name="fred_cpiaucsl",
        target_path=Path("data/raw/macro_fred_monthly/fred_CPIAUCSL_cpi.csv"),
        source_url=f"{FRED_BASE}CPIAUCSL",
    ),
    DatasetSpec(
        name="fred_indpro",
        target_path=Path("data/raw/macro_fred_monthly/fred_INDPRO_industrial_production.csv"),
        source_url=f"{FRED_BASE}INDPRO",
    ),
    DatasetSpec(
        name="fred_aaa",
        target_path=Path("data/raw/macro_fred_monthly/fred_AAA_aaa_corporate_yield.csv"),
        source_url=f"{FRED_BASE}AAA",
    ),
    DatasetSpec(
        name="fred_baa",
        target_path=Path("data/raw/macro_fred_monthly/fred_BAA_baa_corporate_yield.csv"),
        source_url=f"{FRED_BASE}BAA",
    ),
    DatasetSpec(
        name="french_ff3_monthly",
        target_path=Path("data/raw/equity_factor_returns_monthly/ff3_monthly.zip"),
        source_url=(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Research_Data_Factors_CSV.zip"
        ),
    ),
    DatasetSpec(
        name="french_short_term_reversal_monthly",
        target_path=Path(
            "data/raw/equity_factor_returns_monthly/short_term_reversal_monthly.zip"
        ),
        source_url=(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_ST_Reversal_Factor_CSV.zip"
        ),
    ),
    DatasetSpec(
        name="nikkei_225_fred",
        target_path=Path(
            "data/raw/international_equity_indices/nikkei_225_fred_raw.csv"
        ),
        source_url=f"{FRED_BASE}NIKKEI225",
    ),
    DatasetSpec(
        name="russell_2000_yahoo",
        target_path=Path(
            "data/raw/international_equity_indices/russell_2000_yahoo_raw.csv"
        ),
        yahoo_symbol="^RUT",
        start_date="1987-01-01",
        end_date="2026-06-05",
    ),
    DatasetSpec(
        name="ftse_100_yahoo",
        target_path=Path(
            "data/raw/international_equity_indices/ftse_100_yahoo_raw.csv"
        ),
        yahoo_symbol="^FTSE",
        start_date="1984-01-01",
        end_date="2026-06-05",
    ),
]


def main() -> None:
    results = acquire_external_datasets(
        SPECS,
        manifest_path=Path("data/raw/external_dataset_manifest.json"),
    )
    for result in results:
        print(f"{result.name}: {result.status} -> {result.path}")
        if result.error:
            print(f"  remote error: {result.error}")


if __name__ == "__main__":
    main()
