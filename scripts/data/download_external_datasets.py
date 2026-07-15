from __future__ import annotations

from pathlib import Path

from data.download_external_datasets import DatasetSpec, acquire_external_datasets

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
RAW = Path("data/raw")
FALLBACK = Path("data/fallback")


def _spec(
    *,
    name: str,
    relative_path: str,
    source_url: str | None = None,
    yahoo_symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        target_path=RAW / relative_path,
        fallback_path=FALLBACK / relative_path,
        source_url=source_url,
        yahoo_symbol=yahoo_symbol,
        start_date=start_date,
        end_date=end_date,
    )


SPECS = [
    _spec(
        name="sp500_yahoo",
        relative_path="monthly_market_features/sp500_yahoo_raw.csv",
        yahoo_symbol="^GSPC",
        start_date="1927-01-01",
        end_date="2026-06-05",
    ),
    _spec(
        name="fred_tb3ms",
        relative_path="macro_fred_monthly/fred_TB3MS_three_month_tbill.csv",
        source_url=f"{FRED_BASE}TB3MS",
    ),
    _spec(
        name="fred_cpiaucsl",
        relative_path="macro_fred_monthly/fred_CPIAUCSL_cpi.csv",
        source_url=f"{FRED_BASE}CPIAUCSL",
    ),
    _spec(
        name="fred_indpro",
        relative_path="macro_fred_monthly/fred_INDPRO_industrial_production.csv",
        source_url=f"{FRED_BASE}INDPRO",
    ),
    _spec(
        name="fred_aaa",
        relative_path="macro_fred_monthly/fred_AAA_aaa_corporate_yield.csv",
        source_url=f"{FRED_BASE}AAA",
    ),
    _spec(
        name="fred_baa",
        relative_path="macro_fred_monthly/fred_BAA_baa_corporate_yield.csv",
        source_url=f"{FRED_BASE}BAA",
    ),
    _spec(
        name="french_ff3_monthly",
        relative_path="equity_factor_returns_monthly/ff3_monthly.zip",
        source_url=(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Research_Data_Factors_CSV.zip"
        ),
    ),
    _spec(
        name="french_short_term_reversal_monthly",
        relative_path=(
            "equity_factor_returns_monthly/short_term_reversal_monthly.zip"
        ),
        source_url=(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_ST_Reversal_Factor_CSV.zip"
        ),
    ),
    _spec(
        name="nikkei_225_fred",
        relative_path="international_equity_indices/nikkei_225_fred_raw.csv",
        source_url=f"{FRED_BASE}NIKKEI225",
    ),
    _spec(
        name="russell_2000_yahoo",
        relative_path="international_equity_indices/russell_2000_yahoo_raw.csv",
        yahoo_symbol="^RUT",
        start_date="1987-01-01",
        end_date="2026-06-05",
    ),
    _spec(
        name="ftse_100_yahoo",
        relative_path="international_equity_indices/ftse_100_yahoo_raw.csv",
        yahoo_symbol="^FTSE",
        start_date="1984-01-01",
        end_date="2026-06-05",
    ),
]


def main() -> None:
    results = acquire_external_datasets(
        SPECS,
        manifest_path=RAW / "external_dataset_manifest.json",
    )
    for result in results:
        print(f"{result.name}: {result.status} -> {result.path}")
        if result.error:
            print(f"  remote error: {result.error}")
            print(f"  fallback: {result.fallback_path}")


if __name__ == "__main__":
    main()
