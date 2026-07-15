import json
from pathlib import Path
from zipfile import ZipFile

from data.inventory_raw_datasets import inventory_raw_datasets


def test_inventory_reports_csv_and_zip_metadata(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    csv_path = raw_root / "series.csv"
    csv_path.write_text("date,value\n2020-01-01,1\n2020-01-02,2\n", encoding="utf-8")

    zip_path = raw_root / "factors.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("factors.csv", "date,factor\n202001,0.1\n")

    output = tmp_path / "inventory.json"
    results = inventory_raw_datasets(raw_root, output_path=output, sample_size=1)

    assert len(results) == 2
    csv_result = next(result for result in results if result.suffix == ".csv")
    zip_result = next(result for result in results if result.suffix == ".zip")

    assert csv_result.columns == ["date", "value"]
    assert csv_result.sample_rows == [["2020-01-01", "1"]]
    assert csv_result.sha256
    assert zip_result.zip_members == ["factors.csv"]

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["file_count"] == 2
    assert len(payload["files"]) == 2
