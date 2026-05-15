from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_markdown_report(
    output_path: str | Path,
    title: str,
    metrics: pd.DataFrame,
    notes: list[str] | None = None,
) -> Path:
    """Write a compact markdown report. Convert to PDF separately if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"# {title}", ""]
    if notes:
        lines.append("## Notes")
        lines.extend([f"- {note}" for note in notes])
        lines.append("")

    lines.append("## Model comparison")
    lines.append("")
    lines.append(metrics.to_markdown(index=False))
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
