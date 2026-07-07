from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def write_markdown_report(
    output_path: str | Path,
    title: str,
    metrics: pd.DataFrame,
    notes: list[str] | None = None,
) -> Path:
    """Write a compact markdown report."""
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


def dataframe_page(
    title: str,
    df: pd.DataFrame,
    max_rows: int = 18,
    max_cols: int = 8,
    figsize: tuple[float, float] = (11, 8.5),
):
    """Render a compact DataFrame preview as a matplotlib figure."""
    preview = df.head(max_rows).copy()
    if len(preview.columns) > max_cols:
        preview = preview.iloc[:, :max_cols]

    preview = preview.fillna("")
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=14)
    table = ax.table(
        cellText=preview.astype(str).values,
        colLabels=list(preview.columns),
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)
    fig.tight_layout()
    return fig


def text_page(title: str, lines: list[str], figsize: tuple[float, float] = (11, 8.5)):
    """Render text into a matplotlib figure page."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=16, loc="left", pad=14)
    ax.text(
        0.02,
        0.92,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        wrap=True,
        family="monospace",
    )
    fig.tight_layout()
    return fig


def build_pdf_report(
    output_pdf: str | Path,
    title: str,
    notes: list[str],
    tables: dict[str, pd.DataFrame],
    figure_paths: list[str | Path] | None = None,
) -> Path:
    """Build a self-contained PDF report from text, tables, and saved figures."""
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        fig = text_page(title, notes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for table_title, table_df in tables.items():
            fig = dataframe_page(table_title, table_df)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for path in figure_paths or []:
            path = Path(path)
            if not path.exists():
                continue
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(path.name, fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return output_pdf
