"""
scripts/plot_performance.py

Performance comparison visualizations for Unit 3.

This script expects:
  - results/complexity_parta_compare.csv  (from scripts.complexity_parta_compare)
  - results/benchmark_runtime.csv         (from scripts.benchmark_runtime)

It produces:
  - results/figures/perf_parta_runtime_vs_n.{png,pdf}
  - results/figures/perf_components_baseline_vs_optimized.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.plotstyle import apply_plot_style


def parta_runtime_vs_n(ax: plt.Axes, csv_path: Path) -> None:
    """Line plot: Part A runtime vs n (baseline vs optimized)."""
    df = pd.read_csv(csv_path).sort_values("n")
    ax.plot(
        df["n"],
        df["runtime_baseline"],
        marker="o",
        linestyle="-",
        label="baseline",
    )
    ax.plot(
        df["n"],
        df["runtime_optimized"],
        marker="o",
        linestyle="-",
        label="optimized",
    )
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel("runtime (seconds)")
    ax.set_title("Part A runtime vs $n$")
    ax.legend()


def component_runtime_bar(ax: plt.Axes, csv_path: Path) -> None:
    """Bar chart: baseline vs optimized runtimes for components."""
    df = pd.read_csv(csv_path)

    # Pivot to wide form: rows = component, columns = variant
    wide = df.pivot(index="component", columns="variant", values="runtime_sec")

    components = list(wide.index)
    baseline = wide.get("baseline")
    optimized = wide.get("optimized")

    x = range(len(components))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        baseline.values,
        width=width,
        label="baseline",
    )
    if optimized is not None:
        ax.bar(
            [i + width / 2 for i in x],
            optimized.fillna(0).values,
            width=width,
            label="optimized",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(components)
    ax.set_ylabel("runtime (seconds)")
    ax.set_title("Runtime by component: baseline vs optimized")
    ax.legend()

    # Annotate bars with values
    for i, comp in enumerate(components):
        b = baseline.loc[comp]
        ax.text(i - width / 2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
        if optimized is not None and not pd.isna(optimized.loc[comp]):
            o = optimized.loc[comp]
            ax.text(i + width / 2, o, f"{o:.1f}", ha="center", va="bottom", fontsize=8)


def main() -> None:
    outdir = Path("results/figures")
    outdir.mkdir(parents=True, exist_ok=True)

    complex_csv = Path("results/complexity_parta_compare.csv")
    bench_csv = Path("results/benchmark_runtime.csv")

    if not complex_csv.exists():
        raise SystemExit(
            f"{complex_csv} not found. Run `make complexity` (or the corresponding "
            "Python script) first."
        )
    if not bench_csv.exists():
        raise SystemExit(
            f"{bench_csv} not found. Run `make benchmark` (or the corresponding "
            "Python script) first."
        )

    apply_plot_style()

    # Figure 1: Part A complexity
    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    parta_runtime_vs_n(ax1, complex_csv)
    fig1.tight_layout()
    png1 = outdir / "perf_parta_runtime_vs_n.png"
    pdf1 = outdir / "perf_parta_runtime_vs_n.pdf"
    fig1.savefig(png1, dpi=300)
    fig1.savefig(pdf1)
    print(f"[ok] wrote {png1}")
    print(f"[ok] wrote {pdf1}")

    # Figure 2: component bar chart
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    component_runtime_bar(ax2, bench_csv)
    fig2.tight_layout()
    png2 = outdir / "perf_components_baseline_vs_optimized.png"
    pdf2 = outdir / "perf_components_baseline_vs_optimized.pdf"
    fig2.savefig(png2, dpi=300)
    fig2.savefig(pdf2)
    print(f"[ok] wrote {png2}")
    print(f"[ok] wrote {pdf2}")


if __name__ == "__main__":
    main()
