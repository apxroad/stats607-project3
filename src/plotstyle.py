def use_nice_style():
    """Apply a clean, consistent Matplotlib theme (visual only).

    Notes
    -----
    - This is a convenience preset for interactive work and quick figures.
    - It tweaks rcParams globally; call early in a session or before plotting.
    - Purely cosmetic: does not alter data or plotting semantics.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150,           # sharper on-screen rendering
        "savefig.dpi": 150,          # higher-res saved figures
        "font.size": 12,             # base font size
        "axes.titlesize": 13,        # slightly larger titles
        "axes.labelsize": 12,        # axis label size
        "axes.titleweight": "semibold",
        "axes.grid": True,           # light grid improves readability
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,    # remove top/right spines for a cleaner look
        "axes.spines.right": False,
        "lines.linewidth": 2.0,      # make lines easier to see
        "lines.markersize": 6.5,
        "legend.frameon": False,     # frameless legends
    })

# --- central plotting style helper ---
# --- central plotting style helper ---
def apply_plot_style(rc_overrides: dict | None = None):
    """Set a project-wide Matplotlib style with sensible defaults.

    Parameters
    ----------
    rc_overrides : dict | None
        Optional dictionary of rcParams to override the defaults here.
        Use to tweak styles per-figure without editing global presets.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt  # noqa: F401  (backend init; also gives plt.cycler)

    rc = {
        # --- Export / typography ---
        "pdf.fonttype": 42,              # embed TrueType; text stays selectable
        "ps.fonttype": 42,
        "savefig.bbox": "tight",

        # --- Figure / axes sizing ---
        "figure.dpi": 150,
        "savefig.dpi": 150,

        # --- Grid & spines ---
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # --- Titles & labels (consistent across the project) ---
        "figure.titlesize": 14,
        "figure.titleweight": "regular",
        "axes.titlesize": 12,
        "axes.titleweight": "regular",
        "axes.titlepad": 8.0,
        "axes.labelsize": "medium",
        "axes.labelweight": "regular",

        # --- Lines & markers ---
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,

        # --- Legend ---
        "legend.frameon": False,
        "legend.loc": "best",
        "legend.borderaxespad": 0.8,

        # --- Ticks ---
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,

        # --- Math text: Computer Modern-style for LaTeX-like labels ---
        "mathtext.fontset": "cm",
        "mathtext.default": "regular",

        # --- Color cycle (subtle, colorblind-friendly) ---
        "axes.prop_cycle": plt.cycler("color", [
            "#1f77b4", "#ff7f0e", "#2ca02c",
            "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]),
    }

    if rc_overrides:
        rc.update(rc_overrides)  # user-provided tweaks take precedence
    mpl.rcParams.update(rc)      # apply globally
