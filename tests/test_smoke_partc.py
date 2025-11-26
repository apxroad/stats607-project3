import pathlib, subprocess

def test_partc_pooled_z_pipeline(tmp_path):
    # Ensure standard result directories exist for the pipeline.
    raw = pathlib.Path("results/raw")
    figs = pathlib.Path("results/figures")
    raw.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # --- Step 1: generate Proposition 2.6 raw CSV (tiny config for speed) ---
    subprocess.run(
        "python -m src_cli.partc_log_prop26 --alpha 5 --t 0.5 "
        "--n 100 --M 100 --seed 1 --base uniform",
        shell=True, check=True
    )
    
    # Verify that at least one matching CSV was written.
    csvs = list(raw.glob("prop26_*_a5.0_*_uniform.csv"))
    assert csvs, "expected prop26 raw CSV"

    # --- Step 2: render pooled-Z figure from the CSV ---
    subprocess.run(
        f"python -m src_cli.partc_figures_prop26 --csv {csvs[0]} --title 'test'",
        shell=True, check=True
    )
    
    # Confirm a pooled-Z PNG was produced in the figures directory.
    pngs = list(figs.glob("prop26_zcheck_*.png"))
    assert pngs, "expected pooled-Z PNG"
