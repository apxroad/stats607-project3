import pathlib, subprocess, sys

def test_parta_prior_runs(tmp_path):
    # Create an output directory (not strictly needed by the CLI, but tidy).
    out = tmp_path / "figures"
    out.mkdir(parents=True, exist_ok=True)
    
    # Run the Part A CLI with tiny M,N so the test is fast.
    # Here n=0 triggers *prior* panels (no observed data).
    cmd = (
        "python -m src_cli.parta_panels "
        "--base uniform --t 0.5 --alpha 5 --n 0 --M 50 --N 50 --seed 1"
    )
    subprocess.run(cmd, shell=True, check=True)
    
    # Confirm at least one prior panel image was written in the standard location.
    figs = list(pathlib.Path("results/figures").glob("post_panels_cont_n0_*.png"))
    assert figs, "expected a prior panel PNG to be written"
