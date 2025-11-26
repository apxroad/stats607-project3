from pathlib import Path
import pandas as pd
from src.simulation import run_stream

def test_reproducible_output(tmp_path: Path):
    # Prepare two distinct output files in a temporary directory.
    out1 = tmp_path / "a.parquet"
    out2 = tmp_path / "b.parquet"

    # Tiny Pólya run (Uniform base, to match our config).
    # Using identical kwargs (including seed) should yield identical outputs.
    kwargs = dict(
        method_name="polya_dp",
        n=120,
        J=20,
        tmin=0.0,
        tmax=1.0,
        record_every=10,
        seed=123,
        alpha=5.0,
        base="uniform",
    )

    # Run the stream twice with the same random seed and parameters.
    run_stream(out_path=str(out1), **kwargs)
    run_stream(out_path=str(out2), **kwargs)

    # Read back the parquet files and compare.
    df1 = pd.read_parquet(out1, engine="fastparquet")
    df2 = pd.read_parquet(out2, engine="fastparquet")
    # exact equality with fixed RNG + same code path
    assert df1.equals(df2)
