# src_cli/simulate.py
from __future__ import annotations
import os, argparse, yaml
from pathlib import Path
from src.simulation import run_stream

def main():
    ap = argparse.ArgumentParser(description="Run simulations from config")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    raw_dir = cfg["io"]["raw_dir"]
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    Ns      = cfg["n"]
    reps    = int(cfg["reps"])
    J       = int(cfg["grid"]["J"])
    tmin    = float(cfg["grid"]["tmin"])
    tmax    = float(cfg["grid"]["tmax"])
    methods = list(cfg["methods"])
    record_every = int(cfg["metrics"]["record_every"])

    base_seed = int(cfg["seeding"]["base"])
    method_params = cfg.get("method_params", {})  

    for n in Ns:
        for rep in range(reps):
            seed = base_seed + 1000 * int(n) + int(rep)
            for method in methods:
                params = method_params.get(method, {}) 
                out = os.path.join(raw_dir, f"{method}_n{int(n)}_rep{int(rep)}.parquet")
                print(f"[simulate] method={method} n={n} rep={rep} seed={seed} -> {out}")
                run_stream(
                    method_name=method,
                    n=int(n),
                    J=J,
                    tmin=tmin,
                    tmax=tmax,
                    record_every=record_every,
                    seed=seed,
                    out_path=out,
                    **params,                
                )

if __name__ == "__main__":
    main()
