from __future__ import annotations
import argparse, subprocess, yaml, shutil
from pathlib import Path

def run(cmd: list[str]):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="Sweep Monte Carlo reps (M) and rerun simulate+analyze.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--Ms", nargs="+", type=int, default=[50, 100, 200, 500])
    ap.add_argument("--t", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--level", type=float, default=0.95)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.config, "r"))
    Path("results").mkdir(exist_ok=True)

    for M in args.Ms:
        cfg = dict(base_cfg)  # shallow copy ok
        cfg["reps"] = M
        # isolate IO per M
        io = dict(cfg.get("io", {}))
        io["raw_dir"] = f"results/raw_M{M}"
        io["fig_dir"] = f"results/figures_M{M}"
        cfg["io"] = io

        # temp config file
        tmp_cfg = Path(f"config/_tmp_polya_M{M}.yaml")
        tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
        tmp_cfg_str = str(tmp_cfg)

        # simulate + analyze
        run(["python", "-m", "src_cli.simulate", "--config", tmp_cfg_str])
        run(["python", "-m", "src_cli.analyze_polya", "--config", tmp_cfg_str, "--t", *map(str, args.t), "--level", str(args.level)])

        # stash the CSV with an M-suffixed name
        src = Path("results/polya_checks.csv")
        dst = Path(f"results/polya_checks_M{M}.csv")
        if src.exists():
            shutil.copy2(src, dst)
            print("[ok]", dst)
        else:
            print("[warn] missing", src)

if __name__ == "__main__":
    main()
