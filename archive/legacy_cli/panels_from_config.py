from __future__ import annotations
import argparse, subprocess, yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    ns = cfg.get("n", [100])
    base = cfg.get("method_params",{}).get("polya_dp",{}).get("base","uniform")
    # run the panel script; it produces both prior and posterior panels
    cmd = ["python","-m","examples.polya_panel",
           "--base", base,
           "--ts","0.25","0.5","0.75",
           "--alphas","1","5","20",
           "--ns", *map(str, ns),
           "--reps","4000"]
    print("[panels] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
