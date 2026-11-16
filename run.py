"""Einstiegspunkt des Proof of Concept (Kommandozeile).

Beispiele:
  python run.py download
  python run.py all --run lauf-02 --seeds 0 1 2 3 4 --timesteps 300000
  python run.py evaluate --run lauf-02
  streamlit run dashboard.py          Dashboard im Browser

Ohne --run wird der Lauf "standard" mit den Standardwerten verwendet.
"""
import argparse
import json
import traceback

import pandas as pd

from poc import config, data, pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "train", "evaluate", "all"], nargs="?", default="all")
    ap.add_argument("--run", default="standard")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--algorithm", choices=["PPO", "A2C", "DQN"])
    ap.add_argument("--fee", type=float)
    a = ap.parse_args()

    if a.cmd == "download":
        data.check(data.download())
        return

    config.migrate_legacy()
    cfg = config.load_run_config(a.run)
    for key in ("seeds", "timesteps", "algorithm", "fee"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    run_dir = config.save_run_config(a.run, cfg)
    pipeline._write_status(run_dir, state="gestartet", progress=0.0, message="")
    try:
        df = data.load()
        sections = pipeline.prepare(df, cfg)
        if a.cmd in ("train", "all"):
            pipeline.train(sections, cfg, run_dir)
        if a.cmd in ("evaluate", "all"):
            pipeline._write_status(run_dir, state="auswertung", progress=1.0)
            pd.set_option("display.float_format", "{:.3f}".format)
            for name in ("validation", "test"):
                print(f"\n=== {pipeline.SECTION_LABELS[name]} ===")
                print(pipeline.evaluate(sections, cfg, run_dir, name).to_string(), flush=True)
        pipeline._write_status(run_dir, state="fertig", progress=1.0)
        print(f"\nErgebnisse in {run_dir}")
    except Exception as exc:  # Status für das Dashboard festhalten
        pipeline._write_status(run_dir, state="fehler", message=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
