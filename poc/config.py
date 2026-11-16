"""Zentrale Konfiguration des Proof of Concept.

DEFAULTS enthält alle Parameter, die ein Experiment bestimmen. Jeder Lauf
speichert seine eigene Kopie davon in runs/<name>/config.json, damit
Ergebnisse immer ihrer Konfiguration zugeordnet bleiben.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"

# Daten (für alle Läufe gleich)
SYMBOL = "BTCUSDT"
INTERVAL = "1d"
FIRST_MONTH = "2017-08"   # BTCUSDT-Historie bei Binance beginnt am 17.08.2017
LAST_MONTH = "2025-12"

DEFAULTS = {
    "algorithm": "PPO",            # PPO, A2C oder DQN (diskreter Aktionsraum)
    "train": ["2017-09-01", "2022-12-31"],
    "validation": ["2023-01-01", "2023-12-31"],
    "test": ["2024-01-01", "2025-12-31"],
    "window": 30,                  # Anzahl vergangener Tage im State
    "fee": 0.001,                  # 0.1 % pro gehandeltem Volumen (Binance Spot)
    "episode_length": 365,         # Trainingsepisoden in Tagen, zufälliger Startpunkt
    "annualization": 365,          # Bitcoin wird an 365 Tagen gehandelt
    "seeds": [0, 1, 2],            # PoC: drei Seeds. Masterarbeit: mindestens zehn.
    "timesteps": 200_000,
}

# Rückwärtskompatible Einzelwerte
WINDOW = DEFAULTS["window"]
FEE = DEFAULTS["fee"]
ANNUALIZATION = DEFAULTS["annualization"]


def run_dir(name: str) -> Path:
    return RUNS_DIR / name


def save_run_config(name: str, cfg: dict) -> Path:
    d = run_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return d


def load_run_config(name: str) -> dict:
    path = run_dir(name) / "config.json"
    cfg = dict(DEFAULTS)
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    return cfg


def list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    runs = [p for p in RUNS_DIR.iterdir() if (p / "config.json").exists()]
    return [p.name for p in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)]


def migrate_legacy() -> None:
    """Ergebnisse aus der ersten PoC-Version (direkt in results/) als Lauf übernehmen."""
    legacy_models = RESULTS_DIR / "models"
    target = run_dir("lauf-01-ppo-standard")
    if not legacy_models.exists() or target.exists():
        return
    import shutil
    target.mkdir(parents=True)
    for item in list(RESULTS_DIR.iterdir()):
        if item.name in ("runs",):
            continue
        shutil.move(str(item), str(target / item.name))
    for f in target.glob("*_ppo_seed*.csv"):   # alte Dateinamen vereinheitlichen
        f.rename(f.with_name(f.name.replace("_ppo_seed", "_seed")))
    save_run_config(target.name, dict(DEFAULTS))
    (target / "status.json").write_text(json.dumps({"state": "fertig", "message": "aus erster PoC-Version übernommen"}), encoding="utf-8")
