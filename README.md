# Proof of Concept: RL-Agent für Bitcoin (DIFA-Projekt)

Ein RL-Agent lernt auf Bitcoin-Tagesdaten, ob er investiert ist oder nicht (long-only), und wird nach Transaktionskosten mit Buy-and-Hold und mit Zufallsstrategien verglichen. Läufe lassen sich über ein lokales Dashboard starten und auswerten.

## Installation (Windows)

Voraussetzung ist Python 3.11 oder neuer.

```powershell
cd D:\_temp\difa-poc
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Wer die erste Version bereits installiert hat, führt nur `pip install -r requirements.txt` erneut aus (neu: Streamlit und Plotly).

## Dashboard

```powershell
.venv\Scripts\activate
streamlit run dashboard.py
```

Der Browser öffnet sich automatisch unter http://localhost:8501. Beenden mit Ctrl+C im Terminal.

| Tab | Inhalt |
|---|---|
| Ergebnisse | Kennzahlen nach Kosten, Portfolioentwicklung, Drawdown, Positionen pro Seed, Handelsverhalten und Kosten, Vergleich mit Zufallsstrategien, Lernkurven |
| Neuer Lauf | Algorithmus (PPO, A2C, DQN), Anzahl Seeds, Trainingsschritte, Gebühr, Fenster und Episodenlänge wählen und Lauf starten |
| Training | Fortschritt und Protokoll des laufenden Trainings, aktualisiert sich alle drei Sekunden |
| Vergleich | Mehrere abgeschlossene Läufe nebeneinander, Streuung der Sharpe Ratio über die Seeds |
| Daten | Kursverlauf mit Trainings-, Validierungs- und Testabschnitt, Kennzahlen pro Abschnitt, Download der Daten |

Ein gestarteter Lauf trainiert im Hintergrund weiter, auch wenn die Browserseite geschlossen wird. Das Terminal mit Streamlit muss offen bleiben. Es läuft immer nur ein Lauf gleichzeitig.

Beim ersten Start übernimmt das Dashboard die Ergebnisse der ersten PoC-Version als Lauf `lauf-01-ppo-standard`. Die Dateien werden dafür von `results` nach `results\runs\lauf-01-ppo-standard` verschoben. Für diesen Lauf gibt es keine Lernkurven, weil die erste Version sie nicht aufgezeichnet hat.

## Kommandozeile

Alles, was das Dashboard kann, geht auch ohne Browser:

```powershell
python -m pytest tests                                   # Prüfungen
python run.py download                                   # Daten laden
python run.py all --run lauf-02 --seeds 0 1 2 3 4 5 6 7 8 9 --timesteps 300000
python run.py all --run lauf-03 --algorithm A2C
python run.py evaluate --run lauf-02                     # nur neu auswerten
```

## Aufbau

| Datei | Inhalt |
|---|---|
| `poc/config.py` | Standardwerte, Zeiträume, Verwaltung der Läufe |
| `poc/data.py` | Download der Tagesdaten BTCUSDT von Binance Vision, Qualitätsprüfung |
| `poc/features.py` | Zehn Merkmale aus Kurs und Volumen, Standardisierung nur auf Trainingsdaten |
| `poc/env.py` | Trading-Umgebung nach Gymnasium-Standard |
| `poc/metrics.py` | Gesamtrendite, Sharpe, Sortino, Max Drawdown, Positionswechsel |
| `poc/pipeline.py` | Aufteilung, Training, Auswertung, Zufallsbenchmark |
| `run.py` | Kommandozeile |
| `dashboard.py` | Streamlit-Dashboard |
| `tests/test_poc.py` | Prüfungen gegen Look-Ahead Bias und Rechenfehler, Durchlauf mit Testdaten |

Jeder Lauf liegt in `results\runs\<name>` mit `config.json`, Modellen, Lernkurven (`monitor`), Tagesverläufen pro Seed und Abschnitt, Kennzahlen und Grafiken. Die PNG-Grafiken eignen sich direkt für den Bericht.

## Versuchsaufbau (Standardwerte)

- **Daten:** BTC/USDT, Tagesdaten, Binance Spot, 09/2017 bis 12/2025
- **Aufteilung:** Training 2017–2022, Validierung 2023, Test 2024–2025 (zeitlich getrennt)
- **State:** zehn Merkmale über die letzten 30 Tage plus aktuelle Position
- **Aktion:** 0 = nicht investiert, 1 = voll investiert (diskret, long-only)
- **Reward:** logarithmische Tagesrendite nach 0.1 % Gebühr pro Positionswechsel
- **Training:** Stable-Baselines3 mit Standardparametern, 200'000 Schritte, Episoden von 365 Tagen mit zufälligem Start
- **Benchmarks:** Buy-and-Hold mit einmaliger Kaufgebühr; Zufallsstrategien mit gleicher mittlerer Haltedauer und Gebühr wie der Agent

Der Agent handelt zum Schlusskurs ohne Slippage. Sharpe und Sortino sind ohne risikolosen Zins berechnet und mit 365 Tagen annualisiert.

## Bewusste Vereinfachungen gegenüber der Masterarbeit

- Keine Hyperparameter-Optimierung
- Fester Datensplit statt Walk-Forward-Analyse
- Nur Bitcoin, nur diskreter Aktionsraum (SAC daher nicht enthalten)
- Keine formalen Signifikanztests
