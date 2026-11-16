"""Lokales Dashboard für den Proof of Concept.

Start:  streamlit run dashboard.py
"""
import json
import subprocess
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from poc import config, pipeline

st.set_page_config(page_title="RL-Trading PoC", page_icon="📈", layout="wide")

RUNNING = ("gestartet", "training", "auswertung")
SECTION_NAMES = {"Test 2024–2025": "test", "Validierung 2023": "validation"}
PCT_COLS = ["Gesamtrendite", "Rendite p.a.", "Volatilität p.a.", "Max Drawdown", "Investitionsgrad"]
BH_COLOR = "#111111"
PALETTE = ["#2a6fdb", "#e8833a", "#2e9e5b", "#c0392b", "#8e44ad", "#16a085",
           "#d4a017", "#7f8c8d", "#e84393", "#00a8cc", "#6d4c41", "#3949ab"]

config.migrate_legacy()


# ---------------------------------------------------------------- Hilfsfunktionen
def data_path() -> Path:
    return config.DATA_DIR / f"{config.SYMBOL.lower()}_{config.INTERVAL}.csv"


@st.cache_data(show_spinner=False)
def load_prices(mtime: float) -> pd.DataFrame:
    return pd.read_csv(data_path(), index_col="date", parse_dates=True)


def read_status(run: str) -> dict:
    p = config.run_dir(run) / "status.json"
    if not p.exists():
        return {"state": "unbekannt"}
    s = json.loads(p.read_text(encoding="utf-8"))
    # Ohne Lebenszeichen seit 15 Minuten gilt ein Lauf als abgebrochen (z. B. Neustart des Rechners)
    if s.get("state") in RUNNING and time.time() - p.stat().st_mtime > 15 * 60:
        s["state"] = "abgebrochen"
    return s


def read_log(run: str, lines: int = 25) -> str:
    p = config.run_dir(run) / "log.txt"
    if not p.exists():
        return ""
    return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def any_running() -> str | None:
    for r in config.list_runs():
        if read_status(r).get("state") in RUNNING:
            return r
    return None


def start_run(name: str, cfg: dict):
    run_dir = config.save_run_config(name, cfg)
    (run_dir / "status.json").write_text(json.dumps({"state": "gestartet", "progress": 0.0}), encoding="utf-8")
    log = open(run_dir / "log.txt", "w", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    subprocess.Popen([sys.executable, "-u", str(config.ROOT / "run.py"), "all", "--run", name],
                     cwd=config.ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)


def section_files(run: str, section: str) -> dict:
    d = config.run_dir(run)
    cfg = config.load_run_config(run)
    files = {s: d / f"{section}_seed{s}.csv" for s in cfg["seeds"]}
    return {s: p for s, p in files.items() if p.exists()}


@st.cache_data(show_spinner=False)
def load_section(run: str, section: str, stamp: float):
    cfg = config.load_run_config(run)
    logs = {s: pd.read_csv(p, index_col=0, parse_dates=True) for s, p in section_files(run, section).items()}
    bh_path = config.run_dir(run) / f"{section}_buy_and_hold.csv"
    if bh_path.exists():
        bh = pd.read_csv(bh_path, index_col=0, parse_dates=True)
    else:  # ältere Läufe: Buy-and-Hold aus den Daten nachrechnen
        prices = load_prices(data_path().stat().st_mtime)
        sec = pipeline.prepare(prices, cfg)
        bh = pipeline.buy_and_hold(pipeline.make_env(sec[section], cfg))
    table = pd.read_csv(config.run_dir(run) / f"{section}_kennzahlen.csv", index_col=0)
    return cfg, logs, bh, table


@st.cache_data(show_spinner=False)
def random_sims(returns: tuple, holding: float, fee: float, n: int = 3000) -> np.ndarray:
    return pipeline.random_benchmark(np.array(returns), holding, fee, n=n)


def ch(n) -> str:
    """Ganze Zahl mit Schweizer Tausendertrennzeichen."""
    return f"{int(n):,}".replace(",", "'")


def fmt_table(t: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=t.index)
    for c in t.columns:
        if c in PCT_COLS:
            out[c] = t[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
        elif c == "Positionswechsel":
            out[c] = t[c].map(lambda v: f"{v:.0f}" if pd.notna(v) else "")
        else:
            out[c] = t[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    return out


def line_chart(series: dict, title: str, log_y=False, y_title="", pct=False) -> go.Figure:
    fig = go.Figure()
    i = 0
    for name, s in series.items():
        is_bh = name == "Buy-and-Hold"
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines",
                                 line=dict(width=3 if is_bh else 1.4,
                                           color=BH_COLOR if is_bh else PALETTE[i % len(PALETTE)])))
        i += 0 if is_bh else 1
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    fig.update_yaxes(title=y_title, type="log" if log_y else "linear",
                     tickformat=".0%" if pct else None)
    return fig


def run_label(run: str) -> str:
    cfg = config.load_run_config(run)
    state = read_status(run).get("state", "")
    mark = {"fertig": "✓", "fehler": "✗"}.get(state, "…" if state in RUNNING else "")
    return f"{mark} {run} ({cfg['algorithm']}, {len(cfg['seeds'])} Seeds)"


# ---------------------------------------------------------------- Seitenleiste
st.sidebar.title("RL-Trading PoC")
st.sidebar.caption("BTC/USDT, Tagesdaten, Long-only")
runs = config.list_runs()
selected = st.sidebar.selectbox("Lauf", runs, format_func=run_label) if runs else None
if selected:
    @st.fragment(run_every="3s")
    def sidebar_status():
        s = read_status(selected)
        state = s.get("state", "–")
        st.markdown(f"**Status:** {state}")
        if state in RUNNING:
            st.progress(float(s.get("progress", 0)),
                        text=f"Seed {s.get('seed', '–')} · {float(s.get('progress', 0)):.0%}")

    with st.sidebar:
        sidebar_status()
    cfg_sel = config.load_run_config(selected)
    with st.sidebar.expander("Konfiguration des Laufs"):
        st.markdown(f"""
| | |
|---|---|
| Algorithmus | {cfg_sel['algorithm']} |
| Seeds | {', '.join(map(str, cfg_sel['seeds']))} |
| Schritte pro Seed | {ch(cfg_sel['timesteps'])} |
| Gebühr | {cfg_sel['fee']:.2%} |
| Fenster | {cfg_sel['window']} Tage |
| Episodenlänge | {cfg_sel['episode_length']} Tage |
""")
if st.sidebar.button("Aktualisieren", width="stretch"):
    st.cache_data.clear()
    st.rerun()

tab_res, tab_new, tab_train, tab_cmp, tab_data = st.tabs(
    ["Ergebnisse", "Neuer Lauf", "Training", "Vergleich", "Daten"])

# ---------------------------------------------------------------- Ergebnisse
with tab_res:
    if not selected:
        st.info("Noch kein Lauf vorhanden. Starte einen im Tab «Neuer Lauf».")
    else:
        sec_label = st.radio("Abschnitt", list(SECTION_NAMES), horizontal=True)
        section = SECTION_NAMES[sec_label]
        kpath = config.run_dir(selected) / f"{section}_kennzahlen.csv"
        if not kpath.exists():
            st.info("Für diesen Lauf liegen noch keine Ergebnisse vor.")
        elif not data_path().exists():
            st.warning("Kursdaten fehlen. Lade sie im Tab «Daten».")
        else:
            cfg, logs, bh, table = load_section(selected, section, kpath.stat().st_mtime)
            algo = cfg["algorithm"]
            agents = table.loc[[i for i in table.index if "Seed" in i]]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe Buy-and-Hold", f"{table.loc['Buy-and-Hold', 'Sharpe Ratio']:.2f}")
            c2.metric(f"Sharpe {algo} (Mittel)", f"{agents['Sharpe Ratio'].mean():.2f}",
                      f"{agents['Sharpe Ratio'].mean() - table.loc['Buy-and-Hold', 'Sharpe Ratio']:+.2f}")
            c3.metric("Rendite Buy-and-Hold", f"{table.loc['Buy-and-Hold', 'Gesamtrendite']:.0%}")
            c4.metric(f"Rendite {algo} (Spanne)",
                      f"{agents['Gesamtrendite'].min():.0%} bis {agents['Gesamtrendite'].max():.0%}")

            st.subheader("Kennzahlen nach Kosten")
            st.dataframe(fmt_table(table), width="stretch")
            st.caption("Sharpe und Sortino ohne risikolosen Zins, annualisiert mit "
                       f"{cfg['annualization']} Tagen. Gebühr {cfg['fee']:.2%} pro Positionswechsel.")

            curves = {"Buy-and-Hold": bh["value"]} | {f"{algo} Seed {s}": l["value"] for s, l in logs.items()}
            st.plotly_chart(line_chart(curves, "Portfolioentwicklung (Start = 1)", log_y=True,
                                       y_title="Portfoliowert, log. Skala"), width="stretch")

            dd = {k: v / v.cummax() - 1 for k, v in curves.items()}
            st.plotly_chart(line_chart(dd, "Drawdown", y_title="Rückgang vom Höchststand", pct=True),
                            width="stretch")

            st.subheader("Positionen")
            pos = pd.DataFrame({f"Seed {s}": l["position"] for s, l in logs.items()})
            fig = go.Figure(go.Heatmap(z=pos.T.values, x=pos.index, y=pos.columns,
                                       colorscale=[[0, "#eeeeee"], [1, "#2a6fdb"]], showscale=False,
                                       hovertemplate="%{y}, %{x|%d.%m.%Y}: %{z}<extra></extra>"))
            fig.update_layout(height=60 + 40 * len(pos.columns), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Blau = investiert, grau = nicht investiert.")

            st.subheader("Handelsverhalten und Kosten")
            rows = []
            for s, l in logs.items():
                cost_factor = (1 - l["cost"]).prod()
                net = (1 + l["return"]).prod()
                rows.append({"Seed": s,
                             "Positionswechsel": int(table.loc[f"{algo} Seed {s}", "Positionswechsel"]),
                             "Mittlere Haltedauer (Tage)": round(pipeline.mean_holding_period(l["position"]), 1),
                             "Kosten in % der Rendite": f"{1 - cost_factor:.1%}",
                             "Endwert ohne Kosten": round(net / cost_factor, 2),
                             "Endwert nach Kosten": round(net, 2)})
            st.dataframe(pd.DataFrame(rows).set_index("Seed"), width="stretch")

            st.subheader("Vergleich mit Zufallsstrategien")
            holding = float(np.mean([pipeline.mean_holding_period(l["position"]) for l in logs.values()]))
            market = bh["return"].reset_index(drop=True).copy()  # Marktrendite pro Tag
            market.iloc[0] = (1 + market.iloc[0]) / (1 - cfg["fee"]) - 1   # Kaufgebühr herausrechnen
            sims = random_sims(tuple(market.tolist()), holding, cfg["fee"])
            fig = go.Figure(go.Histogram(x=sims, nbinsx=60, marker_color="#bbbbbb", name="Zufallsstrategien"))
            for i, (s, l) in enumerate(logs.items()):
                v = l["value"].iloc[-1]
                fig.add_vline(x=v, line_color=PALETTE[i % len(PALETTE)], line_width=2,
                              annotation_text=f"Seed {s}", annotation_position="top")
            fig.add_vline(x=bh["value"].iloc[-1], line_color=BH_COLOR, line_dash="dash", line_width=2,
                          annotation_text="Buy-and-Hold", annotation_position="top left")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), bargap=0.05,
                              xaxis_title="Endwert (Start = 1)", yaxis_title="Anzahl")
            st.plotly_chart(fig, width="stretch")
            share = {f"Seed {s}": f"{(sims >= l['value'].iloc[-1]).mean():.1%}" for s, l in logs.items()}
            st.caption(f"{ch(len(sims))} Zufallsstrategien mit gleicher mittlerer Haltedauer "
                       f"({holding:.1f} Tage) und gleicher Gebühr. Anteil der Zufallsstrategien, "
                       "die mindestens so gut abschneiden: " + ", ".join(f"{k} {v}" for k, v in share.items())
                       + ". Ein hoher Anteil heisst: Das Ergebnis ist von Zufall kaum zu unterscheiden.")

            st.subheader("Lernkurven")
            mon_dir = config.run_dir(selected) / "monitor"
            mons = sorted(mon_dir.glob("seed*.monitor.csv")) if mon_dir.exists() else []
            if not mons:
                st.caption("Für diesen Lauf wurden keine Lernkurven aufgezeichnet "
                           "(erst ab der Dashboard-Version verfügbar).")
            else:
                fig = go.Figure()
                for i, m in enumerate(mons):
                    ep = pd.read_csv(m, skiprows=1)
                    y = np.exp(ep["r"]).rolling(20, min_periods=1).mean()
                    fig.add_trace(go.Scatter(x=ep["l"].cumsum(), y=y, mode="lines",
                                             name=m.name.split(".")[0].replace("seed", "Seed "),
                                             line=dict(color=PALETTE[i % len(PALETTE)])))
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10),
                                  title="Endwert pro Trainingsepisode (gleitender Mittelwert über 20 Episoden)",
                                  xaxis_title="Trainingsschritte", yaxis_title="Endwert der Episode")
                st.plotly_chart(fig, width="stretch")
                st.caption("Steigt die Kurve am Ende noch deutlich, war das Training zu kurz.")

# ---------------------------------------------------------------- Neuer Lauf
with tab_new:
    st.subheader("Neuen Lauf starten")
    busy = any_running()
    if not data_path().exists():
        st.warning("Zuerst die Kursdaten im Tab «Daten» laden.")
    elif busy:
        st.info(f"Lauf «{busy}» ist noch aktiv. Ein neuer Lauf kann erst danach gestartet werden.")
    else:
        d = config.DEFAULTS
        with st.form("new_run"):
            c1, c2 = st.columns(2)
            algorithm = c1.selectbox("Algorithmus", ["PPO", "A2C", "DQN"])
            n_seeds = c2.number_input("Anzahl Seeds", 1, 30, len(d["seeds"]))
            timesteps = c1.number_input("Trainingsschritte pro Seed", 10_000, 2_000_000, d["timesteps"], step=50_000)
            fee = c2.number_input("Gebühr pro Positionswechsel (%)", 0.0, 1.0, d["fee"] * 100, step=0.05, format="%.2f")
            window = c1.number_input("Fenster (Tage im State)", 5, 120, d["window"])
            episode = c2.number_input("Länge Trainingsepisode (Tage)", 60, 2000, d["episode_length"], step=30)
            n = len(config.list_runs()) + 1
            name = st.text_input("Name des Laufs", f"lauf-{n:02d}")
            st.caption(f"Zeiträume: Training {d['train'][0]} bis {d['train'][1]}, "
                       f"Validierung {d['validation'][0]} bis {d['validation'][1]}, "
                       f"Test {d['test'][0]} bis {d['test'][1]} (in poc/config.py änderbar).")
            submitted = st.form_submit_button("Lauf starten", type="primary")
        if submitted:
            name = name.strip().replace(" ", "-")
            if not name or (config.run_dir(name) / "config.json").exists():
                st.error("Name ist leer oder bereits vergeben.")
            else:
                cfg = dict(d, algorithm=algorithm, seeds=list(range(int(n_seeds))), timesteps=int(timesteps),
                           fee=fee / 100, window=int(window), episode_length=int(episode))
                start_run(name, cfg)
                st.success(f"Lauf «{name}» gestartet. Fortschritt im Tab «Training».")
                st.cache_data.clear()

# ---------------------------------------------------------------- Training
with tab_train:
    if not selected:
        st.info("Kein Lauf ausgewählt.")
    else:
        @st.fragment(run_every="3s")
        def live():
            s = read_status(selected)
            state = s.get("state", "unbekannt")
            st.markdown(f"**Lauf:** {selected}  ·  **Status:** {state}  ·  zuletzt aktualisiert {s.get('updated', '–')}")
            if state in RUNNING:
                st.progress(float(s.get("progress", 0)),
                            text=f"Seed {s.get('seed', '–')}  ·  {float(s.get('progress', 0)):.0%}")
            if state == "abgebrochen":
                st.warning("Seit 15 Minuten kein Fortschritt mehr. Der Lauf wurde vermutlich abgebrochen. "
                           "Starte ihn unter neuem Namen erneut.")
            if state == "fehler":
                st.error(s.get("message", "Unbekannter Fehler, siehe Protokoll."))
            log = read_log(selected)
            if log:
                st.code(log, language="text")
            elif state in RUNNING:
                st.caption("Training läuft. Das Protokoll erhält nach jedem abgeschlossenen Seed einen Eintrag.")
            else:
                st.caption("Für diesen Lauf gibt es kein Protokoll.")
            if state == "fertig":
                st.caption("Lauf abgeschlossen. Ergebnisse im Tab «Ergebnisse» (ggf. «Aktualisieren»).")
        live()

# ---------------------------------------------------------------- Vergleich
with tab_cmp:
    done = [r for r in runs if read_status(r).get("state") == "fertig"]
    if len(done) < 1:
        st.info("Noch keine abgeschlossenen Läufe.")
    else:
        chosen = st.multiselect("Läufe", done, default=done[:4])
        sec_label = st.radio("Abschnitt ", list(SECTION_NAMES), horizontal=True, key="cmp_sec")
        section = SECTION_NAMES[sec_label]
        rows, points = [], []
        bh_sharpe = None
        for r in chosen:
            p = config.run_dir(r) / f"{section}_kennzahlen.csv"
            if not p.exists():
                continue
            t = pd.read_csv(p, index_col=0)
            cfg = config.load_run_config(r)
            ag = t.loc[[i for i in t.index if "Seed" in i]]
            bh_sharpe = t.loc["Buy-and-Hold", "Sharpe Ratio"]
            rows.append({"Lauf": r, "Algorithmus": cfg["algorithm"], "Seeds": len(ag),
                         "Schritte": ch(cfg['timesteps']), "Gebühr": f"{cfg['fee']:.2%}",
                         "Sharpe Mittel": round(ag["Sharpe Ratio"].mean(), 2),
                         "Sharpe Std": round(ag["Sharpe Ratio"].std(), 2),
                         "Rendite Mittel": f"{ag['Gesamtrendite'].mean():.0%}",
                         "Max DD Mittel": f"{ag['Max Drawdown'].mean():.0%}",
                         "Wechsel Mittel": round(ag["Positionswechsel"].mean())})
            points += [{"Lauf": r, "Sharpe": v} for v in ag["Sharpe Ratio"]]
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Lauf"), width="stretch")
            pts = pd.DataFrame(points)
            fig = go.Figure()
            for i, r in enumerate(pts["Lauf"].unique()):
                y = pts.loc[pts["Lauf"] == r, "Sharpe"]
                fig.add_trace(go.Box(y=y, name=r, boxpoints="all", jitter=0.3, pointpos=0,
                                     marker_color=PALETTE[i % len(PALETTE)]))
            fig.add_hline(y=bh_sharpe, line_dash="dash", line_color=BH_COLOR,
                          annotation_text="Buy-and-Hold", annotation_position="top left")
            fig.update_layout(height=420, showlegend=False, yaxis_title="Sharpe Ratio (ein Punkt pro Seed)",
                              margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Daten
with tab_data:
    if not data_path().exists():
        st.warning("Noch keine Kursdaten vorhanden.")
        if st.button("Daten von Binance Vision laden", type="primary"):
            from poc import data
            with st.spinner("Lade Monatsdateien …"):
                data.download()
            st.cache_data.clear()
            st.rerun()
    else:
        prices = load_prices(data_path().stat().st_mtime)
        d = config.DEFAULTS
        fig = go.Figure(go.Scatter(x=prices.index, y=prices["close"], mode="lines",
                                   line=dict(color=BH_COLOR, width=1.2), name="Schlusskurs"))
        for key, color, label in [("train", "#2a6fdb", "Training"), ("validation", "#e8833a", "Validierung"),
                                  ("test", "#2e9e5b", "Test")]:
            fig.add_vrect(x0=d[key][0], x1=d[key][1], fillcolor=color, opacity=0.12, line_width=0,
                          annotation_text=label, annotation_position="top left")
        fig.update_layout(height=420, title="BTC/USDT Schlusskurs (log. Skala)",
                          margin=dict(l=10, r=10, t=50, b=10))
        fig.update_yaxes(type="log", title="USDT")
        st.plotly_chart(fig, width="stretch")
        rows = []
        for key, label in [("train", "Training"), ("validation", "Validierung"), ("test", "Test")]:
            c = prices["close"].loc[d[key][0]:d[key][1]]
            r = c.pct_change().dropna()
            rows.append({"Abschnitt": label, "Von": c.index[0].date(), "Bis": c.index[-1].date(),
                         "Tage": len(c), "Kursveränderung": f"{c.iloc[-1] / c.iloc[0] - 1:.0%}",
                         "Volatilität p.a.": f"{r.std() * np.sqrt(365):.0%}",
                         "Max Drawdown": f"{(c / c.cummax() - 1).min():.0%}"})
        st.dataframe(pd.DataFrame(rows).set_index("Abschnitt"), width="stretch")
        full = pd.date_range(prices.index.min(), prices.index.max(), freq="D")
        st.caption(f"{ch(len(prices))} Tage von {prices.index.min().date()} bis {prices.index.max().date()}, "
                   f"{len(full.difference(prices.index))} fehlende Tage. Quelle: data.binance.vision")
