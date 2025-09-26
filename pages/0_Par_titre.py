# pages/01-Fiche_Titre.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# scipy (optionnel) pour Student-t, tests, etc.
try:
    from scipy import stats as sstats
except Exception:
    sstats = None

st.set_page_config(page_title="Fiche Titre — Dashboard complet", layout="wide")

# ====== Style & constantes ======
PRICE_H, CUM_H, VOL_H, CANDLE_H = 380, 300, 220, 880
AX_LABEL_F, AX_TITLE_F = 12, 13
DIV_POINT_SIZE = 200

PALETTE = {
    "c1":"#FC4E00", "c2":"#8BA503", "c3":"#F3BB00", "c4":"#EB9F04", "c5":"#BD3100",
    "blue":"#2563eb", "teal":"#10b981", "violet":"#7c3aed", "slate":"#64748b"
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&display=swap');
:root{{ --c1:{PALETTE['c1']}; --c2:{PALETTE['c2']}; --c3:{PALETTE['c3']}; --c4:{PALETTE['c4']}; --c5:{PALETTE['c5']};
       --txt:#0f172a; --muted:#64748b; }}
html, body, [class*="css"] {{ font-family:'Montserrat',sans-serif !important; color:var(--txt); }}
h1,h2,h3,h4 {{ font-weight:800; }}
.hero{{font-size:2rem; line-height:1.1; font-weight:800; margin:2px 0 0 0;}}
.sub{{color:#64748b; margin-bottom: 6px;}}
.kpi-label{{ color:#475569; font-size:.85rem; margin-bottom:2px;}}
.kpi-value{{ font-weight:800; font-size:1.2rem;}}
.kpi-main{{ color:var(--c1);}}
.kpi-pos{{ color:var(--c2);}} .kpi-neg{{ color:var(--c5);}} .kpi-cap{{ color:var(--c4);}} .kpi-perf{{ color:var(--c3);}}
.num{{ font-family:'Inter','Montserrat',sans-serif; font-variant-numeric: tabular-nums; }}
.hr{{ height:1px; background:#e5e7eb; margin:10px 0 14px 0; }}
.note{{ color:#475569; font-size:.92rem; }}
.expl{{ color:#374151; font-size:.93rem; margin-top:8px; border-left:3px solid #e5e7eb; padding:8px 12px; background:#fafafa;}}
.expl b{{ color:#111827;}}
.param{{ color:#111827; font-weight:700; margin-top:2px; margin-bottom:4px; }}
.small{{ color:#6b7280; font-size:.85rem; }}
</style>
""", unsafe_allow_html=True)

# ---- helper pour explications
def explain_block(show, definition, interpretation, theory=None):
    if not show: 
        return
    html = "<div class='expl'>"
    if definition:
        html += f"<b>Définition.</b> {definition}<br>"
    if interpretation:
        html += f"<b>Interprétation.</b> {interpretation}"
    if theory:
        html += f"<br><b>Fondement théorique.</b> {theory}"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ====== Data (chargée dans Home) ======
data = st.session_state.get("data", {})
base_df    = data.get("base")     # Base_complète.csv
indices_df = data.get("indices")  # 60_Cours_indices.csv
ratios_df  = data.get("ratios")   # ratios.csv

if base_df is None:
    st.error("Charge d’abord Base_complète.csv dans l’onglet Home.")
    st.stop()

# Normalisation
base_df["Date"] = pd.to_datetime(base_df["Date"], errors="coerce")
if indices_df is not None:
    indices_df["Date"] = pd.to_datetime(indices_df["Date"], errors="coerce")

# ====== Fcts indicateurs ======
def fr_fmt(x, nd=0):
    if pd.isna(x): return "—"
    s = f"{x:,.{nd}f}"
    return s.replace(",", " ").replace(".", ",")

def fr_pct_from_pct(x, nd=2):
    if pd.isna(x): return "—"
    s = f"{x:.{nd}f}".replace(".", ",")
    return s + " %"

def axis_left(title):  return alt.Axis(title=title, orient="left",  labelFontSize=AX_LABEL_F, titleFontSize=AX_TITLE_F)
def axis_right(title=None): return alt.Axis(title=title, orient="right", labelFontSize=AX_LABEL_F, titleFontSize=AX_TITLE_F)

def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def rsi(s, period=14):
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    rs=ema(g,period)/ema(l,period).replace(0,np.nan)
    return 100 - (100/(1+rs))
def macd(s, fast=12, slow=26, signal=9):
    mf, ms = ema(s, fast), ema(s, slow)
    line = mf - ms; sig = ema(line, signal); hist = line - sig
    return line, sig, hist
def bollinger(s, period=20, nstd=2):
    mid=s.rolling(period, min_periods=1).mean()
    std=s.rolling(period, min_periods=1).std(ddof=0)
    return mid, mid+nstd*std, mid-nstd*std

def theil_sen_logprice(prices: pd.Series):
    y = np.log(prices.values.astype(float))
    x = np.arange(len(y)).astype(float)
    if sstats is not None and hasattr(sstats, "theilslopes"):
        slope, intercept, lo, hi = sstats.theilslopes(y, x)
        return slope, intercept, lo, hi, True
    # fallback robuste approx
    n = len(y); k = min(n, 400)
    idx = np.linspace(0, n-1, k).astype(int)
    xv, yv = idx.astype(float), y[idx]
    slopes = []
    for i in range(k):
        for j in range(i+1, k):
            dx = xv[j] - xv[i]
            if dx != 0:
                slopes.append((yv[j] - yv[i]) / dx)
    slope = np.median(slopes) if slopes else 0.0
    intercept = np.median(yv - slope * xv)
    return slope, intercept, np.nan, np.nan, False

def mann_kendall_test(series: pd.Series):
    x = np.asarray(series.values, dtype=float)
    n = len(x)
    if n < 10: return np.nan, np.nan
    S = 0
    for i in range(n-1):
        S += np.sum(np.sign(x[i+1:] - x[i]))
    varS = (n*(n-1)*(2*n+5)) / 18.0
    if S > 0:   Z = (S - 1) / np.sqrt(varS)
    elif S < 0: Z = (S + 1) / np.sqrt(varS)
    else:       Z = 0.0
    p = 2 * (1 - sstats.norm.cdf(abs(Z))) if sstats is not None else np.nan
    return Z, p

def adx14(df_ohlc: pd.DataFrame):
    need = {"High","Low","Close"}
    if not need.issubset(df_ohlc.columns): return None
    high = df_ohlc["High"].values.astype(float)
    low  = df_ohlc["Low"].values.astype(float)
    close= df_ohlc["Close"].values.astype(float)
    n = len(close)
    if n < 15: return None
    tr = np.zeros(n); plusDM = np.zeros(n); minusDM = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        up, down = high[i]-high[i-1], low[i-1]-low[i]
        plusDM[i]  = up   if (up > down and up   > 0) else 0.0
        minusDM[i] = down if (down > up  and down > 0) else 0.0
    def wilder(arr, period=14):
        out = np.zeros_like(arr)
        out[period] = np.nansum(arr[1:period+1])
        for i in range(period+1, n):
            out[i] = out[i-1] - out[i-1]/period + arr[i]
        return out
    period = 14
    atr = wilder(tr, period)
    pdi = 100 * (wilder(plusDM, period) / atr)
    mdi = 100 * (wilder(minusDM, period) / atr)
    dx  = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = np.full_like(dx, np.nan)
    if n > period*2:
        adx[period*2] = np.nanmean(dx[period+1:period*2+1])
        for i in range(period*2+1, n):
            adx[i] = ((adx[i-1]*(period-1)) + dx[i]) / period
    return pd.Series(adx, index=df_ohlc.index, name="ADX14")

def rolling_levels(df: pd.DataFrame, n: int = 20):
    out = pd.DataFrame({"Date": df["Date"]})
    if "High" in df.columns and "Low" in df.columns:
        out["HighN"] = df["High"].rolling(n, min_periods=1).max()
        out["LowN"]  = df["Low"].rolling(n, min_periods=1).min()
    else:
        p = df.iloc[:,1]
        out["HighN"] = p.rolling(n, min_periods=1).max()
        out["LowN"]  = p.rolling(n, min_periods=1).min()
    return out

def cusum_changes(returns: pd.Series, k: float = 0.0, h_sigmas: float = 5.0):
    r = returns.dropna().values
    if len(r) < 10: return [], []
    sigma = np.std(r, ddof=0)
    if sigma == 0: return [], []
    h = h_sigmas * sigma
    s_pos = 0.0; s_neg = 0.0
    pos_idx = []; neg_idx = []
    for i, rt in enumerate(r):
        s_pos = max(0.0, s_pos + (rt - k))
        s_neg = min(0.0, s_neg + (rt + k))
        if s_pos > h:   pos_idx.append(i); s_pos = 0.0
        if s_neg < -h:  neg_idx.append(i); s_neg = 0.0
    return pos_idx, neg_idx

def hurst_exponent(prices: pd.Series):
    x = np.log(prices.dropna().values.astype(float))
    N = len(x)
    if N < 100: return np.nan
    lags = np.unique(np.logspace(np.log10(2), np.log10(min(N//5, 200)), num=20, dtype=int))
    taus = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        taus.append(np.sqrt(np.mean(diff**2)))
    lags = lags[~np.isnan(taus)]
    taus = np.array(taus)[~np.isnan(taus)]
    if len(lags) < 5: return np.nan
    slope = np.polyfit(np.log(lags), np.log(taus), 1)[0]
    return float(slope)

def fr_to_datetime(s):
    if pd.isna(s): return pd.NaT
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        try:
            return pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
        except Exception:
            return pd.NaT

# ====== Contrôles globaux ======
tickers = sorted(base_df["Ticker"].dropna().unique())
sel_ticker = st.radio("Titre", options=tickers, horizontal=True)

use_cols = [c for c in ["Cours Ajuste","Close","Cours Normal"] if c in base_df.columns]
if not use_cols:
    st.error("Série de prix introuvable."); st.stop()
price_col = use_cols[0]

df_t = base_df[base_df["Ticker"] == sel_ticker].sort_values("Date").dropna(subset=["Date"]).copy()
if df_t.empty: 
    st.warning("Aucune donnée pour ce ticker."); 
    st.stop()

min_date, max_date = df_t["Date"].min(), df_t["Date"].max()

# --- NOUVEAU: préréglages de période avec 1w / WTD / MTD ---
presets = ["1w", "WTD", "MTD", "1m", "3m", "6m", "YTD", "1y", "2y", "3y", "5y", "10y", "Custom"]
default_idx = presets.index("YTD")  # garde YTD par défaut
choice = st.radio("Période", options=presets, horizontal=True, index=default_idx)

def clamp(d): 
    return min(max(d, min_date), max_date)

if choice == "Custom":
    start_default, end_default = min_date.to_pydatetime(), max_date.to_pydatetime()
    start, end = st.slider(
        "Sélection (Custom)", 
        min_value=start_default, max_value=end_default,
        value=(start_default, end_default), 
        step=datetime.timedelta(days=1)
    )
    start, end = pd.to_datetime(start), pd.to_datetime(end)

else:
    # Semaine/mois/année à date et fenêtres fixes
    if choice == "WTD":
        # Lundi de la semaine ISO contenant max_date
        start = max_date.normalize() - pd.Timedelta(days=max_date.weekday())
    elif choice == "MTD":
        start = pd.Timestamp(year=max_date.year, month=max_date.month, day=1)
    elif choice == "YTD":
        start = pd.Timestamp(year=max_date.year, month=1, day=1)
    elif choice == "1w":
        start = max_date - pd.Timedelta(days=7)
    else:
    # Fenêtres FIXES "date à date" (ancrées sur max_date)
        offsets = {
            "1w":  pd.DateOffset(weeks=1),
            "1m":  pd.DateOffset(months=1),
            "3m":  pd.DateOffset(months=3),
            "6m":  pd.DateOffset(months=6),
            "1y":  pd.DateOffset(years=1),
            "2y":  pd.DateOffset(years=2),
            "3y":  pd.DateOffset(years=3),
            "5y":  pd.DateOffset(years=5),
            "10y": pd.DateOffset(years=10),
        }
        start = max_date - offsets[choice]
        # borne dans l'historique disponible

    start = clamp(start)
    end = max_date

# Fenêtre filtrée commune à toute la page
df_win = df_t[(df_t["Date"] >= start) & (df_t["Date"] <= end)].copy()
if df_win.empty:
    st.info("Aucune donnée dans l’intervalle sélectionné.")
    st.stop()


# Switch global pour explications
show_expl = st.toggle("Afficher les explications (définitions, interprétations, théorie)", value=True)

# ====== Carte d'identité & KPIs ======
last = df_win.tail(1)
nom_soc = last.get("Société", last.get("Company name", pd.Series(["—"]))).iloc[0]
secteur = last.get("Secteur d'activité", last.get("Sector", pd.Series(["—"]))).iloc[0]
pays    = last.get("Pays", last.get("Country", pd.Series(["—"]))).iloc[0]
intro   = last.get("Introduction à la BRVM", pd.Series([""])).iloc[0]
nb_tit  = last.get("Nombre de titres", last.get("nb_actions_estime", pd.Series([np.nan]))).iloc[0]
flott   = last.get("Flottant", pd.Series([np.nan])).iloc[0]
site    = last.get("Site Web", pd.Series([""])).iloc[0]

st.markdown(f"<div class='hero'>{sel_ticker} — {nom_soc}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub'>{secteur} · {pays}</div>", unsafe_allow_html=True)
row1 = st.columns(5)
row1[0].markdown(f"<div class='kpi-label'>Intro BRVM</div><div class='kpi-value num'>{intro if isinstance(intro,str) and intro else '—'}</div>", unsafe_allow_html=True)
row1[1].markdown(f"<div class='kpi-label'>Nombre de titres</div><div class='kpi-value num'>{fr_fmt(nb_tit,0)}</div>", unsafe_allow_html=True)
row1[2].markdown(f"<div class='kpi-label'>Flottant</div><div class='kpi-value num'>{fr_fmt(flott,2)} %</div>", unsafe_allow_html=True)
if isinstance(site,str) and site.strip():
    link = site if site.startswith(("http://","https://")) else "http://"+site
    row1[3].markdown(f"<div class='kpi-label'>Site</div><div class='kpi-value'><a class='info' href='{link}' target='_blank'>{site}</a></div>", unsafe_allow_html=True)
else:
    row1[3].markdown(f"<div class='kpi-label'>Site</div><div class='kpi-value'>—</div>", unsafe_allow_html=True)

prix    = last[price_col].iloc[0]
var_pct = last.get("Variation", pd.Series([np.nan])).iloc[0]
capi    = last.get("Capitalisation", pd.Series([np.nan])).iloc[0]
first_p = df_win[price_col].iloc[0]
cum_pct = (prix/first_p - 1.0)*100.0

row2 = st.columns(4)
row2[0].markdown(f"<div class='kpi-label'>Cours ajusté</div><div class='kpi-value kpi-main num'>{fr_fmt(prix,0)}</div>", unsafe_allow_html=True)
row2[1].markdown(f"<div class='kpi-label'>Variation jour</div><div class='kpi-value {'kpi-pos' if (pd.notna(var_pct) and var_pct>=0) else 'kpi-neg'} num'>{fr_pct_from_pct(var_pct,2)}</div>", unsafe_allow_html=True)
row2[2].markdown(f"<div class='kpi-label'>Capitalisation</div><div class='kpi-value kpi-cap num'>{fr_fmt(capi,0)}</div>", unsafe_allow_html=True)
row2[3].markdown(f"<div class='kpi-label'>Rendement cumulé</div><div class='kpi-value {'kpi-pos' if cum_pct>=0 else 'kpi-neg'} num'>{fr_pct_from_pct(cum_pct,2)}</div>", unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ====== RÉSUMÉ EXÉCUTIF ======
def colored_pct(x, nd=2):
    if pd.isna(x): return "—"
    cls = 'kpi-pos' if x >= 0 else 'kpi-neg'
    return f"<span class='num {cls}'>{fr_pct_from_pct(x, nd)}</span>"

def colored_num(x, nd=0, pos_class='kpi-pos', neg_class='kpi-neg'):
    if pd.isna(x): return "—"
    cls = pos_class if x >= 0 else neg_class
    return f"<span class='num {cls}'>{fr_fmt(x, nd)}</span>"

# 1) Tendance (Theil–Sen + Mann–Kendall)
px_win = df_win[["Date", price_col]].rename(columns={price_col:"P"}).dropna().sort_values("Date")
mu_ann_pct = np.nan; mk_z = np.nan; mk_p = np.nan; trend_label = "Neutre"
if len(px_win) >= 30:
    slope, intercept, lo, hi, robust = theil_sen_logprice(px_win["P"])
    mu_ann_pct = (np.exp(slope*252)-1.0)*100.0
    mk_z, mk_p = mann_kendall_test(px_win["P"])
    if pd.notna(mu_ann_pct) and pd.notna(mk_p):
        if (mu_ann_pct > 5) and (mk_p < 0.10):
            trend_label = "Haussière"
        elif (mu_ann_pct < -5) and (mk_p < 0.10):
            trend_label = "Baissière"
        else:
            trend_label = "Neutre"

# 2) Performance
def perf_between(df_all, t0, t1):
    d = df_all[(df_all["Date"]>=t0) & (df_all["Date"]<=t1)]
    if d.empty: return np.nan
    p0 = d.iloc[0][price_col]; p1 = d.iloc[-1][price_col]
    return (p1/p0 - 1.0)*100.0 if (pd.notna(p0) and p0!=0) else np.nan

perf_win = (df_win[price_col].iloc[-1]/df_win[price_col].iloc[0] - 1.0)*100.0
start_ytd = pd.Timestamp(year=max_date.year, month=1, day=1)
start_ytd = max(start_ytd, min_date)
perf_ytd  = perf_between(df_t, start_ytd, max_date)
start_1y  = max_date - pd.Timedelta(days=365)
start_1y  = max(start_1y, min_date)
perf_1y   = perf_between(df_t, start_1y, max_date)

# 3) Risque (vol, Sharpe~0%, DD max)
rs_tmp = df_win[["Date", price_col]].rename(columns={price_col:"P"}).sort_values("Date")
rs_tmp["r"] = np.log(rs_tmp["P"]/rs_tmp["P"].shift(1))
mu_d = rs_tmp["r"].mean(); sd_d = rs_tmp["r"].std(ddof=0)
vol_ann = sd_d*np.sqrt(252)*100 if pd.notna(sd_d) else np.nan
sharpe  = (mu_d/sd_d)*np.sqrt(252) if (pd.notna(mu_d) and sd_d and sd_d>0) else np.nan
dd_tmp = px_win.copy()
dd_tmp["cummax"]   = dd_tmp["P"].cummax()
dd_tmp["drawdown"] = (dd_tmp["P"]/dd_tmp["cummax"] - 1.0)*100
max_dd = dd_tmp["drawdown"].min() if not dd_tmp.empty else np.nan

# 4) Liquidité (volume, turnover, Amihud)
vol_series = df_win.get("Volume", df_win.get("Volume Normal Total", df_win.get("Volume Ajuste Total", pd.Series(np.nan, index=df_win.index))))
vol_median = np.nanmedian(vol_series.values) if vol_series is not None else np.nan
flot_v = (last.get("Flottant", pd.Series([np.nan]))).iloc[0]
nb_v   = (last.get("Nombre de titres", last.get("nb_actions_estime", pd.Series([np.nan])))).iloc[0]
denom  = (flot_v/100.0)*nb_v if (pd.notna(flot_v) and pd.notna(nb_v) and flot_v>0 and nb_v>0) else np.nan
turnover_mean = np.nan
if pd.notna(denom):
    turnover_mean = np.nanmean(vol_series.values/denom)
# Amihud médian
amihud_med = np.nan
try:
    ptmp = df_win[price_col].astype(float).values
    rabs = np.abs(np.diff(np.log(ptmp)))
    v2   = vol_series.shift(1).values[1:]
    am   = rabs / (ptmp[1:]*np.where(v2==0, np.nan, v2))
    amihud_med = np.nanmedian(am)
except Exception:
    pass

# 5) Relative (vs indice si dispo)
rel_txt = "n.d."
if indices_df is not None and not indices_df.empty:
    idx_df = indices_df.copy()
    cand_ticker_cols = ["Ticker","Symbole","Index","Indice"]
    idx_ticker_col = next((c for c in cand_ticker_cols if c in idx_df.columns), None)
    price_cols_idx = [c for c in ["Cours Ajuste","Close","Price","Dernier"] if c in idx_df.columns]
    if idx_ticker_col and price_cols_idx:
        idx_price_col = price_cols_idx[0]
        cs = sorted(idx_df[idx_ticker_col].dropna().astype(str).unique().tolist())
        default_idx = next((x for x in cs if str(x).upper() in {"BRVM-C","BRVM COMPOSITE","COMPOSITE","BRVMCOM"}), cs[0] if cs else None)
        ref_sym = default_idx
        ref = idx_df[idx_df[idx_ticker_col].astype(str)==str(ref_sym)].sort_values("Date")
        ref = ref[["Date", idx_price_col]].rename(columns={idx_price_col:"I"}).dropna()
        if not ref.empty:
            t = df_win[["Date", price_col]].rename(columns={price_col:"P"}).copy()
            m = pd.merge_asof(t.sort_values("Date"), ref.sort_values("Date"), on="Date", direction="backward").dropna()
            if not m.empty:
                rs_base = (m["P"]/m["I"]) / (m["P"].iloc[0]/m["I"].iloc[0])
                delta = (rs_base.iloc[-1]-1.0)*100.0
                rel_txt = f"{'sur' if delta>=0 else 'sous'}-performance vs <b>{ref_sym}</b> : {colored_pct(delta)}"

# 6) Dividendes (DY12M, dernière ex-date, payout si dispo)
dy12 = np.nan; last_ex_date = "—"; last_div = np.nan; payout = np.nan
div_col = next((c for c in ["Dividende ajusté","Dividende ajusté (FCFA)","Dividende"] if c in df_t.columns), None)
if div_col:
    dy = df_t[["Date", price_col, div_col]].rename(columns={price_col:"P", div_col:"D"}).copy()
    dy["D"] = dy["D"].fillna(0.0); dy = dy.sort_values("Date")
    # DY 12M
    s12 = []
    for i in range(len(dy)):
        end_i = dy.iloc[i]["Date"]; start_i = end_i - pd.Timedelta(days=365)
        window = dy[(dy["Date"]>start_i) & (dy["Date"]<=end_i)]
        s12.append(window["D"].sum())
    dy["Div12M"] = s12
    if not dy.empty:
        prc = dy.iloc[-1]["P"]
        dy12 = (dy.iloc[-1]["Div12M"]/prc)*100 if prc and prc>0 else np.nan
    # Dernière ex-date
    ex_rows = dy[dy["D"]>0]
    if not ex_rows.empty:
        last_ex_date = ex_rows.iloc[-1]["Date"].strftime("%d/%m/%Y")
        last_div = ex_rows.iloc[-1]["D"]
    # Payout si BNPA dispo
    if (ratios_df is not None) and ("Bénéfice net par action (BNPA)" in ratios_df.columns):
        bnpa_row = ratios_df.loc[ratios_df["Ticker"].astype(str)==str(sel_ticker), "Bénéfice net par action (BNPA)"]
        if not bnpa_row.empty and pd.notna(bnpa_row.iloc[0]) and bnpa_row.iloc[0]!=0:
            payout = (dy.iloc[-1]["Div12M"]/bnpa_row.iloc[0])*100.0

# 7) Valorisation (Z-MAD vs secteur)
val_txt = "n.d."
if ratios_df is not None and not ratios_df.empty:
    r = ratios_df.copy()
    if "Secteur d'activité" in r.columns: r.rename(columns={"Secteur d'activité":"Secteur"}, inplace=True)
    for src,dst in [("Price Earning Ratio (PER)","PER"),("Price-to-Book Ratio (PBR)","PBR"),("Price-to-Sales (PSR)","PSR")]:
        if src in r.columns: r.rename(columns={src:dst}, inplace=True)
    if {"Ticker","Secteur"}.issubset(r.columns):
        sec_val = df_t["Secteur d'activité"].dropna().iloc[-1] if "Secteur d'activité" in df_t.columns and not df_t["Secteur d'activité"].dropna().empty else \
                  (df_t["Sector"].dropna().iloc[-1] if "Sector" in df_t.columns and not df_t["Sector"].dropna().empty else None)
        rf = r[r["Secteur"].astype(str)==str(sec_val)].copy() if sec_val is not None else r.copy()
        row_me = r[r["Ticker"].astype(str)==str(sel_ticker)].head(1)
        txts=[]
        for m in ["PER","PBR","PSR"]:
            if (m in rf.columns) and (m in row_me.columns) and not row_me.empty:
                med = rf[m].median(skipna=True); mad = (rf[m]-med).abs().median(skipna=True)
                x = row_me[m].iloc[0]
                z = (x-med)/(mad if mad>0 else np.nan)
                if pd.notna(x) and pd.notna(z):
                    txts.append(f"{m} {fr_fmt(x,2)} ({'+' if z>=0 else ''}{z:.2f}σ)")
        if txts:
            val_txt = " · ".join(txts)

# 8) Alertes/Régimes (squeeze, ruptures, Hurst, SMA)
alert_bits = []
if "Close" in df_t.columns:
    z = df_t[["Date","Close"]].dropna().sort_values("Date").copy()
    mid, up, low = bollinger(z["Close"], 20, 2)
    z["BW"] = (up-low)/mid
    z = z[(z["Date"]>=start)&(z["Date"]<=end)]
    if not z.empty and z["BW"].notna().any():
        p20 = np.nanpercentile(z["BW"], 20)
        last_bw = z["BW"].iloc[-1]
        if pd.notna(last_bw) and pd.notna(p20) and last_bw <= p20:
            alert_bits.append("squeeze (volatilité comprimée)")

r_eff = rs_tmp.dropna(subset=["r"]).reset_index(drop=True)
pos_idx, neg_idx = cusum_changes(r_eff["r"], k=0.0, h_sigmas=5.0) if not r_eff.empty else ([],[])
nb_breaks = len(pos_idx)+len(neg_idx)
if nb_breaks >= 1:
    alert_bits.append(f"{nb_breaks} rupture(s) CUSUM")

H = hurst_exponent(df_win[price_col])
if pd.notna(H):
    if H>0.55: alert_bits.append(f"Hurst {H:.2f} (persistance)")
    elif H<0.45: alert_bits.append(f"Hurst {H:.2f} (anti-persistance)")

sig_txt = ""
if "Close" in df_t.columns and len(df_t) >= 50:
    tmp = df_t[["Date","Close"]].sort_values("Date").copy()
    tmp["SMA20"] = tmp["Close"].rolling(20, min_periods=1).mean()
    tmp["SMA50"] = tmp["Close"].rolling(50, min_periods=1).mean()
    tsel = tmp[(tmp["Date"]>=start)&(tmp["Date"]<=end)]
    if not tsel.empty:
        last_sma20, last_sma50 = tsel["SMA20"].iloc[-1], tsel["SMA50"].iloc[-1]
        if pd.notna(last_sma20) and pd.notna(last_sma50):
            if last_sma20 > last_sma50: sig_txt = "SMA20 > SMA50 (bias haussier)"
            elif last_sma20 < last_sma50: sig_txt = "SMA20 < SMA50 (bias baissier)"
if sig_txt: alert_bits.append(sig_txt)

alerts = " · ".join(alert_bits) if alert_bits else "—"

# Rendu résumé
st.markdown("### Résumé exécutif")
resume = []
trend_str = f"**Tendance** : {trend_label}"
if pd.notna(mu_ann_pct): trend_str += f" (pente Theil–Sen ≈ {colored_pct(mu_ann_pct,1)})"
if pd.notna(mk_z):       trend_str += f", test Mann–Kendall z={mk_z:.2f}" + (f", p≈{mk_p:.3f}" if pd.notna(mk_p) else "")
resume.append(trend_str)

perf_str = f"**Performance** : {colored_pct(perf_win,2)} sur la fenêtre"
if pd.notna(perf_ytd): perf_str += f" · YTD {colored_pct(perf_ytd,2)}"
if pd.notna(perf_1y):  perf_str += f" · 1 an {colored_pct(perf_1y,2)}"
resume.append(perf_str)

risk_str = f"**Risque** : σ≈ {colored_pct(vol_ann,1)} (ann.)"
if pd.notna(sharpe): risk_str += f", Sharpe≈ <span class='num'>{sharpe:.2f}</span>"
if pd.notna(max_dd): risk_str += f", DD max {colored_pct(max_dd,2)}"
resume.append(risk_str)

liq_parts = []
if pd.notna(vol_median):      liq_parts.append(f"vol médian {colored_num(vol_median,0)}")
if pd.notna(turnover_mean):   liq_parts.append(f"turnover {colored_pct(100*turnover_mean,2)}")
if pd.notna(amihud_med):      liq_parts.append(f"Amihud {fr_fmt(amihud_med, 6)}")
resume.append("**Liquidité** : " + (" · ".join(liq_parts) if liq_parts else "n.d."))

resume.append("**Relative** : " + rel_txt)

div_parts = []
if pd.notna(dy12):      div_parts.append(f"DY12M {colored_pct(dy12,2)}")
if isinstance(last_ex_date,str) and last_ex_date!="—":  div_parts.append(f"dernière ex-date {last_ex_date}")
if pd.notna(last_div):  div_parts.append(f"dernier dividende {fr_fmt(last_div,2)}")
if pd.notna(payout):    div_parts.append(f"payout ≈ {colored_pct(payout,1)}")
resume.append("**Dividendes** : " + (" · ".join(div_parts) if div_parts else "n.d."))

resume.append("**Valorisation** : " + val_txt)
resume.append(f"**Alertes / Régimes** : {alerts}")

st.markdown("<br>".join(resume), unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
# ====== FIN RÉSUMÉ EXÉCUTIF ======

# ====================== TABS ======================
tabs = st.tabs([
    "📊 Prix & rendements",
    "📈 Perf & risque",
    "🔗 Relative",
    "💸 Dividendes & Total Return",
    "🧪 Microstructure",
    "📐 Valorisation",
    "🧭 Régimes & signaux",
    "🗓️ Saisonnalité",
    "🎛️ Gestion & sizing",
    "🧰 Qualité des données"
])

# ---------- TAB 1: Prix & rendements ----------
with tabs[0]:
    # Cours ajusté + ex-dividendes
    plot_df = df_win[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).copy()
    plot_df["PriceAdj_fr"] = plot_df["PriceAdj"].apply(lambda x: fr_fmt(x,0))
    div_col = next((c for c in ["Dividende ajusté","Dividende ajusté (FCFA)","Dividende"] if c in df_win.columns), None)
    div_df = pd.DataFrame(columns=["Date","PriceAdj","DivAdj"])
    if div_col is not None:
        tmp = df_win.loc[df_win[div_col].notna(), ["Date", price_col, div_col]].rename(columns={price_col:"PriceAdj", div_col:"DivAdj"})
        if not tmp.empty:
            tmp["DivAdj_fr"] = tmp["DivAdj"].apply(lambda x: fr_fmt(x,2))
            div_df = tmp

    line_price = alt.Chart(plot_df).mark_line(strokeWidth=2, color=PALETTE["c1"]).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("PriceAdj:Q", axis=axis_left("Cours ajusté (FCFA)")),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("PriceAdj_fr:N", title="Cours ajusté")]
    ).properties(height=PRICE_H)
    axis_right_layer = alt.Chart(plot_df).mark_line(opacity=0).encode(x="Date:T", y=alt.Y("PriceAdj:Q", axis=axis_right()))
    layers = [line_price, axis_right_layer]
    if not div_df.empty:
        layers.append(
            alt.Chart(div_df).mark_point(size=DIV_POINT_SIZE, filled=True, color=PALETTE["c2"],
                                         stroke="#0f172a", strokeWidth=0.5).encode(
                x="Date:T", y="PriceAdj:Q",
                tooltip=[alt.Tooltip("Date:T", title="Ex-dividende"),
                         alt.Tooltip("DivAdj_fr:N", title="Dividende ajusté")]
            )
        )
    st.altair_chart(alt.layer(*layers), use_container_width=True)
    explain_block(
        show_expl,
        definition="Le <i>cours ajusté</i> incorpore mécaniquement les effets de dividendes, splits, regroupements, etc. Les points marquent les dates ex-dividende.",
        interpretation="L’évolution reflète la valeur économique pour l’actionnaire. Des marches à la baisse au point vert traduisent l’ajustement ex-dividende.",
        theory="L’ajustement suit l’égalité prix ex-date = prix cum-div − dividende (absence d’arbitrage). On utilise cet historique pour des rendements additifs/stationnaires."
    )

    # Rendement cumulé
    df_cum = df_win[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).copy()
    p0 = df_cum["PriceAdj"].iloc[0]
    df_cum["CumPct"] = (df_cum["PriceAdj"]/p0 - 1.0)*100.0
    df_cum["CumPct_fr"] = df_cum["CumPct"].apply(lambda x: fr_pct_from_pct(x,2))
    cum_left = alt.Chart(df_cum).mark_line(strokeWidth=2, color=PALETTE["c3"]).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("CumPct:Q", axis=axis_left("Rendement cumulé (%)")),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("CumPct_fr:N", title="Cumulé")]
    ).properties(height=CUM_H)
    cum_right = alt.Chart(df_cum).mark_line(opacity=0).encode(x="Date:T", y=alt.Y("CumPct:Q", axis=axis_right()))
    st.altair_chart(alt.layer(cum_left, cum_right), use_container_width=True)
    explain_block(
        show_expl,
        definition="Performance relative au début de la fenêtre : (P/P<sub>0</sub> − 1)×100.",
        interpretation="Au-dessus de 0 % → gain ; en dessous → perte. Permet de situer le niveau actuel par rapport au début de période.",
        theory="On compare des niveaux de prix homogénéisés (ajustés) pour éviter les biais d’événements corporates (dividendes, splits)."
    )

    # Volumes (ligne)
    vol_series = df_win.get("Volume", df_win.get("Volume Normal Total", df_win.get("Volume Ajuste Total", pd.Series(np.nan, index=df_win.index))))
    vol_df = df_win[["Date"]].copy(); vol_df["Volume"] = vol_series
    vol_df["Volume_fr"] = vol_df["Volume"].apply(lambda x: fr_fmt(x,0))
    vol_left = alt.Chart(vol_df).mark_line(strokeWidth=2, color=PALETTE["c3"]).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Volume:Q", axis=axis_left("Volume (titres)")),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Volume_fr:N", title="Volume")]
    ).properties(height=VOL_H)
    vol_right = alt.Chart(vol_df).mark_line(opacity=0).encode(x="Date:T", y=alt.Y("Volume:Q", axis=axis_right()))
    st.altair_chart(alt.layer(vol_left, vol_right), use_container_width=True)
    explain_block(
        show_expl,
        definition="Titres échangés par jour.",
        interpretation="Pics = intérêt/breakout/annonce ; creux persistants = moindre liquidité, exécution potentiellement difficile.",
        theory="Activité de marché : le volume reflète agrégation de préférences/contraintes des intervenants ; volatilité et volume sont souvent corrélés."
    )

# ---------- TAB 2: Perf & risque ----------
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='param'>Paramètres Sharpe/Sortino</div>", unsafe_allow_html=True)
        rf = st.number_input("Taux sans risque annualisé (%)", value=0.0, step=0.25)
        window = st.slider("Fenêtre (jours)", 20, 120, 60, 5)
    with c2:
        st.markdown("<div class='param'>VaR / ES & Histogramme</div>", unsafe_allow_html=True)
        var_lvl = st.select_slider("Niveau de confiance", options=[90,95,97.5,99], value=95)
        bins = st.slider("Classes histogramme", 20, 80, 40, 2)

    # Sharpe & Sortino
    rs = df_win[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).sort_values("Date").copy()
    rs["r"] = np.log(rs["PriceAdj"]/rs["PriceAdj"].shift(1))
    rs = rs.dropna()
    if len(rs) >= window + 5:
        roll_mean = rs["r"].rolling(window).mean()
        roll_std  = rs["r"].rolling(window).std(ddof=0)
        neg = rs["r"].copy(); neg[neg>0] = 0.0
        roll_down_std = neg.rolling(window).std(ddof=0)
        rf_daily = (rf/100.0)/252.0
        sharpe  = (roll_mean - rf_daily) / roll_std * np.sqrt(252)
        sortino = (roll_mean - rf_daily) / roll_down_std * np.sqrt(252)
        out = rs.copy()
        out["Sharpe"]  = sharpe.replace([np.inf,-np.inf], np.nan)
        out["Sortino"] = sortino.replace([np.inf,-np.inf], np.nan)
        ch1 = alt.Chart(out).mark_line(strokeWidth=2, color=PALETTE["teal"]).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Sharpe:Q", axis=axis_left("Sharpe (ann.)")),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Sharpe:Q", title="Sharpe", format=".2f")]
        ).properties(height=240)
        ch2 = alt.Chart(out).mark_line(strokeWidth=2, color=PALETTE["c4"]).encode(
            x="Date:T",
            y=alt.Y("Sortino:Q", axis=axis_right()),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Sortino:Q", title="Sortino", format=".2f")]
        )
        st.altair_chart(alt.layer(ch1, ch2), use_container_width=True)
        explain_block(
            show_expl,
            definition="Sharpe = (µ − r<sub>f</sub>)/σ, Sortino remplace σ par la volatilité baissière (seulement les rendements négatifs).",
            interpretation=">1 bon, >2 très bon. Un Sortino > Sharpe implique des baisses relativement moins fréquentes/intenses.",
            theory="Ratio utilité quadratique/espérance-variance (Markowitz). Sortino utilise un <i>downside risk</i> proche des préférences asymétriques des investisseurs."
        )
    else:
        st.info("Pas assez d’observations pour Sharpe/Sortino.")

    # Volatilité réalisée
    w_vol = st.slider("Fenêtre volatilité (jours)", 10, 120, 30, 5)
    rv = df_win[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).copy()
    rv["r"] = np.log(rv["PriceAdj"]/rv["PriceAdj"].shift(1))
    rv["vol_ann"] = rv["r"].rolling(w_vol, min_periods=max(5, w_vol//3)).std(ddof=0)*np.sqrt(252)*100
    rv = rv.dropna(subset=["vol_ann"])
    if not rv.empty:
        rv["vol_fr"] = rv["vol_ann"].apply(lambda v: fr_pct_from_pct(v,2))
        ch_vol = alt.Chart(rv).mark_line(strokeWidth=2, color=PALETTE["c4"]).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("vol_ann:Q", axis=axis_left("Volatilité annualisée (%)")),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("vol_fr:N", title="Vol")]
        ).properties(height=260)
        st.altair_chart(ch_vol, use_container_width=True)
        explain_block(
            show_expl,
            definition="Écart-type des rendements, annualisé (√252).",
            interpretation="Plus la courbe est élevée, plus l’amplitude quotidienne est grande → risque de prix plus fort.",
            theory="Mesure de dispersion sous hypothèse (quasi) stationnaire des rendements. L’annualisation suppose l’additivité des variances en iid."
        )

    # Drawdown
    dd = df_win[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).copy()
    dd["cummax"] = dd["PriceAdj"].cummax()
    dd["drawdown"] = (dd["PriceAdj"]/dd["cummax"] - 1.0)*100
    ch_dd = alt.Chart(dd).mark_area(opacity=0.6, color=PALETTE["c5"]).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("drawdown:Q", axis=axis_left("Drawdown (%)"))
    ).properties(height=260)
    st.altair_chart(ch_dd, use_container_width=True)
    max_dd = dd["drawdown"].min()
    under = dd["drawdown"] < 0
    max_len = 0; cur = 0
    for v in under:
        cur = cur+1 if v else 0
        max_len = max(max_len, cur)
    explain_block(
        show_expl,
        definition="Baisse depuis le dernier plus-haut (en %).",
        interpretation=f"Min ≈ {fr_pct_from_pct(max_dd,2)} ; durée max ≈ {max_len} jours. Les longues périodes sous l’eau testent la tolérance au risque.",
        theory="Risque de trajectoire non capturé par la simple variance ; lié à l’utilité dépendant de la richesse et à l’aversion aux pertes."
    )

    # VaR / ES (empiriques) + Histogramme + fits
    rets = df_win[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).sort_values("Date").copy()
    rets["ret_pct"] = 100*np.log(rets["PriceAdj"]/rets["PriceAdj"].shift(1))
    rets = rets.dropna()
    if not rets.empty:
        q = var_lvl/100.0
        var_emp = -np.percentile(rets["ret_pct"], (1-q)*100)
        tail = rets["ret_pct"][rets["ret_pct"] <= -var_emp]
        es_emp = -tail.mean() if len(tail)>0 else np.nan
        st.write(f"{var_lvl}% → VaR hist {var_emp:.2f}% · ES hist {es_emp:.2f}%")

        counts, edges = np.histogram(rets["ret_pct"].values, bins=bins, density=True)
        hist_df = pd.DataFrame({"x0": edges[:-1], "x1": edges[1:], "density": counts})
        mu, sigma = rets["ret_pct"].mean(), rets["ret_pct"].std(ddof=0)
        xs = np.linspace(hist_df["x0"].min(), hist_df["x1"].max(), 500)
        normal_pdf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mu)/sigma)**2)
        pdf_df = pd.DataFrame({"x": xs, "Normal": normal_pdf})
        if sstats is not None:
            try:
                nu, loc, scale = sstats.t.fit(rets["ret_pct"].values)
                pdf_df["Student-t"] = sstats.t.pdf(xs, nu, loc=loc, scale=scale)
            except Exception:
                pass
        hist_chart = alt.Chart(hist_df).mark_bar(opacity=0.55, color=PALETTE["c3"]).encode(
            x=alt.X("x0:Q", title="Rendement (%)"), x2="x1:Q",
            y=alt.Y("density:Q", title="Densité")
        ).properties(height=260)
        layers = [hist_chart,
                  alt.Chart(pdf_df).mark_line(strokeWidth=2, color=PALETTE["blue"]).encode(x="x:Q", y="Normal:Q")]
        if "Student-t" in pdf_df.columns:
            layers.append(alt.Chart(pdf_df).mark_line(strokeWidth=2, color=PALETTE["teal"]).encode(x="x:Q", y="Student-t:Q"))
        st.altair_chart(alt.layer(*layers).resolve_scale(y='shared'), use_container_width=True)
        explain_block(
            show_expl,
            definition=f"VaR {int(var_lvl)}% = perte seuil ; ES = perte moyenne conditionnelle au-delà de la VaR. Histogramme = distribution empirique ; courbes = lois ajustées.",
            interpretation="Queue plus épaisse que Normal → risques extrêmes plus fréquents ; ES > VaR capture la sévérité des mauvaises queues.",
            theory="VaR/ES sont des mesures de risque basées sur la distribution des rendements. ES est cohérente (subadditive), la VaR peut ne pas l’être."
        )

    # Skewness / Kurtosis & Tail ratio
    st.markdown("<div class='param'>Moments & queues (roulants)</div>", unsafe_allow_html=True)
    w_mom = st.slider("Fenêtre (jours) pour moments", 20, 120, 60, 5)
    if len(rs) >= w_mom+5:
        def skew_roll(x): return pd.Series(x).skew()
        def kurt_roll(x): return pd.Series(x).kurt()
        def tail_ratio(x):
            x = pd.Series(x).dropna()
            if len(x)<10: return np.nan
            q05 = np.percentile(x, 5); q95 = np.percentile(x, 95)
            if q95==0: return np.nan
            return abs(q05)/q95
        mom = rs.copy()
        mom["Skew"]  = mom["r"].rolling(w_mom).apply(skew_roll, raw=True)
        mom["Kurt"]  = mom["r"].rolling(w_mom).apply(kurt_roll, raw=True)
        mom["TR"]    = mom["r"].rolling(w_mom).apply(tail_ratio, raw=True)
        m1 = alt.Chart(mom).mark_line(strokeWidth=2, color=PALETTE["violet"]).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Skew:Q", axis=axis_left("Skewness")),
        ).properties(height=200)
        m2 = alt.Chart(mom).mark_line(strokeWidth=2, color=PALETTE["c5"]).encode(
            x="Date:T", y=alt.Y("Kurt:Q", axis=axis_right("Kurtosis"))
        )
        st.altair_chart(alt.layer(m1, m2), use_container_width=True)
        tr = alt.Chart(mom).mark_line(strokeWidth=2, color=PALETTE["c2"]).encode(
            x=alt.X("Date:T"), y=alt.Y("TR:Q", axis=axis_left("Tail ratio |5%|/95%"))
        ).properties(height=160)
        st.altair_chart(tr, use_container_width=True)
        explain_block(
            show_expl,
            definition="Skew = asymétrie ; Kurtosis = épaisseur de queue ; Tail-ratio = |5e centile| / 95e centile.",
            interpretation="Skew < 0 + Kurt > 3 = risques de sell-off asymétriques ; TR > 1 = pertes extrêmes dominantes.",
            theory="Moments d’ordre 3 et 4 des distributions ; les rendements financiers montrent couramment asymétrie négative et leptokurtose."
        )

    # Recovery time
    px_rt = df_win[["Date", price_col]].rename(columns={price_col:"P"}).copy()
    px_rt["cummax"] = px_rt["P"].cummax()
    rec = []
    peak_val = px_rt["P"].iloc[0]
    cur = 0
    for val in px_rt["P"].iloc[1:]:
        if val >= peak_val:
            if cur>0: rec.append(cur)
            peak_val = val; cur = 0
        else:
            cur += 1
    if rec:
        rec_df = pd.DataFrame({"jours": rec})
        hist = alt.Chart(rec_df).mark_bar(color=PALETTE["c1"], opacity=.7).encode(
            x=alt.X("jours:Q", bin=alt.Bin(step=1), title="Jours de recouvrement"),
            y=alt.Y("count()", title="Fréquence")
        ).properties(height=180)
        st.altair_chart(hist, use_container_width=True)
        explain_block(
            show_expl,
            definition="Nombre de jours nécessaires pour revenir à un plus-haut précédent.",
            interpretation="Distribution focalisée = reprises rapides ; queue longue = phases de convalescence prolongées.",
            theory="Mesure le <i>temps de ruine inverse</i> sous trajectoires non-gaussiennes ; lié au profil de drawdown."
        )

    # Fan chart (bootstrap iid)
    st.markdown("<div class='param'>Fan chart (bootstrap)</div>", unsafe_allow_html=True)
    n_paths = st.slider("Nombre de scénarios", 50, 300, 100, 10)
    horizon = st.slider("Horizon (jours)", 20, 200, 60, 5)
    rvals = rs["r"].dropna().values
    if len(rvals) >= 50:
        sim = []
        for _ in range(n_paths):
            draw = np.random.choice(rvals, size=horizon, replace=True)
            path = np.exp(draw.cumsum())
            sim.append(path)
        sim = np.array(sim)
        last_price = df_win[price_col].iloc[-1]
        q5  = np.percentile(sim, 5, axis=0)*last_price
        q25 = np.percentile(sim, 25, axis=0)*last_price
        q50 = np.percentile(sim, 50, axis=0)*last_price
        q75 = np.percentile(sim, 75, axis=0)*last_price
        q95 = np.percentile(sim, 95, axis=0)*last_price
        fc = pd.DataFrame({"t": np.arange(1, horizon+1),
                           "q05": q5, "q25": q25, "q50": q50, "q75": q75, "q95": q95})
        band1 = alt.Chart(fc).mark_area(opacity=.18, color=PALETTE["blue"]).encode(x="t:Q", y="q25:Q", y2="q75:Q")
        band2 = alt.Chart(fc).mark_area(opacity=.1, color=PALETTE["blue"]).encode(x="t:Q", y="q05:Q", y2="q95:Q")
        med   = alt.Chart(fc).mark_line(strokeWidth=2, color=PALETTE["blue"]).encode(x="t:Q", y="q50:Q")
        st.altair_chart(band2 + band1 + med, use_container_width=True)
        explain_block(
            show_expl,
            definition="Projection d’éventail de prix simulés par ré-échantillonnage des rendements historiques (bootstrap iid).",
            interpretation="Largeur = incertitude. La médiane donne la trajectoire centrale sans hypothèse paramétrique forte.",
            theory="Bootstrap non-paramétrique (Efron) supposant indépendance et identique distribution des rendements ; ignore l’hétéroscédasticité conditionnelle."
        )

# ---------- TAB 3: Relative ----------
with tabs[2]:
    st.subheader("Force relative vs indice (base 100)")
    if indices_df is None:
        st.info("Charge 60_Cours_indices.csv dans Home.")
    else:
        idx_df = indices_df.copy()
        cand_ticker_cols = ["Ticker","Symbole","Index","Indice"]
        idx_ticker_col = next((c for c in cand_ticker_cols if c in idx_df.columns), None)
        price_cols_idx = [c for c in ["Cours Ajuste","Close","Price","Dernier"] if c in idx_df.columns]
        if idx_ticker_col and price_cols_idx:
            idx_price_col = price_cols_idx[0]
            choices = sorted(idx_df[idx_ticker_col].dropna().astype(str).unique().tolist())
            default_idx = next((x for x in choices if str(x).upper() in {"BRVM-C","BRVM COMPOSITE","COMPOSITE","BRVMCOM"}), choices[0] if choices else None)
            ref_sym = st.selectbox("Indice de référence", choices, index=choices.index(default_idx) if default_idx in choices else 0)
            ref = idx_df[idx_df[idx_ticker_col].astype(str)==str(ref_sym)].sort_values("Date")
            ref = ref[["Date", idx_price_col]].rename(columns={idx_price_col:"IndexPrice"}).dropna()
            tmp = df_win[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).copy()
            m = pd.merge_asof(tmp.sort_values("Date"), ref.sort_values("Date"), on="Date", direction="backward")
            m = m.dropna(subset=["IndexPrice"])
            if not m.empty:
                m["RS_base100"] = 100 * ( (m["PriceAdj"]/m["IndexPrice"]) / (m["PriceAdj"].iloc[0]/m["IndexPrice"].iloc[0]) )
                rs_left = alt.Chart(m).mark_line(strokeWidth=2, color=PALETTE["blue"]).encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("RS_base100:Q", axis=axis_left("Force relative (base 100)"))
                ).properties(height=260)
                st.altair_chart(rs_left, use_container_width=True)
                explain_block(
                    show_expl,
                    definition="Rapport prix du titre / indice, rebases à 100 au départ.",
                    interpretation="Ligne montante = surperformance ; descendante = sous-performance versus l’indice choisi.",
                    theory="Comparaison en espace multiplicatif : RS_t = (P_t/I_t) ÷ (P_0/I_0). Invariance à l’échelle et aux niveaux."
                )

            st.markdown("<div class='param'>Alpha/Beta roulants</div>", unsafe_allow_html=True)
            w_ab = st.slider("Fenêtre (jours) alpha/beta", 20, 120, 60, 5)
            price_t = df_t[["Date", price_col]].rename(columns={price_col:"P"}).sort_values("Date")
            price_i = ref[["Date","IndexPrice"]].rename(columns={"IndexPrice":"I"}).sort_values("Date")
            z = pd.merge_asof(price_t, price_i, on="Date", direction="backward").dropna()
            z["rt"] = np.log(z["P"]/z["P"].shift(1))
            z["ri"] = np.log(z["I"]/z["I"].shift(1))
            z = z.dropna()
            if len(z) >= w_ab+5:
                beta = z["rt"].rolling(w_ab).cov(z["ri"]) / z["ri"].rolling(w_ab).var()
                alpha_daily = z["rt"] - beta*z["ri"]
                alpha_ann = (1+alpha_daily.rolling(w_ab).mean())**252 - 1
                corr = z["rt"].rolling(w_ab).corr(z["ri"])
                out = z[["Date"]].copy()
                out["Beta"] = beta
                out["Alpha%"] = alpha_ann*100
                out["Corr"] = corr
                chb = alt.Chart(out).mark_line(strokeWidth=2, color=PALETTE["c1"]).encode(
                    x=alt.X("Date:T"), y=alt.Y("Beta:Q", axis=axis_left("Beta"))
                ).properties(height=200)
                cha = alt.Chart(out).mark_line(strokeWidth=2, color=PALETTE["c2"]).encode(
                    x="Date:T", y=alt.Y("Alpha%:Q", axis=axis_right("Alpha ann. (%)"))
                )
                st.altair_chart(alt.layer(chb, cha), use_container_width=True)
                explain_block(
                    show_expl,
                    definition="β = sensibilité du titre au marché ; α = performance excédentaire annualisée ; Corr = co-mouvement.",
                    interpretation="β>1 → plus réactif que le marché ; α>0 → génération de performance hors bêta.",
                    theory="Régression de marché (CAPM) : r_t = α + β·r_m + ε ; β = Cov(r, r_m)/Var(r_m)."
                )
                chc = alt.Chart(out).mark_line(strokeWidth=2, color=PALETTE["violet"]).encode(
                    x=alt.X("Date:T"), y=alt.Y("Corr:Q", axis=axis_left("Corrélation"))
                ).properties(height=160)
                st.altair_chart(chc, use_container_width=True)
                explain_block(
                    show_expl,
                    definition="Corrélation roulante entre le titre et l’indice.",
                    interpretation="Corr ↑ = co-mouvement accru (diversification moindre) ; Corr ↓ = comportement plus idiosyncratique.",
                    theory="Coefficient de Bravais-Pearson sur fenêtre glissante ; borné entre −1 et +1."
                )

    # Force relative sectorielle (vs médiane du secteur)
    st.subheader("Force relative sectorielle (vs médiane)")
    sec_col = "Secteur d'activité" if "Secteur d'activité" in base_df.columns else "Sector"
    if sec_col in base_df.columns:
        sect_val = df_t[sec_col].dropna().iloc[-1] if not df_t[sec_col].dropna().empty else None
        if sect_val is not None:
            peer = base_df[(base_df[sec_col]==sect_val) & (base_df["Date"]>=start) & (base_df["Date"]<=end)].copy()
            if not peer.empty:
                peer["P"] = peer[price_col]
                med = peer.groupby("Date", as_index=False)["P"].median().rename(columns={"P":"P_med"})
                cur = df_win[["Date", price_col]].rename(columns={price_col:"P"}).copy()
                m2 = pd.merge_asof(cur.sort_values("Date"), med.sort_values("Date"), on="Date", direction="backward").dropna()
                m2["RS_sec"] = 100 * ( (m2["P"]/m2["P_med"]) / (m2["P"].iloc[0]/m2["P_med"].iloc[0]) )
                ch = alt.Chart(m2).mark_line(strokeWidth=2, color=PALETTE["teal"]).encode(
                    x=alt.X("Date:T"), y=alt.Y("RS_sec:Q", axis=axis_left("Base 100"))
                ).properties(height=200)
                st.altair_chart(ch, use_container_width=True)
                explain_block(
                    show_expl,
                    definition="Prix du titre rapporté à la médiane du secteur (base 100).",
                    interpretation="Montée = surperformance vs pairs ; baisse = sous-performance.",
                    theory="La médiane est robuste aux extrêmes (outliers) et mieux adaptée qu’une moyenne sectorielle biaisée."
                )
        else:
            st.info("Secteur non renseigné pour ce titre.")
    else:
        st.info("Colonne secteur absente.")

# ---------- TAB 4: Dividendes & Total Return ----------
with tabs[3]:
    # Total Return (prix + dividendes)
    px = df_win[["Date", price_col]].rename(columns={price_col:"P"}).sort_values("Date").copy()
    div_col = next((c for c in ["Dividende ajusté","Dividende ajusté (FCFA)","Dividende"] if c in df_win.columns), None)
    px["DivAdj"] = df_win[div_col].values if div_col else 0.0
    px["DivAdj"] = px["DivAdj"].fillna(0.0)
    px["logret"] = np.log(px["P"]/px["P"].shift(1))
    px["income"] = px["DivAdj"] / px["P"].shift(1)
    px["tr"] = px["logret"].fillna(0.0) + px["income"].fillna(0.0)
    px["TR_base100"] = 100*np.exp(px["tr"].cumsum())
    px["PX_base100"] = 100*np.exp(px["logret"].fillna(0.0).cumsum())
    ch_tr = alt.Chart(px).transform_fold(
        ["TR_base100","PX_base100"], as_=["Série","Valeur"]
    ).mark_line(strokeWidth=2).encode(
        x=alt.X("Date:T"), y=alt.Y("Valeur:Q", axis=axis_left("Base 100")),
        color=alt.Color("Série:N", scale=alt.Scale(range=[PALETTE["blue"], PALETTE["c1"]]), title="")
    ).properties(height=260)
    st.altair_chart(ch_tr, use_container_width=True)
    explain_block(
        show_expl,
        definition="Indice base 100 comparant <i>Total Return</i> (prix + dividendes réinvestis) au prix seul.",
        interpretation="Écart croissant = contribution significative du dividende à la performance long terme.",
        theory="Sous absence d’arbitrage, réinvestir les coupons maximise la croissance composée (log-utility)."
    )

    # Dividend yield trailing 12M
    if div_col:
        dy = df_t[["Date", price_col, div_col]].rename(columns={price_col:"P", div_col:"D"}).copy()
        dy["D"] = dy["D"].fillna(0.0)
        dy = dy.sort_values("Date")
        s12 = []
        for i in range(len(dy)):
            end_i = dy.iloc[i]["Date"]; start_i = end_i - pd.Timedelta(days=365)
            window = dy[(dy["Date"]>start_i) & (dy["Date"]<=end_i)]
            s12.append(window["D"].sum())
        dy["Div12M"] = s12
        dy["DY%"] = 100*dy["Div12M"]/dy["P"].replace(0,np.nan)
        dwin = dy[(dy["Date"]>=start) & (dy["Date"]<=end)]
        ch_dy = alt.Chart(dwin).mark_line(strokeWidth=2, color=PALETTE["c2"]).encode(
            x=alt.X("Date:T"), y=alt.Y("DY%:Q", axis=axis_left("Dividend Yield (12M) %"))
        ).properties(height=200)
        st.altair_chart(ch_dy, use_container_width=True)
        explain_block(
            show_expl,
            definition="Rendement du dividende courant = (dividendes cumulés 12 mois) / prix.",
            interpretation="Plus haut → profil de rendement plus généreux, mais attention à la soutenabilité.",
            theory="Proche du <i>cash yield</i> ; avec BNPA on mesure un payout et un <i>dividend coverage</i>."
        )

    # Études d’événement : ex-date
    L = st.slider("Fenêtre événement ±J", 3, 20, 10, 1)
    if div_col and df_win[div_col].notna().any():
        events = df_win.loc[df_win[div_col].notna(), ["Date"]].reset_index(drop=True)
        px_all = df_t[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).sort_values("Date").reset_index(drop=True)
        arr = []
        for evd in events["Date"]:
            idx = px_all[px_all["Date"]<=evd].index.max()
            if pd.isna(idx): continue
            idx = int(idx); lo = max(0, idx-L); hi = min(len(px_all)-1, idx+L)
            window_e = px_all.iloc[lo:hi+1].copy()
            if len(window_e) < 2: continue
            anchor_price = px_all.iloc[idx]["PriceAdj"]
            window_e["tau"] = np.arange(lo-idx, hi-idx+1)
            window_e["ret_cum_pct"] = 100*(window_e["PriceAdj"]/anchor_price - 1.0)
            arr.append(window_e[["tau","ret_cum_pct"]])
        if arr:
            A = pd.concat(arr, ignore_index=True)
            grp = A.groupby("tau", as_index=False).agg(mean=("ret_cum_pct","mean"), std=("ret_cum_pct","std"), count=("ret_cum_pct","count"))
            grp["stderr"] = grp["std"]/np.sqrt(grp["count"].replace(0,np.nan))
            grp2 = grp.copy(); grp2["up"] = grp["mean"] + grp["stderr"]; grp2["dn"] = grp["mean"] - grp["stderr"]
            base_rule = alt.Chart(pd.DataFrame({"tau":[-L,0,L]})).mark_rule(color="#94a3b8").encode(x="tau:Q")
            mean_line = alt.Chart(grp).mark_line(strokeWidth=2, color=PALETTE["blue"]).encode(
                x=alt.X("tau:Q", title="Jours vs ex-date"), y=alt.Y("mean:Q", axis=axis_left("Rendement cumulé moyen (%)"))
            ).properties(height=220)
            band = alt.Chart(grp2).mark_area(opacity=0.2, color=PALETTE["blue"]).encode(x="tau:Q", y="dn:Q", y2="up:Q")
            st.altair_chart(base_rule + mean_line + band, use_container_width=True)
            explain_block(
                show_expl,
                definition="Moyenne des performances alignées autour des ex-dates (J0).",
                interpretation="Décrochage à J0 conforme à l’ajustement ; dynamique post-J0 → sur/sous-réaction potentielle.",
                theory="Études d’événements (Brown-Warner) : on aligne, on moyenne, on teste l’anormalité vs modèle d’attente."
            )

    # Payment-date (si dispo)
    pay_col = "Date paiement" if "Date paiement" in df_t.columns else None
    if pay_col and df_t[pay_col].notna().any():
        def _fr_to_datetime(s):
            if pd.isna(s): return pd.NaT
            for fmt in [None, "%d/%m/%Y", "%Y-%m-%d"]:
                try:
                    return pd.to_datetime(s, format=fmt, errors="raise") if fmt else pd.to_datetime(s, errors="raise")
                except Exception:
                    continue
            return pd.NaT
        df_t["_pay_dt"] = df_t[pay_col].apply(_fr_to_datetime)
        pay_events = df_t[(df_t["_pay_dt"].notna()) & (df_t["_pay_dt"]>=start) & (df_t["_pay_dt"]<=end)][["_pay_dt"]].rename(columns={"_pay_dt":"Date"})
        if not pay_events.empty:
            px_all = df_t[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).sort_values("Date").reset_index(drop=True)
            arr = []
            for evd in pay_events["Date"]:
                idx = px_all[px_all["Date"]<=evd].index.max()
                if pd.isna(idx): continue
                idx = int(idx); lo = max(0, idx-L); hi = min(len(px_all)-1, idx+L)
                window_p = px_all.iloc[lo:hi+1].copy()
                if len(window_p) < 2: continue
                anchor_price = px_all.iloc[idx]["PriceAdj"]
                window_p["tau"] = np.arange(lo-idx, hi-idx+1)
                window_p["ret_cum_pct"] = 100*(window_p["PriceAdj"]/anchor_price - 1.0)
                arr.append(window_p[["tau","ret_cum_pct"]])
            if arr:
                A = pd.concat(arr, ignore_index=True)
                grp = A.groupby("tau", as_index=False).agg(mean=("ret_cum_pct","mean"), std=("ret_cum_pct","std"), count=("ret_cum_pct","count"))
                grp["stderr"] = grp["std"]/np.sqrt(grp["count"].replace(0,np.nan))
                grp2 = grp.copy(); grp2["up"] = grp["mean"] + grp["stderr"]; grp2["dn"] = grp["mean"] - grp["stderr"]
                mean_line = alt.Chart(grp).mark_line(strokeWidth=2, color=PALETTE["c2"]).encode(
                    x=alt.X("tau:Q", title="Jours vs paiement"), y=alt.Y("mean:Q", axis=axis_left("Rendement cumulé moyen (%)"))
                ).properties(height=200)
                band = alt.Chart(grp2).mark_area(opacity=0.2, color=PALETTE["c2"]).encode(x="tau:Q", y="dn:Q", y2="up:Q")
                st.altair_chart(mean_line + band, use_container_width=True)
                explain_block(
                    show_expl,
                    definition="Moyenne des performances alignées autour des dates de paiement.",
                    interpretation="Réactions plus diffuses que l’ex-date ; parfois pas d’effet car <i>anticipé</i>.",
                    theory="Même protocole d’étude d’événements ; l’effet <i>cash</i> peut influencer la demande après paiement."
                )

# ---------- TAB 5: Microstructure ----------
with tabs[4]:
    vol = df_win.get("Volume", df_win.get("Volume Normal Total", df_win.get("Volume Ajuste Total", pd.Series(np.nan, index=df_win.index))))
    px_m = df_win[["Date", price_col]].rename(columns={price_col:"P"}).copy()
    px_m["Vol"] = vol.values
    # Turnover = Volume / (Flottant% × Nombre de titres)
    flot = (last.get("Flottant", pd.Series([np.nan]))).iloc[0]
    nb = (last.get("Nombre de titres", last.get("nb_actions_estime", pd.Series([np.nan])))).iloc[0]
    denom = (flot/100.0)*nb if (pd.notna(flot) and pd.notna(nb) and (flot>0) and (nb>0)) else np.nan
    px_m["Turnover"] = px_m["Vol"]/denom if pd.notna(denom) else np.nan
    ch_tov = alt.Chart(px_m).mark_line(strokeWidth=2, color=PALETTE["c1"]).encode(
        x=alt.X("Date:T"), y=alt.Y("Turnover:Q", axis=axis_left("Turnover (du flottant)"))
    ).properties(height=220)
    st.altair_chart(ch_tov, use_container_width=True)
    explain_block(
        show_expl,
        definition="Part du flottant qui a changé de mains (jour).",
        interpretation="Pics = rotation forte, intérêt spéculatif ou repositionnements institutionnels.",
        theory="Mesure de <i>market depth</i> pratique ; plus le flottant est faible, plus le turnover est sensible."
    )

    # Amihud illiquidity
    px_m["ret_abs"] = np.abs(np.log(px_m["P"]/px_m["P"].shift(1)))
    px_m["Amihud"] = px_m["ret_abs"] / (px_m["P"]*px_m["Vol"].replace(0,np.nan))
    ch_ami = alt.Chart(px_m).mark_line(strokeWidth=2, color=PALETTE["c5"]).encode(
        x=alt.X("Date:T"), y=alt.Y("Amihud:Q", axis=axis_left("Amihud (|r|/P·Vol)"))
    ).properties(height=220)
    st.altair_chart(ch_ami, use_container_width=True)
    explain_block(
        show_expl,
        definition="Impact prix/volume d’Amihud : |r|/(P·Vol).",
        interpretation="Plus élevé = plus illiquide (fort impact pour échanger).",
        theory="Amihud (2002) : <i>illiquidity proxy</i> reliant variation de prix et volume sous microstructure frictionnelle."
    )

    # Z-score de volume
    wvz = st.slider("Fenêtre Z-score volume", 10, 60, 20, 1)
    px_m["Vol_ma"] = px_m["Vol"].rolling(wvz).mean()
    px_m["Vol_sd"] = px_m["Vol"].rolling(wvz).std(ddof=0)
    px_m["Vol_z"]  = (px_m["Vol"]-px_m["Vol_ma"])/px_m["Vol_sd"]
    ch_vz = alt.Chart(px_m).mark_line(strokeWidth=2, color=PALETTE["c3"]).encode(
        x=alt.X("Date:T"), y=alt.Y("Vol_z:Q", axis=axis_left("Z-score volume"))
    ).properties(height=180)
    st.altair_chart(ch_vz, use_container_width=True)
    explain_block(
        show_expl,
        definition="Écart du volume à sa moyenne locale (en écarts-types).",
        interpretation="Z > 2 = volume inhabituel (news/breakout) ; Z < −2 = atonie.",
        theory="Standardisation locale supposant stationnarité locale ; utile pour <i>anomaly detection</i>."
    )

# ---------- TAB 6: Valorisation ----------
with tabs[5]:
    if ratios_df is None:
        st.info("Charge ratios.csv dans Home.")
    else:
        r = ratios_df.copy()
        col_map = {
            "Ticker":"Ticker", "Secteur d'activité":"Secteur",
            "Price Earning Ratio (PER)":"PER", "Price-to-Book Ratio (PBR)":"PBR",
            "Price-to-Sales (PSR)":"PSR", "Return on equity (ROE)":"ROE",
            "Bénéfice net par action (BNPA)":"BNPA", "Cours de l'action":"Price",
            "Capitalisation boursière":"Cap"
        }
        for k,v in list(col_map.items()):
            if k in r.columns: r.rename(columns={k:v}, inplace=True)
        if "Secteur" in r.columns:
            sec = df_t["Secteur d'activité"].dropna().iloc[-1] if "Secteur d'activité" in df_t.columns and not df_t["Secteur d'activité"].dropna().empty else None
            sec = sec if sec is not None else (df_t["Sector"].dropna().iloc[-1] if "Sector" in df_t.columns and not df_t["Sector"].dropna().empty else None)
            rf = r[r["Secteur"].astype(str)==str(sec)].copy() if sec is not None else r.copy()
            metrics = [c for c in ["PER","PBR","PSR"] if c in rf.columns]
            if rf.empty or not metrics:
                st.info("Ratios indisponibles pour ce secteur.")
            else:
                row_me = r[r["Ticker"].astype(str)==str(sel_ticker)].head(1)
                cols = st.columns(len(metrics))
                for i,m in enumerate(metrics):
                    med = rf[m].median(skipna=True)
                    mad = (rf[m]-med).abs().median(skipna=True)
                    x = row_me[m].iloc[0] if (not row_me.empty and m in row_me.columns) else np.nan
                    z = (x-med)/(mad if mad>0 else np.nan)
                    cols[i].markdown(f"**{m}**: {fr_fmt(x,2)} · Z₍MAD₎≈ **{fr_fmt(z,2)}**  \n<span class='small'>Médiane secteur: {fr_fmt(med,2)}</span>", unsafe_allow_html=True)
                explain_block(
                    show_expl,
                    definition="Comparaison des multiples (PER/PBR/PSR) au secteur via Z-score robuste (MAD).",
                    interpretation="Z>0 = plus cher que la médiane ; Z<0 = plus “value”.",
                    theory="MAD (Median Absolute Deviation) robuste aux outliers contrairement à l’écart-type ; bonne pratique cross-sectionnelle."
                )

                if {"PBR","ROE"}.issubset(rf.columns):
                    rf_plot = rf[["Ticker","PBR","ROE"]].dropna()
                    me_color = alt.condition(alt.datum.Ticker==sel_ticker, alt.value(PALETTE["c1"]), alt.value(PALETTE["blue"]))
                    sc = alt.Chart(rf_plot).mark_point(size=80).encode(
                        x=alt.X("PBR:Q", title="PBR"),
                        y=alt.Y("ROE:Q", title="ROE"),
                        color=me_color,
                        tooltip=["Ticker","PBR","ROE"]
                    ).properties(height=260)
                    st.altair_chart(sc, use_container_width=True)
                    explain_block(
                        show_expl,
                        definition="Carte PBR vs ROE pour situer le titre parmi ses pairs.",
                        interpretation="Quadrant ROE↑ & PBR↓ = <i>value de qualité</i> potentielle.",
                        theory="Logique de Gordon-Shapiro simplifiée : P/B ≈ (ROE − g)/(k − g) ; ROE haut justifie PBR plus élevé."
                    )

        if ("BNPA" in r.columns) and (div_col is not None):
            bnpa = r.loc[r["Ticker"].astype(str)==str(sel_ticker), "BNPA"]
            bnpa = bnpa.iloc[0] if not bnpa.empty else np.nan
            dy = df_t[["Date", price_col, div_col]].rename(columns={price_col:"P", div_col:"D"}).copy()
            dy["D"] = dy["D"].fillna(0.0)
            dy = dy.sort_values("Date")
            s12 = []
            for i in range(len(dy)):
                end_i = dy.iloc[i]["Date"]; start_i = end_i - pd.Timedelta(days=365)
                window = dy[(dy["Date"]>start_i) & (dy["Date"]<=end_i)]
                s12.append(window["D"].sum())
            dy["Div12M"] = s12
            last_row = dy.iloc[-1]
            payout = (last_row["Div12M"]/bnpa) if (pd.notna(bnpa) and bnpa!=0) else np.nan
            st.write(f"Payout (12M/BNPA) ≈ **{fr_fmt(payout,2)}**")
            explain_block(
                show_expl,
                definition="Part du bénéfice distribué (trailing).",
                interpretation=">100 % non soutenable ; trop faible peut indiquer des besoins d’investissement/désendettement.",
                theory="Politique de distribution et cycle d’investissement de l’entreprise ; lien avec coût du capital et opportunités de ROIC."
            )

# ---------- TAB 7: Régimes & signaux ----------
with tabs[6]:
    st.markdown("### Chandelier + indicateurs")
    colA, colB, colC, colD, colE, colF, colG = st.columns(7)
    with colA: opt_sma  = st.checkbox("SMA20/50", value=True)
    with colB: opt_bb   = st.checkbox("Bandes de Bollinger", value=False)
    with colC: opt_adj  = st.checkbox("Cours ajusté (ligne)", value=True)
    with colD: opt_div  = st.checkbox("Ex-dividendes", value=True)
    with colE: pnl_vol  = st.checkbox("Panel Volumes", value=True)
    with colF: pnl_rsi  = st.checkbox("RSI(14)", value=False)
    with colG: pnl_macd = st.checkbox("MACD (12-26-9)", value=False)

    need = {"Open","High","Low","Close","Volume"}
    if need.issubset(df_win.columns):
        df_all = df_t.sort_values("Date").copy()
        df_all["Adj"] = df_all.get("Cours Ajuste", df_all.get(price_col, df_all["Close"]))
        if opt_sma or opt_bb or pnl_rsi or pnl_macd:
            df_all["SMA20"] = df_all["Close"].rolling(20, min_periods=1).mean()
        if opt_sma:
            df_all["SMA50"] = df_all["Close"].rolling(50, min_periods=1).mean()
        if opt_bb:
            mid, up, low = bollinger(df_all["Close"], 20, 2)
            df_all["BB_UP"], df_all["BB_LOW"] = up, low
        if pnl_rsi:
            df_all["RSI14"] = rsi(df_all["Close"], 14)
        if pnl_macd:
            m, s, h = macd(df_all["Close"], 12, 26, 9)
            df_all["MACD"], df_all["MACD_SIG"], df_all["MACD_HIST"] = m, s, h

        df_cdl = df_all[(df_all["Date"]>=start) & (df_all["Date"]<=end)].copy()
        div_c = pd.DataFrame(columns=["Date","Close","DivAdj"])
        if opt_div:
            dcol = next((c for c in ["Dividende ajusté","Dividende ajusté (FCFA)","Dividende"] if c in df_cdl.columns), None)
            if dcol: div_c = df_cdl.loc[df_cdl[dcol].notna(), ["Date","Close",dcol]].rename(columns={dcol:"DivAdj"})

        row_defs=[("price",0.62)]
        if pnl_vol: row_defs.append(("volume",0.18))
        if pnl_rsi: row_defs.append(("rsi",0.11))
        if pnl_macd:row_defs.append(("macd",0.09))
        row_index={n:i+1 for i,(n,_) in enumerate(row_defs)}
        heights=[h for _,h in row_defs]
        fig = make_subplots(rows=len(row_defs), cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)

        r=row_index["price"]
        fig.add_trace(go.Candlestick(x=df_cdl["Date"], open=df_cdl["Open"], high=df_cdl["High"],
                                     low=df_cdl["Low"], close=df_cdl["Close"],
                                     name="Cours", increasing_line_color="#0a8f5b",
                                     decreasing_line_color=PALETTE["c5"], showlegend=False), row=r, col=1)
        if opt_sma:
            fig.add_trace(go.Scatter(x=df_cdl["Date"], y=df_cdl["SMA20"], mode="lines",
                                     line=dict(width=1.6, color=PALETTE["blue"]), name="SMA20"), row=r, col=1)
            fig.add_trace(go.Scatter(x=df_cdl["Date"], y=df_cdl["SMA50"], mode="lines",
                                     line=dict(width=1.6, color=PALETTE["violet"]), name="SMA50"), row=r, col=1)
        if opt_bb:
            fig.add_trace(go.Scatter(x=df_cdl["Date"], y=df_cdl["BB_UP"],  mode="lines",
                                     line=dict(width=1, color="#94a3b8"), name="BB Up", showlegend=False), row=r, col=1)
            fig.add_trace(go.Scatter(x=df_cdl["Date"], y=df_cdl["BB_LOW"], mode="lines",
                                     line=dict(width=1, color="#94a3b8"), name="BB Low", showlegend=False), row=r, col=1)
        if opt_adj and "Adj" in df_cdl:
            fig.add_trace(go.Scatter(x=df_cdl["Date"], y=df_cdl["Adj"], mode="lines",
                                     line=dict(width=2, color=PALETTE["c1"]), name="Cours ajusté"), row=r, col=1)
        if opt_div and not div_c.empty:
            fig.add_trace(go.Scatter(x=div_c["Date"], y=div_c["Close"], mode="markers",
                                     marker=dict(size=13, color=PALETTE["c2"], line=dict(color="#0f172a", width=0.6)),
                                     name="Ex-dividende", showlegend=True,
                                     hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Dividende ajusté: %{customdata:.2f} CFA<extra></extra>",
                                     customdata=div_c["DivAdj"].values), row=r, col=1)
        fig.update_yaxes(title_text="Cours (FCFA)", side="right", showgrid=True, row=r, col=1)

        if "volume" in row_index:
            rv=row_index["volume"]
            fig.add_trace(go.Bar(x=df_cdl["Date"], y=df_cdl["Volume"],
                                 marker_color=PALETTE["c3"], name="Volume", opacity=0.6), row=rv, col=1)
            fig.update_yaxes(title_text="Volume (titres)", side="right", showgrid=True, row=rv, col=1)
        if "rsi" in row_index:
            rr=row_index["rsi"]
            fig.add_trace(go.Scatter(x=df_cdl["Date"], y=df_cdl["RSI14"], mode="lines",
                                     line=dict(width=1.6, color=PALETTE["teal"]), name="RSI(14)"), row=rr, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#94a3b8", row=rr, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#94a3b8", row=rr, col=1)
            fig.update_yaxes(title_text="RSI (14)", side="right", range=[0,100], row=rr, col=1)
        if "macd" in row_index:
            rm=row_index["macd"]
            fig.add_trace(go.Scatter(x=df_cdl["Date"], y=df_cdl["MACD"], mode="lines",
                                     line=dict(width=1.6, color="#0ea5e9"), name="MACD"), row=rm, col=1)
            fig.add_trace(go.Scatter(x=df_cdl["Date"], y=df_cdl["MACD_SIG"], mode="lines",
                                     line=dict(width=1.6, color="#f59e0b"), name="Signal"), row=rm, col=1)
            fig.add_trace(go.Bar(x=df_cdl["Date"], y=df_cdl["MACD_HIST"],
                                 marker_color=PALETTE["slate"], name="Hist", opacity=0.5), row=rm, col=1)
            fig.update_yaxes(title_text="MACD", side="right", row=rm, col=1)
        for name,_ in row_defs:
            rr = row_index[name]
            fig.update_xaxes(rangeslider=dict(visible=False), row=rr, col=1)
        fig.update_layout(height=CANDLE_H, hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                          margin=dict(l=10, r=10, t=10, b=10), title=sel_ticker)
        st.plotly_chart(fig, use_container_width=True, config={"locale":"fr"})
        explain_block(
            show_expl,
            definition="Chandelier OHLC avec overlays (SMA, bandes de Bollinger, RSI, MACD) et volumes.",
            interpretation="Cross SMA20/50 = momentum ; Bollinger serrées = squeeze ; RSI>70/<30 = sur-achat/sur-vente ; MACD croise signal = changement de dynamique.",
            theory="SMA/Bollinger (moyennes/volatilité) ; RSI = oscillateur de force relative ; MACD = différence de moyennes exponentielles (12-26) et signal (9)."
        )
    else:
        st.info("Chandelier indisponible (colonnes OHLC/Volume manquantes).")

    # Supports / Résistances (n-jours)
    st.markdown("### Supports / Résistances")
    n_sr = st.slider("Période n (jours)", 10, 100, 20, 5)
    if {"High","Low"}.issubset(df_win.columns):
        sr = df_win[["Date","High","Low"]].copy()
    else:
        st.info("OHLC incomplet : fallback sur le prix ajusté.")
        sr = df_win[["Date", price_col]].rename(columns={price_col:"High"}); sr["Low"] = sr["High"]
    lev = rolling_levels(sr, n=n_sr)
    px_line = df_win[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).copy()
    line_p = alt.Chart(px_line).mark_line(strokeWidth=2, color=PALETTE["c1"]).encode(
        x=alt.X("Date:T"), y=alt.Y("PriceAdj:Q", axis=axis_left("Cours (FCFA)"))
    ).properties(height=260)
    line_hi = alt.Chart(lev).mark_line(strokeWidth=1.2, color="#0ea5e9").encode(x="Date:T", y="HighN:Q")
    line_lo = alt.Chart(lev).mark_line(strokeWidth=1.2, color="#10b981").encode(x="Date:T", y="LowN:Q")
    st.altair_chart(alt.layer(line_p, line_hi, line_lo), use_container_width=True)
    explain_block(
        show_expl,
        definition=f"Plus-hauts/bas glissants sur {n_sr} jours.",
        interpretation="Cassure au-dessus du plus-haut N-j = breakout haussier ; sous le plus-bas N-j = signal baissier.",
        theory="Mesure <i>non paramétrique</i> de la tendance ; proche des systèmes Donchian/Turtle."
    )

    # Squeeze (Bollinger bandwidth)
    st.markdown("### Squeeze (Bollinger bandwidth)")
    z = df_t[["Date","Close"]].dropna().sort_values("Date").copy()
    mid, up, low = bollinger(z["Close"], 20, 2)
    z["BW"] = (up-low)/mid
    z = z[(z["Date"]>=start)&(z["Date"]<=end)]
    qth = st.slider("Seuil de squeeze (percentile)", 5, 40, 20, 1)
    thr = np.nanpercentile(z["BW"], qth) if not z["BW"].dropna().empty else np.nan
    ch_bw = alt.Chart(z).mark_line(strokeWidth=2, color=PALETTE["blue"]).encode(
        x=alt.X("Date:T"), y=alt.Y("BW:Q", axis=axis_left("Bandwidth")),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("BW:Q", format=".4f")]
    ).properties(height=180)
    st.altair_chart(ch_bw, use_container_width=True)
    if pd.notna(thr):
        st.caption(f"Squeeze si BW ≤ {thr:.4f} (percentile {qth})")
    explain_block(
        show_expl,
        definition="Largeur des bandes de Bollinger / moyenne.",
        interpretation="Très bas = compression → probabilité accrue de breakout (volatilité future).",
        theory="Hypothèse de <i>volatility clustering</i> : faiblesse prolongée de la volatilité tend à être suivie d’un regain."
    )

    # Breakouts n-jours
    st.markdown("### Breakouts n-jours")
    n_bo = st.slider("n pour plus-haut/bas", 10, 100, 20, 5)
    pxh = df_win[["Date","High","Low",price_col]].rename(columns={price_col:"P"}).copy()
    if "High" not in pxh or pxh["High"].isna().all():
        pxh["High"], pxh["Low"] = pxh["P"], pxh["P"]
    pxh["HighN"] = pxh["High"].rolling(n_bo, min_periods=1).max().shift(1)
    pxh["LowN"]  = pxh["Low"].rolling(n_bo, min_periods=1).min().shift(1)
    sig_up  = pxh[pxh["P"]>pxh["HighN"]][["Date","P"]]
    sig_dn  = pxh[pxh["P"]<pxh["LowN"]][["Date","P"]]
    st.write("Breakouts haussier:", sig_up.tail(10))
    st.write("Breakouts baissier:", sig_dn.tail(10))
    explain_block(
        show_expl,
        definition="Détection des cassures de plus-haut/bas à N jours.",
        interpretation="Outils simples de système suiveur de tendance.",
        theory="Règles non-paramétriques robustes au bruit, inspirées des filtres de canaux de prix."
    )

    # CUSUM
    st.markdown("### CUSUM (ruptures)")
    colC1, colC2 = st.columns(2)
    with colC1: k_bps = st.slider("Drift k (bps/j)", 0, 50, 0, 1)
    with colC2: h_sig = st.slider("Seuil h (σ)", 2.0, 8.0, 5.0, 0.5)
    r_series = df_win[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).copy()
    r_series["r"] = np.log(r_series["PriceAdj"]/r_series["PriceAdj"].shift(1))
    r_eff = r_series.dropna(subset=["r"]).reset_index(drop=True)
    pos_idx, neg_idx = cusum_changes(r_eff["r"], k=k_bps/10000.0, h_sigmas=h_sig)
    marks = pd.DataFrame({"Date": pd.concat([r_eff["Date"].iloc[pos_idx], r_eff["Date"].iloc[neg_idx]]).values,
                          "type": ["up"]*len(pos_idx)+["down"]*len(neg_idx)})
    base_line = alt.Chart(r_series).mark_line(strokeWidth=2, color=PALETTE["c1"]).encode(
        x=alt.X("Date:T"), y=alt.Y("PriceAdj:Q", axis=axis_left("Cours (FCFA)"))
    ).properties(height=200)
    if not marks.empty:
        mk = alt.Chart(marks).mark_rule(color="#111827").encode(x="Date:T")
        st.altair_chart(base_line + mk, use_container_width=True)
    else:
        st.altair_chart(base_line, use_container_width=True)
    explain_block(
        show_expl,
        definition="Cumulative Sum pour détecter des ruptures de moyenne.",
        interpretation="Traits verticaux = points de rupture possibles (changement de régime).",
        theory="CUSUM (Page, 1954) détecte des shifts sous hypothèse de bruit centré ; seuil en fonction de σ."
    )

    # Gaps
    st.markdown("### Gaps d’ouverture")
    if {"Open","Close"}.issubset(df_win.columns):
        gp = df_win[["Date","Open","Close"]].sort_values("Date").copy()
        gp["Close_prev"] = gp["Close"].shift(1)
        gp["gap_pct"] = 100*(gp["Open"]/gp["Close_prev"] - 1.0)
        gp = gp.dropna(subset=["gap_pct"])
        line_gap = alt.Chart(gp).mark_line(strokeWidth=2, color=PALETTE["violet"]).encode(
            x=alt.X("Date:T"), y=alt.Y("gap_pct:Q", axis=axis_left("Gap (%)"))
        ).properties(height=160)
        zero = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(color="#94a3b8").encode(y="y:Q")
        st.altair_chart(line_gap + zero, use_container_width=True)
        explain_block(
            show_expl,
            definition="Écart d’ouverture vs clôture de la veille (en %).",
            interpretation="Gaps récurrents = annonces / faible profondeur ; comblement rapide = inefficiences court-terme.",
            theory="L’ouverture incorpore la nouvelle information overnight ; microstructure : <i>order imbalance</i> et frictions."
        )

    # Hurst
    st.markdown("### Exposant de Hurst")
    H = hurst_exponent(df_win[price_col])
    if np.isnan(H):
        st.info("Série trop courte pour Hurst.")
    else:
        interp = "Persistance" if H>0.55 else ("Anti-persistance" if H<0.45 else "Quasi marche aléatoire")
        st.write(f"H ≈ **{H:.2f}** — {interp}")
        explain_block(
            show_expl,
            definition="Mesure de mémoire longue des séries (H=0,5 ≈ marche aléatoire).",
            interpretation="H>0,5 = tendances persistantes ; H<0,5 = alternance/reversion.",
            theory="Analyse R/S & scaling des accroissements : Var(X_{t+τ}−X_t) ∝ τ^{2H}."
        )

# ---------- TAB 8: Saisonnalité ----------
with tabs[7]:
    # Mensuelle
    pr = df_win[["Date", price_col]].rename(columns={price_col:"PriceAdj"}).dropna().copy()
    if len(pr) >= 30:
        pr = pr.set_index("Date").resample("M").last().dropna().reset_index()
        pr["Month"] = pr["Date"].dt.month
        pr["Year"]  = pr["Date"].dt.year
        pr["MoRet"] = 100*(pr["PriceAdj"]/pr["PriceAdj"].shift(1) - 1.0)
        pr = pr.dropna(subset=["MoRet"])
        m_bar = pr.groupby("Month", as_index=False)["MoRet"].mean()
        bar = alt.Chart(m_bar).mark_bar(color=PALETTE["c1"]).encode(
            x=alt.X("Month:O", title="Mois"),
            y=alt.Y("MoRet:Q", title="Rendement moyen (%)"),
            tooltip=[alt.Tooltip("MoRet:Q", title="Moyenne", format=".2f")]
        ).properties(height=120)
        heat = alt.Chart(pr).mark_rect().encode(
            x=alt.X("Month:O", title="Mois"),
            y=alt.Y("Year:O", title="Année"),
            color=alt.Color("MoRet:Q", title="%", scale=alt.Scale(scheme='redblue', domainMid=0)),
            tooltip=[alt.Tooltip("Year:O",title="Année"),
                     alt.Tooltip("Month:O",title="Mois"),
                     alt.Tooltip("MoRet:Q", title="Rendement", format=".2f")]
        ).properties(height=140)
        st.altair_chart(bar & heat, use_container_width=True)
        explain_block(
            show_expl,
            definition="Rendements moyens par mois + carte année×mois.",
            interpretation="Mois systématiquement positifs/négatifs → biais saisonniers (à confirmer).",
            theory="Saisonnalités documentées (effet janvier/weekend) mais instables, sujettes à data-snooping."
        )

    # Jour de semaine
    prd = df_win[["Date", price_col]].rename(columns={price_col:"P"}).dropna().copy()
    prd["r%"] = 100*np.log(prd["P"]/prd["P"].shift(1))
    try:
        prd["weekday"] = prd["Date"].dt.day_name(locale="fr_FR")
    except Exception:
        prd["weekday"] = prd["Date"].dt.weekday
    prd = prd.dropna(subset=["r%"])
    if not prd.empty:
        w = prd.groupby("weekday", as_index=False)["r%"].mean()
        ch = alt.Chart(w).mark_bar(color=PALETTE["c3"]).encode(
            x=alt.X("weekday:N", sort=None, title="Jour"),
            y=alt.Y("r%:Q", title="Rendement moyen (%)")
        ).properties(height=180)
        st.altair_chart(ch, use_container_width=True)
        explain_block(
            show_expl,
            definition="Rendements moyens par jour de semaine.",
            interpretation="Effets lundi/vendredi éventuels ; souvent faibles et instables.",
            theory="Anomalies comportementales et microstructure (flux d’ordres, news)."
        )

# ---------- TAB 9: Gestion & sizing ----------
with tabs[8]:
    # Kelly
    rs2 = df_win[["Date", price_col]].rename(columns={price_col:"P"}).sort_values("Date")
    rs2["r"] = np.log(rs2["P"]/rs2["P"].shift(1))
    mu = rs2["r"].mean(); sig2 = rs2["r"].var(ddof=0)
    f_kelly = (mu/sig2) if sig2>0 else np.nan
    st.write(f"Kelly ≈ **{fr_pct_from_pct(100*f_kelly,2)}** (½ Kelly ≈ **{fr_pct_from_pct(50*f_kelly,2)}**)")
    explain_block(
        show_expl,
        definition="Taille de mise optimale théorique pour maximiser la croissance composée.",
        interpretation="Utiliser prudemment (½ Kelly) : très sensible aux estimations et aux queues épaisses.",
        theory="Critère de Kelly (1956) : max E[log(wealth)] sous iid ; équivalent à µ/σ² pour petits rendements."
    )

    # Stop-loss via VaR k-jours (empirique)
    k = st.slider("Horizon stop (jours)", 1, 20, 5, 1)
    r_k = rs2["r"].dropna()
    if len(r_k) >= k+5:
        kret = r_k.rolling(k).sum()   # log-returns k-jours
        VaR95 = -np.percentile(100*kret.dropna(), 5)
        ES95  = -np.mean(100*kret[kret<=-VaR95/100.0]) if not kret.dropna().empty else np.nan
        st.write(f"VaR {k}j 95% ≈ **{VaR95:.2f}%** ; ES ≈ **{ES95:.2f}%**")
        explain_block(
            show_expl,
            definition="Perte seuil (VaR) et perte moyenne au-delà (ES) sur k jours.",
            interpretation="Calibrage de stop cohérent avec l’historique ; ES donne une idée de la sévérité au-delà de la VaR.",
            theory="Mesures issues de la distribution empirique des rendements k-jours ; ES est une mesure cohérente du risque."
        )

# ---------- TAB 10: Qualité des données ----------
with tabs[9]:
    sub = df_win.copy()
    na_rate = sub.isna().mean().sort_values(ascending=False)
    q = na_rate[na_rate>0]
    if not q.empty:
        dfq = q.reset_index(); dfq.columns = ["Colonne","Taux_NA"]
        dfq["Taux_%"] = (100*dfq["Taux_NA"]).round(2)
        ch = alt.Chart(dfq).mark_bar(color=PALETTE["c5"]).encode(
            x=alt.X("Taux_%:Q", title="% NA"),
            y=alt.Y("Colonne:N", sort="-x", title=""),
            tooltip=[alt.Tooltip("Taux_%:Q", title="% NA")]
        ).properties(height=220)
        st.altair_chart(ch, use_container_width=True)
        explain_block(
            show_expl,
            definition="Part de valeurs manquantes par colonne sur la fenêtre analysée.",
            interpretation="Colonnes très incomplètes → prudence dans les conclusions sur ces mesures.",
            theory="Biais de sélection : l’information manquante n’est pas forcément aléatoire (MNAR)."
        )
    else:
        st.success("Aucune valeur manquante visible sur la fenêtre.")

# ===== Période affichée =====
st.caption(f"Période : {start.date().strftime('%d/%m/%Y')} → {end.date().strftime('%d/%m/%Y')} "
           f"(données : {min_date.date().strftime('%d/%m/%Y')} → {max_date.date().strftime('%d/%m/%Y')})")
