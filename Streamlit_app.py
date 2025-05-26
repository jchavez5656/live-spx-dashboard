# streamlit_app.py
import time
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from datetime import datetime
from math import log, sqrt
from scipy.stats import norm
from scipy.interpolate import make_interp_spline, interp1d
from requests.auth import HTTPBasicAuth
import statsmodels.api as sm

# ──────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━ 1) CONFIGURATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ──────────────────────────────────────────────────────────────────────────────

TOKEN_FILE    = 'schwab_tokens.json'
CLIENT_ID     = 'TeyurKNMQmDOlYa8v534SYvGDgnbt1H6'
CLIENT_SECRET = 'cYHDTdzmQX4nKrGM'
REDIRECT_URI  = 'https://jchavez5656.github.io/chavez-redirect/'
UNDERLYING    = '$SPX'

STRIKE_COUNT  = 50
STRIKE_MIN    = 5600
STRIKE_MAX    = 6100

HISTORICAL_CSV = 'historical_exposures.csv'

DEFAULT_WEIGHTS_BY_BUCKET = {
    'Early': pd.Series({
        'const':    0.0,
        'n_gex':    1.2, 'n_dex':    1.0, 'n_vex':    0.8,
        'n_chx':    0.7, 'n_vax':    0.5, 'n_vol':    0.3,
        'n_dg':     1.0, 'n_dd':     1.0, 'n_dv':     0.8,
        'n_dc':     0.7, 'n_dva':    0.6,
    }),
    'Mid': pd.Series({
        'const':    0.0,
        'n_gex':    2.0, 'n_dex':    1.5, 'n_vex':    0.6,
        'n_chx':    0.5, 'n_vax':    0.3, 'n_vol':    0.2,
        'n_dg':     1.8, 'n_dd':     1.4, 'n_dv':     0.6,
        'n_dc':     0.5, 'n_dva':    0.3,
    }),
    'Late': pd.Series({
        'const':    0.0,
        'n_gex':    3.0, 'n_dex':    2.5, 'n_vex':    0.1,
        'n_chx':    0.0, 'n_vax':    0.0, 'n_vol':    0.1,
        'n_dg':     2.8, 'n_dd':     2.3, 'n_dv':     0.1,
        'n_dc':     0.0, 'n_dva':    0.0,
    }),
}


# ──────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━ 2) OAUTH TOKEN MANAGEMENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ──────────────────────────────────────────────────────────────────────────────

def load_tokens():
    with open(TOKEN_FILE) as f:
        return json.load(f)

def save_tokens(tokens):
    with open(TOKEN_FILE, 'w') as f:
        json.dump(tokens, f)

def refresh_access_token(tok):
    resp = requests.post(
        'https://api.schwabapi.com/v1/oauth/token',
        data={
            'grant_type':    'refresh_token',
            'refresh_token': tok['refresh_token'],
            'redirect_uri':  REDIRECT_URI
        },
        auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET),
        headers={'Content-Type': 'application/x-www-form-urlencoded'}
    )
    resp.raise_for_status()
    new_tok = resp.json()
    save_tokens(new_tok)
    return new_tok


# ──────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━ 3) HELPERS & BLACK‐SCHOLES GREEKS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ──────────────────────────────────────────────────────────────────────────────

def safe_normalize(s: pd.Series) -> pd.Series:
    m = s.abs().max()
    return s * 0 if (not m or np.isnan(m)) else s / m

def dynamic_width(strikes):
    u = np.sort(np.unique(strikes))
    return 10 if len(u) < 2 else 0.8 * np.mean(np.diff(u))

def calc_time_to_expiration(exp_str: str) -> float:
    try:
        d = datetime.strptime(exp_str[:10], '%Y-%m-%d')
        return max((d - datetime.utcnow()).days / 365.0, 0)
    except:
        return 0

def black_scholes_gamma(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return norm.pdf(d1) / (S * sigma * sqrt(T))

def black_scholes_charm(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    dd1 = (r + 0.5 * sigma**2)/(2 * sigma * sqrt(T)) - (log(S/K))/(2 * sigma * T**1.5)
    return norm.pdf(d1) * dd1

def black_scholes_vanna(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return -d2 * norm.pdf(d1) / sigma

def get_smoothed_iv(strikes, ivs):
    s = np.array(strikes)
    v = np.array(ivs)
    mask = ~np.isnan(v)
    s, v = s[mask], v[mask]
    if len(s) < 4:
        return interp1d(s, v, kind='linear', fill_value='extrapolate')
    idx = np.argsort(s)
    s_u = np.unique(s[idx])
    v_u = [v[idx][np.where(s[idx] == strike)[0][0]] for strike in s_u]
    return make_interp_spline(s_u, v_u, k=3)


# ──────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━ 4) BUILD RAW EXPOSURES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ──────────────────────────────────────────────────────────────────────────────

def chain_to_df(chain, spot):
    rows = []
    # Calls
    for exp, strikes in chain.get('callExpDateMap', {}).items():
        for K, rec in strikes.items():
            d = rec[0]
            rows.append({
                'exp': exp[:10],
                'strike':   float(K),
                'call_open_int': d['openInterest'],
                'call_delta':    d['delta'],
                'call_gamma':    d['gamma'],
                'call_vega':     d['vega'],
                'call_iv':       d.get('volatility', np.nan)/100,
                'call_bid':      d.get('bid', np.nan),
                'call_ask':      d.get('ask', np.nan),
                'call_mark':     d.get('mark', np.nan),
                'call_volume':   d['totalVolume']
            })
    # Puts
    for exp, strikes in chain.get('putExpDateMap', {}).items():
        for K, rec in strikes.items():
            d = rec[0]
            rows.append({
                'exp': exp[:10],
                'strike':   float(K),
                'put_open_int': d['openInterest'],
                'put_delta':    -abs(d['delta']),
                'put_gamma':    d['gamma'],
                'put_vega':     d['vega'],
                'put_iv':       d.get('volatility', np.nan)/100,
                'put_bid':      d.get('bid', np.nan),
                'put_ask':      d.get('ask', np.nan),
                'put_mark':     d.get('mark', np.nan),
                'put_volume':   d['totalVolume']
            })
    df = pd.DataFrame(rows)
    df['exp_t'] = df['exp'].apply(calc_time_to_expiration)

    # Filter to strike window
    df = df[(df['strike'] >= STRIKE_MIN) & (df['strike'] <= STRIKE_MAX)]

    # Smooth & average IV
    df['call_iv'] = df['call_iv'].fillna(0)
    df['put_iv']  = df['put_iv'].fillna(0)
    iv_c = get_smoothed_iv(df['strike'], df['call_iv'])
    iv_p = get_smoothed_iv(df['strike'], df['put_iv'])
    df['avg_iv'] = df['strike'].apply(lambda k: (iv_c(k) + iv_p(k)) / 2)

    # Compute Black‐Scholes Greeks
    df['bs_gamma'] = df.apply(lambda r: black_scholes_gamma(spot, r['strike'], r['exp_t'], r['avg_iv']), axis=1)
    df['bs_charm'] = df.apply(lambda r: black_scholes_charm(spot, r['strike'], r['exp_t'], r['avg_iv']), axis=1)
    df['bs_vanna'] = df.apply(lambda r: black_scholes_vanna(spot, r['strike'], r['exp_t'], r['avg_iv']), axis=1)

    # Aggregate per exp/strike
    df = df.groupby(['exp','exp_t','strike'], as_index=False).sum()

    # Raw exposures
    df['gex'] = (df['bs_gamma'] * (df['call_open_int'] - df['put_open_int'])) * 100
    df['dex'] = (df['call_delta'] * df['call_open_int'] + df['put_delta'] * df['put_open_int']) * 100
    df['vex'] = (df['call_vega']  * df['call_open_int'] - df['put_vega']  * df['put_open_int']) * 100
    df['chx'] = (df['bs_charm']  * df['call_open_int'] - df['bs_charm']  * df['put_open_int']) * 100
    df['vax'] = (df['bs_vanna']  * (df['call_open_int'] - df['put_open_int']))             * 100
    df['vol'] = df['call_volume'].fillna(0) + df['put_volume'].fillna(0)

    # Directional volume Greeks
    df['call_td'] = np.where(
        (df['call_mark'] - df['call_bid']).abs() < (df['call_ask'] - df['call_mark']).abs(),
        1, -1
    )
    df['put_td']  = np.where(
        (df['put_ask'] - df['put_mark']).abs() < (df['put_mark'] - df['put_bid']).abs(),
        -1, 1
    )
    df['dir_vol_gamma'] = (
        df['call_volume']*df['bs_gamma']*df['call_delta']*df['call_td']
      + df['put_volume'] *df['bs_gamma']*df['put_delta'] *df['put_td']
    )
    df['dir_vol_delta'] = (
        df['call_volume']*df['call_delta']*df['call_td']
      + df['put_volume'] *df['put_delta'] *df['put_td']
    )
    df['dir_vol_vega']  = (
        df['call_volume']*df['call_vega'] *df['call_td']
      - df['put_volume'] *df['put_vega']  *df['put_td']
    )
    df['dir_vol_charm'] = (
        df['call_volume']*df['bs_charm']*df['call_delta']*df['call_td']
      - df['put_volume'] *df['bs_charm']*df['put_delta'] *df['put_td']
    )
    df['dir_vol_vanna'] = (
        df['call_volume']*df['bs_vanna']*df['call_delta']*df['call_td']
      - df['put_volume'] *df['bs_vanna']*df['put_delta'] *df['put_td']
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━ 5) GAMMA FLIP FINDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ──────────────────────────────────────────────────────────────────────────────

def find_gamma_flip(df):
    df2 = df.sort_values('strike').reset_index(drop=True)
    df2['cum_gex'] = df2['gex'].cumsum()
    for i in range(1, len(df2)):
        if df2.loc[i-1, 'cum_gex'] < 0 < df2.loc[i, 'cum_gex']:
            x0, y0 = df2.loc[i-1, ['strike','cum_gex']]
            x1, y1 = df2.loc[i,   ['strike','cum_gex']]
            return x0 + (x1 - x0) * (-y0 / (y1 - y0))
    return None


# ──────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━ 6) PLOTTING FUNCTIONS (RETURNING FIGURES) ━━━━━━━━━━━━━━━━━━━━━━
# ──────────────────────────────────────────────────────────────────────────────

def plot_bar_exposure_figure(df, col, title, ylabel, gf=None, xtick_interval=25):
    """
    Returns a matplotlib Figure for total {col} per strike, with optional gamma‐flip line.
    """
    if df[col].abs().max() == 0:
        return None

    w = dynamic_width(df['strike'])
    colors = ['blue' if v >= 0 else 'red' for v in df[col]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df['strike'], df[col], width=w, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=1)

    # X‐axis ticks/grid
    smin, smax = df['strike'].min(), df['strike'].max()
    ticks = np.arange(
        start = np.floor(smin / xtick_interval) * xtick_interval,
        stop  = np.ceil(smax / xtick_interval) * xtick_interval + 1,
        step  = xtick_interval
    )
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', linestyle=':',  alpha=0.4)

    # If gamma‐flip provided, draw vertical line
    if (gf is not None) and (col == 'gex'):
        ax.axvline(gf, ls='--', color='green', label=f'Flip: {gf:.2f}')
        ax.legend()

    ax.set_xlabel('Strike Price')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def plot_exposure_with_directional_volume_figure(df, col, dircol, label, title, xtick_interval=25):
    """
    Returns a matplotlib Figure showing directional‐volume (faint) plus raw exposure (solid).
    """
    if df[col].abs().max() == 0 and df[dircol].abs().max() == 0:
        return None

    w = dynamic_width(df['strike'])
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Directional‐volume (left axis, faint bars)
    dc = ['blue' if v >= 0 else 'red' for v in df[dircol]]
    ax1.bar(df['strike'], df[dircol], color=dc, alpha=0.3, width=w)
    ax1.set_ylabel('Directional Volume')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Raw exposure (right axis, opaque bars)
    ax2 = ax1.twinx()
    eo = ['green' if v >= 0 else 'maroon' for v in df[col]]
    ax2.bar(df['strike'], df[col], width=w * 0.8, color=eo, edgecolor='black', alpha=0.7)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylabel(label)

    # X‐axis ticks/grid
    smin, smax = df['strike'].min(), df['strike'].max()
    ticks = np.arange(
        start = np.floor(smin / xtick_interval) * xtick_interval,
        stop  = np.ceil(smax / xtick_interval) * xtick_interval + 1,
        step  = xtick_interval
    )
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(ticks, rotation=45)
    ax1.grid(axis='x', linestyle=':', alpha=0.4)

    plt.title(title)
    plt.tight_layout()
    return fig

def plot_composite_and_magnets_figure(df, xtick_interval=25):
    """
    Returns a matplotlib Figure of “flat composite” (equal weights) aggregated across strikes.
    """
    # Normalize each exposure‐type across strikes
    for c in ['gex','dex','vex','chx','vax','vol']:
        df['n_' + c] = safe_normalize(df[c])
    for c in ['gamma','delta','vega','charm','vanna']:
        df['n_dir_' + c] = safe_normalize(df['dir_vol_' + c])

    # Flat composite = sum of normalized raw exposures + normalized directional‐volume
    df['final_composite'] = (
        df[['n_gex','n_dex','n_vex','n_chx','n_vax','n_vol']].sum(axis=1)
      + df[['n_dir_gamma','n_dir_delta','n_dir_vega','n_dir_charm','n_dir_vanna']].sum(axis=1)
    )
    df['abs_final_composite'] = df['final_composite'].abs()

    w = dynamic_width(df['strike'])
    colors = ['blue' if v >= 0 else 'red' for v in df['final_composite']]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df['strike'], df['final_composite'], width=w, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=1)

    # Annotate top‐3 magnet strikes
    top3 = df.nlargest(3, 'abs_final_composite')
    for _, r in top3.iterrows():
        offset = np.sign(r['final_composite']) * 0.05 * np.max(df['abs_final_composite'])
        ax.annotate(
            f"{int(r['strike'])}",
            xy=(r['strike'], r['final_composite']),
            xytext=(r['strike'], r['final_composite'] + offset),
            ha='center',
            arrowprops=dict(arrowstyle='->', lw=1.2)
        )

    # X‐axis ticks/grid
    smin, smax = df['strike'].min(), df['strike'].max()
    ticks = np.arange(
        start = np.floor(smin / xtick_interval) * xtick_interval,
        stop  = np.ceil(smax / xtick_interval) * xtick_interval + 1,
        step  = xtick_interval
    )
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', linestyle=':', alpha=0.4)

    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Composite Score')
    ax.set_title('Flat Composite Exposure & Magnet Levels (Aggregated)')
    plt.tight_layout()
    return fig

def plot_dynamic_composite_and_magnets_figure(df: pd.DataFrame, weights_by_bucket: dict, xtick_interval=25):
    """
    Returns a matplotlib Figure of “dynamic composite” (weighted by DTE bucket) aggregated across strikes.
    """
    # Normalize per‐row exposures
    for c in ['gex','dex','vex','chx','vax','vol']:
        df['n_' + c] = safe_normalize(df[c])
    for c in ['gamma','delta','vega','charm','vanna']:
        df['n_d' + c] = safe_normalize(df['dir_vol_' + c])

    # Compute weighted composite per row
    composites = []
    for _, row in df.iterrows():
        w = get_dynamic_weights(row['exp_t'], weights_by_bucket)
        terms = []
        for key in w.index:
            if key == 'const':
                continue
            if key in row:
                terms.append(w[key] * row[key])
        comp = sum(terms) + w.get('const', 0.0)
        composites.append(comp)
    df['weighted_composite'] = composites

    # Aggregate weighted_composite across expirations by strike
    df_strike = df.groupby('strike', as_index=False)['weighted_composite'].sum()
    df_strike['abs_weighted_composite'] = df_strike['weighted_composite'].abs()

    w = dynamic_width(df_strike['strike'])
    colors = ['blue' if v >= 0 else 'red' for v in df_strike['weighted_composite']]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_strike['strike'], df_strike['weighted_composite'], width=w, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=1)

    # Annotate top‐3 magnet strikes
    top3 = df_strike.nlargest(3, 'abs_weighted_composite')
    for _, r in top3.iterrows():
        offset = np.sign(r['weighted_composite']) * 0.05 * np.max(df_strike['abs_weighted_composite'])
        ax.annotate(
            f"{int(r['strike'])}",
            xy=(r['strike'], r['weighted_composite']),
            xytext=(r['strike'], r['weighted_composite'] + offset),
            ha='center',
            arrowprops=dict(arrowstyle='->', lw=1.2)
        )

    # X‐axis ticks/grid
    smin, smax = df_strike['strike'].min(), df_strike['strike'].max()
    ticks = np.arange(
        start = np.floor(smin / xtick_interval) * xtick_interval,
        stop  = np.ceil(smax / xtick_interval) * xtick_interval + 1,
        step  = xtick_interval
    )
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', linestyle=':', alpha=0.4)

    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Dynamic Composite Score')
    ax.set_title('Dynamic Composite Exposure & Magnet Levels')
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━ 7) REGRESSION TO FIT DYNAMIC WEIGHTS ━━━━━━━━━━━━━━━━━━━━━━━━━━
# ──────────────────────────────────────────────────────────────────────────────

def fit_weights_by_bucket(historical_df: pd.DataFrame) -> dict:
    """
    Given a DataFrame with columns:
      ['exp_t','gex','dex','vex','chx','vax','vol',
       'dir_vol_gamma','dir_vol_delta','dir_vol_vega','dir_vol_charm','dir_vol_vanna',
       'next_day_return']
    Fit OLS regressions in three DTE buckets: Early (>30), Mid (8–30), Late (≤7).
    Return dict: {'Early': Series, 'Mid': Series, 'Late': Series}
    """
    dfh = historical_df.copy()
    dfh['days_to_exp'] = dfh['exp_t'] * 365
    dfh['bucket'] = pd.cut(dfh['days_to_exp'], bins=[-1, 7, 30, 365], labels=['Late','Mid','Early'])

    weights_by_bucket = {}
    for bucket in ['Late','Mid','Early']:
        sub = dfh[dfh['bucket']==bucket].copy()
        if sub.empty:
            continue
        for c in ['gex','dex','vex','chx','vax','vol']:
            sub['n_'+c] = safe_normalize(sub[c])
        for c in ['gamma','delta','vega','charm','vanna']:
            sub['n_d' + c] = safe_normalize(sub['dir_vol_'+c])
        X = pd.DataFrame({
            'n_gex':  sub['n_gex'],
            'n_dex':  sub['n_dex'],
            'n_vex':  sub['n_vex'],
            'n_chx':  sub['n_chx'],
            'n_vax':  sub['n_vax'],
            'n_vol':  sub['n_vol'],
            'n_dg':   sub['n_dgamma'],
            'n_dd':   sub['n_ddelta'],
            'n_dv':   sub['n_dvega'],
            'n_dc':   sub['n_dcharm'],
            'n_dva':  sub['n_dvanna'],
        })
        X = sm.add_constant(X)
        y = sub['next_day_return']
        model = sm.OLS(y, X).fit()
        weights_by_bucket[bucket] = model.params
    return weights_by_bucket

def get_dynamic_weights(exp_t: float, weights_by_bucket: dict) -> pd.Series:
    days = exp_t * 365
    if days <= 7:
        return weights_by_bucket.get('Late', pd.Series(dtype=float))
    elif days <= 30:
        return weights_by_bucket.get('Mid', pd.Series(dtype=float))
    else:
        return weights_by_bucket.get('Early', pd.Series(dtype=float))


# ──────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━ 8) STREAMLIT APP LAYOUT & MAIN LOOP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Live SPX Exposures", layout="wide")
st.title("Live SPX Composite Exposure & Greeks Dashboard")

placeholder = st.empty()

while True:
    try:
        # ─── (A) LOAD & REFRESH TOKENS ──────────────────────────────────────────
        tok = load_tokens()
        tok = refresh_access_token(tok)
        hdr = {'Authorization': f"Bearer {tok['access_token']}"}

        # ─── (B) FETCH SPX SPOT & OPTION CHAIN ─────────────────────────────────
        res_q = requests.get(
            f'https://api.schwabapi.com/marketdata/v1/{UNDERLYING}/quotes',
            headers=hdr
        )
        res_q.raise_for_status()
        spot = res_q.json().get(UNDERLYING, {}).get('quote', {}).get('lastPrice')

        res_c = requests.get(
            'https://api.schwabapi.com/marketdata/v1/chains',
            params={'symbol': UNDERLYING, 'strikeCount': STRIKE_COUNT},
            headers=hdr
        )
        res_c.raise_for_status()
        chain = res_c.json()

        st.markdown(f"**Current SPX Spot:** {spot:.2f}")

        # ─── (C) BUILD RAW EXPOSURES DATAFRAME ─────────────────────────────────
        df = chain_to_df(chain, spot)
        gf = find_gamma_flip(df)

        # ─── (D) AGGREGATE ACROSS EXPIRATIONS ───────────────────────────────────
        df_agg = df.groupby('strike', as_index=False).sum()

        # ─── (E) FIT OR LOAD DYNAMIC WEIGHTS ───────────────────────────────────
        try:
            hist = pd.read_csv(HISTORICAL_CSV, parse_dates=['date'])
            required_cols = [
                'exp_t','gex','dex','vex','chx','vax','vol',
                'dir_vol_gamma','dir_vol_delta','dir_vol_vega','dir_vol_charm','dir_vol_vanna',
                'next_day_return'
            ]
            missing = [c for c in required_cols if c not in hist.columns]
            if missing:
                st.warning(f"Historical CSV is missing columns {missing}. Using default weights.")
                weights_by_bucket = DEFAULT_WEIGHTS_BY_BUCKET
            else:
                st.info("Fitting dynamic weights by DTE bucket from historical data…")
                weights_by_bucket = fit_weights_by_bucket(hist)
                st.success("Dynamic weight calibration complete.")
        except FileNotFoundError:
            st.warning(f"'{HISTORICAL_CSV}' not found. Using default weight buckets.")
            weights_by_bucket = DEFAULT_WEIGHTS_BY_BUCKET

        # ─── (F) PLOT & DISPLAY EACH FIGURE ────────────────────────────────────
        # We use a Streamlit “container” so we can clear/redraw on each loop iteration.
        with placeholder.container():
            # 1) GEX Bar + Directional Volume Overlay
            fig_gex = plot_bar_exposure_figure(df_agg, 'gex', 'Total Gamma Exposure (GEX) per Strike', 'GEX', gf)
            if fig_gex:
                st.pyplot(fig_gex)
                plt.close(fig_gex)

            fig_gex_dir = plot_exposure_with_directional_volume_figure(
                df_agg, 'gex', 'dir_vol_gamma', 'GEX', 'Total GEX + Directional Volume'
            )
            if fig_gex_dir:
                st.pyplot(fig_gex_dir)
                plt.close(fig_gex_dir)

            # 2) DEX Bar + Directional Volume Overlay
            fig_dex = plot_bar_exposure_figure(df_agg, 'dex', 'Total Delta Exposure (DEX) per Strike', 'DEX')
            if fig_dex:
                st.pyplot(fig_dex)
                plt.close(fig_dex)

            fig_dex_dir = plot_exposure_with_directional_volume_figure(
                df_agg, 'dex', 'dir_vol_delta', 'DEX', 'Total DEX + Directional Volume'
            )
            if fig_dex_dir:
                st.pyplot(fig_dex_dir)
                plt.close(fig_dex_dir)

            # 3) VEX Bar + Directional Volume Overlay
            fig_vex = plot_bar_exposure_figure(df_agg, 'vex', 'Total Vega Exposure (VEX) per Strike', 'VEX')
            if fig_vex:
                st.pyplot(fig_vex)
                plt.close(fig_vex)

            fig_vex_dir = plot_exposure_with_directional_volume_figure(
                df_agg, 'vex', 'dir_vol_vega', 'VEX', 'Total VEX + Directional Volume'
            )
            if fig_vex_dir:
                st.pyplot(fig_vex_dir)
                plt.close(fig_vex_dir)

            # 4) CHX Bar + Directional Volume Overlay
            fig_chx = plot_bar_exposure_figure(df_agg, 'chx', 'Total Charm Exposure (CHX) per Strike', 'CHX')
            if fig_chx:
                st.pyplot(fig_chx)
                plt.close(fig_chx)

            fig_chx_dir = plot_exposure_with_directional_volume_figure(
                df_agg, 'chx', 'dir_vol_charm', 'CHX', 'Total CHX + Directional Volume'
            )
            if fig_chx_dir:
                st.pyplot(fig_chx_dir)
                plt.close(fig_chx_dir)

            # 5) VAX Bar + Directional Volume Overlay
            fig_vax = plot_bar_exposure_figure(df_agg, 'vax', 'Total Vanna Exposure (VAX) per Strike', 'VAX')
            if fig_vax:
                st.pyplot(fig_vax)
                plt.close(fig_vax)

            fig_vax_dir = plot_exposure_with_directional_volume_figure(
                df_agg, 'vax', 'dir_vol_vanna', 'VAX', 'Total VAX + Directional Volume'
            )
            if fig_vax_dir:
                st.pyplot(fig_vax_dir)
                plt.close(fig_vax_dir)

            # 6) Flat Composite & Magnet Levels
            fig_comp = plot_composite_and_magnets_figure(df_agg)
            if fig_comp:
                st.pyplot(fig_comp)
                plt.close(fig_comp)

            # 7) Dynamic Composite & Magnet Levels
            fig_dyn = plot_dynamic_composite_and_magnets_figure(df, weights_by_bucket)
            if fig_dyn:
                st.pyplot(fig_dyn)
                plt.close(fig_dyn)

            # Footer / last run timestamp
            st.markdown(f"*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*")

        # ─── (G) WAIT 30 SECONDS BEFORE REFRESH ─────────────────────────────────
        time.sleep(30)

    except Exception as e:
        # If anything errors (e.g. token expired, network fail), show it and retry after 30s.
        st.error(f"Error: {e}")
        time.sleep(30)
        continue

