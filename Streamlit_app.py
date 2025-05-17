# streamlit_app.py
import time, json, requests, pandas as pd, numpy as np, os
import matplotlib.pyplot as plt, streamlit as st
from datetime import datetime
from math import log, sqrt
from scipy.stats import norm
from scipy.interpolate import make_interp_spline, interp1d
from requests.auth import HTTPBasicAuth

# ---------------------------
# Configuration (from Streamlit secrets)
# ---------------------------
TOKEN_FILE    = 'schwab_tokens.json'
CLIENT_ID     = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
REDIRECT_URI  = st.secrets["REDIRECT_URI"]
UNDERLYING    = '$SPX'
STRIKE_COUNT  = 50
STRIKE_MIN    = 5600
STRIKE_MAX    = 6100

# ---------------------------
# OAuth token management
# ---------------------------
def load_tokens():
    if not os.path.exists(TOKEN_FILE):
        st.error("Token file not found. Please upload 'schwab_tokens.json' via the file uploader below.")
        st.stop()
    with open(TOKEN_FILE) as f:
        return json.load(f)

def save_tokens(tokens):
    with open(TOKEN_FILE, 'w') as f:
        json.dump(tokens, f)

def refresh_access_token(tok):
    resp = requests.post(
        'https://api.schwabapi.com/v1/oauth/token',
        data={
            'grant_type': 'refresh_token',
            'refresh_token': tok['refresh_token'],
            'redirect_uri': REDIRECT_URI
        },
        auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET),
        headers={'Content-Type': 'application/x-www-form-urlencoded'}
    )
    resp.raise_for_status()
    new_tok = resp.json()
    save_tokens(new_tok)
    return new_tok

# ---------------------------
# Helper functions & Greeks
# ---------------------------
def safe_normalize(s):
    m = s.abs().max()
    return s * 0 if not m or np.isnan(m) else s / m

def dynamic_width(strikes):
    u = np.sort(np.unique(strikes))
    return 10 if len(u) < 2 else 0.8 * np.mean(np.diff(u))

def calc_time_to_expiration(exp_str):
    try:
        d = datetime.strptime(exp_str[:10], '%Y-%m-%d')
        return max((d - datetime.utcnow()).days / 365.0, 0)
    except:
        return 0

def black_scholes_gamma(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * sqrt(T))
    return norm.pdf(d1) / (S * sigma * sqrt(T))

def black_scholes_charm(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * sqrt(T))
    dd1 = (r + 0.5*sigma**2)/(2*sigma*sqrt(T)) - (log(S/K))/(2*sigma*(T**1.5))
    return norm.pdf(d1) * dd1

def black_scholes_vanna(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return -d2 * norm.pdf(d1) / sigma

def get_smoothed_iv(strikes, ivs):
    s = np.array(strikes); v = np.array(ivs)
    mask = ~np.isnan(v)
    s, v = s[mask], v[mask]
    if len(s) < 4:
        return interp1d(s, v, kind='linear', fill_value='extrapolate')
    idx = np.argsort(s)
    s_u = np.unique(s[idx])
    v_u = [v[idx][np.where(s[idx]==k)[0][0]] for k in s_u]
    return make_interp_spline(s_u, v_u, k=3)

def chain_to_df(chain, spot):
    rows = []
    for exp, strikes in chain.get('callExpDateMap', {}).items():
        for K, rec in strikes.items():
            d = rec[0]
            rows.append({
                'exp': exp[:10], 'strike': float(K),
                'call_open_int': d['openInterest'], 'call_delta': d['delta'],
                'call_gamma': d['gamma'], 'call_vega': d['vega'],
                'call_iv': d.get('volatility', np.nan)/100,
                'call_bid': d.get('bid', np.nan), 'call_ask': d.get('ask', np.nan),
                'call_mark': d.get('mark', np.nan), 'call_volume': d['totalVolume']
            })
    for exp, strikes in chain.get('putExpDateMap', {}).items():
        for K, rec in strikes.items():
            d = rec[0]
            rows.append({
                'exp': exp[:10], 'strike': float(K),
                'put_open_int': d['openInterest'], 'put_delta': -abs(d['delta']),
                'put_gamma': d['gamma'], 'put_vega': d['vega'],
                'put_iv': d.get('volatility', np.nan)/100,
                'put_bid': d.get('bid', np.nan), 'put_ask': d.get('ask', np.nan),
                'put_mark': d.get('mark', np.nan), 'put_volume': d['totalVolume']
            })
    df = pd.DataFrame(rows)
    df['exp_t'] = df['exp'].apply(calc_time_to_expiration)
    df = df[(df['strike']>=STRIKE_MIN)&(df['strike']<=STRIKE_MAX)]
    df['call_iv'].fillna(0, inplace=True)
    df['put_iv'].fillna(0, inplace=True)
    iv_c = get_smoothed_iv(df['strike'], df['call_iv'])
    iv_p = get_smoothed_iv(df['strike'], df['put_iv'])
    df['avg_iv'] = df['strike'].apply(lambda k: (iv_c(k)+iv_p(k))/2)
    df['bs_gamma'] = df.apply(lambda r: black_scholes_gamma(spot, r['strike'], r['exp_t'], r['avg_iv']), axis=1)
    df['bs_charm'] = df.apply(lambda r: black_scholes_charm(spot, r['strike'], r['exp_t'], r['avg_iv']), axis=1)
    df['bs_vanna'] = df.apply(lambda r: black_scholes_vanna(spot, r['strike'], r['exp_t'], r['avg_iv']), axis=1)
    df = df.groupby(['exp','exp_t','strike'], as_index=False).sum()
    df['gex'] = (df['bs_gamma']*(df['call_open_int']-df['put_open_int']))*100
    df['dex'] = (df['call_delta']*df['call_open_int']+df['put_delta']*df['put_open_int'])*100
    df['vex'] = (df['call_vega']*df['call_open_int']-df['put_vega']*df['put_open_int'])*100
    df['call_td'] = np.where((df['call_mark']-df['call_bid']).abs()<(df['call_ask']-df['call_mark']).abs(),1,-1)
    df['put_td']  = np.where((df['put_ask']-df['put_mark']).abs()<(df['put_mark']-df['put_bid']).abs(),-1,1)
    df['dir_vol_gamma'] = df['call_volume']*df['bs_gamma']*df['call_delta']*df['call_td']\
                        + df['put_volume']*df['bs_gamma']*df['put_delta']*df['put_td']
    df['n_gex'] = safe_normalize(df['gex'])
    df['n_dex'] = safe_normalize(df['dex'])
    df['n_vex'] = safe_normalize(df['vex'])
    df['n_dir_gamma'] = safe_normalize(df['dir_vol_gamma'])
    df['final_composite'] = df[['n_gex','n_dex','n_vex','n_dir_gamma']].sum(axis=1)
    return df

def make_composite_figure(df):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['strike'], df['final_composite'], marker='o', linewidth=2)
    ax.axhline(0, color='black')
    ax.set_xlabel('Strike'); ax.set_ylabel('Composite')
    ax.set_title('Live Composite Exposure & Magnet Levels')
    ax.grid(True)
    return fig

@st.cache_data(ttl=30)
def fetch_data():
    tok = load_tokens()
    tok = refresh_access_token(tok)
    hdr = {'Authorization': f"Bearer {tok['access_token']}"}
    spot = requests.get(f'https://api.schwabapi.com/marketdata/v1/{UNDERLYING}/quotes', headers=hdr)\
                   .json()[UNDERLYING]['quote']['lastPrice']
    chain = requests.get(
        'https://api.schwabapi.com/marketdata/v1/chains',
        params={'symbol': UNDERLYING, 'strikeCount': STRIKE_COUNT},
        headers=hdr
    ).json()
    return chain_to_df(chain, spot)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Live SPX Composite Exposure")

if not os.path.exists(TOKEN_FILE):
    uploaded = st.file_uploader("Upload 'schwab_tokens.json'", type='json')
    if uploaded:
        with open(TOKEN_FILE, "wb") as f:
            f.write(uploaded.read())
        st.experimental_rerun()
    else:
        st.stop()

placeholder = st.empty()
while True:
    df = fetch_data()
    fig = make_composite_figure(df)
    placeholder.pyplot(fig)
    time.sleep(30)

