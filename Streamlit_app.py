import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

# ------------------- CONFIG -------------------
CLIENT_ID = "TeyurKNMQmDOlYa8v534SYvGDgnbt1H6"
CLIENT_SECRET = "cYHDTdzmQX4nKrGM"
REDIRECT_URI = "https://jchavez5656.github.io/chavez-redirect/"
TOKEN_FILE = "token.json"

# ------------------- AUTH HELPERS -------------------
def load_tokens():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return json.load(f)
    return None

def save_tokens(tokens):
    with open(TOKEN_FILE, 'w') as f:
        json.dump(tokens, f)

def refresh_access_token(refresh_token):
    url = "https://api.schwabapi.com/v1/oauth/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI
    }
    response = requests.post(url, headers=headers, data=data)
    if response.ok:
        tokens = response.json()
        tokens['refresh_token'] = refresh_token
        save_tokens(tokens)
        return tokens['access_token']
    else:
        st.error("Token refresh failed.")
        st.stop()

def get_access_token():
    tokens = load_tokens()
    if tokens:
        return refresh_access_token(tokens['refresh_token'])
    else:
        st.error("No token found. Please authenticate first.")
        st.stop()

# ------------------- SCHWAB API -------------------
def fetch_option_chain(symbol, access_token):
    url = f"https://api.schwabapi.com/marketdata/v1/chains"
    params = {
        "symbol": symbol,
        "includeQuotes": "FALSE",
        "strategy": "SINGLE",
        "contractType": "ALL",
        "range": "ALL"
    }
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers, params=params)
    if response.ok:
        return response.json()
    else:
        st.error("Failed to fetch option chain.")
        st.stop()

# ------------------- METRIC CALCS -------------------
def calc_exposure_data(chain_data):
    calls = []
    puts = []

    strike_prices = set()

    for strike, contracts in chain_data.get("callExpDateMap", {}).items():
        for strike_price, data in contracts.items():
            option = data[0]
            calls.append({
                "strike": float(option["strikePrice"]),
                "openInterest": option["openInterest"],
                "delta": option.get("delta", 0),
                "gamma": option.get("gamma", 0),
                "vega": option.get("vega", 0),
                "type": "call"
            })
            strike_prices.add(float(option["strikePrice"]))

    for strike, contracts in chain_data.get("putExpDateMap", {}).items():
        for strike_price, data in contracts.items():
            option = data[0]
            puts.append({
                "strike": float(option["strikePrice"]),
                "openInterest": option["openInterest"],
                "delta": option.get("delta", 0),
                "gamma": option.get("gamma", 0),
                "vega": option.get("vega", 0),
                "type": "put"
            })
            strike_prices.add(float(option["strikePrice"]))

    df = pd.DataFrame(calls + puts)
    df["DEX"] = df["openInterest"] * df["delta"]
    df["GEX"] = df["openInterest"] * df["gamma"]
    df["VEX"] = df["openInterest"] * df["vega"]
    return df

def aggregate_metrics(df):
    grouped = df.groupby("strike").agg({
        "DEX": "sum",
        "GEX": "sum",
        "VEX": "sum"
    }).reset_index()
    return grouped

def find_top_magnets(df):
    df["absGEX"] = df["GEX"].abs()
    top = df.sort_values("absGEX", ascending=False).head(3)
    return top["strike"].tolist()

def round_to_nearest_25(x):
    return int(round(x / 25.0) * 25)

# ------------------- PLOTTING -------------------
def plot_exposure(df, magnet_levels):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["strike"], y=df["GEX"],
        name="Gamma Exposure", marker_color="blue"
    ))

    fig.add_trace(go.Bar(
        x=df["strike"], y=df["DEX"],
        name="Delta Exposure", marker_color="orange"
    ))

    fig.add_trace(go.Bar(
        x=df["strike"], y=df["VEX"],
        name="Vega Exposure", marker_color="green"
    ))

    for level in magnet_levels:
        fig.add_vline(
            x=level,
            line=dict(color="red", dash="dash"),
            annotation_text=f"Magnet: {level}",
            annotation_position="top"
        )

    fig.update_layout(
        barmode="overlay",
        title="SPX Options Exposure (DEX / GEX / VEX)",
        xaxis_title="Strike Price",
        yaxis_title="Exposure",
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(
                round_to_nearest_25(df["strike"].min()) - 50,
                round_to_nearest_25(df["strike"].max()) + 50,
                25
            )
        ),
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------- MAIN -------------------
def main():
    st.title("SPX Options Exposure Dashboard (DEX/GEX/VEX)")

    with st.sidebar:
        st.header("Schwab OAuth")
        code = st.text_input("Paste the Authorization Code Here")
        if st.button("Authorize and Save Tokens"):
            if not code:
                st.warning("Paste the code first.")
                return

            token_url = "https://api.schwabapi.com/v1/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI
            }

            response = requests.post(token_url, headers=headers, data=data)
            if response.ok:
                tokens = response.json()
                save_tokens(tokens)
                st.success("Tokens saved successfully!")
            else:
                st.error("Failed to exchange code for tokens.")

    if not os.path.exists(TOKEN_FILE):
        st.warning("Please authorize Schwab API using sidebar first.")
        return

    access_token = get_access_token()
    data = fetch_option_chain("SPX", access_token)
    df_raw = calc_exposure_data(data)
    df = aggregate_metrics(df_raw)
    magnets = find_top_magnets(df)

    plot_exposure(df, magnets)

if __name__ == "__main__":
    main()

