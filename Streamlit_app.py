import streamlit as st
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- Your existing imports and helper functions --- 
# (chain_to_df, find_gamma_flip, plot_bar_exposure_figure, 
# plot_exposure_with_directional_volume_figure, plot_composite_and_magnets_figure, 
# plot_dynamic_composite_and_magnets_figure, load_tokens, save_tokens, refresh_access_token,
# fit_weights_by_bucket, get_authorization_url, exchange_code_for_token etc.)

UNDERLYING = "SPX"
STRIKE_COUNT = 50
HISTORICAL_CSV = "historical_data.csv"
DEFAULT_WEIGHTS_BY_BUCKET = {...}  # Your existing default weights dict

def main():
    st.set_page_config(page_title="Live SPX Exposures", layout="wide")
    st.title("Live SPX Composite Exposure & Greeks Dashboard")

    # --- Sidebar: OAuth2 Authorization ---
    with st.sidebar:
        st.header("Schwab API Authorization")

        # Load saved tokens if any
        tokens = load_tokens()  # Your function to load saved tokens (e.g. from disk or session_state)
        authorized = tokens and 'access_token' in tokens

        if not authorized:
            st.info("You must authorize access to Schwab API to view live data.")
            auth_url = get_authorization_url()
            st.markdown(f"[Click here to authorize Schwab API]({auth_url})", unsafe_allow_html=True)
            auth_code = st.text_input("Paste the authorization code here:")

            if auth_code:
                try:
                    tokens = exchange_code_for_token(auth_code)
                    save_tokens(tokens)
                    st.success("Authorization successful! Please refresh the page.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Authorization failed: {e}")
        else:
            st.success("Schwab API Authorized ✅")

    if not authorized:
        st.warning("Waiting for Schwab API authorization in sidebar...")
        return  # Skip rest of app until authorized

    placeholder = st.empty()

    while True:
        try:
            # ─── (A) LOAD & REFRESH TOKENS ──────────────────────────────────────────
            tok = load_tokens()
            tok = refresh_access_token(tok)
            save_tokens(tok)
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
                    st.warning(f"Historical CSV missing columns {missing}. Using default weights.")
                    weights_by_bucket = DEFAULT_WEIGHTS_BY_BUCKET
                else:
                    st.info("Fitting dynamic weights by DTE bucket from historical data…")
                    weights_by_bucket = fit_weights_by_bucket(hist)
                    st.success("Dynamic weight calibration complete.")
            except FileNotFoundError:
                st.warning(f"'{HISTORICAL_CSV}' not found. Using default weight buckets.")
                weights_by_bucket = DEFAULT_WEIGHTS_BY_BUCKET

            # ─── (F) PLOT & DISPLAY EACH FIGURE ────────────────────────────────────
            with placeholder.container():
                # 1) GEX Bar + Directional Volume Overlay
                try:
                    fig_gex = plot_bar_exposure_figure(df_agg, 'gex', 'Total Gamma Exposure (GEX) per Strike', 'GEX', gf)
                    if fig_gex:
                        st.pyplot(fig_gex)
                        plt.close(fig_gex)
                except Exception as e:
                    st.error(f"GEX plot error: {e}")

                try:
                    fig_gex_dir = plot_exposure_with_directional_volume_figure(
                        df_agg, 'gex', 'dir_vol_gamma', 'GEX', 'Total GEX + Directional Volume'
                    )
                    if fig_gex_dir:
                        st.pyplot(fig_gex_dir)
                        plt.close(fig_gex_dir)
                except Exception as e:
                    st.error(f"GEX directional volume plot error: {e}")

                # 2) DEX Bar + Directional Volume Overlay
                try:
                    fig_dex = plot_bar_exposure_figure(df_agg, 'dex', 'Total Delta Exposure (DEX) per Strike', 'DEX')
                    if fig_dex:
                        st.pyplot(fig_dex)
                        plt.close(fig_dex)
                except Exception as e:
                    st.error(f"DEX plot error: {e}")

                try:
                    fig_dex_dir = plot_exposure_with_directional_volume_figure(
                        df_agg, 'dex', 'dir_vol_delta', 'DEX', 'Total DEX + Directional Volume'
                    )
                    if fig_dex_dir:
                        st.pyplot(fig_dex_dir)
                        plt.close(fig_dex_dir)
                except Exception as e:
                    st.error(f"DEX directional volume plot error: {e}")

                # 3) VEX Bar + Directional Volume Overlay
                try:
                    fig_vex = plot_bar_exposure_figure(df_agg, 'vex', 'Total Vega Exposure (VEX) per Strike', 'VEX')
                    if fig_vex:
                        st.pyplot(fig_vex)
                        plt.close(fig_vex)
                except Exception as e:
                    st.error(f"VEX plot error: {e}")

                try:
                    fig_vex_dir = plot_exposure_with_directional_volume_figure(
                        df_agg, 'vex', 'dir_vol_vega', 'VEX', 'Total VEX + Directional Volume'
                    )
                    if fig_vex_dir:
                        st.pyplot(fig_vex_dir)
                        plt.close(fig_vex_dir)
                except Exception as e:
                    st.error(f"VEX directional volume plot error: {e}")

                # 4) CHX Bar + Directional Volume Overlay
                try:
                    fig_chx = plot_bar_exposure_figure(df_agg, 'chx', 'Total Charm Exposure (CHX) per Strike', 'CHX')
                    if fig_chx:
                        st.pyplot(fig_chx)
                        plt.close(fig_chx)
                except Exception as e:
                    st.error(f"CHX plot error: {e}")

                try:
                    fig_chx_dir = plot_exposure_with_directional_volume_figure(
                        df_agg, 'chx', 'dir_vol_charm', 'CHX', 'Total CHX + Directional Volume'
                    )
                    if fig_chx_dir:
                        st.pyplot(fig_chx_dir)
                        plt.close(fig_chx_dir)
                except Exception as e:
                    st.error(f"CHX directional volume plot error: {e}")

                # 5) VAX Bar + Directional Volume Overlay
                try:
                    fig_vax = plot_bar_exposure_figure(df_agg, 'vax', 'Total Vanna Exposure (VAX) per Strike', 'VAX')
                    if fig_vax:
                        st.pyplot(fig_vax)
                        plt.close(fig_vax)
                except Exception as e:
                    st.error(f"VAX plot error: {e}")

                try:
                    fig_vax_dir = plot_exposure_with_directional_volume_figure(
                        df_agg, 'vax', 'dir_vol_vanna', 'VAX', 'Total VAX + Directional Volume'
                    )
                    if fig_vax_dir:
                        st.pyplot(fig_vax_dir)
                        plt.close(fig_vax_dir)
                except Exception as e:
                    st.error(f"VAX directional volume plot error: {e}")

                # 6) Flat Composite & Magnet Levels
                try:
                    fig_comp = plot_composite_and_magnets_figure(df_agg)
                    if fig_comp:
                        st.pyplot(fig_comp)
                        plt.close(fig_comp)
                except Exception as e:
                    st.error(f"Composite plot error: {e}")

                # 7) Dynamic Composite & Magnet Levels (Always show)
                try:
                    st.info("Generating Dynamic Composite & Magnet Levels plot…")
                    fig_dyn = plot_dynamic_composite_and_magnets_figure(df, weights_by_bucket)
                    if fig_dyn:
                        st.pyplot(fig_dyn)
                        plt.close(fig_dyn)
                    else:
                        st.warning("Dynamic composite plot returned None.")
                except Exception as e:
                    st.error(f"Dynamic composite plot error: {e}")

                # Footer / last run timestamp
                st.markdown(f"*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*")

            # ─── (G) WAIT 30 SECONDS BEFORE REFRESH ─────────────────────────────────
            time.sleep(30)

        except Exception as e:
            st.error(f"Error: {e}")
            time.sleep(30)
            continue

if __name__ == "__main__":
    main()

