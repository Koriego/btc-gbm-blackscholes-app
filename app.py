import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm

# --- Función Black-Scholes ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# --- Función para cargar datos BTC ---
@st.cache_data
def load_btc_data():
    today = datetime.today().strftime('%Y-%m-%d')
    btc = yf.download("BTC-USD", start="2021-01-01", end=today, interval="1d", auto_adjust=True)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    if 'Close' not in btc.columns:
        return pd.Series(dtype='float64')
    btc = btc.dropna(subset=['Close'])
    return btc['Close']

# --- Función simulación GBM ---
def simulate_gbm(S0, mu, sigma, dt, days, simulations):
    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = S0
    for t in range(1, days + 1):
        Z = np.random.standard_normal(simulations)
        price_paths[t] = price_paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return price_paths

# --- UI Streamlit ---
st.title("Simulación Bitcoin + Calculadora Black-Scholes")

# Parámetros simulación
num_simulations = st.sidebar.slider("Número de simulaciones", 10, 1000, 100, 10)
days_ahead = st.sidebar.slider("Días a simular", 30, 730, 365, 30)

prices = load_btc_data()
if prices.empty:
    st.error("No se pudieron cargar datos BTC.")
    st.stop()

log_returns = np.log(prices / prices.shift(1)).dropna()
mu = log_returns.mean()
sigma = log_returns.std()
S0 = prices[-1]
dt = 1

st.write(f"Último precio BTC: ${S0:,.2f}")
st.write(f"Media diaria (μ): {mu:.6f}")
st.write(f"Volatilidad diaria (σ): {sigma:.6f}")

# Simulación GBM
simulated_prices = simulate_gbm(S0, mu, sigma, dt, days_ahead, num_simulations)

# Mostrar gráfico simulación
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(simulated_prices, color='grey', alpha=0.3, lw=1)
ax.set_xlabel("Días")
ax.set_ylabel("Precio (USD)")
ax.grid(True)
st.pyplot(fig)

# --- Calculadora Black-Scholes ---
st.header("Calculadora Black-Scholes")

S = st.number_input("Precio actual del activo (S)", value=float(S0))
K = st.number_input("Precio de ejercicio (K)", value=52000.0)
T_days = st.number_input("Días hasta vencimiento", value=30)
r = st.number_input("Tasa libre de riesgo anual (decimal)", value=0.03)
volatility_input = st.number_input("Volatilidad anual (desviación estándar)", value=sigma * np.sqrt(252))
option_type = st.selectbox("Tipo de opción", ["call", "put"])

T = T_days / 365

def greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta = theta_call if option_type == 'call' else theta_put
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2)
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    rho = rho_call if option_type == 'call' else rho_put
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, theta, rho, vega

if st.button("Calcular precio de opción"):
    price = black_scholes(S, K, T, r, volatility_input, option_type)
    delta, gamma, theta, rho, vega = greeks(S, K, T, r, volatility_input, option_type)

    st.write(f"### 🎯 Precio de la opción {option_type.upper()}: **${price:.2f}**")
    st.write("### ⚖️ Greeks")
    st.markdown(f"""
    - **Δ Delta:** `{delta:.4f}`  
    - **Γ Gamma:** `{gamma:.4f}`  
    - **Θ Theta (por día):** `{theta / 365:.4f}`  
    - **𝜌 Rho:** `{rho:.4f}`  
    - **Vega:** `{vega:.4f}`
    """)

    st.write(f"Precio opción {option_type.upper()}: **${price:.2f}**")

# Puedes expandir esta app integrando más análisis o descargables si quieres :)

