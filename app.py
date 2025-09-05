import streamlit as st
import numpy as np
from scipy.stats import norm

st.set_page_config(page_title="Calculadora Black-Scholes", layout="centered")

st.title("📈 Calculadora Black-Scholes para Opciones Europeas")

st.markdown("""
Esta app calcula el **precio teórico** de una opción europea usando el modelo de Black-Scholes, y también muestra las **Greeks** (Delta, Gamma, Theta, Vega y Rho).

""")

# --- Inputs ---
st.sidebar.header("Parámetros de entrada")

S = st.sidebar.number_input("Precio actual del activo (S)", min_value=0.01, value=50000.0)
K = st.sidebar.number_input("Precio de ejercicio (K)", min_value=0.01, value=52000.0)
days = st.sidebar.number_input("Días hasta el vencimiento", min_value=1, value=30)
T = days / 365.0  # Convertir a años
r = st.sidebar.number_input("Tasa libre de riesgo anual (r)", min_value=0.0, max_value=1.0, value=0.03)
sigma = st.sidebar.number_input("Volatilidad anual (σ)", min_value=0.01, max_value=2.0, value=0.6)
option_type = st.sidebar.selectbox("Tipo de opción", options=["call", "put"])

# --- Función de precio Black-Scholes ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# --- Greeks ---
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

# --- Cálculo ---
if st.button("Calcular precio de opción"):
    price = black_scholes(S, K, T, r, sigma, option_type)
    delta, gamma, theta, rho, vega = greeks(S, K, T, r, sigma, option_type)

    st.markdown(f"### 🎯 Precio de la opción **{option_type.upper()}**: ${price:,.2f}")
    
    # In the Money check
    if option_type == 'call':
        if S > K:
            st.success("✅ Esta opción CALL está **In The Money**.")
        elif S < K:
            st.info("ℹ️ Esta opción CALL está **Out of The Money**.")
        else:
            st.warning("⚠️ Esta opción CALL está **At The Money**.")
    else:
        if S < K:
            st.success("✅ Esta opción PUT está **In The Money**.")
        elif S > K:
            st.info("ℹ️ Esta opción PUT está **Out of The Money**.")
        else:
            st.warning("⚠️ Esta opción PUT está **At The Money**.")

    # Mostrar las Greeks
    st.subheader("📊 Greeks")
    st.markdown(f"""
    - **Δ Delta:** `{delta:.4f}`  
    - **Γ Gamma:** `{gamma:.4f}`  
    - **Θ Theta (por día):** `{theta / 365:.4f}`  
    - **𝜌 Rho:** `{rho:.4f}`  
    - **Vega:** `{vega:.4f}`
    """)


