"""
Tablero Streamlit v2.1 — API de Análisis de Riesgo Financiero Global
30 activos · 4 regiones · 6 sectores
Perfil global → VaR y Markowitz se alinean con la recomendación
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Risk Analytics v2 — USTA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = "http://localhost:8000"

COLORES_REGION = {
    "Norteamérica": "#58a6ff",
    "Europa":       "#3fb950",
    "LatAm":        "#d29922",
    "Asia":         "#bc8cff",
}

PERFIL_CONFIG = {
    "conservador": {"emoji": "🛡️", "color": "#3fb950"},
    "moderado":    {"emoji": "⚖️", "color": "#58a6ff"},
    "agresivo":    {"emoji": "🚀", "color": "#f85149"},
}

st.markdown("""
<style>
div[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 10px; padding: 16px 20px;
}
.portafolio-banner {
    background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.3);
    border-radius: 8px; padding: 12px 16px; color: #3fb950;
    font-size: 13px; margin-bottom: 16px;
}
.portafolio-banner-base {
    background: rgba(88,166,255,0.08); border: 1px solid rgba(88,166,255,0.2);
    border-radius: 8px; padding: 12px 16px; color: #58a6ff;
    font-size: 13px; margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Estado global del portafolio ──────────────────────────────────────────────
# Se inicializa con el portafolio base y se actualiza desde Recomendaciones
if "portafolio_activo" not in st.session_state:
    st.session_state.portafolio_activo = {
        "tickers":  ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        "pesos":    [0.30, 0.25, 0.20, 0.15, 0.10],
        "fuente":   "base",
        "perfil":   "moderado",
        "label":    "Portafolio base (AAPL · MSFT · GOOGL · AMZN · TSLA)",
    }

if "perfil_global" not in st.session_state:
    st.session_state.perfil_global = "moderado"


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_api(path):
    try:
        r = requests.get(f"{API}{path}", timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=300)
def post_api(path, payload):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def plotly_theme():
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="monospace", size=11),
        margin=dict(l=10, r=10, t=40, b=10),
    )

def color_region(region):
    return COLORES_REGION.get(region, "#7d8590")

def banner_portafolio():
    """Muestra banner del portafolio activo con estado visual."""
    p = st.session_state.portafolio_activo
    tickers_str = " · ".join(p["tickers"][:4]) + ("..." if len(p["tickers"]) > 4 else "")
    perfil_cfg  = PERFIL_CONFIG.get(p["perfil"], PERFIL_CONFIG["moderado"])

    if p["fuente"] == "recomendacion":
        st.success(
            f"✓ **Portafolio recomendado activo** — "
            f"Perfil {perfil_cfg['emoji']} **{p['perfil']}** · "
            f"{len(p['tickers'])} activos: {tickers_str}"
        )
    else:
        st.info(
            f"ℹ️ Usando **portafolio base**: {tickers_str} — "
            f"Ve a **Recomendaciones** para generar un portafolio óptimo."
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Risk Analytics v2")
    st.markdown("**30 activos · 4 regiones · 6 sectores**")
    st.divider()

    health, err = get_api("/")
    catalogo, _ = get_api("/catalogo")

    if health:
        st.success(f"✓ API v{health.get('version','?')}")
    else:
        st.error(f"✗ Desconectada\n{err}")
        st.info("Ejecuta: `python main.py`")

    st.divider()

    # Selector de perfil global
    st.markdown("**Perfil de riesgo global**")
    perfil_sel = st.selectbox(
        "Perfil",
        ["conservador", "moderado", "agresivo"],
        index=["conservador","moderado","agresivo"].index(st.session_state.perfil_global),
        format_func=lambda x: {"conservador":"🛡️ Conservador","moderado":"⚖️ Moderado","agresivo":"🚀 Agresivo"}[x],
        label_visibility="collapsed",
        key="perfil_sidebar"
    )

    # Si cambia el perfil, resetear al portafolio base
    if perfil_sel != st.session_state.perfil_global:
        st.session_state.perfil_global = perfil_sel
        st.session_state.portafolio_activo = {
            "tickers":  ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "pesos":    [0.30, 0.25, 0.20, 0.15, 0.10],
            "fuente":   "base",
            "perfil":   perfil_sel,
            "label":    f"Portafolio base — genera recomendación para perfil {perfil_sel}",
        }
        st.cache_data.clear()
        st.rerun()

    # Estado del portafolio activo
    p = st.session_state.portafolio_activo
    if p["fuente"] == "recomendacion":
        st.success(f"✓ Portafolio recomendado activo\n{len(p['tickers'])} activos")
    else:
        st.warning("Portafolio base activo")

    st.divider()

    pagina = st.radio("Sección", [
        "Dashboard",
        "Precios e Indicadores",
        "VaR & CVaR",
        "Markowitz & CAPM",
        "Señales",
        "Comparar activos",
        "Recomendaciones",
        "Macroeconómico",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown(f"<small style='color:#7d8590'>API: `{API}`</small>", unsafe_allow_html=True)
    if st.button("🔄 Limpiar caché"):
        st.cache_data.clear()
        st.rerun()


# ════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════
if pagina == "Dashboard":
    st.title("Dashboard — Portafolio Global")

    if not health or not catalogo:
        st.error("No se puede conectar con la API.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activos globales",   catalogo["total_activos"])
    c2.metric("Regiones",           len(catalogo["regiones"]))
    c3.metric("Sectores",           len(catalogo["sectores"]))
    cfg = PERFIL_CONFIG[st.session_state.perfil_global]
    c4.metric("Perfil activo", f"{cfg['emoji']} {st.session_state.perfil_global.capitalize()}")

    # Estado portafolio
    banner_portafolio()

    st.divider()

    col_f1, col_f2 = st.columns(2)
    filtro_region = col_f1.selectbox("Filtrar por región", ["Todas"] + catalogo["regiones"])
    filtro_sector = col_f2.selectbox("Filtrar por sector", ["Todos"] + catalogo["sectores"])

    activos = [a for r in catalogo["por_region"].values() for a in r]
    if filtro_region != "Todas": activos = [a for a in activos if a["region"] == filtro_region]
    if filtro_sector != "Todos": activos = [a for a in activos if a["sector"] == filtro_sector]

    st.subheader(f"Catálogo — {len(activos)} activos")
    df = pd.DataFrame(activos)[["ticker","nombre","sector","pais","region","moneda"]]
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Distribución del catálogo")
    c_a, c_b = st.columns(2)
    with c_a:
        conteo_reg = pd.DataFrame(activos).groupby("region").size().reset_index(name="count")
        fig = px.pie(conteo_reg, names="region", values="count",
                     color="region", color_discrete_map=COLORES_REGION,
                     template="plotly_dark", hole=0.4, title="Por región")
        fig.update_layout(**plotly_theme(), height=280)
        st.plotly_chart(fig, use_container_width=True)
    with c_b:
        conteo_sec = pd.DataFrame(activos).groupby("sector").size().reset_index(name="count")
        fig2 = px.bar(conteo_sec, x="sector", y="count", color="sector",
                      template="plotly_dark", title="Por sector", height=280)
        fig2.update_layout(**plotly_theme())
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════
# PRECIOS E INDICADORES
# ════════════════════════════════════════
elif pagina == "Precios e Indicadores":
    st.title("Precios e Indicadores Técnicos")

    if not catalogo:
        st.error("Sin conexión a la API.")
        st.stop()

    todos = [f"{a['ticker']} — {a['nombre']} ({a['pais']})"
             for r in catalogo["por_region"].values() for a in r]

    col1, col2, col3 = st.columns([3, 2, 1])
    seleccion = col1.selectbox("Activo", todos)
    ticker    = seleccion.split(" — ")[0]
    fecha_ini = col2.selectbox("Horizonte",
                               ["2023-01-01","2022-01-01","2020-01-01"],
                               index=1,
                               format_func=lambda x: {"2023-01-01":"1 año","2022-01-01":"2 años","2020-01-01":"5 años"}[x])
    col3.button("Cargar", type="primary", use_container_width=True)

    with st.spinner(f"Descargando datos de {ticker}..."):
        ind_data, err = get_api(f"/indicadores/{ticker}?fecha_inicio={fecha_ini}")

    if err:
        st.error(f"Error: {err}")
        st.stop()

    datos   = pd.DataFrame(ind_data["datos"]).dropna(subset=["cierre"])
    señales = ind_data.get("señales", [])
    resumen = ind_data.get("resumen", {})

    precio_actual = datos["cierre"].iloc[-1]
    cambio_pct    = (precio_actual - datos["cierre"].iloc[0]) / datos["cierre"].iloc[0] * 100
    rsi_actual    = resumen.get("rsi_actual", 0) or 0
    vols          = np.log(datos["cierre"] / datos["cierre"].shift(1)).dropna()
    vol_anual     = vols.std() * np.sqrt(252) * 100

    info = next((a for r in catalogo["por_region"].values() for a in r if a["ticker"] == ticker), {})
    st.markdown(f"**{info.get('nombre',ticker)}** · {info.get('sector','N/A')} · {info.get('pais','N/A')} · {info.get('region','N/A')}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio actual",     f"${precio_actual:.2f}", f"{cambio_pct:+.2f}%")
    c2.metric("RSI actual",        f"{rsi_actual:.1f}",
              "Sobrecomprado" if rsi_actual>70 else "Sobrevendido" if rsi_actual<30 else "Neutral")
    c3.metric("Volatilidad anual", f"{vol_anual:.1f}%")
    c4.metric("Observaciones",     f"{len(datos)} días")

    if señales:
        st.divider()
        st.subheader("Señales activas")
        cols = st.columns(min(len(señales), 4))
        for i, s in enumerate(señales[:4]):
            with cols[i]:
                icon = "🟢" if s["tipo"]=="COMPRA" else "🔴"
                st.info(f"{icon} **{s['tipo']}** — {s['indicador']}\n\n{s['descripcion']}")

    color = color_region(info.get("region","Norteamérica"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos["fecha"], y=datos["cierre"],
        name=ticker, line=dict(color=color, width=2)))
    if "sma_20" in datos.columns:
        fig.add_trace(go.Scatter(x=datos["fecha"], y=datos["sma_20"],
            name="SMA 20", line=dict(color="#3fb950", width=1, dash="dot")))
    if "sma_50" in datos.columns:
        fig.add_trace(go.Scatter(x=datos["fecha"], y=datos["sma_50"],
            name="SMA 50", line=dict(color="#d29922", width=1, dash="dot")))
    fig.update_layout(**plotly_theme(), height=340, title=f"Precio histórico — {ticker}",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=datos["fecha"], y=datos["rsi_14"],
            name="RSI", line=dict(color="#bc8cff", width=1.5)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="rgba(248,81,73,0.5)", annotation_text="70")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="rgba(63,185,80,0.5)",  annotation_text="30")
        fig_rsi.update_layout(**plotly_theme(), height=240, title="RSI (14 días)", yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col_b:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=datos["fecha"], y=datos["macd"],
            name="MACD", line=dict(color="#58a6ff", width=1.5)))
        fig_macd.add_trace(go.Scatter(x=datos["fecha"], y=datos["macd_señal"],
            name="Signal", line=dict(color="#f85149", width=1)))
        hist = datos["macd_hist"].fillna(0)
        fig_macd.add_trace(go.Bar(x=datos["fecha"], y=hist, name="Hist.",
            marker_color=["rgba(63,185,80,0.6)" if v>=0 else "rgba(248,81,73,0.6)" for v in hist]))
        fig_macd.update_layout(**plotly_theme(), height=240, title="MACD",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_macd, use_container_width=True)

    fig_boll = go.Figure()
    fig_boll.add_trace(go.Scatter(x=datos["fecha"], y=datos["boll_superior"],
        name="Banda sup", line=dict(color="rgba(248,81,73,0.6)", width=1)))
    fig_boll.add_trace(go.Scatter(x=datos["fecha"], y=datos["boll_inferior"],
        name="Banda inf", line=dict(color="rgba(63,185,80,0.6)", width=1),
        fill="tonexty", fillcolor="rgba(88,166,255,0.03)"))
    fig_boll.add_trace(go.Scatter(x=datos["fecha"], y=datos["cierre"],
        name=ticker, line=dict(color="#fff", width=1.5)))
    fig_boll.update_layout(**plotly_theme(), height=280, title="Bandas de Bollinger",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_boll, use_container_width=True)


# ════════════════════════════════════════
# VaR & CVaR
# ════════════════════════════════════════
elif pagina == "VaR & CVaR":
    st.title("Valor en Riesgo (VaR) y CVaR")

    # Banner del portafolio activo
    banner_portafolio()

    p         = st.session_state.portafolio_activo
    confianza = st.select_slider("Nivel de confianza", [0.90, 0.95, 0.99], value=0.95,
                                  format_func=lambda x: f"{x*100:.0f}%")

    with st.spinner("Calculando VaR con 3 métodos..."):
        var_data, err = post_api("/var", {
            "tickers":          p["tickers"],
            "pesos":            p["pesos"],
            "nivel_confianza":  confianza,
        })

    if err:
        st.error(f"Error: {err}")
        st.stop()

    vh=var_data["var_historico"]; vp=var_data["var_parametrico"]
    vm=var_data["var_monte_carlo"]; st_p=var_data["estadisticas_portafolio"]
    kp=var_data["backtesting_kupiec"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("VaR Histórico",   vh["var_porcentaje"], f"${abs(vh['var_monetario']):,.0f}")
    c2.metric("CVaR Histórico",  vh["cvar_porcentaje"],f"${abs(vh['cvar_monetario']):,.0f}")
    c3.metric("Sharpe Ratio",    f"{st_p['sharpe_ratio']:.3f}")
    c4.metric("Volatilidad anual",f"{st_p['volatilidad_anual']*100:.1f}%")

    st.info(f"📌 {vh['interpretacion']}")

    df_var = pd.DataFrame([
        {"Método":"Histórico",   "VaR (%)":vh["var_porcentaje"],"VaR (USD)":f"${abs(vh['var_monetario']):,.0f}","CVaR (%)":vh["cvar_porcentaje"],"CVaR (USD)":f"${abs(vh['cvar_monetario']):,.0f}","Supuesto":"Sin supuesto distribucional"},
        {"Método":"Paramétrico", "VaR (%)":vp["var_porcentaje"],"VaR (USD)":f"${abs(vp['var_monetario']):,.0f}","CVaR (%)":vp["cvar_porcentaje"],"CVaR (USD)":f"${abs(vp['cvar_monetario']):,.0f}","Supuesto":"Distribución normal"},
        {"Método":"Monte Carlo", "VaR (%)":vm["var_porcentaje"],"VaR (USD)":f"${abs(vm['var_monetario']):,.0f}","CVaR (%)":vm["cvar_porcentaje"],"CVaR (USD)":f"${abs(vm['cvar_monetario']):,.0f}","Supuesto":"Normal (10k sim.)"},
    ])
    st.dataframe(df_var, use_container_width=True, hide_index=True)

    c_a, c_b = st.columns(2)
    labs = ["Histórico","Paramétrico","Monte Carlo"]
    vv = [abs(vh["var_decimal"])*100, abs(vp["var_decimal"])*100, abs(vm["var_decimal"])*100]
    cv = [abs(vh["cvar_decimal"])*100, abs(vp["cvar_decimal"])*100, abs(vm["cvar_decimal"])*100]

    with c_a:
        fig = go.Figure(go.Bar(x=labs, y=vv,
            marker_color=["#f85149","#d29922","#58a6ff"],
            text=[f"{v:.2f}%" for v in vv], textposition="outside"))
        fig.update_layout(**plotly_theme(), height=300, title="VaR por método", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="VaR",  x=labs, y=vv, marker_color="rgba(248,81,73,0.6)"))
        fig2.add_trace(go.Bar(name="CVaR", x=labs, y=cv, marker_color="rgba(248,81,73,1.0)"))
        fig2.update_layout(**plotly_theme(), height=300, title="VaR vs CVaR", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Backtesting Kupiec")
    c1,c2,c3 = st.columns(3)
    c1.metric("Excedencias", kp["excedencias_observadas"], f"de {kp['total_observaciones']} días")
    c2.metric("Tasa real",    f"{kp['tasa_excedencias_real']*100:.2f}%")
    c3.metric("Tasa esperada",f"{kp['tasa_excedencias_esperada']*100:.2f}%")
    if kp["modelo_adecuado"]:
        st.success(f"✓ {kp['interpretacion']}")
    else:
        st.warning(f"⚠ {kp['interpretacion']}")


# ════════════════════════════════════════
# MARKOWITZ & CAPM
# ════════════════════════════════════════
elif pagina == "Markowitz & CAPM":
    st.title("Frontera Eficiente de Markowitz & CAPM")

    # Banner del portafolio activo
    banner_portafolio()

    p = st.session_state.portafolio_activo
    st.caption(f"Portafolio: {' · '.join(p['tickers'])} — ~30s")

    with st.spinner("Optimizando portafolios y calculando Beta..."):
        frontera, err1 = post_api("/frontera-eficiente", {
            "tickers": p["tickers"],
            "pesos":   p["pesos"],
        })
        capm_params = "&".join(f"tickers={t}" for t in p["tickers"])
        capm_data, err2 = get_api(f"/capm?{capm_params}")

    if err1 or err2:
        st.error(f"Error: {err1 or err2}")
        st.stop()

    c1,c2,c3 = st.columns(3)
    for col, titulo, port in [
        (c1, "Mínima varianza",  frontera["portafolio_min_varianza"]),
        (c2, "Máximo Sharpe",    frontera["portafolio_max_sharpe"]),
        (c3, "Igual ponderado",  frontera["portafolio_igual_ponderado"]),
    ]:
        with col:
            col.metric(titulo, port["retorno_pct"],
                       f"Vol: {port['volatilidad_pct']} · Sharpe: {port['sharpe_ratio']:.3f}")
            df_p = pd.DataFrame(list(port["pesos"].items()), columns=["Ticker","Peso"])
            df_p["Peso (%)"] = (df_p["Peso"]*100).round(1).astype(str)+"%"
            st.dataframe(df_p[["Ticker","Peso (%)"]], hide_index=True, use_container_width=True)

    sim=frontera["simulacion"]; front=frontera["frontera_eficiente"]
    minV=frontera["portafolio_min_varianza"]; maxS=frontera["portafolio_max_sharpe"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[v*100 for v in sim["volatilidades"]], y=[r*100 for r in sim["retornos"]],
        mode="markers", name="Portafolios simulados",
        marker=dict(size=5, color=sim["sharpes"], colorscale="Blues",
                    showscale=True, colorbar=dict(title="Sharpe",thickness=12), opacity=0.6)))
    fig.add_trace(go.Scatter(
        x=[p["volatilidad"]*100 for p in front], y=[p["retorno"]*100 for p in front],
        mode="lines", name="Frontera eficiente", line=dict(color="#3fb950", width=2.5)))
    fig.add_trace(go.Scatter(
        x=[minV["volatilidad_anual"]*100], y=[minV["retorno_anual"]*100],
        mode="markers", name="Mínima varianza",
        marker=dict(size=14, color="#3fb950", symbol="star")))
    fig.add_trace(go.Scatter(
        x=[maxS["volatilidad_anual"]*100], y=[maxS["retorno_anual"]*100],
        mode="markers", name="Máximo Sharpe",
        marker=dict(size=14, color="#58a6ff", symbol="triangle-up")))
    fig.update_layout(**plotly_theme(), height=420,
                      xaxis_title="Volatilidad (%)", yaxis_title="Retorno (%)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    activos_capm = capm_data.get("activos",{})
    rows = [{
        "Ticker":             t,
        "Beta":               round(d.get("beta",0),4),
        "Alpha anual (%)":    round((d.get("alpha_anual",0) or 0)*100,2),
        "Rend. CAPM (%)":     round((d.get("rendimiento_esperado_capm",0) or 0)*100,2),
        "Volatilidad (%)":    round((d.get("volatilidad_anual",0) or 0)*100,1),
        "R²":                 round(d.get("r_cuadrado",0) or 0,4),
        "Interpretación Beta":d.get("interpretacion_beta","—"),
    } for t,d in activos_capm.items()]
    st.subheader("CAPM — Beta y rendimiento esperado")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Gráfico Beta
    fig_beta = go.Figure(go.Bar(
        x=[r["Ticker"] for r in rows],
        y=[r["Beta"] for r in rows],
        marker_color=[COLORES_REGION.get(
            next((a["region"] for reg in catalogo["por_region"].values()
                  for a in reg if a["ticker"]==r["Ticker"]), "Norteamérica"),
            "#58a6ff") for r in rows] if catalogo else ["#58a6ff"]*len(rows),
        text=[f"{r['Beta']:.3f}" for r in rows],
        textposition="outside",
    ))
    fig_beta.add_hline(y=1, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                       annotation_text="Beta = 1 (mercado)")
    fig_beta.update_layout(**plotly_theme(), height=300, title="Beta por activo", showlegend=False)
    st.plotly_chart(fig_beta, use_container_width=True)


# ════════════════════════════════════════
# SEÑALES
# ════════════════════════════════════════
elif pagina == "Señales":
    st.title("Señales automáticas de trading")

    if not catalogo:
        st.error("Sin conexión a la API.")
        st.stop()

    col1, col2 = st.columns([2,1])
    region_sel = col1.selectbox(
        "Región a analizar",
        ["Portafolio activo"] + catalogo["regiones"]
    )
    actualizar = col2.button("Actualizar señales", type="primary")
    if actualizar:
        st.cache_data.clear()

    p = st.session_state.portafolio_activo
    if region_sel == "Portafolio activo":
        tickers_señales = p["tickers"]
        st.caption(f"Analizando: {' · '.join(tickers_señales)}")
    else:
        tickers_señales = [a["ticker"] for a in catalogo["por_region"].get(region_sel,[])]
        st.caption(f"Analizando región {region_sel}: {len(tickers_señales)} activos")

    url = "/alertas?" + "&".join(f"tickers={t}" for t in tickers_señales)
    with st.spinner("Analizando indicadores técnicos..."):
        alertas_data, err = get_api(url)

    if err:
        st.error(f"Error: {err}")
        st.stop()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total alertas",  alertas_data["total_alertas"])
    c2.metric("Señales compra", alertas_data["alertas_compra"],  delta="alcistas")
    c3.metric("Señales venta",  alertas_data["alertas_venta"],   delta="bajistas")
    c4.metric("Fecha análisis", alertas_data["fecha_analisis"])

    st.divider()
    resumen = alertas_data.get("resumen",{})
    n_cols  = min(len(resumen), 5)
    if n_cols > 0:
        cols = st.columns(n_cols)
        for i,(ticker,d) in enumerate(list(resumen.items())[:n_cols]):
            if d.get("error"): continue
            with cols[i]:
                señal = d.get("señal_neta","NEUTRAL")
                icon  = "🟢" if señal=="COMPRA" else "🔴" if señal=="VENTA" else "⚪"
                st.metric(f"{icon} {ticker}", f"${d.get('precio_actual',0):.2f}",
                          f"RSI: {d.get('rsi_actual',0):.1f}")
                if señal=="COMPRA":   st.success(f"**{señal}**")
                elif señal=="VENTA":  st.error(f"**{señal}**")
                else:                 st.info(f"**{señal}**")

    alertas = alertas_data.get("alertas",[])
    if alertas:
        df_a = pd.DataFrame(alertas)
        df_a["tipo"] = df_a["tipo"].map({"COMPRA":"🟢 COMPRA","VENTA":"🔴 VENTA"})
        st.dataframe(
            df_a[["ticker","tipo","indicador","fuerza","descripcion","valor","fecha"]],
            use_container_width=True, hide_index=True)

        st.subheader("Distribución por indicador")
        conteo = df_a.groupby(["indicador","tipo"]).size().reset_index(name="count")
        fig_dist = px.bar(conteo, x="indicador", y="count", color="tipo",
                          color_discrete_map={"🟢 COMPRA":"#3fb950","🔴 VENTA":"#f85149"},
                          template="plotly_dark", height=300)
        fig_dist.update_layout(**plotly_theme())
        st.plotly_chart(fig_dist, use_container_width=True)


# ════════════════════════════════════════
# COMPARAR ACTIVOS
# ════════════════════════════════════════
elif pagina == "Comparar activos":
    st.title("Comparar activos globales")
    st.info("Compara activos de distintas regiones, sectores y países lado a lado.")

    if not catalogo:
        st.error("Sin conexión a la API.")
        st.stop()

    todos = [f"{a['ticker']} — {a['nombre']} ({a['region']})"
             for r in catalogo["por_region"].values() for a in r]

    col1, col2 = st.columns([3,1])
    seleccionados = col1.multiselect(
        "Selecciona activos (mín. 2, máx. 8)",
        todos,
        default=[
            "AAPL — Apple Inc. (Norteamérica)",
            "SAP.DE — SAP SE (Europa)",
            "TM — Toyota Motor (Asia)",
            "EC — Ecopetrol S.A. (LatAm)",
        ]
    )
    fecha_cmp = col2.selectbox(
        "Horizonte",
        ["2022-01-01","2020-01-01","2023-01-01"],
        format_func=lambda x: {"2022-01-01":"2 años","2020-01-01":"5 años","2023-01-01":"1 año"}[x]
    )

    if len(seleccionados) < 2:
        st.warning("Selecciona al menos 2 activos.")
        st.stop()

    tickers_cmp = [s.split(" — ")[0] for s in seleccionados]
    params = "&".join(f"tickers={t}" for t in tickers_cmp)
    url    = f"/comparar?{params}&fecha_inicio={fecha_cmp}"

    with st.spinner(f"Comparando {len(tickers_cmp)} activos globales..."):
        comp_data, err = get_api(url)

    if err:
        st.error(f"Error: {err}")
        st.stop()

    comp = comp_data["comparacion"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mejor Sharpe",      comp_data.get("mejor_sharpe","—"))
    c2.metric("Mejor retorno",     comp_data.get("mejor_retorno","—"))
    c3.metric("Menor volatilidad", comp_data.get("menor_volatilidad","—"))
    c4.metric("Activos",           comp_data["total_activos"])

    vals   = list(comp.values())
    labels = [v["ticker"] for v in vals]
    colors = [COLORES_REGION.get(v["region"],"#7d8590") for v in vals]

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure(go.Bar(
            x=labels, y=[(v["retorno_total"] or 0)*100 for v in vals],
            marker_color=colors,
            text=[v["retorno_total_pct"] for v in vals], textposition="outside"))
        fig.update_layout(**plotly_theme(), height=300, title="Retorno total (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = go.Figure(go.Bar(
            x=labels, y=[v["sharpe_ratio"] or 0 for v in vals],
            marker_color=colors,
            text=[f"{v['sharpe_ratio']:.3f}" if v['sharpe_ratio'] else '—' for v in vals],
            textposition="outside"))
        fig2.update_layout(**plotly_theme(), height=300, title="Sharpe Ratio", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig3 = go.Figure(go.Bar(
            x=labels, y=[(v["volatilidad_anual"] or 0)*100 for v in vals],
            marker_color=colors))
        fig3.update_layout(**plotly_theme(), height=260, title="Volatilidad (%)", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = go.Figure(go.Bar(
            x=labels, y=[abs(v["max_drawdown"] or 0)*100 for v in vals],
            marker_color=colors))
        fig4.update_layout(**plotly_theme(), height=260, title="Máx. Drawdown (%)", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Tabla comparativa completa")
    rows = sorted(vals, key=lambda x: x["ranking_sharpe"] or 99)
    df_comp = pd.DataFrame([{
        "Ranking":       f"#{v['ranking_sharpe']}",
        "Ticker":        v["ticker"],
        "Región":        v["region"],
        "Sector":        v["sector"],
        "País":          v["pais"],
        "Retorno total": v["retorno_total_pct"],
        "Retorno anual": v["retorno_anual_pct"],
        "Volatilidad":   v["volatilidad_anual_pct"],
        "Sharpe":        f"{v['sharpe_ratio']:.3f}" if v["sharpe_ratio"] else "—",
        "Max DD":        v["max_drawdown_pct"],
        "RSI":           f"{v['rsi_actual']:.1f}" if v["rsi_actual"] else "—",
        "Tendencia":     v["tendencia_ema"],
    } for v in rows])
    st.dataframe(df_comp, use_container_width=True, hide_index=True)


# ════════════════════════════════════════
# RECOMENDACIONES
# ════════════════════════════════════════
elif pagina == "Recomendaciones":
    st.title("Motor de recomendaciones")
    st.markdown("**Scoring multifactor:** Sharpe Ratio · Señales técnicas · Momentum 3M · Volatilidad")
    st.info("Al generar una recomendación, el portafolio se aplica automáticamente a **VaR** y **Markowitz**.")

    perfil_actual = st.session_state.perfil_global
    col1, col2, col3 = st.columns([2, 2, 1])
    perfil = col1.selectbox(
        "Perfil de riesgo",
        ["conservador","moderado","agresivo"],
        index=["conservador","moderado","agresivo"].index(perfil_actual),
        format_func=lambda x: {"conservador":"🛡️ Conservador","moderado":"⚖️ Moderado","agresivo":"🚀 Agresivo"}[x],
        key="perfil_rec"
    )
    region_rec = col2.selectbox(
        "Universo de activos",
        ["Global (recomendado)"] + (catalogo["regiones"] if catalogo else [])
    )
    calcular = col3.button("Recomendar", type="primary", use_container_width=True)

    # Sincronizar perfil global si cambia aquí
    if perfil != st.session_state.perfil_global:
        st.session_state.perfil_global = perfil

    url = f"/recomendar?perfil_riesgo={perfil}"
    if region_rec != "Global (recomendado)":
        url += f"&region={region_rec}"

    with st.spinner("Analizando mercado y calculando scores (~60s)..."):
        rec_data, err = get_api(url)

    if err:
        st.error(f"Error: {err}")
        st.stop()

    port = rec_data["portafolio_recomendado"]
    just = rec_data["justificacion"]
    met  = rec_data["metricas_portafolio"]
    div  = rec_data["diversificacion"]

    # ── ACTUALIZAR ESTADO GLOBAL ──────────────────────────────────────────────
    tickers_rec = list(port.keys())
    pesos_rec   = [port[t]["peso"] for t in tickers_rec]
    st.session_state.portafolio_activo = {
        "tickers": tickers_rec,
        "pesos":   pesos_rec,
        "fuente":  "recomendacion",
        "perfil":  perfil,
        "label":   f"{perfil} · {' · '.join(tickers_rec[:3])}{'...' if len(tickers_rec)>3 else ''}",
    }
    st.session_state.perfil_global = perfil
    # Limpiar caché para que VaR y Markowitz recalculen
    st.cache_data.clear()

    cfg_perfil = PERFIL_CONFIG[perfil]
    st.success(
        f"✓ Portafolio guardado — perfil {cfg_perfil['emoji']} **{perfil}** · "
        f"Retorno esperado: **{met['retorno_esperado_pct']}** · "
        f"VaR y Markowitz actualizados automáticamente."
    )

    # Tabla portafolio
    rows_p = sorted(port.items(), key=lambda x: -x[1]["peso"])
    df_port = pd.DataFrame([{
        "Ticker":      t,
        "Empresa":     d["nombre"],
        "Región":      d["region"],
        "Sector":      d["sector"],
        "País":        d["pais"],
        "Peso (%)":    d["peso_pct"],
        "Score":       f"{d['score']:.3f}",
        "Sharpe":      f"{d['sharpe']:.3f}",
        "Retorno":     d["retorno_anual"],
        "Volatilidad": d["volatilidad"],
        "Momentum 3M": d["momentum_3m"],
    } for t,d in rows_p])
    st.dataframe(df_port, use_container_width=True, hide_index=True)

    # Diversificación
    col_a, col_b = st.columns(2)
    with col_a:
        df_reg = pd.DataFrame(list(div["por_region"].items()), columns=["Región","Peso"])
        df_reg["Peso (%)"] = df_reg["Peso"]*100
        fig_reg = px.pie(df_reg, names="Región", values="Peso (%)",
                         color="Región", color_discrete_map=COLORES_REGION,
                         template="plotly_dark", hole=0.4, title="Por región")
        fig_reg.update_layout(**plotly_theme(), height=300)
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_b:
        df_sec = pd.DataFrame(list(div["por_sector"].items()), columns=["Sector","Peso"])
        df_sec["Peso (%)"] = df_sec["Peso"]*100
        fig_sec = px.bar(df_sec, x="Sector", y="Peso (%)",
                         template="plotly_dark", title="Por sector")
        fig_sec.update_layout(**plotly_theme(), height=300)
        st.plotly_chart(fig_sec, use_container_width=True)

    # Justificación
    st.divider()
    st.subheader("Justificación")
    st.write(just["resumen"])

    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown(f"**Activo destacado:** `{just['activo_destacado']['ticker']}`")
        st.caption(just["activo_destacado"]["razon"])
        st.markdown(f"**Diversificación:** {just['diversificacion']}")
    with col_f:
        for alerta in just.get("alertas_riesgo",[]):
            st.warning(f"⚠ {alerta}")
        st.info(f"💡 {just['recomendacion_accion']}")

    # Metodología
    st.divider()
    st.subheader("Metodología — pesos del scoring")
    factores = rec_data.get("metodologia",{}).get("factores",{})
    cols = st.columns(len(factores))
    for i,(k,v) in enumerate(factores.items()):
        cols[i].metric(k, v)


# ════════════════════════════════════════
# MACROECONÓMICO
# ════════════════════════════════════════
elif pagina == "Macroeconómico":
    st.title("Indicadores Macroeconómicos — FRED")

    macro_data, err = get_api("/macro")
    if err:
        st.error(f"Error: {err}")
        st.stop()

    datos = macro_data.get("datos",{})
    config = {
        "DGS3MO":   ("🏦","T-Bills 3M (Rf)"),
        "DGS10":    ("📊","Tesoro 10 años"),
        "CPIAUCSL": ("📈","Inflación CPI"),
        "UNRATE":   ("👥","Desempleo"),
        "FEDFUNDS": ("⚙️","Tasa Fed"),
        "VIXCLS":   ("⚡","VIX"),
    }

    keys = [k for k in config if k in datos and not k.startswith("_")]
    cols = st.columns(len(keys))
    for i,k in enumerate(keys):
        d, cfg = datos[k], config[k]
        val    = d.get("valor")
        unit   = "pts" if k=="VIXCLS" else "%"
        cols[i].metric(f"{cfg[0]} {cfg[1]}",
                       f"{val:.2f}{unit}" if val is not None else "—",
                       d.get("fecha",""))

    if macro_data.get("nota"):
        st.info(f"ℹ️ {macro_data['nota']}")
        st.markdown("[Obtener clave FRED gratis →](https://fred.stlouisfed.org/docs/api/api_key.html)")

    ctx = macro_data.get("contexto_macro")
    if ctx:
        st.divider()
        st.subheader("Contexto macroeconómico")
        st.write(ctx.get("descripcion",""))
        for item in ctx.get("impacto_portafolio",[]):
            st.markdown(f"→ {item}")

        tasas_keys = [k for k in ["DGS3MO","DGS10","FEDFUNDS","CPIAUCSL","UNRATE"] if k in datos]
        if tasas_keys:
            fig = go.Figure(go.Bar(
                x=[config[k][1] for k in tasas_keys],
                y=[datos[k].get("valor",0) for k in tasas_keys],
                marker_color=["#58a6ff","#3fb950","#d29922","#f85149","#bc8cff"],
                text=[f"{datos[k].get('valor',0):.2f}%" for k in tasas_keys],
                textposition="outside"))
            fig.update_layout(**plotly_theme(), height=320,
                              title="Comparación de tasas", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
