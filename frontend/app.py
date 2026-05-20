"""
app.py — Dashboard Shiny v2 CORREGIDO
======================================
Correcciones respecto a v1:
  1. Selector de activos carga dinámicamente desde GET /activos (no hardcodeado)
  2. VaR: fix matmul — el backend maneja el cálculo, el frontend solo muestra
  3. Black-Scholes: gráfico de payoff removido (era el que intentaba importar app.services)
  4. ML: tabla de features ahora muestra los valores correctamente
  5. Comparación: botón ahora dispara correctamente el reactive
  6. Portafolio default usa los 8 activos recomendados

Ejecución:
  cd frontend
  shiny run app.py --reload --port 8501
"""

from shiny import App, ui, render, reactive
import requests
import pandas as pd
from datetime import date

API = "http://localhost:8000"

TICKERS_DEFAULT   = ["AAPL", "JPM", "XOM", "JNJ", "EC", "CIB", "SAP.DE", "NOVN.SW"]
PESOS_DEFAULT     = "0.15,0.15,0.12,0.12,0.12,0.10,0.12,0.12"
TICKERS_STR       = ",".join(TICKERS_DEFAULT)


def api_get(path, params=None):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=45)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Sin conexión al backend ({API})"
    except Exception as e:
        try:
            detail = r.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail


def api_post(path, payload):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=90)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Sin conexión al backend ({API})"
    except Exception as e:
        try:
            detail = r.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail


def fmt_pct(v):
    if v is None: return "—"
    try: return f"{float(v)*100:.2f}%"
    except: return str(v)

def fmt_usd(v):
    if v is None: return "—"
    try: return f"${float(v):,.2f}"
    except: return str(v)

def color_num(v, positive_good=True):
    if v is None: return "—"
    try:
        f = float(v)
        color = "#34d399" if (f > 0) == positive_good else "#f87171"
        return f'<span style="color:{color};font-weight:600">{f:.4f}</span>'
    except:
        return str(v)


# ─────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────

app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.style("""
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        body{font-family:'IBM Plex Sans',sans-serif;background:#0f1117;color:#e2e8f0;margin:0}
        .card{background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;padding:20px;margin-bottom:16px}
        .card-title{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#94a3b8;margin-bottom:10px}
        .metric-big{font-family:'IBM Plex Mono',monospace;font-size:26px;font-weight:700;color:#60a5fa}
        .metric-pos{color:#34d399!important} .metric-neg{color:#f87171!important} .metric-neu{color:#60a5fa!important}
        .header-bar{background:linear-gradient(135deg,#1e40af,#7c3aed 50%,#db2777);padding:18px 28px;border-radius:12px;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between}
        .header-title{font-size:22px;font-weight:700;color:white;margin:0}
        .header-sub{font-size:13px;color:rgba(255,255,255,.75);margin:4px 0 0}
        .status-ok{background:#064e3b;color:#34d399;padding:4px 14px;border-radius:8px;font-size:12px;font-weight:600}
        .status-err{background:#450a0a;color:#f87171;padding:4px 14px;border-radius:8px;font-size:12px;font-weight:600}
        .badge-buy{display:inline-block;background:#064e3b;color:#34d399;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:600;margin:2px}
        .badge-sell{display:inline-block;background:#450a0a;color:#f87171;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:600;margin:2px}
        .badge-neu{display:inline-block;background:#1e293b;color:#94a3b8;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:600;margin:2px}
        table{width:100%;border-collapse:collapse;font-size:13px}
        th{background:#1e293b;color:#94a3b8;text-align:left;padding:9px 12px;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
        td{padding:9px 12px;border-bottom:1px solid #1e293b;font-family:'IBM Plex Mono',monospace}
        tr:hover td{background:#1e293b55}
        hr{border:none;border-top:1px solid #2d3748;margin:16px 0}
        .shiny-input-container label{color:#94a3b8!important;font-size:12px!important;font-weight:500!important;text-transform:uppercase!important;letter-spacing:.04em!important}
        .form-control,.selectize-input{background:#1a1f2e!important;border:1px solid #2d3748!important;color:#e2e8f0!important;border-radius:8px!important}
        .btn-primary{background:linear-gradient(135deg,#1e40af,#7c3aed)!important;border:none!important;border-radius:8px!important;font-weight:600!important;width:100%!important;padding:10px!important}
        .err{color:#f87171;font-size:13px;padding:8px;background:#450a0a22;border-radius:6px;border-left:3px solid #f87171}
    """)),

    # Header
    ui.div(
        ui.div(
            ui.tags.h1("Risk Analytics", class_="header-title"),
            ui.tags.p("Proyecto Integrador — Teoría del Riesgo · USTA 2026-I", class_="header-sub"),
        ),
        ui.output_ui("status_badge"),
        class_="header-bar",
    ),

    ui.navset_tab(

        # ═══ TAB 1: INDICADORES ═════════════════════════
        ui.nav_panel("📊 Indicadores",
            ui.row(
                ui.column(3,
                    ui.div(
                        ui.tags.p("Activo", class_="card-title"),
                        ui.output_ui("ind_selector"),
                        ui.input_date_range("ind_fechas", "Período",
                            start="2023-01-01", end=str(date.today())),
                        ui.input_action_button("ind_btn", "Calcular indicadores", class_="btn-primary"),
                        class_="card",
                    ),
                    ui.div(
                        ui.tags.p("Señales activas", class_="card-title"),
                        ui.output_ui("ind_signals"),
                        class_="card",
                    ),
                ),
                ui.column(9,
                    ui.div(ui.output_ui("ind_metrics"), class_="card"),
                    ui.div(
                        ui.tags.p("Precio + SMA20 + SMA50 + Bollinger", class_="card-title"),
                        ui.output_ui("ind_chart"),
                        class_="card",
                    ),
                    ui.row(
                        ui.column(6, ui.div(ui.tags.p("RSI (14)", class_="card-title"), ui.output_ui("ind_rsi"), class_="card")),
                        ui.column(6, ui.div(ui.tags.p("MACD + Histograma", class_="card-title"), ui.output_ui("ind_macd"), class_="card")),
                    ),
                ),
            ),
        ),

        # ═══ TAB 2: RIESGO ══════════════════════════════
        ui.nav_panel("⚠️ Riesgo",
            ui.row(
                ui.column(3,
                    ui.div(
                        ui.tags.p("Portafolio", class_="card-title"),
                        ui.input_text("r_tickers", "Tickers (coma)", value=TICKERS_STR),
                        ui.input_text("r_pesos",   "Pesos (coma)",   value=PESOS_DEFAULT),
                        ui.input_slider("r_conf", "Confianza VaR", 0.90, 0.99, 0.95, step=0.01),
                        ui.input_select("r_tipo", "Análisis", choices={
                            "var":        "VaR & CVaR + Kupiec",
                            "capm":       "CAPM & Beta",
                            "markowitz":  "Markowitz QP",
                            "volatilidad":"Volatilidad EWMA + GARCH",
                        }),
                        ui.input_action_button("r_btn", "Calcular", class_="btn-primary"),
                        class_="card",
                    ),
                ),
                ui.column(9, ui.div(ui.output_ui("riesgo_out"), class_="card")),
            ),
        ),

        # ═══ TAB 3: RENTA FIJA & OPCIONES ═══════════════
        ui.nav_panel("💰 Renta Fija & Opciones",
            ui.navset_pill(
                ui.nav_panel("Curva de Rendimiento",
                    ui.row(
                        ui.column(4,
                            ui.div(
                                ui.input_action_button("rf_btn", "Cargar curva FRED", class_="btn-primary"),
                                ui.tags.br(), ui.tags.br(),
                                ui.output_ui("rf_params"),
                                class_="card",
                            ),
                        ),
                        ui.column(8, ui.div(ui.output_ui("rf_chart"), class_="card")),
                    ),
                ),
                ui.nav_panel("Bono Sintético",
                    ui.row(
                        ui.column(4,
                            ui.div(
                                ui.input_numeric("b_cupon", "Cupón anual (%)", 5.0, min=0.1, max=20, step=0.1),
                                ui.input_numeric("b_venc",  "Vencimiento (años)", 10, min=1, max=30),
                                ui.input_numeric("b_ytm",   "YTM (%)", 5.0, min=0.1, max=20, step=0.1),
                                ui.input_action_button("b_btn", "Calcular duración", class_="btn-primary"),
                                class_="card",
                            ),
                        ),
                        ui.column(8, ui.div(ui.output_ui("bono_out"), class_="card")),
                    ),
                ),
                ui.nav_panel("Black-Scholes & Greeks",
                    ui.row(
                        ui.column(4,
                            ui.div(
                                ui.input_numeric("bs_S",     "S — Subyacente",     150.0, min=1),
                                ui.input_numeric("bs_K",     "K — Strike",         150.0, min=1),
                                ui.input_numeric("bs_T",     "T — Vencimiento (años)", 0.5, min=0.01, step=0.1),
                                ui.input_numeric("bs_r",     "r — Tasa libre (%)",  5.0, min=0),
                                ui.input_numeric("bs_sigma", "σ — Volatilidad (%)", 20.0, min=1),
                                ui.input_radio_buttons("bs_tipo", "Tipo",
                                    choices={"call": "Call", "put": "Put"}),
                                ui.input_action_button("bs_btn", "Calcular", class_="btn-primary"),
                                class_="card",
                            ),
                        ),
                        ui.column(8, ui.div(ui.output_ui("bs_out"), class_="card")),
                    ),
                ),
                ui.nav_panel("Stress Testing",
                    ui.row(
                        ui.column(4,
                            ui.div(
                                ui.input_text("st_tickers", "Tickers", value=TICKERS_STR),
                                ui.input_text("st_pesos",   "Pesos",   value=PESOS_DEFAULT),
                                ui.input_numeric("st_var",   "VaR base (decimal)", 0.0185, min=0.001, step=0.001),
                                ui.input_numeric("st_sigma", "Volatilidad diaria", 0.012,  min=0.001, step=0.001),
                                ui.input_numeric("st_valor", "Valor portafolio (USD)", 100000, min=1000),
                                ui.input_action_button("st_btn", "Aplicar escenarios", class_="btn-primary"),
                                class_="card",
                            ),
                        ),
                        ui.column(8, ui.div(ui.output_ui("stress_out"), class_="card")),
                    ),
                ),
            ),
        ),

        # ═══ TAB 4: ML ══════════════════════════════════
        ui.nav_panel("🤖 ML Predicción",
            ui.row(
                ui.column(4,
                    ui.div(
                        ui.tags.p("Features del modelo", class_="card-title"),
                        ui.output_ui("ml_ticker_sel"),
                        ui.input_numeric("ml_rsi",   "RSI (14)",             45.0, min=0, max=100),
                        ui.input_numeric("ml_macd",  "MACD Histogram",        0.5, step=0.1),
                        ui.input_numeric("ml_ewma",  "Vol. EWMA diaria",     0.012, step=0.001),
                        ui.input_numeric("ml_r5",    "Ret. 5 días (%)",       1.2, step=0.1),
                        ui.input_numeric("ml_r21",   "Ret. 21 días (%)",      3.5, step=0.1),
                        ui.input_numeric("ml_pctb",  "Bollinger %B",         0.55, min=0, max=1, step=0.01),
                        ui.input_numeric("ml_estoc", "Estocástico %K",       55.0, min=0, max=100),
                        ui.input_action_button("ml_btn", "Predecir régimen", class_="btn-primary"),
                        class_="card",
                    ),
                    ui.div(ui.output_ui("ml_status"), class_="card"),
                ),
                ui.column(8,
                    ui.div(ui.output_ui("ml_out"), class_="card"),
                    ui.div(
                        ui.tags.p("Sobre el modelo", class_="card-title"),
                        ui.tags.p("Propósito: Clasificación de régimen de mercado (alcista / lateral / bajista).", style="color:#94a3b8;font-size:13px"),
                        ui.tags.p("Algoritmo: RandomForestClassifier — 200 árboles, max_depth=8, balanced classes.", style="color:#94a3b8;font-size:13px"),
                        ui.tags.p("Entrenamiento: 80% datos históricos. Partición temporal sin shuffle para evitar data leakage.", style="color:#94a3b8;font-size:13px"),
                        ui.tags.p("Singleton: el modelo se carga una sola vez al levantar el servidor (verificar en logs de uvicorn).", style="color:#94a3b8;font-size:13px"),
                        class_="card",
                    ),
                ),
            ),
        ),

        # ═══ TAB 5: MACRO & COMPARACIÓN ═════════════════
        ui.nav_panel("🌐 Macro & Comparación",
            ui.navset_pill(
                ui.nav_panel("Indicadores Macro",
                    ui.div(
                        ui.input_action_button("macro_btn", "Actualizar desde FRED", class_="btn-primary"),
                        ui.tags.br(), ui.tags.br(),
                        ui.output_ui("macro_out"),
                        class_="card",
                    ),
                ),
                ui.nav_panel("Comparar activos",
                    ui.row(
                        ui.column(4,
                            ui.div(
                                ui.input_text("comp_tickers", "Tickers (coma)", value="AAPL,SAP.DE,EC,TM"),
                                ui.input_date_range("comp_fechas", "Período",
                                    start="2022-01-01", end=str(date.today())),
                                ui.input_action_button("comp_btn", "Comparar", class_="btn-primary"),
                                class_="card",
                            ),
                        ),
                        ui.column(8, ui.div(ui.output_ui("comp_out"), class_="card")),
                    ),
                ),
            ),
        ),
    ),
)


# ─────────────────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────────────────

def server(input, output, session):

    # ── Cargar catálogo de activos al inicio ──────────────
    @reactive.calc
    def _catalogo():
        data, _ = api_get("/activos")
        if data:
            return [a["ticker"] for a in data.get("activos", [])]
        return TICKERS_DEFAULT

    # ── Status ────────────────────────────────────────────
    @output
    @render.ui
    def status_badge():
        h, err = api_get("/")
        if h:
            return ui.span(f"✓ API v{h.get('version','?')} · {h.get('mensaje','ok')}", class_="status-ok")
        return ui.span(f"✗ {err}", class_="status-err")

    # ── Selector dinámico de activos ──────────────────────
    @output
    @render.ui
    def ind_selector():
        tickers = _catalogo()
        return ui.input_select("ind_ticker", "Activo", choices=tickers, selected="AAPL")

    @output
    @render.ui
    def ml_ticker_sel():
        tickers = _catalogo()
        return ui.input_select("ml_ticker", "Activo", choices=tickers, selected="AAPL")

    # ════════════════════════════════════════════════════
    # TAB 1 — INDICADORES
    # ════════════════════════════════════════════════════

    @reactive.calc
    @reactive.event(input.ind_btn)
    def _ind():
        ticker = input.ind_ticker() if hasattr(input, 'ind_ticker') else "AAPL"
        fi = str(input.ind_fechas()[0])
        ff = str(input.ind_fechas()[1])
        return api_get(f"/indicadores/{ticker}", {"fecha_inicio": fi, "fecha_fin": ff})

    @output
    @render.ui
    def ind_signals():
        data, err = _ind()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Presiona Calcular indicadores", style="color:#64748b;font-size:12px")
        señales = data.get("señales", [])
        if not señales: return ui.span("Sin señales activas", class_="badge-neu")
        items = []
        for s in señales[:8]:
            cls = "badge-buy" if s.get("tipo") == "COMPRA" else "badge-sell"
            items.append(ui.div(
                ui.span(s.get("tipo",""), class_=cls),
                ui.tags.span(f" {s.get('indicador','')} — {s.get('descripcion','')[:45]}",
                             style="font-size:11px;color:#94a3b8;margin-left:4px"),
                style="margin-bottom:6px",
            ))
        return ui.div(*items)

    @output
    @render.ui
    def ind_metrics():
        data, err = _ind()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Selecciona activo y presiona Calcular", style="color:#64748b")
        res  = data.get("resumen", {})
        ult  = (data.get("datos") or [{}])[-1]
        rsi  = res.get("rsi_actual")
        price = ult.get("cierre")
        macd_p = res.get("macd_positivo", False)
        vs20   = res.get("precio_vs_sma20", "—")
        pct_b  = res.get("boll_pct_b")

        rsi_color = "#f87171" if (rsi or 50)>70 else ("#34d399" if (rsi or 50)<30 else "#60a5fa")

        def m(title, val, color="#60a5fa"):
            return ui.column(2, ui.div(
                ui.tags.p(title, class_="card-title"),
                ui.tags.p(str(val) if val is not None else "—",
                          class_="metric-big", style=f"color:{color}"),
            ))
        return ui.row(
            m("Precio", f"${price:.2f}" if price else "—"),
            m("RSI (14)", f"{rsi:.1f}" if rsi else "—", rsi_color),
            m("MACD", "Alcista ↑" if macd_p else "Bajista ↓", "#34d399" if macd_p else "#f87171"),
            m("vs SMA20", vs20.upper(), "#34d399" if vs20=="sobre" else "#f87171"),
            m("Bollinger %B", f"{pct_b:.2f}" if pct_b else "—",
              "#f87171" if (pct_b or 0)>0.8 else ("#34d399" if (pct_b or 0)<0.2 else "#60a5fa")),
            m("Días datos", data.get("total_dias","—"), "#94a3b8"),
        )

    def _plotly_chart(html_str):
        return ui.HTML(html_str)

    @output
    @render.ui
    def ind_chart():
        data, err = _ind()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Sin datos", style="color:#64748b")
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            df = pd.DataFrame(data.get("datos", [])).dropna(subset=["cierre"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["cierre"], name="Precio",
                                     line=dict(color="#60a5fa", width=2)))
            for col, color, name in [("sma_20","#f59e0b","SMA 20"),("sma_50","#a78bfa","SMA 50"),
                                      ("boll_superior","#475569","BB Sup"),("boll_inferior","#475569","BB Inf")]:
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df[col], name=name,
                                             line=dict(color=color, width=1,
                                                       dash="dot" if "boll" in col else "solid")))
            fig.update_layout(height=280, paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                              font=dict(color="#94a3b8", size=11), margin=dict(t=20,b=20,l=40,r=10),
                              legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"))
            fig.update_xaxes(gridcolor="#2d3748")
            fig.update_yaxes(gridcolor="#2d3748")
            return ui.HTML(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
        except Exception as e:
            return ui.tags.p(str(e), style="color:#f87171;font-size:12px")

    @output
    @render.ui
    def ind_rsi():
        data, err = _ind()
        if err or data is None: return ui.tags.p(err or "Sin datos", style="color:#64748b;font-size:12px")
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            df = pd.DataFrame(data.get("datos",[])).dropna(subset=["rsi_14"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["rsi_14"], line=dict(color="#a78bfa", width=1.5)))
            fig.add_hline(y=70, line_dash="dash", line_color="#f87171", annotation_text="70")
            fig.add_hline(y=30, line_dash="dash", line_color="#34d399", annotation_text="30")
            fig.update_layout(height=180, paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                              font=dict(color="#94a3b8",size=10), showlegend=False,
                              margin=dict(t=10,b=20,l=40,r=10), yaxis=dict(range=[0,100]))
            fig.update_xaxes(gridcolor="#2d3748"); fig.update_yaxes(gridcolor="#2d3748")
            return ui.HTML(pio.to_html(fig, include_plotlyjs=False, full_html=False))
        except Exception as e:
            return ui.tags.p(str(e), style="color:#f87171;font-size:11px")

    @output
    @render.ui
    def ind_macd():
        data, err = _ind()
        if err or data is None: return ui.tags.p(err or "Sin datos", style="color:#64748b;font-size:12px")
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            df = pd.DataFrame(data.get("datos",[])).dropna(subset=["macd"])
            colors = ["#34d399" if v>=0 else "#f87171" for v in df["macd_hist"].fillna(0)]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["fecha"], y=df["macd_hist"], name="Hist.", marker_color=colors))
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["macd"],       name="MACD",  line=dict(color="#60a5fa",width=1.5)))
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["macd_señal"], name="Señal", line=dict(color="#f59e0b",width=1.5)))
            fig.update_layout(height=180, paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                              font=dict(color="#94a3b8",size=10),
                              legend=dict(orientation="h",bgcolor="rgba(0,0,0,0)",font=dict(size=10)),
                              margin=dict(t=10,b=20,l=40,r=10))
            fig.update_xaxes(gridcolor="#2d3748"); fig.update_yaxes(gridcolor="#2d3748")
            return ui.HTML(pio.to_html(fig, include_plotlyjs=False, full_html=False))
        except Exception as e:
            return ui.tags.p(str(e), style="color:#f87171;font-size:11px")

    # ════════════════════════════════════════════════════
    # TAB 2 — RIESGO
    # ════════════════════════════════════════════════════

    @reactive.calc
    @reactive.event(input.r_btn)
    def _riesgo():
        try:
            tickers = [t.strip().upper() for t in input.r_tickers().split(",") if t.strip()]
            pesos   = [float(p.strip()) for p in input.r_pesos().split(",") if p.strip()]
        except ValueError:
            return "error", "Pesos deben ser números separados por coma"
        tipo    = input.r_tipo()
        conf    = input.r_conf()
        payload = {"tickers": tickers, "pesos": pesos, "nivel_confianza": conf}
        return tipo, tickers, pesos, payload

    @output
    @render.ui
    def riesgo_out():
        result = _riesgo()
        if result is None or result[0] == "error":
            msg = result[1] if result else "Configura el portafolio y presiona Calcular"
            return ui.tags.p(msg, style="color:#64748b" if result is None else "color:#f87171")

        tipo, tickers, pesos, payload = result

        if tipo == "var":
            data, err = api_post("/var", payload)
            if err: return ui.div(ui.tags.p(err, class_="err"))
            return _render_var(data)
        elif tipo == "capm":
            data, err = api_get("/capm", {"tickers": tickers, "tasa_libre_riesgo": 0.0525})
            if err: return ui.div(ui.tags.p(err, class_="err"))
            return _render_capm(data)
        elif tipo == "markowitz":
            data, err = api_post("/frontera-eficiente", payload)
            if err: return ui.div(ui.tags.p(err, class_="err"))
            return _render_markowitz(data)
        elif tipo == "volatilidad":
            data, err = api_get(f"/volatilidad/{tickers[0]}", {"lambda_ewma": 0.94})
            if err: return ui.div(ui.tags.p(err, class_="err"))
            return _render_garch(data)
        return ui.tags.p("Selecciona un análisis", style="color:#64748b")

    def _render_var(data):
        if not data: return ui.tags.p("Sin datos", style="color:#64748b")
        vp  = data.get("var_parametrico", {})
        vh  = data.get("var_historico",   {})
        vmc = data.get("var_montecarlo",  {})
        rk  = data.get("resumen_kupiec",  {})

        def fila(label, d):
            kup  = d.get("kupiec", {})
            adec = kup.get("modelo_adecuado")
            badge = "Pasa" if adec else ("Falla" if adec is not None else "—")
            lr   = kup.get("LR_POF", "—")
            lr_s = f"{lr:.4f}" if isinstance(lr, float) else str(lr)
            color = "#34d399" if adec else "#f87171"
            return f"""<tr>
                <td style="color:#e2e8f0">{label}</td>
                <td style="color:#60a5fa">{d.get("var_porcentaje") or "—"}</td>
                <td style="color:#e2e8f0">{fmt_usd(d.get("var_monetario_usd"))}</td>
                <td style="color:#a78bfa">{d.get("cvar_porcentaje") or "—"}</td>
                <td style="color:#e2e8f0">{fmt_usd(d.get("cvar_monetario_usd"))}</td>
                <td style="color:{color};font-weight:600">{badge}</td>
                <td style="color:#94a3b8">{lr_s}</td>
            </tr>"""

        tabla_html = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#1e293b">
                {"".join(f'<th style="color:#94a3b8;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
                         for h in ["Metodo","VaR %","VaR USD","CVaR %","CVaR USD","Kupiec","LR_POF"])}
            </tr></thead>
            <tbody>
                {fila("Parametrico", vp)}
                {fila("Historico", vh)}
                {fila("Montecarlo", vmc)}
            </tbody>
        </table>"""

        return ui.div(
            ui.tags.p("VaR y CVaR — 3 metodos", class_="card-title"),
            ui.HTML(tabla_html),
            ui.tags.hr(),
            ui.tags.p("Backtesting de Kupiec (LR_POF — umbral 3.841)", class_="card-title"),
            ui.tags.p(str(rk.get("recomendacion","—")), style="color:#60a5fa;font-size:13px"),
            ui.tags.p(str(data.get("interpretacion_general","—")), style="color:#94a3b8;font-size:12px;margin-top:8px"),
        )


    def _render_capm(data):
        if not data or "activos" not in data:
            return ui.tags.p("Sin datos CAPM", style="color:#64748b")
        activos = data.get("activos", {})
        if not activos:
            return ui.tags.p("Sin activos calculados (datos insuficientes)", style="color:#f87171")

        filas = ""
        for t, d in activos.items():
            beta  = d.get("beta") or 0
            er    = fmt_pct(d.get("rendimiento_esperado_capm"))
            alpha = fmt_pct(d.get("alpha_anual"))
            r2    = f"{float(d.get('r_cuadrado') or 0):.4f}"
            tipo  = str(d.get("interpretacion_beta",""))[:45]
            bc    = "#f87171" if beta>1.2 else ("#34d399" if beta<0.8 else "#f59e0b")
            filas += f"""<tr>
                <td style="color:#e2e8f0;padding:9px 12px">{t}</td>
                <td style="color:{bc};font-weight:600;padding:9px 12px">{beta:.4f}</td>
                <td style="color:#60a5fa;padding:9px 12px">{er}</td>
                <td style="color:#a78bfa;padding:9px 12px">{alpha}</td>
                <td style="color:#94a3b8;padding:9px 12px">{r2}</td>
                <td style="color:#94a3b8;padding:9px 12px;font-size:11px">{tipo}</td>
            </tr>"""

        tabla_html = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#1e293b">
                {"".join(f'<th style="color:#94a3b8;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
                         for h in ["Ticker","Beta","E(R) CAPM","Alpha Jensen","R2","Tipo"])}
            </tr></thead>
            <tbody>{filas}</tbody>
        </table>"""

        rf    = fmt_pct(data.get("tasa_libre_riesgo_anual"))
        prima = fmt_pct(data.get("prima_riesgo_mercado") or data.get("prima_riesgo_mercado_pct"))
        return ui.div(
            ui.tags.p(f"Benchmark: {data.get('benchmark','SPY')} | Rf: {rf} | Prima mercado: {prima}",
                      style="color:#94a3b8;font-size:12px;margin-bottom:10px"),
            ui.HTML(tabla_html),
        )


    def _render_markowitz(data):
        if not data: return ui.tags.p("Sin datos", style="color:#64748b")
        ms = data.get("portafolio_max_sharpe", {})
        mv = data.get("portafolio_min_varianza", {})

        def tabla_pesos(port):
            pesos = port.get("pesos", {})
            if not pesos:
                return "<tr><td style='color:#94a3b8;padding:8px'>Sin datos</td></tr>"
            return "".join(
                f"<tr><td style='color:#e2e8f0;padding:7px 12px'>{t}</td>"
                f"<td style='color:#60a5fa;padding:7px 12px;font-weight:600'>{float(v)*100:.1f}%</td></tr>"
                for t, v in pesos.items()
            )

        def color_ret(port):
            return "#34d399" if (port.get("retorno_anual") or 0) > 0 else "#f87171"

        html = f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div>
                <p style="color:#94a3b8;font-size:11px;font-weight:600;text-transform:uppercase;margin-bottom:6px">Maximo Sharpe</p>
                <p style="color:{color_ret(ms)};font-size:12px;margin-bottom:8px">
                    Retorno: {fmt_pct(ms.get("retorno_anual"))} | Vol: {fmt_pct(ms.get("volatilidad_anual"))} | Sharpe: {ms.get("sharpe_ratio") or 0:.3f}
                </p>
                <table style="width:100%;border-collapse:collapse;font-size:13px">
                    <thead><tr style="background:#1e293b">
                        <th style="color:#94a3b8;padding:7px 12px;text-align:left;font-size:11px">Ticker</th>
                        <th style="color:#94a3b8;padding:7px 12px;text-align:left;font-size:11px">Peso</th>
                    </tr></thead>
                    <tbody>{tabla_pesos(ms)}</tbody>
                </table>
            </div>
            <div>
                <p style="color:#94a3b8;font-size:11px;font-weight:600;text-transform:uppercase;margin-bottom:6px">Minima Varianza</p>
                <p style="color:{color_ret(mv)};font-size:12px;margin-bottom:8px">
                    Retorno: {fmt_pct(mv.get("retorno_anual"))} | Vol: {fmt_pct(mv.get("volatilidad_anual"))} | Sharpe: {mv.get("sharpe_ratio") or 0:.3f}
                </p>
                <table style="width:100%;border-collapse:collapse;font-size:13px">
                    <thead><tr style="background:#1e293b">
                        <th style="color:#94a3b8;padding:7px 12px;text-align:left;font-size:11px">Ticker</th>
                        <th style="color:#94a3b8;padding:7px 12px;text-align:left;font-size:11px">Peso</th>
                    </tr></thead>
                    <tbody>{tabla_pesos(mv)}</tbody>
                </table>
            </div>
        </div>"""

        return ui.div(
            ui.tags.p("Frontera eficiente — Programacion Cuadratica (sin ventas en corto)", class_="card-title"),
            ui.HTML(html),
        )


    def _render_garch(data):
        if not data: return ui.tags.p("Sin datos GARCH", style="color:#64748b")
        garch   = data.get("garch", {})
        modelos = garch.get("modelos", [])
        mejor   = garch.get("mejor_por_aic","—")
        ewma_d  = data.get("ewma",{}).get("modelos_ewma",{})
        ewma94  = ewma_d.get("ewma_lambda_0.94",{})

        filas = ""
        for m in modelos:
            star = "* " if m.get("modelo") == mejor else ""
            if "error" in m:
                filas += f"<tr><td style='color:#e2e8f0;padding:9px 12px'>{star}{m['modelo']}</td><td colspan='5' style='color:#f87171;padding:9px 12px'>Error: {str(m['error'])[:60]}</td></tr>"
            else:
                filas += f"""<tr>
                    <td style="color:#e2e8f0;padding:9px 12px;font-weight:{'600' if star else '400'}">{star}{m['modelo']}</td>
                    <td style="color:#60a5fa;padding:9px 12px">{float(m.get('aic') or 0):.2f}</td>
                    <td style="color:#94a3b8;padding:9px 12px">{float(m.get('bic') or 0):.2f}</td>
                    <td style="color:#a78bfa;padding:9px 12px">{float(m.get('alpha') or 0):.4f}</td>
                    <td style="color:#94a3b8;padding:9px 12px">{float(m.get('beta') or 0):.4f}</td>
                    <td style="color:#34d399;padding:9px 12px">{fmt_pct(m.get('vol_pronostico_anual'))}</td>
                </tr>"""

        tabla_html = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#1e293b">
                {"".join(f'<th style="color:#94a3b8;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
                         for h in ["Modelo","AIC","BIC","Alpha","Beta","Vol. Pronostico"])}
            </tr></thead>
            <tbody>{filas}</tbody>
        </table>"""

        vol_anual = fmt_pct(ewma94.get("vol_ultimo_anual"))
        return ui.div(
            ui.tags.p(f"EWMA lambda=0.94 — Volatilidad anualizada: {vol_anual}",
                      style="color:#34d399;font-weight:600;margin-bottom:12px"),
            ui.tags.p("ARCH/GARCH — Tabla comparativa AIC/BIC", class_="card-title"),
            ui.HTML(tabla_html),
            ui.tags.p(str(garch.get("interpretacion","—")),
                      style="color:#60a5fa;font-size:12px;margin-top:10px"),
        )


    # TAB 3 — RENTA FIJA & OPCIONES
    # ════════════════════════════════════════════════════

    @reactive.calc
    @reactive.event(input.rf_btn)
    def _rf(): return api_get("/curva-rendimiento")

    @output
    @render.ui
    def rf_params():
        data, err = _rf()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Presiona el botón", style="color:#64748b;font-size:12px")
        params = data.get("parametros", {})
        interp = data.get("interpretacion_parametros", {})
        return ui.div(
            ui.tags.p(f"RMSE: {data.get('rmse_ajuste_pct','—'):.4f}%", style="color:#34d399;font-weight:600"),
            ui.tags.p(f"Forma: {data.get('forma_curva','—')}", style="color:#f59e0b;font-size:12px"),
            ui.tags.hr(),
            *[ui.tags.p(v, style="color:#94a3b8;font-size:11px;margin-bottom:4px") for v in interp.values()],
        )

    @output
    @render.ui
    def rf_chart():
        data, err = _rf()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Presiona Cargar curva FRED", style="color:#64748b")
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            obs = pd.DataFrame(data.get("curva_observada", []))
            fit = pd.DataFrame(data.get("curva_ajustada",  []))
            fig = go.Figure()
            if not obs.empty:
                fig.add_trace(go.Scatter(x=obs["vencimiento"], y=obs["rendimiento_obs"],
                    mode="markers+lines", name="Observada (FRED)",
                    marker=dict(color="#60a5fa",size=8), line=dict(color="#60a5fa",dash="dot")))
            if not fit.empty:
                fig.add_trace(go.Scatter(x=fit["vencimiento"], y=fit["rendimiento_fit"],
                    mode="lines", name="Nelson-Siegel", line=dict(color="#34d399",width=2.5)))
            fig.update_layout(height=320, paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                              font=dict(color="#94a3b8",size=11), xaxis_title="Vencimiento (años)",
                              yaxis_title="Rendimiento (%)",
                              legend=dict(orientation="h",bgcolor="rgba(0,0,0,0)"),
                              margin=dict(t=20,b=40,l=50,r=10))
            fig.update_xaxes(gridcolor="#2d3748"); fig.update_yaxes(gridcolor="#2d3748")
            return ui.HTML(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
        except Exception as e:
            return ui.tags.p(str(e), style="color:#f87171;font-size:12px")

    @reactive.calc
    @reactive.event(input.b_btn)
    def _bono():
        return api_post("/bono/duracion", {
            "cupon_anual":       input.b_cupon() / 100,
            "vencimiento_anios": int(input.b_venc()),
            "valor_nominal":     1000.0,
            "ytm":               input.b_ytm() / 100,
            "pagos_por_anio":    2,
        })

    @output
    @render.ui
    def bono_out():
        data, err = _bono()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Configura y presiona Calcular duración", style="color:#64748b")
        shocks = data.get("sensibilidad_shocks", [])
        filas_bono = ""
        for s in shocks:
            shock_pb = int(s.get("shock_pb", 0))
            color_s = "#34d399" if shock_pb < 0 else "#f87171"
            dp_lin = s.get("dp_lineal", 0) or 0
            dp_dc  = s.get("dp_duracion_convexidad", 0) or 0
            dp_ex  = s.get("dp_reprice_exacto", 0) or 0
            p_new  = s.get("precio_nuevo", 0) or 0
            filas_bono += f"""<tr>
                <td style="color:{color_s};font-weight:600;padding:9px 12px">{shock_pb:+d} pb</td>
                <td style="color:#94a3b8;padding:9px 12px">{float(dp_lin)*100:.3f}%</td>
                <td style="color:#60a5fa;padding:9px 12px">{float(dp_dc)*100:.3f}%</td>
                <td style="color:#34d399;padding:9px 12px">{float(dp_ex)*100:.3f}%</td>
                <td style="color:#e2e8f0;padding:9px 12px">${float(p_new):.2f}</td>
            </tr>"""
        tabla_bono = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#1e293b">
                {"".join(f'<th style="color:#94a3b8;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
                         for h in ["Shock","dP Lineal","dP D+C","dP Exacto","Precio nuevo"])}
            </tr></thead>
            <tbody>{filas_bono}</tbody>
        </table>"""
        return ui.div(
            ui.row(
                ui.column(3, ui.div(ui.tags.p("Precio", class_="card-title"), ui.tags.p(f"${data.get('precio',0):.4f}", class_="metric-big"))),
                ui.column(3, ui.div(ui.tags.p("D. Macaulay (anos)", class_="card-title"), ui.tags.p(f"{data.get('duracion_macaulay_anios',0):.4f}", class_="metric-big metric-pos"))),
                ui.column(3, ui.div(ui.tags.p("D. Modificada", class_="card-title"), ui.tags.p(f"{data.get('duracion_modificada',0):.4f}", class_="metric-big metric-neu"))),
                ui.column(3, ui.div(ui.tags.p("Convexidad", class_="card-title"), ui.tags.p(f"{data.get('convexidad',0):.2f}", class_="metric-big", style="color:#a78bfa"))),
            ),
            ui.tags.hr(),
            ui.tags.p("Sensibilidad ante shocks — 3 aproximaciones", class_="card-title"),
            ui.HTML(tabla_bono),
        )

    @reactive.calc
    @reactive.event(input.bs_btn)
    def _bs():
        return api_post("/opcion/precio", {
            "S": float(input.bs_S()), "K": float(input.bs_K()),
            "T": float(input.bs_T()), "r": input.bs_r()/100,
            "sigma": input.bs_sigma()/100, "tipo": input.bs_tipo(),
        })

    @output
    @render.ui
    def bs_out():
        data, err = _bs()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Ingresa parámetros y presiona Calcular", style="color:#64748b")
        g = data.get("greeks", {})
        interp = data.get("interpretacion_greeks", {})
        paridad = data.get("paridad_put_call", {})
        return ui.div(
            ui.row(
                ui.column(3, ui.div(ui.tags.p("Precio opción", class_="card-title"), ui.tags.p(f"${data.get('precio',0):.4f}", class_="metric-big metric-pos"))),
                ui.column(3, ui.div(ui.tags.p("d₁", class_="card-title"), ui.tags.p(f"{data.get('d1',0):.4f}", class_="metric-big"))),
                ui.column(3, ui.div(ui.tags.p("d₂", class_="card-title"), ui.tags.p(f"{data.get('d2',0):.4f}", class_="metric-big"))),
                ui.column(3, ui.div(
                    ui.tags.p("Paridad put-call", class_="card-title"),
                    ui.tags.p("✅ Verificada" if paridad.get("verificada") else "❌ Error",
                              style=f"color:{'#34d399' if paridad.get('verificada') else '#f87171'};font-weight:600"),
                    ui.tags.p(f"Error numérico: {paridad.get('error_numerico','—'):.2e}", style="color:#64748b;font-size:11px"),
                )),
            ),
            ui.tags.hr(),
            ui.tags.p("Las 5 Greeks", class_="card-title"),
            ui.tags.table(
                ui.tags.thead(ui.tags.tr(*[ui.tags.th(h) for h in ["Greek","Símbolo","Valor","Interpretación"]])),
                ui.tags.tbody(*[
                    ui.tags.tr(
                        ui.tags.td(n.title()), ui.tags.td(sym),
                        ui.tags.td(ui.HTML(f'<span style="color:#60a5fa;font-weight:600">{g.get(n,"—"):.6f}</span>')),
                        ui.tags.td(interp.get(n,"—"), style="font-size:11px;color:#94a3b8"),
                    ) for n, sym in [("delta","Δ"),("gamma","Γ"),("vega","ν"),("theta","Θ"),("rho","ρ")]
                ]),
            ),
        )

    @reactive.calc
    @reactive.event(input.st_btn)
    def _stress():
        try:
            tickers = [t.strip().upper() for t in input.st_tickers().split(",") if t.strip()]
            pesos   = [float(p.strip()) for p in input.st_pesos().split(",") if p.strip()]
        except ValueError:
            return None, "Pesos inválidos"
        return api_post("/stress", {
            "tickers": tickers, "pesos": pesos, "betas": {},
            "var_base": float(input.st_var()), "sigma_base": float(input.st_sigma()),
            "valor_portafolio": float(input.st_valor()),
        })

    @output
    @render.ui
    def stress_out():
        data, err = _stress()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Configura y presiona Aplicar escenarios", style="color:#64748b")
        escenarios = data.get("escenarios", {})
        labels = {
            "tasa_menos_200pb":"Tasa −200pb","tasa_mas_200pb":"Tasa +200pb",
            "caida_mercado_20pct":"Caída −20%","caida_mercado_30pct":"Caída −30%",
            "volatilidad_doble":"Volatilidad ×2","combinado_tormenta_perfecta":"⚡ Tormenta perfecta",
        }
        filas_stress = ""
        for key, esc in escenarios.items():
            perdida = float(esc.get("perdida_total_pct") or esc.get("perdida_portafolio_pct") or 0)
            usd     = float(esc.get("perdida_total_usd") or esc.get("perdida_portafolio_usd") or 0)
            color_p = "#34d399" if perdida > 0 else "#f87171"
            filas_stress += f"""<tr>
                <td style="color:#e2e8f0;padding:9px 12px;font-weight:500">{labels.get(key, key)}</td>
                <td style="color:#94a3b8;padding:9px 12px;font-size:11px">{str(esc.get("shock_descripcion","—"))[:60]}</td>
                <td style="color:{color_p};font-weight:600;padding:9px 12px">{perdida:.2f}%</td>
                <td style="color:#e2e8f0;padding:9px 12px">{fmt_usd(usd)}</td>
            </tr>"""
        tabla_stress = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#1e293b">
                {"".join(f'<th style="color:#94a3b8;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
                         for h in ["Escenario","Shock","Perdida %","Perdida USD"])}
            </tr></thead>
            <tbody>{filas_stress}</tbody>
        </table>"""
        return ui.div(
            ui.tags.p(f"VaR base: {data.get('var_base_pct',0):.3f}% | Portafolio: {fmt_usd(data.get('valor_portafolio_usd'))}", style="color:#94a3b8;font-size:12px;margin-bottom:10px"),
            ui.HTML(tabla_stress),
            ui.tags.p(str(data.get("interpretacion","")), style="color:#94a3b8;font-size:12px;margin-top:10px"),
        )

    # ════════════════════════════════════════════════════
    # TAB 4 — ML
    # ════════════════════════════════════════════════════

    @output
    @render.ui
    def ml_status():
        data, err = api_get("/predict/status")
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if not data: return ui.tags.p("Sin estado", style="color:#64748b;font-size:12px")
        disp = data.get("disponible", False)
        return ui.div(
            ui.tags.p("✅ Modelo cargado" if disp else "⚠️ Modelo no disponible",
                      style=f"color:{'#34d399' if disp else '#f59e0b'};font-weight:600"),
            ui.tags.p(f"Versión: {data.get('model_version','—')}", style="color:#94a3b8;font-size:11px"),
            ui.tags.p(f"Singleton ID: {data.get('singleton_id','—')}", style="color:#64748b;font-size:10px"),
            ui.tags.p("Llamar /predict 3 veces → 'modelo cargado' aparece solo 1 vez en logs de uvicorn",
                      style="color:#64748b;font-size:10px;margin-top:4px"),
        )

    @reactive.calc
    @reactive.event(input.ml_btn)
    def _ml():
        ticker = input.ml_ticker() if hasattr(input, 'ml_ticker') else "AAPL"
        features = [
            float(input.ml_rsi()),
            float(input.ml_macd()),
            float(input.ml_ewma()),
            float(input.ml_r5())  / 100,
            float(input.ml_r21()) / 100,
            float(input.ml_pctb()),
            float(input.ml_estoc()),
        ]
        return api_post("/predict", {"ticker": ticker, "features": features}), features

    @output
    @render.ui
    def ml_out():
        result = _ml()
        if result is None:
            return ui.tags.p("Ingresa features y presiona Predecir régimen", style="color:#64748b")
        (data, err), features = result
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Sin respuesta del modelo", style="color:#64748b")

        pred  = data.get("prediction", 0)
        label = data.get("prediction_label", "—")
        colors = {1.0:"#34d399", 0.0:"#f59e0b", -1.0:"#f87171"}
        color  = colors.get(float(pred), "#60a5fa")
        fnames = data.get("feature_names") or ["rsi_14","macd_hist","ewma_vol","ret_5d","ret_21d","pct_b_bollinger","estocastico_k"]
        if not fnames:
            fnames = ["rsi_14","macd_hist","ewma_vol","ret_5d","ret_21d","pct_b_bollinger","estocastico_k"]

        filas_ml = "".join(
            f'<tr><td style="color:#94a3b8;padding:8px 12px">{fn}</td><td style="color:#60a5fa;padding:8px 12px;font-family:monospace">{fv:.6f}</td></tr>'
            for fn, fv in zip(fnames, features)
        )
        tabla_ml = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#1e293b">
                <th style="color:#94a3b8;padding:8px 12px;text-align:left;font-size:11px">FEATURE</th>
                <th style="color:#94a3b8;padding:8px 12px;text-align:left;font-size:11px">VALOR</th>
            </tr></thead>
            <tbody>{filas_ml}</tbody>
        </table>"""
        # Limpiar emojis del label para evitar encoding issues en Windows
        label_clean = label.replace("📈","(alcista)").replace("➡️","(lateral)").replace("📉","(bajista)")
        return ui.div(
            ui.tags.p(label_clean, style=f"color:{color};font-size:36px;font-weight:700;text-align:center;margin:16px 0"),
            ui.tags.p(f"Activo: {data.get('ticker','—')} | Modelo: {data.get('model_version','—')}",
                      style="color:#94a3b8;font-size:12px;text-align:center"),
            ui.tags.hr(),
            ui.tags.p("Features enviadas al modelo", class_="card-title"),
            ui.HTML(tabla_ml),
        )

    # ════════════════════════════════════════════════════
    # TAB 5 — MACRO & COMPARACIÓN
    # ════════════════════════════════════════════════════

    @reactive.calc
    @reactive.event(input.macro_btn)
    def _macro(): return api_get("/macro")

    @output
    @render.ui
    def macro_out():
        data, err = _macro()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Presiona Actualizar desde FRED", style="color:#64748b")
        datos = data.get("datos", {})
        iconos = {"DGS3MO":"[Rf]","DGS10":"[T10]","CPIAUCSL":"[CPI]","UNRATE":"[U]","FEDFUNDS":"[Fed]","VIXCLS":"[VIX]"}
        cards = []
        for serie, info in datos.items():
            if serie.startswith("_"): continue
            valor = info.get("valor")
            cards.append(ui.column(4, ui.div(
                ui.tags.p(f"{iconos.get(serie,'📌')} {info.get('nombre',serie)}", class_="card-title"),
                ui.tags.p(f"{valor:.2f}%" if valor else "—", class_="metric-big"),
                ui.tags.p(info.get("fecha","—"), style="color:#64748b;font-size:11px"),
                ui.tags.p(info.get("interpretacion",""), style="color:#94a3b8;font-size:11px;margin-top:4px"),
                class_="card",
            )))
        ctx = data.get("contexto_macro", {})
        return ui.div(
            ui.row(*cards),
            ui.div(
                ui.tags.p("Contexto macro integrado", class_="card-title"),
                ui.tags.p(ctx.get("descripcion",""), style="color:#94a3b8;font-size:13px"),
                *[ui.tags.p(f"→ {i}", style="color:#60a5fa;font-size:12px") for i in ctx.get("impacto_portafolio",[])],
                class_="card",
            ),
        )

    @reactive.calc
    @reactive.event(input.comp_btn)
    def _comp():
        tickers = [t.strip().upper() for t in input.comp_tickers().split(",") if t.strip()]
        fi = str(input.comp_fechas()[0])
        ff = str(input.comp_fechas()[1])
        return api_get("/comparar", {"tickers": tickers, "fecha_inicio": fi, "fecha_fin": ff})

    @output
    @render.ui
    def comp_out():
        data, err = _comp()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Ingresa tickers y presiona Comparar", style="color:#64748b")
        comp = data.get("comparacion", {})
        filas_comp = ""
        for ticker, d in sorted(comp.items(), key=lambda x: x[1].get("sharpe_ratio") or 0, reverse=True):
            ret    = float(d.get("retorno_total") or 0)
            sharpe = float(d.get("sharpe_ratio") or 0)
            rc = "#34d399" if ret > 0 else "#f87171"
            sc = "#34d399" if sharpe > 0 else "#f87171"
            dd = fmt_pct(d.get("max_drawdown"))
            vol = fmt_pct(d.get("volatilidad_anual"))
            tendencia = str(d.get("tendencia_ema") or "—")
            nombre = str(d.get("nombre") or "")[:22]
            pais = str(d.get("pais") or "—")
            ranking = int(d.get("ranking_sharpe") or 0)
            filas_comp += f"""<tr>
                <td style="color:#e2e8f0;padding:9px 12px;font-weight:500">#{ranking} {ticker}</td>
                <td style="color:#94a3b8;padding:9px 12px">{nombre}</td>
                <td style="color:#94a3b8;padding:9px 12px">{pais}</td>
                <td style="color:{rc};font-weight:600;padding:9px 12px">{ret*100:.1f}%</td>
                <td style="color:#94a3b8;padding:9px 12px">{vol}</td>
                <td style="color:{sc};font-weight:600;padding:9px 12px">{sharpe:.3f}</td>
                <td style="color:#f87171;padding:9px 12px">{dd}</td>
                <td style="color:#60a5fa;padding:9px 12px;font-size:11px">{tendencia}</td>
            </tr>"""
        tabla_comp = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#1e293b">
                {"".join(f'<th style="color:#94a3b8;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
                         for h in ["Ranking","Nombre","Pais","Retorno","Volatilidad","Sharpe","Max DD","Tendencia"])}
            </tr></thead>
            <tbody>{filas_comp}</tbody>
        </table>"""
        return ui.div(
            ui.tags.p(f"Mejor Sharpe: {data.get('mejor_sharpe','—')} | Mayor retorno: {data.get('mejor_retorno','—')} | Menor vol: {data.get('menor_volatilidad','—')}",
                      style="color:#60a5fa;font-size:12px;font-weight:600;margin-bottom:10px"),
            ui.HTML(tabla_comp),
        )


app = App(app_ui, server)
