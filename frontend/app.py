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
        color = "#2A9D8F" if (f > 0) == positive_good else "#f87171"
        return f'<span style="color:{color};font-weight:600">{f:.4f}</span>'
    except:
        return str(v)


# ─────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────

app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.style("""
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500&family=Inter+Tight:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        :root{
            --navy:#0F1B33;--navy-mid:#1B2A4A;
            --teal:#2A9D8F;--teal-bright:#34BFB0;
            --coral:#E76F51;--coral-soft:#F2856A;
            --gold:#F4A261;
            --cream:#F7F3EB;--paper:#FBFAF6;
            --ink:#0E1326;--ink-soft:#3A4256;
            --rule:#D3CCBC;--rule-soft:#E8E2D2;
        }
        *{box-sizing:border-box}
        body{font-family:'Inter Tight',sans-serif;background:var(--paper);color:var(--ink);margin:0;line-height:1.55}
        .card{background:#fff;border:1px solid var(--rule-soft);border-radius:8px;padding:18px;margin-bottom:14px;box-shadow:0 1px 4px rgba(14,19,38,.06)}
        .card-title{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.14em;color:var(--coral);margin-bottom:10px;display:flex;align-items:center;gap:8px}
        .card-title::before{content:'';display:inline-block;width:18px;height:1px;background:var(--coral)}
        .metric-big{font-family:'Fraunces',serif;font-size:26px;font-weight:300;color:var(--navy)}
        .metric-pos{color:var(--teal)!important}
        .metric-neg{color:var(--coral)!important}
        .metric-neu{color:var(--gold)!important}
        .header-bar{background:var(--navy);padding:16px 28px;margin-bottom:0;display:flex;align-items:center;justify-content:space-between}
        .header-title{font-family:'Fraunces',serif;font-size:20px;font-weight:400;color:var(--paper);margin:0;letter-spacing:-.02em}
        .header-sub{font-family:'JetBrains Mono',monospace;font-size:10px;color:rgba(251,250,246,.5);margin:3px 0 0;letter-spacing:.1em;text-transform:uppercase}
        .status-ok{background:rgba(42,157,143,.12);color:var(--teal);padding:4px 12px;border-radius:100px;font-family:'JetBrains Mono',monospace;font-size:11px;border:1px solid rgba(42,157,143,.3)}
        .status-err{background:rgba(231,111,81,.1);color:var(--coral);padding:4px 12px;border-radius:100px;font-family:'JetBrains Mono',monospace;font-size:11px;border:1px solid rgba(231,111,81,.25)}
        .badge-buy{display:inline-block;background:rgba(42,157,143,.1);color:var(--teal);padding:3px 10px;border-radius:100px;font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:500;margin:2px;border:1px solid rgba(42,157,143,.2)}
        .badge-sell{display:inline-block;background:rgba(231,111,81,.1);color:var(--coral);padding:3px 10px;border-radius:100px;font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:500;margin:2px;border:1px solid rgba(231,111,81,.2)}
        .badge-neu{display:inline-block;background:rgba(244,162,97,.1);color:#B8772F;padding:3px 10px;border-radius:100px;font-family:'JetBrains Mono',monospace;font-size:11px;margin:2px;border:1px solid rgba(244,162,97,.2)}
        table{width:100%;border-collapse:collapse;font-size:13px}
        th{background:var(--cream);color:var(--ink-soft);text-align:left;padding:9px 12px;font-family:'JetBrains Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.1em;border-bottom:1px solid var(--rule)}
        td{padding:9px 12px;border-bottom:1px solid var(--rule-soft);font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--ink)}
        tr:hover td{background:var(--cream)}
        hr{border:none;border-top:1px solid var(--rule-soft);margin:16px 0}
        .shiny-input-container label{color:var(--ink-soft)!important;font-family:'JetBrains Mono',monospace!important;font-size:10px!important;font-weight:500!important;text-transform:uppercase!important;letter-spacing:.1em!important}
        .form-control,.selectize-input{background:#fff!important;border:1px solid var(--rule)!important;color:var(--ink)!important;border-radius:6px!important;font-family:'Inter Tight',sans-serif!important}
        .form-control:focus,.selectize-input.focus{border-color:var(--teal)!important;box-shadow:0 0 0 3px rgba(42,157,143,.1)!important}
        .btn-primary{background:var(--navy)!important;border:none!important;border-radius:100px!important;font-family:'Inter Tight',sans-serif!important;font-weight:600!important;color:var(--paper)!important;width:100%!important;padding:10px!important;letter-spacing:.02em!important;transition:all .2s!important;cursor:pointer!important}
        .btn-primary:hover{background:#1B2A4A!important;transform:translateY(-1px)!important}
        .err{color:var(--coral);font-size:13px;padding:10px 14px;background:rgba(231,111,81,.08);border-radius:6px;border-left:3px solid var(--coral)}
        /* Portafolio interactivo */
        .ticker-chip{display:inline-flex;align-items:center;gap:5px;background:rgba(42,157,143,.08);color:var(--teal);border:1px solid rgba(42,157,143,.2);border-radius:100px;padding:4px 12px;font-family:'JetBrains Mono',monospace;font-size:11px;margin:2px 2px 4px;cursor:default}
        .ticker-chip .rm{color:rgba(42,157,143,.5);cursor:pointer;font-size:13px;line-height:1;margin-left:2px;transition:color .15s}
        .ticker-chip .rm:hover{color:var(--coral)}
        .peso-row{display:flex;align-items:center;gap:10px;padding:5px 0;border-bottom:1px solid var(--rule-soft)}
        .peso-row:last-child{border-bottom:none}
        .peso-label{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--teal);width:72px;flex-shrink:0;font-weight:500}
        .peso-num{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--ink-soft);width:38px;text-align:right;flex-shrink:0}
        .sum-badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:11px;padding:4px 12px;border-radius:100px;text-align:center;margin-top:8px;width:100%}
        .sum-ok{background:rgba(42,157,143,.1);color:var(--teal);border:1px solid rgba(42,157,143,.25)}
        .sum-warn{background:rgba(231,111,81,.1);color:var(--coral);border:1px solid rgba(231,111,81,.25)}
        /* Nav tabs */
        .nav-tabs{border-bottom:1px solid var(--rule)!important}
        .nav-tabs .nav-link{color:var(--ink-soft)!important;border:none!important;padding:10px 16px!important;font-family:'Inter Tight',sans-serif!important;font-size:13px!important;font-weight:500!important;border-radius:0!important;transition:all .2s!important}
        .nav-tabs .nav-link.active{color:var(--ink)!important;border-bottom:2px solid var(--coral)!important;font-weight:600!important}
        .nav-tabs .nav-link:hover:not(.active){color:var(--ink)!important;background:var(--cream)!important}
        .nav-pills .nav-link{color:var(--ink-soft)!important;font-family:'Inter Tight',sans-serif!important;font-size:12px!important;border-radius:100px!important;padding:5px 14px!important}
        .nav-pills .nav-link.active{background:var(--navy)!important;color:var(--paper)!important}
    """)),

    # Header
    ui.div(
        ui.div(
            ui.tags.h1("Risk Analytics", class_="header-title"),
            ui.tags.p("PROYECTO INTEGRADOR — TEORIA DEL RIESGO · USTA 2026-I", class_="header-sub"),
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
                    ui.div(
                        ui.tags.p("Distribucion de rendimientos logaritmicos", class_="card-title"),
                        ui.output_ui("ind_hist_rend"),
                        class_="card",
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
                        ui.output_ui("port_chips_ui"),
                        ui.output_ui("port_pesos_ui"),
                        ui.input_slider("r_conf", "Confianza VaR", 0.90, 0.99, 0.95, step=0.01),
                        ui.input_select("r_tipo", "Analisis", choices={
                            "var":          "VaR & CVaR + Kupiec",
                            "capm":         "CAPM & Beta",
                            "markowitz":    "Markowitz QP",
                            "volatilidad":  "Volatilidad EWMA + GARCH",
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
                                ui.output_ui("stress_port_info"),
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
                        ui.tags.p("Propósito: Clasificación de régimen de mercado (alcista / lateral / bajista).", style="color:#3A4256;font-size:13px"),
                        ui.tags.p("Algoritmo: RandomForestClassifier — 200 árboles, max_depth=8, balanced classes.", style="color:#3A4256;font-size:13px"),
                        ui.tags.p("Entrenamiento: 80% datos históricos. Partición temporal sin shuffle para evitar data leakage.", style="color:#3A4256;font-size:13px"),
                        ui.tags.p("Singleton: el modelo se carga una sola vez al levantar el servidor (verificar en logs de uvicorn).", style="color:#3A4256;font-size:13px"),
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
        if data is None: return ui.tags.p("Presiona Calcular indicadores", style="color:#6B7280;font-size:12px")
        señales = data.get("señales", [])
        if not señales: return ui.span("Sin señales activas", class_="badge-neu")
        items = []
        for s in señales[:8]:
            cls = "badge-buy" if s.get("tipo") == "COMPRA" else "badge-sell"
            items.append(ui.div(
                ui.span(s.get("tipo",""), class_=cls),
                ui.tags.span(f" {s.get('indicador','')} — {s.get('descripcion','')[:45]}",
                             style="font-size:11px;color:#3A4256;margin-left:4px"),
                style="margin-bottom:6px",
            ))
        return ui.div(*items)

    @output
    @render.ui
    def ind_metrics():
        data, err = _ind()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Selecciona activo y presiona Calcular", style="color:#6B7280")
        res  = data.get("resumen", {})
        ult  = (data.get("datos") or [{}])[-1]
        rsi  = res.get("rsi_actual")
        price = ult.get("cierre")
        macd_p = res.get("macd_positivo", False)
        vs20   = res.get("precio_vs_sma20", "—")
        pct_b  = res.get("boll_pct_b")

        rsi_color = "#f87171" if (rsi or 50)>70 else ("#2A9D8F" if (rsi or 50)<30 else "#60a5fa")

        def m(title, val, color="#60a5fa"):
            return ui.column(2, ui.div(
                ui.tags.p(title, class_="card-title"),
                ui.tags.p(str(val) if val is not None else "—",
                          class_="metric-big", style=f"color:{color}"),
            ))
        return ui.row(
            m("Precio", f"${price:.2f}" if price else "—"),
            m("RSI (14)", f"{rsi:.1f}" if rsi else "—", rsi_color),
            m("MACD", "Alcista ↑" if macd_p else "Bajista ↓", "#2A9D8F" if macd_p else "#E76F51"),
            m("vs SMA20", vs20.upper(), "#2A9D8F" if vs20=="sobre" else "#E76F51"),
            m("Bollinger %B", f"{pct_b:.2f}" if pct_b else "—",
              "#f87171" if (pct_b or 0)>0.8 else ("#2A9D8F" if (pct_b or 0)<0.2 else "#60a5fa")),
            m("Días datos", data.get("total_dias","—"), "#94a3b8"),
        )

    def _plotly_chart(html_str):
        return ui.HTML(html_str)

    @output
    @render.ui
    def ind_chart():
        data, err = _ind()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Sin datos", style="color:#6B7280")
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            df = pd.DataFrame(data.get("datos", [])).dropna(subset=["cierre"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["cierre"], name="Precio",
                                     line=dict(color="#2A9D8F", width=2)))
            for col, color, name in [("sma_20","#f59e0b","SMA 20"),("sma_50","#a78bfa","SMA 50"),
                                      ("boll_superior","#3A4256","BB Sup"),("boll_inferior","#3A4256","BB Inf")]:
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df[col], name=name,
                                             line=dict(color=color, width=1,
                                                       dash="dot" if "boll" in col else "solid")))
            fig.update_layout(height=280, paper_bgcolor="#FBFAF6", plot_bgcolor="#FFFFFF",
                              font=dict(color="#3A4256", size=11), margin=dict(t=20,b=20,l=40,r=10),
                              legend=dict(orientation="h", y=1.05, bgcolor="rgba(255,255,255,0)"))
            fig.update_xaxes(gridcolor="#E8E2D2")
            fig.update_yaxes(gridcolor="#E8E2D2")
            return ui.HTML(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
        except Exception as e:
            return ui.tags.p(str(e), style="color:#E76F51;font-size:12px")

    @output
    @render.ui
    def ind_rsi():
        data, err = _ind()
        if err or data is None: return ui.tags.p(err or "Sin datos", style="color:#6B7280;font-size:12px")
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            df = pd.DataFrame(data.get("datos",[])).dropna(subset=["rsi_14"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["rsi_14"], line=dict(color="#7C3AED", width=1.5)))
            fig.add_hline(y=70, line_dash="dash", line_color="#E76F51", annotation_text="70")
            fig.add_hline(y=30, line_dash="dash", line_color="#2A9D8F", annotation_text="30")
            fig.update_layout(height=180, paper_bgcolor="#FBFAF6", plot_bgcolor="#FFFFFF",
                              font=dict(color="#3A4256",size=10), showlegend=False,
                              margin=dict(t=10,b=20,l=40,r=10), yaxis=dict(range=[0,100]))
            fig.update_xaxes(gridcolor="#E8E2D2"); fig.update_yaxes(gridcolor="#E8E2D2")
            return ui.HTML(pio.to_html(fig, include_plotlyjs=False, full_html=False))
        except Exception as e:
            return ui.tags.p(str(e), style="color:#E76F51;font-size:11px")

    @output
    @render.ui
    def ind_macd():
        data, err = _ind()
        if err or data is None: return ui.tags.p(err or "Sin datos", style="color:#6B7280;font-size:12px")
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            df = pd.DataFrame(data.get("datos",[])).dropna(subset=["macd"])
            colors = ["#2A9D8F" if v>=0 else "#f87171" for v in df["macd_hist"].fillna(0)]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["fecha"], y=df["macd_hist"], name="Hist.", marker_color=colors))
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["macd"],       name="MACD",  line=dict(color="#60a5fa",width=1.5)))
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["macd_señal"], name="Señal", line=dict(color="#F4A261",width=1.5)))
            fig.update_layout(height=180, paper_bgcolor="#FBFAF6", plot_bgcolor="#FFFFFF",
                              font=dict(color="#3A4256",size=10),
                              legend=dict(orientation="h",bgcolor="rgba(255,255,255,0)",font=dict(size=10)),
                              margin=dict(t=10,b=20,l=40,r=10))
            fig.update_xaxes(gridcolor="#E8E2D2"); fig.update_yaxes(gridcolor="#E8E2D2")
            return ui.HTML(pio.to_html(fig, include_plotlyjs=False, full_html=False))
        except Exception as e:
            return ui.tags.p(str(e), style="color:#E76F51;font-size:11px")

    @output
    @render.ui
    def ind_hist_rend():
        data, err = _ind()
        if err or data is None: return ui.tags.p(err or "Sin datos", style="color:#6B7280;font-size:12px")
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            import numpy as np
            from scipy import stats

            df = pd.DataFrame(data.get("datos", [])).dropna(subset=["cierre"])
            if len(df) < 10:
                return ui.tags.p("Insuficientes datos para histograma", style="color:#6B7280;font-size:11px")

            precios = df["cierre"].astype(float)
            rend_log = np.log(precios / precios.shift(1)).dropna()

            mu  = float(rend_log.mean())
            sig = float(rend_log.std())

            fig = go.Figure()
            # Histograma
            fig.add_trace(go.Histogram(
                x=rend_log,
                nbinsx=50,
                name="Rendimientos",
                marker_color="#60a5fa",
                opacity=0.7,
                histnorm="probability density",
            ))
            # Curva normal superpuesta
            x_range = np.linspace(rend_log.min(), rend_log.max(), 200)
            y_norm  = stats.norm.pdf(x_range, mu, sig)
            fig.add_trace(go.Scatter(
                x=x_range, y=y_norm,
                mode="lines", name=f"Normal(mu={mu:.4f}, sig={sig:.4f})",
                line=dict(color="#E76F51", width=2.5),
            ))
            # Líneas VaR 95% y 99%
            var95 = mu + stats.norm.ppf(0.05) * sig
            var99 = mu + stats.norm.ppf(0.01) * sig
            fig.add_vline(x=var95, line_color="#B8772F", line_dash="dash",
                          annotation_text="VaR 95%", annotation_font_color="#F4A261")
            fig.add_vline(x=var99, line_color="#E76F51", line_dash="dash",
                          annotation_text="VaR 99%", annotation_font_color="#E76F51")

            # Estadísticas en anotación
            kurt = float(rend_log.kurtosis())
            skew = float(rend_log.skew())
            fig.add_annotation(
                x=0.02, y=0.97, xref="paper", yref="paper",
                text=f"Kurt: {kurt:.2f} | Skew: {skew:.2f}<br>Media: {mu*100:.4f}% | Vol: {sig*100:.4f}%",
                showarrow=False, align="left",
                font=dict(size=10, color="#94a3b8"),
                bgcolor="#F7F3EB", bordercolor="#D3CCBC", borderwidth=1,
            )
            fig.update_layout(
                height=260,
                paper_bgcolor="#FBFAF6", plot_bgcolor="#FFFFFF",
                font=dict(color="#3A4256", size=11),
                legend=dict(orientation="h", y=-0.2, bgcolor="rgba(255,255,255,0)", font=dict(size=10)),
                margin=dict(t=20, b=60, l=50, r=10),
                xaxis_title="Rendimiento logaritmico diario",
                yaxis_title="Densidad",
                bargap=0.02,
            )
            fig.update_xaxes(gridcolor="#E8E2D2")
            fig.update_yaxes(gridcolor="#E8E2D2")
            return ui.HTML(pio.to_html(fig, include_plotlyjs=False, full_html=False))
        except Exception as e:
            return ui.tags.p(str(e), style="color:#E76F51;font-size:11px")

    # ════════════════════════════════════════════════════
    # TAB 2 — RIESGO
    # ════════════════════════════════════════════════════

    # ── Estado reactivo del portafolio ──────────────────────────────────────────
    _port_rv = reactive.Value({
        t: round(1/len(TICKERS_DEFAULT), 4) for t in TICKERS_DEFAULT
    })

    @output
    @render.ui
    def port_chips_ui():
        port = _port_rv.get()
        tickers = list(port.keys())
        # Solo activos del catálogo que no están ya en el portafolio
        catalogo = ["AAPL","MSFT","JPM","XOM","JNJ","SAP.DE","NOVN.SW","EC","CIB","TM",
                    "GOOGL","AMZN","TSLA","BAC","GS","CVX","PFE","WMT","F"]
        disponibles = {"": "— Agregar activo —"}
        disponibles.update({t: t for t in catalogo if t not in tickers})

        chips = [
            ui.tags.span(
                t,
                ui.tags.span(
                    " ×",
                    onclick=f"Shiny.setInputValue('_rm_ticker','{t}',{{priority:'event'}})",
                    class_="rm",
                ),
                class_="ticker-chip",
            )
            for t in tickers
        ]
        return ui.div(
            ui.tags.p("ACTIVOS EN EL PORTAFOLIO", style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.12em;color:#888;text-transform:uppercase;margin-bottom:6px"),
            ui.div(*chips, style="margin-bottom:10px"),
            ui.input_select("_add_ticker", None, choices=disponibles, width="100%"),
            style="margin-bottom:12px",
        )

    @output
    @render.ui
    def port_pesos_ui():
        port  = _port_rv.get()
        total = sum(port.values())
        ok    = abs(total - 1.0) <= 0.005
        rows  = []
        for t, w in port.items():
            tid = t.replace(".","_").replace("-","_")
            rows.append(ui.div(
                ui.tags.span(t, class_="peso-label"),
                ui.input_slider(f"_w_{tid}", None, 0.0, 1.0, w, step=0.01, width="100%"),
                ui.tags.span(f"{w:.2f}", class_="peso-num", id=f"_wv_{tid}"),
                class_="peso-row",
            ))
        badge_cls = "sum-badge sum-ok" if ok else "sum-badge sum-warn"
        badge_txt = f"Suma: {total:.3f} ✓" if ok else f"Suma: {total:.3f} — debe ser 1.0"
        return ui.div(
            ui.tags.p("PESOS (deben sumar 1.0)", style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.12em;color:#888;text-transform:uppercase;margin-bottom:6px"),
            *rows,
            ui.tags.span(badge_txt, class_=badge_cls),
            ui.input_action_button("_norm_btn", "Normalizar pesos", class_="btn-primary",
                                   style="margin-top:6px;font-size:11px;padding:6px!important"),
            style="margin-bottom:12px",
        )

    # Agregar ticker
    @reactive.effect
    @reactive.event(input._add_ticker)
    def _on_add():
        val = input._add_ticker()
        if not val or val == "": return
        port = dict(_port_rv.get())
        if val in port: return
        port[val] = round(1 / (len(port) + 1), 4)
        # Renormalizar
        total = sum(port.values())
        port = {k: round(v/total, 4) for k, v in port.items()}
        _port_rv.set(port)

    # Quitar ticker
    @reactive.effect
    @reactive.event(input._rm_ticker)
    def _on_remove():
        val = input._rm_ticker()
        if not val: return
        port = {k: v for k, v in _port_rv.get().items() if k != val}
        if len(port) < 2: return
        total = sum(port.values()) or 1
        port = {k: round(v/total, 4) for k, v in port.items()}
        _port_rv.set(port)

    # Normalizar pesos
    @reactive.effect
    @reactive.event(input._norm_btn)
    def _on_norm():
        port   = dict(_port_rv.get())
        tickers = list(port.keys())
        nuevos = {}
        for t in tickers:
            tid = t.replace(".","_").replace("-","_")
            try:
                w = float(getattr(input, f"_w_{tid}")())
            except Exception:
                w = port[t]
            nuevos[t] = max(0.0, w)
        total = sum(nuevos.values()) or 1
        nuevos = {k: round(v/total, 4) for k, v in nuevos.items()}
        _port_rv.set(nuevos)

    @reactive.calc
    @reactive.event(input.r_btn)
    def _riesgo():
        port    = dict(_port_rv.get())
        tickers = list(port.keys())
        # Leer sliders actuales
        pesos = []
        for t in tickers:
            tid = t.replace(".","_").replace("-","_")
            try:
                w = float(getattr(input, f"_w_{tid}")())
            except Exception:
                w = port[t]
            pesos.append(max(0.0, w))
        # Normalizar silenciosamente
        total = sum(pesos) or 1
        pesos = [round(p/total, 4) for p in pesos]
        tipo  = input.r_tipo()
        conf  = input.r_conf()
        payload = {"tickers": tickers, "pesos": pesos, "nivel_confianza": conf}
        return tipo, tickers, pesos, payload

    @output
    @render.ui
    def riesgo_out():
        result = _riesgo()
        if result is None or result[0] == "error":
            msg = result[1] if result else "Configura el portafolio y presiona Calcular"
            return ui.tags.p(msg, style="color:#6B7280" if result is None else "color:#E76F51")

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
        return ui.tags.p("Selecciona un análisis", style="color:#6B7280")

    def _render_var(data):
        if not data: return ui.tags.p("Sin datos", style="color:#6B7280")
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
            color = "#2A9D8F" if adec else "#f87171"
            return f"""<tr>
                <td style="color:#0E1326">{label}</td>
                <td style="color:#0F1B33">{d.get("var_porcentaje") or "—"}</td>
                <td style="color:#0E1326">{fmt_usd(d.get("var_monetario_usd"))}</td>
                <td style="color:#F4A261">{d.get("cvar_porcentaje") or "—"}</td>
                <td style="color:#0E1326">{fmt_usd(d.get("cvar_monetario_usd"))}</td>
                <td style="color:{color};font-weight:600">{badge}</td>
                <td style="color:#3A4256">{lr_s}</td>
            </tr>"""

        tabla_html = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#F7F3EB">
                {"".join(f'<th style="color:#3A4256;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
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
            ui.tags.p(str(rk.get("recomendacion","—")), style="color:#0F1B33;font-size:13px"),
            ui.tags.p(str(data.get("interpretacion_general","—")), style="color:#3A4256;font-size:12px;margin-top:8px"),
        )


    def _render_capm(data):
        if not data or "activos" not in data:
            return ui.tags.p("Sin datos CAPM", style="color:#6B7280")
        activos = data.get("activos", {})
        if not activos:
            return ui.tags.p("Sin activos calculados (datos insuficientes)", style="color:#E76F51")

        filas = ""
        for t, d in activos.items():
            beta  = d.get("beta") or 0
            er    = fmt_pct(d.get("rendimiento_esperado_capm"))
            alpha = fmt_pct(d.get("alpha_anual"))
            r2    = f"{float(d.get('r_cuadrado') or 0):.4f}"
            tipo  = str(d.get("interpretacion_beta",""))[:45]
            bc    = "#f87171" if beta>1.2 else ("#2A9D8F" if beta<0.8 else "#f59e0b")
            filas += f"""<tr>
                <td style="color:#0E1326;padding:9px 12px">{t}</td>
                <td style="color:{bc};font-weight:600;padding:9px 12px">{beta:.4f}</td>
                <td style="color:#0F1B33;padding:9px 12px">{er}</td>
                <td style="color:#F4A261;padding:9px 12px">{alpha}</td>
                <td style="color:#3A4256;padding:9px 12px">{r2}</td>
                <td style="color:#3A4256;padding:9px 12px;font-size:11px">{tipo}</td>
            </tr>"""

        tabla_html = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#F7F3EB">
                {"".join(f'<th style="color:#3A4256;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
                         for h in ["Ticker","Beta","E(R) CAPM","Alpha Jensen","R2","Tipo"])}
            </tr></thead>
            <tbody>{filas}</tbody>
        </table>"""

        rf    = fmt_pct(data.get("tasa_libre_riesgo_anual"))
        prima = fmt_pct(data.get("prima_riesgo_mercado") or data.get("prima_riesgo_mercado_pct"))

        # Gráfico dispersión Beta — datos sintéticos de regresión por activo
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            import numpy as np
            fig_capm = go.Figure()
            beta_vals = []
            er_vals   = []
            tickers_list = []
            for t, d in activos.items():
                b  = float(d.get("beta") or 0)
                er = float(d.get("rendimiento_esperado_capm") or 0)
                vol = float(d.get("volatilidad_anual") or 0)
                beta_vals.append(b)
                er_vals.append(er * 100)
                tickers_list.append(t)
                fig_capm.add_trace(go.Scatter(
                    x=[b], y=[er * 100],
                    mode="markers+text",
                    name=t,
                    text=[t],
                    textposition="top center",
                    marker=dict(size=10 + vol * 30),
                    hovertemplate=f"<b>{t}</b><br>Beta: {b:.4f}<br>E(R): {er*100:.2f}%<br>Vol: {vol*100:.1f}%<extra></extra>"
                ))
            # Línea SML (Security Market Line)
            rf_val  = float(data.get("tasa_libre_riesgo_anual") or 0.0364) * 100
            prima_v = float(data.get("prima_riesgo_mercado") or -0.06) * 100
            b_range = [min(beta_vals) - 0.05, max(beta_vals) + 0.05]
            sml_y   = [rf_val + b * prima_v for b in b_range]
            fig_capm.add_trace(go.Scatter(
                x=b_range, y=sml_y,
                mode="lines", name="SML",
                line=dict(color="#3A4256", width=1.5, dash="dot"),
                showlegend=True
            ))
            fig_capm.add_vline(x=0, line_color="#D3CCBC", line_width=1)
            fig_capm.add_hline(y=rf_val, line_color="#D3CCBC", line_width=1,
                               annotation_text=f"Rf={rf_val:.2f}%", annotation_font_color="#6B7280")
            fig_capm.update_layout(
                height=320,
                paper_bgcolor="#FBFAF6", plot_bgcolor="#FFFFFF",
                font=dict(color="#3A4256", size=11),
                xaxis_title="Beta (riesgo sistematico)",
                yaxis_title="E(R) CAPM (%)",
                legend=dict(orientation="h", y=-0.2, bgcolor="rgba(255,255,255,0)", font=dict(size=10)),
                margin=dict(t=20, b=60, l=50, r=10),
                showlegend=True,
            )
            fig_capm.update_xaxes(gridcolor="#E8E2D2")
            fig_capm.update_yaxes(gridcolor="#E8E2D2")
            capm_chart_html = pio.to_html(fig_capm, include_plotlyjs=False, full_html=False)
        except Exception as e:
            capm_chart_html = f"<p style='color:#f87171;font-size:11px'>Error grafico: {e}</p>"

        return ui.div(
            ui.tags.p(f"Benchmark: {data.get('benchmark','SPY')} | Rf: {rf} | Prima mercado: {prima}",
                      style="color:#3A4256;font-size:12px;margin-bottom:10px"),
            ui.HTML(tabla_html),
            ui.tags.hr(style="margin:16px 0"),
            ui.tags.p("Security Market Line (SML) — Beta vs E(R) CAPM", class_="card-title"),
            ui.tags.p("Tamano del punto = volatilidad anual del activo", style="color:#6B7280;font-size:11px;margin-bottom:6px"),
            ui.HTML(capm_chart_html),
        )


    def _render_markowitz(data):
        if not data: return ui.tags.p("Sin datos", style="color:#6B7280")
        ms = data.get("portafolio_max_sharpe", {})
        mv = data.get("portafolio_min_varianza", {})

        def tabla_pesos(port):
            pesos = port.get("pesos", {})
            if not pesos:
                return "<tr><td style='color:#3A4256;padding:8px'>Sin datos</td></tr>"
            return "".join(
                f"<tr><td style='color:#0E1326;padding:7px 12px'>{t}</td>"
                f"<td style='color:#60a5fa;padding:7px 12px;font-weight:600'>{float(v)*100:.1f}%</td></tr>"
                for t, v in pesos.items()
            )

        def color_ret(port):
            return "#2A9D8F" if (port.get("retorno_anual") or 0) > 0 else "#f87171"

        html = f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div>
                <p style="color:#3A4256;font-size:11px;font-weight:600;text-transform:uppercase;margin-bottom:6px">Maximo Sharpe</p>
                <p style="color:{color_ret(ms)};font-size:12px;margin-bottom:8px">
                    Retorno: {fmt_pct(ms.get("retorno_anual"))} | Vol: {fmt_pct(ms.get("volatilidad_anual"))} | Sharpe: {ms.get("sharpe_ratio") or 0:.3f}
                </p>
                <table style="width:100%;border-collapse:collapse;font-size:13px">
                    <thead><tr style="background:#F7F3EB">
                        <th style="color:#3A4256;padding:7px 12px;text-align:left;font-size:11px">Ticker</th>
                        <th style="color:#3A4256;padding:7px 12px;text-align:left;font-size:11px">Peso</th>
                    </tr></thead>
                    <tbody>{tabla_pesos(ms)}</tbody>
                </table>
            </div>
            <div>
                <p style="color:#3A4256;font-size:11px;font-weight:600;text-transform:uppercase;margin-bottom:6px">Minima Varianza</p>
                <p style="color:{color_ret(mv)};font-size:12px;margin-bottom:8px">
                    Retorno: {fmt_pct(mv.get("retorno_anual"))} | Vol: {fmt_pct(mv.get("volatilidad_anual"))} | Sharpe: {mv.get("sharpe_ratio") or 0:.3f}
                </p>
                <table style="width:100%;border-collapse:collapse;font-size:13px">
                    <thead><tr style="background:#F7F3EB">
                        <th style="color:#3A4256;padding:7px 12px;text-align:left;font-size:11px">Ticker</th>
                        <th style="color:#3A4256;padding:7px 12px;text-align:left;font-size:11px">Peso</th>
                    </tr></thead>
                    <tbody>{tabla_pesos(mv)}</tbody>
                </table>
            </div>
        </div>"""

        # Heatmap de correlación + nube de portafolios
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            import numpy as np
            from plotly.subplots import make_subplots

            tickers_list = list((ms.get("pesos") or mv.get("pesos") or {}).keys())
            sim = data.get("simulacion", {})
            vols_sim  = sim.get("volatilidades", [])
            rets_sim  = sim.get("retornos", [])
            sharpes_sim = sim.get("sharpes", [])
            frontera  = data.get("frontera_eficiente", [])

            # ── Subplot: nube + frontera (izquierda) | heatmap correlación (derecha)
            fig_m = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Conjunto factible y frontera eficiente", "Correlacion entre activos"],
                column_widths=[0.55, 0.45],
                horizontal_spacing=0.08,
            )

            # Nube de portafolios simulados
            if vols_sim and rets_sim:
                n = min(len(vols_sim), len(rets_sim), len(sharpes_sim)) if sharpes_sim else min(len(vols_sim), len(rets_sim))
                colors_sim = sharpes_sim[:n] if sharpes_sim else [0] * n
                fig_m.add_trace(go.Scatter(
                    x=[v*100 for v in vols_sim[:n]],
                    y=[r*100 for r in rets_sim[:n]],
                    mode="markers",
                    marker=dict(size=3, color=colors_sim, colorscale=[[0,"#E8E2D2"],[0.5,"#2A9D8F"],[1,"#0F1B33"]],
                                showscale=True, colorbar=dict(title="Sharpe", x=0.52, len=0.8)),
                    name="Portafolios simulados",
                    hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>",
                    showlegend=False,
                ), row=1, col=1)

            # Frontera eficiente
            if frontera:
                fv = [p["volatilidad"]*100 for p in frontera]
                fr = [p["retorno"]*100 for p in frontera]
                fig_m.add_trace(go.Scatter(
                    x=fv, y=fr, mode="lines",
                    line=dict(color="#2A9D8F", width=2.5),
                    name="Frontera eficiente",
                ), row=1, col=1)

            # Portafolio max Sharpe
            ms_v = float(ms.get("volatilidad_anual") or 0) * 100
            ms_r = float(ms.get("retorno_anual") or 0) * 100
            mv_v = float(mv.get("volatilidad_anual") or 0) * 100
            mv_r = float(mv.get("retorno_anual") or 0) * 100
            fig_m.add_trace(go.Scatter(
                x=[ms_v], y=[ms_r], mode="markers+text",
                text=["Max Sharpe"], textposition="top right",
                marker=dict(size=12, color="#E76F51", symbol="star"),
                name="Max Sharpe",
            ), row=1, col=1)
            fig_m.add_trace(go.Scatter(
                x=[mv_v], y=[mv_r], mode="markers+text",
                text=["Min Var"], textposition="top right",
                marker=dict(size=12, color="#F4A261", symbol="diamond"),
                name="Min Varianza",
            ), row=1, col=1)

            # Heatmap correlación — llamar /comparar para obtener matriz real
            corr_matrix = None
            if tickers_list:
                try:
                    comp_data, _ = api_get("/comparar", {
                        "tickers": tickers_list,
                        "fecha_inicio": "2022-01-01"
                    })
                    if comp_data:
                        corr_matrix = comp_data.get("matriz_correlacion")
                except Exception:
                    pass

            if corr_matrix and tickers_list:
                z_vals = [[float(corr_matrix.get(t1, {}).get(t2, 0) or 0) for t2 in tickers_list] for t1 in tickers_list]
            else:
                n = len(tickers_list)
                z_vals = [[1.0 if i==j else 0.3 for j in range(n)] for i in range(n)]

            if tickers_list:
                fig_m.add_trace(go.Heatmap(
                    z=z_vals,
                    x=tickers_list, y=tickers_list,
                    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                    text=[[f"{v:.2f}" for v in row] for row in z_vals],
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                    showscale=True,
                    colorbar=dict(title="r", x=1.01, len=0.8),
                ), row=1, col=2)

            fig_m.update_layout(
                height=360,
                paper_bgcolor="#FBFAF6", plot_bgcolor="#FFFFFF",
                font=dict(color="#3A4256", size=10),
                legend=dict(orientation="h", y=-0.18, bgcolor="rgba(255,255,255,0)", font=dict(size=10)),
                margin=dict(t=40, b=60, l=50, r=60),
            )
            fig_m.update_xaxes(gridcolor="#E8E2D2")
            fig_m.update_yaxes(gridcolor="#E8E2D2")
            mko_html = pio.to_html(fig_m, include_plotlyjs=False, full_html=False)
        except Exception as e:
            mko_html = f"<p style='color:#f87171;font-size:11px'>Error grafico: {e}</p>"

        return ui.div(
            ui.tags.p("Frontera eficiente — Programacion Cuadratica (sin ventas en corto)", class_="card-title"),
            ui.HTML(html),
            ui.tags.hr(style="margin:16px 0"),
            ui.tags.p("Conjunto factible, frontera eficiente y correlaciones", class_="card-title"),
            ui.HTML(mko_html),
        )


    def _render_garch(data):
        if not data: return ui.tags.p("Sin datos GARCH", style="color:#6B7280")
        garch   = data.get("garch", {})
        modelos = garch.get("modelos", [])
        mejor   = garch.get("mejor_por_aic","—")
        ewma_d  = data.get("ewma",{}).get("modelos_ewma",{})
        ewma94  = ewma_d.get("ewma_lambda_0.94",{})

        filas = ""
        for m in modelos:
            star = "* " if m.get("modelo") == mejor else ""
            if "error" in m:
                filas += f"<tr><td style='color:#0E1326;padding:9px 12px'>{star}{m['modelo']}</td><td colspan='5' style='color:#f87171;padding:9px 12px'>Error: {str(m['error'])[:60]}</td></tr>"
            else:
                filas += f"""<tr>
                    <td style="color:#0E1326;padding:9px 12px;font-weight:{'600' if star else '400'}">{star}{m['modelo']}</td>
                    <td style="color:#0F1B33;padding:9px 12px">{float(m.get('aic') or 0):.2f}</td>
                    <td style="color:#3A4256;padding:9px 12px">{float(m.get('bic') or 0):.2f}</td>
                    <td style="color:#F4A261;padding:9px 12px">{float(m.get('alpha') or 0):.4f}</td>
                    <td style="color:#3A4256;padding:9px 12px">{float(m.get('beta') or 0):.4f}</td>
                    <td style="color:#2A9D8F;padding:9px 12px">{fmt_pct(m.get('vol_pronostico_anual'))}</td>
                </tr>"""

        tabla_html = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#F7F3EB">
                {"".join(f'<th style="color:#3A4256;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
                         for h in ["Modelo","AIC","BIC","Alpha","Beta","Vol. Pronostico"])}
            </tr></thead>
            <tbody>{filas}</tbody>
        </table>"""

        vol_anual = fmt_pct(ewma94.get("vol_ultimo_anual"))
        return ui.div(
            ui.tags.p(f"EWMA lambda=0.94 — Volatilidad anualizada: {vol_anual}",
                      style="color:#2A9D8F;font-weight:600;margin-bottom:12px"),
            ui.tags.p("ARCH/GARCH — Tabla comparativa AIC/BIC", class_="card-title"),
            ui.HTML(tabla_html),
            ui.tags.p(str(garch.get("interpretacion","—")),
                      style="color:#0F1B33;font-size:12px;margin-top:10px"),
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
        if data is None: return ui.tags.p("Presiona el botón", style="color:#6B7280;font-size:12px")
        params = data.get("parametros", {})
        interp = data.get("interpretacion_parametros", {})
        return ui.div(
            ui.tags.p(f"RMSE: {data.get('rmse_ajuste_pct','—'):.4f}%", style="color:#2A9D8F;font-weight:600"),
            ui.tags.p(f"Forma: {data.get('forma_curva','—')}", style="color:#B8772F;font-size:12px"),
            ui.tags.hr(),
            *[ui.tags.p(v, style="color:#3A4256;font-size:11px;margin-bottom:4px") for v in interp.values()],
        )

    @output
    @render.ui
    def rf_chart():
        data, err = _rf()
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Presiona Cargar curva FRED", style="color:#6B7280")
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
            fig.update_layout(height=320, paper_bgcolor="#FBFAF6", plot_bgcolor="#FFFFFF",
                              font=dict(color="#3A4256",size=11), xaxis_title="Vencimiento (años)",
                              yaxis_title="Rendimiento (%)",
                              legend=dict(orientation="h",bgcolor="rgba(255,255,255,0)"),
                              margin=dict(t=20,b=40,l=50,r=10))
            fig.update_xaxes(gridcolor="#E8E2D2"); fig.update_yaxes(gridcolor="#E8E2D2")
            return ui.HTML(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
        except Exception as e:
            return ui.tags.p(str(e), style="color:#E76F51;font-size:12px")

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
        if data is None: return ui.tags.p("Configura y presiona Calcular duración", style="color:#6B7280")
        shocks = data.get("sensibilidad_shocks", [])
        filas_bono = ""
        for s in shocks:
            shock_pb = int(s.get("shock_pb", 0))
            color_s = "#2A9D8F" if shock_pb < 0 else "#f87171"
            dp_lin = s.get("dp_lineal", 0) or 0
            dp_dc  = s.get("dp_duracion_convexidad", 0) or 0
            dp_ex  = s.get("dp_reprice_exacto", 0) or 0
            p_new  = s.get("precio_nuevo", 0) or 0
            filas_bono += f"""<tr>
                <td style="color:{color_s};font-weight:600;padding:9px 12px">{shock_pb:+d} pb</td>
                <td style="color:#3A4256;padding:9px 12px">{float(dp_lin)*100:.3f}%</td>
                <td style="color:#0F1B33;padding:9px 12px">{float(dp_dc)*100:.3f}%</td>
                <td style="color:#2A9D8F;padding:9px 12px">{float(dp_ex)*100:.3f}%</td>
                <td style="color:#0E1326;padding:9px 12px">${float(p_new):.2f}</td>
            </tr>"""
        tabla_bono = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#F7F3EB">
                {"".join(f'<th style="color:#3A4256;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
                         for h in ["Shock","dP Lineal","dP D+C","dP Exacto","Precio nuevo"])}
            </tr></thead>
            <tbody>{filas_bono}</tbody>
        </table>"""
        return ui.div(
            ui.row(
                ui.column(3, ui.div(ui.tags.p("Precio", class_="card-title"), ui.tags.p(f"${data.get('precio',0):.4f}", class_="metric-big"))),
                ui.column(3, ui.div(ui.tags.p("D. Macaulay (anos)", class_="card-title"), ui.tags.p(f"{data.get('duracion_macaulay_anios',0):.4f}", class_="metric-big metric-pos"))),
                ui.column(3, ui.div(ui.tags.p("D. Modificada", class_="card-title"), ui.tags.p(f"{data.get('duracion_modificada',0):.4f}", class_="metric-big metric-neu"))),
                ui.column(3, ui.div(ui.tags.p("Convexidad", class_="card-title"), ui.tags.p(f"{data.get('convexidad',0):.2f}", class_="metric-big", style="color:#F4A261"))),
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
        if data is None: return ui.tags.p("Ingresa parámetros y presiona Calcular", style="color:#6B7280")
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
                    ui.tags.p("Verificada" if paridad.get("verificada") else "Error",
                              style=f"color:{'#34d399' if paridad.get('verificada') else '#f87171'};font-weight:600"),
                    ui.tags.p(f"Error numérico: {paridad.get('error_numerico','—'):.2e}", style="color:#6B7280;font-size:11px"),
                )),
            ),
            ui.tags.hr(),
            ui.tags.p("Las 5 Greeks", class_="card-title"),
            ui.HTML(f"""
            <table style="width:100%;border-collapse:collapse;font-size:13px">
                <thead><tr style="background:#F7F3EB">
                    <th style="color:#3A4256;padding:9px 12px;text-align:left;font-size:11px">GREEK</th>
                    <th style="color:#3A4256;padding:9px 12px;text-align:left;font-size:11px">SIMBOLO</th>
                    <th style="color:#3A4256;padding:9px 12px;text-align:left;font-size:11px">VALOR</th>
                    <th style="color:#3A4256;padding:9px 12px;text-align:left;font-size:11px">INTERPRETACION</th>
                </tr></thead>
                <tbody>
                    {"".join(
                        f'<tr><td style="color:#0E1326;padding:9px 12px;font-weight:500">{n.title()}</td>'
                        f'<td style="color:#F4A261;padding:9px 12px;font-size:16px">{sym}</td>'
                        f'<td style="color:#0F1B33;padding:9px 12px;font-weight:600;font-family:monospace">{float(g.get(n) or 0):.6f}</td>'
                        f'<td style="color:#3A4256;padding:9px 12px;font-size:11px">{str(interp.get(n,"—"))}</td></tr>'
                        for n, sym in [("delta","d"),("gamma","G"),("vega","v"),("theta","T"),("rho","r")]
                    )}
                </tbody>
            </table>"""),
        )

    @output
    @render.ui
    def stress_port_info():
        port    = _port_rv.get()
        tickers = list(port.keys())
        pesos   = list(port.values())
        chips   = "".join(
            f'<span style="display:inline-block;background:rgba(42,157,143,.08);color:#2A9D8F;'
            f'border:1px solid rgba(42,157,143,.2);border-radius:100px;padding:3px 10px;'
            f'font-family:JetBrains Mono,monospace;font-size:11px;margin:2px">{t} {w:.0%}</span>'
            for t, w in zip(tickers, pesos)
        )
        return ui.div(
            ui.tags.p("PORTAFOLIO ACTIVO", style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.12em;color:#888;text-transform:uppercase;margin-bottom:6px"),
            ui.HTML(f'<div style="margin-bottom:8px;line-height:2.2">{chips}</div>'),
            ui.tags.p("Activos y pesos del tab Riesgo.",
                      style="font-size:11px;color:#6B7280;font-style:italic;margin-bottom:8px"),
        )

    @reactive.calc
    @reactive.event(input.st_btn)
    def _stress():
        port    = _port_rv.get()
        tickers = list(port.keys())
        pesos_d = list(port.values())
        total   = sum(pesos_d) or 1
        pesos   = [round(p/total, 4) for p in pesos_d]
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
        if data is None: return ui.tags.p("Configura y presiona Aplicar escenarios", style="color:#6B7280")
        escenarios = data.get("escenarios", {})
        labels = {
            "tasa_menos_200pb":"Tasa −200pb","tasa_mas_200pb":"Tasa +200pb",
            "caida_mercado_20pct":"Caida -20%","caida_mercado_30pct":"Caida -30%",
            "volatilidad_doble":"Volatilidad ×2","combinado_tormenta_perfecta":"⚡ Tormenta perfecta",
        }
        filas_stress = ""
        for key, esc in escenarios.items():
            perdida = float(esc.get("perdida_total_pct") or esc.get("perdida_portafolio_pct") or 0)
            usd     = float(esc.get("perdida_total_usd") or esc.get("perdida_portafolio_usd") or 0)
            color_p = "#2A9D8F" if perdida > 0 else "#f87171"
            filas_stress += f"""<tr>
                <td style="color:#0E1326;padding:9px 12px;font-weight:500">{labels.get(key, key)}</td>
                <td style="color:#3A4256;padding:9px 12px;font-size:11px">{str(esc.get("shock_descripcion","—"))[:60]}</td>
                <td style="color:{color_p};font-weight:600;padding:9px 12px">{perdida:.2f}%</td>
                <td style="color:#0E1326;padding:9px 12px">{fmt_usd(usd)}</td>
            </tr>"""
        tabla_stress = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#F7F3EB">
                {"".join(f'<th style="color:#3A4256;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
                         for h in ["Escenario","Shock","Perdida %","Perdida USD"])}
            </tr></thead>
            <tbody>{filas_stress}</tbody>
        </table>"""
        return ui.div(
            ui.tags.p(f"VaR base: {data.get('var_base_pct',0):.3f}% | Portafolio: {fmt_usd(data.get('valor_portafolio_usd'))}", style="color:#3A4256;font-size:12px;margin-bottom:10px"),
            ui.HTML(tabla_stress),
            ui.tags.p(str(data.get("interpretacion","")), style="color:#3A4256;font-size:12px;margin-top:10px"),
        )

    # ════════════════════════════════════════════════════
    # TAB 4 — ML
    # ════════════════════════════════════════════════════

    @output
    @render.ui
    def ml_status():
        data, err = api_get("/predict/status")
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if not data: return ui.tags.p("Sin estado", style="color:#6B7280;font-size:12px")
        disp = data.get("disponible", False)
        return ui.div(
            ui.tags.p("Modelo cargado" if disp else "Modelo no disponible",
                      style=f"color:{'#34d399' if disp else '#f59e0b'};font-weight:600"),
            ui.tags.p(f"Versión: {data.get('model_version','—')}", style="color:#3A4256;font-size:11px"),
            ui.tags.p(f"Singleton ID: {data.get('singleton_id','—')}", style="color:#6B7280;font-size:10px"),
            ui.tags.p("Llamar /predict 3 veces → 'modelo cargado' aparece solo 1 vez en logs de uvicorn",
                      style="color:#6B7280;font-size:10px;margin-top:4px"),
        )

    @reactive.effect
    @reactive.event(input.ml_ticker)
    def _auto_load_ml():
        """Cargar indicadores actuales del activo seleccionado automáticamente."""
        ticker = input.ml_ticker() if hasattr(input, 'ml_ticker') else "AAPL"
        if not ticker: return
        try:
            data, err = api_get(f"/indicadores/{ticker}", {"fecha_inicio": "2024-01-01"})
            if err or not data: return
            datos = data.get("datos", [])
            if not datos: return
            ultimo = datos[-1]
            resumen = data.get("resumen", {})
            # Actualizar inputs con valores reales del activo
            from shiny import ui as _ui
            # Los updates de inputs se hacen con session.send_input_message
            # pero en Shiny for Python usamos reactive.Value para pasar los valores
            _ml_features_rv.set({
                "rsi":   round(float(resumen.get("rsi_actual") or 50), 1),
                "macd":  round(float(ultimo.get("macd_hist") or 0), 4),
                "ewma":  round(float(resumen.get("boll_pct_b") or 0.5) * 0.02, 4),
                "r5":    1.2,
                "r21":   3.5,
                "pctb":  round(float(resumen.get("boll_pct_b") or 0.5), 2),
                "estoc": round(float(ultimo.get("esto_k") or 50), 1),
            })
        except Exception:
            pass

    _ml_features_rv = reactive.Value({
        "rsi": 45.0, "macd": 0.5, "ewma": 0.012,
        "r5": 1.2, "r21": 3.5, "pctb": 0.55, "estoc": 55.0,
    })

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
            return ui.tags.p("Ingresa features y presiona Predecir régimen", style="color:#6B7280")
        (data, err), features = result
        if err: return ui.div(ui.tags.p(err, class_="err"))
        if data is None: return ui.tags.p("Sin respuesta del modelo", style="color:#6B7280")

        pred  = data.get("prediction", 0)
        label = data.get("prediction_label", "—")
        colors = {1.0:"#34d399", 0.0:"#f59e0b", -1.0:"#f87171"}
        color  = colors.get(float(pred), "#60a5fa")
        fnames = data.get("feature_names") or ["rsi_14","macd_hist","ewma_vol","ret_5d","ret_21d","pct_b_bollinger","estocastico_k"]
        if not fnames:
            fnames = ["rsi_14","macd_hist","ewma_vol","ret_5d","ret_21d","pct_b_bollinger","estocastico_k"]

        filas_ml = "".join(
            f'<tr><td style="color:#3A4256;padding:8px 12px">{fn}</td><td style="color:#0F1B33;padding:8px 12px;font-family:monospace">{fv:.6f}</td></tr>'
            for fn, fv in zip(fnames, features)
        )
        tabla_ml = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#F7F3EB">
                <th style="color:#3A4256;padding:8px 12px;text-align:left;font-size:11px">FEATURE</th>
                <th style="color:#3A4256;padding:8px 12px;text-align:left;font-size:11px">VALOR</th>
            </tr></thead>
            <tbody>{filas_ml}</tbody>
        </table>"""
        # Limpiar emojis del label para evitar encoding issues en Windows
        label_clean = label.replace("📈","(alcista)").replace("➡️","(lateral)").replace("📉","(bajista)")
        return ui.div(
            ui.tags.p(label_clean, style=f"color:{color};font-size:36px;font-weight:700;text-align:center;margin:16px 0"),
            ui.tags.p(f"Activo: {data.get('ticker','—')} | Modelo: {data.get('model_version','—')}",
                      style="color:#3A4256;font-size:12px;text-align:center"),
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
        if data is None: return ui.tags.p("Presiona Actualizar desde FRED", style="color:#6B7280")
        datos = data.get("datos", {})
        iconos = {"DGS3MO":"[Rf]","DGS10":"[T10]","CPIAUCSL":"[CPI]","UNRATE":"[U]","FEDFUNDS":"[Fed]","VIXCLS":"[VIX]"}
        cards = []
        for serie, info in datos.items():
            if serie.startswith("_"): continue
            valor = info.get("valor")
            cards.append(ui.column(4, ui.div(
                ui.tags.p(f"{iconos.get(serie,'📌')} {info.get('nombre',serie)}", class_="card-title"),
                ui.tags.p(f"{valor:.2f}%" if valor else "—", class_="metric-big"),
                ui.tags.p(info.get("fecha","—"), style="color:#6B7280;font-size:11px"),
                ui.tags.p(info.get("interpretacion",""), style="color:#3A4256;font-size:11px;margin-top:4px"),
                class_="card",
            )))
        ctx = data.get("contexto_macro", {})
        # Construir cards como HTML puro (ui.row(*cards) falla con lista Python en Shiny)
        cards_html = ""
        for serie, info in datos.items():
            if serie.startswith("_"): continue
            valor = info.get("valor")
            label = iconos.get(serie, "[?]")
            nombre = str(info.get("nombre", serie))
            fecha  = str(info.get("fecha", "—"))
            interp = str(info.get("interpretacion", ""))
            val_str = f"{valor:.2f}%" if valor is not None else "—"
            cards_html += f"""
            <div style="background:#FFFFFF;border:1px solid #2d3748;border-radius:12px;padding:16px;margin-bottom:12px">
                <p style="font-size:11px;font-weight:600;text-transform:uppercase;color:#3A4256;margin-bottom:6px">{label} {nombre}</p>
                <p style="font-family:monospace;font-size:26px;font-weight:700;color:#60a5fa;margin:4px 0">{val_str}</p>
                <p style="color:#6B7280;font-size:11px">{fecha}</p>
                <p style="color:#3A4256;font-size:11px;margin-top:4px">{interp}</p>
            </div>"""
        impacto_html = "".join(
            f'<p style="color:#0F1B33;font-size:12px;margin:4px 0">→ {i}</p>'
            for i in ctx.get("impacto_portafolio", [])
        )
        return ui.div(
            ui.HTML(f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px">
                {cards_html}
            </div>
            <div style="background:#FFFFFF;border:1px solid #2d3748;border-radius:12px;padding:16px">
                <p style="font-size:11px;font-weight:600;text-transform:uppercase;color:#3A4256;margin-bottom:8px">CONTEXTO MACRO INTEGRADO</p>
                <p style="color:#3A4256;font-size:13px;margin-bottom:8px">{str(ctx.get("descripcion",""))}</p>
                {impacto_html}
            </div>"""),
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
        if data is None: return ui.tags.p("Ingresa tickers y presiona Comparar", style="color:#6B7280")
        comp = data.get("comparacion", {})
        filas_comp = ""
        for ticker, d in sorted(comp.items(), key=lambda x: x[1].get("sharpe_ratio") or 0, reverse=True):
            ret    = float(d.get("retorno_total") or 0)
            sharpe = float(d.get("sharpe_ratio") or 0)
            rc = "#2A9D8F" if ret > 0 else "#f87171"
            sc = "#2A9D8F" if sharpe > 0 else "#f87171"
            dd = fmt_pct(d.get("max_drawdown"))
            vol = fmt_pct(d.get("volatilidad_anual"))
            tendencia = str(d.get("tendencia_ema") or "—")
            nombre = str(d.get("nombre") or "")[:22]
            pais = str(d.get("pais") or "—")
            ranking = int(d.get("ranking_sharpe") or 0)
            filas_comp += f"""<tr>
                <td style="color:#0E1326;padding:9px 12px;font-weight:500">#{ranking} {ticker}</td>
                <td style="color:#3A4256;padding:9px 12px">{nombre}</td>
                <td style="color:#3A4256;padding:9px 12px">{pais}</td>
                <td style="color:{rc};font-weight:600;padding:9px 12px">{ret*100:.1f}%</td>
                <td style="color:#3A4256;padding:9px 12px">{vol}</td>
                <td style="color:{sc};font-weight:600;padding:9px 12px">{sharpe:.3f}</td>
                <td style="color:#E76F51;padding:9px 12px">{dd}</td>
                <td style="color:#0F1B33;padding:9px 12px;font-size:11px">{tendencia}</td>
            </tr>"""
        tabla_comp = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#F7F3EB">
                {"".join(f'<th style="color:#3A4256;padding:9px 12px;text-align:left;font-size:11px;text-transform:uppercase">{h}</th>'
                         for h in ["Ranking","Nombre","Pais","Retorno","Volatilidad","Sharpe","Max DD","Tendencia"])}
            </tr></thead>
            <tbody>{filas_comp}</tbody>
        </table>"""
        # Gráfico rendimiento acumulado base 100
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            import numpy as np
            fig_comp = go.Figure()
            colors_line = ["#2A9D8F","#E76F51","#F4A261","#60a5fa","#a78bfa","#34d399","#f59e0b","#f87171"]
            for i, (ticker, d) in enumerate(sorted(comp.items(), key=lambda x: x[1].get("sharpe_ratio") or 0, reverse=True)):
                hist = d.get("retornos_acumulados_base100") or d.get("precio_base100")
                fechas_h = d.get("fechas_acumulado") or d.get("fechas")
                if hist and fechas_h and len(hist) == len(fechas_h):
                    color = colors_line[i % len(colors_line)]
                    ret_total = float(d.get("retorno_total") or 0)
                    fig_comp.add_trace(go.Scatter(
                        x=fechas_h, y=hist,
                        mode="lines", name=f"{ticker} ({ret_total*100:+.1f}%)",
                        line=dict(color=color, width=1.8),
                        hovertemplate=f"<b>{ticker}</b><br>%{{x}}<br>Base 100: %{{y:.1f}}<extra></extra>",
                    ))
            if fig_comp.data:
                fig_comp.add_hline(y=100, line_color="#D3CCBC", line_width=1,
                                   line_dash="dot", annotation_text="Base 100")
                fig_comp.update_layout(
                    height=300,
                    paper_bgcolor="#FBFAF6", plot_bgcolor="#FFFFFF",
                    font=dict(color="#3A4256", size=11),
                    yaxis_title="Rendimiento acumulado (base 100)",
                    legend=dict(orientation="h", y=-0.22, bgcolor="rgba(255,255,255,0)", font=dict(size=10)),
                    margin=dict(t=20, b=70, l=50, r=10),
                )
                fig_comp.update_xaxes(gridcolor="#E8E2D2")
                fig_comp.update_yaxes(gridcolor="#E8E2D2")
                comp_chart_html = pio.to_html(fig_comp, include_plotlyjs=False, full_html=False)
            else:
                comp_chart_html = "<p style='color:#6B7280;font-size:11px'>Datos de rendimiento acumulado no disponibles en esta respuesta del backend.</p>"
        except Exception as e:
            comp_chart_html = f"<p style='color:#f87171;font-size:11px'>Error grafico acumulado: {e}</p>"

        return ui.div(
            ui.tags.p(f"Mejor Sharpe: {data.get('mejor_sharpe','—')} | Mayor retorno: {data.get('mejor_retorno','—')} | Menor vol: {data.get('menor_volatilidad','—')}",
                      style="color:#0F1B33;font-size:12px;font-weight:600;margin-bottom:10px"),
            ui.HTML(tabla_comp),
            ui.tags.hr(style="margin:16px 0"),
            ui.tags.p("Rendimiento acumulado base 100", class_="card-title"),
            ui.HTML(comp_chart_html),
        )


app = App(app_ui, server)
