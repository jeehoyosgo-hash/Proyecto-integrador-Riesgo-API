# ──────────────────────────────────────────────────────────
# Dockerfile multi-stage — Proyecto Integrador Riesgo
# Base: python:3.11.9-slim-bookworm (requerida por la rúbrica)
#
# Stage 1 (builder): instala todas las dependencias con compiladores
# Stage 2 (runtime): imagen final SIN compiladores — ~150 MB vs ~600 MB
#
# Beneficio del multi-stage:
#   - Imagen más pequeña → cold-start más rápido en Render free-tier
#   - Sin build-essential ni gcc en producción → superficie de ataque menor
# ──────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────
FROM python:3.11.9-slim-bookworm AS builder

WORKDIR /build

# Instalar compiladores solo en el stage de build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias en --user (sin sudo en runtime)
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────
FROM python:3.11.9-slim-bookworm

WORKDIR /app

# Copiar paquetes instalados del builder
COPY --from=builder /root/.local /root/.local

# Configurar PATH para los paquetes de --user
ENV PATH=/root/.local/bin:$PATH
# No escribir .pyc → imagen más limpia
ENV PYTHONDONTWRITEBYTECODE=1
# Logs sin buffer → visibles en tiempo real en Render
ENV PYTHONUNBUFFERED=1

# Copiar solo el código de la aplicación (no requirements, no cache)
COPY app ./app

# Puerto de FastAPI/uvicorn
EXPOSE 8000

# Comando de inicio
# --host 0.0.0.0 es necesario para Render (bind a todas las interfaces)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
