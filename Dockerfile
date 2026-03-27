# ─────────────────────────────────────────────────────────────────────────────
# MetalForming Edge AI – Inspeção Industrial
# Alvo: Raspberry Pi 5 (ARM64 / aarch64)
# OS compatível: Debian Trixie, Bookworm ou Raspberry Pi OS 64-bit
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bookworm

# ── Dependências do sistema ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependências Python ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código-fonte ──────────────────────────────────────────────────────────────
COPY main.py .

# ── Pré-download dos pesos do modelo (bake em build time) ────────────────────
# Garante funcionamento em redes industriais sem acesso à internet.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# Worker único: YOLOv8n não é thread-safe no nível do PyTorch.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
