"""
MetalForming Edge AI – Inspeção Industrial por Visão Computacional
Serviço de inferência embarcado na Raspberry Pi 5 com YOLOv8n pré-treinado.

Endpoints:
    GET /detect/json   – retorna detecções em JSON estruturado
    GET /detect/image  – retorna imagem PNG anotada com bounding boxes
"""

import time
import io
from contextlib import asynccontextmanager

import httpx
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from ultralytics import YOLO


# ── Carregamento do modelo no lifespan ────────────────────────────────────────
# O modelo é carregado ANTES de aceitar requisições.
# Se falhar, a aplicação não sobe — evita erros silenciosos em produção.

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = YOLO("yolov8n.pt")
    # Warm-up: compila os grafos PyTorch antes da primeira requisição real.
    # Sem isso, a primeira chamada levaria 3–10x mais tempo (JIT compilation).
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    yield


# ── Aplicação ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MetalForming Edge AI Inspection",
    description=(
        "Serviço de visão computacional embarcado na Raspberry Pi 5. "
        "Detecta objetos em imagens usando o modelo pré-treinado YOLOv8n."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Auxiliar: download assíncrono da imagem ───────────────────────────────────

async def download_image(url: str) -> np.ndarray:
    """Baixa uma imagem da URL e retorna um array NumPy BGR."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Erro HTTP ao buscar imagem ({e.response.status_code}): {url}",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Erro de rede ao buscar imagem: {e}",
        )

    img_array = np.frombuffer(response.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail="Não foi possível decodificar a imagem da URL fornecida.",
        )
    return img


def _validar_confianca(confidence: float) -> None:
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=422,
            detail="O parâmetro 'confidence' deve estar entre 0.0 e 1.0.",
        )


# ── Endpoint 1: JSON com detecções ───────────────────────────────────────────

@app.get("/detect/json", summary="Detectar objetos e retornar JSON")
async def detect_json(
    url: str = Query(..., description="URL da imagem a ser analisada"),
    confidence: float = Query(0.25, description="Confiança mínima para aceitar uma detecção (0–1)"),
):
    """
    Executa inferência YOLOv8n sobre a imagem da URL e retorna:
    - Lista completa de detecções (classe, confiança, bounding box)
    - Classes detectadas únicas
    - Contagem de instâncias por classe
    - Metadados (modelo, tempo de inferência, URL recebida)
    """
    _validar_confianca(confidence)
    img = await download_image(url)

    inicio = time.perf_counter()
    # iou=0.45: NMS mais agressivo — elimina bounding boxes sobrepostas do mesmo objeto.
    # O padrão do YOLOv8 é 0.7, que permite sobreposição excessiva e gera caixas duplicadas.
    results = model.predict(source=img, conf=confidence, iou=0.45, verbose=False)[0]
    tempo_ms = round((time.perf_counter() - inicio) * 1000, 2)

    detections = []
    class_counts = {}

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf     = float(box.conf[0])
        bbox     = box.xyxy[0].tolist()

        detections.append({
            "class":      cls_name,
            "confidence": round(conf, 2),
            "bbox":       bbox,
        })
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    return {
        "detections":       detections,
        "classes_detected": list(class_counts.keys()),
        "quant_detect":     class_counts,
        "metadata": {
            "model":             "yolov8n.pt",
            "inference_time_ms": tempo_ms,
            "image_url":         url,
            "bbox_format":       "xyxy",
        },
    }


# ── Endpoint 2: Imagem PNG anotada ───────────────────────────────────────────

@app.get("/detect/image", summary="Detectar objetos e retornar imagem anotada")
async def detect_image(
    url: str = Query(..., description="URL da imagem a ser analisada"),
    confidence: float = Query(0.25, description="Confiança mínima para aceitar uma detecção (0–1)"),
):
    """
    Executa inferência YOLOv8n sobre a imagem da URL e retorna uma imagem
    PNG anotada com bounding boxes e labels desenhados pela Ultralytics.
    """
    _validar_confianca(confidence)
    img = await download_image(url)

    # iou=0.45: NMS mais agressivo — elimina bounding boxes sobrepostas do mesmo objeto.
    # O padrão do YOLOv8 é 0.7, que permite sobreposição excessiva e gera caixas duplicadas.
    results = model.predict(source=img, conf=confidence, iou=0.45, verbose=False)[0]
    annotated = results.plot()  # anotação nativa da Ultralytics (BGR)

    _, encoded = cv2.imencode(".png", annotated)
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/png")
