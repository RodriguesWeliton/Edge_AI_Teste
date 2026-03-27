"""
tests/test_api.py

Testes de integração para o serviço Edge AI – MetalForming Brasil S.A.

Execute com:
    pytest tests/ -v

Os testes usam o TestClient do FastAPI (transporte ASGI síncrono) — nenhum
servidor HTTP real é necessário. O modelo YOLOv8n é carregado uma única vez
por sessão via fixture de escopo session, evitando recarregamentos desnecessários.
"""

import pytest
from fastapi.testclient import TestClient

from main import app


# ── Fixture compartilhada ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """Inicializa a aplicação e carrega o modelo uma única vez para toda a sessão."""
    with TestClient(app) as c:
        yield c


# ── URL de referência ─────────────────────────────────────────────────────────

# Imagem canônica do YOLOv8 — contém ônibus e pessoas em posições estáveis.
# Garante testes determinísticos e reproduzíveis em qualquer ambiente.
BUS_URL = "https://ultralytics.com/images/bus.jpg"


# ── Testes do endpoint /detect/json ──────────────────────────────────────────

class TestDetectJson:

    def test_status_200_com_url_valida(self, client):
        resp = client.get("/detect/json", params={"url": BUS_URL})
        assert resp.status_code == 200

    def test_estrutura_da_resposta(self, client):
        body = client.get("/detect/json", params={"url": BUS_URL, "confidence": 0.3}).json()
        assert "detections" in body
        assert "classes_detected" in body
        assert "quant_detect" in body
        assert "metadata" in body

    def test_campos_obrigatorios_no_metadata(self, client):
        meta = client.get("/detect/json", params={"url": BUS_URL}).json()["metadata"]
        assert "model" in meta
        assert "inference_time_ms" in meta
        assert "image_url" in meta
        assert "bbox_format" in meta
        assert meta["bbox_format"] == "xyxy"

    def test_schema_de_cada_deteccao(self, client):
        body = client.get("/detect/json", params={"url": BUS_URL, "confidence": 0.3}).json()
        if body["detections"]:
            det = body["detections"][0]
            assert "class" in det
            assert "confidence" in det
            assert "bbox" in det
            assert len(det["bbox"]) == 4

    def test_filtro_de_confianca_reduz_deteccoes(self, client):
        """Com threshold menor, deve haver >= detecções que com threshold maior."""
        baixo = client.get("/detect/json", params={"url": BUS_URL, "confidence": 0.10}).json()
        alto  = client.get("/detect/json", params={"url": BUS_URL, "confidence": 0.90}).json()
        assert len(baixo["detections"]) >= len(alto["detections"])

    def test_contagem_por_classe_corresponde_as_deteccoes(self, client):
        body = client.get("/detect/json", params={"url": BUS_URL, "confidence": 0.3}).json()
        assert sum(body["quant_detect"].values()) == len(body["detections"])

    def test_classes_detected_contem_apenas_classes_presentes(self, client):
        body = client.get("/detect/json", params={"url": BUS_URL, "confidence": 0.3}).json()
        assert set(body["classes_detected"]) == set(body["quant_detect"].keys())

    def test_tempo_de_inferencia_positivo(self, client):
        body = client.get("/detect/json", params={"url": BUS_URL}).json()
        assert body["metadata"]["inference_time_ms"] > 0

    def test_url_ausente_retorna_422(self, client):
        assert client.get("/detect/json").status_code == 422

    def test_confidence_acima_de_1_retorna_422(self, client):
        resp = client.get("/detect/json", params={"url": BUS_URL, "confidence": 1.5})
        assert resp.status_code == 422

    def test_confidence_abaixo_de_0_retorna_422(self, client):
        resp = client.get("/detect/json", params={"url": BUS_URL, "confidence": -0.1})
        assert resp.status_code == 422

    def test_url_inacessivel_retorna_422(self, client):
        resp = client.get("/detect/json", params={"url": "https://host.invalido.xyz/img.jpg"})
        assert resp.status_code == 422


# ── Testes do endpoint /detect/image ─────────────────────────────────────────

class TestDetectImage:

    def test_status_200_com_url_valida(self, client):
        assert client.get("/detect/image", params={"url": BUS_URL}).status_code == 200

    def test_content_type_e_image_png(self, client):
        resp = client.get("/detect/image", params={"url": BUS_URL})
        assert resp.headers["content-type"] == "image/png"

    def test_resposta_possui_magic_bytes_do_png(self, client):
        resp = client.get("/detect/image", params={"url": BUS_URL})
        assert resp.content[:4] == b"\x89PNG"

    def test_imagem_retornada_nao_esta_vazia(self, client):
        resp = client.get("/detect/image", params={"url": BUS_URL})
        assert len(resp.content) > 1000

    def test_confidence_0_retorna_imagem_valida(self, client):
        resp = client.get("/detect/image", params={"url": BUS_URL, "confidence": 0.0})
        assert resp.status_code == 200
        assert resp.content[:4] == b"\x89PNG"

    def test_url_ausente_retorna_422(self, client):
        assert client.get("/detect/image").status_code == 422

    def test_confidence_acima_de_1_retorna_422(self, client):
        resp = client.get("/detect/image", params={"url": BUS_URL, "confidence": 2.0})
        assert resp.status_code == 422

    def test_url_inacessivel_retorna_422(self, client):
        resp = client.get("/detect/image", params={"url": "https://host.invalido.xyz/img.jpg"})
        assert resp.status_code == 422


# ── Swagger / OpenAPI ─────────────────────────────────────────────────────────

class TestDocs:

    def test_swagger_ui_acessivel(self, client):
        assert client.get("/docs").status_code == 200

    def test_openapi_json_contem_ambos_endpoints(self, client):
        schema = client.get("/openapi.json").json()
        assert "/detect/json" in schema["paths"]
        assert "/detect/image" in schema["paths"]
