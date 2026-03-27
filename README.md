# MetalForming Edge AI – Inspeção Industrial por Visão Computacional

Sistema de visão computacional embarcado para segurança em prensas industriais da **MetalForming Brasil S.A.**

Executa o modelo pré-treinado **YOLOv8n** dentro de um container Docker na **Raspberry Pi 5** e expõe dois endpoints HTTP para detecção de objetos em tempo real, sem dependência de nenhuma infraestrutura de nuvem.

---

## Arquitetura

```
Câmera (URL de snapshot)
        │
        ▼
┌──────────────────────────────────┐
│  Raspberry Pi 5  (ARM64)         │
│  ┌───────────────────────────┐   │
│  │  Container Docker         │   │
│  │  FastAPI + Uvicorn        │◄──┼── GET /detect/json   :8000
│  │  main.py + YOLOv8n        │◄──┼── GET /detect/image  :8000
│  └───────────────────────────┘   │
└──────────────────────────────────┘
```

---

## Estrutura do Projeto

```
.
├── main.py               # Aplicação FastAPI com os dois endpoints
├── Dockerfile            # Imagem ARM64 com pesos do modelo embutidos
├── docker-compose.yml    # restart:always + healthcheck
├── requirements.txt      # Dependências com versões fixas
├── pyproject.toml        # Configuração do pytest
├── tests/
│   ├── conftest.py
│   └── test_api.py       # 22 testes automatizados
└── README.md
```

---

## Endpoints

Ambos aceitam os mesmos parâmetros de consulta:

| Parâmetro | Tipo | Obrigatório | Padrão | Descrição |
|-----------|------|-------------|--------|-----------|
| `url` | string | ✅ | — | URL da imagem a analisar |
| `confidence` | float | ❌ | `0.25` | Confiança mínima (0.0–1.0) |

### `GET /detect/json`

Retorna JSON com a lista de detecções, contagem por classe e metadados de inferência.

**Exemplo de requisição:**
```bash
curl "http://localhost:8000/detect/json?url=https://ultralytics.com/images/bus.jpg&confidence=0.5"
```

**Exemplo de resposta:**
```json
{
  "detections": [
    { "class": "bus",    "confidence": 0.87, "bbox": [22, 230, 804, 757] },
    { "class": "person", "confidence": 0.85, "bbox": [669, 391, 809, 879] }
  ],
  "classes_detected": ["bus", "person"],
  "quant_detect": { "bus": 1, "person": 3 },
  "metadata": {
    "model": "yolov8n.pt",
    "inference_time_ms": 187.4,
    "image_url": "https://ultralytics.com/images/bus.jpg",
    "bbox_format": "xyxy"
  }
}
```

### `GET /detect/image`

Retorna uma imagem **PNG anotada** com bounding boxes e labels desenhados sobre cada objeto detectado.

**Exemplo de requisição:**
```bash
curl "http://localhost:8000/detect/image?url=https://ultralytics.com/images/bus.jpg&confidence=0.5" \
     --output anotada.png
```

---

## Instalação e Deploy

### 1. Instalar Docker na Raspberry Pi 5 (Debian Trixie)

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Clonar o repositório

```bash
git clone https://github.com/RodriguesWeliton/Edge_AI_Tste.git
cd Edge_AI_Teste
```

### 3. Subir a aplicação

```bash
docker compose up --build -d
```

O primeiro build baixa as dependências e incorpora os pesos do modelo à imagem (~15–30 min). Builds seguintes usam cache e ficam prontos em segundos.

### 4. Verificar se está rodando

```bash
# Aguardar status 'healthy' (~60 s)
docker compose ps

# Acompanhar os logs
docker compose logs -f
```

### 5. Acessar a documentação interativa (Swagger UI)

```
http://<IP_DA_RASPBERRY>:8000/docs
```

---

## Testes Automatizados

```bash
# Instalar dependências
pip install -r requirements.txt pytest

# Executar os testes
pytest tests/ -v
```

Os testes cobrem:

- Schema completo da resposta JSON (`detections`, `classes_detected`, `quant_detect`, `metadata`)
- Filtro de confiança (threshold menor → mais detecções)
- Consistência entre `quant_detect` e `detections`
- Formato PNG válido (magic bytes `\x89PNG`)
- Tratamento de erros: URL ausente, confidence inválida, host inacessível → HTTP 422
- Disponibilidade do Swagger UI em `/docs`

Os testes também rodam automaticamente no **GitHub Actions** a cada push.

---

## Tratamento de Erros

| Cenário | Código HTTP |
|---------|-------------|
| URL ausente | 422 |
| `confidence` fora de `[0.0, 1.0]` | 422 |
| Host DNS não resolvível | 422 |
| URL retorna status 4xx/5xx | 422 |
| Bytes recebidos não são imagem válida | 422 |
| Erro interno ao codificar PNG | 500 |

---

## Desempenho na Raspberry Pi 5

| Métrica | Valor esperado |
|---------|----------------|
| Processador | ARM Cortex-A76 @ 2,4 GHz (quad-core, 64-bit) |
| Latência de inferência (YOLOv8n) | ~150–400 ms por imagem |
| Consumo de memória do container | ~400–600 MB |

> Para latência abaixo de 50 ms, utilize o **Raspberry Pi AI Kit** com acelerador Hailo-8L (26 TOPS NPU), compatível com a Pi 5 via slot PCIe Gen 2.

---

## Comandos Úteis

```bash
docker compose ps                # status do container
docker compose logs -f           # logs em tempo real
docker compose down              # parar
docker compose restart           # reiniciar sem rebuild
docker compose up --build -d     # rebuild completo e reiniciar
docker stats metalforming_vision # uso de CPU e memória
vcgencmd measure_temp            # temperatura da CPU da Pi 5
```
