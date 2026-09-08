from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from backend.pipeline.ENCOD import SchemaEx

from backend.pipeline.TENSORING import (
    make_tensor,
    load_autoencoder,
    make_embedding,
    get_or_train_autoencoder
)

from backend.pipeline.slm_inference import (
    generate_text
)

# =============================================================
# MODEL
# =============================================================

MODEL_PATH = "backend/data/autoencoder.pth"

autoencoder_27, device = load_autoencoder(
    MODEL_PATH,
    27
)

# =============================================================
# FASTAPI
# =============================================================

app = FastAPI(
    title="Sameer PRODigy 🧨🧨",
    description="Backend for JSON encoding, tensoring, RAG and SLM",
    version="0.1.0"
)

# =============================================================
# REQUEST
# =============================================================

class AskRequest(BaseModel):
    data: list[dict]
    question: str

class EncodeRequest(BaseModel):
    data: list[dict]

# =============================================================
# ROOT
# =============================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "SchemaX backend is alive"
    }

# =============================================================
# HEALTH
# =============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# =============================================================
# ENCODE
# =============================================================

@app.post("/encode")
def encode(request: EncodeRequest):
    df = pd.DataFrame(request.data)
    df = df.dropna()

    encoder = SchemaEx()
    encoded_data = encoder.detect_and_encode(df)

    print("ENCODE:", encoded_data.shape)

    return {
        "rows": encoded_data.shape[0],
        "features": encoded_data.shape[1],
        "encoded_data": encoded_data.tolist()
    }

# =============================================================
# TENSOR
# =============================================================

@app.post("/tensor")
def tensor(request: EncodeRequest):
    df = pd.DataFrame(request.data)
    df = df.dropna()

    encoder = SchemaEx()
    encoded_data = encoder.detect_and_encode(df)
    tensor_data = make_tensor(encoded_data)

    print("TENSOR:", tensor_data.shape)

    return {
        "rows": tensor_data.shape[0],
        "features": tensor_data.shape[1],
        "dtype": str(tensor_data.dtype),
        "tensor": tensor_data.tolist()
    }

# =============================================================
# EMBED
# =============================================================

@app.post("/embed")
def embed(request: EncodeRequest):
    df = pd.DataFrame(request.data)
    df = df.dropna()

    if df.empty:
        return {"error": "No valid rows remain after removing missing values."}

    encoder = SchemaEx()
    encoded_data = encoder.detect_and_encode(df)

    print("EMBED encoded shape:", encoded_data.shape)

    tensor_data = make_tensor(encoded_data)

    if tensor_data.ndim != 2:
        return {"error": "Tensor must be 2-dimensional.", "tensor_shape": list(tensor_data.shape)}

    model, device_used = get_or_train_autoencoder(
        tensor_data, device, pretrained_model=autoencoder_27, pretrained_features=27
    )
    embedding = make_embedding(tensor_data, model, device_used)

    print("EMBEDDING shape:", embedding.shape)

    return {
        "rows": embedding.shape[0],
        "embedding_features": embedding.shape[1],
        "embedding": embedding.tolist()
    }

# =============================================================
# ASK
# =============================================================

@app.post("/ask")
def ask(request: AskRequest):
    df = pd.DataFrame(request.data)
    df = df.dropna()

    if df.empty:
        return {"error": "No valid data remains after removing missing values."}

    encoder = SchemaEx()
    encoded_data = encoder.detect_and_encode(df)

    print("ASK encoded shape:", encoded_data.shape)

    tensor_data = make_tensor(encoded_data)

    model, device_used = get_or_train_autoencoder(
        tensor_data, device, pretrained_model=autoencoder_27, pretrained_features=27
    )
    embedding = make_embedding(tensor_data, model, device_used)

    print("ASK embedding shape:", embedding.shape)

    prompt = (
        request.question
        + "\n\n"
        + "Data embedding: "
        + ", ".join(f"{float(x):.3f}" for x in embedding[0])
        + "\nAnswer:"
    )

    answer = generate_text(prompt)

    return {
        "question": request.question,
        "embedding_features": embedding.shape[1],
        "answer": answer
    }