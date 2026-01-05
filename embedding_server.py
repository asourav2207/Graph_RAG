"""
Local Embedding Server using Sentence-Transformers
Provides OpenAI-compatible /v1/embeddings endpoint
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
import uvicorn

app = FastAPI(title="Local Embedding Server")

# Load model once at startup
print("Loading embedding model...")
model = SentenceTransformer('all-mpnet-base-v2')
print("Model loaded!")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "all-mpnet-base-v2"


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: dict


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    try:
        # Handle single string or list
        texts = [request.input] if isinstance(request.input, str) else request.input
        
        # Generate embeddings
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        
        # Format response like OpenAI
        data = [
            EmbeddingObject(embedding=emb, index=i)
            for i, emb in enumerate(embeddings)
        ]
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model": "all-mpnet-base-v2"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
