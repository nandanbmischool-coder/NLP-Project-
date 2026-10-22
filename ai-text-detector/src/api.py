import json
import os
import re
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import predict_text

BASE = Path(__file__).resolve().parents[1]
DIST = BASE / "frontend" / "dist"

app = FastAPI(title="AI Text Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")


class Item(BaseModel):
    text: str = Field(max_length=30000)


@app.get("/")
def index():
    page = DIST / "index.html"
    if page.exists():
        return FileResponse(page)
    return {
        "message": "Frontend not built. Run `npm run dev` in frontend, or `npm run build`."
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/metrics")
def metrics():
    path = BASE / "models" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@app.post("/predict")
def predict(item: Item):
    return predict_text(item.text)


@app.post("/humanize")
async def humanize(item: Item):
    token = os.getenv("HF_TOKEN")
    if not token:
        return {"text": local_human_style(item.text), "mode": "local"}
    prompt = (
        "Rewrite the following text in a natural, personal human voice. Preserve its meaning, "
        "facts, and approximate length. Do not add claims or commentary. Text:\n\n" + item.text
    )
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-large",
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 700, "temperature": 0.7}},
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail="Hugging Face rewrite service returned an error.")
    payload = response.json()
    if not isinstance(payload, list) or not payload or "generated_text" not in payload[0]:
        raise HTTPException(status_code=502, detail="The rewrite service did not return usable text.")
    return {"text": payload[0]["generated_text"]}


def local_human_style(text: str) -> str:
    """Make conservative readability edits without inventing or removing facts."""
    result = re.sub(r"[ \t]+", " ", text.strip())
    result = re.sub(r"\n{3,}", "\n\n", result)
    replacements = {
        "Furthermore, ": "Also, ",
        "Moreover, ": "Also, ",
        "In conclusion, ": "To sum up, ",
        "It is important to note that ": "",
        "It is worth noting that ": "",
        "utilize": "use",
        "approximately": "about",
    }
    for old, new in replacements.items():
        result = result.replace(old, new).replace(old.lower(), new.lower())
    return result
