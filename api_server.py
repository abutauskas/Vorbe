"""
Vorbe - AI Coding Assistant Backend
Built to help Vortex developers code faster.

Runs on Groq's hosted API rather than a locally-loaded model, so this can
deploy as a plain serverless function (no GPU, no multi-GB model download).
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from typing import Optional
import json
import logging
import os

# Loads variables from a local .env file if one exists (no-op otherwise, so
# this is safe in production where Vercel injects env vars directly).
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vorbe", version="1.0")

# Wide open by design: this is a public API with no user accounts or
# cookies, gated only by the optional bearer token below.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# Only used when a request includes an image - gpt-oss-120b is text-only.
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Groq's own cap for inline base64 images. The frontend compresses images
# client-side to stay well under this, but a direct API call could bypass
# that, so enforce it server-side too.
MAX_IMAGE_DATA_URL_BYTES = 4 * 1024 * 1024

system_prompts = {}


class GenerateRequest(BaseModel):
    """What the user is asking for"""
    prompt: str
    task_type: str = "script_generation"
    max_tokens: int = 8000
    temperature: float = 0.7
    image: Optional[str] = None  # data:image/...;base64,... - never a remote URL


class GenerateResponse(BaseModel):
    """What we send back"""
    response: str
    task_type: str
    tokens_used: int


# Opt-in bearer token for /generate. Unset by default so local dev needs no setup.
API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN")


def check_auth_token(authorization: str = Header(default=None)) -> None:
    if API_AUTH_TOKEN is None:
        return
    if authorization != f"Bearer {API_AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")


@app.on_event("startup")
async def load_system_prompts():
    """Load the per-task system prompts built from Vortex training data"""
    global system_prompts

    try:
        with open("vortex_training_data/system_prompts.json", 'r') as f:
            prompts_list = json.load(f)
            system_prompts = {p['task']: p['system'] for p in prompts_list}
        logger.info(f"Loaded {len(system_prompts)} task prompts")
    except Exception as e:
        logger.warning(f"Couldn't load task prompts: {e}")
        system_prompts = {}

    if groq_client is None:
        logger.warning("GROQ_API_KEY isn't set - /generate will fail until it is")
    else:
        logger.info(f"Using Groq model: {GROQ_MODEL}")


@app.get("/health")
async def health():
    """Check if the server is alive"""
    return {"status": "healthy", "backend": "groq", "configured": groq_client is not None}


@app.get("/api/info")
async def info():
    """Basic info about the API"""
    return {
        "name": "Vorbe",
        "version": "1.0",
        "status": "running",
        "docs": "/docs"
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, _auth: None = Depends(check_auth_token)):
    """Generate a response for the user's prompt via Groq."""

    if groq_client is None:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY isn't configured")

    model = GROQ_MODEL
    user_content = request.prompt

    if request.image:
        if not request.image.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="image must be a data:image/...;base64,... URL")
        if len(request.image) > MAX_IMAGE_DATA_URL_BYTES:
            raise HTTPException(status_code=413, detail="Attached image is too large")
        model = GROQ_VISION_MODEL
        user_content = [
            {"type": "text", "text": request.prompt},
            {"type": "image_url", "image_url": {"url": request.image}},
        ]

    try:
        system = system_prompts.get(
            request.task_type,
            "You are an expert Vortex Luau programmer."
        )

        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        response_text = completion.choices[0].message.content
        tokens_used = completion.usage.total_tokens if completion.usage else 0

        return GenerateResponse(
            response=response_text,
            task_type=request.task_type,
            tokens_used=tokens_used
        )

    except Exception as e:
        logger.error(f"Something went wrong: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def get_tasks():
    """List what task modes are available"""
    return {
        "tasks": list(system_prompts.keys()),
        "default": "script_generation"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
