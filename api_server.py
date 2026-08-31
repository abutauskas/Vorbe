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
import httpx
import json
import logging
import os
import re

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
# Groq is only used for vision now - text generation runs on OpenRouter (below).
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Text generation runs on two free OpenRouter models, routed by task type:
# a coding-focused model for script generation/bug fixing/security review,
# and a documentation-focused model for API/concept explanations. Both are
# free-tier ($0/request, confirmed via usage.cost) and OpenAI-compatible.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_CODING_MODEL = os.environ.get("OPENROUTER_CODING_MODEL", "poolside/laguna-s-2.1:free")
OPENROUTER_DOC_MODEL = os.environ.get("OPENROUTER_DOC_MODEL", "minimax/minimax-m3:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DOCUMENTATION_TASK_TYPES = {"documentation"}

# Groq's own cap for inline base64 images. The frontend compresses images
# client-side to stay well under this, but a direct API call could bypass
# that, so enforce it server-side too.
MAX_IMAGE_DATA_URL_BYTES = 4 * 1024 * 1024

system_prompts = {}
qa_pairs = []

# There's a real, unrelated product also called "Vortex Studio" (CM Labs'
# robotics/vehicle physics simulator, C++/Lua-based). Without this, the
# model blends real facts about that product with invented details for
# this platform, producing confident-sounding but wrong answers - e.g.
# claiming Vortex uses "Lua 5.3, not Luau" or citing fabricated vtx::
# C++ classes. Prepended to every system prompt below.
PLATFORM_CONTEXT = (
    "Vortex here is a Roblox-like game development platform: scripts are "
    "written in Luau, and the object model uses services such as Workspace, "
    "ReplicatedStorage, ServerScriptService, and StarterPlayerScripts, with "
    "Instance-based objects (Parts with properties like Position, Size, "
    "Color, Anchored, CanCollide, and methods like SetAttribute/GetAttribute). "
    "This is NOT CM Labs' \"Vortex Studio\" physics/robotics simulator - "
    "never reference that product's APIs (a C++ vtx:: namespace, lua_State, "
    "RemoteEventServer, vsWorld, and similar), since none of that applies here. "
    "Vortex has NO documented GUI/UI system as of the current public docs - "
    "no ScreenGui, Frame, TextLabel, TextButton, ImageLabel, or any other "
    "GuiObject-style class exists in the class reference. If asked to build "
    "a GUI, shop UI, menu, HUD, or similar on-screen interface, say plainly "
    "that this isn't a documented Vortex feature yet rather than generating "
    "Roblox-style ScreenGui code, which does not carry over. "
    "If you're not certain about a specific class, method, or property, say "
    "so plainly rather than inventing plausible-sounding details."
)


class GenerateRequest(BaseModel):
    """What the user is asking for"""
    prompt: str
    task_type: str = "script_generation"
    # Groq's free tier caps openai/gpt-oss-120b at 8,000 tokens/minute total
    # (prompt + completion, across all requests). Keeping this well under
    # that leaves room for more than one request per minute instead of one
    # long reply eating the whole budget and 429ing the next call.
    max_tokens: int = 1500
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
    """Load the per-task system prompts and reference docs built from Vortex training data"""
    global system_prompts, qa_pairs

    try:
        with open("vortex_training_data/system_prompts.json", 'r') as f:
            prompts_list = json.load(f)
            system_prompts = {p['task']: p['system'] for p in prompts_list}
        logger.info(f"Loaded {len(system_prompts)} task prompts")
    except Exception as e:
        logger.warning(f"Couldn't load task prompts: {e}")
        system_prompts = {}

    try:
        with open("vortex_training_data/vortex_qa_pairs.json", 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        logger.info(f"Loaded {len(qa_pairs)} reference QA pairs for grounding")
    except Exception as e:
        logger.warning(f"Couldn't load reference QA pairs: {e}")
        qa_pairs = []

    if groq_client is None:
        logger.warning("GROQ_API_KEY isn't set - image attachments will fail until it is")
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY isn't set - text /generate will fail until it is")
    else:
        logger.info(f"Using OpenRouter models: {OPENROUTER_CODING_MODEL} (coding), {OPENROUTER_DOC_MODEL} (documentation)")


def stem(word: str) -> str:
    """Trivial plural normalizer ("events" -> "event") so a prompt asking
    about "remote events" still matches a topic named "RemoteEvent"."""
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def find_relevant_docs(prompt: str, max_entries: int = 4, max_chars: int = 900) -> str:
    """Keyword-overlap lookup over the real Vortex QA pairs, so answers can
    be grounded in verified docs instead of invented from scratch. No
    embeddings - the corpus (~130 entries) is small enough that keyword
    matching is good enough, and it's zero extra dependencies or cost.
    """
    if not qa_pairs:
        return ""

    prompt_lower = prompt.lower()
    # Whole-word matching, not substring containment - substring checks let
    # short words (e.g. "part") spuriously match inside unrelated longer
    # words, and let single-letter topics like "R"/"G"/"B" match almost
    # anything.
    prompt_words = {stem(w) for w in re.findall(r"[a-z0-9]+", prompt_lower) if len(w) > 3}

    scored = []
    for entry in qa_pairs:
        topic = entry.get("topic", "")
        topic_lower = topic.lower()
        answer_lower = entry.get("answer", "").lower()
        if "stub" in answer_lower:
            continue  # placeholder pages with no real content - would only waste budget
        answer_words = {stem(w) for w in re.findall(r"[a-z0-9]+", answer_lower)}
        # CamelCase topics ("RemoteEvent") split into their own words
        # ("remote", "event") so they match a spaced-out prompt ("remote
        # events") - a straight substring check misses that entirely.
        topic_words = {stem(w.lower()) for w in re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", topic)}

        score = 0
        if topic_lower and len(topic_lower) > 2 and topic_lower in prompt_lower:
            score += 5  # exact topic name mentioned - strong signal
        # Only match against topic/answer, not `questions` - every entry's
        # questions follow the same "What is X? / Tell me about X?" template,
        # so common words there would spuriously match every single entry.
        score += 2 * len(prompt_words & topic_words)
        score += len(prompt_words & answer_words)

        if score >= 2:  # filter out single-coincidence matches
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    lines = []
    total_chars = 0
    for _, entry in scored[:max_entries]:
        line = f"- {entry.get('topic', '')}: {entry.get('answer', '')}"
        if total_chars + len(line) > max_chars:
            break
        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


@app.get("/health")
async def health():
    """Check if the server is alive"""
    return {
        "status": "healthy",
        "backend": "openrouter+groq",
        "configured": bool(OPENROUTER_API_KEY) and groq_client is not None,
    }


@app.get("/api/info")
async def info():
    """Basic info about the API"""
    return {
        "name": "Vorbe",
        "version": "1.0",
        "status": "running",
        "docs": "/docs"
    }


async def call_openrouter(model: str, system: str, user_content, max_tokens: int, temperature: float):
    """Call an OpenRouter chat completion and return (text, tokens_used)."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    # Reasoning models (Poolside included) can burn the whole token budget on
    # an internal `reasoning` field and leave `content` null if max_tokens is
    # too tight - fall back to that rather than returning nothing.
    text = message.get("content") or message.get("reasoning")
    tokens_used = (data.get("usage") or {}).get("total_tokens", 0)
    return text, tokens_used


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, _auth: None = Depends(check_auth_token)):
    """Generate a response for the user's prompt - OpenRouter for text, Groq for vision."""

    try:
        system = system_prompts.get(
            request.task_type,
            "You are an expert Vortex Luau programmer."
        )
        system = f"{system}\n\n{PLATFORM_CONTEXT}"

        relevant_docs = find_relevant_docs(request.prompt)
        if relevant_docs:
            system += (
                "\n\nRelevant reference documentation (verified - treat this "
                "as source of truth over anything you'd otherwise assume):\n"
                + relevant_docs
            )

        if request.image:
            if groq_client is None:
                raise HTTPException(status_code=503, detail="GROQ_API_KEY isn't configured")
            if not request.image.startswith("data:image/"):
                raise HTTPException(status_code=400, detail="image must be a data:image/...;base64,... URL")
            if len(request.image) > MAX_IMAGE_DATA_URL_BYTES:
                raise HTTPException(status_code=413, detail="Attached image is too large")

            user_content = [
                {"type": "text", "text": request.prompt},
                {"type": "image_url", "image_url": {"url": request.image}},
            ]
            completion = groq_client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            response_text = completion.choices[0].message.content
            tokens_used = completion.usage.total_tokens if completion.usage else 0
        else:
            if not OPENROUTER_API_KEY:
                raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY isn't configured")

            model = (
                OPENROUTER_DOC_MODEL
                if request.task_type in DOCUMENTATION_TASK_TYPES
                else OPENROUTER_CODING_MODEL
            )
            response_text, tokens_used = await call_openrouter(
                model, system, request.prompt, request.max_tokens, request.temperature
            )

        return GenerateResponse(
            response=response_text,
            task_type=request.task_type,
            tokens_used=tokens_used
        )

    except HTTPException:
        raise
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
