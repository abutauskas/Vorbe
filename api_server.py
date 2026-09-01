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

# Text generation runs on free OpenRouter models, routed by task type: a
# coding-focused model for script generation/bug fixing/security review, and
# a documentation-focused model for API/concept explanations. Each has a
# backup - OpenRouter's free tier is capped at 20 requests/minute and
# 50/day per account (1000/day once any credits are purchased), shared
# across every free model, so the primary model alone can and does 429
# under real use. Falls back only on 429/5xx (see call_openrouter_with_
# fallback) - a different model has a real shot at those, not at a 400/401.
# All confirmed free ($0/request) and working via real test calls.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_CODING_MODEL = os.environ.get("OPENROUTER_CODING_MODEL", "poolside/laguna-s-2.1:free")
OPENROUTER_CODING_MODEL_BACKUP = os.environ.get("OPENROUTER_CODING_MODEL_BACKUP", "cohere/north-mini-code:free")
OPENROUTER_DOC_MODEL = os.environ.get("OPENROUTER_DOC_MODEL", "minimax/minimax-m3:free")
OPENROUTER_DOC_MODEL_BACKUP = os.environ.get("OPENROUTER_DOC_MODEL_BACKUP", "minimax/minimax-m2.7:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Absolute last resort - a self-hosted Cloudflare Worker (see
# scripts/setup-cloudflare-fallback.sh), only reached if every OpenRouter
# model above has failed. Running on entirely separate infrastructure from
# OpenRouter/Groq is the point - it's meant to survive an outage that takes
# both of those down at once. Unset by default, so this is a no-op until
# someone actually runs the setup script.
CLOUDFLARE_WORKER_URL = os.environ.get("CLOUDFLARE_WORKER_URL")
CLOUDFLARE_WORKER_TOKEN = os.environ.get("CLOUDFLARE_WORKER_TOKEN")
DOCUMENTATION_TASK_TYPES = {"documentation"}

# Groq's own cap for inline base64 images. The frontend compresses images
# client-side to stay well under this, but a direct API call could bypass
# that, so enforce it server-side too.
MAX_IMAGE_DATA_URL_BYTES = 4 * 1024 * 1024

system_prompts = {}
qa_pairs = []
# Loaded at startup from vortex_training_data/confirmed_classes.json and
# confirmed_datatypes.json (kept fresh by update_docs.py) - concrete
# allowlists of what the docs actually confirm exists, so the model has a
# hard boundary to check against instead of a vague "if unsure" instruction,
# which doesn't help when it's confidently *wrong* (assuming Vortex has
# Roblox's GUI system, or its BrickColor - Vortex only has Color3 - rather
# than actually being uncertain). CONFIRMED_CLASSES/_DATATYPES (sets) back
# find_unconfirmed_apis()'s output scan; the _CONTEXT strings are what get
# injected into the system prompt.
CONFIRMED_CLASSES = set()
CONFIRMED_DATATYPES = set()
CONFIRMED_CLASSES_CONTEXT = ""
CONFIRMED_DATATYPES_CONTEXT = ""

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
    "so plainly rather than inventing plausible-sounding details. "
    "Keep any internal reasoning brief and get to the actual answer quickly - "
    "don't re-derive the same constraints repeatedly or second-guess a "
    "settled decision at length before writing the response."
)


class GenerateRequest(BaseModel):
    """What the user is asking for"""
    prompt: str
    task_type: str = "script_generation"
    # Text generation runs on OpenRouter now, which has no token-rate limit
    # (only a request-count cap - see OPENROUTER_CODING_MODEL's comment) so
    # this no longer needs to be squeezed the way it did under Groq's old
    # 8K-tokens/minute cap. Sized generously because reasoning models can
    # spend a lot of their budget on internal deliberation before producing
    # real content - and how much varies run to run (temperature=0.7), so
    # this is risk reduction, not a hard fix. A live A/B on the identical
    # prompt/system-prompt showed reasoning alone using ~1740 tokens on one
    # run; 1500 and then 4000 both still got routed into ReasoningExhausted
    # on real prompts. See call_openrouter_with_fallback's retry pass for
    # the other half of the mitigation.
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


def _load_json_list(path: str, label: str) -> list:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Couldn't load {label}: {e}")
        return []


@app.on_event("startup")
async def load_system_prompts():
    """Load the per-task system prompts and reference docs built from Vortex training data"""
    global system_prompts, qa_pairs, CONFIRMED_CLASSES, CONFIRMED_DATATYPES
    global CONFIRMED_CLASSES_CONTEXT, CONFIRMED_DATATYPES_CONTEXT

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

    confirmed_classes = _load_json_list("vortex_training_data/confirmed_classes.json", "confirmed_classes.json")
    confirmed_datatypes = _load_json_list("vortex_training_data/confirmed_datatypes.json", "confirmed_datatypes.json")
    CONFIRMED_CLASSES = set(confirmed_classes)
    CONFIRMED_DATATYPES = set(confirmed_datatypes)

    if confirmed_classes:
        CONFIRMED_CLASSES_CONTEXT = (
            "The ONLY Vortex classes confirmed in the current documentation "
            "are: " + ", ".join(confirmed_classes) + ". If a script would "
            "need a class not on this list - this includes any GUI, "
            "ScreenGui, Frame, TextButton, or other on-screen UI class - say "
            "plainly that it isn't confirmed to exist in Vortex rather than "
            "assuming it works like the Roblox equivalent. Only go beyond "
            "this list if the user's own prompt or the reference "
            "documentation below explicitly confirms something else."
        )
        logger.info(f"Confirmed-classes allowlist: {len(confirmed_classes)} classes")

    if confirmed_datatypes:
        CONFIRMED_DATATYPES_CONTEXT = (
            "The ONLY Vortex datatypes confirmed in the current documentation "
            "are: " + ", ".join(confirmed_datatypes) + " (each constructed via "
            "its own .new(...) or documented factory method, e.g. "
            "Color3.fromRGB(...)). Vortex does NOT have Roblox's BrickColor - "
            "use Color3 instead. Don't use a datatype not on this list unless "
            "the user's own prompt or the reference documentation below "
            "explicitly confirms it."
        )
        logger.info(f"Confirmed-datatypes allowlist: {len(confirmed_datatypes)} datatypes")

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


def find_relevant_docs(prompt: str, max_entries: int = 8, max_chars: int = 2500) -> str:
    """Keyword-overlap lookup over the real Vortex QA pairs, so answers can
    be grounded in verified docs instead of invented from scratch. No
    embeddings - the corpus (~150 entries) is small enough that keyword
    matching is good enough, and it's zero extra dependencies or cost.

    Budget was tighter (4 entries / 900 chars) back when this ran on Groq's
    8K-tokens/minute free tier; text generation now runs on OpenRouter,
    which has no such per-minute token cap (see GenerateRequest.max_tokens),
    so there's room to surface more grounding per request.
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


def find_unconfirmed_apis(code_text: str) -> list:
    """Scans generated text for Instance.new("X") and X.new(...) calls
    referencing a class/datatype that isn't in the confirmed allowlists -
    a safety net for when the model ignores the system prompt's warning.
    Confirmed to happen in practice, not just in theory: a real response
    used TextLabel (Instance.new("TextLabel")) despite CONFIRMED_CLASSES_
    CONTEXT explicitly listing the confirmed classes and calling out GUI
    classes by name as unconfirmed.
    """
    if not CONFIRMED_CLASSES and not CONFIRMED_DATATYPES:
        return []  # allowlists not loaded - nothing to check against

    found = set()

    for cls in re.findall(r'Instance\.new\(\s*["\']([A-Za-z0-9_]+)["\']', code_text):
        if CONFIRMED_CLASSES and cls not in CONFIRMED_CLASSES:
            found.add(cls)

    # X.new(...) - covers datatype construction (Color3.new, CFrame.new,
    # etc.) and, notably, catches things like BrickColor.new(...) that
    # aren't created via Instance.new at all. "Instance" itself is the
    # pattern above, not this one.
    for dtype in re.findall(r'\b([A-Z][A-Za-z0-9]*)\.new\(', code_text):
        if dtype != "Instance" and CONFIRMED_DATATYPES and dtype not in CONFIRMED_DATATYPES and dtype not in CONFIRMED_CLASSES:
            found.add(dtype)

    return sorted(found)


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


class ReasoningExhausted(Exception):
    """Raised when a reasoning model (Poolside included) burns its entire
    token budget on internal deliberation and never produces real `content`
    - a different failure mode from an HTTP error (this is a normal 200),
    but one a *different*, non-reasoning model in the fallback chain has a
    real shot at avoiding entirely, rather than just retrying the same
    model that just proved it wants more room than it's getting."""
    pass


async def call_openrouter(model: str, system: str, user_content, max_tokens: int, temperature: float):
    """Call an OpenRouter chat completion and return (text, tokens_used).
    Raises ReasoningExhausted instead of returning raw internal monologue as
    if it were the answer - see that class's docstring."""
    async with httpx.AsyncClient(timeout=90) as client:
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
    text = message.get("content")
    tokens_used = (data.get("usage") or {}).get("total_tokens", 0)

    if not text:
        if message.get("reasoning"):
            raise ReasoningExhausted()
        text = ""  # genuinely empty response, not a reasoning-budget issue

    return text, tokens_used


async def _try_models_once(models: list, system: str, user_content, max_tokens: int, temperature: float):
    """One pass through the model list, falling back to the next model on a
    rate limit (429), a provider-side error (5xx), or the model exhausting
    its budget on reasoning without producing content - those are the cases
    where a different free model actually has a real shot at succeeding.
    Anything else (a 400 from a malformed request, a 401 from a bad key)
    will fail on every model identically, so it's raised immediately
    instead of wasting the fallback models on a request that was never
    going to work.

    If the pass ends because every model hit the reasoning-exhausted case,
    raises ReasoningExhausted itself (distinct from an HTTP failure) so the
    caller can decide whether a full retry pass is worth it - see
    call_openrouter_with_fallback, which is the only caller.
    """
    last_error = None
    # Reflects only the LAST model tried, not "did any model in this pass
    # ever hit this" - a pass where model 1 exhausts its reasoning budget
    # but model 2 then 429s should surface as a rate-limit error (accurate,
    # and the outer handler already has good messaging for it), not as
    # reasoning-exhaustion, which isn't what actually happened last.
    hit_reasoning_wall = False
    for i, model in enumerate(models):
        try:
            return await call_openrouter(model, system, user_content, max_tokens, temperature)
        except ReasoningExhausted:
            hit_reasoning_wall = True
            if i < len(models) - 1:
                logger.warning(f"{model} burned its whole budget on reasoning, trying {models[i + 1]}")
                continue
            logger.warning(f"{model} burned its whole budget on reasoning, no more fallback models this pass")
            break
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            last_error = e
            hit_reasoning_wall = False
            if status == 429 or status >= 500:
                if i < len(models) - 1:
                    logger.warning(f"{model} failed ({status}), falling back to {models[i + 1]}")
                    continue
                logger.warning(f"{model} failed ({status}), no more fallback models this pass")
                break
            raise
    if hit_reasoning_wall:
        raise ReasoningExhausted()
    raise last_error


async def call_openrouter_with_fallback(models: list, system: str, user_content, max_tokens: int, temperature: float):
    """Runs _try_models_once, and if every model in that pass hit the
    reasoning wall, tries the whole chain again once before giving up.

    That retry is specifically for the reasoning-exhausted case, not for
    HTTP failures (those propagate immediately, no retry) - confirmed live
    that how much of the budget reasoning eats is genuinely stochastic
    (temperature=0.7): the identical prompt against the identical system
    prompt failed once and then succeeded cleanly on a second try with real
    output, so a second full pass has a real shot where an HTTP failure
    retried immediately mostly wouldn't. Only if BOTH passes exhaust every
    model does this return a plain apology - there's nothing further
    upstream that would handle it better, and showing the raw reasoning
    trace to the user (a wall of "We need to..." internal monologue) is
    worse than admitting it didn't work.
    """
    try:
        return await _try_models_once(models, system, user_content, max_tokens, temperature)
    except ReasoningExhausted:
        logger.warning("Every model hit the reasoning wall - retrying the whole chain once")
        try:
            return await _try_models_once(models, system, user_content, max_tokens, temperature)
        except ReasoningExhausted:
            return (
                "Sorry, I ran out of room thinking this one through and didn't "
                "reach a real answer. Try again, or ask for something more specific.",
                0,
            )


async def call_cloudflare_worker(system: str, user_content, max_tokens: int):
    """Absolute last resort - a self-hosted Cloudflare Worker running
    Workers AI, on completely separate infrastructure from OpenRouter and
    Groq. Only called when both of those have already failed."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            CLOUDFLARE_WORKER_URL,
            headers={
                "Authorization": f"Bearer {CLOUDFLARE_WORKER_TOKEN}",
                "Content-Type": "application/json",
            },
            json={"system": system, "prompt": user_content, "max_tokens": max_tokens},
        )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return text, 0  # Workers AI doesn't return a token count in this response shape


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, _auth: None = Depends(check_auth_token)):
    """Generate a response for the user's prompt - OpenRouter for text, Groq for vision."""

    try:
        system = system_prompts.get(
            request.task_type,
            "You are an expert Vortex Luau programmer."
        )
        system = f"{system}\n\n{PLATFORM_CONTEXT}"
        if CONFIRMED_CLASSES_CONTEXT:
            system += f"\n\n{CONFIRMED_CLASSES_CONTEXT}"
        if CONFIRMED_DATATYPES_CONTEXT:
            system += f"\n\n{CONFIRMED_DATATYPES_CONTEXT}"

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
            models = (
                [OPENROUTER_DOC_MODEL, OPENROUTER_DOC_MODEL_BACKUP]
                if request.task_type in DOCUMENTATION_TASK_TYPES
                else [OPENROUTER_CODING_MODEL, OPENROUTER_CODING_MODEL_BACKUP]
            )

            if OPENROUTER_API_KEY:
                try:
                    response_text, tokens_used = await call_openrouter_with_fallback(
                        models, system, request.prompt, request.max_tokens, request.temperature
                    )
                except httpx.HTTPStatusError:
                    if not CLOUDFLARE_WORKER_URL:
                        raise
                    logger.warning("All OpenRouter models failed, trying the Cloudflare Worker fallback")
                    response_text, tokens_used = await call_cloudflare_worker(system, request.prompt, request.max_tokens)
            elif CLOUDFLARE_WORKER_URL:
                logger.warning("OPENROUTER_API_KEY isn't set, going straight to the Cloudflare Worker fallback")
                response_text, tokens_used = await call_cloudflare_worker(system, request.prompt, request.max_tokens)
            else:
                raise HTTPException(status_code=503, detail="Neither OPENROUTER_API_KEY nor CLOUDFLARE_WORKER_URL is configured")

            # Safety net for when the model ignores CONFIRMED_CLASSES_CONTEXT/
            # CONFIRMED_DATATYPES_CONTEXT above - confirmed to happen in
            # practice (TextLabel got generated despite the warning). This
            # can't fix the code, but it stops a silent runtime crash from
            # being the user's first sign anything was wrong.
            unconfirmed = find_unconfirmed_apis(response_text) if response_text else []
            if unconfirmed:
                names = ", ".join(f"`{n}`" for n in unconfirmed)
                plural = "s aren't" if len(unconfirmed) > 1 else " isn't"
                response_text = (
                    f"⚠️ **Heads up:** this uses {names}, which{plural} in Vortex's "
                    f"confirmed class/datatype list and may not actually exist. "
                    f"Test before relying on it.\n\n---\n\n{response_text}"
                )

        return GenerateResponse(
            response=response_text,
            task_type=request.task_type,
            tokens_used=tokens_used
        )

    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        logger.error(f"Upstream AI provider error: {e}")
        if status == 429:
            # Real 429, not collapsed to 500 - the frontend has specific
            # rate-limit messaging for this exact status.
            raise HTTPException(status_code=429, detail="All available models are currently rate-limited - try again in a moment.")
        raise HTTPException(status_code=502, detail=f"The AI provider returned an error ({status}).")
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
