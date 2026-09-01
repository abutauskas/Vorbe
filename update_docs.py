"""
Vorbe - Automated docs sync

Keeps vortex_training_data/vortex_qa_pairs.json fresh against
https://github.com/TheHaloDeveloper/vortex-docs without anyone having to do
it by hand. Meant to run on a schedule (see .github/workflows/update-docs.yml)
so the grounding data self-heals even if nobody's actively maintaining it.

What it does, per run:
1. Lists every markdown file in the source repo's guides/reference folders.
2. Compares each file's git blob sha against what we saw last time
   (vortex_training_data/docs_manifest.json) - only changed or new files
   get reprocessed.
3. For each changed file, asks an LLM to extract factual QA-style entries
   in the same paraphrased, attributed style used for the hand-written
   entries already in the corpus - never copies doc text verbatim.
4. Replaces only the entries this script previously generated from that
   file (tracked per-file in the manifest), so hand-curated entries are
   never touched and edits to a doc don't leave stale duplicates behind.
5. Rebuilds vortex_training_data/confirmed_classes.json and
   confirmed_datatypes.json from every class/datatype page's exact
   frontmatter `title:` field - these are the allowlists api_server.py
   uses to stop the model from assuming Vortex has something (Roblox's
   GUI system; BrickColor, which Vortex doesn't have - only Color3)
   that isn't actually documented. Cheap: a raw-file fetch per page, no
   LLM call, so both are rebuilt in full every run regardless of what
   changed.
6. Writes the updated QA pairs and manifest back to disk. The GitHub
   Action wrapping this script commits the result if anything changed,
   which triggers Vercel to redeploy - fully hands-off.

Safe to re-run any time (`python update_docs.py`) - a no-op if nothing in
the source repo changed since the last run.
"""

import json
import logging
import os
import re
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DOCS_REPO = "TheHaloDeveloper/vortex-docs"
DOCS_BRANCH = "main"
DOCS_PREFIXES = ("content/guides/", "content/reference/")

QA_PAIRS_PATH = "vortex_training_data/vortex_qa_pairs.json"
MANIFEST_PATH = "vortex_training_data/docs_manifest.json"
CONFIRMED_CLASSES_PATH = "vortex_training_data/confirmed_classes.json"
CLASSES_PREFIX = "content/reference/classes/"
CONFIRMED_DATATYPES_PATH = "vortex_training_data/confirmed_datatypes.json"
DATATYPES_PREFIX = "content/reference/datatypes/"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_DOC_MODEL", "minimax/minimax-m3:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# GitHub's own token, if this runs in a GitHub Action - raises the API
# rate limit from 60/hr to 5,000/hr. Works fine without it too.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# OpenRouter's free tier is capped at 20 requests/minute and 50/day (1000/day
# once any credits have been purchased on the account). A cold run with an
# empty manifest would otherwise try to process every doc file at once and
# blow through that in one go - cap it and let later runs pick up the rest.
MAX_FILES_PER_RUN = 40
SECONDS_BETWEEN_REQUESTS = 3.5

EXTRACTION_SYSTEM_PROMPT = """You are helping build a factual reference corpus for an AI coding \
assistant. You'll be given one page of documentation for "Vortex", a \
Roblox-like Luau game development platform.

Extract every distinct fact, class, property, method, or concept covered \
on the page as separate entries. For each one, respond with an object \
matching this exact shape:

{"topic": "ShortName", "questions": ["What is X?", "Explain X in Vortex", \
"How do I use X?", "Tell me about X"], "answer": "A paraphrased, factual \
explanation in your own words - never copy sentences verbatim from the \
source text."}

Rules:
- "topic" is the concept's own name (e.g. a class or method name), not a \
generic label.
- "questions" always has exactly those four phrasings, with X replaced by \
the topic.
- "answer" must be entirely reworded in your own words - paraphrase, don't \
quote. Keep it factual and concise (2-5 sentences). Include concrete \
details (property names, types, method signatures) where the source gives \
them.
- Skip anything that's just navigation, a table of contents, or boilerplate \
with no real technical content.
- If the page has no extractable technical content, return an empty array.

Respond with ONLY a JSON array of these objects - no prose, no markdown \
code fences, nothing else."""


def gh_headers():
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def extract_title(content):
    """The page's exact `title:` from its YAML frontmatter - the authoritative
    name for a class (correctly cased, e.g. "SpawnLocation"), far more
    reliable than guessing from a kebab-case filename or mining citation
    text elsewhere."""
    match = re.match(r"^---\s*\n(.*?\n)---", content, re.S)
    if not match:
        return None
    title_match = re.search(r"^title:\s*(.+)$", match.group(1), re.M)
    return title_match.group(1).strip() if title_match else None


def rebuild_titles_list(doc_files, prefix, out_path, label):
    """Full rebuild, every run, from every page under `prefix` currently in
    the docs repo - not just the ones that changed. Cheap (raw fetch, no LLM
    call). Shared by confirmed_classes.json and confirmed_datatypes.json -
    same reliability requirement either way: this becomes an allowlist
    api_server.py leans on to tell the model what's real, so a bad write
    here would make it wrongly refuse things that actually exist."""
    matching_files = [f for f in doc_files if f["path"].startswith(prefix)]
    titles = []
    fetch_failures = 0
    for f in matching_files:
        try:
            content = fetch_raw(f["path"])
        except httpx.HTTPError as e:
            logger.warning(f"  couldn't fetch {f['path']} for the {label} list: {e}")
            fetch_failures += 1
            continue
        title = extract_title(content)
        if title:
            titles.append(title)
        else:
            # Expected and permanent for a method-level sub-page (e.g.
            # cframe.lookat.md) - not a top-level type, so no title to grab.
            # Not a failure worth counting toward the guard below.
            logger.info(f"  no title frontmatter in {f['path']} (likely a method sub-page), skipping")

    # Guard against actual fetch failures (e.g. GitHub having a bad moment
    # mid-run) corrupting a good file - NOT against pages that fetched fine
    # but structurally have no title (that's permanent, not transient, so
    # it shouldn't block a write).
    if matching_files and fetch_failures > 0.2 * len(matching_files):
        logger.error(
            f"{fetch_failures}/{len(matching_files)} {label} pages failed to fetch - "
            f"leaving the existing {out_path} untouched"
        )
        return None

    titles = sorted(set(titles))
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(titles, out, indent=2)
    logger.info(f"{out_path}: {len(titles)} {label}(s)")
    return titles


def rebuild_confirmed_classes(doc_files):
    return rebuild_titles_list(doc_files, CLASSES_PREFIX, CONFIRMED_CLASSES_PATH, "class")


def rebuild_confirmed_datatypes(doc_files):
    return rebuild_titles_list(doc_files, DATATYPES_PREFIX, CONFIRMED_DATATYPES_PATH, "datatype")


def list_doc_files():
    """Every markdown file in the source repo's guides/reference folders,
    with its git blob sha (used to detect changes)."""
    url = f"https://api.github.com/repos/{DOCS_REPO}/git/trees/{DOCS_BRANCH}?recursive=1"
    resp = httpx.get(url, headers=gh_headers(), timeout=30)
    resp.raise_for_status()
    tree = resp.json()["tree"]
    return [
        {"path": item["path"], "sha": item["sha"]}
        for item in tree
        if item["type"] == "blob"
        and item["path"].endswith(".md")
        and item["path"].startswith(DOCS_PREFIXES)
    ]


def fetch_raw(path):
    url = f"https://raw.githubusercontent.com/{DOCS_REPO}/{DOCS_BRANCH}/{path}"
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_entries(path, content):
    """Ask the LLM to turn one doc page into paraphrased QA entries."""
    if len(content.strip()) < 40:
        return []  # not enough real content to bother with

    source_url = f"https://github.com/{DOCS_REPO}/blob/{DOCS_BRANCH}/{path}"
    resp = httpx.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": content[:8000]},
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
        },
        timeout=60,
    )
    if resp.status_code == 429:
        raise RateLimited()
    resp.raise_for_status()

    text = resp.json()["choices"][0]["message"].get("content") or "[]"
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    try:
        raw_entries = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"  couldn't parse model output as JSON, skipping {path}")
        return []

    entries = []
    for e in raw_entries[:20]:  # sane cap in case the model runs long
        if not isinstance(e, dict):
            continue
        topic = str(e.get("topic", "")).strip()
        answer = str(e.get("answer", "")).strip()
        questions = e.get("questions")
        if not topic or not answer or not isinstance(questions, list) or not questions:
            continue
        entries.append({
            "topic": topic,
            "questions": [str(q) for q in questions],
            "answer": f"{answer} Source: {source_url}",
        })
    return entries


class RateLimited(Exception):
    pass


def main():
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY isn't set - can't extract anything.")
        sys.exit(1)

    try:
        with open(QA_PAIRS_PATH, encoding="utf-8") as f:
            qa_pairs = json.load(f)
    except FileNotFoundError:
        qa_pairs = []

    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        manifest = {}

    try:
        doc_files = list_doc_files()
    except httpx.HTTPError as e:
        logger.error(f"Couldn't list docs from GitHub: {e}")
        sys.exit(1)

    # Always rebuilt in full, regardless of what's "changed" below - it's
    # cheap (no LLM calls) and it's the allowlist api_server.py leans on to
    # stop the model from assuming an undocumented class or datatype exists
    # (BrickColor, which real Vortex doesn't have - only Color3 - was a
    # real hallucination caught this way), so it shouldn't silently go
    # stale just because no QA pairs needed updating.
    rebuild_confirmed_classes(doc_files)
    rebuild_confirmed_datatypes(doc_files)

    changed = [
        f for f in doc_files
        if manifest.get(f["path"], {}).get("sha") != f["sha"]
    ]
    logger.info(f"{len(doc_files)} doc files total, {len(changed)} new or changed")

    if not changed:
        logger.info("No QA pairs to update.")
        return

    if len(changed) > MAX_FILES_PER_RUN:
        logger.info(f"Capping this run to {MAX_FILES_PER_RUN} files - the rest will pick up next run")
        changed = changed[:MAX_FILES_PER_RUN]

    processed = 0
    for i, f in enumerate(changed):
        path = f["path"]
        logger.info(f"[{i + 1}/{len(changed)}] {path}")
        try:
            content = fetch_raw(path)
            entries = extract_entries(path, content)
        except RateLimited:
            logger.info("Hit OpenRouter's rate limit - stopping here, the rest will pick up next run")
            break
        except httpx.HTTPError as e:
            logger.warning(f"  failed to fetch/process, skipping: {e}")
            continue

        # Drop only the entries this script previously generated from this
        # exact file - hand-curated entries and entries from other files
        # (even same-named topics) are never touched.
        old_topics = set(manifest.get(path, {}).get("topics", []))
        if old_topics:
            qa_pairs = [q for q in qa_pairs if not (q.get("topic") in old_topics and path in _entry_sources(q))]

        qa_pairs.extend(entries)
        manifest[path] = {"sha": f["sha"], "topics": [e["topic"] for e in entries]}
        processed += 1

        if i < len(changed) - 1:
            time.sleep(SECONDS_BETWEEN_REQUESTS)

    with open(QA_PAIRS_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Done. Processed {processed} files, {len(qa_pairs)} total QA entries.")


def _entry_sources(entry):
    """Best-effort: pull the "Source: <url>" tail an entry's answer carries,
    so we can tell whether it came from a given doc file before dropping it."""
    return entry.get("answer", "")


if __name__ == "__main__":
    main()
