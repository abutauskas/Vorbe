# Deploying Vorbe to Vercel

Vorbe runs entirely on Vercel: the landing page (`public/index.html`) and the chat UI (`public/app.html`) are served as static assets, and the backend (`api_server.py`, exposed via `api/index.py`) runs as a single Vercel Python function. Text generation calls OpenRouter's free-tier models; image attachments call Groq's vision model. Neither loads a model locally, so there's no separate GPU host to set up.

## 1. Get free API keys

1. **OpenRouter (required, text generation)** - go to [openrouter.ai/keys](https://openrouter.ai/keys), sign up, create a key.
2. **Groq (optional, image attachments only)** - go to [console.groq.com](https://console.groq.com), sign up (email or Google account, no phone or card needed), create a key. Skip this if you don't need image uploads.

## 2. Deploy

1. Go to [github.com/abutauskas/Vorbe](https://github.com/abutauskas/Vorbe)
2. Click "Deploy with Vercel" and import the repo.
3. Before the first deploy, add environment variables:
   - `OPENROUTER_API_KEY` = the key from step 1
   - `GROQ_API_KEY` = your Groq key, if you added one (image attachments won't work without it, everything else still will)
   - Optional: `API_AUTH_TOKEN` = a random string of your choosing, if you want to gate `/generate` against random bots hitting the endpoint directly
   - Optional: `OPENROUTER_CODING_MODEL` = a different model for script generation/bug fixing/security review (default `poolside/laguna-s-2.1:free`)
   - Optional: `OPENROUTER_CODING_MODEL_BACKUP` = falls back to this if the coding model hits a rate limit or errors (default `cohere/north-mini-code:free`)
   - Optional: `OPENROUTER_DOC_MODEL` = a different model for documentation questions (default `minimax/minimax-m3:free`)
   - Optional: `OPENROUTER_DOC_MODEL_BACKUP` = falls back to this if the doc model hits a rate limit or errors (default `minimax/minimax-m2.7:free`)
   - Optional: `GROQ_VISION_MODEL` = a different vision-capable model, used only for requests with an attached image (default `qwen/qwen3.8-27b`)
   - Optional: `CLOUDFLARE_WORKER_URL` / `CLOUDFLARE_WORKER_TOKEN` = an absolute last-resort fallback, only reached if every OpenRouter model above has failed. Runs on Cloudflare's own infrastructure (independent of OpenRouter/Groq) via its free Workers AI tier. Set both up with `./scripts/setup-cloudflare-fallback.sh` - see that script rather than configuring this by hand.
4. Deploy.

Manual deploy from the terminal works too:

```bash
npm install -g vercel
vercel login
vercel env add OPENROUTER_API_KEY
vercel env add GROQ_API_KEY
vercel --prod
```

## 3. If you set API_AUTH_TOKEN

Open `public/app.html`, find:

```js
const API_TOKEN = ""; // set to match API_AUTH_TOKEN if you configured one
```

Set it to the same value as `API_AUTH_TOKEN`, then redeploy.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and paste in your OpenRouter key (and Groq key, if you added one), then:

```bash
vercel dev
```

`vercel dev` serves the static frontend and the Python function together on one local port, so `public/app.html`'s relative `fetch("/generate", ...)` calls work locally exactly as they do in production - no separate backend URL to configure. The landing page is at `/`, the chat UI at `/app` (clean URLs are on via `vercel.json`'s `cleanUrls`, so the `.html` extension doesn't show).

## Verification

1. `curl https://<your-deployment>.vercel.app/health` → `{"status": "healthy", "backend": "openrouter+groq", "configured": true}`. If `configured` is `false`, `OPENROUTER_API_KEY` and/or `GROQ_API_KEY` aren't set in Vercel's environment variables.
2. `curl -X POST https://<your-deployment>.vercel.app/generate -H "Content-Type: application/json" -d '{"prompt": "spawn a red part", "task_type": "script_generation"}'` → a 200 response with a real `response` field.
3. Open the deployed site, send a real chat message, confirm a response renders.

## Troubleshooting

**`/health` shows `"configured": false`** - `OPENROUTER_API_KEY` isn't set (breaks all text generation) or `GROQ_API_KEY` isn't set (breaks image attachments only). Add whichever's missing under Vercel's Project Settings → Environment Variables, then redeploy.

**`/generate` returns a model-related error** - free-tier model availability on OpenRouter/Groq changes over time; a model may have been renamed, deprecated, or hit its free-tier rate limit (OpenRouter's free models are capped at 20 requests/minute, 50/day per account unless you've purchased credits, which raises the daily cap to 1000). Each text model already has a backup it falls back to on a 429 or 5xx, so a single retired/rate-limited model shouldn't take the whole thing down - but if you're seeing this often, set `OPENROUTER_CODING_MODEL` / `OPENROUTER_CODING_MODEL_BACKUP` / `OPENROUTER_DOC_MODEL` / `OPENROUTER_DOC_MODEL_BACKUP` / `GROQ_VISION_MODEL` to current model names.

**Frontend loads but chat doesn't respond** - check the browser console. A failed request to `/generate` on the same origin usually means the function errored (check Vercel's function logs) rather than a networking/CORS issue.

**Deploy fails on the Python function** - `requirements.txt` at the repo root is what Vercel installs for `api/index.py`. Keep it limited to what `api_server.py` actually imports; the heavier ML stack in `requirements-finetune.txt` is for the separate self-hosting path and isn't needed here.
