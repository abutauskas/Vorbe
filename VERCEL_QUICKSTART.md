# Deploying Vorbe to Vercel

Vorbe runs entirely on Vercel: the landing page (`public/index.html`) and the chat UI (`public/app.html`) are served as static assets, and the backend (`api_server.py`, exposed at `main.py`) runs as a single Vercel Python function. The backend calls Groq's hosted API for inference rather than loading a model itself, so there's no separate GPU host to set up.

## 1. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com) and sign up (email or Google account, no phone or card needed).
2. Create an API key from the console.

## 2. Deploy

1. Go to [github.com/abutauskas/Vorbe](https://github.com/abutauskas/Vorbe)
2. Click "Deploy with Vercel" and import the repo.
3. Before the first deploy, add an environment variable:
   - `GROQ_API_KEY` = the key from step 1
   - Optional: `API_AUTH_TOKEN` = a random string of your choosing, if you want to gate `/generate` against random bots hitting the endpoint directly
   - Optional: `GROQ_MODEL` = a different model name if you don't want the default (see Troubleshooting)
   - Optional: `GROQ_VISION_MODEL` = a different vision-capable model, used only for requests with an attached image (default `qwen/qwen3.8-27b`)
4. Deploy.

Manual deploy from the terminal works too:

```bash
npm install -g vercel
vercel login
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

Open `.env` and paste in your Groq key, then:

```bash
vercel dev
```

`vercel dev` serves the static frontend and the Python function together on one local port, so `public/app.html`'s relative `fetch("/generate", ...)` calls work locally exactly as they do in production - no separate backend URL to configure. The landing page is at `/`, the chat UI at `/app` (clean URLs are on via `vercel.json`'s `cleanUrls`, so the `.html` extension doesn't show).

## Verification

1. `curl https://<your-deployment>.vercel.app/health` → `{"status": "healthy", "backend": "groq", "configured": true}`. If `configured` is `false`, `GROQ_API_KEY` isn't set in Vercel's environment variables.
2. `curl -X POST https://<your-deployment>.vercel.app/generate -H "Content-Type: application/json" -d '{"prompt": "spawn a red part", "task_type": "script_generation"}'` → a 200 response with a real `response` field.
3. Open the deployed site, send a real chat message, confirm a response renders.

## Troubleshooting

**`/health` shows `"configured": false`** - `GROQ_API_KEY` isn't set. Add it under Vercel's Project Settings → Environment Variables, then redeploy.

**`/generate` returns a model-related error** - Groq's model lineup changes over time; the default text model (`openai/gpt-oss-120b`) or vision model (`qwen/qwen3.8-27b`) may have been renamed or deprecated. Check [console.groq.com](https://console.groq.com) for current model names and set `GROQ_MODEL` / `GROQ_VISION_MODEL` accordingly.

**Frontend loads but chat doesn't respond** - check the browser console. A failed request to `/generate` on the same origin usually means the function errored (check Vercel's function logs) rather than a networking/CORS issue.

**Deploy fails on the Python function** - `requirements.txt` at the repo root is what Vercel installs for `main.py`. Keep it limited to what `api_server.py` actually imports; the heavier ML stack in `requirements-finetune.txt` is for the separate self-hosting path and isn't needed here.
