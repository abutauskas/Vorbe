# Vorbe

![Vorbe](public/res/GithubBannerVorbe.png)

An AI assistant built for Vortex developers. It helps you write code faster.

Need a script? Describe what you want and get working code. Have a bug? Paste it and get help fixing it. Want to learn an API? Ask and get examples with explanations. Need to generate a procedural world? Get a script for that too.

That's what Vorbe does.

## Why it exists

Vortex is awesome for game dev, but writing scripts takes time. We built Vorbe to speed that up. Instead of googling or trial-and-error, you describe what you need and get working code.

It's free, open source, and built by developers who actually use Vortex.

## What you can do with it

**Generate scripts** - "Create a script that spawns a red part at position (0, 10, 0)" → working code  
**Fix bugs** - Paste broken code, get explanations and fixes  
**Learn APIs** - Ask about RemoteEvents, TweenService, whatever → detailed explanations with examples  
**Generate worlds** - "Create a procedural dungeon" → generation script  
**Review code** - Check your scripts for security issues  

## Getting started

### Fastest way (2 minutes)

```bash
git clone https://github.com/abutauskas/Vorbe.git
cd vorbe
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and paste in a free key from [openrouter.ai/keys](https://openrouter.ai/keys) (text generation). Add a [console.groq.com](https://console.groq.com) key too if you want image attachments to work. Then:

```bash
vercel dev
```

Then open the local URL `vercel dev` prints and click through to the chat, or go straight to `/app`.

### Deploy to the internet

Vorbe deploys entirely to Vercel for free - frontend and backend both, powered by free-tier LLM APIs. See [VERCEL_QUICKSTART.md](./VERCEL_QUICKSTART.md).

## How it works

Vorbe sends your prompt to a free OpenRouter model, picked by the task you selected - a coding-focused model for script generation, bug fixing, and security review, a documentation-focused model for API/concept questions - along with a system prompt built from the Vortex training data in `vortex_training_data/`. Image attachments go through Groq's vision model instead. The web interface is a simple chat UI; the backend is a thin FastAPI layer that talks to both APIs. Everything is open source, so you can see exactly how it works.

Want the assistant running on an actual fine-tuned model instead of a system prompt? `finetune_deepseek_coder.py` and `finetune_codeqwen.py` are still here for that - see "Self-hosting the fine-tuned model" below.

## System requirements

**Default (API-backed):** Just Python and a free OpenRouter API key (plus optionally Groq for image attachments) - no GPU, no heavy RAM, no model download.  
**Self-hosting the fine-tuned model instead:** Python 3.10, 16GB RAM, GPU with 8GB VRAM (see `requirements-finetune.txt` and `app_hf_spaces.py`).  
**Cloud:** Deploys free to Vercel (see [VERCEL_QUICKSTART.md](./VERCEL_QUICKSTART.md))  

## What's included

- **Training data** - 548 conversation pairs from Vortex docs, including fact-extracted entries from [TheHaloDeveloper/vortex-docs](https://github.com/TheHaloDeveloper/vortex-docs)
- **Scripts** - Fine-tuning, web interface, everything
- **Docs** - Setup guides, deployment guides, contribution guides
- **Open source** - MIT License, do what you want with it

## Keeping the docs fresh automatically

A scheduled GitHub Action (`.github/workflows/update-docs.yml`) runs `update_docs.py` weekly: it checks [TheHaloDeveloper/vortex-docs](https://github.com/TheHaloDeveloper/vortex-docs) for changed pages, asks an LLM to extract paraphrased, attributed QA entries from anything new, and commits the result straight to `vortex_training_data/vortex_qa_pairs.json` - which Vercel then redeploys automatically. No one has to do this by hand. It needs an `OPENROUTER_API_KEY` repo secret (Settings → Secrets and variables → Actions) to run; it's a no-op commit-free run if nothing in the source docs changed.

## Contributing

Found a bug? Have an idea? Want to improve something?

Read [CONTRIBUTING.md](./CONTRIBUTING.md) - it's actually pretty short and explains how to help.

We'd genuinely love contributions. Whether it's fixing docs, improving the code, adding examples, or sharing Vorbe with others - it all helps.

## Community

This is for the Vortex community. We built it open source because tools should be free and everyone should be able to help make them better.

Have questions? Open a GitHub issue or start a discussion. Something not working? Let us know.

## Deployment options

**Local** - Run on your machine via `vercel dev`  
**Vercel** - Free cloud deployment, share with others  
**Self-hosted fine-tune** - Use Docker + `app_hf_spaces.py` if you want the actual fine-tuned model instead of Groq  

Each one has a guide. Pick whatever fits your needs.

## License

MIT License. That means you can use it however you want - commercially, personally, whatever. Just keep the license in your code.

## Thanks

Built on [OpenRouter](https://openrouter.ai/), [Groq](https://groq.com/), [FastAPI](https://fastapi.tiangolo.com/), and [Vercel](https://vercel.com/). The optional self-hosted path also uses [DeepSeek-Coder](https://github.com/deepseek-ai/deepseek-coder) and [Transformers](https://huggingface.co/docs/transformers).

For Vortex - https://playvortex.io

## Let's build something

If you find Vorbe useful, star this repo. If you have ideas, open an issue. If you want to contribute, check out CONTRIBUTING.md.

And if you build something cool with Vorbe, let us know. We'd love to hear about it.

---

Questions? Open an issue. Want to chat? Start a discussion. Have a bug to report? You know where to go.

Let's make Vortex game dev faster and more fun.
