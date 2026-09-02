# Contributing to Vorbe

Thanks for thinking about contributing. This project only gets better when people like you help out.

## Ways to help

**Found a bug?** Open an issue and describe what went wrong. Include steps to reproduce if you can.

**Have an idea?** Start a discussion about it first. We can talk through whether it's a good fit for Vorbe.

**Want to fix something?** Awesome. Pick an issue, fork the repo, make your changes, and submit a PR.

**Good at writing?** Docs can always be better. Fix typos, clarify confusing sections, add examples.

**Want to help others?** Answer questions in discussions, help debug issues, review PRs.

**Want to spread the word?** Write about Vorbe, make a tutorial, post on social media. Share it with the Vortex community.

---

## Getting your hands dirty

**1. Fork and clone**
```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/vorbe.git
cd vorbe
```

**2. Create a branch**
```bash
git checkout -b fix/your-fix-name
# or: git checkout -b feature/your-feature-name
```

**3. Make your changes**
```bash
# Edit files, test locally
python app_hf_spaces.py
```

**4. Commit and push**
```bash
git add .
git commit -m "What you changed and why"
git push origin your-branch-name
```

**5. Open a PR**
Go back to GitHub and submit a pull request. Describe what you changed.

---

## Before you submit a PR

- Code is clean and makes sense
- You tested it (run the app, make sure it works)
- Documentation is updated if needed
- Commit message is clear about what you did
- No random changes that aren't related to the fix/feature

## PR title format

Keep it simple. Examples:
- `Fix: Model loading timeout` 
- `Add: New task mode for code optimization`
- `Docs: Update deployment guide`
- `Improve: API response time`

## What to include in your PR

Describe what you changed and why. If it fixes an issue, mention it. If you changed something that users will see, say so. If you tested it, say how.

---

## Development setup

### Requirements
- Python 3.10+
- Git
- A free [Groq API key](https://console.groq.com) (no phone or card needed)
- Node.js, for the `vercel` CLI

### Setup
```bash
# Clone repo
git clone https://github.com/abutauskas/Vorbe.git
cd vorbe

# Install dependencies
pip install -r requirements.txt
npm install -g vercel

# Add your Groq key
cp .env.example .env
# then open .env and paste it in

# Test locally
vercel dev
```

---

## Code style

### Python
- Follow PEP 8
- Use type hints
- Add docstrings
- 88-character line length (black format)

```python
def generate_response(prompt: str, task: str) -> str:
    """
    Generate AI response for given prompt.
    
    Args:
        prompt: User input text
        task: Task type (script_generation, bug_fixing, etc)
    
    Returns:
        Generated response string
    """
    # Implementation
```

### Comments
- Explain WHY, not WHAT
- Keep comments updated
- Remove debug code

### Commits
- One logical change per commit
- Clear commit messages
- Reference issues: "Fixes #123"

---

## Testing

Make sure your changes actually work:

```bash
vercel dev
```

Then try using it. Does it start? Does the web interface load? Does it generate responses? Good. If you're working on the self-hosted fine-tune path instead, test that with `python app_hf_spaces.py` (needs `requirements-finetune.txt`).

If you changed something in the API, test that too. Just make sure nothing broke.

---

## Project structure

```
vorbe/
├── api_server.py              # FastAPI backend (calls Groq's API)
├── main.py                    # Thin entrypoint Vercel auto-detects (re-exports api_server's app)
├── public/index.html          # Landing page, served as a static asset
├── public/app.html            # Chat UI, served as a static asset
├── public/res/                # Logo, icons, and other image assets
├── app_hf_spaces.py           # Gradio interface (optional self-hosted fine-tune path)
├── finetune_deepseek_coder.py # Training script (optional self-hosted fine-tune path)
├── Dockerfile                 # Docker config for the self-hosted path
├── vortex_training_data/      # Training data
├── requirements.txt           # Deps for the deployed Groq-backed backend
├── requirements-finetune.txt  # Deps for the optional self-hosted fine-tune path
├── .env.example                # Copy to .env and fill in your Groq key
├── vercel.json                # Vercel function config
├── LICENSE                    # MIT License
├── README.md                  # Project overview
└── CONTRIBUTING.md            # This file
```

---

## Where we could use help

**Big things:**
- Make the AI better at answering questions (train on more data, try new models)
- Add new task modes (whatever the Vortex community is asking for)
- Make it faster (especially for cloud deployments)
- Better error messages

**Medium things:**
- Improve the web interface
- Write better docs and tutorials
- Add more examples

**Small things:**
- Fix typos
- Clean up code
- Update comments
- Improve documentation

---

## Questions?

Setup issues are covered in README.md, and this file covers how to contribute. Otherwise: open an issue for a bug, or start a discussion for anything else, including if you're just stuck.

---

## How we treat each other

**Be cool.** Treat people with respect. Everyone's here trying to make something better.

**Be helpful.** If you see someone confused, help them. If you're giving feedback, make it constructive. "Hey, could we do this differently?" not "This is wrong."

**Welcome everyone.** New contributors bring ideas and perspectives a project wouldn't get otherwise.

**Share knowledge.** Explain your thinking. Ask questions if something's unclear. Help others learn.

---

## Recognition

We'll credit your contributions in the README and release notes, and mention you when we talk about Vorbe.

---

## License

By contributing, you agree your code will be licensed under MIT License.

---

## Links

- [GitHub Issues](https://github.com/abutauskas/Vorbe/issues) - Bug reports and feature ideas
- [GitHub Discussions](https://github.com/abutauskas/Vorbe/discussions) - Chat about features and ideas
- [README](./README.md) - Project overview
- [Setup Guide](./OPEN_SOURCE_SETUP.md) - How to get started

---

Thanks for helping make Vorbe better - even a small fix counts.
