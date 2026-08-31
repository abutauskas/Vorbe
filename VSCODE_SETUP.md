# Adding Vorbe to GitHub via VSCode

Here's the easiest way to add all the files:

## Step 1: Download Everything

All files are in `/mnt/user-data/outputs/`

Download these specific folders/files:
- `README.md`
- `LICENSE`
- `.gitignore`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `requirements.txt`
- `requirements-vercel.txt`
- `Dockerfile`
- `vercel.json`
- `api_server.py`
- `app_hf_spaces.py`
- `finetune_deepseek_coder.py`
- `finetune_codeqwen.py`
- `prepare_vortex_training_data.py`
- `index.html`
- `vortex_training_data/` (entire folder)
- `.github/ISSUE_TEMPLATE/` (entire folder)

Plus optionally:
- `VERCEL_QUICKSTART.md`
- `DEPLOYMENT_GUIDE.md`

(The other .md files are reference docs - you don't need them in the repo)

## Step 2: Open VSCode

Open your Vorbe folder in VSCode.

## Step 3: Create Folder Structure

In VSCode, create these folders:
- `.github/ISSUE_TEMPLATE/`
- `vortex_training_data/`

## Step 4: Add Files

Drag and drop the files into VSCode, or:
1. Right-click folder → New File
2. Paste content
3. Save

Key files to add:

### Root Level (add to repo root)
```
README.md
LICENSE
.gitignore
CONTRIBUTING.md
CODE_OF_CONDUCT.md
requirements.txt
requirements-vercel.txt
Dockerfile
vercel.json
```

### Python Scripts (add to repo root)
```
api_server.py
app_hf_spaces.py
finetune_deepseek_coder.py
finetune_codeqwen.py
prepare_vortex_training_data.py
```

### Web (add to repo root)
```
index.html
```

### Training Data (add vortex_training_data/ folder)
Inside `vortex_training_data/`:
```
vortex_conversations.jsonl
vortex_qa_pairs.json
vortex_code_examples.json
system_prompts.json
```

### GitHub Templates (add .github/ISSUE_TEMPLATE/ folder)
Inside `.github/ISSUE_TEMPLATE/`:
```
bug_report.md
feature_request.md
```

## Step 5: Commit and Push

```bash
git add .
git commit -m "Initial commit: Vorbe - AI Coding Assistant for Vortex"
git push origin main
```

## Step 6: Verify on GitHub

Go to https://github.com/abutauskas/Vorbe and make sure all files are there.

## Step 7: Update Vercel

Go to Vercel dashboard for your vorber.vercel.app project:
1. Settings → GitHub
2. Redeploy (or it auto-deploys when you push)

That's it. You're live.

---

**Questions about a file?** Check `FILES_TO_ADD.txt` for what everything does.
