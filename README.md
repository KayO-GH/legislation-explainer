---
title: Legislation Explainer
emoji: 📚
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.17.3
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Simplify complex legislation that affects you!
tags:
  - track:backyard
  - sponsor:openai
  - achievement:fieldnotes
models:
  - Qwen/Qwen3-14B
  - sentence-transformers/all-MiniLM-L6-v2
---

# Legislation Explainer

Legislation Explainer is a Gradio app for helping regular people understand public-interest legislation and how it affects them.

## Links

- Hugging Face Space: https://huggingface.co/spaces/build-small-hackathon/legislation-explainer
- Live app: https://build-small-hackathon-legislation-explainer.hf.space/
- GitHub repo: https://github.com/KayO-GH/legislation-explainer
- Demo video: https://www.loom.com/share/ed7b682ab6774d5499fca4c06c290bfc
- Social post: https://www.linkedin.com/posts/kwadwo-agyapon-ntra_rag-huggingface-gradio-activity-7470763368237244416-JseD/
- Blog post: https://kayo-gh.github.io/building-legislation-explainer/

## Motivation

In May 2026, the Ghana tech community on X/Twitter reacted strongly to the contents of a draft National Information Technology Authority bill proposed by the Ministry of Communication, Digital Technology and Innovations. The draft bill, and 14 others, had been public for months, but the legal language made it hard for many people in the developer ecosystem to evaluate quickly. When people noticed proposals around licensing fees, revenue-based charges, and restrictions affecting company formation, the discussion became urgent.

This project is my attempt to help civic groups, technologists, journalists, students, entrepreneurs, policy watchers, and regular people understand legislation as soon as they come across it.

It is created for the Hugging Face Build Small Hackathon under the `Backyard AI` track: a practical, small-model assistant for people who need to understand a real bill quickly without reading every clause first.

## Hackathon Fit

- Track: `Backyard AI`
- Real user: Ghanaian citizens and digital-policy stakeholders who need a clearer view of a bill's practical effects.
- Small-model constraint: each model used by the app is individually below the hackathon's `<= 32B` cap.
- Required surface: Gradio app, ready for Hugging Face Spaces through `app.py`.

## Try The Demo

1. Open the live app.
2. Paste this public NITA bill URL into `Document URL`: `https://moc.gov.gh/wp-content/uploads/2023/03/NITA-NATIONAL-INFORMATION-TECHNOLOGY-AUTHORITY-BILL_-10-07-25.pdf`.
3. Click `Run analysis` to load the precomputed example analysis.
4. Ask a follow-up question such as `What should startup founders pay attention to first?`.

_**Note:** The ministry's website does not have consistent uptime. In the event that the file cannot be accessed from the site, you can [download it from here](https://drive.google.com/file/d/1P-cvgp-bX42QU2zijRPpmnE-MNPgXuGD/view?usp=sharing). Aslo feel free to experiment with any relevant documents you have. Nothing is saved beyond a session._

## What It Does

- Ingests PDF, DOCX, TXT, Markdown, or document URLs.
- Produces a structured policy brief:
  - executive summary
  - bill summary
  - implementation implications
  - critique and recommendations
  - SWOT analysis
- Supports follow-up Q&A over the generated analysis.
- Offers deeper full-document answering when the summary is not enough.
- Includes bundled example-bill assets for faster demos.

## Model And Provider Notes

The hackathon-safe default is Qwen3 14B through the Hugging Face router, with a small embedding model for retrieval.

- Default generator: `Qwen/Qwen3-14B`
- Embeddings and chunk retrieval: `sentence-transformers/all-MiniLM-L6-v2`
- Default credential path: `HF_TOKEN`
- Default provider: `qwen`
- Parameter disclosure: `Qwen3-14B` is under 32B, and `all-MiniLM-L6-v2` is far below the cap at ~22.7M.

_A smaller Qwen model could have been used successfully, but considering the real-world importance, a model with a higher capactity for reasoning is desired._

## Local Run

```bash
# activate virtual env
pip install -r requirements.txt   # alternatively, uv sync to install from pyproject.toml
source .venv/bin/activate
gradio app.py                     # alternatively, uv run gradio app.py
```

Set these environment variables for the default Qwen path:

- `HF_TOKEN`

## Space Deployment

This directory is structured as a Hugging Face Space:

- `app.py` exposes `demo` and `app`.
- `requirements.txt` lists runtime dependencies.
- `assets/` contains bundled example documents and precomputed artifacts.
- `services/` contains ingestion, provider, and RAG logic.

Upload the contents of `legislation-explainer/` to a Gradio Space under the hackathon organization.
