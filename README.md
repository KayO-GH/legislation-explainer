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
models:
  - nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  - nvidia/llama-nemotron-embed-vl-1b-v2
  - nvidia/llama-nemotron-rerank-1b-v2
---

# Legislation Explainer

Legislation Explainer is a Gradio app for helping regular people understand public-interest legislation and how it affects them.

## Motivation

In May of 2026, there was a huge public outcry online through the Ghana 🇬🇭 X (formerly twitter) community over the contents of a draft bill (the NITA bill) proposed by the nation's Ministry of Communication, Digital Technology and Innovations. The draft bill, and 14 others, had been around for well over 6 months, but being shrouded in legalese, not many regualr folks in the developer ecosystem had taken note of it. When we found out that it contained proposals for exorbitant licensing fees and taxes on revenue (yes, revenue, not profit) of startups, and restrictions on co-founding tech companies with foreign nationals, everyone panicked and things got really tense between the tech community and the ministry.

This project is my attempt tp help civic groups, technologists, journalists, students, entrepreneurs, policy watchers, and regualr people understand legislation as soon as they come across it.

It is created for the Hugging Face Build Small Hackathon under the `Backyard AI` track: a practical, small-model assistant for people who need to understand a real bill quickly without reading every clause first.

## Hackathon Fit

- Track: `Backyard AI`
- Real user: Ghanaian citizens and digital-policy stakeholders who need a clearer view of a bill's practical effects.
- Small-model constraint: default generation uses `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, with Nemotron retriever and reranker models for grounded QA.
- Required surface: Gradio app, ready for Hugging Face Spaces through `app.py`.
- GitHub repo: https://github.com/KayO-GH/legislation-explainer

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

The default QA stack is now Nemotron-first:

- Generator: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- Retriever: `nvidia/llama-nemotron-embed-vl-1b-v2`
- Reranker: `nvidia/llama-nemotron-rerank-1b-v2`

The generator uses an OpenAI-compatible endpoint, so it can run through the Hugging Face router when available or through a Modal-hosted / NIM-style deployment when you want a dedicated path. The retriever also uses an OpenAI-compatible embeddings endpoint. The reranker uses a separate HTTP endpoint so it can be pointed at a hosted Nemotron reranking service.

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

Set these environment variables for the default Nemotron stack:

- `NEMOTRON_API_KEY`
- `NEMOTRON_BASE_URL` optional, defaults to `https://router.huggingface.co/v1`
- `NEMOTRON_RETRIEVER_API_KEY` optional, defaults to `NEMOTRON_API_KEY`
- `NEMOTRON_RETRIEVER_BASE_URL` optional, defaults to `NEMOTRON_BASE_URL`
- `NEMOTRON_RERANKER_API_KEY` optional, defaults to `NEMOTRON_API_KEY`
- `NEMOTRON_RERANKER_URL` optional but recommended for the dedicated reranker path

`HF_TOKEN` still works as a fallback credential for Hugging Face-routed Nemotron endpoints.

## Space Deployment

This directory is structured as a Hugging Face Space:

- `app.py` exposes `demo` and `app`.
- `requirements.txt` lists runtime dependencies.
- `assets/` contains bundled example documents and precomputed artifacts.
- `services/` contains ingestion, provider, and RAG logic.

Upload the contents of `legislation-explainer/` to a Gradio Space under the hackathon organization.
