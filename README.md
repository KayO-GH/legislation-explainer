---
title: Legislation Explainer
emoji: 📚
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Simplify complex legislation that affects you!
models:
  - Qwen/Qwen3-32B
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
- Small-model constraint: default provider is `Qwen/Qwen3-32B:cheapest` through the Hugging Face router, staying within the hackathon's `<= 32B` model cap. I could go lower than 32B, but since this deals with legal documents I want the best cognition possible.
- Required surface: Gradio app, ready for Hugging Face Spaces through `app.py`.
<<<<<<< HEAD
- Submission assets still needed: Space link, short demo video, and social post.
=======
- GitHub repo: https://github.com/KayO-GH/legislation-explainer
>>>>>>> hf

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

The hackathon-safe default is Qwen3 32B through the Hugging Face router. Bring-your-own provider support is a proposed future expansion and is currently commented out so the hackathon build stays focused on one documented `<= 32B` model path.

For future expansion, we will have a bring your own provider setting allowing users to connect to other models eg. from OpenAI, ANthropic, etc. if they so wish.

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

Set `HF_TOKEN` in your environment for the default Qwen provider.

## Space Deployment

This directory is structured as a Hugging Face Space:

- `app.py` exposes `demo` and `app`.
- `requirements.txt` lists runtime dependencies.
- `assets/` contains bundled example documents and precomputed artifacts.
- `services/` contains ingestion, provider, and RAG logic.

Upload the contents of `legislation-explainer/` to a Gradio Space under the hackathon organization.
