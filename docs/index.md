---
layout: home
title: Winter
titleTemplate: Chat with your videos

hero:
  name: "Winter"
  text: "Chat with your videos"
  tagline: "Multimodal retrieval over speech and keyframes so every answer is cited by timestamp."
  image:
    src: /hero.png
    alt: "Winter product screenshot"
  actions:
  - theme: brand
    text: Get started
    link: /getting-started
  - theme: alt
    text: Guide
    link: /guide
  - theme: alt
    text: GitHub
    link: https://github.com/mediuswareltd/winter

features:
  - title: "Any video source"
    details: "Drag-and-drop upload (mp4, mov, mkv, webm, mp3, wav\u2026) or paste a YouTube URL"
  - title: "Timestamped transcripts"
    details: "`faster-whisper` speech-to-text with voice-activity filtering; runs on CPU, faster on a GPU, 90+ languages"
  - title: "Visual understanding"
    details: "Scene-change keyframe detection (OpenCV) plus CLIP image embeddings, so text queries find matching frames"
  - title: "Hybrid retrieval"
    details: "Dense embeddings (BGE) and BM25 keyword search, merged with Reciprocal Rank Fusion"
  - title: "Multimodal answers"
    details: "Transcript excerpts and the matching keyframes go to Claude, which answers with inline `[n]` citations"
  - title: "Clickable citations"
    details: "Each citation is a timestamp chip; clicking it seeks the player. Source cards show the frame and excerpt"
---

<div class="vp-doc" style="max-width: 1152px; margin: 48px auto 0; padding: 0 24px 64px">

## What you get

Multimodal retrieval over speech and keyframes so every answer is cited by timestamp.

This documentation is the same material as the repository README, expanded for reading in the browser: install steps, how the pieces fit together, and how to run it yourself.

## Stack

`Python` · `Whisper` · `CLIP` · `Claude`

## Next

1. [Getting started](/getting-started): install and run locally
2. [Guide](/guide): features, architecture, and usage notes from the project README
3. [About](/about): who maintains it and how to contribute

</div>
