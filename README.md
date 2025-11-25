🔐 AI Text Watermarking
Invisible, Contextual, and Neural Watermarking for AI-Generated Text

This repository contains a complete implementation of invisible watermarking techniques for AI-generated text, including both Unicode-based watermarking and a more advanced contextual neural watermarking system using PyTorch and HuggingFace Transformers.

The goal of this project is to embed a watermark during generation so that text remains untouched visually but can be reliably detected later — even after copy/paste or light editing.

📌 Features
✅ 1. Invisible Unicode Watermarking (Basic Method)

Uses zero-width Unicode characters

Fully invisible to users

Survives copy/paste into plain editors

Good baseline watermarking method

✅ 2. Contextual Neural Watermarking (Advanced Method)

Built using your Python implementation:

EnhancedHashNet: neural context-based hashing

Green/Red token lists per decoding step

Logit manipulation using custom LogitsProcessor

Dynamic watermark embedding

Statistical detection using Z-score and p-values

Visualization of watermark patterns

This method provides higher security, robustness, and stealth.

🧠 How It Works
🔹 Watermark Encoding

During generation:

The model takes previous tokens (context window)

A neural hash network generates a seed

Vocabulary is permuted using this seed

“Green tokens” are boosted, “red tokens” are penalized

Model is more likely to choose green tokens → invisible pattern

🔹 Watermark Detection

Given a text:

Detector reconstructs the seed per position

Rebuilds green-lists exactly

Checks how often text chooses green tokens

Computes:

z-score

p-value

confidence
