# 🔐 AI Text Watermarking

Invisible, Contextual, and Neural Watermarking for AI-Generated Text

This repository contains a complete implementation of invisible watermarking techniques for AI-generated text, including both Unicode-based watermarking and a more advanced contextual neural watermarking system using PyTorch and HuggingFace Transformers.

The goal of this project is to embed a watermark during generation so that text remains untouched visually but can be reliably detected later — even after copy/paste or light editing.

---

## 📌 Features

### ✅ 1. Invisible Unicode Watermarking (Basic Method)
- Uses zero-width Unicode characters
- Fully invisible to users
- Survives copy/paste into plain editors
- Good baseline watermarking method

### ✅ 2. Contextual Neural Watermarking (Advanced Method)
Built using state-of-the-art techniques:
- **EnhancedHashNet**: Neural context-based hashing
- **Green/Red token lists** per decoding step
- **Logit manipulation** using custom LogitsProcessor
- **Dynamic watermark embedding**
- **Statistical detection** using Z-score and p-values
- **Visualization** of watermark patterns

This method provides higher security, robustness, and stealth.

---

## 🧠 How It Works

### 🔹 Watermark Encoding

During text generation:

1. **Context Analysis**: The model takes previous tokens (context window)
2. **Neural Hashing**: A neural hash network generates a unique seed based on context
3. **Vocabulary Permutation**: Vocabulary is permuted using this seed
4. **Token Biasing**: "Green tokens" are boosted, "red tokens" are penalized
5. **Natural Selection**: Model is more likely to choose green tokens → creates invisible pattern

The watermark is embedded seamlessly without affecting text quality or fluency.

### 🔹 Watermark Detection

Given a text to verify:

1. **Seed Reconstruction**: Detector reconstructs the seed at each position
2. **Green-list Rebuilding**: Rebuilds green-lists exactly as during generation
3. **Pattern Matching**: Checks how often text chooses green tokens
4. **Statistical Analysis**: Computes:
   - **z-score**: Measures deviation from random selection
   - **p-value**: Statistical significance of watermark presence
   - **confidence**: Overall detection confidence score

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- HuggingFace Transformers
- NumPy, SciPy

### Install Dependencies

```bash
pip install torch transformers numpy scipy matplotlib
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

### Watermark Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `context_width` | Number of previous tokens used for hashing | 5 |
| `gamma` | Proportion of vocabulary marked as "green" | 0.25 |
| `delta` | Logit bias added to green tokens | 2.0 |
| `detection_threshold` | Z-score threshold for detection | 4.0 |

### Tuning Guidelines

- **Higher `gamma`**: More tokens marked green → stronger watermark, potentially less natural
- **Higher `delta`**: Stronger bias → more detectable but may affect quality
- **Larger `context_width`**: More secure but slower detection

---

## 📊 Detection Metrics

The detector provides several metrics:

- **Z-score**: Measures how unusual the green token frequency is
  - `z > 4.0`: Strong watermark detected
  - `2.0 < z < 4.0`: Weak signal
  - `z < 2.0`: No watermark

- **P-value**: Probability of observing this pattern by chance
  - `p < 0.0001`: Very high confidence
  - `p < 0.05`: Significant detection

- **Green Token Ratio**: Percentage of tokens that are green
  - Expected ratio without watermark: `gamma` (e.g., 0.25)
  - With watermark: typically > 0.5

---

## 🛡️ Robustness

### What the Watermark Survives
✅ Copy/paste operations  
✅ Light paraphrasing  
✅ Minor edits  
✅ Format changes  

### Limitations
❌ Heavy rewriting or summarization  
❌ Translation to another language  
❌ Adversarial attacks specifically designed to remove watermarks  

---

## 📈 Visualization

Generate watermark pattern visualizations to analyze the detection results and see the distribution of green tokens throughout the text.

---

## 📚 References

This implementation is based on research in:

- "A Watermark for Large Language Models" (Kirchenbauer et al., 2023)
- "On the Reliability of Watermarks for Large Language Models" (Christ et al., 2023)
- Zero-width character steganography techniques

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
