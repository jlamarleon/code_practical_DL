# 🧠 Transformer Model Exploration with Qwen2 (1.5B Instruct)

## 📌 Project Overview

This project demonstrates how to:

- Install and configure required libraries
- Check GPU availability in Google Colab
- Load a HuggingFace Large Language Model (LLM)
- Run inference with temperature and sampling
- Inspect tokenizer vocabulary
- Explore transformer architecture internals
- Visualize forward-pass module execution
- Generate text step-by-step with logits and probabilities
- Visualize token embeddings using t-SNE

The default model used:

```
Qwen/Qwen2-1.5B-Instruct
```

You may switch to larger variants if GPU memory allows.

---


👤 Author

Dr. Javier Lamar León – Investigador

Laboratório BigData@UE, Escola de Ciências e Tecnologia, Universidade de Évora, Portugal

Email: jlamarleon@gmail.com   jlamarleon@uevora.pt


# 📁 Main File

The main notebook for inference and exploration is:

```
notebooks/01_inference.ipynb
```

It can be run directly in **Google Colab**.

---

# 📦 Installation

Install required packages:

```bash
pip install -q transformers datasets peft accelerate
```

### Package Purpose

- **torch** → PyTorch backend (GPU + tensors)
- **transformers** → Model & tokenizer loading
- **datasets** → Optional dataset handling
- **peft** → LoRA / parameter-efficient fine-tuning
- **accelerate** → Optimized device handling
- **matplotlib** → t-SNE visualization plotting
- **scikit-learn** → TSNE (from sklearn.manifold)

---

# 🖥️ Running in Google Colab

1. Open Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Upload `notebooks/01_inference.ipynb` or open it directly from GitHub.
3. Change runtime type to GPU: `Runtime → Change runtime type → GPU`
4. Run all cells to install dependencies, check GPU, load the model, and perform inference.

Optional: Save outputs to Google Drive for persistence.

---

# 🤖 Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "Qwen/Qwen2-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if DEVICE=="cuda" else torch.float32,
    device_map="auto" if DEVICE=="cuda" else None,
    trust_remote_code=True
)

model.eval()
```

---

# ✨ Text Generation

Basic generation helper:

```python
def generate_text(prompt, max_length=128, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

# 🔎 Transformer Architecture Overview

### Processing Flow

1. Tokenization  
2. Embeddings  
3. Positional Encoding  
4. Multi-Head Self-Attention  
5. Feed-Forward Layers  
6. Output Projection  
7. Softmax & Token Generation

---

# 📚 Vocabulary Exploration

Access vocabulary:

```python
vocab = tokenizer.get_vocab()
print(len(vocab))
```

Random sampling:

```python
sample_vocab(tokenizer, N=100)
```

Language-filtered sampling:
- English (ASCII tokens)
- French / Spanish (non-ASCII tokens)

---

# 🏗️ Inspecting Model Internals

The notebook includes:
- Forward hook tracing
- Module execution order
- Weight shapes
- Trainable status
- Output tensor shapes
- Functional role comments

---

# 📊 Step-by-Step Token Generation

Includes token-by-token generation with:
- Logits
- Softmax probabilities
- Token ID
- Decoded token
- Running sequence

Supports:
- Greedy decoding (temperature=0)
- Temperature sampling
- Top-k sampling

---

# 📈 Embedding Visualization (t-SNE)

Visualizes selected token embeddings in 2D:

```python
visualize_token_embeddings_tsne(model, tokenizer, tokens, device=DEVICE)
```

Example tokens:
- king / queen
- man / woman
- dog / cat
- happy / sad

---

# ⚙️ Requirements

- Python 3.8+
- PyTorch 2.x
- CUDA-enabled GPU (recommended)
- 8GB+ VRAM for 0.5B model

---

# 📜 License

This project is for educational and research purposes.
Refer to the original model license for commercial usage restrictions.

---

# 🙌 Acknowledgements

- HuggingFace Transformers
- Qwen Team
- PyTorch

---

# 🚀 Summary

This repository provides a hands-on, transparent exploration of:

- How transformers work internally
- How token probabilities are computed
- How embeddings represent meaning
- How GPU acceleration affects performance

Designed for students, researchers, and practitioners who want to go beyond black-box usage and understand LLM mechanics.

---

Happy experimenting! 🧠✨

