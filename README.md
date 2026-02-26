# 📚 Deep Learning Practical: Transformers & LLMs

Welcome to the **Deep Learning Practical** focused on **Transformers and Large Language Models (LLMs)**.  
This course will guide you **from understanding the Transformer architecture to running inference and fine-tuning models** like Qwen and TinyLlama.

---

## 🎯 Learning Objectives

By the end of this practical session, students will be able to:

1. Understand the **core Transformer architecture** (self-attention, MLP, residuals, layer norms).  
2. Load a **pre-trained 1B+ parameter LLM** and run **text generation**.  
3. Explore **internal Transformer components** to understand weights, projections, and layers.  
4. Perform **first experiments in model fine-tuning** using full parameter updates.  
5. Apply **LoRA (Low-Rank Adaptation)** for **parameter-efficient fine-tuning**.  
6. Use professional workflow with **VS Code, GitHub, and Google Colab**.  

---

## 📂 Repository Structure
```text
DL_LLM_Practical/
│
├── notebooks/
│   ├── 01_inference.ipynb         # Load model, run prompts, explore Transformer internals
│   ├── 02_full_finetuning.ipynb   # Full fine-tuning on small dataset
│   ├── 03_lora_finetuning.ipynb   # LoRA fine-tuning experiments
│
├── data/
│   └── small_dataset.json          # Sample instruction dataset
│
├── requirements.txt                # Python dependencies for the notebooks
└── README.md


---

## ⚡ Getting Started

### 1️⃣ Open Notebook in Google Colab

Click the badge to launch the notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USER/DL_LLM_Practical/blob/main/notebooks/01_inference.ipynb)

Or manually:

1. Go to [Google Colab](https://colab.research.google.com)  
2. Click **File → Open notebook → GitHub**  
3. Paste the repository URL:  


4. Select `01_inference.ipynb`

---

### 2️⃣ Set Runtime Type

To use GPU:

- **Runtime → Change runtime type → Hardware accelerator → GPU**  

This ensures the LLM runs efficiently.

---

### 3️⃣ Run the First Cell

The first cell installs all required packages:

```python
# 🚀 Install required packages for running the model
# - transformers: for loading and running LLMs
# - datasets: optional, for dataset handling
# - peft: for later LoRA experiments
# - accelerate: for optimized GPU usage in Colab
!pip install -q transformers datasets peft accelerate