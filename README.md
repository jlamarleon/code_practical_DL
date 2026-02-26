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
