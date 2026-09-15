# 🎨 Text-to-Image Generation with Stable Diffusion — Self-Hosted, Open-Weights Image AI

Generating images from plain-English prompts using **Stable Diffusion v1.5**, run entirely locally on GPU through HuggingFace's `diffusers` library — no hosted API, no per-image billing, and no black box. The model weights, the architecture, and the entire generation loop are open and inspectable.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-Stable%20Diffusion-FFD21E)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Hub-FFD21E?logo=huggingface&logoColor=black)

---

## 📌 Overview

Part 1 of my GenAI series generated images through **OpenAI's hosted `gpt-image-1-mini` API** — a single function call, someone else's servers, someone else's model weights. This project generates images a completely different way: **downloading actual open-weights model files from HuggingFace Hub and running the entire diffusion process on local (or Colab) GPU hardware.** Same output category — an image from a text prompt — fundamentally different pipeline underneath.

---

## 🧠 How Stable Diffusion Actually Works (Briefly)

Unlike the autoregressive, next-token-prediction approach behind every text LLM in this series, Stable Diffusion generates images through **iterative denoising**:

1. Start with **pure random noise** in a compressed latent space
2. A neural network (U-Net) predicts what noise to *remove*, guided by the text prompt's embedding
3. Repeat the denoising step dozens of times, each pass refining the image slightly
4. Decode the final latent representation into a real pixel-space image

This is a genuinely different generative paradigm from every text-generation notebook elsewhere in this series — worth understanding as a category of its own, not just "the image version of GPT."

---

## 🔬 The Pipeline, Step by Step

### 1️⃣ Installing the Stack
```bash
pip install diffusers transformers
pip install accelerate    # efficient GPU memory management
pip install hf_transfer    # faster large-model downloads from HuggingFace Hub
```
Each dependency has a distinct job: `diffusers` provides the pipeline abstraction itself, `transformers` supplies the text encoder that turns prompts into embeddings, `accelerate` optimizes how model weights are placed and moved across GPU memory, and `hf_transfer` speeds up pulling multi-gigabyte model files.

### 2️⃣ Loading the Pretrained Pipeline
```python
from diffusers import DiffusionPipeline
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```
`from_pretrained()` downloads the complete Stable Diffusion v1.5 model — text encoder, U-Net, and VAE decoder — directly from HuggingFace Hub. `use_safetensors=True` is a deliberate security-conscious choice: `.safetensors` is a weight-storage format that cannot execute arbitrary code on load, unlike older pickle-based checkpoint formats.

### 3️⃣ Moving to GPU
```python
pipeline.to("cuda")
```
Diffusion models are **computationally heavy** — dozens of denoising passes through a large neural network per image. This single line moves the entire pipeline onto GPU memory, without which generation would be impractically slow on CPU.

### 4️⃣ Generating Images from Prompts
```python
image = pipeline("An image of a duck sitting on a mountain").images[0]
image
```
The pipeline call itself is deceptively simple — one string in, one `PIL.Image` object out. Underneath, that single call runs the full noise-embedding-denoise-decode loop described above.

---


### Recommended Generation Parameters (Not Yet Tuned in This Notebook)

| Parameter | Default | What It Controls |
|---|---|---|
| `num_inference_steps` | 50 | More steps = higher quality, slower generation |
| `guidance_scale` | 7.5 | How closely the image follows the prompt vs. creative freedom |
| `negative_prompt` | None | Elements to explicitly steer the model away from |

The notebook runs every prompt with pipeline defaults — a natural next step is tuning these explicitly per use case.

## 🖼️ Prompts Tested

The notebook runs several prompts to explore the model's range:

| Prompt | What It Tests |
|---|---|
| *"An image of a duck sitting on a mountain"* | Basic object + setting composition |
| *"An image of cat eating a fish"* | Action/interaction between two subjects |
| *"An image of bollywood actor sitting in a restaurant eating biryani"* | Complex scene: person, setting, action, cultural specificity |
| *"An image of indian man standing"* | Simple single-subject generation |

Running a spread of prompts — from simple to compositionally complex — is a reasonable way to sanity-check a newly loaded pipeline before relying on it for anything more deliberate.

---

## ⚠️ A Note on Hardware Requirements

Unlike every text-based notebook elsewhere in this series, this one has a **hard hardware dependency**. Stable Diffusion v1.5 runs dozens of forward passes through a multi-billion-parameter U-Net per image — on CPU alone, a single generation can take several minutes; on a T4 GPU (freely available on Google Colab), it takes seconds. If `pipeline.to("cuda")` raises an error, the runtime almost certainly does not have GPU access enabled.

---

## 🗂️ Repository Structure

```
stable-diffusion-image-generation/
├── GenAI_26_Stable_Diffusion.ipynb   # Main notebook
├── requirements.txt                    # Dependencies
├── .gitignore                          # Keeps generated images & model cache out of git
└── README.md                           # This documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- **A CUDA-capable GPU** (locally or via Google Colab — this notebook will not run practically on CPU alone)
- ~5GB free disk space for the downloaded model weights

### Installation

```bash
git clone https://github.com/Kailaswadje/stable-diffusion-image-generation.git
cd stable-diffusion-image-generation

pip install -r requirements.txt

jupyter notebook GenAI_26_Stable_Diffusion.ipynb
```

> 💡 No API key needed — Stable Diffusion v1.5 is a public model on HuggingFace Hub. If running on Google Colab, select a GPU runtime (`Runtime → Change runtime type → T4 GPU`) before executing.

---

## 🧠 Key Takeaways

- **Diffusion models generate through iterative denoising, not next-token prediction** — a genuinely different generative mechanism from every LLM elsewhere in this series
- **`use_safetensors=True` is a real security decision**, not a performance flag — it avoids the arbitrary-code-execution risk of older pickle-based model formats
- **Self-hosted, open-weights generation trades convenience for control** — no per-image API cost and full model access, at the cost of needing real GPU compute
- **`accelerate` and `hf_transfer` solve practical infrastructure problems** — GPU memory management and large-file download speed are real engineering concerns once models leave the world of simple API calls
- This project sits alongside Part 1's hosted `gpt-image-1-mini` calls as a direct **hosted-API vs. self-hosted-open-weights** comparison — the same task, two fundamentally different infrastructure choices

---

## 🔮 Possible Extensions

- [ ] Experiment with `num_inference_steps` and `guidance_scale` to control generation quality vs. speed
- [ ] Compare output quality against Part 1's `gpt-image-1-mini` on identical prompts
- [ ] Try a negative prompt to steer the model away from unwanted elements
- [ ] Explore a newer model checkpoint (SDXL, SD 3) and compare output quality
- [ ] Wrap the pipeline in a small Streamlit UI, following this series' earlier app-building pattern

---

## 👤 Author

**Kailas Wadje**
MSc Data Science & AI, University of Liverpool

- GitHub: [@Kailaswadje](https://github.com/Kailaswadje)
- LinkedIn: [linkedin.com/in/kwadaje](https://www.linkedin.com/in/kwadaje/)

---

⭐ If this demystified diffusion-based image generation for you, consider giving it a star!
