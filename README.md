# 🎙️ Backroom Group Audio Deepfake Detector

An advanced, locally-hosted machine learning system designed to forensically analyze audio files and microphone inputs to detect synthetic (AI-generated) voice artifacts. 

Built on a fine-tuned **HuBERT** foundation model and a custom **Digital Signal Processing (DSP)** pipeline, this system successfully bridges the "domain gap" between pristine studio audio and noisy real-world environments.

---

## 🗂️ Project Structure & Pipeline

This repository contains a modular, end-to-end pipeline covering raw data ingestion, model training, and live deployment.

### 1. Dataset Preparation
* **`flatten_dataset.py`**: Restructures and flattens complex nested audio directories into a clean, unified format for the pipeline.
* **`sort_dataset.py`**: Organizes the raw audio files into binary classification folders (e.g., `human_gen` vs. `ai_gen`).
* **`augment_dataset.py`**: Injects mathematical variations (like codec compression and noise) into pristine audio to close the domain gap and prevent false positives.
* **`build_dataset.py`**: Compiles the final, balanced dataset and prepares it for the PyTorch dataloader.

### 2. Model Training
* **`train.py`**: The core training loop utilizing the Hugging Face `Trainer` API. Automatically handles mixed-precision (fp16) and GPU offloading to fine-tune the final classification layers of the HuBERT model.

### 3. Deployment
* **`live_demo.py`**: The final production application. Runs the DSP cleaning pipeline (including harmonic isolation and noise gating) and local inference via a clean Gradio web interface.

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.8+
* An NVIDIA GPU (CUDA) is highly recommended for real-time training and inference, though the inference script will safely fall back to CPU if necessary.

### Environment Setup
Clone the repository and create an isolated Python virtual environment:

```bash
git clone [https://github.com/your-username/backroom-deepfake-detector.git](https://github.com/your-username/backroom-deepfake-detector.git)
cd backroom-deepfake-detector
python -m venv venv
