import gradio as gr
import torch
import librosa
from transformers import AutoFeatureExtractor, HubertForSequenceClassification

# Load the fine-tuned model and feature extractor
MODEL_PATH = "./final_hubert_model"

print("Loading model into memory...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = HubertForSequenceClassification.from_pretrained(MODEL_PATH)

# Move model to GPU if available for faster inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def classify_audio(audio_filepath):
    if audio_filepath is None:
        return "No audio provided."
    
    # 1. Load and resample audio from the microphone to 16kHz
    speech, _ = librosa.load(audio_filepath, sr=16000)
    
    # 2. Extract features
    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 3. Predict
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # 4. Convert logits to probabilities using Softmax
    probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
    
    # 5. Format for Gradio Label output
    labels = model.config.id2label
    result = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
    
    return result

# Build the Gradio Interface
demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="Record Your Voice"),
    outputs=gr.Label(num_top_classes=2, label="Detection Confidence"),
    title="🎙️ Deepfake Audio Detector (HuBERT)",
    description="Record yourself speaking. The model will analyze the audio to determine if it sounds human or AI-generated.",
    # Removed allow_flagging as it is no longer supported in Gradio 6.0
)

if __name__ == "__main__":
    # Added inbrowser=True to automatically open your default web browser
    demo.launch(share=False, theme=gr.themes.Soft(), inbrowser=True)
