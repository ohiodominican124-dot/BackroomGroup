import gradio as gr
import torch
import librosa
import numpy as np
from scipy.signal import butter, lfilter
from transformers import AutoFeatureExtractor, HubertForSequenceClassification

MODEL_PATH = "./final_hubert_model"

print("Loading model and Neural DSP pipeline...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = HubertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- DSP AUDIO FILTERS ---
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=85, fs=16000, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_audio(audio_filepath):
    # 1. Load audio (Forces Mono and 16kHz)
    speech, sr = librosa.load(audio_filepath, sr=16000, mono=True)
    
    # 2. DC Offset Removal (Centers the electrical baseline)
    speech = speech - np.mean(speech)
    
    # 3. Highpass filter (Deletes low-end desk bumps and fan rumble)
    speech = highpass_filter(speech, cutoff=85, fs=16000)
    
    # 4. Silence Trap & Auto-Normalize 
    if np.max(np.abs(speech)) < 0.005:
        raise ValueError("Audio is completely empty. Mic is muted or blocked.")
    speech = librosa.util.normalize(speech)
    
    # 5. Precision Trimming (Cuts dead air from the absolute beginning and end)
    speech, _ = librosa.effects.trim(speech, top_db=40)
    
    # 6. Absolute Zero Check (Prevents the model from crashing on a literal 0-second array)
    if len(speech) == 0:
        raise ValueError("Audio contains only silence after trimming.")
    
    return speech

def classify_audio(audio_filepath):
    if audio_filepath is None:
        return None
    
    try:
        speech = preprocess_audio(audio_filepath)
        inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits
            
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
        raw_labels = model.config.id2label
        
        display_map = {
            "ai_gen": "AI Generated",
            "human_gen": "Human Generated",
            "fake": "AI Generated", 
            "real": "Human Generated" 
        }
        
        result = {}
        for i in range(len(raw_labels)):
            raw_name = raw_labels[i]
            pretty_name = display_map.get(raw_name, raw_name) 
            result[pretty_name] = float(probabilities[i])
            
        return result
        
    except ValueError as ve:
        print(f"DSP Warning: {ve}")
        raise gr.Error(str(ve))
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        raise gr.Error("System Error Processing Audio")

def reset_ui():
    """Natively clears the audio and label outputs."""
    return None, None

# --- UI: NATIVE GRADIO (Clean & Professional) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🎙️ Backroom Group Audio Deepfake Detector
        Record your voice or upload a `.wav` file to analyze it for synthetic generation artifacts. 
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Input Audio"
            )
            clear_btn = gr.Button("Clear Results", variant="secondary")
            
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=2, label="Detection Confidence")

    # Native Gradio Event Triggers
    audio_input.stop_recording(fn=classify_audio, inputs=audio_input, outputs=label_output)
    audio_input.upload(fn=classify_audio, inputs=audio_input, outputs=label_output)
    
    clear_btn.click(fn=reset_ui, inputs=None, outputs=[audio_input, label_output])

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
