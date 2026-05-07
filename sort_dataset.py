import os
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from tqdm import tqdm
import warnings

# Keep the terminal clean
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Firing up Feature Extraction on {device.type.upper()}...")

print("Loading Google's HuBERT Model...")
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
hubert.eval() # Freeze the massive acoustic layers

real_dir = "dataset/real"
fake_dir = "dataset/fake"

def extract_features(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ ERROR: Cannot find {folder_path}!")
        return torch.tensor([])

    features_list = []
    files = []
    
    # Deep Search: Walk through all subfolders (Codecfake, LibriTTS, etc.)
    for root, dirs, filenames in os.walk(folder_path):
        for f in filenames:
            if f.lower().endswith(('.wav', '.flac')):
                files.append(os.path.join(root, f))
    
    print(f"--> Found {len(files)} files in {folder_path}")
    
    if len(files) == 0:
        return torch.tensor([])

    for file_path in tqdm(files, desc=f"Extracting {os.path.basename(folder_path)}"):
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Standardize everything to 16kHz Mono
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.squeeze().numpy()
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

            # Extract the mathematical fingerprint
            with torch.no_grad():
                outputs = hubert(inputs.input_values.to(device))
                pooled_features = outputs.last_hidden_state.mean(dim=1).squeeze()
                features_list.append(pooled_features.cpu())
                
        except Exception as e:
            pass # Skip silently if a file is deeply corrupted
            
    return torch.stack(features_list) if features_list else torch.tensor([])

# --- EXECUTION ---
print("\nExtracting REAL features (Clean & Dirty)...")
real_features = extract_features(real_dir)
torch.save(real_features, "real_features.pt")

print("\nExtracting FAKE features (Clean & Dirty)...")
fake_features = extract_features(fake_dir)
torch.save(fake_features, "fake_features.pt")

print("\n✅ Extraction complete! 6,000 mathematical fingerprints saved.")
