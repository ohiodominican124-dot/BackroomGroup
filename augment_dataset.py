import os
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from pathlib import Path

# Define augmentations
augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

def augment_folder(input_folder):
    classes = ['ai_gen', 'human_gen']
    
    for cls in classes:
        folder_path = Path(input_folder) / cls
        print(f"Processing {cls}...")
        
        for file_path in folder_path.glob("*.wav"):
            if "_aug" in file_path.name:
                continue # Skip already augmented files
                
            try:
                # Load audio
                samples, sample_rate = librosa.load(file_path, sr=16000)
                
                # Apply augmentation
                augmented_samples = augmenter(samples=samples, sample_rate=sample_rate)
                
                # Save new file
                new_file_name = file_path.stem + "_aug.wav"
                new_file_path = folder_path / new_file_name
                sf.write(new_file_path, augmented_samples, sample_rate)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Point this to your main dataset folder
    augment_folder("dataset") 
    print("Augmentation complete!")
