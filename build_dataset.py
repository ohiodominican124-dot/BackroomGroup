import os
import shutil
import random
from pathlib import Path

def sample_and_copy(source_dir, target_dir, num_samples, prefix):
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Ensure the target directory exists
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {source_path} for .wav files...")
    # Grab all wav files (rglob searches subfolders too, just in case)
    all_wavs = list(source_path.rglob("*.wav"))
    
    if len(all_wavs) < num_samples:
        print(f"⚠️ Warning: Only found {len(all_wavs)} files in {source_dir}. Copying all of them.")
        sampled_wavs = all_wavs
    else:
        # Randomly sample the exact amount needed
        sampled_wavs = random.sample(all_wavs, num_samples)
        
    print(f"Copying {len(sampled_wavs)} files to {target_dir}...")
    for i, wav_path in enumerate(sampled_wavs):
        # Add a prefix to the filename so files from different datasets don't overwrite each other
        new_name = f"{prefix}_{wav_path.name}"
        dest_path = target_path / new_name
        
        # Copy the file (leaves original intact)
        shutil.copy2(wav_path, dest_path)
        
        # Print progress every 5000 files so you know it hasn't frozen
        if (i + 1) % 5000 == 0:
            print(f"  -> Copied {i + 1} / {len(sampled_wavs)} files...")

if __name__ == "__main__":
    # Your Source Paths
    path_real_codec = r"C:\Users\LGS76\Downloads\Real\Codecfake"
    path_real_libri = r"C:\Users\LGS76\Downloads\Real\LibriTTS"
    path_fake_codec = r"C:\Users\LGS76\Downloads\Fake\Codecfake"
    
    # Your VS Code Target Paths
    target_human = "dataset/human_gen"
    target_ai = "dataset/ai_gen"
    
    print("Starting dataset compilation. This will take a few minutes...")
    
    # 1. 20k Real Codecfake -> Human Gen
    sample_and_copy(path_real_codec, target_human, 20000, "real_codec")
    
    # 2. 20k Real LibriTTS -> Human Gen
    sample_and_copy(path_real_libri, target_human, 20000, "real_libri")
    
    # 3. 40k Fake Codecfake -> AI Gen
    sample_and_copy(path_fake_codec, target_ai, 40000, "fake_codec")
    
    print("✅ Dataset compilation complete! You have 80,000 files ready for training.")
