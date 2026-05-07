import os
import shutil
from pathlib import Path

def flatten_directory(base_dir):
    base_path = Path(base_dir)
    # UPDATED: Matches the exact folder names from your new screenshot
    classes = ['ai_gen', 'human_gen']

    for cls in classes:
        cls_path = base_path / cls
        if not cls_path.exists():
            print(f"Could not find folder: {cls_path} - check your spelling!")
            continue
            
        print(f"Flattening {cls} folder...")
        
        # Find all wav files in subdirectories
        for wav_file in cls_path.rglob("*.wav"):
            # If the file is already directly in the 'ai_gen' or 'human_gen' folder, skip it
            if wav_file.parent == cls_path:
                continue
                
            # Create a new unique name so files don't overwrite each other
            subfolder_name = wav_file.parent.name
            new_name = f"{subfolder_name}_{wav_file.name}"
            new_path = cls_path / new_name
            
            # Move the file up
            shutil.move(str(wav_file), str(new_path))
            
        # Clean up the now-empty subfolders (like ASVspoof, LibriTTS)
        for subfolder in list(cls_path.iterdir()):
            if subfolder.is_dir():
                try:
                    shutil.rmtree(str(subfolder)) # Forces deletion of the subfolder
                    print(f"Removed subfolder: {subfolder.name}")
                except Exception as e:
                    print(f"Could not remove {subfolder.name}: {e}")

if __name__ == "__main__":
    flatten_directory("dataset")
    print("Dataset flattened successfully!")
