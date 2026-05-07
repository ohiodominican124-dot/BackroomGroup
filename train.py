import torch
import evaluate
import numpy as np
import librosa
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, HubertForSequenceClassification, TrainingArguments, Trainer

# 1. Configuration
MODEL_ID = "facebook/hubert-base-ls960"
DATA_DIR = "dataset" 
MAX_DURATION = 5.0 

# 2. Load Dataset
print("Loading 80k dataset... this may take a moment.")
dataset = load_dataset("audiofolder", data_dir=DATA_DIR)
# Standard split
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42) # 10% test is enough for 80k
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 3. Preprocessing
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

def preprocess_function(examples):
    audio_arrays = []
    for x in examples["audio"]:
        # Trim silence
        trimmed_audio, _ = librosa.effects.trim(x["array"], top_db=30)
        audio_arrays.append(trimmed_audio)

    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        max_length=int(16000 * MAX_DURATION), 
        truncation=True, 
        padding="max_length"
    )
    return inputs

print("Tokenizing and trimming (Batched processing)...")
# batched=True and batch_size help manage RAM usage
encoded_dataset = dataset.map(
    preprocess_function, 
    remove_columns=["audio"], 
    batched=True, 
    batch_size=100
)

# 4. Model Setup
print("Loading model...")
class_names = dataset["train"].features["label"].names
label2id = {name: i for i, name in enumerate(class_names)}
id2label = {i: name for i, name in enumerate(class_names)}

model = HubertForSequenceClassification.from_pretrained(
    MODEL_ID, 
    num_labels=len(class_names),
    label2id=label2id,
    id2label=id2label
)

# Freeze base, unfreeze last 2 layers
for param in model.hubert.parameters():
    param.requires_grad = False
for param in model.hubert.encoder.layers[-2:].parameters():
    param.requires_grad = True

# 5. Metrics & Training
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

training_args = TrainingArguments(
    output_dir="./hubert_deepfake_detector",
    eval_strategy="steps",        # Evaluating by steps instead of epochs for large data
    eval_steps=500,               # Check accuracy every 500 steps
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,           # ONLY KEEP THE 2 BEST MODELS (Saves disk space!)
    learning_rate=3e-5,           # Lower LR for massive dataset stability
    warmup_steps=500,             # Gradually ramp up the learning rate
    per_device_train_batch_size=8, 
    gradient_accumulation_steps=4,
    num_train_epochs=3,           # 3 epochs is plenty for 80,000 samples
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(), # Use mixed precision if you have a modern GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,
)

print("Starting deep training on 80,000 samples...")
trainer.train()

trainer.save_model("./final_hubert_model")
print("Training complete!")
