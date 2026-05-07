import torch
import evaluate
import numpy as np
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, HubertForSequenceClassification, TrainingArguments, Trainer

# 1. Configuration
MODEL_ID = "facebook/hubert-base-ls960"
DATA_DIR = "dataset" # Path to your dataset folder
MAX_DURATION = 5.0 # Truncate audio longer than 5 seconds

# 2. Load Dataset
print("Loading dataset...")
dataset = load_dataset("audiofolder", data_dir=DATA_DIR)

# Split into train and test sets (80/20)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

# Ensure audio is 16kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 3. Preprocessing
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        max_length=int(16000 * MAX_DURATION), 
        truncation=True, 
        padding="max_length"
    )
    return inputs

print("Tokenizing dataset...")
encoded_dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True)

# ... (Keep Step 1, 2, and 3 the same) ...

# 4. Model Setup
print("Loading model...")

# AUTO-DETECT LABELS: Let's see exactly what folders `datasets` found
class_names = dataset["train"].features["label"].names
print(f"Found the following classes in your dataset: {class_names}")

# Dynamically build the label mappings based on what was actually found
label2id = {name: i for i, name in enumerate(class_names)}
id2label = {i: name for i, name in enumerate(class_names)}

model = HubertForSequenceClassification.from_pretrained(
    MODEL_ID, 
    num_labels=len(class_names), # Dynamically set the number of classes
    label2id=label2id,
    id2label=id2label
)

# FREEZE the base model (only train the classification head)
for param in model.hubert.parameters():
    param.requires_grad = False

# 5. Metrics & Training
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

training_args = TrainingArguments(
    output_dir="./hubert_deepfake_detector",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4, 
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    processing_class=feature_extractor, # <--- Fixed the tokenizer warning here!
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# Save final model
trainer.save_model("./final_hubert_model")
print("Training complete and model saved to ./final_hubert_model")
