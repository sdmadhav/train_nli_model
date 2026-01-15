
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
import torch

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
# ========================================================================
# DATA PREPARATION
# ========================================================================

def read_json(file):
    data = json.load(open(file, encoding='utf-8'))
    return pd.DataFrame(data)

# Load data
finetune_data = read_json('Processed_complete_dataset.json')
print(f"Total data loaded: {len(finetune_data)}")
print(finetune_data.head())
print(finetune_data.label.value_counts())

# Replace label
finetune_data["label"] = finetune_data["label"].str.replace(
    "Half True/False",
    "Conflicting",
    regex=False
)

print("\nAfter label replacement:")
print(finetune_data["label"].value_counts())

# Filter out test data
finetune_data = finetune_data[finetune_data['Category'] != 'test']
print(f"\nAfter removing test category:")
print(finetune_data.Category.value_counts())
# ========================================================================
# BUILD PREMISE AND HYPOTHESIS
# ========================================================================

def build_premise(row):
    parts = [f"Claim: {row['claim']}"]
    
    for i, ev in enumerate(row['reranked_our_evidences'], start=1):
        q = ev.get("questions", "").strip()
        docs = ev.get("top_k_doc", [])
        a = " ".join(docs).strip()
        
        parts.append(f"Question {i}: {q}")
        parts.append(f"Answer {i}: {a}")
    
    return "\n".join(parts)

LABEL2HYPOTHESIS = {
    "True": "The claim is true.",
    "False": "The claim is false.",
    "Conflicting": "The claim is conflicting or partially incorrect."
}

# Split data
train_df = finetune_data[finetune_data["Category"] == "train"]
val_df   = finetune_data[finetune_data["Category"] == "validation"]

print(f"\nTrain size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")

# Create examples
def row_to_example(row):
    return {
        "premise": build_premise(row),
        "hypothesis": LABEL2HYPOTHESIS[row["label"]],
        "label": list(LABEL2HYPOTHESIS.keys()).index(row["label"])
    }

train_examples = [row_to_example(row) for _, row in train_df.iterrows()]
val_examples   = [row_to_example(row) for _, row in val_df.iterrows()]

# Create datasets
train_dataset = Dataset.from_list(train_examples)
val_dataset   = Dataset.from_list(val_examples)

print(f"\nTrain dataset: {len(train_dataset)} examples")
print(f"Val dataset: {len(val_dataset)} examples")
# ========================================================================
# TOKENIZATION
# ========================================================================

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

def tokenize_fn(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

print("\nTokenizing datasets...")
train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset   = val_dataset.map(tokenize_fn, batched=True)

# Remove unnecessary columns and rename label
train_dataset = train_dataset.remove_columns(["premise", "hypothesis"])
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format("torch")

val_dataset = val_dataset.remove_columns(["premise", "hypothesis"])
val_dataset = val_dataset.rename_column("label", "labels")
val_dataset.set_format("torch")

print("Tokenization complete!")
print(f"Train dataset columns: {train_dataset.column_names}")
# ========================================================================
# MODEL LOADING
# ========================================================================

print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large-mnli",
    num_labels=3  # True, False, Conflicting
)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded and moved to: {device}")
# ========================================================================
# TRAINING CONFIGURATION
# ========================================================================

training_args = TrainingArguments(
    output_dir="./roberta-factcheck",
    per_device_train_batch_size=8,      # Good for 32GB GPU
    per_device_eval_batch_size=16,      # Can use larger batch for eval
    gradient_accumulation_steps=2,       # Effective batch size = 16
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    logging_dir='./logs',
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,                  # Keep only 2 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,                           # Mixed precision for faster training
    dataloader_num_workers=4,            # Parallel data loading
    report_to="none"
)

print("\nTraining Configuration:")
print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  FP16: {training_args.fp16}")

# ========================================================================
# TRAINER
# ========================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

print("\n" + "="*70)
print("Starting Training...")
print("="*70)

# Train
trainer.train()

print("\n" + "="*70)
print("Training Complete!")
print("="*70)

# Save final model
trainer.save_model("./roberta-factcheck-final")
print("Model saved to './roberta-factcheck-final'")
# ========================================================================
# EVALUATION
# ========================================================================

print("\nEvaluating on validation set...")
eval_results = trainer.evaluate()

print("\nEvaluation Results:")
for key, value in eval_results.items():
    print(f"  {key}: {value}")

# ========================================================================
# PREDICTION FUNCTION
# ========================================================================

def predict(row):
    """Predict label for a single row"""
    premise = build_premise(row)
    hypotheses = list(LABEL2HYPOTHESIS.values())
    
    inputs = tokenizer(
        [premise] * len(hypotheses),
        hypotheses,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
    
    predicted_idx = probs.argmax().item()
    return list(LABEL2HYPOTHESIS.keys())[predicted_idx]

print("\n" + "="*70)
print("Training pipeline completed successfully!")
print("="*70)
print("\nNow use the predict() function to make predictions on new data.")
print("Example: predicted_label = predict(test_row)")
