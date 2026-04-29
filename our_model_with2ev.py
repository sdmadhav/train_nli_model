import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import os
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).parent.resolve()


class ClaimVerificationDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512, num_evidences=2, label_map=None):
        json_path = SCRIPT_DIR / json_file if not Path(json_file).is_absolute() else Path(json_file)

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_evidences = num_evidences

        if label_map is not None:
            self.label_map = label_map
        else:
            unique_labels = sorted(set(item['label'] for item in self.data))
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        print(f"Loaded {len(self.data)} examples from {json_path}")
        print(f"Using {self.num_evidences} evidence(s) per claim")
        print(f"Label distribution: {Counter([item['label'] for item in self.data])}")
        print(f"Label mapping: {self.label_map}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        claim = item['claim']
        label = self.label_map[item['label']]

        questions, evidences = [], []
        for i, ev in enumerate(item['evidences']):
            if i >= self.num_evidences:
                break
            if 'questions' in ev:
                questions.append(ev['questions'])
            if 'top_k_doc' in ev and ev['top_k_doc']:
                evidences.append(ev['top_k_doc'][0])

        feature = (
            f"[Claim]: {claim} "
            f"[Questions]: {' '.join(questions)} "
            f"[Evidences]: {' '.join(evidences)}"
        )

        encoding = self.tokenizer(
            feature,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'claim': claim
        }


def train_model(model, train_loader, val_loader, device, config):
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'],
                      weight_decay=config['weight_decay'])

    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_f1 = 0
    best_epoch = 0
    save_dir = SCRIPT_DIR / "models" / "model1_our_only"

    for epoch in range(config['epochs']):
        # Training
        model.train()
        total_loss = 0
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            train_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels_list = [], []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels_list, val_preds)
        val_f1 = f1_score(val_labels_list, val_preds, average='weighted')
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"  💾 Saved best model (F1: {best_f1:.4f})")

    print(f"\n✅ Training complete. Best F1: {best_f1:.4f} at epoch {best_epoch}")
    return best_f1, best_epoch


def evaluate_model(model, test_loader, device, label_map):
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    idx_to_label = {v: k for k, v in label_map.items()}
    label_names = [idx_to_label[i] for i in range(len(label_map))]

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))


def main():
    CONFIG = {
        'batch_size': 8,
        'learning_rate': 2e-5,
        'epochs': 10,
        'max_length': 512,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'seed': 42,
        'num_evidences': 2
    }

    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Load datasets
    train_dataset = ClaimVerificationDataset(
        'our_train.json', tokenizer,
        max_length=CONFIG['max_length'],
        num_evidences=CONFIG['num_evidences']
    )
    LABEL_MAP = train_dataset.label_map

    val_dataset = ClaimVerificationDataset(
        'our_val.json', tokenizer,
        max_length=CONFIG['max_length'],
        num_evidences=CONFIG['num_evidences'],
        label_map=LABEL_MAP
    )
    test_dataset = ClaimVerificationDataset(
        'our_test.json', tokenizer,
        max_length=CONFIG['max_length'],
        num_evidences=CONFIG['num_evidences'],
        label_map=LABEL_MAP
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Train
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=len(LABEL_MAP)
    )
    train_model(model, train_loader, val_loader, device, CONFIG)

    # Evaluate best model
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)

    best_model = RobertaForSequenceClassification.from_pretrained(
        str(SCRIPT_DIR / 'models' / 'model1_our_only')
    )
    evaluate_model(best_model, test_loader, device, LABEL_MAP)


if __name__ == "__main__":
    main()
