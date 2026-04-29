"""
Fine-tune FinQA-RoBERTa-large on "our" dataset.

Pipeline
--------
  Step 1 (done by QuanTemp):
      roberta-large-mnli  -->  fine-tuned on FinQA  -->  model_weights.zip

  Step 2 (this script):
      Load model_weights.zip into MultiClassClassifier
      --> fine-tune on our dataset (our_train / our_val / our_test)

Setup
-----
  Place the QuanTemp model folder next to this script:
      finqa_roberta_claimdecomp_early_stop_2/
          model_weights.zip          <- the weights file
          tokenizer_config.json      <- (optional, we use roberta-large-mnli tokenizer)
          ...

  Or set FINQA_WEIGHTS_PATH in CONFIG to any absolute path.

Architecture (MultiClassClassifier — must match QuanTemp notebook exactly)
---------------------------------------------------------------------------
  backbone : roberta-large-mnli   (hidden=1024, 24 layers)
  head     : Dropout(0.1) -> Linear(1024->768) -> ReLU -> Linear(768->3)
  pooling  : pooler_output

  The saved state_dict keys look like:
      roberta.*          <- backbone
      dropout.*          <- head dropout
      mlp.0.*            <- Linear(1024->768)
      mlp.2.*            <- Linear(768->3)

Training config (mirrors QuanTemp notebook exactly)
----------------------------------------------------
  max_length          : 256
  batch_size          : 16
  lr                  : 2e-5,  adam eps=1e-8
  scheduler           : defined but step() NEVER called  <- notebook quirk
  grad clipping       : DISABLED                         <- notebook quirk
  epochs              : up to 20
  early_stopping      : patience=2, monitors val LOSS
  frozen layers       : encoder layers 0-4 (first 5 of 24)
  loss                : CrossEntropyLoss
  seed                : 42
  label encoding      : sklearn LabelEncoder (alphabetical)
                        Conflicting=0, False=1, True=2
  num_evidences cap   : 2
"""

import json, os, random, time, datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).parent.resolve()

CONFIG = {
    "base_model":      "roberta-large-mnli",

    # Path to the model_weights.zip from QuanTemp drive
    # (finqa_roberta_claimdecomp_early_stop_2/model_weights.zip)
    "finqa_weights":   "finqa_roberta_claimdecomp_early_stop_2/model_weights.zip",

    "max_length":      256,
    "batch_size":      16,
    "lr":              2e-5,
    "adam_eps":        1e-8,
    "epochs":          20,
    "early_stop_patience": 2,
    "freeze_first_n_layers": 5,
    "hidden_dim":      1024,
    "mlp_dim":         768,
    "dropout":         0.1,
    "num_evidences":   2,
    "seed":            42,
}


# ── Helpers ───────────────────────────────────────────────────────────
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def flat_accuracy(preds, labels):
    return np.sum(np.argmax(preds, axis=1).flatten() == labels.flatten()) / len(labels.flatten())


# ── Early Stopping (exact copy from QuanTemp notebook) ────────────────
class EarlyStopping:
    def __init__(self, patience=2, verbose=True, delta=0, path="checkpoint.pt"):
        self.patience = patience; self.verbose = verbose
        self.counter = 0; self.best_score = None
        self.early_stop = False; self.val_loss_min = np.inf
        self.delta = delta; self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score; self._save(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score; self._save(val_loss, model); self.counter = 0

    def _save(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ── Model (exact architecture from QuanTemp notebook) ─────────────────
class MultiClassClassifier(nn.Module):
    def __init__(self, base_model_path, num_classes,
                 hidden_dim=1024, mlp_dim=768, dropout=0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(
            base_model_path, output_hidden_states=True, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, tokens, masks):
        output = self.roberta(tokens, attention_mask=masks)
        return self.mlp(self.dropout(output["pooler_output"]))


def load_finqa_weights(model, weights_path, device):
    """
    Load the QuanTemp-released FinQA weights into the model.
    The file is model_weights.zip — a torch.save() of state_dict.
    PyTorch can load it directly with torch.load().
    """
    print(f"\nLoading FinQA weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)

    # The state_dict was saved from MultiClassClassifier directly,
    # so keys are: roberta.*, dropout.*, mlp.0.*, mlp.2.*
    # strict=True should work since architectures match exactly.
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"  Loaded successfully.")
    print(f"  Missing keys  : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    return model


# ── Feature extraction ────────────────────────────────────────────────
def get_features(data, num_evidences=None):
    features = []
    for fact in data:
        claim = fact["claim"]
        evidences, questions = [], []
        for i, ev in enumerate(fact["evidences"]):
            if num_evidences is not None and i >= num_evidences:
                break
            if ev.get("top_k_doc"):
                evidences.append(ev["top_k_doc"][0])
            if ev.get("questions"):
                questions.append(ev["questions"])
        questions = list(set(questions))
        evidences = list(set(evidences))
        features.append(
            "[Claim]:" + claim +
            "[Questions]:" + " ".join(questions) +
            "[Evidences]:" + " ".join(evidences))
    return features

def tokenize_features(features, tokenizer, max_length):
    input_ids, attention_masks = [], []
    for sent in features:
        enc = tokenizer(
            sent,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt")
        input_ids.append(enc["input_ids"])
        attention_masks.append(enc["attention_mask"])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

# ── Training loop ─────────────────────────────────────────────────────
def train(model, train_loader, val_loader, device, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "model_weights.pt")

    optimizer = AdamW(model.parameters(), lr=config["lr"], eps=config["adam_eps"])

    # Defined but NEVER stepped — exact QuanTemp notebook behaviour
    _ = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_loader) * config["epochs"])

    loss_func = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config["early_stop_patience"],
                                   verbose=True, path=ckpt_path)
    total_t0 = time.time()

    for epoch_i in range(config["epochs"]):
        print(f"\n======== Epoch {epoch_i+1} / {config['epochs']} ========")
        print("Training...")
        t0 = time.time()
        model.train()
        total_loss = 0; total_acc = 0

        for step, batch in enumerate(train_loader):
            if step % 40 == 0 and step != 0:
                print(f"  Batch {step:>5,}  of  {len(train_loader):>5,}.    "
                      f"Elapsed: {format_time(time.time()-t0)}.")
            b_ids    = batch[0].to(device)
            b_mask   = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            logits = model(b_ids, b_mask)
            loss = loss_func(logits, b_labels)
            total_loss += loss.item()
            loss.backward()
            # grad clipping DISABLED — matches notebook (line is commented out there)
            optimizer.step()
            # scheduler.step() NOT called — matches notebook
            total_acc += flat_accuracy(logits.detach().cpu().numpy(), b_labels.cpu().numpy())

        print(f" Train Accuracy: {total_acc/len(train_loader):.2f}")
        print(f"  Average training loss: {total_loss/len(train_loader):.2f}")
        print(f"  Training epoch took: {format_time(time.time()-t0)}")

        # Validation
        print("\nRunning Validation...")
        t0 = time.time(); model.eval()
        val_loss = 0; val_acc = 0

        with torch.no_grad():
            for batch in val_loader:
                b_ids    = batch[0].to(device)
                b_mask   = batch[1].to(device)
                b_labels = batch[2].to(device)
                logits = model(b_ids, b_mask)
                val_loss += loss_func(logits, b_labels).item()
                val_acc  += flat_accuracy(logits.cpu().numpy(), b_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f"  Accuracy: {val_acc/len(val_loader):.2f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping"); break

        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {format_time(time.time()-t0)}")

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time()-total_t0)} (h:mm:ss)")
    return ckpt_path


# ── Evaluation ────────────────────────────────────────────────────────
def evaluate(model, test_loader, device, label_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            b_ids    = batch[0].to(device)
            b_mask   = batch[1].to(device)
            b_labels = batch[2].to(device)
            logits = model(b_ids, b_mask)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())

    print(f"\nAccuracy:    {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Weighted F1: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
    print(f"Macro F1:    {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    random.seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"]); torch.cuda.manual_seed_all(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def load_json(name):
        with open(SCRIPT_DIR / name) as f:
            return json.load(f)

    print("\nLoading data...")
    train_data = load_json("our_train.json")
    val_data   = load_json("our_val.json")
    test_data  = load_json("our_test.json")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Train labels: {Counter(d['label'] for d in train_data)}")

    # Alphabetical LabelEncoder: Conflicting=0, False=1, True=2
    LE = LabelEncoder()
    train_labels_enc = LE.fit_transform([d["label"] for d in train_data])
    val_labels_enc   = LE.transform([d["label"] for d in val_data])
    test_labels_enc  = LE.transform([d["label"] for d in test_data])
    num_classes = len(LE.classes_)
    label_names = list(LE.classes_)
    print(f"Label mapping: { {c: i for i, c in enumerate(LE.classes_)} }")

    # Tokenizer
    print(f"\nLoading tokenizer: {CONFIG['base_model']}")
    tokenizer = RobertaTokenizer.from_pretrained(CONFIG["base_model"])

    # Features
    print(f"\nBuilding features (num_evidences={CONFIG['num_evidences']}, max_length={CONFIG['max_length']})...")
    print("Tokenizing train...")
    train_ids, train_masks = tokenize_features(get_features(train_data, CONFIG["num_evidences"]), tokenizer, CONFIG["max_length"])
    print("Tokenizing val...")
    val_ids,   val_masks   = tokenize_features(get_features(val_data,   CONFIG["num_evidences"]), tokenizer, CONFIG["max_length"])
    print("Tokenizing test...")
    test_ids,  test_masks  = tokenize_features(get_features(test_data,  CONFIG["num_evidences"]), tokenizer, CONFIG["max_length"])

    train_loader = DataLoader(TensorDataset(train_ids, train_masks, torch.tensor(train_labels_enc)),
                              sampler=RandomSampler(TensorDataset(train_ids, train_masks, torch.tensor(train_labels_enc))),
                              batch_size=CONFIG["batch_size"])
    val_loader   = DataLoader(TensorDataset(val_ids,  val_masks,  torch.tensor(val_labels_enc)),
                              sampler=SequentialSampler(TensorDataset(val_ids, val_masks, torch.tensor(val_labels_enc))),
                              batch_size=CONFIG["batch_size"])
    test_loader  = DataLoader(TensorDataset(test_ids, test_masks, torch.tensor(test_labels_enc)),
                              sampler=SequentialSampler(TensorDataset(test_ids, test_masks, torch.tensor(test_labels_enc))),
                              batch_size=CONFIG["batch_size"])

    # Build model with roberta-large-mnli architecture
    print(f"\nBuilding model: {CONFIG['base_model']}")
    model = MultiClassClassifier(CONFIG["base_model"], num_classes,
                                  CONFIG["hidden_dim"], CONFIG["mlp_dim"], CONFIG["dropout"])

    # Load FinQA fine-tuned weights on top
    weights_path = SCRIPT_DIR / CONFIG["finqa_weights"]
    model = load_finqa_weights(model, str(weights_path), device)

    # Freeze encoder layers 0-4
    print(f"\nFreezing encoder layers 0-{CONFIG['freeze_first_n_layers']-1} ...")
    for param in model.roberta.encoder.layer[:CONFIG["freeze_first_n_layers"]].parameters():
        param.requires_grad = False

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_p:,} | Trainable: {trainable_p:,}")
    model.to(device)

    # Train
    save_dir  = str(SCRIPT_DIR / "models" / "finqa_roberta_finetuned_our_dataset")
    ckpt_path = train(model, train_loader, val_loader, device, CONFIG, save_dir)

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET (best checkpoint by val loss)")
    print("="*60)
    best_model = MultiClassClassifier(CONFIG["base_model"], num_classes,
                                       CONFIG["hidden_dim"], CONFIG["mlp_dim"], CONFIG["dropout"])
    best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    best_model.to(device)
    evaluate(best_model, test_loader, device, label_names)


if __name__ == "__main__":
    main()
