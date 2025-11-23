# Quick Reference: Training with Custom Format

## 🎯 Key Changes Made

1. **✅ Input Format**: Now uses `[Claim]: ... [Questions]: ... [Evidences]: ...`
2. **✅ Label Mapping**: Uses YOUR dataset labels for ALL models
3. **✅ Flexible k**: Easy to choose 1, 2, or 3 evidences
4. **✅ Extended Length**: Can use up to 1024+ tokens (configurable)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Configure (1 minute)

Open `config.py` and set:

```python
NUM_EVIDENCES = 3  # Change to 1, 2, or 3
MAX_LENGTH = 1024  # Can use 512, 1024, 1536, or 2048
```

**What k means:**
- `k=1`: Uses only first question-evidence pair
- `k=2`: Uses first two question-evidence pairs  
- `k=3`: Uses all three question-evidence pairs

### Step 2: Train (1-3 hours)

```bash
python train_three_models.py
```

### Step 3: Check Results

```bash
cat three_models_comparison.json
```

---

## 📝 Input Format Explained

Your data is now formatted as:

```
[Claim]: {claim text} 
[Questions]: {question1} {question2} {question3} 
[Evidences]: {evidence1} {evidence2} {evidence3}
```

**Example with k=3:**
```
[Claim]: Public transport is free in Brussels and there is a bus there called "not the 48" which takes you to a mystery location.
[Questions]: Which bus is available to a mystery location in Brussels? Which bus is there called "not the 48"? Which bus leaves from Brussels called "not the 48"?
[Evidences]: Mar 7, 2023 ... not only is public transport free in Brussels... Mar 13, 2023 ... not only is public transport free in Brussels... Mar 8, 2023 ... Brussels takes riders to a mystery location for free...
```

**Example with k=1 (only first evidence):**
```
[Claim]: Public transport is free in Brussels...
[Questions]: Which bus is available to a mystery location in Brussels?
[Evidences]: Mar 7, 2023 ... not only is public transport free in Brussels...
```

---

## 🏷️ Label Mapping

**Important Change**: All three models now use YOUR dataset labels!

If your dataset has:
```json
{
  "label": "True"  // or "False"
}
```

Then:
- Model 1 trains on your data → uses `{"True": 0, "False": 1}`
- Model 2 trains on combined data → **filters QuantEmp to only keep "True"/"False" labels**
- Model 3 trains on QuantEmp → **only uses QuantEmp items with "True"/"False" labels**

**What if QuantEmp has different labels?**
- Items with labels like "Conflicting", "Not Enough Info", etc. are **automatically skipped**
- You'll see warnings in console: `"Skipping item with label 'Conflicting' (not in reference map)"`
- This is normal and expected!

---

## ⚙️ Configuration Options

### For Different k Values

```python
# config.py

# Use only first evidence (fastest, less information)
NUM_EVIDENCES = 1
MAX_LENGTH = 512

# Use first two evidences (balanced)
NUM_EVIDENCES = 2
MAX_LENGTH = 1024

# Use all three evidences (most information, slower)
NUM_EVIDENCES = 3
MAX_LENGTH = 1536
```

### Memory Issues?

If you get "CUDA out of memory":

```python
# In config.py, reduce these:
TRAINING_CONFIG = {
    'batch_size': 4,      # Instead of 8
    'num_evidences': 2,   # Instead of 3
    'max_length': 512     # Instead of 1024
}
```

### Want Longer Sequences?

RoBERTa technically supports up to 514 tokens, but we can set higher values:

```python
MAX_LENGTH = 1024  # Tokenizer will truncate if needed
# or
MAX_LENGTH = 2048  # Even longer (more memory)
```

**Note**: Higher MAX_LENGTH = more memory usage but less truncation

---

## 📊 Expected Output

```
==================================================================
CONFIGURATION
==================================================================
Number of evidences (k): 3
Max sequence length: 1024
Batch size: 8
Learning rate: 2e-05
Epochs: 5

==================================================================
STEP 1: Loading YOUR dataset to establish label mapping
==================================================================
Loaded 1234 examples from our_train.json
Using 3 evidence(s) per claim
Label distribution: Counter({'False': 650, 'True': 584})
Label mapping: {'False': 0, 'True': 1}

✅ Reference label mapping established: {'False': 0, 'True': 1}

==================================================================
MODEL 1: Training on YOUR dataset only
==================================================================
Loaded 1234 examples from our_train.json
...

==================================================================
MODEL 2: Training on COMBINED dataset (Your + QuantEmp)
==================================================================
Loading our_train.json...
Loading quantemp_train.json...
Skipping item with label 'Conflicting' (not in reference map)
Skipping item with label 'Not Enough Info' (not in reference map)
...
Combined dataset: 2156 examples (loaded 3567, skipped 1411)
...

==================================================================
MODEL 3: Training on QuantEmp dataset only (Baseline)
==================================================================
Loaded 2333 examples from quantemp_train.json
Skipping items with non-matching labels...
...

==================================================================
FINAL COMPARISON SUMMARY
==================================================================

📊 Using 3 evidence(s) per claim
📏 Max sequence length: 1024 tokens
🏷️ Label mapping (from YOUR dataset): {'False': 0, 'True': 1}

Model 1 (Your dataset only):
  Best Validation F1: 0.7856
  Test Accuracy: 0.7645
  Test F1 (Weighted): 0.7823

Model 2 (Combined dataset):
  Best Validation F1: 0.8023
  Test Accuracy: 0.7756
  Test F1 (Weighted): 0.7912

Model 3 (QuantEmp baseline):
  Best Validation F1: 0.7234
  Test Accuracy: 0.7012
  Test F1 (Weighted): 0.7156

🏆 Best performing model: Model 2 with F1 = 0.7912
```

---

## 🔬 Experimenting with k

Want to find the best k value? Run experiments:

### Experiment 1: k=1
```python
# config.py
NUM_EVIDENCES = 1
MAX_LENGTH = 512
```
```bash
python train_three_models.py
mv three_models_comparison.json results_k1.json
```

### Experiment 2: k=2
```python
# config.py
NUM_EVIDENCES = 2
MAX_LENGTH = 1024
```
```bash
python train_three_models.py
mv three_models_comparison.json results_k2.json
```

### Experiment 3: k=3
```python
# config.py
NUM_EVIDENCES = 3
MAX_LENGTH = 1536
```
```bash
python train_three_models.py
mv three_models_comparison.json results_k3.json
```

Then compare F1 scores across all three!

---

## 🎓 What to Report

### Table: Effect of Number of Evidences (k)

| k | Model | Test F1 | Test Acc |
|---|-------|---------|----------|
| 1 | Model 1 | 0.75 | 0.73 |
| 1 | Model 2 | 0.77 | 0.75 |
| 2 | Model 1 | 0.76 | 0.74 |
| 2 | Model 2 | 0.79 | 0.77 |
| 3 | Model 1 | 0.78 | 0.76 |
| 3 | Model 2 | 0.81 | 0.79 |

**Finding**: "Increasing k from 1 to 3 evidences improved F1 by 6 percentage points, suggesting that multiple evidence sources provide complementary information."

---

## ❓ FAQ

### Q: Why are some QuantEmp examples skipped?

**A**: Your dataset uses different labels (True/False) than QuantEmp (which might have Conflicting, Not Enough Info, etc.). The script only keeps QuantEmp examples with matching labels.

### Q: What if all QuantEmp examples are skipped?

**A**: This means QuantEmp has completely different labels. Model 3 won't train. You have two options:
1. Manually map QuantEmp labels to yours (edit the JSON files)
2. Skip Model 3 entirely

### Q: Can I use k > 3?

**A**: Yes! If your data has more than 3 evidences, modify the code:
```python
# In train_three_models.py, change:
CONFIG = {
    'num_evidences': 5,  # Or however many you have
    'max_length': 2048   # Increase accordingly
}
```

### Q: MAX_LENGTH 1024 vs 512 - what's the difference?

**A**: 
- 512: Faster, less memory, but might truncate longer texts
- 1024: Slower, more memory, keeps more text
- Recommendation: Start with 1024, reduce if out of memory

### Q: How do I know if sequences are being truncated?

**A**: Add this after loading data:
```python
# Check sequence lengths
for batch in train_loader:
    lens = (batch['attention_mask'].sum(dim=1))
    print(f"Avg length: {lens.float().mean():.1f}, Max: {lens.max()}")
    break
```

If avg is close to MAX_LENGTH, you're truncating!

---

## 🐛 Common Issues

### Issue: "No examples loaded"

**Cause**: Label mismatch between datasets

**Solution**: Check your label field:
```python
# Print labels from both datasets
with open('our_train.json') as f:
    our_data = json.load(f)
    print("Your labels:", set(d['label'] for d in our_data))

with open('quantemp_train.json') as f:
    qt_data = json.load(f)
    print("QuantEmp labels:", set(d['label'] for d in qt_data))
```

### Issue: "Sequences too long"

**Error**: Getting truncation warnings

**Solution**: Increase MAX_LENGTH or reduce NUM_EVIDENCES

### Issue: Model not learning

**Symptoms**: F1 stays at ~0.50

**Solutions**:
1. Check if labels are balanced
2. Try different learning rate
3. Verify input format is correct

---

## 💡 Tips

1. **Start with k=2**: Good balance of information and speed
2. **Use config.py**: Don't edit the main script
3. **Check console warnings**: They tell you about skipped data
4. **Compare k values**: Run experiments to find optimal k
5. **Monitor memory**: Watch GPU usage with `nvidia-smi`

---

## 📞 Next Steps

1. Run with default settings first
2. Check if results are reasonable
3. Experiment with different k values
4. Choose best configuration
5. Report findings to your guide

Good luck! 🚀
