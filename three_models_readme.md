# Training Three RoBERTa Models for Claim Verification

## Overview

This setup trains and compares **three different models** for your numerical claim verification task:

1. **Model 1**: Trained ONLY on your dataset
2. **Model 2**: Trained on BOTH your dataset AND QuantEmp (combined)
3. **Model 3**: Trained ONLY on QuantEmp (baseline for comparison)

## Why Three Models?

### Model 1: Your Dataset Only
- **Purpose**: See how well the model performs when trained specifically on your data
- **Best for**: When your data has unique characteristics different from QuantEmp
- **Risk**: Might overfit if your dataset is small

### Model 2: Combined Dataset
- **Purpose**: Leverage more training data by combining both datasets
- **Best for**: When both datasets share similar patterns
- **Benefit**: More training data → potentially better generalization
- **Risk**: If datasets are too different, might hurt performance

### Model 3: QuantEmp Only (Baseline)
- **Purpose**: Establish a baseline performance on standard benchmark
- **Best for**: Comparing against existing research
- **Use**: Reference point to understand if your data is harder/easier

## Quick Start

### Step 1: Ensure Your Files Are Ready

You need these 6 JSON files:
```
our_train.json
our_val.json
our_test.json
quantemp_train.json
quantemp_val.json
quantemp_test.json
```

### Step 2: Train All Three Models

```bash
python train_three_models.py
```

**What happens:**
1. Trains Model 1 on your data (saves to `./models/model1_our_dataset/`)
2. Trains Model 2 on combined data (saves to `./models/model2_combined/`)
3. Trains Model 3 on QuantEmp (saves to `./models/model3_quantemp_baseline/`)
4. Evaluates all models on appropriate test sets
5. Saves comparison results to `three_models_comparison.json`

**Time estimate:** 1-3 hours depending on dataset size and GPU

### Step 3: Visualize Results

```bash
python visualize_comparison.py
```

**Output:**
- `three_models_comparison.png` - Comprehensive visual comparison
- Console output with detailed summary

## Understanding the Results

### Key Metrics Explained

| Metric | What it means | When to use |
|--------|---------------|-------------|
| **Accuracy** | % of correct predictions | Overall performance, balanced classes |
| **F1 (Weighted)** | Balance of precision/recall, weighted by class size | Imbalanced classes |
| **F1 (Macro)** | Average F1 across all classes | When all classes matter equally |

### What to Look For

**1. Best Validation F1**
- Shows which model learned best during training
- Higher = better learning

**2. Test Metrics**
- Shows real-world performance
- Compare these across models

**3. Model 2 on Combined vs Your Data Only**
- If Model 2 does better on your data → combined training helps!
- If Model 2 does worse → datasets might be too different

## Expected Training Output

```
==================================================================
MODEL 1: Training on YOUR dataset only
==================================================================
Loaded 1234 examples from our_train.json
Label distribution: Counter({'False': 650, 'True': 584})
Loaded 234 examples from our_val.json

Epoch 1/5
Training Loss: 0.6234
Val Loss: 0.5123
Val Accuracy: 0.7234
Val F1: 0.7156
💾 Saved best model (F1: 0.7156)

Epoch 2/5
...

✅ Training completed for model1_our_dataset
   Best F1: 0.7856 at epoch 4

==================================================================
MODEL 2: Training on COMBINED dataset (Your + QuantEmp)
==================================================================
...

==================================================================
MODEL 3: Training on QuantEmp dataset only (Baseline)
==================================================================
...

==================================================================
EVALUATION PHASE: Testing all models
==================================================================
...

==================================================================
FINAL COMPARISON SUMMARY
==================================================================

Model 1 (Your dataset only):
  Best Validation F1: 0.7856
  Test Accuracy: 0.7645
  Test F1 (Weighted): 0.7823

Model 2 (Combined dataset):
  Best Validation F1: 0.8123
  Test Accuracy (Combined): 0.7934
  Test F1 (Combined, Weighted): 0.8045
  Test Accuracy (Your data only): 0.7756
  Test F1 (Your data only, Weighted): 0.7912

Model 3 (QuantEmp baseline):
  Best Validation F1: 0.7234
  Test Accuracy: 0.7012
  Test F1 (Weighted): 0.7156

✅ All models trained and evaluated!
📊 Results saved to 'three_models_comparison.json'
💾 Models saved in './models/' directory
```

## Interpreting the Comparison

### Scenario 1: Model 2 > Model 1
```
Model 1 (Your data): F1 = 0.75
Model 2 (Combined):  F1 = 0.82 on your data
```
**Conclusion**: ✅ Combined training helps! More diverse data improves performance.

### Scenario 2: Model 1 > Model 2
```
Model 1 (Your data): F1 = 0.78
Model 2 (Combined):  F1 = 0.72 on your data
```
**Conclusion**: ⚠️ Datasets might be too different. Stick with Model 1.

### Scenario 3: Model 1 ≈ Model 2
```
Model 1 (Your data): F1 = 0.76
Model 2 (Combined):  F1 = 0.77 on your data
```
**Conclusion**: → Marginal improvement. Either model works.

## What to Report in Your Paper/Thesis

### Table Format

| Model | Training Data | Test Accuracy | F1 (Weighted) | F1 (Macro) |
|-------|---------------|---------------|---------------|------------|
| RoBERTa (Your data) | 1,234 | 0.7645 | 0.7823 | 0.7612 |
| RoBERTa (Combined) | 3,567 | 0.7756 | 0.7912 | 0.7734 |
| RoBERTa (QuantEmp) | 2,333 | 0.7012 | 0.7156 | 0.6989 |

### Key Points to Mention

1. **Data size effect**: "Training on combined data (3,567 examples) improved F1 from 0.78 to 0.79 compared to our dataset alone (1,234 examples)"

2. **Generalization**: "Model 2 achieved X% F1 on combined test set, demonstrating ability to generalize across datasets"

3. **Baseline comparison**: "Our best model outperforms the QuantEmp baseline by X percentage points"

## File Structure After Training

```
.
├── our_train.json
├── our_val.json
├── our_test.json
├── quantemp_train.json
├── quantemp_val.json
├── quantemp_test.json
├── train_three_models.py
├── visualize_comparison.py
├── models/
│   ├── model1_our_dataset/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   ├── model2_combined/
│   │   └── ...
│   └── model3_quantemp_baseline/
│       └── ...
├── three_models_comparison.json
└── three_models_comparison.png
```

## Advanced: Using Individual Models

### Load and Use Model 1
```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('./models/model1_our_dataset')

# Make prediction
claim = "Your claim here"
evidence = "Evidence text here"

encoding = tokenizer(claim, evidence, return_tensors='pt', max_length=512, truncation=True)
outputs = model(**encoding)
prediction = torch.argmax(outputs.logits, dim=1)
```

### Cross-Dataset Testing

Want to test Model 1 (trained on your data) on QuantEmp test set?

```python
# Load Model 1
model1 = RobertaForSequenceClassification.from_pretrained('./models/model1_our_dataset')

# Load QuantEmp test data with Model 1's label mapping
# You'll need to manually map QuantEmp labels to Model 1's label space
# This is tricky if label sets differ!
```

## Common Issues

### Issue 1: Label Mismatch
**Problem**: Your dataset has `['True', 'False']` but QuantEmp has `['Supported', 'Refuted', 'Not Enough Info']`

**Solution**: The script handles this automatically for Model 2 by creating a unified label space.

### Issue 2: Memory Error
**Problem**: `CUDA out of memory`

**Solution**:
```python
# In train_three_models.py, reduce batch size
CONFIG = {
    'batch_size': 4,  # Instead of 8
    # ...
}
```

### Issue 3: Different Dataset Sizes
**Problem**: One dataset is much larger, dominating training

**Solution**: Use weighted sampling (advanced):
```python
from torch.utils.data import WeightedRandomSampler

# Calculate sample weights
# Give smaller dataset higher weights
```

## Hyperparameter Optimization (Optional)

Want to optimize Model 2 (the best one)?

```bash
# Create a new optimize script for Model 2
python optimize_model2.py
```

Just modify the Optuna script to use combined datasets instead of single dataset.

## Next Steps

1. **Run training**: `python train_three_models.py`
2. **Check results**: `python visualize_comparison.py`
3. **Analyze**: Look at the comparison visualization
4. **Choose best model**: Based on your test set performance
5. **Report**: Use metrics in your paper/thesis
6. **(Optional)** Optimize the best model with Optuna

## Questions for Your Guide

After running these experiments, discuss with your guide:

1. "Model X performed best on our data. Should we use this for final evaluation?"
2. "Combined training helped/hurt. What does this say about our dataset?"
3. "Should we try data augmentation or other techniques to improve further?"
4. "What other baselines should we compare against?"

## Time Budgeting

| Task | Time (with GPU) | Time (CPU only) |
|------|----------------|-----------------|
| Train all 3 models | 1-3 hours | 10-30 hours |
| Evaluation | 10-20 minutes | 1-2 hours |
| Visualization | 1 minute | 1 minute |
| **Total** | **~2-4 hours** | **~15-35 hours** |

**Recommendation**: Use Google Colab if you don't have a GPU!

---

Good luck with your experiments! 🚀
