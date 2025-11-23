# Complete Workflow: Training Three RoBERTa Models

## 🎯 Your Goal

Train and compare three models to find the best approach for your numerical claim verification task:

1. **Model 1**: Your dataset only → See if specialized training works best
2. **Model 2**: Combined datasets → Leverage more training data
3. **Model 3**: QuantEmp only → Establish baseline for comparison

## 📋 Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] All 6 JSON files ready (our_train/val/test, quantemp_train/val/test)
- [ ] Required packages installed (`pip install -r requirements.txt`)
- [ ] GPU available (recommended) or access to Google Colab
- [ ] ~50GB free disk space (for models and results)

## 🚀 Step-by-Step Workflow

### Phase 1: Setup (5-10 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify your data files
ls -lh *.json

# 3. Check configuration
python config.py

# 4. Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Phase 2: Training (1-3 hours)

```bash
# Run the main training script
python train_three_models.py
```

**What happens:**
- ✅ Loads all datasets
- ✅ Trains Model 1 (your data)
- ✅ Trains Model 2 (combined)
- ✅ Trains Model 3 (QuantEmp)
- ✅ Evaluates all models
- ✅ Saves results and models

**Monitor progress:**
```
Look for:
- ✅ "Saved best model" messages
- ✅ Validation F1 increasing
- ⚠️ "Possible overfitting" warnings
```

**What you'll see in console:**
```
==================================================================
MODEL 1: Training on YOUR dataset only
==================================================================
Loaded 1234 examples from our_train.json
Label distribution: Counter({'False': 650, 'True': 584})

Epoch 1/5
100%|██████████| 154/154 [02:15<00:00, 1.14it/s, loss=0.432]
Epoch 1:
  Train Loss: 0.4321
  Val Loss: 0.3876
  Val Accuracy: 0.7456
  Val F1: 0.7389
  💾 Saved best model (F1: 0.7389)

... (continues for all models)

✅ All models trained and evaluated!
📊 Results saved to 'three_models_comparison.json'
💾 Models saved in './models/' directory
```

### Phase 3: Visualization (1 minute)

```bash
# Generate comparison plots
python visualize_comparison.py
```

**Output:**
- `three_models_comparison.png` - Visual comparison
- Console summary with key findings

### Phase 4: Analysis (10-15 minutes)

Open `three_models_comparison.png` and look at:

1. **Validation F1 Scores** (top-left)
   - Which model learned best during training?

2. **Test Accuracy** (top-middle)
   - Which model generalizes best?

3. **Test F1 Scores** (top-right)
   - Most reliable metric for imbalanced data

4. **Model 2 Breakdown** (bottom-left)
   - Does combined training help or hurt?

5. **Side-by-side comparison** (bottom-middle)
   - Direct comparison of all metrics

6. **Summary Table** (bottom-right)
   - Quick reference, best values highlighted

## 📊 Understanding Your Results

### Scenario A: Model 2 Wins! 🎉

```
Model 1 (Your data):     F1 = 0.75
Model 2 (Combined):      F1 = 0.82
Model 3 (QuantEmp):      F1 = 0.71
```

**Interpretation:**
- ✅ Combined training significantly helps
- ✅ Both datasets share similar patterns
- ✅ More training data = better generalization

**Next Steps:**
- Use Model 2 for final experiments
- Consider using Model 2 for your paper
- Mention: "Training on combined dataset improved performance by 9.3%"

### Scenario B: Model 1 Wins

```
Model 1 (Your data):     F1 = 0.78
Model 2 (Combined):      F1 = 0.72
Model 3 (QuantEmp):      F1 = 0.69
```

**Interpretation:**
- ⚠️ Datasets are too different
- ⚠️ QuantEmp data might confuse the model
- ✅ Your data has unique characteristics

**Next Steps:**
- Use Model 1 for final experiments
- Analyze why datasets differ (check label distributions)
- Mention: "Specialized training on our dataset outperformed combined approach"

### Scenario C: All Similar

```
Model 1 (Your data):     F1 = 0.76
Model 2 (Combined):      F1 = 0.77
Model 3 (QuantEmp):      F1 = 0.75
```

**Interpretation:**
- → Performance is comparable
- → Small datasets, high variance
- → Could go either way

**Next Steps:**
- Run statistical significance test
- Train 3-5 times with different seeds
- Average results for more reliable comparison

## 🎓 For Your Research Paper/Thesis

### What to Report

#### Table 1: Model Performance Comparison

| Model | Training Data | #Examples | Test Acc | F1-W | F1-M |
|-------|--------------|-----------|----------|------|------|
| RoBERTa-Your | Your dataset | 1,234 | 0.765 | 0.782 | 0.761 |
| RoBERTa-Combined | Combined | 3,567 | 0.779 | 0.791 | 0.773 |
| RoBERTa-QuantEmp | QuantEmp | 2,333 | 0.701 | 0.716 | 0.699 |

*F1-W: Weighted F1, F1-M: Macro F1*

#### Key Findings to Mention

1. **Dataset characteristics:**
   > "Our dataset consists of X examples with Y label distribution, while QuantEmp contains Z examples focused on numerical claims."

2. **Training approach:**
   > "We trained three RoBERTa-base models: (1) on our dataset alone, (2) on combined datasets, and (3) on QuantEmp as baseline."

3. **Best performance:**
   > "Model X achieved the highest F1 score of Y on our test set, representing an Z% improvement over the baseline."

4. **Combined vs individual:**
   > "Training on combined data [improved/did not improve] performance, suggesting [datasets share patterns / our data is unique]."

5. **Baseline comparison:**
   > "Our best model outperforms the QuantEmp baseline by X percentage points, demonstrating effectiveness on our task."

### Statistical Significance (Advanced)

If you want to be rigorous, run each model 5 times:

```bash
# Modify config.py to set different seeds
for seed in 42 43 44 45 46; do
    python train_three_models.py --seed $seed
done

# Then average results and compute standard deviation
```

Report as: **F1 = 0.782 ± 0.014** (mean ± std)

## ⚙️ Advanced: Hyperparameter Optimization

Want to squeeze out maximum performance?

### Step 1: Choose Best Model
Based on your results, pick the winning model (usually Model 1 or 2)

### Step 2: Optimize It

```bash
# Modify optimize.py to use your chosen dataset
# For Model 1:
CONFIG['train_file'] = 'our_train.json'

# For Model 2:
CONFIG['train_file'] = ['our_train.json', 'quantemp_train.json']

# Then run optimization
python optimize.py
```

This will take 3-8 hours but can improve F1 by 2-5 percentage points.

### Step 3: Train Final Model

```bash
# After Optuna finds best parameters, train one final time
python train_with_best_params.py
```

## 🐛 Troubleshooting Common Issues

### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solution:**
```python
# In config.py, reduce batch size
TRAINING_CONFIG = {
    'batch_size': 4,  # Instead of 8
    'max_length': 256  # Instead of 512
}
```

### Issue 2: Labels Don't Match

**Error:**
```
KeyError: 'Conflicting' not in label_map
```

**Cause:** QuantEmp has different labels than your dataset

**Solution:** The script handles this automatically for Model 2. If still issues:
1. Check both JSON files for unique labels
2. Make sure label field is named 'label' (not 'labels' or 'class')

### Issue 3: Training Not Improving

**Symptoms:**
```
Epoch 1: Val F1=0.5123
Epoch 2: Val F1=0.5145
Epoch 3: Val F1=0.5139
...
```

**Solutions:**
1. Increase learning rate: `'learning_rate': 3e-5`
2. Train longer: `'epochs': 10`
3. Check if data loaded correctly:
   ```python
   # Print first batch
   for batch in train_loader:
       print(batch['labels'])
       break
   ```

### Issue 4: Model Overfitting

**Symptoms:**
```
Train Loss: 0.12 ↓
Val Loss: 0.65 ↑
```

**Solutions:**
1. Early stopping (already implemented)
2. Add dropout:
   ```python
   model = RobertaForSequenceClassification.from_pretrained(
       'roberta-base',
       hidden_dropout_prob=0.2  # Add this
   )
   ```
3. Use weight decay: `'weight_decay': 0.01`
4. Get more training data

### Issue 5: Too Slow Training

**On CPU:**
- Expected: 10-20x slower than GPU
- Solution: Use Google Colab (free GPU) or cloud GPU

**On GPU but still slow:**
- Increase batch size: `'batch_size': 16`
- Reduce max_length: `'max_length': 256`
- Use `pin_memory=True` in DataLoader

## 🔄 Google Colab Setup

If you don't have a GPU, use Google Colab:

```python
# 1. In Colab notebook, enable GPU
# Runtime → Change runtime type → GPU

# 2. Install packages
!pip install transformers torch scikit-learn optuna tqdm

# 3. Upload your JSON files
from google.colab import files
uploaded = files.upload()  # Upload all 6 JSON files

# 4. Upload Python scripts
# Or clone from GitHub if you have the code there
!git clone [your-repo-url]
%cd [your-repo-name]

# 5. Run training
!python train_three_models.py

# 6. Download results
files.download('three_models_comparison.json')
files.download('three_models_comparison.png')
```

## 📝 Checklist: Before You're Done

- [ ] All three models trained successfully
- [ ] Results saved to `three_models_comparison.json`
- [ ] Visualization generated (`three_models_comparison.png`)
- [ ] Models saved in `./models/` directory
- [ ] Best model identified based on test F1
- [ ] Results documented for paper/thesis
- [ ] (Optional) Best model optimized with Optuna
- [ ] (Optional) Statistical significance tested with multiple runs

## 🎯 Quick Command Reference

```bash
# Complete workflow
pip install -r requirements.txt
python config.py                    # Check configuration
python train_three_models.py        # Train all models (1-3 hours)
python visualize_comparison.py      # Generate plots

# Optional: Optimize best model
python optimize.py                  # Find best hyperparameters (3-8 hours)

# Test individual model
python test.py --model ./models/model1_our_dataset
```

## 📧 Questions for Your Guide

After completing training, discuss:

1. **Performance:** "Model X achieved F1 of Y. Is this good for claim verification?"
2. **Combined training:** "Model 2 [helped/hurt]. What does this tell us about our data?"
3. **Next steps:** "Should we try other models (BERT, DeBERTa) or focus on error analysis?"
4. **Baselines:** "What other baselines should we compare against?"
5. **Paper:** "How should we present these results in the paper?"

## 🎉 Success Criteria

You're done when you can answer:

✅ Which model performs best on your data?
✅ Does combined training help or hurt?
✅ How does your model compare to the QuantEmp baseline?
✅ What's the best F1 score you achieved?
✅ What are the main error patterns?

---

**Good luck with your experiments!** 🚀

If you get stuck, check:
1. Console error messages
2. This guide's troubleshooting section
3. Ask your guide
4. Check GitHub issues for transformers library
