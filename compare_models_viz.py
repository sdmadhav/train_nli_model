import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load results
with open('three_models_comparison.json', 'r') as f:
    results = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparison of Three RoBERTa Models for Claim Verification', 
             fontsize=16, fontweight='bold')

# ============================================================================
# 1. Validation F1 Scores Comparison
# ============================================================================
ax1 = axes[0, 0]
models = ['Model 1\n(Your Data)', 'Model 2\n(Combined)', 'Model 3\n(QuantEmp)']
val_f1s = [
    results['model1_our_dataset']['training']['best_val_f1'],
    results['model2_combined']['training']['best_val_f1'],
    results['model3_quantemp_baseline']['training']['best_val_f1']
]
colors = ['#3498db', '#2ecc71', '#e74c3c']

bars1 = ax1.bar(models, val_f1s, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax1.set_title('Best Validation F1 Scores', fontsize=13, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontweight='bold')

# ============================================================================
# 2. Test Accuracy Comparison
# ============================================================================
ax2 = axes[0, 1]
test_accs = [
    results['model1_our_dataset']['testing']['accuracy'],
    results['model2_combined']['testing_on_our_only']['accuracy'],  # Model 2 on your data
    results['model3_quantemp_baseline']['testing']['accuracy']
]

bars2 = ax2.bar(models, test_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Test Accuracy', fontsize=13, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontweight='bold')

# ============================================================================
# 3. Test F1 (Weighted) Comparison
# ============================================================================
ax3 = axes[0, 2]
test_f1s = [
    results['model1_our_dataset']['testing']['f1_weighted'],
    results['model2_combined']['testing_on_our_only']['f1_weighted'],
    results['model3_quantemp_baseline']['testing']['f1_weighted']
]

bars3 = ax3.bar(models, test_f1s, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('F1 Score (Weighted)', fontsize=12, fontweight='bold')
ax3.set_title('Test F1 Scores (Weighted)', fontsize=13, fontweight='bold')
ax3.set_ylim([0, 1])
ax3.grid(axis='y', alpha=0.3)

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontweight='bold')

# ============================================================================
# 4. Model 2 Performance: Combined vs Your Data Only
# ============================================================================
ax4 = axes[1, 0]
model2_comparison = ['Combined\nTest Set', 'Your Data\nOnly']
model2_accs = [
    results['model2_combined']['testing_on_combined']['accuracy'],
    results['model2_combined']['testing_on_our_only']['accuracy']
]

bars4 = ax4.bar(model2_comparison, model2_accs, color=['#9b59b6', '#f39c12'], 
                alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('Model 2 (Combined) - Test Performance', fontsize=13, fontweight='bold')
ax4.set_ylim([0, 1])
ax4.grid(axis='y', alpha=0.3)

for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontweight='bold')

# ============================================================================
# 5. All Metrics Side-by-Side
# ============================================================================
ax5 = axes[1, 1]
metrics = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)']
x = np.arange(len(metrics))
width = 0.25

model1_metrics = [
    results['model1_our_dataset']['testing']['accuracy'],
    results['model1_our_dataset']['testing']['f1_weighted'],
    results['model1_our_dataset']['testing']['f1_macro']
]

model2_metrics = [
    results['model2_combined']['testing_on_our_only']['accuracy'],
    results['model2_combined']['testing_on_our_only']['f1_weighted'],
    results['model2_combined']['testing_on_our_only']['f1_macro']
]

model3_metrics = [
    results['model3_quantemp_baseline']['testing']['accuracy'],
    results['model3_quantemp_baseline']['testing']['f1_weighted'],
    results['model3_quantemp_baseline']['testing']['f1_macro']
]

bars_m1 = ax5.bar(x - width, model1_metrics, width, label='Model 1 (Your Data)', 
                  color='#3498db', alpha=0.7, edgecolor='black')
bars_m2 = ax5.bar(x, model2_metrics, width, label='Model 2 (Combined)', 
                  color='#2ecc71', alpha=0.7, edgecolor='black')
bars_m3 = ax5.bar(x + width, model3_metrics, width, label='Model 3 (QuantEmp)', 
                  color='#e74c3c', alpha=0.7, edgecolor='black')

ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
ax5.set_title('All Metrics Comparison', fontsize=13, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(metrics)
ax5.legend(loc='lower right', fontsize=9)
ax5.set_ylim([0, 1])
ax5.grid(axis='y', alpha=0.3)

# ============================================================================
# 6. Summary Table
# ============================================================================
ax6 = axes[1, 2]
ax6.axis('tight')
ax6.axis('off')

table_data = [
    ['Metric', 'Model 1', 'Model 2', 'Model 3'],
    ['Training Data', 'Your Data', 'Combined', 'QuantEmp'],
    ['Best Val F1', 
     f"{results['model1_our_dataset']['training']['best_val_f1']:.4f}",
     f"{results['model2_combined']['training']['best_val_f1']:.4f}",
     f"{results['model3_quantemp_baseline']['training']['best_val_f1']:.4f}"],
    ['Test Accuracy',
     f"{results['model1_our_dataset']['testing']['accuracy']:.4f}",
     f"{results['model2_combined']['testing_on_our_only']['accuracy']:.4f}",
     f"{results['model3_quantemp_baseline']['testing']['accuracy']:.4f}"],
    ['Test F1 (Weighted)',
     f"{results['model1_our_dataset']['testing']['f1_weighted']:.4f}",
     f"{results['model2_combined']['testing_on_our_only']['f1_weighted']:.4f}",
     f"{results['model3_quantemp_baseline']['testing']['f1_weighted']:.4f}"],
    ['Test F1 (Macro)',
     f"{results['model1_our_dataset']['testing']['f1_macro']:.4f}",
     f"{results['model2_combined']['testing_on_our_only']['f1_macro']:.4f}",
     f"{results['model3_quantemp_baseline']['testing']['f1_macro']:.4f}"]
]

table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style first column
for i in range(1, 6):
    table[(i, 0)].set_facecolor('#ecf0f1')
    table[(i, 0)].set_text_props(weight='bold')

# Highlight best values in each row
for row_idx in range(2, 6):
    values = [float(table_data[row_idx][i]) for i in range(1, 4)]
    best_col = values.index(max(values)) + 1
    table[(row_idx, best_col)].set_facecolor('#d5f4e6')
    table[(row_idx, best_col)].set_text_props(weight='bold', color='green')

ax6.set_title('Performance Summary Table\n(Best values highlighted)', 
              fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('three_models_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to 'three_models_comparison.png'")
plt.show()

# ============================================================================
# Print text summary
# ============================================================================
print("\n" + "="*70)
print("THREE MODELS COMPARISON SUMMARY")
print("="*70)

print("\n📊 MODEL 1: Trained on YOUR dataset only")
print(f"   Best Validation F1: {results['model1_our_dataset']['training']['best_val_f1']:.4f}")
print(f"   Test Accuracy: {results['model1_our_dataset']['testing']['accuracy']:.4f}")
print(f"   Test F1 (Weighted): {results['model1_our_dataset']['testing']['f1_weighted']:.4f}")
print(f"   Test F1 (Macro): {results['model1_our_dataset']['testing']['f1_macro']:.4f}")

print("\n📊 MODEL 2: Trained on COMBINED dataset")
print(f"   Best Validation F1: {results['model2_combined']['training']['best_val_f1']:.4f}")
print("   Performance on Combined Test Set:")
print(f"      Accuracy: {results['model2_combined']['testing_on_combined']['accuracy']:.4f}")
print(f"      F1 (Weighted): {results['model2_combined']['testing_on_combined']['f1_weighted']:.4f}")
print("   Performance on Your Test Set Only:")
print(f"      Accuracy: {results['model2_combined']['testing_on_our_only']['accuracy']:.4f}")
print(f"      F1 (Weighted): {results['model2_combined']['testing_on_our_only']['f1_weighted']:.4f}")
print(f"      F1 (Macro): {results['model2_combined']['testing_on_our_only']['f1_macro']:.4f}")

print("\n📊 MODEL 3: Trained on QuantEmp dataset only (BASELINE)")
print(f"   Best Validation F1: {results['model3_quantemp_baseline']['training']['best_val_f1']:.4f}")
print(f"   Test Accuracy: {results['model3_quantemp_baseline']['testing']['accuracy']:.4f}")
print(f"   Test F1 (Weighted): {results['model3_quantemp_baseline']['testing']['f1_weighted']:.4f}")
print(f"   Test F1 (Macro): {results['model3_quantemp_baseline']['testing']['f1_macro']:.4f}")

# Determine which model performs best on your data
your_data_performances = {
    'Model 1': results['model1_our_dataset']['testing']['f1_weighted'],
    'Model 2': results['model2_combined']['testing_on_our_only']['f1_weighted'],
}

best_model = max(your_data_performances, key=your_data_performances.get)
best_score = your_data_performances[best_model]

print("\n" + "="*70)
print("🏆 CONCLUSION")
print("="*70)
print(f"Best performing model on YOUR test data: {best_model}")
print(f"F1 Score: {best_score:.4f}")

improvement = ((your_data_performances['Model 2'] - your_data_performances['Model 1']) 
               / your_data_performances['Model 1'] * 100)
if improvement > 0:
    print(f"\n✨ Model 2 (Combined) improves over Model 1 by {improvement:.2f}%")
    print("   → Training on combined data helps!")
else:
    print(f"\n⚠️ Model 2 (Combined) is {abs(improvement):.2f}% worse than Model 1")
    print("   → Training on your data alone might be better")
    print("   → QuantEmp data might be too different from yours")
