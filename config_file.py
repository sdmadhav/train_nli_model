"""
Configuration file for training three RoBERTa models
Easily adjust k (number of evidences) and max_length here
"""

# ============================================================================
# KEY PARAMETERS - CHANGE THESE!
# ============================================================================

# Number of evidences to use per claim (k parameter)
# k=1 means use only first evidence
# k=2 means use first two evidences
# k=3 means use all three evidences
NUM_EVIDENCES = 1  # Change to 1, 2, or 3

# Maximum sequence length
# RoBERTa has a HARD LIMIT of 512 tokens (514 with special tokens)
# Cannot go higher than this - the model architecture doesn't support it
# If your sequences are longer, they will be automatically truncated
MAX_LENGTH = 512  # Must be <= 512 for RoBERTa

# ============================================================================
# FILE PATHS
# ============================================================================
PATHS = {
    # Your dataset files
    'our_train': 'our_train.json',
    'our_val': 'our_val.json',
    'our_test': 'our_test.json',
    
    # QuantEmp dataset files
    'quantemp_train': 'train_claimdecomp_evidence_question_mapping.json',
    'quantemp_val': 'val_claimdecomp_evidence_question_mapping.json',
    'quantemp_test': 'test_claimdecomp_evidence_question_mapping.json',
    
    # Output directories
    'model_dir': './models',
    'results_file': 'three_models_comparison.json',
}

# ============================================================================
# INPUT FORMAT
# ============================================================================
# The input format is:
# "[Claim]: {claim} [Questions]: {question1 question2 ...} [Evidences]: {evidence1 evidence2 ...}"
# 
# Example with k=2:
# "[Claim]: Brussels has free transport [Questions]: What bus exists? Where is it? [Evidences]: Public transport is free... There is a bus called..."

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
TRAINING_CONFIG = {
    # Basic settings
    'batch_size': 8,                # Reduce to 4 if out of memory, increase to 16 if you have good GPU
    'learning_rate': 2e-5,          # Typical range: 1e-5 to 5e-5
    'epochs': 5,                    # Number of training epochs (3-10 typical)
    'warmup_ratio': 0.1,            # Warmup steps as fraction of total steps
    
    # Regularization
    'weight_decay': 0.01,           # L2 regularization (0.0 to 0.1)
    'max_grad_norm': 1.0,           # Gradient clipping threshold
    
    # Random seed for reproducibility
    'seed': 42,
    
    # Evidence and length settings (from above)
    'num_evidences': NUM_EVIDENCES,
    'max_length': MAX_LENGTH
}

# ============================================================================
# PRESETS FOR DIFFERENT SCENARIOS
# ============================================================================

def get_config_for_k(k):
    """
    Get configuration for specific k value
    
    Args:
        k: Number of evidences to use (1, 2, or 3)
    """
    config = TRAINING_CONFIG.copy()
    config['num_evidences'] = k
    
    # Adjust max_length based on k
    # More evidences = need longer sequences
    if k == 1:
        config['max_length'] = 512  # Shorter is fine
    elif k == 2:
        config['max_length'] = 1024  # Medium length
    else:  # k == 3
        config['max_length'] = 1536  # Longer to fit all evidences
    
    return config


def get_memory_efficient_config():
    """Configuration for limited GPU memory (< 8GB)"""
    return {
        'batch_size': 4,
        'num_evidences': 2,         # Use fewer evidences
        'max_length': 512,          # Shorter sequences
        'epochs': 5,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'seed': 42
    }


def get_fast_test_config():
    """Quick configuration for testing the pipeline"""
    return {
        'batch_size': 16,
        'num_evidences': 1,         # Fastest with just 1 evidence
        'max_length': 256,          # Short sequences
        'epochs': 2,                # Quick training
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'seed': 42
    }


def get_best_performance_config():
    """Configuration for best performance (slower but better)"""
    return {
        'batch_size': 8,
        'num_evidences': 3,         # Use all evidences
        'max_length': 1536,         # Long sequences
        'epochs': 10,               # More training
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'seed': 42
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_config():
    """Print current configuration"""
    print("="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)
    
    print(f"\n🔢 Number of evidences (k): {NUM_EVIDENCES}")
    print(f"📏 Max sequence length: {MAX_LENGTH} tokens")
    print(f"📝 Input format: [Claim]: ... [Questions]: ... [Evidences]: ...")
    
    print("\n📁 File Paths:")
    for key, value in PATHS.items():
        print(f"  {key}: {value}")
    
    print("\n🎯 Training Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("="*70)


def estimate_sequence_length(k):
    """
    Estimate average sequence length based on number of evidences
    
    Args:
        k: Number of evidences (1, 2, or 3)
    
    Returns:
        Estimated average tokens needed
    """
    # Rough estimates based on your data
    avg_claim_tokens = 30
    avg_question_tokens = 15
    avg_evidence_tokens = 100
    
    # Format: [Claim]: claim [Questions]: q1 q2... [Evidences]: e1 e2...
    special_tokens = 10  # For [Claim], [Questions], [Evidences], etc.
    
    total = special_tokens + avg_claim_tokens + (k * avg_question_tokens) + (k * avg_evidence_tokens)
    
    return total


def recommend_max_length(k):
    """
    Recommend appropriate max_length based on k
    
    Args:
        k: Number of evidences
    """
    estimated = estimate_sequence_length(k)
    
    # Add 20% buffer
    recommended = int(estimated * 1.2)
    
    # Round to nice numbers
    if recommended <= 512:
        return 512
    elif recommended <= 1024:
        return 1024
    elif recommended <= 1536:
        return 1536
    else:
        return 2048


def check_config_validity():
    """Check if configuration is valid"""
    issues = []
    warnings = []
    
    # Check num_evidences
    if NUM_EVIDENCES < 1 or NUM_EVIDENCES > 3:
        issues.append(f"NUM_EVIDENCES should be 1, 2, or 3 (got {NUM_EVIDENCES})")
    
    # Check max_length
    if MAX_LENGTH < 128:
        issues.append("MAX_LENGTH too small (minimum 128)")
    
    # Check if max_length is appropriate for k
    recommended = recommend_max_length(NUM_EVIDENCES)
    if MAX_LENGTH < recommended * 0.8:
        warnings.append(f"MAX_LENGTH ({MAX_LENGTH}) might be too small for {NUM_EVIDENCES} evidences. Recommended: {recommended}")
    
    # Check batch size
    if TRAINING_CONFIG['batch_size'] < 1:
        issues.append("Batch size must be at least 1")
    
    # Check learning rate
    if not (1e-6 <= TRAINING_CONFIG['learning_rate'] <= 1e-3):
        warnings.append("Learning rate seems unusual (typical: 1e-5 to 5e-5)")
    
    # Print results
    if issues:
        print("❌ Configuration Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    if warnings:
        print("⚠️ Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("✅ Configuration is valid")
    
    return len(issues) == 0


def compare_k_values():
    """Compare different k values"""
    print("\n" + "="*70)
    print("COMPARISON: Different k values")
    print("="*70)
    
    for k in [1, 2, 3]:
        estimated = estimate_sequence_length(k)
        recommended = recommend_max_length(k)
        
        print(f"\nk = {k} evidence(s):")
        print(f"  Estimated avg tokens: ~{estimated}")
        print(f"  Recommended max_length: {recommended}")
        print(f"  Input format: [Claim] + {k} Questions + {k} Evidences")
        
        # Memory and speed estimates
        if k == 1:
            print(f"  Memory: Low")
            print(f"  Speed: Fast")
            print(f"  Information: Minimal (only first evidence)")
        elif k == 2:
            print(f"  Memory: Medium")
            print(f"  Speed: Medium")
            print(f"  Information: Good balance")
        else:  # k == 3
            print(f"  Memory: High")
            print(f"  Speed: Slower")
            print(f"  Information: Maximum (all evidences)")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ROBERTA CLAIM VERIFICATION - CONFIGURATION TOOL")
    print("="*70)
    
    # Show current config
    print_config()
    
    # Check validity
    print("\n" + "="*70)
    print("Configuration Validation:")
    print("="*70)
    check_config_validity()
    
    # Show recommendations
    print("\n" + "="*70)
    print("Recommendations:")
    print("="*70)
    recommended = recommend_max_length(NUM_EVIDENCES)
    current = MAX_LENGTH
    
    if current < recommended:
        print(f"⚠️ Your MAX_LENGTH ({current}) is less than recommended ({recommended})")
        print(f"   Some sequences might be truncated")
        print(f"   Consider increasing MAX_LENGTH to {recommended}")
    elif current > recommended * 1.5:
        print(f"ℹ️ Your MAX_LENGTH ({current}) is higher than needed (~{recommended})")
        print(f"   This is okay but uses more memory")
    else:
        print(f"✅ Your MAX_LENGTH ({current}) is appropriate for k={NUM_EVIDENCES}")
    
    # Compare different k values
    compare_k_values()
    
    # Show preset configs
    print("\n" + "="*70)
    print("AVAILABLE PRESETS:")
    print("="*70)
    
    print("\n1. Memory Efficient (for GPU < 8GB):")
    mem_config = get_memory_efficient_config()
    for key, value in mem_config.items():
        print(f"   {key}: {value}")
    
    print("\n2. Fast Test (for pipeline testing):")
    fast_config = get_fast_test_config()
    for key, value in fast_config.items():
        print(f"   {key}: {value}")
    
    print("\n3. Best Performance (slower but better):")
    best_config = get_best_performance_config()
    for key, value in best_config.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*70)
    print("QUICK START:")
    print("="*70)
    print("1. Adjust NUM_EVIDENCES (k) and MAX_LENGTH at the top of this file")
    print("2. Run: python train_three_models.py")
    print("3. Check results: three_models_comparison.json")
    print("="*70)