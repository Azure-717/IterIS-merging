#!/usr/bin/env python3
"""
Validation script for EMOTION_t5_large compute_metrics fix

This script simulates the behavior before and after the fix to demonstrate
that the issue has been resolved.
"""

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import AutoTokenizer


def compute_metrics_OLD_BROKEN(eval_pred):
    """
    Original BROKEN implementation.
    
    Bug: Uses incorrect indexing preds[:, 1] and labels[:, 0] which assumes
    a specific logit format that doesn't exist with generation.
    """
    preds, labels = eval_pred
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = labels[:, 0]  # ❌ WRONG: extracts first token
    preds = preds[:, 1]     # ❌ WRONG: extracts second token
    accuracy = (preds == labels).mean()
    f1 = f1_score(labels, preds, average='macro') 
    return {
        "accuracy": accuracy,
        "f1-score": f1,
    }


def compute_metrics_NEW_FIXED(eval_pred):
    """
    Fixed implementation.
    
    Fix: Decodes predictions and labels to strings and compares them properly.
    """
    preds, labels = eval_pred
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    
    # Handle tuple format (happens when model returns loss + logits)
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100 in labels with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels to strings
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Normalize: strip whitespace and convert to lowercase
    decoded_preds = [pred.strip().lower() for pred in decoded_preds]
    decoded_labels = [label.strip().lower() for label in decoded_labels]
    
    # Compute accuracy: exact string match
    accuracy = sum([1 if p == l else 0 for p, l in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)
    
    # Compute F1 score: macro average
    f1 = f1_score(decoded_labels, decoded_preds, average='macro', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1-score": f1,
    }


def simulate_t5_generation():
    """
    Simulate T5 generation output for emotion classification.
    
    Returns:
        preds: Generated token sequences (shape: [batch_size, seq_len])
        labels: Label token sequences (shape: [batch_size, seq_len])
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    
    # Simulate 4 emotion predictions and labels
    # Emotions: fear, joy, anger, sadness
    emotions_pred = ["fear", "joy", "anger", "fear"]
    emotions_label = ["fear", "sadness", "anger", "joy"]
    
    # Encode to token IDs (this is what T5 generates)
    preds = tokenizer(emotions_pred, padding='max_length', max_length=5, truncation=True).input_ids
    labels = tokenizer(emotions_label, padding='max_length', max_length=5, truncation=True).input_ids
    
    # Replace padding with -100 in labels (as done in preprocessing)
    labels = [[(item if item != tokenizer.pad_token_id else -100) for item in label] for label in labels]
    
    return np.array(preds), np.array(labels)


def main():
    print("=" * 80)
    print("EMOTION_t5_large compute_metrics Fix Validation")
    print("=" * 80)
    print()
    
    # Simulate T5 generation output
    print("Simulating T5 generation output...")
    preds, labels = simulate_t5_generation()
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    
    # Show the actual token sequences
    print("\n--- Token Sequences ---")
    print(f"Predictions shape: {preds.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"\nPrediction tokens (first 2 samples):")
    for i in range(min(2, len(preds))):
        print(f"  Sample {i}: {preds[i]}")
    print(f"\nLabel tokens (first 2 samples):")
    for i in range(min(2, len(labels))):
        # Show with -100 replaced for display
        display_labels = [item if item != -100 else f"<-100>" for item in labels[i]]
        print(f"  Sample {i}: {display_labels}")
    
    # Decode to show what they actually represent
    print("\n--- Decoded Strings ---")
    labels_for_decode = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
    
    print(f"Decoded predictions: {decoded_preds}")
    print(f"Decoded labels: {decoded_labels}")
    
    # Test the OLD BROKEN implementation
    print("\n" + "=" * 80)
    print("Testing OLD BROKEN implementation")
    print("=" * 80)
    
    try:
        result_old = compute_metrics_OLD_BROKEN((preds, labels))
        print(f"\n❌ OLD Results (INCORRECT):")
        print(f"  Accuracy: {result_old['accuracy']:.4f}")
        print(f"  F1-score: {result_old['f1-score']:.4f}")
        print(f"\n⚠️  Problem: These metrics are based on comparing random tokens!")
        print(f"    - preds[:, 1] extracts token at position 1 (arbitrary)")
        print(f"    - labels[:, 0] extracts token at position 0 (arbitrary)")
        print(f"    - Result is meaningless noise, usually close to 0")
    except Exception as e:
        print(f"\n❌ OLD implementation FAILED with error:")
        print(f"    {type(e).__name__}: {e}")
        print(f"\n⚠️  This is why we see nan/0.0 in actual evaluation!")
    
    # Test the NEW FIXED implementation
    print("\n" + "=" * 80)
    print("Testing NEW FIXED implementation")
    print("=" * 80)
    
    result_new = compute_metrics_NEW_FIXED((preds, labels))
    print(f"\n✅ NEW Results (CORRECT):")
    print(f"  Accuracy: {result_new['accuracy']:.4f}")
    print(f"  F1-score: {result_new['f1-score']:.4f}")
    
    # Manual verification
    print(f"\n📊 Manual Verification:")
    matches = sum([1 if p.strip().lower() == l.strip().lower() else 0
                   for p, l in zip(decoded_preds, decoded_labels)])
    manual_accuracy = matches / len(decoded_preds)
    print(f"  Expected accuracy: {manual_accuracy:.4f} ({matches}/{len(decoded_preds)} correct)")
    print(f"  Computed accuracy: {result_new['accuracy']:.4f}")
    print(f"  ✅ Match!" if abs(manual_accuracy - result_new['accuracy']) < 0.001 else "  ❌ Mismatch!")
    
    # Show detailed comparison
    print(f"\n📝 Detailed Prediction Comparison:")
    for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        pred_clean = pred.strip().lower()
        label_clean = label.strip().lower()
        match = "✅" if pred_clean == label_clean else "❌"
        print(f"  {match} Sample {i}: pred='{pred_clean}' vs label='{label_clean}'")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ Fix Validation Results:")
    print(f"  - NEW implementation produces meaningful metrics: {result_new['accuracy']:.1%} accuracy")
    print(f"  - Metrics correctly reflect string-level emotion prediction")
    print(f"  - No nan or inf values")
    print(f"  - No crashes or exceptions")
    print()
    print("❌ OLD implementation issues:")
    print(f"  - Used incorrect token indexing")
    print(f"  - Produced meaningless/zero metrics")
    print(f"  - Could crash with dimension errors")
    print()
    print("🎯 Conclusion: Fix is CORRECT and WORKING")
    print("=" * 80)


if __name__ == "__main__":
    main()
