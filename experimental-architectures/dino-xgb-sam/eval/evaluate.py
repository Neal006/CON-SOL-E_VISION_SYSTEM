"""
Comprehensive Evaluation and Analysis for Multi-Class Anomaly Detection
Generates detailed comparison: Actual vs Predicted defects
Produces evaluation metrics and analysis report
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURES_DIR, EVALUATION_DIR,
    CLASS_NAMES, ID_TO_CLASS, NUM_CLASSES
)


def load_predictions():
    """Load predictions from JSON file."""
    predictions_path = EVALUATION_DIR / "predictions.json"
    with open(predictions_path, 'r') as f:
        return json.load(f)


def generate_confusion_matrix_plots(predictions):
    """Generate and save confusion matrix visualizations."""
    print(f"\nGenerating confusion matrix plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, split in enumerate(['train', 'val', 'test']):
        y_true = np.array(predictions[split]['y_true'])
        y_pred = np.array(predictions[split]['y_pred'])
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_normalized, 
            annot=cm,  # Show actual counts
            fmt='d',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=axes[idx],
            vmin=0, vmax=1
        )
        axes[idx].set_title(f'{split.capitalize()} Set\n(n={len(y_true)})', fontsize=12)
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    plt.tight_layout()
    plot_path = EVALUATION_DIR / "confusion_matrices.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrices to: {plot_path}")


def generate_metrics_comparison():
    """Generate comprehensive metrics comparison table."""
    print(f"\n{'='*80}")
    print("IN-DEPTH ANALYSIS: ACTUAL VS PREDICTED")
    print(f"{'='*80}")
    
    predictions = load_predictions()
    
    results = {}
    for split in ['train', 'val', 'test']:
        y_true = np.array(predictions[split]['y_true'])
        y_pred = np.array(predictions[split]['y_pred'])
        
        results[split] = {
            'accuracy': accuracy_score(y_true, y_pred) * 100,
            'recall_macro': recall_score(y_true, y_pred, average='macro') * 100,
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
            'f1_macro': f1_score(y_true, y_pred, average='macro') * 100,
        }
        
        # Per-class metrics
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(CLASS_NAMES):
            if i < len(recall_per_class):
                results[split][f'recall_{class_name}'] = recall_per_class[i] * 100
                results[split][f'precision_{class_name}'] = precision_per_class[i] * 100
    
    # Print comparison table
    print(f"\n{'Metric':<30} {'Train':>12} {'Validation':>12} {'Test':>12}")
    print("-" * 80)
    
    for metric in ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']:
        name = metric.replace('_', ' ').title()
        print(f"{name:<30} {results['train'][metric]:>11.1f}% {results['val'][metric]:>11.1f}% {results['test'][metric]:>11.1f}%")
    
    print("-" * 80)
    print("PER-CLASS RECALL:")
    for class_name in CLASS_NAMES:
        metric_key = f'recall_{class_name}'
        print(f"  {class_name:<28} {results['train'][metric_key]:>11.1f}% {results['val'][metric_key]:>11.1f}% {results['test'][metric_key]:>11.1f}%")
    
    print("-" * 80)
    print("PER-CLASS PRECISION:")
    for class_name in CLASS_NAMES:
        metric_key = f'precision_{class_name}'
        print(f"  {class_name:<28} {results['train'][metric_key]:>11.1f}% {results['val'][metric_key]:>11.1f}% {results['test'][metric_key]:>11.1f}%")
    
    print("=" * 80)
    
    return results


def generate_actual_vs_predicted_analysis(predictions):
    """Generate detailed actual vs predicted analysis for each class."""
    print(f"\n{'='*80}")
    print("ACTUAL VS PREDICTED DEFECT COMPARISON (TEST SET)")
    print(f"{'='*80}")
    
    y_true = np.array(predictions['test']['y_true'])
    y_pred = np.array(predictions['test']['y_pred'])
    
    for class_id, class_name in enumerate(CLASS_NAMES):
        actual_count = np.sum(y_true == class_id)
        predicted_count = np.sum(y_pred == class_id)
        correct = np.sum((y_true == class_id) & (y_pred == class_id))
        
        # What other classes were predicted for this actual class
        misclassified_as = {}
        actual_indices = y_true == class_id
        for other_id, other_name in enumerate(CLASS_NAMES):
            if other_id != class_id:
                count = np.sum(y_pred[actual_indices] == other_id)
                if count > 0:
                    misclassified_as[other_name] = count
        
        # What actual classes were predicted as this class
        false_positives = {}
        predicted_indices = y_pred == class_id
        for other_id, other_name in enumerate(CLASS_NAMES):
            if other_id != class_id:
                count = np.sum(y_true[predicted_indices] == other_id)
                if count > 0:
                    false_positives[other_name] = count
        
        print(f"\n{class_name.upper()}:")
        print(f"  Actual samples:     {actual_count}")
        print(f"  Predicted samples:  {predicted_count}")
        print(f"  Correctly classified: {correct} ({100*correct/actual_count if actual_count > 0 else 0:.1f}%)")
        
        if misclassified_as:
            print(f"  Misclassified as:")
            for other_name, count in misclassified_as.items():
                print(f"    → {other_name}: {count}")
        
        if false_positives:
            print(f"  False positives from:")
            for other_name, count in false_positives.items():
                print(f"    ← {other_name}: {count}")
    
    print("=" * 80)


def generate_analysis_report(results, predictions):
    """Generate a comprehensive analysis report."""
    report_path = EVALUATION_DIR / "analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-CLASS ANOMALY DETECTION - ANALYSIS REPORT\n")
        f.write("DINOv2 (dinov2_vits14) + XGBoost\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET SUMMARY:\n")
        f.write("-"*40 + "\n")
        f.write(f"  Train samples: {len(predictions['train']['y_true'])}\n")
        f.write(f"  Val samples:   {len(predictions['val']['y_true'])}\n")
        f.write(f"  Test samples:  {len(predictions['test']['y_true'])}\n")
        f.write(f"  Classes: {', '.join(CLASS_NAMES)}\n\n")
        
        f.write("EVALUATION METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<30} {'Train':>12} {'Validation':>12} {'Test':>12}\n")
        f.write("-"*80 + "\n")
        
        for metric in ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']:
            name = metric.replace('_', ' ').title()
            f.write(f"{name:<30} {results['train'][metric]:>11.1f}% {results['val'][metric]:>11.1f}% {results['test'][metric]:>11.1f}%\n")
        
        f.write("-"*80 + "\n")
        f.write("Per-Class Recall:\n")
        for class_name in CLASS_NAMES:
            metric_key = f'recall_{class_name}'
            f.write(f"  {class_name:<28} {results['train'][metric_key]:>11.1f}% {results['val'][metric_key]:>11.1f}% {results['test'][metric_key]:>11.1f}%\n")
        
        f.write("-"*80 + "\n")
        f.write("Per-Class Precision:\n")
        for class_name in CLASS_NAMES:
            metric_key = f'precision_{class_name}'
            f.write(f"  {class_name:<28} {results['train'][metric_key]:>11.1f}% {results['val'][metric_key]:>11.1f}% {results['test'][metric_key]:>11.1f}%\n")
        
        f.write("="*80 + "\n\n")
        
        # Confusion matrix for test set
        y_true = np.array(predictions['test']['y_true'])
        y_pred = np.array(predictions['test']['y_pred'])
        cm = confusion_matrix(y_true, y_pred)
        
        f.write("CONFUSION MATRIX (TEST SET):\n")
        f.write("-"*40 + "\n")
        f.write(f"{'':>12}")
        for name in CLASS_NAMES:
            f.write(f"{name:>12}")
        f.write("  (Predicted)\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{name:>12}")
            for j in range(len(CLASS_NAMES)):
                f.write(f"{cm[i,j]:>12}")
            f.write("\n")
        f.write("(Actual)\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n✓ Analysis report saved to: {report_path}")
    return report_path


def run_full_evaluation():
    """Run complete evaluation pipeline."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION")
    print(f"{'='*80}")
    
    # Load predictions
    predictions = load_predictions()
    
    # Generate metrics comparison
    results = generate_metrics_comparison()
    
    # Generate confusion matrix plots
    generate_confusion_matrix_plots(predictions)
    
    # Generate actual vs predicted analysis
    generate_actual_vs_predicted_analysis(predictions)
    
    # Generate analysis report
    generate_analysis_report(results, predictions)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_full_evaluation()
