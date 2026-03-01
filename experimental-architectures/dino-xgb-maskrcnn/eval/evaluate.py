import os
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
try:
    _current_dir = Path(__file__).parent.parent
except NameError:
    _current_dir = Path(os.getcwd())
sys.path.insert(0, str(_current_dir))
from config import (
    FEATURES_DIR, EVALUATION_DIR,
    PATCH_CLASS_NAMES as CLASS_NAMES, PATCH_ID_TO_CLASS as ID_TO_CLASS, PATCH_NUM_CLASSES as NUM_CLASSES
)


def load_predictions():
    predictions_path = EVALUATION_DIR / "predictions.json"
    with open(predictions_path, 'r') as f:
        return json.load(f)

def generate_confusion_matrix_plots(predictions):
    print("Generating confusion matrix plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, split in enumerate(['train', 'val', 'test']):
        y_true = np.array(predictions[split]['y_true'])
        y_pred = np.array(predictions[split]['y_pred'])
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[idx])
        axes[idx].set_title(f'{split.capitalize()} Set (n={len(y_true)})', fontsize=12)
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    plt.tight_layout()
    plot_path = EVALUATION_DIR / "confusion_matrices.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrices saved to: {plot_path}")

def generate_metrics_comparison():
    print("IN-DEPTH ANALYSIS: ACTUAL VS PREDICTED")
    predictions = load_predictions()
    results = {}
    for split in ['train', 'val', 'test']:
        y_true = np.array(predictions[split]['y_true'])
        y_pred = np.array(predictions[split]['y_pred'])
        results[split] = {
            'accuracy': accuracy_score(y_true, y_pred) * 100,
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
            'f1_macro': f1_score(y_true, y_pred, average='macro') * 100,
        }
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        for i, class_name in enumerate(CLASS_NAMES):
            if i < len(recall_per_class):
                results[split][f'recall_{class_name}'] = recall_per_class[i] * 100
                results[split][f'precision_{class_name}'] = precision_per_class[i] * 100
    for metric in ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']:
        name = metric.replace('_', ' ').title()
        print(f"{name}: Train={results['train'][metric]:.2f}%, Val={results['val'][metric]:.2f}%, Test={results['test'][metric]:.2f}%")
    print("PER-CLASS RECALL:")
    for class_name in CLASS_NAMES:
        metric_key = f'recall_{class_name}'
        if metric_key in results['test']:
            print(f"  {class_name}: Train={results['train'][metric_key]:.2f}%, Val={results['val'][metric_key]:.2f}%, Test={results['test'][metric_key]:.2f}%")
    print("PER-CLASS PRECISION:")
    for class_name in CLASS_NAMES:
        metric_key = f'precision_{class_name}'
        if metric_key in results['test']:
            print(f"  {class_name}: Train={results['train'][metric_key]:.2f}%, Val={results['val'][metric_key]:.2f}%, Test={results['test'][metric_key]:.2f}%")
    return results

def generate_actual_vs_predicted_analysis(predictions):
    print("ACTUAL VS PREDICTED DEFECT COMPARISON (TEST SET)")
    y_true = np.array(predictions['test']['y_true'])
    y_pred = np.array(predictions['test']['y_pred'])
    for class_id, class_name in enumerate(CLASS_NAMES):
        actual_count = np.sum(y_true == class_id)
        predicted_count = np.sum(y_pred == class_id)
        correct = np.sum((y_true == class_id) & (y_pred == class_id))
        misclassified_as = {}
        actual_indices = y_true == class_id
        for other_id, other_name in enumerate(CLASS_NAMES):
            if other_id != class_id:
                count = np.sum(y_pred[actual_indices] == other_id)
                if count > 0:
                    misclassified_as[other_name] = count
        false_positives = {}
        predicted_indices = y_pred == class_id
        for other_id, other_name in enumerate(CLASS_NAMES):
            if other_id != class_id:
                count = np.sum(y_true[predicted_indices] == other_id)
                if count > 0:
                    false_positives[other_name] = count
        print(f"{class_name}: Actual={actual_count}, Predicted={predicted_count}, Correct={correct}")
        if misclassified_as:
            for other_name, count in misclassified_as.items():
                print(f"    -> Misclassified as {other_name}: {count}")
        if false_positives:
            for other_name, count in false_positives.items():
                print(f"    <- False positive from {other_name}: {count}")
    return results

def generate_analysis_report(results, predictions):
    report_path = EVALUATION_DIR / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("MULTI-CLASS ANOMALY DETECTION - ANALYSIS REPORT\n")
        f.write("DINOv2 (dinov2_vits14) + XGBoost\n")
        f.write("DATASET SUMMARY:\n")
        f.write(f"  Train samples: {len(predictions['train']['y_true'])}\n")
        f.write(f"  Val samples:   {len(predictions['val']['y_true'])}\n")
        f.write(f"  Test samples:  {len(predictions['test']['y_true'])}\n")
        f.write("EVALUATION METRICS:\n")
        for metric in ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']:
            name = metric.replace('_', ' ').title()
            f.write(f"{name}: Train={results['train'][metric]:.2f}%, Val={results['val'][metric]:.2f}%, Test={results['test'][metric]:.2f}%\n")
        f.write("Per-Class Recall:\n")
        for class_name in CLASS_NAMES:
            metric_key = f'recall_{class_name}'
            if metric_key in results['test']:
                f.write(f"  {class_name}: Train={results['train'][metric_key]:.2f}%, Val={results['val'][metric_key]:.2f}%, Test={results['test'][metric_key]:.2f}%\n")
        f.write("Per-Class Precision:\n")
        for class_name in CLASS_NAMES:
            metric_key = f'precision_{class_name}'
            if metric_key in results['test']:
                f.write(f"  {class_name}: Train={results['train'][metric_key]:.2f}%, Val={results['val'][metric_key]:.2f}%, Test={results['test'][metric_key]:.2f}%\n")
        y_true = np.array(predictions['test']['y_true'])
        y_pred = np.array(predictions['test']['y_pred'])
        cm = confusion_matrix(y_true, y_pred)
        f.write("CONFUSION MATRIX (TEST SET):\n")
        f.write("".rjust(12))
        for name in CLASS_NAMES:
            f.write(name.rjust(12))
        f.write(" (Predicted)\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(name.rjust(12))
            for j in range(len(CLASS_NAMES)):
                f.write(str(cm[i,j]).rjust(12))
            f.write("\n")
        f.write("(Actual)\n\n")
    print(f"Analysis report saved to: {report_path}")
    return report_path

def run_full_evaluation():
    print("COMPREHENSIVE EVALUATION")
    predictions = load_predictions()
    results = generate_metrics_comparison()
    generate_confusion_matrix_plots(predictions)
    generate_actual_vs_predicted_analysis(predictions)
    generate_analysis_report(results, predictions)
    print("EVALUATION COMPLETE")

if __name__ == "__main__":
    run_full_evaluation()
