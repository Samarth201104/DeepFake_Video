"""Evaluate frame-level and video-level classifier predictions."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from backend.app.vision.preprocessing import normalize_image

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "models" / "xception_best.h5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "evaluation")
    return parser.parse_args()


def plot_and_save_metrics(y_true, y_pred, y_prob, level_name, output_dir):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print(f"\n{level_name} metrics:")
    print(
        f"Accuracy: {accuracy:.4f} | Balanced accuracy: {balanced_accuracy:.4f} | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}"
    )
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))

    matrix = confusion_matrix(y_true, y_pred)
    for name, values, fmt in (
        ("counts", matrix, "d"),
        ("percent", matrix.astype(float) / matrix.sum(axis=1)[:, np.newaxis] * 100, ".2f"),
    ):
        plt.figure(figsize=(6, 5))
        sns.heatmap(values, annot=True, fmt=fmt, cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        plt.title(f"Confusion Matrix ({name}) - {level_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_{name}_{level_name}.png")
        plt.close()

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {level_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_curve_{level_name}.png")
    plt.close()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(str(args.checkpoint))
    print(f"Loaded model: {args.checkpoint}")
    generator = ImageDataGenerator(preprocessing_function=normalize_image).flow_from_directory(
        args.data_dir / "test",
        target_size=(256, 256),
        batch_size=16,
        class_mode="binary",
        shuffle=False,
        classes=["real", "fake"],
    )
    probabilities = model.predict(generator)
    predictions = (probabilities > 0.5).astype(int).flatten()
    labels = generator.classes
    filenames = np.asarray(generator.filenames)
    plot_and_save_metrics(labels, predictions, probabilities, "Frame-Level", args.output_dir)

    video_names = [Path(filename).name.rsplit("_0_", 1)[0] for filename in filenames]
    unique_videos = np.unique(video_names)
    video_probabilities, video_labels = [], []
    for video in unique_videos:
        mask = np.asarray(video_names) == video
        video_probabilities.append(np.mean(probabilities[mask]))
        video_labels.append(labels[mask][0])
    video_probabilities = np.asarray(video_probabilities)
    video_predictions = (video_probabilities > 0.5).astype(int)
    plot_and_save_metrics(video_labels, video_predictions, video_probabilities, "Video-Level", args.output_dir)


if __name__ == "__main__":
    main()
