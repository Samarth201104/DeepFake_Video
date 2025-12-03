# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import (
#     confusion_matrix, classification_report,
#     accuracy_score, precision_score, recall_score,
#     f1_score, roc_auc_score, roc_curve, balanced_accuracy_score
# )

# # =========================
# # Enable interactive mode in VS Code
# # =========================
# plt.ion()

# # =========================
# # Paths
# # =========================
# base_dir = "preprocessed_split"
# test_dir = os.path.join(base_dir, "test")  # should contain 'real' and 'fake' folders
# output_dir = "evaluation_results"
# os.makedirs(output_dir, exist_ok=True)

# # =========================
# # Load trained model
# # =========================
# model_path = "xception_best.h5"
# model = load_model(model_path)
# print(f"✅ Loaded model: {model_path}")

# # =========================
# # Prepare test data
# # =========================
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(256, 256),
#     batch_size=16,
#     class_mode='binary',
#     shuffle=False,
#     classes=['real', 'fake']
# )

# # =========================
# # Predictions
# # =========================
# print("\n🔍 Predicting on test data...")
# y_pred_prob = model.predict(test_generator)
# y_pred = (y_pred_prob > 0.5).astype(int).flatten()
# y_true = test_generator.classes

# # =========================
# # Metrics
# # =========================
# accuracy = accuracy_score(y_true, y_pred)
# bal_acc = balanced_accuracy_score(y_true, y_pred)
# precision = precision_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)
# roc_auc = roc_auc_score(y_true, y_pred_prob)

# print("\n📊 Evaluation Metrics:")
# print(f"Accuracy           : {accuracy:.4f}")
# print(f"Balanced Accuracy  : {bal_acc:.4f}")
# print(f"Precision          : {precision:.4f}")
# print(f"Recall             : {recall:.4f}")
# print(f"F1-Score           : {f1:.4f}")
# print(f"ROC-AUC            : {roc_auc:.4f}")

# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, target_names=['Real', 'Fake'], digits=4))

# # =========================
# # Confusion Matrix (Counts)
# # =========================
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
# plt.title('Confusion Matrix - Counts')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.tight_layout()
# counts_path = os.path.join(output_dir, "confusion_matrix_counts.png")
# plt.savefig(counts_path)
# print(f"✅ Confusion Matrix (Counts) saved at {counts_path}")
# plt.show()

# # =========================
# # Confusion Matrix (Percentage)
# # =========================
# cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
# plt.figure(figsize=(6,5))
# sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
#             xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
# plt.title('Confusion Matrix - Percentage')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.tight_layout()
# percent_path = os.path.join(output_dir, "confusion_matrix_percent.png")
# plt.savefig(percent_path)
# print(f"✅ Confusion Matrix (Percentage) saved at {percent_path}")
# plt.show()

# # =========================
# # ROC Curve
# # =========================
# fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
# plt.figure(figsize=(6,5))
# plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
# plt.plot([0, 1], [0, 1], color='red', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve - DeepFake Detection')
# plt.legend()
# plt.tight_layout()
# roc_path = os.path.join(output_dir, "roc_curve.png")
# plt.savefig(roc_path)
# print(f"✅ ROC Curve saved at {roc_path}")
# plt.show()

# # =========================
# # Keep plots interactive in VS Code
# # =========================
# plt.ioff()








import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, balanced_accuracy_score
)

# =========================
# Enable interactive mode in VS Code
# =========================
plt.ion()

# =========================
# Paths
# =========================
base_dir = "preprocessed_split"
test_dir = os.path.join(base_dir, "test")  # should contain 'real' and 'fake' folders
output_dir = "evaluation_results"
os.makedirs(output_dir, exist_ok=True)

# =========================
# Load trained model
# =========================
model_path = "xception_best.h5"
model = load_model(model_path)
print(f"✅ Loaded model: {model_path}")

# =========================
# Prepare test data
# =========================
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=16,
    class_mode='binary',
    shuffle=False,
    classes=['real', 'fake']
)

# =========================
# Frame-level Predictions
# =========================
print("\n🔍 Predicting on test data (frame-level)...")
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_generator.classes
filenames = np.array(test_generator.filenames)

# =========================
# Function: Plot & Save Metrics
# =========================
def plot_and_save_metrics(y_true, y_pred, y_prob, level_name):
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"\n📊 {level_name} Evaluation Metrics:")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Balanced Accuracy  : {bal_acc:.4f}")
    print(f"Precision          : {prec:.4f}")
    print(f"Recall             : {rec:.4f}")
    print(f"F1-Score           : {f1:.4f}")
    print(f"ROC-AUC            : {roc_auc:.4f}")

    print(f"\nClassification Report ({level_name}):")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake'], digits=4))

    # Confusion Matrix - Counts
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix (Counts) - {level_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    counts_path = os.path.join(output_dir, f"confusion_matrix_counts_{level_name}.png")
    plt.savefig(counts_path)
    plt.show()

    # Confusion Matrix - Percent
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
    plt.title(f'Confusion Matrix (Percentage) - {level_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    percent_path = os.path.join(output_dir, f"confusion_matrix_percent_{level_name}.png")
    plt.savefig(percent_path)
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {level_name}')
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(output_dir, f"roc_curve_{level_name}.png")
    plt.savefig(roc_path)
    plt.show()
    print(f"✅ Saved plots for {level_name} at {output_dir}\n")

# =========================
# Frame-Level Evaluation
# =========================
plot_and_save_metrics(y_true, y_pred, y_pred_prob, level_name="Frame-Level")

# =========================
# Video-Level Evaluation
# =========================
print("\n🎥 Aggregating predictions to video-level...")

# Extract video names from filenames based on your naming convention
# e.g., 01_02__outside_talking_still_laughing__YVGY8LOK.mp4_0_0.jpg -> 01_02__outside_talking_still_laughing__YVGY8LOK.mp4
video_names = [os.path.basename(f).rsplit("_0_", 1)[0] for f in filenames]

unique_videos = np.unique(video_names)
video_preds = []
video_true = []
video_probs = []

for vid in unique_videos:
    mask = np.array(video_names) == vid
    probs = y_pred_prob[mask]
    mean_prob = np.mean(probs)  # average probability for all frames in this video
    video_probs.append(mean_prob)
    video_preds.append(int(mean_prob > 0.5))
    video_true.append(int(y_true[mask][0]))  # all frames have the same true label

video_preds = np.array(video_preds)
video_true = np.array(video_true)
video_probs = np.array(video_probs)

# Video-level Evaluation
plot_and_save_metrics(video_true, video_preds, video_probs, level_name="Video-Level")

# =========================
# Keep plots interactive in VS Code
# =========================
plt.ioff()
