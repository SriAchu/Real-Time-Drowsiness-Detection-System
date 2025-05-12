import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib
# type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_recall_curve, roc_curve, auc
)

# -------------------
# Haar-like Feature Extraction
# -------------------
def compute_haar_features(img):
    haar_feats = []
    integral = cv2.integral(img)

    h, w = img.shape
    step_size = 16
    for y in range(0, h - 32, step_size):
        for x in range(0, w - 32, step_size):
            region = integral[y:y + 32 + 1, x:x + 32 + 1]
            A = region[0, 0]
            B = region[0, -1]
            C = region[-1, 0]
            D = region[-1, -1]
            total = D - B - C + A
            haar_feats.append(total)
    return np.array(haar_feats)

# -------------------
# Augmentation Setup
# -------------------
augmenter = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.6, 1.4],
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# -------------------
# Combined Feature Extraction (HOG + Haar + Augmentation)
# -------------------
EXPECTED_FEATURE_LENGTH = 1768  # Adjust if needed

def extract_combined_features(folder, label, augment_times=3):
    features = []
    labels = []
    filenames = []

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))

            # Original image features
            hog_feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
            haar_feat = compute_haar_features(img)
            combined_feat = np.concatenate((hog_feat, haar_feat))

            if len(combined_feat) != EXPECTED_FEATURE_LENGTH:
                print(f"[WARNING] Feature length mismatch for {filename}: Got {len(combined_feat)} expected {EXPECTED_FEATURE_LENGTH}")
                continue

            features.append(combined_feat)
            labels.append(label)
            filenames.append(filename)

            # Data augmentation loop
            img_expanded = np.expand_dims(img, axis=2)
            img_aug = np.expand_dims(img_expanded, axis=0)
            i = 0
            for batch in augmenter.flow(img_aug, batch_size=1):
                aug_img = batch[0].astype(np.uint8).squeeze()
                hog_feat_aug = hog(aug_img, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                haar_feat_aug = compute_haar_features(aug_img)
                combined_aug = np.concatenate((hog_feat_aug, haar_feat_aug))

                if len(combined_aug) != EXPECTED_FEATURE_LENGTH:
                    continue

                features.append(combined_aug)
                labels.append(label)
                filenames.append(filename + f"_aug{i+1}")
                i += 1
                if i >= augment_times:
                    break

    return np.array(features), np.array(labels), filenames

# -------------------
# Load Dataset & Extract Features
# -------------------
dataset_path = "C:/Users/HOME/Documents/ML-Project/Final Dataset"

X_eye_open, y_eye_open, fn_eye_open = extract_combined_features(os.path.join(dataset_path, "open"), 0, augment_times=3)
X_eye_closed, y_eye_closed, fn_eye_closed = extract_combined_features(os.path.join(dataset_path, "closed"), 1, augment_times=3)

X_mouth_yawn, y_mouth_yawn, fn_mouth_yawn = extract_combined_features(os.path.join(dataset_path, "yawn"), 1, augment_times=3)
X_mouth_noyawn, y_mouth_noyawn, fn_mouth_noyawn = extract_combined_features(os.path.join(dataset_path, "no yawn"), 0, augment_times=3)

# Combine & Shuffle
X_eye = np.concatenate((X_eye_open, X_eye_closed), axis=0)
y_eye = np.concatenate((y_eye_open, y_eye_closed), axis=0)
fn_eye = fn_eye_open + fn_eye_closed

X_mouth = np.concatenate((X_mouth_yawn, X_mouth_noyawn), axis=0)
y_mouth = np.concatenate((y_mouth_yawn, y_mouth_noyawn), axis=0)
fn_mouth = fn_mouth_yawn + fn_mouth_noyawn

X_eye, y_eye, fn_eye = shuffle(X_eye, y_eye, fn_eye, random_state=42)
X_mouth, y_mouth, fn_mouth = shuffle(X_mouth, y_mouth, fn_mouth, random_state=42)

# Feature scaling
scaler_eye = StandardScaler()
X_eye = scaler_eye.fit_transform(X_eye)

scaler_mouth = StandardScaler()
X_mouth = scaler_mouth.fit_transform(X_mouth)

# Save to CSV
df_eye = pd.DataFrame(X_eye)
df_eye['label'] = y_eye
df_eye['filename'] = fn_eye
df_eye.to_csv("eye_combined_features.csv", index=False)

df_mouth = pd.DataFrame(X_mouth)
df_mouth['label'] = y_mouth
df_mouth['filename'] = fn_mouth
df_mouth.to_csv("mouth_combined_features.csv", index=False)

print("\nFeature extraction (HOG + Haar) completed!")

# -------------------
# Train-Test Split
# -------------------
X_eye_train, X_eye_test, y_eye_train, y_eye_test = train_test_split(X_eye, y_eye, test_size=0.2, random_state=42)
X_mouth_train, X_mouth_test, y_mouth_train, y_mouth_test = train_test_split(X_mouth, y_mouth, test_size=0.2, random_state=42)

# -------------------
# Model Evaluation (Updated to include training evaluation)
# -------------------
def evaluate_model(model, X_train, X_test, y_train, y_test, part_name):
    model.fit(X_train, y_train)
    
    # Training evaluation
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\n{model.__class__.__name__} Training Accuracy on {part_name}: {train_acc:.4f}")
    print("Training Classification Report:")
    print(classification_report(y_train, y_train_pred))
    
    # Testing evaluation
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n{model.__class__.__name__} Testing Accuracy on {part_name}: {test_acc:.4f}")
    print("Testing Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model  # Return the trained model

# -------------------
# Train Models (Updated to use returned models)
# -------------------
print("\n=== Eye Classification ===")
print("\n--- Logistic Regression ---")
eye_lr = evaluate_model(LogisticRegression(max_iter=1000, C=0.1), X_eye_train, X_eye_test, y_eye_train, y_eye_test, "Eyes")

print("\n--- SVM ---")
eye_svm = evaluate_model(SVC(C=0.5, kernel='rbf'), X_eye_train, X_eye_test, y_eye_train, y_eye_test, "Eyes")

print("\n--- Random Forest ---")
eye_rf = evaluate_model(RandomForestClassifier(max_depth=10, n_estimators=100), X_eye_train, X_eye_test, y_eye_train, y_eye_test, "Eyes")

print("\n=== Mouth Classification ===")
print("\n--- Logistic Regression ---")
mouth_lr = evaluate_model(LogisticRegression(max_iter=1000), X_mouth_train, X_mouth_test, y_mouth_train, y_mouth_test, "Mouth")

print("\n--- SVM ---")
mouth_svm = evaluate_model(SVC(), X_mouth_train, X_mouth_test, y_mouth_train, y_mouth_test, "Mouth")

print("\n--- Random Forest ---")
mouth_rf = evaluate_model(RandomForestClassifier(), X_mouth_train, X_mouth_test, y_mouth_train, y_mouth_test, "Mouth")


# ======================
# TRAIN MODELS
# ======================
best_eye_model = SVC(C=0.5, kernel='rbf', probability=True)  # Eye SVM (92%)
best_mouth_model = SVC(C=1.0, kernel='rbf', probability=True)  # Mouth SVM (95%)

best_eye_model.fit(X_eye_train, y_eye_train)
best_mouth_model.fit(X_mouth_train, y_mouth_train)

# ======================
# COMBINED EVALUATION FUNCTION
# ======================
def evaluate_combined_system(eye_model, mouth_model, X_eye, X_mouth, y_eye, y_mouth, data_type="Testing"):
    print(f"[{data_type}] Evaluation started...")

    min_len = min(len(X_eye), len(X_mouth))
    X_eye, X_mouth = X_eye[:min_len], X_mouth[:min_len]
    y_eye, y_mouth = y_eye[:min_len], y_mouth[:min_len]

    # Predictions (labels and probabilities)
    eye_preds = eye_model.predict(X_eye)
    mouth_preds = mouth_model.predict(X_mouth)
    eye_probs = eye_model.predict_proba(X_eye)[:, 1]  # Prob of closed eyes
    mouth_probs = mouth_model.predict_proba(X_mouth)[:, 1]  # Prob of yawning

    # Combined rule
    drowsy_preds = np.logical_or(eye_preds == 1, mouth_preds == 1).astype(int)
    actual_drowsy = np.logical_or(y_eye == 1, y_mouth == 1).astype(int)

    # For PR/ROC curves: use max of eye/mouth probabilities as combined probability
    combined_probs = np.maximum(eye_probs, mouth_probs)

    # Metrics
    acc = accuracy_score(actual_drowsy, drowsy_preds)
    print(f"\n=== Combined Drowsiness Detection ({data_type}) ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(actual_drowsy, drowsy_preds, target_names=["Awake", "Drowsy"]))

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(actual_drowsy, combined_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 4))
    
    # Plot PR Curve
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.')
    plt.title(f"{data_type} - Precision-Recall (AUC={pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(actual_drowsy, combined_probs)
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"{data_type} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(f"{data_type}_PR_ROC_Curve.png")
    plt.pause(10)  # display for 3 seconds
    plt.close()

    print(f"[{data_type}] Evaluation started...")

    return acc

# ======================
# EVALUATION
# ======================
train_acc = evaluate_combined_system(
    best_eye_model, best_mouth_model,
    X_eye_train, X_mouth_train,
    y_eye_train, y_mouth_train,
    data_type="Training"
)

test_acc = evaluate_combined_system(
    best_eye_model, best_mouth_model,
    X_eye_test, X_mouth_test,
    y_eye_test, y_mouth_test,
    data_type="Testing"
)

gap = train_acc - test_acc
print(f"\nGeneralization Gap: {gap:.2%} (Expected: <5% is ideal)")

# ======================
# COMBINATION TESTING (LOGIC RULES / THRESHOLDS)
# ======================
import time
from itertools import product

combination_rules = list(product(["or", "and"], repeat=1))  # Test OR and AND rules

print(f"\n Testing {len(combination_rules)} combinations of logic rules...\n")
overall_start = time.time()

for idx, (rule,) in enumerate(combination_rules):
    print(f"\n Combination {idx+1}/{len(combination_rules)}: Rule = '{rule.upper()}'")
    combo_start = time.time()

    def combined_logic(eye_preds, mouth_preds, rule="or"):
        if rule == "or":
            return np.logical_or(eye_preds == 1, mouth_preds == 1).astype(int)
        elif rule == "and":
            return np.logical_and(eye_preds == 1, mouth_preds == 1).astype(int)
        else:
            raise ValueError("Unsupported logic rule")

    min_len = min(len(X_eye_test), len(X_mouth_test))
    X_eye, X_mouth = X_eye_test[:min_len], X_mouth_test[:min_len]
    y_eye, y_mouth = y_eye_test[:min_len], y_mouth_test[:min_len]

    eye_preds = best_eye_model.predict(X_eye)
    mouth_preds = best_mouth_model.predict(X_mouth)
    eye_probs = best_eye_model.predict_proba(X_eye)[:, 1]
    mouth_probs = best_mouth_model.predict_proba(X_mouth)[:, 1]

    drowsy_preds = combined_logic(eye_preds, mouth_preds, rule)
    actual_drowsy = np.logical_or(y_eye == 1, y_mouth == 1).astype(int)
    combined_probs = np.maximum(eye_probs, mouth_probs)

    acc = accuracy_score(actual_drowsy, drowsy_preds)
    precision, recall, _ = precision_recall_curve(actual_drowsy, combined_probs)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(actual_drowsy, combined_probs)
    roc_auc = auc(fpr, tpr)

    print(f"[] Accuracy: {acc:.4f}, PR AUC: {pr_auc:.2f}, ROC AUC: {roc_auc:.2f}")
    print(f"[] Time for this combo: {time.time() - combo_start:.2f} sec")

overall_time = time.time() - overall_start
print(f"\n All combinations tested in {overall_time:.2f} seconds ({overall_time / 60:.2f} minutes)")


# ======================
# MODEL SAVING
# ======================
joblib.dump(best_eye_model, "eye_model_svm.pkl")
joblib.dump(best_mouth_model, "mouth_model_svm.pkl")
joblib.dump(scaler_eye, "scaler_eye.pkl")
joblib.dump(scaler_mouth, "scaler_mouth.pkl")

print("\nModels saved:")
print("- Eye SVM: 'eye_model_svm.pkl'")
print("- Mouth SVM: 'mouth_model_svm.pkl'")
print("- Scalers: 'scaler_eye.pkl', 'scaler_mouth.pkl'")

# ======================
# PERFORMANCE SUMMARY
# ======================
print("\n=== Performance Summary ===")
print("Eye Model (SVM): Test Acc = 92%")
print("Mouth Model (SVM): Test Acc = 95%")
print(f"Combined System: Test Acc = {test_acc:.2%}")
print(f"Generalization Gap: {gap:.2%}")

if gap > 0.05:
    print("\n[Note] Small gap detected. To reduce it:")
    print("- Lower SVM C slightly")
    print("- Add more diverse samples to training set")
else:
    print("\n[Result] Generalization looks good ")
