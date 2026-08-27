from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR
    / "datasets"
    / "extracted"
    / "kadid10k"
    / "kadid10k"
)

CSV_PATH = DATASET_DIR / "dmos.csv"
IMAGES_DIR = DATASET_DIR / "images"

MODEL_DIR = BASE_DIR / "backend" / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(image_path: Path):
    """
    Extract simple image quality features.

    Returns:
        List of 5 numerical features, or None if the image
        cannot be loaded.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    # Resize for consistent feature extraction
    image = cv2.resize(image, (224, 224))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Brightness
    brightness = float(np.mean(gray))

    # 2. Contrast
    contrast = float(np.std(gray))

    # 3. Sharpness using Laplacian variance
    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    # 4. Saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))

    # 5. Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    return [
        brightness,
        contrast,
        sharpness,
        saturation,
        edge_density,
    ]


# -----------------------------
# Validate paths
# -----------------------------
print("Checking dataset paths...")

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV file not found:\n{CSV_PATH}"
    )

if not IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Images directory not found:\n{IMAGES_DIR}"
    )

print(f"CSV path: {CSV_PATH}")
print(f"Images path: {IMAGES_DIR}")


# -----------------------------
# Load dataset
# -----------------------------
print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Total images in CSV: {len(df)}")
print(f"CSV columns: {list(df.columns)}")

required_columns = {"dist_img", "dmos"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"CSV must contain columns: {required_columns}\n"
        f"Found columns: {list(df.columns)}"
    )


# -----------------------------
# Extract features
# -----------------------------
X = []
y = []

missing_images = 0
failed_images = 0

for position, (_, row) in enumerate(df.iterrows()):

    # Get distorted image name
    image_name = str(row["dist_img"])

    # Build full image path
    image_path = IMAGES_DIR / image_name

    # Check image exists
    if not image_path.exists():
        missing_images += 1

        if missing_images <= 5:
            print(
                f"Missing image: {image_path}"
            )

        continue

    # Extract image features
    features = extract_features(image_path)

    # Skip unreadable images
    if features is None:
        failed_images += 1

        if failed_images <= 5:
            print(
                f"Failed to process image: {image_path}"
            )

        continue

    # Store features and target score
    X.append(features)
    y.append(float(row["dmos"]))

    # Progress update
    if (position + 1) % 500 == 0:
        print(
            f"Processed {position + 1}/{len(df)} images"
        )


# -----------------------------
# Convert to NumPy arrays
# -----------------------------
X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.float64)

print("\nFeature extraction complete.")
print(f"Valid samples: {len(X)}")
print(f"Missing images: {missing_images}")
print(f"Failed feature extraction: {failed_images}")
print(f"Feature shape: {X.shape}")


# -----------------------------
# Validate extracted samples
# -----------------------------
if len(X) < 2:
    raise ValueError(
        "Not enough valid samples for training. "
        "Check dataset paths and image extraction."
    )

if len(X) < 10:
    print(
        "\nWarning: Very few valid samples were found. "
        "Check the dataset."
    )


# -----------------------------
# Train / test split
# -----------------------------
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# -----------------------------
# Train model
# -----------------------------
print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)


# -----------------------------
# Evaluation
# -----------------------------
print("\nEvaluating model...")

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions,
    )
)

r2 = r2_score(
    y_test,
    predictions,
)


print("\nModel Evaluation")
print("-" * 30)
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# -----------------------------
# Save model
# -----------------------------
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    {
        "model": model,
        "feature_names": [
            "brightness",
            "contrast",
            "sharpness",
            "saturation",
            "edge_density",
        ],
    },
    MODEL_PATH,
)


print("\nTraining completed successfully.")
print(f"Model saved to:\n{MODEL_PATH}")