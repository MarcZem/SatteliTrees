import glob
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore', message='.*All-NaN.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom.*')

# 1. Wir nehmen jetzt absichtlich eine extrem schwierige, wolkige Tropen-Kachel aus Asien!
TILE_ID = "48PWV_7_8"  
MODEL_PATH = "makeathon_rf_model_v3_with2025.joblib"
AEF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
AEF_DIM = 64
CHUNK_SIZE = 500_000

print(f"Lade Modell '{MODEL_PATH}'...")
clf = joblib.load(MODEL_PATH)

def read_and_aggregate_timeseries(file_paths, crs, transform, shape):
    if not file_paths: return None
    stack = []
    for f in file_paths:
        with rasterio.open(f) as src:
            reproj_data = np.zeros((src.count, shape[0], shape[1]), dtype=np.float32)
            reproject(source=src.read(), destination=reproj_data, src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=crs, resampling=Resampling.bilinear)
            reproj_data[reproj_data == 0] = np.nan
            stack.append(reproj_data)
    if not stack: return None
    stack = np.array(stack)
    return np.nanmedian(stack, axis=0), np.nanstd(stack, axis=0)

print(f"Lade Ground Truth für Kachel {TILE_ID}...")
RADD_PATH = f"data/makeathon-challenge/labels/train/radd/radd_{TILE_ID}_labels.tif"
GLADS2_PATH = f"data/makeathon-challenge/labels/train/glads2/glads2_{TILE_ID}_alert.tif"

with rasterio.open(RADD_PATH) as src_radd:
    radd_data = src_radd.read(1)
    target_meta = src_radd.meta.copy()
    target_shape = (src_radd.shape[0], src_radd.shape[1])

if os.path.exists(GLADS2_PATH):
    with rasterio.open(GLADS2_PATH) as src_glads2:
        glads2_reproj = np.zeros(target_shape, dtype=np.uint8)
        reproject(source=src_glads2.read(1), destination=glads2_reproj, src_transform=src_glads2.transform, src_crs=src_glads2.crs, dst_transform=target_meta['transform'], dst_crs=target_meta['crs'], resampling=Resampling.nearest)
    ground_truth_img = ((radd_data > 0) & (glads2_reproj > 0)).astype(np.uint8)
else:
    ground_truth_img = (radd_data > 0).astype(np.uint8)

print("Extrahiere Features...")
features_batch = []

# AEF
for year in AEF_YEARS:
    aef_path = f"data/makeathon-challenge/aef-embeddings/train/{TILE_ID}_{year}.tiff"
    if os.path.exists(aef_path):
        with rasterio.open(aef_path) as src_aef:
            aef_reproj = np.zeros((AEF_DIM, target_shape[0], target_shape[1]), dtype=np.float32)
            reproject(source=src_aef.read(), destination=aef_reproj, src_transform=src_aef.transform, src_crs=src_aef.crs, dst_transform=target_meta['transform'], dst_crs=target_meta['crs'], resampling=Resampling.bilinear)
        features_batch.append(aef_reproj.reshape(AEF_DIM, -1).T)
    else:
        features_batch.append(np.zeros((target_shape[0]*target_shape[1], AEF_DIM)))

# Sentinel-2
s2_files = glob.glob(f"data/makeathon-challenge/sentinel-2/train/{TILE_ID}__s2_l2a/*.tif")
s2_median, s2_std = read_and_aggregate_timeseries(s2_files, target_meta['crs'], target_meta['transform'], target_shape)
features_batch.append(np.hstack([s2_median.reshape(s2_median.shape[0], -1).T, s2_std.reshape(s2_std.shape[0], -1).T]))

# Sentinel-1
s1_files = glob.glob(f"data/makeathon-challenge/sentinel-1/train/{TILE_ID}__s1_rtc/*.tif")
s1_median, s1_std = read_and_aggregate_timeseries(s1_files, target_meta['crs'], target_meta['transform'], target_shape)
features_batch.append(np.hstack([s1_median.reshape(s1_median.shape[0], -1).T, s1_std.reshape(s1_std.shape[0], -1).T]))

X_test = np.hstack(features_batch)
X_test = np.nan_to_num(X_test, nan=0.0)

total_pixels = X_test.shape[0]
print("Modell berechnet Vorhersagen...")
predictions = np.zeros(total_pixels, dtype=np.uint8)
for start_idx in range(0, total_pixels, CHUNK_SIZE):
    end_idx = min(start_idx + CHUNK_SIZE, total_pixels)
    predictions[start_idx:end_idx] = clf.predict(X_test[start_idx:end_idx])

prediction_img = predictions.reshape(target_shape)

# ==========================================
# NEU: METRIKEN FÜR DIESES SPEZIFISCHE BILD
# ==========================================
print("\n" + "="*50)
print(f"STATISTIK FÜR KACHEL {TILE_ID}")
print("="*50)
# Wir vergleichen das 1D-Array der echten Werte mit dem 1D-Array der Vorhersagen
gt_flat = ground_truth_img.flatten()
pred_flat = prediction_img.flatten()

acc = accuracy_score(gt_flat, pred_flat)
print(f"Overall Accuracy: {acc*100:.2f}%\n")
print(classification_report(gt_flat, pred_flat, target_names=["Kein Event (0)", "Entwaldung (1)"], zero_division=0))


# ==========================================
# VISUALISIERUNG (TRUE COLOR - GOOGLE MAPS STYLE)
# ==========================================
print("Erstelle Visualisierung...")

# 1. Echte Farben (True Color): Rot, Grün und Blau (Meist Index 2, 1, 0 bei Sentinel-2)
# Falls die Farben immer noch leicht falsch sind, probiere: np.stack([s2_median[0], s2_median[1], s2_median[2]]...)
rgb_bg = np.stack([s2_median[2], s2_median[1], s2_median[0]], axis=-1)
rgb_bg = np.nan_to_num(rgb_bg, nan=0.0)

# 2. Den Leak-Effekt fixen: Wir berechnen den Kontrast NUR auf den echten Pixeln, nicht auf den Rändern!
valid_mask = (rgb_bg[..., 0] > 0) | (rgb_bg[..., 1] > 0) | (rgb_bg[..., 2] > 0)

# Schöne Kontrast-Streckung (Ignoriert grelle Wolken und tiefe Schatten)
if np.any(valid_mask):
    p2, p98 = np.percentile(rgb_bg[valid_mask], (2, 98))
    rgb_bg = np.clip((rgb_bg - p2) / (p98 - p2), 0, 1)

# Leere Ränder sauber auf Schwarz (oder Weiß) setzen
rgb_bg[~valid_mask] = 0

cmap_truth = ListedColormap(['none', 'red'])
cmap_pred = ListedColormap(['none', 'yellow'])

fig, axes = plt.subplots(1, 3, figsize=(20, 8), dpi=120)

axes[0].imshow(rgb_bg)
axes[0].set_title(f"1. Normales Foto ({TILE_ID})", fontsize=14, fontweight='bold')
axes[0].axis("off")

axes[1].imshow(rgb_bg)
axes[1].imshow(ground_truth_img, cmap=cmap_truth, interpolation='nearest', alpha=0.6)
axes[1].set_title("2. Ground Truth (Wahrheit = Rot)", fontsize=14, fontweight='bold')
axes[1].axis("off")

axes[2].imshow(rgb_bg)
axes[2].imshow(prediction_img, cmap=cmap_pred, interpolation='nearest', alpha=0.6)
axes[2].set_title("3. Vorhersage (Modell = Gelb)", fontsize=14, fontweight='bold')
axes[2].axis("off")

plt.tight_layout()
plt.show()