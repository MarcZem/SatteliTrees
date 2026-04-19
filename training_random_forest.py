import glob
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings

# --- WARNUNGEN UNTERDRÜCKEN ---
warnings.filterwarnings('ignore', message='.*All-NaN.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom.*')

print("1. Suche alle Trainingskacheln...")
radd_files = glob.glob("data/makeathon-challenge/labels/train/radd/radd_*_labels.tif")
all_tiles = [os.path.basename(f).replace("radd_", "").replace("_labels.tif", "") for f in radd_files]

# --- TRAIN / VALIDATION SPLIT ---
np.random.seed(42)
np.random.shuffle(all_tiles)

TRAIN_SPLIT = 12
train_tiles = all_tiles[:TRAIN_SPLIT]
val_tiles = all_tiles[TRAIN_SPLIT:]

print(f"Gefunden: {len(all_tiles)} Kacheln insgesamt.")
print(f"-> Nutze {len(train_tiles)} Kacheln fürs Training.")
print(f"-> Nutze {len(val_tiles)} Kacheln für die Evaluation.\n")

# HIER IST DAS NEUE JAHR 2025 INTEGRIERT
AEF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
AEF_DIM = 64
trees_per_tile = 10

# Das Modell wird so konfiguriert, dass es in der Cloud alle Kerne nutzt (n_jobs=-1)
clf = RandomForestClassifier(
    n_estimators=0,
    warm_start=True,
    max_depth=20, 
    random_state=42, 
    n_jobs=-1
)

def read_and_aggregate_timeseries(file_paths, crs, transform, shape):
    """Liest alle Bilder eines Satelliten (egal wie viele) und berechnet Median & Schwankung"""
    if not file_paths: return None
    stack = []
    for f in file_paths:
        with rasterio.open(f) as src:
            reproj_data = np.zeros((src.count, shape[0], shape[1]), dtype=np.float32)
            reproject(
                source=src.read(), destination=reproj_data,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=crs,
                resampling=Resampling.bilinear
            )
            reproj_data[reproj_data == 0] = np.nan
            stack.append(reproj_data)
            
    if not stack: return None
    stack = np.array(stack)
    median_img = np.nanmedian(stack, axis=0)
    std_img = np.nanstd(stack, axis=0)
    
    return np.hstack([
        median_img.reshape(median_img.shape[0], -1).T,
        std_img.reshape(std_img.shape[0], -1).T
    ])

def extract_features_for_tile(tile, n_samples_per_class=25000):
    RADD_PATH = f"data/makeathon-challenge/labels/train/radd/radd_{tile}_labels.tif"
    GLADS2_PATH = f"data/makeathon-challenge/labels/train/glads2/glads2_{tile}_alert.tif"
    
    # Basis-Wahrheit (Ground Truth) aus RADD laden
    with rasterio.open(RADD_PATH) as src_radd:
        radd_data = src_radd.read(1)
        meta = src_radd.meta
        
    # Fallback-Logik: Falls GLADS2 existiert, verfeinern wir die Wahrheit, sonst nutzen wir nur RADD
    if os.path.exists(GLADS2_PATH):
        with rasterio.open(GLADS2_PATH) as src_glads2:
            glads2_reproj = np.zeros(radd_data.shape, dtype=np.uint8)
            reproject(
                source=src_glads2.read(1), destination=glads2_reproj,
                src_transform=src_glads2.transform, src_crs=src_glads2.crs,
                dst_transform=meta['transform'], dst_crs=meta['crs'],
                resampling=Resampling.nearest
            )
        y_tile = ((radd_data > 0) & (glads2_reproj > 0)).astype(np.uint8).flatten()
    else:
        print(f"  -> Hinweis: GLAD-S2 fehlt für {tile}. Nutze nur RADD.")
        y_tile = (radd_data > 0).astype(np.uint8).flatten()
    
    # Ausbalancieren (genauso viel Wald wie Entwaldung)
    idx_deforest = np.where(y_tile == 1)[0]
    idx_forest = np.where(y_tile == 0)[0]
    
    if len(idx_deforest) == 0: 
        print(f"  -> Keine Entwaldung in {tile} gefunden. Überspringe Kachel.")
        return None, None
        
    n_samples = min(len(idx_deforest), n_samples_per_class)
    sampled_indices = np.concatenate([
        np.random.choice(idx_deforest, size=n_samples, replace=False),
        np.random.choice(idx_forest, size=n_samples, replace=False)
    ])
    
    y_batch = y_tile[sampled_indices]
    features_batch = []
    
    # Feature 1: AlphaEarth Foundations (jetzt inkl. 2025)
    for year in AEF_YEARS:
        aef_path = f"data/makeathon-challenge/aef-embeddings/train/{tile}_{year}.tiff"
        if os.path.exists(aef_path):
            with rasterio.open(aef_path) as src_aef:
                aef_reproj = np.zeros((AEF_DIM, radd_data.shape[0], radd_data.shape[1]), dtype=np.float32)
                reproject(source=src_aef.read(), destination=aef_reproj, src_transform=src_aef.transform, src_crs=src_aef.crs, dst_transform=meta['transform'], dst_crs=meta['crs'], resampling=Resampling.bilinear)
            features_batch.append(aef_reproj.reshape(AEF_DIM, -1).T[sampled_indices])
        else:
            features_batch.append(np.zeros((len(sampled_indices), AEF_DIM)))

    # Feature 2 & 3: Sentinel-2 & Sentinel-1 (Alle Bilder im Ordner verarbeiten)
    s2_feat = read_and_aggregate_timeseries(glob.glob(f"data/makeathon-challenge/sentinel-2/train/{tile}__s2_l2a/*.tif"), meta['crs'], meta['transform'], radd_data.shape)
    features_batch.append(s2_feat[sampled_indices] if s2_feat is not None else np.zeros((len(sampled_indices), 8)))
    
    s1_feat = read_and_aggregate_timeseries(glob.glob(f"data/makeathon-challenge/sentinel-1/train/{tile}__s1_rtc/*.tif"), meta['crs'], meta['transform'], radd_data.shape)
    features_batch.append(s1_feat[sampled_indices] if s1_feat is not None else np.zeros((len(sampled_indices), 4)))

    X_batch = np.hstack(features_batch)
    X_batch = np.nan_to_num(X_batch, nan=0.0)
    return X_batch, y_batch

# ==========================================
# PHASE 1: TRAINING
# ==========================================
print("\n" + "="*50 + "\nPHASE 1: TRAINING\n" + "="*50)
for i, tile in enumerate(train_tiles):
    print(f"Trainiere an Kachel {i+1}/{len(train_tiles)}: {tile}")
    X_batch, y_batch = extract_features_for_tile(tile)
    
    if X_batch is not None:
        clf.n_estimators += trees_per_tile
        clf.fit(X_batch, y_batch)
        print(f"-> Modell hat nun {clf.n_estimators} Bäume.")

# ==========================================
# PHASE 2: VALIDATION (METRIKEN BERECHNEN)
# ==========================================
print("\n" + "="*50 + "\nPHASE 2: EVALUATION AUF UNBEKANNTEN DATEN\n" + "="*50)
all_y_true = []
all_y_pred = []

for i, tile in enumerate(val_tiles):
    print(f"Teste Kachel {i+1}/{len(val_tiles)}: {tile}")
    X_val, y_val = extract_features_for_tile(tile, n_samples_per_class=30000)
    
    if X_val is not None:
        y_pred = clf.predict(X_val)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

print("\n--- FINALE EVALUATION ---")
if all_y_true:
    print(classification_report(all_y_true, all_y_pred, target_names=["Intakter Wald (0)", "Entwaldung (1)"]))
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    print("\nKonfusionsmatrix (absolut):")
    print(f"Richtig Wald erkannt (True Negatives)    : {cm[0,0]}")
    print(f"Falscher Alarm (False Positives)         : {cm[0,1]}")
    print(f"Entwaldung übersehen (False Negatives)   : {cm[1,0]}")
    print(f"Richtig Entwaldung erkannt (True Pos.)   : {cm[1,1]}")
else:
    print("Nicht genügend Validierungsdaten gefunden.")

# Das Modell wird unter einem neuen Namen gespeichert, um das alte nicht zu überschreiben
FINAL_MODEL_NAME = "makeathon_rf_model_v3_with2025.joblib"
joblib.dump(clf, FINAL_MODEL_NAME)
print(f"\n🚀 Modell inklusive 2025er Daten trainiert und unter '{FINAL_MODEL_NAME}' gespeichert!")