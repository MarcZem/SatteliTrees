import os
import glob
import json
import torch
import joblib
import rasterio
import rasterio.features
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing
from rasterio.warp import reproject, Resampling, transform_geom
from shapely.geometry import shape, Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

warnings.filterwarnings('ignore')

print("🚀 STARTE PRODUCT-OF-EXPERTS ENSEMBLE (Multiplikativer Veto-Filter) 🚀")

# ==========================================
# 1. MODELLE LADEN
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class DeforestationUNet(nn.Module):
    def __init__(self, n_channels=154, n_classes=1):
        super(DeforestationUNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128) 
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x = self.up1(x3); x = torch.cat([x2, x], dim=1); x = self.conv_up1(x)
        x = self.up2(x); x = torch.cat([x1, x], dim=1); x = self.conv_up2(x)
        return self.outc(x)

print("Lade U-Net (.pth)...")
unet_model = DeforestationUNet(n_channels=154).to(device)
unet_model.load_state_dict(torch.load("unet_deforestation_model_v1.pth", map_location=device))
unet_model.eval()

print("Lade Random Forest (.joblib)...")
rf_model = joblib.load("makeathon_rf_model_v3_with2025.joblib")

# ==========================================
# 2. HELPER FUNKTIONEN
# ==========================================
def get_hanning_window(size):
    w = np.hanning(size)
    return np.outer(w, w)

def remove_small_components(mask, min_size=50):
    labeled_mask, num_features = ndimage.label(mask)
    if num_features == 0: return mask
    sizes = ndimage.sum(mask, labeled_mask, range(num_features + 1))
    mask_sizes = sizes < min_size
    remove_pixel = mask_sizes[labeled_mask]
    mask[remove_pixel] = 0
    return mask

def extract_wgs84_features(mask, src_crs, transform):
    shapes = rasterio.features.shapes(mask, transform=transform)
    features = []
    for geom, val in shapes:
        if val == 1:
            poly = shape(geom)
            if isinstance(poly, (Polygon, MultiPolygon)):
                try:
                    geom_wgs84 = transform_geom(src_crs, 'EPSG:4326', geom)
                    features.append({"type": "Feature", "geometry": geom_wgs84, "properties": {}})
                except Exception: pass
    return features

BASE_DIR = "/shared-docker/makeathon-challenge-2026/data/makeathon-challenge"

# ==========================================
# 3. KERN-INFERENZ FUNKTION
# ==========================================
def process_tile(tid, sub_folder, h, w, transform, src_crs):
    s2_f = sorted(glob.glob(f"{BASE_DIR}/sentinel-2/{sub_folder}/{tid}__s2_l2a/*.tif"))
    s1_f = sorted(glob.glob(f"{BASE_DIR}/sentinel-1/{sub_folder}/{tid}__s1_rtc/*.tif"))
    
    def load_m(p, t): 
        with rasterio.open(p) as s: return np.nan_to_num(s.read(out_shape=(s.count, t[0], t[1])).astype(np.float32), 0)
    def safe_med(files, t_shape, start, end):
        sub = files[start:end] if end != 0 else files[start:]
        if not sub: sub = files
        return np.median([load_m(f, t_shape) for f in sub], axis=0)

    s2_o = safe_med(s2_f, (h,w), 0, 3); s2_n = safe_med(s2_f, (h,w), -3, 0)
    s1_o = safe_med(s1_f, (h,w), 0, 3); s1_n = safe_med(s1_f, (h,w), -3, 0)
    
    aef_f = sorted(glob.glob(f"{BASE_DIR}/aef-embeddings/{sub_folder}/{tid}_*.tiff"))
    aef_o = safe_med(aef_f, (h,w), 0, 3); aef_n = safe_med(aef_f, (h,w), -3, 0)
    
    img_stack = np.concatenate([s1_o, s2_o, aef_o, s1_n, s2_n, aef_n], 0)
    for c in range(154): img_stack[c] = (img_stack[c] - img_stack[c].mean()) / (img_stack[c].std() + 1e-6)
    
    img_stack = torch.from_numpy(img_stack)
    unet_pred_full = np.zeros((h,w), dtype=np.float32); weights = np.zeros((h,w), dtype=np.float32)
    patch_size = 256; step = 128
    window = torch.from_numpy(get_hanning_window(patch_size)).to(device)
    
    # --- U-NET INFERENZ ---
    for y in range(0, max(1, h-patch_size+1), step):
        for x in range(0, max(1, w-patch_size+1), step):
            patch = img_stack[:, y:y+patch_size, x:x+patch_size]
            ph, pw = patch.shape[1], patch.shape[2]
            pad_h, pad_w = patch_size - ph, patch_size - pw
            if pad_h > 0 or pad_w > 0: patch = F.pad(patch, (0, pad_w, 0, pad_h))
            
            patch = patch.unsqueeze(0).to(device)
            with torch.no_grad(): out = torch.sigmoid(unet_model(patch)).squeeze() * window 
            
            unet_pred_full[y:y+ph, x:x+pw] += out[:ph, :pw].cpu().numpy()
            weights[y:y+ph, x:x+pw] += window[:ph, :pw].cpu().numpy()
            
    unet_probs = unet_pred_full / (weights + 1e-6)

    # --- RANDOM FOREST INFERENZ ---
    def get_rf_features(files, channels):
        if not files: return np.zeros((h * w, channels * 2))
        stack = []
        for f in files:
            with rasterio.open(f) as src:
                reproj_data = np.zeros((src.count, h, w), dtype=np.float32)
                reproject(source=src.read(), destination=reproj_data, src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=src_crs, resampling=Resampling.bilinear)
                reproj_data[reproj_data == 0] = np.nan
                stack.append(reproj_data)
        stack = np.array(stack)
        median_img = np.nanmedian(stack, axis=0).reshape(-1, h * w).T
        std_img = np.nanstd(stack, axis=0).reshape(-1, h * w).T
        return np.hstack([median_img, std_img])

    features_rf = []
    for year in [2020, 2021, 2022, 2023, 2024, 2025]:
        aef_p = f"{BASE_DIR}/aef-embeddings/{sub_folder}/{tid}_{year}.tiff"
        if os.path.exists(aef_p):
            with rasterio.open(aef_p) as src_aef:
                aef_reproj = np.zeros((64, h, w), dtype=np.float32)
                reproject(source=src_aef.read(), destination=aef_reproj, src_transform=src_aef.transform, src_crs=src_aef.crs, dst_transform=transform, dst_crs=src_crs, resampling=Resampling.bilinear)
            features_rf.append(aef_reproj.reshape(64, -1).T)
        else:
            features_rf.append(np.zeros((h * w, 64)))
            
    features_rf.append(get_rf_features(s2_f, 4))
    features_rf.append(get_rf_features(s1_f, 2))

    X_rf = np.nan_to_num(np.hstack(features_rf), nan=0.0)
    rf_probs_flat = np.zeros(h * w, dtype=np.float32)
    chunk_size = 50000
    for i in range(0, h * w, chunk_size):
        rf_probs_flat[i:i+chunk_size] = rf_model.predict_proba(X_rf[i:i+chunk_size])[:, 1]
    rf_probs = rf_probs_flat.reshape((h, w))

    # ========================================================
    # C: PRODUCT OF EXPERTS (Der mathematische Wolken-Killer)
    # ========================================================
    # Wir multiplizieren die Wahrscheinlichkeiten und ziehen die Wurzel!
    # Wenn ein Modell 0% sagt (z.B. RF sieht Wald im Radar), wird alles genullt.
    ensemble_probs = np.sqrt(unet_probs * rf_probs)
    
    # Schwellenwert auf 0.40 (Multiplikation zieht Werte natürlich nach unten)
    preds = (ensemble_probs > 0.40).astype(np.uint8)
    
    # 1. Formen glätten und Lücken schließen
    preds = binary_closing(preds, structure=np.ones((5,5))).astype(np.uint8)
    
    # 2. Gnadenloser 0.5 Hektar Filter (50 Pixel) gegen letztes Rauschen
    preds = remove_small_components(preds, min_size=50)

    return preds, s2_n

# ==========================================
# PHASE 1: EVALUIERUNG (TRAIN FOLDER)
# ==========================================
eval_tiles = ['47QQV_2_4', '18NWG_6_6', '48PUT_0_8', '18NWH_1_4', '48PWV_7_8']
ious = []

print("\n" + "="*50 + "\nPHASE 1: VISUELLE EVALUIERUNG\n" + "="*50)
for tid in eval_tiles:
    print(f"\n--- Evaluiere Kachel {tid} ---")
    sub = 'train'
    s2_f = sorted(glob.glob(f"{BASE_DIR}/sentinel-2/{sub}/{tid}__s2_l2a/*.tif"))
    with rasterio.open(s2_f[-1]) as src:
        h, w = src.shape; transform = src.transform; src_crs = src.crs
        
    preds, s2_n = process_tile(tid, sub, h, w, transform, src_crs)
    
    lab_p = glob.glob(f"{BASE_DIR}/labels/train/radd/radd_{tid}_labels.tif")[0]
    
    # Fix für Label-Größen
    with rasterio.open(lab_p) as s:
        lab = np.nan_to_num(s.read(1, out_shape=(h, w))).astype(np.float32) > 0
    
    inter = np.logical_and(lab, preds).sum()
    union = np.logical_or(lab, preds).sum()
    iou = inter/union if union > 0 else 0
    ious.append(iou)
    print(f"  ✅ IoU: {iou:.4f} (Gefunden: {preds.sum()}px | Echt: {lab.sum()}px)")

    # ----------------------------------------------------
    # DEINE KORREKTE VISUALISIERUNG (RGB 2, 1, 0)
    # ----------------------------------------------------
    rgb_bg = np.stack([s2_n[2], s2_n[1], s2_n[0]], axis=-1)
    rgb_bg = np.nan_to_num(rgb_bg, nan=0.0)

    valid_mask = (rgb_bg[..., 0] > 0) | (rgb_bg[..., 1] > 0) | (rgb_bg[..., 2] > 0)

    if np.any(valid_mask):
        p2, p98 = np.percentile(rgb_bg[valid_mask], (2, 98))
        rgb_bg = np.clip((rgb_bg - p2) / (p98 - p2 + 1e-6), 0, 1)

    rgb_bg[~valid_mask] = 0

    cmap_truth = ListedColormap(['none', 'red'])
    cmap_pred = ListedColormap(['none', 'yellow'])

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), dpi=120)

    axes[0].imshow(rgb_bg)
    axes[0].set_title(f"1. Normales Foto ({tid})", fontsize=14, fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(rgb_bg)
    axes[1].imshow(lab, cmap=cmap_truth, interpolation='nearest', alpha=0.6)
    axes[1].set_title("2. Ground Truth (Wahrheit = Rot)", fontsize=14, fontweight='bold')
    axes[1].axis("off")

    axes[2].imshow(rgb_bg)
    axes[2].imshow(preds, cmap=cmap_pred, interpolation='nearest', alpha=0.6)
    axes[2].set_title("3. Vorhersage (Geometrisches Mittel = Gelb)", fontsize=14, fontweight='bold')
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

print(f"\n🏆 DURCHSCHNITTLICHE EVAL IoU: {np.mean(ious):.4f} ({np.mean(ious)*100:.1f}%)")

# ==========================================
# PHASE 2: LEADERBOARD GEOJSON GENERIERUNG
# ==========================================
print("\n" + "="*50 + "\nPHASE 2: GENERIERE LEADERBOARD SUBMISSION\n" + "="*50)

output_dir = "final_submission_ready"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "FINAL_PRODUCT_OF_EXPERTS_SUBMISSION.geojson")

test_tiles = [os.path.basename(f).replace("__s2_l2a", "") for f in glob.glob(f"{BASE_DIR}/sentinel-2/test/*__s2_l2a")]
all_submission_features = []

for tid in test_tiles:
    print(f"Verarbeite Leaderboard-Kachel {tid}...")
    sub = 'test'
    s2_f = sorted(glob.glob(f"{BASE_DIR}/sentinel-2/{sub}/{tid}__s2_l2a/*.tif"))
    with rasterio.open(s2_f[-1]) as src:
        h, w = src.shape; transform = src.transform; src_crs = src.crs
        
    preds, _ = process_tile(tid, sub, h, w, transform, src_crs)
    
    features = extract_wgs84_features(preds, src_crs, transform)
    all_submission_features.extend(features)
    print(f"  ✅ {len(features)} saubere Polygone extrahiert.")

with open(output_file, 'w') as f:
    json.dump({"type": "FeatureCollection", "features": all_submission_features}, f)

print("\n" + "="*50)
print(f"🎉 FERTIG! Datei gespeichert unter:\n📂 {os.path.abspath(output_file)}")
print("="*50)