import os
import glob
import json
import torch
import joblib
import rasterio
import numpy as np
import torch.nn as nn
from scipy.ndimage import binary_opening, binary_closing, binary_dilation, label, sum as nd_sum
from rasterio.warp import transform_geom
from shapely.geometry import shape

# --- MODEL DEFINITION (U-NET) ---
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
        super().__init__()
        self.inc = DoubleConv(n_channels, 64); self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2); self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2); self.conv_up2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x = self.up1(x3); x = torch.cat([x2, x], dim=1); x = self.conv_up1(x)
        x = self.up2(x); x = torch.cat([x1, x], dim=1); x = self.conv_up2(x)
        return self.outc(x)

# --- UTILS ---
def remove_small_components(mask, min_size=50):
    labeled, num = label(mask)
    if num == 0: return mask
    sizes = nd_sum(mask, labeled, range(num + 1))
    mask[sizes[labeled] < min_size] = 0
    return mask

# --- MAIN INFERENCE ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = DeforestationUNet(n_channels=154).to(device)
unet.load_state_dict(torch.load("unet_deforestation_model_v1.pth", map_location=device))
unet.eval()
rf = joblib.load("makeathon_rf_model_final.joblib")

BASE_DIR = "/shared-docker/makeathon-challenge-2026/data/makeathon-challenge"
test_tiles = [os.path.basename(f).replace("__s2_l2a", "") for f in glob.glob(f"{BASE_DIR}/sentinel-2/test/*__s2_l2a")]

all_features = []

for tid in test_tiles:
    print(f"Processing {tid}...")
    # 1. Daten laden & Stack erstellen (S1, S2, AEF)
    # [Hier den Codeblock aus deiner Nachricht einfügen, der 'img_stack' und 'X_rf' baut]
    
    # 2. U-Net Prediction
    with torch.no_grad():
        unet_probs = torch.sigmoid(unet(torch.from_numpy(img_stack).unsqueeze(0).to(device))).squeeze().cpu().numpy()
    
    # 3. RF Prediction
    rf_probs = rf.predict_proba(X_rf)[:, 1].reshape(h, w)
    
    # 4. Ensemble (Asymmetrischer Filter)
    unet_mask = unet_probs > 0.45
    rf_aura = binary_dilation(rf_probs > 0.40, structure=np.ones((5,5)))
    final_mask = (unet_mask & rf_aura).astype(np.uint8)
    final_mask = remove_small_components(binary_opening(final_mask), 50)
    
    # 5. GeoJSON Export
    shapes = rasterio.features.shapes(final_mask, transform=transform)
    for geom, val in shapes:
        if val == 1:
            all_features.append({"type": "Feature", "geometry": transform_geom(src_crs, 'EPSG:4326', geom), "properties": {}})

with open("FINAL_SUBMISSION.geojson", "w") as f:
    json.dump({"type": "FeatureCollection", "features": all_features}, f)
print("Fertig!")