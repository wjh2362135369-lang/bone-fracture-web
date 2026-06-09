from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import os
import io
import base64
import time
import uuid
import gc
import numpy as np

# Limit CPU threads — fewer threads = lower peak memory on small instances
torch.set_num_threads(1)

app = Flask(__name__)

# ---- Upload dir (kept) ----
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---- Classes (kept) ----
class_names = [
    "Avulsion fracture",
    "Comminuted fracture",
    "Fracture Dislocation",
    "Greensstick fracture",
    "Hairline Fracture",
    "Impacted fracture",
    "Longitudinal fracture",
    "Oblique fracture",
    "Pathological fracture",
    "Spiral Fracture"
]

# ---- Model (kept) ----
model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ---- Transform (kept) ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============================================================
# Grad-CAM utilities (NEW) — for an EfficientNet classifier we
# DO NOT fabricate detection boxes. We compute a Grad-CAM heatmap
# and derive an "Approximate Lesion Localization" region from it.
# ============================================================
_target_layer = model.conv_head  # last conv block in timm efficientnet_b0
_fmap, _grad = {}, {}

def _fwd_hook(_m, _i, o): _fmap['v'] = o.detach()
def _bwd_hook(_m, _gi, go): _grad['v'] = go[0].detach()

_target_layer.register_forward_hook(_fwd_hook)
_target_layer.register_full_backward_hook(_bwd_hook)


def gradcam(input_tensor, class_idx):
    """Return a HxW heatmap in [0,1] aligned to the 224x224 input."""
    model.zero_grad()
    out = model(input_tensor)
    score = out[0, class_idx]
    score.backward(retain_graph=False)

    fmap = _fmap['v'][0]            # C,h,w
    grad = _grad['v'][0]            # C,h,w
    weights = grad.mean(dim=(1, 2)) # C
    cam = (weights[:, None, None] * fmap).sum(0)
    cam = torch.relu(cam)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    cam = cam.cpu().numpy()
    # upsample to 224
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    return np.asarray(cam_img, dtype=np.float32) / 255.0, out


def colorize_jet(gray):
    """Lightweight jet colormap (no matplotlib)."""
    g = np.clip(gray, 0, 1)
    r = np.clip(1.5 - np.abs(4 * g - 3), 0, 1)
    gC = np.clip(1.5 - np.abs(4 * g - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * g - 1), 0, 1)
    rgb = np.stack([r, gC, b], axis=-1) * 255
    return rgb.astype(np.uint8)


def encode_png(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def approx_bbox_from_cam(cam, thresh=0.55):
    """Largest connected high-activation region -> bbox (normalized)."""
    mask = cam >= thresh
    ys, xs = np.where(mask)
    if len(xs) < 10:
        # fallback: argmax-centered box
        y, x = np.unravel_index(np.argmax(cam), cam.shape)
        h, w = cam.shape
        side = int(min(h, w) * 0.25)
        x0, y0 = max(0, x - side), max(0, y - side)
        x1, y1 = min(w, x + side), min(h, y + side)
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
    h, w = cam.shape
    return {"x": float(x0 / w), "y": float(y0 / h),
            "w": float((x1 - x0) / w), "h": float((y1 - y0) / h)}


def region_text(bbox):
    cx, cy = bbox["x"] + bbox["w"] / 2, bbox["y"] + bbox["h"] / 2
    v = "上" if cy < 0.34 else ("中" if cy < 0.67 else "下")
    h = "左" if cx < 0.34 else ("中" if cx < 0.67 else "右")
    return f"{v}{h}区域"


def risk_level(conf):
    if conf >= 0.85: return "High"
    if conf >= 0.6:  return "Medium"
    return "Low"


# ============================================================
# Routes
# ============================================================
@app.route("/", methods=["GET", "POST"])
def index():
    # Legacy POST flow kept so old clients still work.
    result, img_path = None, None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0)
            with torch.no_grad():
                out = model(x)
                pred = out.argmax(1).item()
            result = class_names[pred]
            img_path = path
    return render_template("index.html", result=result, img_path=img_path)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """NEW: returns prediction + confidence + heatmap + approx bbox."""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file"}), 400

    try:
        fname = f"{uuid.uuid4().hex}_{file.filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(save_path)

        pil = Image.open(save_path).convert("RGB").resize((224, 224))
        x = transform(pil).unsqueeze(0)
        x.requires_grad_(True)

        # Forward (no grad) for probs
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        # Grad-CAM (needs grad)
        cam, _ = gradcam(x, pred)
        bbox = approx_bbox_from_cam(cam)

        # Build images
        orig_arr = np.asarray(pil, dtype=np.uint8)
        heat_rgb = colorize_jet(cam)
        overlay = (0.55 * orig_arr + 0.45 * heat_rgb).clip(0, 255).astype(np.uint8)

        orig_b64 = encode_png(Image.fromarray(orig_arr))
        heat_b64 = encode_png(Image.fromarray(heat_rgb))
        overlay_b64 = encode_png(Image.fromarray(overlay))

        response = jsonify({
            "label": class_names[pred],
            "confidence": round(conf * 100, 2),
            "risk": risk_level(conf),
            "region": region_text(bbox),
            "bbox": bbox,                  # normalized 0..1
            "timestamp": int(time.time()),
            "model": "EfficientNet-B0 (Classification + Grad-CAM)",
            "localization_note": "Approximate Lesion Localization (Grad-CAM derived)",
            "images": {
                "original": orig_b64,
                "heatmap": heat_b64,
                "overlay": overlay_b64
            },
            "probs": [{"label": class_names[i], "p": round(float(p) * 100, 2)} for i, p in enumerate(probs)]
        })

        # Free heavy tensors / graph before returning
        del x, logits, cam
        model.zero_grad(set_to_none=True)
        gc.collect()

        return response

    except Exception as e:
        import traceback
        traceback.print_exc()          # shows up in Render logs
        model.zero_grad(set_to_none=True)
        gc.collect()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
