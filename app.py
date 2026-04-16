from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os

app = Flask(__name__)

# 上传目录
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 分类类别
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

# 模型
model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# 图片处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            img = Image.open(path).convert("RGB")
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(img)
                pred = output.argmax(1).item()

            result = class_names[pred]
            img_path = path

    return render_template("index.html", result=result, img_path=img_path)

# 🚀 云部署关键（只改这一行！）
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))