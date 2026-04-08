from flask import Flask, render_template, request
import os
import shutil
import torch
import torch.nn.functional as F
import requests
import gc

app = Flask(__name__)

UPLOAD_DIR = 'static/uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= DEVICE =================
device = torch.device("cpu")

# ================= OPTIONAL IMPORTS =================
MONAI_AVAILABLE = True

try:
    import dicom2nifti
    import dicom2nifti.settings as settings
except Exception as e:
    print("❌ DICOM import error:", e)
    MONAI_AVAILABLE = False

try:
    from monai.networks.nets import DenseNet121
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, Orientationd
except Exception as e:
    print("❌ MONAI import error:", e)
    MONAI_AVAILABLE = False

# ================= DOWNLOAD MODEL =================
MODEL_URL = "https://huggingface.co/pujitha15/medio/resolve/main/model.pth"

if not os.path.exists("model.pth"):
    print("⬇️ Downloading model from Hugging Face...")
    try:
        r = requests.get(MODEL_URL, timeout=60)
        with open("model.pth", "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded")
    except Exception as e:
        print("❌ Download failed:", e)

# ================= MODEL =================
model = None

def load_model():
    global model
    if model is None:
        print("🔄 Loading model...")
        try:
            model = DenseNet121(
                spatial_dims=3,
                in_channels=1,
                out_channels=3
            ).to(device)

            model.load_state_dict(torch.load("model.pth", map_location=device))
            model.eval()

            print("✅ Model loaded successfully")
        except Exception as e:
            print("❌ Model load failed:", e)
            model = None

# ================= TRANSFORMS =================
if MONAI_AVAILABLE:
    predict_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear"),
    ])

# ================= ROUTES =================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        files = request.files.getlist("file")

        if not files or files[0].filename == "":
            return "No files uploaded"

        session_path = os.path.join(UPLOAD_DIR, "scan")

        if os.path.exists(session_path):
            shutil.rmtree(session_path)
        os.makedirs(session_path)

        # ========= SAVE FILES =========
        for f in files:
            f.save(os.path.join(session_path, f.filename))

        print("Uploaded files:", os.listdir(session_path))

        # ========= CHECK MONAI =========
        if not MONAI_AVAILABLE:
            return render_template('advice.html',
                                   diagnosis="Demo Result",
                                   note="MONAI not available")

        # ========= LOAD MODEL LAZILY =========
        if model is None:
            load_model()

        if model is None:
            return render_template('advice.html',
                                   diagnosis="Demo Result",
                                   note="Model failed to load")

        # ========= DICOM → NIFTI =========
        try:
            settings.disable_validate_slice_increment()

            dicom2nifti.convert_directory(
                session_path,
                UPLOAD_DIR,
                compression=True,
                reorient=True
            )

            nifti_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".nii.gz")]

            if not nifti_files:
                return "No valid DICOM series found"

            nifti_path = os.path.join(UPLOAD_DIR, nifti_files[0])

        except Exception as e:
            print("❌ Conversion Error:", e)
            return f"DICOM conversion failed: {str(e)}"

        # ========= PREPROCESS =========
        data_dict = {"image": nifti_path}
        processed = predict_transforms(data_dict)
        input_tensor = processed["image"].unsqueeze(0).to(device)

        print("Input shape:", input_tensor.shape)

        # ========= PREDICTION =========
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)

        p_ad = probs[0][2].item() * 100
        p_mci = probs[0][1].item() * 100
        p_cn = probs[0][0].item() * 100

        # ========= DECISION =========
        if p_ad > 70:
            diagnosis = "Alzheimer (AD)"
            note = "High Confidence Alzheimer Detection"

        elif p_cn > 70:
            diagnosis = "Healthy (CN)"
            note = "High Confidence Healthy Brain"

        elif p_mci > 30:
            diagnosis = "MCI"
            note = "Early Stage Cognitive Impairment Detected"

        else:
            probs_list = [p_ad, p_mci, p_cn]
            classes = ["Alzheimer (AD)", "MCI", "Healthy (CN)"]
            diagnosis = classes[probs_list.index(max(probs_list))]
            note = "Moderate Confidence Prediction"

        # ========= MEMORY CLEANUP =========
        del input_tensor
        del logits
        del probs
        gc.collect()

        return render_template('advice.html',
                               diagnosis=diagnosis,
                               note=note)

    except Exception as e:
        print("SERVER ERROR:", e)
        return "Internal Server Error"


# ================= RUN =================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)