from flask import Flask, render_template, request
import os
import numpy as np
import nibabel as nib
import onnxruntime as ort

app = Flask(__name__)

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= MODEL =================
session = None
input_name = None

def load_model():
    global session, input_name
    if session is None:
        print("🔄 Loading ONNX model...")
        session = ort.InferenceSession("model_final.onnx")
        input_name = session.get_inputs()[0].name
        print("✅ Model loaded")


# ================= PREPROCESS =================
def preprocess_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()

    # Handle NaN
    data = np.nan_to_num(data)

    # Normalize
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    # Resize
    data = resize_volume(data, (64, 64, 64))

    # Add channel + batch
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=0)

    return data.astype(np.float32)


def resize_volume(img, target_shape):
    from scipy.ndimage import zoom

    factors = (
        target_shape[0] / img.shape[0],
        target_shape[1] / img.shape[1],
        target_shape[2] / img.shape[2],
    )

    return zoom(img, factors, order=1)


# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        if not file.filename.endswith((".nii", ".nii.gz")):
            return "Please upload a valid NIFTI file (.nii or .nii.gz)"

        filepath = os.path.join(UPLOAD_DIR, file.filename)
        file.save(filepath)

        load_model()

        input_data = preprocess_nifti(filepath)

        outputs = session.run(None, {input_name: input_data})
        logits = outputs[0]

        # Stable softmax
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(logits)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        cn, mci, ad = probs[0]

        if ad > 0.6:
            diagnosis = "Alzheimer (AD)"
            note = "Signs of Alzheimer's detected. Please consult a neurologist."

        elif mci > 0.4:
            diagnosis = "Mild Cognitive Impairment (MCI)"
            note = "Early stage cognitive decline detected."

        else:
            diagnosis = "Healthy (CN)"
            note = "No significant abnormalities detected."

        os.remove(filepath)

        return render_template("advice.html",
                               diagnosis=diagnosis,
                               note=note)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return "Error: " + str(e)


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)