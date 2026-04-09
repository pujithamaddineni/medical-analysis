Alzheimer’s Disease Detection Using Deep Learning

Deploy link::https://medical-analysis.onrender.com

Concept Summary

This project implements an automated computer-aided diagnostic (CAD) 
system for early detection of Alzheimer’s Disease (AD) using multimodal brain imaging.

Data Used:

3D structural MRI scans
Optional Amyloid PET scans

Objective:

Automatically classify subjects into:
Cognitively Normal (CN)
Mild Cognitive Impairment (MCI)
Alzheimer’s Disease (AD)

Approach:

Preprocessing: Normalize, resize, and convert medical images into structured 3D format.
Feature Extraction: Use advanced CNNs (VGG16, InceptionV3) to extract structural brain features.
3D-CNN Classification: Analyze full volumetric brain scans to capture spatial relationships.
Model Interpretation: Highlight brain regions influencing predictions for clinical transparency.

Key Advantages:

Supports early diagnosis
Reduces human error
Improves consistency across scans
Provides visual explanations for clinical decisions

How to Run the Model
1. Clone the Repository
git clone <your-repo-link>
cd <repo-folder>
2. Set Up Environment

Install required Python packages (preferably in a virtual environment):

pip install -r requirements.txt

Typical packages used:

torch, torchvision (for PyTorch and CNNs)
monai (medical imaging tools)
nibabel or pydicom (for MRI/DICOM handling)
numpy, scikit-learn, matplotlib (data processing and visualization)

3. Prepare Data
Place 3D MRI scans in a designated folder (e.g., data/MRI/).
Ensure all images are preprocessed (normalized, resized, co-registered).
Optional: Use dicom2nifti if converting DICOM to NIfTI format.

5. Run Preprocessing Script
python preprocess.py

This will:

Convert raw images into structured 3D arrays
Normalize intensity
Save processed images for model training or inference

5. Run Prediction / Train Model
To predict using pre-trained model:
python predict.py --input data/MRI/subject1.nii.gz
To train from scratch:
python train.py --data_dir data/MRI/ --epochs 50 --batch_size 8

7. View Results
Predictions will display as CN, MCI, or AD (percentage optional).
Model interpretation module will highlight influential brain regions.

9. Optional: Run Flask Web App
Start the server:
python app.py


Open your browser at http://127.0.0.1:5000/
Upload MRI or PET scans and get automated predictions with visual explanations.
