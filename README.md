🎂 AgeVision — Facial Age Estimation with Deep Learning
AgeVision is a deep-learning web application that estimates a person's numerical age from a face image and assigns the estimate to an age category. It uses a Convolutional Neural Network (CNN) trained on the UTKFace dataset and a Streamlit interface for photo upload or webcam capture.

🌐 Try the live AgeVision app · [📓 Training notebook](Age_Prediction_UTKFace.ipynb)
Important: AgeVision estimates apparent age from images. It does not verify anyone's age, and predictions can be inaccurate. Do not use it for age verification or other high-stakes decisions.

✨ Features
- Numerical age estimation: Predicts an approximate age in years using a CNN regression model.
- Age category: Converts the estimated age into Child, Teenager, Young Adult, Middle Aged, or Senior.
- Photo upload: Accepts JPG, JPEG, and PNG images.
- Webcam support: Captures a photo directly in the browser.
- Face detection: Uses OpenCV's Haar Cascade to detect and crop the largest visible face.
- Cropped-image mode: Lets users skip face detection for already-cropped UTKFace examples.
- Visual results: Displays the original image, the processed 64 × 64 face, estimated age, and category.
📊 Model performance
The following results were measured using a held-out test set of 3,557 images:
Metric	Result
Mean Absolute Error (MAE)	6.54 years
Root Mean Squared Error (RMSE)	9.04 years
Age-category accuracy	73.15%
Age-category macro F1-score	0.63


How to read MAE: Across the test images, numerical predictions differed from the labelled ages by about 6.54 years on average. This is not a guarantee of accuracy on an individual photo.
Error by actual age category
Actual category	Age range	Test images	Age MAE
Child	0–12	512	4.48 years
Teenager	13–19	177	7.33 years
Young Adult	20–39	1,782	5.41 years
Middle Aged	40–59	682	8.26 years
Senior	60+	404	10.92 years


The model was least accurate for Seniors in numerical age estimation. Age-category classification was especially challenging for Teenagers, many of whom were predicted as Young Adults.
🗂️ Dataset and preprocessing
- Dataset: UTKFace — Kaggle dataset
- Images used: 23,708 images from the dataset's UTKFace folder.
- Target: Numerical age extracted from the image filename.
- Age-group labels: Derived from numerical ages for analysis and presentation; the CNN itself learns to predict numerical age, not five softmax classes.
- Image input: 64 × 64 RGB (three-channel) images.
- Normalization: Pixel values scaled from 0–255 to 0–1.
- Data augmentation: Random horizontal flips during training.
We used stratified sampling by age group to maintain similar age-group proportions across the splits:
Split	Images	Share
Training	16,594	70%
Validation	3,557	15%
Testing	3,557	15%
Total	23,708	100%


The test set was held out during training and model selection.
🧠 CNN architecture
Input: 64 × 64 × 3 RGB image
   ↓
RandomFlip("horizontal") — training augmentation
   ↓
Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
   ↓
Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
   ↓
Conv2D(128, 3×3, ReLU) → MaxPooling2D(2×2)
   ↓
Flatten → Dense(128, ReLU) → Dropout(0.3)
   ↓
Dense(1, linear) → Estimated age
   ↓
Age-range rules → Age category
The model has 1,142,081 trainable parameters. It was trained with:
- Optimizer: Adam, learning rate 0.001
- Loss: Huber loss, delta=10.0
- Reported metric: Mean Absolute Error (mae)
- Batch size: 32
- Maximum epochs: 20; training stopped after 13 epochs
- Early stopping: Monitor validation MAE, patience 4, minimum improvement 0.1, restore selected best weights
🏷️ Age categories
Predicted age (years)	Displayed category
0–12	Child
13–19	Teenager
20–39	Young Adult
40–59	Middle Aged
60–116	Senior


The app rounds and clips the CNN's numerical prediction into the age range before assigning a category.
🚀 Run locally
Requirements
- Python 3.11 recommended
- Files from this repository
- A webcam (optional; photo upload works without one)
Instructions
1. Clone the repository:
   git clone https://github.com/abdullaaihaiki/agevision-utkface.git
   cd agevision-utkface
2. Create and activate a virtual environment (recommended):
   Windows:
   python -m venv .venv
   .venv\Scripts\activate
   macOS / Linux:
   python3 -m venv .venv
   source .venv/bin/activate
3. Install the dependencies:
   pip install -r requirements.txt
4. Launch the app:
   python -m streamlit run streamlit_app.py
5. Open the local address displayed by Streamlit (usually http://localhost:8501).
Note: streamlit_app.py expects the saved model utkface_age_cnn.h5 to be in the same directory. The full training image dataset is not required to run the app.
📁 Repository files
agevision-utkface/
├── Age_Prediction_UTKFace.ipynb   # Training, evaluation, visualizations
├── streamlit_app.py              # Streamlit user interface and inference
├── utkface_age_cnn.h5            # Trained CNN model
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
🔎 How prediction works
1. The user uploads a photo or takes a webcam picture.
2. For a standard photograph, OpenCV attempts to detect a face and crops the largest one. For an already-cropped UTKFace image, the user can opt to skip detection.
3. The selected face is resized to 64 × 64, converted to RGB, and normalized to 0–1.
4. The trained CNN predicts one numerical age.
5. The age is rounded, constrained to the supported range, and mapped to a category.
6. Streamlit displays the estimated age, category, original image, and processed face.
Evaluation note: The notebook's test metrics use the dataset images directly; the live app may produce different predictions because OpenCV cropping and resizing can alter its inputs.
⚠️ Limitations and responsible use
- Predictions are approximate: An average test error of 6.54 years means individual errors can be much larger.
- Age-group boundaries: Small numerical errors near ages 12/13, 19/20, 39/40, and 59/60 can change the predicted category.
- Uneven performance: The training data is concentrated in Young Adults; older faces and teenagers are more challenging.
- Image quality matters: Blurriness, occlusions, unusual camera angles, and poor lighting can reduce accuracy.
- Dataset imperfections: Error analysis identified some questionable or incorrectly cropped images.
- Face detection is imperfect: Haar Cascade may miss faces, especially low-resolution or unusually framed photos.
- Generalization and fairness: Performance may vary across people and conditions; this project is not validated for identity checks, eligibility decisions, or medical/legal use.
🔮 Potential improvements
- Evaluate a larger or better-cleaned training dataset.
- Add stronger but age-appropriate augmentation and compare CNN architectures.
- Investigate sampling or age-balanced training strategies.
- Explore transfer learning using a pretrained image backbone.
- Improve face detection and make app preprocessing match notebook preprocessing more closely.
- Evaluate errors across additional subgroups where responsible and feasible.
🛠️ Technology stack
Python · TensorFlow/Keras · NumPy · Pandas · scikit-learn · OpenCV · Pillow · Matplotlib · Streamlit · Jupyter Notebook
📚 Project purpose
Created as a hands-on deep learning project to practice image data preparation, CNN regression, early stopping, model evaluation, error analysis, and deployment of a trained model as an interactive web application.

Dataset credit: UTKFace. This repository contains a trained model and code; it does not redistribute the full source dataset.
