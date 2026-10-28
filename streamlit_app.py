
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AgeVision",
    page_icon="🎂",
    layout="wide"
)

# --------------------------------------------------
# 2. Load CNN Model and Face Detector
# --------------------------------------------------

@st.cache_resource
def load_resources():

    model_path = Path(__file__).resolve().parent / "utkface_age_cnn.h5"

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    if detector.empty():
        raise RuntimeError("Face detector could not be loaded.")

    return model, detector


model, detector = load_resources()


# --------------------------------------------------
# 3. Convert Predicted Age to Age Category
# --------------------------------------------------

def get_age_category(age):

    if age <= 12:
        return "Child"

    elif age <= 19:
        return "Teenager"

    elif age <= 39:
        return "Young Adult"

    elif age <= 59:
        return "Middle Aged"

    else:
        return "Senior"


# --------------------------------------------------
# 4. Application Interface
# --------------------------------------------------

st.title("🎂 AgeVision")

st.subheader(
    "AI-Powered Age Estimation Using Deep Learning"
)

st.write(
    "Upload a face image or capture a webcam photo "
    "to estimate the person's age and age category."
)

# Sidebar
st.sidebar.header("About AgeVision")

st.sidebar.write("Dataset: UTKFace")

st.sidebar.write(
    "Architecture: Convolutional Neural Network"
)

st.sidebar.metric(
    "Test MAE",
    "6.54 years"
)

st.sidebar.metric(
    "Age Group Accuracy",
    "73.15%"
)

# --------------------------------------------------
# 5. Choose Image Input Method
# --------------------------------------------------

mode = st.radio(
    "Choose your input method",
    ["Upload Image", "Use Webcam"],
    horizontal=True
)

if mode == "Upload Image":

    uploaded = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"]
    )

else:

    uploaded = st.camera_input(
        "Take a photo"
    )


# --------------------------------------------------
# 6. Process Image
# --------------------------------------------------

if uploaded is not None:

    # Open image and convert to RGB
    image = Image.open(uploaded).convert("RGB")

    rgb = np.array(image)

    # Determine whether to skip face detection
    use_full_image = False

    if mode == "Upload Image":

        use_full_image = st.checkbox(
            "This image is already cropped (UTKFace test image)"
        )

    # Option A: Already cropped image
    if use_full_image:

        face = rgb

        st.info(
            "Using the entire image for age prediction."
        )

    # Option B: Detect face with OpenCV
    else:

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:

            st.warning(
                "No face detected. "
                "Try a clearer, front-facing photo. "
                "For an already-cropped UTKFace image, "
                "select the cropped-image checkbox."
            )

            st.stop()

        # Select the largest detected face
        x, y, w, h = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        # Crop the selected face
        face = rgb[y:y+h, x:x+w]


    # --------------------------------------------------
    # 7. Prepare Image for Our CNN
    # --------------------------------------------------

    # Resize to 64 x 64
    face_resized = cv2.resize(
        face,
        (64, 64),
        interpolation=cv2.INTER_LINEAR
    )

    # Normalize pixel values between 0 and 1
    input_data = (
        face_resized.astype("float32") / 255.0
    )

    # Add batch dimension
    # (64, 64, 3) -> (1, 64, 64, 3)
    input_data = np.expand_dims(
        input_data,
        axis=0
    )


    # --------------------------------------------------
    # 8. Predict Age Using Trained CNN
    # --------------------------------------------------

    prediction = model.predict(
        input_data,
        verbose=0
    )

    # Extract predicted numerical age
    estimated_age = float(prediction[0][0])

    # Clip to dataset age range and round
    predicted_age = int(
        np.rint(
            np.clip(estimated_age, 0, 116)
        )
    )

    # Determine age category
    category = get_age_category(predicted_age)


    # --------------------------------------------------
    # 9. Display Prediction Results
    # --------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

        st.image(
            face_resized,
            caption="Processed Face (64 x 64)",
            width=180
        )

    with right:

        st.success(
            "Age estimation completed successfully!"
        )

        st.metric(
            "Estimated Age",
            f"{predicted_age} years"
        )

        st.metric(
            "Age Category",
            category
        )

        st.write(
            "Our CNN examined the facial features "
            "and generated an estimated numerical age."
        )

        st.info(
            "Prediction method: "
            + (
                "Already-cropped image"
                if use_full_image
                else "OpenCV face detection"
            )
        )


    # --------------------------------------------------
    # 10. Disclaimer
    # --------------------------------------------------

    st.caption(
        "AgeVision predicts approximate age from facial "
        "appearance. Results may be inaccurate and should "
        "not be treated as verified age or used for "
        "important decisions."
    )
