import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SkinSight AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 2.2rem 2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 12px 35px rgba(15, 23, 42, 0.18);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.9;
    margin-bottom: 18px;
}

.badge {
    display: inline-block;
    padding: 7px 14px;
    margin-right: 8px;
    margin-bottom: 5px;
    border-radius: 20px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.2);
    font-size: 13px;
}

.card {
    background: white;
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.07);
    margin-bottom: 1rem;
}

.card-title {
    font-size: 21px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 10px;
}

.card-text {
    color: #475569;
    line-height: 1.7;
}

.prediction-card {
    background: linear-gradient(135deg, #eff6ff, #ffffff);
    padding: 1.8rem;
    border-radius: 20px;
    border: 1px solid #bfdbfe;
    text-align: center;
}

.prediction-label {
    color: #64748b;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prediction-class {
    font-size: 42px;
    font-weight: 800;
    color: #1d4ed8;
    margin: 8px 0;
}

.confidence {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
}

.top-item {
    padding: 12px 15px;
    margin: 8px 0;
    border-radius: 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.section-title {
    font-size: 28px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.section-subtitle {
    color: #64748b;
    margin-bottom: 1.5rem;
}

.info-box {
    padding: 1.2rem;
    border-radius: 15px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.info-value {
    font-size: 22px;
    font-weight: 750;
    color: #1e3a8a;
}

.info-label {
    font-size: 13px;
    color: #64748b;
    margin-top: 5px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding: 2rem 0 1rem 0;
}

.disclaimer {
    padding: 1.2rem 1.4rem;
    border-radius: 15px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #7c2d12;
    margin-top: 2rem;
}

[data-testid="stFileUploader"] {
    background: white;
    border-radius: 18px;
    padding: 1rem;
    border: 2px dashed #93c5fd;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC"
]

class_descriptions = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AKIEC": "Actinic Keratosis / Intraepithelial Carcinoma",
    "BKL": "Benign Keratosis-like Lesion",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion"
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = models.efficientnet_b0(weights=None)

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        len(class_names)
    )

    model_path = "models/efficientnet_b0_skin_lesion.pth"

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()


# ============================================================
# TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):

        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):

        self.model.zero_grad()

        output = self.model(input_tensor)

        score = output[:, class_idx]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()

        if cam.max() != 0:
            cam /= cam.max()

        return cam


target_layer = model.features[-1]

grad_cam = GradCAM(
    model,
    target_layer
)


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image):

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    confidence, predicted_idx = torch.max(
        probabilities,
        dim=1
    )

    predicted_idx = predicted_idx.item()

    confidence = confidence.item()

    return (
        predicted_idx,
        confidence,
        probabilities[0].cpu().numpy(),
        image_tensor
    )


# ============================================================
# GRAD-CAM VISUALIZATION
# ============================================================

def create_gradcam(image, cam):

    original = np.array(
        image.convert("RGB")
    )

    height, width = original.shape[:2]

    cam = cv2.resize(
        cam,
        (width, height)
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    return overlay


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🔬 SkinSight AI")

    st.write(
        "An explainable image classification "
        "demonstration using deep learning."
    )

    st.divider()

    st.markdown("### Model")

    st.write("""
    **Architecture:** EfficientNet-B0

    **Explainability:** Grad-CAM

    **Input Size:** 224 × 224

    **Classes:** 7
    """)

    st.divider()

    st.markdown("### Dataset")

    st.write("""
    **ISIC 2018**

    Skin lesion image dataset used for
    educational machine-learning experimentation.
    """)

    st.divider()

    st.caption(
        "Educational AI/ML internship project."
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        🔬 SkinSight AI
    </div>

    <div class="hero-subtitle">
        Explainable Skin Lesion Classification
        using Deep Learning
    </div>

    <span class="badge">EfficientNet-B0</span>
    <span class="badge">Grad-CAM</span>
    <span class="badge">ISIC 2018</span>
    <span class="badge">Explainable AI</span>

</div>
""", unsafe_allow_html=True)


# ============================================================
# ANALYZE IMAGE
# ============================================================

st.markdown(
    '<div class="section-title">Analyze an Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Upload a skin-lesion image to see the model prediction '
    'and the visual regions that influenced the prediction.'
    '</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "📤 Upload a skin lesion image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)


# ============================================================
# RESULTS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    predicted_idx, confidence, probabilities, image_tensor = (
        predict_image(image)
    )

    predicted_class = class_names[
        predicted_idx
    ]

    cam = grad_cam.generate(
        image_tensor,
        predicted_idx
    )

    overlay = create_gradcam(
        image,
        cam
    )


    # --------------------------------------------------------
    # IMAGE + PREDICTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Analysis Results</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    with col1:

        st.markdown(
            '<div class="card">'
            '<div class="card-title">'
            '📷 Uploaded Image'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            '<div class="prediction-card">'
            '<div class="prediction-label">'
            'Model Prediction'
            '</div>'
            f'<div class="prediction-class">'
            f'{predicted_class}'
            '</div>'
            f'<div>'
            f'{class_descriptions[predicted_class]}'
            f'</div>'
            '<br>'
            '<div class="confidence">'
            f'Confidence: {confidence * 100:.2f}%'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.progress(
            float(confidence)
        )

        st.markdown(
            '<div class="card">'
            '<div class="card-title">'
            '🏆 Top 3 Predictions'
            '</div>',
            unsafe_allow_html=True
        )

        top_indices = np.argsort(
            probabilities
        )[::-1][:3]

        for rank, idx in enumerate(
            top_indices
        ):

            probability = probabilities[idx]

            st.markdown(
                f"""
                <div class="top-item">

                    <strong>
                    #{rank + 1}
                    &nbsp; {class_names[idx]}
                    </strong>

                    <span style="float:right;">
                    {probability * 100:.2f}%
                    </span>

                    <br>

                    <small>
                    {class_descriptions[class_names[idx]]}
                    </small>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🧠 Explainable AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Grad-CAM highlights image regions that contributed '
        'to the model prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    cam_col1, cam_col2 = st.columns(
        2,
        gap="large"
    )

    with cam_col1:

        st.markdown(
            '<div class="card">'
            '<div class="card-title">'
            'Original Image'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with cam_col2:

        st.markdown(
            '<div class="card">'
            '<div class="card-title">'
            'Grad-CAM Visualization'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            overlay,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    st.info(
        "Grad-CAM is an explainability technique that "
        "visualizes image regions that contributed to "
        "the model's prediction."
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Model Information'
        '</div>',
        unsafe_allow_html=True
    )

    info1, info2, info3, info4 = st.columns(4)

    with info1:

        st.markdown(
            """
            <div class="info-box">
                <div class="info-value">
                    EfficientNet-B0
                </div>
                <div class="info-label">
                    Architecture
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info2:

        st.markdown(
            """
            <div class="info-box">
                <div class="info-value">
                    7
                </div>
                <div class="info-label">
                    Classes
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info3:

        st.markdown(
            """
            <div class="info-box">
                <div class="info-value">
                    224×224
                </div>
                <div class="info-label">
                    Input Size
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info4:

        st.markdown(
            """
            <div class="info-box">
                <div class="info-value">
                    Grad-CAM
                </div>
                <div class="info-label">
                    Explainability
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown("""
    <div class="card"
         style="text-align:center; padding:3rem;">

        <div style="font-size:55px;">
            🖼️
        </div>

        <div class="card-title">
            Ready to Analyze
        </div>

        <div class="card-text">
            Upload a JPG, JPEG, or PNG image above
            to start the AI analysis.
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ABOUT
# ============================================================

st.markdown(
    '<div class="section-title">'
    'ℹ️ About the Project'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="card">

<div class="card-text">

<b>SkinSight AI</b> is an educational deep-learning
application developed to demonstrate image classification
and explainable artificial intelligence.

The system uses <b>EfficientNet-B0</b> with transfer learning
to classify skin-lesion images into seven categories.

The application also uses <b>Grad-CAM</b> to provide a visual
explanation of the regions that influenced the model's
prediction.

The project demonstrates the complete machine-learning
workflow, including image preprocessing, transfer learning,
classification, evaluation, and explainability.

</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("""
<div class="disclaimer">

<b>⚠️ Educational Disclaimer</b>

<br><br>

This application is an educational machine-learning
demonstration and is <b>not a medical diagnostic tool</b>.
Model predictions should not be used for medical decisions.
For health concerns, consult a qualified healthcare professional.

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class=hero>

    <div class=hero-title>
        🔬 SkinSight AI
    </div>

    <div class=hero-subtitle>
        Explainable Skin Lesion Classification
        using Deep Learning
    </div>

    <span class=badge>EfficientNet-B0</span>
    <span class=badge>Grad-CAM</span>
    <span class=badge>ISIC 2018</span>
    <span class=badge>Explainable AI</span>

</div>
""", unsafe_allow_html=True)

