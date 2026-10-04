
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Skin Lesion Classification",
    page_icon="🔬",
    layout="wide"
)


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
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AKIEC": "Actinic keratoses / intraepithelial carcinoma",
    "BKL": "Benign keratosis-like lesion",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion"
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORMATION
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
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = models.efficientnet_b0(
        weights=None
    )

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        len(class_names)
    )

    model_path = "models/efficientnet_b0_skin_lesion.pth" 

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):
        self.activations = output.detach()

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor,
        class_idx
    ):

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

        cam = (
            weights * activations
        ).sum(dim=1)

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

def predict(image):

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    model.eval()

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    confidence, predicted_idx = torch.max(
        probabilities,
        dim=1
    )

    return (
        input_tensor,
        predicted_idx.item(),
        confidence.item(),
        probabilities[0].cpu().numpy()
    )


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    cam
):

    image_np = np.array(image)

    cam = cv2.resize(
        cam,
        (image.width, image.height)
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(
        image_bgr,
        0.6,
        heatmap,
        0.4,
        0
    )

    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return overlay


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔬 Explainable Skin Lesion Classification"
)

st.subheader(
    "EfficientNet-B0 + Grad-CAM"
)

st.write(
    """
    Upload a skin-lesion image to obtain an educational
    image-classification prediction and a Grad-CAM
    visualization.
    """
)


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    """
    ⚠️ Educational Project Only

    This application is intended for educational and
    demonstration purposes only. It is not a medical
    diagnostic tool and should not be used to make
    healthcare decisions.
    """
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        width=400
    )

    (
        input_tensor,
        predicted_idx,
        confidence,
        probabilities
    ) = predict(image)

    predicted_class = class_names[
        predicted_idx
    ]


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Class",
            predicted_class
        )

        st.write(
            class_descriptions[
                predicted_class
            ]
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )


    # ========================================================
    # TOP 3
    # ========================================================

    st.subheader("Top 3 Predictions")

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    for idx in top_indices:

        st.write(
            f"**{class_names[idx]}** — "
            f"{class_descriptions[class_names[idx]]}"
        )

        st.progress(
            float(probabilities[idx])
        )

        st.write(
            f"{probabilities[idx]:.2%}"
        )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.subheader(
        "Grad-CAM Explainability"
    )

    with st.spinner(
        "Generating Grad-CAM..."
    ):

        cam = grad_cam.generate(
            input_tensor,
            predicted_idx
        )

        overlay = create_gradcam_overlay(
            image,
            cam
        )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    with col2:

        st.image(
            overlay,
            caption="Grad-CAM Visualization",
            use_container_width=True
        )

    st.info(
        """
        Grad-CAM highlights image regions that contributed
        more strongly to the model's classification. It
        should not be interpreted as a clinical explanation
        or diagnosis.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Internship Project | EfficientNet-B0 | Grad-CAM"
)
