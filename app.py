import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Explainable Skin Lesion Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Explainable Skin Lesion Classification")
st.write(
    "Educational image classification demo using EfficientNet-B0 "
    "and Grad-CAM."
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Model path
# --------------------------------------------------

model_path = "models/efficientnet_b0_skin_lesion.pth"


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():

    checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=False
)

    class_names = checkpoint["class_names"]

    model = models.efficientnet_b0(weights=None)

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, class_names


model, class_names = load_model()


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------

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

        self.activations = output


    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]


    def generate(self, input_tensor, class_idx):

        self.model.zero_grad()

        output = self.model(input_tensor)

        score = output[:, class_idx]

        score.backward()

        gradients = self.gradients.detach()

        activations = self.activations.detach()

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


# --------------------------------------------------
# Create Grad-CAM
# --------------------------------------------------

target_layer = model.features[-1]

grad_cam = GradCAM(
    model,
    target_layer
)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict(image):

    input_tensor = transform(image).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    input_tensor.requires_grad_(True)

    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        predicted_index
    ].item()

    return (
        input_tensor,
        probabilities.detach().cpu().numpy(),
        predicted_index,
        confidence
    )


# --------------------------------------------------
# Grad-CAM visualization
# --------------------------------------------------

def create_gradcam(image, cam):

    original = np.array(image)

    original = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
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

    overlay = cv2.addWeighted(
        original,
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


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------
    # Display uploaded image
    # --------------------------------------------------

    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    with st.spinner("Analyzing image..."):

        (
            input_tensor,
            probabilities,
            predicted_index,
            confidence
        ) = predict(image)


        cam = grad_cam.generate(
            input_tensor,
            predicted_index
        )


        gradcam_image = create_gradcam(
            image,
            cam
        )


    # --------------------------------------------------
    # Prediction result
    # --------------------------------------------------

    predicted_class = class_names[
        predicted_index
    ]

    st.subheader("Prediction")

    st.success(
        f"Predicted Class: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence * 100:.2f}%"
    )


    # --------------------------------------------------
    # Top 3 predictions
    # --------------------------------------------------

    st.subheader("Top 3 Predictions")

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    for index in top_indices:

        probability = probabilities[index]

        st.write(
            f"**{class_names[index]}** — "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )


    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------

    st.subheader("Grad-CAM Explanation")

    st.image(
        gradcam_image,
        caption="Grad-CAM visualization",
        use_container_width=True
    )


    st.info(
        "The Grad-CAM visualization highlights image regions "
        "that contributed to the model's prediction."
    )


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.divider()

st.warning(
    "Disclaimer: This application is an educational machine-learning "
    "demonstration and is not a medical diagnostic tool. Model predictions "
    "should not be used for medical decisions."
)
