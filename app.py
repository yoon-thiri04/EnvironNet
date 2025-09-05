import streamlit as st
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
from collections import deque
import numpy as np

# config
DEVICE = "cpu"
CKPT_PATH = "environ_net.pt"
IMG_SIZE = 224

# load the model weight
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)

# Rebuild MobileNetV2
model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, checkpoint["config"]["num_classes"])
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

class_names = list(checkpoint["class_to_idx"].keys())

# Image transform
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# prediction
def classify_image(image: Image.Image, top_k=3):
    """
    Takes a PIL image and returns top_k class labels with probabilities.
    """
    model.eval()
    img_t = base_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        # output proba
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
        # top k 
        topk_idx = probs.argsort()[-top_k:][::-1]
        topk_labels = [class_names[i] for i in topk_idx]
        topk_probs = probs[topk_idx]
    return list(zip(topk_labels, topk_probs))

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Upload Photo", "Take Photo", "Live Webcam"])

# Home
if page == "Home":
    # Project Title
    st.title("🌍 EnvironNet")
    st.subheader("Trash Classification AI")
    st.write(
         "EnvironNet is a deep learning model designed to classify everyday waste into **10 categories**: "
        "**Battery, Plastic, Shoe, Cardboard, Clothes, Metal, Organic, Glass, Paper, and Trash.**"
    )


    st.markdown("---")

    st.image("demo/environNet.jpg", width="stretch")

    
    st.header("💡 How EnvironNet Helps")
    st.write(
        "Our AI model uses deep convolutional neural networks (CNNs) to analyze waste images and "
        "classify them automatically. This saves time, reduces human error, and ensures that waste "
        "is sorted properly. By supporting recycling efforts, EnvironNet contributes to a cleaner environment "
        "and builds the foundation for smarter waste management systems."
    )

    st.markdown("---")

    # Goals
    st.header("🎯 Our Goals")
    goal_cols = st.columns(3)
    with goal_cols[0]:
        st.success("**Time-Saving**\n\nMake the trash classification process faster and more efficient by integrating AI.")
    with goal_cols[1]:
        st.success("**Real-World Impact**\n\nUse AI to create meaningful impact for the environment with recycling practices.\n\n")
    with goal_cols[2]:
        st.success("**Awareness**\n\nEncourage people to identify and sort waste responsibly and systematically..\n\n")

    st.markdown("---")

    # Future Plans
    st.header("🚀 Future Plans")
    future_cols = st.columns(2)
    with future_cols[0]:
        st.info("**IoT Integration**\n\nConnect EnvironNet with IoT-enabled bins to make sorting more effective and scalable.")
    with future_cols[1]:
        st.info("**Advanced Features**\n\nExpand classification to include categories like recyclable and non-recyclable waste.")

    st.markdown("---")

    # Options explanation
    st.header("🔍 How to Use")
    st.write(
        "You can classify waste in three different ways (available in the navigation sidebar):\n"
        "- **Upload Photo** – Upload an image from your device.\n"
        "- **Take Photo** – Capture a new photo directly with your camera.\n"
        "- **Live Webcam** – Get real-time classification with labels."
    )

    st.markdown("---")
    # Footer
    st.markdown(
        "<div style='text-align:center; color:gray; padding:10px;'>"
        "EnvironNet © 2025"
        "</div>",
        unsafe_allow_html=True
    )
# upload section
elif page == "Upload Photo":
    st.header(" 📤 Easily classify your everyday waste with EnvironNet!")
    
    # Demo images (paths only)
    demo_images = [
        "demo/metal_2389.jpg",
        "demo/glass381.jpg",
        "demo/no5.jpeg",
        "demo/paper701.jpg",
        "demo/cardboard160.jpg",
        "demo/clothes.jpg",
        "demo/biological336.jpg"
    ]

    image = None

    # Upload from device
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")

    st.write("Or pick a demo image:")
    cols = st.columns(len(demo_images))
    for i, path in enumerate(demo_images):
        with cols[i]:
            demo_img = Image.open(path).convert("RGB")
            demo_img = demo_img.resize((150, 150))
            if st.button("Select", key=f"demo_{i}"):
                image = Image.open(path).convert("RGB")

            st.image(demo_img, width="stretch")

    # Display the selected/uploaded image
    if image is not None:
        st.image(image, caption="Selected Image", width="stretch")
        # Automatic scanning animation
        placeholder = st.empty()
        for i in range(5):
            placeholder.info(f"🔄 Scanning... {'.' * (i % 3 + 1)}")
            time.sleep(0.4)
        placeholder.empty()

        results = classify_image(image, top_k=3)
        if results:
            top_label, top_prob = results[0]
            st.success(f"✅ Predicted: **{top_label}** ({top_prob*100:.2f}%)")
        
        for label, prob in results[1:3]:
            st.write(f"**{label}**: {prob*100:.2f}%")

        if st.button("❌ Cancel"):
            st.warning("Cancelled. Please upload or choose another image.")
            image = None
            results= None

# take with camera
elif page == "Take Photo":
    st.header("📸 Snap your waste and let EnvironNet do the rest!")
    
    img_file = st.camera_input("Take a picture")  # opens phone camera

    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Captured Photo",width="stretch")
        placeholder = st.empty()
        for i in range(5):
            placeholder.info(f"🔄 Scanning... {'.' * (i % 3 + 1)}")
            time.sleep(0.4)
        placeholder.empty()

        results = classify_image(image, top_k=3)
        if results:
            top_label, top_prob = results[0]
            st.success(f"✅ Predicted: **{top_label}** ({top_prob*100:.2f}%)")
        
        for label, prob in results[1:3]:
            st.write(f"**{label}**: {prob*100:.2f}%")

        if st.button("❌ Cancel"):
            st.warning("Cancelled. Please take another photo.")
            
# live webcam with OpenCV
elif page == "Live Webcam":
    st.header("📹 Live Webcam Classification")

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    # Webcam setup
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        frame_count = 0
        skip_frames = 5          # classify every 5th frame
        smooth_window = 10       # moving average window
        probs_buffer = deque(maxlen=smooth_window)
        top_label, top_prob = "", 0.0  # last prediction

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1

            # Only classify every Nth frame
            if frame_count % skip_frames == 0:
                image = Image.fromarray(frame_rgb)
                results = classify_image(image, top_k=1)
                if results:
                    top_label, top_prob = results[0]
                    probs_buffer.append(top_prob)

            # Compute smoothed probability
            if probs_buffer:
                avg_prob = np.mean(probs_buffer)
                cv2.putText(frame_rgb,
                            f"{top_label} ({avg_prob*100:.1f}%)",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

            FRAME_WINDOW.image(frame_rgb)

    if cap:
        cap.release()