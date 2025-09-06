
# 🌍 EnvironNet - Trash Classification AI

EnvironNet is a deep learning-powered application that classifies everyday waste into **10 categories** using a MobileNetV2-based convolutional neural network with accuracy of **92.2** . It aims to make waste management faster, smarter, and more environmentally friendly.

---

## Contents
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Future Plans](#future-plans)
- [License](#license)

---

## 🔹 Features
- Classifies images of waste into **10 categories**: Battery, Plastic, Shoe, Cardboard, Clothes, Metal, Organic, Glass, Paper, and Trash.
- Supports **image upload** from devices.
- Supports **camera input** .
- Supports **live webcam** feature but only on local devices , so clone this project to test. 
- Provides **top-3 predictions** with probabilities.
- Smooth and interactive UI built with Streamlit.
- Designed for **fast inference**, even on mobile devices.

---

##  Demo

[EnvironNet Demo](https://environnet.streamlit.app/)

---

##  Installation

1. Clone the repository:

```bash
git clone https://github.com/yoon-thiri04/EnvironNet.git
````

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app locally:

```bash
streamlit run app.py
```

> Note: The live webcam feature works only on local devices, not on Streamlit Cloud.

---

## 🚀 Usage

1. **Home Page:** Overview of EnvironNet and project goals.
2. **Upload Photo:** Upload an image from your device or select from demo images.
3. **Take Photo:** Use your device camera to capture a new image.
4. **Live Webcam:** Real-time classification using webcam (local only).
5. The app shows **top-3 predicted classes** with probabilities for each image.

---

## 📁 Project Structure

```
environnet/
│
├─ app.py                  # Main Streamlit app
├─ environ_net.pt          # Trained PyTorch model
├─ requirements.txt        # Python dependencies
├─ demo/                   # Demo images for testing
│   ├─ environNet.jpg
│   ├─ metal_2389.jpg
│   ├─ glass381.jpg
│   └─ ... 
```

---

## 🛠 Technologies

* **Python**
* **Streamlit** – Frontend UI
* **PyTorch** – Deep learning framework
* **Torchvision** – Model architecture & image transforms
* **Pillow** – Image processing
* **OpenCV** – Image/video handling (local webcam)

---

## 🌟 Future Plans

* IoT integration: Connect EnvironNet with smart bins for automatic sorting.
* Expand classification to include recyclable vs non-recyclable categories.
* Optimize for larger datasets and faster inference.

---

## 📄 License

This project is licensed under the MIT License.

---

