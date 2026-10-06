#  Multi-Modal Age & Gender Detection with Health & Behavioral Prediction

 This is an AI-based facial analysis project that detects Age, Gender, Emotion, Skin Type, and Behaviour from a face.

##  Live Demo

https://multi-modal-age-and-gender-detection-0z0h.onrender.com

---

##  Features

*  Capture an image using the webcam
*  Upload an image from the device
*  Age prediction
*  Gender prediction
*  Emotion detection
*  Skin type estimation
*  Behaviour prediction
*  Personalized suggestions
*  Face detection with bounding box

### Skin Types

The application gives an approximate skin type:

* Dry
* Oily
* Normal
* Combination

### Emotions

The emotion model can detect:

* Neutral
* Happy
* Surprise
* Sad
* Angry
* Disgust
* Fear
* Contempt

---

##  Technologies Used

* **Python**
* **Flask**
* **OpenCV**
* **InsightFace**
* **ONNX Runtime**
* **NumPy**
* **HTML**
* **CSS**
* **JavaScript**
* **Gunicorn**
* **Render**

---

## Project Workflow

```text
Webcam / Upload Image
          ↓
     Image Processing
          ↓
     Face Detection
          ↓
   ┌──────┴───────┐
   ↓              ↓
Age & Gender    Emotion
   ↓              ↓
   └──────┬───────┘
          ↓
     Skin Type
          ↓
      Behaviour
          ↓
   Suggestions
          ↓
    Display Results
```

---
## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Varshabachu050/Multi-Modal-Age-and-Gender-Detection-with-Health-Behavioral-Prediction.git
```

### 2. Open the Project Folder

```bash
cd Multi-Modal-Age-and-Gender-Detection-with-Health-Behavioral-Prediction
```

### 3. Create Virtual Environment

```bash
python3 -m venv .venv
```

### 4. Activate Virtual Environment

For macOS/Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Packages

```bash
pip install -r requirements.txt
```

---

## Run the Project

Run:

```bash
python main.py
```
---
## 👨‍💻 Author

**Bachu Varsha**
