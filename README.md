# 🧠 Multi-Model Age & Gender Classification with Healthcare and Behavioral Prediction

A real-time AI-powered facial analysis system that captures a user's face through a webcam and predicts **Age**, **Gender**, **Emotion**, and **Skin Type**, while providing personalized **Healthcare**, **Skincare**, and **Behavioral Recommendations**.

This project combines Computer Vision and Deep Learning techniques using **Python**, **OpenCV**, **DeepFace**, and **NumPy** to deliver intelligent insights from facial features in real time.

---

## 🚀 Features

* 📷 Real-Time Face Detection using Webcam
* 🧠 Age Estimation
* 🚻 Gender Classification
* 😊 Emotion Detection
* 🧴 Skin Type Analysis

  * Dry Skin
  * Oily Skin
  * Normal Skin
  * Combination Skin
* 💡 Personalized Healthcare Recommendations
* 🌟 Skincare Suggestions Based on Skin Type
* 📊 Behavioral Insights Based on Emotion Analysis

---

## 🛠️ Technologies Used

* Python 3.x
* OpenCV
* DeepFace
* NumPy

---

## 🧠 How It Works

### 1. Face Capture

* Accesses the webcam feed.
* Detects the largest face using OpenCV.
* Displays a live video stream with a face bounding box.

### 2. Facial Analysis

* Captures the detected face image.
* Uses DeepFace to analyze:

  * Age
  * Gender
  * Emotion

### 3. Skin Type Detection

* Converts the facial image to HSV color space.
* Evaluates brightness and saturation levels.
* Classifies skin type as:

  * Dry
  * Oily
  * Normal
  * Combination

### 4. Recommendation Engine

Generates personalized suggestions based on:

* Detected age group
* Emotional state
* Skin type characteristics

---

## 📁 Project Structure

```text
Multi_Model_Age_Gender_Classification/
│
├── main.py
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/varshabachu050/Multi_Model_Age_Gender_Classification.git
cd Multi_Model_Age_Gender_Classification
```

### Install Required Dependencies

```bash
pip install opencv-python numpy deepface
```

---

## ▶️ Run the Project

```bash
python main.py
```

### Controls

* Press **Q** to capture the detected face.
* The system will generate age, gender, emotion, and skin-type predictions along with recommendations.

---

## 📋 Dependencies

```text
Python 3.x
OpenCV
NumPy
DeepFace
```

---

## 📊 Sample Output

```text
Age: 25

Gender: Male

Emotion: Happy

Skin Type: Normal
```

### 💡 Recommendations

```text
✓ Maintain a balanced skincare routine
✓ Stay hydrated throughout the day
✓ Use a lightweight moisturizer
✓ Continue positive lifestyle habits
```

---

## 🎯 Applications

* Personalized Healthcare Assistance
* AI-Based Skincare Recommendation Systems
* Emotion and Behavioral Monitoring
* Human-Computer Interaction Research
* Educational AI and Computer Vision Projects
* Smart Wellness and Self-Care Applications

---

## 🛡️ Security & Ethical Considerations

* Webcam access is used only during application execution.
* No facial images are permanently stored.
* User privacy should always be respected.
* Obtain user consent before collecting or processing facial data.
* This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**Varsha Bachu**
