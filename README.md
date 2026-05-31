🧠 Multi-Model Age & Gender Classification with Healthcare and Behavior Predictions

-> A real-time face analysis system that captures a user’s face using a webcam and performs:
1. Age estimation
2. Gender classification
3. Emotion detection
4. Skin type analysis
5. Personalized healthcare and skincare recommendations
6. Built with Python, OpenCV, DeepFace, and NumPy.

 🚀 Features
- 📷 Real-Time Face Capture using webcam
- 🧠 Age Prediction
- 🚻 Gender Classification
- 😊 Emotion Detection
- 🧴 Skin Type Detection (Dry, Oily, Normal, Combination)
- 💡 Personalized Skincare & Health Suggestions.
  
🧠 How It Works
1. Face Capture 
2. Captures webcam feed
3. Detects largest face using OpenCV
4. Shows live video with face bounding box
5. Facial Analysis
6. Uses DeepFace to predict age, gender, emotion
7. Saves temporary image for analysis
8. Skin Type Detection
9. Converts face to HSV
10. Uses brightness and saturation to classify skin type
11. Healthcare & Behavior Recommendations
12. Emotion-based suggestions
13. Age-based advice
14. Skin-type-based skincare guidance
    
-> 📁 Project Structure

--> Multi_Model_Age_Gender_Classification/

├── multi_model_age&gender_classification_with_healthcare_using_CNN.py       
└── README.md 

⚠️ Note: This project is a standalone Python script.
It does not include a web interface or Flask integration.

-> ⚙️ How to Run
- Clone the repository:
  
  git clone https://github.com/varshabachu050/Multi_Model_Age_Gender_Classification.git
  
  cd Multi_Model_Age_Gender_Classification

-> Install dependencies: 
- pip install opencv-python numpy deepface
- Run the script:
- python Multi_Model_Age_Gender_Classification.py
- Press Q to capture the face and generate analysis results.

📋 Dependencies: 
- Python 3.x
- OpenCV (cv2)
- NumPy
- DeepFace
 
-> 🛡️ Security & Ethical Notes
-  Webcam access is only used during script execution
-  No facial data is stored permanently
-  Avoid using real biometric data in production without consent
 
-> 📊 Sample Output
- Age: 25
- Gender: Male
- Emotion: Happy
- Skin Type: Normal

-> 💡 Recommendations:
- Use a balanced skincare routine
- Maintain hydration
- Light moisturizer for skin protection
  
-> 🎯 Applications :
- Personalized healthcare suggestions.
- Emotion and behavior monitoring.
- AI-powered personal care insights. 
