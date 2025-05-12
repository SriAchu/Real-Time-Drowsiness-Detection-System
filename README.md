This project builds a real-time drowsiness detection system to decrease the risk of accidents due to driver fatigue. 
It employs a webcam to monitor a driver's eye and mouth movement constantly, processing their state based on Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) measurements. 
Machine learning is utilized in this system, integrating Histogram of Oriented Gradients (HOG) and Haar features for accurate facial feature extraction. 
Pre-trained Support Vector Machine (SVM) models are employed to identify eye closure and yawning, while real-time detection is processed using OpenCV and MediaPipe. 
Alerts are initiated using sound notifications and Pushbullet messages in order to guarantee timely intervention of the driver. 
Data augmentation is also included in the project to enhance model robustness and it has a modular design for simplicity in integration with vehicle safety systems. 
