## 🤖 YOLOv8 AI Agent Detection System
This project is a Streamlit-based object detection system powered by YOLOv8, enhanced with an AI Agent that performs deep analysis of detections and provides a feedback loop for low-confidence predictions.
Developed as part of a 5th semester mini-project.

## 🚀 Features
🔍 Real-time or image-based object detection
🧠 AI-powered expert analysis using structured JSON responses
🔄 Automatic feedback loop for improving low-confidence detection results
🎙️ Voice feedback using text-to-speech
📊 Interactive visualizations (confidence graphs, summary tables, frequency analysis)
💾 Downloadable full detection summary reports
🎨 A beautifully designed UI using advanced custom CSS

## 🛠️ Tech Stack
- Framework:	Streamlit
- Object Detection:	YOLOv8 (Ultralytics)
- AI Agent:	Claude API / LLM JSON Responses
- Voice Output:	pyttsx3
- Plotting:	Plotly
- Image Processing:	OpenCV, Pillow
- Data:	Pandas, Numpy
  
## 📦 Installation & Setup
1️⃣ Clone the repository
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
2️⃣ Install dependencies
```
pip install -r requirements.txt

```
Packages you need include:
```
streamlit
ultralytics
opencv-python
numpy
pillow
pyttsx3
plotly
pandas
requests
```
3️⃣ Download YOLOv8 model weights

Weights load automatically (yolov8n.pt, yolov8s.pt, yolov8m.pt).
But you can also download manually from Ultralytics.

4️⃣ Run the app
```
streamlit run app.py
```
```
📁 Project Structure
📂 YOLOv8-AI-Agent-Detection
│── app.py               # Main Streamlit application
│── requirements.txt     # Python dependencies
│── README.md            # Project documentation
└── (models auto-download during runtime)
```
Screenshots :
<img width="1362" height="580" alt="Screenshot 2025-11-26 101603" src="https://github.com/user-attachments/assets/6d8f9dc0-54a0-4488-8c1e-f6dc0452779a" />

<img width="1359" height="588" alt="Screenshot 2025-11-26 101750" src="https://github.com/user-attachments/assets/3053d2e2-6b8f-486e-b60b-0dda24336f23" />

<img width="1044" height="296" alt="Screenshot 2025-11-26 101816" src="https://github.com/user-attachments/assets/94eab74c-8539-49cf-be82-1e35714fff6e" />

<img width="1355" height="522" alt="Screenshot 2025-11-26 101843" src="https://github.com/user-attachments/assets/01d07154-cd1c-49d7-91ee-787c15687478" />


## Future Enhancements
- Video / real-time webcam detection
- Multi-model support
- Export results to PDF
- Mobile-friendly UI

** If you find it helpful, leave a STAR ⭐ **
