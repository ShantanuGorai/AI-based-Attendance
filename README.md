# 🧠 AI-Based Face Recognition Attendance System

An intelligent attendance system that uses Artificial Intelligence and Computer Vision to automatically mark attendance using real-time face recognition. This system replaces manual attendance methods with a fast, accurate, and contactless solution.

---

## 🚀 Overview
This project captures live video through a webcam, detects faces, and recognizes individuals using pre-trained encodings. Once identified, attendance is automatically recorded with name, date, and timestamp.

---

## ✨ Features
- 🎥 Real-time face detection using webcam  
- 🧠 Face recognition using trained encodings  
- 📝 Automatic attendance marking (Name + Date + Time)  
- 📊 Stores attendance in CSV format  
- ❌ Prevents duplicate entries  
- ⚡ Fast and efficient system  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:**
  - OpenCV (Computer Vision)
  - face_recognition (Face Detection & Recognition)
  - NumPy (Numerical Computing)
  - Pandas (Data Handling)

---

## 📂 Project Structure
AI-Attendance-System/
│── images/ # Images of known faces
│── attendance.csv # Attendance records
│── main.py # Main script
│── requirements.txt # Dependencies
│── README.md # Documentation


---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/ShantanuGorai/AI-based-Attendance

# Navigate to project folder
cd AI-Attendance-System

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py