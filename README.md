# Cervical Cancer Detection Using Cytopathology Images

This project uses **Convolutional Neural Networks (CNN)** under **Machine Learning** to detect cervical cancer stages from cytopathology images.

---

## 🚀 Project Overview

Cervical cancer is one of the most common and deadly diseases in women, especially in developing countries. Early detection can save lives. This system:
- Processes cervical cytopathology images
- Extracts features from cells
- Uses CNN to classify the stage of cervical cancer (Stage 1–4)

---

## 🧠 Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- Flask (for web interface)
- Google Colab / VS Code

---

## 🗃️ Project Structure
├── src/
│   ├── train_model.py          # Trains and saves the CNN model
│   └── utils.py                # (Optional) Helper functions if any
├── app.py                      # Flask app for frontend/backend integration
├── model/
│   └── cnn_model.h5            # Trained CNN model (tracked via Git LFS)
├── dataset/                    # Folder containing organized cervical cancer images
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview and instructions

