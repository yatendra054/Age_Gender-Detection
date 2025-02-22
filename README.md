## Age and Gender Predict..

## Overview
This project is a deep learning-based application that uses Convolutional Neural Networks (CNNs) to predict the age and gender of a person from an uploaded image or webcam feed. The model is integrated into a **Streamlit** web application for user-friendly interaction.

## Key Features
- **Deep Learning Model**: Built using **Keras** and **TensorFlow** from scratch.
- **Dataset**: Trained on a **12K dataset** sourced from Kaggle.
- **Data Preprocessing**: Preprocessing steps were performed to clean and standardize the dataset before training.
- **Web Application**: Implemented using **Streamlit** to allow users to upload images or use a webcam for predictions.

## Technologies Used
- Python
- TensorFlow & Keras
- OpenCV (for image processing)
- NumPy & Pandas
- Matplotlib & Seaborn (for data visualization)
- Streamlit (for the web interface)

## Installation
Follow these steps to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/age-gender-predict.git
cd age-gender-predict
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
