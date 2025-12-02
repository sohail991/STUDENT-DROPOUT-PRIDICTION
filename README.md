
# Student Dropout Prediction — Flask Web App

## Overview

This project is a Machine Learning–based **Student Dropout Prediction System** built using **Flask (Python)**.  
It takes student-related input values and predicts whether a student is likely to **Dropout** or **Continue**.

The prediction is made using a trained **Support Vector Classifier (SVC)** stored in `class_svc.pkl`.

---

## Project Files

```
app.py                 → Flask backend application
templates/index.html   → Frontend form for user input
student-dropout-prediction.ipynb → Model training notebook
class_svc.pkl          → Trained ML model (must be in same folder)
README.md              → Documentation
```

---

## Features

- User-friendly web interface  
- Collects various academic and demographic inputs  
- Handles one-hot encoded features programmatically  
- Uses a trained ML model for predictions  
- Returns clear result:  
  - **YES — Student may Dropout**  
  - **NO — Student will Continue**

---

## How It Works

1. User enters required student information in HTML form  
2. `app.py` maps these inputs to **34 model features**  
3. The input is converted into NumPy array  
4. Flask sends input to the SVC model  
5. The model predicts dropout status  
6. Output is displayed on the webpage

---

## How to Run the Project

### 1. Install Required Libraries

```
pip install flask numpy pickle-mixin
```

### 2. Run Flask Application

```
python app.py
```

### 3. Open Browser

```
http://127.0.0.1:5000/
```

---

## Important Notes

- `class_svc.pkl` **must** be present in the same folder as `app.py`  
- Dummy feature names should be updated depending on your real dataset  
- The feature count is dynamically handled in the code  
- Errors are printed in console for debugging  

---

## Author

This project was created to demonstrate Flask-based ML model deployment.

