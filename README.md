# 🩺 Diabetes Risk Prediction System (Deployment Repository)

## 📖 About the Project
The **Diabetes Risk Prediction System** is a machine learning application that predicts whether a person has a **High** or **Low** risk of diabetes based on 8 medical metrics (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function and Age). It uses a **Random Forest Classifier** to analyze the profile and also shows the estimated risk probability.

⚠️ **Disclaimer:** This is an educational screening tool. It is **not** a medical diagnosis. Always consult a doctor for proper testing.

⚠️ **Important Note:** This repository is structured and configured for **Render Deployment/Hosting**.

The complete training workflow is in `train_model.py`. It trains on the real Pima Indians Diabetes dataset if you place `diabetes.csv` in the project folder; otherwise it generates a Pima-style dataset automatically. To retrain, run:

```
python train_model.py
```

## 👩‍💻 Developers
**Mamta** (Roll No. 24011041049) and **Kirti** (Roll No. 24011041041)

## 🛠️ Tools & Technologies Used
* **Machine Learning:** Scikit-learn (Random Forest Classifier)
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Joblib
* **Web UI Framework:** Gradio
* **Cloud Hosting:** Render
* **Development Environment:** Python 3.12

## 📁 Project Files
* `app.py` – Gradio web app with input validation and prediction
* `diabetes_prediction_model.pkl` – trained Random Forest model
* `train_model.py` – script to train/retrain the model
* `requirements.txt` – dependencies
* `.python-version` – Python version for Render

## 💻 Run Locally
```
pip install -r requirements.txt
python app.py
```
Then open `http://localhost:10000`.

## 🚀 How to Host on Render
### Step 1: Prepare Your Code
1. Create a new **public** repository on GitHub.
2. Upload all files from this project to the repository.

### Step 2: Deploy on Render
1. Go to [Render.com](https://render.com/) and Log In or Sign Up.
2. Click **New** and select **Web Service**.
3. Connect your GitHub account and select your repository.
4. Fill in the deployment details:
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `python app.py`
5. Select the **Free** instance type.
6. Click **Create Web Service**.

Render will build your environment and give you a live URL.

## 🌐 Live Demo
👉 [Diabetes Risk Prediction System - Live App](https://diabetes-risk-prediction-system-by.onrender.com)
