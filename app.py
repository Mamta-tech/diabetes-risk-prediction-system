import os
import joblib
import gradio as gr

# ==========================================================
# Load the trained model
# ==========================================================
try:
    deployed_rf = joblib.load("diabetes_prediction_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Model not found or error loading. {e}")
    deployed_rf = None


# ==========================================================
# Prediction Function with Error Handling
# ==========================================================
def predict_diabetes_risk(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age,
):
    values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age,
    ]

    # 1. Empty input check
    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please fill in all the input fields."

    # 2. Type casting
    try:
        pregnancies = int(pregnancies)
        glucose = float(glucose)
        blood_pressure = float(blood_pressure)
        skin_thickness = float(skin_thickness)
        insulin = float(insulin)
        bmi = float(bmi)
        diabetes_pedigree_function = float(diabetes_pedigree_function)
        age = int(age)
    except (ValueError, TypeError):
        return "❌ Please enter valid numeric values."

    numeric_values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age,
    ]

    # 3. Negative value check
    if any(v < 0 for v in numeric_values):
        return "❌ Negative values are not allowed for medical metrics."

    # 4. Range Validations
    if not (40 <= glucose <= 400):
        return "❌ Glucose must be between 40 and 400 mg/dL."

    if not (20 <= blood_pressure <= 200):
        return "❌ Blood pressure must be between 20 and 200 mm Hg."

    if not (10 <= bmi <= 70):
        return "❌ BMI must be between 10 and 70."

    if not (1 <= age <= 120):
        return "❌ Age must be between 1 and 120 years."

    if pregnancies > 20:
        return "❌ Number of pregnancies seems unusually high (Maximum 20)."

    if skin_thickness > 100:
        return "❌ Skin thickness must be 100 mm or less."

    if insulin > 900:
        return "❌ Insulin must be 900 mu U/ml or less."

    if diabetes_pedigree_function > 2.5:
        return "❌ Diabetes Pedigree Function must be 2.5 or less."

    if deployed_rf is None:
        return "❌ Model failed to load. Please check your diabetes_prediction_model.pkl file."

    try:
        # Input data in same order as model training
        input_data = [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree_function,
            age,
        ]]

        prediction = deployed_rf.predict(input_data)
        probability = float(deployed_rf.predict_proba(input_data)[0][1]) * 100

        if prediction[0] == 1:
            return (
                "🔴 Prediction Result\n\n"
                "Diabetes Risk: HIGH RISK\n\n"
                f"Estimated risk probability: {probability:.1f}%\n\n"
                "The applicant shows a high risk of diabetes. "
                "Please consult a doctor for proper medical testing."
            )
        else:
            return (
                "🟢 Prediction Result\n\n"
                "Diabetes Risk: LOW RISK\n\n"
                f"Estimated risk probability: {probability:.1f}%\n\n"
                "The applicant shows a low risk of diabetes. "
                "Maintain a healthy lifestyle and go for regular check-ups."
            )

    except Exception as e:
        return f"❌ Prediction failed.\n\nError:\n{str(e)}"


# ==========================================================
# Interface Setup
# ==========================================================
DESCRIPTION = """
# 🩺 Diabetes Risk Prediction System

This application predicts whether a person has a **High** or **Low** risk of diabetes using a trained **Random Forest Machine Learning Model**.

### Features
- ✅ Random Forest Machine Learning Model
- ✅ Input Validation
- ✅ Error Handling
- ✅ Instant Prediction with Risk Probability
- ✅ Easy-to-use Interface

⚠️ This is a screening tool made for educational purposes. It is not a medical diagnosis.
"""

developer_info = """
## 👩‍💻 About the Developers

**Created By:** Mamta and Kirti

**Roll No. (Kirti):** 24011041041

**Roll No. (Mamta):** 24011041049

**Project:** Diabetes Risk Prediction System

**Technology Used:**
- Python
- Scikit-learn
- Gradio
- Joblib
- Random Forest Classifier
"""

interface = gr.Interface(
    fn=predict_diabetes_risk,
    inputs=[
        gr.Number(label="Number of Pregnancies"),

        gr.Number(label="Glucose Level (mg/dL)"),

        gr.Number(label="Blood Pressure (mm Hg)"),

        gr.Number(label="Skin Thickness (mm)"),

        gr.Number(label="Insulin Level (mu U/ml)"),

        gr.Number(label="BMI (Body Mass Index)"),

        gr.Number(label="Diabetes Pedigree Function"),

        gr.Number(label="Age (years)"),
    ],

    outputs=gr.Textbox(
        label="Assessment Result",
        lines=8,
    ),

    title="🩺 Diabetes Risk Prediction System",

    description=DESCRIPTION,

    article=developer_info,

    theme="soft",
)

# ==========================================================
# Launch Configuration
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    print(f"Starting Gradio server on port {port}...")

    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
