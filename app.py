import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import io

# PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE ----------------
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

st.title("🏥 Diabetes Prediction System")
st.markdown("### AI-Powered Risk Assessment Tool")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler_svm.pkl")
    return model, scaler

model, scaler = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Patient Information")

patient_name = st.sidebar.text_input("Patient Name")

st.sidebar.subheader("Demographics")
age = st.sidebar.number_input("Age", 1, 100, 30)
pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 0)

st.sidebar.subheader("Medical Measurements")
glucose = st.sidebar.number_input("Glucose", 0, 200, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 130, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 10.0, 70.0, 25.0)
dpf = st.sidebar.number_input("DPF", 0.0, 2.5, 0.5)

predict_btn = st.sidebar.button("🔮 Predict")

# ---------------- SESSION ----------------
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# ---------------- PREDICTION ----------------
if predict_btn:
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_std = scaler.transform(input_data)

    prediction = model.predict(input_std)[0]

    try:
        prob = model.predict_proba(input_std)[0]
        prob_neg = prob[0] * 100
        prob_pos = prob[1] * 100
    except:
        prob_pos = 100 if prediction == 1 else 0
        prob_neg = 100 - prob_pos

    st.session_state.prediction = prediction
    st.session_state.prob_pos = prob_pos
    st.session_state.prob_neg = prob_neg
    st.session_state.predicted = True

# ---------------- DISPLAY ----------------
if st.session_state.predicted:

    prediction = st.session_state.prediction
    prob_pos = st.session_state.prob_pos
    prob_neg = st.session_state.prob_neg

    st.markdown("---")
    st.header("🎯 Prediction Result")

    if prediction == 1:
        st.error("🔴 HIGH RISK - DIABETIC")
    else:
        st.success("🟢 LOW RISK - NOT DIABETIC")

    col1, col2 = st.columns([1,2])

    with col1:
        st.metric("Non-Diabetic", f"{prob_neg:.1f}%")
        st.metric("Diabetic", f"{prob_pos:.1f}%")

        risk_label = "🟢 LOW RISK" if prob_pos < 30 else "🟡 MODERATE RISK" if prob_pos < 70 else "🔴 HIGH RISK"
        st.markdown(f"### {risk_label}")

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pos,
            number={'suffix': "%"},
            title={'text': "Risk %"},

            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},

                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],

                'threshold': {
                    'line': {'color': "black", 'width': 5},
                    'value': prob_pos
                }
            }
        ))

        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Risk Levels:
    🟢 Low Risk (0–30%)  
    🟡 Moderate Risk (30–70%)  
    🔴 High Risk (70–100%)
    """)

    # ---------------- PRECAUTIONS ----------------
    st.markdown("### 💡 Precautions & Recommendations")

    if prob_pos > 70:
        precautions = [
            "Consult a doctor immediately",
            "Monitor glucose daily",
            "Strict low sugar diet",
            "Regular exercise",
            "Avoid junk food"
        ]
    elif prob_pos > 30:
        precautions = [
            "Maintain healthy diet",
            "Exercise regularly",
            "Reduce sugar intake",
            "Routine health check"
        ]
    else:
        precautions = [
            "Maintain healthy lifestyle",
            "Regular exercise",
            "Balanced diet",
            "Routine checkups"
        ]

    for p in precautions:
        st.markdown(f"- {p}")

# ---------------- PDF ----------------
st.markdown("---")
st.subheader("Download Printable Report")

if st.session_state.predicted:

    prediction = st.session_state.prediction
    prob_pos = st.session_state.prob_pos
    prob_neg = st.session_state.prob_neg

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("<b>DIABETES MEDICAL REPORT</b>", styles['Title']))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles['Normal']))
    elements.append(Paragraph(f"<b>Age:</b> {age}", styles['Normal']))
    elements.append(Spacer(1, 15))

    data = [
        ["Parameter", "Your Value", "Male Normal", "Female Normal"],
        ["Pregnancies", pregnancies, "N/A", "0-10"],
        ["Glucose", glucose, "70-140", "70-140"],
        ["Blood Pressure", bp, "90-120", "90-120"],
        ["Skin Thickness", skin, "10-40", "10-40"],
        ["Insulin", insulin, "16-166", "16-166"],
        ["BMI", bmi, "18.5-24.9", "18.5-24.9"],
        ["DPF", dpf, "0.1-2.5", "0.1-2.5"],
    ]

    table = Table(data, colWidths=[120, 80, 120, 120])

    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    result_text = "HIGH RISK - DIABETIC" if prediction == 1 else "LOW RISK - NOT DIABETIC"

    elements.append(Paragraph(f"<b>Result:</b> {result_text}", styles['Heading2']))
    elements.append(Paragraph(f"Diabetic Probability: {prob_pos:.1f}%", styles['Normal']))
    elements.append(Paragraph(f"Non-Diabetic Probability: {prob_neg:.1f}%", styles['Normal']))

    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>Precautions</b>", styles['Heading2']))
    for p in precautions:
        elements.append(Paragraph(f"• {p}", styles['Normal']))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Consult a doctor for medical advice.", styles['Italic']))

    doc.build(elements)

    pdf = buffer.getvalue()
    buffer.close()

    st.download_button(
        "📥 Download PDF Report",
        data=pdf,
        file_name=f"{patient_name}_report.pdf",
        mime="application/pdf"
    )

else:
    st.info("👈 Fill details and click Predict")