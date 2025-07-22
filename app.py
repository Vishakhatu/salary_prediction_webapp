import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import base64
from io import BytesIO
import plotly.express as px  # required for visualizations

# Load trained model
model = joblib.load("best_model.pkl")

# Optional: Load the dataset for visuals
df = pd.read_csv("/content/adult 3 (1).csv")  # Make sure to update with correct file path

# Set page config
st.set_page_config(page_title=" Employee Salary Classification App", page_icon="💼", layout="centered")

# Title
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
workclass = st.sidebar.selectbox("Work Class", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"])
education_num = st.sidebar.slider("Education Number (Years)", 1, 16, 10)
occupation = st.sidebar.selectbox("Occupation", ["Tech-support","Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", ["United-States","Mexico", "Philippines", "Germany", "Canada", "India",
    "England", "Cuba", "Jamaica", "China", "Italy", "Puerto-Rico",
    "Vietnam", "South", "Columbia", "Japan", "Poland", "Iran",
    "Honduras", "Ireland", "Cambodia", "Thailand", "Laos", "Taiwan", "Haiti", "Portugal",
    "Dominican-Republic", "France", "Greece", "Hong", "Ecuador", "Peru", "Trinadad&Tobago", "Nicaragua",
    "Scotland", "Guatemala", "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Hungary", "Holand-Netherlands"])

# Input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'educational-num': [education_num],
    'occupation': [occupation],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Prediction
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    label = ">50K" if prediction == 1 else "<=50K"
    st.success(f"✅ Predicted Salary Category: {label}")

# Salary Optimization
st.markdown("---")
st.markdown("#### 💡 Salary Optimization Insights")

if st.button("Show Suggestions to Earn >50K"):
    insight_pred = model.predict(input_df)
    current_class = insight_pred[0]

    st.info(f"📊 Current Predicted Class: {'>50K' if current_class == 1 else '<=50K'}")

    suggestions = []
    if current_class == 0:
        st.markdown("#### Suggestions to potentially increase salary:")
        if input_df['educational-num'][0] < 13:
            suggestions.append("Consider pursuing higher education (like Bachelors or Masters).")
        if input_df['capital-gain'][0] < 5000:
            suggestions.append("Explore investment or gain-generating activities.")
        if input_df['hours-per-week'][0] < 40:
            suggestions.append("Increase work hours per week (40+ may lead to higher salary prediction).")
        if input_df['occupation'][0] in ['Other-service', 'Handlers-cleaners']:
            suggestions.append("Consider shifting to tech or professional specialties.")
        if input_df['workclass'][0] in ['Private', 'State-gov']:
            suggestions.append("Try to move into federal jobs or self-employment.")

        for s in suggestions:
            st.write(f"🔹 {s}")
    else:
        st.success("You're already in the >50K salary bracket!")

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Salary Optimization Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction: {'>50K' if current_class == 1 else '<=50K'}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Suggestions:", ln=True)
    for s in suggestions:
        pdf.multi_cell(0, 10, s)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_buffer = BytesIO(pdf_bytes)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name="salary_report.pdf",
        mime='application/pdf'
    )

# Visualization Section
st.markdown("### 📊 Explore Salary Insights")
analysis_option = st.selectbox(
    "Choose an insight to visualize:",
    [
        "Income Distribution Across Age Groups",
        "Top 10 Countries with Highest % Earning >50K",
        "Education Level vs Earning Capacity",
        "Gender-wise Salary Distribution"
    ]
)

if analysis_option == "Income Distribution Across Age Groups":
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100],
                             labels=['<25', '25–35', '35–45', '45–55', '55–65', '65+'])
    fig = px.histogram(df, x='age_group', color='income', barmode='group',
                       title="Income Distribution Across Age Groups")
    st.plotly_chart(fig)

elif analysis_option == "Top 10 Countries with Highest % Earning >50K":
    country_income = df.groupby('native-country')['income'].value_counts(normalize=True).unstack().fillna(0)
    country_income = country_income.sort_values('>50K', ascending=False).head(10)
    fig = px.bar(country_income['>50K'], title="Top 10 Countries with Highest % of >50K Earners")
    st.plotly_chart(fig)

elif analysis_option == "Education Level vs Earning Capacity":
    edu_income = df.groupby('education')['income'].value_counts(normalize=True).unstack().fillna(0)
    edu_income = edu_income.sort_values('>50K', ascending=False)
    fig = px.bar(edu_income['>50K'], title="Income >50K by Education Level")
    st.plotly_chart(fig)
    top_edu = edu_income['>50K'].idxmax()
    st.info(f"💡 Individuals with **{top_edu}** education are most likely to earn >50K.")

elif analysis_option == "Gender-wise Salary Distribution":
    gender_income = df.groupby('gender')['income'].value_counts(normalize=True).unstack().fillna(0)
    fig = px.bar(gender_income, barmode='group', title="Gender-wise Income Distribution")
    st.plotly_chart(fig)
    st.info("💡 Note: Gender disparity is visible in >50K earnings.")

