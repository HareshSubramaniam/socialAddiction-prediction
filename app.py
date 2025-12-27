import streamlit as st
import numpy as np
from trainmodel import train_model

# Page config
st.set_page_config(
    page_title="Social Media Addiction Predictor",
    layout="wide"
)

# Train model
model, scaler, le_dict, categorical_cols, scale_cols = train_model()

# 🔹 MAIN TITLE
st.markdown(
    "<h1 style='text-align:center; color:#2C3E50;'>📱 Social Media Addiction Health Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
 
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# 🔹 SIDEBAR INPUTS
st.sidebar.header("🧾 User Inputs")

age = st.sidebar.slider("Age", 15, 30, 20)
usage = st.sidebar.slider("Average Daily Usage (Hours)", 0.0, 15.0, 5.0)
sleep = st.sidebar.slider("Sleep Hours Per Night", 0.0, 10.0, 6.5)
mental = st.sidebar.slider("Mental Health Score (1–10)", 1, 10, 6)

# Categorical inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
academic = st.sidebar.selectbox("Academic Level", ["High School", "Undergraduate", "Postgraduate"])



st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Predict Health Status")

# 🔹 MAIN CONTENT AREA
if predict_btn:
    # Prepare user data
    user_dict = {
        "Avg_Daily_Usage_Hours": usage,
        "Sleep_Hours_Per_Night": sleep,
        "Mental_Health_Score": mental,
        "Gender": gender,
        "Academic_Level": academic,
        "Most_Used_Platform": platform,
        "Affects_Academic_Performance": affects_academic
    }

    # Encode categorical features
    for col in categorical_cols:
        le = le_dict[col]
        user_dict[col] = le.transform([user_dict[col]])[0]

    # Scale numerical features
    import numpy as np
    num_features = np.array([user_dict[col] for col in scale_cols]).reshape(1, -1)
    num_features_scaled = scaler.transform(num_features)

    # Combine features
    user_features = np.hstack([
        num_features_scaled,
        np.array([user_dict[col] for col in categorical_cols]).reshape(1, -1)
    ])

    # Predict
    prediction = model.predict(user_features)[0]

    st.subheader("📊 Prediction Result")

    if prediction == "Low":
        st.success("✅ HEALTH STATUS: GOOD")
        st.markdown("""
        ### 🟢 Professional Health Advice
        - You maintain **balanced social media usage**
        - Your sleep pattern and mental health are in a healthy range
        - Continue prioritizing **offline activities and real-world connections**
        - Regular breaks from screens help sustain long-term well-being
        """)
    else:
        st.error("⚠️ HEALTH STATUS: NEED IMPROVEMENT")
        advice_text = "- You need **more listening and awareness** to improve your mental health." if mental < 7 else "- Your screen time may be **negatively impacting your health**"
        st.markdown(f"""
        ### 🔴 Professional Health Advice
        {advice_text}
        - High social media usage is linked with sleep disruption
        - Consider **digital detox hours** daily
        - Increase physical activity and reduce late-night screen exposuregit add .

        - Seek professional guidance if mental stress persists
        """)

else:
    st.info(
        "👈 Use the inputs on the left and click **Predict Health Status** to view your results."
    )

st.markdown("---")

# 🔹 FOOTER
st.markdown(
    "<p style='text-align:center; font-size:14px; color:gray;'>"
    "⚙️ | Logistic Regression Model"
    "</p>",
    unsafe_allow_html=True
)
