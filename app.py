import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load("heart_columns.pkl")
st.title("Heart Stroke Prediction")
st.markdown("Provide the following details to predict the risk of heart stroke.")


age = st.slider("Age",18,100,140)
sex = st.selectbox("Sex",["Male","Female"])
chest_pain_type = st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resting_bp = st.number_input("Resting Blood Pressure (in mm Hg)", 80, 200, 120) 
cholestrol = st.number_input("Cholestrol (in mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
resting_ecg = st.selectbox("Resting ECG",["Normal","ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina",["Yes","No"])
oldpeak = st.slider("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 10.0, 1.0)


if st.button("predict"):
    rew_input = {
        'Age': age,
        'Resting bp': resting_bp,
        'Cholestrol': cholestrol,
        'Fasting bs': fasting_bs,
        'Max hr': max_hr,
        'Oldpeak': oldpeak,
        'Sex_'+sex: 1,
        'chestpaintype_'+ chest_pain_type:1,
        'RestingECG_' + resting_ecg: 1,
        'exerciseAngina_'+ exercise_angina: 1,
        'st_slope_'+ st_slope : 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0       

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0] 

    if prediction == 1:
        st.error("high risk of heart stroke")

    else:
        st.success("low risk of heart stroke")
        
               
