import streamlit as st
import pandas as pd
import joblib


st.title('Salary Prediction ')


#load_model
model = joblib.load('salary_mul_model.pkl')
def predict_salary(experience, bachelor, masters, phd):
    user_data = {'Years of Experience': [experience],
                 "Bachelor's": [bachelor],
            "Master's": [masters],
            'PhD': [phd]}
    user_df = pd.DataFrame(user_data)
    predicted_salary = model.predict(user_df)
    return predicted_salary[0]

# User input

experience = st.slider("Years of Experience", min_value=0.0, max_value=20.0, value=10.5)
st.markdown('Select Your Highest Level of qualification:')
bachelor = st.checkbox("Bachelor's")
masters = st.checkbox("Master's")
phd = st.checkbox("PhD")
if st.button('Predict Salary'):
    predicted_salary = predict_salary(experience, bachelor, masters, phd)
    st.write('Salary = ', predicted_salary)

    

