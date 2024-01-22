import streamlit as st
import numpy as np
import joblib

st.title('Salary Prediction')

regression = joblib.load('salary_prediction_model.pkl')
inp = st.number_input('Enter years of experience:')
inp = np.array([[inp]])
if st.button('Predict'):
    predicted_salary = regression.predict(inp)[0]
    st.write('Salary = ', predicted_salary)