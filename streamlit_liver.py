import pickle
import numpy as np
import streamlit as st


liver_model = pickle.load(open('liver_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

st.title('Data Mining Prediksi Liver')


col1, col2 = st.columns(2)

with col1:
    Age = st.text_input('Input Umur', value='0')

with col2:
    Gender = st.text_input('Input Gender (L:1 / P:0)', value='0')

with col1:
    Total_Bilirubin = st.text_input('Input Nilai Total Bilirubin', value='0')

with col2:
    Direct_Bilirubin = st.text_input('Input Nilai Direct Bilirubin', value='0')

with col1:
    Alkaline_Phosphotase = st.text_input('Input Nilai Alkaline Phosphotase', value='0')

with col2:
    Alamine_Aminotransferase = st.text_input('Input Nilai Alamine Aminotransferase', value='0')

with col1:
    Aspartate_Aminotransferase = st.text_input('Input Nilai Aspartate Aminotransferase', value='0')

with col2:
    Total_Protiens = st.text_input('Input Nilai Total Protein', value='0')

with col1:
    Albumin = st.text_input('Input Nilai Albumin', value='0')

with col2:
    Albumin_and_Globulin_Ratio = st.text_input('Input Nilai Albumin and Globulin Ratio', value='0')


liv_diagnosis = ''



if st.button('Test Prediksi Liver'):
    try:
        input_data = np.array([[
            float(Age),
            float(Gender),
            float(Total_Bilirubin),
            float(Direct_Bilirubin),
            float(Alkaline_Phosphotase),
            float(Alamine_Aminotransferase),
            float(Aspartate_Aminotransferase),
            float(Total_Protiens),
            float(Albumin),
            float(Albumin_and_Globulin_Ratio)
        ]])

        std_data = scaler.transform(input_data)

        liv_prediction = liver_model.predict(std_data)

        if liv_prediction[0] == 1:
            liv_diagnosis = 'Pasien terkena penyakit liver'
        else:
            liv_diagnosis = 'Pasien tidak terkena penyakit liver'

        st.success(liv_diagnosis)
    except ValueError:
        st.error("Pastikan semua input berupa angka yang valid!")