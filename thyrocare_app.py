import streamlit as st
import numpy as np
import joblib
import pandas as pd
from PIL import Image
import time


path_model_risk='model_risk.pkl'
path_model_diag='model_diag1.pkl'
path_scaler='scaler.pkl'
path_thresholds_risk='thresholds_risk.npy'
path_thresholds_diag='threshold_diag.npy'
path_model_diag2='model_diag2.pkl'

model_risk = joblib.load(path_model_risk)
model_diag = joblib.load(path_model_diag)
scaler = joblib.load(path_scaler)
thresholds_risk = np.load(path_thresholds_risk)
threshold_diag = np.load(path_thresholds_diag)
model_diag2=joblib.load(path_model_diag2)

st.set_page_config(page_title="ThyroCare", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Quicksand', sans-serif;
    }
    .main {
        background-color: #f5f8fa;
    }
    .title {
        color: #cc521d;
        font-size: 42px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 20px;
        color: #ed5c29;
        margin-bottom: 30px;
    }
    .card {
        background-color: #fff;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.07);
        margin-bottom: 30px;
        border: 2px solid #ffd3c5;
    }
    .result-risk {
        background-color: #ffecd9;
        border-left: 6px solid #ed5c29;
        padding: 20px;
        border-radius: 12px;
    }
    .result-diagnosis {
        background-color: #ffe9e4;
        border-left: 6px solid #cc521d;
        padding: 20px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

logo_path = "logo_super.png"
logo = Image.open(logo_path)

        
col1, col2 = st.columns([1, 5])

with col1:
    st.image(logo, width=60)

with col2:
    st.markdown("<div class='title'>ThyroCare</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Виявити ризик — зменшити загрозу</div>", unsafe_allow_html=True)


st.markdown("""
<div style='background-color:#fff4ed; padding:25px 40px; border-radius:18px; margin-top:10px; margin-bottom:30px;'>
    <p style='font-size:17px; line-height:1.6; margin:0; color:#3c3c3c;'>
        <strong>ThyroCare</strong> — це перша AI-система в Україні, яка допомагає оцінити
        ризик онкологічного захворювання щитовидної залози на основі ваших медичних та персональних даних.</br>
        Наша платформа вирішує проблему тривалості діагностики, відсутності персоналізованого прогнозу та можливості самоконтролю для пацієнтів. </br> 
        Лише з нами
        ви зможете швидко дізнатись свій потенційний ризик, не змарнувавши цінний час для лікування.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("<div class='card'>Введіть основні медичні показники, щоб оцінити ризик появи захворювання:</div>", unsafe_allow_html=True)

age = st.number_input("Вік", 0, 120, step=1, index=None)
gender = st.radio("Стать", ["Жіноча", "Чоловіча"],index=None)
gender_code = 0 if gender == "Жіноча" else 1


# Початкові мапінги 
country_mapping = {
    10: 'Україна', 0: 'Бразилія', 1: 'Китай', 2: 'Німеччина', 3: 'Індія', 4: 'Японія',
    5: 'Нігерія', 6: 'росія', 7: 'Південна Корея', 8: 'Велика Британія', 9: 'США'
}
ethnicity_mapping = {
    5: "Слов'янин", 0: 'Африканець', 1: 'Азієць', 2: 'Кавказець', 3: 'Латиноамериканець', 4: 'Близькосхідний'
}
country_to_code = {v.lower(): k for k, v in country_mapping.items()}
visible_countries = [name for name in country_mapping.values() if name.lower() != "росія"]

ethnicity_to_code = {v.lower(): k for k, v in ethnicity_mapping.items()}

st.markdown("""
    <style>
        .country-label, .ethnicity-label {
            font-size: 19px;  
            font-weight: 600;  
        }
    </style>
""", unsafe_allow_html=True)

# Країна 
# Країна
st.markdown('<div class="country-label">Вкажіть країну вашого проживання.</div>', unsafe_allow_html=True)

country_input = st.selectbox(
    "Виберіть країну", 
    options=[""] + visible_countries + ["Інше"], 
    help="Виберіть вашу країну зі списку або введіть нову."
)

if country_input == "Інше":
    country_input = st.text_input("Введіть назву країни").strip().lower()
    if country_input:
        country_code = max(country_to_code.values()) + 1  
        country_to_code[country_input] = country_code  
    else:
        country_code = -1  
elif country_input:
    country_code = country_to_code.get(country_input.lower(), -1)
else:
    country_code = -1  

# Етнічна група
st.markdown('<div class="ethnicity-label">Вкажіть ваше етнічне походження.</div>', unsafe_allow_html=True)

ethnicity_input = st.selectbox(
    "Виберіть етнічну групу", 
    options=[""] + list(ethnicity_to_code.keys()) + ["Інше"],  
    help="Виберіть вашу етнічну групу зі списку або введіть нову."
)

if ethnicity_input == "Інше":
    ethnicity_input = st.text_input("Введіть етнічну групу").strip().lower()
    if ethnicity_input:
        ethnicity_code = max(ethnicity_to_code.values()) + 1  
        ethnicity_to_code[ethnicity_input] = ethnicity_code  
    else:
        ethnicity_code = -1  
elif ethnicity_input:
    ethnicity_code = ethnicity_to_code.get(ethnicity_input.lower(), -1)
else:
    ethnicity_code = -1  


# Медичні показники 

family_history = st.radio("Чи є у вас сімейна історія захворювання?", ["Так","Ні"], index=None)
radiation = st.radio("Чи мали випадок значного радіаційного опромінення?", ["Так","Ні"], index=None)
iodine = st.radio("Чи є у вас дефіцит йоду?", ["Так","Ні"], index=None)
smoking = st.radio("Чи палите ви?", ["Так","Ні"], index=None)
obesity = st.radio("Наявність ожиріння", ["Так","Ні"], index=None)
diabetes = st.radio("Наявність цукрового діабету", ["Так","Ні"], index=None)

tsh = st.number_input(
    "Рівень тиреотропного гормону (у мкМО/мл):", 
    min_value=0.0, 
    format="%.2f",
    step=0.1,
    index=None,
    help="Тиреотропний гормон регулює роботу щитовидної залози.\nНормальний рівень для дорослої людини: від 0,3-0,4 до 4,0-4,2 мкМО/мл."
)

t3 = st.number_input(
    "Рівень трийодтироніну (у мкг/дл):", 
    min_value=0.0, 
    format="%.2f", 
    step=0.1,
    index=None,
    help="Трійодтиронін (T3) є активною формою тиреоїдних гормонів і важливий для регуляції метаболізму в організмі.\nНормальний рівень для дорослої людини: 0,8–2,0 мкг/дл. "
)

t4 = st.number_input(
    "Рівень тироксину (у мкг/дл):", 
    min_value=0.0, 
    format="%.2f", 
    step=0.1,
    index=None,
    help="Тироксин (T4) є основним гормоном, що виробляється щитовидною залозою. Нормальний рівень для дорослої людини: 5–12 мкг/дл."
)

nodule = st.number_input(
    "Розмір вузла (у см):", 
    min_value=0.0, 
    format="%.2f", 
    step=0.1,
    index=None,
    help="Вузол — це аномальне утворення в тканині щитовидної залози, яке може бути доброякісним або злоякісним. Вкажіть розмір цього утворення, якщо воно у вас є.")

exclude_country_ethnicity = st.checkbox("Не використовувати дані про країну та етнічність для діагностики.")

if st.button("Отримати прогноз"):
    if (age == 0 or gender is None or country_code == -1 or ethnicity_code == -1 or
        family_history is None or radiation is None or iodine is None or smoking is None or 
        obesity is None or diabetes is None or tsh == 0.0 or t3 == 0.0 or t4 == 0.0 ):
        
        st.warning("❗ Будь ласка, заповніть всі поля для отримання прогнозу.")
    else:
        with st.spinner('АІ-система створює персональний прогноз...'):
            time.sleep(4)
        
        user_input = {
            'Age': age,
            'Gender': gender_code,
            'Country': country_code,
            'Ethnicity': ethnicity_code,
            'Family_History': 1 if family_history == "Так" else 0,
            'Radiation_Exposure': 1 if radiation == "Так" else 0,
            'Iodine_Deficiency': 1 if iodine == "Так" else 0,
            'Smoking': 1 if smoking == "Так" else 0,
            'Obesity': 1 if obesity == "Так" else 0,
            'Diabetes': 1 if diabetes == "Так" else 0,
            'TSH_Level': tsh,
            'T3_Level': t3,
            'T4_Level': t4,
            'Nodule_Size': nodule
        }
        try:
            user_input['Combination'] = str(user_input['Smoking']) + str(user_input['Obesity']) + str(user_input['Diabetes'])
            user_input['Combination'] = int(user_input['Combination'], 2)
        except Exception as e:
            st.error(f"Помилка при обчисленні ознак: {e}")
        
        X_user = pd.DataFrame([user_input])
        X_user = X_user.drop(columns=['Diabetes', 'Obesity', 'Smoking'])  

        cols_to_scale = ['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
        X_user[cols_to_scale] = scaler.transform(X_user[cols_to_scale])

        y_prob_risk = model_risk.predict_proba(X_user)[0]
        thresholds_risk = np.load(path_thresholds_risk)
        if y_prob_risk[0] >= 0.5:  # Клас 1 
            predicted_risk = 1
        elif y_prob_risk[1] >= 0.5:  # Клас 2 
            predicted_risk = 2
        elif y_prob_risk[2] >= 0.5:  # Клас 3 
            predicted_risk = 3
        else:
            predicted_risk = 3
    
        risk_labels = {1: "низький", 2: "середній", 3: "високий"}
    
        if exclude_country_ethnicity:
            X_user = X_user.drop(columns=['Country', 'Ethnicity'])  
            model_diag_used = model_diag2  
            threshold_diag = 0.355  
        else:
            model_diag_used = model_diag  
            threshold_diag = 0.24  
    
        y_prob_diag = model_diag_used.predict_proba(X_user)[0]

        if y_prob_diag[1] >= threshold_diag:
            predicted_diag = 1  
        else:
            predicted_diag = 0  

        diagnosis_labels = {0: "доброякісний", 1: "злоякісний"}
    

        st.markdown("### 🩺 Результати прогнозу:")
        st.success(f"**Ризик розвитку онкологічного захворювання:** {risk_labels.get(predicted_risk, '???')}")
        st.info(f"**Ймовірний характер вузлів/пухлини:** {diagnosis_labels.get(predicted_diag, '???')}")
    
        if predicted_diag == 1:
            st.warning("⚠️ **Рекомендація:** Ймовірне злоякісне утворення. Рекомендується звернутися до лікаря для додаткових обстежень та консультацій.")

        
