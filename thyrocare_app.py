import streamlit as st
import numpy as np
import joblib
import pandas as pd
from PIL import Image

path_model_risk='model_risk.pkl'
path_model_diag='model_diag.pkl'
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
        <strong>ThyroCare</strong> — це перша AI-система в Україні, яка допомагає оцінити ризик появи злоякісної пухлини раку щитоподібної залози на основі 
        ваших медичних та персональних показників.</br>
        Наша платформа вирішує проблему затримки діагностики, відсутності персоналізованого прогнозу та складності самоконтролю для пацієнтів. </br> 
        Лише з нами
        ви зможете швидко та без зайвих витрат дізнатись свій ризик і не витрачати тижні на обстеження.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("<div class='card'>Введіть основні медичні показники, щоб оцінити ризик появи захворювання:</div>", unsafe_allow_html=True)

age = st.number_input("Вік", 0, 120, step=1)
gender = st.radio("Стать", ["Жіноча", "Чоловіча"])
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
st.markdown('<div class="country-label">Вкажіть країну вашого проживання.</div>', unsafe_allow_html=True)
st.markdown("Наприклад, " + ", ".join(f"`{c}`" for c in visible_countries))
country_input = st.text_input("Введіть без пробілів, з великої літери.").strip().lower()

if country_input:
    if country_input in country_to_code:
        country_code = country_to_code[country_input]
    else:
        country_code = max(country_to_code.values()) + 1
        country_to_code[country_input] = country_code
        st.info(f"Нова країна '{country_input}' додана з кодом {country_code}")
else:
    st.warning("❗ Введіть назву країни")
    country_code = -1  


# Етнічна група 
st.markdown('<div class="ethnicity-label">Вкажіть ваше етнічне походження.</div>', unsafe_allow_html=True)
st.markdown("Наприклад, `" + "`, `".join(ethnicity_to_code.keys()) + "`")
ethnicity_input = st.text_input("Введіть без пробілів.").strip().lower()

if ethnicity_input:
    if ethnicity_input in ethnicity_to_code:
        ethnicity_code = ethnicity_to_code[ethnicity_input]
    else:
        ethnicity_code = max(ethnicity_to_code.values()) + 1
        ethnicity_to_code[ethnicity_input] = ethnicity_code
        st.info(f"Нова етнічна група '{ethnicity_input}' додана з кодом {ethnicity_code}")
else:
    st.warning("❗ Введіть назву етнічної групи")
    ethnicity_code = -1


# Медичні показники 

family_history = st.radio("Чи є у вас сімейна історія захворювання?", ["Ні", "Так"])
radiation = st.radio("Чи мали випадок значного радіаційного опромінення?", ["Ні", "Так"])
iodine = st.radio("Чи є у вас дефіцит йоду?", ["Ні", "Так"])
smoking = st.radio("Чи палите ви?", ["Ні", "Так"])
obesity = st.radio("Наявність жиріння", ["Ні", "Так"])
diabetes = st.radio("Наявність цукрового діабету", ["Ні", "Так"])

tsh = st.number_input("TSH рівень", min_value=0.0, format="%.2f")
t3 = st.number_input("T3 рівень", min_value=0.0, format="%.2f")
t4 = st.number_input("T4 рівень", min_value=0.0, format="%.2f")
nodule = st.number_input("Розмір вузла (мм)", min_value=0.0, format="%.2f")

exclude_country_ethnicity = st.checkbox("Не використовувати дані про країну та етнічність для діагностики.")

if st.button("Продовжити"):
    with st.spinner('Модель прогнозує...'):
        # Затримка для симуляції часу обчислень (4 секунди)
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
    
    risk_labels = {1: "Низький ризик", 2: "Середній ризик", 3: "Високий ризик"}
    
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

    diagnosis_labels = {0: "Доброякісний вузол", 1: "Злоякісне утворення"}
    

    st.markdown("### 🩺 Результати прогнозу:")
    st.success(f"**Рівень ризику:** {risk_labels.get(predicted_risk, '???')}")
    st.info(f"**Ймовірний тип утворення:** {diagnosis_labels.get(predicted_diag, '???')}")
    
    if predicted_diag == 1:
        st.warning("⚠️ **Рекомендація:** Ймовірне злоякісне утворення. Рекомендується звернутися до лікаря для додаткових обстежень та консультацій.")
