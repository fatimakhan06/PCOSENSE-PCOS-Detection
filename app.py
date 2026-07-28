import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from PIL import Image

# 1. PAGE CONFIGURATION & CUSTOM CSS
st.set_page_config(page_title="PCOSense", page_icon="🌸", layout="centered")

st.markdown("""
    <style>
        /* 1. Global Background */
        .stApp { 
            background: linear-gradient(135deg, #FFF5F8 0%, #F0F4F8 100%); 
        }
        
        /* 2. Typography & Headers */
        h2 { 
            text-align: center; 
            color: #D81159 !important; /* Deep vibrant pink */
            font-family: 'Inter', sans-serif; 
            padding-top: 20px; 
            font-size: 46px !important;
            font-weight: 800 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
        }
        .subtitle { 
            text-align: center; 
            color: #5C677D; 
            font-size: 20px; 
            margin-bottom: 40px; 
            font-weight: 600;
        }
        label { 
            font-size: 16px !important; 
            color: #2B2D42 !important; 
            font-weight: 700 !important; 
        }
        
        /* 3. Styling the Input Fields (Numbers & Dropdowns) */
        div[data-baseweb="input"], div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 2px solid #FCD5CE !important;
            border-radius: 10px !important;
            transition: border-color 0.3s ease;
        }
        div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
            border-color: #FA8072 !important; /* Salmon color on focus */
            box-shadow: 0 0 5px rgba(250, 128, 114, 0.3) !important;
        }
        .stNumberInput input, div[data-baseweb="select"] { 
            font-size: 16px !important; 
            color: #2B2D42 !important;
        }
        
        /* 4. Styling the File Uploader */
        section[data-testid="stFileUploadDropzone"] {
            background-color: #FFFFFF;
            border: 2px dashed #2A9D8F; /* Teal border */
            border-radius: 15px;
            padding: 20px;
        }
        section[data-testid="stFileUploadDropzone"]:hover {
            background-color: #F0FAF9;
            border-color: #21867a;
        }
        
        /* 5. Custom Button Styling (Gradient & Shadow) */
        div.stButton > button:first-child { 
            background: linear-gradient(90deg, #D81159 0%, #FA8072 100%); 
            color: white; 
            border-radius: 30px; 
            padding: 10px 20px; 
            border: none; 
            margin-top: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(216, 17, 89, 0.3);
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover { 
            background: linear-gradient(90deg, #FA8072 0%, #D81159 100%); 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(216, 17, 89, 0.4);
        }
        div.stButton > button:first-child p { 
            color: white !important; 
            font-size: 22px !important; 
            font-weight: bold !important;
        }
        
        /* 6. Results Styling */
        .badge-low { 
            background-color: #E6F4EA; color: #1E8E3E; 
            padding: 12px 28px; border-radius: 30px; 
            font-weight: bold; font-size: 20px; 
            display: inline-block; border: 2px solid #A8DAB5; 
            margin-bottom: 20px; box-shadow: 0 4px 10px rgba(30, 142, 62, 0.1);
        }
        .badge-high { 
            background-color: #FCE8E6; color: #D93025; 
            padding: 12px 28px; border-radius: 30px; 
            font-weight: bold; font-size: 20px; 
            display: inline-block; border: 2px solid #F6B2AB; 
            margin-bottom: 20px; box-shadow: 0 4px 10px rgba(217, 48, 37, 0.1);
        }
        .confidence-track { 
            width: 100%; height: 16px; 
            background-color: #E2E8F0; 
            border-radius: 8px; overflow: hidden; 
            margin-top: 5px; margin-bottom: 25px;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* 7. Recommendation Boxes */
        .rec-box-safe { 
            background-color: #FFFFFF; padding: 25px; 
            border-radius: 15px; border-left: 6px solid #2A9D8F; 
            text-align: left; font-size: 18px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        .rec-box-safe li { color: #2A9D8F !important; margin-bottom: 8px; font-weight: 500;}
        
        .rec-box-risk { 
            background-color: #FFFFFF; padding: 25px; 
            border-radius: 15px; border-left: 6px solid #D81159; 
            text-align: left; font-size: 18px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        .rec-box-risk li { color: #D81159 !important; margin-bottom: 8px; font-weight: 500;}
        
        /* Divider Line Color */
        hr { border-top: 2px dashed #FCD5CE; }
    </style>
""", unsafe_allow_html=True)

# 2. MODEL ARCHITECTURE & LOADING
def se_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.multiply([input_tensor, se])

def spatial_attention(input_tensor):
    avg_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_tensor)
    max_pool = layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_tensor)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = layers.Conv2D(1, (7,7), padding='same', activation='sigmoid')(concat)
    return layers.multiply([input_tensor, attention])

def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), padding='same')(shortcut)
    x = layers.add([x, shortcut])
    return layers.Activation('relu')(x)

def build_image_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = base_model.output
    x = residual_block(x, 128)
    x = se_block(x)
    x = spatial_attention(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=base_model.input, outputs=output)

@st.cache_resource
def load_all_models():
    scaler = joblib.load('scaler.pkl')
    clinical_model = joblib.load('clinical_hybrid_model.pkl')
    img_model = build_image_model()
    img_model.load_weights('ultrasound_model.weights.h5')
    return scaler, clinical_model, img_model

try:
    scaler, clinical_model, image_model = load_all_models()
except Exception as e:
    st.error("Model files missing. Please ensure 'scaler.pkl', 'clinical_hybrid_model.pkl', and 'ultrasound_model.weights.h5' are in the script directory.")
    st.stop()

# 3. USER INTERFACE
st.markdown("<h2>🌸 PCOSense</h2>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Multimodal Polycystic Ovarian Syndrome Prediction System</div>", unsafe_allow_html=True)

# TOPIC 1: Clinical Data 
st.markdown("<h3 style='color: #D81159; margin-bottom: 15px;'>Clinical Data</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    pulse_rate = st.number_input("Pulse Rate (bpm)", min_value=0.00, step=1.00, value=0.00, format="%.2f")
    follicle_l = st.number_input("Follicle Count (Left)", min_value=0.00, step=1.00, value=0.00, format="%.2f")
    cycle_length = st.number_input("Cycle Length (days)", min_value=0.00, step=1.00, value=0.00, format="%.2f")
    amh = st.number_input("AMH Level (ng/mL)", min_value=0.00, step=0.10, value=0.00, format="%.2f")
    weight_gain = st.selectbox("Weight Gain", ["No", "Yes"]) 
    fast_food = st.selectbox("Fast Food Consumption", ["No", "Yes"])

with col2:
    follicle_r = st.number_input("Follicle Count (Right)", min_value=0.00, step=1.00, value=0.00, format="%.2f")
    cycle_ri = st.selectbox("Cycle Regularity", ["Regular", "Irregular"]) 
    prl = st.number_input("PRL Level (ng/mL)", min_value=0.00, step=0.10, value=0.00, format="%.2f")
    skin_darkening = st.selectbox("Skin Darkening", ["No", "Yes"])
    hair_growth = st.selectbox("Hair Growth", ["No", "Yes"])
    hip = st.number_input("Hip Circumference (inches)", min_value=0.00, step=1.00, value=0.00, format="%.2f")

st.markdown("---")

#  TOPIC 2: Ultrasound Data
st.markdown("<h3 style='color: #D81159; margin-bottom: 15px;'>Ultrasound Data</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Ultrasound Scan", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Scan Uploaded", width=300)

# 4. PREDICTION LOGIC
st.markdown("<br>", unsafe_allow_html=True)
c_left, c_mid, c_right = st.columns([1, 1, 1])

with c_mid:
    run_prediction = st.button("🔍 Predict PCOS", use_container_width=True)

if run_prediction:
    if uploaded_file is None:
        st.warning("⚠️ Please upload an ultrasound image to proceed.")
    else:
        with st.spinner("Analyzing patient data..."):
            map_yn = {"No": 0, "Yes": 1}
            cycle_map = {"Regular": 2, "Irregular": 4}
            
            input_data = pd.DataFrame([{
                'Follicle No. (L)': follicle_l,
                'Follicle No. (R)': follicle_r,
                'Weight gain(Y/N)': map_yn[weight_gain],
                'hair growth(Y/N)': map_yn[hair_growth],
                'Cycle length(days)': cycle_length,
                'Skin darkening (Y/N)': map_yn[skin_darkening],
                'AMH(ng/mL)': amh,
                'Hip(inch)': hip,
                'Fast food (Y/N)': map_yn[fast_food],
                'Cycle(R/I)': cycle_map[cycle_ri],
                'PRL(ng/mL)': prl,
                'Pulse rate(bpm) ': pulse_rate
            }])

            # Calculate Clinical Probability
            scaled_input = scaler.transform(input_data)
            clinical_prob = clinical_model.predict_proba(scaled_input)[0][1]

            # Calculate Ultrasound Probability
            image = Image.open(uploaded_file).convert('RGB')
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            image_prob = image_model.predict(img_array)[0][0]

            # Apply Weighted Likelihood (60% Image, 40% Clinical)
            final_prob = ((0.6 * image_prob) + (0.4 * clinical_prob)) * 100

            #DISPLAY RESULTS
            st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
            
            if final_prob >= 50.0:
                st.markdown(f"<div class='badge-high'>⚠ High Risk of PCOS</div>", unsafe_allow_html=True)
                bar_color = "linear-gradient(90deg, #FA8072 0%, #D81159 100%)"
            else:
                st.markdown(f"<div class='badge-low'>✓ Low Risk of PCOS</div>", unsafe_allow_html=True)
                bar_color = "linear-gradient(90deg, #2A9D8F 0%, #1E8E3E 100%)"

            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; font-size: 18px; font-weight: bold; color: #1E293B;">
                    <span>PCOS Likelihood</span>
                    <span style="font-size: 24px; color: #D81159;">{final_prob:.1f}%</span>
                </div>
                <div class="confidence-track">
                    <div style="height: 100%; background: {bar_color}; border-radius: 8px; width: {final_prob}%;"></div>
                </div>
            """, unsafe_allow_html=True)

            if final_prob >= 50.0:
                st.markdown("""
                    <div class="rec-box-risk">
                        <h4 style="margin-top: 0; color: #D81159; font-size: 22px;">Recommendations</h4>
                        <ul>
                            <li>Consult an endocrinologist or gynecologist for a formal diagnosis</li>
                            <li>Schedule hormonal blood tests (Testosterone, LH/FSH ratio)</li>
                            <li>Discuss potential lifestyle and dietary interventions with a professional</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="rec-box-safe">
                        <h4 style="margin-top: 0; color: #2A9D8F; font-size: 22px;">Recommendations</h4>
                        <ul>
                            <li>Maintain a healthy diet rich in whole grains, fruits, and vegetables</li>
                            <li>Engage in regular physical activity (at least 30 minutes daily)</li>
                            <li>Consult a gynecologist if symptoms persist or worsen</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)