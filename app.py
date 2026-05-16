import streamlit as st
import cv2
import numpy as np
import os
import joblib
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

# กำหนดสไตล์หน้าเว็บ
st.set_page_config(page_title="Lung Cancer Classification", page_icon="🫁", layout="centered")

st.title("🫁 Lung Cancer Classification App")
st.write("ระบบวินิจฉัยมะเร็งปอดจากภาพถ่าย CT Scan ด้วยโมเดล (GLCM + SIFT + DenseNet121 + SVM)")
st.markdown("---")

# ====================================================================
# 1. ฟังก์ชันเตรียมและปรับสภาพรูปภาพ (Preprocessing)
# ====================================================================
def apply_preprocessing(img_gray):
    denoised = cv2.GaussianBlur(img_gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced

# ====================================================================
# 2. โหลดโมเดลเข้ามาทำงานในระบบ (และเคลียร์ปัญหาแคชค้าง)
# ====================================================================
@st.cache_resource
def load_all_models():
    # โหลดโครงสร้างโมเดลหลักเพื่อดึง 1,024 ฟีเจอร์ตามที่ StandardScaler คาดหวัง
    dense_extractor = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    svm_pipeline = joblib.load('lung_cancer_svm_pipeline.pkl')
    return dense_extractor, svm_pipeline

try:
    densenet_extractor, svm_pipeline = load_all_models()
    st.success("✅ โหลดระบบประมวลผลและโมเดลทำนายผลเรียบร้อยแล้ว!")
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    st.info("กรุณาตรวจสอบว่ามีไฟล์ `lung_cancer_svm_pipeline.pkl` อยู่ในโฟลเดอร์เดียวกับโค้ดนี้")

# ====================================================================
# 3. ส่วนอินเตอร์เฟซการรับรูปภาพและแสดงผลลัพธ์
# ====================================================================
uploaded_file = st.file_uploader("เลือกรูปภาพผลเอกซเรย์คอมพิวเตอร์ (CT Scan)...", type=["jpg", "jpeg", "png"])

CLASS_NAMES = ['Normal cases (ปกติ)', 'Bengin cases (เนื้องอกชนิดธรรมดา)', 'Malignant cases (มะเร็งปอด)']

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption='รูปภาพที่อัปโหลดเข้าสู่ระบบ', use_container_width=True)
    
    if st.button("🔍 เริ่มกระบวนการวิเคราะห์ภาพถ่าย"):
        with st.spinner("🔄 กำลังประมวลผลและสกัดลักษณะเด่นเชิงลึก (1,024 ฟีเจอร์)..."):
            # 3.1 ดำเนินการ Preprocessing รูปภาพ
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_preprocessed = apply_preprocessing(img_gray)
            
            # 3.2 ปรับขนาดรูปภาพให้ตรงสเปคอินพุตโมเดล (224x224x3)
            img_gray_resized = cv2.resize(img_preprocessed, (224, 224))
            img_rgb_resized = cv2.cvtColor(img_gray_resized, cv2.COLOR_GRAY2RGB)
            
            # 3.3 สกัด Deep Features ด้วย DenseNet121 (ได้ผลลัพธ์ขนาด 1,024 ฟีเจอร์พอดี)
            x_dl = np.expand_dims(img_rgb_resized, axis=0).astype(np.float32)
            x_dl = densenet_preprocess(x_dl)
            deep_features = densenet_extractor.predict(x_dl, verbose=0).reshape(1, -1)
            
        with st.spinner("🧠 โมเดลกำลังประเมินผลลัพธ์..."):
            try:
                # 3.4 สั่งรันโมเดลทำนายผลลัพธ์ผ่านตัวแปร 1,024 ฟีเจอร์
                prediction = svm_pipeline.predict(deep_features)[0]
                
                # แสดงผลลัพธ์ทางหน้าจอ
                st.markdown("---")
                st.subheader("📊 ผลการวิเคราะห์จากระบบ:")
                
                if prediction == 0:
                    st.success(f"**ผลลัพธ์:** {CLASS_NAMES[prediction]}")
                elif prediction == 1:
                    st.warning(f"**ผลลัพธ์:** {CLASS_NAMES[prediction]}")
                else:
                    st.error(f"**ผลลัพธ์:** {CLASS_NAMES[prediction]}")
                    
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในขั้นตอนทำนายผล: {e}")
