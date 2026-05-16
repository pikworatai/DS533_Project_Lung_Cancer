import streamlit as st
import cv2
import numpy as np
import os
import joblib
import tensorflow as tf
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

# กำหนดสไตล์หน้าเว็บ
st.set_page_config(page_title="Lung Cancer Classification", page_icon="🫁", layout="centered")

st.title("🫁 Lung Cancer Classification App")
st.write("ระบบวินิจฉัยมะเร็งปอดจากภาพถ่าย CT Scan ด้วยโมเดล Hybrid (GLCM + SIFT + DenseNet121 + SVM)")
st.markdown("---")

# ====================================================================
# 1. ฟังก์ชันดึงและสกัดลักษณะเด่น (Feature Extraction) ให้มิติตรงกับ Notebook
# ====================================================================
def apply_preprocessing(img_gray):
    denoised = cv2.GaussianBlur(img_gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced

def extract_glcm_features(gray_img_array):
    features = []
    for img in gray_img_array:
        # ใช้ระยะเดียว distances=[1] ตามสเปคที่สกัดใน Notebook เพื่อให้ได้ 5,302 ฟีเจอร์หลัก
        glcm = graycomatrix(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        asm = graycoprops(glcm, 'ASM').flatten()
        
        # นำลักษณะเด่นมารวมกันต่อ 1 รูปภาพ (จะได้ขนาดเวกเตอร์คงที่ตามที่โมเดลต้องการ)
        img_features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, asm])
        features.append(img_features)
    return np.array(features)

def extract_sift_bovw_live(img_gray, kmeans_model):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    
    # จำนวนคลาสเตอร์ที่ได้จากไฟล์ sift_kmeans.pkl (100 คลาสเตอร์)
    n_clusters = kmeans_model.n_clusters
    bovw_feature = np.zeros(n_clusters)
    
    if descriptors is not None:
        predictions = kmeans_model.predict(descriptors.astype(float))
        for pred in predictions:
            bovw_feature[pred] += 1
            
        # ทำ Normalization เวกเตอร์ความถี่ของคำ
        sum_feat = np.sum(bovw_feature)
        if sum_feat > 0:
            bovw_feature = bovw_feature / sum_feat
            
    return bovw_feature.reshape(1, -1)

def combine_features(feat_glcm, feat_sift, feat_densenet):
    # ปรับตำแหน่งมิติให้แบนและต่อกันเป็นมิติเดียวความยาวรวม 6,426 ฟีเจอร์พอดี
    feat_glcm_flat = feat_glcm.reshape(1, -1)
    feat_sift_flat = feat_sift.reshape(1, -1)
    feat_densenet_flat = feat_densenet.reshape(1, -1)
    
    return np.hstack([feat_glcm_flat, feat_sift_flat, feat_densenet_flat])

# ====================================================================
# 2. โหลดโมเดลทั้งหมดเข้ามาทำงานในระบบ
# ====================================================================
@st.cache_resource
def load_all_models():
    # 2.1 โหลดตัวสกัดฟีเจอร์สำเร็จรูปจาก Keras เพื่อหลีกเลี่ยงปัญหาเวอร์ชันโมเดลในไฟล์ .h5/.keras
    dense_extractor = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    
    # 2.2 โหลดตัวแปลงคำของ SIFT และ Pipeline ของ SVM
    kmeans_model = joblib.load('sift_kmeans.pkl')
    svm_pipeline = joblib.load('lung_cancer_svm_pipeline.pkl')
    
    return dense_extractor, kmeans_model, svm_pipeline

try:
    densenet_extractor, kmeans_model, svm_pipeline = load_all_models()
    st.success("✅ โหลดระบบประมวลผลและโมเดลทั้งหมดเรียบร้อยแล้ว!")
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    st.info("กรุณาตรวจสอบว่ามีไฟล์ `lung_cancer_svm_pipeline.pkl` และ `sift_kmeans.pkl` อยู่ในโฟลเดอร์เดียวกับโค้ดนี้แล้ว")

# ====================================================================
# 3. ส่วนอินเตอร์เฟซการรับรูปภาพและแสดงผลลัพธ์
# ====================================================================
uploaded_file = st.file_uploader("เลือกรูปภาพผลเอกซเรย์คอมพิวเตอร์ (CT Scan)...", type=["jpg", "jpeg", "png"])

CLASS_NAMES = ['Normal cases (ปกติ)', 'Benign cases (เนื้องอกชนิดธรรมดา)', 'Malignant cases (มะเร็งปอด)']

if uploaded_file is not None:
    # อ่านและแสดงผลรูปภาพ
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption='รูปภาพที่อัปโหลดเข้าสู่ระบบ', use_container_width=True)
    
    if st.button("🔍 เริ่มกระบวนการวิเคราะห์ภาพถ่าย"):
        with st.spinner("🔄 กำลังประมวลผลและหลอมรวมคุณลักษณะเด่น (Fused Features)..."):
            # ขั้นตอนเตรียมรูปภาพ
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_preprocessed = apply_preprocessing(img_gray)
            
            # ย่อรูปภาพสำหรับนำไปใช้งานสกัดฟีเจอร์หลัก
            img_gray_resized = cv2.resize(img_preprocessed, (224, 224))
            img_rgb_resized = cv2.cvtColor(img_gray_resized, cv2.COLOR_GRAY2RGB)
            
            # สกัดลักษณะเด่นย่อยทั้ง 3 รูปแบบ
            feat_glcm = extract_glcm_features([img_gray_resized])
            feat_sift = extract_sift_bovw_live(img_gray_resized, kmeans_model)
            
            # สกัดลักษณะเด่นเชิงลึกด้วย DenseNet121
            x_dl = np.expand_dims(img_rgb_resized, axis=0).astype(np.float32)
            x_dl = densenet_preprocess(x_dl)
            feat_densenet = densenet_extractor.predict(x_dl, verbose=0)
            
            # หลอมรวมเวกเตอร์ลักษณะเด่นทั้งหมดเข้าด้วยกัน
            fused_features = combine_features(feat_glcm, feat_sift, feat_densenet)
            
        with st.spinner("🧠 โมเดลกำลังประเมินผลลัพธ์..."):
            try:
                # ทำนายผลด้วย SVM Pipeline (มี StandardScaler อยู่ภายในแล้ว)
                prediction = svm_pipeline.predict(fused_features)
