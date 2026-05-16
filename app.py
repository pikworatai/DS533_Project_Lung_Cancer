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

# สร้างตัวแปร SIFT สำหรับประมวลผล
sift = cv2.SIFT_create()
CLASS_NAMES = ['Normal cases', 'Benign cases', 'Malignant cases']

# ====================================================================
# ฟังก์ชันดึงและสกัดลักษณะเด่น
# ====================================================================
def apply_preprocessing(img_gray):
    denoised = cv2.GaussianBlur(img_gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced

def extract_glcm_features(gray_img_array):
    features = []
    for img in gray_img_array:
        glcm = graycomatrix(img, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        feature_vector = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
        features.append(feature_vector)
    return np.array(features, dtype=np.float32)

def extract_sift_bovw_live(gray_img, kmeans_model):
    # ดึงจำนวน Cluster จากตัว KMeans โดยตรง ป้องกันปัญหาเรื่องคลาสหาย
    n_clusters = kmeans_model.n_clusters
    _, descriptors = sift.detectAndCompute(gray_img, None)
    
    histogram = np.zeros(n_clusters, dtype=np.float32)
    if descriptors is not None and len(descriptors) > 0:
        preds = kmeans_model.predict(descriptors)
        for p in preds:
            histogram[p] += 1.0
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
            
    return np.array([histogram], dtype=np.float32)

def combine_features(*arrays):
    return np.concatenate(arrays, axis=1).astype(np.float32)

# ====================================================================
# โหลดโมเดลเข้าสู่ระบบแอปพลิเคชัน (Cached เพื่อความรวดเร็ว)
# ====================================================================
@st.cache_resource
def load_all_models():
    densenet_extractor = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
    
    pipeline_path = 'lung_cancer_svm_pipeline.pkl'
    kmeans_path = 'sift_kmeans.pkl'
    
    if os.path.exists(pipeline_path) and os.path.exists(kmeans_path):
        classifier = joblib.load(pipeline_path)
        kmeans_model = joblib.load(kmeans_path)
    else:
        st.error("❌ ไม่พบไฟล์โมเดลพยากรณ์ (.pkl) ในโฟลเดอร์ทำงานหลักของ GitHub!")
        classifier, kmeans_model = None, None
        
    return densenet_extractor, classifier, kmeans_model

densenet_extractor, classifier_model, kmeans_model = load_all_models()

# ====================================================================
# โครงสร้างหน้าต่าง UI แอพเว็บ
# ====================================================================
st.title("🫁 Lung Cancer Classification App")
st.write("ระบบวิเคราะห์จำแนกโรคจากภาพเอกซเรย์ปอดด้วยเทคนิค Feature Fusion (GLCM + SIFT + DenseNet121)")

uploaded_file = st.file_uploader("กรุณาอัปโหลดไฟล์ภาพเอกซเรย์ของคุณ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and classifier_model is not None and kmeans_model is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption='รูปภาพที่อัปโหลดเข้าสู่ระบบ', use_container_width=True)
        
        with st.spinner("🔄 กำลังประมวลผลวิเคราะห์รูปภาพความละเอียดสูง..."):
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_preprocessed = apply_preprocessing(img_gray)
            img_gray_resized = cv2.resize(img_preprocessed, (224, 224))
            img_rgb_resized = cv2.cvtColor(img_gray_resized, cv2.COLOR_GRAY2RGB)
            
            # สกัดคุณลักษณะเด่นแบบผสม
            feat_glcm = extract_glcm_features([img_gray_resized])
            feat_sift = extract_sift_bovw_live(img_gray_resized, kmeans_model)
            
            x_dl = np.expand_dims(img_rgb_resized, axis=0).astype(np.float32)
            x_dl = densenet_preprocess(x_dl)
            feat_densenet = densenet_extractor.predict(x_dl, verbose=0)
            
            # หลอมรวมฟีเจอร์ 
            fused_features = combine_features(feat_glcm, feat_sift, feat_densenet)
            
            # ส่งทำนายผลลัพธ์
            prediction = classifier_model.predict(fused_features)
            predicted_class_id = int(prediction[0])
            result_label = CLASS_NAMES[predicted_class_id]
            
        st.success(f"🎯 ผลการวินิจฉัยจากโมเดลคำนวณคือ: **{result_label}**")
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {e}")
