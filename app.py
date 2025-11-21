import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import pandas as pd

# Page config
st.set_page_config(
    page_title="Disaster Classification",
    page_icon="🚨",
    layout="wide"
)

# --- CSS Styles ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stat-box {
        background-color: #f8f9fa;
        border-left: 4px solid #FF4B4B;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    /* Force images to have rounded corners */
    img {
        border-radius: 8px;
    }
    .feature-box {
        background-color: #ffffff;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        transition: transform 0.2s;
    }
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'selected_image_path' not in st.session_state:
    st.session_state.selected_image_path = None
if 'input_source' not in st.session_state:
    st.session_state.input_source = None 

# --- Helper Function for Gallery ---
def make_square_thumbnail(img):
    """
    Crops an image to a square from the center to ensure
    uniform display in the gallery.
    """
    width, height = img.size
    new_size = min(width, height)
    
    left = (width - new_size)/2
    top = (height - new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2
    
    img = img.crop((left, top, right, bottom))
    img = img.resize((300, 300))
    return img

# --- Load Model ---
@st.cache_resource
def load_tflite_model_and_classes():
    try:
        interpreter = tf.lite.Interpreter(model_path='densenet.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        return interpreter, input_details, output_details, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

interpreter, input_details, output_details, class_names = load_tflite_model_and_classes()

# --- Functions ---
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def predict_tflite(interpreter, input_details, output_details, image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div class="info-card">
        <h2 style='margin:0; text-align:center;'>🚨 Disaster AI</h2>
        <p style='margin:5px 0 0 0; text-align:center; font-size:0.9rem;'>Advanced Disaster Classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("ℹ️ About This App")
    st.write("""
    This intelligent system uses **Deep Learning** and **Computer Vision** to automatically classify disaster types from images.
    
    **Supported Disaster Types:**
    """)
    
    disaster_info = {
        "🏚️ Earthquake": "Structural damage from seismic activity",
        "🔥 Urban Fire": "Fire incidents in urban areas",
        "🏔️ Land Slide": "Soil and rock movement disasters",
        "🌊 Water Disaster": "Floods and water-related emergencies"
    }
    
    for disaster, description in disaster_info.items():
        st.markdown(f"""
        <div class="feature-box">
            <strong>{disaster}</strong><br>
            <small>{description}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("🎯 How It Works")
    st.write("""
    1. **Upload** or select an example image
    2. **AI analyzes** the image using DenseNet121
    3. **Get instant** classification results
    4. **View confidence** scores for all categories
    """)
    
    st.markdown("---")
    
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Accuracy", "90%")
    with col2:
        st.metric("Test Accuracy", "88%")
    
    st.markdown("""
    <div class="stat-box">
        <strong>🏗️ Architecture</strong><br>
        DenseNet121 with Transfer Learning
    </div>
    <div class="stat-box">
        <strong>⚡ Model Type</strong><br>
        TensorFlow Lite (Optimized)
    </div>
    <div class="stat-box">
        <strong>🖼️ Input Size</strong><br>
        224 × 224 pixels
    </div>
    <div class="stat-box">
        <strong>📦 Model Size</strong><br>
        ~20 MB (Compressed)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("🔬 Technology Stack")
    st.write("""
    - **TensorFlow Lite** - Model inference
    - **DenseNet121** - CNN architecture
    - **Streamlit** - Web interface
    - **Python** - Backend processing
    """)
    
    st.markdown("---")
    
    st.header("⚠️ Disclaimer")
    st.warning("""
    This tool is for **educational and research purposes**. For real disaster response, always consult professional emergency services and authorities.
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Try different images to see how the model performs on various disaster scenarios!")

# --- Header ---
st.markdown('<h1 class="main-header">🚨 Disaster Classification System</h1>', unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;'>
    AI-powered disaster image classification using deep learning • Real-time analysis • High accuracy predictions
</p>
""", unsafe_allow_html=True)

# --- Main Layout ---
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📥 Step 1: Select Input")
    
    tab1, tab2 = st.tabs(["📤 Upload Image", "🖼️ Example Gallery"])
    
    with tab1:
        st.write("Upload your own disaster image for classification")
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], help="Supported formats: JPG, JPEG, PNG")
        if uploaded_file:
            st.session_state.input_source = 'upload'
            st.session_state.uploaded_file = uploaded_file
            st.success("✅ Image uploaded successfully!")

    with tab2:
        st.write("Click on any example image below to test the classifier:")
        example_images = {
            "earthquake.png": "Earthquake",
            "landslide.png": "Land Slide",
            "urbanfire.png": "Urban Fire",
            "waterdisaster.png": "Water Disaster"
        }
        example_dir = "example_images"
        
        cols = st.columns(2)
        for idx, (img_file, label) in enumerate(example_images.items()):
            img_path = os.path.join(example_dir, img_file)
            if os.path.exists(img_path):
                with cols[idx % 2]:
                    raw_img = Image.open(img_path)
                    square_img = make_square_thumbnail(raw_img)
                    
                    st.image(square_img, use_column_width=True)
                    
                    if st.button(f"Select {label}", key=f"btn_{idx}", use_container_width=True):
                        st.session_state.input_source = 'example'
                        st.session_state.selected_image_path = img_path
                        st.success(f"✅ {label} selected!")

# --- Processing Logic ---
image_to_process = None
if st.session_state.input_source == 'upload' and 'uploaded_file' in st.session_state:
    image_to_process = Image.open(st.session_state.uploaded_file)
elif st.session_state.input_source == 'example' and st.session_state.selected_image_path:
    image_to_process = Image.open(st.session_state.selected_image_path)

# --- Results Column ---
with col_results:
    st.subheader("🔍 Step 2: Analysis Results")
    
    if image_to_process:
        with st.spinner("🤖 AI is analyzing your image..."):
            processed_img = preprocess_image(image_to_process)
            prediction = predict_tflite(interpreter, input_details, output_details, processed_img)
            
            # Process results
            pred_idx = np.argmax(prediction[0])
            pred_class = class_names[pred_idx]
            confidence = prediction[0][pred_idx] * 100
            
            icon_map = {'Earthquake': '🏚️', 'Urban_Fire': '🔥', 'Land_Slide': '🏔️', 'Water_Disaster': '🌊'}
            icon = icon_map.get(pred_class, '🚨')

            # --- Result Layout ---
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                 st.image(image_to_process, caption="Analyzed Image", width=200)

            with res_col2:
                st.markdown(f"""
                <div class="prediction-card" style="margin-top:0;">
                    <h3 style='margin:0; color:#333;'>{icon} {pred_class}</h3>
                    <p style='margin-top:5px; color:#666;'>Confidence Score</p>
                    <h1 style='color:#FF4B4B; margin:0; font-size: 2rem;'>{confidence:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence indicator
                if confidence > 80:
                    st.success("✅ High confidence prediction!")
                elif confidence > 60:
                    st.warning("⚠️ Moderate confidence - results may vary")
                else:
                    st.error("❌ Low confidence - consider a clearer image")
            
            st.markdown("---")
            
            # --- Detailed Chart ---
            st.markdown("### 📊 Detailed Class Probabilities")
            st.write("Breakdown of confidence scores across all disaster categories:")
            
            df_probs = pd.DataFrame({
                'Class': class_names,
                'Probability': prediction[0]
            }).sort_values(by='Probability', ascending=False)
            
            for index, row in df_probs.iterrows():
                prob_percent = row['Probability'] * 100
                class_icon = icon_map.get(row['Class'], '📷')
                
                col_text, col_bar, col_val = st.columns([2, 5, 1])
                with col_text:
                    st.write(f"**{class_icon} {row['Class']}**")
                with col_bar:
                    st.progress(float(row['Probability']))
                with col_val:
                    st.write(f"{prob_percent:.1f}%")
                    
    else:
        st.markdown("""
        <div style='text-align: center; padding: 50px 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 20px;'>
            <h3 style='color: #666;'>👈 Get Started!</h3>
            <p style='color: #888;'>Upload an image or select an example from the left panel to begin disaster classification.</p>
            <br>
            <p style='color: #999; font-size: 0.9rem;'>The AI model will analyze your image and provide instant results with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 30px;'>
    <p style='margin: 10px 0; color: #666; font-size : 20px;'>Built with TensorFlow Lite & Streamlit | DenseNet121 Transfer Learning</p>
    <p style='margin: 10px 0; color: #666; font-size : 20px;'>Garent Ecklesia | Github: @GarentEcklesia

</p>
</div>
""", unsafe_allow_html=True)