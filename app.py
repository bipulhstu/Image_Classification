import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os

# Configure page
st.set_page_config(
    page_title='🌸 Bangladeshi Flower Classifier',
    page_icon='🌸',
    layout='wide'
)

# Custom CSS
st.markdown('''
<style>
.main-header {
    font-size: 3rem;
    color: #2E7D32;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.description {
    font-size: 1.2rem;
    text-align: center;
    color: #555;
    margin-bottom: 2rem;
}
.prediction-box {
    background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and class names"""
    try:
        # Load model
        if os.path.exists('best_model.h5'):
            model = tf.keras.models.load_model('best_model.h5')
        elif os.path.exists('flower_classifier.h5'):
            model = tf.keras.models.load_model('flower_classifier.h5')
        else:
            st.error('❌ Model file not found! Please ensure the model file exists.')
            return None, None
            
        # Load class names
        if os.path.exists('class_names.pkl'):
            with open('class_names.pkl', 'rb') as f:
                class_names = pickle.load(f)
        else:
            # Default class names for Bangladeshi flowers
            class_names = [
                'Chandramallika', 'Cosmos Phul', 'Gada', 'Golap', 'Jaba',
                'Kagoj Phul', 'Noyontara', 'Radhachura', 'Rangan', 
                'Salvia', 'Sandhyamani', 'Surjomukhi', 'Zinnia'
            ]
            
        return model, class_names
    except Exception as e:
        st.error(f'❌ Error loading model: {str(e)}')
        return None, None

def preprocess_image(image):
    """Preprocess the image for prediction"""
    # Ensure RGB mode (handles PNG with alpha, grayscale, CMYK, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize image to 224x224
    image = image.resize((224, 224))
    # Convert to array
    image_array = np.array(image)
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    # Normalize pixel values
    image_array = image_array.astype('float32') / 255.0
    return image_array

def render_image(image, caption=None):
    """Safely render an image across all versions of Streamlit"""
    try:
        # Streamlit 1.43+ / 1.61+ (modern parameter)
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        try:
            # Streamlit 1.20 - 1.42
            st.image(image, caption=caption, use_container_width=True)
        except TypeError:
            # Legacy Streamlit fallback
            st.image(image, caption=caption)

def predict_flower(model, image, class_names):
    """Make prediction on the image"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [(class_names[i], float(prediction[0][i])) for i in top_3_idx]
        
        return predicted_class, confidence, top_3_predictions
    except Exception as e:
        st.error(f'❌ Error making prediction: {str(e)}')
        return None, None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🌸 Bangladeshi Flower Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Identify 13 species of Bangladeshi flowers in real time using our fine-tuned MobileNetV2 deep learning model</p>', unsafe_allow_html=True)
    
    # Load model
    model, class_names = load_model()
    
    if model is None or class_names is None:
        st.stop()
    
    # Sidebar with info
    with st.sidebar:
        st.header('📋 Model Information')
        st.write('**Architecture:** MobileNetV2 with Transfer Learning')
        st.write('**Accuracy:** 99.87% (Validation)')
        st.write('**Classes:** 13 Bangladeshi Flower Types')
        st.write('**Image Size:** 224x224 pixels')
        
        st.header('🌺 Supported Flowers')
        for i, flower in enumerate(class_names, 1):
            st.write(f'{i}. {flower}')
    
    # Discover available sample images
    sample_dir = 'sample_images'
    sample_files = []
    if os.path.exists(sample_dir):
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
        sample_files = [f for f in sorted(os.listdir(sample_dir)) if f.lower().endswith(valid_exts)]
    
    # Initialize default sample so the app immediately shows a live prediction
    if 'selected_sample' not in st.session_state and sample_files:
        st.session_state['selected_sample'] = sample_files[0]
    
    # Sample Images Showcase Section
    if sample_files:
        st.markdown('### 🖼️ Test with Sample Images')
        st.caption('Click any sample photo below to immediately classify it, or upload your own image below.')
        
        cols_per_row = 4
        for row_start in range(0, len(sample_files), cols_per_row):
            row_samples = sample_files[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_samples))
            for idx, sample_name in enumerate(row_samples):
                sample_idx = row_start + idx
                sample_path = os.path.join(sample_dir, sample_name)
                with cols[idx]:
                    try:
                        thumb = Image.open(sample_path)
                        render_image(thumb)
                    except Exception:
                        st.write(f"Sample #{sample_idx + 1}")
                    
                    is_current = (st.session_state.get('selected_sample') == sample_name)
                    btn_label = f"✓ Sample {sample_idx + 1}" if is_current else f"Test #{sample_idx + 1}"
                    btn_type = "primary" if is_current else "secondary"
                    if st.button(btn_label, key=f"btn_sample_{sample_idx}", type=btn_type, use_container_width=True):
                        st.session_state['selected_sample'] = sample_name
                        st.rerun()
        st.markdown('---')

    # Main content layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader('📤 Upload Custom Image')
        uploaded_file = st.file_uploader(
            'Upload a flower image from your device...',
            type=['jpg', 'jpeg', 'png', 'webp'],
            help='Supported formats: JPG, JPEG, PNG, WEBP'
        )
        
        # Determine active image (uploaded image takes priority over sample)
        active_image = None
        source_label = ""
        
        if uploaded_file is not None:
            active_image = Image.open(uploaded_file)
            source_label = f"📁 Uploaded: {uploaded_file.name}"
        elif st.session_state.get('selected_sample'):
            selected_path = os.path.join(sample_dir, st.session_state['selected_sample'])
            if os.path.exists(selected_path):
                active_image = Image.open(selected_path)
                sample_num = sample_files.index(st.session_state['selected_sample']) + 1 if st.session_state['selected_sample'] in sample_files else 1
                source_label = f"🌸 Sample Flower #{sample_num}"
        
        if active_image is not None:
            st.markdown(f"**Selected Source:** `{source_label}`")
            # Image details card
            st.info(f"📐 **Resolution:** {active_image.width} × {active_image.height} px | **Mode:** {active_image.mode}")
            render_image(active_image, caption=source_label)
        else:
            st.info('👆 Choose a sample image above or upload an image to begin.')
    
    with col2:
        st.subheader('🔍 Classification & Live Preview')
        
        if active_image is not None:
            # Display prominent image preview directly in the classification panel
            st.markdown("**Specimen Under Evaluation:**")
            render_image(active_image, caption=f"Active Input: {source_label}")
            
            with st.spinner('🌸 MobileNetV2 analyzing floral morphology and features...'):
                predicted_class, confidence, top_3_predictions = predict_flower(model, active_image, class_names)
            
            if predicted_class is not None:
                # Main prediction card
                st.markdown(f'''
                <div class="prediction-box">
                    <p style="text-align: center; font-size: 1rem; color: #555; margin: 0;">Predicted Species</p>
                    <h2 style="text-align: center; color: #2E7D32; margin: 0.5rem 0 1rem 0;">
                        🌺 {predicted_class}
                    </h2>
                    <p style="text-align: center; font-size: 1.4rem; color: #222; margin: 0;">
                        Confidence: <strong>{confidence:.2%}</strong>
                    </p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Top 3 breakdown
                st.markdown('#### 📊 Top 3 Class Probabilities')
                for i, (flower, conf) in enumerate(top_3_predictions, 1):
                    col_name, col_conf = st.columns([3, 1])
                    with col_name:
                        st.write(f"**{i}. {flower}**")
                    with col_conf:
                        st.write(f"**{conf:.2%}**")
                    st.progress(float(conf))
        else:
            st.info('👈 Select a sample above or upload an image to see prediction results.')
    
    # Footer
    st.markdown('---')
    st.markdown('''
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🌸 Bangladeshi Flower Classifier | Fine-tuned MobileNetV2 on ColoredFlowersBD
    </div>
    ''', unsafe_allow_html=True)

if __name__ == '__main__':
    main()