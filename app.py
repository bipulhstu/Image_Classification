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
    # Resize image to 224x224
    image = image.resize((224, 224))
    # Convert to array
    image_array = np.array(image)
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    # Normalize pixel values
    image_array = image_array.astype('float32') / 255.0
    return image_array

def predict_flower(model, image, class_names):
    """Make prediction on the image"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        predicted_class = class_names[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [(class_names[i], prediction[0][i]) for i in top_3_idx]
        
        return predicted_class, confidence, top_3_predictions
    except Exception as e:
        st.error(f'❌ Error making prediction: {str(e)}')
        return None, None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🌸 Bangladeshi Flower Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Upload an image of a flower to identify its species using our trained MobileNetV2 model</p>', unsafe_allow_html=True)
    
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
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader('📤 Upload Image')
        uploaded_file = st.file_uploader(
            'Choose a flower image...',
            type=['jpg', 'jpeg', 'png'],
            help='Supported formats: JPG, JPEG, PNG'
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        if uploaded_file is not None:
            st.subheader('🔍 Prediction Results')
            
            # Make prediction
            with st.spinner('🌸 Analyzing flower...'):
                predicted_class, confidence, top_3_predictions = predict_flower(model, image, class_names)
            
            if predicted_class is not None:
                # Display main prediction
                st.markdown(f'''
                <div class="prediction-box">
                    <h2 style="text-align: center; color: #2E7D32; margin-bottom: 1rem;">
                        🌺 {predicted_class}
                    </h2>
                    <p style="text-align: center; font-size: 1.5rem; color: #333;">
                        Confidence: <strong>{confidence:.2%}</strong>
                    </p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Display top 3 predictions
                st.subheader('📊 Top 3 Predictions')
                for i, (flower, conf) in enumerate(top_3_predictions, 1):
                    st.write(f'{i}. **{flower}**: {conf:.2%}')
                    st.progress(float(conf))
        else:
            st.info('👆 Please upload an image to get started!')
    
    # Footer
    st.markdown('---')
    st.markdown('''
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🌸 Bangladeshi Flower Classifier | Built with Streamlit & TensorFlow
    </div>
    ''', unsafe_allow_html=True)

if __name__ == '__main__':
    main()