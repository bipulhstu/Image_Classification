# 🌸 Bangladeshi Flower Classifier

A deep learning-powered web application for classifying 13 different types of Bangladeshi flowers using MobileNetV2 architecture with transfer learning.

## 🌐 Live Demo

**🚀 Try the app now:** [https://flower-image-classify.streamlit.app/](https://flower-image-classify.streamlit.app/)

Upload any flower image and get instant classification results with confidence scores!

## 🌺 Overview

This project uses a fine-tuned MobileNetV2 model to identify and classify Bangladeshi flowers with an impressive **99.87% validation accuracy**. The model was trained on the ColoredFlowersBD dataset containing 13 different flower species commonly found in Bangladesh.

## 🎯 Supported Flower Classes

1. **Chandramallika** - Chrysanthemum
2. **Cosmos Phul** - Cosmos flower
3. **Gada** - Marigold
4. **Golap** - Rose
5. **Jaba** - Hibiscus
6. **Kagoj Phul** - Bougainvillea
7. **Noyontara** - Vinca/Periwinkle
8. **Radhachura** - Flame of the Forest
9. **Rangan** - Ixora
10. **Salvia** - Sage flower
11. **Sandhyamani** - Four o'clock flower
12. **Surjomukhi** - Sunflower
13. **Zinnia** - Zinnia flower

## 🏗️ Model Architecture

- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Transfer Learning**: Fine-tuned last 50 layers
- **Input Size**: 224×224×3
- **Output**: 13 classes (softmax activation)
- **Custom Head**: GlobalAveragePooling2D → Dropout(0.4) → Dense(13)
- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Function**: Categorical Crossentropy

## 📊 Model Performance

- **Training Accuracy**: 99.89%
- **Validation Accuracy**: 99.87%
- **Dataset Split**: 80% Training, 20% Validation
- **Total Images**: 7,927 images
- **Training Images**: 6,332 images
- **Validation Images**: 1,595 images

## 🔧 Data Preprocessing

- **Data Augmentation**: Rotation, width/height shift, shear, zoom, horizontal flip
- **Normalization**: Pixel values scaled to [0,1]
- **Image Resizing**: All images resized to 224×224 pixels
- **Batch Size**: 32

## 🚀 Deployment

### 🔧 Requirements Fix

If you encounter TensorFlow version compatibility issues during deployment, the requirements.txt has been updated with flexible version constraints that work across different Python versions and deployment platforms.

### Local Deployment

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Image_Classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files exist**:
   - `best_model.h5` or `flower_classifier.h5`
   - `class_names.pkl` (optional - defaults will be used)

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

### Cloud Deployment Options

#### Streamlit Community Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Community Cloud
3. Deploy directly from GitHub

#### Heroku
1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy using Heroku CLI or GitHub integration

## 🛌 Troubleshooting

### Common Deployment Issues

#### TensorFlow Version Compatibility
If you encounter TensorFlow version errors:
1. The requirements.txt uses flexible versioning (`tensorflow` instead of `tensorflow==2.13.0`)
2. This allows the deployment platform to choose the compatible version
3. For local development, you may need: `pip install tensorflow>=2.15.0`

#### Model File Issues
- Ensure either `best_model.h5` or `flower_classifier.h5` exists in the project directory
- Model files are large (24MB) - some platforms may have size limits
- Consider using Git LFS for model files in version control

#### Memory Issues
- TensorFlow models require sufficient RAM (recommend at least 1GB)
- Consider using smaller model architectures for resource-constrained environments

#### Google Cloud Run
1. Create Dockerfile
2. Build and push container image
3. Deploy to Cloud Run

## 📁 Project Structure

```
Image_Classification/
├── Image_Classification_Improved_Model.ipynb  # Main training notebook
├── app.py                                     # Streamlit web application
├── requirements.txt                           # Python dependencies
├── best_model.h5                             # Trained model (24MB)
├── flower_classifier.h5                      # Alternative model file
├── class_names.pkl                           # Class names (optional)
├── dataset/                                  # Dataset folder (excluded from git)
│   ├── train/                               # Training images
│   └── val/                                 # Validation images
└── README.md                                # This file
```

## 🖥️ Web Application Features

- **🔍 Real-time Classification**: Upload and classify flower images instantly
- **📊 Confidence Scores**: View prediction confidence and top-3 results
- **🎨 Beautiful UI**: Modern, responsive design with custom CSS
- **📱 Mobile Friendly**: Works seamlessly on desktop and mobile devices
- **ℹ️ Model Information**: Sidebar with model details and supported classes
- **🖼️ Multiple Formats**: Supports JPG, JPEG, and PNG image formats

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **Web Framework**: Streamlit
- **Image Processing**: PIL, OpenCV
- **Data Manipulation**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Model Architecture**: MobileNetV2

## 📈 Training Process

1. **Data Collection**: ColoredFlowersBD dataset from Kaggle
2. **Data Preprocessing**: Image augmentation and normalization
3. **Model Building**: MobileNetV2 with custom classification head
4. **Transfer Learning**: Fine-tuning approach with frozen initial layers
5. **Training**: With callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
6. **Evaluation**: 99.87% validation accuracy achieved

## 🔄 Model Training Workflow

```python
# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Base Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tuning: Unfreeze last 50 layers
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Custom Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(13, activation='softmax')(x)
```

## 📊 Performance Metrics

- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Validation Strategy**: 20% holdout
- **Early Stopping**: Patience of 10 epochs
- **Learning Rate Reduction**: Factor of 0.2 when plateau detected

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: ColoredFlowersBD from Kaggle
- **Architecture**: MobileNetV2 by Google
- **Framework**: TensorFlow/Keras team
- **Deployment**: Streamlit team
- **Inspiration**: Bangladeshi flora and biodiversity

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out through the repository issues or discussions.

---

**Happy Flower Classification! 🌸🌺🌻**