# 🌸 Flower Recognition using Convolutional Neural Network (CNN)

This project implements a flower recognition system using TensorFlow and Keras. It trains a CNN model on a dataset of flower images and classifies them into different categories.

---

## 📁 Project Structure

**flower_recognition/**
 - ├── flowers/ # Dataset folder with subfolders for each flower class
 - ├── model/ # Folder to save the trained model
 - ├── flower.py # Main Python script
 - └── README.md # Project documentation
   
---

## 🧠 Model Architecture

The CNN model uses the following layers:

- Rescaling (Normalization)
- Conv2D + MaxPooling (3 times)
- Flatten
- Dense (128 units)
- Output Dense (Number of classes)

---

## 📦 Requirements

Install the required Python packages:

```bash
pip install tensorflow matplotlib numpy
```

📂 **Dataset**
The dataset should be organized like this inside the flowers/ directory:
  -  flowers/
  -  ├── daisy/
  -  ├── dandelion/
  -  ├── roses/
  -  ├── sunflowers/
  -  └── tulips/

# How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/flower-recognition-cnn.git
   cd flower-recognition-cnn
   ```
2. Put the flower image dataset inside the flowers/ directory.
3. Run the training script:
   ```bash
   python flower.py
   ```
4. The trained model will be saved as:
      ```bash
   model/flower_model.h5
   ```
5. 📊 Results
   - The training script will output:
   - Training & validation accuracy
   - Training & validation loss
   - Accuracy/Loss graphs

🧠 **Future Improvements**
- Implement data augmentation

- Deploy model with a web interface using Flask/Streamlit

- Use transfer learning with pre-trained models like MobileNet or ResNet

    **Let me know if you want me to include badges, contribution guidelines at kunalrangnekar22@gmail.com, or deployment instructions in the README.**
  


   
   

