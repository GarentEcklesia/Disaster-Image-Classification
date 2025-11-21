# 🚨 Disaster Classification System

An intelligent, real-time computer vision application designed to assist in rapid disaster response. This project utilizes **Transfer Learning** with the **DenseNet121** architecture to classify disaster scenes into critical categories (Earthquake, Fire, Flood, Landslide) via a lightweight web interface.

## 🚀 Live Demo

Test the model in real-time with our interactive web app:

[**➡️ Click here to launch the Streamlit App**](https://disaster-image-classification-garent-ecklesia.streamlit.app/)

## 💡 Features

* **Multi-Class Detection:** Capable of distinguishing between four distinct disaster types: **Earthquake**, **Urban Fire**, **Water Disaster**, and **Landslide**.
* **Edge-Optimized Performance:** Powered by **TensorFlow Lite**, ensuring fast inference speeds and low memory usage suitable for web and edge deployment.
* **Interactive Gallery:** Includes a built-in gallery of test images for instant demonstration without needing to upload files.
* **Visual Analytics:** Provides a confidence score and a probability distribution chart for every prediction to ensure transparency in results.
* **User-Friendly Interface:** A clean, responsive Streamlit UI that works seamlessly across desktop and mobile devices.

## ⚙️ Tech Stack

* **Deep Learning Framework:** TensorFlow, Keras
* **Model Architecture:** DenseNet121 (Transfer Learning)
* **Inference Engine:** TensorFlow Lite (TFLite)
* **Image Processing:** Pillow (PIL), NumPy
* **Data Manipulation:** Pandas
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Cloud

## 🧠 Model & Research Details

* **Architecture:** **DenseNet121** (Densely Connected Convolutional Networks). This architecture was chosen for its parameter efficiency and feature reuse, making it highly effective for image classification tasks with limited computational resources.
* **Methodology:**
    1.  **Preprocessing:** Input images are resized to **224x224** pixels and normalized to a [0,1] range.
    2.  **Transfer Learning:** Utilizes weights pre-trained on **ImageNet** to extract robust visual features, with a custom classification head trained on the disaster dataset.
    3.  **Optimization:** The final model is converted to `.tflite` format, optimizing the computation graph for lower latency inference.
* **Output:** The model outputs a probability distribution across the 4 classes, with the highest confidence score determining the final classification.

## 🛠️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/GarentEcklesia/disaster-classification-streamlit](https://github.com/GarentEcklesia/disaster-classification-streamlit)
    cd disaster-classification-streamlit
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## 📬 Contact

Created by **Garent Ecklesia**. Feel free to reach out for collaboration or questions!
