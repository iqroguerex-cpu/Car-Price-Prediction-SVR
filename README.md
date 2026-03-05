# 🚗 Car Price Prediction using SVR

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?logo=streamlit)](https://inhznfjq8txzv6j9gj8yax.streamlit.app/)

A simple **Machine Learning web application** that predicts **car prices based on engine size** using **Support Vector Regression (SVR)**.

The model is trained on a dataset of engine sizes and corresponding car prices and visualizes the relationship using an SVR regression curve.

---

## 🚀 Features

* Predict car price based on **engine size**
* Interactive **Streamlit web interface**
* **Support Vector Regression (RBF Kernel)**
* Visualization of **actual data vs model predictions**
* Simple ML project demonstrating regression workflows

---

## 🛠 Tech Stack

* Python
* Streamlit
* Scikit-learn
* NumPy
* Pandas
* Matplotlib

---

## 📈 How It Works

1. The dataset contains **engine sizes and car prices**.
2. Data is **scaled using StandardScaler**.
3. A **Support Vector Regression (SVR)** model is trained with an **RBF kernel**.
4. Users select an **engine size**, and the model predicts the **car price**.
5. The app visualizes **actual data points and the SVR prediction curve**.

---

## ▶️ Run the App Locally

Clone the repository:

```bash
git clone https://github.com/yourusername/car-price-svr.git
cd car-price-svr
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
car-price-svr
│
├── app.py
├── Car_Price.csv
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Author

**Chinmay V Chatradamath**

---

⭐ If you found this project useful, consider giving it a **star on GitHub**.
