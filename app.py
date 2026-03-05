import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


# Page config
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction using Machine Learning")

st.markdown(
"""
This application predicts **car price based on engine size** using **Machine Learning regression models**.

Models used in this project:

• **Support Vector Regression (SVR)**  
• **Linear Regression**

You can interact with the model by selecting an **engine size** and seeing the predicted price.
"""
)

st.divider()


# Load dataset
dataset = pd.read_csv("Car_Price.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(-1,1)


# Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()

X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)


# Train SVR
svr = SVR(kernel="rbf")
svr.fit(X_scaled, y_scaled)


# Train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_scaled, y_scaled)


# Sidebar controls
st.sidebar.header("⚙️ User Input")

engine_size = st.sidebar.slider(
    "Select Engine Size (Liters)",
    1.0,
    4.0,
    2.5,
    0.1
)


# Predictions
svr_pred = sc_y.inverse_transform(
    svr.predict(sc_X.transform([[engine_size]])).reshape(-1,1)
)

lin_pred = sc_y.inverse_transform(
    lin_reg.predict(sc_X.transform([[engine_size]])).reshape(-1,1)
)


# Display predictions
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="SVR Predicted Price",
        value=f"${svr_pred[0][0]:,.0f}"
    )

with col2:
    st.metric(
        label="Linear Regression Price",
        value=f"${lin_pred[0][0]:,.0f}"
    )


st.divider()


# Model visualization
st.header("📈 Model Visualization")

X_original = sc_X.inverse_transform(X_scaled)

X_grid = np.arange(
    np.min(X_original),
    np.max(X_original),
    0.01
)

X_grid = X_grid.reshape((len(X_grid),1))


fig, ax = plt.subplots()

ax.scatter(
    sc_X.inverse_transform(X_scaled),
    sc_y.inverse_transform(y_scaled),
    color="red",
    label="Actual Prices"
)

ax.plot(
    X_grid,
    sc_y.inverse_transform(
        svr.predict(sc_X.transform(X_grid)).reshape(-1,1)
    ),
    color="blue",
    label="SVR Prediction"
)

ax.plot(
    X_grid,
    sc_y.inverse_transform(
        lin_reg.predict(sc_X.transform(X_grid)).reshape(-1,1)
    ),
    color="green",
    label="Linear Regression"
)

ax.set_xlabel("Engine Size (L)")
ax.set_ylabel("Car Price")
ax.set_title("Car Price Prediction Models")
ax.legend()

st.pyplot(fig)


st.divider()


# Dataset section
st.header("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(dataset)

with col2:
    st.subheader("Dataset Statistics")
    st.write(dataset.describe())


st.divider()


# Extra charts
st.header("📉 Exploratory Data Analysis")


# Scatter plot
fig2, ax2 = plt.subplots()

ax2.scatter(dataset["EngineSize"], dataset["Price"])
ax2.set_xlabel("Engine Size")
ax2.set_ylabel("Price")
ax2.set_title("Engine Size vs Price")

st.pyplot(fig2)


# Histogram
fig3, ax3 = plt.subplots()

ax3.hist(dataset["Price"], bins=10)
ax3.set_title("Price Distribution")

st.pyplot(fig3)


# Correlation
st.subheader("Feature Correlation")

corr = dataset.corr()

fig4, ax4 = plt.subplots()

cax = ax4.matshow(corr)
fig4.colorbar(cax)

ax4.set_xticks(range(len(corr.columns)))
ax4.set_yticks(range(len(corr.columns)))

ax4.set_xticklabels(corr.columns)
ax4.set_yticklabels(corr.columns)

st.pyplot(fig4)


st.divider()


# About section
st.header("ℹ️ About the Project")

st.markdown(
"""
### Machine Learning Workflow

1️⃣ Load dataset  
2️⃣ Extract features and target  
3️⃣ Apply **feature scaling**  
4️⃣ Train regression models  
5️⃣ Predict price for new engine sizes  
6️⃣ Visualize model predictions  

### Why SVR?

**Support Vector Regression (SVR)** is effective for modeling **nonlinear relationships** between variables.

In this dataset, **car price increases nonlinearly with engine size**, making SVR a strong choice.

### Technologies Used

• Python  
• Streamlit  
• Scikit-learn  
• NumPy  
• Pandas  
• Matplotlib
"""
)


st.caption("Created by Chinmay V Chatradamath 🚀")
