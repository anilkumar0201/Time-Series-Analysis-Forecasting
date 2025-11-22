# Google Stock Price Forecasting

### *Time Series Modeling using ARIMA & Hybrid CNN–LSTM*

This project builds a complete **time series forecasting pipeline** using both **statistical** and **deep learning** approaches to predict Google’s future stock prices.
It includes **EDA, trend/seasonality analysis, ADF stationarity testing, ARIMA modeling**, and a **1D CNN–LSTM hybrid deep learning model**.

The goal of this project is to compare traditional time series models with deep learning architectures and understand how each performs on financial time-series data.

---

## Project Highlights

### Exploratory Data Analysis (EDA)

* Visualized trends in closing prices.
* Identified **moving averages (MA20, MA100)** for smoothing.
* Computed **returns & rolling volatility**.
* Built **monthly and weekday seasonality plots**.
* Generated a **correlation heatmap** for OHLC + volume.
* Plotted **candlestick charts** for financial visualization.

---

## Time Series Analysis

* Performed **Augmented Dickey–Fuller (ADF)** test to evaluate stationarity.
* Applied **differencing** (1st & 2nd order) to achieve stationarity.
* Plotted **ACF & PACF** for ARMA component selection.
* Automatically suggested ARIMA hyperparameters *(p, d, q)* using custom logic.

---

## ARIMA Forecasting

* Trained ARIMA model using identified (p, d, q) values.
* Compared **actual vs fitted values**.
* Computed:

  * **MAPE** =  **2.5%**

**→ ARIMA achieved strong forecasting performance with low MAPE.**

---

## Hybrid Deep Learning Model (1D CNN–LSTM)

This hybrid model extracts both:

* **Local temporal patterns** (via 1D CNN), and
* **Long-term dependencies** (via LSTM).

### Model architecture:

* 1D CNN layer (64 filters)
* MaxPooling
* LSTM (50 units)
* Dropout (0.2)
* Dense output layer
* Loss function: MSE

### Results:

* Achieved **Adjusted R² ≈ 0.67**
* CNN–LSTM predictions closely follow actual values.

---

##  Dataset

Dataset already uploaded 
Contains historical Google stock data:

* Date
* Open
* High
* Low
* Close
* Adj Close
* Volume

---

##  Tech Stack

### **Languages & Libraries**

* Python
* Pandas, NumPy
* Matplotlib, Seaborn, Plotly
* Statsmodels (ADF, ARIMA, ACF/PACF)
* Scikit-learn (metrics, scaling)
* TensorFlow / Keras (CNN-LSTM)

---

##  Workflow Overview

1. **Load & Preprocess Data**
2. **EDA + Visualization**
3. **Check stationarity (ADF test)**
4. **Differencing + ACF/PACF**
5. **Fit ARIMA model & forecast**
6. **Scale data for DL**
7. **Train CNN–LSTM model**
8. **Inverse transform predictions**
9. **Compare both models**


