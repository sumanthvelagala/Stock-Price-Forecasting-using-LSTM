# Microsoft Stock Price Prediction Using LSTM

In this project, I have used an LSTM-based deep learning model to predict Microsoft’s (MSFT) closing stock prices. This was a personal experiment to see how well a model could perform with just a few days of historical prices—and surprisingly, I achieved a **MAE of 10.5**, even with just the closing price as input.

# Timeline
May 2023

# Objective

I wanted to build a time-series model that could forecast the next day’s closing stock price using only the past few days of closing prices. I used a **sliding window** of size 3, meaning the model looks at the last 3 days to predict the next one.

# Packages
pip install numpy pandas matplotlib tensorflow scikit-learn

# Dataset Details

- Used Microsoft stock CSV file (`MSFT.csv`) with `Date` and `Close` columns  
- I used only the `Close` column as feature  
- The `Date` column was converted into a datetime format and set as the index for visualization purpose

# Pre-Processing

To train the model, I had to convert the continuous time-series data into supervised learning format using a **windowing** function.

# Preprocessing Steps:

1. **Windowing**: Created sequences of 3 consecutive days of closing prices → used them to predict the 4th day.  
2. **Data Split**:
   - 80% for training
   - 10% for validation
   - 10% for testing  
3. **Scaling**: Used `MinMaxScaler` to scale the values between 0 and 1, then reshaped the data into 3D format required by LSTM: `(samples, time_steps, features)`.


# Model Architecture

I built the model using **TensorFlow/Keras**, with two stacked LSTM layers and three dense layers at the end for regression:

```python
model = Sequential([
    Input(shape=(3, 1)),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
```
Loss Function: Mean Squared Error (mse)
Optimizer: Adam with learning rate 0.0001
Epochs: 100
Metrics: Mean Absolute Error (mae)

# Evaluation 
A Mean Absoulte Error Score of 10.5 was observed


