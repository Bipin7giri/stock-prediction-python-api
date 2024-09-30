import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model

# Load the CSV data
df = pd.read_csv("data.csv")
df['date'] = pd.to_datetime(df['t'], unit='s')
df.set_index('date', inplace=True)

# Filter for the closing price
data = df.filter(['c'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.8)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)


# In simple terms, this code is **rescaling** the stock price data to fit within a range of **0 to 1**. 
# Think of it like this: if the stock prices range from $10 to $1000, the neural network might struggle to learn properly
# because the numbers are so spread out. By scaling the data, we're making sure all the prices fit within the same range (0 to 1),
# which helps the model understand the patterns better and train more efficiently.
# ### Steps:
# 1. **MinMaxScaler(feature_range=(0, 1))**: Sets up a tool to scale all the data so that it falls between 0 and 1.
# 2. **fit_transform(dataset)**: This applies the scaling to the data, converting each stock price based
# on its value relative to the smallest and largest prices in the dataset.
# In short, it makes the data easier for the model to work with by putting everything on the same scale.


# Create the training dataset
train_data = scaled_data[0:training_data_len, :]
# previous 60 days 
x_train = [] 

# future 60 days 
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert the training data to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)


# This part of the code is preparing data to train the LSTM model in a **very specific way**—by using past stock prices to predict future prices.

# ### 1. **Extracting the Training Data**
# ```python
# train_data = scaled_data[0:training_data_len, :]
# ```
# - This line selects the first 80% of the data (`scaled_data`) to be used as the training set. The training data consists of stock prices that have already been scaled to fall between 0 and 1.

# ### 2. **Creating `x_train` (Input) and `y_train` (Output)**
# python
# x_train = []  # This will hold the previous 60 days of data (input)
# y_train = []  # This will hold the price on the next day (output)
# 
# - **`x_train`**: This will contain the previous 60 days of stock prices (input data for the model).
# - **`y_train`**: This will contain the stock price on the **next** day (the output the model is trying to predict).

# ### 3. **Filling `x_train` and `y_train` with Data**
# ```python
# for i in range(60, len(train_data)):
#     x_train.append(train_data[i-60:i, 0])
#     y_train.append(train_data[i, 0])
# ```
# This loop goes through the training data and does the following:

# - **For `x_train`**: It grabs the **previous 60 days** of stock prices from `train_data` and appends that sequence as one input sample.
# - **For `y_train`**: It grabs the **next day’s stock price** right after the 60-day period and appends it as the corresponding output (the value you want to predict).

# ### Example:
# If your stock price data looks like this:

# | Day  | Stock Price |
# |------|-------------|
# |  0   | $100        |
# |  1   | $101        |
# |  ... | ...         |
# |  59  | $120        |
# |  60  | $121        |
# |  61  | $122        |
# |  62  | $123        |

# - **First `x_train`**: Contains the prices from day 0 to day 59: `[100, 101, ..., 120]`.
# - **First `y_train`**: Contains the stock price for day 60: `121`.

# - **Second `x_train`**: Contains the prices from day 1 to day 60: `[101, 102, ..., 121]`.
# - **Second `y_train`**: Contains the stock price for day 61: `122`.

# The loop continues, shifting the 60-day window each time, creating many examples for training.

# ### 4. **Convert Lists to Numpy Arrays**
# ```python
# x_train, y_train = np.array(x_train), np.array(y_train)
# ```
# - After the loop is done, the data in `x_train` and `y_train` are converted from Python lists into **numpy arrays**, which is the format required to feed data into machine learning models.

# ### Summary (in Simple Terms):
# - **`x_train`**: Contains chunks of **60 days of stock prices** (this is what the model uses to learn).
# - **`y_train`**: Contains the **price of the next day** (this is what the model is trying to predict).
# - The loop creates many examples of this 60-day input and next-day output to train the model on how to predict future stock prices based on past prices.





# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
#Sequential(): This initializes a new neural network model. 
# The Sequential model is a linear stack of layers, meaning you can add one layer after another in a straightforward way.
model = Sequential()



# LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)):
# LSTM(50): This adds an LSTM layer with 50 units (or neurons). The number of units determines how much information the LSTM layer can hold.
# return_sequences=True: This means that the layer will return the full sequence of outputs for each input sequence (useful when stacking multiple LSTM layers).
# input_shape=(x_train.shape[1], 1): This specifies the shape of the input data.
# x_train.shape[1] is 60 (the number of past days used for predictions),
# and 1 indicates that we have one feature (the stock price). This input shape tells the model what to expect when training.
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))



# Adding Another LSTM Layer
# This adds a second LSTM layer with 50 units as well.
# return_sequences=False: This means that this layer will only return the last output for each input sequence, 
# which is typical for the final LSTM layer when preparing to connect to dense layers for prediction.
model.add(LSTM(50, return_sequences=False))

# Adding Dense Layers
#Dense(25): This adds a fully connected (dense) layer with 25 units. 
# This layer helps the model learn complex patterns by processing the output from the previous LSTM layer.
model.add(Dense(25))

#Dense(1): This is the final layer that outputs a single value (the predicted stock price for the next day).
# Since we're predicting one continuous value (the price), this layer has 1 unit.
model.add(Dense(1))

#Input (60 days) --> LSTM Layer (50 units) --> LSTM Layer (50 units) --> Dense Layer (25 units) --> Output (Next day price)




# Compile the model

# model.compile(...):
# This method configures the model for training.
# It tells the model how to update its weights during training.

# optimizer='adam':
# The Adam optimizer is an advanced optimization algorithm that adjusts the learning rate during 
# training based on the gradients of the loss function.
# It combines the benefits of two other optimizers: AdaGrad and RMSProp. 
# Adam is popular because it works well in practice for many types of neural networks, 
# including LSTMs, and tends to converge faster compared to other optimizers.


# loss='mean_squared_error'
#The loss function measures
# how well the model's predictions match the actual target 
# values (the stock prices for the next day in this case).
# Mean Squared Error (MSE) is calculated as the average of the squares of the differences between predicted values and actual values. 
# It is commonly used for regression problems, like predicting stock prices.

# A lower MSE indicates a better fit of the model to the training data.

model.compile(optimizer='adam', loss='mean_squared_error')


# model.fit(...): This function is called to train the model on the provided data.
#Parameters Explained
  # x_train: This is the input data (the previous 60 days of stock prices) that the model will use to learn from.

  # y_train: This is the output data (the next day's stock price) that the model will try to predict.

  # batch_size=1:

  # This specifies how many samples (or data points) to process at one time before updating the model's weights.
  # A batch size of 1 means that the model will update its weights after processing each individual training example. 
  # This can lead to more frequent updates but may be less stable compared to larger batch sizes. However, 
  # it can also help the model learn faster in some cases.


# epochs=1:

# An epoch is one complete pass through the entire training dataset. 
# In this case, specifying epochs=1 means that the model will train for 
# only one full iteration over the training data.
# Generally, you would train for more epochs (like 10, 50, or even hundreds) 
# to allow the model to learn better from the data. Training for just one epoch might 
# not be sufficient to 
# fully train the model, 
# especially in complex tasks.

# model.fit(x_train, y_train, batch_size=1, epochs=1)

# Save the trained model
model.save('stock_lstm_model.h5')
