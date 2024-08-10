#!/usr/bin/env python
# coding: utf-8

# # NLP Disaster Tweets Kaggle Mini-Project

# Data Source: https://www.kaggle.com/c/nlp-getting-started/overview

# Git: https://github.com/AnushaK7018/introduction-to-deep-learning-wk4

# # Project Description:
# 
# ## Project Objective: 
# 
# The project involves building a binary classification model to predict whether a tweet is about a real disaster or not. This problem falls under the category of Natural Language Processing (NLP). The task is to analyze the text data provided in the dataset and predict whether the context is related to a disaster.
# 
# ## **Project Goal:** 
# 
# The goal of this NLP project is to build a model that can classify whether a tweet is about a real disaster or not. This binary classification problem involves natural language processing techniques to process and analyze text data, followed by building a deep learning model to make predictions.
# 
# ## **Data Description:**
# 
# The dataset contains three CSV files: train.csv, test.csv, and sample_submission.csv.
# 
# **train.csv:** This is the training dataset. It includes columns: id, keyword, location, text, and target.
# 
# id: Unique identifier for each tweet.
# 
# keyword: A specific keyword from the tweet.
# 
# location: The location from which the tweet was sent.
# 
# text: The content of the tweet.
# 
# target: Binary label indicating if the tweet is about a disaster (1) or not (0).
# 
# **test.csv:** This is the test dataset. It includes columns: id, keyword, location, and text. The target column is absent because this is the data on which predictions will be made.
# 
# **sample_submission.csv:** A sample submission file showing the expected format for the final predictions.

# ## **Importing Data**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
sample_submission_df = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# # **Exploratory Data Analysis (EDA)**

# Inspect, Visualize, and Clean the Data

# In[ ]:


# Display basic information about the datasets
print("Training Data Info:")
print(train_df.info())
print("\nTest Data Info:")
print(test_df.info())
print("\nSample Submission Data Info:")
print(sample_submission_df.info())

# Display first few rows of the datasets
print("\nTraining Data Sample:")
print(train_df.head())
print("\nTest Data Sample:")
print(test_df.head())
print("\nSample Submission Data Sample:")
print(sample_submission_df.head())


# In[ ]:


train_df["length"] = train_df["text"].apply(lambda x : len(x))
test_df["length"] = test_df["text"].apply(lambda x : len(x))

print("Train Length Stat")
print(train_df["length"].describe())
print()

print("Test Length Stat")
print(test_df["length"].describe())


# Visualize the distribution of the tweet lengths and check the balance of the target variable in the training dataset.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of tweet lengths in the training data
plt.figure(figsize=(10, 6))
sns.histplot(train_df['length'], bins=30, kde=True, color='blue')
plt.title('Distribution of Tweet Lengths in Training Data')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of the target variable in the training data
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=train_df, palette='viridis')
plt.title('Distribution of Target Variable in Training Data')
plt.xlabel('Target (0 = Not Disaster, 1 = Disaster)')
plt.ylabel('Count')
plt.show()

# Check the balance of the target variable
target_counts = train_df['target'].value_counts()
target_counts


# **The EDA visualizations provide the following insights:**
# 
# **Distribution of Tweet Lengths:** The length of tweets in the training data varies, with most tweets having around 100 characters. This distribution is fairly consistent, suggesting that the text length does not drastically differ across samples.
# 
# **Distribution of the Target Variable:** The target variable shows some imbalance, with 4342 tweets labeled as non-disaster (0) and 3271 tweets labeled as disaster-related (1). While the imbalance isn't extreme, it may still require consideration during model training.

# # **Model Architecture**

# **Preprocessing the Data**

# In[ ]:


import nltk
nltk.download('stopwords')


# In[ ]:


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize objects
stop = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = Tokenizer(num_words=20000, lower=True)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

# Apply preprocessing to train and test data
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# Fit tokenizer on the processed text
tokenizer.fit_on_texts(train_df['text'].values)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_df['text'].values)
test_sequences = tokenizer.texts_to_sequences(test_df['text'].values)

# Pad sequences
max_len = 100
train_padded = pad_sequences(train_sequences, maxlen=max_len)
test_padded = pad_sequences(test_sequences, maxlen=max_len)

train_padded.shape, test_padded.shape


# **The output ((7613, 100), (3263, 100)) indicates the shapes of the padded sequences for both the training and test datasets:**
# 
# **Training Data:** There are 7,613 samples, and each sample has been padded or truncated to a length of 100 tokens (words).
# 
# **Test Data:** There are 3,263 samples, and each sample is also padded or truncated to 100 tokens.
# 
# This confirms that the text data has been successfully preprocessed and transformed into sequences of uniform length, which is crucial for feeding into the neural network model.

# **Building and Training the Model:**

# **Build the Model**

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional

# Define model architecture with input_length specified
model = Sequential()
model.add(Embedding(input_dim=20000, output_dim=128, input_length=100))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))

# Manually build the model (optional, usually not necessary if input_length is specified)
model.build(input_shape=(None, 100))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()


# **Train the Model**

# In[ ]:


from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_padded, train_df['target'].values, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), verbose=2)


# **Evaluate the Model**

# In[ ]:


# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")


# **Analysis:**
# 
# **Overfitting:** The model's training accuracy improved significantly with each epoch, reaching over 95% by the final epoch. However, the validation accuracy started to plateau and even dropped slightly, while the validation loss increased. This is a classic sign of overfitting, where the model performs well on the training data but struggles to generalize to unseen data.
# 
# **Validation Performance:** Although the validation accuracy started at a reasonable level (80.50% in the first epoch), it declined slightly as the model continued to train, indicating that the model is memorizing the training data rather than learning generalizable patterns.

# **Next Steps:**
# 
# **Hyperparameter Tuning:** We can perform hyperparameter tuning to optimize the model further. This includes adjusting the LSTM units, dropout rates, learning rates, batch sizes, and epochs to find the best combination that minimizes overfitting.
# 
# **Regularization Techniques:** Introduce regularization techniques such as L2 regularization or increase dropout rates to prevent overfitting.
# 
# **Model Complexity:** Consider simplifying the model by reducing the number of LSTM units or layers to prevent the model from becoming too complex for the given data.
# 
# **Early Stopping:** Implement early stopping during training to halt the process when the validation loss stops improving, which can prevent overfitting.

# # **Results and Analysis**

# **Visualize Training History**

# In[ ]:


import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# # **Hyperparameter Tuning**

# **Random Hyperparameter Search:**
# 
# Perform hyperparameter tuning to optimize the model's performance by using the Random Search approach to explore different combinations of hyperparameters such as LSTM units, dropout rates, learning rates, batch sizes, and epochs.

# In[ ]:


import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define the ranges of hyperparameters
lstm_units = [32, 64, 128]
dropouts = [0.2, 0.3, 0.5]
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
epochs = [5, 10, 20]

# Randomly select a combination of hyperparameters
random_lstm_units = random.choice(lstm_units)
random_dropout = random.choice(dropouts)
random_learning_rate = random.choice(learning_rates)
random_batch_size = random.choice(batch_sizes)
random_epochs = random.choice(epochs)

# Print selected hyperparameters
print(f"Selected Hyperparameters: LSTM Units={random_lstm_units}, Dropout={random_dropout}, "
      f"Learning Rate={random_learning_rate}, Batch Size={random_batch_size}, Epochs={random_epochs}")

# Define and compile the model
model = Sequential()
model.add(Embedding(input_dim=20000, output_dim=128, input_length=100))
model.add(SpatialDropout1D(random_dropout))
model.add(Bidirectional(LSTM(random_lstm_units, dropout=random_dropout, recurrent_dropout=random_dropout)))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=random_learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_padded, train_df['target'].values, test_size=0.2, random_state=42)

# Train the model with the selected hyperparameters
history = model.fit(X_train, y_train, epochs=random_epochs, batch_size=random_batch_size, validation_data=(X_val, y_val), verbose=2)


# # **Analysis:**
# 
# **Selected Hyperparameters:** LSTM Units=64, Dropout=0.2, Learning Rate=0.01, Batch Size=128, Epochs=10
# 
# **Model Performance:** The selected hyperparameters yielded an initial improvement in validation accuracy, reaching up to 80.30% in the first epoch. However, as training progressed, the validation accuracy declined, and the validation loss increased. This pattern suggests that the model began to overfit the training data after the first epoch.
# 
# **Dropout and Regularization:** The high dropout rate (0.2) was intended to help prevent overfitting, but it appears that even with this dropout, the model overfits after a few epochs. The learning rate of 0.01 might be too high, leading to rapid convergence on the training data but poor generalization to the validation data.
# 
# **Impact of Hyperparameters:**
# 
# **LSTM Units:** The lower number of LSTM units (32) may have restricted the model's capacity to learn complex patterns, contributing to the early overfitting.
# 
# **Learning Rate:** A lower learning rate could potentially allow the model to learn more gradually and avoid overfitting so quickly.

# **Updated Hyperparameter Tuning Code with Lower Learning Rate:**
# 
# Testing a lower learning rate to see if it improves the model's performance and helps mitigate overfitting. We'll adjust the learning rate to 0.001 and keep the other hyperparameters the same.

# In[ ]:


from tensorflow.keras.optimizers import Adam

# Updated learning rate
lower_learning_rate = 0.0001

# Define and compile the model with the lower learning rate
model = Sequential()
model.add(Embedding(input_dim=20000, output_dim=128, input_length=100))
model.add(SpatialDropout1D(0.5))  # Keeping dropout the same as before
model.add(Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.5)))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=lower_learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model with the same batch size and epochs
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_val, y_val), verbose=2)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")


# The model's performance didn't improve with the lower learning rate. In fact, the model appears to be stuck, with both the training and validation accuracy hovering around 57%. This suggests that the model might not be learning effectively, possibly due to the lower learning rate being too conservative.

# **Updated Hyperparameter Tuning Code with Keras Tuner:**

# This code is to automatically find the best combination of hyperparameters (dropout rates, LSTM units, learning rate) that lead to the highest validation accuracy. Instead of manually trying out different values, the RandomSearch tuner efficiently explores the hyperparameter space to find a good model configuration.

# In[ ]:


# Install the keras-tuner library
import keras_tuner
from kerastuner.tuners import RandomSearch


# In[ ]:


from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional

# Define a model-building function
def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=20000, output_dim=128, input_length=100))
    model.add(SpatialDropout1D(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32),
                                 dropout=hp.Float('dropout_lstm', 0.2, 0.5, step=0.1),
                                 recurrent_dropout=hp.Float('recurrent_dropout', 0.2, 0.5, step=0.1))))
    model.add(Dense(1, activation='sigmoid'))

    # Adjust the learning rates to explore a wider range
    optimizer = Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 5e-4]))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Initialize the tuner with more trials for better exploration
tuner = RandomSearch(build_model,
                     objective='val_accuracy',
                     max_trials=10,  # Increased number of trials for better exploration
                     executions_per_trial=1,
                     directory='my_dir',
                     project_name='disaster_tweets')

# Run the search with a slightly longer training duration
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))


# **Evaluate the Best Model:**

# In[ ]:


# Get the best hyperparameters and the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

# Print the best hyperparameters
print(f"""
The optimal number of LSTM units: {best_hps.get('units')}
The optimal dropout rate: {best_hps.get('dropout')}
The optimal LSTM dropout rate: {best_hps.get('dropout_lstm')}
The optimal recurrent dropout rate: {best_hps.get('recurrent_dropout')}
The optimal learning rate: {best_hps.get('learning_rate')}
""")

# Evaluate the best model on the validation set
val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
print(f"Best Model Validation Loss: {val_loss}")
print(f"Best Model Validation Accuracy: {val_accuracy}")


# In[ ]:


# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]
# Evaluate the best model on the validation set
val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
print(f"Best Model Validation Loss: {val_loss}")
print(f"Best Model Validation Accuracy: {val_accuracy}")

# Retrain the best model on the full training data
history = best_model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val), verbose=2)

# Evaluate the model again after additional training
val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss after additional training: {val_loss}")
print(f"Validation Accuracy after additional training: {val_accuracy}")


# # **Analysis:**
# 
# **Model Generalization:** The model's validation accuracy of 81.55% and relatively low validation loss of 0.4368 indicate that the chosen hyperparameters allow the model to generalize well on unseen data. The lower dropout rates and the learning rate of 0.001 seem to have helped the model avoid overfitting while still learning effectively.
# 
# **Stable Performance:** The consistent performance across the trials suggests that this configuration strikes a good balance between model complexity and the ability to generalize to new data.

# **Make Predictions on the Test Data**

# In[ ]:


# Make predictions on the test data
test_predictions = best_model.predict(test_padded)
test_predictions = (test_predictions > 0.5).astype(int).flatten()


# **Prepare the Submission File**

# In[ ]:


# Prepare the Submission File
submission = sample_submission_df.copy()
submission['target'] = test_predictions


# **Verify and Save the submission file**

# In[ ]:


# Check the first few rows of the submission file
print(submission.head())
# Save the file
submission.to_csv('/kaggle/working/submission_a4.csv', index=False)


# # **Conclusion**
# 
# The project focused on developing a deep learning model for binary text classification, specifically to determine whether a tweet is about a real disaster or not. Using a combination of data preprocessing, model architecture design, and hyperparameter tuning, we achieved a final validation accuracy of 81.55%.
# 
# **Key Takeaways:**
# 
# ***Data Preprocessing:*** Preprocessing steps, including text normalization, tokenization, and sequence padding, were essential for transforming raw text data into a format suitable for input into the neural network. The handling of stopwords and stemming helped reduce noise in the data.
# 
# ***Model Architecture:*** The chosen model architecture, which included an embedding layer followed by a bidirectional LSTM with dropout layers, proved to be effective for this task. The use of bidirectional LSTMs allowed the model to capture context from both directions in the text, improving its ability to understand the semantics of the tweets.
# 
# ***Hyperparameter Tuning:*** Systematic hyperparameter tuning using Random Search was crucial in optimizing the model’s performance. The optimal configuration included:
# 
# 32 LSTM units
# 
# 0.2 dropout rate
# 
# 0.2 LSTM dropout and recurrent dropout rates
# 
# 0.001 learning rate
# 
# This configuration provided the best balance between model complexity and generalization, leading to the highest validation accuracy.
# 
# ***Model Performance:*** The model achieved a validation accuracy of 81.55%, with a validation loss of 0.4368. This indicates that the model generalizes well to unseen data, making it reliable for real-world application in identifying disaster-related tweets.
# 
# ***Challenges:*** Some challenges encountered during the project included initial overfitting, which was addressed by adjusting the dropout rates and using a lower learning rate. Additionally, ensuring that the model did not get stuck during training required careful tuning of the learning rate and other hyperparameters.
# 
# 

# ***Future Work:***
# 
# * Further Tuning: While the model achieved good results, further tuning, such as
# exploring other neural network architectures (e.g., GRU, transformer-based models) or combining multiple models in an ensemble, could potentially yield better performance.
# 
# * Handling Imbalance: Exploring techniques to handle the slight imbalance in the dataset could improve the model’s performance further, especially in correctly identifying disaster-related tweets.
# 
# * Feature Engineering: Incorporating additional features, such as sentiment analysis or keyword presence, could enhance the model’s ability to classify tweets more accurately.
# 
# * Model Deployment: In a real-world scenario, the next step would involve deploying the model in a production environment, potentially integrating it with a real-time tweet monitoring system for disaster response efforts.
# 
# ***Final Thoughts:***
# 
# The project successfully demonstrated the application of deep learning techniques to natural language processing tasks, highlighting the importance of data preprocessing, model selection, and hyperparameter tuning in building effective models. The results achieved are promising and provide a strong foundation for further work and potential deployment in disaster management systems.
