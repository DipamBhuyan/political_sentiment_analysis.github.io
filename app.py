from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

data1 = pd.read_csv('bjp_tweets.csv')
data2 = pd.read_csv('congress_tweets.csv')
data = pd.concat([data1,data2])
# Preprocess the text by converting it to lowercase and removing unnecessary characters
data['text'] = data['tweet'].str.lower()
data['text'] = data['tweet'].str.replace('[^\w\s]', '')
count_target = data['target']
data.drop('tweet', inplace=True, axis=1)
data.head()

app = Flask(__name__)
model = load_model('pt7.h5')

train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

max_len = 100  # Maximum length of a sequence
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data['text'])  # Dummy text to initialize tokenizer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    statement = request.form['statement']
    preprocessed_statement = statement.lower().replace('[^\w\s]', '')
    sequence = tokenizer.texts_to_sequences([preprocessed_statement])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    sentiment = prediction
    float_sentiment = float(sentiment[0][0])
    if float_sentiment > 0.5:
        sentiment = 'Positive'
    elif float_sentiment < 0.5:
        sentiment = 'Negative'
    else:
        sentiment = 'Cannot determine'


    #sentiment = 'Negative' if prediction > 0.5 else 'Negative'
    return render_template('result.html', statement=statement, prediction=float_sentiment, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True,port=8000)
