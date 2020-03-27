#Importing Necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

#Importing the dataset
df = pd.read_csv('spam.csv')

#Mapping Spam messages as '1' and Ham messages as '0'
df['target'] = df['label'].map( {'spam':1, 'ham':0 })

#Splitting the data for training and testing set
X_train,X_test,y_train,y_test = train_test_split(df['sms'].values,df['target'].values, test_size=0.2, random_state = 19)

#Tokenization of words in the messages
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_dict = tokenizer.index_word

#Generating the sequences upon the tokenized words
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

#Applying padding to the generated sequence of words
X_train_pad = pad_sequences(X_train_seq, maxlen=20, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=20, padding='post')

pad_length = 20

#Using the Sequential class from keras module to apply Long short-term memory algorithm on the dataset
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=len(word_dict)+1, output_dim=20, input_length=pad_length))
lstm_model.add(LSTM(400))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fitting the data into model
lstm_model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_data=(X_test_pad, y_test))

#Make prediction upon the input provided
msg = str(input("Enter the text message:"))
sms_seq = tokenizer.texts_to_sequences([msg])
sms_pad = pad_sequences(sms_seq, maxlen=20, padding='post')
tokenizer.index_word
prediction = lstm_model.predict_classes(sms_pad)
if prediction == [[0]]:
    print(" '{}' is a Ham".format(msg))
else:
    print(" '{}' is a Spam".format(msg))

