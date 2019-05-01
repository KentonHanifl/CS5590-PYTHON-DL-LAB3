# PART 4,5,6

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
import re
from sklearn.preprocessing import LabelEncoder

data = pd.read_table('train.tsv')
# Keeping only the neccessary columns
data = data[:15000]
data = data[['Phrase', 'Sentiment']]

data['Phrase'] = data['Phrase'].apply(lambda x: x.lower())
data['Phrase'] = data['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

max_fatures = 10000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['Phrase'].values)
X = tokenizer.texts_to_sequences(data['Phrase'].values)
X = pad_sequences(X)
embed_dim = 128
lstm_out = 196


def create_lstm_model():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_cnn_model():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(Conv1D(32, 1, input_shape=(1,), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_cnn_model_tuned():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(Conv1D(32, 1, input_shape=(1,), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv1D(32, 1, activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Conv1D(64, 1, activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 1, activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Conv1D(128, 1, activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 1, activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='relu', kernel_constraint=maxnorm(3)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['Sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

lstmmodel = create_lstm_model()
lstmmodel.fit(X_train, Y_train, batch_size=20, epochs=2)
lstmscores = lstmmodel.evaluate(X_test, Y_test, verbose=0)

cnnmodel = create_cnn_model()
cnnmodel.fit(X_train, Y_train, batch_size=20, epochs=2)
cnnscores = cnnmodel.evaluate(X_test, Y_test, verbose=0)

cnnmodel_tuned = create_cnn_model_tuned()
cnnmodel_tuned.fit(X_train, Y_train, batch_size=20, epochs=2)
cnnscores_tuned = cnnmodel_tuned.evaluate(X_test, Y_test, verbose=0)

print('Test accuracy lstm:', lstmscores[1], '\n')
print('Test accuracy cnn:', cnnscores[1], '\n')
print('Test accuracy cnn tuned:', cnnscores_tuned[1], '\n')

