from utils import prepare_data_seq, prepare_data_cnn
import numpy as np

from keras.models import load_model


def predict_seq(x_pred, y_pred):
    model = load_model('model.h5')
    counter = 0
    for i in range(2573):
        if y_pred.iat[i, 0] == np.argmax(model.predict(x_pred.iloc[i:i + 1, :])):
            counter += 1
    print(counter/2573)


def predict_seq_2(x_pred, y_pred):
    model = load_model('model_2.h5')
    counter = 0
    for i in range(2573):
        if y_pred.iat[i, 0] == np.argmax(model.predict(x_pred.iloc[i:i + 1, :])):
            counter += 1
    print(counter/2573)


def predict_lstm(x_pred, y_pred):
    model = load_model('model_lstm.h5')
    counter = 0
    for i in range(2573):
        if y_pred[i, :] == np.argmax(model.predict(x_pred[i:i + 1, :])):
            counter += 1
    print(counter/2573)


def predict_cnn(x_pred, y_pred):
    model = load_model('model_cnn.h5')
    counter = 0
    for i in range(2573):
        if y_pred[i, :] == np.argmax(model.predict(x_pred[i:i + 1, :])):
            counter += 1
    print(counter/2573)


def predict_cnn_rnn(x_pred, y_pred):
    model = load_model('model_cnn_rnn.h5')
    counter = 0
    for i in range(2573):
        if y_pred[i, :] == np.argmax(model.predict(x_pred[i:i + 1, :])):
            counter += 1
    print(counter/2573)


x_test_med_seq, y_test_med_seq = prepare_data_seq()
x_test_med_cnn, y_test_med_cnn = prepare_data_cnn()

predict_cnn(x_test_med_cnn, y_test_med_cnn)
predict_cnn_rnn(x_test_med_cnn, y_test_med_cnn)
predict_lstm(x_test_med_cnn, y_test_med_cnn)
predict_seq(x_test_med_seq, y_test_med_seq)
predict_seq_2(x_test_med_seq, y_test_med_seq)

# print(x_test_med_seq.shape, y_test_med_seq.shape)
