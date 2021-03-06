from utils import prepare_data_seq, prepare_data_cnn, AUDIO_DIR
import numpy as np

from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def predict_seq(x_pred, y_pred):
    model = load_model('model.h5')
    counter = 0
    for i in range(2573):
        if y_pred.iat[i, 0] == np.argmax(model.predict(x_pred.iloc[i:i + 1, :])):
            counter += 1
    print(counter / 2573)


def predict_seq_2(x_pred, y_pred):
    model = load_model('model_2.h5')
    counter = 0
    for i in range(2573):
        if y_pred.iat[i, 0] == np.argmax(model.predict(x_pred.iloc[i:i + 1, :])):
            counter += 1
    print(counter / 2573)


def predict_lstm(x_pred, y_pred):
    model = load_model('model_lstm.h5')
    counter = 0
    for i in range(2573):
        if y_pred[i, :] == np.argmax(model.predict(x_pred[i:i + 1, :])):
            counter += 1
    print(counter / 2573)


def predict_gru(x_pred, y_pred):
    model = load_model('model_gru.h5')
    counter = 0
    for i in range(2573):
        if y_pred[i, :] == np.argmax(model.predict(x_pred[i:i + 1, :])):
            counter += 1
    print(counter / 2573)


def predict_cnn(x_pred, y_pred):
    model = load_model('model_cnn.h5')
    counter = 0
    for i in range(2573):
        if y_pred[i, :] == np.argmax(model.predict(x_pred[i:i + 1, :])):
            counter += 1
    print(counter / 2573)


def report(x_pred, y_pred):
    dict_genres = {'Hip-Hop': 0, 'Pop': 1, 'Folk': 2, 'Rock': 3,
                   'Experimental': 4, 'International': 5, 'Electronic': 6, 'Instrumental': 7}

    # dict_genres_med = {'Hip-Hop': 0, 'Pop': 1, 'Folk': 2, 'Rock': 3, 'Experimental': 4,
    #      'International': 5, 'Electronic': 6, 'Instrumental': 7, 'Jazz': 8, 'Spoken': 9,
    #      'Country': 10, 'Blues': 11, 'Old-Time / Historic': 12, 'Soul-RnB': 13,
    #      'Classical': 14, 'Easy Listening': 15}

    # dict_genres_2 = {'Electronic': 0, 'Experimental': 1, 'Folk': 2, 'Hip-Hop': 3,
    #                  'Instrumental': 4, 'International': 5, 'Pop': 6, 'Rock': 7}

    model = load_model('model_gru_small.h5')
    x_test = model.predict(x_pred)
    # x_test = np.expand_dims(x_test, axis=-1)
    y_test = np.argmax(x_test, axis=1)
    y_pred = np.squeeze(y_pred)

    # print(y_pred)
    # print(y_test)
    print(classification_report(y_pred, y_test, target_names=dict_genres.keys()))
    print(accuracy_score(y_pred, y_test))
    mat = confusion_matrix(y_pred, y_test)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=dict_genres.keys(),
                yticklabels=dict_genres.keys())
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()


# x_test_med_seq, y_test_med_seq = prepare_data_seq()
x_test_cnn, y_test_cnn = prepare_data_cnn()
# x_test_spec, y_test_spec = prepare_data_spec()

# report(x_test_med_seq, y_test_med_seq)
report(x_test_cnn, y_test_cnn)
# report(x_test_spec, y_test_spec)

# predict_cnn(x_test_med_cnn, y_test_med_cnn)
# predict_cnn_rnn(x_test_med_cnn, y_test_med_cnn)
# predict_lstm(x_test_med_cnn, y_test_med_cnn)
# predict_gru(x_test_med_cnn, y_test_med_cnn)
# predict_seq(x_test_med_seq, y_test_med_seq)
# predict_seq_2(x_test_med_seq, y_test_med_seq)

# print(x_test_med_seq.shape, y_test_med_seq.shape)
