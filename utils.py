import ast
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


def shuffle(x_train, y_train):
    perm = tf.random.shuffle(tf.range(tf.shape(x_train)[0]))
    x_train = tf.gather(x_train, perm, axis=0)
    y_train = tf.gather(y_train, perm, axis=0)

    return x_train, y_train


def load_file(filepath):
    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                CategoricalDtype(categories=SUBSETS, ordered=True))
        except ValueError:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def load():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    tracks = load_file('C:/Users/Sebastian/Desktop/Mgr/fma_metadata/fma_metadata/tracks.csv')
    # genres = utils.load('C:/Users/Sebastian/Desktop/Mgr/fma_metadata/fma_metadata/genres.csv')
    features = load_file('C:/Users/Sebastian/Desktop/Mgr/fma_metadata/fma_metadata/features.csv')
    echonest = load_file('C:/Users/Sebastian/Desktop/Mgr/fma_metadata/fma_metadata/echonest.csv')

    np.testing.assert_array_equal(features.index, tracks.index)
    assert echonest.index.isin(tracks.index).all()

    small = tracks['set', 'subset'] <= 'small'
    medium = tracks['set', 'subset'] <= 'medium'

    train = tracks['set', 'split'] == 'training'
    val = tracks['set', 'split'] == 'validation'
    test = tracks['set', 'split'] == 'test'

    y_train = tracks.loc[small & train, ('track', 'genre_top')]
    y_test = tracks.loc[small & test, ('track', 'genre_top')]
    y_train_med = tracks.loc[medium & train, ('track', 'genre_top')]
    y_test_med = tracks.loc[medium & test, ('track', 'genre_top')]

    x_train = features.loc[small & train, ['mfcc', 'chroma_cens', 'spectral_contrast']]
    x_test = features.loc[small & test, ['mfcc', 'chroma_cens', 'spectral_contrast']]
    x_train_med = features.loc[medium & train, ['mfcc', 'chroma_cens', 'spectral_contrast']]
    x_test_med = features.loc[medium & test, ['mfcc', 'chroma_cens', 'spectral_contrast']]

    x_val = features.loc[small & val, ['mfcc', 'chroma_cens', 'spectral_contrast']]
    y_val = tracks.loc[small & val, ('track', 'genre_top')]
    x_val_med = features.loc[medium & val, ['mfcc', 'chroma_cens', 'spectral_contrast']]
    y_val_med = tracks.loc[medium & val, ('track', 'genre_top')]

    # print(x_train.info())
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_val.shape)
    # print(y_val.shape)
    # print(features.head())

    return x_train, y_train, x_test, y_test, x_val, y_val, x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med


def encode(y_train, y_test, y_val, y_train_med, y_test_med, y_val_med):
    cat = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    y_train = y_train.map(
        {'Hip-Hop': cat[0], 'Pop': cat[1], 'Folk': cat[2], 'Rock': cat[3], 'Experimental': cat[4],
         'International': cat[5], 'Electronic': cat[6], 'Instrumental': cat[7]})
    y_test = y_test.map(
        {'Hip-Hop': cat[0], 'Pop': cat[1], 'Folk': cat[2], 'Rock': cat[3], 'Experimental': cat[4],
         'International': cat[5], 'Electronic': cat[6], 'Instrumental': cat[7]})
    y_val = y_val.map(
        {'Hip-Hop': cat[0], 'Pop': cat[1], 'Folk': cat[2], 'Rock': cat[3], 'Experimental': cat[4],
         'International': cat[5], 'Electronic': cat[6], 'Instrumental': cat[7]})

    y_train_med = y_train_med.map(
        {'Hip-Hop': cat[0], 'Pop': cat[1], 'Folk': cat[2], 'Rock': cat[3], 'Experimental': cat[4],
         'International': cat[5], 'Electronic': cat[6], 'Instrumental': cat[7], 'Jazz': cat[8], 'Spoken': cat[9],
         'Country': cat[10], 'Blues': cat[11], 'Old-Time / Historic': cat[12], 'Soul-RnB': cat[13],
         'Classical': cat[14], 'Easy Listening': cat[15]})
    y_test_med = y_test_med.map(
        {'Hip-Hop': cat[0], 'Pop': cat[1], 'Folk': cat[2], 'Rock': cat[3], 'Experimental': cat[4],
         'International': cat[5], 'Electronic': cat[6], 'Instrumental': cat[7], 'Jazz': cat[8], 'Spoken': cat[9],
         'Country': cat[10], 'Blues': cat[11], 'Old-Time / Historic': cat[12], 'Soul-RnB': cat[13],
         'Classical': cat[14], 'Easy Listening': cat[15]})
    y_val_med = y_val_med.map(
        {'Hip-Hop': cat[0], 'Pop': cat[1], 'Folk': cat[2], 'Rock': cat[3], 'Experimental': cat[4],
         'International': cat[5], 'Electronic': cat[6], 'Instrumental': cat[7], 'Jazz': cat[8], 'Spoken': cat[9],
         'Country': cat[10], 'Blues': cat[11], 'Old-Time / Historic': cat[12], 'Soul-RnB': cat[13],
         'Classical': cat[14], 'Easy Listening': cat[15]})

    y_train = y_train.rename(None).to_frame()
    y_test = y_test.rename(None).to_frame()
    y_val = y_val.rename(None).to_frame()
    y_train_med = y_train_med.rename(None).to_frame()
    y_test_med = y_test_med.rename(None).to_frame()
    y_val_med = y_val_med.rename(None).to_frame()

    tf.keras.backend.set_floatx('float64')

    return y_train, y_test, y_val, y_train_med, y_test_med, y_val_med


def cnn_dim(x_train, y_train, x_test, y_test_med, x_val_med, y_val_med):
    x_train = np.expand_dims(x_train, 2)
    y_train = np.expand_dims(y_train, 2)
    x_val_med = np.expand_dims(x_val_med, 2)
    y_val_med = np.expand_dims(y_val_med, 2)
    x_test = np.expand_dims(x_test, 2)
    y_test_med = np.expand_dims(y_test_med, 2)
    return x_train, y_train, x_test, y_test_med, x_val_med, y_val_med

    # print(X_train.info())
    # print(X_train.dtypes)
    # print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
    # print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))


def plot(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()


def prepare_data_seq_train():
    x_train, y_train, x_test, y_test, x_val, y_val, x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med = load()
    y_train, y_test, y_val, y_train_med, y_test_med, y_val_med = encode(y_train, y_test, y_val, y_train_med, y_test_med,
                                                                        y_val_med)

    return x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med


def prepare_data_cnn_train():
    x_train, y_train, x_test, y_test, x_val, y_val, x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med = load()
    y_train, y_test, y_val, y_train_med, y_test_med, y_val_med = encode(y_train, y_test, y_val, y_train_med, y_test_med,
                                                                        y_val_med)
    x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med = cnn_dim(x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med)

    return x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med


def prepare_data_cnn():
    x_train, y_train, x_test, y_test, x_val, y_val, x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med = load()
    y_train, y_test, y_val, y_train_med, y_test_med, y_val_med = encode(y_train, y_test, y_val, y_train_med, y_test_med,
                                                                        y_val_med)
    x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med = cnn_dim(x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med)

    return x_test_med, y_test_med


def prepare_data_seq():
    x_train, y_train, x_test, y_test, x_val, y_val, x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med = load()
    y_train, y_test, y_val, y_train_med, y_test_med, y_val_med = encode(y_train, y_test, y_val, y_train_med, y_test_med,
                                                                        y_val_med)

    return x_test_med, y_test_med
