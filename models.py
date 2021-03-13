import os

from utils import plot, prepare_data_seq_train, prepare_data_cnn_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import regularizers


# def model_seq(x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med):
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(128, activation='relu'),
#         #tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(64, activation='relu'),
#         #tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(64, activation='relu'),
#         #tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(64, activation='relu'),
#         #tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(64, activation='relu'),
#         #tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(16, activation='softmax')
#     ])
#
#     adam = tf.keras.optimizers.Adam(lr=0.0005)
#     model.compile(optimizer=adam,
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     history = model.fit(x_train_med, y_train_med, epochs=20, batch_size=128, validation_data=(x_val_med, y_val_med))
#     # batch_size = 128
#
#     print(model.evaluate(x_test_med, y_test_med, verbose=2))
#     print(model.evaluate(x_val_med, y_val_med, verbose=2))
#     # print(model.summary())
#
#     plot(history)
#     model.save('model_seq.h5')

def model_seq(x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='softmax')
    ])

    adam = tf.keras.optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train_med, y_train_med, epochs=20, validation_data=(x_val_med, y_val_med))
    # batch_size = 128

    print(model.evaluate(x_test_med, y_test_med, verbose=2))
    print(model.evaluate(x_val_med, y_val_med, verbose=2))
    # print(model.summary())

    plot(history)
    model.save('model_seq.h5')


# def model_seq(x_train_med, y_train_med, x_val_med, y_val_med, params):
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(params['neuron_first'], activation='relu'),
#         tf.keras.layers.Dropout(params['dropout']),
#         tf.keras.layers.Dense(params['neuron_second'], activation='relu'),
#         tf.keras.layers.Dropout(params['dropout']),
#         tf.keras.layers.Dense(params['neuron_second'], activation='relu'),
#         tf.keras.layers.Dropout(params['dropout']),
#         tf.keras.layers.Dense(params['neuron_second'], activation='relu'),
#         tf.keras.layers.Dropout(params['dropout']),
#         tf.keras.layers.Dense(params['neuron_second'], activation='relu'),
#         tf.keras.layers.Dropout(params['dropout']),
#         tf.keras.layers.Dense(16, activation='softmax')
#     ])
#
#     adam = tf.keras.optimizers.Adam(lr=0.0005)
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     history = model.fit(x_train_med, y_train_med, epochs=20, batch_size=params['batch_size'],
#                         validation_data=(x_val_med, y_val_med))
#     # history = model.fit(x_train_med, y_train_med, epochs=20, validation_data=(x_val_med, y_val_med))
#     # batch_size = 128
#
#     # print(model.evaluate(x_test_med, y_test_med, verbose=2))
#     # print(model.evaluate(x_val_med, y_val_med, verbose=2))
#     # print(model.summary())
#
#     # plot(history)
#     # model.save('model_seq.h5')
#
#     return history, model


# x_train_med_seq, y_train_med_seq, x_test_med_seq, y_test_med_seq, x_val_med_seq, y_val_med_seq = prepare_data_seq_train()
# talos.Scan(x=x_train_med_seq, y=y_train_med_seq, params=p, x_val=x_val_med_seq, y_val=y_val_med_seq, model=model_seq, experiment_name='model_test')


def model_seq_2(x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train_med, y_train_med, epochs=20, validation_data=(x_val_med, y_val_med))

    print(model.evaluate(x_test_med, y_test_med, verbose=2))
    print(model.evaluate(x_val_med, y_val_med, verbose=2))
    # print(model.summary())

    plot(history)
    model.save('model_seq_2.h5')


def model_lstm(x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, kernel_regularizer=regularizers.l2(0.0001), activation='tanh',
                             recurrent_activation='sigmoid', recurrent_dropout=0.0, unroll=False, use_bias=True,
                             input_shape=(140, 1), return_sequences=True),
        tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0.0, unroll=False, use_bias=True),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='softmax')
    ])

    # adam = tf.keras.optimizers.Adam(lr=0.0005)
    # adam = tf.keras.optimizers.Adam(lr=0.0005)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train_med, y_train_med, epochs=20, validation_data=(x_val_med, y_val_med))

    print(model.evaluate(x_test_med, y_test_med, verbose=2))
    print(model.evaluate(x_val_med, y_val_med, verbose=2))

    plot(history)
    model.save('model_lstm_small.h5')


# 140 mfcc, 273, 315, 518 all
def model_gru(x_train, y_train, x_test, y_test, x_val, y_val):
    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(32, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0.0, unroll=False,
                            use_bias=True,
                            input_shape=(518, 1), return_sequences=True),
        tf.keras.layers.BatchNormalization(momentum=0.0),
        # tf.keras.layers.Activation(activation='tanh'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.GRU(16, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0.0, unroll=False,
                            use_bias=True, return_sequences=True),
        # tf.keras.layers.BatchNormalization(momentum=0.0),
        # tf.keras.layers.Activation(activation='tanh'),
        tf.keras.layers.Dropout(0.3),

        # tf.keras.layers.BatchNormalization(momentum=0.0),

        # tf.keras.layers.Dense(32, activation='relu'),
        # tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(momentum=0.0),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.Dropout(0.3),

        # tf.keras.layers.Dense(32, activation='relu'),
        # tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(momentum=0.0),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.Dropout(0.3),

        # tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
        # tf.keras.layers.Dense(16),
        # tf.keras.layers.BatchNormalization(momentum=0.0),
        # tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=100, batch_size=512, validation_data=(x_val, y_val))

    print(model.evaluate(x_test, y_test, verbose=2))
    print(model.evaluate(x_val, y_val, verbose=2))

    plot(history)
    model.save('model_gru_small.h5')

    
def model_cnn(x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(273, 1)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        # tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        # tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='softmax'),
    ])

    # adam = tf.keras.optimizers.Adam(lr=0.0005)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train_med, y_train_med, epochs=20, validation_data=(x_val_med, y_val_med))

    print(model.evaluate(x_test_med, y_test_med, verbose=2))
    print(model.evaluate(x_val_med, y_val_med, verbose=2))

    plot(history)
    model.save('model_cnn.h5')
