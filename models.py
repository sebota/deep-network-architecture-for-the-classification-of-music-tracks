import os

from utils import plot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import regularizers


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


def model_seq_2(x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
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
                             input_shape=(273, 1), return_sequences=True),
        tf.keras.layers.Dropout(0.6),
        # tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0.0, unroll=False, use_bias=True),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='softmax')
    ])

    # adam = tf.keras.optimizers.Adam(lr=0.0005)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train_med, y_train_med, epochs=20, validation_data=(x_val_med, y_val_med))

    print(model.evaluate(x_test_med, y_test_med, verbose=2))
    print(model.evaluate(x_val_med, y_val_med, verbose=2))

    plot(history)
    model.save('model_lstm.h5')


def model_gru(x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med):
    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(128, kernel_regularizer=regularizers.l2(0.0001), activation='tanh',
                            recurrent_activation='sigmoid', recurrent_dropout=0.0, unroll=False, use_bias=True,
                            input_shape=(273, 1), return_sequences=True),
        tf.keras.layers.Dropout(0.6),
        # tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0.0, unroll=False, use_bias=True),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='softmax')
    ])

    # adam = tf.keras.optimizers.Adam(lr=0.0005)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train_med, y_train_med, epochs=20, validation_data=(x_val_med, y_val_med))

    print(model.evaluate(x_test_med, y_test_med, verbose=2))
    print(model.evaluate(x_val_med, y_val_med, verbose=2))

    plot(history)
    model.save('model_gru.h5')


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

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train_med, y_train_med, epochs=20, validation_data=(x_val_med, y_val_med))

    print(model.evaluate(x_test_med, y_test_med, verbose=2))
    print(model.evaluate(x_val_med, y_val_med, verbose=2))

    plot(history)
    model.save('model_cnn.h5')


def model_cnn_rnn(x_train_med, y_train_med, x_test_med, y_test_med, x_val_med, y_val_med):
    num_classes = 16
    n_layers = 3
    filter_length = 5
    conv_filter_count = 56
    # BATCH_SIZE = 32
    lstm_count = 96
    # EPOCH_COUNT = 70
    num_hidden = 64
    l2_regularization = 0.001

    input_shape = (273, 1)
    model_input = tf.keras.layers.Input(input_shape, name='input')
    layer = model_input

    for i in range(n_layers):
        layer = tf.keras.layers.Conv1D(
            filters=conv_filter_count,
            kernel_size=filter_length,
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
            name='convolution_' + str(i + 1)
        )(layer)
        layer = tf.keras.layers.BatchNormalization(momentum=0.9)(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.MaxPooling1D(2)(layer)
        layer = tf.keras.layers.Dropout(0.4)(layer)

    layer = tf.keras.layers.LSTM(lstm_count, return_sequences=False)(layer)
    layer = tf.keras.layers.Dropout(0.4)(layer)

    layer = tf.keras.layers.Dense(num_hidden, kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                                  name='dense1')(layer)
    layer = tf.keras.layers.Dropout(0.4)(layer)

    layer = tf.keras.layers.Dense(num_classes)(layer)
    layer = tf.keras.layers.Activation('softmax', name='output_realtime')(layer)
    model_output = layer
    model = tf.keras.models.Model(model_input, model_output)

    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(x_train_med, y_train_med, epochs=20, validation_data=(x_val_med, y_val_med))

    print(model.evaluate(x_test_med, y_test_med, verbose=2))
    print(model.evaluate(x_val_med, y_val_med, verbose=2))

    plot(history)
    model.save('model_cnn_rnn.h5')


def model_cnn_spec(x_train, y_train, x_val, y_val):
    batch_size = 32
    num_classes = 8
    n_features = x_train.shape[2]
    n_time = x_train.shape[1]

    N_LAYERS = 3
    FILTER_LENGTH = 5
    CONV_FILTER_COUNT = 56
    BATCH_SIZE = 32
    LSTM_COUNT = 96
    EPOCH_COUNT = 70
    NUM_HIDDEN = 64
    L2_regularization = 0.001

    # input_shape = (273, 1)
    # model_input = tf.keras.layers.Input(input_shape, name='input')
    # layer = model_input

    n_features = x_train.shape[2]
    input_shape = (None, n_features)
    model_input = tf.keras.layers.Input(input_shape, name='input')
    layer = model_input

    ### 3 1D Convolution Layers
    for i in range(N_LAYERS):
        # give name to the layers
        layer = tf.keras.layers.Conv1D(
            filters=CONV_FILTER_COUNT,
            kernel_size=FILTER_LENGTH,
            kernel_regularizer=regularizers.l2(L2_regularization),  # Tried 0.001
            name='convolution_' + str(i + 1)
        )(layer)
        layer = tf.keras.layers.BatchNormalization(momentum=0.9)(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.MaxPooling1D(2)(layer)
        layer = tf.keras.layers.Dropout(0.4)(layer)

    ## LSTM Layer
    layer = tf.keras.layers.LSTM(LSTM_COUNT, return_sequences=False)(layer)
    layer = tf.keras.layers.Dropout(0.4)(layer)

    ## Dense Layer
    layer = tf.keras.layers.Dense(NUM_HIDDEN, kernel_regularizer=regularizers.l2(L2_regularization), name='dense1')(layer)
    layer = tf.keras.layers.Dropout(0.4)(layer)

    ## Softmax Output
    layer = tf.keras.layers.Dense(num_classes)(layer)
    layer = tf.keras.layers.Activation('softmax', name='output_realtime')(layer)
    model_output = layer
    model = tf.keras.models.Model(model_input, model_output)

    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

    # print(model.evaluate(x_test_med, y_test_med, verbose=2))
    # print(model.evaluate(x_val_med, y_val_med, verbose=2))

    plot(history)
    model.save('model_cnn_spec.h5')

