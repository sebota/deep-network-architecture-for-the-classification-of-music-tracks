import models
from utils import prepare_data_seq_train, shuffle, prepare_data_cnn_train

# x_train_med_seq, y_train_med_seq, x_test_med_seq, y_test_med_seq, x_val_med_seq, y_val_med_seq = prepare_data_seq_train()
x_train_cnn, y_train_cnn, x_test_cnn, y_test_cnn, x_val_cnn, y_val_cnn = prepare_data_cnn_train()
# x_train_med_seq, y_train_med_seq = shuffle(x_train_med_seq, y_train_med_seq)
# x_train_med_cnn, y_train_med_cnn = shuffle(x_train_med_cnn, y_train_med_cnn) # ?

models.model_gru(x_train_cnn, y_train_cnn, x_test_cnn, y_test_cnn, x_val_cnn, y_val_cnn)
# models.model_lstm(x_train_med_cnn, y_train_med_cnn, x_test_med_cnn, y_test_med_cnn, x_val_med_cnn, y_val_med_cnn)
# models.model_cnn(x_train_med_cnn, y_train_med_cnn, x_test_med_cnn, y_test_med_cnn, x_val_med_cnn, y_val_med_cnn)
# models.model_seq(x_train_med_seq, y_train_med_seq, x_test_med_seq, y_test_med_seq, x_val_med_seq, y_val_med_seq)
# models.model_seq_2(x_train_med_seq, y_train_med_seq, x_test_med_seq, y_test_med_seq, x_val_med_seq, y_val_med_seq)
# models.model_cnn_spec(x_train, y_train, x_valid, y_valid)
# x_train, y_train, x_valid, y_valid = prepare_data_spec_train()
