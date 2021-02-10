import talos

from models import model_seq
from utils import prepare_data_seq_train


def model_seq_tuning():
    p = {'neuron': [8, 16, 32, 64, 128, 256],
         'batch_size': [64, 128],
         'epochs': [10, 15, 20],
         'dropout': [0, 0.1, 0.2, 0.3, 0.4],
         'optimizer': ['Nadam', 'Adam'],
         'activation': ['relu', 'elu']}

    x_train_med_seq, y_train_med_seq, x_test_med_seq, y_test_med_seq, x_val_med_seq, y_val_med_seq = prepare_data_seq_train()
    t = talos.Scan(x=x_train_med_seq, y=y_train_med_seq, params=p, model=model_seq, experiment_name='model_seq')

    return t


model_seq_tuning()
