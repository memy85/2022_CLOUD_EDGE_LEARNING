import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.losses import mean_squared_error
import numpy as np
import math

class BottomNetwork(object):
    '''
    bottom cloud에서의 뉴런
    '''
    def __init__(self, config):
        self.task: str = config['task']
        self.bottom_input_size: int = config['bottom_input_size']
        self.layers: int = config['layers']
        self.hidden_units: int = config['hidden_units']
        self.random_seed: int = config['random_seed']
        self.bottom_network: Model = None # create_bottom_network
        self.bottom_weights_grads: tf.Tensor = None # bottom_split_gradients

    def create_bottom_network(self):
        input_layer = Input(shape=self.bottom_input_size, name="bottom_input")
        dense = input_layer
        for i in range(self.layers):
            dense = Dense(
                units=self.hidden_units,
                kernel_initializer=glorot_uniform(seed=self.random_seed),
                activation='relu',
                name='bottom_dense_{}'.format(i + 1)
            )(dense)
        self.bottom_network = Model(input_layer, dense, name=f"{self.task}_bottom_model")
        return self.bottom_network

    def bottom_split_gradients(self, bottom_input: np.ndarray, h_grad_from_top: tf.Tensor):
        with tf.GradientTape(persistent=True) as tape:
            bottom_input_tf = tf.constant(bottom_input)
            bottom_weights = self.bottom_network.trainable_weights
            tape.watch(bottom_input_tf)
            tape.watch(bottom_weights)
            h = self.bottom_network(bottom_input) # TODO h 두 번 계산됨. self에 넣고. bottom class 계산 --> Top Network에 전달?
        self.bottom_weights_grads = tape.gradient(h, bottom_weights, output_gradients=h_grad_from_top)
        return self.bottom_weights_grads

class TopNetwork(object):
    '''
    top 네트워크는 client 상황
    '''
    def __init__(self, config):
        self.task: str = config['task']
        self.top_input_size: int = config['top_input_size']
        self.bottom_output_h_size: int = config['bottom_output_h_size'] #BottomNetowk.hidden_units
        self.layers: int = config['layers']
        self.hidden_units: int = config['hidden_units']
        self.num_outputs: int = config['num_outputs']
        self.random_seed: int = config['random_seed']
        self.loss: tf.keras.losses = config['loss']
        self.top_network: Model = None
        self.y_hat: tf.Tensor = None
        self.h_grad: tf.Tensor = None
        self.top_weights_grads: tf.Tensor = None

    def create_top_model(self):
        top_input_layer = Input(shape=self.top_input_size, name="top_input_layer")
        h_embed_layer = Input(shape=self.bottom_output_h_size, name="h_input_layer") # receive h from bottom
        top_embed_layer = Dense(units=self.hidden_units,
                                kernel_initializer=glorot_uniform(seed=self.random_seed),
                                activation='relu',
                                name='top_dense_embed')(top_input_layer) # embedding top input for balance with h
        concat_input_layer = Concatenate(axis=1)([top_embed_layer, h_embed_layer])
        # TODO Considering location and method of concat layer
        dense = concat_input_layer
        for i in range(self.layers):
            dense = Dense(
                units=self.hidden_units,
                kernel_initializer=glorot_uniform(seed=self.random_seed),
                activation='relu',
                name='top_dense_{}'.format(i + 1)
            )(dense)

        output_layer = Dense(self.num_outputs,
                             kernel_initializer=glorot_uniform(seed=self.random_seed),
                             activation="linear" if self.task == 'regression' else 'softmax',
                             name="regressor" if self.task == 'regression' else 'classifier')(dense)

        self.top_network = Model([top_input_layer, h_embed_layer], output_layer, name=f"{self.task}_top_model")
        return self.top_network

    def top_split_gradients(self, top_input: np.ndarray, h_from_bottom: tf.Tensor, label_true: np.ndarray):
        with tf.GradientTape(persistent=True) as tape:
            top_weights = self.top_network.trainable_weights
            tape.watch(h_from_bottom)
            self.y_hat = self.top_network([top_input, h_from_bottom])
            empirical_loss = tf.reduce_mean(self.loss(label_true, self.y_hat))
            self.h_grad = tape.gradient(empirical_loss, h_from_bottom)
            self.top_weights_grads = tape.gradient(empirical_loss, top_weights)
        return self.h_grad, self.top_weights_grads

if __name__ == "__main__":
    from pathlib import Path
    import sys
    import matplotlib.pyplot as plt
    # try:
    #     PROJECT_PATH = Path(__file__).parents[1]
    # except NameError:
    #     PROJECT_PATH = Path('.').absolute().parents[1]
    
    PROJECT_PATH = '/home/wonseok/2021_edge_cloud_learning/'
    DATA_PATH = Path(PROJECT_PATH, 'data', 'raw')
    HISTORY_PATH = Path(PROJECT_PATH, 'history')
    FIGURE_PATH = Path(PROJECT_PATH, 'figure')
    sys.path.append(PROJECT_PATH)
    from utils.save_load import *
    from simulation.simulation_base_model import censored_model

    cloud_data = load_pickle(Path(DATA_PATH, 'simulation_cloud_data.pkl')).reshape(-1, 2)[:1000]
    edge_data = load_pickle(Path(DATA_PATH, 'simulation_edge_data.pkl')).reshape(-1, 2)[:1000]
    label_data = load_pickle(Path(DATA_PATH, 'simulation_label_data.pkl')).reshape(-1)[:1000]

    # config
    bottom_config = {}
    bottom_config['task'] = 'regression'
    bottom_config['bottom_input_size'] = 2
    bottom_config['layers'] = 1
    bottom_config['hidden_units'] = 10
    bottom_config['random_seed'] = 42

    top_config = {}
    top_config['task'] = 'regression'
    top_config['top_input_size'] = 2
    top_config['bottom_output_h_size'] = 10
    top_config['layers'] = 2
    top_config['hidden_units'] = 10
    top_config['num_outputs'] = 1
    top_config['random_seed'] = 42
    top_config['loss'] = mean_squared_error
    # cloud
    bottom = BottomNetwork(bottom_config)
    bottom.create_bottom_network()
    # edge
    top = TopNetwork(top_config)
    top.create_top_model()
    # train
    lr = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    batch_size = 1
    loss_metric = tf.keras.metrics.MeanSquaredError()
    rmse_metric = tf.keras.metrics.RootMeanSquaredError()
    loss_per_epoch = []
    rmse_per_epoch = []
    # TODO train 반복문에서 bottom / top dataset 분리
    train_dataset = tf.data.Dataset.from_tensor_slices((cloud_data, edge_data, label_data))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    EPOCHS = 100
    for epoch in range(EPOCHS):
        print("====="*10,f"epoch {epoch+1}: ")
        for step, (cloud_batch, edge_batch, label_batch) in enumerate(train_dataset):
            # TODO h 계산 2번 되는 것 수정
            h = bottom.bottom_network(edge_batch)  # first compute h

            top.top_split_gradients(top_input=edge_batch, h_from_bottom=h, label_true=label_batch)
            bottom.bottom_split_gradients(bottom_input=cloud_batch, h_grad_from_top=top.h_grad) # second compute h

            optimizer.apply_gradients(zip(top.top_weights_grads, top.top_network.trainable_variables))
            optimizer.apply_gradients(zip(bottom.bottom_weights_grads, bottom.bottom_network.trainable_variables))

            loss_metric.update_state(label_batch, top.y_hat)
            rmse_metric.update_state(label_batch, top.y_hat)
        loss = loss_metric.result()
        loss_per_epoch.append(loss)
        rmse = rmse_metric.result()
        rmse_per_epoch.append(rmse)
        print(f"train loss: {loss}, train rmse: {rmse}")

    cloud_history = load_pickle(Path(HISTORY_PATH, 'cloud_history.pkl'))
    edge_history = load_pickle(Path(HISTORY_PATH, 'edge_history.pkl'))
    integrate_history = load_pickle(Path(HISTORY_PATH, 'integrate_history.pkl'))

    plt.plot(cloud_history['loss'], label='cloud_loss(mse)')
    plt.plot(edge_history['loss'], label='edge_loss(mse)')
    plt.plot(integrate_history['loss'], label='edge_loss(mse)')
    plt.plot(loss_per_epoch, label='vertical_loss(mse)')
    plt.legend()
    plt.show()
    plt.savefig(Path(FIGURE_PATH, f'{lr}_vertical_lr_simulation_loss.png'))

    plt.plot(cloud_history['root_mean_squared_error'], label='cloud_rmse')
    plt.plot(edge_history['root_mean_squared_error'], label='edge_rmse')
    plt.plot(integrate_history['root_mean_squared_error'], label='integrate_rmse')
    plt.plot(rmse_per_epoch, label='vertical_rmse')
    plt.legend()
    plt.show()
    plt.savefig(Path(FIGURE_PATH, f'{lr}_vertical_rmse_simulation_loss.png'))