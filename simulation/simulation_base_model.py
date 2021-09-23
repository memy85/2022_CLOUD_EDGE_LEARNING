import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

class censored_model(object):
    def __init__(self, config):
        self.task = config['task']
        self.random_seed = config['random_seed']

        self.input_size = config['input_size']
        self.bottom_layers = config['bottom_layers']
        self.bottom_hidden_units = config['bottom_hidden_units']

        self.top_input_size = self.create_bottom_model().layers[-1].output.shape[1]
        self.top_layers = config['top_layers']
        self.top_hidden_units = config['top_hidden_units']
        self.num_output = config['num_output']
    
    def create_bottom_model(self):
        input_layer = Input(shape=self.input_size, name="bottom_input")
        dense = input_layer
        for i in range(self.bottom_layers):
            dense = Dense(
                units=self.bottom_hidden_units,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.random_seed),
                activation='relu',
                name='bottom_dense_{}'.format(i + 1)
            )(dense)
        model = Model(input_layer, dense, name="bottom_model")
        return model

    def create_top_model(self):
        input_layer = Input(shape=self.top_input_size, name="h=top_input")
        dense = input_layer
        for i in range(self.top_layers):
            dense = Dense(
                units=self.top_hidden_units,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.random_seed),
                activation='relu',
                name='top_dense_{}'.format(i + 1)
            )(dense)

        output_layer = Dense(self.num_output, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.random_seed),
                             activation="linear" if self.task=='regression' else 'softmax',
                             name="regressor" if self.task=='regression' else 'classifier')(dense)
        model = Model(input_layer, output_layer, name="top_model")
        return model

    def create_full_model(self):
        input_layer = Input(shape=self.input_size, name="bottom_input")
        dense = input_layer
        for i in range(self.bottom_layers):
            dense = Dense(
                units=self.bottom_hidden_units,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.random_seed),
                activation='relu',
                name='bottom_dense_{}'.format(i + 1)
            )(dense)
        for i in range(self.top_layers):
            dense = Dense(
                units=self.top_hidden_units,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.random_seed),
                activation='relu',
                name='top_dense_{}'.format(i + 1)
            )(dense)

        output_layer = Dense(self.num_output, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.random_seed),
                             activation="linear" if self.task=='regression' else 'softmax',
                             name="regressor" if self.task=='regression' else 'classifier')(dense)
        model = Model(input_layer, output_layer, name="top_model")
        return model

class base_model(object):
    def __init__(self, base_config):
        self.task = base_config['task']
        self.base_input_size = base_config['base_input_size']
        self.base_layers = base_config['base_layers']
        self.base_hidden_units = base_config['base_hidden_units']
        self.base_epochs = base_config['base_epochs']
        self.base_lr = base_config['base_lr']
        self.base_random_seed = base_config['base_random_seed']

    def create_regression_model(self):
        """
        create base mlp model
        self.task can be set as 'regression' or 'classification'
        :return:
        """
        input_layer = Input(shape=self.base_input_size, name="base_input")
        dense = input_layer
        for i in range(self.base_layers):
            dense = Dense(
                units=self.base_hidden_units,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.base_random_seed),
                activation='relu',
                name='base_dense_{}'.format(i + 1)
            )(dense)

        output_layer = Dense(1, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.base_random_seed), activation="linear", name="base_regressor")(dense)
        model = Model(input_layer, output_layer, name="base_regression_model")

        adam = tf.keras.optimizers.Adam(
            learning_rate=self.base_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam')
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return model

    def create_classification_model(self):
        """
        create base mlp model
        self.task can be set as 'regression' or 'classification'
        :return:
        """
        input_layer = Input(shape=self.base_input_size, name="base_input")
        dense = input_layer
        for i in range(self.base_layers):
            dense = Dense(
                units=self.base_hidden_units,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.base_random_seed),
                activation='relu',
                name='base_dense_{}'.format(i + 1)
            )(dense)

        output_layer = Dense(1, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.base_random_seed), activation="sigmoid", name="base_classifier")(dense)
        model = Model(input_layer, output_layer, name="base_classification_model")

        adam = tf.keras.optimizers.Adam(
            learning_rate=self.base_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam')
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
        return model

    def build_model(self, model, X_train, y_train, valid_data=None, verbose=1):
        model.fit(x=X_train, y= y_train, epochs=self.base_epochs,
                  validation_data=valid_data)
        return model

if __name__ == "__main__":
    from utils.save_load import *
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    # try:
    #     PROJECT_PATH = Path(__file__).parents[1]
    # except NameError:
    #     PROJECT_PATH = Path('.').absolute().parents[1]
    PROJECT_PATH = '/Users/taehyun/PycharmProjects/vertical_cloud_edge_learning/'
    DATA_PATH = Path(PROJECT_PATH, 'data', 'raw')
    HISTORY_PATH = Path(PROJECT_PATH, 'history')

    cloud_data = load_pickle(Path(DATA_PATH, 'simulation_cloud_data.pkl')).reshape(-1, 2)
    edge_data = load_pickle(Path(DATA_PATH, 'simulation_edge_data.pkl')).reshape(-1, 2)
    integrate_data = np.concatenate([edge_data, cloud_data], axis=1)

    label_data = load_pickle(Path(DATA_PATH, 'simulation_label_data.pkl')).reshape(-1)

    # only edge data
    edge_config = dict()
    edge_config['task'] = 'regression'
    edge_config['base_input_size'] = 2
    edge_config['base_layers'] = 3
    edge_config['base_hidden_units'] = 10
    edge_config['base_epochs'] = 100
    edge_config['base_lr'] = 0.0001
    edge_config['base_random_seed'] = 42

    edge = base_model(edge_config)
    edge_model = edge.create_regression_model()
    edge_model = edge.build_model(model = edge_model, X_train=edge_data, y_train=label_data)

    # only server data
    cloud_config = dict()
    cloud_config['task'] = 'regression'
    cloud_config['base_input_size'] = 2
    cloud_config['base_layers'] = 3
    cloud_config['base_hidden_units'] = 10
    cloud_config['base_epochs'] = 100
    cloud_config['base_lr'] = 0.0001
    cloud_config['base_random_seed'] = 42

    cloud = base_model(cloud_config)
    cloud_model = cloud.create_regression_model()
    cloud_model = cloud.build_model(model=cloud_model, X_train=cloud_data, y_train=label_data)
    
    # integrate data
    integrate_config = dict()
    integrate_config['task'] = 'regression'
    integrate_config['base_input_size'] = 4
    integrate_config['base_layers'] = 3
    integrate_config['base_hidden_units'] = 10
    integrate_config['base_epochs'] = 100
    integrate_config['base_lr'] = 0.0001
    integrate_config['base_random_seed'] = 42

    integrate = base_model(integrate_config)
    integrate_model = integrate.create_regression_model()
    integrate_model = integrate.build_model(model=integrate_model, X_train=integrate_data, y_train=label_data)

    save_pickle(cloud_model.history.history, Path(HISTORY_PATH, 'cloud_history.pkl'))
    save_pickle(edge_model.history.history, Path(HISTORY_PATH, 'edge_history.pkl'))
    save_pickle(integrate_model.history.history, Path(HISTORY_PATH, 'integrate_history.pkl'))

    plt.plot(cloud_model.history.history['loss'], label='cloud_loss(mse)')
    plt.plot(edge_model.history.history['loss'], label='edge_loss(mse)')
    plt.plot(integrate_model.history.history['loss'], label='edge_loss(mse)')
    plt.legend()
    plt.show()

    plt.plot(cloud_model.history.history['root_mean_squared_error'], label='cloud_rmse')
    plt.plot(edge_model.history.history['root_mean_squared_error'], label='edge_rmse')
    plt.plot(integrate_model.history.history['root_mean_squared_error'], label='integrate_rmse')
    plt.legend()
    plt.show()

