import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras import optimizers


class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr

        # ORIGINAL NETWORK
        self.model = Sequential() # make sequential model

	# input layer
        self.model.add(LSTM(256, input_shape=(1, input_dim),
                            return_sequences=True, stateful=False, dropout=0.5))
	# hidden layer
        self.model.add(BatchNormalization())

        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))

        self.model.add(BatchNormalization())

        self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.5))

        self.model.add(BatchNormalization())

	# output layer
        self.model.add(Dense(output_dim))

        self.model.add(Activation('sigmoid'))

	# Optimizer
        self.model.compile(optimizer=optimizers.sgd(lr=lr), loss='mse')

        self.prob = None

        """
        # DQN Network in SAMPLE CODE
        self.model = Sequential() # make sequential model

	# input layer
        self.model.add(Dense(64, input_dim=input_dim, activation='relu'))


	# hidden layer
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))

	# output layer
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))

	# Optimizer
        self.model.compile(optimizer=Adam(lr=lr), loss='mse')

        self.prob = None
        """

    def reset(self):
        self.prob = None

    def predict(self, sample): # predict() : input to model, make output.
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        return self.prob

    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def save_model(self, model_path): # save model
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path): # load model
        if model_path is not None:
            self.model.load_weights(model_path)
