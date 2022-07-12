import random
#import gym
import numpy as np
from collections import deque
import tensorflow as tf
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras.utils import plot_model
import os  # for creating directories


tf.compat.v1.disable_eager_execution()

LearningRate = 0.001


def softmax(w, t=1.0):
    e = np.exp((w-np.min(w))/np.max(np.abs(w)))
    dist = e / (np.sum(e))
    return dist


class DQNAgent:
    def __init__(self, action_size, state_shape):
        self.action_size = action_size
        self.input_shape = state_shape
        # double-ended queue; acts like list, but elements can be added/removed from either end
        self.memory = deque(maxlen=10000)
        self.gamma = 0.75  # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon = 1
        # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.1  # minimum amount of random exploration permitted
        # rate at which NN adjusts models parameters via SGD to reduce cost
        self.learning_rate = 0.01
        self.model = self._build_model()  # private method

    def _build_model(self):

        kernel = 64
        units = 128
        Initializer = keras.initializers.lecun_uniform(seed=None)
        Xin = keras.layers.Input(shape=self.input_shape)
        y = Conv2D(kernel, kernel_size=3, padding='same',
                   kernel_initializer=Initializer)(Xin)
        y = keras.layers.LeakyReLU(alpha=0.3)(y)
        y = Conv2D(kernel, kernel_size=3, padding='same',
                   kernel_initializer=Initializer)(y)
        #x = keras.layers.add([Xin, y])
        x = keras.layers.LeakyReLU(alpha=0.3)(y)
        x = keras.layers.MaxPool2D((2, 2))(x)
        #x = BatchNormalization(momentum=Mo)(x)

        y = Conv2D(kernel, kernel_size=3,
                   padding='same', kernel_initializer=Initializer)(x)
        y = keras.layers.LeakyReLU(alpha=0.3)(y)
        y = Conv2D(kernel, kernel_size=1, padding='same',
                   kernel_initializer=Initializer)(y)
        x = keras.layers.add([x, y])
        x = keras.layers.LeakyReLU(alpha=0.3)(y)
        x = keras.layers.MaxPool2D((2, 2))(x)
        #x = BatchNormalization(momentum=Mo)(x)

        y = Flatten()(x)
        y = Dense(units, kernel_initializer=Initializer)(y)
        y = keras.layers.LeakyReLU(alpha=0.3)(y)
        #y = keras.layers.Dropout(0.3)(y)
        y = Dense(units, kernel_initializer=Initializer)(y)
        y = keras.layers.LeakyReLU(alpha=0.3)(y)
        #y = keras.layers.Dropout(0.3)(y)
        output = Dense(self.action_size, activation='linear',
                       kernel_initializer=Initializer)(y)
        model = Model(inputs=Xin, outputs=output)
        print(model.summary())
        # plot graph
        #plot_model(model, to_file='ResNet.png')
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(
            lr=LearningRate), metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        # list of previous experiences, enabling re-training later
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # if acting randomly, take random action
            return random.randrange(self.action_size)
        # if not acting randomly, predict reward value based on current state
        act_values = self.model.predict(state,verbose=0)
        # pick the action that will give the highest reward (i.e., go left or right?)
        return np.argmax(act_values[0])
        #A = np.random.choice(self.action_size, p=softmax(act_values[0]))
        # return A

    def actmod(self, state, actlist):
        act_values = self.model.predict(state,verbose=0)[0]
        for index in actlist:
            act_values[index] = -1000*np.abs(np.min(act_values))
        return np.random.choice(self.action_size, p=softmax(act_values))

    def replay(self, batch_size):  # method that trains NN with experiences sampled from memory
        # sample a minibatch from memory
        minibatch = random.sample(self.memory, batch_size)
        # mb_size = batch_size
        # inputs_shape = (mb_size,) + self.input_shape
        # inputs = np.zeros(inputs_shape)
        # targets = np.zeros((mb_size, self.action_size))
        # i = 0
        for state, action, reward, next_state, done in minibatch:  # extract data for each minibatch sample
            # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            # target = reward
            if not done:  # if not done, then predict future discounted reward
                # (maximum target Q based on future action a')
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state,verbose=0)[0]))
            else:
                target = reward
            # approximately map current state to future discounted reward
            # inputs[i:i+1] = state
            # targets[i] = self.model.predict(state)
            # Q_sa = self.model.predict(next_state)
            # targets[i, action] = reward + self.gamma * np.max(Q_sa)
            target_f = self.model.predict(state,verbose=0)
            target_f[0][action] = target
            # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
            self.model.fit(state, target_f, epochs=1, verbose=0)
            # self.model.model.train_on_batch(inputs, targets)
            # i += 1

    def updateEps(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
