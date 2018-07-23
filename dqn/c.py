# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from tensorflow.tools.api.generator.api.keras.models import Sequential
from tensorflow.tools.api.generator.api.keras.layers import Dense
from tensorflow.tools.api.generator.api.keras.optimizers import Adam
from tensorflow.tools.api.generator.api.keras import backend as K
import tensorflow as tf

from game import Game
import config
import cv2

EPISODES = 10000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def getInput(image):
    # 转换为灰度值
    image = cv2.cvtColor(cv2.resize(image, (100, 80)), cv2.COLOR_BGR2GRAY)
    # 转换为二值
    ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    input_image_data = np.stack((image, image, image, image), axis=2)

    return input_image_data


if __name__ == "__main__":
    # env = gym.make('CartPole-v1')
    game = Game()
    state_size = game.cur_state().shape[0]
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    n = 0
    total_reward = 0

    while 1:
        state = game.cur_state()
        state = np.reshape(state, [1, state_size])
        n = n + 1
        # 从上到下图中大概128格，所以这里先设置成256
        for time in range(256):
            game.render()
            action = agent.act(state)
            next_state, reward, done = game.step(game.toAction(action))
            game.dealEvent()
            total_reward = (total_reward + reward) if time > 0 else total_reward
            next_state = np.reshape((next_state), [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}, {}"
                      .format(n, EPISODES, time, agent.epsilon, total_reward))
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # print("total_reward: ", total_reward, " n=", n)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")
