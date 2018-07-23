# -*- coding: utf-8 -*-
import tensorflow as tf
import random
import gym
import numpy as np
from collections import deque
from game import Game

# from tensorflow.keras import Sequential # import Sequential
# from tf.keras.layers import Dense
# from tf.keras.optimizers import Adam

Sequential = tf.keras.Sequential
Dense = tf.keras.layers.Dense
Adam = tf.keras.optimizers.Adam

EPISODES = 100000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.996
        self.learning_rate = 0.001
        self.observe = 0
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        state_batch = []
        target_f_batch = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            # state_batch.append(state)
            # target_f_batch.append(target_f)
        # state_batch = np.reshape(state_batch, (batch_size, self.state_size))
        # target_f_batch = np.reshape(target_f_batch, (batch_size, self.action_size))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # env = gym.make('CartPole-v1')
    env = Game()
    state_size = env.cur_state().shape[0]
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    reward_sum = 0

    for e in range(EPISODES):
        state = env.cur_state()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.dealEvents()
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # reward = reward if not done else -10
            reward_sum += reward

            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, reward_sum, agent.epsilon))
                break
            if len(agent.memory) > batch_size and e > agent.observe:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")