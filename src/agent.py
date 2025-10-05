from . import nn
from collections import deque
import numpy as np
import random


MINI_BATCH_SIZE = 64
MEMORY_SIZE = 256


class Agent:
    def __init__(self):
        self.model = nn.create_model()
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.action_size = 2
        self.memory = deque(maxlen=MEMORY_SIZE)
        nn.compile_model(self.model)

    def take_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = nn.predict(self.model, state) # expected rewards on all actions
        # print(f"q_values.shape: {q_values.shape}")
        return np.argmax(q_values[0])

    def store_transition(self, state, action, reward, next_state, end_episode):
        self.memory.append((state, action, reward, next_state, end_episode))

    def replay(self):
        if len(self.memory) < MINI_BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, MINI_BATCH_SIZE)
        for (state, action, reward, next_state, end_episode) in minibatch:
            expected_reward = reward
            if not end_episode:
                q_values = nn.predict(self.model, next_state)
                expected_reward = reward + self.gamma * np.amax(q_values[0])
            y = nn.predict(self.model, state)
            y[0][action] = expected_reward
            nn.train(self.model, state, y)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay