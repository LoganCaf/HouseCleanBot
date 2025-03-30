import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQAgent:
    def __init__(self,inputShape,outputShape):
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.epsilon = 1
        self.epsilonDecay = .99
        self.learningRate = 0.00025
        self.epsilonMin = 0.1
        self.gamma = 0.95
        self.memory = deque(10240)
        self.minMemorySize = 256
        self.sampleSize = 64
        self.actionModel = self.buildModel()
        self.targetModel = self.buildModel()
        self.updateTime = 0
        self.updateTargetModel()
    
    def updateTargetModel(self):
        self.updateTime -=1
        if self.updateTime > 0:
            return
        self.targetModel.set_weights(self.actionModel.get_weights())
        self.updateTime = 10000

    def reset(self):
        pass

    def buildModel(self):
        model = Sequential()
        model.add(Conv2D(64, (8, 8), strides=(4, 4), activation='relu', input_shape=self.inputShape))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.outputShape, activation='linear'))
        model.compile(optimizer=Adam(lr=0.00025), loss='mse')
        return model

    # call to take an action in the environment
    def act(self,state):
        self.lastState = state
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.outputShape)
        act_values = self.actionModel.predict(self.lastState)
        self.lastAction = np.argmax(act_values[0])
        self.epsilon *= self.epsilonDecay
        if self.epsilon < self.epsilonMin:
            self.epsilon = self.epsilonMin
        return self.lastAction
    
    # call to remember the last action taken and the reward received
    # this is called after the action has been taken and a new state has been received
    def remember(self,state,reward,done):
        self.memory.append((self.lastState,self.lastAction,reward,state,done))
        self.train()
    
    def train(self):
        if len(self.memory) < self.minMemorySize:
            return
        batch = np.array(np.random.sample(self.memory, self.sampleSize))
        currTargets = self.targetModel.predict(batch[:,0])
        nextTargets = self.targetModel.predict(batch[:,3])

        for i, state, action, reward, nextState, done in enumerate(batch):
            if done:
                currTargets[i][0][action] = reward
            else:
                currTargets[i][0][action] = reward + self.gamma * np.amax(nextTargets[i][0])
        
        self.actionModel.fit(state, currTargets, epochs=1, verbose=0)
        self.updateTargetModel()

