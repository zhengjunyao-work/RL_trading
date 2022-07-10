import random
from collections import deque

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from utils import Portfolio

class Agent(Portfolio):
    def __init__(self,state_dim,balance,is_eval=False,model_name=""):
        super().__init__(balance=balance)
        self.model_type="DQN"
        self.state_dim = state_dim
        self.action_dim = 3
        self.memory=deque(maxlen=100)
        self.buffer_size=60

        self.gamma=0.95
        self.epsilon=1.0
        self.epsilon_min= 0.01
        self.epsilon_decay = 0.995
        self.is_eval = is_eval
        self.model = load_model("save_models/{}.h5".format(model_name)) if is_eval else self.model()
        self.print_f()
        
        
    def print_f(self):
        print("hello world.")
        return
    
    
    def model(self):
        print("test")
        model=Sequential()
        model.add(Dense(units=64,input_dim = self.state_dim,activation="relu"))
        model.add(Dense(units =32,activation="relu"))
        model.add(Dense(units = 8,activation='relu'))
        model.add(Dense(self.action_dim,activation='softmax'))
        model.compile(loss='mse',optimizer=Adam(lr=0.01))
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon=1.0

    def remember(self,state,actions,reward,next_state,done):
        self.memory.append((state,actions,reward,next_state,done))

    def act(self,state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def experience_replay(self):

        mini_batch = [self.memory[i] for i in range(len(self.memory)-self.buffer_size+1,len(self.memory))]

        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                Q_target_value = reward+self.gamma*np.amax(self.model.predict(next_state)[0])
            else:
                Q_target_value = reward

            next_actions = self.model.predict(state)
            next_actions[0][np.argmax(actions)]=Q_target_value
            history = self.model.fit(state,next_actions,epochs=1,verbose =1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]









