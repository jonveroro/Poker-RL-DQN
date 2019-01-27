import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import model_from_json
from keras import backend as K
#from keras.models import load_weights



class brain:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.955
        self.learning_rate = 0.001
        #self.model = self._build_model()
        self.model = self.load()
        self.tau = .125
        #self.target_model = self._build_model()
        self.target_model = self.loadTargetModel()
        #self.target_model = self.load('Brain1')
       	#self.model = self.load('Models/Brain1.h5')

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model   = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size))
        model.compile(loss=self._huber_loss,
            optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def replay(self, batch_size):

        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)


    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)



    def load(self):
        
        json_file = open('Models/Brain1.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        #load woeights into new model
        loaded_model.load_weights("Models/Brain1.h5")
        print("Loaded Model from disk")

        #compile and evaluate loaded model
        loaded_model.compile(loss=self._huber_loss,
            optimizer=Adam(lr=self.learning_rate))


        return loaded_model

    def save(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("Models/Brain1.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("Models/Brain1.h5")
        self.saveTargetModel()
        print("Saved model to disk")

    def saveTargetModel(self):
        model_json = self.target_model.to_json()
        with open("Models/TargetModel1.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("Models/TargetModel1.h5")
        print("Saved model to disk")


    def loadTargetModel(self):
        json_file = open('Models/TargetModel1.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        #load woeights into new model
        loaded_model.load_weights("Models/TargetModel1.h5")
        print("Loaded Model from disk")

        #compile and evaluate loaded model
        loaded_model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return loaded_model

    def loadModel(self):

        self.model = self.load()
        self.target_model = self.loadTargetModel()

