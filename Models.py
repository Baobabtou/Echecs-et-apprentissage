import argparse
import os
import shutil
import random
import numpy as np
import math
import sys
import time
import argparse
import abc
import joblib

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Reshape, Flatten, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import activations, regularizers
from tensorflow.keras.optimizers import Adam

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import wandb
from wandb.keras import WandbCallback

class CustomModel():
    def getInputTargets(self, examples):
        """
        Preprocessing des données pour la compatibilité avec tensorflow
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards=np.asarray(input_boards)
        pi=[]
        for move in target_pis:
            if type(list(move.keys())[0])==int:
                temp = np.zeros(7)
                for (u,v) in move.items():
                    temp[u]=v
            else:
                temp = np.zeros((9,9))
                for (u,v) in move.items():     
                    temp[int(u[0]),int(u[1])]=v
            pi.append(temp)
        target_pis = np.asarray(pi)
        target_vs = np.asarray(target_vs)
        return input_boards, target_pis, target_vs

    def predictBatch(self, boards):
        """
        Estimation des politques et valeurs à partir d'un dictionnaire de plateaux
        """
        numpyBoards = np.asarray([boards[b].representation for b in boards])
        pi, v = self.prediction(numpyBoards)
        finalpi = dict()
        finalv = dict()
        for e,board in enumerate(boards):
            movespi=dict()
            for m in boards[board].get_legal_moves():
                if type(m)==int:
                    movespi[m] = pi[e,m]
                else:
                    movespi[m] = pi[e,m[0],m[1]]
            total = sum(movespi.values())
            if total<=1E-5:
                print("Shit")
                for move in movespi:
                    movespi[move]=1
            finalpi[board]=movespi
            finalv[board]=v[e][0]
        return finalpi, finalv

    @abc.abstractmethod
    def prediction(self, numpyBoards):
        """
        Estimation de la politique et de la valeur avec le modèle
        """
        return

class NeuralNetwork(CustomModel):

    def residual_block(self, x, filters, kernel_size = 3):
        y = Conv2D(kernel_size=kernel_size,filters=filters,padding="same")(x)
        y = BatchNormalization()(y)
        y = Activation(activations.relu)(y)
        y = Conv2D(kernel_size=kernel_size,filters=filters,padding="same")(y)
        y = BatchNormalization()(y)
        y = Add()([x, y])
        y = Activation(activations.relu)(y)
        return y

    def __init__(self):

        self.config = {
            'learning_rate': 0.001,
            'epochs': 1,
            'batch_size': 256,
            'filters': 256,
            'residualDepth': 10
        }

        self.inputs = Input(shape=(7,6,2))
        #self.inputs = Input(shape=(9,9,3))

        x = Conv2D(filters=self.config['filters'],kernel_size=(3,3),padding="same")(self.inputs)
        #x = Conv2D(filters=self.config['filters'], kernel_size=(3,3), strides=(3,3),padding="valid")(self.inputs)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        for _ in range(self.config['residualDepth']):
            x = self.residual_block(x,self.config['filters'],kernel_size=3)

        valueHead = Conv2D(filters=8, kernel_size=(1,1), padding="same")(x)
        valueHead = BatchNormalization()(valueHead)
        valueHead = Activation(activations.relu)(valueHead)
        valueHead = Flatten()(valueHead)
        valueHeadLastLayer= Dense(256,activation="relu")(valueHead)
        valueHeadLastLayer = Dropout(0.3)(valueHeadLastLayer)
        valueHead = Dense(1,activation="tanh", name="v")(valueHeadLastLayer)

        policyHead = Conv2D(filters=32, kernel_size=(1,1), padding="same")(x)
        policyHead = BatchNormalization()(policyHead)
        policyHead = Activation(activations.relu)(policyHead)
        policyHeadLastLayer = Flatten()(policyHead)
        policyHead = Dropout(0.3)(policyHeadLastLayer)
        policyHead = Dense(7,activation="softmax",name="pi")(policyHead)
        #policyHead = Dense(9*9,activation="softmax", kernel_regularizer=regularizers.l2(1e-7))(policyHead)
        #policyHead = Reshape(target_shape=(9,9),name="pi")(policyHead)


        
        self.model = Model(inputs=self.inputs, outputs=[policyHead, valueHead, policyHeadLastLayer, valueHeadLastLayer])
        self.compile()
        #self.model.summary()

        config = wandb.config

    def compile(self):
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error', None, None], optimizer=Adam(self.config['learning_rate']), loss_weights=[1,1,1,1])

    def start_wandb(self,resume):
        wandb.init(project='alphazero', entity='naaranja',config=self.config, resume=resume)

    def train(self, examples, warm_start=False):
        input_boards, target_pis, target_vs = self.getInputTargets(examples)
        if warm_start:
            self.model.fit(x = input_boards, y = [target_pis, target_vs, np.zeros(target_vs.shape), np.zeros(target_vs.shape)], batch_size = self.config['batch_size'], epochs = 1)
        else:
            self.model.fit(x = input_boards, y = [target_pis, target_vs, np.zeros(target_vs.shape), np.zeros(target_vs.shape)], batch_size = self.config['batch_size'], epochs = self.config['epochs'],callbacks=[WandbCallback()])

    def prediction(self, numpyBoards):
        return self.model.predict(numpyBoards)[:2]

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        self.model.save(filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        self.model = tf.keras.models.load_model(filepath)



class NeuralXGBoost(CustomModel):

    """
    NeuralXGBoost utilise les sorties intermédiaires du réseau de neurones comme entrées pour les XGBoost
    L'utilisation des sortie alternatives obtenues avec XGBoost est optionnelle avec les paramètres use_pi et use_v.
    """

    def __init__(self, use_pi=True, use_v=True):
        self.use_pi = use_pi
        self.use_v = use_v
        self.NN = NeuralNetwork()
        self.NN.load_checkpoint(folder='data', filename='daily_donkey/save48.h5')
        if self.use_pi:
            self.xgb_pi = xgb.XGBClassifier(use_label_encoder=False, colsample_bytree = 0.5, n_estimators = 100)
        if self.use_v:
            self.xgb_v = xgb.XGBRegressor(colsample_bytree = 0.5, n_estimators = 100)
    
    def train(self, examples):
        input_boards, target_pis, target_vs = self.getInputTargets(examples)
        policyHeadLastLayer, valueHeadLastLayer = [],[]
        for i in range(1+len(input_boards)//100000):
            print(f'Step #{i}')
            data = input_boards[i*100000:min((i+1)*100000,len(input_boards)+1)]
            p, v = self.NN.model.predict(data)[2:]
            policyHeadLastLayer+=list(p)
            valueHeadLastLayer+=list(v)
        policyHeadLastLayer = np.array(policyHeadLastLayer)
        valueHeadLastLayer = np.array(valueHeadLastLayer)

        target_pi_class = []
        for i in range(len(target_pis)):
            target_pi_class.append(np.argmax(target_pis[i]))

        if self.use_pi:
            self.xgb_pi.fit(policyHeadLastLayer, target_pi_class, eval_metric='logloss')
            joblib.dump(self.xgb_pi, 'data/xgb_pi')

        if self.use_v:
            self.xgb_v.fit(valueHeadLastLayer, target_vs)
            joblib.dump(self.xgb_v, 'data/xgb_v')
        
    def prediction(self, numpyBoards):
        pi, v, policyHeadLastLayer, valueHeadLastLayer = self.NN.model.predict(numpyBoards)
        if self.use_pi:
            pi = self.xgb_pi.predict_proba(policyHeadLastLayer)
        if self.use_v:
            v = self.xgb_v.predict(valueHeadLastLayer)
            v = [[e] for e in v]
        return pi,v

    def load(self, xgb_folder, nn_folder, nn_filename):
        self.NN.load_checkpoint(folder=nn_folder, filename=nn_filename)
        self.xgb_pi = joblib.load(xgb_folder+'/xgb_pi')
        self.xgb_v = joblib.load(xgb_folder+'/xgb_v')