from Models import NeuralXGBoost
import pickle
from random import shuffle
import logging
import time
import coloredlogs
import numpy as np

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

import xgboost as xgb

def main():
    
    model = NeuralXGBoost()

    log.info('Loading warm start data...')
    data_file = open("data/data_daily_donkey.plk", "rb")
    trainExamplesHistory = pickle.load(data_file)
    data_file.close()
    log.info('Data succesfully loaded')
    trainExamples = []
    for i,examples in enumerate(trainExamplesHistory[:]):
        trainExamples.extend(examples)
    shuffle(trainExamples)
    
    model.train(trainExamples)
    

if __name__ == "__main__":
    main()