from Models import *
from Arena import Arena

from MCTS import MCTS
from Parallel_MCTS import Parallel_MCTS
from Games import Connect4 as Game

args = {
    'numMCTSSims': [50,50],
    'arenaBatch': 1,         
    'numParallelGameArena': 50,
    'cpuct': [1,1]
}

def main():
    player1 = NeuralXGBoost(use_v=False)
    player1.load(xgb_folder='data/xgb', nn_folder='data/daily_donkey', nn_filename='save48.h5')
    #player1 = NeuralNetwork()
    #player1.load_checkpoint(folder=args['checkpoint'], filename='daily_donkey/save48.h5')
    #player1 = "mcts"
    #player1 = None

    player2 = NeuralNetwork()
    player2.load_checkpoint(folder='data', filename='daily_donkey/save48.h5')
    #player2 = "mcts"
    #player2 = None

    arena = Arena(Game,[player1,MCTS(args['cpuct'][1])],[player2,MCTS(args['cpuct'][1])])
    winP1, WinP2, draw = arena.compare(args)
    print('Player 1 : %d, Player 2 : %d, Draw : %d' % (winP1, WinP2, draw))
    
if __name__ == "__main__":
    main()