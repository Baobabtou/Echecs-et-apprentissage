import copy
from tqdm import tqdm
import numpy as np
from random import shuffle
from MCTS import MCTS
from Arena import Arena
from Models import NeuralNetwork
import logging
import time
import coloredlogs
import pickle
import wandb

from Games import TTT as Game

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

args = {
    'numIters': 200,
    'numEps': 256,
    'numParallelGame': 256,
    'numMCTSSims': 20,
    'arenaBatch': 1,
    'numParallelGameArena':50,
    'cpuct': 1,

    'checkpoint': './data/history',
    'resume_wandb': False,
    'resume_model_and_data': False,
    'warm_start': False,
    'resume_iteration': 0,
    'model_file_name':'best.h5',
}

def main():
    train()

def generate_data(NN):
    #Initialisation des variables
    iterationTrainExamples=[]
    pbar = tqdm(total=args['numEps'], desc="Self Play")
    boards=[]
    mcts=[]
    trainExamples=[]
    ended=[0]*args['numParallelGame']
    currentPlayer = [0]*args['numParallelGame']
    episodeStep = [0]*args['numParallelGame']
    nbrGameStarted = args['numParallelGame']
    for i in range(args['numParallelGame']):
        boards.append(Game())
        mcts.append([MCTS(args['cpuct']),MCTS(args['cpuct'])])
        trainExamples.append([])      
    
    while not min(ended):   #Boucle tant qu'un des processus n'est pas terminé

        for i in range(args['numParallelGame']):
            episodeStep[i]+=1

        for _ in range(args['numMCTSSims']): #Un tour de boucle par simulation
            #On initialise un dictionnaire auquel on va ajouter tous les etats de jeu à predict avec le réseau de neurone
            boardsToPredict=dict()
            for i in [k for k in range(args['numParallelGame']) if not ended[k]]:
                #On fait une descente dans chaque mcts pour trouvé un état de jeu à predict
                boardToPredict = mcts[i][currentPlayer[i]].selection(boards[i])
                if boardToPredict!=None:
                    boardsToPredict[i]=boardToPredict
            if len(boardsToPredict)>0:
                #On estime pi et v pour tous les jeux puis on met à jour les mcts
                pi, v = NN.predictBatch(boardsToPredict) 
                for i in boardsToPredict:
                    mcts[i][currentPlayer[i]].backpropagation(pi[i],v[i])

        for i in [k for k in range(args['numParallelGame']) if not ended[k]]:
            if False and i%2==0 and episodeStep[i]<6:
                boards[i].playRandomMove()
            else:
                pi = mcts[i][currentPlayer[i]].getActionProb(boards[i],temp=(1 if episodeStep[i]<20 else 0)) #probabilité d'action
                #On sauvegarde l'état du jeu et ses symétries
                for sym,pis in boards[i].get_symmetries(pi):
                    trainExamples[i].append((sym,pis,currentPlayer[i]))
                #trainExamples[i].append((boards[i].representation,pi,currentPlayer[i]))

                #Selection d'un coup à jouer
                d = {i:e for i,e in enumerate(list(pi.keys()))}
                move = np.random.choice(list(d.keys()), p=list(pi.values()))
                move = d[move]

                #On joue le coup
                boards[i].push(move)

            if boards[i].is_game_over():      #Si la partie est terminée, ajoute des données aux données d'entraînement      
                r=boards[i].result()
                iterationTrainExamples+=[(x[0], x[1], np.round(r) * ((-1) ** (x[2]!=currentPlayer[i]))) for (k,x) in enumerate(trainExamples[i])]

                if nbrGameStarted<args['numEps']:       #Si il y a d'autres partie à jouer, on réinitialise les variables
                    nbrGameStarted+=1
                    currentPlayer[i] = 0
                    episodeStep[i] = 0
                    boards[i]=Game()
                    mcts[i]=[MCTS(args['cpuct']),MCTS(args['cpuct'])]
                    trainExamples[i]=[]
                else:
                    ended[i]=1
                pbar.update(1)
            else:      #Sinon on inverse le plateau et on change de joueur
                boards[i] = boards[i].mirror()
                currentPlayer[i] = (currentPlayer[i]+1)%2
    pbar.close()
    return iterationTrainExamples

def train():
    log.info('START OF TRAINING IN 5 SECONDS...')
    time.sleep(5)

    log.info('Initialization')
    #Initialisation du réseau de neurones et chargement des poids
    NN = NeuralNetwork()
    if args['resume_model_and_data']:
        log.info('Loading model...')
        NN.load_checkpoint(folder=args['checkpoint'], filename=args['model_file_name'])
        NN.compile()
        log.info('Model succesfully loaded')
    NN.save_checkpoint(folder=args['checkpoint'], filename='best.h5')
    NN.save_checkpoint(folder=args['checkpoint'], filename='save0.h5')
    
    #Récupération des données
    if args['warm_start']:
        log.info('Loading warm start data...')
        data_file = open("data/warm_start_data.plk", "rb")
        trainExamplesHistory = pickle.load(data_file)
        data_file.close()
        log.info('Data succesfully loaded')
        trainExamples = []
        for e in trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)
        log.info('Starting warm start')
        NN.train(trainExamples, warm_start=True)
    else:
        if args['resume_model_and_data']:
            log.info('Loading data...')
            data_file = open("data/data.plk", "rb")
            trainExamplesHistory = pickle.load(data_file)
            data_file.close()
            log.info('Data succesfully loaded')
        else:
            trainExamplesHistory=[]

    #Activation du tracker
    NN.start_wandb(args['resume_wandb'])
    
    #Début de l'entrainement
    if args['resume_model_and_data']:
        start_iter = args['resume_iteration']
    else:
        start_iter=0
    for iteration in range(start_iter,args['numIters']):
        log.info(f'Iteration #{iteration}')

        iterationTrainExamples = generate_data(NN)  #Chaque iteration fait jouer 'numParallelGame' parties en parallèle

        #Limitation du nombre de données d'entrainement
        if args['warm_start']:
            limit = 20
        else:
            limit = max(5,min(20,3+iteration//2))
        trainExamplesHistory.append(iterationTrainExamples)
        while len(trainExamplesHistory)>limit:
            trainExamplesHistory.pop(0)

        #Sauvegarde des données d'entrainement
        data_file = open("data/data.plk", "wb")
        pickle.dump(trainExamplesHistory,data_file)
        data_file.close()
        
        #Création du dataset de train
        trainExamples = []
        for e in trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)

        NN.save_checkpoint(folder=args['checkpoint'], filename='temp.h5')  
        lastNN = NeuralNetwork()
        lastNN.load_checkpoint(folder=args['checkpoint'], filename='temp.h5')
        NN.train(trainExamples)

        arena = Arena(Game,[NN,MCTS(args['cpuct'])],["mcts",MCTS(args['cpuct'])])
        mctswinsNew, mctswinsLast, mctsdraw = arena.compare(args)
        log.info('New wins : %d ; Mcts wins %d ; Draws : %d' % (mctswinsNew, mctswinsLast, mctsdraw))

        arena = Arena(Game,[NN,MCTS(args['cpuct'])],[lastNN,MCTS(args['cpuct'])])
        winsNew, winsLast, draw = arena.compare(args)
        log.info('New wins : %d ; Previous wins %d ; Draws : %d' % (winsNew, winsLast, draw))

        wandb.log({
            "window_size": len(trainExamplesHistory),
            "wins_against_mcts": mctswinsNew,
            "losses_against_mcts": mctswinsLast,
            "draws_against_mcts": mctsdraw,
            "wins_against_self": winsNew,
            "losses_against_self": winsLast,
            "draws_against_self": draw
        },commit=False)

        if winsNew>winsLast:
            log.info('ACCEPTING NEW MODEL')
            NN.save_checkpoint(folder=args['checkpoint'], filename='best.h5')
            NN.save_checkpoint(folder=args['checkpoint'], filename=f'save{iteration+1}.h5')
        else:
            log.info('REJECTING NEW MODEL')
            NN.load_checkpoint(folder=args['checkpoint'], filename='best.h5')

if __name__ == "__main__":
    main()