import logging

import numpy as np
from tqdm import tqdm
from MCTS import MCTS
from Parallel_MCTS import Parallel_MCTS

log = logging.getLogger(__name__)


class Arena():
    def __init__(self, Game, player1, player2):
        self.GameClass = Game
        self.player1 = player1
        self.player2 = player2

    def start(self):
        players = [self.player1,self.player2]
        pbar = tqdm(total=self.args['arenaBatch']*self.args['numParallelGameArena'], desc="Arena")
        results=[]
        for _ in range(self.args['arenaBatch']):
            boards=[]
            mcts=[]
            ended=[0]*self.args['numParallelGameArena']
            currentPlayer = 0
            episodeStep = 0
            for i in range(self.args['numParallelGameArena']):
                boards.append(self.GameClass())
                mcts.append([self.player1[1].copy(),self.player2[1].copy()])
            while not min(ended):
                episodeStep+=1
                if players[currentPlayer][0]!=None: 
                    for _ in range(self.args['numMCTSSims'][currentPlayer]):

                        #MCTS player
                        if players[currentPlayer][0]=="mcts":
                            for i in [k for k in range(self.args['numParallelGameArena']) if not ended[k]]:
                                mcts[i][currentPlayer].search(boards[i])

                        #Neural Network player
                        else:
                            #Simple MCTS
                            if isinstance(players[currentPlayer][1],MCTS):
                                boardsToPredict=dict()
                                for i in [k for k in range(self.args['numParallelGameArena']) if not ended[k]]:
                                    boardToPredict = mcts[i][currentPlayer].selection(boards[i])
                                    if boardToPredict!=None:
                                        boardsToPredict[i]=boardToPredict
                                if len(boardsToPredict)>0:
                                    pi, v = players[currentPlayer][0].predictBatch(boardsToPredict) 
                                    for i in boardsToPredict:
                                        mcts[i][currentPlayer].backpropagation(pi[i],v[i])
                            #Parellel MCTS
                            else:
                                for i in [k for k in range(self.args['numParallelGameArena']) if not ended[k]]:
                                    boardsToPredict = mcts[i][currentPlayer].selection(boards[i])
                                    if len(boardsToPredict)>0:
                                        pi, v = players[currentPlayer][0].predictBatch(boardsToPredict)
                                        mcts[i][currentPlayer].parallel_backpropagation(pi,v)

                for i in [k for k in range(self.args['numParallelGameArena']) if not ended[k]]:
                    if players[currentPlayer][0]!=None: 
                        pi = mcts[i][currentPlayer].getActionProb(boards[i],temp=(2 if episodeStep<10 else 0))
                        d = {i:e for i,e in enumerate(list(pi.keys()))}
                        move = np.random.choice(list(d.keys()), p=list(pi.values()))
                        move = d[move]
                        boards[i].push(move)
                    else:
                        boards[i].playRandomMove()
                    r=boards[i].result()
                    if r != 0:
                        results.append(r*(-1)**currentPlayer)
                        ended[i]=1
                        pbar.update(1)
                    else:
                        boards[i] = boards[i].mirror()
                currentPlayer = (currentPlayer+1)%2
        pbar.close()
        return results

    def compare(self, args, verbose=False):
        self.args = args.copy()
        if type(self.args['numMCTSSims'])==int:
            self.args['numMCTSSims']=[self.args['numMCTSSims'],self.args['numMCTSSims']]
        if type(self.args['cpuct'])==int:
            self.args['cpuct']=[self.args['cpuct'],self.args['cpuct']]
        winNew=0
        winLast=0
        draw=0

        results = self.start()
        for r in results:
            if r==1:
                winNew+=1
            elif r==-1:
                winLast+=1
            else:
                draw+=1
     
        self.player1, self.player2 = self.player2, self.player1
        self.args['numMCTSSims'] = self.args['numMCTSSims'][::-1]
        results = self.start()
        for r in results:
            if r==1:
                winLast+=1
            elif r==-1:
                winNew+=1
            else:
                draw+=1

        return winNew, winLast, draw