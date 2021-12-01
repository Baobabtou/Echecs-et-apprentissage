import logging
import math
import time
import copy

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class Parallel_MCTS():
    def __init__(self, cpuct, parallel_search):
        self.cpuct = cpuct
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}    
        self.Ps = {}    #Politique
        self.Es = {}    #Valeur des états finaux
        self.Ls = {}    #Compteur de feuilles filles non atteintes par d'autres descentes

        self.parallel_search = parallel_search  #Nombre de descentes
        self.SAhistory=[]

    def copy(self):
        return copy.deepcopy(self)

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Renvoie les probabilités des coups à jouer en fonction de la température
        Pour une température de 0, renvoie le meilleur coup avec une probabilité de 1
        """
        s = canonicalBoard.fen()
        counts = {a:self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in self.Ps[s]}
        if temp == 0:
            bestN = max(counts.values())
            bestAs = {key:value for (key,value) in counts.items() if value==bestN}
            d = {i:e for i,e in enumerate(list(bestAs.keys()))}
            bestA = np.random.choice(list(d.keys()))
            bestA = d[bestA]
            probs = {bestA:1}
            return probs

        for move in counts:
            counts[move]=counts[move]**(1./temp)
        counts_sum = float(sum(counts.values()))
        probs = {move:x/counts_sum for (move,x) in counts.items()}
        return probs

    def selection(self, canonicalBoard):
        """
        Descente dans l'arbre pour selectionner une feuille
        """
        self.SAhistories=[]
        self.boardsToPredict=[]
        fen = canonicalBoard.fen()
        for _ in range(self.parallel_search):
            if (fen in self.Ps) and (max([self.Ls[(fen,a)] for a in self.Ps[fen]])==0):
                break
            self.SAhistory=[]
            board = self.findNext(canonicalBoard)
            if board!=None:
                self.SAhistories.append(self.SAhistory.copy())
                self.boardsToPredict.append(board.copy())
        return {i:e for (i,e) in enumerate(self.boardsToPredict)}

    def findNext(self, canonicalBoard):
        """
        Descente récursive pour selectionner une feuille
        """
        s = canonicalBoard.fen()

        if s not in self.Es:
            self.Es[s] = canonicalBoard.result()
        if self.Es[s] != 0:
            # Noeud terminal
            self.backpropagation(canonicalBoard,self.SAhistory,None,self.Es[s])
            return None

        if s not in self.Ps:
            # Feuille
            for (i,j) in self.SAhistory:
                self.Ls[(i,j)]-=1
            self.Ns[s] = 0
            return canonicalBoard

        cur_best = -float('inf')
        best_move = -1

        # On choisit l'action avec la plus grande valeur ucb
        for a in [e for e in self.Ps[s] if self.Ls[(s,e)]>0]:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)

            if u > cur_best:
                cur_best = u
                best_move = a

        a = best_move
        self.SAhistory.append((s,a))

        canonicalBoard = canonicalBoard.copy()
        canonicalBoard.push(best_move)
        next_s = canonicalBoard.mirror()

        return self.findNext(next_s)

    def backpropagation(self,board,history,pi,v):
        """
        Mise à jour des Qsa, Nsa et Ls de toutes les actions choisies lors d'une descente
        """
        newNodes = 0
        if pi!=None:
            fen = board.fen()
            self.Ps[fen] = pi
            newNodes = len(pi)
            for a in pi:
                self.Ls[(fen,a)] = 1
        v=-v
        for (s,a) in history[::-1]:
            self.Ls[(s,a)] += newNodes
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)] = v
                self.Nsa[(s, a)] = 1
            self.Ns[s] += 1
            v=-v

    def parallel_backpropagation(self,pi,v):
        """
        Mise à jour des Qsa, Nsa et Ls de toutes les descentes
        """
        for i in range(len(self.boardsToPredict)):
            self.backpropagation(self.boardsToPredict[i],self.SAhistories[i],pi[i],v[i])