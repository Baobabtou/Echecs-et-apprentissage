from Games import Connect4
from Models import NeuralXGBoost
from MCTS import MCTS

import tkinter as tk
import tkinter.messagebox
import time
import numpy as np

def politique(plateau):
    return NN.predictBatch({0:plateau})[0][0]

def coupajouer(plateau,joueur,episodeStep):
    global mcts
    if joueur == 1:
        plateau=plateau.mirror()


    for k in range(numMCTSSims):
        boardToPredict = mcts[joueur].selection(plateau)
        #print(boardToPredict)
        if boardToPredict != None:
            pi, v = NN.predictBatch({0 : boardToPredict}) 
            #print(pi,v)
            mcts[joueur].backpropagation(pi[0],v[0])         
    for i in plateau.get_legal_moves():
        plat = plateau.copy()
        plat.push(i,(plat.k+1) % 2)
        if plat.winner in [-1,1] :
            return i
    pi = mcts[joueur].getActionProb(plateau,temp=0)
    d = {i:e for i,e in enumerate(list(pi.keys()))}
    move = np.random.choice(list(d.keys()), p=list(pi.values()))
    move = d[move]
    return move

#Le joueur rouge commence toujours

coup = ()

def coup_si_clique_gauche(event):
    global coup
    coup = (event.x,event.y)


def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)

tk.Canvas.create_circle = _create_circle

def affichePlateau(game,can,IA = False):
    #Ligne de séparation :
    can.create_line(400,0,400,350, fill="black", width = 3)

    #Quadrillage :
    #Lignes :
    l = [25+50*k for k in range(0,7)]
    for k in l:
        can.create_line(25,k,375,k, fill="black")
    
    #Colonnes :
    l = [25+50*k for k in range(0,8)]
    for k in l:
        can.create_line(k,25,k,325, fill="black")    
    
    #Placement des pions :
    bo=game.board
    r=20
    for x in range(7):
        for y in range(6):
            if bo[x][y][0]==1 :
                can.create_circle(50+50*x,50+50*(5-y),r, fill = 'red')
            if bo[x][y][1]==1 :
                can.create_circle(50+50*x,50+50*(5-y),r, fill = 'gold')                

    #initiative
    if game.k %2 ==1:
        can.create_circle(425, 25, 5, fill = 'red')
    if game.k %2 ==0:
        can.create_circle(425, 25, 5, fill = 'gold')

    #Affichage des idées de l'ordi :
    joueur = joueurs[(game.k + 1) % 2]
    if joueur == "humain_conseillé" : 
        pol = politique(game)
        txt="\n"
        for k in pol :
            txt+=(str(k+1) + " : " + str(round(pol[k],2)) + "\n")
        can.create_text(430,300, text= txt, font="Arial 8")

    #Affichage de la préférence et de l'estim de l'ordi (modifier le premier if... pour masquer/afficher la flèche) :
    m = 0
    if joueur == "humain_conseillé" :
        #Préférence
        fle = coupajouer(game,(game.k +1) % 2,game.k)
        can.create_line(50+50*fle,350,50+50*fle,330,arrow=tk.LAST)

        #Estimation
        if (game.str,fle) in mcts[(game.k + 1) % 2].Qsa:
            esti=mcts[(game.k + 1) % 2].Qsa[(game.str,fle)]*((-1)**((game.k + 1) % 2))
            #print(mcts[(game.k + 1) % 2].Nsa[(game.str,fle)])
        else :
            print("attention_esti_non_trouvée")
            esti = NN.predictBatch({0:game})[1][0]
        txt2 = str(round(esti,2))
        can.create_text(425,45, text= txt2, font="Arial 8")
    return ()


def conversion(coup):
    a, b = coup[0], coup[1]
    if 25<=a and a<=375 and 25<=b and b<=325:
        return (a-25) // 50
    else :
        return "Illegal"


def partie(game):
    global coup
    if (joueurs[0] == "IA" and (game.k + 1)% 2 == 0) or (joueurs[1] == "IA" and (game.k + 1)% 2 == 1):
        time.sleep(0.1)
        coup = coupajouer(game,(game.k +1) % 2,game.k)
        game.push(coup,(game.k+1)%2)
        can.delete('all')
        affichePlateau(game,can, IA = True)
        can.update()
        coup = ()        

    elif ((joueurs[0] == "humain_seul" or joueurs[0] == "humain_conseillé") and (game.k + 1)% 2 == 0) or \
    ((joueurs[1] == "humain_seul" or joueurs[1] == "humain_conseillé") and (game.k + 1)% 2 == 1):
        fen.bind ("<Button-1>", coup_si_clique_gauche)
        if coup != ():
            coup = conversion(coup)
            if coup == "Illegal":
                print("Clic raté !")
            elif game.legal_moves[coup]==1:
                game.push(coup,(game.k+1)%2)
                can.delete('all')
                affichePlateau(game,can)
                can.update()
            else :
                print("Coup illégal !")
            coup = ()

    if game.game_over :
        print("Fini !")
        affichePlateau(game,can)
        can.update()
        time.sleep(0.5)
        if game.draw:
            print("Match nul !")
            tk.messagebox.showinfo('Jeu terminé', 'Match nul !')
            return 0
        elif game.k % 2 == 0 :
            print("Bravo ! Le joueur rouge a gagné" )
            tk.messagebox.showinfo('Jeu terminé', 'Le joueur rouge a gagné !')
            return 1
        elif game.k % 2 == 1 :
            print("Bravo ! Le joueur jaune a gagné" )
            tk.messagebox.showinfo('Jeu terminé', 'Le joueur jaune a gagné !')
            return -1
    can.after(20,partie,game)

def main(iterations, assist):
    """
    Starts main function
    """

    global numMCTSSims, NN, joueurs, mcts
    
    ### Paramètres ###
    cpuct = 1
    numMCTSSims = iterations

    #Joueurs possibles : humain_seul, humain_conseillé, IA #
    if assist:
        joueur1 = "humain_conseillé"
    else:

        joueur1 = "humain_seul"
    joueur2 = "IA"

    NN = NeuralXGBoost(use_v=False)
    NN.load(xgb_folder='data/xgb', nn_folder='data', nn_filename='daily_donkey/save48.h5')
    cpuct = 1
    mcts = [MCTS(cpuct),MCTS(cpuct)]
    joueurs = [joueur1,joueur2]

    global fen, can
    fen = tk.Tk(baseName = "Puissance 4 " + joueur1 + " " + joueur2)
    can = tk.Canvas (fen, bg='light grey', height=350, width=450) 
    can.pack() 
    game = Connect4() 
    affichePlateau(game,can)
    can.update()
    partie(game)
    fen.mainloop()

