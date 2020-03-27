import Environment
import Agent
import numpy as np
import random


GAME_TYPE = "Ledge" # OR LEDGE
VERBOSE_MODE = True

G = 10 # NUMBER OF GAMES IN A BATCH
P = 1
 # STARTING PLAYER OPTION
M = 50# NUMBER OF SIMULATIONS
EXPLORATION_RATE = 1

N = 50 #STARTING NUMBER OF PIECES IN EACH GAME
K = 10# MAXIMUM NUMBER OF PIECES THAT PLAYER CAN REMOVE

B_INIT = [0,1,0,1,0,1,0,1,0,0,2,0,0,1,1,1]#[0,2,0,1,0,1,1,1,0,0,0,0,1,0,1,0,1]# # BOARD INIT FOR LEDGE


done = False
num_times_p1_won = 0

def play_one_game(P, M, EXPLORATION_RATE,N,K,B_INIT,verbose_mode):
    done=False
    if P == 3:
        if random.random() < 0.5:
            P=2
        else:
            P= 1

    if GAME_TYPE == "NIM":
        if verbose_mode:
            game = Environment.NIM(N,K,P, True)
        else:
            game = Environment.NIM(N,K,P,False)
        rollout_game = Environment.NIM(N,K,P, False)
    else:
        if verbose_mode:
            game = Environment.Ledge(len(B_INIT), B_INIT.count(1), B_INIT,P,True)
        else:
            game = Environment.Ledge(len(B_INIT), B_INIT.count(1), B_INIT,P,False)
        rollout_game = Environment.Ledge(len(B_INIT), B_INIT.count(1), B_INIT,P,True)
    MCTS = Agent.MCTS(EXPLORATION_RATE, game, rollout_game)

    while not done:
        #MCTS.node_expansion()
        #print("before choose action :",game.get_state(),game.get_legal_actions())
        if game.get_last_player() == None:
            if P == 1:
                action = MCTS.get_action(M, game.get_state(), 2)
            else:

                action = MCTS.get_action(M, game.get_state(), 1)
        else:
            action = MCTS.get_action(M, game.get_state(), game.get_last_player())
        if game.get_last_player() == None:
            reward, winner, done = game.make_move(P,action)
        else:
            if game.get_last_player() == 1:
                reward, winner, done = game.make_move(2,action)
            else:
                #print("- 0")
                reward, winner, done = game.make_move(1,action)
                #print(game.get_state())
        #print(done)
        #break
        if done:
            #print("DONE!",winner)
            #print(MCTS.get_dictionary(),len(MCTS.get_dictionary()))
            if winner == 1:
                return 1
            else:
                return 0




if G == 1:
    num_times_p1_won = 0
    num_times_p1_won += play_one_game(P, M, EXPLORATION_RATE,N,K,B_INIT,True)
else:
    num_times_p1_won = 0
    for i in range(0,G):
        print("------------------------------")
        num_times_p1_won += play_one_game(P, M, EXPLORATION_RATE,N,K,B_INIT,True)




print("Player 1 won ",(num_times_p1_won/G)*100," % of the games.")
