import Environment
import Agent
import numpy as np


GAME_TYPE = "Ledge" # OR LEDGE
VERBOSE_MODE = True

G = 100 # NUMBER OF GAMES IN A BATCH
P = 1 # STARTING PLAYER OPTION
M = 500 # NUMBER OF SIMULATIONS

N = 6 #STARTING NUMBER OF PIECES IN EACH GAME
K = 2 # MAXIMUM NUMBER OF PIECES THAT PLAYER CAN REMOVE

B_INIT = [1,0,1,0,0,2,1,0] # BOARD INIT FOR LEDGE


done = False
num_times_p1_won = 0
for g in range(0,G):
    if GAME_TYPE == "NIM":
        game = Environment.NIM(N,K,P, VERBOSE_MODE)
        rollout_game = Environment.NIM(N,K,P, False)
    else:
        game = Environment.Ledge(len(B_INIT), B_INIT.count(1), B_INIT,P,VERBOSE_MODE)
        rollout_game = Environment.Ledge(len(B_INIT), B_INIT.count(1), B_INIT,P,False)
    MCTS = Agent.MCTS(P, game, rollout_game)

    while not done:
        MCTS.node_expansion()

        #print("before choose action :",game.get_state(),game.get_legal_actions())
        if game.get_last_player() == None:
            if P == 1:
                action = MCTS.choose_next_action(M, game.get_state(), 2)
            else:

                action = MCTS.choose_next_action(M, game.get_state(), 1)
        else:
            action = MCTS.choose_next_action(M, game.get_state(), game.get_last_player())
        #print("After choose action : ",game.get_state(),game.get_legal_actions())
        #print("CHOSEN ACTION : ",action, ", FOR STATE :" ,game.get_state())
        if game.get_last_player() == None:
            game.make_move(P,action)
        else:
            if game.get_last_player() == 1:
                #print("- 1")
                game.make_move(2,action)
            else:
                #print("- 0")
                game.make_move(1,action)
                #print(game.get_state())
        done,winner=game.is_game_done()
        if done:
            if winner == 1:
                num_times_p1_won += 1
            break
    if g == G:
        break


print("Player 1 won ", (num_times_p1_won/G)*100.0," % of the games.")
