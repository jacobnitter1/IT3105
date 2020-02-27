import Environment
import Agent

GAME_TYPE = "NIM" # OR LEDGE
VERBOSE_MODE = True

G = 1 # NUMBER OF GAMES IN A BATCH
P = 1 # STARTING PLAYER OPTION
M = 10 # NUMBER OF SIMULATIONS

N = 4 #STARTING NUMBER OF PIECES IN EACH GAME
K = 2 # MAXIMUM NUMBER OF PIECES THAT PLAYER CAN REMOVE

B_INIT = [0,1,0,0,2,1,0,0] # BOARD INIT FOR LEDGE

if GAME_TYPE == "NIM":
    game = Environment.NIM(N,K,P, VERBOSE_MODE)
    rollout_game = Environment.NIM(N,K,P, False)
else:
    game = Environment.LEDGE(len(B_INIT), B_INIT.count(1), B_INIT,P,VERBOSE_MODE)
    rollout_game = Environment.LEDGE(len(B_INIT), B_INIT.count(1), B_INIT,P,False)

MCTS = Agent.MCTS(1, game, rollout_game)
