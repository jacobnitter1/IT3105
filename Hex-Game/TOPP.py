import Agent
import Environment

import numpy as np
import random

boardsize =3
starting_player = 1
RBUF_size = 100
minibatch_size = 10

game= Environment.HexGame("Diamond",boardsize, starting_player)
rollout_game = Environment.HexGame("Diamond",boardsize,starting_player)

policy_network = Agent.Policy_Network(boardsize)
mcts =Agent.MCTS( starting_player, game, rollout_game)

RBUF = Agent.replay_buffer(RBUF_size)
num_tournament_games = 25

policies=[]
for i in range(0,5):
    policies.append(Agent.Policy_Network(boardsize))
    policies[i].load_weights('./checkpoints/chkp_size_'+str(boardsize)+"_"+str(i))
print(policies)
results = []
for i in range(0,5):
    for j in range(i,5):
        if i  != j:
            result = 0
            for k in range(0,num_tournament_games):
                #print("------")
                game.reset_board()
                if k%2 == 0:
                    game.set_starting_player(2)
                else:
                    game.set_starting_player(1)
                while not game.is_game_done()[0]:

                    #game.print_boardState()
                    state=game.get_NN_state()
                    legal_actions = game.get_legal_actions()
                    if game.get_last_player() == 1:
                        a = policies[i].get_action(state,legal_actions,0.99)

                        game.do_action(a,2)
                    else:
                        a = policies[j].get_action(state,legal_actions,0.99)
                        game.do_action(a,1)
                if game.get_winner() == 1:
                    #print("Player 1 won", game.get_winner())
                    result+=1
                #else:
                #    print("player 2 won", game.get_winner())
            results.append([i,j,result/num_tournament_games])
#print(results[0][0]," vs. ", results[0][1], " : ",results[0][2])

for i in range(0,len(results)):
    print(results[i][0]," vs. ", results[i][1], " : ",results[i][2])
