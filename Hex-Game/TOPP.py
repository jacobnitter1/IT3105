import Agent
import Environment

import numpy as np
import random

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

boardsize =4
starting_player = 1
RBUF_size = 100
minibatch_size = 10

game= Environment.HexGame("Diamond",boardsize, starting_player)
rollout_game = Environment.HexGame("Diamond",boardsize,starting_player)

policy_network = Agent.Policy_Network(boardsize, conv_bool = False)
#mcts =Agent.MCTS( starting_player, game, rollout_game)

#RBUF = Agent.replay_buffer(RBUF_size)
num_tournament_games = 10



learning_rate = 0.001
NN_structure = [128,128]
optimizer_ = 'Adam'
activation_function_ = 'sigmoid'

if True:
    save_path = "./checkpoints/2020-04-24/nn_struct_"+str(NN_structure[0])+"_"+str(len(NN_structure))+"_"+optimizer_+"_"+activation_function_+"_size_"+str(boardsize)+"_lr_"+str(learning_rate)+"_chkp_"
    print(save_path)
    policies=[]
    num_agents = 5
    for i in range(0,num_agents):
        policies.append(Agent.Policy_Network(boardsize,nn_struct=NN_structure,conv_bool = False))
        policies[i].load_weights(save_path+str(i))#"/home/vilde/Code/IT3105_/Hex-Game/checkpoints/2020-04-21/3_nn_struct_100_3_Adam_sigmoid_size_3_chkp_"+str(i))
        #status.assert_existing_objects_matched()
    #print(policies)
    results = []
    points = np.zeros(num_agents)
    points_played = np.zeros(num_agents)
    for i in range(0,num_agents):
        game.reset_board()
        state=game.get_NN_state()
        legal_actions = game.get_legal_actions()
        print(i," distribution : ")
        print(state)
        policies[i].print_distribution(state, legal_actions)
        for j in range(0,num_agents):
            if i  != j:
                result = 0

                #print(i," vs. ", j, " : ")
                for k in range(0,num_tournament_games):
                    #print("------")
                    game.reset_board()
                    if k%1 == 0:
                        game.set_starting_player(1)
                    else:
                        game.set_starting_player(2)
                    #game.print_state()
                    while not game.is_game_done()[0]:

                        #game.print_boardState()
                        state=game.get_NN_state()
                        legal_actions = game.get_legal_actions()
                        #print("-",game.get_last_player())
                        if game.get_last_player() == 1:
                            #print(state)
                            a = policies[j].get_distributed_action(state,legal_actions,0)

                            game.do_action(a,2)
                        else:
                            a = policies[i].get_distributed_action(state,legal_actions,0)
                            game.do_action(a,1)
                        #game.print_state()

                    #policies[i].print_distribution(state, legal_actions)
                    if game.get_winner() == 1:
                        #print("Player 1 won", game.get_winner())
                        result+=1
                        points[i] +=1
                        points_played[i] +=1
                        points_played[j] +=1
                    else:
                        #print(game.get_winner())
                        points[j] +=1

                        points_played[i] +=1
                        points_played[j] +=1
                    #else:
                    #    print("player 2 won", game.get_winner())
                #game.print_state()
                #if game.get_winner() == 1:
                #    print(i, " won!")
                #else:
                #    print(j, " won!")
                #print("---")

                results.append([i,j,result/num_tournament_games])
    #print(results[0][0]," vs. ", results[0][1], " : ",results[0][2])

    for i in range(0,len(results)):
        print(results[i][0]," vs. ", results[i][1], " : ",results[i][2])

    for i in range(0,len(points)):
        print("Player ",i," points : ", points[i]/points_played[i])


if False:
    optimizers = ['Adam']
    activation_functions = ['sigmoid','relu','tanh']
    learning_rates =[0.01,0.001,0.1]
    NN_structures=[[100]]
    for optimizer_ in optimizers:
        for activation_function_ in activation_functions:
            for learning_rate in learning_rates:
                for NN_structure in NN_structures:
                    save_path = "./checkpoints/2020-04-22/hyperparms_test/nn_struct_"+str(NN_structure[0])+"_"+str(len(NN_structure))+"_"+optimizer_+"_"+activation_function_+"_size_"+str(boardsize)+"_lr_"+str(learning_rate)+"_chkp_"
                    print(save_path)
                    policies=[]
                    num_agents = 5
                    for i in range(0,num_agents):
                        policies.append(Agent.Policy_Network(boardsize,nn_struct=NN_structure))
                        policies[i].load_weights(save_path+str(i))#"/home/vilde/Code/IT3105_/Hex-Game/checkpoints/2020-04-21/3_nn_struct_100_3_Adam_sigmoid_size_3_chkp_"+str(i))
                        #status.assert_existing_objects_matched()
                    #print(policies)
                    results = []
                    points = np.zeros(num_agents)
                    for i in range(0,num_agents):
                        game.reset_board()
                        state=game.get_NN_state()
                        legal_actions = game.get_legal_actions()
                        #policies[i].print_distribution(state, legal_actions)
                        for j in range(0,num_agents):
                            if i  != j:
                                result = 0

                                #print(i," vs. ", j, " : ")
                                for k in range(0,num_tournament_games):
                                    #print("------")
                                    game.reset_board()
                                    if k%1 == 0:
                                        game.set_starting_player(1)
                                    else:
                                        game.set_starting_player(2)
                                    #game.print_state()
                                    while not game.is_game_done()[0]:

                                        #game.print_boardState()
                                        state=game.get_NN_state()
                                        legal_actions = game.get_legal_actions()
                                        #print("-",game.get_last_player())
                                        if game.get_last_player() == 1:
                                            a = policies[j].get_distributed_action(state,legal_actions,0)

                                            game.do_action(a,2)
                                        else:
                                            a = policies[i].get_distributed_action(state,legal_actions,0)
                                            game.do_action(a,1)
                                        #game.print_state()

                                    #policies[i].print_distribution(state, legal_actions)
                                    if game.get_winner() == 1:
                                        #print("Player 1 won", game.get_winner())
                                        result+=1
                                        points[i] +=1
                                    else:
                                        points[j] +=1
                                    #else:
                                    #    print("player 2 won", game.get_winner())
                                #game.print_state()
                                #if game.get_winner() == 1:
                                #    print(i, " won!")
                                #else:
                                #    print(j, " won!")
                                #print("---")

                                results.append([i,j,result/num_tournament_games])
                    #print(results[0][0]," vs. ", results[0][1], " : ",results[0][2])

                    for i in range(0,len(results)):
                        print(results[i][0]," vs. ", results[i][1], " : ",results[i][2])

                    for i in range(0,len(points)):
                        print("Player ",i," points : ", points[i]/((num_agents-1)*2.0*num_tournament_games+1))
