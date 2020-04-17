import Agent
import Environment

import numpy as np
import random

boardsize =3
starting_player = 1
RBUF_size = 10_000
minibatch_size = 10
epsilon = 0.5

game= Environment.HexGame("Diamond",boardsize, starting_player)
rollout_game = Environment.HexGame("Diamond",boardsize,starting_player)

policy_network = Agent.Policy_Network(boardsize)
policy_network.save_weights('./checkpoints/chkp_size_'+str(boardsize)+'_-1')
mcts =Agent.MCTS( starting_player, game, rollout_game)

RBUF = Agent.replay_buffer(RBUF_size)



num_actual_games = 1_000
i_s = num_actual_games//5
num_simulated_games = 100
for g_a in range(1,num_actual_games+1):
    print("Game number ", g_a ," of total ", num_actual_games, " games.")
    game.reset_board()
    rollout_game.reset_board()
    S_init = game.get_NN_state()
    mcts = Agent.MCTS(1,game, rollout_game)
    #print(game.is_game_done(), game.get_NN_state())
    while not game.is_game_done()[0]:
        #game.print_state()
        rollout_game.set_state(game.get_NN_state(),game.get_last_player())
        state=rollout_game.get_NN_state()
        last_player = rollout_game.get_last_player()
        mcts.run_simulations(num_simulated_games, state, last_player, policy_network)
        legal_actions = game.get_legal_actions()
        a = policy_network.get_action(state,legal_actions,epsilon)
        D = mcts.get_distribution()
        #print(legal_actions,D)
        #print("Saving shapes : ",np.shape(state), np.shape(D))
        RBUF.save_experience(state,D)
        #RBUF.print_RBUF()


        if game.last_player == 1:

            game.do_action(a,2)
        else:
            game.do_action(a,1)
    if g_a > 2:
        mb_s , mb_D = RBUF.get_minibatch(minibatch_size)
        #print(mb_s, mb_D)
        #print("SHAPES " ,np.shape(mb_s), np.shape(mb_D))
        #print(mb_s.shape, mb_D.shape)
        policy_network.train(mb_s,mb_D)
    if g_a % i_s == 0:
        policy_network.save_weights('./checkpoints/chkp_size_'+str(boardsize)+"_"+str(g_a//i_s))
