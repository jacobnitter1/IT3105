import Agent
import Environment
#from Environment import show_policy

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

def show_policy( policy, boardShape,boardSize, startingPlayer,env,DELAY_BETWEEN_MOVE):
	t_states = []
	t_states.append(np.copy(env.get_boardState()))
	#print(env.is_game_done())
	while not env.is_game_done()[0]:

		state=env.get_NN_state()
		last_player = env.get_last_player()
		legal_actions = env.get_legal_actions()
		a= policy_network.get_action(state,legal_actions,0)
		print("->",state,a)
		if env.last_player == 1:
			env.do_action(a,2)
		else:
			env.do_action(a,1)
		t_states.append(np.copy(env.get_boardState()))
	print("All states recorded.")
	vis = Environment.VisualizeHexGame(env.get_boardState(),"Diamond")
	vis.show_played_game(t_states,DELAY_BETWEEN_MOVE,"Diamond",boardSize,type)


num_actual_games = 500
i_s = num_actual_games//5
num_simulated_games = 1
policy_network.load_weights('./checkpoints/chkp_size'+str(boardsize)+"_"+str(4))
show_policy(policy_network, "Diamond",boardsize, starting_player,game,1)

for i in range(-1,-1):
	policy_network.load_weights('./checkpoints/chkp_size'+str(boardsize)+"_"+str(i))
	game.reset_board()
	rollout_game.reset_board()
	S_init = game.get_NN_state()
	print("---------------------------------------------------------------")
	while not game.is_game_done()[0]:
		game.print_state()
		#rollout_game.set_state(game.get_NN_state(),game.get_last_player())
		state=game.get_NN_state()
		#last_player = rollout_game.get_last_player()
		legal_actions = game.get_legal_actions()
		a= policy_network.get_action(state,legal_actions,0)
		#RBUF.print_RBUF()


		if game.last_player == 1:

			game.do_action(a,2)
		else:
			game.do_action(a,1)
		print("Game done ? ", game.is_game_done())
