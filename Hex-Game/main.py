import Agent
import Environment

import numpy as np
import random
import matplotlib.pyplot as plt


#ENVIRONMENT PIVOTAL PARAMETERS
boardsize =4
starting_player = 1

#REPLAY BUFFER PIVOTAL ARGUMENTS
RBUF_size = 100
minibatch_size =50

#ANET PIVOTAL ARGUMENTS

def epsilon(current_e, total_e):
	#return 0.1
	e = np.clip(0.5 - (current_e/total_e)/1.2,0.1,1)
	print("Current epsilon : ",e)
	return 0
learning_rate = 0.001
NN_structure = [128,128]
optimizer_ = 'Adam'
activation_function_ = 'sigmoid'

#MCTS PIVOTAL PARAMETERS
num_episodes = 1000
num_simulated_games_pr_move = 100

#SAVING PARAMS
num_saves = 4

def train(boardsize,RBUF_size, minibatch_size,num_episodes,num_simulated_games_pr_move,learning_rate, NN_structure,conv_bool,optimizer_, activation_function_, num_saves, save_path):


	game= Environment.HexGame("Diamond",boardsize, starting_player)
	rollout_game = Environment.HexGame("Diamond",boardsize,starting_player)
	policy_network = Agent.Policy_Network( boardsize,lr = learning_rate, nn_struct = NN_structure, activation_function = activation_function_,optimizer = optimizer_, conv_bool=False)
	mcts =Agent.MCTS( starting_player, game, rollout_game,1)

	RBUF = Agent.replay_buffer(RBUF_size)
	print("OK")

	num_actual_games = num_episodes
	i_s = num_actual_games//num_saves
	num_simulated_games = num_simulated_games_pr_move
	num_actual_games = i_s*num_saves
	correct_actions = []
	print("OK")
	for g_a in range(1,num_actual_games+1):
		print("Game number ", g_a ," of total ", num_actual_games, " games.")
		game.reset_board()
		rollout_game.reset_board()
		S_init = game.get_NN_state()
		if g_a%2 == 0:
			mcts = Agent.MCTS(1,game, rollout_game,epsilon(g_a, num_actual_games+1))
			game.set_starting_player(1)
			rollout_game.set_starting_player(1)
		else:
			mcts = Agent.MCTS(2,game, rollout_game,epsilon(g_a, num_actual_games+1))
			game.set_starting_player(2)
			rollout_game.set_starting_player(2)
 #print(game.is_game_done(), game.get_NN_state())
		ca = 0
		ta=0
		print("OK")

		while not game.is_game_done()[0]:
			print("In game ", g_a)
			#game.print_state()
			rollout_game.set_state(game.get_NN_state(),game.get_last_player())
			state=rollout_game.get_NN_state()
			last_player = rollout_game.get_last_player()
			#print(num_simulated_games)
			mcts.run_simulations(num_simulated_games, state, last_player, policy_network)
			legal_actions = game.get_legal_actions()
			#a = policy_network.get_distributed_action(state,legal_actions)#,epsilon(g_a, num_actual_games+1)/4.0)
			D = mcts.get_distribution(legal_actions)
			d=policy_network.predict([[game.get_NN_state()]])
			if np.argmax(D) == np.argmax(d):
				ca += 1
			ta += 1
	 #print(" ------------------!!!!!!!!!!!!!!!!!!!!!!!!")
	 #print(D)
	 #game.print_state()

			visit_counts=mcts.get_visit_counts_tree(rollout_game)
			a = np.random.choice(len(D),1,p=D)
			RBUF.save_experience(state,D)
			#print(visit_counts)
			RBUF.save_experiences(visit_counts)


			if game.last_player == 1:

				game.do_action(a,2)
			else:
				game.do_action(a,1)

			if RBUF.RBUF_ready(minibatch_size):
				for i in range(0,10):
					mb_s , mb_D = RBUF.get_minibatch(minibatch_size)
					policy_network.train(mb_s,mb_D)
			if game.is_game_done()[0]:
				break

		print(game.is_game_done())
		game.print_state()
		correct_actions.append(ca/ta)
		if g_a % i_s == 0:
			print("SAVE")
			policy_network.save_weights(save_path+str(g_a//i_s))


if True:
	save_path = "./checkpoints/2020-04-24/nn_struct_"+str(NN_structure[0])+"_"+str(len(NN_structure))+"_"+optimizer_+"_"+activation_function_+"_size_"+str(boardsize)+"_lr_"+str(learning_rate)+"_chkp_"
	#save_path = "./demo_agents/"+str(boardsize)

	game= Environment.HexGame("Diamond",boardsize, starting_player)
	rollout_game = Environment.HexGame("Diamond",boardsize,starting_player)
	policy_network = Agent.Policy_Network( boardsize,lr = learning_rate, nn_struct = NN_structure, activation_function = activation_function_,optimizer = optimizer_, conv_bool=False)
	critic_network = Agent.Critic_Network(boardsize,lr = learning_rate, nn_struct = NN_structure, activation_function = activation_function_,optimizer = optimizer_, conv_bool=False)
	#policy_network.init_weights()
	#policy_network.save_weights(save_path)
	policy_network.save_weights(save_path+'0')
	mcts =Agent.MCTS( starting_player, game, rollout_game,1,0)

	RBUF = Agent.replay_buffer(RBUF_size)
	print("OK")

	num_actual_games = num_episodes
	i_s = num_actual_games//num_saves
	num_simulated_games = num_simulated_games_pr_move
	num_actual_games = i_s*num_saves
	correct_actions = []
	print("OK")
	for g_a in range(1,num_actual_games+1):
		print("Game number ", g_a ," of total ", num_actual_games, " games.")
		game.reset_board()
		rollout_game.reset_board()
		S_init = game.get_NN_state()
		if g_a%2 == 0:
			mcts = Agent.MCTS(1,game, rollout_game,epsilon(g_a, num_actual_games+1),0)
			game.set_starting_player(1)
			rollout_game.set_starting_player(1)
		else:
			mcts = Agent.MCTS(2,game, rollout_game,epsilon(g_a, num_actual_games+1),0)
			game.set_starting_player(2)
			rollout_game.set_starting_player(2)
		#print(game.is_game_done(), game.get_NN_state())
		ca = 0
		ta=0
		print("OK")

		while not game.is_game_done()[0]:
			print("In game ", g_a)
			#game.print_state()
			rollout_game.set_state(game.get_NN_state(),game.get_last_player())
			state=rollout_game.get_NN_state()
			last_player = rollout_game.get_last_player()
			#print(num_simulated_games)
			mcts.run_simulations(num_simulated_games, state, last_player, policy_network,critic_network)
			legal_actions = game.get_legal_actions()
			#a = policy_network.get_distributed_action(state,legal_actions)#,epsilon(g_a, num_actual_games+1)/4.0)
			D = mcts.get_distribution(legal_actions)
			d=policy_network.predict([[game.get_NN_state()]])
			if np.argmax(D) == np.argmax(d):
				ca += 1
			ta += 1
			visit_counts=mcts.get_visit_counts_tree(rollout_game)
			critic_targets = mcts.get_critic_targets(legal_actions)
			#print("!!!!",critic_targets)
			a = np.random.choice(len(D),1,p=D)
			RBUF.save_experience(state,D,critic_targets)
			#print(visit_counts)
			RBUF.save_experiences(visit_counts)
			#RBUF.print_RBUF()
			#print(game.is_game_done())


			if game.last_player == 1:

				game.do_action(a,2)
			else:
				game.do_action(a,1)

			if RBUF.RBUF_ready(minibatch_size):
				for i in range(0,10):
					mb_s , mb_D ,C= RBUF.get_minibatch(minibatch_size)
					policy_network.train(mb_s,mb_D)
					#critic_network.train(mb_s,C)
			if game.is_game_done()[0]:
				break

		print(game.is_game_done())
		game.print_state()
		correct_actions.append(ca/ta)
		if g_a % i_s == 0:
			print("SAVE")
			policy_network.save_weights(save_path+str(g_a//i_s))
#RBUF.print_RBUF()
equals = []
for j in range(0,0):
	print("Train round : ",j)
	policy_network.train(mb_s,mb_D)
	all_equal = True
	for i in range(0,len(mb_s)):
		D = mb_D[i]
		d=policy_network.predict([[mb_s[i]]])
		if np.argmax(d) != np.argmax(D):
			all_equal = False
	if all_equal:
		print("All equal after ",j)
		equals.append(j)
		all_equal
		#break
	#print(j)
#print(mb_s,mb_D)
print(equals)
for i in range(0,len(mb_s)):
	d=policy_network.predict([[mb_s[i]]])
	print("---")
	print(mb_s[i])
	print( np.argmax(d), np.argmax(mb_D[i]))
	print(d)
	print(mb_D[i])
plt.plot(correct_actions)
plt.show()

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


					game= Environment.HexGame("Diamond",boardsize, starting_player)

					rollout_game = Environment.HexGame("Diamond",boardsize,starting_player)
					policy_network = Agent.Policy_Network( boardsize,lr = learning_rate, nn_struct = NN_structure, activation_function = activation_function_,optimizer = optimizer_)
					policy_network.init_weights()
					policy_network.save_weights(save_path+'0')
					mcts =Agent.MCTS( starting_player, game, rollout_game,1)

					RBUF = Agent.replay_buffer(RBUF_size)

					num_actual_games = num_episodes
					i_s = num_actual_games//num_saves
					num_simulated_games = num_simulated_games_pr_move
					num_actual_games = i_s*num_saves
					for g_a in range(1,num_actual_games+1):
						print("Game number ", g_a ," of total ", num_actual_games, " games.")
						game.reset_board()
						rollout_game.reset_board()
						S_init = game.get_NN_state()
						if g_a%2 == 0:
							mcts = Agent.MCTS(1,game, rollout_game,epsilon(g_a, num_actual_games+1))
							game.set_starting_player(1)
							rollout_game.set_starting_player(1)
						else:
							mcts = Agent.MCTS(2,game, rollout_game,epsilon(g_a, num_actual_games+1))
							game.set_starting_player(2)
							rollout_game.set_starting_player(2)
						#print(game.is_game_done(), game.get_NN_state())
						while not game.is_game_done()[0]:
							#game.print_state()
							#print("1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
							rollout_game.set_state(game.get_NN_state(),game.get_last_player())
							state=rollout_game.get_NN_state()
							last_player = rollout_game.get_last_player()
							mcts.run_simulations(num_simulated_games, state, last_player, policy_network)
							legal_actions = game.get_legal_actions()
							#a = policy_network.get_distributed_action(state,legal_actions)#,epsilon(g_a, num_actual_games+1)/4.0)
							D = mcts.get_distribution(legal_actions)
							#print("main.py :  ", D, D.sum())
							if D.sum() == 0:
								a = np.random.choice(legal_actions,1)
							else:
								a = np.random.choice(len(D),1,p=D)#.argmax(D)
							#print(D)
							#game.print_state()
							print(game.get_winner())
							#print(legal_actions,D)
							#print("Saving shapes : ",np.shape(state), np.shape(D))
							RBUF.save_experience(state,D)
							RBUF.save_experiences(mcts.get_visit_counts_tree(rollout_game))
							#RBUF.print_RBUF()
							print(game.is_game_done())


							if game.last_player == 1:

								game.do_action(a,2)
							else:
								game.do_action(a,1)

							if RBUF.RBUF_ready(minibatch_size):
								mb_s , mb_D ,C= RBUF.get_minibatch(minibatch_size)

								policy_network.train(mb_s,mb_D)
								policy_network.train(mb_s,mb_D)

							#print(mb_s, mb_D)
							#print("SHAPES " ,np.shape(mb_s), np.shape(mb_D))
							#print(mb_s.shape, mb_D.shape)
							if game.is_game_done()[0]:

								break
						print(game.is_game_done())
						game.print_state()
						if g_a % i_s == 0:
							print("SAVE")
							policy_network.save_weights(save_path+str(g_a//i_s))
