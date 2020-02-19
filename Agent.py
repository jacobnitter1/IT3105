import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras.activations import relu,tanh,linear,sigmoid
import SplitGD
import math
#import splitgd


class Critic_NN(SplitGD.SplitGD):
	def __init__(self, obs_space, hidden_layers_dim, gamma,lr,e_decay_factor):
		#Critic_NN( int(obs_space),NN_structure, gamma,alpha)
		self.obs_space = obs_space
		self.hidden_layers_dim = hidden_layers_dim
		self.e_decay_factor = e_decay_factor
		self.lr = lr
		self.e_traces = self.init_e_traces()
		self.model = SplitGD.SplitGD(self.createNN(),self.e_decay_factor,self.e_traces,self.lr)
		#self.model = self.createNN()
		self.gamma = gamma
		#??? Hvilken størrelse skal eligibility tracesene ha?????

	def init_e_traces(self):
		weights_pr_layer=[]
		weights_pr_layer.append([self.obs_space,self.hidden_layers_dim[0]])
		for i in range(0,len(self.hidden_layers_dim)-1):
			weights_pr_layer.append([self.hidden_layers_dim[i],self.hidden_layers_dim[i+1]])
		weights_pr_layer.append([self.hidden_layers_dim[-1],1])
		#return np.zeros(weights_pr_layer)

	def createNN(self,output_dim =1,activation_function = 'sigmoid'):
		model = Sequential()
		nn_struct = self.hidden_layers_dim
		num_layers = len(nn_struct)
		model.add(Dense(nn_struct[0],input_dim=self.obs_space))
		for i in range(1,num_layers):
			model.add(Dense(nn_struct[i], activation = activation_function))
		model.add(Dense(output_dim, activation = 'sigmoid'))
		sgd = SGD(learning_rate = self.lr,momentum = 0.0, nesterov=False)
		model.compile(loss='mse',optimizer=sgd)
		print(model.summary())
		return model

	def get_state_value(self, state):
		#print(np.shape(state))
		#if np.shape(state) == (1,16):
		#	print(np.shape(state[0]))
		#	return self.model.get_model().predict(state[0])
		#else:
		return self.model.get_model().predict([state])

	def get_TD_error(self,reward,state,next_state):
		same_booll= True
		for i in range(0,len(state)):
			if state[i] != next_state[i]:
				same_booll = False
		if same_booll:
			TD = -1
		else:
			#print(np.shape(state), state)
			#print(np.shape(next_state),next_state)
			#if np.shape(state) == (16,):
			#	print("shape (16,)")
			#	s_val = self.get_state_value([state])
			#	ns_val = self.get_state_value([next_state])
			#	return reward + self.gamma*s_val-ns_val
			#else:
			TD =reward + self.gamma*self.get_state_value(state)-self.get_state_value(next_state)
		if math.isnan(TD):
			print( "TD = ", reward, " + ", self.gamma, " * ", self.get_state_value(state)," - ", self.get_state_value(next_state))
		#print("TD error is " ,TD[0][0])
		return TD


	def update_critic(self, trajectory_states, TD):

		if len(trajectory_states)==1:
			states=[trajectory_states]
			y = TD
			#print(TD)
		else:
			states = np.zeros((len(trajectory_states),self.obs_space))#trajectory_states
			y = np.zeros(len(trajectory_states))
			for i in range(0,len(trajectory_states)):
				states[i] = trajectory_states[i]
				#print(TD)
				y[i] = TD

		#print(len(states),	states,y)
		#SplitGD.fit()
		self.model.fit_SplitGD(states,y,verbose=False)#, batch_size=len(trajectory_states), epochs=10,verbose=0
		#self.model.fit(states,y,verbose=False, batch_size=len(trajectory_states),epochs=10)

	def reset_e_traces(self):
		self.model.reset_e_traces()
	def update_e_traces(self,state):
		pass
		### HOW SHOULD THIS BE DONE? Update every step and then for every step in trajectory? Do it like this for now


class Actor_tab():
	def __init__(self, obs_space, action_space, epsilon, learning_rate, e_discount):
		self.table = np.random.rand(2**((obs_space)),obs_space,action_space)
		print("policy table made")
		#for i in range(0, 2**obs_space):
		#	for j in range(0, obs_space):
		#		for k in range(0,action_space):
		#			self.table[i,j,k] += np.random.rand()/4
		self.num_entries_obs = 2**((obs_space)*(obs_space))
		self.num_entries_act=2**(action_space)
		self.e_traces = np.zeros((2**((obs_space)),(obs_space),(action_space)))
		self.action_space = action_space
		self.obs_space = obs_space

		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.e_discount = e_discount

	def save_policy(self,shape,size, method):
		name = str(shape)+"_"+str(size)+"_"+str(method)#+"_"+str(np.datetime64("today",'D'))
		np.save(name,self.table)

	def set_policy(self,q_value_table):
		self.table = q_value_table

	def update_illegal_move(self,state,action):
		state_num = self.bin_state_to_state_num(state)
		self.table[state_num][action[0]][action[1]]-=1

	def print_table(self,state):
		state = self.bin_state_to_state_num(state)
		print(self.table[state])

	def get_table(self):
		return self.table

	def bin_state_to_state_num(self, state):

		s = ''.join(str(int(elem)) for elem in state)
		return int(s,2)

	def binary_to_int_list(self,state,action):
		a_b = np.binary_repr(action)
		a_l = [int(a) for a in a_b]
		bin_sap = state
		for i in a_l:
			bin_sap.append(i)
		return bin_sap

	def merge_obs_action(self, obs, action):
		actions = np.zeros(len(self.action_space))
		#actions[action]=1
        #obs = np.append(obs,actions)
		return obs

	def get_action(self, state,epsilon):
		if np.random.rand()<epsilon:#EPSILON
			peg = np.random.randint(0,self.obs_space)
			action = np.random.randint(0,self.action_space)
			#print("GET_ACTION - random: ", state, peg,action)
		else:

			state_num = self.bin_state_to_state_num(state)
			#print("state_num : ",state_num)
			q_vals = self.table[state_num]
			best_val = q_vals[0][0]
			peg = 0
			action = 0
			#print(self.e_traces[state_num])
			#print(q_vals)
			for r in range(0,(self.obs_space)):
				for c in range(0,(self.action_space)):
					if q_vals[r][c] >= best_val:
						best_val = q_vals[r][c]
						peg =r
						action = c
			#print("GET ACTION : ",state, peg, action)


		return peg, action

	def update_e_traces(self,state,action):
		#print("BEFORE UPDATE", self.e_traces)
		num =1
		if num == 0:
			return None
		elif num == 1:
			state_num = self.bin_state_to_state_num(state)
			self.e_traces[state_num][action[0]][action[1]]=1
			#print("---->",self.e_traces[state_num])
		else:
			#print(state,num, len(state))
			for i in range(0,num):
				state_num = self.bin_state_to_state_num(state[i])
				for j in range(0,i):
					state_num = self.bin_state_to_state_num(state[j])
					self.e_traces[state_num][action[j][0]][action[j][1]] *= self.e_discount

					#print("---->",self.e_traces[state_num])
				self.e_traces[state_num][action[i][0]][action[i][1]] =1
		#print("AFTER",self.e_traces)
	def reset_e_traces(self):

		self.e_traces = np.zeros((2**((self.obs_space)),(self.obs_space),(self.action_space)))

	def update_actor(self, states, actions, TD_error):
		#print(states, actions)
		for i in range(0, len(states)):
			state =self.bin_state_to_state_num(states[i])
			action = actions[i]
			self.table[state][action[0]][action[1]] = self.table[state][action[0]][action[1]] + self.learning_rate*TD_error*self.e_traces[state][action[0]][action[1]]

class Critic_tab():
	def __init__(self, obs_space,gamma, alpha, lamda):
		self.value_table = np.random.rand(2**(obs_space))
		self.value_table = np.zeros(2**obs_space)
		self.e_traces = np.zeros(np.shape(self.value_table))
		self.gamma= gamma
		self.alpha = alpha
		self.lamda = lamda
		self.e_discount =lamda
	def print_values(self):
		print(self.value_table)
	def get_TD_error(self,reward,state,next_state):
		if state == next_state:
			delta = -1
		else:
			delta = reward + self.gamma*self.value_table[self.bin_state_to_state_num(next_state)] - self.value_table[self.bin_state_to_state_num(state)]
		return delta

	def update_critic(self, states,TD_error):
		if len(states) == 0:
			return None
		elif len(states) == 1:
			state = self.bin_state_to_state_num(states[0])
			self.value_table[state] += self.alpha*TD_error*self.e_traces[state]
		else:
			for i in range(0,len(states)):
				state = self.bin_state_to_state_num(states[i])
				self.value_table[state] += self.alpha*TD_error*self.e_traces[state]

	def update_e_traces(self,state):
		#print(state)
		for elem in self.e_traces:
			elem = self.e_discount*elem
		#self.e_traces = self.e_discount*self.e_traces

		self.e_traces[self.bin_state_to_state_num(state)] = 1

	def reset_e_traces(self):
		self.e_traces = np.zeros(np.shape(self.value_table))

	def bin_state_to_state_num(self, state):
		s = ''.join(str(int(elem)) for elem in state)
		return int(s,2)

class Agent():
	def __init__(self,critic_type, NN_structure, obs_space, action_space, critic_lerning_rate, actor_learning_rate, critic_e_decay_rate, actor_e_decay_rate,gamma, epsilon=0.0):
		self.critic_type = critic_type
		if critic_type == "NN":
			self.critic = Critic_NN(obs_space,NN_structure, critic_e_decay_rate,critic_lerning_rate,gamma)
		else:
			self.critic = Critic_tab(obs_space, gamma, critic_lerning_rate, critic_e_decay_rate)
		self.actor= Actor_tab(obs_space = obs_space, action_space=action_space, epsilon=epsilon, learning_rate=actor_learning_rate, e_discount=actor_e_decay_rate)
		self.trajectory_states = []
		self.trajectory_actions= []
		self.obs_space = obs_space
		self.action_space = action_space
		self.num_memorized = 0

		self.replay_trajectories_states = []
		self.replay_trajectories_actions = []

	def set_policy(self,policy_table):
		self.actor.set_policy(policy_table)

	def update_replay_trajectories(self, state,action):
		self.replay_trajectories_states.append(state)
		self.replay_trajectories_actions.append(action)

	def get_replay_trajectories(self):
		return self.replay_trajectories_states, self.replay_trajectories_actions

	def reset_replay_trajectories(self):
		self.replay_trajectories_states = []
		self.replay_trajectories_actions = []


	def reset_trajectories(self):
		self.trajectory_states = []
		self.trajectory_actions= []
		self.num_memorized = 0
	def reset_e_traces(self):
		self.actor.reset_e_traces()
		self.critic.reset_e_traces()

	def update_trajectories(self,state,action):
		#print(self.trajectory_states," and ", self.trajectory_actions)
		self.trajectory_states.append(state)
		#print(self.trajectory_states)
		self.trajectory_actions.append(action)

		#print(self.trajectory_states," and ", self.trajectory_actions)
		self.actor.update_e_traces(state,action)
		#self.critic.update_e_traces(state)
		self.num_memorized +=1

	def get_trajectories(self):
		return self.trajectory_states,self.trajectory_actions

	def get_TD_error(self,reward,state,next_state):
		return self.critic.get_TD_error(reward,state,next_state)
	def update_agent(self, state, next_state, reward):
		#print(state, next_state)
		TD_error = self.critic.get_TD_error(reward,state,next_state)
		self.critic.update_critic(self.trajectory_states, TD_error)
		#self.critic.update_critic([state], TD_error)
		self.actor.update_actor(self.trajectory_states,self.trajectory_actions,TD_error)

	def get_action(self,state,epsilon):
		action = self.actor.get_action(state, epsilon)
		return action

	def reset_memories(self):
		self.reset_trajectories()
		self.reset_e_traces()

	def print_values(self):
		self.critic.print_values()
	def print_q_values(self,state):
		self.actor.print_table(state)

	def update_illegal_move(self,state,action):
		self.actor.update_illegal_move(state,action)

	def save_policy(self,shape,size, method):
		self.actor.save_policy(shape,size,method)
