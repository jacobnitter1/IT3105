import numpy as np
import tensorflow as tf
#import splitgd


class Critic_NN():
	def __init__(self, hidden_layers_dim, obs_space, action_space):
		self.value_function = self.createNN(obs_space,hidden_layers_dim)
		self.q_function = self.createNN(obs_space+action_space, hidden_layers_dim)
		self.weights_val = (self.value_function).kernel
		self.e_traces = np.zeros(shape(weights_val))
		#??? Hvilken størrelse skal eligibility tracesene ha?????


	def createNN(input_dim, hidden_layers_dim, output_dim =1,activation_function = tf.nn.relu):
		model = tf.keras.models.Sequential()
		model.add(Dense(hidden_layers_dim[0], input_shape=(input_shape,)))
		for i in range(1,len(hidden_layers_dim)):
			if len(activation_function) == len(hidden_layers_dim):
				model.add(Dense(hidden_layers_dim[i]), activation = activation_function)
			else:
				model.add(Dense(hidden_layers_dim[i], activation = activation_function))
		model.add(Dense(output_dim), activation = tf.nn.sigmoid)
		return model

class Actor_tab():
	def __init__(self, obs_space, action_space, epsilon, learning_rate, e_discount):
		self.table = np.zeros((2**((obs_space)),(obs_space),(action_space)))
		for i in range(0, 2**obs_space):
			for j in range(0, obs_space):
				for k in range(0,action_space):
					self.table[i,j,k] += np.random.rand()/4
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
			print(state,num, len(state))
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
			return -1
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
	def __init__(self,critic_type, NN_structure, obs_space, action_space , gamma, alpha, lamda):
		if critic_type == "NN":
			self.critic = Critic_NN(dim_NN, obs_space, action_space)
		else:
			self.critic = Critic_tab(obs_space, gamma, alpha, lamda)
		self.actor= Actor_tab(obs_space = obs_space, action_space=action_space, epsilon=0.3, learning_rate=alpha, e_discount=lamda)
		self.trajectory_states = []
		self.trajectory_actions= []
		self.obs_space = obs_space
		self.action_space = action_space
		self.num_memorized = 0

		self.replay_trajectories_states = []
		self.replay_trajectories_actions = []

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
		self.trajectory_actions.append(action)

		#print(self.trajectory_states," and ", self.trajectory_actions)
		self.actor.update_e_traces(state,action)
		self.critic.update_e_traces(state)
		self.num_memorized +=1

	def get_trajectories(self):
		return self.trajectory_states,self.trajectory_actions

	def update_agent(self, state, next_state, reward):
		TD_error = self.critic.get_TD_error(reward,state,next_state)
		self.critic.update_critic(self.trajectory_states, TD_error)
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
