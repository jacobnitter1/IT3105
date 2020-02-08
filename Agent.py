import numpy as np
import tensorflow as tf
#import splitgd

class Actor():
	def __init__(self, obs_space, action_space,epsilon =.2, gamma=.95, alpha=.1, lamda=0.5, hidden_layers_dim=[8,8] ):
		self.epsilon = epsilon
		self.gamma = gamma
		self.alpha = alpha
		self.lamda = lamda
		#self.policy_function = self.createNN(obs_space+action_space, hidden_layers_dim)
		self.e_traces = np.zeros(2**(obs_space+action_space))
		self.obs_space = obs_space
		self.action_space = action_space
		# ???? Størrelse ???? self.e_traces=np.zeros

	def get_next_action(self, bin_obs):
		actions = np.zeros(self.action_space)
		r = np.random.rand()
		if epsilon >= r:
			best_a = actions
			best_val =-np.inf
			for a in range (0,len(actions)):
				bin_input = bin_obs
				actions = 0
				actions[a]=1
				for ac in actions:
					bin_input.append(ac)
				val = runNN(bin_input)
				if val >= best_val:
					best_a = actions
		else:
			actions = 0
			action[np.random.randint(0,len(actions))] = 1
			best_val = None

		return actions, best_val


	def runNN(self,bin_sap):
		return self.policy_function(bin_sap)


	def createNN(input_dim, hidden_layers_dim, output_dim,activation_function = tf.nn.relu):
		model = tf.keras.models.Sequential()
		model.add(Dense(hidden_layers_dim[0], input_shape=(input_shape,)))
		for i in range(1,len(hidden_layers_dim)):
			if len(activation_function) == len(hidden_layers_dim):
				model.add(Dense(hidden_layers_dim[i]), activation = activation_function)
			else:
				model.add(Dense(hidden_layers_dim[i], activation = activation_function))
		model.add(Dense(output_dim, activation = tf.nn.sigmoid))
		return model


	def binary_to_int_list(self,state,action):
		a_b = np.binary_repr(action)
		a_l = [int(a) for a in a_b]
		bin_sap = state
		for i in a_l:
			bin_sap.append(i)
		return bin_sap

	def update_policy(self):
		pass

	def calc_loss(self, state, action, state_):
		pass


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
		self.table = np.zeros((2**(len(obs_space)),len(action_space)))
		self.num_entries_obs = 2**(len(obs_space)*len(obs_space))
		self.num_entries_act=2**len(action_space)
		self.e_traces = np.zeros((2**(len(obs_space)), len(action_space)))
		self.action_space = action_space
		self.obs_space = obs_space

		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.e_discount = e_discount

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
		if np.random.rand()<epsilon:
			action = np.random.sample(range(0,len(self.action_space)))
		else:
			predictions=np.zeros(len(self.action_space))
			for i in range(0,len(self.action_space)):
				predictions[i]=self.table[self.bin_state_to_state_num(state)][i]
			action= np.argmax(predictions)
		return action

	def update_e_traces(self,state,action):
		state =self.bin_state_to_state_num(states[i])
		self.e_traces[state][action] = self.e_discount*self.e_traces[state][action]
		self.e_traces[s]=1

	def reset_e_traces(self):
		self.e_traces = 0

	def update_actor(self, states, actions, TD_error):
		for i in range(0, len(states)):
			state =self.bin_state_to_state_num(states[i])
			action = actions[i]
			self.table[state][action] = self.table[state][action] + self.learning_rate*TD_error*self.e_traces[state][action]





class Critic_tab():
	def __init__(self, obs_space,gamma, alpha, lamda):
		self.value_table = np.rand.random(2**len(obs_space))
		self.e_traces = np.zeros(shape(value_table))
		self.gamma= gamma
		self.alpha = alpha
		self.lamda = lamda
		self.e_discount = e_discount

	def get_TD_error(self,reward,state,next_state):
		delta = reward + self.gamma*self.value_table[self.bin_state_to_state_num(next_state)] - self.value_table[self.bin_state_to_state_num(state)]
		return delta

	def update_critic(self, states,TD_error):
		for i in range(0,len(states)):
			state = self.bin_state_to_state_num(states[i])
			self.value_table[state] += self.alpha*TD_error*self.e_traces[state]

	def update_e_traces(self,state):
		self.e_traces = self.e_discount*self.e_traces
		self.e_traces[self.bin_state_to_state_num(state)] = 1

	def reset_e_traces(self):
		self.e_traces = 0

	def bin_state_to_state_num(self, state):
		s = ''.join(str(int(elem)) for elem in state)
		return int(s,2)




class Agent():
	def __init__(self,critic_type, NN_structure, obs_space, action_space , gamma, alpha, lamda):
		if critic_type == "NN":
			self.critic = Critic_NN(dim_NN, obs_space, action_space)
		else:
			self.critic = Critic_tab(obs_space, gamma, alpha, lamda)
		self.actor= Actor_tab()
		self.trajectory_states = []
		self.trajetory_actions= []

	def reset_trajectories(self):
		self.trajectory_states = []
		self.trajetory_actions= []
	def reset_e_traces(self):
		actor.reset_e_traces()
		critic.reset_e_traces()

	def update_trajectories(self,state,action):
		self.trajectory_states.append(state)
		self.trajectory_actions.append(action)
		actor.update_e_traces(state,action)
		critic.update_e_traces(state)

	def update_agent(self, TD_error):
		critic.update_critic(self.trajectory_states, TD_error)
		actor.update_actor(self.trajectory_states,self.trajetory_actions,TD_error)

	def get_action(self,state):
		action = actor.get_action(state)
		return action

	def reset_memories(self):
		self.reset_trajectories()
		self.reset_e_traces()
