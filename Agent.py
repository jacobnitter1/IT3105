import numpy as np
import tensorflow as tf
import splitgd

class Actor():
	def __init__(self, obs_space, action_space,epsilon =.2, gamma=.95, alpha=.1, lamda=0.5, hidden_layers_dim=[8,8] ):
		self.epsilon = epsilon
		self.gamma = gamma
		self.alpha = alpha
		self.lamda = lamda
		self.policy_function = self.createNN(obs_space+action_space, hidden_layers_dim)
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



class Critic_tab():
	def __init__(self, obs_space,gamma, alpha, lamda):
		self.value_table = np.rand.random(2**(obs_space*obs_space))
		self.e_traces = np.zeros(shape(value_table))
		self.gamma= gamma
		self.alpha = alpha
		self.lamda = lamda

	def get_TD_error(self,reward,state,next_state):
		delta = reward + self.gamma*self.value_table[self.bin_state_to_state_num(next_state)] - self.value_table[self.bin_state_to_state_num(state)]
		return delta	

	def update_values(self, reward, state, next_state):
		delta = self.get_TD_error(reward,state, next_state)
		(self.value_table)[self.bin_state_to_state_num(state)] += self.alpha*delta*self.e_traces[self.bin_state_to_state_num(state)]

	def update_e_traces(self,state):
		for elem in self.e_traces:
			elem *=self.gamma*self.lamda
			if elem ==bin_state_to_state_num(state):
				elem = 1

	def bin_state_to_state_num(self, state):
		s = ''.join(str(elem) for elem in state)
		return int(s,2)



class Agent():
	def __init__(self,critic_type, NN_structure, obs_space, action_space , gamma, alpha, lamda):
		if critic_type == "NN":
			self.critic = Critic_NN(dim_NN, obs_space, action_space)
		else:
			self.critic = Critic_tab(obs_space, gamma, alpha, lamda)
		self.actor= Actor()
