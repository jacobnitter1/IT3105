import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras.activations import relu,tanh,linear,sigmoid
import math
import scipy
#import splitgd
DEBUGGING_VAL = False
class MCTS:

	def __init__(self, exploration_rate, game, rollout_game):
		self.root_node = Node(game.get_NN_state(),game.get_last_player(), None)
		self.exploration_rate = exploration_rate
		self.game = game
		self.rollout_game = rollout_game

		#self.previously_visited_nodes = []
	def get_distribution(self):
		Q_values = np.zeros(len(self.game.get_boardState())*len(self.game.get_boardState()))
		children_nodes = self.root_node.get_children()
		#print(children_nodes)
		Q = self.root_node.get_childrens_Q_values()
		for i in range(0,len(children_nodes)):
			if children_nodes[i] != None:
				a = self.rollout_game.action_from_s1_to_s2(self.root_node.get_state(), children_nodes[i].get_state())

				#print("!!!",self.root_node.get_state(), children_nodes[i].get_state(),a)
				Q_values[a]=Q[i]

		#print("hehehehehehe",Q_values)
		#print("hehehehehehe2",softmax(Q_values))
		return softmax(Q_values)


	def set_exploration_rate(self,e):
		self.exploration_rate = e
	def default_policy(self): # ROLLOUT-POLICY
		children_states = self.rollout_game.get_child_states()
		if len(children_states) == 1:
			return children_states[0]
		else:
			idx = np.random.randint(0,len(children_states),1)[0]
		return children_states[idx]

	def rollout_policy(self,model):
		actions = model.predict([[self.rollout_game.get_NN_state()]])
		#print("1 - Action distribution : ", actions)
		legal_actions = self.rollout_game.get_legal_actions()
		#print(self.rollout_game.get_NN_state(), self.rollout_game.get_legal_actions())
		for i in range(0, len(actions[0])):
			#print( i , " in legal actions = ",legal_actions," ? ", i in legal_actions)
			if i not in legal_actions:
				#print(i, " not in ", legal_actions)
				actions[0][i] = 0
		#print("2 - Action distribution : ", actions)
		actions = softmax(actions)
		#print("run get child states")

		#print("3 - Action distribution : ", actions)
		children_states = self.rollout_game.get_padded_child_states()
		idx = np.argmax(actions[0])
		#print(" -",actions, len(actions[0]))
		#print("-",children_states,len(children_states))
		return children_states[idx]


	def tree_policy(self,root_node):
		Q_vals = root_node.get_childrens_Q_values()
		UCT_vals = root_node.get_childrens_UCT_values()
		#print("Root node has been played ", root_node.get_num_played())
		if root_node.get_last_player() == 2:
			if DEBUGGING_VAL :
				print("P1 max : ", Q_vals)#,np.multiply(self.exploration_rate,UCT_vals), Q_vals+np.multiply(self.exploration_rate,UCT_vals))
			idx = np.argmax(Q_vals+np.multiply(self.exploration_rate,UCT_vals))
		else:

			if DEBUGGING_VAL :
				print("P2 min : ", Q_vals, np.multiply(self.exploration_rate,UCT_vals),Q_vals-np.multiply(self.exploration_rate,UCT_vals))
			idx = np.argmin(Q_vals-np.multiply(self.exploration_rate,UCT_vals))

		return idx

	def get_target_values(self,root_node,action_space):
		Q_vals = root_node.get_childrens_Q_values()
		children = root_node.get_child_states()
		target_values = np.zeros(action_space)
		p_s = root_node.get_state()
		for i in range(0,len(children)):
			c_s = children[i]
			action = self.rollout_game.action_from_s1_to_s2(p_s,c_s)
			target_values[action] = Q_vals[i]

		return softmax(target_values)

	def tree_search(self):
		last_node = self.root_node
		current_node = self.root_node
		#print("current node states : ",current_node.get_state())
		self.rollout_game.set_state(current_node.get_state(), current_node.get_last_player())
		#print("???")
		#self.rollout_game.print_state()
		current_node.get_children()
		#self.rollout_game.print_state()
		#print("+???")
		if len(current_node.get_children()) == 0:
			print("Node expansion happening")
			self.node_expansion(current_node)
		next_node_idx = None
		#print("!!!")
		#self.rollout_game.print_state()
		d,l_p = self.rollout_game.is_game_done()
		if d:
			if DEBUGGING_VAL:
				print("CHecking before starting ", current_node.get_state(), self.rollout_game.get_state())
			return current_node.get_state(), None, current_node
		while len(current_node.get_children()) != 0 :

			if DEBUGGING_VAL :
				print("Tree search : ", current_node.get_state())

			last_node = current_node
			next_node_idx = self.tree_policy(current_node)
			current_node = current_node.get_children()[next_node_idx]

			if current_node == None:
				self.rollout_game.set_state(last_node.get_state(),last_node.get_last_player())
				children_states = self.rollout_game.get_child_states()
				if last_node.get_last_player() == 1:
					last_node.add_child(Node(children_states[next_node_idx],2,last_node), next_node_idx)
				else:
					last_node.add_child(Node(children_states[next_node_idx],1,last_node), next_node_idx)
				current_node = last_node.get_children()[next_node_idx]
				self.node_expansion(current_node)
				#print("<------------Tree search : ", current_node.get_state(), " breaking because currend node was None")
				break
			elif current_node.get_num_played() == 0:
				self.rollout_game.set_state(last_node.get_state(),last_node.get_last_player())
				children_states = self.rollout_game.get_child_states()
				if last_node.get_last_player() == 1:
					last_node.add_child(Node(children_states[next_node_idx],2,last_node), next_node_idx)
				else:
					last_node.add_child(Node(children_states[next_node_idx],1,last_node), next_node_idx)
				current_node = last_node.get_children()[next_node_idx]
				self.node_expansion(current_node)
				#print("------------->Tree search : ", current_node.get_state(), "breaking because current node has never been played before")
				break

			self.rollout_game.set_state(current_node.get_state(), current_node.get_last_player())
			d,l_p = self.rollout_game.is_game_done()
			if DEBUGGING_VAL:
				print("CHecking in while loop:",d, current_node.get_state(), self.rollout_game.get_state())
			if d:

				return current_node.get_state(), None, current_node
		#self.node_expansion(current_node)
		if DEBUGGING_VAL :
			print(current_node.get_state(), " is leaf node and has been played ", current_node.get_num_played(), " times.")
		return current_node.get_state(), next_node_idx, current_node


	def leaf_evaluation(self,leaf_node, last_player, model):
		num_wins = 0
		num_plays = 1
		for i in range(0,num_plays):
			num_wins += self.rollout(leaf_node,last_player,model)
		#print("ROLLOUT FINISHED")
		return num_wins, num_plays

	def run_simulation(self,model):
		#print("NEW SIMULATION")
		leaf_node_state, idx,leaf_node = self.tree_search()
		if leaf_node.get_last_player() == 1:
			result, num_plays = self.leaf_evaluation(leaf_node_state, 2,model)
		else:
			result, num_plays = self.leaf_evaluation(leaf_node_state, 1,model)
		if leaf_node == None: #state, player_took_last_turn,parent
			if self.root_node.get_last_player() == 1:
				self.root_node.add_child(Node(leaf_node_state,2,self.root_node), idx)
			else:
				self.root_node.add_child(Node(leaf_node_state,1,self.root_node), idx)
			leaf_node = self.root_node.get_child(idx)
		#print("Current leaf node is ", leaf_node_state,"with last player from leaf node ", leaf_node.get_last_player()," with result ", result)
		self.backprop(result,num_plays,leaf_node)
		#print(leaf_node.get_state(), " is leaf node and has been played ", leaf_node.get_num_played(), " times.")


	def choose_greedy_action(self):
		#print("Choose greedy action: ",self.root_node.get_state(), self.root_node.get_last_player())
		Q_vals = self.root_node.get_childrens_Q_values()
		UCT_vals = self.root_node.get_childrens_UCT_values()
		#print(Q_vals)
		if self.root_node.get_last_player() == 2:
			idx = np.argmax(Q_vals)#+np.multiply(self.exploration_rate,UCT_vals))
		else:
			idx = np.argmin(Q_vals)#-np.multiply(self.exploration_rate,UCT_vals))
		if self.game.get_last_player() == 1:
			player = 2
		else:
			player = 1
		actions = self.game.get_legal_actions()
		action=actions[idx]
		#print("Choose between actions : ", actions," with q vals : ",Q_vals)
		#self.game.make_move(player,action)
		#self.root_node = Node(self.game.get_state(),self.game.get_last_player())
		return action

	def get_action(self, M, state, last_player):
		self.game.set_state(state,last_player)
		self.root_node = Node(state,last_player,None)
		#self.previously_visited_nodes.append(self.root_node)
		for i in range(0,M):
			#if i < M/4 :
			#	self.exploration_rate = 4
			#elif i < M/2:
			#	self.exploration_rate = 2
			#elif i < M/1.5:
			#	self.exploration_rate = 1
			#else:
			#	self.exploration_rate = 0.8
			self.rollout_game.set_state(state,last_player)
			#print(" SIM #",i,state," == ",self.rollout_game.get_state())
			#print("NEW SIMULATION ", self.rollout_game.get_state())
			self.run_simulation()
		#print("Root node info - State : ",self.root_node.get_state(), " and last player : ",self.root_node.get_last_player())
		if DEBUGGING_VAL :

			print(self.game.get_legal_actions())
			print(self.root_node.get_state()," has the following Q stuffs :")
		#self.root_node.print_Q_properties()
		action = self.choose_greedy_action()
		return action

	def run_simulations(self, M, state, last_player, rollout_policy):
		self.game.set_state(state,last_player)
		self.root_node = Node(state,last_player,None)
		#self.previously_visited_nodes.append(self.root_node)
		for i in range(0,M):
			self.rollout_game.set_state(state,last_player)
			self.run_simulation(rollout_policy)
		#print("Root node info - State : ",self.root_node.get_state(), " and last player : ",self.root_node.get_last_player())
		if DEBUGGING_VAL :

			print(self.game.get_legal_actions())
			print(self.root_node.get_state()," has the following Q stuffs :")
		#self.root_node.print_Q_properties()
		action = self.choose_greedy_action()
		return action

	def node_expansion(self,parent_node,child_node_state=None, idx=None):
		#print(" mhmhmhmh : ", parent_node.get_state())
		self.rollout_game.set_state(parent_node.get_state(),parent_node.get_last_player())#, parent_node.get_last_player())
		#self.rollout_game.print_state()
		#print("mmhmhmhmmhmhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
		#self.rollout_game.print_state()
		#print("mmhmhmhmmhmhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh2")
		children = self.rollout_game.get_child_states()
		#self.rollout_game.print_state()
		#print("mmhmhmhmmhmhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh3")
		if len(parent_node.get_children()) == 0:
			for i in range(0,len(children)):
				parent_node.add_child(None,i)
		else:
			for i in range(0,len(children)):
				if children[i] == child_node_state:
					if parent_node.get_last_player() == 1:
						parent_node.add_child(Node(child_node_state,2,parent_node))
					else:
						parent_node.add_child(Node(child_node_state,1,parent_node))

	def rollout(self,root_node,last_player, model):
		#print(" root node : ", root_node)
		if last_player == 1:
			self.rollout_game.set_state(root_node,2)
		else:
			self.rollout_game.set_state(root_node,1)
		return self.rollout_step(model)

	def __backprop(self,result, root_node, child_node_idx, child_state):
		#for node in self.previously_visited_nodes:
		#	node.update_stats(result)
		#print("In backprop ",root_node, type(root_node), self.root_node.get_state())
		#print("Backpropped ", result, " to state ", child_state, " last player : ",root_node.get_last_player())
		root_node.update_stats(result)
		root_node.update_child_stats(result,child_node_idx,child_state,root_node.get_last_player())

	def backprop(self,result,num_plays,leaf_node):

		leaf_node.update_stats(result,num_plays)

		if DEBUGGING_VAL :
			print("This is the leafe node ",leaf_node.get_state(), " with result : ", result)
		node = leaf_node.get_parent()

		parent_line = []
		parent_line.append(leaf_node.get_state())
		while node is not None:
			node.update_stats(result,num_plays)
			if DEBUGGING_VAL :
				print("This is the parent node ",node.get_state(), node is None)
			parent_line.append(node.get_state())
			node = node.get_parent()
			if DEBUGGING_VAL :
				print(node)

		if DEBUGGING_VAL :
			print("BACKPROPPING ", parent_line)
	def rollout_step(self, model):
		#print("Rollout step, last player ", self.rollout_game.get_last_player())
		#self.rollout_game.print_state()
		children = self.rollout_game.get_child_states()
		if self.rollout_game.is_game_done()[0]:#print("End of Rollout : ",self.rollout_game.get_state(), self.rollout_game.get_winner(), self.rollout_game.is_game_done())
			winner = self.rollout_game.get_winner()
			#print(winner, " won!!", self.rollout_game.is_game_done())
			#self.rollout_game.print_state()
			#print("Rollout game is done : ", self.rollout_game.get_state(), " last player was : ", self.rollout_game.get_last_player(), self.rollout_game.get_winner())
			if winner == 1:
				return 1
			elif winner == 2:
				return 0
		else:
			last_state = self.rollout_game.get_state()
			#self.rollout_game.print_state()

			next_state = self.rollout_policy(model)#self.default_policy()#self.rollout_policy()#[0]
			#print("Last player befor emove : ", self.rollout_game.get_last_player())
			if self.rollout_game.get_last_player() == 1:
				self.rollout_game.set_state(next_state,2)
				#print("Last player : 2  == ", self.rollout_game.get_last_player())
			else:
				self.rollout_game.set_state(next_state,1)
				#print("last player : 1 == ",  self.rollout_game.get_last_player())
			if DEBUGGING_VAL :
				print("Rollout step from : ",last_state," to : ",next_state," with last player ", self.rollout_game.get_last_player())
			#self.rollout_game.print_state()
			return self.rollout_step(model)


class Node():
	def __init__(self, state, player_took_last_turn,parent):
		self.state = state
		self.num_wins = 0
		self.num_played = 0
		self.children = []
		self.player_took_last_turn = player_took_last_turn
		self.leaf_node = False
		self.parent = parent

	def add_child(self, node,idx):
		if node is None:
			self.children.append(node)
		else:
			#print("Add child : ","idx ",idx," node ", node)
			self.children[idx] = node

	def get_children(self):
		return self.children

	def update_stats(self, result,num_plays = 1):
		self.num_wins += result
		self.num_played += num_plays
	def update_child_stats(self,result, child_idx,child_state,parent_last_player):
		if self.children[child_idx] == None:
			if parent_last_player == 1:
				self.children[child_idx] =  Node(child_state,2,self)
			else:
				self.children[child_idx] = Node(child_state,1,self)
		self.children[child_idx].update_stats(result)


	def get_state(self):
		return self.state

	def set_is_leaf_node(self, bool):
		self.leaf_node = bool

	def get_num_played(self):
		return self.num_played

	def get_num_wins(self):
		return self.num_wins

	def get_childrens_UCT_values(self):
		UCT_values=[]
		if self.num_played == 0:
			#print("!!!!!!!!!!!!!!!!!!!!!!!!!!!1Choosing between children when node itself is not visited?")
			return 0
		else:
			for node in self.children:
				if node is None:
					UCT_values.append(0.5+np.sqrt(np.log(self.num_played)))
				else:
					UCT_values.append(np.sqrt(np.log(self.num_played)/(1+node.get_num_played())))
		return UCT_values
	def get_last_player(self):
		return self.player_took_last_turn

	def get_childrens_Q_values(self):
		Q_values=[]
		for node in self.children:
			if node is None:
				Q_values.append(0)
			else:
				Q_values.append(node.get_num_wins()/node.get_num_played())
				#print(node.get_state()," state with ", node.get_last_player()," as last player leads to " ,node.get_num_wins()," wins out of ", node.get_num_played()," times = ",node.get_num_wins()/node.get_num_played())
		return Q_values

	def get_parent(self):
		return self.parent
	def get_child(self,idx):
		return self.children[idx]
	def print_Q_properties(self):
		str=[]
		str2=[]
		for node in self.children:
			if node is None:
				str.append(0)
				str2.append([None,None])
			else:
				str.append(node.get_num_wins()/node.get_num_played())
				str2.append([node.get_num_wins(),node.get_num_played()])
		print("Q Values : ",str)
		print("Node statistics : ",str2)

class replay_buffer():
	def __init__(self, max_size):
		self.states = []
		self.action_distributions = []
		self.max_size = max_size
		self.is_full = False
		self.current_size = 0

	def get_minibatch(self, size):
		if size >self.max_size or (not self.is_full and size > self.current_size):
			print("Requesting to big minibatch!")
		indices = random.sample(range(0,len(self.states)),size)
		return_States = []
		return_Ds = []
		for i in range(0,len(indices)):
			return_States.append(self.states[indices[i]])
			return_Ds.append(self.action_distributions[indices[i]])
		#print(indices, type(indices), type(indices[0]))
		#return self.states[indices], self.action_distributions[indices]
		#print(return_States)
		#print(np.shape(self.action_distributions)," ---> ", np.shape(return_Ds))
		#print(np.shape(self.states)," ---> ", np.shape(return_States))
		return np.array(return_States),np.array(return_Ds)

	def print_RBUF(self):
		if self.is_full:
			I = self.max_size
		else:
			I = self.current_size

		for i in range(0,I):
			print("(State, D): ", self.states[i], self.action_distributions[i])

	def save_experience(self, state, D):
		if self.is_full:
			self.states[self.current_size] = state
			self.action_distributions[self.current_size] = D
		else:
			self.states.append(state)
			self.action_distributions.append(D)
		if self.current_size == self.max_size:
			self.is_full = True
			self.current_size=0
		else:
			self.current_size +=1

	def reset_buffer(self):
		self.states = []
		self.action_distributions=[]
		self.current_size = 0



def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	sum = x.sum()
	if sum == 0:
		sum=1.0

	return x/sum

class Policy_Network():

	def __init__(self, boardSize,lr = 0.001, nn_struct = [100,100]):
		self.boardSize = boardSize
		self.output_dim = boardSize*boardSize
		self.input_dim = boardSize*boardSize*2+2
		self.nn_struct = nn_struct
		self.lr = lr
		self.model = self.createNN()

	def createNN(self,activation_function = 'sigmoid'):
		model = Sequential()
		nn_struct = self.nn_struct
		num_layers = len(nn_struct)
		model.add(Dense(nn_struct[0],input_dim=self.input_dim))
		for i in range(1,num_layers):
			model.add(Dense(nn_struct[i], activation = activation_function))
		model.add(Dense(self.output_dim, activation = 'softmax'))
		sgd = SGD(learning_rate = self.lr,momentum = 0.0, nesterov=False)
		model.compile(loss='mse',optimizer=sgd)
		print(model.summary())
		return model

	def predict(self,state):
		return self.model.predict(state)

	def get_action(self,state, legal_actions,epsilon):
		if random.random() < epsilon:
			a= random.sample(legal_actions,1)
			#print("random ", a)
			return a[0]
		else:
			actions= self.model.predict([[state]])
			for i in range(0, len(actions[0])):
				#print( i , " in legal actions = ",legal_actions," ? ", i in legal_actions)
				if i not in legal_actions:
					#print(i, " not in ", legal_actions)
					actions[0][i] = 0
			#print("2 - Action distribution : ", actions)
			actions = softmax(actions)
			return np.argmax(actions)

	def get_distribution_and_action(self, state, legal_actions):
		actions= self.model.predict([[state]])
		for i in range(0, len(actions[0])):
			#print( i , " in legal actions = ",legal_actions," ? ", i in legal_actions)
			if i not in legal_actions:
				#print(i, " not in ", legal_actions)
				actions[0][i] = 0
		#print("2 - Action distribution : ", actions)
		actions = softmax(actions)
		return actions, np.argmax(actions)

	def save_weights(self, path):
		self.model.save_weights(path)

	def load_weights(self,path):
		self.model.load_weights(path)

	def train(self, state_batch, target_batch):
		self.model.fit(state_batch,target_batch)
