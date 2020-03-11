import numpy as np
import random
class MCTS:

	def __init__(self, exploration_rate, game, rollout_game):
		self.root_node = Node(game.get_state(),game.get_last_player(), None)
		self.exploration_rate = exploration_rate
		self.game = game
		self.rollout_game = rollout_game
		#self.previously_visited_nodes = []

	def default_policy(self): # ROLLOUT-POLICY
		children_states = self.rollout_game.get_child_states()
		if len(children_states) == 1:
			return children_states[0]
		else:
			idx = np.random.randint(0,len(children_states),1)[0]
		return children_states[idx]

	def rollout_policy(self):
		children_states = self.rollout_game.get_child_states()
		idx= 0
		for i in range(0,len(children_states)):
			if self.rollout_game.is_state_end_state(children_states[i]):
				#print("----> ", children_states[i]," is an end state!")
				return children_states[i]
		any_good_actions = False
		good_action_idx = []
		for i in range(0,len(children_states)):
			if not self.rollout_game.is_next_state_too_stupid(children_states[i]):
				any_good_actions = True
				good_action_idx.append(i)
		if any_good_actions:
			idx = np.random.randint(0,len(good_action_idx),1)[0]
		else:
			idx = np.random.randint(0,len(children_states),1)[0]
		return children_states[idx]




	def tree_policy(self,root_node):
		Q_vals = root_node.get_childrens_Q_values()
		UCT_vals = root_node.get_childrens_UCT_values()
		if root_node.get_last_player() == 2:
			idx = np.argmax(Q_vals+np.multiply(self.exploration_rate,UCT_vals))
		else:
			idx = np.argmin(Q_vals-np.multiply(self.exploration_rate,UCT_vals))
		return idx

	def tree_search(self):
		node_children = self.root_node.get_children()
		children_states=self.game.get_child_states()
		if len(node_children) == 0:
			self.node_expansion()
			node_children = self.root_node.get_children()
		for i in range(0,len(node_children)):
			if node_children[i] is None:
				chosen_node = i
				#self.node_expansion(children_states[chosen_node],chosen_node)
				#print("TREE POLICY -> ",children_states[chosen_node])
				return children_states[chosen_node],chosen_node,self.root_node.get_child(chosen_node)
		idx = self.tree_policy(self.root_node)
		leaf_node_state=children_states[idx]
		leaf_node=self.root_node.get_child(idx)
		return leaf_node_state, idx,leaf_node

	def leaf_evaluation(self,leaf_node):
		#print("In leaf_eval " ,leaf_node, type(leaf_node))
		return self.rollout(leaf_node)

	def run_simulation(self):
		leaf_node_state, idx,leaf_node = self.tree_search()
		result = self.leaf_evaluation(leaf_node_state)
		#print("Result to be backpropped : ", result)
		#print(result)
		if leaf_node == None: #state, player_took_last_turn,parent
			if self.root_node.get_last_player() == 1:
				self.root_node.add_child(Node(leaf_node_state,2,self.root_node), idx)
			else:
				self.root_node.add_child(Node(leaf_node_state,1,self.root_node), idx)
			leaf_node = self.root_node.get_child(idx)
		self.backprop(result,leaf_node)

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
			self.rollout_game.set_state(state,last_player)
			#print(" SIM #",i,state," == ",self.rollout_game.get_state())
			#print("NEW SIMULATION ", self.rollout_game.get_state())
			self.run_simulation()
		print(self.game.get_legal_actions())
		self.root_node.print_Q_properties()
		action = self.choose_greedy_action()
		return action


	def node_expansion(self,child_node_state=None, idx=None):
		children = self.game.get_child_states()
		if len(self.root_node.get_children()) == 0:
			for i in range(0,len(children)):
				self.root_node.add_child(None,i)
		else:
			for i in range(0,len(children)):
				if children[i] == child_node_state:
					if self.root_node.get_last_player() == 1:
						self.root_node.add_child(Node(child_node_state,2,self.root_node))
					else:
						self.root_node.add_child(Node(child_node_state,1,self.root_node))

	def rollout(self,root_node):
		#print("NEW ROLLOUT - ", root_node)
		last_player = self.rollout_game.get_last_player()
		if last_player == 1:
			self.rollout_game.set_state(root_node,2)
		else:
			self.rollout_game.set_state(root_node,1)
		return self.rollout_step()

	def __backprop(self,result, root_node, child_node_idx, child_state):
		#for node in self.previously_visited_nodes:
		#	node.update_stats(result)
		#print("In backprop ",root_node, type(root_node), self.root_node.get_state())
		#print("Backpropped ", result, " to state ", child_state, " last player : ",root_node.get_last_player())
		root_node.update_stats(result)
		root_node.update_child_stats(result,child_node_idx,child_state,root_node.get_last_player())

	def backprop(self,result,leaf_node):

		leaf_node.update_stats(result)
		node = leaf_node.get_parent()
		if node is not None:
			node.update_stats(result)
			node = node.get_parent()

	def rollout_step(self):
		children = self.rollout_game.get_child_states()
		#print("Rollout step : ", self.rollout_game.get_state(), self.rollout_game.get_last_player())
		#print("Rollout : ", self.rollout_game.get_state(), self.rollout_game.is_game_done())
		#print(self.rollout_game.is_game_done(), self.rollout_game.get_state())
		if self.rollout_game.is_game_done()[0]:#len(children) == 0:
			#print("End of Rollout : ",self.rollout_game.get_state(), self.rollout_game.get_winner(), self.rollout_game.is_game_done())
			winner = self.rollout_game.get_winner()
			if winner == 1:
				return 1
			elif winner == 2:
				return 0
		else:
			#print("Rollout step : ", self.rollout_game.get_state())
			next_state = self.rollout_policy()#self.default_policy()#self.rollout_policy()#[0]
			if self.rollout_game.get_last_player() == 1:
				self.rollout_game.set_state(next_state,2)
			else:
				self.rollout_game.set_state(next_state,1)
			#print(next_state)
			return self.rollout_step()



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

	def update_stats(self, result):
		if result == 1:
			self.num_wins += 1
		else:
			self.num_wins -=0
		self.num_played += 1
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
		for node in self.children:
			if node is None:
				UCT_values.append(np.sqrt(2.0*np.log(self.num_played)))
			else:
				UCT_values.append(np.sqrt(2.0*np.log(self.num_played)/(1+node.get_num_played())))
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
