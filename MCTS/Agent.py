import numpy as np
import random
class MCTS:

	def __init__(self, exploration_rate, game, rollout_game):
		self.root_node = Node(game.get_state(),game.get_last_player())
		self.exploration_rate = exploration_rate
		self.game = game
		self.rollout_game = rollout_game

	def default_policy(self,state): # ROLLOUT-POLICY
		actions = self.game.get_legal_actions(state)
		idx = np.random.randint(0,len(actions),1)
		return actions[idx]

	def choose_next_action(self, M, state, last_player):
		self.game.set_state(state,last_player)
		#print("first - ", self.game.get_state())
		self.root_node = Node(state,last_player)
		for i in range(0,M):
			self.rollout(self.root_node)

		#print("after rollout - ", self.game.get_state())
		Q_vals = self.root_node.get_childrens_Q_values()
		UCT_vals = self.root_node.get_childrens_UCT_values()
		if self.root_node.get_last_player() == 1:
			idx = np.argmax(Q_vals+np.multiply(self.exploration_rate,UCT_vals))
		else:
			idx = np.argmin(Q_vals-np.multiply(self.exploration_rate,UCT_vals))
		if self.game.get_last_player() == 1:
			player = 2
		else:
			player = 1
		actions = self.game.get_legal_actions()
		action=actions[idx]
		#self.game.make_move(player,action)
		#self.root_node = Node(self.game.get_state(),self.game.get_last_player())
		return action

	def node_expansion(self,child_node_state=None, idx=None):
		children = self.game.get_child_states()
		if len(self.root_node.get_children()) == 0:
			for i in range(0,len(children)):
				self.root_node.add_child(None,i)
#		else:

#			if self.game.get_last_player() == 1:
#				child_node = Node(child_node_state,0)
#			else:
#				child_node = Node(child_node_state,1)
#			self.root_node.add_child(child_node,idx)


	def rollout(self, root_node):
		children_states=self.game.get_child_states()
		node_children = root_node.get_children()
		chosen_node = None
		if len(node_children) == 0:
			self.node_expansion()
			node_children = self.root_node.get_children()
		for i in range(0,len(node_children)):
			if node_children[i] is None:
				chosen_node = i
				self.node_expansion(children_states[chosen_node],chosen_node)
				break
		if chosen_node is None:
			Q_vals = root_node.get_childrens_Q_values()
			UCT_vals = root_node.get_childrens_UCT_values()
			if root_node.get_last_player() == 1:
				idx = np.argmax(Q_vals+np.multiply(self.exploration_rate,UCT_vals))
				chosen_node = idx
			else:
				idx = np.argmin(Q_vals-np.multiply(self.exploration_rate,UCT_vals))
				chosen_node = idx
		child_state = children_states[chosen_node]#[0] #root_node.get_state()
		last_player = self.rollout_game.get_last_player()
		if last_player == 1:
			self.rollout_game.set_state(child_state,2)
		else:
			self.rollout_game.set_state(child_state,1)
		result = self.rollout_step()
		root_node.update_stats(result)
		root_node.update_child_stats(result,chosen_node,child_state,root_node.get_last_player())

	def rollout_step(self):
		children = self.rollout_game.get_child_states()
		if len(children) == 0:
			winner = self.rollout_game.get_winner()
			if winner == 1:
				return 1
			elif winner == 2:
				return 0
			else:
				return 0
		else:
			next_state = random.sample(children,1)#[0]
			if self.rollout_game.get_last_player() == 1:
				self.rollout_game.set_state(next_state[0],2)
			else:
				self.rollout_game.set_state(next_state[0],1)
			return self.rollout_step()



class Node():
	def __init__(self, state, player_took_last_turn):
		self.state = state
		self.num_wins = 0
		self.num_played = 0
		self.children = []
		self.player_took_last_turn = player_took_last_turn
		self.leaf_node = False

	def add_child(self, node,idx):
		if node is None:
			self.children.append(node)
		else:
			print("idx ",idx," node ", node)
			self.children[idx] = node

	def get_children(self):
		return self.children

	def update_stats(self, result):
		if result == 1:
			self.num_wins += 1
		self.num_played += 1
	def update_child_stats(self,result, child_idx,child_state,parent_last_player):
		if self.children[child_idx] == None:
			if parent_last_player == 1:
				self.children[child_idx] =  Node(child_state,2)
			else:
				self.children[child_idx] = Node(child_state,1)
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
				UCT_values.append(None)
			else:
				UCT_values.append(np.sqrt(np.log(self.num_played)/(1+node.get_num_played())))
		return UCT_values
	def get_last_player(self):
		return self.player_took_last_turn

	def get_childrens_Q_values(self):
		Q_values=[]
		for node in self.children:
			if node is None:
				Q_values.append(None)
			else:
				Q_values.append(node.get_num_wins()/node.get_num_played())
		return Q_values
