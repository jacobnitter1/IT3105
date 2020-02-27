
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

	def choose_next_action(self, M):
		for i in range(0,M):
			self.rollout(self.root_node)
		if root_node.get_last_player() == 1:
			idx = np.argmax(Q_vals+np.multiply(self.exploration_rate,UCT_vals))
		else:
			idx = np.argmin(Q_vals-np.multiply(self.exploration_rate,UCT_vals))
		action = game.get_legal_actions()

	def node_expansion(self, root_node,child_node_state,game):
		children = self.game.get_child_states()
		if len(root_node.get_children()) == 0:
			for i in range(0,len(children)):
				root_node.add_child(None,i)
		else:
			idx = children.index(child_node_state)
			child_node = Node(child_node_state)
			root_node.add_child(child_node, idx)


	def rollout(self, root_node):
		Q_vals = root_node.get_childrens_Q_values()
		UCT_vals = root_node.get_childrens_UCT_values()
		node_children = root_node.get_children()
		chosen_node = None
		for i in range(0,len(node_children)):
			if node_children[i] == None:
				chosen_node = i
				break
		if chosen_node == None:
			if root_node.get_last_player() == 1:
				idx = np.argmax(Q_vals+np.multiply(self.exploration_rate,UCT_vals))
				chosen_node = idx
			else:
				idx = np.argmin(Q_vals-np.multiply(self.exploration_rate,UCT_vals))
				chosen_node = idx

		child_state = game.get_child_states(root_node.get_state())[chosen_node]
		self.rollout_game.set_state(child_state)
		result = self.rollout_step(self.rollout_game)
		root_node.update_stats(result)
		root_node.update_child_stats(result,chosen_node)

	def rollout_step(self,game):
		children = game.get_children(current_state)
		if len(children) == 0:
			winner = game.get_winner()
			if winner == 1:
				return 1
			elif winner == 2:
				return -1
			else:
				return 0
		else:
			next_state = np.random.sample(children)
			game.set_state(next_state)
			return self.rollout_step(game)



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
			self.children[idx] = node

	def get_children(self):
		return self.children

	def update_stats(self, result):
		if result == 1:
			self.num_wins += 1
		self.num_played += 1
	def update_child_stats(self,result, child_idx):
		self.children[idx].update_stats(result)


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
