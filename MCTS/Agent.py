import numpy as np
import random
DEBUGGING_VAL = False
class MCTS:

	def __init__(self, exploration_rate, game, rollout_game):
		self.root_node = Node(game.get_state(),game.get_last_player(), None)
		self.exploration_rate = exploration_rate
		self.game = game
		self.rollout_game = rollout_game
		#self.previously_visited_nodes = []

	def set_exploration_rate(self,e):
		self.exploration_rate = e
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

	def tree_search(self):
		last_node = self.root_node
		current_node = self.root_node
		self.rollout_game.set_state(current_node.get_state(), current_node.get_last_player())
		if len(current_node.get_children()) == 0:
			self.node_expansion(current_node)
		next_node_idx = None
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

	def _tree_search(self):
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

	def leaf_evaluation(self,leaf_node, last_player):
		num_wins = 0
		num_plays = 1
		for i in range(0,num_plays):
			num_wins += self.rollout(leaf_node,last_player)
		return num_wins, num_plays

	def run_simulation(self):
		leaf_node_state, idx,leaf_node = self.tree_search()
		if leaf_node.get_last_player() == 1:
			result, num_plays = self.leaf_evaluation(leaf_node_state, 2)
		else:
			result, num_plays = self.leaf_evaluation(leaf_node_state, 1)
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
		#print(self.game.get_legal_actions())
		#print("Root node info - State : ",self.root_node.get_state(), " and last player : ",self.root_node.get_last_player())
		if DEBUGGING_VAL :
			print(self.root_node.get_state()," has the following Q stuffs :")
		#self.root_node.print_Q_properties()
		action = self.choose_greedy_action()
		return action


	def node_expansion(self,parent_node,child_node_state=None, idx=None):
		self.rollout_game.set_state(parent_node.get_state(),parent_node.get_last_player())#, parent_node.get_last_player())
		children = self.rollout_game.get_child_states()
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

	def rollout(self,root_node,last_player):
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

	def rollout_step(self):
		children = self.rollout_game.get_child_states()
		if self.rollout_game.is_game_done()[0]:#print("End of Rollout : ",self.rollout_game.get_state(), self.rollout_game.get_winner(), self.rollout_game.is_game_done())
			winner = self.rollout_game.get_winner()
			#print("Rollout game is done : ", self.rollout_game.get_state(), " last player was : ", self.rollout_game.get_last_player(), self.rollout_game.get_winner())
			if winner == 1:
				return 1
			elif winner == 2:
				return 0
		else:
			last_state = self.rollout_game.get_state()
			next_state = self.rollout_policy()#self.default_policy()#self.rollout_policy()#[0]
			if self.rollout_game.get_last_player() == 1:
				self.rollout_game.set_state(next_state,2)
			else:
				self.rollout_game.set_state(next_state,1)
			if DEBUGGING_VAL :
				print("Rollout step from : ",last_state," to : ",next_state," with last player ", self.rollout_game.get_last_player())
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
		#self.all_visited_states = {}

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
