import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from IPython.display import clear_output

class HexBoard():
	def __init__(self, boardShape, boardSize, numHoles, placementHoles):
		self.boardShape = boardShape
		self.boardSize = boardSize
		self.numHoles= numHoles
		self.placementHoles = placementHoles # if None -> if numHoles <4 -> center, else random
		self.boardState = self.init_boardState(boardSize, boardShape, numHoles)
		self.place_holes()

	def init_boardState(self,boardSize, boardShape, numHoles):
		#1 = peg in place, 0 = empty hole, -1 = hole not in game
		boardState = np.ones([boardSize, boardSize])
		if boardShape == "Triangle":
			boardState+=1
			boardState=np.tril(boardState,k=0)-1
		# add some randomness here, or default at center
		return boardState
	def place_holes(self):
		placementHoles=[]
		for i in self.placementHoles:
			r,c = self.hole_num_to_placement(i)
			self.boardState[r,c]=0



	def get_boardState(self):
		listBoardState = list(filter(lambda a: a != -1, (self.boardState).flatten()))
		return listBoardState

	def get_org_boardState(self):
		return self.boardState

	def get_num_pegs_left(self):
		num_pegs = 0
		for r in range(0,self.boardSize):
			for c in range(0,self.boardSize):
				i = self.boardState[r,c]
				if i == 1:
					num_pegs+=1

		return num_pegs

	def hole_num_to_placement(self, hole_num):
		k = 0
		bs = self.get_org_boardState()
		for r in range(0,len(bs)):
			for c in range(0,len(bs)):
				if bs[r,c] != -1:
					if k == hole_num:
						return r,c
					else:
						k +=1

	def placement_to_hole_num(self,r,c):
		k = 0
		for r_ in range(0,self.boardSize):
			for c_ in range(0,self.boardSize):
				if self.boardState[r_,c_] != -1:
					if r==r_ and c==c_ :
						#print("placement_to_hole_num returned : ",k," with boardsize", self.boardSize)
						return k
					else:
						k+=1

	def print_boardState(self):
		print(self.boardState)


class HexGame(HexBoard):
	def __init__(self, boardShape,boardSize, startingPlayer):
		self.boardShape = boardShape
		self.boardSize = boardSize
		self.boardState = np.zeros((self.boardSize,self.boardSize))
		self.startingPlayer = startingPlayer
		if startingPlayer == 1:
			self.last_player = 2
		else:
			self.last_player = 1
		self.neighbor_paths = np.array([[1,1],[-1,-1],[-1,0],[1,0],[0,1],[0,-1]])

	def reset_board(self):
		self.boardState = np.zeros((self.boardSize,self.boardSize))
		if self.startingPlayer == 1:
			self.last_player = 2
		else:
			self.last_player = 1
	def set_starting_player(self,p):
		self.startingPlayer = p
		if self.startingPlayer == 1:
			self.last_player = 2
		else:
			self.last_player = 1




	def do_action(self, action, player):
		if player == self.last_player:
			print("Not player ", player,"'s turn!")
			return None
		elif action >= self.boardSize*self.boardSize:
			print("Too big action, not legal.")
			return None
		else:
			for i in range(0,len(self.boardState)):
				for j in range(0,len(self.boardState)):
					if action == i*len(self.boardState)+j:
						if self.boardState[i,j] == 0:
							self.boardState[i,j] = player
						else:
							print("Hole already filled! Not legal action.")
							#print(self.boardState)
							print("Action : ", action)
							print("--------------------")
							return None
		self.last_player = player
		return self.boardState, self.get_winner()

	def get_state(self):
		return self.get_NN_state()
	def get_winner(self):
		for i in range(0,len(self.boardState)):
			#print(" Starting point ", i)

			if self.go_to_neighbor_p1(0,i,[]):
				return 1
			if self.go_to_neighbor_p2(i,0,[]):
				return 2
		return None

	def _go_to_neighbor_p1(self,i,j,prev_visited):
		connection = False
		if self.boardState[i,j] != 1:
			return False
		if i== len(self.boardState)-1:

			prev_visited.append([i,j])
			print(prev_visited)
			print("1 : ",[i,j])
			return True
		active_neighbours = []
		for p in self.neighbor_paths:
			if i + p[0] >= 0 and i + p[0] < len(self.boardState):
				if j + p[1] >= 0 and j + p[1] < len(self.boardState):
					if self.boardState[i+p[0],j+p[1]] == 1:
						if [i+p[0],j+p[1]] not in prev_visited:
							active_neighbours.append([i+p[0],j+p[1]])
		if active_neighbours == []:
			return False
		for n in active_neighbours:
			prev_visited.append([i,j])
			if self.go_to_neighbor_p1(n[0],n[1],prev_visited):
				connection = True
		return connection

	def _go_to_neighbor_p2(self,i,j,prev_visited):
		#print([i,j])
		connection = False
		prev_visited.append([i,j])
		if self.boardState[i,j] != 2:
			return False
		if j== len(self.boardState)-1:
			print(prev_visited)
			print("2 : ",[i,j])
			return True
		active_neighbours = []
		for p in self.neighbor_paths:
			if i + p[0] >= 0 and i + p[0] < len(self.boardState):
				if j + p[1] >= 0 and j + p[1] < len(self.boardState):
					if self.boardState[i+p[0],j+p[1]] == 2:
						if [i+p[0],j+p[1]] not in prev_visited:
							active_neighbours.append([i+p[0],j+p[1]])
		if active_neighbours == []:
			print(prev_visited)
			return False
		for n in active_neighbours:
			if self.go_to_neighbor_p2(n[0],n[1],prev_visited):
				connection = True
		print(prev_visited)
		return connection




	def go_to_neighbor_p1(self,i,j, prev_visited): # i = 0 at beginning
		connection = False
		#print("BoardState " , self.boardState)
		if self.boardState[i,j] != 1:
			#print("her", i,j,self.boardState[i,j])
			return False
		else:
			#print("Init check : ",i, " == ", len(self.boardState)-1, " ?  [",i,",",j,"]")
			if i == (len(self.boardState)-1):
				return True
		#print("Paths ; ", self.neighbor_paths)
		for p in self.neighbor_paths:
			copy_prev_visited = prev_visited
			#print(" check neighbor [",i+p[0],",",j+p[1],"]")
			if i + p[0] >= 0 and i + p[0] < len(self.boardState):
				if j + p[1] >= 0 and j + p[1] < len(self.boardState):
					if self.boardState[i+p[0],j+p[1]] == 1:
						#print("--> [",i+p[0],",",j+p[1],"] is a neighbor!")
						if [i+p[0],j+p[1]] in prev_visited:
							continue
							#print( [i+p[0],j+p[1]], " is  in prev_visitided ", prev_visited)
							#return False
						else:
							#print( [i+p[0],j+p[1]], " is NOT in prev_visitided ", prev_visited)
							if i+p[0] == (len(self.boardState)-1):
								return True
							else:
								copy_prev_visited.append([i+p[0],j+p[1]])
								c =self.go_to_neighbor_p1(i+p[0],j+p[1], copy_prev_visited)
								if c == True:
									connection = True
		return connection

	def go_to_neighbor_p2(self,i,j, prev_visited):
		connection = False
		if self.boardState[i,j] != 2:
			return False
		else:
			if j == (len(self.boardState)-1):
				return True
		for p in self.neighbor_paths:
			copy_prev_visited = prev_visited
			if i + p[0] >= 0 and i + p[0] < len(self.boardState):
				if j + p[1] >= 0 and j + p[1] < len(self.boardState):
					if self.boardState[i+p[0],j+p[1]] == 2:
						if [i+p[0],j+p[1]] in prev_visited:
							continue
							#return False
						else:
							if j+p[1] == (len(self.boardState)-1):
								return True
							else:
								copy_prev_visited.append([i+p[0],j+p[1]])
								c =self.go_to_neighbor_p2(i+p[0],j+p[1], prev_visited)
								if c == True:
									connection = True
		return connection

	def action_from_s1_to_s2(self,s1,s2):

		_s1=np.zeros(self.boardSize*self.boardSize)
		_s2 = np.zeros(self.boardSize*self.boardSize)
		for i in range(0,int(len(s1)/2)-1):
			if s1[i*2] == 0:
				if s1[i*2+1] == 0:
					pass
				else:
					_s1[i]=2
			else:
				_s1[i] = 1

			if s2[i*2] == 0:
				if s2[i*2+1] == 0:
					pass
				else:
					_s2[i]=2
			else:
				_s2[i] = 1
		for i in range(0,len(_s1)):
			if _s1[i] != _s2[i]:
				return i
		return None

	def print_state(self):
		print(self.boardState)
	def get_boardState(self):
		return self.boardState

	def get_child_states(self):
		#print("!!!!!!!!!!! get child states")
		org_state = self.get_NN_state()
		l_p = self.last_player
		children_states = []
		actions = self.get_legal_actions()
		#print("Actions :: ",actions, actions[0])
		for i in actions:
			self.set_state(org_state,l_p)
			self.last_player = l_p
			if l_p == 1:
				self.do_action(i,2)
			else:
				self.do_action(i,1)
			children_states.append(self.get_NN_state())
		self.set_state(org_state,l_p)
		self.last_player= l_p
		return children_states

	def get_padded_child_states(self):
		org_state = self.get_NN_state()
		l_p = self.last_player
		children_states = []
		actions = self.get_legal_actions()
		#print("Actions :: ",actions)
		for i in range(0,self.boardSize*self.boardSize):
			if i in actions:
				self.set_state(org_state,l_p)
				self.last_player = l_p
				if l_p == 1:
					self.do_action(i,2)
				else:
					self.do_action(i,1)
				children_states.append(self.get_NN_state())
				self.set_state(org_state,l_p)
			else:
				children_states.append([])
		self.last_player= l_p
		return children_states

	def any_child_state_winning(self):
		org_state = self.get_NN_state()
		l_p = self.last_player
		children_states = []
		actions = self.get_legal_actions()
		#print("Actions :: ",actions)
		for i in range(0,self.boardSize*self.boardSize):
			if i in actions:
				self.set_state(org_state,l_p)
				self.last_player = l_p
				if l_p == 1:
					self.do_action(i,2)
				else:
					self.do_action(i,1)
				if self.get_winner() != None:
					return True,i
		return False, None



	def get_legal_actions(self):
		a,w = self.get_action_space()
		return a

	def get_action_space(self):
		actions=[]
		for i in range(0,len(self.boardState)):
			for j in range(0,len(self.boardState)):
				if self.boardState[i,j] == 0:
					actions.append(i*len(self.boardState)+j)
		winner = self.get_winner()
		return actions, winner



	###FUNCTIONS NEEDED BY RL
	def get_NN_state(self):
		state = []
		for i in range(0,len(self.boardState)):
			for j in range(0,len(self.boardState)):
				if self.boardState[i,j] == 0:
					state.append(0)
					state.append(0)
				elif self.boardState[i,j] == 1:
					state.append(1)
					state.append(0)
				else:
					state.append(0)
					state.append(1)
		if self.last_player == 1:
			state.append(0)
			state.append(1)
		else:
			state.append(1)
			state.append(0)
		return state

	###FUNCTIONS NEEDED BY MCTS
	def get_last_player(self):
		return self.last_player

	def _get_child_states(self):
		actions = self.get_action_space()
		children_states = []
		for a in actions:
			child = []
			for i in range(0,len(self.boardState)):
				for j in range(0,len(self.boardState)):
					if  a == i*len(self.boardState)+j:
						if self.last_player == 1:
							child.append(0)
							child.append(1)
						else:
							child.append(1)
							child.append(0)
					else:
						if self.boardState[i,j] == 0:
							child.append(0)
							child.append(0)
						elif self.boardState[i,j] == 1:
							child.append(1)
							child.append(0)
						else:
							child.append(0)
							child.append(1)
			if self.last_player == 1:
				child.append(0)
				child.append(1)
			else:
				child.append(1)
				child.append(0)
			children_states.append(child)
		return children_states

	def is_game_done(self):
		winner = self.get_winner()
		if  winner == None:
			return False, None
		else:
			return True, winner

	def is_state_end_state(self, state):
		temp_state = np.copy(self.boardState)
		self.set_state(state, self.last_player)
		d,l_p = self.is_game_done()
		self.boardState = temp_state
		return d

	def get_MCTS_state(self):
		return self.get_NN_state(), self.last_player

	def set_state(self, state,l_p):
		s = []
		#print(state)
		#print(state, len(state),len(state)/2,type(len(state)/2))
		for i in range(0,int(len(state)/2)):
			if state[i*2] == 0:
				if state[i*2+1] == 0:
					s.append(int(0))
				else:
					s.append(int(2))
			else:
				s.append(int(1))
		#print("Set state : ",state, s)
		for i in range(0,len(self.boardState)):
			for j in range(0,len(self.boardState)):
				self.boardState[i,j] = s[i*len(self.boardState)+j]
		#print(self.boardState)
		self.last_player = l_p
		return self.boardState

class VisualizeHexGame():
	def __init__(self,board, boardShape):
		self.boardState = board
		self.boardShape = boardShape
		self.colormap = self.pegColorMap()
		self.node_sizes = self.node_sizes_()
		self.pegPositions, self.nodelist = self.pegPositionsNX()

	def update_boardState(self,boardState):
		self.boardState = boardState

	def pegColorMap(self):
		boardState = self.boardState

		#print("---> pegColorMap : ",boardState)
		#print( len(self.boardState))
		#print( (self.boardState[0][0]))
		colormap = []
		for r in range(0, len(self.boardState)):
			for c in range(0, len(self.boardState)):
				i = int(self.boardState[r][c])
				if i == 1 :
					colormap.append('blue')
				elif i == 2:
					colormap.append('red')
				else:
					colormap.append('black')

		#print(colormap)
		return colormap

	def node_sizes_(self):
		org_boardState= self.boardState
		boardState = list(filter(lambda a: a != -1, org_boardState.flatten()))
		node_sizes = []
		for i in boardState:
			node_sizes.append(1000)
		return node_sizes

	def pegPositionsNX(self):
		pos={}
		nodelist=[]
		peg_num=0
		boardSize =len(self.boardState)
		first_pos =[0,0]
		for dia_row in range(0,boardSize):
			for dia_col in range(0,boardSize):
				first_pos[0] = -dia_row+dia_col
				first_pos[1]= dia_row+dia_col
				pos[peg_num] = [first_pos[0],first_pos[1]]
				#print(peg_num,first_pos)
				nodelist.append(peg_num)
				peg_num+=1
		return pos, nodelist

	def _pegPositionsNX(self):
		pos={}
		nodelist=[]
		peg_num=0
		boardSize = len(self.boardState)
		for r in range(0,boardSize):
			if r == 0:
				pos[peg_num]=[0,0]
				nodelist.append(peg_num)
				peg_num +=1
			elif r%2 != 0:#Oddetallsrader
				for c in range(int((r+1)/2),0,-1):
					pos[peg_num]=[-1-2*(c-1),r] #pos[peg_num]=[r, -1 -2*(c-1)]
					nodelist.append(peg_num)
					peg_num+=1
				for c in range(0,int((r+1)/2)):
					pos[peg_num]=[1+2*c,r]#pos[peg_num] = [r,1+2*c]
					nodelist.append(peg_num)
					peg_num +=1
			elif r%2 == 0: #partallsrader
				for c in range(int(r/2),0,-1):
					pos[peg_num]=[-2-2*(c-1),r]#pos[peg_num]=[r,-2-2*(c-1)]
					nodelist.append(peg_num)
					peg_num +=1
				pos[peg_num]=[0,r] # [r,0]
				nodelist.append(peg_num)
				peg_num+=1
				for c in range(0,int(r/2)):
					pos[peg_num]=[2+2*c,r]#pos[peg_num]=[r,2+2*c]
					nodelist.append(peg_num)
					peg_num+=1

		num_pegs_in_row = r
		#print(num_pegs_in_row)

		for row in range(r+1, boardSize*2-1):
			if num_pegs_in_row <= 1:
				pos[peg_num]=[0,row]
				nodelist.append(peg_num)
				break
			else:
				if row%2 == 0:
					for c in range(int(num_pegs_in_row/2),0,-1):
						pos[peg_num]=[-2-2*(c-1),row]#[row,-2-2*(c-1)]#
						nodelist.append(peg_num)
						peg_num +=1
					pos[peg_num]=[0,row] # [r,0]
					nodelist.append(peg_num)
					peg_num+=1
					for c in range(0,int(num_pegs_in_row/2)):
						pos[peg_num]=[2+2*c,row]
						nodelist.append(peg_num)
						peg_num+=1
				elif row%2 != 0:
					for c in range(int(num_pegs_in_row/2),0,-1):
						pos[peg_num]=[ -1 -2*(c-1),row]#[-1-2*(c-1),row] #pos[peg_num]=
						nodelist.append(peg_num)
						peg_num+=1
					for c in range(0,int(num_pegs_in_row/2)):
						pos[peg_num]=[1+2*c,row]#[1+2*c,row]#pos[peg_num] =
						nodelist.append(peg_num)
						peg_num +=1
				num_pegs_in_row-=1

		return pos,nodelist

	def update_vis_params(self, boardState):

		self.update_boardState(boardState)
		self.colormap = self.pegColorMap()
		self.node_sizes = self.node_sizes_()
		self.pegPositions, self.nodelist = self.pegPositionsNX()

	def get_vis_params(self,boardState):
		self.update_vis_params(boardState)
		return self.pegPositions, self.node_sizes, self.nodelist, self.colormap


	def drawBoard(self,boardState):
		self.update_vis_params(boardState)
		plt.figure(figsize =(124,124))
		g = nx.Graph()

		#print("POSITIONS:",self.pegPositions)
		#print("SIZES:", self.node_sizes)
		#print("NODELIST:",self.nodelist)
		#print("COLORMAP : ", self.colormap)
		nx.draw_networkx_nodes(g, self.pegPositions, node_size = self.node_sizes, nodelist=self.nodelist, node_color=self.colormap)

		plt.draw()
		plt.pause(0.2)
		#plt.close()

	def show_played_game(self,states, delay,shape,size,type):
		print("TRAJECTORY HAS ", len(states), " TRANSITIONS!")
		print(states, states[0])

		plt.ion()

		if len(states)== 0:
			#print("0 transitions")
			return None
		elif len(states)==1:
			#print("1 transitions")
			#print(len(states), states[0])
			self.drawBoard(states[0])
		else:
			#print("many transitions")
			#self.update_vis_params(states[0],action[0])
			g = nx.Graph()
			#nx.draw_networkx_nodes(g, self.pegPositions, node_size = self.node_sizes, nodelist=self.nodelist, node_color=self.colormap)

			#plt.show()
			for i in range(0,len(states)):
				#print("---->",states[i])
				#print("--->",states[i], actions[i])
				#print("state -> ",states[i])
				#print("show ", actions[i][0])
				print("---",states[i])
				self.update_vis_params(states[i])
				nx.draw_networkx_nodes(g, self.pegPositions, node_size = self.node_sizes, nodelist=self.nodelist, node_color=self.colormap)
				plt.title(str(type)+" "+str(shape)+" "+str(size))
				plt.draw()
				#print("before pause")
				plt.pause(delay)
				#print("after pause : ", delay)
				plt.clf()

def show_policy( policy, boardShape,boardSize, startingPlayer,env):
	t_states = []
	t_states.append(np.copy(env.get_boardState()))
	while not env.is_game_done():

		state=env.get_NN_state()
		last_player = env.get_last_player()
		legal_actions = env.get_legal_actions()
		a= policy_network.get_action(state,legal_actions,0)
		if env.last_player == 1:
			env.do_action(a,2)
		else:
			env.do_action(a,1)
		t_states.append(np.copy(env.get_boardState()))
	#print(t_states)
	vis = Environment.VisualizePegSolitaire(env.get_boardState(),shape)
	vis.show_played_game(t_states,DELAY_BETWEEN_MOVE,shape,size,type)
