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

class NIM():
	def __init__(self,N,K, P,verbose_mode):
		self.N = N
		self.K = K
		self.n = N
		self.last_player = None # 1 or 2
		self.starting_player = P
		self.done = False

	def reset(self):
		self.n = N
		self.last_player = None

	def reset(self,N,K):
		self.N = N
		self.K = K
		self.n = N
		self.last_player = None

	def set_state(self, N, K ,n, last_player):
		self.N = N
		self.K = K
		self.n = N
		self.last_player = last_player
	def set_state(self,n,last_player):
		self.n = int(n)
		self.last_player = last_player

	def get_state(self):
		return str(self.n)

	def make_move(self, player, num_stones):
		winner = None
		done = False
		if self.last_player == player:
			print("Not player ", player, "s turn!")
			return None
		else:
			if num_stones < 1 or num_stones > self.K:
				print("Cannot remove ", num_stones," from the board! Must be within range 0 and ", self.K,".")
				return None
			elif self.n - num_stones < 0:
				print("Cannot remove more stones than there is on the board!")
				return None
			else:
				self.n -= num_stones
				self.last_player = player
				if self.n == 0:
					winner = player
					reward = 1
					done = True
				return self.n,reward, winner, done

	def get_legal_actions(self):
		return np.range(1,np.min(self.K,self.n))

	def get_legal_actions(self, board_state):
		return np.range(1,np.min(self.K,board_state))


	def get_last_player(self):
		return self.last_player

	def get_winner(self):
		if self.done:
			return self.last_player
		else:
			return None


	def get_child_states(self):
		actions = self.get_legal_actions()
		children=[]
		for action in actions:
			n = np.copy(self.n)
			n-=action
			if n == 0:
				#children.append([n, True, True])
				children.append([str(n)])
			else:
				#children.append([n, False, None])
				children.append([str(n)])
		if len(children) == 0:
			self.done = True
			return []
		else:
			return children


class Ledge():
	def __init__(self, L, NC, B_init,P,verbose_mode):
		self.board = np.zeros(L)
		if B_init == None:
			self.random_reset(L,NC,verbose_mode)
		else:
			self.reset(L,NC, verbose_mode)
		self.starting_player = P
		self.last_player = None
		self.verbose_mode = verbose_mode
		self.done = False

	def random_reset(self, L, NC, verbose_mode):
		str=""
		idx = np.random.randint(0,L)
		self.board[idx] = 2
		empty_slots = [i for i, e in enumerate(self.board) if e == 0]
		idxs = np.random.sample(empty_slots,NC)
		self.verbose_mode = verbose_mode
		self.done = False
		for i in idxs:
			self.board[i] = 1

		self.last_player = None
		if verbose_mode:
			str = "Start Board: "+str(self.board)
		return str

	def get_state(self):
		return str(self.board)
	def set_state(self,B_init_str, last_player):
		self.state = [int(s) for s in B_init_str]
		self.last_player = last_player

	def get_last_player(self):
		return self.last_player

	def reset(self, L, NC,B_init):
		str=""
		self.board = B_init
		self.verbose_mode = verbose_mode
		self.last_player = None
		if verbose_mode:
			str = "Start Board: "+str(self.board)
		return str

	def is_legal_jump(self, from_spot,num_jumps):
		if self.board[from_spot] != 0:
			legal_move = True
			for i in range(1,num_jumps):
				if self.board[from_spot-i] != 0:
					legal_move = False
		else:
			legal_move= False
		return legal_move

	def is_legal_jump(self, from_spot,num_jumps, board_state):
		if board_state[from_spot] != 0:
			legal_move = True
			for i in range(1,num_jumps):
				if board_state[from_spot-i] != 0:
					legal_move = False
		else:
			legal_move= False
		return legal_move

	def make_move(self,player,from_spot, num_jumps):
		done= False
		winner = None
		reward = 0
		str= ""
		if self.last_player == player:
			print("Not player ", player, "s turn!")
			return None

		if from_spot == 0:
			if self.board[0] == 2:
				self.board[0]=0
				self.last_player = player
				winner = player
				done = True
				reward = 1
		else:
			if self.is_legal_jump(from_spot,num_jumps):
				if self.verbose_mode:
					str = ""
				self.board[from_spot-num_jumps]= self.board[from_spot]
				self.board[from_spot]=0
				self.last_player = player
		if verbose_mode :
			if self.last_player == 1:
				str += "P1"
			else:
				str += "P2"
			if done:
				str += "picks up gold: "+str(self.board)+"\n Player" + str(self.last_player) + "wins!"
			str += " moves "
			if self.board[from_spot-num_jumps] == 1:
				str += "gold from cell "+str(from_spot)+" to "+str(from_spot-num_jumps)+": "+ str(self.board)
			else:
				str += "cobber from cell "+str(from_spot)+" to "+str(from_spot-num_jumps)+": "+ str(self.board)
		return self.board, reward,winner, done,str

	def get_legal_actions(self):
		nonempty_slots = [i for i, e in enumerate(self.board) if e != 0]
		legal_actions = []
		for i in range(0,nonempty_slots):
			if self.boardState[i]!=0:
				legal_actions.append([from_spot,None])
			else:
				for j in range(1, i):
					if self.is_legal_jump(i,j):
						legal_actions.append([i,j])
					else:
						break
		if len(legal_actions) == 0:
			return []
		else:
			return legal_actions

	def get_legal_actions(self,board_state):
		nonempty_slots = [i for i, e in enumerate(board_state) if e != 0]
		legal_actions = []
		for i in range(0,nonempty_slots):
			if board_state[i]!=0:
				legal_actions.append([from_spot,None])
			else:
				for j in range(1, i):
					if self.is_legal_jump(i,j):
						legal_actions.append([i,j])
					else:
						break
		if len(legal_actions) == 0:
			return []
		else:
			return legal_actions

	def get_child_states(self):
		actions = get_legal_actions()
		children = []
		for action in actions:
			board_state = np.copy(self.board)
			if action[0] == 0:
				coin = board_state[0]
				if coin == 2:
					board_state[0] = 0
					#children.append( [board_state,True, True])
					children.append(str(boardState))
				else:
					board_state[0]=0
					num_coins_left = np.count_nonzero(board_state)
					if num_coins_left == 0:
						done = True
					else:
						done = False
					#children.append([board_state, done, None])
					children.append(str(board_state))
			else:
				board_state = np.copy(self.board)
				board_state[action[0]-action[1]]=board_state[action[0]]
				board_state[action[0]]=0
				#children.append([board_state,False, None])
				children.append(str(board_state))
		if len(children) == 0:
			self.done = True
			return []
		else:
			return children

	def get_winner(self):
		if self.done:
			return self.last_player
		else:
			return None
