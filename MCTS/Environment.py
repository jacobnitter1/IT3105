import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import ast
from IPython.display import clear_output

class NIM():
	def __init__(self,N,K, P,verbose_mode):
		self.N = N
		self.K = K
		self.n = N
		self.last_player = None # 1 or 2
		self.starting_player = P
		self.done = False
		self.verbose_mode = verbose_mode
		if self.verbose_mode:
			print("Start Pile: ",self.N," stones")

	def reset(self):
		self.n = N
		self.last_player = None
		if self.verbose_mode:
			print("Start Pile: ",self.N," stones")

	def reset(self,N,K):
		self.N = N
		self.K = K
		self.n = N
		self.last_player = None
		if self.verbose_mode:
			print("Start Pile: ",self.N," stones")

	def set_state(self, N, K ,n, last_player):
		self.N = N
		self.K = K
		self.n = n
		self.last_player = last_player
	def set_state(self,n,last_player):
		self.n = int(n)
		self.last_player = last_player

	def get_state(self):
		return str(self.n)

	def make_move(self, player, num_stones):
		winner = None
		done = False
		reward =0
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
				if self.verbose_mode:
					print("Player ",player," selects ",num_stones," stone(s) : Remaining stones  = ", self.n)
				if self.n == 0:
					winner = player
					reward = 1
					self.done = True
					print("Player ", winner," wins")
					done=True
				return reward, winner, done

	def get_legal_actions(self):
		actions = [i for i in range(1,np.minimum(self.K,self.n)+1)]
		return actions

	#def get_legal_actions(self, board_state):
	#	return np.range(1,np.min(self.K,board_state))


	def get_last_player(self):
		return self.last_player

	def get_winner(self):
		if self.done:
			return self.last_player
		else:
			return None

	def is_game_done(self):
		if self.n == 0:

			self.done = True
		else:
			self.done = False
		return self.done, self.last_player

	def is_state_end_state(self,state):
		if state == 0:
			return True
		else:
			return False
	def is_next_state_too_stupid(self,state):
		if state < self.K:
			return True
		else:
			return False

	def get_child_states(self):
		actions = self.get_legal_actions()
		children=[]
		for action in actions:
			n = int(np.copy(self.n))
			n-=action
			if n == 0:
				self.done = True
				children.append(n)
			else:
				children.append(n)
		if len(children) == 0:
			self.done = True
			return []
		else:
			return children


class Ledge():
	def __init__(self, L, NC, B_init,P,verbose_mode):
		self.board = np.zeros(L)

		self.verbose_mode = verbose_mode
		if B_init == None:
			self.random_reset(L,NC,verbose_mode)
		else:
			self.reset(L,NC, B_init, verbose_mode)
		self.starting_player = P
		self.last_player = None
		self.done = False
		if self.verbose_mode:
			print(self.verbose_mode,"Start Board: "+str(self.board))
	def is_game_done(self):
		state = self.get_state()
		for i in range(0,len(state)):
			if state[i] == '2':
				self.done = False

				#print("Inside game done func = ",state[i],self.get_state(),self.done)
				return self.done, self.last_player
		self.done = True


		#print("Inside game done func",self.get_state(),self.done)
		return self.done,self.last_player

	def is_state_end_state(self,state):
		for i in range(0,len(state)):
			if state[i] == '2':
				return False
		return True


	def is_next_state_too_stupid(self,state):
		nonzeros= 0
		for l in state:
			if l != '0':
				nonzeros += 1
		if state[0]==2:
			return True
		elif nonzeros==2:
			if state[0] == 1 and state[1] == 2:
				return True
			elif state[1]==1 and state[3] == 2:
				return True
		else:
			return False


	def random_reset(self, L, NC, verbose_mode):
		idx = np.random.randint(0,L)
		self.board[idx] = 2
		empty_slots = [i for i, e in enumerate(self.board) if e == 0]
		idxs = np.random.sample(empty_slots,NC)
		self.verbose_mode = verbose_mode
		self.done = False
		for i in idxs:
			self.board[i] = 1

		self.last_player = None
		if self.verbose_mode:
			print("Start Board: "+str(self.board))


	def get_state(self):
		return self.state_to_str(self.board)

	def set_state(self,B_init_str, last_player):
		if B_init_str is None:
			print("board in set state is None")
			return None
		B = []
		for letter in B_init_str:
			B.append(int(letter))
		self.board = B
		self.last_player = last_player

	def get_last_player(self):
		return self.last_player

	def reset(self, L, NC,B_init,verbose_mode):
		self.board = B_init
		self.last_player = None
		#if self.verbose_mode:
			#print("Start Board: "+str(self.board))

	def is_legal_jump(self, from_spot,num_jumps):

		legal_move = True
		#print(self.board)
		if self.board[from_spot] != 0 :#and from_spot-num_jumps>= 0:

			if num_jumps == 1:
				if self.board[from_spot-1] != 0:
					return False
			else:
				for i in range(1,num_jumps):
					if self.board[from_spot-i] != 0:
			#			print("something between")
						legal_move = False
						#print(self.board[from_spot-i], " gives ", legal_move)
						break
			if self.board[from_spot-num_jumps] != 0:
				legal_move = False
				#print(from_spot,num_jumps,self.board[from_spot-num_jumps], " gives ", legal_move)
		else:
			legal_move= False
		#print("From spot: ",from_spot," to spot : ", from_spot-num_jumps," : ", legal_move)
		return legal_move



	def make_move(self,player,move): # move = [from_smot, num jumps]
		from_spot = move[0]
		num_jumps = move[1]
		done= False
		winner = None
		reward = 0
		str_p= ""
		if self.last_player == player:
			print("Not player ", player, "s turn!")
			return None

		if from_spot == 0:
			if self.board[0] == 2:
				self.board[0]=0
				self.last_player = player
				winner = player
				self.done = True
				done = True
				reward = 1
				if self.verbose_mode :
					if self.last_player == 1:
						str_p += "P1"
					else:
						str_p += "P2"
					str_p += " picks up gold: "+str(self.board)+"\nPlayer " + str(self.last_player) + " wins!"
			elif self.board[0] == 1:
				#print("HERHER ", self.verbose_mode)
				self.board[0]=0
				self.last_player = player
				winner = None
				self.done = False
				done = False
				reward = 0
				if self.verbose_mode :
					if self.last_player == 1:
						str_p += "P1"
					else:
						str_p += "P2"
					str_p += " picks up cobber: "+str(self.board)
					#print(str_p)


		else:
			if self.is_legal_jump(from_spot,num_jumps):
				#print("----> from to :",from_spot, from_spot-num_jumps)
				if self.verbose_mode:
					str_p = ""
				self.board[from_spot-num_jumps]= self.board[from_spot]
				self.board[from_spot]=0
				self.last_player = player
				if self.verbose_mode :
					if self.done:
						print("Player ", self.last_player," wins!")
					else:
						if self.last_player == 1:
							str_p += "P1"
						else:
							str_p += "P2"
						str_p += " moves "
						if self.board[from_spot-num_jumps] == 2:
							str_p += "gold from cell "+str(from_spot)+" to "+str(from_spot-num_jumps)+": "+ str(self.board)
						else:
							str_p += "cobber from cell "+str(from_spot)+" to "+str(from_spot-num_jumps)+": "+ str(self.board)

		if self.verbose_mode:
				print(str_p)
		return reward,winner, done

	def get_legal_actions(self):
		legal_actions = []
		for i in range(0,len(self.board)):
			if i == 0:
				if self.board[i] != 0:
					legal_actions.append([i,None])
			else:
				if self.board[i]!= 0:
					for j in range(1,i+1):
						if self.is_legal_jump(i,j):
							legal_actions.append([i,j])
						else:
							break

		if len(legal_actions) == 0:
			return []
		else:
			return legal_actions


	def state_to_str(self,state):
		state_str = ""
		for i in state:
			state_str += str(i)
		return state_str

	def get_child_states(self):
		actions = self.get_legal_actions()
		children = []
		for action in actions:
			board_state = np.copy(self.board)
			if action[0] == 0:
				coin = board_state[0]
				if coin == 2:
					board_state[0] = 0
					#children.append( [board_state,True, True])
					children.append(self.state_to_str(board_state))
					done = True
					self.done = True
				else:
					board_state[0]=0
					num_coins_left = np.count_nonzero(board_state)
					if num_coins_left == 0:
						done = True
						self.done = True
					else:
						done = False
						self.done = False
					#children.append([board_state, done, None])
					children.append(self.state_to_str(board_state))
			else:
				board_state = np.copy(self.board)
				board_state[action[0]-action[1]]=board_state[action[0]]
				board_state[action[0]]=0
				children.append(self.state_to_str(board_state))
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
