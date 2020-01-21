import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
				if self.boardState[r,c] != -1:
					if r==r_ and c==c_ :
						return k
					else:
						k+=1


	def print_boardState(self):
		print(self.boardState)





class PegSolitaire(HexBoard):
	def __init__(self, boardShape, boardSize, numHoles, placementHoles):
		self.boardShape = boardShape
		self.boardSize = boardSize
		self.numHoles= numHoles
		self.placementHoles = placementHoles # if None -> if numHoles <4 -> center, else random
		self.boardState = self.init_boardState(boardSize, boardShape, numHoles)
		self.place_holes()
		self.actionSpace= np.array([[-1,0],[-1,1],[0,1],[1,0],[1,-1],[0,-1]])
		self.lastAction=[None,None] #from numhole, to numhole
		# self.numPegsLeft=[] # for printing må til agent!
	def get_lastAction(self):
		return self.lastAction

	def is_legal_action(self, hole_num, action):
		boardState = self.get_org_boardState()
		a_r = self.actionSpace[action][0]
		a_c = self.actionSpace[action][1]
		ok_move= True
		r,c = self.hole_num_to_placement(hole_num)
		#print("Place: ", hole_num," and action ", action)
		if boardState[r,c] == 0:
			ok_move =False
			#print(r,c," No peg in hole wanting to jump")
		else:
			if not self.inside_board(r+a_r,c+a_c):
			#	print(r,c," Neighbor hole outside board")
				ok_move = False
			else:
				if boardState[r+a_r,c+a_c] != 1:
			#		print(r,c," No peg in neighbor hole/neighbor hole = -1")
					ok_move = False
				else:
					if (not self.inside_board(r+2*a_r,c+2*a_c)):
			#			print(r,c," Landing hole outside board")
						ok_move = False
					else:
						if boardState[r+2*a_r,c+2*a_c] != 0:
			#				print(r,c," Landing hole not empty/landin hole = -1")
							ok_move = False
		if ok_move:
			print("OK action jumping from ", hole_num," to ", self.placement_to_hole_num(r+2*a_r,c+2*a_c)," by action ", action,)
		return ok_move






	def check(self):
		#hole_num_to_placement OK
		bs = self.boardState
		for r in range(0,len(bs)):
			for c in range(0,len(bs)):
				k = self.placement_to_hole_num(r,c)
				r_,c_ = self.hole_num_to_placement(k)
				print(r,c," = ", k, " = ",r_,c_)
		print("OK")
	def inside_board(self,r,c):
		if r >= 0 and r < self.boardSize:
			if c >= 0 and c < self.boardSize:
				return True
			else:
				return False
		else:
			return False

	def move_peg(self, hole_num, action):
		reward = 0
		move_done = False
		game_done = False
		if hole_num > 2**self.boardSize:
			print("There is no such place!")
			reward = -10
		else:
			if self.is_legal_action(hole_num, action):
				a_r = self.actionSpace[action][0]
				a_c = self.actionSpace[action][1]
				r,c = self.hole_num_to_placement(hole_num)
				self.boardState[r,c] = 0
				self.boardState[r+a_r,c+a_c] = 0
				self.boardState[r+2*a_r,c+2*a_c] = 1
				self.lastAction = [[r,c],[r+2*a_r,c+2*a_c]]
				print("Jumped from ",hole_num," to ", self.placement_to_hole_num(r+2*a_r,c+2*a_c))
				move_done = True
				reward = 1
			else:
				reward = -10
				move_done = False

		ended, won = self.is_game_done()
		if won:
			reward = 100
		new_boardState = self.get_boardState()
		return move_done, reward, new_boardState, ended

	def moves_left(self):
		boardState = self.get_boardState()
		actions = self.actionSpace
		moves_left = False
		for s in range(0,len(boardState)):
			if boardState[s] == 1:
				for a in range(0,len(actions)):
					if self.is_legal_action(s,a):
						moves_left = True
		return moves_left

	def is_game_done(self):
		if self.get_num_pegs_left() == 1:
			return True, True
		elif not self.moves_left():
			return True, False
		else:
			return False, False

	def print_board(self):
		print(self.boardState)








class VisualizePegSolitaire():
	def __init__(self,board, boardShape):
		self.boardState = board
		self.lastAction = [None,None]
		self.boardShape = boardShape
		self.colormap = self.pegColorMap()
		self.node_sizes = self.node_sizes_()
		self.pegPositions, self.nodelist = self.pegPositionsNX()
		
	def update_last_action(self, action):
		self.lastAction = action

	def update_boardState(self,boardState):
		self.boardState = boardState

	def pegColorMap(self):
		boardState = self.boardState
		#print(boardState)
		colormap = []
		for r in range(0, len(self.boardState)):
			for c in range(0, len(self.boardState)):
				i = self.boardState[r,c]
				if i == 1:
					colormap.append('red')
				elif i == 0 :
					colormap.append('black')
				
		#print(colormap)
		return colormap

	def node_sizes_(self):
		org_boardState= self.boardState
		lastAction = self.lastAction
		if lastAction[0] != None:
			org_boardState[lastAction[0][0],lastAction[0][1]]=3
			org_boardState[lastAction[1][0],lastAction[1][1]]=2
		boardState = list(filter(lambda a: a != -1, org_boardState.flatten()))
		node_sizes = []
		for i in boardState:
			if i == 0 or i == 1:
				node_sizes.append(1000)
			elif i == 2: # The one that moved last
				node_sizes.append(3000)
			elif i == 3: #The one that was last left
				node_sizes.append(200)
		return node_sizes

	def pegPositionsNX(self):
		pos={}
		nodelist=[]
		peg_num=0
		boardSize =len(self.boardState)
		print(boardSize)
		if self.boardShape == "Diamond":
			first_pos =[0,0]
			for dia_row in range(0,boardSize):
				for dia_col in range(0,boardSize):
					first_pos[0] = -dia_row+dia_col
					first_pos[1]= dia_row+dia_col
					pos[peg_num] = [first_pos[0],first_pos[1]]
					print(peg_num,first_pos)
					nodelist.append(peg_num)
					peg_num+=1
		print(pos, nodelist)
		return pos, nodelist

	def _pegPositionsNX(self):
		pos={}
		nodelist=[]
		peg_num=0
		boardSize = len(self.boardState)
		if self.boardShape == "Triangle" or self.boardShape == "Diamond":
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
		else:
			print("No/unknown board shape!")
			return None, None
		if self.boardShape == "Diamond":
			num_pegs_in_row = r
			print(num_pegs_in_row)
			
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

	def update_vis_params(self, boardState, lastAction):
		self.update_last_action(lastAction)
		self.update_boardState(boardState)
		self.colormap = self.pegColorMap()
		self.node_sizes = self.node_sizes_()
		self.pegPositions, self.nodelist = self.pegPositionsNX()


	def drawBoard(self,boardState,lastAction):
		self.update_vis_params(boardState, lastAction)
		plt.figure(figsize =(124,124))
		g = nx.Graph()

		print(self.pegPositions, self.node_sizes, self.nodelist, self.colormap)
		nx.draw_networkx_nodes(g, self.pegPositions, node_size = self.node_sizes, nodelist=self.nodelist, node_color=self.colormap)
		#plt.figure()
		plt.show()




