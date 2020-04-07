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
        if startingPlayer == 1:
            self.last_player = 2
        else:
            self.last_player = 1
        self.neighbor_paths = np.array([[1,1],[-1,-1],[-1,0],[1,0],[0,1],[0,-1]])

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
                            return None
        self.last_player = player
        return self.boardState, self.get_winner()


    def get_winner(self):
        for i in range(0,len(self.boardState)):
            #print(" Starting point ", i)
            if self.go_to_neighbor_p1(0,i,[[0,i]]):
                return 1
            if self.go_to_neighbor_p2(i,0,[[i,0]]):
                return 2
        return None

    def go_to_neighbor_p1(self,i,j, prev_visited): # i = 0 at beginning
        connection = False
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
                    state.append(0)
                    state.append(1)
                else:
                    state.append(1)
                    state.append(0)
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

    def get_child_states(self):
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
        print(state, len(state),len(state)/2,type(len(state)/2))
        for i in range(0,int(len(state)/2)):
            if state[i*2] == 0:
                if state[i*2+1] == 0:
                    s.append(int(0))
                else:
                    s.append(int(2))
            else:
                s.append(int(1))

        for i in range(0,len(self.boardState)):
            for j in range(0,len(self.boardState)):
                self.boardState[i,j] = s[i*len(self.boardState)+j]

        self.last_player = l_p
        return self.boardState
