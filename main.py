
import Environment
import Agent
import numpy as np

boardShape = "Triangle" # "Diamond" or "Triangle"
boardSize = 3 # Must be int of size 3 or larger
placementHoles = [0] #Preferrably center
numHoles = len(placementHoles)
visualizeGameBool = False # Boolean
fps = 400 # how many ms between each frame when playing game, if None it waits for keyboard input
AI_player = False #Bool, if False keyboard is playing
gamma = 0.9
alpha = 0.2
lamda = 0.5

PegSolitaire = Environment.PegSolitaire(boardShape, boardSize, numHoles, placementHoles)
agent = Agent.agent()

for e in range(0,NUM_EPISODES):
    agent.reset_memories()
    state = PegSolitaire.reset()
    done = PegSolitaire.is_game_done()
    while not done:
        action =agent.get_action(state)
        state_next = PegSolitaire.move
        done = PegSolitaire.is_game_done()



















#PegSolitaire.print_boardState(
board = PegSolitaire.get_org_boardState()


boardState=PegSolitaire.get_boardState()
print(boardState[0])
vis = Environment.VisualizePegSolitaire(board,boardShape)
#print("boardstate:",len(PegSolitaire.get_boardState()),"actionspace",len(PegSolitaire.get_action_space()))
actor = Agent.Actor_tab(PegSolitaire.get_boardState(),PegSolitaire.get_action_space())
q = actor.get_table()
print(PegSolitaire.get_boardState())
print(actor.get_action(PegSolitaire.get_boardState(),0.001))



#vis.drawBoard(PegSolitaire.get_org_boardState(),PegSolitaire.get_lastAction())
#PegSolitaire.check()
#PegSolitaire.move_peg(2,5)
#print("-------------------- BOARDSTATE")
#PegSolitaire.print_boardState()


#print("check github")
#vis.drawBoard(PegSolitaire.get_org_boardState(),PegSolitaire.get_lastAction())
#print("_>_>_>__>_>__>_>_",PegSolitaire.get_boardState())
