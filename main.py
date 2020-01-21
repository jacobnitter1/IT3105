
import Environment

boardShape = "Diamond" # "Diamond" or "Triangle"
boardSize = 5 # Must be int of size 3 or larger
placementHoles = [0] #Preferrably center
numHoles = len(placementHoles)
visualizeGameBool = False # Boolean
fps = 400 # how many ms between each frame when playing game, if None it waits for keyboard input
AI_player = False #Bool, if False keyboard is playing
gamma = 0.9
alpha = 0.2
lamda = 0.5

PegSolitaire = Environment.PegSolitaire(boardShape, boardSize, numHoles, placementHoles)
PegSolitaire.print_boardState()
#print(PegSolitaire.moves_left())

board = PegSolitaire.get_org_boardState()
vis = Environment.VisualizePegSolitaire(board,boardShape)
vis.drawBoard(PegSolitaire.get_org_boardState(),PegSolitaire.get_lastAction())
#PegSolitaire.check()
PegSolitaire.move_peg(2,5)
PegSolitaire.print_boardState()


vis.drawBoard(PegSolitaire.get_org_boardState(),PegSolitaire.get_lastAction())