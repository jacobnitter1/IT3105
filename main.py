
import Environment
import Agent
import numpy as np

import matplotlib.pyplot as plt

boardShape = "Diamond" # "Diamond" or "Triangle"
boardSize = 4 # Must be int of size 3 or larger
placementHoles = [1,2,4,5] #Preferrably center
placementHoles = [2]
numHoles = len(placementHoles)
visualizeGameBool = False # Boolean
fps = 400 # how many ms between each frame when playing game, if None it waits for keyboard input
AI_player = False #Bool, if False keyboard is playing
gamma = 0.9
alpha = 0.2
lamda = 0.5
random_inits =[[5],[0],[3]]#,[5,4],[0,2],[1,3],[0,5],[0,3],[1,2,3],[2,3],[4]
NUM_EPISODES = 1000
NUM_STEPS = 25
TRAIN_BOOL = False
method = "tab"

#PegSolitaire = Environment.PegSolitaire(boardShape, boardSize, numHoles, placementHoles)

#obs_space= PegSolitaire.get_obs_space()
#agent = Agent.Agent("Tabular", [25,25,25], obs_space= obs_space, action_space = 6 , gamma = 0.9, alpha = 0.01, lamda = 0.85)
#vis = Environment.VisualizePegSolitaire(PegSolitaire.get_org_boardState(),boardShape)
test= False

boardShapes = ["Diamond", "Triangle"]
boardSizes = [3,4]

for shape in boardShapes:
    for size in boardSizes:
        placementHoles = [0]
        numHoles = len(placementHoles)
        PegSolitaire = Environment.PegSolitaire(shape,size,numHoles,placementHoles)
        obs_space = PegSolitaire.get_obs_space()
        random_inits = list(range(0,obs_space))
        #print(random_inits)

        agent = Agent.Agent("Tabular",[1,1,1], obs_space = obs_space, action_space = 6, gamma = 0.9, alpha = 0.01, lamda = 0.85)

        episode_returns=[]
        episode_num_steps = []
        iteration = 0
        for e in range(0,NUM_EPISODES):
            if e%1000 == 0:
                print(e)
                #agent.save_policy(shape,size,"tab")
                name = "episode_return_"+shape+"_"+str(size)+"_tab_"+str(iteration)
                #np.save(name,episode_returns)
                name = "episode_num_steps_"+shape+"_"+str(size)+"_tab_"+str(iteration)
                #np.save(name, episode_num_steps)
                print(shape, size, "trained through iteration ", iteration)
            for step in range(0,NUM_STEPS):
                agent.reset_memories()
                idx = np.random.randint(0,len(random_inits))
                placementHoles=[random_inits[idx]]
                PegSolitaire.reset(numHoles=len(placementHoles), placementHoles = placementHoles)
                state = PegSolitaire.get_boardState()
                done = PegSolitaire.is_game_done()
                returned = 0
                num_steps_taken = 0
                while not done[0]:
                    if np.random.rand() < 0.4:
                        legal_actions = PegSolitaire.get_legal_actions(obs_space)
                        idx = np.random.randint(0,len(legal_actions))
                        action = legal_actions[idx]
                    else:
                        action =agent.get_action(state,0.0)
                    move_done, reward, state_next, ended = PegSolitaire.move_peg(action[0],action[1])

                    if not move_done:
                        agent.update_illegal_move(state,action)

                    else:
                        num_steps_taken+=1
                        returned += reward
                        agent.update_trajectories(state, action)
                        agent.update_agent(state, state_next, reward)
                        done = PegSolitaire.is_game_done()
                        #print(done)
                        if done[0]:
                            episode_num_steps.append(num_steps_taken)
                            break
                        state=state_next
                    episode_returns.append(returned)


    #states, actions = agent.get_replay_trajectories()

    #vis.show_played_game(states,actions)
if TRAIN_BOOL:
    episode_returns=[]
    episode_num_steps = []
    for e in range(0,NUM_EPISODES):
        agent.reset_memories()
        idx = np.random.randint(0,len(random_inits))
        placementHoles=random_inits[idx]
        PegSolitaire.reset(numHoles=len(placementHoles), placementHoles = placementHoles)
        #print(PegSolitaire.get_org_boardState())
        state = PegSolitaire.get_boardState()
        done = PegSolitaire.is_game_done()
        returned = 0
        #agent.print_q_values(state)
        num_steps_taken = 0
        while not done[0]:
            #print("her")
            #print(state)
            if np.random.rand() < 0.4:
                print("random")
                list = PegSolitaire.get_legal_actions(obs_space)
                print("LEGAL MOVES ",list)
                idx = np.random.randint(0,len(list))
                action = list[idx]
            else:
                print("chosen")
                action =agent.get_action(state,0.0) # FIKS EPSILON
            #action = [5,5]
            #print("Chosen action " ,action)
            print(PegSolitaire.get_org_boardState())
            move_done, reward, state_next, ended = PegSolitaire.move_peg(action[0],action[1])
            print(PegSolitaire.get_org_boardState())

            if not move_done:
                #print("illegal move")
                agent.update_illegal_move(state,action)

            else:
                #print("Sucessfull move")
                #print(move_done," gives reward ", reward)
                #print(move_done,action,reward)
                #vis.drawBoard(PegSolitaire.get_org_boardState(), PegSolitaire.get_lastAction())
                num_steps_taken+=1
                returned += reward
                agent.update_trajectories(state, action)
                agent.update_agent(state, state_next, reward)
                done = PegSolitaire.is_game_done()
                print(done)
                if done[0]:
                    episode_num_steps.append(num_steps_taken)
                    break

                #agent.print_q_values(state)
                state=state_next
            episode_returns.append(returned)
    plt.plot(episode_returns)
    plt.show()
    plt.plot(episode_num_steps)
    plt.show()

agent.save_policy(boardShape,boardSize,method)
