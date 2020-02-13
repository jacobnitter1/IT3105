import numpy as np
import Environment
import Agent
import matplotlib.pyplot as plt

PLAY_ALL_GAMES = False
PLOT_TRAINING = False

boardShapes = ["Diamond", "Triangle"]
boardSizes = [3,4]

NUM_STEPS = 25
if PLAY_ALL_GAMES:
    for shape in boardShapes:
        for size in boardSizes:
            name = shape+"_"+str(size)+"_tab.npy"

            placementHoles = [0]
            numHoles = len(placementHoles)
            PegSolitaire = Environment.PegSolitaire(shape,size,numHoles,placementHoles)
            obs_space = PegSolitaire.get_obs_space()
            random_inits = list(range(0,obs_space))
            #print(random_inits)

            agent = Agent.Agent("Tabular",[1,1,1], obs_space = obs_space, action_space = 6, gamma = 0.9, alpha = 0.01, lamda = 0.85)
            vis = Environment.VisualizePegSolitaire(PegSolitaire.get_org_boardState(),shape)

            idx = np.random.randint(0,len(random_inits))
            placementHoles=[random_inits[idx]]
            state =PegSolitaire.reset(numHoles=len(placementHoles), placementHoles = placementHoles)
            vis.drawBoard(state,[None,None])

            state= PegSolitaire.get_boardState()

            for s in range(0,NUM_STEPS):
                action =agent.get_action(state,0.0)
                move_done, reward, state_next, ended = PegSolitaire.move_peg(action[0],action[1])
                if not move_done:
                    print("ILLEGAL ACTION :( ", state, action)
                    break
                else:
                    print("OK ACTION :)", state,action)

                    vis.drawBoard(PegSolitaire.get_org_boardState(), PegSolitaire.get_lastAction())
                    state = state_next
                    lastAction = action

if PLOT_TRAINING:

        for shape in boardShapes:
            for size in boardSizes:
                name = shape+"_"+str(size)+"_tab.npy"
                num_steps = np.load("episode_num_steps_"+name)
                returns = np.load("episode_return_"+name)
                n=100
                num_steps_avg = [sum(num_steps[i:i+n])//n for i in range(0,len(num_steps),n)]
                returns_avg = list2 = [sum(returns[i:i+n])//n for i in range(0,len(returns),n)]
                plt.figure(figsize=(60,60))#figsize=(124, 248)

                plt.subplot(131)
                plt.plot(num_steps_avg)
                plt.ylabel("Num steps avg every "+str(n)+" episode.")

                plt.subplot(132)
                plt.plot(returns_avg)
                plt.ylabel("Return avg every "+str(n)+" episode.")
                plt.suptitle(shape+" "+ str(size))
                plt.show()
