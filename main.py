
import Environment
import Agent
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

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
NUM_EPISODES = 2_00
NUM_STEPS = 25
TRAIN_BOOL = True
method = "tab"


#vis = Environment.VisualizePegSolitaire(PegSolitaire.get_org_boardState(),boardShape)
TRAIN_TABULAR = 0
TESTING_NN = 1

def random_init(environment):
    pop = list(range(0,environment.get_obs_space()))
    placementHoles = random.sample(pop, np.random.randint(1,int(environment.get_obs_space()/1.5)))
    return placementHoles

def visualize_game(agent, environment, vis):
    idx = np.random.randint(0,environment.get_obs_space())
    placementHoles=[idx]
    PegSolitaire.reset(numHoles=len(placementHoles), placementHoles = placementHoles)
    #print(PegSolitaire.get_org_boardState())

    state = PegSolitaire.get_boardState()
    done = PegSolitaire.is_game_done()
    states = []
    actions = []
    states.append(PegSolitaire.get_org_boardState())
    actions.append([None,None])
    while not done[0]:
        #vis.drawBoard(PegSolitaire.get_org_boardState(),PegSolitaire.get_lastAction())
        action =agent.get_action(state,0.0) # FIK
        move_done, reward, state_next, ended = PegSolitaire.move_peg(action[0],action[1])
        if done[0]:
            print("No more moves")
            break
        if state == state_next:
            print("illegal move")
            break
        states.append(PegSolitaire.get_org_boardState())
        print("checkhcekc",PegSolitaire.get_org_boardState())
        actions.append(PegSolitaire.get_lastAction())
        state = state_next
    print(len(states))
    vis.show_played_game(states,actions)

def epsilon_greedy(ep_num):
    if ep_num <= 0.25*NUM_EPISODES:
        return 0.5
    elif ep_num <= 0.5*NUM_EPISODES:
        return 0.3
    elif ep_num <= 0.75*NUM_EPISODES:
        return 0.15
    else:
        return 0.05

def train_tabular(shape,size, gamma, alpha, lamda):
    placementHoles=[0]
    PegSolitaire = Environment.PegSolitaire(shape, size, numHoles, placementHoles)
    obs_space= PegSolitaire.get_obs_space()
    agent = Agent.Agent("Tabular", [None, None,None], obs_space= obs_space, action_space = 6 , gamma = gamma, alpha = alpha, lamda = lamda)

    episode_returns=[]
    episode_num_steps = []
    for e in range(0,NUM_EPISODES):
        agent.reset_memories()
        placementHoles=random_init(PegSolitaire)
        PegSolitaire.reset(numHoles=len(placementHoles), placementHoles = placementHoles)
        #print(PegSolitaire.get_org_boardState())
        state = PegSolitaire.get_boardState()
        done = PegSolitaire.is_game_done()
        returned = 0
        num_steps_taken = 0
        while not done[0]:
            if np.random.rand() < epsilon_greedy(e):
                actions = PegSolitaire.get_legal_actions(obs_space)
                idx = np.random.randint(0,len(actions))
                action = actions[idx]
            else:
                print("chosen")
                action =agent.get_action(state,0.0)
            print(PegSolitaire.get_org_boardState())
            move_done, reward, state_next, ended = PegSolitaire.move_peg(action[0],action[1])
            print(PegSolitaire.get_org_boardState())

            if not move_done:
                agent.update_illegal_move(state,action)

            else:
                num_steps_taken+=1
                returned += reward
                agent.update_trajectories(state, action)
                agent.update_agent(state, state_next, reward)
                done = PegSolitaire.is_game_done()
                print(done)
                if done[0]:
                    episode_num_steps.append(num_steps_taken)
                    break
                state=state_next
            episode_returns.append(returned)
    return episode_returns, episode_num_steps, agent, PegSolitaire

def train_pretrained_tabular(shape,size, gamma, alpha, lamda,name):
    placementHoles=[0]
    PegSolitaire = Environment.PegSolitaire(shape, size, numHoles, placementHoles)
    obs_space= PegSolitaire.get_obs_space()
    agent = Agent.Agent("Tabular", [None, None,None], obs_space= obs_space, action_space = 6 , gamma = gamma, alpha = alpha, lamda = lamda)
    policy = np.load(name)
    agent.set_policy(policy)

    episode_returns=[]
    episode_num_pegs_left = []
    for e in range(0,NUM_EPISODES):
        agent.reset_memories()
        #placementHoles=random_init(PegSolitaire)
        placementHoles = random.sample([4,5,6,7],1)
        PegSolitaire.reset(numHoles=len(placementHoles), placementHoles = placementHoles)
        #print(PegSolitaire.get_org_boardState())
        state = PegSolitaire.get_boardState()
        done = PegSolitaire.is_game_done()
        returned = 0
        num_steps_taken = 0
        while not done[0]:
            if np.random.rand() < epsilon_greedy(e):
                actions = PegSolitaire.get_legal_actions(obs_space)
                idx = np.random.randint(0,len(actions))
                action = actions[idx]
            else:
                print("chosen")
                action =agent.get_action(state,0.0)
            print(PegSolitaire.get_org_boardState())
            move_done, reward, state_next, ended = PegSolitaire.move_peg(action[0],action[1])
            print(PegSolitaire.get_org_boardState())

            if not move_done:
                agent.update_illegal_move(state,action)

            else:
                num_steps_taken+=1
                returned += reward
                agent.update_trajectories(state, action)
                agent.update_agent(state, state_next, reward)
                done = PegSolitaire.is_game_done()
                print(done)
                if done[0]:
                    episode_num_pegs_left.append(PegSolitaire.num_pegs_left())
                    break
                state=state_next
                episode_returns.append(returned)
                episode_num_pegs_left.append(PegSolitaire.num_pegs_left())
    return episode_returns, episode_num_pegs_left, agent, PegSolitaire

PegSolitaire = Environment.PegSolitaire(boardShape, boardSize, numHoles, placementHoles)

obs_space= PegSolitaire.get_obs_space()
agent = Agent.Agent("Tabular", [25,25,25], obs_space= obs_space, action_space = 6 , gamma = 0.9, alpha = 0.01, lamda = 0.85)

if TRAIN_TABULAR:
    boardShapes = ["Diamond", "Triangle"]
    boardSizes = [3,4]
    for shape in boardShapes:
        for size in boardSizes:
            loading_name = shape +"_"+str(size)+"_tab.npy"
            episode_returns, episode_num_pegs_left, agent, PegSolitaire = train_pretrained_tabular(shape,size,0.9,0.1,0.7,loading_name)
            saving_name = shape +"_"+str(size)+"_tab"
            np.save(str(saving_name+"returns"),episode_returns)
            np.save(str(saving_name+"pegs_left"),episode_num_pegs_left)
            agent.save_policy(shape,size,"tab_v2")
        #vis = Environment.VisualizePegSolitaire(PegSolitaire.get_org_boardState(),boardShape)
        #visualize_game(agent, PegSolitaire, vis)
#agent.save_policy(boardShape,boardSize,method)
if False:
    shape = "Triangle"
    size = 5

    loading_name = shape +"_"+str(size)+"_tab.npy"
    episode_returns, episode_num_pegs, agent, PegSolitaire = train_pretrained_tabular(shape,size,0.9,0.1,0.7,loading_name)

    saving_name = shape +"_"+str(size)+"_tab"
    #np.save(str(saving_name+"returns"),episode_returns)
    #np.save(str(saving_name+"pegs"),episode_num_pegs)
    #plt.plot(episode_num_pegs)
    plt.show()
    agent.save_policy(shape,size,"tab")
    vis = Environment.VisualizePegSolitaire(PegSolitaire.get_org_boardState(),shape)
    visualize_game(agent, PegSolitaire,vis)


def train_NN(shape,size,NN_structure,gamma, alpha, lamda):
    placementHoles = [0]
    numHoles = len(placementHoles)
    PegSolitaire = Environment.PegSolitaire(shape,size,numHoles,placementHoles)
    print("NN dim : ", NN_structure, " obs space :", PegSolitaire.get_obs_space())
    agent = Agent.Agent("NN", NN_structure, PegSolitaire.get_obs_space(),6,gamma, alpha,lamda)

    episode_num_pegs_left = []

    for e in range(0,NUM_EPISODES):
        if e%100 == 0:
            print("Episode number ", e, " finished.")
        agent.reset_memories()
        placementHoles=random_init(PegSolitaire)
        PegSolitaire.reset(numHoles=len(placementHoles), placementHoles = placementHoles)
        state = PegSolitaire.get_boardState()
        done = PegSolitaire.is_game_done()
        num_pegs_left = 0
        while not done[0]:
            if np.random.rand() < epsilon_greedy(e):
                actions = PegSolitaire.get_legal_actions(obs_space)
                idx = np.random.randint(0,len(actions))
                action = actions[idx]
            else:
                #print("chosen")
                action =agent.get_action(state,0.0)
            #print(PegSolitaire.get_org_boardState())
            move_done, reward, state_next, ended = PegSolitaire.move_peg(action[0],action[1])
            #print(PegSolitaire.get_org_boardState())

            if not move_done:
                agent.update_illegal_move(state,action)

            else:
                #returned += reward
                agent.update_trajectories(state, action)
                agent.update_agent(state, state_next, reward)
                done = PegSolitaire.is_game_done()
                #print(done)
                if done[0]:

                    #episode_returns.append(returned)
                    episode_num_pegs_left.append(PegSolitaire.num_pegs_left())
                    break
                state=state_next

    return [], episode_num_pegs_left, agent, PegSolitaire

if TESTING_NN:
    shape ="Diamond"
    size=4
    placementHoles = [0]
    numHoles = len(placementHoles)
    PegSolitaire = Environment.PegSolitaire(shape,size,numHoles,placementHoles)
    obs_space = PegSolitaire.get_obs_space()
    placementHoles = random_init(PegSolitaire)

    episode_returns, episode_num_pegs, agent, PegSolitaire = train_NN(shape,size,[5,5,5],0.9,0.01,0.85)
    agent.save_policy(shape,size, "NN")
    plt.plot(episode_num_pegs)
    plt.show()
    vis = Environment.VisualizePegSolitaire(PegSolitaire.get_org_boardState(),shape)
    visualize_game(agent, PegSolitaire,vis)
    iteration = 0
