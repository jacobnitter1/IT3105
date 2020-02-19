
import Environment
import Agent
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()
##Tabular
# Triangle size 5 with holes [4], x episodes :
# Diamond size 4 with holes[x], x episodes :

## NN:
# Triangle size 5 with holes [4], x episodes :
# Diamond size 4 with holes[x], x episodes :

### PIVOTAL ARGUMENTS ###
BOARD_SHAPE = "Diamond" # or "Triangle"
BOARD_SIZE = 5 #any number
PLACEMENT_HOLES = [6]
NUM_EPISODES = 100
NUM_STEPS=100
CRITIC_TYPE = "Tabular" #or "Tabular"
NN_STRUCTURE =[100,50,25]
ACTOR_LEARNING_RATE = 0.3
CRITIC_LEARNING_RATE=0.1
ACTOR_E_DECAY_RATE = 0.8
CRITIC_E_DECAY_RATE =0.8
GAMMA = 0.8
EPSILON = 0.3
EPSILON_DECAY_PR_EPISODE = 0.0001
def epsilon_greedy(ep_num):
    if ep_num <= 0.25*NUM_EPISODES:
        return 0.5
    elif ep_num <= 0.5*NUM_EPISODES:
        return 0.4
    elif ep_num <= 0.75*NUM_EPISODES:
        return 0.1
    elif ep_num <=0.95*NUM_EPISODES:
        return 0.1
    else:
        return 0.05

DISPLAY_TRAINED_POLICY = True
DELAY_BETWEEN_MOVE = 2
### PIVOTAL ARGUMENTS ###

TRAIN_TABULAR = 0
TESTING_NN = 0

def test_GUI():
    placementHoles=[6]
    numHoles = len(placementHoles)
    boardShape="Diamond"
    boardSize = 4

    t_states = []
    t_actions = []
    env = Environment.PegSolitaire(boardShape,boardSize,numHoles,placementHoles)
    t_states.append(np.copy(env.get_org_boardState()))
    t_actions.append(np.copy(env.get_lastAction()))
    for i in range(0,NUM_STEPS):

        pos_actions =env.get_legal_actions(env.get_obs_space())
        if len(pos_actions)==0:
            break
        idx = np.random.randint(0,len(pos_actions))
        action = pos_actions[idx]

        move_done, reward, state_next,done = env.move_peg(action[0],action[1])
        t_states.append(np.copy(env.get_org_boardState()))
        t_actions.append(np.copy(env.get_lastAction()))
    vis = Environment.VisualizePegSolitaire(env.get_org_boardState(),boardShape)
    vis.show_played_game(t_states,t_actions,DELAY_BETWEEN_MOVE)

def show_policy(shape, size, placement_holes, agent):
    numHoles = len(placement_holes)

    t_states = []
    t_actions = []
    env = Environment.PegSolitaire(shape,size,numHoles,placement_holes)
    t_states.append(np.copy(env.get_org_boardState()))
    t_actions.append(np.copy(env.get_lastAction()))
    for i in range(0,NUM_STEPS):
        action = agent.get_action(env.get_boardState(),0.0)
        move_done, reward, state_next,done = env.move_peg(action[0],action[1])
        t_states.append(np.copy(env.get_org_boardState()))
        t_actions.append(np.copy(env.get_lastAction()))
        if done or not move_done:
            print(i," transitions were made.")
            break
    print(t_states)
    vis = Environment.VisualizePegSolitaire(env.get_org_boardState(),shape)
    vis.show_played_game(t_states,t_actions,DELAY_BETWEEN_MOVE)

def random_init(environment):
    pop = list(range(0,environment.get_obs_space()))
    placementHoles = random.sample(pop, np.random.randint(1,int(environment.get_obs_space()/1.5)))
    return placementHoles

def visualize_game(agent, environment, vis):
    idx = np.random.randint(0,environment.get_obs_space())
    placementHoles=[6]
    PegSolitaire.reset(numHoles=len(placementHoles), placementHoles = placementHoles)
    #print(PegSolitaire.get_org_boardState())

    state = PegSolitaire.get_boardState()
    done = [PegSolitaire.is_game_done(),False]
    states = []
    actions = []
    states.append(PegSolitaire.get_org_boardState())
    actions.append([None,None])
    while not done[0]:
        #vis.drawBoard(PegSolitaire.get_org_boardState(),PegSolitaire.get_lastAction())
        action =agent.get_action(state,0.0) # FIK
        move_done, reward, state_next,state_org_next, done = PegSolitaire.move_peg(action[0],action[1]) #!!!! endred fra ended til done
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
    return 0.5
    if ep_num <= 0.25*NUM_EPISODES:
        return 0.5
    elif ep_num <= 0.5*NUM_EPISODES:
        return 0.5
    elif ep_num <= 0.75*NUM_EPISODES:
        return 0.3
    else:
        return 0.2

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
                #print("chosen")
                action =agent.get_action(state,0.0)
            #print(PegSolitaire.get_org_boardState())
            move_done, reward, state_next, ended = PegSolitaire.move_peg(action[0],action[1])
            #print(PegSolitaire.get_org_boardState())

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
                #print("chosen")
                action =agent.get_action(state,0.0)
            #print(PegSolitaire.get_org_boardState())
            move_done, reward, state_next, ended = PegSolitaire.move_peg(action[0],action[1])
            #print(PegSolitaire.get_org_boardState())

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
                    episode_num_pegs_left.append(PegSolitaire.num_pegs_left())
                    break
                state=state_next
                episode_returns.append(returned)
                episode_num_pegs_left.append(PegSolitaire.num_pegs_left())
    return episode_returns, episode_num_pegs_left, agent, PegSolitaire

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
        placementHoles=[6]#random_init(PegSolitaire)
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

def train_all_types():
    game_type=[["Diamond",4,9],["Diamond",4,6],["Triangle",5,4],["Triangle",5,7],["Triangle",5,8]]
    critic_type=["NN","Tabular"]
    for g in range(0,len(game_type)):
        for c in range(0,len(critic_type)):
            env = Environment.PegSolitaire(game_type[g][0], game_type[g][1], 1,[game_type[g][2]])
            obs_space= env.get_obs_space()
            agent = Agent.Agent(critic_type = critic_type[c], NN_structure=[25,25,25], obs_space= obs_space, action_space = 6 , critic_lerning_rate= CRITIC_LEARNING_RATE, actor_learning_rate = ACTOR_LEARNING_RATE, critic_e_decay_rate = CRITIC_E_DECAY_RATE, actor_e_decay_rate=ACTOR_E_DECAY_RATE, gamma=0.85)
            episode_num_pegs_left=[]
            for e in range(0,NUM_EPISODES):
                if e%100 == 0:
                    print("Episode number ", e, " finished.")
                env.reset(numHoles=len(PLACEMENT_HOLES),placementHoles = PLACEMENT_HOLES)
                state = env.get_boardState()
                done = env.is_game_done()
                num_pegs_left = 0
                for s in range(0,NUM_STEPS):
                        if np.random.rand() < epsilon_greedy(e):
                            actions = env.get_legal_actions(obs_space)
                            #actions = env.get_action_space()
                            #action_idx = np.random.randint(0,len(actions))
                            idx = np.random.randint(0,len(actions))
                            #peg = np.random.randint(0,obs_space)
                            #action = actions[idx]
                            action= actions[idx]
                        else:
                            action = agent.get_action(state,0.0)

                        move_done, reward, state_next, ended = env.move_peg(action[0],action[1])
                        if not move_done:
                            agent.update_illegal_move(state,action)

                        else:
                            #returned += reward
                            agent.update_trajectories(state, action)
                            agent.update_agent(state, state_next, reward)
                        if ended:
                            episode_num_pegs_left.append(int(env.num_pegs_left()))
                            break
                        state=state_next
            name = game_type[g][0]+"_boardsize_"+ str(game_type[g][1])+"_"+ str(game_type[g][2])+"_"+ critic_type[c]
            np.save(str(name + "_learning_plot"), episode_num_pegs_left)
            agent.save_policy(str(game_type[g][0]+"_"+ str(game_type[g][2])),str("_board_size")+str(game_type[g][1]), critic_type[c])


def train_one(DISPLAY_TRAINED_POLICY):
    env = Environment.PegSolitaire(BOARD_SHAPE, BOARD_SIZE, len(PLACEMENT_HOLES),PLACEMENT_HOLES)
    print("env made")
    obs_space= env.get_obs_space()
    agent = Agent.Agent(critic_type = CRITIC_TYPE, NN_structure=NN_STRUCTURE, obs_space= obs_space, action_space = 6 , critic_lerning_rate= CRITIC_LEARNING_RATE, actor_learning_rate = ACTOR_LEARNING_RATE, critic_e_decay_rate = CRITIC_E_DECAY_RATE, actor_e_decay_rate=ACTOR_E_DECAY_RATE, gamma=GAMMA)
    print("agent made")
    episode_num_pegs_left=[]
    for e in range(0,NUM_EPISODES):
        agent.reset_memories()
        #if np.random.rand()<= 0.3:
        #    placement_holes = random_init(env)
        #else:
        placement_holes = PLACEMENT_HOLES
        env.reset(numHoles=len(placement_holes),placementHoles = placement_holes)

        state = env.get_boardState()
        done = env.is_game_done()
        num_pegs_left = 0
        if e%20 == 0:
            print("Episode number ", e, " finished.")
            placement_holes = PLACEMENT_HOLES
            for s in range(0,NUM_STEPS):
                action = agent.get_action(state,0.0)

                move_done, reward, state_next, ended = env.move_peg(action[0],action[1])
                #print(state)
                #print(agent.get_TD_error(reward,state,state_next))
                if not move_done:
                    agent.update_illegal_move(state,action)
                    #agent.update_trajectories(state, action)
                    #agent.update_agent(state, state_next, reward)

                else:
                    #returned += reward
                    agent.update_trajectories(state, action)
                    agent.update_agent(state, state_next, reward)
                if ended:
                    episode_num_pegs_left.append(int(env.num_pegs_left()))
                    break
                state=state_next
            print(env.num_pegs_left()," pegs were left.")
            if int(env.num_pegs_left()) == 1:
                agent.save_policy(BOARD_SHAPE, BOARD_SIZE, str("NN_hole_place"+str(PLACEMENT_HOLES[0])))
                np.save(str(BOARD_SHAPE+str(BOARD_SIZE)+ str("NN_hole_place")+str(PLACEMENT_HOLES[0])),episode_num_pegs_left)
                break


        for s in range(0,NUM_STEPS):
                if np.random.rand() < epsilon_greedy(e):
                    actions = env.get_legal_actions(obs_space)
                    #actions = env.get_action_space()
                    #action_idx = np.random.randint(0,len(actions))
                    if len(actions) == 0:
                        episode_num_pegs_left.append(int(env.num_pegs_left()))
                        break

                    idx = np.random.randint(0,len(actions))
                    #peg = np.random.randint(0,obs_space)
                    #action = actions[idx]
                    action= actions[idx]
                else:
                    action = agent.get_action(state,0.0)

                move_done, reward, state_next, ended = env.move_peg(action[0],action[1])
                if not move_done:
                    agent.update_illegal_move(state,action)

                else:
                    #returned += reward
                    agent.update_trajectories(state, action)
                    agent.update_agent(state, state_next, reward)
                if ended:
                    episode_num_pegs_left.append(int(env.num_pegs_left()))
                    break
                state=state_next


    if DISPLAY_TRAINED_POLICY:
        plt.plot(episode_num_pegs_left)
        plt.show()
        plt.pause(0.3)
        plt.close()

        show_policy(BOARD_SHAPE,BOARD_SIZE,PLACEMENT_HOLES,agent)
        plt.close()
#test_GUI()
train_one(DISPLAY_TRAINED_POLICY)

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
    size = 6

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


if TESTING_NN:
    shape ="Diamond"
    size=4
    placementHoles = [6]
    numHoles = len(placementHoles)
    PegSolitaire = Environment.PegSolitaire(shape,size,numHoles,placementHoles)
    obs_space = PegSolitaire.get_obs_space()
    placementHoles = random_init(PegSolitaire)

    episode_returns, episode_num_pegs, agent, PegSolitaire = train_NN(shape,size,[5,5,5],0.9,0.01,0.85)
    #agent.save_policy(shape,size, "NN")
    plt.plot(episode_num_pegs)
    plt.show()
    show_policy(shape,size,[6],agent)
#test_GUI()
