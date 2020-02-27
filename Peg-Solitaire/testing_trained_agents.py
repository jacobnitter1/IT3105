import numpy as np
import Environment
import Agent
import matplotlib.pyplot as plt
import random

PLAY_GAME = 1
PLAY_ALL_GAMES = 0
PLOT_TRAINING = 0

BOARD_SHAPE = "Triangle" # or "Triangle"
BOARD_SIZE = 5 #any number
PLACEMENT_HOLES = [4]
NUM_EPISODES = 2_00
NUM_STEPS=25
CRITIC_TYPE = "Tabular" #or "Tabular"
NN_STRUCTURE =[5,5,5]
ACTOR_LEARNING_RATE = 0.1
CRITIC_LEARNING_RATE=0.1
ACTOR_E_DECAY_RATE = 0.9
CRITIC_E_DECAY_RATE =0.9
EPSILON = 0.5
EPSILON_DECAY_PR_EPISODE = 0.0001
def epsilon_greedy(ep_num):
    if ep_num <= 0.25*NUM_EPISODES:
        return 0.3
    elif ep_num <= 0.5*NUM_EPISODES:
        return 0.2
    elif ep_num <= 0.75*NUM_EPISODES:
        return 0.1
    else:
        return 0.05

DISPLAY_TRAINED_POLICY = True
DELAY_BETWEEN_MOVE = 1

def random_init(environment):
    pop = list(range(0,environment.get_obs_space()))
    placementHoles = random.sample(pop, np.random.randint(1,int(environment.get_obs_space()/1.5)))
    return placementHoles

def show_policy(shape, size, placement_holes, agent,type):
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
    print(len(t_states))
    vis = Environment.VisualizePegSolitaire(env.get_org_boardState(),shape)
    vis.show_played_game(t_states,t_actions,DELAY_BETWEEN_MOVE,shape,size,type)

if PLAY_GAME:
    shape = BOARD_SHAPE
    size = BOARD_SIZE
    placementHoles = PLACEMENT_HOLES


    name = str(shape+"_"+str(size)+"_"+CRITIC_TYPE+"_hole_place_"+str(placementHoles[0])+".npy")

    policy = np.load(name)
    numHoles = len(placementHoles)
    env = Environment.PegSolitaire(shape,size,numHoles,placementHoles)
    obs_space = env.get_obs_space()
    agent = Agent.Agent(critic_type =CRITIC_TYPE, NN_structure=NN_STRUCTURE, obs_space= obs_space, action_space = 6 , critic_lerning_rate= CRITIC_LEARNING_RATE, actor_learning_rate = ACTOR_LEARNING_RATE, critic_e_decay_rate = CRITIC_E_DECAY_RATE, actor_e_decay_rate=ACTOR_E_DECAY_RATE, gamma=0.85)
    #agent = Agent.Agent(critic_type[c],[25,25,25], obs_space = obs_space, action_space = 6, gamma = 0.9, lamda = 0.85)
    agent.set_policy(policy)

    show_policy(shape,size,placementHoles,agent,CRITIC_TYPE)




if PLAY_ALL_GAMES:
    game_type=[["Diamond",4,10],["Diamond",4,5],["Triangle",5,4],["Triangle",5,7],["Triangle",5,8]]
    critic_type=["NN","Tabular"]

    for g in range(0,len(game_type)):
        for c in range(0,len(critic_type)):
            shape = game_type[g][0]
            size = game_type[g][1]
            placementHoles = [game_type[g][2]]
            type = critic_type[c]
            name2 = str(shape+"_"+str(size)+"_"+type+"_hole_place_"+str(placementHoles[0])+".npy")
            #name2=str(game_type[g][0]+"_"+str(game_type[g][2])+"__board_size"+str(game_type[g][1])+"_"+critic_type[c]+".npy")
            print(name2)
            policy = np.load(name2)
            if True:
                numHoles = len(placementHoles)
                env = Environment.PegSolitaire(shape,size,numHoles,placementHoles)
                obs_space = env.get_obs_space()
                agent = Agent.Agent(critic_type =type, NN_structure=NN_STRUCTURE, obs_space= obs_space, action_space = 6 , critic_lerning_rate= CRITIC_LEARNING_RATE, actor_learning_rate = ACTOR_LEARNING_RATE, critic_e_decay_rate = CRITIC_E_DECAY_RATE, actor_e_decay_rate=ACTOR_E_DECAY_RATE, gamma=0.85)
                #agent = Agent.Agent(critic_type[c],[25,25,25], obs_space = obs_space, action_space = 6, gamma = 0.9, lamda = 0.85)
                agent.set_policy(policy)

                show_policy(shape,size,placementHoles,agent,type)
