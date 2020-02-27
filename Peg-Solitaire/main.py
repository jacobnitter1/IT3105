
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
BOARD_SHAPE = "Triangle" # or "Triangle"
BOARD_SIZE = 5 #any number
PLACEMENT_HOLES = [7]
NUM_EPISODES = 150
NUM_STEPS=1_000
CRITIC_TYPE = "NN" #or "Tabular"
NN_STRUCTURE =[25,25,25]
ACTOR_LEARNING_RATE = 0.01
CRITIC_LEARNING_RATE=0.01
ACTOR_E_DECAY_RATE = 0.01
CRITIC_E_DECAY_RATE =0.1
GAMMA = 0.9
EPSILON = 0.4
EPSILON_DECAY_PR_EPISODE = 1
def epsilon_greedy(ep_num,epsilon):
    if ep_num < 0.5*NUM_EPISODES:
        return epsilon
    else:
        epsilon = EPSILON_DECAY_PR_EPISODE*epsilon
    return epsilon

DISPLAY_TRAINED_POLICY = True
DELAY_BETWEEN_MOVE = 0.3
### PIVOTAL ARGUMENTS ###

TRAIN_TABULAR = 0
TESTING_NN = 0

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
    #print(t_states)
    vis = Environment.VisualizePegSolitaire(env.get_org_boardState(),shape)
    vis.show_played_game(t_states,t_actions,DELAY_BETWEEN_MOVE,shape,size,type)

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
    #print("env made")
    obs_space= env.get_obs_space()
    agent = Agent.Agent(critic_type = CRITIC_TYPE, NN_structure=NN_STRUCTURE, obs_space= obs_space, action_space = 6 , critic_lerning_rate= CRITIC_LEARNING_RATE, actor_learning_rate = ACTOR_LEARNING_RATE, critic_e_decay_rate = CRITIC_E_DECAY_RATE, actor_e_decay_rate=ACTOR_E_DECAY_RATE, gamma=GAMMA)
    #print("agent made")
    episode_num_pegs_left=[]
    for e in range(0,NUM_EPISODES):

        if e%10 == 0:
            agent.reset_memories()
            epsilon = EPSILON
            placement_holes = PLACEMENT_HOLES
            env.reset(numHoles=len(placement_holes),placementHoles = placement_holes)

            state = env.get_boardState()
            done = env.is_game_done()
            num_pegs_left = 0
            print("Episode number ", e, " finished.")
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
            #print(env.num_pegs_left()," pegs were left.")
            #if int(env.num_pegs_left()) == 1 :
                #agent.save_policy(BOARD_SHAPE, BOARD_SIZE, str(CRITIC_TYPE+"_hole_place"+str(PLACEMENT_HOLES[0])))
                #print("Found solutions after ",e," episodes!")
                #np.save(str(BOARD_SHAPE+str(BOARD_SIZE)+ str(CRITIC_TYPE+"_hole_place")+str(PLACEMENT_HOLES[0])),episode_num_pegs_left)
                #break

        agent.reset_memories()
        epsilon = EPSILON
        placement_holes = PLACEMENT_HOLES
        env.reset(numHoles=len(placement_holes),placementHoles = placement_holes)

        state = env.get_boardState()
        done = env.is_game_done()
        num_pegs_left = 0

        for s in range(0,NUM_STEPS):
                epsilon = epsilon_greedy(e,epsilon)
                if np.random.rand() < epsilon:
                    actions = env.get_legal_actions(obs_space)
                    #actions = env.get_action_space()
                    #action_idx = np.random.randint(0,len(actions))
                    if len(actions) == 0:
                        print(move_done, reward, state, ended)
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
                    episode_num_pegs_left.append(int(env.get_num_pegs_left()))
                    break
                state=state_next


    if DISPLAY_TRAINED_POLICY:
        #n=NUM_EPISODES//100
        #episode_num_pegs_left2 = [sum(episode_num_pegs_left[i:i+n])/n for i in range(0,len(episode_num_pegs_left),n)]
        plt.plot(episode_num_pegs_left)
        plt.show()
        plt.pause(3)
        plt.close()

        show_policy(BOARD_SHAPE,BOARD_SIZE,PLACEMENT_HOLES,agent,CRITIC_TYPE)
        plt.close()
#test_GUI()

train_one(True)
