import math
from BasicClientActorAbs import BasicClientActorAbs
import Environment
import Agent
import numpy as np

class BasicClientActor(BasicClientActorAbs):

    def __init__(self, IP_address=None, verbose=False):
        self.series_id = -1
        self.game = None
        self.policy_network = None
        self.board_size = -1
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)

    def server_state_to_NN_state(self,srv_state):
        NN_state=[]
        for i in range(1,len(srv_state)):
            #print(i)
            if srv_state[i] == 0:
                NN_state.append(0)
                NN_state.append(0)
            elif srv_state[i] == 1:
                NN_state.append(1)
                NN_state.append(0)
            elif srv_state[i] == 2:
                NN_state.append(0)
                NN_state.append(1)
        #APPEND PLAYER AT END
        if srv_state[0] == self.starting_player:
            NN_state.append(1)
            NN_state.append(0)
        else:
            NN_state.append(0)
            NN_state.append(1)
        return NN_state

    def NN_state_to_server_state(self,NN_state):
        srv_state=[]
        if NN_state[-2] == 0:
            if NN_state[-1] == 1:
                srv_state.append(2)
        else:
            srv_state.append(1)
        for i in range(0,len(NN_state)-2,2):
            #print(i)
            if NN_state[i] == 0:
                if NN_state[i+1] == 1:
                    srv_state.append(2)
                else:
                    srv_state.append(0)
            else:
                srv_state.append(1)


        return srv_state

    def action_to_tuple_action(self,action):
        for i in range(0,self.board_size):
            for j in range(0,self.board_size):
                if action == i*self.board_size+j:
                    return (i,j)

    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """

        # This is an example player who picks random moves. REMOVE THIS WHEN YOU ADD YOUR OWN CODE !!

        #next_move = tuple(self.pick_random_free_cell(
        #    state, size=int(math.sqrt(len(state)-1))))
        #############################
        #
        #
        NN_state = self.server_state_to_NN_state(state)
        predictions = self.policy_network.predict([[NN_state]])
        next_move = np.argmax(predictions)
        self.game.set_state(NN_state,1)
        legal_actions = self.game.get_legal_actions()
        if next_move not in legal_actions:
            next_move = np.random.choice(legal_actions,1)
        next_move = self.action_to_tuple_action(next_move)

        #
        # next_move = ???
        ##############################
        return next_move

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_id = series_id
        self.board_size = game_params[0]
        #############################
        #
        #
        learning_rate = 0.001
        NN_structure = [128,128,128]
        optimizer_ = 'Adam'
        activation_function_ = 'sigmoid'
        self.game = Environment.HexGame("Diamond",game_params[0],0)
        self.policy_network = Agent.Policy_Network(game_params[0], lr = learning_rate, nn_struct = NN_structure, activation_function = activation_function_,optimizer = optimizer_, conv_bool=False)
        self.policy_network.load_weights("./demo_agents/"+str(game_params[0]))
        #
        #
        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state: ' + str(end_state))

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))

        #self.game.set_state(,game.get_last_player())


if __name__ == '__main__':
    bsa = BasicClientActor(verbose=True)
    bsa.connect_to_server()
