from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import _pick_unused_card, _fill_community_card, gen_cards
import Brain
import time
from pprint import pprint
import numpy as np



class RLPlayer(BasePokerPlayer):
    
    def __init__(self):

        self.agentName = 'Agent'
        self.resetVariables()

        self.brain1 = Brain.brain(4,4)
        #self.brain2 = Brain2.brain(4,1000)

    def resetVariables(self):

        self.playerMoves = []
        self.playerCards = []
        self.communityCards = []
        self.playerList = []
        self.phases = ['preflop','flop','turn','river']
        self.uuids = []
        self.playerCount = 0
        self.street = ''
        self.agentUuid = ''
        self.neuronData = [0,0,0,0]
        self.recentMove = 0
        self.blind = 0

    def setAgentUuid(self,data):

        seats = data['seats']

        for i in range(0,len(seats)):

            row = seats[i]

            if row['name'] == self.agentName:

                self.agentUuid = row['uuid']


    def turnToOneHotCards(self,cards):

        suits = {
                    'C':1,
                    'D':2,
                    'H':3,
                    'S':4
                 }


        ranks = {
                    'A':0,
                    '1':1,
                    '2':2,
                    '3':3,
                    '4':4,
                    '5':5,
                    '6':6,
                    '7':7,
                    '8':8,
                    '9':9,
                    'T':10,
                    'J':11,
                    'Q':12,
                    'K':13
                }

        oneHotEncoded = []
        for i in range(0,len(cards)):
            card = cards[i]
            suit = suits[card[0]]
            rank = ranks[card[1]]

            oneHotEncoded.append(suit)
            oneHotEncoded.append(rank)

        return oneHotEncoded

    def estimate_win_rate(self,nb_simulation, nb_player, hole_card, community_card=None):
        if not community_card: community_card = []

        # Make lists of Card objects out of the list of cards
        community_card = gen_cards(community_card)
        hole_card = gen_cards(hole_card)

        # Estimate the win count by doing a Monte Carlo simulation
        win_count = sum([self.montecarlo_simulation(nb_player, hole_card, community_card) for _ in range(nb_simulation)])
        return 1.0 * win_count / nb_simulation


    def montecarlo_simulation(self,nb_player, hole_card, community_card):
        # Do a Monte Carlo simulation given the current state of the game by evaluating the hands
        community_card = _fill_community_card(community_card, used_card=hole_card + community_card)
        unused_cards = _pick_unused_card((nb_player - 1) * 2, hole_card + community_card)
        opponents_hole = [unused_cards[2 * i:2 * i + 2] for i in range(nb_player - 1)]
        opponents_score = [HandEvaluator.eval_hand(hole, community_card) for hole in opponents_hole]
        my_score = HandEvaluator.eval_hand(hole_card, community_card)
        return 1 if my_score >= max(opponents_score) else 0


    def turnToOneHotMoves(self,move):

        moves = {
                    'fold':0,
                    'call':1,
                    'raise':2
                }

        move = moves[move]

        return move

    def setVariables(self,seats):

        for i in range(0,len(seats)):

            self.playerMoves.append([0,0,0,0])


        for i in range(0,len(seats)):

            row = seats[i]

            self.uuids.append(row['uuid'])

        # print(self.uuids)

        # print(self.playerMoves)


    def setMove(self,uuid,move):

        index = self.uuids.index(uuid)

        move = self.turnToOneHotMoves(move)

        playerMoves = self.playerMoves[index]

        streetIndex = self.phases.index(self.street)

        playerMoves[streetIndex] = move

        self.playerMoves[index] = playerMoves

    
    def declare_action(self, valid_actions, hole_card, round_state):

        state = []
        #print(round_state)

        reward = 0.0

        done = False

        batch_size = 32

        call_action_info = valid_actions[1]

        raise_amount_options = [item for item in valid_actions if item['action'] == 'raise'][0]['amount']

        winRate = self.estimate_win_rate(500, self.playerCount, hole_card, round_state['community_card'])

        #print("AGENT WIN RATE:",winRate)
        phaseIndex = self.phases.index(self.street)
        self.neuronData[phaseIndex] = winRate


        
        state = np.reshape(self.neuronData, [1, len(self.neuronData)])
        print('>>>>>>>>>',state)
        action = self.brain1.act(state)
        #pprint(round_state)
        actionStr,amount = self.getAction(action,raise_amount_options,self.blind)

        self.brain1.remember(state, action, reward, state, done)

        ######  BRAIN 1######
        if len(self.brain1.memory) > batch_size:
            #self.brain.replay()       # internally iterates default (prediction) model
            self.brain1.replay(batch_size)
            self.brain1.target_train() # iterates target model

        #action, amount = call_action_info["action"], call_action_info["amount"]
        self.recentMove = action
        self.writeAction(actionStr)
        #print(state)
        #time.sleep(2)
        return actionStr, amount


    def writeAction(self,action):

        file = open('logs/Moves','a+')
        file.write(str(action)+'\n')
        file.close()

    def getAction(self,action,raise_amount,sbAmount):

        if action == 0:
            action = 'fold'
            amount = 0
        if action == 1:
            action = 'call'
            amount = sbAmount
        if action == 2:
            action = 'raise'
            amount = raise_amount['min']
        if action == 3:
            action = 'raise'
            amount = int(raise_amount['max']/2)


        return action,amount




    def receive_game_start_message(self, game_info):
        self.setAgentUuid(game_info)
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        #print("STREET",seats)
        self.playerCards = hole_card
        self.playerCards = self.turnToOneHotCards(self.playerCards)
        self.playerCount =  len(seats)
        #print(seats)
        self.setVariables(seats)
        pass

    def receive_street_start_message(self, street, round_state):
        self.street = street
        pass

    def receive_game_update_message(self, action, round_state):
        
        if action['action'] == 'call':
            self.blind = action['amount']
        

        #pprint(round_state)

        #time.sleep(3)

        # self.communityCards = round_state['community_card']
        # self.communityCards = self.turnToOneHotCards(self.communityCards)
        # self.setMove(action['player_uuid'],action['action'])
        pass

    def setRewardState(self,winners):
        winners =  winners[0]
        if winners['name'] == 'Agent':
            reward = 5

        elif self.recentMove == 0:
            reward = -1
        else:
            reward = -2

        done = True
        batch_size = 32

        state = np.reshape(self.neuronData, [1, len(self.neuronData)])
        action = self.recentMove
        self.brain1.remember(state, action, reward, state, done)

        if len(self.brain1.memory) > batch_size:
            #self.brain.replay()       # internally iterates default (prediction) model
            self.brain1.replay(batch_size)
            self.brain1.target_train() # iterates target model



    def receive_round_result_message(self, winners, hand_info, round_state):
        result_file = open('logs/result.csv','a+')
        result_file.write(str(winners)+'\n')
        result_file.close()
        self.setRewardState(winners)
        self.resetVariables()
        self.brain1.save()
        #time.sleep(1)
        pass


