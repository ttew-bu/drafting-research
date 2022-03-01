##Sample Agent A
##In theory, we will iterate on certain lines of code in different agent files like this.
#E.g. there will be 10-15 Agent.py files with different rules we should be able to plug them into the sim

#Agent A uses the original pick making formula

#Import dependencies
import numpy as np
import pandas as pd
import random

class Basic_Agent:
  def decision_function(pack,draft,drafter_position):
    '''Define how the basic agent calculates the optimal draft pick.
    Pulling from common literature, the basic agent simply takes arrays of archetype preferences and
    card fits, and performs the follow transformation: '''

    pack_archetype_weights = (
            pack.reshape((draft.set.n_cards,1)) *
            draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

    preferences = np.einsum("ca,a->c",pack_archetype_weights, draft.drafter_preferences[drafter_position].reshape((draft.n_archetypes)))

    return preferences

class Naive_Pass_Agent:
  def decision_function(pack,draft,drafter_position):
    '''Define how the NP Agent calculates the optimal draft pack.
    This function mirrors what is done in the basic agent, then adds in a multiplier
    that reduces one's propensity to pick a card by 1/(n_times seen + 1)
    '''

    pack_archetype_weights = (
            pack.reshape((draft.set.n_cards,1)) *
            draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

    preferences = np.einsum("ca,a->c",pack_archetype_weights, draft.drafter_preferences[drafter_position].reshape((draft.n_archetypes)))
      
    already_seen = np.sum(draft.passes[drafter_position], axis=(1))
    already_seen_adj = 1 / (already_seen + 1)
    array = already_seen_adj * preferences
    preferences = np.reshape(array,(1,array.shape[0]))

    return preferences

# class Naive_Receiver_Agent:
#   def decision_function(pack,draft,drafter_position):

# class Dummy_Agent:
#   def decision_function(pack,draft,drafter_position,dummy_choice='random'):
#     '''Define how the NP Agent calculates the optimal draft pack.
#     This function is a dummy agent that picks some random card in the pick (or the first/last card in the pack);
#     the primary advantage to having a bot like this in the draft is to simulate
#     highly inexperienced drafters in the same way SKlearn uses the dummyclassifier()
#     '''

#     if dummy_choice == 'first':
#       first_selection = np.where(pack.reshape((1, draft.set.n_cards))[0]==1)[0][0]
#       return first_selection

#     if dummy_choice == 'last':
#       last_selection = np.where(pack.reshape((1, draft.set.n_cards))[0]==1)[0][-1]
#       return last_selection

#     if dummy_choice == 'random':
#       rand_pack = np.where(pack.reshape((1, draft.set.n_cards))[0]==1)[0]
#       random_index = random.randrange(len(rand_pack))
#       random_number = rand_pack[random_index]
#       return random_number
