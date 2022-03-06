
#Agents.py 



#Import dependencies
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
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

class Naive_Receiver_Agent:
  def decision_function(pack,draft,drafter_position):
    '''Define how the NR Agent calculates its pick.
    This function places a higher weight on the archetypes that are passed to it most.
    E.g. if I get a ton of red cards, I will have a bias toward that archetype 
    '''

    #Create new array that sums up what has been passed to the player

    #329, 1 where 1 is num copies
    passed_array = np.sum(draft.options[drafter_position], axis=1).reshape((draft.set.n_cards,1))

    #Pull card values at each archetype out 
    standard_archweights = draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes))

    #Array of shape 329,10 showing the # of times a card has been seen * the value to each archetype
    receptions = np.multiply(passed_array, standard_archweights)

    #Agg receptions by archetype and then scale everything by arch

    #Now multiply in the pack where 1 represents a card in the pack and a 0 is a missing card
    pack_weights = receptions * pack.reshape((draft.set.n_cards,1))

    #Utilize passes array to compute weights, the more times archetype-compatible cards are seen, the more they're taken
    preferences = np.einsum("ca,a->c",pack_weights, draft.drafter_preferences[drafter_position].reshape((draft.n_archetypes)))
    
    return preferences

# class Arch_Pass_Agent:
#   def decision_function(pack,draft,drafter_position):
#     '''Define how the AP Agent calculates the optimal draft pack.
#     This function mirrors is similar to the naive pass agent, but rolls up the passing bias 
#     to the archetype level. The intuition for this agent is that every time I pass a card of an 
#     archetype (especially good cards in those archetypes), I should be biased against taking those cards
#     in the future
#     '''

#     drafter_pool_prefs = draft.drafter_preferences[drafter_position].reshape(draft.n_archetypes,1)
#     #Keep same setup as before for the pack weights per card 
#     pack_archetype_weights = (
#             pack.reshape((draft.set.n_cards,1)) *
#             draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

#     #create 2d array of length n_cards_in_draft with a value for each time each card has passed
#     already_seen = np.sum(draft.passes[drafter_position], axis=(1))

#     #Now we will multiply in the card weights (if I pass a certain good card in a set multiple times, I should have
#     # a high value for it)
#     already_seen_by_value = already_seen.reshape((329,1)) * draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes))

#     #Now we roll the cards passed based on their relative value to each archetype to the archetype level
#     #Like with the naive agent for passes, we will add 1 to all cards so that we do not run into multiply/divide by 0 errors
#     already_seen_archs = 1 + already_seen_by_value.sum(axis=0).reshape(draft.n_archetypes).reshape((draft.n_archetypes,1))

#     #Once archetype weights are included, we need to scale them so the minimum is 0.5 and the max is 1
#     #The idea here is that you will be less likely to take an arch you've already passed a ton, but
#     #There is no increase in probability based on seeing something a lot
#     scaler = MinMaxScaler(feature_range=(.01,.99))

#     #scale archs
#     transformed_archs = scaler.fit_transform(already_seen_archs)

#     #Combine scaled bias term for the archetype at this pick given the passes and the 
#     #drafter preferences given their pool

#     #We will take 1 - the bias agenst a class term and treat this as the multiplier against the class passed more often
#     drafter_prefs_with_bias = (drafter_pool_prefs *  (1-(transformed_archs))).reshape(draft.n_archetypes)
    
#     preferences = np.einsum("ca,a->c",pack_archetype_weights, drafter_prefs_with_bias)

#     return preferences


# class Arch_Reciever_Agent:
#   def decision_function(pack,draft,drafter_position):
#     '''Define how the AP Agent calculates the optimal draft pack.
#     This function mirrors is similar to the naive pass agent, but rolls up the passing bias 
#     to the archetype level. The intuition for this agent is that every time I pass a card of an 
#     archetype (especially good cards in those archetypes), I should be biased against taking those cards
#     in the future
#     '''

#     drafter_pool_prefs = draft.drafter_preferences[drafter_position].reshape(draft.n_archetypes,1)

#     #329, 1 where 1 is num copies; note this will never be 0, so we do not need to worry about adding 1 at any point
#     passed_array = np.sum(draft.options[drafter_position], axis=1).reshape((draft.set.n_cards,1))

#     #Pull card values at each archetype out 
#     standard_archweights = draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes))

#     #Array of shape 329,10 showing the # of times a card has been seen * the value to each archetype
#     receptions = np.multiply(passed_array, standard_archweights)

#     #Now we roll the cards passed based on their relative value to each archetype to the archetype level
#     #Like with the naive agent for passes, we will add 1 to all cards so that we do not run into multiply/divide by 0 errors
#     receptions_archs = receptions.sum(axis=0).reshape((1,draft.n_archetypes))

#     #Once archetype weights are included, we need to scale them so the minimum is 0.5 and the max is 1
#     #The idea here is that you will be less likely to take an arch you've already passed a ton, but
#     #There is no increase in probability based on seeing something a lot
#     scaler = MinMaxScaler(feature_range=(1,2))

#     #scale archs
#     transformed_archs = scaler.fit_transform(receptions_archs)

#     #Create new pack weights based on our scaled receptions criteria
#     #Pull card values at each archetype out 
#     pack_options = standard_archweights * pack.reshape((draft.set.n_cards,1))

#     new_pack_weights = pack_options * (1+transformed_archs)

    
#     preferences = np.einsum("ca,a->c",new_pack_weights, drafter_pool_prefs.reshape(draft.n_archetypes))

#     return preferences


# class Dummy_Agent:
#   def decision_function(pack,draft,drafter_position,dummy_choice='random'):
#     '''Define how the NP Agent calculates the optimal draft pack.
#     This function is a dummy agent that picks some random card in the pick (or the first/last card in the pack);
#     the primary advantage to having a bot like this in the draft is to simulate
#     highly inexperienced drafters in the same way SKlearn uses the dummyclassifier(). 
#     We can also simulate 'trolls' who do an ineffective strategy using these unsophisticated rules
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
