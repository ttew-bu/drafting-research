
#Agents.py 
#Import dependencies
import math
import numpy as np
import random
import bezier

##PASSING AND RECEIVING HEURISTIC BOTS
#The bots below either have a bias toward cards that they see a lot of (or archetypes), or against cards/archs that the user has continually passed up
#Potential next iteration could be a bias toward each color and abstract the archetypes to the color level
class Naive_Pass_Agent:
  def __init__(self):
       self.name = 'naive_pass_agent'
  def decision_function(self,pack,draft,drafter_position):
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

    #Last card, we just take what's left
    if np.sum(pack)==1:
      preferences = pack

    else:
      preferences = np.reshape(array,(1,array.shape[0]))

    return preferences

class Naive_Receiver_Agent:
  def __init__(self):
       self.name = 'naive_receiver_agent'

  def decision_function(self,pack,draft,drafter_position):
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

    #Last card, we just take what's left
    if np.sum(pack)==1:
      preferences = pack

    else:
    #Utilize passes array to compute weights, the more times archetype-compatible cards are seen, the more they're taken
      preferences = np.einsum("ca,a->c",pack_weights, draft.drafter_preferences[drafter_position].reshape((draft.n_archetypes)))

    return preferences

class Arch_Pass_Agent:

  def __init__(self):
       self.name = 'arch_pass_agent'

  def decision_function(self,pack,draft,drafter_position):
    '''Define how the AP Agent calculates the optimal draft pack.
    This function mirrors is similar to the naive pass agent, but rolls up the passing bias 
    to the archetype level. The intuition for this agent is that every time I pass a card of an 
    archetype (especially good cards in those archetypes), I should be biased against taking those cards
    in the future
    '''

    drafter_pool_prefs = draft.drafter_preferences[drafter_position].reshape(draft.n_archetypes,1)
    #Keep same setup as before for the pack weights per card 
    pack_archetype_weights = (
            pack.reshape((draft.set.n_cards,1)) *
            draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

    #create 2d array of length n_cards_in_draft with a value for each time each card has passed
    already_seen = np.sum(draft.passes[drafter_position], axis=(1))

    #Now we will multiply in the card weights (if I pass a certain good card in a set multiple times, I should have
    # a high value for it)
    already_seen_by_value = already_seen.reshape((draft.set.n_cards,1)) * draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes))

    # n cards (# times seen summed per card), n archetypes (standard val * n cards)

    #Now we roll the cards passed based on their relative value to each archetype to the archetype level
    #Like with the naive agent for passes, we will add 1 to all cards so that we do not run into multiply/divide by 0 errors

    already_seen_archs = already_seen_by_value.sum(axis=0).reshape((draft.n_archetypes,1))

    #Once archetype weights are included, we need to scale them so the minimum is 0.5 and the max is 1
    #The idea here is that you will be less likely to take an arch you've already passed a ton, but
    #There is no increase in probability based on seeing something a lot

    #div by max

    #scale archs
    #scaler = MinMaxScaler(feature_range=(1,2))
    #transformed_archs = scaler.fit_transform(already_seen_archs)

    #If it is the last pick, then auto-pick the last card left
    if np.sum(pack)==1:
        preferences = pack

        return preferences

    #if it isn't the first pick, use the arch flow
    if np.sum(draft.picks) > 0:
      
      #Constant * vector of zeros + already seen archs/asa.max()
      transformed_archs = already_seen_archs / already_seen_archs.max()

      #Combine scaled bias term for the archetype at this pick given the passes and the 
      #drafter preferences given their pool

      #We will take 1 - the bias agenst a class term and treat this as the multiplier against the class passed more often
      #Add in constant here to dampen effect of thi bias term

      drafter_prefs_with_bias = (drafter_pool_prefs * (transformed_archs)).reshape((draft.n_archetypes))
      
      preferences = np.einsum("ca,a->c",pack_archetype_weights, drafter_prefs_with_bias)

      return preferences

    #if it is either pick 1 or the last pick, use special logic
    else:
      pack_archetype_weights = (
        pack.reshape((draft.set.n_cards,1)) *
        draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

        
      #Pick using basic strategy for pick 1 since we have nothing passed to us yet
      preferences = np.einsum("ca,a->c",pack_archetype_weights, draft.drafter_preferences[drafter_position].reshape((draft.n_archetypes)))
      
      return preferences

class Arch_Reciever_Agent:
  def __init__(self):
       self.name = 'arch_receiver_agent'

  def decision_function(self,pack,draft,drafter_position):
    '''Define how the AP Agent calculates the optimal draft pack.
    This function mirrors is similar to the naive pass agent, but rolls up the passing bias 
    to the archetype level. The intuition for this agent is that every time I pass a card of an 
    archetype (especially good cards in those archetypes), I should be biased against taking those cards
    in the future
    '''

    drafter_pool_prefs = draft.drafter_preferences[drafter_position].reshape(draft.n_archetypes,1)

    #329, 1 where 1 is num copies; note this will never be 0, so we do not need to worry about adding 1 at any point
    passed_array = np.sum(draft.options[drafter_position], axis=1).reshape((draft.set.n_cards,1))

    #Pull card values at each archetype out 
    standard_archweights = draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes))

    #Array of shape 329,10 showing the # of times a card has been seen * the value to each archetype
    receptions = np.multiply(passed_array, standard_archweights)

    #Now we roll the cards passed based on their relative value to each archetype to the archetype level
    #Like with the naive agent for passes, we will add 1 to all cards so that we do not run into multiply/divide by 0 errors
    receptions_archs = receptions.sum(axis=0).reshape((1,draft.n_archetypes))

    
    #If it is the last pick, then auto-pick the last card left
    if np.sum(pack)==1:
        preferences = pack

        return preferences

    if np.sum(draft.picks) > 0:
      #We can't really do anything besides the basic pick here since our strategy depends on other cards

      transformed_archs = receptions_archs / receptions_archs.max()

      #Once archetype weights are included, we need to scale them so the minimum is 0.5 and the max is 1
      #The idea here is that you will be less likely to take an arch you've already passed a ton, but
      #There is no increase in probability based on seeing something a lot

      #Create new pack weights based on our scaled receptions criteria
      #Pull card values at each archetype out 
      pack_options = standard_archweights * pack.reshape((draft.set.n_cards,1))

      #Direct multiplication to have clear interpretation of dampening effect (e.g. 20% less of arch, 20% reduction)
      new_pack_weights = pack_options * (transformed_archs)

      preferences = np.einsum("ca,a->c",new_pack_weights, drafter_pool_prefs.reshape(draft.n_archetypes))

      return preferences

    else:
      pack_archetype_weights = (
        pack.reshape((draft.set.n_cards,1)) *
        draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))
      preferences = np.einsum("ca,a->c",pack_archetype_weights, draft.drafter_preferences[drafter_position].reshape((draft.n_archetypes)))

      return preferences

##STANDARD HEURTISTC BOTS
#The bots below use common drafting heuristics including: picking the 'best' card always, taking the best card for n turns then using that pool to as your arch prefs
#simply using your current pool and available cards to find the max, and a dummy bot that serves as a 'random' baseline. 

class Basic_Agent:
  def __init__(self):
       self.name = 'basic_agent'

  def decision_function(self,pack,draft,drafter_position):
    '''Define how the basic agent calculates the optimal draft pick.
    Pulling from common literature, the basic agent simply takes arrays of archetype preferences and
    card fits, and performs the follow transformation: '''

    #Last card, we just take what's left
    if np.sum(pack)==1:
      preferences = pack
#       print('lastcard basic')
      return preferences

    else:
      pack_archetype_weights = (pack.reshape((draft.set.n_cards,1)) *
            draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

      preferences = np.einsum("ca,a->c",pack_archetype_weights, draft.drafter_preferences[drafter_position].reshape((draft.n_archetypes)))
      
      return preferences

class Greedy_Agent:
  def __init__(self, 
  k_forced=0):
       self.name = 'Greedy Agent' + '_turns_greedy_' + str(k_forced)
       self.k_forced=k_forced
       

  def decision_function(self,pack,draft,drafter_position):
    '''Define how the Greedy Agent calculates the optimal draft pack.
    The greedy agent just takes the best card irrespective of preferences 
    every time
    '''

    # if self.k_forced == 0:
    #   picknum = 0

    # else:
    #   picknum = np.sum(draft.picks[drafter_position])

    # if picknum <= self.k_forced:
      #Use preset weights for cards and whether or not they are in pack to make array 329,10 (n_cards,n_archs)
    weights = (
            pack.reshape((draft.set.n_cards,1)) *
            draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

    if np.sum(pack)==1:
      preferences = pack

    else:
    #Reshape weights to 329,1 so that it cqn fit into the work
      preferences = weights.max(axis=1).reshape((draft.set.n_cards,1))

    # #If you're not in a special case, then we just pick normally
    # else:

    #   pack_archetype_weights = (
    #         pack.reshape((draft.set.n_cards,1)) *
    #         draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

    #   preferences = np.einsum("ca,a->c",pack_archetype_weights, draft.drafter_preferences[drafter_position].reshape((draft.n_archetypes)))

    return preferences

class Force_K_Agent:
  '''An agent that wlll choose the best card for k turns, then will return to the standard strategy for the rest of draft'''
  def __init__(self,
       name='force_k_agent',
       k_forced=5,
       n_archs=10,
       custom_weight_coefficient=1
       ):
       self.name = name + '_k_' + str(k_forced)
       self.k_forced = k_forced
       self.n_archs = n_archs
       self.custom_weight_array = np.ones((self.n_archs,1))
       self.custom_weight_coefficient=custom_weight_coefficient

  def decision_function(self,pack,draft,drafter_position):
    '''Define how the AP Agent calculates the optimal draft pack.
    This function mirrors is similar to the naive pass agent, but rolls up the passing bias 
    to the archetype level. The intuition for this agent is that every time I pass a card of an 
    archetype (especially good cards in those archetypes), I should be biased against taking those cards
    in the future
    '''

    #If the pick is lower than the # of turns chosen to force
    #Note that this does NOT reset per pack, but rather at the
    #draft-drafter level 
    picknum = np.sum(draft.picks[drafter_position]) + 1
    #Last card in pack always get picked, no math needed (returns an array with 1 zero and we'll automatically pick that as max)
    
    #Last card in pack always get picked, no math needed (returns an array with 1 zero and we'll automatically pick that as max)
    if np.sum(pack)==1:

      #Again, this is an array with only one card with a value of one; this is the auto-max and gets chosen
      preferences = pack
#       print(np.where(pack==1))
#       print('lastcard force')

      return preferences

    #If we haven't reached the threshold we start forcing at, pick normally and update the internal array to track preferences
    elif picknum <= self.k_forced:

      #Use preset weights for cards and whether or not they are in pack to make array 329,10 (n_cards,n_archs)
      weights = (
            pack.reshape((draft.set.n_cards,1)) *
            draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

      #Reshape weights to 329,1 so that it cqn fit into the work
      preferences = weights.max(axis=1).reshape((draft.set.n_cards,1))

      #Do pick calculation custom in here in order to store the pick array
      pick_idx = np.argmax(preferences)
      picks = np.zeros((draft.set.n_cards, 1))
      picks[pick_idx] = 1
      
      #Create custom array for weighting that we can use for force agents that cut off based on specific 
      self.custom_weight_array = draft.drafter_preferences[drafter_position] + (self.custom_weight_coefficient * np.einsum('ca,dc->da', draft.archetype_weights, picks.reshape(1,draft.set.n_cards)))

      return preferences

    #Once we get beyond the specified number of picks, use custom array that does not update for picks. 
    else:

      #Take our regular card weights
      pack_archetype_weights = (
            pack.reshape((draft.set.n_cards,1)) *
            draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

      #Now run an einsum with our custom array to account for individual agent preferences
      preferences = np.einsum("ca,da->c",pack_archetype_weights, self.custom_weight_array)

      return preferences

class Dummy_Agent:

  def __init__(self, 
      name='dummy_agent',
      dummy_choice='random',):
      self.dummy_choice=dummy_choice
      self.name=str(name + '_'+dummy_choice)

  def decision_function(self,pack,draft,drafter_position):
      '''Define how the NP Agent calculates the optimal draft pack.
      This function is a dummy agent that picks some random card in the pick (or the first/last card in the pack);
      the primary advantage to having a bot like this in the draft is to simulate
      highly inexperienced drafters in the same way SKlearn uses the dummyclassifier(). 
      We can also simulate 'trolls' who do an ineffective strategy using these unsophisticated rules
      '''

      choice_array = np.zeros((draft.set.n_cards, 1))

      if self.dummy_choice == 'first':
        first_selection = np.where(pack.reshape((1, draft.set.n_cards))[0]==1)[0][0]
        choice_array[first_selection] = 1
        return choice_array

      if self.dummy_choice == 'last':
        last_selection = np.where(pack.reshape((1, draft.set.n_cards))[0]==1)[0][-1]
        choice_array[last_selection] = 1
        return choice_array

      if self.dummy_choice == 'random':
        #Choose random card that is actually in the pack
        rand_array = np.where(pack.reshape((1, draft.set.n_cards))[0]==1)[0]
        rand_selection = random.choice(rand_array)
        choice_array[rand_selection] = 1
        return choice_array

##ARTICLE-BASED BOTS
#These articles are inspired by particular works found in articles
#Drafting the hard way is described by Ryan Saxe's blog article, for example
class Med_Agent:
  '''‘drafting with preferences/drafting the medium way’ bot: 
  Pick an initial 2-3 archetypes, ignore all others. 
  Draft the usual way (stay open, weigh by pool etc), 
  but everything is evaluated wrt the 2-3 archetypes you care about. 
  '''

  def __init__(self,
       name='med_agent',
       arch_prefs:list=[0,1,2],
       archs:list=['WU','WB','WR','WG','UB','UR','UG','BR','BG','RG']):
       self.arch_prefs = arch_prefs
       self.archs_selected = ''.join([str(archs[x]) + "_" for x in arch_prefs])
       self.name = name + '_archs_'+str(self.archs_selected)

  def decision_function(self,pack,draft,drafter_position):
    '''Create decision function for agent that only looks at 2-3 archetypes'''

    #Create the custom array that only considers certain archetypes in a draft
    drafter_archs = np.zeros((draft.n_archetypes,1))

    #Iterate through list of selected archs and replace 0's with 1's where
    #the drafter wants to draft in that archetype
    for idx in self.arch_prefs:
       #pick_idx = np.random.choice(self.set.n_cards, p=row)
        drafter_archs[idx] = 1

    pack_archetype_weights = (
            pack.reshape((draft.set.n_cards,1)) *
            draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

    #There were some issues where the last card fails to calculate if the last card has a 0 (can't find an index for 0/0)
    #So let's add in some code here to take the last card in the pack if there are NO other options

    #If there is one card left in my deck, let's take the card in that index
    if np.sum(pack)==1:
      preferences = pack

    else:
      preferences = np.einsum("ca,a->c",pack_archetype_weights, drafter_archs.reshape((draft.n_archetypes)))

    

    return preferences

class Hard_Agent:
  """Following the idea in Ryan Saxe's article here: https://draftsim.com/ryan-saxe-bot-model/#What_Does_the_Data_Look_Like
  A player drafts the hard way as the bias term for one's pool shifts over time. Our forcing agent captures this phenomenom 
  with a set # of turns and then a set array of preferences. The Hard Agent differs because one's preference array still changes
  over time and there is a bias term applied to how much weight the pool gets over time
  
  For the linear pattern, our bias function is pick_number / turns_hard so that you will favor best card available 
  until a point, then you are more concerned with what lines up with the arch you have bias for. Note that we still have a bias 
  term, even if it isn't super influential with this activation function

  For the ln pattern, our bias function will be the ln of the pick num; same logic applies to the log10 choice (except it'll be a log with base 10
  """

  def __init__(self,
  turn_range_one_end:int=8,
  turn_range_two_start:int=16,
  turn_range_two_end:int=24,
  bias_start:int = 5,
  bias_plateau:int = 1,
  n_picks:int = 42,
  bias_function = 'linear'):
       self.name = 'hard_agent' + "_" + str(turn_range_one_end) + '_' +str(turn_range_two_start) + '_bstart_' + str(bias_start) +'_bplateau_' + str(bias_plateau) + "_" + str(bias_function)
       self.turn_range_one_end = turn_range_one_end
       self.turn_range_two_start = turn_range_two_start
       self.bias_function = bias_function
       self.bias_start = bias_start
       self.bias_plateau = bias_plateau
       self.turn_range_two_end = turn_range_two_end
       self.n_picks = n_picks

  def decision_function(self,pack,draft,drafter_position):
    '''Define how the basic agent calculates the optimal draft pick.
    Pulling from common literature, the basic agent simply takes arrays of archetype preferences and
    card fits, and performs the follow transformation: '''

    #Use the same internal pick # tracking we've used in other agents
    #Again, picks would be 0 for pick 1, so we add in 1 here
    picknum = np.sum(draft.picks[drafter_position]) + 1

    #Last card, we just take what's left
    if np.sum(pack)==1:
      preferences = pack
      return preferences

    #logic for linear model bias term
    if self.bias_function == 'linear':

      #if we are before the bias term is supposed to plateau, we will use a linear relationship
      #to bring us from the bias start to the plateau value in turn_range_one_end turns
      if picknum<=self.turn_range_one_end:
        bias_coefficient = self.bias_start - ((self.bias_start - self.bias_plateau) * (picknum/self.turn_range_one_end))


      #If we are in the bias plateau (e.g between the two ranges in the piecewise function), then spit
      #out the bias plateau
      elif picknum>self.turn_range_one_end and picknum<self.turn_range_two_start:

        bias_coefficient = self.bias_plateau

      #If we are in the second range, we see a linear relationship going from the bias plateau to 0
      elif picknum>=self.turn_range_two_start and picknum < self.turn_range_two_end:

        bias_coefficient = self.bias_plateau - self.bias_plateau * ((picknum - self.turn_range_two_start)/(self.turn_range_two_end-self.turn_range_two_start))
      
      else:
        bias_coefficient = 0

    #If we want a smooth curve between the pieces of the function, use a bezier curve
    #across a linear space of n_turns in each range
    if self.bias_function == 'bezier':

      #generate a bezier curve object for both of our curves in the piecewise function
      initial_bezier_components = np.array([
        [1.0, self.turn_range_one_end/2, float(self.turn_range_one_end)],
        [float(self.bias_start), float(self.bias_plateau), float(self.bias_plateau)],
    ])
      initial_curve = bezier.Curve(initial_bezier_components, degree=2)

      #Since the bezier curve spits out our expected bias value (y) in the range 0-1 for x
      #Create a linspace between those bounds and pull out equally spaced points to get 
      #our values at each pick number between pick 1 and the plateau point
      initial_linspace = np.linspace(0,1,self.turn_range_one_end)

      #generate a bezier curve object for both of our curves in the piecewise function
      secondary_bezier_components = np.array([
        [float(self.turn_range_two_start), float((self.turn_range_two_start + self.turn_range_two_end)/2), float(self.turn_range_two_end)],
        [float(self.bias_start), float(self.bias_plateau), float(self.bias_plateau)],
    ])

      #Create another bezier curve that represents the decay that occurs in the post-plateau range
      secondary_curve = bezier.Curve(secondary_bezier_components, degree=2)

      #Create a linespace w/ equally spaced positions across the number of picks in range 2 to estimate the bias coefficient
      secondary_linspace = np.linspace(0,1,(self.turn_range_two_end - self.turn_range_two_start))

      #If we are before the plateau, generate utilize the bezier curve
      if picknum<self.turn_range_one_end:

        #Here, we'll take picknumber as our index on the curve (e.g. what is the value in the linspace for pick 4 and solve for that bias term)
        #We pull position 1 out of the evaluation output here because we only want one coordinate, not the array pairs, from the Bezier curve
        bias_coefficient = initial_curve.evaluate(initial_linspace[picknum])[1]

      #Spit out the bias coefficient as the bias plateau if we are in the plateau range
      elif picknum>=self.turn_range_one_end and picknum<self.turn_range_two_start:

        bias_coefficient = self.bias_plateau

      #If we are in the second range, we use the secondary linspace
      elif picknum>=self.turn_range_two_start and picknum < self.turn_range_two_end:

        #Here, we use picknum - range start to get the position in the linspace because that will spit out a 0 for the first
        #index in our list as we need and will give us the correct indexes from turn_range_two_start onward
        bias_coefficient = secondary_curve.evaluate(secondary_linspace[(picknum - self.turn_range_two_start)])[1]

      else:
        bias_coefficient = 0
   
   #Once we spit out the bias coeffcient, normally calculate our cards except we will add in the bias constant on top of all arch scores
    pack_archetype_weights = (
            pack.reshape((draft.set.n_cards,1)) *
            draft.archetype_weights.reshape((draft.set.n_cards, draft.n_archetypes)))

    #Score our prefs as the einsum of the arrays cards x archs (e.g. 272, 10) -> flat array of length cards with values for their scores given the 
    #archetype preference scores.
    preferences = np.einsum("ca,a->c",pack_archetype_weights, (bias_coefficient+draft.drafter_preferences[drafter_position].reshape((draft.n_archetypes))))

    return preferences
