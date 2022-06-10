from Agents import *
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime
import uuid
import json

#Note that some classes/functions in this module have a note that
#they draw inspiration from the Matt Drury repo. You can find this link here:
#https://github.com/madrury/mtg-draftbot

##CLASS DEFS THAT ALLOW US TO INSTANTIATE DRAFT INSTANCES (BOTH EQUILIBRIUM AND ON HUMAN PICKS)
class PseudoSet:
    def __init__(self, weights_df):
            self.n_cards = weights_df.shape[0]

class PseudoDraft:
    """Create a mock-draft agent that will store drafter preferences, set/card information, and hold tracking for cards picked/passed/options
    An instance of this class holds the data necessary for the agents to be programmed in a way that lets them work in both equilibrium and 
    human-analyzing experiments
    """

    def __init__(self,
    weights, 
    packs,
    cards_per_draft=42):
        self.n_archetypes = weights.shape[1]
        self.set = PseudoSet(weights)
        self.archetype_weights = weights
        self.packs = packs
        self.picks = np.zeros((1,self.set.n_cards,cards_per_draft))
        self.options = np.zeros((1,self.set.n_cards,cards_per_draft))
        self.passes = np.zeros((1,self.set.n_cards,cards_per_draft))
        self.drafter_preferences = np.ones((1,weights.shape[1]))

##Set is an adapted version of the Set in Matt Drury's repository
class Set:
    """A set of cards to draft from.

    Parameters
    ----------
    cards: List[Dict[str, object]]
      Data representing all the cards in the set, created by processing data
      from mtgjson.com. Each card is represented by a dictionary inside this
      list. Important keys in the dictionary are:
        - name: The name of the card.
        - rarity: The rarity of the card, common, uncommon, rare, or mythic.
        - colorIdentity: The colors of the card.

    card_names: List[str]
      A list of card names in the set. This is used here to create a canonical
      ordering of the cards in the set, since different sources may be
      inconsistent about the ordering.
    """
    def __init__(self, cards, card_names):
        self.cards = cards
        self.commons, self.uncommons, self.rares = self.split_by_rarity(cards)
        self.card_names = card_names
        self.n_cards = len(self.card_names)

    def random_packs_array(self, n_packs=8, pack_size=14):
        """Make some random packs of cards from the set. Each pack contains one
        rare/mytic, thre uncommons, and however many commons are needed to
        round out the pack.

        Parameters
        ----------
        n_packs: int
          The number of packs to create.

        pack_size: int
          The number of cards in each pack.

        Returns
        -------
        packs: np.array, shape (n_packs, n_cards)
          The number of each card in each pack.
        """
        
        packs = [self.random_pack_dict(size=pack_size) for _ in range(n_packs)]
        cards_in_pack_df = pd.DataFrame(np.zeros(shape=(n_packs, self.n_cards), dtype=int),
                                        columns=self.card_names)
        
        #Basically generate packs until they are correct in the event names do not line up
        while cards_in_pack_df.values.sum().sum()!= n_packs * pack_size:
            for idx, pack in enumerate(packs):
                for card in pack:
                    name = card['name']
                    cards_in_pack_df.loc[cards_in_pack_df.index[idx], name] += 1

            if cards_in_pack_df.values.sum().sum()== n_packs * pack_size:
                  return cards_in_pack_df.values
            else:
                packs = [self.random_pack_dict(size=pack_size) for _ in range(n_packs)]
        cards_in_pack_df = pd.DataFrame(np.zeros(shape=(n_packs, self.n_cards), dtype=int),
                                        columns=self.card_names)
        
        

    def random_pack_dict(self, size=14):
        n_rares, n_uncommons, n_commons = 1, 3, size - 4
        pack = []
        count_commons = 0
        count_uncs = 0
        packnames = []
        while len(pack) != size:
            while count_commons<n_commons:
                choice = random.choice(self.commons)
                if choice['name'] not in packnames:
                  pack.append(choice)
                  packnames.append(choice['name'])
                  count_commons +=1
                  
            while count_uncs<n_uncommons:
                choice = random.choice(self.uncommons)
                if choice['name'] not in packnames:
                  pack.append(choice)
                  packnames.append(choice['name'])
                  count_uncs +=1

            #Add in our 1 rare/mythic
            pack.append(random.choice(self.rares))
            
            if len(pack) == size:
                return pack
            else:
                pack=[]
                
    @staticmethod
    def split_by_rarity(cards):
        commons, uncommons, rares = [], [], []
        for card in cards:
            rarity = card['rarity']
            if rarity == 'common':
                commons.append(card)
            elif rarity == 'uncommon':
                uncommons.append(card)
            elif rarity in {'rare', 'mythic'}:
                rares.append(card)
        return commons, uncommons, rares

#MSD is also an loose adaptation of the draft class from the repo
class MultiStratDraft:
    """ An adaptation of the Draft class that allows different seats at the draft table
    to use different strategies. The primary difference here is that you must pass in the number of drafters 
    that use each strategy and a file with the strategy functions. Also includes optional argument that 
    would allow the user to pass in their own packs. 
    
    This creates two benefits: 
    1) we can use the exact same packs to test different agents and compare behavior
    2) we can pass in human data (if properly formatted) and use data from multiple sources
    
    
    """
    def __init__(self, 
                 n_rounds:int=3,
                 n_cards_in_pack:int=14,
                 cards_path:str=None,
                 card_values_path:str=None,
                 packs_input_file:str= None,
                 agent_list:list=None,
                 rotate:int = 1,
                 packs_idx:int=0,
                 ):
        self.draft_id = uuid.uuid4()
        self.agent_list=agent_list
        self.n_drafters=len(agent_list)
        self.packs_input = packs_input_file
        self.n_rounds = n_rounds
        self.n_cards_in_pack = n_cards_in_pack
        self.packs_idx = packs_idx
        self.all_packs = self.read_draft_packs(self.packs_input)
        self.packs_for_draft = self.all_packs[self.packs_idx]
        
        # These archetype weights could be learned from human draft data.
        self.archetype_weights, self.archetype_names, self.card_names = (
            self.make_archetype_weights_array(json.load(open(card_values_path))))
        self.n_archetypes = len(self.archetype_names)
        self.set = Set(
            cards=json.load(open(cards_path))['cards'],
            card_names=self.card_names)
        # Internal algorithmic data structure.
        self.drafter_preferences = np.ones(shape=(self.n_drafters, self.n_archetypes))
        self.round = 0

        #New, used to rotate in between rounds; you can enter either 1 or -1 and the algo will work fine
        self.rotate = rotate

        self.archetype_weights[self.archetype_weights==0] = 0.001

        # Output data structures.
        self.options = np.zeros(
            (self.n_drafters, self.set.n_cards, self.n_cards_in_pack * self.n_rounds),
            dtype=int)
        self.picks = np.zeros(
            (self.n_drafters, self.set.n_cards, self.n_cards_in_pack * self.n_rounds),
            dtype=int)
        self.preferences = np.zeros(
            (self.n_drafters, self.n_archetypes, self.n_cards_in_pack * self.n_rounds))

        #Here is how we will keep track of what cards somebody passes on; this will allow us to capture 'signal' 
        self.passes = np.zeros((self.n_drafters, self.set.n_cards, self.n_cards_in_pack * self.n_rounds), dtype=int)

    def read_draft_packs(self, filename:str):
        '''Helper function that allows us to unpack packs.txt files'''
        with open(filename) as file:
            packs = json.load(file)
            array= np.asarray(packs)
        return array

    def rotate_array(self, arrays, switch:int):
        """Given our packs array and a binary switch (1 -> -1 -> 1 etc.)"""
        newx = np.zeros(arrays.shape)

        #If positive, we go right (e.g. cards go from seat 1 to seat 2)
        if switch>0:
            newx[0, :] = arrays[-1, :]
            newx[1:, :] = arrays[:-1, :]

        #If negative, go left (e.g. cards go from seat 2 to seat 1)
        else:
            newx[-1, :] = arrays[0, :]
            newx[:-1, :] = arrays[1:, :]

        #In either case, return the rotated packs
        return newx
    
    #Use draft method from above for pack generation
    def draft(self):
      '''The draft function either generates 3 new packs based on the cards and weights or takes them from the import in 
      the function call'''

      #Iterate through the number of rounds passed into the function       
      for n in range(self.n_rounds):

          try:
            #Pull packs from input
            #Read the packs for each round and store them in-place
            packs = self.packs_for_draft[n]

          except:
            #Create packs using the packs array function
            print('read failed')
            exit()
            #The packs for a given round get stored in the packs_for_draft variable in the proper position 
            
          #Iterate through each pack
          for n_pick in range(0,self.n_cards_in_pack):
              packs = self.draft_packs(packs, n_pick)

          #After each round, move onto to the next pack. 
          self.round += 1
    
    def draft_packs(self, packs, n_pick):
        """Draft a single pick from a set of packs, one pick for each drafter.

        Parameters
        ----------
        packs: np.array, shape (n_drafters, n_cards)
          Array indicating how many of each card remains in each pack.

        n_pick: int
          Which pick of the draft is this?  Note that the first pick of the
          draft is pick zero.

        Modifies
        --------
        self.options: np.array 
                      shape (n_drafters, n_cards, n_cards_in_pack * n_rounds)
          The initial packs array (before any picks are made) is copied into an
          (n_drafters, n_cards) slice of this output array.

        self.picks: np.array
                    shape (n_drafters, n_cards, n_cards_in_pack * n_rounds)
          After picks are computed for each drafter, these are copied into an
          (n_drafters, n_cards) slice of this output array.

        self.preferences: np.array
                          shape (n_drafters, n_archetypes, n_cards_in_pack * n_rounds)
          After archytype preferences are re-computed for each drafter post
          pick, these are copied into an (n_drafters, n_archetypes) slice of
          this output array.f

        Returns
        -------
        packs: np.array, shape (n_drafters, n_cards)
          The input packs array, transformed given the results of the pick algorithm:
            - The cards picked by each drafter are removed from the corresponding packs.
            - The packs array is rotated (in the first dimension), to prepare
              for the next pick (packs are passed around the drafters
              circularly pick-to-pick.
        """
        
        #Tell us the options available
        options = packs.copy()
        self.options[:, :, n_pick + self.n_cards_in_pack * self.round] = options
        
        #1's or 0's if card is in pack, this is where we're gonna delineate from the intended algorithm

        card_is_in_pack = np.sign(packs)
        
        lst = []
        #print(card_is_in_pack)
        for idx, pack in enumerate(card_is_in_pack): 

          array = self.agent_list[idx].decision_function(pack,self,idx)

          if np.sum(array)==0:
            print('error in card selection')
            print(self.agent_list[idx])
            print(pack)
            print(array)

          array = array.reshape(1,self.set.n_cards)

          lst.append(array)

        prefs = np.concatenate(lst, axis=0)
        
        #This needs to depend on the agent 
        picks = self.make_picks(prefs)

        #Create passes variable
        turn_passes = np.subtract(options, picks)
        self.passes[:, :, n_pick + self.n_cards_in_pack * self.round] = turn_passes.copy()

        assert np.all(packs >= picks)
        
        #We will rotate and then change direction for next time
        packs = self.rotate_array(packs - picks, switch=self.rotate)

        #Since our switch is -1/1, we just multiply by -1 to swap every time. 
        self.rotate = self.rotate * -1
        
        #Take the cards+archetypes and the drafters+cards to get an array shaped drafters x archetypes
        #Takes current values and then update them based on math above

        self.drafter_preferences = (
            self.drafter_preferences +
            np.einsum('ca,dc->da', self.archetype_weights, picks))

        #Update the picks array in place
        self.picks[:, :, n_pick + self.n_cards_in_pack * self.round] = picks.copy()

        #Update preferences array based on the drafter preferences for the round
        self.preferences[:, :, n_pick + self.n_cards_in_pack * self.round] = (
            self.drafter_preferences.copy())
        
        return packs

    def make_archetype_weights_array(self, card_values):
      """Convert card values to arrays for the weights, archnames, and cardnames """
      
      #Create dataframe from the card arrays pulled from our JSON file
      archetype_weights_df = pd.DataFrame(card_values).T
      archetype_names = archetype_weights_df.columns
      card_names = archetype_weights_df.index

      #Return the three sets of arrays separately
      return archetype_weights_df.values, archetype_names, card_names

    def make_picks(self, pick_probs):
      """Compute definate picks given an array of pick probabilities.

      Parameters
      ----------
      pick_probs: np.array, shape (n_drafters, n_cards)
        Pick probabilities for each drafter.

      Returns
      -------
      picks: np.array, shape (n_drafters, n_cards)
        Definate picks for each drafter, computed by sampling from according
        to the each row of the pick probability array. 
      """
      picks = np.zeros((self.n_drafters, self.set.n_cards), dtype=int)
      for ridx, row in enumerate(pick_probs):
          pick_idx = np.argmax(row)
          picks[ridx, pick_idx] = 1
      return picks



##EQUILIBRIUM EXPERIMENT FUNCTIONS:
def generate_simulation_packs(n_packs:int=3,n_players:int=8,
n_cards_in_pack:int=14,n_iter:int=100,n_cards_in_set:int=272,
cards_path:str='json_files/VOW_writeback.json',
weights_df_w_names:str='weights_data/source_weights/VOM_Weights_default__seen_rates_df.csv'):
    """Generate a random sample of packs to run closed-circuit simulations on given
    the parameters you intend to use on the draft"""

    generator_set = Set(cards=json.load(open(cards_path))['cards'],
    card_names=pd.read_csv(weights_df_w_names)['Name'])

    draft_list = []
    for x in range(0,n_iter):
        print('pack '+str(x)+' start')
        accum = []
        for z in range(0,n_packs):
            pack = generator_set.random_packs_array()
            accum.append(pack)
        draft_list.append(accum)
    draft_list = np.array(draft_list)

    name = 'draft_txt_files/draft_packs.txt'

    name = name.replace('.txt',"_"+str(n_iter)+"_"+str(datetime.datetime.today().strftime('%Y_%m_%d')+".txt"))

    with open(name, 'w') as filehandle:
        json.dump(draft_list.tolist(),filehandle)

    return None

##HELPER FUNCTIONS AND CLASSES TO RUN SIMULATIONS OF DRAFTS (e.g. many iterations of the classes above and write to files)
#FUNCTIONS THAT POWER INTERNAL PROCESSES TO RUN HUMAN PICK DRAFTS
def create_pack_pick_pass_arrays(df:pd.DataFrame,id:str,pick_index:int=11):
    """Take our dump dataframe, id for the draft to check, and pick index to generate arrays
    for our simulations to run; this function will run every time we check out a new draft in 
    our large-scale experiments
    
    Output here is a dictionary that holds the real picks from the draft, the passes, and the pool
    at each stage in the draft. These allow for the agents to make picks with the conditions the real 
    player had at every single pick."""

    #create a df based on the ID
    new_df = df[df['draft_id']==id]

    #Pull out the picks and packs and store them in dataframes based on their colnames in the input data dump
    packs_df = new_df.filter(regex='pack_card')
    picks_df = new_df.filter(regex='pool')

    #Our picks df has pool_cardname as a structure, so str strip so we just have card names
    card_names= [s.replace("pool_","") for s in picks_df.columns if "pool" in s]

    #Picks seem like a mess coming from 17lands... let's fix that. 
    picks = new_df.iloc[:,pick_index]

    #Create an accumulator 
    actual_pick_arrays = []

    #Create array of zeros except for the card thats picked as our actual value
    for p in picks:

        #Create a new array of length cards
        new_array = np.zeros((len(card_names)))

        #Substitute a 1 in the correct position of the card; we use strip here to avoid any weird spacing on the string match 
        new_array[card_names.index(p.strip())] = 1

        #Add new picks 
        actual_pick_arrays.append(new_array)

    #Packs array is trustworthy, no data issues; this will make a nested array of shape n_cards, n_archs (e.g. 272,10)
    packs_array = packs_df.to_numpy()

    #Convert our actual picks to an array to do operations
    picks_array = np.asarray(actual_pick_arrays)

    #Passes array is the options - the picks, leaving an array w/ 1 less card than are 
    # available as options to be used in the algorithm for certain bots
    passes_array = packs_array - picks_array

    #Create a dict with these outputs so we can easily access these arrays in other functions
    output_dict = {"packs":packs_array,
    "picks":picks_array,
    "passes":passes_array
    }

    return output_dict

def generate_agent_selection_data(psd:object, agent:object, pick_n:int, picks:list,
int_scores:list, int_array_keys:list):
    """Given the draft, agent, pick, and accumulator variables, generate the selection for an agent given a certain 
    pick in the draft. The draft/agent objects hold information on pool/preferences/selection calculations.
    
    The end result of this function is to append the pick made by the agent, the top 3 cards ranked by the agent, 
    the raw scores the agent used for selection, and the index of each card in the array (e.g. card in position 2-> card 12) """

    #Create a bot score array to track top picks
    bot_score_array = agent.decision_function(psd.packs[pick_n],psd,0).reshape((psd.set.n_cards))

    #Create an array that tracks the norm card strength (e.g. )

    #T1 DATA WORKFLOW
    #Add the largest item in the array, 
    #argmax returns first item (e.g. lowest index) with tie
    picks.append(bot_score_array.argmax())

    #T3 DATA WORKFLOW DEPRECATED 
    #Split pull indexes of top 3 values
    #top3 = np.argpartition(bot_score_array, -3)[-3:]
    #top3_picks.append(top3)

    #WORKFLOW FOR ORDER OF CARDS PICKED
    #First, only take the cards that are in the pack
    array_indices = np.where(psd.packs[pick_n]>0)

    #Then, let's find the bot scores
    scores_unsorted = (bot_score_array[array_indices])

    #Instead of doing operations here, just cross check in the index given indices and actual pick later in script 
    int_scores.append(scores_unsorted)
    int_array_keys.append(array_indices)

def update_weights(psd:object, draft_arrays:dict, n:int):
    """Given a draft and arrays, update the drafter's preferences, as well as the picks/options/passes array
    so that the next selection can be made under the same conditions the real player had
    
    No output here, simply an update to picks/options/passes/drafter_preferences on the pseudodraft object"""
    #Now we need to input the data from what actually happened so the bots can start from a net new position 
    psd.picks = draft_arrays['picks'][0:n+1].reshape(1,psd.set.n_cards,n+1)
    psd.options = draft_arrays['picks'][0:n+1].reshape(1,psd.set.n_cards,n+1)
    psd.passes = draft_arrays['picks'][0:n+1].reshape(1,psd.set.n_cards,n+1)


    #Now let's update the drafter preferences so we're not eternally multiplying by 1
    #We will use the dot project of an array of cards with archetype weights (e.g. shape 272,10 )
    #and an array of shape drafters,archs (e.g. 1, 10) -> gives us array (1 drafter, 10 arch preferences)
    psd.drafter_preferences = (
    psd.drafter_preferences +
    np.einsum('ca,dc->da', psd.archetype_weights, draft_arrays['picks'][n].reshape(1,psd.set.n_cards)))

def generate_index_residual_delta_df(df_matches:pd.DataFrame, psd:object, scores:list,array_keys:list,
agent_names:list, output_name:list,suffix:str,abs_dir_path:str):

    #We will start with one df with the deltas
     idx_df = pd.DataFrame(scores,columns=agent_names)
     keys_df = pd.DataFrame(array_keys,columns=[a + '_y' for a in agent_names])
    #Create an array of shape n_cards that is the sum of the card's weights in the x number of archetypes
     norm = psd.archetype_weights.sum(axis=1)

     #We will add the real and ID columns + picknumbers to our data so we can join them
     idx_df['Real'] = df_matches['Real']
     idx_df['id'] = df_matches['id']
     idx_df['picknumbers'] = df_matches['picknumbers']

    #Combine the two dfs horizontally
     idx_df = pd.concat([idx_df,keys_df],axis=1)

     #Now that df is merged, let's do some column operations per agent
     for a in agent_names:

         #First, let's get the relevant indexes by creating a col with an index mapping 
         idx_colname = a + '_index'
         agent_col_values = a
         agent_col_key = a + '_y'

         #Create a column that tells us which position in the given pack for a selection the selected card was in
         #With this, we can manipulate scores and other variables, then pass in this index to pull out the data for the card that was really selected
         #We can answer questions such as how many picks off, how far off was our bot, and how far off was the pick given the norm of card strength?
         idx_df[idx_colname] = idx_df.apply(lambda x: np.where(x[agent_col_key][0]==x['Real'])[0], axis=1)

         #The logic here is that if we have X number of cards in pack, we take the index of the actual pick and see where the agent put it in the array
         picks_off_colname = a + '_picks_off'

         #The code is a bit messy here, but basically, we sort out all of the scores for a given agent in-place, take the index of the card selected
         #and since the sorting is from high to low, our card will always be the picks off (e.g. if it is card #3 in the pack, then we are always 2 off,
         # which the index accounts for here)
         idx_df[picks_off_colname] = idx_df.apply(lambda x: np.argsort(np.argsort(-1 * x[agent_col_values]))[tuple(x[idx_colname])], axis=1)

         #The logic here is that we have an array w/ scores, so we take the max - value of the card selected
         #This is primarily an assessment of the deltas between our top scores and score of selected card; will be 0 when values match and a nonzero value when we don't make the real pick
         internal_residual_colname = a + '_internal_residual'
         idx_df[internal_residual_colname] = idx_df.apply(lambda x: (x[agent_col_values].max() - x[agent_col_values][tuple(x[idx_colname])]), axis=1)

         #Now, let's see how far off the results are from the norm (e.g. sum of all archs for a card)
         #This is a more natural assessment of how our bots are doing with "natural card strength" as sort of a criteria as opposed to including pool bias et al. 
         norm_delta_colname = a + '_norm_delta'
         idx_df[norm_delta_colname] = idx_df.apply(lambda x: (norm[tuple(x[agent_col_key])].max() - norm[tuple(x[idx_colname])]), axis=1)

     #Write the dataframe here that includes all of our simulation picks
     index_str = abs_dir_path + "index_data/"+output_name.replace(r'{suffix}','_index_{suffix}'.format(suffix=suffix))
     idx_df.to_csv(index_str, mode='a',header=(not os.path.exists(index_str)), index=False)

##MODULE FUNCTIONS 
#COMPLETE FUNCTIONS TO CALL FOR WEIGHT AND SIMULATIONS 
def run_human_pick_experiment(suffix:str, draft_dump_path:str, abs_dir_path:str, weights_path:str, agents:list, 
n_iter:int=1000, pick_index:int=11, cards_per_draft:int=42):
    """Run experiments on dump of human picks, a weights file, a list of instantiated agent classes, the number of rows to select, 
    the position of the pick column in the dump file, and the cards in the draft to validate that we do not assess partial/incomplete 
    drafts to keep data high quality.

    The prefix is a label you can add to label experiments e.g. "medium_" + path tells me I did the medium exp. with certain files

    This function returns the following:

    index_data/index data csv -> rows are a single pick in a single draft (e.g. ID, p1), rows are the picks off from the correct card (e.g. we ranked the card 3rd but
    it was first, so we were 2 picks off), the bot score delta (e.g. our bots top score was 1.2, the actual selected card was scored 1.1 by the bot, so we have a delta
    of 0.1 representing how close our bots calculations were), and the norm delta (e.g. given an array where one item is sum of all archetype scores for each card to
    represent the "strength/value" of the card, how far off was our actual pick from the strongest card based on this norm?). The norm delta matches the paper's
    definition of how far off we are from making the "right" pick because it accounts for some measure of absolute strength, as opposed to the scores spat out by each
    bot, which are partially a function of each bot's decision function. 
    """

    #Read weights csv
    df = pd.read_csv(draft_dump_path,nrows=n_iter*cards_per_draft)

    #Add automated stamps to our data so we can version it; start by taking weight set used, then add n_drafts
    output_name = weights_path.replace(r'.csv','_{num}_{suffix}.csv').format(num=str(n_iter), suffix=suffix)
    output_name = output_name.replace(r'weights_data/processed_weights/',"")

    #Pull our weights df and send to array
    weights_df =  pd.read_csv(abs_dir_path + weights_path)
    weights = weights_df.to_numpy()

    #Get the IDs to iterate through
    unique_ids = df.draft_id.unique()
    agent_names = [agent.name for agent in agents]

    #Create a column to track matches w/ real data here
    match_cols = agent_names + ['Real']

    #let's iterate through every draft that we pulled from the dump file 
    for id in unique_ids:
        try:
            draft_arrays = create_pack_pick_pass_arrays(df, id, pick_index)

            #Add try/except logic for drafts that have 42 picks. For incomplete drafts, print out drafts in fail list and skip
            #try:
            #Instantiate the object
            psd = PseudoDraft(weights, draft_arrays['packs'], cards_per_draft)

            #Create lists to hold totals for picks
            totals = []
            #totals_top3 = []
            scores = []
            array_keys = []

            #Iterate through all the selections in a given draft 
            for n in range(0,cards_per_draft):

                #Create accumulator lists for the picks made by the bots (and top 3)
                picks = []
                #top3_picks = [] 
                int_scores = []
                int_array_keys = []

                #For each of our n picks, iterate through each of the agents
                for a in agents: 

                    #This function will generate the picks and add them accumulator
                    generate_agent_selection_data(psd, a, n, picks, int_scores, int_array_keys)

                #Add in the actual pick to the end of the picks array in the final position
                picks.append(draft_arrays['picks'][n].argmax())

                #Append everything to our accumulator data structures
                totals.append(picks)

                #T3 WORKFLOW DEPRECATED 
                #totals_top3.append(top3_picks)

                array_keys.append(int_array_keys)
                scores.append(int_scores)
                update_weights(psd, draft_arrays, n)


            #Let's create some dataframes
            #The first one will hold whether or not we matched the picks
            df_matches = pd.DataFrame(totals, columns=match_cols)

            #Add in the id and picknumbers for us to do more robust analysis
            df_matches['id'] = id
            df_matches['picknumbers'] = [x for x in range(0,42)]

            #INDEX FILE IS NEW SOURCE TABLE FOR PREPROCESSING FUNCTION
            generate_index_residual_delta_df(df_matches,psd,scores,array_keys,agent_names,output_name,suffix,abs_dir_path)
                
        except IndexError:
            print(id)
            pass

def human_pick_experiment_runner(dictionaries:list,weight_file_name):
    """Iterate through a list dictionaries containing weights, filenames, and agents to automatically generate our experiment data given
    The idea here being if you want to run many experiments, it may be better to put all the inputs together in a dictionary as opposed to writing
    many long function calls"""

    #for each item in our processed weights directory
    for d in dictionaries:
        #Iterate through our files
        string = 'weights_data/processed_weights/'+ weight_file_name
        run_human_pick_experiment(d['suffix'],d['draft_str'],string, d['agents'], d['n_iter'], 11)

#class equilibrium_experimentation_util():

def process_simulation_results(weight_path:str,df:pd.DataFrame,
arch_list:list=[x for x in range(0,10)],
n_largest_for_norm:int=23):
    """Take our weights array, get the weights of chosen cards per archetype and 
    assess how strong/weak a deck is in archetypes"""

    #Pull our weights and store in array
    weights_df = pd.read_csv(weight_path)
    norm_arrays = weights_df.to_numpy()

    #Run accumulator pattern to see who selected which cards and pull values for their arch scores out of the norm arrays
    accum = []
    for index,row in df.iterrows():
        accum.append(list(norm_arrays[row['card']]))
    norm_vals = pd.DataFrame(accum)
    merged = pd.concat([df,norm_vals],axis=1)

    #To get the archetype score per player, we will use the top N cards in each arch (this defaults to 23)
    merged = merged.groupby(['simid','player','draft_number'])[arch_list].agg(lambda grp: grp.nlargest(n_largest_for_norm).sum())

    #Create column to show top arch score and which arch was the top score
    merged['top_archetype_score'] = merged.max(axis=1)
    merged['top_archetype_index'] = merged.idxmax(axis=1)
    merged.reset_index(inplace=True)
    return merged

def equilibrium_experiment_runner(agents:list,
baseline_agents:list,
cards_path:str,
weights_json_path:str,
weights_df_path:str, 
packs_input_file:str,
abs_dir_path:str,
batch_id:uuid=uuid.uuid4(),
n_iter:int=10,
n_cards_in_pack:int=14,
rounds:int=3,
n_largest_for_norm:int=23,
deviating_seat:int=0,
rotate_option:int=0,
archetypes:list=['WU','WB','WR','WG','UB','UR','UG','BR','BG','RG']):
    '''Streamline comparison of drafts and create outputs with player prefs;
    this gives us quick glances into how different arrangements '''

    output_str = "equilibrium_data_{num}.csv".format(num=n_iter)
    print('start' + str(datetime.datetime.now()))
    #Create two accumulator lists to hold the preferences 
    ss_picks = []
    ms_picks = []

    #Create loop that will instantiate two draft objects and run them iteratively 
    for r in range(0,n_iter):

        #Instantiate the single strategy draft, which we will use as a baseline to compare the MultiStrat draft
        draft_ss = MultiStratDraft(
        n_rounds=rounds,
        n_cards_in_pack=n_cards_in_pack,
        cards_path=cards_path,
        card_values_path=weights_json_path,
        packs_input_file=packs_input_file,
        agent_list=baseline_agents,
        rotate=rotate_option,
        packs_idx=r)

        #Instantiate the multi strategy draft, this object will simulate what happens with the exact same
        #card packs, but when the drafters in each sit may or may not use different strategies (e.g. seat 1 uses 
        # strategy A, while the rest use strategy B)
        draft_ms = MultiStratDraft(
        n_rounds=rounds,
        n_cards_in_pack=n_cards_in_pack,
        cards_path=cards_path,
        card_values_path=weights_json_path,
        packs_input_file=packs_input_file,
        agent_list=agents,
        rotate=rotate_option,
        packs_idx=r)
       
        #Calling this function will simulate a draft for us. This will allow us to call picks, options, etc. from the draft on a seat by seat basis
        draft_ss.draft()

        # ss_prefs.append(result_array.reshape((draft_ss.n_drafters,draft_ss.n_archetypes+2)))
        ss_picks.append(draft_ss.picks)

        #Now, run the multistrat draft using the exact same cards as before
        draft_ms.draft()

        ms_picks.append(draft_ms.picks)

    #Create accumulators for both of the dfs in our experiment (control/experiment data)
    df_list_baseline = []
    df_list_diff = []

    #Iterate through the ms_picks list (actually we'll go through ss_picks concurrently using the index in ms_picks)
    for index,draft in enumerate(ms_picks):
        sim_id = uuid.uuid4()
        
        #First we will iterate through the base case
        for idx,iter in enumerate(draft):
            testarray = np.where(iter!=0)
            z = np.array((testarray[0],testarray[1])).T
            df = pd.DataFrame(z, columns=['card','picknum'])
            df['simid'] = sim_id
            df['player'] = idx
            df['draft_number'] = index
            df_list_diff.append(df)

        #Then, we iterate through the draft with some level of deviation. 
        for idx2, iterable in enumerate(ss_picks[index]):
            testarray = np.where(iterable!=0)
            z = np.array((testarray[0],testarray[1])).T
            df = pd.DataFrame(z, columns=['card','picknum'])
            df['simid'] = sim_id
            df['player'] = idx2
            df['draft_number'] = index

            df_list_baseline.append(df)


    df_diff = pd.concat(df_list_diff).groupby(['simid','picknum','player','draft_number'])['card'].first().reset_index()
    df_baseline = pd.concat(df_list_baseline).groupby(['simid','picknum','player','draft_number'])['card'].first().reset_index()

    #Put dfs through workflow that will adjust column names, generate norm-based strengths for the cards, 
    output_diff = process_simulation_results(weights_df_path,df_diff,n_largest_for_norm=n_largest_for_norm)
    output_baseline = process_simulation_results(weights_df_path,df_baseline,n_largest_for_norm=n_largest_for_norm)

    #Create a remapping dictionary to overwrite the numbers for arch names as the actual archetype
    arch_remapping_dictionary = {}
    for idx,l in enumerate([x for x in range(len(archetypes))]):
        arch_remapping_dictionary[l] = archetypes[idx]

    #Now we add in some housekeeping columns such as what the experiment was, when it was etc. 
    #As of now, we always deviate in seat 0
    for df in [output_diff,output_baseline]:
        #First seat in baseline agents should always be the same as others, so we will use index 0 of that list
        df['majority_strategy'] = str(baseline_agents[0].name)
        df['deviating_strategy'] = str(agents[deviating_seat].name)
        df['deviating_seat'] = deviating_seat

        #If we want to include a different # of cards to determine norm strength, this will be important
        df['n_cards_incl_norm'] = n_largest_for_norm
        df['experiment_date'] = str(datetime.datetime.now())
        df['weights_name']=weights_json_path
        df['pack_name']=packs_input_file
        df['batch_id'] = batch_id
        df['n_rounds'] = rounds
        df['n_cards_in_pack'] = n_cards_in_pack
        df['rotate'] = rotate_option

        #Use our renaming dictionary to put the archnames in our dataframes
        df.rename(columns=arch_remapping_dictionary, inplace=True)

        
    #Write the dataframe here that includes all of our simulation picks
    diff_str = abs_dir_path + "equilibrium_data/"+output_str.replace(".csv","_diff.csv")
    output_diff.to_csv(diff_str, mode='a',header=(not os.path.exists(diff_str)), index=False)

    baseline_str = abs_dir_path + "equilibrium_data/"+output_str.replace(".csv","_baseline.csv")
    output_baseline.to_csv(baseline_str, mode='a',header=(not os.path.exists(baseline_str)), index=False)

    print('complete' + str(datetime.datetime.now()))

def simulation_batch_runner(experiment_list:list, 
baseline_list:list,
cards_path:str,
weights_json_path:str,
weights_df_path:str, 
packs_input_file:str,
abs_dir_path:str,
n_iter:int=10,
n_cards_in_pack:int=14,
n_rounds:int=3,
n_largest_for_norm:int=23,
deviating_seat:int=0,
rotate_option:int=0,
archetypes:list=['WU','WB','WR','WG','UB','UR','UG','BR','BG','RG']):
    """Given ordered lists of experiments and baseline agents to compare the experiments to, 
    run the experiments and write to the CSV; WARNING, 1 EXPERIMENT (e.g. 1 baseline and 1 set of 
    agents to test), TAKES ~30 MINUTES ON LAPTOP WITH 1000 ITERATIONS:
    
    INPUTS: baseline_list -> list of lists containing agents for baselines 
    (e.g. list containing a list of 8 basic agents, a list of 8 hard agents, etc.
    
    experiment_list-> 3-level nested sublists; structure is a list of experiments for one agent 
    (e.g. all of the hard experiments are a sublist at this level). 
    The next level is the agents that are used in the simualtions themselves
    
    OUTPUTS: /equilibrium_data/filename.csv -> data will be put in the appropriate folder given the naming
    in comp generator"""


    print('batch start at '+str(datetime.datetime.now()))
    batch_id = uuid.uuid4()
    #For every sublist in our experiment list
    for index,experiment_subset in enumerate(experiment_list):

        #identify the baseline for this subset of experiments
        baseline_subset = baseline_list[index]

        #For each subset in the experiment (e.g. all hards, iterate through each hard)
        for experiment in experiment_subset:
            print(experiment[0],baseline_subset[0])
            equilibrium_experiment_runner(experiment,
            baseline_subset,
            n_cards_in_pack,
            n_rounds,
            cards_path,
            weights_json_path, 
            weights_df_path,
            packs_input_file,
            abs_dir_path,
            n_iter=n_iter,
            batch_id=batch_id,
            deviating_seat=deviating_seat,
            n_largest_for_norm=n_largest_for_norm,
            rotate_option=rotate_option,
            archetypes=archetypes)
            
    print('batch complete at '+str(datetime.datetime.now()))
