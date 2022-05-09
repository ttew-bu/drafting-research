from Agents import *
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

##CLASS DEFS THAT ALLOW US TO USE BOTS BUILT FOR CLOSED-CIRCUIT SIMULATIONS 
##TO RUN ON HUMAN DATA... ASK TRISTAN QUESTIONS ABOUT DOCUMENTATION AND WORKFLOW

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

##HELPER FUNCTIONS
#FUNCTIONS THAT EITHER DRIVE FOR WEIGHT/SIMS
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

##DEPRECATED, NEW PREPROCESSING FUNCTION DOES ALL OF THIS GIVEN THE INDEX FILE
# def generate_accuracy_score_df(df_matches:pd.DataFrame, dft3:pd.DataFrame, id:str, agent_names:list, output_name:str):
#     """Given the matches dataframe of top1 pick performance, top3 pick performance, the draft id, agents used in the sim,
#     and a target filename, create a dataframe holding t1/t3 pick information at the draft level
    
#     Output here is 3 dfs described below (they're all written to filepaths as opposed to being returned):
    
#     First is a df with columns for t1/t3 performance per agent where a row is the number of times (out of 42)
#     the agent selected the correct card given a certain draft ID. This is a relatively small df compared to the others
#     and is primarily good for generalizing performance at the draft level, classifying drafts, and high-level analyses joining in player data.
    
#     The next two dfs are the t1/t3 performance where a row is a given pick of a draft (e.g. ID 1, pick 1) where each column is the t1/t3
#     accuracy at a given pick, represented with a boolean 1/0. The value from this df is to understand behavior at certain picks and create
#     the accuracy-by-turn graphics"""
#     #For every agent we're evaluating, let's get a boolean column that tells us whether or not we matched
#     t1_sums = []
#     t3_sums = []

#     for agent in agent_names:

#         #Dynamically creat these cols with same suffix
#         colname = agent + '_match'
#         df_matches[colname] = df_matches[agent] == df_matches['Real']
#         t1_sums.append(df_matches[colname].sum())

#         t3_colname = agent + '_t3_match'
#         dft3[t3_colname] = dft3.apply(lambda x: x['Real'] in x[agent], axis=1)
#         t3_sums.append(dft3[t3_colname].sum())


#     #List with sublists for values
#     final_output = [id]
#     final_output.extend(t1_sums)
#     final_output.extend(t3_sums)

#     #Our columns will be ID, then agent t1 scores, then agent t3 scores
#     cols =["id"]
#     cols.extend([x+ "_t1_score" for x in agent_names])
#     cols.extend([x+ "_t3_score" for x in agent_names])

#     #Results dataframe with agents, t1sum, t3sum, and id)
#     results_dataframe = pd.DataFrame([final_output], columns=cols)

#     #Write the dataframe here that includes all of our simulation picks
#     results_str = "results_data/"+output_name
#     results_dataframe.to_csv(results_str, mode='a', header=(not os.path.exists(results_str)),index=False)

#     #Add ID Data to our Key Files
#     df_matches['id'] = id
#     dft3['id'] = id

#     #let's also send our pick by pick performance data to our performance folder
#     #Create files to track t1/t3 accuracy and place them in the appropriate folder
#     t1_str = 'performance_data/' + output_name.replace(r'.csv','_t1_performance.csv')
#     df_matches.to_csv(t1_str,mode='a',header=(not os.path.exists(t1_str)), index=False)

#     #Same process as aobve, but for t3 performance
#     t3_str = 'performance_data/' +output_name.replace(r'.csv','_t3_performance.csv')
#     dft3.to_csv(t3_str, mode='a',header=(not os.path.exists(t3_str)), index=False)

def generate_index_residual_delta_df(df_matches:pd.DataFrame, psd:object, scores:list,array_keys:list,agent_names:list, output_name:list,suffix:str):

    #We will start with one df with the deltas
     idx_df = pd.DataFrame(scores,columns=agent_names)
     keys_df = pd.DataFrame(array_keys,columns=[a + '_y' for a in agent_names])
    #Create an array of shape n_cards that is the sum of the card's weights in the x number of archetypes
     norm = psd.archetype_weights.sum(axis=1)

     #We will add the real and ID columns to our data so we can join them
     idx_df['Real'] = df_matches['Real']
     idx_df['id'] = df_matches['id']

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
     index_str = "index_data/"+output_name.replace(r'{suffix}','_index_{suffix}'.format(suffix=suffix))
     idx_df.to_csv(index_str, mode='a',header=(not os.path.exists(index_str)), index=False)

##MODULE FUNCTIONS 
#COMPLETE FUNCTIONS TO CALL FOR WEIGHT AND SIMULATIONS 
def weight_generator(df_path:str, zero_out_noncolor_archs:bool=False, 
min_max_scale:bool=False,color_str:str='IWD',min_max_range:tuple=(1,5)):
    '''Take a dataframe created from our 17lands web scraper and convert it into 
    an array that we can feed into our simulator generator. By default, this function
    will take our weights array from 17lands' IWD metric (effectively wins above replacement
    in a given archetype or in any deck) and then apply a min-max scale across the archetype 
    before zeroing out weights for archs the card does not represent. 
    
    For example, a blue-black card
    will first get scaled across the 10 archetypes along with the other cards in a set, then values will 
    be replaced whenever the archetype does not contain blue or black'''

    #Read df
    df = pd.read_csv(df_path)

    #Remove nomenclature from the input file and conv
    output_filename = df_path.replace(r'_default_','').replace(r'_df.csv','.csv')

    #Strip out IWD from titles so we can regex match the color column to the archs
    df.columns = df.columns.str.replace(r'{}'.format(color_str), '')

    #Archs will be all cols but the name and color coming from the scraper
    n_archs = df.loc[:, ~df.columns.isin(['Name', 'Color'])].shape[1]

    #store the df without name and color since we'll use it a few times throughout the function
    validation_df = df.loc[:, ~df.columns.isin(['Name', 'Color'])]

    #Note that the length of the df will always be the cards in the set and the archs will be everything but the name and colors
    weights_array = validation_df.to_numpy().reshape((df.shape[0],n_archs)) 

    #Apply additional transforms 
    if zero_out_noncolor_archs == True:

        #store the colnames that are not name or color to confirm they're the right datatype
        cols = df.loc[:, ~df.columns.isin(['Name','Color'])].columns.values
        #print(validation_df.columns.values)
        for c in cols:
            validation_df[c] = validation_df[c].astype('float')

        #Let's use iterrows to rip through the columns that are colors
        color_cols = validation_df.columns.values

        #We will create arrays w 0 and 1 here to see what's good 
        master_list = []

        #Iterate thrugh each row in the df 
        for index, row in df.iterrows():

            #We will accumulate rows in an array and then convert this later
            rowlist = []

            #Store the color for the row in a variable (e.g. W or B)
            row_color = row['Color']

            #iterate through all indices that are for the archweights
            for column in color_cols:

                #now iterate through the 10 columns w color combos to find matches 
                try:
                    if row_color in column:
                        rowlist.append(1)
                    else:
                        rowlist.append(0)

                #Cards without colors will just get a 0 by default for now
                except TypeError:
                    rowlist.append(0)

            #Now that we have that, let's regex match to our cols array
            master_list.append(rowlist)

        #Convert our list of 1's and 0's for color into arrays so we can multiply our weight values by this
        color_check_array = np.asarray(master_list)

        #Now, multiply the array of weights (ranging from 0,5) by 1's and 0's to 
        #zero out card weights in arrays that are not inclusive of the card's color
        weights_array = weights_array * color_check_array

        #Add in the naming convention if we generated a set like this
        output_filename = output_filename.replace(r'.csv','_colors.csv')
        print(output_filename)

    #Use default mm scaler
    if min_max_scale == True:

        scaler = MinMaxScaler(feature_range=min_max_range)

        weights_array = scaler.fit_transform(weights_array.reshape((10,272)))

        weights_array = weights_array.reshape((272,10))

        #Add in the naming convention if we generated a set like this
        output_filename = output_filename.replace(r'.csv','_minmax.csv')
        print(output_filename)
        
    df_output = pd.DataFrame(weights_array)
    df_output.to_csv(output_filename, index=False)

#ADDING IN NEW DATA OUTPUTS 4/10/22
def simulation_generator(suffix:str, draft_dump_path:str, weights_path:str, agents:list, 
nrows:int=42000, pick_index:int=11, cards_per_draft:int=42):
    """Run simulations with out closed circuit bots on real 17lands data by taking a dump of picks, a weights file,
    a list of instantiated agent classes, the number of rows to select, the position of the pick column in the dump file,
    and the cards in the draft to validate that we do not assess partial/incomplete drafts to keep data high quality.

    The prefix is a label you can add to label experiments e.g. "medium_" + path tells me I did the medium exp. with certain files

    This function returns several CSV files and will store them in certain folders:
    results_data/results csv -> rows represent a single draft (marked by ID), columns are # picks correct by agent and # times the bot matches the top 3 in the pack.
    Can be used for high level performance/accuracy estimations, relatively small compared to other files

    performance_data/t1/t3 performance csv -> rows are a single pick in a single draft (e.g. ID, p1), rows are booleans as to whether or not the bot made the right pick or 
    placed the correct card in the t3 in a given pick. Will always be 42x (or however many cards are in the draft) the number of rows of the results csv

    index_data/index data csv -> rows are a single pick in a single draft (e.g. ID, p1), rows are the picks off from the correct card (e.g. we ranked the card 3rd but
    it was first, so we were 2 picks off), the bot score delta (e.g. our bots top score was 1.2, the actual selected card was scored 1.1 by the bot, so we have a delta
    of 0.1 representing how close our bots calculations were), and the norm delta (e.g. given an array where one item is sum of all archetype scores for each card to
    represent the "strength/value" of the card, how far off was our actual pick from the strongest card based on this norm?). The norm delta matches the paper's
    definition of how far off we are from making the "right" pick because it accounts for some measure of absolute strength, as opposed to the scores spat out by each
    bot, which are partially a function of each bot's decision function. 
    """

    #Read weights csv
    df = pd.read_csv(draft_dump_path,nrows=nrows)

    #Add automated stamps to our data so we can version it; start by taking weight set used, then add n_drafts
    output_name = weights_path.replace(r'.csv','_{num}_{suffix}.csv').format(num=str(int(nrows/cards_per_draft)), suffix=suffix)
    output_name = output_name.replace(r'weights_data/processed_weights',"")

    #Pull our weights df and send to array
    weights_df =  pd.read_csv(weights_path)
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

            #T3 workflow deprecated in favor of processing post-simulation.
            #dft3 = pd.DataFrame(totals_top3, columns = agent_names)

            #We can port over the real column for our top 3 df
            #dft3['Real'] = df_matches['Real']

            #generate_accuracy_score_df(df_matches, dft3, id, agent_names, output_name)

            #INDEX FILE IS NEW SOURCE TABLE FOR PREPROCESSING FUNCTION
            generate_index_residual_delta_df(df_matches,psd,scores,array_keys,agent_names,output_name,suffix)
                
        except IndexError:
            print(id)
            pass

def run_experiments(dictionaries:list,string:str,weight_file_name):
    """Iterate through a list dictionaries containing weights, filenames, and agents to automatically generate our experiment data given
    The idea here being"""

    #for each item in our processed weights directory
    for d in dictionaries:
        #Iterate through our files
        string = 'weights_data/processed_weights' + '/' + weight_file_name
        simulation_generator(d['suffix'],d['draft_str'],string, d['agents'], d['n_iter'], 11)

##ANALYSIS HELPER FUNCTIONS:
def display_draftwise_results(result_path:str,include_t3_data=False,n_picks:int=42):
    """Present results that show what % the bots match 
    human picks given the filename holding the picks"""

    #Read csv
    df = pd.read_csv(result_path)

    #Get rid of dummy index column
    df = df.iloc[:,1:]

    #If you also want to know about t3 data, you can set this flag to true
    if include_t3_data == False:
        
        #Remove all cols that include the str t3
        df = df.loc[:,~df.columns.str.contains('t3')]
    
    #Mean here is avg # of picks matched, so the rate is the avg/total picks
    results = df.mean()/n_picks
    results = results.sort_values(ascending=False)

    print(results)

def display_pickwise_results(performance_string:str, visualization:str='table'):
    """Present results that show % bots match human picks at the pick level. 
    That is, if there are 3 packs of 14 cards, then there are 42 picks at which
    the bots will have different accuracies. Each multiple of 14 will have an
    accuracy of 100% since you can only pick from one card at these points"""

    df = pd.read_csv(performance_string)

    df.rename(columns={'Unnamed: 0':'match_index'}, inplace=True)

    #only include cols that keep track of our match boolean values
    #also set all colnames to lowercase because some old files have diff capitilzation patterns
    df.columns = df.columns.str.lower()
    df = df.loc[:,df.columns.str.contains('match')]

    #Sub out the text from our input file (Python isn't reading the T/F as boolean by default)
    #df = df.replace({False: 0, True: 1})
    output_df = df.groupby('match_index').mean()

    #Print a df where we have 42 rows (1 per pick in a standard draft), where each bot has an accuracy %
    if visualization=='table':
        return output_df

    #If you also want a plot, add this logic in here
    else:
    
        df.groupby('match_index').mean().plot()

        #Move legend off to the side
        plt.legend(loc=(1.04,.35))

        #Show off the plot
        plt.show()

def open_index_file_and_preprocess(index_file_path:str,source_weights_file_path:str):
    """Engineer all the features that we had in the other 3 filetypes, but do that from 1 source
    file to reduce clutter in the repository. Given an index file, generate t1, t3, accuracies.
    
    OUTPUT: Df containing raw data for picks off, norm delta, bot score residual, t1 accuracy boolean, t3 accuracy boolean,
    for each pick of each draft.There is one column for each of these per bot in the simulation (e.g. norm delta, picks off, accuracy for bot x),
    so our data can get a bit wide. We can use regex search on the column names to pull out the subsets most relevant to our analysis (or future feature
    engineering). Currently, some common analyses here would be to:
    regex on t1_match (and create a column by cloning the id column and other relevant attributes from the original df) for t1 accuracy analysis
    regex on picks off (and create necessary columns by cloning from source df) for understanding how good the bots are at each pick (can do this for norm + residual too)


    Can join in data from the dump file to include win rates etc using the ID column for cross-sectional
    analysis, and can join colors in given the weights source file. Since all of our source weight files have a convenient color column,
    we can just use any of those files from the same set to get our weight columns without having to parse the set's JSON file"""

    df = pd.read_csv(index_file_path)

    #Iterate through each of our picks off columns to get the agent name (in theory could've been done w/ any other autogenerated column,
    #since we can strip the suffix off whatever column as long as there is 1 per bot)
    for x in list(df.filter(regex='picks_off').columns): 

        #Set each column to numeric before doing the number matching here
        df[x] = df[x].apply(pd.to_numeric, errors='coerce')

        #Strip out the picks off part of the column name to get our agents themselves
        x_str = x.replace('_picks_off','')

        #Dynamically make t1 accuracy columns and names for the cols
        t1_string = (x_str + "_t1_match")
        df[t1_string] = np.where(df[x]==0,1,0)

        #Dynamically make t3 accuracy columns
        t3_string = (x_str + "_t3_match")
        df[t3_string] = np.where(df[x]<=3,1,0)

    #Now, let's add in the colors for each card in the dataset from our source weights file
    source_df = pd.read_csv(source_weights_file_path)

    colors = source_df['Color'].reset_index()

    #Pull out the item from the colors index corresponding with the real pick
    df['Real_colors'] = df.apply(lambda x: colors['Color'].iloc[x['Real']], axis=1)

    #Replace nan with a single character N, so we can do single character color matching later
    df['Real_colors'].replace({np.nan:"N"}, inplace=True)

    return df

#WANTS -> Pull, simulate 17lands style + equilibrium style, analyze