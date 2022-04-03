from draftbot_sim_adapted import *
import sqlite3
import pandas as pd
import numpy as np
import json
import os
from Agents import *
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, normalize


##CLASS DEFS THAT ALLOW US TO USE BOTS BUILT FOR CLOSED-CIRCUIT SIMULATIONS 
##TO RUN ON HUMAN DATA... ASK TRISTAN QUESTIONS ABOUT DOCUMENTATION AND WORKFLOW
class PseudoSet:
    def __init__(self, card_names):
            self.n_cards = len(card_names)

class PseudoDraft:
    def __init__(self,
    n_archs,
    weights, 
    packs,
    card_names):
        self.n_archetypes = n_archs
        self.set = PseudoSet(card_names)
        self.archetype_weights = weights
        self.packs = packs
        self.picks = np.zeros((1,self.set.n_cards,42))
        self.options = np.zeros((1,self.set.n_cards,42))
        self.passes = np.zeros((1,self.set.n_cards,42))
        self.drafter_preferences = np.ones((1,n_archs))

#COMPLETE
def weight_generator(df_path, output_filename, zero_out_noncolor_archs=False, 
min_max_scale=False,min_max_range=(1,5)):
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

    #Strip out IWD from titles so we can regex match the color column to the archs
    df.columns = df.columns.str.replace(r'IWD', '')

    #Archs will be all cols but the name and color coming from the scraper
    n_archs = df.loc[:, ~df.columns.isin(['Name', 'Color'])].shape[1]

    #store the df without name and color since we'll use it a few times throughout the function
    validation_df = df.loc[:, ~df.columns.isin(['Name', 'Color'])]

    #Note that the length of the df will always be the cards in the set and the archs will be everything but the name and colors
    weights_array = validation_df.to_numpy().reshape((df.shape[0],n_archs))
    #if  scale_min_max == True and scale_by_max == True: throw assertion error 

    #Scale converting the card weights as a fraction of the best card in each arch
    # if scale_by_max == True:

    #     #Now Scale the archetypes so the card weights are a function of the best arch + best card
    #     weights_array_archs = weights_array.sum(axis=0)

    #     wmax = weights_array_archs.max()

    #     weights_array_archs_final = (weights_array_archs / wmax).reshape((1,10))

    #     weights_array = weights_array * weights_array_archs_final

    #Use default mm scaler
    if min_max_scale == True:

        scaler = MinMaxScaler(feature_range=min_max_range)

        weights_array = scaler.fit_transform(weights_array.reshape((10,272)))

        weights_array = weights_array.reshape((272,10))

        

    #Apply additional transforms 
    if zero_out_noncolor_archs == True:

        #store the colnames that are not name or color to confirm they're the right datatype
        cols = df.loc[:, ~df.columns.isin(['Name','Color'])].columns.values
        print(validation_df.columns.values)
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

    df_output = pd.DataFrame(weights_array)
    df_output.to_csv('weights_data/'+output_filename, index=False)



#dump path 'C:/Users/trist/OneDrive/Documents/draft_data_public.VOW.PremierDraft.csv'
#nrows really 420000
# IN PROGRESS 

def simulation_generator(draft_dump_path, output_name, weights_path, agents, nrows=42000, pick_index=11):
    """Run simulations with out closed circuit bots on real 17lands data
    """

    df = pd.read_csv(draft_dump_path,nrows=nrows)

    weights_df =  pd.read_csv(weights_path)
    weights = weights_df.to_numpy()

    unique_ids = df.draft_id.unique()

    agent_names = [agent.name for agent in agents]
    final_output = []

    match_cols = agent_names + ['Real']

    #let's iterate through every draft that we pulled from the dump file 
    for id in unique_ids:

        #Add try/except logic for drafts that have 42 picks. For incomplete drafts, print out drafts in fail list and skip
        try:
            new_df = df[df['draft_id']==id]
            packs_df = new_df.filter(regex='pack_card')
            picks_df = new_df.filter(regex='pool')

            cards = []
            for c in df.columns:
                if 'pool' in c:
                    cards.append(c.replace('pool_',''))

            #Picks seem like a mess coming from 17lands... let's fix that. 
            picks = new_df.iloc[:,pick_index]

            cards_names = sorted(set(cards))

            master_array = []

            #Create array of zeros except for the card thats picked as our actual value
            for p in picks:

                #
                new_array = np.zeros((len(cards_names)))
                # try:
                new_array[cards_names.index(p.strip())] = 1
                # except:
                #     new_array[cards_names.index(p+' ')] = 1

                master_array.append(new_array)


            #Create a df out of our picks using the cardnames as column values
            picks_df = pd.DataFrame(master_array, columns=cards)

            #Packs array is trustworthy, no data issues 
            packs_array = packs_df.to_numpy()

            #Packs array is now clean to go
            picks_array = picks_df.to_numpy()

            #Passes array is a function of these 
            passes_array = packs_array - picks_array

            card_names = picks_df.columns.values

            #Instantiate the object
            psd = PseudoDraft(10, weights, 
            packs_array,
            card_names)

            #Create lists to hold totals for picks
            totals = []
            totals_top3 = []

            #Iterate through all the rounds 
            for n in range(0,42):

                #Create accumulator lists for the picks made by the bots (and top 3)
                picks = []
                top3_picks = [] 
                for a in agents: 
                    print(psd.packs[n])
                    array = a.decision_function(psd.packs[n],psd,0).reshape((psd.set.n_cards))

                    picks.append(array.argmax())


                    top3 = np.argpartition(array, -3)[-3:]
                    top3_picks.append(top3)

                #Add in the actual pick 
                picks.append(picks_array[n].argmax())
                totals.append(picks)
                totals_top3.append(top3_picks)


                #Now we need to input the data from what actually happened so the bots can start from a net new position 
                psd.picks = picks_array[0:n+1].reshape(1,psd.set.n_cards,n+1)
                psd.options = packs_array[0:n+1].reshape(1,psd.set.n_cards,n+1)
                psd.passes = passes_array[0:n+1].reshape(1,psd.set.n_cards,n+1)


            #Now let's update the drafter preferences so we're not eternally multiplying by 1
                psd.drafter_preferences = (
                psd.drafter_preferences +
                np.einsum('ca,dc->da', psd.archetype_weights, picks_array[n].reshape(psd.set.n_cards,1)))

            #Let's create some dataframes

            #The first one will hold whether or not we matched the picks

            #Make sure nothing is a string when converting
            df_matches = pd.DataFrame(totals, columns=match_cols).astype(int)


            #Create an accumulator so we can index on our match cols later
            sumcol_names = []
            #For every agent we're evaluating, let's get a boolean column that tells us whether or not we matched
            for agent in agent_names:

                #Dynamically creat these cols with same suffix
                colname = agent + '_Match'

                df_matches[colname] = df_matches[agent] == df_matches['Real']

                sumcol_names.append(colname)


            #Number of cases where our bots predict the taken card in the top 1
            #Note that the baseline here would be 3/42 since we would always be right with 1 option left in the pack

            #See if at least one of our bots got this pick right
            t1_sum = sum(df_matches.loc[:,sumcol_names].sum(axis=1)>0)

            #like with column creation, lets create a loop that peforms operations and names dynamically
            #Let's get a sum per model this time


            #Create an accumulator that holds the sums based on sumcol names 
            t1_sums = []
            for col in sumcol_names:

                #Value here is the number of picks (out of 42) a given agent gets right
                df_matches[col].sum()

                t1_sums.append(df_matches[col].sum())


            dft3 = pd.DataFrame(totals_top3, columns = agent_names)

            #We can just inherit the real column and port it over here
            dft3['Real'] = df_matches['Real']

            #Now let's iterate like before 
            t3_sumcol_names = []

            for agent in agent_names:

                colname = agent + '_t3_match'
                dft3[colname] = dft3.apply(lambda x: x['Real'] in x[agent], axis=1)

                t3_sumcol_names.append(colname)
            

            #Number of cases where our bots predict the taken card in the top 3
            #Note that the baseline here would be 9/42 since the packs where there are 3 or less cards left would be troublesome
            t3_sum = sum(dft3.loc[:,t3_sumcol_names].sum(axis=1)>0)

            #Let's get a sum per model

            #Create an accumulator that holds 
            t3_sums = []
            for col in t3_sumcol_names:

                #Value here is the number of picks (out of 42) a given agent gets right
                t3_sums.append(dft3[col].sum())

            #List with sublists for values

            fo = [id, 
            t1_sum]
            fo.extend(t1_sums)
            fo.append(t3_sum)
            fo.extend(t3_sums)
            final_output.append(fo)

            #List with sublists for colnames
            cols =["id",
            "t1_sum"]
            cols.extend(sumcol_names)
            cols.append("t3_sum")
            cols.extend(t3_sumcol_names)

            resdef = pd.DataFrame(final_output, columns=cols)

            resdef.to_csv(output_name)

            df_matches.to_csv('t1_performance.csv', mode='a')

            dft3.to_csv('t3_performance.csv', mode='a')

        
        except IndexError:
            print(id)
            pass
