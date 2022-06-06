from selenium import webdriver 
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import re

class mtg_etl_client: 
    """Create a class that will instantiate a webdriver object using Chrome (selenium)
    and collect data from various subpages within 17lands; 
    
    REQUIRES SELENIUM SETUP AND THE CHROMEDRIVER.EXE FILE TO BE ACCESSIBLE IN SOME DIRECTORY
    
    This client is capable of pulling data, extracting key columns to generate new source dataframes from,
    and cleanly generating source dataframes containing a score per arch + name + color. 
    
    The output of this ETL client should be further preprocessed so that there is a corresponding file 
    in source_weights and processed weights (if you want to perform different scaling or rules to the source data,
    you can do that on new processed_weights files, but you do not need to redo this workflow since the source data
    is already saved in the source_weights folder"""

    def __init__(self, url:str='https://www.17lands.com/card_ratings', 
    archetypes:list=['WU','WB','WR','WG','UB','UR','UG','BR','BG','RG'], 
    expansion:str="VOW",headless:bool=True):
        self.url = url
        self.archetypes = archetypes
        self.expansion = expansion
        self.headless = headless

    def pull_raw_card_rankings(self,chromedriver_path:str):
        """Given the target 17lands URL, archetypes, and expansion on the client,
        pull the raw data. Also need to add path to selenium chromedriver executable file
        to run this function"""

        if self.headless == True:

            #If we're going headless, initialize your settings and add headless browsing to the driver
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            driver = webdriver.Chrome(executable_path=chromedriver_path,options=chrome_options)

        else:
            driver = webdriver.Chrome(executable_path=chromedriver_path)

            try:

                #get the url in the driver, use either URL or URL_two depending on which page you're trying to scrape
                driver.get(self.url)

                #Now let's make sure that we are choosing the right set
                select = Select(driver.find_element_by_id('expansion'))

                # select by visible text
                select.select_by_visible_text(self.expansion)

                #Wait for our table body to load
                table = WebDriverWait(driver, 20).until(EC.visibility_of_element_located(
                    (By.XPATH, '//tbody')))

                #Grab the page source
                html=driver.page_source

                #5 second cooldown 
                time.sleep(5)

                #Create list of html tables that we will later iterate through
                html_list = []
                for arch in self.archetypes:
                    #Now let's make sure that we are choosing the right set
                    select = Select(driver.find_element_by_id('deck_color'))

                    # select by visible text
                    select.select_by_visible_text(arch)

                    #Wait for our table body to load
                    table = WebDriverWait(driver, 20).until(EC.visibility_of_element_located(
                    (By.XPATH, '//tbody')))

                    #Grab the page source
                    new_html=driver.page_source

                    html_list.append(new_html)

                    #5 second cooldown between refreshes to be nice to target page
                    time.sleep(5)

                driver.quit()

                #Once we have shut down our driver, return the list of html stored in list as part of the client
                self.html_list = html_list
    
    #If something times out, let us know
            except TimeoutError as tie:
                #Error Handling 
                print("Exception " +tie+ 'has occured')
                driver.quit()

                self.html_list = None
        
    def parse_html_to_raw_df(self, target_cols:list):
        """Given pipeline object with valid html list, 
        generate dataframes with data for each archetype given
        certain target columns for use in weight generation"""

        #Create a row of the table for the data generated when there is no archetype selected
        div = soup.select_one('table')

        #This also allows us to keep color info rarity, etc. 
        source_table = pd.read_html(div.prettify())[0]

        #Iterate through the 17lands dataframe for each
        for idx, tbl in enumerate(self.html_list):

            #Since we need name as a join key later, make sure it is in the dataframe as it is processed
            if 'Name' in target_cols == False:
                target_cols.append('Name')


            #Use beautifulsoup to parse for our card ranking table and select it
            soup = BeautifulSoup(tbl,'html.parser')
            div = soup.select_one('table')


            #Convert the html to a dataframe
            new_table = pd.read_html(div.prettify())[0].loc[:,target_cols]
            new_table = new_table.loc[:,target_cols]

            #pp and % are the only non-number characters that could occur in the columns we want to be numeric
            #as a result, we should probably parse those out, but we don't want to affect the name column
            target_cols_noname = target_cols
            target_cols_noname.remove('Name')

            for col in target_cols_noname:
                new_table[col].str.replace('[a-zA-Z%]', '', regex=True)

            source_table = source_table.merge(right=new_table,how='inner',on='Name',suffixes=[None," " + self.archetypes[idx]])

        #Now that we are done parsing, let's apply basic cleaning to make everything numeric
        target_cols


        #Once we are done parsing together the dataframes and doing basic cleaning return the source table
        #Don't store it on the class so you can build multiple datasets from one 17lands pull
        return source_table

    # def generate_computed_column_from_raw_targets(self:object, target_column1:str, 
    # target_column2:str, output_colname:str):
    #     """OPTIONAL PIPELINE STEP: if you took >1 column in the parsing step because you wanted to 
    #     create a column by interacting >1 column, use the basic operations here (or add in logic as use cases arise)
    #     to perform your transformations. For now, logic will just work for 2 columns since that was the basic use case"""

        
    def process_raw_client_df_to_source_csv(self:object,df:pd.DataFrame,output_colname:str):
        """Given a raw df processed by the etl class, create a source dataframe ready to be preprocessed further,
        used in single pick experiments or be JSONified used in equilibrium experiments"""

        filename = self.expansion + "_Weights_default_" + output_colname + "_df.csv"
        #Create value to algorithmically pull the generated correct columns, whether optional step is used or not
        alg_posn = -1 * len(self.archetypes)
        
        df_archvals = df.iloc[:,alg_posn:].columns.values

        #add in name and color columns to our source dataframe
        df_archvals = df_archvals.append(df_archvals,['Name','Color'])

        #Now we deal with the weights of the missing lands by always giving them a 0
        #the intuition here being there is no incentive to take a basic land instead of a regular card
        lands = ['Swamp', 'Forest', 'Island', 'Plains', 'Mountain']
        colors =['B','G','U','W','R']
        zeros = [0 for x in self.archetypes]

        #For each of the basic five lands, append an array of 0's, the color and the land to an array
        #we will use this to make 5 new entries to the final dataframe for these cards to appear in packs
        output = []

        #For each item in the lands list, pull the corresponding color in and iterate through workflow
        for idx, l in enumerate(lands):
            new_zeros = zeros.copy()
            new_zeros.append(l)
            new_zeros.append(colors[idx])
            output.append(new_zeros)

        #Create df of our land outputs
        dummies_df = pd.DataFrame(output, columns=df_archvals.columns.values)

        #Add the lands to the df and sort all by name; create new index too
        final_weights_df = pd.concat([df_archvals, dummies_df]).sort_values(by='Name').reset_index(drop=True)
        final_weights_df = final_weights_df.loc[:, final_weights_df.columns != 'IWD']

        #Since we are done creating our source weight file, ship it off to the source weights folder
        final_weights_df.to_csv("weights_data/source_weights/"+filename, index=False)

class mtg_processed_csv_json_generator:
    def __init__(self, default_df_name:str, 
    target_col_suffixes:str,
    archetypes:list=['WU','WB','WR','WG','UB','UR','UG','BR','BG','RG']
    ):
        self.archetypes = archetypes
        self.target_col_suffixes = target_col_suffixes
        self.df = pd.read_csv("weights_data/source_weights/"+self.default_df_name)

    def convert_source_df_to_processed_df(self,zero_out_noncolor_archs:bool=False,
    min_max_scale:bool=False, min_max_range:tuple=(1,5)):
        '''REQUIRED STEP TO CONVERT SOURCE_WEIGHT FILE TO A FILE THAT CAN BE CONVERTED TO JSON 
        OR ELIGIBLE FOR EXPERIMENTATION AGAINST HUMAN PICKS (and put in processed weights folder)
        
        Take a dataframe created from our 17lands web scraper and convert it into 
        an array that we can feed into our simulator generator. By default, it will simply reshape 
        the source_dataframe and make it ready to be used in experiments comparing human picks to 
        bot picks (so you can make files for the processed_weights folder with this even if you 
        don't apply the two processing steps that we iterated through). You only need to do this
        when you are creating a net new transformation on a dataset as it will write the new file
        to a csv in the source_weights folder. 
        
        The two current workflows in this preprocessor are to apply a minmax scaler from sklearn 
        on some range (default is min of 1 max of 5 since that worked best in testing) and to 
        create 0 values for archs that are outside of the card's color schema (e.g. a blue-black card
        will get archetype values replaced whenever the archetype does not contain blue or black)'''

        #Remove nomenclature from the input file and conv
        output_filename = self.default_df_name.replace(r'_default_','').replace(r'_df.csv','.csv')

        #Strip out IWD from titles so we can regex match the color column to the archs
        self.df.columns = self.df.columns.str.replace(r'{}'.format(self.target_col_suffixes), '')

        #store the df without name and color since we'll use it a few times throughout the function
        validation_df = self.df.loc[:, ~self.df.columns.isin(['Name', 'Color'])]

        #Note that the length of the df will always be the cards in the set and the archs will be everything but the name and colors
        weights_array = validation_df.to_numpy().reshape((self.df.shape[0],len(self.archetypes))) 

        #Apply additional transforms 
        if zero_out_noncolor_archs == True:

            #store the colnames that are not name or color to confirm they're the right datatype
            cols = self.df.loc[:, ~self.df.columns.isin(['Name','Color'])].columns.values
            #print(validation_df.columns.values)
            for c in cols:
                validation_df[c] = validation_df[c].astype('float')

            #Let's use iterrows to rip through the columns that are colors
            color_cols = validation_df.columns.values

            #We will create arrays w 0 and 1 here to see what's good 
            master_list = []

            #Iterate thrugh each row in the df 
            for index, row in self.df.iterrows():

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

        #Use default mm scaler
        if min_max_scale == True:

            scaler = MinMaxScaler(feature_range=min_max_range)

            weights_array = scaler.fit_transform(weights_array.reshape((len(self.archetypes),self.df.shape[0])))

            weights_array = weights_array.reshape((self.df.shape[0],len(self.archetypes)))

            #Add in the naming convention if we generated a set like this
            output_filename = output_filename.replace(r'.csv','_minmax.csv')

        self.processed_filename = output_filename
        self.processed_df = df_output = pd.DataFrame(weights_array)


    def write_to_processed_csv(self):
            """Write stored processed_df in class to a csv file in the processed tab
            to be used in pick accuracy/human pick experiments"""

            #If the filename is currently for a json ending, return it to csv ending
            self.processed_filename = self.processed_filename.replace('.json','.csv')
            self.processed_df.to_csv(self.processed_filename, index=False)

    def write_to_json_weights_writeback(self):
        """Write our weights array to a JSON that can be used in equilibrium experiments"""

        #Since we do not pull out rows, nor do we reorder any indexes, 
        #we can just add the name column back in to set as an index for 
        #the JSON writeback of our weights
        self.processed_df['name'] = self.df['Name']

        #Make the name the dict key with the weight arrays being the values
        json_content = self.processed_df['name'].set_index('name').T.to_dict()

        #Replace CSV ending with JSON
        json_filename = self.processed_filename.replace('.csv','.json')
        #Dump the file to the designated path
        with open(json_filename,"w+", encoding='utf-8') as fp:
            json.dump(json_content, fp, ensure_ascii=False)

        #Drop the name column in case you're writing to JSON first
        self.processed_df.drop(columns='name',inplace=True)

class mtg_set_writeback_generator:
    """Sometimes, the mtg_json file for the set is not super clean and there are duplicates of cards
    (usually due to arena variants being added as new cards to the JSON file and starting with the prefix A-.
    
    Sometimes cards also get removed from the set when we look for unique cards in the set, so we will use the 
    target weights file as a validation for the writeback for a given set. You only need to create one writeback 
    file per set that you want to run equilibrium experiments on. 
    """

    def __init__(self, mtg_json_path:str, 
    default_weight_df_path:str,
    expansion:str="VOW"):
        self.mtg_json_path = mtg_json_path
        self.expansion = expansion
        self.default_weight_df_path = default_weight_df_path


    def generate_writeback(self):
        """Given the raw mtg_json file path, get writeback information for
        every card that we have weight information on. Warning messages will be given 
        if card names are missing or if there are other idiosyncrasies with the writeback 
        generation"""
        cardnames = []
        cards = []
        with open(self.mtg_json_path, encoding="utf8") as f:
            data = json.load(f)
            
            #Iterate through each card in the raw json file
            for c in data['data']['cards']:

                #Create regex var for the pattern A-, which denotes some duplicates in the mtgjson file
                p = re.compile('[A]-')

                #If you find the letter A- pattern as the start of the cardname, remove that pattern and add the regular name
                #to our list of cards
                if len(p.findall(c['name']))>0:
                    c['name'] = c['name'].replace('A-','').lstrip()

                    if c['name'] in cardnames:
                        print('passed')
                    else:
                        print(c['name']+" getting name fixed and added to list")
                        cardnames.append(c['name'])
                        cards.append(c)

                elif '//' in c['name']:
                    c['name'] = c['name'].split("//")[0]
                    c['name'] = c['name'].strip()
                    print(c['name'])
                    
                    if c['name'] in cardnames:
                        print('passed')
                    else:
                        print('added')
                        cardnames.append(c['name'])
                        cards.append(c)

                else:
                    if c['name'] in cardnames:
                        print('passed')
                    else:
                        print(c['name'])
                        print("added normally")
                        cardnames.append(c['name'])

            f.close()
        
        self.raw_writeback = cards
        self.writeback_cardsnames = cardnames
    
    def validate_writeback(self:object, default_weights_df_path_for_set:pd.DataFrame):
        """Compare the cards in our writeback to the cards in the cards in the weights df
        and highlight any descrepancies in cardnames/cards. With descrepancies highlighted,
        you can manually edit/remove items from this class' cardlist before writing to json"""

        #Load in weight files for the set and get sets of the cardnames
        df = pd.read_csv(default_weights_df_path_for_set)
        nameset = set(df['Name'])
        cardset = set(self.writeback_cardsnames)

        #For each comparison of lists, highlight # of problem cards and then highlight the names
        if len(nameset-cardset)>0:
            print("There are {} items in the default weights df, but not in our JSON writeback").format(str(len(nameset-cardset)))
            print('The items in weight df, but not JSON are:')
            for x in nameset-cardset:
                print(x)

        if len(cardset-nameset)>0:
            print("There are {} items in the JSON writeback, but not our weights").format(str(len(cardset-nameset)))
            print('The items in JSON, but not df are:')
            for x in cardset-nameset:
                print(x)

        #If there are no descrepancies, our validation passes; else, we fail
        if len(cardset-nameset)==0 and len(nameset-cardset)==0:
            self.validation_flag = True
            print('validation successful')

        else:
            self.validation_flag = False
            print('validation failed')

    def writeback_to_json(self:object,prefix:str):
        """If validation worked, then we can dump our file with a prefix"""

        if self.validation_flag == True:
            #Autogenerate the name based on expansion in init function
            fname = "json_files/" + prefix + "_" + self.expansion + "_writeback.json"

            #open our new file and dump the validated writeback in
            with open(fname, "w+", encoding="utf8") as f:
                json.dump(self.raw_writeback,f, ensure_ascii=False)
                f.close()

        else:
            print('Either you have not yet validated this output or validation failed, please check your data')