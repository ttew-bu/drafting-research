# drafting-research
Computational experimentation repository for Magic the Gathering drafts both against human picks experiments and against other bots. The intent of this repository is to hold the tools to collect/clean data for weights and cards to run the simulations, flexible infrastructure for both experiment types, and helper functions/rich data to run analyses on to answer questions around strategy selection in drafts. This repository powers research by Gerdus Benade and Tristan Tew of Boston University's Questrom School of Business. 

## There are 3 main modules in this repository:

### mtg_etl_pipeline.py
Holds classes/methods surrounding creating compatible weights files (in csv/json format) and card writebacks (json) to be used as inputs. If you are trying to make a relatively complex transformation for a weight input, it is recommended you look closely at the implementation of generate_computed_column_from_raw_targets() and convert_source_df_to_processed_df() to make sure you are doing what you want to the data and have what you need. The intent of the functions in this module are to generate reliable input data that can be fet into simulations. 

### mtg_experimentation_utilities.py
Holds functions, classes, and methods for running both kinds of experiments listed above, and analyzing output datasets. This ranges from having classes for the entire draft and the set from which the cards are generated to unbound functions that convert dataframes into visualizations or apply other transformations. The intent with this file is that you are taking input data and generating output data/insights for analysis and the paper. 

### Agents.py
Holds the classes and decision functions that constitute the strategies we want to test; all agents are classes of a similar structure and the primary distinction between bots is how their decision functions play out. Additionally, there are hyperparameters that can be adjusted to create more or less extreme versions of the class' strategy, which leaves ample room for further experimentation. The intent of this file is to house similarly structured classes for different drafting strategies and improve development time of other agents, should other strategies become of interest. 

## There are several key folders to be aware of:

### notebooks
This holds the demos of each experiment workflow (e.g. human picks v. equilibrium/all bots), relevant past research on using different weights and rough benchmarks for accuracy on human picks, old notebooks for the etl workflow (there are some minor changes to get things clean, but it is broken down in a step by step manner), and other notebooks containing relevant visualizations/functions relevant to our paper. When working in notebooks/creating new notebooks, note that it is important to add a variable dir_path that is the absolute path to the repository on your device/virtual machine (e.g. C://users/person/drafting-research); this will allow you to write new files to the correct folders or access current ones. 

### json_files
This holds the raw JSON file (e.g. cards and all associated metadata) from MTG_JSON for Crimson Vow, the cleaned and validated JSON file match the set of cards from 17Lands/in the weights files (this is called the VOW_writeback.json file), and weights files that are preprocessed for use in equilibrium experiment (e.g. nested dictionary with card names as key and the value being a dictionary with a value for each archetype; the current one is called "weights_writeback"). For each set experimented upon, you would need one raw json file, one json_set_writeback file and however many weights writebacks as you'd like to test (the functions now automatically name these, so it should be clear enough which file is which). The workflow for writebacks can be found in the 17lands notebook.

### draft_txt_files
This holds txt_files of sample drafts with the following breakdown: 1 mythic/rare, 3 uncommons, 10 commons (you can tweak these in future pack generations). By using the pack_generator functions in the experimentation_utilities module, you can generate however many packs as you'd like and run experiments on different sets. The txt files are all arrays that are readily able to be read directly by the equilibrium experiment infrastructure. The structure of each array is: n_iterations, n_rounds, n_players, n_cards in draft. Note that the final level (n_cards in draft), will contain 1's or 0's to signify which cards are in the pack based on the distribution/number of cards given to the generator. As of now, all packs generated do not allow for duplicates in a single pack. 

### equilibrium_data
This folder holds 2 kinds of CSV files: baseline and diff files. The baseline data contains results of the experiments where there is the same strategy in all 8 seats of the draft; diff contains the results where 1 seat deviates to a given strategy. These datasets can tell us the 'benefit' associated with choosing a certain strategy given that of others and more generally, can show us how preferences change at a table under different circumstances. Note that the functions in the demo workbook for equilibrium has a couple helper methods and examples of ways to manipulate this data to get meaningful outputs. The schema for each of the tables is as follows:

simid: id for the individual draft (e.g. 8 players, 3 rounds, 14 packs; there are 1000 simids in a simulation with 1000 iterations)

player: seat # (e.g. one of the 8 players)

0-10: scores for each archetype (these scores are calculated with the norms of each card in each archetype; e.g. adding app the scores in each archetype for the 42 cards selected)

top_archetype_score: the max # from the 10 archetype columns

top_arch_index: which archetype was the best (e.g. arch #3)

majority_strategy: string of the bot name used by 7/8 seats 

deviating_strategy: string of the bot used by 1/8 the seats

deviating_seat: seat index of the deviating strategy

experiment_date: timestamp of experiment (if run in batch, you can use this to see exactly which simulations you ran at a given point in time)

weights_name: filepath of the JSON weights file used to compute picks for each agent

packs_name: filepath of the txt file used to generate the picks (this is good for replicability of anything)

batch_id: id for the entire batch of expirements (e.g. if you run 20 experiments at once, this number will be the same for all of the rows for those 20 experiments)

### index_data
This folder holds highly raw files from experiments involving human pick data; it is recommended that you use the open_index_file_and_preprocess() function to parse the data into more usable data. These files are good to understand (and tune) performance for the agents in terms of matching human picks (both top 1 and top 3 accuracy) in addition to assessing accuracy under different input weight criteria. The unprocessed schema for these tables looks as follows:

real: the real card selected (it will be the index within the N cards in the set, e.g. card 171 in the set)
id: the id of the draft from 17lands

PER AGENT YOU EXPERIMENT YOU WILL HAVE ONE OF THE FOLLOWING
{bot_name} = the scores for the 14 cards in the set that are generated by the bot (e.g. array of 14 items)

{bot_name__y} = the indexes of the cards within the set in an array (e.g. [29,14,etc...] would be the first card was card index 29, second was 14)

{bot_name__index} = the card index of the card that was selected in the pack in real life (e.g. the 'correct answer')

{bot_name__picks_off} = the rank of the card selected (e.g. {bot_name__index}) within {bot_name} (will be zero if the bot selected the right card)

{bot_name__internal_residual} = the difference between the card selected and the correct card in terms of the scores given by the bot (e.g. bot_name[correct_idx] - bot_name[agent_selection_idx]) - this will also be zero if the correct card is selected

bot_name__norm_delta = the difference in "norm strength" (e.g. the sum of weights for a given card across all 10 archetypes) between the correct card and card given by bot. This will be zero if the correct card selected and is intended to show some sense of card strength (e.g. when we are wrong, does the bot's strategy tend to choose cards that are generally weaker or generally stronger? Are they making the wrong pick because of bias toward/against an archetype?)

There are several helper functions in the human_pick notebook that simplify this information to answer questions such as: accuracy rates, are bots better on certain picks? There is also a demo workflow in that notebook to simplify this raw data output. 

### Results Data and Performance Data
These are both deprecated folders that were used to generate accuracy scores at a grouped level in the past and then look at the more granular view. If you are using the March_weights testing notebook, you can pull these folders in from the Drive and load up that notebook. However, none of the current workflows will write to this directory/in that table schema. 



## Repository Install Notes

### These packages work on Python 3.8.10 or higher and you can install the packages using the following:

pip install -r requirements.txt

### Due to size constraints with GitHub, only a subset of the data is saved directly with this repository. 
To access the draft dump, complete weights, etc. please check out this Google Drive folder: https://drive.google.com/drive/folders/1X1DBADANd_ADcV1aHnYlpdkXgTf28pD_?usp=sharing (Note you'll need 5-10 gigabytes of storage for everything hosted here; place all files in the Google Drive at the top level of this repo to use the files)

### Note that Selenium may require slightly more work (especially the install of Chromedriver)
check out this link for more detail on complete install directions: https://selenium-python.readthedocs.io/installation.html

### It is recommended that you use these modules in a virtualenv with a Python installation that is not downloaded via Anaconda. 
This is because the Bezier module used to create smoother contours for one of the bot's bias terms struggles to work with Anaconda 
