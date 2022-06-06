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
This holds the demos of each experiment workflow (e.g. human picks v. equilibrium/all bots), relevant past research on using different weights and rough benchmarks for accuracy on human picks, old notebooks for the etl workflow (there are some minor changes to get things clean, but it is broken down in a step by step manner), and other notebooks containing relevant visualizations/functions relevant to our paper.

### json_files
This holds the raw JSON file (e.g. cards and all associated metadata) from MTG_JSON for Crimson Vow, the cleaned and validated JSON file match the set of cards from 17Lands/in the weights files (this is called the VOW_writeback.json file), and weights files that are preprocessed for use in equilibrium experiment (e.g. nested dictionary with card names as key and the value being a dictionary with a value for each archetype; the current one is called "weights_writeback"). For each set experimented upon, you would need one raw json file, one json_set_writeback file and however many weights writebacks as you'd like to test (the functions now automatically name these, so it should be clear enough which file is which). 

### draft_txt_files
This holds txt_files of sample drafts with the following breakdown: 1 mythic/rare, 3 uncommons, 10 commons (you can tweak these in future pack generations). By using the pack_generator functions in the experimentation_utilities module, you can generate however many packs as you'd like and run experiments on different sets. The txt files are all arrays that are readily able to be read directly by the equilibrium experiment infrastructure. 

### equilibrium_data
This folder holds 2 kinds of CSV files: baseline and diff files. The baseline data contains results of the experiments where there is the same strategy in all 8 seats of the draft; diff contains the results where 1 seat deviates to a given strategy. These datasets can tell us the 'benefit' associated with choosing a certain strategy given that of others and more generally, can show us how preferences change at a table under different circumstances. The schema for each of the tables is as follows:

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
This folder holds highly raw files from experiments involving human pick data; it is recommended that you use the open_index_file_and_preprocess() function to parse the data into more usable data. These files are good to understand (and tune) performance for the agents in terms of matching human picks (both top 1 and top 3 accuracy) in addition to assessing accuracy under different input weight criteria. The (processed) schema for these tables looks as follows:

# Repository Install Notes
### These packages work on Python 3.8.10 or higher and you can install the packages using the following:

pip install -r requirements.txt

### Due to size constraints with GitHub, only a subset of the data is saved directly with this repository. 
To access the draft dump, complete weights, etc. please reach out to ttew@bu.edu for the complete data (note that you probably want 5-10 gigabytes handy if you really want every file). 

### Note that Selenium may require slightly more work (especially the install of Chromedriver)
check out this link for more detail on complete install directions: https://selenium-python.readthedocs.io/installation.html

### It is recommended that you use these modules in a virtualenv with a Python installation that is not downloaded via Anaconda. 
This is because the Bezier module used to create smoother contours for one of the bot's bias terms struggles to work with Anaconda 
