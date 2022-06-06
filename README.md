# drafting-research
Computational experimentation repository for Magic the Gathering drafts both against human picks experiments and against other bots. The intent of this repository is to hold the tools to collect/clean data for weights and cards to run the simulations, flexible infrastructure for both experiment types, and helper functions/rich data to run analyses on to answer questions around strategy selection in drafts. 

There are 3 main modules in this repository:

1) mtg_etl_pipeline.py, which holds classes/methods surrounding creating compatible weights files (in csv/json format) and card writebacks (json) to be used as inputs. 
2) mtg_experimentation_utilities.py, which holds functions, classes, and methods for running both kinds of experiments listed above, and analyzing output datasets. 
3) Agents.py, which holds the classes and decision functions that constitute the strategies we want to test; all agents are classes of a similar structure and the primary distinction between bots is how their decision functions play out. Additionally, there are hyperparameters that can be adjusted to create more or less extreme versions of the class' strategy, which leaves ample room for further experimentation. 

These packages work on Python 3.8.10 or higher and you can install the packages using the following:

pip install -r requirements.txt

## Note that Selenium may require slightly more work (especially the install of Chromedriver), so check out this link for more detail:
https://selenium-python.readthedocs.io/installation.html

## Note that the Bezier module used to create smoother contours for one of the bot's bias terms struggles to work with Anaconda; thus it is recommended that you use these modules in a virtualenv with a Python installation that is not downloaded via Anaconda. 
