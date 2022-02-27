##Sample Agent A
##In theory, we will iterate on certain lines of code in different agent files like this.
#E.g. there will be 10-15 Agent.py files with different rules we should be able to plug them into the sim

#Agent A uses the original pick making formula

#Import dependencies
import json
import random
import uuid
import sqlite3
import numpy as np
import pandas as pd
from draftbot_sim import *

class Basic_Agent:
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
        this output array.

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
      self.options[:, :, n_pick + self.n_cards_in_pack * self.round] = packs.copy()
      
      #1's or 0's if card is in pack 
      card_is_in_pack = np.sign(packs)
      
      
      pack_archetype_weights = (
          card_is_in_pack.reshape((self.n_drafters, self.set.n_cards, 1)) *
          self.archetype_weights.reshape((1, self.set.n_cards, self.n_archetypes)))
      
      preferences = np.einsum(
          'dca,da->dc', pack_archetype_weights, self.drafter_preferences)
    
      pick_probs = non_zero_softmax(preferences)
      
      #This needs to depend on the agent 
      picks = self.make_picks(pick_probs)
      
      # We should not be able to pick a card that does not exist.
      assert np.all(packs >= picks)
      
      packs = rotate_array(packs - picks, forward=True)
      
      self.drafter_preferences = (
          self.drafter_preferences +
          np.einsum('ca,dc->da', self.archetype_weights, picks))
      
      self.picks[:, :, n_pick + self.n_cards_in_pack * self.round] = picks.copy()

      
      self.preferences[:, :, n_pick + self.n_cards_in_pack * self.round] = (
          self.drafter_preferences.copy())

      self.passes[:, :, n_pick + self.n_cards_in_pack * self.round] = self.picks[:, :, n_pick + self.n_cards_in_pack * self.round] - self.options[:, :, n_pick + self.n_cards_in_pack * self.round]

      return packs

class Agent_Signal:
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
        this output array.

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
      self.options[:, :, n_pick + self.n_cards_in_pack * self.round] = packs.copy()
      
      #1's or 0's if card is in pack 
      card_is_in_pack = np.sign(packs)
      
      
      pack_archetype_weights = (
          card_is_in_pack.reshape((self.n_drafters, self.set.n_cards, 1)) *
          self.archetype_weights.reshape((1, self.set.n_cards, self.n_archetypes)))
      
      preferences = np.einsum(
          'dca,da->dc', pack_archetype_weights, self.drafter_preferences)
    
      pick_probs = non_zero_softmax(preferences)
      
      #This needs to depend on the agent 
      picks = self.make_picks(pick_probs)
      
      # We should not be able to pick a card that does not exist.
      assert np.all(packs >= picks)
      
      packs = rotate_array(packs - picks, forward=True)
      
      self.drafter_preferences = (
          self.drafter_preferences +
          np.einsum('ca,dc->da', self.archetype_weights, picks))
      
      self.picks[:, :, n_pick + self.n_cards_in_pack * self.round] = picks.copy()

      
      self.preferences[:, :, n_pick + self.n_cards_in_pack * self.round] = (
          self.drafter_preferences.copy())

      self.passes[:, :, n_pick + self.n_cards_in_pack * self.round] = self.picks[:, :, n_pick + self.n_cards_in_pack * self.round] - self.options[:, :, n_pick + self.n_cards_in_pack * self.round]

      return packs