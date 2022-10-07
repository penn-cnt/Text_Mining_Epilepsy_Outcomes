import pickle
import numpy as np
import pandas as pd
import string
import timeline_utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.preprocessing import normalize
sns.set_theme(style='ticks')

#get the quantified data
with open(r'all_patients_and_visits_outcome_measures_py38.pkl', 'rb') as f:
    datapull = pickle.load(f)
    
#get aggregate patients
all_agg_pats = timeline_utils.aggregate_patients_and_visits(datapull['all_pats'])
all_pat_ids = [agg_pat.pat_id for agg_pat in all_agg_pats]

#Kaplan Meier Curve: Probability of seizure freedom given the previous 6 months
six_mo = timeline_utils.generate_markovian_pat_history_survival_curve(all_agg_pats, history_months=6, survival_cutoff_years=1, plot_xlim_years=10, save_path=f"Fig_4")

#Calculate the number of visits used for the kaplan meier curve
hasSz_prev = np.sum([len(classifications) for classifications in six_mo[3]])
hasSz_future = np.sum([len(classifications) for classifications in six_mo[5]])
szFree_prev = np.sum([len(classifications) for classifications in six_mo[4]])
szFree_future = np.sum([len(classifications) for classifications in six_mo[6]])
print(f"Total number of visits used in 6 month history survival curve: {hasSz_prev + hasSz_future + szFree_prev + szFree_future}")

#Kaplan Meier 50% probability locations
print(f"50% SzFree Interval Chance for Non-SzFree Patients after: {six_mo[1].iloc[np.argmin(np.abs(six_mo[1]['szFree']-0.5))].name} years")
print(f"50% Breakthrough Chance for SzFree Patients after: {six_mo[2].iloc[np.argmin(np.abs(six_mo[2]['szFree']-0.5))].name} years")

#Calculate the probability of seizure freedom given the previous three visits
three_vis = timeline_utils.generate_visit_markovian_table(all_agg_pats, save_path='Fig_3')