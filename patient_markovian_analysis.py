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

#Calculate the probability of seizure freedom given the previous three visits
three_vis = timeline_utils.generate_visit_markovian_table(all_agg_pats, save_path='Fig_3')