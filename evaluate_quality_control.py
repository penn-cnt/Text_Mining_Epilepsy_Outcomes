import pickle
import numpy as np
import seaborn as sns
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta, date
from functools import total_ordering
import json
from timeline_utils import Visit, sort_by_visit_date, Patient, generate_aggregate_visit_table, Aggregate_Visit

#Get predictions
with open(r'all_patients_and_visits_outcome_measures_py38', 'rb') as f:
    datapull = pickle.load(f)

#minimum date allowed.
date_cutoff = datetime.strptime('01-01-2005', '%m-%d-%Y')

#we're considering all patients
min_num_visits = 1

#create the aggregate visit table
plurality_voting_tbl = generate_aggregate_visit_table(datapull['all_pats'], min_num_visits=min_num_visits)

#Find all visits where PQF != 0 but HasSz = 0
contradictory_agg_visits_pqf = []
contradictory_pqfs = []
for MRN in plurality_voting_tbl.index:
    for vis in plurality_voting_tbl.columns:
        #make sure it's an aggregate visit object
        if not isinstance(plurality_voting_tbl.loc[MRN,vis], Aggregate_Visit):
            continue
        try:
            #try to convert the pqf into a float. Then check if it is greater than 0. 
            pqf = float(plurality_voting_tbl.loc[MRN, vis].pqf)
            if (pqf != 0) and (plurality_voting_tbl.loc[MRN,vis].hasSz == 0) and not pd.isnull(pqf):
                contradictory_agg_visits_pqf.append(plurality_voting_tbl.loc[MRN,vis])
                contradictory_pqfs.append(pqf)
        except:
            #if the pqf can't be a float, then non-zero and non-nan. In this case, if hasSz = 0, there must be a contradiction
            if plurality_voting_tbl.loc[MRN,vis].hasSz == 0:
                contradictory_agg_visits_pqf.append(plurality_voting_tbl.loc[MRN,vis])
                contradictory_pqfs.append(plurality_voting_tbl.loc[MRN,vis].pqf)                
print(f"Number of visits where PQF != 0 but the patient is seizure free: {len(contradictory_agg_visits_pqf)}")

#Find all visits where ELO - visit_date < 1 year, and ELO was not before a previous visit, and HasSz = 0
contradictory_agg_visits_elo = []
contradictory_elos = []
for MRN in plurality_voting_tbl.index:
    previous_visit = None
    for vis in plurality_voting_tbl.columns:
        #make usre it's an aggregate visit object
        if not isinstance(plurality_voting_tbl.loc[MRN,vis], Aggregate_Visit):
            continue
        if previous_visit:
            if not pd.isnull(plurality_voting_tbl.loc[MRN,vis].elo) and plurality_voting_tbl.loc[MRN,vis].elo:

                #check for a contradiction
                #hasSz = 0 and #ELO was within 1 year of the visit date and #ELO was not before the previous visit
                if (plurality_voting_tbl.loc[MRN,vis].hasSz == 0) and \
                (plurality_voting_tbl.loc[MRN,vis].visit_date - plurality_voting_tbl.loc[MRN,vis].elo < timedelta(days=365)) and \
                (plurality_voting_tbl.loc[MRN,vis].elo - previous_visit.visit_date >= timedelta(days=0)): 
                    contradictory_agg_visits_elo.append({'current_visit': plurality_voting_tbl.loc[MRN,vis], 'previous_visit': previous_visit})
                    contradictory_elos.append(plurality_voting_tbl.loc[MRN,vis].elo)

                previous_visit = plurality_voting_tbl.loc[MRN,vis]
        else:
            previous_visit = plurality_voting_tbl.loc[MRN,vis]
print(f"Number of visits where ELO contradicts seizure freedom: {len(contradictory_agg_visits_elo)}")
                
#How many notes are both ELO and PQF contradictory?
double_contradiction = []
for elo_contradiction in contradictory_agg_visits_elo:
    if elo_contradiction['current_visit'] in contradictory_agg_visits_pqf:
        double_contradiction.append(elo_contradiction['current_visit'])
print(f"Number of visits with both PQF and ELO contradicting the seizure freedom diagnosis:" {len(double_contradiction)}")