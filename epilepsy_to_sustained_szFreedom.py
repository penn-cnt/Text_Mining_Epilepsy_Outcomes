import pickle
import numpy as np
import json
import pandas as pd
from datetime import datetime, timedelta, date
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
    
#load the raw data and find the new patient visits
epileptologist_notes = pd.read_pickle('<PATH_TO_NOTES>')
new_patient_notes = epileptologist_notes.loc[epileptologist_notes['VISIT_TYPE'] == 'NEW PATIENT VISIT']
new_patient_note_ids = list(new_patient_notes['NOTE_ID'])
    
#aggregate patients and visit predictions
all_agg_pats = timeline_utils.aggregate_patients_and_visits(datapull['all_pats'])

#calculate the average time spent seizure free or having seizures
final_outcomes = {0:0, 1:0, 2:0} #how many starting hasSz end szFree, hasSz, or IDK, accounting for 6 months at the end?
for pat in all_agg_pats:
    #you need at least 3 visits for this
    if len(pat.aggregate_visits) < 3:
        continue
    
    #sort their visits by visit date, descending as we need to do this retrospectively
    visits_sorted = sorted(pat.aggregate_visits, reverse=True, key=lambda x: x.visit_date)
    
    #we only want patients who start with a new patient visit
    if int(visits_sorted[-1].note_id) not in new_patient_note_ids:
        continue
    
    #get their initial visit classification and skip if they didn't initiate hasSz
    if visits_sorted[-1].hasSz != 1:
        continue
        
    #there needs to be at least 182.5 days (6 mo) between the first and last visits
    if visits_sorted[0].visit_date - visits_sorted[-1].visit_date < timedelta(days =182.5):
        continue
    
    #iterate through their visits until there's at least 6 months of time - 182.5 days. 
    interval_time = timedelta(days=0)
    previous_visit_date = None
    visit_classifications = []    
    for vis in visits_sorted:
        if previous_visit_date == None:
            previous_visit_date = vis.visit_date
            visit_classifications.append(vis.hasSz)
            continue
            
        #update the interval time
        interval_time += previous_visit_date - vis.visit_date
        
        #if at least 6 mo passed, stop.
        if interval_time >= timedelta(days=182.5):
            if (1 in visit_classifications):
                final_outcomes[1] += 1
            elif (2 in visit_classifications):
                final_outcomes[2] += 1
            else:
                final_outcomes[0] += 1
            break
        else:
            previous_visit_date = vis.visit_date
            visit_classifications.append(vis.hasSz)
            
print(final_outcomes)