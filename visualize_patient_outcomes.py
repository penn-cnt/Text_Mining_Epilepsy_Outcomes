import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from difflib import SequenceMatcher
from datetime import datetime, timedelta, date
from functools import total_ordering
import json
from scipy.stats import mannwhitneyu, iqr
from timeline_utils import Visit, sort_by_visit_date, Patient, Aggregate_Visit, generate_aggregate_visit_table, aggregate_patients_and_visits
# sns.set_theme(style='ticks')

#figure save path
fig_save_dir = r'<PATH_TO_SAVE_DIR>'

#min number of visits a patient needs to be considered in the analysis
#change this to 5 to generate results for patients with at least 5 visits.
min_num_visits = 1

#get the quantified data (model predictions) for each seed 
#formatted as a dictionary, with each seed as a key, and the data as the values
with open(r'all_patients_and_visits_outcome_measures_py38.pkl', 'rb') as f:
    datapull = pickle.load(f)
    
#load the raw text data
epileptologist_notes = pd.read_pickle('<PATH_TO_NOTES>')
#remove duplicate notes
epileptologist_notes = epileptologist_notes.drop_duplicates(subset=['NOTE_AUTHOR', 'VISIT_DATE', 'NOTE_ID'], keep='first')

#get basic statistics of what was predicted on
pats_predicted_on = datapull['all_pats'][2]
visits_predicted_on = datapull['all_pat_visits'][2]
visit_ids = [f"{vis.Patient.pat_id}_{vis.note_id}_{vis.author}_{str(vis.visit_date)[:10]}" for vis in visits_predicted_on]
visit_authors = set([vis.author for vis in visits_predicted_on])
print(f"Number of notes predicted on: {len(visits_predicted_on)}")
print(f"Number of patients predicted on: {len(pats_predicted_on)}")
print(f"Number of note authors: {len(visit_authors)}")

#find which of the original notes were used
generated_ids = epileptologist_notes.apply(lambda x: f"{x['MRN']}_{x['NOTE_ID']}_{x['NOTE_AUTHOR']}_{str(x['VISIT_DATE'])[:10]}", axis=1)
included_notes = epileptologist_notes.loc[generated_ids[generated_ids.isin(visit_ids)].index]
#get only new patient notes
included_notes['VISIT_TYPE'] = included_notes['VISIT_TYPE'].fillna("NULL")
new_visits = included_notes.loc[(included_notes['VISIT_TYPE'].str.contains('NEW PATIENT')) & (~included_notes['VISIT_TYPE'].str.contains('GI'))]
print(f"Number of new patient notes predicted on: {len(new_visits)} ({len(new_visits)/len(visits_predicted_on)})")

#get the number of patients and visits statistics
total_num_pats = [len(datapull['all_pats'][seed]) for seed in datapull['all_pats']]
total_num_visits = [len(datapull['all_pat_visits'][seed]) for seed in datapull['all_pat_visits']]
num_visits_per_pat = {}
overall_min_visit_dates = []
overall_max_visit_dates = []
for seed in datapull['all_pats']:
    num_visits_per_pat[seed] = []
    for pat in datapull['all_pats'][seed]:
        num_visits_per_pat[seed].append(len(pat.visits))
        overall_min_visit_dates.append(np.min([vis.visit_date for vis in pat.visits]))
        overall_max_visit_dates.append(np.max([vis.visit_date for vis in pat.visits]))
print(f"Total number of patients: {total_num_pats}")
print(f"Total number of visits across all patients: {total_num_visits}")
print(f"Total earliest visits across all patients: {np.min(overall_min_visit_dates)}")
print(f"Total latest of visits across all patients: {np.max(overall_max_visit_dates)}")

#get info for at least 5 visits specifically
print(f"Number of patients with at least {5} visits: {np.sum(np.array(num_visits_per_pat[2]) >= 5)}")
five_visits_or_more = [num_vis if num_vis >= 5 else 0 for num_vis in num_visits_per_pat[2]]
print(f"Number of visits from patients with at least 5 visits: {np.sum(five_visits_or_more)}")
np.sum(np.array(num_visits_per_pat[2]) >= 5) / len(num_visits_per_pat[2])

#get info for over 1 visit specifically
print(f"Number of patients with at least {1} visit: {np.sum(np.array(num_visits_per_pat[2]) > 1)}")
one_visits_or_more = [num_vis if num_vis > 1 else 0 for num_vis in num_visits_per_pat[2]]
print(f"Number of visits from patients with at least 1 visit: {np.sum(one_visits_or_more)}")
np.sum(np.array(num_visits_per_pat[2]) > 1) / len(num_visits_per_pat[2])

#number of visits distribution
print(f"Mean number of visits per patient: {np.mean(num_visits_per_pat[2])}")
print(f"Median number of visits per patient: {np.median(num_visits_per_pat[2])}")

#generate aggregate visits
plurality_voting_tbl = generate_aggregate_visit_table(datapull['all_pats'], min_num_visits=min_num_visits)

#collect descriptive/comprehensive statistics 
all_frac_hasSz = []
all_frac_idk = []
all_hasSz = []
all_szFreq = []
all_szFreqs_float = []
total_num_vis = 0
total_num_notes_classification = {0:0, 1:0, 2:0}
total_num_notes_with_ELO = 0

#keep track of how recent their last seizure was
min_time_since_last_sz = timedelta(days=99999)
min_elo_vis = None
max_time_since_last_sz = timedelta(days=-99999)
max_elo_vis = None
all_elo_times = []

#for each patient, and each visit of the patient
for pat in plurality_voting_tbl.index:
    frac_hasSz = 0
    frac_idk = 0
    num_vis = 0
    
    for vis in plurality_voting_tbl.columns:            
        #skip if it is not a visit
        if not isinstance(plurality_voting_tbl.loc[pat,vis], Aggregate_Visit):
            continue
        
        #scale all visits by the patient's first visit
        plurality_voting_tbl.loc[pat, vis].time_to_first_visit = (plurality_voting_tbl.loc[pat, vis].visit_date - plurality_voting_tbl.loc[pat, 0].visit_date).total_seconds()/(3600*24)
        
        #accumulate their seizure freedom, frequency and and date of last seizure status for this visit
        all_hasSz.append(plurality_voting_tbl.loc[pat, vis].hasSz)
        all_szFreq.append(plurality_voting_tbl.loc[pat, vis].pqf)
        frac_hasSz += plurality_voting_tbl.loc[pat, vis].hasSz if plurality_voting_tbl.loc[pat, vis].hasSz != 2 else 0 
        frac_idk += int(plurality_voting_tbl.loc[pat, vis].hasSz == 2)
        total_num_notes_classification[plurality_voting_tbl.loc[pat, vis].hasSz] += 1
        total_num_notes_with_ELO += 1 if not pd.isnull(plurality_voting_tbl.loc[pat, vis].elo) else 0
        #attempt to get a numerical seizure frequency
        try:
            pqf = float(plurality_voting_tbl.loc[pat, vis].pqf)
            if (not pd.isnull(pqf)) and (pqf != -2): #-2 is an error code in seizure frequencies
                all_szFreqs_float.append(pqf)
        except:
            pass
        
        #if there is an ELO, check if it is in the future
        #if it isn't then get the number of days before and see how it compares to the max or min
        if not pd.isnull(plurality_voting_tbl.loc[pat, vis].elo):
            if plurality_voting_tbl.loc[pat, vis].elo <= plurality_voting_tbl.loc[pat, vis].visit_date:
                time_since_last_sz = plurality_voting_tbl.loc[pat, vis].visit_date - plurality_voting_tbl.loc[pat, vis].elo
                all_elo_times.append(time_since_last_sz.days/365)
                if min_time_since_last_sz > time_since_last_sz:
                    min_time_since_last_sz = time_since_last_sz
                    min_elo_vis = plurality_voting_tbl.loc[pat, vis]
                elif max_time_since_last_sz < time_since_last_sz:
                    max_time_since_last_sz = time_since_last_sz
                    max_elo_vis = plurality_voting_tbl.loc[pat, vis]
                
            
        #raise if an error code was spotted and is being considered
        if (plurality_voting_tbl.loc[pat, vis].pqf == -299.0645) or (plurality_voting_tbl.loc[pat, vis].elo == -299.0645):
            raise
        
        #add to the counter
        num_vis += 1
    
    #get the fraction of hasSz and append to container
    frac_hasSz /= num_vis
    frac_idk /= num_vis
    all_frac_hasSz.append(frac_hasSz)
    all_frac_idk.append(frac_idk)
    total_num_vis += num_vis
    
#collect hasSz vs SzFreq predictions for all visits
all_hasSz_vs_szFreq = pd.DataFrame({'hasSz':all_hasSz, 'szFreq':all_szFreq})
    
#add a new column containing their fraction of having seizures
all_szFreqs_float = np.array(all_szFreqs_float)
plurality_voting_tbl['frac_hasSz'] = all_frac_hasSz
plurality_voting_tbl['frac_idk'] = all_frac_idk

#print outcomes
print(f"""Total number of patients: {len(plurality_voting_tbl)}\n
          Total number of visits: {total_num_vis}\n
          Total number of visits with seizure freedom Yes: {total_num_notes_classification[0]} ({total_num_notes_classification[0]/total_num_vis})\n
          Total number of visits with seizure freedom No: {total_num_notes_classification[1]} ({total_num_notes_classification[1]/total_num_vis})\n
          Total number of visits with seizure freedom IDK: {total_num_notes_classification[2]} ({total_num_notes_classification[2]/total_num_vis})\n
          
          Total number of patients only seizure free: {len(plurality_voting_tbl.loc[(plurality_voting_tbl['frac_hasSz'] == 0) & (plurality_voting_tbl['frac_idk'] == 0)])} ({len(plurality_voting_tbl.loc[(plurality_voting_tbl['frac_hasSz'] == 0) & (plurality_voting_tbl['frac_idk'] == 0)])/len(plurality_voting_tbl)})\n
          Total number of patients only having seizures: {len(plurality_voting_tbl.loc[(plurality_voting_tbl['frac_hasSz'] == 1) & (plurality_voting_tbl['frac_idk'] == 0)])} ({len(plurality_voting_tbl.loc[(plurality_voting_tbl['frac_hasSz'] == 1) & (plurality_voting_tbl['frac_idk'] == 0)])/len(plurality_voting_tbl)})\n
          Total number of patients with at least 50% having seizures: {len(plurality_voting_tbl.loc[(plurality_voting_tbl['frac_hasSz'] >= 0.5)])} ({len(plurality_voting_tbl[plurality_voting_tbl['frac_hasSz'] >= 0.5])/len(plurality_voting_tbl)})\n
          Total number of patients alternating seizure free and having seizures: {len(plurality_voting_tbl.loc[(plurality_voting_tbl['frac_hasSz'] != 0) & (plurality_voting_tbl['frac_hasSz'] != 1)])} ({len(plurality_voting_tbl.loc[(plurality_voting_tbl['frac_hasSz'] != 0) & (plurality_voting_tbl['frac_hasSz'] != 1)])/len(plurality_voting_tbl)})\n
          
          Total number of patients only IDK: {len(plurality_voting_tbl[plurality_voting_tbl['frac_idk'] == 1])} ({len(plurality_voting_tbl[plurality_voting_tbl['frac_idk'] == 1])/len(plurality_voting_tbl)})\n
          Total number of patients no IDK: {len(plurality_voting_tbl[plurality_voting_tbl['frac_idk'] == 0])} ({len(plurality_voting_tbl[plurality_voting_tbl['frac_idk'] == 0])/len(plurality_voting_tbl)})\n
          Total number of patients at least 50% IDK: {len(plurality_voting_tbl[plurality_voting_tbl['frac_idk'] >= 0.5])} ({len(plurality_voting_tbl[plurality_voting_tbl['frac_idk'] >= 0.5])/len(plurality_voting_tbl)})\n
          Total number of visits with PQF: {len(all_hasSz_vs_szFreq[~pd.isnull(all_hasSz_vs_szFreq['szFreq'])])} ({len(all_hasSz_vs_szFreq[~pd.isnull(all_hasSz_vs_szFreq['szFreq'])])/total_num_vis})\n
          Total number of visits with ELO: {total_num_notes_with_ELO} ({total_num_notes_with_ELO/total_num_vis})\n
          Maximum ELO was before a visit: {max_elo_vis}\n
          Minimum ELO was before a visit: {min_elo_vis}\n
          Median ELO distance before visit: {np.median(all_elo_times)} years, IQR {np.percentile(all_elo_times,q=25)} - {np.percentile(all_elo_times,q=75)} years\n
          
          Total number of visits szFree but with PQF != 0: {len(all_hasSz_vs_szFreq.loc[(all_hasSz_vs_szFreq['hasSz'] == 0) & (all_hasSz_vs_szFreq['szFreq'] != 0) & (~pd.isnull(all_hasSz_vs_szFreq['szFreq']))])} ({len(all_hasSz_vs_szFreq.loc[(all_hasSz_vs_szFreq['hasSz'] == 0) & (all_hasSz_vs_szFreq['szFreq'] != 0) & (~pd.isnull(all_hasSz_vs_szFreq['szFreq']))]) / len(all_hasSz_vs_szFreq[~pd.isnull(all_hasSz_vs_szFreq['szFreq'])])})\n
          Total number of visits hasSz but with PQF != 0: {len(all_hasSz_vs_szFreq.loc[(all_hasSz_vs_szFreq['hasSz'] == 1) & (all_hasSz_vs_szFreq['szFreq'] != 0) & (~pd.isnull(all_hasSz_vs_szFreq['szFreq']))])} ({len(all_hasSz_vs_szFreq.loc[(all_hasSz_vs_szFreq['hasSz'] == 1) & (all_hasSz_vs_szFreq['szFreq'] != 0) & (~pd.isnull(all_hasSz_vs_szFreq['szFreq']))]) / len(all_hasSz_vs_szFreq[~pd.isnull(all_hasSz_vs_szFreq['szFreq'])])})\n
          Total number of visits IDK but with PQF != 0: {len(all_hasSz_vs_szFreq.loc[(all_hasSz_vs_szFreq['hasSz'] == 2) & (all_hasSz_vs_szFreq['szFreq'] != 0) & (~pd.isnull(all_hasSz_vs_szFreq['szFreq']))])} ({len(all_hasSz_vs_szFreq.loc[(all_hasSz_vs_szFreq['hasSz'] == 2) & (all_hasSz_vs_szFreq['szFreq'] != 0) & (~pd.isnull(all_hasSz_vs_szFreq['szFreq']))]) / len(all_hasSz_vs_szFreq[~pd.isnull(all_hasSz_vs_szFreq['szFreq'])])})\n""")

#calculate the average time spent seizure free or having seizures
time_szFree = []
time_hasSz = []
all_agg_pats = aggregate_patients_and_visits(datapull['all_pats'])
for pat in all_agg_pats:
    #We'll consider patients with at least 5 visits in this analysis
    if len(pat.aggregate_visits) < 5:
        continue
        
    #we want people with a variety of classifications
    if (plurality_voting_tbl.loc[pat.pat_id].frac_hasSz == 1) or (plurality_voting_tbl.loc[pat.pat_id].frac_idk == 0 and plurality_voting_tbl.loc[pat.pat_id].frac_hasSz == 0) or (plurality_voting_tbl.loc[pat.pat_id].frac_idk == 1):
        continue
        
    #sort their visits by visit date, descending as we need to do this retrospectively
    visits_sorted = sorted(pat.aggregate_visits, reverse=True, key=lambda x: x.visit_date)
    
    #iterate through visits
    is_szFree = None
    initial_visit_date = None
    for i in range(len(visits_sorted)):
        vis = visits_sorted[i]
        
        #if it's the last visit, 
        if is_szFree == None:
            is_szFree = vis.hasSz
            initial_visit_date = vis.visit_date
            
        else:
            #if it's the patient's first visit, or
            #if the patient changes states, then append the amount of time that has passed
            #then terminate accordingly
            if (i == len(visits_sorted) - 1) or vis.hasSz != is_szFree:
                if is_szFree == 0:
                    time_szFree.append((initial_visit_date - vis.visit_date).days)
                elif is_szFree == 1:
                    time_hasSz.append((initial_visit_date - vis.visit_date).days)
                
                #reset of the new state
                is_szFree = vis.hasSz
                initial_visit_date = vis.visit_date
                
#print results
print(f"Average time spent seizure free: {np.mean(time_szFree)/30.4167} months")
print(f"Median time spent seizure free: {np.median(time_szFree)/30.4167} months")
print(f"Stddev time spent seizure free: {np.std(time_szFree)/30.4167} months")
print(f"IQR time spent seizure free: {np.percentile(time_szFree,q=25)/30.4167} - {np.percentile(time_szFree,q=75)/30.4167} months")

#find the average time between visits
#Yes, this code is a bit redundant with the one a few lines above
d_visit_date = []
d_szFree_visit_date = []
d_hasSz_visit_date = []
for pat in all_agg_pats:
    #We'll consider patients with at least 5 visits in this analysis
    if len(pat.aggregate_visits) < 5:
        continue
    
    #sort their visits by visit date
    visits_sorted = sorted(pat.aggregate_visits, reverse=False, key=lambda x: x.visit_date)
    
    #get the time between visits and categorize it
    for i in range(1, len(visits_sorted)):
        d_visit_date.append(visits_sorted[i].visit_date - visits_sorted[i-1].visit_date)
        if visits_sorted[i-1].hasSz == 1:
            d_hasSz_visit_date.append(visits_sorted[i].visit_date - visits_sorted[i-1].visit_date)
        elif visits_sorted[i-1].hasSz == 0:
            d_szFree_visit_date.append(visits_sorted[i].visit_date - visits_sorted[i-1].visit_date)

#get days values and print results
d_visit_date = [dif.days for dif in d_visit_date]
d_szFree_visit_date = [dif.days for dif in d_szFree_visit_date]
d_hasSz_visit_date = [dif.days for dif in d_hasSz_visit_date]
print(f"The median time between visits was {np.median(d_visit_date)/30.4167} (IQR {np.percentile(d_visit_date,q=25)/30.4167} - {np.percentile(d_visit_date,q=75)/30.4167}) months")
print(f"The median time between visits after seizure free was {np.median(d_szFree_visit_date)/30.4167} (IQR {np.percentile(d_szFree_visit_date,q=25)/30.4167} - {np.percentile(d_szFree_visit_date,q=75)/30.4167}) months")
print(f"The median time between visits after not seizure free was {np.median(d_hasSz_visit_date)/30.4167} (IQR {np.percentile(d_hasSz_visit_date,q=25)/30.4167} - {np.percentile(d_hasSz_visit_date,q=75)/30.4167}) months")

#check how the distributions between the upper and lower quintiles of frac_hasSz are different
upper_quint = 0.8
lower_quint = 0.2
upper_quint_total_num_vis = []
lower_quint_total_num_vis = []
upper_quint_total_time = []
lower_quint_total_time = []
upper_quint_visit_freq = []
lower_quint_visit_freq = []
#iterate through patients
for idx, row in plurality_voting_tbl.iterrows():
    #check if they have at least 2 visits
    if pd.isnull(row[1]):
        continue
    
    #get only the visits
    visits = row.dropna().drop(['frac_hasSz', 'frac_idk'])
    total_time = visits[np.max(visits.index)].visit_date - visits[0].visit_date 
    
    #if there isn't any time between first and last visit (at least 1 week), then skip
    if total_time.days < 7:
        continue
        
    #if there are less than 5 visits (len 7, for the two frac_X columns), skip
    if len(row.dropna()) < 7:
        continue
        
    #if they have many hasSz or few hasSz, add their metrics to the arrays
    if row['frac_hasSz'] > upper_quint:
        upper_quint_total_num_vis.append(len(visits))
        upper_quint_total_time.append(total_time.days)
        upper_quint_visit_freq.append(upper_quint_total_num_vis[-1] / (upper_quint_total_time[-1] / 365)) #visits per year
    if row['frac_hasSz'] <= lower_quint:
        lower_quint_total_num_vis.append(len(visits))
        lower_quint_total_time.append(total_time.days)
        lower_quint_visit_freq.append(lower_quint_total_num_vis[-1] / (lower_quint_total_time[-1] / 365)) #visits per year
        
#perform mann whitney u tests between the quintiles on each one.
print("\t\tUpper Quintile\t|\tLower Quintile")

#for total number of visits
u1, p_val_1 = mannwhitneyu(upper_quint_total_num_vis, lower_quint_total_num_vis)
u2, p_val_2 = mannwhitneyu(lower_quint_total_num_vis, upper_quint_total_num_vis)
print(f"Med Num Vis:\t{np.round(np.median(upper_quint_total_num_vis), decimals=8)}\t\t|\t{np.round(np.median(lower_quint_total_num_vis), decimals=8)}")
print(f"n1: {len(upper_quint_total_num_vis)}")
print(f"n2: {len(lower_quint_total_num_vis)}")
print(f"u1: {u1}")
print(f"u2: {u2}")
print(f"p-val: {p_val_1}")
print("\n")

#for the total time spent
u1, p_val_1 = mannwhitneyu(upper_quint_total_time, lower_quint_total_time)
u2, p_val_2 = mannwhitneyu(lower_quint_total_time, upper_quint_total_time)
print(f"Med Time (d):\t{np.round(np.median(upper_quint_total_time), decimals=8)}\t\t|\t{np.round(np.median(lower_quint_total_time), decimals=8)}")
print(f"n1: {len(upper_quint_total_time)}")
print(f"n2: {len(lower_quint_total_time)}")
print(f"u1: {u1}")
print(f"u2: {u2}")
print(f"p-val: {p_val_1}")
print("\n")

#for the visit frequency
u1, p_val_1 = mannwhitneyu(upper_quint_visit_freq, lower_quint_visit_freq)
u2, p_val_2 = mannwhitneyu(lower_quint_visit_freq, upper_quint_visit_freq)
print(f"Med Freq (/yr):\t{np.round(np.median(upper_quint_visit_freq), decimals=8)}\t|\t{np.round(np.median(lower_quint_visit_freq), decimals=8)}")
print(f"n1: {len(upper_quint_visit_freq)}")
print(f"n2: {len(lower_quint_visit_freq)}")
print(f"u1: {u1}")
print(f"u2: {u2}")
print(f"p-val: {p_val_1}")
print("\n")

#----------------------------- Function for Plotting Seizure Frequency Outcomes Across Time -----------------------------#
def plot_visit_dates_and_hasSz(min_frac_hasSz = 0.0, max_frac_hasSz = 1.0, 
                               title="All Patients", use_delta=False, 
                               split_by_proportions=None, split_amount=50, ylabel='Patient Index',
                               xlabel='Date', no_y_ticks=False, save_path='Fig_2'):
    colors = ['#0066cc', '#ffc20a', '#7570b3']
    fig = plt.figure(figsize=(12, 12))
    num_pats = 0
    num_visits = 0

    if split_by_proportions:
        proportions = np.arange(split_by_proportions, 1, split_by_proportions)
    
    #iterate through all patients
    for idx, row in plurality_voting_tbl.iterrows():
        row = row.loc[(row.index !='pat_id') & (~row.isna())]
        
        #ignore this patient if they do not have the right proportion of hasSz
        if (row['frac_hasSz'] < min_frac_hasSz) | (row['frac_hasSz'] > max_frac_hasSz):
            continue

        num_pats += 1
        num_visits += len(row) - 2 #subtracting 2 because one element will be the frac_hasSz and frac_idk
            
        #plot the points
        for vis in row:
            if not isinstance(vis, Aggregate_Visit):
                continue
            #skip unclassified visits
            if vis.hasSz == 2:
                continue
            if use_delta:
                if split_by_proportions:
                    plt.plot(vis.time_to_first_visit/365, num_pats + split_amount*np.sum(proportions<=row['frac_hasSz']), 'o', color=colors[vis.hasSz], markersize=3)
                else:
                    plt.plot(vis.time_to_first_visit/365, num_pats, 'o', color=colors[vis.hasSz], markersize=3)
            else:
                if split_by_proportions:
                    plt.plot(vis.visit_date, num_pats + split_amount*np.sum(proportions<=row['frac_hasSz']), 'o', color=colors[vis.hasSz], markersize=3)
                else:
                    plt.plot(vis.visit_date, num_pats, 'o', color=colors[vis.hasSz], markersize=3)
                    
    #plot legend
    legend_elements = [Line2D([0], [0], color=colors[0], marker='.', lw=0, label='Seizure Free'),
                       Line2D([0], [0], color=colors[1], marker='.', lw=0, label='Has Seizures')]
    plt.legend(handles=legend_elements, prop={'size': 14})
    plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    if no_y_ticks:
        fig.axes[0].set_yticks([])
    sns.despine()
    if save_path != None:
        plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", dpi=600, bbox_inches='tight')
    else:
        plt.show()
    print(f"Number of patients with this fraction range: {num_pats}. Total number of visits: {num_visits}")
#----------------------------- End Function for Plotting Seizure Frequency Outcomes Across Time -----------------------------#

#sort the visits by fraction of having seizures
plurality_voting_tbl = plurality_voting_tbl.sort_values(by='frac_hasSz').reset_index(drop=True)

#plot the seizure frequencies for all patients across time
#change min_num_visits above to create the plot of patients with at least 5 visits, as seen in Figure 2
plot_visit_dates_and_hasSz(min_frac_hasSz=0, max_frac_hasSz=1.0, 
                           title="Patient Seizure Freedom Aligned\nat Their First Recorded Visits", 
                           use_delta=True, xlabel='Years After First Visit', ylabel="Patients",
                           split_by_proportions=0.2, split_amount=200,no_y_ticks=True,
                           save_path=f"{fig_save_dir}/Fig_2")