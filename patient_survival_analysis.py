import pickle
import numpy as np
import json
import pandas as pd
from datetime import datetime, timedelta, date
import string
import timeline_utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency, mannwhitneyu, pearsonr, spearmanr
import seaborn as sns
sns.set_theme(style='ticks')
from lifelines import KaplanMeierFitter

#figure save path
fig_save_dir = r''

#get the quantified data
with open(r'all_patients_and_visits_outcome_measures_py38.pkl', 'rb') as f:
    datapull = pickle.load(f)
    
#get aggregate patients
all_agg_pats = timeline_utils.aggregate_patients_and_visits(datapull['all_pats'])
all_pat_ids = [agg_pat.pat_id for agg_pat in all_agg_pats]

#create an array where rows are patients and columns days on which they remain default until their event
def generate_survival_table(all_final_to_breakpoint_time_diff, all_future_visit_dates, all_future_visit_classifications, all_breakpoint_visit_dates, agg_pats, all_pre_baseline_visits, min_followup_days = None,  max_followup_days = None, default_value=0):
    
    #find which patients meet the inclusion criteria
    pat_indices_used = [i for i in range(len(all_future_visit_dates)) if (all_future_visit_dates[i][-1] - all_breakpoint_visit_dates[i] >= min_followup_days)]
    
    #rows = patients, columns = days. We need to +2, because the range must be inclusive on both size, [0, max_days], and we include one more column for all np.nan after max_days.
    #we add the np.nan column so that patients who do not fall off the curve and are not censored are truncated at max_days
    survival_table = np.ones((len(pat_indices_used), max_followup_days.days+2)) * default_value
    survival_table[:, -1] = np.nan

    #for each patient, add their influence to the survival table
    pat_ct = 0
    pats_in_table = []
    pre_baselines_in_table = []
    for i in pat_indices_used:

        #for each visit of each patient, accumulate
        for j in range(len(all_future_visit_dates[i])):
            
            #calculate how much time has passed since the breakpoint
            days_after = (all_future_visit_dates[i][j] - all_breakpoint_visit_dates[i]).days
            
            #if days_after > max_followup_days, then skip
            if days_after > max_followup_days.days:
                continue
            
            #add in the patient's visit classification. We assume everyone is szFree until their visit.
            survival_table[pat_ct, days_after] = all_future_visit_classifications[i][j]
            
            #if the visit was hasSz, then the patient leaves the survival curve - we stop processing this patient
            if all_future_visit_classifications[i][j] != default_value:
                break
                
        #if the patient is out of visits, or they were hasSz, then set all future values to NaN
        survival_table[pat_ct, days_after+1:] = np.nan
        pats_in_table.append(agg_pats[i])
        pre_baselines_in_table.append(all_pre_baseline_visits[i])
        pat_ct += 1
        
    return survival_table, pats_in_table, pre_baselines_in_table

def generate_kaplan_table(survival_table, pats_in_table, pre_baselines_in_table, default_value):
    """creates a dataframe where rows are patients, and columns are the duration they stay (default_value) and whether the event occurs (!default_value)"""

    #find where in the suvival_table there is a np.nan. 1 day before is the last visit = the duration
    #if there is no nan, then the patient did not fall from having the event, and still followed up.
    last_visit_idx = [np.where(np.isnan(survival_table[i]))[0][0] - 1 for i in range(survival_table.shape[0])]    
    #get the value, either 0 or 1, of the last visit day. Re-format the event such that 1 means they had an event, and 0 means they did not (censored or ended)
    last_visit_val = [int(survival_table[i, last_visit_idx[i]] != default_value) for i in range(len(last_visit_idx))]
    
    #get the patient's MRN
    pat_mrns = [pat.pat_id for pat in pats_in_table]
    
    #get the pre-baseline visit classifications
    baseline_class = [[vis.hasSz for vis in vis_arr] for vis_arr in pre_baselines_in_table]
    
    return pd.DataFrame(data={'duration':last_visit_idx, 'event':last_visit_val, 'MRN':pat_mrns, 'pre_baseline':baseline_class})
    
def get_demographic_counts(df):
    demographic, counts = np.unique(df, return_counts = True)
    return {demographic[i]:counts[i] for i in range(len(demographic)) if not pd.isnull(demographic[i])}

def initialize_missing_keys(df1, df2):
    keys1 = set(df1.keys())
    keys2 = set(df2.keys())
    all_keys = keys1.union(keys2)
    
    for key in all_keys:
        if key not in df1: 
            df1[key] = 0
        if key not in df2:
            df2[key] = 0
            
def normalize_dict(d):
    total_pats = np.sum([d[k] for k in d])
    normalized = {k:d[k]/total_pats for k in d}
    return normalized
    
history_months = 6 #how many months of history do they need
min_followup = timedelta(days=365) #how much minimum followup do they need in days
max_followup = timedelta(days=10*365) #how long are we following up for, in days

#containers to store necessary info
all_breakpoint_visit_dates = []
all_future_visit_dates = []
all_future_visit_classifications = []
all_final_to_breakpoint_time_diff = []
all_six_mo_summaries = []
all_classifications_across_prev_visits = []
all_pre_baseline_visits = []
used_pats = []

#for each patient, count the number of times they were seizure free or having seizures in a roughly history_months month interval
for pat in all_agg_pats:

    #a patient needs to have at least 3 visits to be considered
    if len(pat.aggregate_visits) < 3:
        continue

    #sort the visits by visit date
    aggregate_visits_sorted = sorted(pat.aggregate_visits, key=lambda x:x.visit_date)

    #iterate through their visits - if there is a roughly history_months month gap between their current and previous visit, then it's a useable datapoint
    #or, if there is a roughly history_months month gap across multiple visits, that is also a useable datapoint
    #we will accept +/- half a month
    i = 0
    while i  < len(aggregate_visits_sorted) - 1:

        #internal loop counter
        j=i

        #accumulate time between visits for this window
        interval_time = timedelta(days=0)
        #accumulate the classification between visits
        classifications_across_prev_visits = []

        #while we're still under the history_months month threshold, and we haven't exceeded our visit array
        while interval_time < timedelta(days=history_months*30.4167 - 30.4167/2) and j < len(aggregate_visits_sorted) - 1:

            #accumulate the amount of time between visits
            interval_time += aggregate_visits_sorted[j+1].visit_date - aggregate_visits_sorted[j].visit_date

            #get the next visit classification
            classifications_across_prev_visits.append(aggregate_visits_sorted[j+1].hasSz)

            #go into the future
            j += 1

        #if any of the visits were unclassified,
        #skip
        if (2 in classifications_across_prev_visits):
            i+=1
            continue
        else:
            #skip patients who have no more data
            if j >= len(aggregate_visits_sorted) - 1:
                break

            #get the reminaing visits if they exist
            future_visit_classifications = [vis.hasSz for vis in aggregate_visits_sorted[j+1:]]   

            #skip patients who have idk classifications in their future
            if 2 in future_visit_classifications:
                i+=1
                continue

            all_pre_baseline_visits.append(aggregate_visits_sorted[:i+1])
            all_future_visit_classifications.append(future_visit_classifications)
            all_future_visit_dates.append([vis.visit_date for vis in aggregate_visits_sorted[j+1:]])

            #what is the time at the breakpoint from history_months months in the past to the future?
            all_breakpoint_visit_dates.append(aggregate_visits_sorted[j].visit_date)
            final_to_breakpoint_time_diff = aggregate_visits_sorted[-1].visit_date - aggregate_visits_sorted[j].visit_date
            all_final_to_breakpoint_time_diff.append(final_to_breakpoint_time_diff)

            #get the patient's status from the last six months
            six_mo_summary = int((1 in classifications_across_prev_visits))
            all_classifications_across_prev_visits.append(classifications_across_prev_visits)
            all_six_mo_summaries.append(six_mo_summary)
            
            #store this patient
            used_pats.append(pat)
            break
            
    
#convert to numpy array for indexing
all_final_to_breakpoint_time_diff = np.array(all_final_to_breakpoint_time_diff)
all_future_visit_dates = np.array(all_future_visit_dates, dtype='object')
all_future_visit_classifications = np.array(all_future_visit_classifications, dtype='object')
all_classifications_across_prev_visits = np.array(all_classifications_across_prev_visits, dtype='object')
all_breakpoint_visit_dates = np.array(all_breakpoint_visit_dates)
all_six_mo_summaries = np.array(all_six_mo_summaries)
all_pre_baseline_visits = np.array(all_pre_baseline_visits)
used_pats = np.array(used_pats)


#calculate survival tables. For szFree_start - the default value is 0, and the event is 1
szFree_start = all_six_mo_summaries == 0
szFree_start_survival_table, szFree_start_pats, szFree_pre_baseline = generate_survival_table(all_final_to_breakpoint_time_diff[szFree_start], all_future_visit_dates[szFree_start], all_future_visit_classifications[szFree_start], all_breakpoint_visit_dates[szFree_start], used_pats[szFree_start], all_pre_baseline_visits[szFree_start], min_followup, max_followup, 0)
szFree_start_km_table = generate_kaplan_table(szFree_start_survival_table, szFree_start_pats, szFree_pre_baseline, 0)

#fit kaplan meier and plot 1 - survival functions for szFree start
szFree_start_kmf = KaplanMeierFitter()
szFree_start_kmf.fit(szFree_start_km_table['duration']/365, szFree_start_km_table['event'])
szFree_start_kmf.survival_function_ = 1 - szFree_start_kmf.survival_function_
print(szFree_start_kmf)
plt.figure(figsize=(8,7))
plt.fontsize=16
#plot the curve
kmf_plot = szFree_start_kmf.plot(ci_show=False, legend=False, c='#0066cc', linewidth=2)
#overlay just the censoring
szFree_start_kmf.plot(show_censors=True, censor_styles={'ms':12, 'marker':'|', 'alpha':0.3}, ci_show=False, legend=False, c='#0066cc', linewidth=0, label='_nolegend_')
plt.xlim([0,max_followup.days/365])
plt.ylim([0, 1])
plt.xlabel("Years After 6+ Months Seizure Free", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("Probability of a Breakthrough Seizure", fontsize=16)
plt.legend(['n = 987'], loc='lower right', fontsize=14)
plt.savefig(f'{fig_save_dir}/fig_4.pdf', dpi=600)
plt.show()

#get demographics info
#load patients
pat_info = pd.read_pickle('demographics_information_py38.pkl')

#find all MRNS for patients used in the group
szFree_start_MRNs = [pat.pat_id for pat in szFree_start_pats]

#get patient info for those patients
szFree_start_pat_info = pat_info.loc[pat_info['MRN'].isin(szFree_start_MRNs)].drop_duplicates(subset='MRN')

#get censored and uncensored patients
szFree_start_censored_km = szFree_start_km_table.loc[(szFree_start_km_table['duration'] <= max_followup.days) & (szFree_start_km_table['event'] != 1)]
szFree_start_uncensored_km = szFree_start_km_table.loc[~szFree_start_km_table.index.isin(szFree_start_censored_km.index)]

#get patient info for censored and uncensored patients
szFree_start_censored_pat_info = szFree_start_pat_info.loc[szFree_start_pat_info['MRN'].isin(szFree_start_censored_km['MRN'])][['DOB_YR', 'ZIP', 'RACE', 'GENDER']]
szFree_start_uncensored_pat_info = szFree_start_pat_info.loc[szFree_start_pat_info['MRN'].isin(szFree_start_uncensored_km['MRN'])][['DOB_YR', 'ZIP', 'RACE', 'GENDER']]

#Mann whitney tests between censored and uncensored patients for DOB
print(f"szFree Start: {mannwhitneyu(szFree_start_uncensored_pat_info['DOB_YR'], szFree_start_censored_pat_info['DOB_YR'])}")

#get counts of genders for Chi2
szFree_start_uncensored_genders = get_demographic_counts(szFree_start_uncensored_pat_info['GENDER'].dropna())
szFree_start_censored_genders = get_demographic_counts(szFree_start_censored_pat_info['GENDER'].dropna())
initialize_missing_keys(szFree_start_uncensored_genders, szFree_start_censored_genders)
szFree_start_genders = pd.DataFrame([szFree_start_uncensored_genders, normalize_dict(szFree_start_uncensored_genders), szFree_start_censored_genders, normalize_dict(szFree_start_censored_genders)], index=['uncensored', 'uncensored_normalized', 'censored', 'censored_normalized']).transpose()
print(chi2_contingency(szFree_start_genders[['uncensored', 'censored']])[1])

#get counts of races for Chi2
szFree_start_uncensored_races = get_demographic_counts(szFree_start_uncensored_pat_info['RACE'].dropna())
szFree_start_censored_races = get_demographic_counts(szFree_start_censored_pat_info['RACE'].dropna())
initialize_missing_keys(szFree_start_uncensored_races, szFree_start_censored_races)
szFree_start_races = pd.DataFrame([szFree_start_uncensored_races, normalize_dict(szFree_start_uncensored_races), szFree_start_censored_races, normalize_dict(szFree_start_censored_races)], index=['uncensored', 'uncensored_normalized', 'censored', 'censored_normalized']).transpose()
print(chi2_contingency(szFree_start_races[['uncensored', 'censored']])[1])

#Calculate the pearson correlation between fraction of visits with seizures and length of followup
plt.figure(figsize=(8,6))
plt.scatter(szFree_start_pat_visits.total_followup, szFree_start_pat_visits.frac_hasSz, alpha=0.1)
plt.title("Fraction of Visits With Seizures vs. Length of Followup\n(6+ Months Seizure Free Cohort)")
plt.xlabel("Total Duration of Followup")
plt.ylabel("Fraction of Visits With Seizures")
plt.show()
print(f"Pearson Correlation: {pearsonr(szFree_start_pat_visits.total_followup, szFree_start_pat_visits.frac_hasSz)}")
print(f"Spearman Correlation: {spearmanr(szFree_start_pat_visits.total_followup, szFree_start_pat_visits.frac_hasSz)}")
print("\n")
