import pickle
import numpy as np
import json
import pandas as pd
from datetime import datetime, timedelta
import string
import timeline_utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.preprocessing import normalize
sns.set_theme(style='ticks')

#minimum date allowed
date_cutoff = datetime.strptime('01-01-2005', '%m-%d-%Y')

#file paths for the dataset, predictions and summaries directories
all_elo_summaries_path = '<path_to_date_of_last_seizure_summaries>'
all_pqf_summaries_path = '<path_to_seizure_frequency_summaries>'
all_eqa_predictions_path = r'<path_to_roberta_model_predictions>'
eqa_dataset_path = r'<path_to_roberta_model_dataset'
all_hasSz_classifications_path = r'<path_to_clinicalbert_predictions>'

#container to hold all predictions and summaries
all_predictions_and_summaries = {}
all_pats = {}
all_pat_ids = {}
all_pat_visits = {}

#load the original dataset
with open(eqa_dataset_path, 'r') as f:
    eqa_dataset = json.load(f)['data']

for seed in [2, 17, 42, 97, 136]:
    elo_summaries_path = f'{all_elo_summaries_path}{seed}/generated_predictions.txt'
    pqf_summaries_path = f'{all_pqf_summaries_path}{seed}/generated_predictions.txt'
    eqa_predictions_path = f'{all_eqa_predictions_path}{seed}/predict_predictions.json'
    hasSz_classifications_path = f'{all_hasSz_classifications_path}{seed}/predictions.tsv'
    
    #Load the classifications
    classification_preds = pd.read_csv(hasSz_classifications_path, sep='\t')
    
    #remove the erroneous dates
    #some visits had odd visit_dates, like from the 60s and 70s
    classification_preds = classification_preds[classification_preds['ID'].apply(lambda x: (datetime.strptime(x.split('_')[-2], '%Y-%m-%d') >= date_cutoff))].reset_index(drop=True)

    #load date of last seizure summarizations
    elo_summaries = []
    with open(elo_summaries_path, 'r') as f:
        for line in f.readlines():
            elo_summaries.append(line.splitlines()[0])
    #if the last summarization was blank, it won't show up in the output text. Add it here. 
    elo_summaries.append("")        
    
    #load seizure frequency summarizations
    pqf_summaries = []
    with open(pqf_summaries_path, 'r') as f:
        for line in f.readlines():
            pqf_summaries.append(line.splitlines()[0])

    #load the original eqa predictions
    with open(eqa_predictions_path, 'r') as f:
        eqa_predictions = json.load(f)
        
    #organize the text extraction predictions and summaries together into a single container
    predictions_and_summaries = {}
    
    #some patients have duplicate but different MRNS. Find the analogs
    #get all entries with duplicate note identifiers and list the MRNs associated with them
    duplicate_identifiers = {}
    for i in classification_preds['ID']:
        full_id = i.split("_")
        key = "_".join([s for s in full_id[1:-1]])
        if key not in duplicate_identifiers:
            duplicate_identifiers[key] = [full_id[0]]
        else:
            duplicate_identifiers[key].append(full_id[0])
    duplicate_identifiers = {key: duplicate_identifiers[key] for key in duplicate_identifiers if (len(duplicate_identifiers[key]) > 1)}
    #go through and make a dictionary mapping one mrn to its duplicate
    #enforce - if Z is in the key, then swap it to the value
    mrn_analogs = {}
    for mrn_analog in duplicate_identifiers.values():
        #check if this mrn is a mapper or a mappee already. 
        if mrn_analog[0] not in mrn_analogs.keys() and mrn_analog[0] not in mrn_analogs.values():
            if 'z' in mrn_analog[0].lower():
                mrn_analogs[mrn_analog[0]] = mrn_analog[1]
            else:
                mrn_analogs[mrn_analog[1]] = mrn_analog[0]  

    #for each eqa prediction grab the summary of the prediction, and the prediction 
    pqf_ct = 0
    elo_ct = 0
    dataset_ct = 0
    for i in eqa_predictions:
        full_id = i.split("_")
        visit_date = datetime.strptime(full_id[3].split('.')[0], '%Y-%m-%d')
        filename = "_".join(full_id[:-1])
        pat_id = mrn_analogs[full_id[0]] if full_id[0] in mrn_analogs else full_id[0]

        #if the prediction is for pqf:
        if 'pqf' in i.lower():
            predictions_and_summaries[i] = {
                'prediction':eqa_predictions[i],
                'summarization':pqf_summaries[pqf_ct],
                "sz_per_month":np.nan,
                "last_occurrence":np.nan,
                'visit_date':visit_date,
                'id':i,
                'context':eqa_dataset[dataset_ct]['context'],
                'filename':filename,
                'pat_id':pat_id
            }
            pqf_ct += 1
            dataset_ct += 1
        #if the data is for elo
        elif 'elo' in i.lower():
            predictions_and_summaries[i] = {
                'prediction':eqa_predictions[i],
                'summarization':elo_summaries[elo_ct],
                "sz_per_month":np.nan,
                "last_occurrence":np.nan,
                'visit_date':visit_date,
                'id':i,
                'context':eqa_dataset[dataset_ct]['context'],
                'filename':filename,
                'pat_id':pat_id
            }
            elo_ct += 1
            dataset_ct += 1

    #convert to dataframe
    predictions_and_summaries = pd.DataFrame(predictions_and_summaries).transpose()
    #make sure there are no erroneous visit dates (before the cutoff)
    predictions_and_summaries = predictions_and_summaries[(predictions_and_summaries['visit_date'] >= date_cutoff)]
    
    #translate the summaries into useable numbers
    timeline_utils.translate_summaries(predictions_and_summaries)
    predictions_and_summaries = timeline_utils.process_since_last_visit(predictions_and_summaries)
    
    print("=====================================================================")
    print("Finished quantification. Proceeding with Patient and Visit generation")
    print("=====================================================================")
    
    #create Patient and Visit objects and incorporate each classification and translated summary into them.
    pat_visits = []
    pat_ids = [] 
    pats = []
    debug = 0
    for idx, row in classification_preds.iterrows():
        #get the visit information from this visit
        classification_id = row['ID'].split("_")
        filename = "_".join(classification_id[:-1])
        pat_id = mrn_analogs[classification_id[0]] if classification_id[0] in mrn_analogs else classification_id[0]
        note_id = classification_id[1]
        note_author = classification_id[2]
        visit_date = datetime.strptime(classification_id[3].split('.')[0], '%Y-%m-%d')
        
        #find the corresponding seizure frequency and last occurrence summaries for this visit
        visit_summaries = predictions_and_summaries.loc[predictions_and_summaries['filename'] == filename]

        #get the seizure frequency and last occurrence translations
        for summary_idx in visit_summaries.index:
            if 'pqf' in summary_idx.lower():
                szfreq = visit_summaries.loc[summary_idx, 'sz_per_month']
            else:
                elo = visit_summaries.loc[summary_idx, 'last_occurrence']
        context = visit_summaries['context'].iloc[0]

        #create a new patient if necessary
        if pat_id not in pat_ids:
            pats.append(timeline_utils.Patient(pat_id))
            pat_ids.append(pat_id)

        #find which patient this visit corresponds to
        patient_index = pat_ids.index(pat_id)
        new_visit = timeline_utils.Visit(patient=pats[patient_index],
                                               note_id=note_id,
                                               author=note_author,
                                               visit_date=visit_date,
                                               hasSz=row['argmax'], 
                                               pqf=szfreq,
                                               elo=elo,
                                               context=context)
        
        if new_visit not in pat_visits:
            pat_visits.append(new_visit)
                
    all_pats[seed] = pats
    all_pat_visits[seed] = pat_visits
    all_pat_ids[seed] = pat_ids
    all_predictions_and_summaries[seed] = predictions_and_summaries
    print('\n\n\n')

#condense the data into one structure
full_data = {'all_pats':all_pats, 
             'all_pat_visits': all_pat_visits, 
             'all_pat_ids':all_pat_ids, 
             'all_predictions_and_summaries':all_predictions_and_summaries}
#pickle it
with open(r'all_patients_and_visits_outcome_measures_py38.pkl', 'wb') as f:
    pickle.dump(datapull05192022_full, f)