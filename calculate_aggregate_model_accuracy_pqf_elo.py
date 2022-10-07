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
import scipy
from sklearn.preprocessing import normalize
sns.set_theme(style='ticks')

def isFloat(string):
    try:
        float(string)
        return True
    except:
        return False

#figure save path
fig_save_dir = r'<PATH_TO_SAVE_DIR>'

#the following code is very similar to that in calculate_quantitative_seizure_outcomes.py
#file paths for the dataset, predictions and summaries directories
all_elo_summaries_path = '<path_to_date_of_last_seizure_summaries>'
all_pqf_summaries_path = '<path_to_seizure_frequency_summaries>'
all_eqa_predictions_path = r'<path_to_roberta_model_predictions>'
eqa_dataset_path = r'<path_to_roberta_model_dataset'

#container to hold all predictions and summaries
all_predictions_and_summaries_models = {}
all_pats_models = {}
all_pat_ids_models = {}
all_pat_visits_models = {}

#load the original dataset
with open(eqa_dataset_path, 'r') as f:
    eqa_dataset = json.load(f)['data']

#for each seed
for seed in [2, 17, 42, 97, 136]:
    elo_summaries_path = f'{all_elo_summaries_path}{seed}/generated_predictions.txt'
    pqf_summaries_path = f'{all_pqf_summaries_path}{seed}/generated_predictions.txt'
    eqa_predictions_path = f'{all_eqa_predictions_path}{seed}_eval_predictions.json'

    #load the summarizations
    elo_summaries = []
    with open(elo_summaries_path, 'r') as f:
        for line in f.readlines():
            elo_summaries.append(line.splitlines()[0])     
        if len(elo_summaries) < 300:
            elo_summaries.append("")
    pqf_summaries = []
    with open(pqf_summaries_path, 'r') as f:
        for line in f.readlines():
            pqf_summaries.append(line.splitlines()[0])
        if len(pqf_summaries) < 300:
            pqf_summaries.append("")

    #load the original eqa predictions
    with open(eqa_predictions_path, 'r') as f:
        eqa_predictions = json.load(f)

    #organize the text extraction predictions and summaries together into a single container
    predictions_and_summaries = {}

    #for each eqa prediction grab the summary of the prediction, and the prediction 
    pqf_ct = 0
    elo_ct = 0
    dataset_ct = 0
    for datum in eqa_dataset: 
        full_id = datum['title'].split("_")
        visit_date = datetime.strptime(full_id[2], '%Y-%m-%d')
        filename = "_".join(full_id)
        pat_id = full_id[0]

        #organize into correct format
        for qa in datum['paragraphs'][0]['qas']:
            i = qa['id']
            if int(i) % 2 == 0:
                predictions_and_summaries[i] = {
                    'prediction':eqa_predictions[i],
                    'summarization':pqf_summaries[pqf_ct],
                    "sz_per_month":np.nan,
                    "last_occurrence":np.nan,
                    'visit_date':visit_date,
                    'id':i+"_pqf",
                    'context':datum['paragraphs'][0]['context'],
                    'filename':filename,
                    'pat_id':pat_id
                }
                pqf_ct += 1
                dataset_ct += 1
            else:
                predictions_and_summaries[i] = {
                    'prediction':eqa_predictions[i],
                    'summarization':elo_summaries[elo_ct],
                    "sz_per_month":np.nan,
                    "last_occurrence":np.nan,
                    'visit_date':visit_date,
                    'id':i+"_elo",
                    'context':datum['paragraphs'][0]['context'],
                    'filename':filename,
                    'pat_id':pat_id
                }
                elo_ct += 1
                dataset_ct += 1

    #convert to dataframe
    predictions_and_summaries = pd.DataFrame(predictions_and_summaries).transpose()

    #translate the summaries into useable numbers
    timeline_utils.translate_summaries(predictions_and_summaries)
    predictions_and_summaries = timeline_utils.process_since_last_visit(predictions_and_summaries)
    
    print("=====================================================================")
    print("Finished quantification. Proceeding with Patient and Visit generation")
    print("=====================================================================")
    
    #create Patient and Visit objects and incorporate each translated summary into them.
    pat_visits = []
    pat_ids = [] 
    pats = []
    debug = 0
    for idx, row in predictions_and_summaries.iterrows():
        #get the visit information from this visit
        fn = filename.split("_")
        pat_id = row['id']
        note_id = row['id']
        note_author = fn[1]
        visit_date = datetime.strptime(fn[2], '%Y-%m-%d')
        context = row['context']
        szfreq = row['sz_per_month']
        elo = row['last_occurrence']

        pats.append(timeline_utils.Patient(pat_id))        
        pat_ids.append(pat_id)

        #find which patient this visit corresponds to
        patient_index = pat_ids.index(pat_id)
        new_visit = timeline_utils.Visit(patient=pats[patient_index],
                                               note_id=note_id,
                                               author=note_author,
                                               visit_date=visit_date,
                                               hasSz=0, 
                                               pqf=szfreq,
                                               elo=elo,
                                               context=context)

        if new_visit not in pat_visits:
            pat_visits.append(new_visit)
                
    all_pats_models[seed] = pats
    all_pat_visits_models[seed] = pat_visits
    all_pat_ids_models[seed] = pat_ids
    all_predictions_and_summaries_models[seed] = predictions_and_summaries
    print('\n\n\n')
    
#create aggregate patients
all_agg_pats = np.array(timeline_utils.aggregate_patients_and_visits(all_pats_models))


#For annotator predictions
#file paths for the dataset, predictions and summaries directories
all_elo_summaries_path = '<annotator_summaries_path_elo>'
all_pqf_summaries_path = '<annotator_summaries_path_pqf>'
all_eqa_predictions_path = r'<annotator_predictions_path>'

#container to hold all predictions and summaries
all_predictions_and_summaries_annotators = {}

#for each of the 15 annotators, get their predictions
for annotator in range(0, 15):
    elo_summaries_path = f'{all_elo_summaries_path}{annotator}/generated_predictions.txt'
    pqf_summaries_path = f'{all_pqf_summaries_path}{annotator}/generated_predictions.txt'
    eqa_predictions_path = f'{all_eqa_predictions_path}{annotator}_predictions.json'

    #load the summarizations
    elo_summaries = []
    with open(elo_summaries_path, 'r') as f:
        for line in f.readlines():
            elo_summaries.append(line.splitlines()[0])     
        if len(elo_summaries) < 300:
            elo_summaries.append("")
    pqf_summaries = []
    with open(pqf_summaries_path, 'r') as f:
        for line in f.readlines():
            pqf_summaries.append(line.splitlines()[0])
        if len(pqf_summaries) < 300:
            pqf_summaries.append("")

    #load the original eqa predictions
    with open(eqa_predictions_path, 'r') as f:
        eqa_predictions = json.load(f)

    #organize the text extraction predictions and summaries together into a single container
    predictions_and_summaries = {}

    #for each eqa prediction grab the summary of the prediction, and the prediction 
    pqf_ct = 0
    elo_ct = 0
    dataset_ct = 0
    for datum in eqa_dataset: 
        full_id = datum['title'].split("_")
        visit_date = datetime.strptime(full_id[2], '%Y-%m-%d')
        filename = "_".join(full_id)
        pat_id = full_id[0]

        #organize into correct format
        for qa in datum['paragraphs'][0]['qas']:
            i = qa['id']
            if i not in eqa_predictions:
                continue
            if int(i) % 2 == 0:
                predictions_and_summaries[i] = {
                    'prediction':eqa_predictions[i],
                    'summarization':pqf_summaries[pqf_ct],
                    "sz_per_month":np.nan,
                    "last_occurrence":np.nan,
                    'visit_date':visit_date,
                    'id':i+"_pqf",
                    'context':datum['paragraphs'][0]['context'],
                    'filename':filename,
                    'pat_id':pat_id
                }
                pqf_ct += 1
                dataset_ct += 1
            else:
                predictions_and_summaries[i] = {
                    'prediction':eqa_predictions[i],
                    'summarization':elo_summaries[elo_ct],
                    "sz_per_month":np.nan,
                    "last_occurrence":np.nan,
                    'visit_date':visit_date,
                    'id':i+"_elo",
                    'context':datum['paragraphs'][0]['context'],
                    'filename':filename,
                    'pat_id':pat_id
                }
                elo_ct += 1
                dataset_ct += 1

    #convert to dataframe
    predictions_and_summaries = pd.DataFrame(predictions_and_summaries).transpose()
    
    #translate the summaries into useable numbers
    timeline_utils.translate_summaries(predictions_and_summaries)
    predictions_and_summaries = timeline_utils.process_since_last_visit(predictions_and_summaries)
    all_predictions_and_summaries_annotators[annotator]=predictions_and_summaries
    
    
#get the ground truth values
ground_truth_quantities = pd.read_excel('ground_truths.xlsx')
quantity_cols = [f'Quantity_{i}' for i in range(5)]
all_ans = []
#collect all correct answers for a note
for idx in ground_truth_quantities.index:
    answers = [ans for ans in ground_truth_quantities.loc[idx, quantity_cols] if not pd.isnull(ans)]
    if answers:
        all_ans.append(answers)
    else:
        all_ans.append(np.nan)
ground_truth_quantities['all_answers'] = all_ans


#go through each annotator's quantities and calculate their accuracies
annotator_accuracy = {}
for annotator in all_predictions_and_summaries_annotators:
    annotator_accuracy[annotator] = []
    for idx, row in all_predictions_and_summaries_annotators[annotator].iterrows():
        
        #get the ground truth for this index
        ground_truth = ground_truth_quantities.loc[int(idx), 'all_answers']
        
        #check if it's a pqf question
        if 'pqf' in row['id']:
            #check if it's nan
            if np.any(pd.isnull(ground_truth)):
                if pd.isnull(row['sz_per_month']):
                    annotator_accuracy[annotator].append(1)
                else:
                    annotator_accuracy[annotator].append(0)
            else:
                if pd.isnull(row['sz_per_month']):
                    annotator_accuracy[annotator].append(0)
                else:
                    #for each grounth truth answer, check if it's equal to the predicted quantity. If any of them are, then it is correct
                    annotator_accuracy[annotator].append(int(True in [(np.round(row['sz_per_month'], decimals=2) == np.round(float(gt), decimals=2)) if (isFloat(gt) and isFloat(row['sz_per_month'])) else (row['sz_per_month'] == gt) for gt in ground_truth]))
        #otherwise, it must be a elo quesiton
        else:
            if np.any(pd.isnull(ground_truth)):
                if pd.isnull(row['last_occurrence']):
                    annotator_accuracy[annotator].append(1)
                else:
                    annotator_accuracy[annotator].append(0)
            else:
                if pd.isnull(row['last_occurrence']):
                    annotator_accuracy[annotator].append(0)
                else:
                    #for each grouth truth answer, check if it is within 1 week of the predicted quantity. If any of them are, then it is correct
                    #first, convert ground_truth to datetimes
                    ground_truth = [datetime.strptime(gt, '%B %d %Y') for gt in ground_truth]
                    annotator_accuracy[annotator].append(int(True in [np.abs((row['last_occurrence'] - gt).days) <= 7 for gt in ground_truth]))
                
    annotator_accuracy[annotator] = np.mean(annotator_accuracy[annotator])
    
#go through aggregate patient quantities and calculate their accuracies
aggregate_model_accuracy = []
for pat in all_agg_pats:
    
    #get the ground truth for this index
    ground_truth = ground_truth_quantities.loc[int(pat.pat_id.split("_")[0]), 'all_answers']
    
    #check if it's a pqf question
    if 'pqf' in pat.pat_id:
        #check if it's nan
        if np.any(pd.isnull(ground_truth)):
            if pd.isnull(pat.aggregate_visits[0].pqf):
                aggregate_model_accuracy.append(1)
            else:
                aggregate_model_accuracy.append(0)
        else:
            if pd.isnull(pat.aggregate_visits[0].pqf):
                aggregate_model_accuracy.append(0)
            else:
                #for each ground truth answer, check if it's equal to the predicted quantity. If any of them are, then it is correct
                aggregate_model_accuracy.append(int(True in [(np.round(pat.aggregate_visits[0].pqf, decimals=2) == np.round(float(gt), decimals=2)) if (isFloat(gt) and isFloat(pat.aggregate_visits[0].pqf)) else (pat.aggregate_visits[0].pqf == gt) for gt in ground_truth]))
    #otherwise, it must be a elo quesiton
    else:
        if np.any(pd.isnull(ground_truth)):
            if pd.isnull(pat.aggregate_visits[0].elo):
                aggregate_model_accuracy.append(1)
            else:
                aggregate_model_accuracy.append(0)
        else:
            if pd.isnull(pat.aggregate_visits[0].elo):
                aggregate_model_accuracy.append(0)
            else:
                #for each grouth truth answer, check if it is within 1 week of the predicted quantity. If any of them are, then it is correct
                #first, convert ground_truth to datetimes
                ground_truth = [datetime.strptime(gt, '%B %d %Y') for gt in ground_truth]
                aggregate_model_accuracy.append(int(True in [np.abs((pat.aggregate_visits[0].elo - gt).days) <= 7 for gt in ground_truth]))
                
#go through each models' quantities and calculate their accuracies
model_accuracy = {}
for seed in all_predictions_and_summaries_models:
    model_accuracy[seed] = []
    for idx, row in all_predictions_and_summaries_models[seed].iterrows():
        
        #get the ground truth for this index
        ground_truth = ground_truth_quantities.loc[int(idx), 'all_answers']
        
        #check if it's a pqf question
        if 'pqf' in row['id']:
            #check if it's nan
            if np.any(pd.isnull(ground_truth)):
                if pd.isnull(row['sz_per_month']):
                    model_accuracy[seed].append(1)
                else:
                    model_accuracy[seed].append(0)
            else:
                if pd.isnull(row['sz_per_month']):
                    model_accuracy[seed].append(0)
                else:
                    #for each grounth truth answer, check if it's equal to the predicted quantity. If any of them are, then it is correct
                    model_accuracy[seed].append(int(True in [(np.round(row['sz_per_month'], decimals=2) == np.round(float(gt), decimals=2)) if (isFloat(gt) and isFloat(row['sz_per_month'])) else (row['sz_per_month'] == gt) for gt in ground_truth]))
        #otherwise, it must be a elo quesiton
        else:
            if np.any(pd.isnull(ground_truth)):
                if pd.isnull(row['last_occurrence']):
                    model_accuracy[seed].append(1)
                else:
                    model_accuracy[seed].append(0)
            else:
                if pd.isnull(row['last_occurrence']):
                    model_accuracy[seed].append(0)
                else:
                    #for each grouth truth answer, check if it is within 1 week of the predicted quantity. If any of them are, then it is correct
                    #first, convert ground_truth to datetimes
                    ground_truth = [datetime.strptime(gt, '%B %d %Y') for gt in ground_truth]
                    model_accuracy[seed].append(int(True in [np.abs((row['last_occurrence'] - gt).days) <= 7 for gt in ground_truth]))
                
    model_accuracy[seed] = np.mean(model_accuracy[seed])
    
# Overall annotator stats
overall_accuracy = list(annotator_accuracy.values())
overall_mean = np.mean(overall_accuracy)
overall_95 = scipy.stats.t.interval(0.95, len(overall_accuracy) - 1, overall_mean, np.std(overall_accuracy))

#model accuracies
model_accs = sorted(list(model_accuracy.values()))
model_mean = np.mean(model_accs)
model_95 = scipy.stats.t.interval(0.95, len(model_accs) - 1, model_mean, np.std(model_accs))

#aggregate accuracy
aggregate_acc = np.mean(aggregate_model_accuracy)


#plot the accuracies
x = 1
width = 0.15
sep = width/4
x_pos = [x+i*width for i in range(-3,4)]

#model accuracies
plt.bar(x_pos[0],   model_accs[0], width=width, color='#EDF8FB', edgecolor='black', capsize=3)
plt.bar(x_pos[1],   model_accs[1], width=width, color='#d0d1e6', edgecolor='black', capsize=3)
plt.bar(x_pos[2],   model_accs[2], width=width, color='#a6bddb', edgecolor='black', capsize=3)
plt.bar(x_pos[3],   model_accs[3], width=width, color='#74a9cf', edgecolor='black', capsize=3)
plt.bar(x_pos[4],   model_accs[4], width=width, color='#2b8cbe', edgecolor='black', capsize=3)
plt.bar(x_pos[5]+sep,   aggregate_acc, width = width, color='#045a8d', edgecolor='black')

#annotator agreement
plt.bar(x_pos[6]+sep, overall_mean, yerr = overall_mean-overall_95[0], width=width, color='#FEE090', edgecolor='black', capsize=3)

plt.ylim([0.7, 1.0])
plt.xticks(ticks=x_pos[:-2] + [x_pos[-2] + sep, x_pos[-1] + sep],
           labels=['Model 1', 'Model 2', 'Model 3', 'Model 4 ', 'Model 5', 'Plurality Voting', 'Humans'],
           rotation = -45, ha='left')
# plt.title("Plurality-Voted Seizure Frequency and Date of Last Seizure Extraction\nAccuracy Against Annotator Accuracy (95% Confidence Intervals)\n")
plt.ylabel("Accuracy")


save_path=f'{fig_save_dir}/Fig_5b'
plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{save_path}.pdf", dpi=600, bbox_inches='tight')