import pandas as pd
import json
import numpy as np
import sys
import os
from timeline_utils import replace_implicits

#directory path to the predictions from each seed of the model. Each prediction file should be stored in its own respective subdirectory
eqa_pred_dir = 'eqa_predictions'

#implicit number conversions
implicit_converter = {
    'a couple':2,
    'couple':2,
    'Couple':2,
    'a few':3,
    'few':3,
    'Few':3,
    'several':4,
    'Several':4,
    'multiple':4,
    'Multiple':4,
    'many':5,
    'Many':5
}

#read the prediction files. 
eqa_predictions = {}
for seed_dir in os.listdir(eqa_pred_dir):
    if '.ipynb' in seed_dir:
        continue
    with open(f"{eqa_pred_dir}/{seed_dir}/predict_predictions.json", 'r') as f:
        eqa_predictions[int(seed_dir.split("_")[-1])] = json.load(f)
        
#generate the pqf and elo datasets
pqf_dataset = {}
elo_dataset = {}
for seed in eqa_predictions:
    pqf_dataset[seed] = []
    elo_dataset[seed] = []
    
    #iterate through each of the predictions
    for pred_id in eqa_predictions[seed]:
    
        #check if it is a pqf or elo prediction, and handle accordingly
        if 'pqf' in pred_id.lower():
            pqf_dataset[seed].append({
                'text':replace_implicits(eqa_predictions[seed][pred_id], implicit_converter),
                'summary':"NULL"})
        elif 'elo' in pred_id.lower():
            elo_dataset[seed].append({
                'text':replace_implicits(eqa_predictions[seed][pred_id], implicit_converter),
                'summary':"NULL"})
            
#generate prediction dataset
for seed in eqa_predictions:
    with open(f'RB_pqf_predictions_for_summarization_{seed}.json', 'w') as f:
        for datum in pqf_dataset[seed]:
            json.dump(datum, f)
            f.write('\n')
    with open(f'RB_elo_predictions_for_summarization_{seed}.json', 'w') as f:
        for datum in elo_dataset[seed]:
            json.dump(datum, f)
            f.write('\n')