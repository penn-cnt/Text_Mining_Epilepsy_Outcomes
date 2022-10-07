import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from datetime import datetime, timedelta
import re
import pickle
import json
import sys
from timeline_utils import get_paragraph_with_max_token_length

#what are we saving the dataset(s) to?
classification_output = 'classification_dataset.json'
EQA_output = 'eqa_dataset.json'

#attending addendum string
addendum_string = "I saw and evaluated/examined  on  and reviewed 's notes. I agree with the history, physical exam"

#regex for extracting HPI/Interval History
hpi_whitelist_regex = r"(?im)^(\bHPI\b|\bHistory of Present Illness\b|\bInterval History\b)"
hpi_blacklist_regex = r"(?im)(\b(Past |Prior )?((Social|Surgical|Family|Medical|Psychiatric|Seizure|Disease|Epilepsy) )History\b|\bSemiology\b|\bLab|\bExam|\bDiagnostic|\bImpression|\bPlan\b|\bPE\b|\bRisk Factor|\bMedications|\bAllerg)"

#load the data
epileptologist_notes = pd.read_pickle('<PATH_TO_RAW_NOTES>')

#remove attending addendums
attendings_phrase = epileptologist_notes['NOTE_TEXT'].apply(lambda x: SequenceMatcher(None, x[:len(addendum_string)], addendum_string).ratio() > 0.60)
attendings_explicit = epileptologist_notes['NOTE_TEXT'].apply(lambda x: 'attending addendum' in x.lower()[:500])
attendings_explicit_cased = epileptologist_notes['NOTE_TEXT'].apply(lambda x: 'Addendum:' in x[:500])
epileptologist_notes = epileptologist_notes[~(attendings_explicit | attendings_phrase | attendings_explicit_cased)].reset_index(drop=True)

#remove quick notes
quick_notes = epileptologist_notes['NOTE_TEXT'].apply(lambda x: 'quick note' in x.lower()[:20])
epileptologist_notes = epileptologist_notes[~quick_notes].reset_index(drop=True)

#How are the lines in the notes denoted from each other in raw text form?
splitter = "  "
max_token_length = 512-20 #512 tokens max, with 20 set aside for questions
hpi_paragraphs_CB = []
hpi_paragraphs_RB = []

#path to the models
CB_path = r'<PATH TO CNT-UPenn/Bio_ClinicalBERT_for_seizureFreedom_classification>'
RB_path = r'<PATH TO CNT-UPenn/RoBERTa_for_seizureFrequency_QA>'

for idx, row in epileptologist_notes.iterrows():
    
    #extract the hpi/interval history relevant paragraphs using the CB tokenizer
    hpi_paragraph_CB = get_paragraph_with_max_token_length(hpi_whitelist_regex, hpi_blacklist_regex, row['NOTE_TEXT'], row['NOTE_AUTHOR'], row['MRN'], str(row['VISIT_DATE']), str(row['NOTE_ID']), CB_path, splitter, max_token_length)
    if hpi_paragraph_CB != None:
        hpi_paragraphs_CB.append(hpi_paragraph_CB)
        
    #extract the hpi/interval history relevant paragraphs using the RB tokenizer
    hpi_paragraph_RB = get_paragraph_with_max_token_length(hpi_whitelist_regex, hpi_blacklist_regex, row['NOTE_TEXT'], row['NOTE_AUTHOR'], row['MRN'], str(row['VISIT_DATE']), str(row['NOTE_ID']), RB_path, splitter, max_token_length)
    if hpi_paragraph_RB != None:
        hpi_paragraphs_RB.append(hpi_paragraph_RB)
        
    if idx % 5000 == 0:
        print(idx)
        
#Initialize the eqa questions and the classification questions
eqa_dataset = {'version':'v2.0', 'data':[]}
classification_dataset = []

#what questions are we asking
classification_q = "Has the patient had recent events?"
eqa_q = ["How often does the patient have events?", "When was the patient's last event?"]
eqa_identifer = ["PQF", "ELO"]

#for the classification dataset, format it
for datum in hpi_paragraphs_CB:
    for key in datum.keys():
        if isinstance(key, str):
            continue
        classification_dataset.append({
            'id':datum['filename']+"_hasSz",
            'passage':datum[key],
            'question':classification_q
        })
        #put a break here to ensure that we only take the first entry. This is because later entries may include historical hpi or other sections
        break
        
#for the eqa dataset, format it
for datum in hpi_paragraphs_RB:
    for key in datum.keys():
        if isinstance(key, str):
            continue
        #add eqa questions
        for i in range(len(eqa_q)):
            eqa_dataset['data'].append({
                'id':datum['filename']+"_"+eqa_identifer[i],
                'context':datum[key],
                'question':eqa_q[i]
            })
        #put a break here to ensure that we only take the first entry. This is because later entries may include historical hpi or other sections
        break
        
#save datasets to json
with open(classification_output, 'w') as f:
    for datum in classification_dataset:
        json.dump(datum, f)
        f.write('\n')
with open(EQA_output, 'w') as f:
    json.dump(eqa_dataset, f)