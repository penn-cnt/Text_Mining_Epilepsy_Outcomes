import pickle
import numpy as np
import json
import pandas as pd
from datetime import datetime, timedelta
import string
import timeline_utils
import annotation_utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import scipy
from sklearn.preprocessing import normalize
sns.set_theme(style='ticks')

def isFloat(s):
    try:
        float(s)
        return True
    except:
        return False

#figure save path
fig_save_dir = r'<PATH_TO_SAVE_FIGURES'>


#file paths for the dataset, predictions and summaries directories. Models predicted on the JAMIA 2022 dataset
all_hasSz_classifications_path = r'<MODEL_SEIZURE_FREEDOM_CLASSIFICATION_PREDICTIONS_DIRECTORY>'

#container to hold all predictions
all_predictions_and_summaries = {}
all_pats = {}
all_pat_ids = {}
all_pat_visits = {}
#for each seed, get predictions for visits and patients
for seed in [2, 17, 42, 97, 136]:
    hasSz_classifications_path = f'{all_hasSz_classifications_path}/{seed}_eval_predictions.tsv'

    #Load the classifications
    classification_preds = pd.read_csv(hasSz_classifications_path, sep='\t')
    
    #create Visit objects, with blank elo and pqf
    pat_visits = []
    pat_ids = [] 
    pats = []
    for idx, row in classification_preds.iterrows():
        #parse the hasSz array
        hasSz_arr = [float(x) for x in row.Predictions[1:-1].split(" ") if isFloat(x)]
        hasSz = np.argmax(hasSz_arr)
        
        #create a new patient and new visit
        new_pat = timeline_utils.Patient(idx)
        new_visit = timeline_utils.Visit(patient=new_pat,
                                               note_id="None",
                                               author="None",
                                               visit_date="None",
                                               hasSz=hasSz, 
                                               pqf=0,
                                               elo=0,
                                               context=hasSz_arr)
        pats.append(new_pat)
    
    #store the patients and visits
    all_pats[seed] = pats
    all_pat_visits[seed] = pat_visits    
    
    
#create aggregate patients
all_agg_pats = np.array(timeline_utils.aggregate_patients_and_visits(all_pats))
#convert has-seizures values to text
onehot = {0: 'No', 1: 'Yes', 2: 'no-answer'}

#load the seizure freedom dataset from the JAMIA 2022 paper
hasSz_dataset = '<PATH_TO_DATASET>'
with open(hasSz_dataset, 'r') as f:
    hasSz_dataset = [json.loads(s) for s in f.read().splitlines()]
    
#calculate the aggregate accuracy
aggregate_acc = np.sum([onehot[all_agg_pats[i].aggregate_visits[0].hasSz] == hasSz_dataset[i]['answer'] for i in range(len(all_agg_pats))])/len(all_agg_pats)
print(f"Ensemble Accuracy: {aggregate_acc}")

# Get agreement statistics
#agreement values were calculated using code from the JAMIA 2022 paper but with simple agreement, not cohen's kappa.
g1_agreement = [0.9255014326647565, 0.9364161849710982, 0.9329608938547486]
g1_mean = np.mean(g1_agreement)
g1_95 = scipy.stats.t.interval(0.95, len(g1_agreement) - 1, g1_mean, np.std(g1_agreement))
g2_agreement = [0.8918128654970761, 0.9439775910364145, 0.8783382789317508]
g2_mean = np.mean(g2_agreement)
g2_95 = scipy.stats.t.interval(0.95, len(g2_agreement) - 1, g2_mean, np.std(g2_agreement))
g3_agreement = [0.9228650137741047, 0.9157303370786517, 0.75, 0.9314285714285714, 0.7828947368421053]
g3_mean = np.mean(g3_agreement)
g3_95 = scipy.stats.t.interval(0.95, len(g3_agreement) - 1, g3_mean, np.std(g3_agreement))
g4_agreement = [0.9213483146067416, 0.9152542372881356, 0.9302325581395349]
g4_mean = np.mean(g4_agreement)
g4_95 = scipy.stats.t.interval(0.95, len(g4_agreement) - 1, g4_mean, np.std(g4_agreement))
g5_agreement = [0.9202279202279202, 0.9339080459770115, 0.9022988505747126]
g5_mean = np.mean(g5_agreement)
g5_95 = scipy.stats.t.interval(0.95, len(g5_agreement) - 1, g5_mean, np.std(g5_agreement))

# Overall agreement
overall_agreement = g1_agreement + g2_agreement + g3_agreement + g4_agreement + g5_agreement
overall_mean = np.mean(overall_agreement)
overall_95 = scipy.stats.t.interval(0.95, len(overall_agreement) - 1, overall_mean, np.std(overall_agreement))

#model accuracies
model_accs = [0.82, 0.8267, 0.8367, 0.8433, 0.8633]
model_mean = np.mean(model_accs)
model_95 = scipy.stats.t.interval(0.95, len(model_accs) - 1, model_mean, np.std(model_accs))

#plot the output
x = 1
width = 0.15
sep = width/4
x_pos = [x+i*width for i in range(-3,4)]

#model accuracies
plt.bar(x_pos[0],   0.82, width=width, color='#EDF8FB', edgecolor='black', capsize=3)
plt.bar(x_pos[1],   0.8267, width=width, color='#d0d1e6', edgecolor='black', capsize=3)
plt.bar(x_pos[2],   0.8367, width=width, color='#a6bddb', edgecolor='black', capsize=3)
plt.bar(x_pos[3],   0.8433, width=width, color='#74a9cf', edgecolor='black', capsize=3)
plt.bar(x_pos[4],   0.8633, width=width, color='#2b8cbe', edgecolor='black', capsize=3)
plt.bar(x_pos[5]+sep,   aggregate_acc, width = width, color='#045a8d', edgecolor='black')

#annotator agreement
plt.bar(x_pos[6]+sep, overall_mean, yerr = overall_mean-overall_95[0], width=width, color='#FEE090', edgecolor='black', capsize=3)

#set up the plot
plt.ylim([0.7, 1.0])
plt.xticks(ticks=x_pos[:-2] + [x_pos[-2] + sep, x_pos[-1] + sep],
           labels=['Model 1 Accuracy', 'Model 2 Accuracy', 'Model 3 Accuracy', 'Model 4  Accuracy', 'Model 5  Accuracy', 'Plurality Voting Accuracy', 'Overall Annotator Agreement'],
           rotation = -45, ha='left')
plt.ylabel("Accuracy or Agreement")

#save the figures
save_path=f'{fig_save_dir}/Fig_5a'
plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{save_path}.pdf", dpi=600, bbox_inches='tight')