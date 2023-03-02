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

#container to hold all predictions and summaries
all_pats = {}
all_pat_visits = {}
all_pat_predictions = {}

for seed in [2, 17, 42, 97, 136]:
    hasSz_classifications_path = f'{all_hasSz_classifications_path}/{seed}_eval_predictions.tsv'

    #Load the classifications
    classification_preds = pd.read_csv(hasSz_classifications_path, sep='\t')
    
    #process the classifications
    classification_preds['prediction'] = classification_preds.apply(lambda row: np.argmax([float(x) for x in row.Predictions[1:-1].split(" ") if isFloat(x)]), axis=1)
    
    #create Aggregate Visit objects, with blank elo and pqf
    pat_visits = []
    pat_ids = [] 
    pats = []
    for idx, row in classification_preds.iterrows():
        
        #create a new patient and new visit
        new_pat = timeline_utils.Patient(idx)
        new_visit = timeline_utils.Visit(patient=new_pat,
                                               note_id="None",
                                               author="None",
                                               visit_date="None",
                                               hasSz=row.prediction, 
                                               pqf=0,
                                               elo=0,
                                               context=row.True_Label) #we'll use the context parameter as a place to store the true label
        pats.append(new_pat)
                
    all_pats[seed] = pats
    all_pat_visits[seed] = pat_visits
    all_pat_predictions[seed] = classification_preds
    
#create aggregate patients
all_agg_pats = np.array(timeline_utils.aggregate_patients_and_visits(all_pats))
#convert has-seizures values to text
onehot = {0: 'No', 1: 'Yes', 2: 'no-answer'}

#load the seizure freedom dataset from the JAMIA 2022 paper
hasSz_dataset = '<PATH_TO_DATASET>'
with open(hasSz_dataset, 'r') as f:
    hasSz_dataset = [json.loads(s) for s in f.read().splitlines()]
    
#calculate performance metrics for the ensembles
f1_ensemble = f1_score([all_agg_pats[i].aggregate_visits[0].hasSz for i in range(len(all_agg_pats))], [all_agg_pats[i].aggregate_visits[0].all_visits[0].context for i in range(len(all_agg_pats))], average='weighted')
precision_ensemble = precision_score([all_agg_pats[i].aggregate_visits[0].hasSz for i in range(len(all_agg_pats))], [all_agg_pats[i].aggregate_visits[0].all_visits[0].context for i in range(len(all_agg_pats))], average='weighted')
recall_ensemble = recall_score([all_agg_pats[i].aggregate_visits[0].hasSz for i in range(len(all_agg_pats))], [all_agg_pats[i].aggregate_visits[0].all_visits[0].context for i in range(len(all_agg_pats))], average='weighted')

# Get agreement statistics
#agreement values were calculated using code from the JAMIA 2022 paper
g1_kappa_agree = [0.855592335603794, 0.8739380289154233, 0.8640527839749371]
g1_kappa_mean = np.mean(g1_kappa_agree)
g1_kappa_95 = scipy.stats.t.interval(0.95, len(g1_kappa_agree) - 1, g1_kappa_mean, np.std(g1_kappa_agree))
g1_f1_agree = [0.9240841152299175, 0.9345882654745714, 0.9322871074254783]
g1_f1_mean = np.mean(g1_f1_agree)
g1_f1_95 = scipy.stats.t.interval(0.95, len(g1_f1_agree) - 1, g1_f1_mean, np.std(g1_f1_agree))
g2_kappa_agree = [0.7963401091207571, 0.8897739903668025, 0.7739254217320877]
g2_kappa_mean = np.mean(g2_kappa_agree)
g2_kappa_95 = scipy.stats.t.interval(0.95, len(g2_kappa_agree) - 1, g2_kappa_mean, np.std(g2_kappa_agree))
g2_f1_agree = [0.9450318336941764, 0.8926430492603871, 0.8769728663997853]
g2_f1_mean = np.mean(g2_f1_agree)
g2_f1_95 = scipy.stats.t.interval(0.95, len(g2_f1_agree) - 1, g2_f1_mean, np.std(g2_f1_agree))
g3_kappa_agree = [0.864512516995921, 0.8601215423302598, 0.5570552147239265, 0.8857080657450745, 0.6140944760732421]
g3_kappa_mean = np.mean(g3_kappa_agree)
g3_kappa_95 = scipy.stats.t.interval(0.95, len(g3_kappa_agree) - 1, g3_kappa_mean, np.std(g3_kappa_agree))
g3_f1_agree = [0.7743622180777635, 0.9229928964322336, 0.9298324486163648, 0.7483746130030959, 0.9134647020357367]
g3_f1_mean = np.mean(g3_f1_agree)
g3_f1_95 = scipy.stats.t.interval(0.95, len(g3_f1_agree) - 1, g3_f1_mean, np.std(g3_f1_agree))
g4_kappa_agree = [0.8518606586612769, 0.8416532474503489, 0.8654212919947186]
g4_kappa_mean = np.mean(g4_kappa_agree)
g4_kappa_95 = scipy.stats.t.interval(0.95, len(g4_kappa_agree) - 1, g4_kappa_mean, np.std(g4_kappa_agree))
g4_f1_agree = [0.9297242488495319, 0.9222001452419722, 0.9161013481500624]
g4_f1_mean = np.mean(g4_f1_agree)
g4_f1_95 = scipy.stats.t.interval(0.95, len(g4_f1_agree) - 1, g4_f1_mean, np.std(g4_f1_agree))
g5_kappa_agree = [0.8577898681792531, 0.8863471778487753, 0.8321916351104115]
g5_kappa_mean = np.mean(g5_kappa_agree)
g5_kappa_95 = scipy.stats.t.interval(0.95, len(g5_kappa_agree) - 1, g5_kappa_mean, np.std(g5_kappa_agree))
g5_f1_agree = [0.9188969908516902, 0.9027163305139883, 0.9350209999507928]
g5_f1_mean = np.mean(g5_f1_agree)
g5_f1_95 = scipy.stats.t.interval(0.95, len(g5_f1_agree) - 1, g5_f1_mean, np.std(g5_f1_agree))

# Overall agreement
overall_kappa_agree = g1_kappa_agree + g2_kappa_agree + g3_kappa_agree + g4_kappa_agree + g5_kappa_agree
overall_kappa_mean = np.mean(overall_kappa_agree)
overall_kappa_95 = scipy.stats.t.interval(0.95, len(overall_kappa_agree) - 1, overall_kappa_mean, np.std(overall_kappa_agree))
overall_f1_agree = g1_f1_agree + g2_f1_agree + g3_f1_agree + g4_f1_agree + g5_f1_agree
overall_f1_mean = np.mean(overall_f1_agree)
overall_f1_95 = scipy.stats.t.interval(0.95, len(overall_f1_agree) - 1, overall_f1_mean, np.std(overall_f1_agree))

#Overall agreement precision and recall
overall_precision_mean = np.mean([0.9234945649702095, 0.9355916747744139, 0.9329221189121321, 
                          0.9466966885225196, 0.9036691909064459, 0.8888883903851887, 
                          0.7676414046479836, 0.924076753577478, 0.9308534322820037, 
                          0.7497757177033493,0.9191451531289891, 0.9315395441122185, 
                          0.9234758122543865, 0.9178973531672014, 0.9195820546830648, 
                          0.9038976846495379, 0.9379018118059614])
overall_recall_mean = np.mean([0.9255014326647565, 0.9364161849710982, 0.9329608938547486, 
                       0.9439775910364145, 0.8918128654970761, 0.8783382789317508,
                       0.7828947368421053, 0.9228650137741047, 0.9314285714285714,
                       0.75, 0.9157303370786517, 0.9302325581395349, 0.9213483146067416,
                       0.9152542372881356, 0.9202279202279202, 0.9022988505747126, 0.9339080459770115])

#model accuracies
model_accs = [0.82, 0.8267, 0.8367, 0.8433, 0.8633]
model_mean = np.mean(model_accs)
model_95 = scipy.stats.t.interval(0.95, len(model_accs) - 1, model_mean, np.std(model_accs))

#model f1 scores, precision, and recall
model_f1s = np.sort([f1_score(all_pat_predictions[seed]['True_Label'], all_pat_predictions[seed]['prediction'], average='weighted') for seed in all_pat_predictions])
model_precisions = np.sort([precision_score(all_pat_predictions[seed]['True_Label'], all_pat_predictions[seed]['prediction'], average='weighted') for seed in all_pat_predictions])
model_recalls = np.sort([recall_score(all_pat_predictions[seed]['True_Label'], all_pat_predictions[seed]['prediction'], average='weighted') for seed in all_pat_predictions])

#plot the output
x = 1
width = 0.15
sep = width/4
x_pos = [x+i*width for i in range(-3,4)]
colors = ['#EDF8FB', '#d0d1e6', '#a6bddb', '#74a9cf', '#2b8cbe']
plt.figure(figsize=(8,6))
for i in range(len(model_f1s)):
    plt.bar(x_pos[i],   model_f1s[i], width=width, color=colors[i], edgecolor='black', capsize=3)
#ensemble performance
plt.bar(x_pos[5]+sep,   f1_ensemble, width = width, color='#045a8d', edgecolor='black')

#annotator agreement
plt.bar(x_pos[6]+sep, overall_mean, yerr = overall_mean-overall_95[0], width=width, color='#FEE090', edgecolor='black', capsize=3)

#set up the plot
plt.ylim([0.6, 1])
plt.xticks(ticks=x_pos[:-2] + [x_pos[-2] + sep, x_pos[-1] + sep],
           labels=['Model 1 F$_1$', 'Model 2 F$_1$', 'Model 3 F$_1$', 'Model 4  F$_1$', 'Model 5  F$_1$', 'Plurality Voting F$_1$', "Mean Annotator\nCohen's $\kappa$"],
           rotation = -45, ha='center')
plt.ylabel("F$_1$ or Cohen's $\kappa$")

#save the figures
save_path=f'{fig_save_dir}/Fig_5a'
plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{save_path}.pdf", dpi=600, bbox_inches='tight')