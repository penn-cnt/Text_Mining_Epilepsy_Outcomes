import pickle
import numpy as np
import json
import pandas as pd
import string
import timeline_utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from random import choices
sns.set_theme(style='ticks')
import scipy.stats

#get the quantified data
with open(r'all_patients_and_visits_outcome_measures_py38.pkl', 'rb') as f:
    datapull = pickle.load(f)
    
#create aggregate patients
all_agg_pats = np.array(timeline_utils.aggregate_patients_and_visits(datapull['all_pats']))

#How many bootstraps do we want?
n_boots = 10000

def use_symbol_idx(df):
    df.index = [idx.split('\n')[0] for idx in df.index]
    return df

#for each bootstrap calculate new transition probabilities
all_chains = [use_symbol_idx(timeline_utils.generate_visit_markovian_table(boot_pats, no_plot=True)) for boot_pats in [choices(all_agg_pats, k = len(all_agg_pats)) for i in range(n_boots)]]

#get the distributions of the bootstrapped chains
distributions = {k:{} for k in all_chains[0].index}
for seq in distributions:
    distributions[seq]['arr'] = np.array([chain.loc[seq, r'$\bigcirc$'] for chain in all_chains])
    distributions[seq]['mean'] = np.mean(distributions[seq]['arr'])
    distributions[seq]['95_CI'] = scipy.stats.norm.interval(alpha=0.95, loc=distributions[seq]['mean'], scale=np.std(distributions[seq]['arr']))
    distributions[seq]['PlusMin'] = np.array(distributions[seq]['95_CI']) - distributions[seq]['mean']

#calculate the bootstrapped final transition table
bootstrapped_chain = {k:{} for k in all_chains[0].index}
heatmap_labels = {k:{} for k in all_chains[0].index}
for seq in bootstrapped_chain:
    bootstrapped_chain[seq][r'$\bigcirc$'] = distributions[seq]['mean']
    bootstrapped_chain[seq][r'$\blacksquare$'] = (1-distributions[seq]['mean'])
    heatmap_labels[seq][r'$\bigcirc$'] = f"{int((distributions[seq]['mean']*100).round(decimals=0))}%\n({int((distributions[seq]['95_CI'][0]*100).round())} - {int((distributions[seq]['95_CI'][1]*100).round())}%)"
    heatmap_labels[seq][r'$\blacksquare$'] = f"{int(((1-distributions[seq]['mean'])*100).round(decimals=0))}%\n({int((100*(1-distributions[seq]['95_CI'][1])).round())} - {int((100*(1-distributions[seq]['95_CI'][0])).round())}%)"
    
bootstrapped_chain = pd.DataFrame.from_dict(bootstrapped_chain).transpose()
heatmap_labels = pd.DataFrame.from_dict(heatmap_labels).transpose()

#plot the table
fig = plt.figure(figsize = (6,15))
seq_to_sym={'0':r"$\bigcirc$", '1':r"$\blacksquare$"}
ax = sns.heatmap(bootstrapped_chain, annot=heatmap_labels, vmin=0, vmax=1, linewidth=0.25, linecolor='#303030', fmt="",
                 cbar_kws={'label': 'Probability'}, cmap=sns.color_palette("Blues", as_cmap=True))
ax.xaxis.tick_top()
ax.set_xlabel(f"Next Visit Classification\n{seq_to_sym['0']} = Seizure Free\n{seq_to_sym['1']} = Has Seizures\n")
ax.xaxis.set_label_position('top') 
plt.ylabel('(Ordered) Previous Visit Classifications')
plt.yticks(rotation=0)
plt.title(f"Mean and 95% Confidence Interval of Probability of Having\nSeizures or Being Seizure Free Given a Patient's Previous 3 Visits\nUsing Non-Parametric Bootstrapping (10,000 Iterations)\n\n")
plt.show()