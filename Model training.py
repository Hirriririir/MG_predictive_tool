"""# Import the training data"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#pd.set_option('display.max_rows', 100)
#pd.set_option('display.max_columns', 100)

DB_status = pd.read_excel("MG database derivation 2022.07.25.xlsx").drop(['Only_ID', 'Stage', 'Duration' ,'Predict_QMG_change', 'VC', 'Visting_date', 'Height', 'Weight', 'Heart_rate','Hipline', 'Waistline'], axis=1)

DB_status['Predict_status'].value_counts()

"""# Preliminary feature selection"""

plt.figure(figsize=(20, 20), dpi=300)
sns.clustermap(DB_status.corr(), cmap="RdBu", vmin=-1, yticklabels=True,  xticklabels=True)
#plt.savefig('Before_selection.svg', dpi=300, transparent=False)

DB_status = DB_status.drop(['Ab-AChR', 'Ab-MuSK','Ab-Negative', 'QOL: Be frustrated by MG','QOL: Having trouble using eyes','QOL: Having trouble eating','QOL: Limited social activity','QOL: Limited ability to enjoy hobbies','QOL: Meeting the needs of family','QOL: Making plans around MG','QOL: Negatively affected job status','QOL: Having difficulty speaking','QOL: Having trouble driving','QOL: Be depressed about MG','QOL: Having trouble walking','QOL: Having trouble getting around','QOL: Be overwhelmed by MG','QOL: Having trouble performing grooming'], axis=1)

DB_status['Predict_status'].value_counts()

plt.figure(figsize=(20, 20), dpi=300)
sns.clustermap(DB_status.corr(), cmap="RdBu", vmin=-1, yticklabels=True,  xticklabels=True)

"""# Import external dataset"""

DB_external = pd.read_excel("MG database validation 2022.07.22 修改后.xlsx").drop(['MG_Center', 'Patient_ID'], axis=1) #
DB_external['Predict_status'].value_counts()

DB_external.columns

MG_score_names = DB_status.columns.values.tolist()[8:33] #
MG_score_names

"""# Initiate a pycaret training pipline """

from pycaret.classification import *
exp_clf102 = setup(data = DB_status, target = 'Predict_status', session_id=223, test_data=DB_external, imputation_type='iterative',  iterative_imputation_iters = 5, #223
                   #categorical_iterative_imputer="et", numeric_iterative_imputer="et",
                  normalize = True, 
                  transformation = True,
                  fix_imbalance = True, 
                  feature_selection = True, feature_selection_threshold = 0.8, feature_selection_method = "classic", 
                  numeric_features = MG_score_names, 
                  ignore_features = ['Clinical_type', 'MGFA_type','ADL: Eyelid droop', 'ADL: Double vision', 'QMG: Vital capacity', 'ADL: Talking', 'ADL: Swallowing', 'QMG: Right arm outstretched', 'QMG: Right leg outstretched', 'QMG: Right hand grip','QMG: Left hand grip'],
                  use_gpu=True, silent = True)#,

"""# After secondary feature selection"""

After_selection = get_config('X_train')

plt.figure(figsize=(20, 20), dpi=300)
sns.clustermap(After_selection.corr(), cmap="RdBu", vmin=-1,  yticklabels=True,  xticklabels=True) 
#plt.savefig('After_selection.png', dpi=300, transparent=True)

from sklearn.metrics import roc_auc_score
add_metric('auc2', 'AUC_ovr', roc_auc_score, target='pred_proba', multi_class= 'ovr', average='weighted')
add_metric('auc3', 'AUC_ovo', roc_auc_score, target='pred_proba', multi_class= 'ovo', average='weighted')

best_model = compare_models(sort='F1', fold=10)

et = create_model('rf')
tuned_model = tune_model(et, optimize = 'F1', fold=10)

"""# Model performance evaluation"""

Evaluation1 = predict_model(tuned_model, DB_status, raw_score=True, encoded_labels=True) 
Evaluation2 = predict_model(tuned_model, DB_external, raw_score=True, encoded_labels=True)

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from math import sqrt

# AUC curve
def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper)

n_classes = 3
classes = ['Improved', 'Unchanged', 'Worse']

y_test1 = label_binarize(Evaluation1['Predict_status'], classes=['Improved', 'Unchanged', 'Worse']) #
y_score1 = Evaluation1[['Score_0', 'Score_1', 'Score_2']].to_numpy()


fpr1 = dict()
tpr1 = dict()
roc_auc1 = dict()
auc_CI1= dict()
lw=2

for i in range(n_classes):
    fpr1[i], tpr1[i], _ = roc_curve(y_test1[:, i], y_score1[:, i])
    roc_auc1[i] = auc(fpr1[i], tpr1[i])
    auc_CI1[i] = roc_auc_ci(y_test1[:, i], y_score1[:, i])


y_test2 = label_binarize(Evaluation2['Predict_status'], classes=['Improved', 'Unchanged', 'Worse']) #
y_score2 = Evaluation2[['Score_0', 'Score_1', 'Score_2']].to_numpy()


fpr2 = dict()
tpr2 = dict()
auc_CI2 = dict()
roc_auc2 = dict()
lw=2

for i in range(n_classes):
    fpr2[i], tpr2[i], _ = roc_curve(y_test2[:, i], y_score2[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])
    auc_CI2[i] = roc_auc_ci(y_test2[:, i], y_score2[:, i])

colors = cycle(['#377eb8', '#4daf4a', '#e4211c'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for i, color in zip(range(n_classes), colors):
    ax1.plot(fpr1[i], tpr1[i], color=color, lw=2,
             label= f' {classes[i]} (AUC = {roc_auc1[i]:0.2f}, 95%CI: {auc_CI1[i][0]:0.2f}-{auc_CI1[i][1]:0.2f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=lw)
ax1.set(title='ROC curve on the training dataset', xlabel='False Positive Rate', ylabel='True Positive Rate', xlim=[-0.05, 1.0], ylim=[0.0, 1.05])
ax1.legend(loc="lower right")

for i, color in zip(range(n_classes), colors):
    ax2.plot(fpr2[i], tpr2[i], color=color, lw=2,
             label= f' {classes[i]} (AUC = {roc_auc2[i]:0.2f}, 95%CI: {auc_CI2[i][0]:0.2f}-{auc_CI2[i][1]:0.2f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=lw)
ax2.set(title='ROC curve on the external dataset', xlabel='False Positive Rate', ylabel='True Positive Rate', xlim=[-0.05, 1.0], ylim=[0.0, 1.05])
ax2.legend(loc="lower right")

plt.savefig('AUC.svg', dpi=300, transparent=True)

# Calibration_curve
from sklearn.calibration import calibration_curve

fraction_of_positives1 = dict()
mean_predicted_value1 = dict()
fraction_of_positives2 = dict()
mean_predicted_value2 = dict()

for i in range(3):
    fraction_of_positives1[i], mean_predicted_value1[i] = calibration_curve(y_test1[:, i],  y_score1[:, i], n_bins=15)
    fraction_of_positives2[i], mean_predicted_value2[i] = calibration_curve(y_test2[:, i],  y_score2[:, i], n_bins=15)

# Plot 
colors = cycle(['#377eb8', '#4daf4a', '#e4211c'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1
for i, color in zip(range(3), colors):
    ax1.plot(mean_predicted_value1[i], fraction_of_positives1[i], 's-', color=color, lw=2,
             label= f' {classes[i]}')

ax1.plot([0, 1], [0, 1], '--', color='gray')
ax1.set(title='Calibration Curve on the training dataset', xlabel='Predicted outcome', ylabel='Actual outcome', xlim=[0.0, 1.0], ylim=[0.0, 1.0])
ax1.legend(loc="lower right")

# Plot 2
for i, color in zip(range(3), colors):
    ax2.plot(mean_predicted_value2[i], fraction_of_positives2[i], 's-', color=color, lw=2,
             label= f' {classes[i]}' )

ax2.plot([0, 1], [0, 1], '--', color='gray')
ax2.set(title='Calibration Curve on the external dataset', xlabel='Predicted outcome', ylabel='Actual outcome', xlim=[0.0, 1.0], ylim=[0.0, 1.0])
ax2.legend(loc="lower right")

#save plots
plt.savefig('Calibration.svg', dpi=300, transparent=True)

Evaluation3 = predict_model(tuned_model, DB_status, raw_score=True, encoded_labels=True) 
Evaluation4 = predict_model(tuned_model, DB_external, raw_score=True, encoded_labels=True) 

Evaluation_combined = pd.concat([Evaluation3,Evaluation4])

Evaluation_AChR = Evaluation_combined[Evaluation_combined['Antibody'] == 'AChR']
Evaluation_MuSK = Evaluation_combined[Evaluation_combined['Antibody'] == 'MuSK']
Evaluation_Negative = Evaluation_combined[Evaluation_combined['Antibody'] == 'Negative']
Evaluation_Ocular = Evaluation_combined[Evaluation_combined['Clinical_type'] == 'ocular']
Evaluation_generalized = Evaluation_combined[Evaluation_combined['Clinical_type'] == 'generalized']

def roc_subtype(Evaluation):

    auc_CI = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    y_test = label_binarize(Evaluation['Predict_status'], classes=['Improved', 'Unchanged', 'Worse']) #
    y_score = Evaluation[['Score_0', 'Score_1', 'Score_2']].to_numpy()

    # micro-average ROC 
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    auc_CI= roc_auc_ci(y_test.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    

    return (fpr, tpr, roc_auc, auc_CI)

roc_subtype(Evaluation_AChR)[3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


ax1.plot(roc_subtype(Evaluation_AChR)[0], roc_subtype(Evaluation_AChR)[1], color='#377eb8', lw= 2,
             label= f'AChR+ MG (AUC = {roc_subtype(Evaluation_AChR)[2]:0.2f}, 95%CI: {roc_subtype(Evaluation_AChR)[3][0]:0.2f}-{roc_subtype(Evaluation_AChR)[3][1]:0.2f})')
ax1.plot(roc_subtype(Evaluation_MuSK)[0], roc_subtype(Evaluation_MuSK)[1], color='#4daf4a', lw= 2,
             label= f'MuSK+ MG (AUC = {roc_subtype(Evaluation_MuSK)[2]:0.2f}, 95%CI: {roc_subtype(Evaluation_MuSK)[3][0]:0.2f}-{roc_subtype(Evaluation_MuSK)[3][1]:0.2f})')
ax1.plot(roc_subtype(Evaluation_Negative)[0], roc_subtype(Evaluation_Negative)[1], color='#36098a', lw= 2,
             label= f'Ab-Negative MG (AUC = {roc_subtype(Evaluation_Negative)[2]:0.2f}, 95%CI: {roc_subtype(Evaluation_Negative)[3][0]:0.2f}-{roc_subtype(Evaluation_Negative)[3][1]:0.2f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=lw)
ax1.set(title='ROC curves', xlabel='False Positive Rate', ylabel='True Positive Rate', xlim=[-0.05, 1.0], ylim=[0.0, 1.05])
ax1.legend(loc="lower right")


ax2.plot(roc_subtype(Evaluation_Ocular)[0], roc_subtype(Evaluation_Ocular)[1], color='#377eb8', lw= 2,
             label= f'Ocular MG (AUC = {roc_subtype(Evaluation_Ocular)[2]:0.2f}, 95%CI: {roc_subtype(Evaluation_Ocular)[3][0]:0.2f}-{roc_subtype(Evaluation_Ocular)[3][1]:0.2f})')
ax2.plot(roc_subtype(Evaluation_generalized)[0], roc_subtype(Evaluation_generalized)[1], color='#4daf4a', lw= 2,
             label= f'Generalized MG (AUC = {roc_subtype(Evaluation_generalized)[2]:0.2f}, 95%CI: {roc_subtype(Evaluation_generalized)[3][0]:0.2f}-{roc_subtype(Evaluation_generalized)[3][1]:0.2f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=lw)
ax2.set(title='ROC curves', xlabel='False Positive Rate', ylabel='True Positive Rate', xlim=[-0.05, 1.0], ylim=[0.0, 1.05])
ax2.legend(loc="lower right")

plt.savefig('Subtype ROC.svg', dpi=300, transparent=True)

plot_model(tuned_model, plot = 'confusion_matrix')

plot_model(tuned_model, use_train_data = True, plot = 'confusion_matrix')

"""# Feature importance"""

len(X_train_transformed.columns)

import shap as shap
import matplotlib.colors

# Generate colormap through matplotlib
cmap =matplotlib.colors.ListedColormap(['#377eb8', '#e4211c', '#4daf4a'], name='from_list', N=None)


import matplotlib.pyplot as pl

class_names= ['Improved', 'Unchanged', 'Worse']

X_train_transformed= get_config('X_train')
X_test_transformed= get_config('X_test')
 
shap_values = shap.TreeExplainer(tuned_model).shap_values(X_train_transformed)
shap.summary_plot(shap_values, X_train_transformed, plot_type="bar", color=cmap, class_names = class_names, show=False,  max_display=40)

plt.savefig('feature importance.svg')

X_test_transformed

# Improved
shap.summary_plot(shap_values[0], X_train_transformed)

# Unchanged
shap.summary_plot(shap_values[1], X_train_transformed)

# worse
shap.summary_plot(shap_values[2], X_train_transformed)

"""# Save the final model"""

save_model(tuned_model, 'MG_predictive_multiclass')


"""# Patient case"""

X_train_transformed

# Load local model
tuned_model = load_model('MG_predictive_multiclass.pkl')

predict_model(tuned_model, DB_status,  raw_score=True).to_excel('MG ML 内部  raw_score=True.xlsx', encoding='utf_8_sig', index=False)

predict_model(tuned_model, DB_external,  raw_score=True).to_excel('MG ML 外部  raw_score=True.xlsx', encoding='utf_8_sig', index=False)

predict_model(tuned_model, DB_status, raw_score=True)

explainer = shap.TreeExplainer(tuned_model)
explainer.expected_value

# shap_values[A][B], in which A as the predicted classification (Improved: 0, Unchanged: 1, Worse: 2) and B as the patient number 
shap.initjs()
Force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][4], X_train_transformed.iloc[[4]]) 
Force_plot

Force_plot = shap.force_plot(explainer.expected_value[2], shap_values[2][41], X_train_transformed.iloc[[41]]) 
shap.save_html('Patient 2: worse.html', Force_plot) # 
Force_plot

Force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][41], X_train_transformed.iloc[[41]]) 
shap.save_html('Patient 2: Unchanged.html', Force_plot) # 
Force_plot

Force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0][41], X_train_transformed.iloc[[41]]) 
shap.save_html('Patient 2: Improved.html', Force_plot) # 
Force_plot

Force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0][31], X_train_transformed.iloc[[31]]) 
shap.save_html('Patient 3: Improved.html', Force_plot) #Stable
Force_plot