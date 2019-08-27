from preprocessing import getEpochs
from preprocessing import getRaw
from preprocessing import getAvarage
from preprocessing import getEpochsICA
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne.decoding import CSP
import numpy as np

# Print score of certain arrangement
#
# Trains and apply 10 fold cross validation over the preprocessed signal
# experiments - number of experiments (1,2,3)
# components for csp
# tasks signal classes ['M consonant','break','Ye sound','M + movement']
# classifier - (lda,svm)
# ica - apply ica, Boolean
#
#
def score(experiments, components, tasks, classifier ,ica):
    if(ica):
        epochs = getEpochsICA(experiments,tasks)
    else:
        epochs = getEpochsICA(experiments,tasks)
    print(epochs.__len__)
    cv = ShuffleSplit(10,test_size=0.2, random_state =42)
    epochs_data = epochs.get_data()
    y = epochs.events[:,2]
    csp = CSP(n_components=components, reg=None,log=False ,norm_trace = False)
    if(classifier == 'lda'):
        lda = LinearDiscriminantAnalysis()
        clf = Pipeline([('CSP',csp),('LDA',lda)])
    else:
        svm = LinearDiscriminantAnalysis()
        clf = Pipeline([('CSP',csp),('SVM',lda)])
    scores = cross_val_score(clf,epochs_data,y, cv=cv, n_jobs=1)
    print('mean accuracy : ica ',ica ,',classifier', classifier,',tasks ', tasks,' ',np.mean(scores))

# Plots CSP components
#
# experiments - number of experiments (1,2,3)
# components - for csp
# tasks - signal classes ['M consonant','break','Ye sound','M + movement']
#
def plotCSP(experiments,components,task):
    epochs = getEpochs(experiments,task)
    y = epochs.events[:,2]
    epochs_data = epochs.get_data()
    csp = CSP(n_components=components, reg=None,log=False ,norm_trace = False)
    csp.fit_transform(epochs_data, y)
    csp.plot_patterns(epochs.info,ch_type='eeg', units = 'Patterns (AU)', size = 1.5)

# Plots raw data for first experiment record
#
#
def visualizeRaw():
    raw,events,event_id,event_color = getRaw()
    raw.plot( duration = 10,events = events, event_id = event_id, event_color = event_color, scalings = {'eeg':75e-6}, block= True)

# Plots epoch mne object with the events marked
#
# experiments - number of experiments (1,2,3)
# tasks - signal classes ['M consonant','break','Ye sound','M + movement']
#
def visualizeEpochs(experiments,tasks):
    event_color = {1:'peachpuff', 2: 'c', 3:'peachpuff', 7:'peachpuff'}
    epochs = getEpochs(1,tasks)
    epochs.plot(n_epochs = 2, event_colors = event_color, scalings = {'eeg':75e-6}, block= True)

# get Epochs object with ica applied
#
# experiments - number of experiments (1,2,3)
# tasks - signal classes ['M consonant','break','Ye sound','M + movement']
#
def icaEpochs(experiments,tasks):
    epochs = getEpochsICA(experiments,tasks)
    return epochs
