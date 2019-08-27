from preprocessing import getEpochs
from preprocessing import getRaw
from preprocessing import getAvarage
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne.decoding import CSP
import numpy as np


def classify(experiments, components, task):
    epochs = getEpochs(experiments,task)
    print(epochs.__len__)
    cv = ShuffleSplit(10,test_size=0.2, random_state =42)
    epochs_data = epochs.get_data()
    epochs.average().plot(spatial_colors=True,time_unit='s')
    y = epochs.events[:,2]
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=components, reg=None,log=False ,norm_trace = False)
    clf = Pipeline([('CSP',csp),('LDA',lda)])
    scores = cross_val_score(clf,epochs_data,y, cv=cv, n_jobs=1)
    print('mean accuracy tasks :', task  ,' ',np.mean(scores))

def plotCSP(experiments,components,task):
    epochs = getEpochs(experiments,task)
    y = epochs.events[:,2]
    epochs_data = epochs.get_data()
    csp = CSP(n_components=components, reg=None,log=False ,norm_trace = False)
    csp.fit_transform(epochs_data, y)
    csp.plot_patterns(epochs.info,ch_type='eeg', units = 'Patterns (AU)', size = 1.5)

def visualizeRaw():
    raw,events,event_id,event_color = getRaw(1)
    raw.plot( duration = 10,events = events, event_id = event_id, event_color = event_color, scalings = {'eeg':75e-6}, block= True)

def visualizeEpochs(tasks):
    epochs = getEpochs(1,tasks)
    # epochs = getAvarage(tasks)
    # epochs.plot(spatial_colors=True,time_unit='s')
    epochs.plot(n_epochs = 2, event_colors = event_color, scalings = {'eeg':75e-6}, block= True)

# visualizeEpochs([])
plotCSP(1,10,['M consonant','Ye sound','break'])
# visualizeRaw()

# visualizeEpochs(['M consonant'])
