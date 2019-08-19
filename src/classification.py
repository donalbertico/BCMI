from preprocessing import getEpochs
from preprocessing import getRaw
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
    y = epochs.events[:,2]
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=components, reg=None,log=False ,norm_trace = False)
    clf = Pipeline([('CSP',csp),('LDA',lda)])
    scores = cross_val_score(clf,epochs_data,y, cv=cv, n_jobs=1)
    print('mean accuracy tasks :', task  ,' ',np.mean(scores))

def visualize():
    raw = getRaw(1)
    raw.plot( duration = 60,events = labels, event_id = event_id, event_color = event_color, scalings = {'eeg':75e-6}, block= True)



classify(1,10,['M consonant','break'])
