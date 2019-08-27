import mne
import numpy as np
from mne.channels import read_layout

ch_dic = np.loadtxt('ch_dic.csv',delimiter = ',', dtype=str)
ch_dic = dict(zip(ch_dic[0],ch_dic[1]))
event_id = {'M consonant': 1, 'Ye sound': 3, 'M + movement' : 7, 'break' : 2 }
event_color = {1:'peachpuff', 2: 'c', 3:'peachpuff', 7:'peachpuff'}
montage = mne.channels.read_montage('biosemi64')

# Filter the unknown events
#
#events from stim channel
#
def reduceEvents(events):
    events[:,2] += -(2**16-255)
    newEvents = []
    last = 0
    lastEvent = []

    for i in range(events.shape[0]):
        eventId = events[i,2]
        if eventId > 0:
            if eventId == 1:
                if last != 1:
                    newEvents.append([events[i,0],last,eventId])
            if eventId == 2 :
                if last ==1:
                    newEvents.append([events[i,0],last,eventId])
            if eventId == 3:
                if last != 3:
                    newEvents.append([events[i,0],last,eventId])
            if eventId == 4 :
                if last ==3:
                    newEvents.append([events[i,0],last,2])
            if eventId == 7:
                if last != 7:
                    newEvents.append([events[i,0],last,eventId])
            if eventId == 8 :
                if last ==7:
                    newEvents.append([events[i,0],last,2])
            last = eventId

    newEvents = np.stack(newEvents)
    return newEvents


# Return epochs object after apply bandpass (7,30)
#
# experiments - number of experiments (1,2,3)
# components - for csp
#
def getEpochs(experiments,tasks):
    if(experiments >= 1):
        raw = mne.io.read_raw_bdf("../data/e01.bdf")
        events = mne.find_events(raw, shortest_event = 1)
        events = reduceEvents(events)
        picks = mne.pick_types(raw.info,eeg = True,stim =True ,exclude = 'bads')
        epochs = mne.Epochs(raw, events, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None, preload = True)
    if(experiments >= 2):
        raw_02 = mne.io.read_raw_bdf("../data/e02.bdf")
        events2 = mne.find_events(raw_02, shortest_event = 1)
        events2 = reduceEvents(events2)
        epochs2 = mne.Epochs(raw_02, events2, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None)
        epochs = mne.concatenate_epochs([epochs,epochs2])
    if(experiments >= 3):
        raw_03 = mne.io.read_raw_bdf("../data/e03.bdf")
        events3 = mne.find_events(raw_03, shortest_event = 1)
        events3 = reduceEvents(events3)
        epochs3 = mne.Epochs(raw_03, events3, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None)
        epochs = mne.concatenate_epochs([epochs,epochs3])

    epochs.rename_channels(ch_dic)
    epochs.set_montage(montage)
    epochs.filter(7.,30.,fir_design='firwin')

    if(tasks):
        return epochs[tasks]
    else :
        return epochs

# Return epochs object after apply ICA and bandapass (1,60)
#
# experiments - number of experiments (1,2,3)
# components - for csp
#
def getEpochsICA(experiments,tasks):
    ica = mne.preprocessing.ICA(method="fastica",random_state=23)
    if(experiments >= 1):
        raw = mne.io.read_raw_bdf("../data/e01.bdf", preload = True)
        picks = mne.pick_types(raw.info,eeg = True,stim =True ,exclude = 'bads')
        events = mne.find_events(raw, shortest_event = 1)
        events = reduceEvents(events)
        raw.notch_filter(np.arange(50, 250, 50), picks=picks, fir_design='firwin')
        epochs = mne.Epochs(raw, events, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None)
    if(experiments >= 2):
        raw_02 = mne.io.read_raw_bdf("../data/e02.bdf", preload = True)
        picks = mne.pick_types(raw_02.info,eeg = True,stim =True ,exclude = 'bads')
        events2 = mne.find_events(raw_02, shortest_event = 1)
        events2 = reduceEvents(events2)
        raw_02.notch_filter(np.arange(50, 250, 50), picks=picks, fir_design='firwin')
        epochs2 = mne.Epochs(raw_02, events2, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None)
        epochs = mne.concatenate_epochs([epochs,epochs2])
    if(experiments >= 3):
        raw_03 = mne.io.read_raw_bdf("../data/e03.bdf", preload = True)
        picks = mne.pick_types(raw_03.info,eeg = True,stim =True ,exclude = 'bads')
        events3 = mne.find_events(raw_03, shortest_event = 1)
        events3 = reduceEvents(events3)
        raw_03.notch_filter(np.arange(50, 250, 50), picks=picks, fir_design='firwin')
        epochs3 = mne.Epochs(raw_03, events3, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None)
        epochs = mne.concatenate_epochs([epochs,epochs3])

    epochs.rename_channels(ch_dic)
    epochs.set_montage(montage)
    epochs.filter(1., 60, n_jobs=1, fir_design='firwin')
    ica.fit(epochs['M consonant','break'],decim=3)
    ica.plot_components()
    ica.exclude = [48]
    ica.apply(epochs)
    epochs.plot(n_epochs = 2, event_colors = event_color, scalings = {'eeg':75e-6}, block= True)
    if(tasks):
        return epochs[tasks]
    else :
        return epochs

# Return raw data from first exeriment record
#
#
def getRaw():
    raw = mne.io.read_raw_bdf("../data/e01.bdf",preload = True)
    picks = mne.pick_types(raw.info,eeg = True,stim =True ,exclude = 'bads')
    events = mne.find_events(raw, shortest_event = 1)
    events = reduceEvents(events)
    raw.rename_channels(ch_dic)
    raw.set_montage(montage)
    raw.notch_filter(np.arange(50, 250, 50), picks=picks, fir_design='firwin')
    return raw,events,event_id,event_color

# Plots ICA compoents from frist recrods
#
#
def plotICA():
    raw = mne.io.read_raw_bdf("../data/e01.bdf",preload = True)
    events = mne.find_events(raw, shortest_event = 1)
    events = reduceEvents(events)
    raw.rename_channels(ch_dic)
    raw.set_montage(montage)
    picks = mne.pick_types(raw.info,eeg = True,stim =True ,exclude = 'bads')
    raw.plot_psd(tmin=93, tmax=98, fmax= 200, picks = picks)
    raw.notch_filter(np.arange(50, 250, 50), picks=picks, fir_design='firwin')
    raw.filter(1.,60., fir_design='firwin')
    ica = mne.preprocessing.ICA(method="fastica",random_state=23,n_components = 35)
    ica.fit(raw,decim=3)
    ica.plot_components(inst = raw)
    ica.exclude = [2]
    ica.apply(raw)
    raw.plot_psd(tmin=93, tmax=98, fmax= 200, picks = picks)
