import mne
import numpy as np
from mne.channels import read_layout

ch_dic = np.loadtxt('src/ch_dic.csv',delimiter = ',', dtype=str)
ch_dic = dict(zip(ch_dic[0],ch_dic[1]))
event_id = {'M consonant': 1, 'Ye sound': 3, 'M + movement' : 7, 'break' : 2 }
event_color = {1:'peachpuff', 2: 'c', 3:'peachpuff', 7:'peachpuff'}
montage = mne.channels.read_montage('biosemi64')

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

def getEpochs(experiments,tasks):
    if(experiments >= 1):
        raw = mne.io.read_raw_bdf("data/e01.bdf", preload = True)
        raw.rename_channels(ch_dic)
        raw.set_montage(montage)
        events = mne.find_events(raw, shortest_event = 1)
        events = reduceEvents(events)
        picks = mne.pick_types(raw.info,eeg = True,stim =True ,exclude = 'bads')
        raw.filter(1., None, n_jobs=1, fir_design='firwin')
        # raw.notch_filter(np.arange(50, 241, 50), picks=picks, fir_design='firwin')
        ica = mne.preprocessing.ICA(method="fastica",random_state=23,n_components = 15)
        ica.fit(raw,decim=3)
        ica.plot_components(inst = raw)
        ica.exclude = [2]
        ica.apply(raw)
        # raw.filter(7, 30, n_jobs=1, fir_design='firwin')
        # epochs = mne.Epochs(raw, events, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None, preload = True)
        raw.filter(7.,30.,fir_design = 'firwin')
        epochs = mne.Epochs(raw, events, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None)
    if(experiments >= 2):
        raw_02 = mne.io.read_raw_bdf("data/e02.bdf")
        events2 = mne.find_events(raw_02, shortest_event = 1)
        events2 = reduceEvents(events2)
        epochs2 = mne.Epochs(raw_02, events2, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None)
        epochs = mne.concatenate_epochs([epochs,epochs2])
    if(experiments >= 3):
        raw_03 = mne.io.read_raw_bdf("data/e03.bdf")
        events3 = mne.find_events(raw_03, shortest_event = 1)
        events3 = reduceEvents(events3)
        epochs3 = mne.Epochs(raw_03, events3, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None)
        epochs = mne.concatenate_epochs([epochs,epochs3])

    # epochs.rename_channels(ch_dic)
    epochs.set_montage(montage)
    # epochs.filter(7.,30.,fir_design='firwin')

    if(tasks):
        return epochs[tasks]
    else :
        return epochs

def getRaw(experiments):
    raw = mne.io.read_raw_bdf("data/e01.bdf",preload = True)
    events = mne.find_events(raw, shortest_event = 1)
    events = reduceEvents(events)
    raw.rename_channels(ch_dic)
    raw.set_montage(montage)
    # raw.filter(7.,30., fir_design='firwin')
    return raw,events,event_id,event_color

def getAvarage(tasks):
    reject = dict(eeg = 80e-6)
    raw = mne.io.read_raw_bdf("data/e01.bdf",preload = True)
    raw.rename_channels(ch_dic)
    raw.set_montage(montage)
    picks = mne.pick_types(raw.info,eeg = True,stim =True ,exclude = 'bads')
    events = mne.find_events(raw, shortest_event = 1)
    events = reduceEvents(events)
    raw.filter(1., None, n_jobs=1, fir_design='firwin')
    # raw.notch_filter(np.arange(50, 241, 50), picks=picks, fir_design='firwin')
    # raw.filter(None, 150., fir_design='firwin')
    ica = mne.preprocessing.ICA(method="fastica",random_state=23,n_components = 15)
    ica.fit(raw,decim=3)
    # ica.plot_components(inst = raw)
    ica.exclude[2]
    ica.apply(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin = -0.1 ,tmax = 2,proj=True ,picks=picks, baseline = (None,0))

    print(epochs.__len__)
    # epochs.filter(7.,30.,fir_design='firwin')
    return epochs[tasks].average()
