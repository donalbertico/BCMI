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
    print(events[:,2])
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
        raw = mne.io.read_raw_bdf("data/e01.bdf")
        events = mne.find_events(raw, shortest_event = 1)
        events = reduceEvents(events)
        picks = mne.pick_types(raw.info,eeg = True,stim =True ,exclude = 'bads')
        epochs = mne.Epochs(raw, events, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None, preload = True)
        # epochs = mne.Epochs(raw, events, event_id, tmin = 0.1, tmax = 5, proj=True, picks=picks, baseline = None)
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

    epochs.rename_channels(ch_dic)
    epochs.set_montage(montage)
    epochs.filter(7.,30.,fir_design='firwin')

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
    raw.filter(7.,30., fir_design='firwin')
    return raw,events,event_id,event_color
#
# raw = mne.io.read_raw_bdf("data/e01.bdf",preload= True)
#
#
# events = mne.find_events(raw ,shortest_event = 1)
#
# events[:,2] += -(2**16-255)
# newEvents = []
# last = 0
# tmin,tmax = 0,60
# fmin,fmax = 2,300
# n_fft = 2048
#
#
# for i in range(events.shape[0]):
#     eventId = events[i,2]
#     if eventId > 0:
#         if eventId == 1:
#             if last == 0 or last == 128:
#                 newEvents.append(events[i,:])
#                 last = eventId
#         if eventId == 2 :
#             if last ==1:
#                 newEvents.append(events[i,:])
#                 last = eventId
#         if eventId == 128 :
#             if last == 2 or last == 4 or last == 8:
#                 newEvents.append(events[i,:])
#                 last = eventId
#         if eventId == 3:
#             if last == 128:
#                 newEvents.append(events[i,:])
#                 last = eventId
#         if eventId == 4 :
#             if last ==3:
#                 newEvents.append(events[i,:])
#                 last = eventId
#         if eventId == 7:
#             if last == 128:
#                 newEvents.append(events[i,:])
#                 last = eventId
#         if eventId == 8 :
#             if last ==7:
#                 newEvents.append(events[i,:])
#                 last = eventId
# newEvents = np.stack(newEvents)
# label  = []
#
# last = 0
# for i in range(newEvents.shape[0]):
#     eventId = newEvents[i,2]
#     if eventId == 1:
#         label.append([newEvents[i,0],last,eventId])
#         last = eventId
#     if eventId == 2:
#         label.append([newEvents[i,0],last,eventId])
#         last = eventId
#     if eventId == 3:
#         label.append([newEvents[i,0],last,eventId])
#         last = eventId
#     if eventId == 4:
#         label.append([newEvents[i,0],last,2])
#         last = eventId
#     if eventId == 7:
#         label.append([newEvents[i,0],last,eventId])
#         last = eventId
#     if eventId == 8:
#         label.append([newEvents[i,0],last,2])
#         last = eventId
#
# labels = np.stack(label)
# y = labels[:,2]
#
# montage = mne.channels.read_montage('biosemi64')
# raw.rename_channels(ch_dic)
# raw.set_montage(montage)
#
# picks = mne.pick_types(raw.info,eeg = True,stim = True, exclude = 'bads')
# picks_ns = mne.pick_types(raw.info,eeg = True, exclude = 'bads')
# raw.filter(7.,30., fir_design='firwin')
# epochs = mne.Epochs(raw, labels, event_id, tmin = 0.1, tmax = 0.2, proj=True, picks=picks_ns, baseline = None, preload = True)
#
#
# mEps = epochs['M consonant','Ye sound']
#
#
# subEpc = mEps.get_data()
# subLbl = mEps.events[:,2]
#
# cv = ShuffleSplit(10,test_size=0.2, random_state =42)
# epochs_data = epochs.get_data()
# # cv_split = cv.split(epochs_data)
# # cv_split = cv.split(subEpc)
# #
# # lda = LinearDiscriminantAnalysis()
# # svmClf = svm.SVC(gamma='scale')
# csp = CSP(n_components=10, reg=None,log=False ,norm_trace = False)
# #
# # clf = Pipeline([('CSP',csp),('LDA',lda)])
# #
# # # scores = cross_val_score(clf,epochs_data,y, cv=cv, n_jobs=1)
# # scores = cross_val_score(clf,subEpc,subLbl, cv=cv, n_jobs=1)
# # print('El score es :::',np.mean(scores))
#
# csp.fit_transform(epochs_data, y)
#
# csp.plot_patterns(epochs.info,ch_type='eeg', units = 'Patterns (AU)', size = 1.5)
# raw.plot( duration = 60,events = labels, event_id = event_id, event_color = event_color, scalings = {'eeg':75e-6}, block= True)
