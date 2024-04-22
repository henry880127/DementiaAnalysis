import numpy as np
import copy
from scipy.signal import decimate, butter, lfilter

class Data_Preprocess_EEGNet:
    def __init__(self, data, info, **kwargs):
        self.data = data
        self.info = info
        

        # use **kwargs to set the new value of below args.
        self.sec_Window = 3
        self.sec_Overlap = 0
        self.Fs = 500
        self.band = [2,40] # band pass to "band" Hz
        self.target_Freq = 125 # downsample to "target_freq" Hz
        self.conduct_filter = True  # Whether to band-pass filter
        self.conduct_decimate = True  # Whether to downsample
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
            
    
    def data_preprocessing(self, data, info, sec_Window, nOverlap):
        # Data segmentaion
        # Note that: Data format: subject x channel x datapoint	
        #            Input of EEGNet: subject x 1 x channel x datapoint as an epoch
        # Output : subject x epochs x 1 x channel x datapoint

        def butter_bandpass(lowcut, highcut, fs, order=2):
            return butter(order, [lowcut, highcut], fs=fs, btype='band')
    
        def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data, axis=1)
            return y
            
        def epochs(signal, sec_Window, sec_Overlap):
            q = 1
            # band-pass filter
            if self.conduct_filter: signal = butter_bandpass_filter(signal, self.band[0], self.band[1], self.Fs, order=5)
            # downsample
            if self.conduct_decimate: 
                q = int(self.Fs/self.target_Freq)
                signal = decimate(signal, q, axis=1)
                nWindow = sec_Window * self.target_Freq
                nOverlap = sec_Overlap * self.target_Freq
            else:
                nWindow = sec_Window * self.Fs
                nOverlap = sec_Overlap * self.Fs
            
            [nRow, nCol] = signal.shape
            pt_start = 0
            pt_end = nWindow
            output=list()
            while pt_end <= nCol:
                if nWindow <= nOverlap:
                    print('Invalid Input: nWindow == nOverlap')
                    break
                output.append(signal[:,pt_start:pt_end])
                pt_start += (nWindow - nOverlap)
                pt_end += (nWindow - nOverlap)
            output = np.array(output)
            # print('(epoch,channel,datapoint):',output.shape)
            return output
        
        def convert_to_input(signal_3d, sec_Window, sec_Overlap):
            [nSubject, nRow, nCol] = signal_3d.shape
            output = list()
            for i in range(nSubject):
                output.append(epochs(signal_3d[i,:,:], sec_Window, sec_Overlap))
            output = np.array(output)
            output = output[:,:,np.newaxis,:,:]
            print('(subject,epoch,1,channel,datapoint):',output.shape)
            return output
        
        data_ndarrays = {}
        for key, value in data.items():
            # Detect if the key is 'Info' and skip it
            if key != 'Info':
                data_ndarrays[key] = convert_to_input(value, self.sec_Window, self.sec_Overlap)
            else:
                data_ndarrays[key] = value

        # Label organizing
        label_ndarrays = {}
        label_ndarrays_CInonCI = {}
        for key, value in info.items():
            if key != 'Info':
                label_ndarrays[key] = value['Label'].values
                label_ndarrays_CInonCI[key] = value['Label'].values
            else:
                label_ndarrays[key] = value
                label_ndarrays_CInonCI[key] = value

        # Replace 2 with 1 in label_ndarrays
        label_ndarrays_CInonCI = {key: np.where(value == 2, 1, value) for key, value in label_ndarrays_CInonCI.items()}

        
        return data_ndarrays, label_ndarrays, label_ndarrays_CInonCI

    def epoch_based_organizing(self, z_score=False):
        [data_ndarrays, label_ndarrays, label_ndarrays_CInonCI] = self.data_preprocessing(self.data, self.info, self.sec_Window, self.sec_Overlap)
        
        # In python, assigning a dict() to variable is just creating a "reference".
        # Use copy.deepcopy to prevent this situation.
        self.data_ndarrays_S = copy.deepcopy(data_ndarrays)
        self.label_ndarrays_CInonCI_S = copy.deepcopy(label_ndarrays_CInonCI)

        # epoch based Conversion: (subject,epoch,1,channel,datapoint) -> (subject*epoch,1,channel,datapoint)
        for key, value in data_ndarrays.items():
            [nSubject,nEpoch,one,nChannel,nDatapoint] = value.shape
            new_shape = (nSubject*nEpoch, 1, nChannel, nDatapoint)
            data_ndarrays[key] = value.reshape(new_shape)
            print(f'{key}: (subject*epoch,1,channel,datapoint):',data_ndarrays[key].shape)
            if z_score==True:
                print(f'z-score normalization of {key}')
                data_ndarrays[key] = self.scaling(data_ndarrays[key])
                print(f'{key}: (subject*epoch,1,channel,datapoint):',data_ndarrays[key].shape)
            else:   pass

            # Repeate the label based on the nEpoch, i.e. [1,0,1] with nEpoch=3 -> [1,1,1,0,0,0,1,1,1] 
            label_ndarrays[key] = np.repeat(label_ndarrays[key], nEpoch)
            label_ndarrays_CInonCI[key] = np.repeat(label_ndarrays_CInonCI[key], nEpoch)
            print(f'epoch based labels of {key}:',label_ndarrays[key].shape)

        self.data_ndarrays_E = data_ndarrays
        self.label_ndarrays_CInonCI_E = label_ndarrays_CInonCI
        return data_ndarrays, label_ndarrays, label_ndarrays_CInonCI

    def scaling(datx):
        ndatx = []
        for dx in datx:
            mean, std = np.average(dx, axis=1), np.std(dx, axis=1)
            ndatx.append((dx-mean)/std)
        return np.array(ndatx)

    
