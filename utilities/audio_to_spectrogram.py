"""
Reads a wav file and splits it ito 500 ms segments.
Stores the spectrograms and creates a file with spectrogram locations
Output spectrogram's filename: original filename + segment index + .spec
"""

from scipy.io import wavfile
from scipy import signal
import os
import numpy as np
import pandas as pd


source_dir = "/net/projects/scratch/winter/valid_until_31_July_2021/0-animal-communication/data_grid/Chimp_IvoryCoast/manually_verified_2s/chimp_only_23112020"
out_dir = "/net/projects/scratch/winter/valid_until_31_July_2021/sbiswas/Data/Spectrograms"
out_file = "/net/projects/scratch/winter/valid_until_31_July_2021/sbiswas/Data/spectrogramList.csv"
data_list = []

source_files = os.listdir(source_dir)
print('Total no. of wav files: {}\n'.format(len(source_files)))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for fname in source_files:
    try:
        srate, sdata = wavfile.read(os.path.join(source_dir, fname))
        clip_len = len(sdata)
        if clip_len/srate >= 0.5:
            num_500ms_clips = int((len(sdata) - (srate * 0.5)) / (srate * 0.2)) + 1
            for i in range(num_500ms_clips):
                this_piece = sdata[int(srate * (0.2 * i)):int(srate * (0.2 * i + 0.5))]
                if srate != 44100:
                    this_piece = signal.resample(this_piece, int(len(this_piece) * 44100 / srate))
                _, _, spectrogram = signal.spectrogram(
                                                        x=this_piece,
                                                        fs=srate,
                                                        nfft=512,
                                                        noverlap=427,
                                                        detrend=False,
                                                        window=signal.get_window('hamming', 512)
                                                        )
                spectrogram = np.log(spectrogram + 1e-9)
                save_location = out_dir + '/' + fname.split('.')[0] + '_' + str(i) + '.spec'
                with open(save_location, 'wb') as f_save:
                    np.save(f_save, spectrogram)
                    data_list.append({'spectrogram': save_location})
        else:
            print("File:{} Error: Clip shorter than 500ms".format(fname))
    except Exception as e:
        print("File:{} Error: {}".format(fname, e))

pd.DataFrame(data_list, columns=['spectrogram']).to_csv(out_file, index=False, header=True)

