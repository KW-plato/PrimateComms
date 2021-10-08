"""
audio_to_periodogram.py
Author: Saurabh Biswas
Institution: University of Osnabrueck

Creates Hilbert Periodogram of the audio clips.
a. Takes a user defined frequency range which has to be captured and split into bins
b. Creates an empty placeholder of (bins, data samples/time points)
each one of the bin frequencies will be the mean of one of the gaussian filters. For each bin does the following steps
1. Fourier transform the original signal
2. Create gaussian filters in frequency domain and multiply
3. Invert to get back to time domain
4. Hilbert transform
5. Log of the power spectra of hilbert transform
5. Save to the corresponding bin in the placeholder created in step b

Input: csv files with location and label of the audio clips
Output: 224 x 224 size jpeg images placed in the output folder. Each label gets its own subfolder.

How to run? Change the following in the code and then submit as usual:
source: path to the csv list of audio files and their label
output: path to the folder where the periodograms are to be placed.
target_call_types : list of call labels of interest. Only audios with these labels will be converted.

Note: program checks if a periodogram corresponding to the audio file is already present
in the provided folder. If yes, that audio isnt converted.

"""
import gc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import fft, signal
from scipy.io import wavfile
matplotlib.use('Agg')


if __name__ == "__main__":

    source = "/net/projects/scratch/summer/valid_until_31_January_2022/sbiswas/Data/clips_4_SPECGRAM_1label.csv"
    output = "/net/projects/scratch/summer/valid_until_31_January_2022/sbiswas/Data/PGRAM"
    target_call_types = ['sm','ph','noise'] # we trained with only these labels

    source_files = pd.read_csv(source)
    # the params for the bins and gaussian filter
    minfreq = 0
    maxfreq = 6500  # determined from the observed data
    numfrex = 900   # determined by experimenting with multiple frequency bin sizes
    gauss_filter_width = 6
    s = gauss_filter_width * (2 * np.pi - 1) / (4 * np.pi)
    pgram_freq_range = np.linspace(minfreq, maxfreq, numfrex)

    for label in target_call_types:
        out_dir = os.path.join(output, label)
        try:
            os.makedirs(out_dir)
        except Exception as e:
            print("Error creating output dir {}".format(e))

        select_files = source_files[source_files["label"] == label]
        select_files = select_files['filename']

        for fn in select_files.tolist():
            pgram_file = os.path.join(out_dir, fn.split(os.sep)[-1].rstrip('.wav') + '.jpg')
            if not os.path.exists(pgram_file):
                try:
                    sampling_rate, input_clip = wavfile.read(fn)
                    if sampling_rate != 44100:  # sampling frequency of all files are made same
                        input_clip = signal.resample(input_clip, int(len(input_clip) * 44100 / sampling_rate))
                        sampling_rate = 44100
                    clip_len = len(input_clip)

                    time_points = np.linspace(0, len(input_clip) / sampling_rate, clip_len)
                    fourier = fft.rfft(input_clip)  # rfft used since we have real valued functions here
                    fourier_freqs = fft.rfftfreq(clip_len, 1.0 / sampling_rate)
                    pgram = np.zeros((numfrex, clip_len))  # empty matrix to hold the periodogram
                    for i in range(numfrex):
                        x = fourier_freqs - pgram_freq_range[i]
                        fx = np.exp(-.5 * ((x / s) ** 2))
                        fx = fx / np.abs(max(fx))
                        filtered_sig = np.real(fft.irfft(fourier * fx))
                        hbert = abs(signal.hilbert(filtered_sig)) ** 2
                        pgram[i, :] = 10 * np.log10(hbert)

                    fig, ax = plt.subplots(1, 1, figsize=(2.9, 2.92))  # to get a 224 by 224 image
                    ax.axis('off')
                    p = ax.pcolormesh(time_points, pgram_freq_range, pgram, cmap=plt.cm.cubehelix, shading='auto')
                    fig.patch.set_visible(False)
                    plt.savefig(pgram_file, bbox_inches='tight', pad_inches=0)

                    # steps below are for memory use management
                    ax.cla()
                    fig.clear()
                    plt.close("all")
                    del pgram, p
                    gc.collect()
                except Exception as e:
                    print("File:{}, err: {}".format(fn.split('/')[-1], e))


