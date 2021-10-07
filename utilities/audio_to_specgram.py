"""
audio_to_specgram.py
Author: Saurabh Biswas
Institution: University of Osnabrueck

Creates Spectrogram of the audio clips.
a. Reads a wav file and splits it ito 500 ms segments with 200ms hop-length.
b. Creates spectrograms of the 500ms segments
b. Stores the spectrograms and creates a file with location and label of each spectrogram

Input:
1. Path to the csv file with location and label of the audio clips
2. (Optional) Output directory. If not provided a folder named SPECGRAM is created in the current directory

Output: Spectrogram saved as a binary .npy file

How to run? Examples are provided below:
audio_to_specgram.py myFolder/audio_list.csv
audio_to_specgram.py myFolder/audio_list.csv -o myOutputFolder
"""

import argparse
import numpy as np
import os
import pandas as pd
from scipy import signal
from scipy.io import wavfile


def parse_cli():
    """
    parses command line arguments
    """
    cmd_parse = argparse.ArgumentParser(
        usage= "%(prog)s [-o OUTPUT] Audio_clips.csv",
        description="Create Spectrograms for training\n e.g. python %(prog)s myFolder/audio_files.csv")
    cmd_parse.add_argument("Samples",
                           help="Location of the CSV file with location and label of each auyio file to be converted")

    cmd_parse.add_argument("-o", "--output",
                           help="Directory where outputs are placed. SPECGRAM folder created in current directory if not provided")

    args = cmd_parse.parse_args()
    if not os.path.isfile(args.Samples):
        print("Audio list file not found. Exiting...")
        exit()

    if args.output is None:
        curr_dir = os.getcwd()
        output_dir = os.path.join(curr_dir,"SPECGRAM")
        print("Outputs will be in folder {}".format(output_dir))
        try:
            os.makedirs(output_dir)
            print("Folder created!")
        except Exception as e:
            print("Cant create folder {} {}".format(output_dir, e))
            exit()
    else:
        if not os.path.isdir(args.output):
            print("Provided Output location either doesn't exist or isn't a directory.")
            exit()
        else:
            output_dir = args.output
            print("Outputs will be in folder {}".format(output_dir))

    return args.Samples, output_dir


if __name__ == "__main__":
    # parse command line inputs, get the input file and the output folder
    data_file, output_directory = parse_cli()

    out_file = os.path.join(output_directory,"SPECGRAM_list.csv")
    error_file = os.path.join(output_directory,"audio_to_specgram_errors.csv")

    try:
        source_files = pd.read_csv(data_file)
    except Exception as e:
        print("Error in reading input csv. ", e)
        exit()

    data_list = []
    error_files = []

    for label in source_files["label"].drop_duplicates().to_list():
        out_dir = os.path.join(output_directory, label)
        try:
            os.mkdir(out_dir)
        except Exception as e:
            print("Error creating directory {} {}".format(out_dir, e)) # to handle existing label folder
        select_files = source_files[source_files["label"] == label]
        select_files = select_files['filename']
        for fn in select_files.tolist():
            try:
                sampling_rate, input_clip = wavfile.read(fn)
                clip_len = len(input_clip)
                if clip_len / sampling_rate >= 0.5:
                    if sampling_rate != 44100:
                        input_clip = signal.resample(input_clip, int(clip_len * 44100 / sampling_rate))
                        sampling_rate = 44100
                    num_500ms_clips = int((len(input_clip) - (sampling_rate * 0.5)) / (sampling_rate * 0.2)) + 1
                    for i in range(num_500ms_clips):
                        clip = input_clip[int(sampling_rate * (0.2 * i)):int(sampling_rate * (0.2 * i + 0.5))]
                        _, _, spectrogram = signal.spectrogram(
                            x=clip,
                            fs=sampling_rate,
                            nfft=512,
                            noverlap=427,
                            detrend=False,
                            window=signal.get_window('hamming', 512)
                        )
                        spectrogram = np.log(spectrogram + 1e-9)
                        specgram_file = os.path.join(out_dir, fn.split('/')[-1].rstrip('.wav') + '_' + str(i) + '.spec')
                        with open(specgram_file, 'wb') as f_save:
                            np.save(f_save, spectrogram)
                            data_list.append({
                                'spectrogram': specgram_file,
                                'label': label
                            })
                else:
                    error_files.append({
                        'filename': fn,
                        'error': "Clip shorter than 500ms"
                    })
            except Exception as e:
                error_files.append({
                    'filename': fn,
                    'error': e
                })

    # saves list of the spectrograms as csv
    if data_list:
        pd.DataFrame(data_list, columns=['spectrogram', 'label']).to_csv(out_file, index=False, header=True)
        print("List of spectrograms saved in {}".format(out_file))

    # list of audio clips which ran into problems during spectrogram conversion
    if error_files:
        pd.DataFrame(error_files, columns=['filename', 'error']).to_csv(error_file, index=False, header=True)




