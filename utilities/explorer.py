"""
explorer.py
Author: Saurabh Biswas
Institution: University of Osnabrueck

Crawls through the audio samples and creates list of them.

Input:
1. Path to the directory containing the audio files
2. Path where to save the summary csv files

Output:
1. csv file with path to every audio clip and the call types found tagged in the clip name
2. csv file with path to every audio clip and the call label for non-compound calls

How to run? Examples are provided below:
explorer.py -i AudioFolder -o OutputFolder
"""

import argparse
import os
import pandas as pd
from scipy.io import wavfile

def parse_cli():
    """
    parses command line arguments
    """
    cmd_parse = argparse.ArgumentParser(
        usage="%(prog)s -i AudioFolder -o OutputFolder",
        description="Crawls through the audio samples and creates list of them.")
    cmd_parse.add_argument("-i", "--input",
                           required=True,
                           help="Directory where audio samples are stored")

    cmd_parse.add_argument("-o", "--output",
                           required=True,
                           help="Directory where outputs are placed")

    args = cmd_parse.parse_args()
    if not os.path.isdir(args.input):
        print("Provided Input location either doesn't exist or isn't a directory.")
        exit()

    if not os.path.isdir(args.output):
        print("Provided Output location either doesn't exist or isn't a directory.")
        exit()

    return args.input, args.output

if __name__ == "__main__":

    input_dir, out_dir =  parse_cli()

    call_categories = ["ab","pg","ph","ng","fg","tb","sm","gr","noise"]
    row_list = []

    for dirName, subdirName, fileNames in os.walk(input_dir): # get the files in the directory, subdirectory
        for fname in fileNames:
            if fname.endswith(".wav"):
                fullpath = os.path.join(dirName,fname)
                try:
                    fs, fholder = wavfile.read(fullpath)
                    f = fname.lower().replace("_", "-")
                    dict1 = {}
                    dict1['filename'] = fullpath
                    dict1['num_labels'] = 0
                    labels = []
                    for x in call_categories:
                        if x in f:
                            dict1['num_labels'] += 1
                            labels.append(x)
                    if dict1['num_labels'] != 0:
                        dict1['label'] = ','.join(labels)
                        dict1['fs'] = fs
                        dict1['duration'] = len(fholder) / fs
                    else:
                        dict1['label'] = 'NA'
                        dict1['fs'] = 0
                        dict1['duration'] = 0
                    row_list.append(dict1)
                except Exception as e:
                    print("{}: Error {}".format(fullpath, e))

    if row_list:
        data_f = pd.DataFrame(
            row_list,
            columns=["filename","fs","duration","num_labels","label"])
        data_f.to_csv(
            os.path.join(out_dir,"all_clips_list.csv"),  index=False, header=True)

        print("List of all wav files is in {}".format(os.path.join(out_dir,"all_clips_list.csv")))
        print('Total no. of audio files: {}'.format(data_f.shape[0]))
        print('Summary of all audio files \n', data_f.groupby(['num_labels'])['filename'].agg(['count']))
        compoundcall_mask = data_f.loc[:, 'num_labels'] > 1
        onecall_mask = data_f.loc[:, 'num_labels'] == 1

        df_temp = data_f.loc[onecall_mask, : ]
        if df_temp.shape[0] > 0:
            print("\nList of target wav files is in {}".format(os.path.join(out_dir, "selected_audio_files.csv")))
            df_temp.to_csv(os.path.join(out_dir, "selected_audio_files.csv"), index=False, header=True)
            print('Summary of one call type clips\n',
                  df_temp.groupby(['label']).agg({'filename':['count'],'duration':['sum','mean','std']}))

        df_temp = data_f.loc[compoundcall_mask, :]
        if df_temp.shape[0] > 0:
            df_temp.to_csv(os.path.join(out_dir, "compoundcall_audio_files.csv"),index=False, header=True)
            print('\nSummary of compound call clips\n',
                  df_temp.groupby(['label']).agg({'filename':['count'],'duration':['sum','mean','std']}))

