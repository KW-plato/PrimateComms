# PrimateComms
Overview: Supervised learning approach to classify chimpanzee call types from audio samples recorded in the wild.

The classifider is adapted from the model in the paper Oikarinen, T., Srinivasan, K. ,Meisner, O., et al. (2019) Deep convolutional network for animal sound
classification and source attribution using dual audio recordings. The Journal of the Acoustical Society of America
145, 654. doi: https://doi.org/10.1121/1.5087827.

The classifier perfomed with 98.3% accuracy on the dataset at disposal. 

To retrain and use the classifier on your data follow the steps below.

1. Collate all audio samples in a folder. The filename must contain the call-type label for the next steps in the pipeline to work. 
Currently call-type labels handled are "ab","pg","ph","ng","fg","tb","sm","gr" and "noise".

2. Inventorise the audio samples.

  Run: explorer.py -i <folder-with-audio-samples> -o <output-folder>
  
  The script saves the list of audio samples with one call label in the file output-folder/selected_audio_files.csv
  The list of compound calls are found in the file output-folder/compoundcall_audio_files.csv

  Note: The in our tests the classifier coundnt deal with compound calls as good as single calls. 
  Besides, in Chimpanzees the cmpound calls can be produced by rather flexible permutations of the single call types. So no effort was made to improve performance  for compound call classification.

3. Create Spectrograms of 500ms clips extracted from the audio samples.
  
  Run: audio_to_specgram.py output-folder/selected_audio_files.csv -o data-folder/SPECGRAM
  
  Script saves the list of the spectrograms in the provided output folder in a file named SPECGRAM_list.csv i.e. in data-folder/SPECGRAM_list.csv in the example above.

4. Train the classifier.
  
  Run: train_spectrogram.py data-folder/SPECGRAM_list.csv -o output-folder.

  The trained model and the metrics are found in the folder output-folder

The notebook "predictor" can be used to perform classification on unseen audio-samples.

Enjoy!


