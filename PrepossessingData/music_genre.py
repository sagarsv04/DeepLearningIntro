
import pandas as pd
import numpy as np
import os
import shutil
import librosa
pd.set_option('display.max_columns', 200) # Set number of columns to show in the notebook
pd.set_option('display.max_rows', 50) # Set number of rows to show in the notebook
pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
import matplotlib.pyplot as plt # Import MatPlotLib Package
%matplotlib inline # Display pictures within the notebook itself


# os.getcwd()
# os.chdir(r'D:\CodeRepo\DeepLearningIntro\PrepossessingData')


annotations_file_path = './annotations_final.csv'


def music_genre_classification():

    newdata = pd.read_csv(annotations_file_path, sep="\t")
    # newdata.head()
    # Get to know the data better
    # newdata.info()
    # Extract the clip_id and mp3_path
    # newdata[["clip_id", "mp3_path"]]
    # Previous command extracted it as a Dataframe. We need it as a matrix to do analyics on.
    # Extract clip_id and mp3_path as a matrix.
    clip_id, mp3_path = newdata[["clip_id", "mp3_path"]].as_matrix()[:,0], newdata[["clip_id", "mp3_path"]].as_matrix()[:,1]


    return 0


def main():




    return 0


if __name__ == '__main__':
    main()
