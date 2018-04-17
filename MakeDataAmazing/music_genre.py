
import pandas as pd
import numpy as np
import os
import shutil
import librosa
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\MakeDataAmazing')

# to download the data use
# http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset  >> Tag annotations

annotations_file_path = './annotations_final.csv'


# some of the tags in the dataset are really close to each other. Lets merge them together
synonyms = [['beat', 'beats'], ['chant', 'chanting'], ['choir', 'choral'], ['classical', 'clasical', 'classic'], ['drum', 'drums'],
            ['electro', 'electronic', 'electronica', 'electric'], ['fast', 'fast beat', 'quick'],
            ['female', 'female singer', 'female singing', 'female vocals', 'female vocal', 'female voice', 'woman', 'woman singing', 'women'],
            ['flute', 'flutes'], ['guitar', 'guitars'], ['hard', 'hard rock'], ['harpsichord', 'harpsicord'], ['heavy', 'heavy metal', 'metal'],
            ['horn', 'horns'], ['india', 'indian'], ['jazz', 'jazzy'], ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
            ['no beat', 'no drums'], ['no singer', 'no singing', 'no vocal','no vocals', 'no voice', 'no voices', 'instrumental'],
            ['opera', 'operatic'], ['orchestra', 'orchestral'], ['quiet', 'silence'], ['singer', 'singing'], ['space', 'spacey'], ['string', 'strings'],
            ['synth', 'synthesizer'], ['violin', 'violins'], ['vocal', 'vocals', 'voice', 'voices'], ['strange', 'weird']]


def merge_synonyms(newdata):
    # Merge the synonyms and drop all other columns than the first one.
    """
    Example: Merge 'beat', 'beats' and save it to 'beat'.
             Merge 'classical', 'clasical', 'classic' and save it to 'classical'.
    """

    for synonym_list in synonyms:
        # synonym_list = synonyms[0]
        newdata[synonym_list[0]] = newdata[synonym_list].max(axis=1)
        newdata.drop(synonym_list[1:], axis=1, inplace=True)

    return newdata


def shuffle_data(newdata):
    # Shuffle the dataframe
    newdata = shuffle(newdata)
    newdata.reset_index(drop=True, inplace=True)
    return newdata


def music_genre_classification():

    newdata = pd.read_csv(annotations_file_path, sep="\t")
    # newdata.head()
    # newdata.info() # Get to know the data better
    # Extract clip_id and mp3_path as a matrix.
    clip_id, mp3_path = newdata[["clip_id", "mp3_path"]].as_matrix()[:,0], newdata[["clip_id", "mp3_path"]].as_matrix()[:,1]

    newdata = merge_synonyms(newdata)
    newdata.drop('mp3_path', axis=1, inplace=True)
    # Save the column names into a variable
    data = newdata.sum(axis=0)
    # Sort the column names
    data.sort_values(axis=0, inplace=True)
    # Find the top tags from the dataframe
    # take top 50 values form 135 dataset
    topindex, topvalues = list(data.index[85:]), data.values[85:]
    del(topindex[-1])
    topvalues = np.delete(topvalues, -1)
    # Get a list of columns to remove
    rem_cols = data.index[:85]
    newdata.drop(rem_cols, axis=1, inplace=True)
    # Use this to revive the dataframe newdata = backup_newdata
    backup_newdata = newdata
    newdata = shuffle_data(newdata)
    final_columns_names = list(newdata.columns)
    # Do it only once to delete the clip_id column
    del(final_columns_names[0])
    # Here, binary 0's and 1's from each column is changed to 'False' and 'True' by using '==' operator on the dataframe.
    final_matrix = pd.concat([newdata['clip_id'], newdata[final_columns_names]==1], axis=1)



    return 0


def main():




    return 0


if __name__ == '__main__':
    main()
