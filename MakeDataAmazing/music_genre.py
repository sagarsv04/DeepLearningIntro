
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import shutil
import librosa
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# os.getcwd()
# os.chdir(r'D:\CodeRepo\DeepLearningIntro\MakeDataAmazing')

# to download the data use
# http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset  >> Tag annotations

annotations_file_path = './annotations_final.csv'

src_path = './mp3_zip/'
dest_path = './dataset_mp3/'
npy_path = './dataset_npy/'


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


def check_file_exist(file_path):
    if os.path.exists(file_path):
        return True
    else:
        return False


def compute_melgram(audio_path):
    # audio_path = dest_path+audio_path
    # Audio preprocessing function
    # Convert all the mp3 files into their corresponding mel-spectrograms (melgrams).
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame
    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load
    '''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]
    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2,
                ref_power=1.0)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


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

    # Iterate over the mp3 files, rename them to the clip_id and save it to another folder.
    print("Renaming and moving all the audio files")
    for id in tqdm(range(len(mp3_path))):
        # id = 0
        src = src_path + mp3_path[id]
        dest = dest_path + str(clip_id[id]) + ".mp3"
        if check_file_exist(src):
            if not check_file_exist(dest):
                shutil.copy2(src,dest)

    # Get the absolute path of all audio files and save it to audio_paths array
    audio_paths = []
    # Variable to save the mp3 files that don't work
    files_that_dont_work=[]

    print("Saving all the audio files to npy")
    for audio_path in tqdm(os.listdir(dest_path)):
        # audio_path = os.listdir(dest_path)[0]
        if check_file_exist(npy_path+audio_path.split(".")[-2]+'.npy'):
            continue
        else:
            if audio_path.split(".")[-1] == "mp3":
                try:
                    melgram = compute_melgram(dest_path+audio_path)
                    dest = npy_path+audio_path.split(".")[-2]+'.npy'
                    np.save(dest, melgram)
                except EOFError:
                    files_that_dont_work.append(audio_path)
                    continue



    return 0


def main():

    music_genre_classification()


    return 0


if __name__ == '__main__':
    main()
