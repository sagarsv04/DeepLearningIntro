
import _pickle as pickle
from collections import Counter
import keras
import os


# download the data from below link
# http://mlg.ucd.ie/datasets/bbc.html


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\TextSummarizer')


data_dir_base = "./bbc/"


def get_list_dir(path):
    list_dir = os.listdir(path)
    return list_dir


def read_files_in_list(data_dir_base):

    headings = []
    contents = []
    folder_list = get_list_dir(data_dir_base)
    files_path = []
    for folder in folder_list:
        # folder = folder_list[0]
        file_list = get_list_dir(data_dir_base+folder)
        for file in file_list:
            # file = file_list[0]
            files_path.append(data_dir_base+folder+'/'+file)
    for file in files_path:
        # file = files_path[0]
        with open(file, 'r') as f:
            content = f.read()
        head = content.split('\n\n')
        headings.append(content[0])
        # add to string
        body = [body for i in range(1,len(content)) ]
        contents.append(content[1])

    return 0



def create_pickle(data_dir_base):
    headings, contents = read_files_in_list(data_dir_base)


    return 0


def main():


    return 0




if __name__ == '__main__':
    main()
