from collections import Counter
import numpy as np

# initialize the data field variables
_train = []
_train_data = ([], [], []) # 0=nums, 1=words, 2=parts
_unique_parts = []
_unique_words = []
_pos_counts = []

_full_file_name = "hw-1/berp-POS-training.txt"
_train_file_name = "hw-1/train.txt"
_dev_file_name = "hw-1/dev.txt"
_results_file_name = "hw-1/results.txt"

        
# preprocess the data to find unique and separate training from dev data
def preprocessForBasicAnalysis():

    # read in the training data
    full_file = open(_full_file_name, "r")

    # create subset files for training and dev
    train_file = open(_train_file_name, "w")
    dev_file = open(_dev_file_name, "w")

    # pull out the lines into a list
    train_lines = []
    count = 0
    for line in full_file:
        count += 1
        train_lines.append(line)

    # sample from the full training set to make a smaller training and dev set
    dev_lines = np.random.choice(train_lines, int(count / 10), False)

    # fill training file
    for line in train_lines:
        train_file.write(line)

    # fill dev file
    for line in dev_lines:
        dev_file.write(line)


# find stats for basic analysis
def findStatsForBasicAnalysis():

    _train_data = getFileData(_train_file_name)

    # get the list of unique words
    _unique_words = list(Counter(_train_data[1]).keys())

    #  get the list of unique parts of speach
    _unique_parts = list(Counter(_train_data[2]).keys())

def getFileData(file_name):

    # read in training data
    lines = open(file_name, "r")

    file_length = 0
    nums = []
    words = []
    parts = []
    for line in lines:
        # increment the count for the total length of the training data
        file_length += 1

        #  pull out the individual collumns of the data
        fields = line.rstrip("\n\r").split("\t")

        #  if the data is not a blank line, add the data to the lists
        if len(fields) > 1:
            nums.append(fields[0])
            words.append(fields[1])
            parts.append(fields[2])

    return (nums, words, parts)


# basic "most frequent tag" system to just make a start on the project
def trainForBasicAnalysis():

    # count the pos tags associated with words
    _pos_counts = np.zeros((len(_unique_words), len(_unique_parts)))
    for i in range(0, len(_train_data[1])):
        word = _train_data[1][i]
        part = _train_data[2][i]

        word_index = _unique_words.index(word)
        part_index = _unique_parts.index(part)

        _pos_counts[word_index][part_index] += 1

# basic model that uses the word counts to predict pos
def modelForBasicAnalysis(word):

    # if the word exists in the training data find the most common pos
    # if not, an exception will be thrown and we'll guess that it's a noun
    pos = "?"
    try:
        word_index = _unique_words.index(word)
        part_index = np.argmax(_pos_counts[word_index])
        pos = _unique_parts[part_index]
    except:
        pos = "NN"

    return pos



#  test using the basic "most frequent tag" technique
def test():

    results_file = open(_results_file_name, "w")
    
    dev_data = getFileData(_dev_file_name)

    count = 0
    for word in dev_data[1]:
        dev_data[2][count]
        count += 1



if  __name__ == "__main__":
    preprocessForBasicAnalysis()

    findStatsForBasicAnalysis()

    trainForBasicAnalysis()


    # train()
