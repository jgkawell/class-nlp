from collections import Counter
import numpy as np
import random

# initialize the data field variables
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

    # pull out the sentences from the lines
    train_sentences = []
    temp_sentence = []
    for line in full_file:
        temp_sentence.append(line)
        if line == "\n":
            train_sentences.append(temp_sentence.copy())
            temp_sentence = []
        
    # get a random sampling of sentence indices
    sample = random.sample(range(len(train_sentences)), int(len(train_sentences) / 2))

    # pull out the sentences into train and dev sets
    dev_sentences = []
    count = 0
    for i in sample:
        i -= count
        dev_sentences.append(train_sentences.pop(i))
        count += 1

    # fill training file
    for sentence in train_sentences:
        for line in sentence:
            train_file.write(line)

    # fill dev file
    for sentence in dev_sentences:
        for line in sentence:
            dev_file.write(line)


# find stats for basic analysis
def findStatsForBasicAnalysis():
    global _train_data
    global _unique_words
    global _unique_parts

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
        else:
            nums.append(0)
            words.append(0)
            parts.append(0)

    return (nums, words, parts)


# basic "most frequent tag" system to just make a start on the project
def trainForBasicAnalysis():
    global _pos_counts

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
    unique = _unique_words
    try:
        word_index = unique.index(word)
        part_index = np.argmax(_pos_counts[word_index])
        pos = _unique_parts[part_index]
    except:
        pos = "NN"

    return pos



#  test using the basic "most frequent tag" technique
def testForBasicAnalysis():

    # create results file
    results_file = open(_results_file_name, "w")
    
    # retrieve data to run through model
    dev_data = getFileData(_dev_file_name)

    # write predictions to test file
    for i in range(0, len(dev_data[1])):
        if dev_data[0][i] == 0:
            results_file.write("\n")
        else:
            line = str(dev_data[0][i]) + "\t" + dev_data[1][i] + "\t" + modelForBasicAnalysis(dev_data[1][i]) + "\n"
            results_file.write(line)



if  __name__ == "__main__":
    preprocessForBasicAnalysis()

    findStatsForBasicAnalysis()

    trainForBasicAnalysis()

    testForBasicAnalysis()
