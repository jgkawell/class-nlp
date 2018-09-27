from collections import Counter
import numpy as np
import random

# initialize the data field variables
_train_data = ([], [], []) # 0=nums, 1=words, 2=parts
_part_types = []
_word_types = []
_pos_counts = []

_full_file_name = "hw-1/berp-POS-training.txt"
_train_file_name = "hw-1/train.txt"
_dev_file_name = "hw-1/dev.txt"
_results_file_name = "hw-1/results.txt"

        
# preprocess the data to create a training and dev set
def createTrainingAndDevData():

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



# find tokens for both words and pos
def findTokens():
    global _train_data
    global _word_types
    global _part_types

    _train_data = getLists(_train_file_name)

    # get the list of word types
    word_counter = Counter(_train_data[1])
    _word_types = list(word_counter.keys())

    #  get the list of unique parts of speach
    part_counter = Counter(_train_data[2])
    _part_types = list(part_counter.keys())

    _word_tokens = []
    for word in _word_types:
        _word_tokens.append(word_counter.get(word))

    _part_tokens = []
    for part in _part_types:
        _part_tokens.append(part_counter.get(part))

# pulls out the nums, words, and pos data as lists
def getLists(file_name):

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
    _pos_counts = np.zeros((len(_word_types), len(_part_types)))
    for i in range(0, len(_train_data[1])):
        word = _train_data[1][i]
        part = _train_data[2][i]

        word_index = _word_types.index(word)
        part_index = _part_types.index(part)

        _pos_counts[word_index][part_index] += 1

# basic model that uses the word counts to predict pos
def modelForBasicAnalysis(word):

    # if the word exists in the training data find the most common pos
    # if not, an exception will be thrown and we'll guess that it's a noun
    pos = "?"
    unique = _word_types
    try:
        word_index = unique.index(word)
        part_index = np.argmax(_pos_counts[word_index])
        pos = _part_types[part_index]
    except:
        pos = "NN"

    return pos



#  test using the basic "most frequent tag" technique
def testForBasicAnalysis():

    # create results file
    results_file = open(_results_file_name, "w")
    
    # retrieve data to run through model
    dev_data = getLists(_dev_file_name)

    # write predictions to test file
    for i in range(0, len(dev_data[1])):
        if dev_data[0][i] == 0:
            results_file.write("\n")
        else:
            line = str(dev_data[0][i]) + "\t" + dev_data[1][i] + "\t" + modelForBasicAnalysis(dev_data[1][i]) + "\n"
            results_file.write(line)



if  __name__ == "__main__":

    createTrainingAndDevData()

    findTokens()

    trainForBasicAnalysis()

    testForBasicAnalysis()
