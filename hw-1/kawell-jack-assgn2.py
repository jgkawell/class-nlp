from collections import Counter
import numpy as np
import random

# initialize the data field variables
_train_data = ([], [], []) # 0=nums, 1=words, 2=parts
_part_types = []
_word_types = []
_words_observation_prob = [[]]
_parts_transition_prob = [[]]


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
def buildProbMatrices():
    global _train_data
    global _word_types
    global _part_types
    global _words_observation_prob
    global _parts_transition_prob

    _train_data = getLists(_train_file_name)

    # get the list of word types and tokens
    word_counter = Counter(_train_data[1])
    _word_types = list(word_counter.keys())
    num_word_types = len(_word_types) + 1 # +1 to account for <unk>, unknown words that will appear in testing

    #  get the list of pos types and tokens
    part_counter = Counter(_train_data[2])
    _part_types = list(part_counter.keys())
    num_part_types = len(_part_types) + 1 # +1 to account for the beginning sentence marker

    # find the transition probability for pos given previous pos
    _parts_transition_prob = findPartsTransitionProb(num_part_types)

    # find the observation likelihood for the pos given words
    _words_observation_prob = findWordsObservationProb(num_word_types, num_part_types)

# find the transition probabilities for the pos
def findPartsTransitionProb(num_part_types):
    parts_transition_count = np.zeros((num_part_types, num_part_types))
    for i in range(0, len(_train_data[1])):
        prev_part = "?"
        cur_part = _part_types.index(_train_data[2][i])
        
        #  for the first position, make the previous pos the beginning of the sentence marker
        if i == 0:
            prev_part = _part_types.index("<s>")
        else:
            prev_part = _part_types.index(_train_data[2][i - 1])

        # increment the count for the transition count
        parts_transition_count[cur_part][prev_part] += 1

    # find the transition probabilities for the pos
    parts_transition_prob = np.zeros((num_part_types, num_part_types))
    for i in range(0, num_part_types):
        for j in range(0, num_part_types):
            parts_transition_prob[i][j] = parts_transition_count[i][j] / (num_part_types * num_part_types)

    return parts_transition_prob

# find the observation probability for the words
def findWordsObservationProb(num_word_types, num_part_types):
    # iterate through training data and count the words with corresponding pos
    word_observation_count = np.zeros((num_word_types, num_part_types))
    for i in range(0, len(_train_data[1])):
        # pull out the current word and pos
        cur_word = _word_types.index(_train_data[1][i])
        cur_part = _part_types.index(_train_data[2][i])
        
        if cur_part == _part_types.index("<s>"):
            continue

        # increment the count for the observation
        word_observation_count[cur_word][cur_part] += 1

    # find the observation probability from the counts
    word_observation_prob = np.zeros((num_word_types, num_part_types))
    for i in range(0, num_word_types):
        for j in range(0, num_part_types):
            word_observation_prob[i][j] = word_observation_count[i][j] / (num_word_types * num_part_types)

    return word_observation_prob

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
            parts.append("<s>")

    return (nums, words, parts)

#  test using the basic "most frequent tag" technique
def generateOutput():

    # create results file
    results_file = open(_results_file_name, "w")
    
    # retrieve data to run through model
    dev_data = getLists(_dev_file_name)

    # write predictions to test file
    for i in range(0, len(dev_data[1])):
        if dev_data[0][i] == 0:
            results_file.write("\n")
        else:
            line = "test"#str(dev_data[0][i]) + "\t" + dev_data[1][i] + "\t" + predict(str(dev_data[1][i])) + "\n"
            results_file.write(line)

# implements the viterbi algorithm
def viterbi():
    global _word_types
    global _part_types
    global _probabilities
    global _train_data
    global _parts_transition_prob
    global _words_observation_prob

    print("Not implemented")

def predict(word):
    print("Not implemented")

if  __name__ == "__main__":

    createTrainingAndDevData()

    buildProbMatrices()

    viterbi()

    generateOutput()
