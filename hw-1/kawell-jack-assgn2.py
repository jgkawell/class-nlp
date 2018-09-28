from collections import Counter
import numpy as np
import random

# initialize the data field variables
_train_data = ([], [], []) # 0=nums, 1=words, 2=parts
_part_types = []
_word_types = []
_observation_prob = [[]]
_transition_prob = [[]]


_full_file_name = "hw-1/berp-POS-training.txt"
_train_file_name = "hw-1/train.txt"
_dev_file_name = "hw-1/dev.txt"
_results_file_name = "hw-1/results.txt"
_blank_line = "blank line"
_beginning_of_sentence = "<s>"

        
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
    sample = random.sample(range(len(train_sentences)), int(len(train_sentences) / 10))

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
            words.append(_blank_line)
            parts.append(_beginning_of_sentence)

    return (nums, words, parts)

# find tokens for both words and pos
def calcProbData():
    global _train_data
    global _word_types
    global _part_types
    global _observation_prob
    global _transition_prob

    _train_data = getLists(_train_file_name)

    # get the list of word types and tokens
    word_counter = Counter(_train_data[1])
    _word_types = list(word_counter.keys())
    num_word_types = len(_word_types)

    #  get the list of pos types and tokens
    part_counter = Counter(_train_data[2])
    _part_types = list(part_counter.keys())
    num_part_types = len(_part_types)

    # iterate through training data and count the transitions for pos
    parts_transition_count = buildCountMatrix(num_part_types, num_part_types, count_type=1)

    # find the transition probability for pos given previous pos
    _transition_prob = buildProbMatrix(num_part_types, num_part_types, parts_transition_count)

    # iterate through training data and count the words with corresponding pos
    word_observation_count = buildCountMatrix(num_part_types, num_word_types, count_type=2)

    # find the observation likelihood for the pos given words
    _observation_prob = buildProbMatrix(num_part_types, num_word_types, word_observation_count)

# build the count matrices needed to build the prob matrices
def buildCountMatrix(num_rows, num_cols, count_type):
    
    count_matrix = np.zeros((num_rows, num_cols))
    if count_type == 1:
            # iterate through training data and count the transitions for pos
            for row in range(0, len(_train_data[1])):
                prev = "?"
                cur = _part_types.index(_train_data[2][row])
                
                #  for the first position, make the previous pos the beginning of the sentence marker
                if row == 0:
                    prev = _part_types.index(_beginning_of_sentence)
                else:
                    prev = _part_types.index(_train_data[2][row - 1])

                # increment the count for the transition count
                count_matrix[cur][prev] += 1
    elif count_type == 2:
            # iterate through training data and count the words with corresponding pos
            for i in range(0, len(_train_data[1])):
                # pull out the current word and pos
                cur_word = _word_types.index(_train_data[1][i])
                cur_part = _part_types.index(_train_data[2][i])

                # increment the count for the observation
                count_matrix[cur_part][cur_word] += 1

    return count_matrix

#  build the prob matrices (both transition and emission)
def buildProbMatrix(num_rows, num_cols, count_matrix):
    # find the transition probabilities for the pos
    prob_matrix = np.zeros((num_rows, num_cols))
    row_sums = np.zeros(num_rows)
    for row in range(0, num_rows):
        for col in range(0, num_cols): 
            row_sums[row] += count_matrix[row][col]

    for row in range(0, num_rows):
        for col in range(0, num_cols):
            prob_matrix[row][col] = count_matrix[row][col] / row_sums[row]

    return prob_matrix

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
    global _transition_prob
    global _observation_prob

    observtions = _word_types
    states = _part_types
    transition_prob = _transition_prob
    observation_prob = _observation_prob

    for i in range(0, len(observation_prob)):
        total = 0
        for j in range(0, len(observation_prob[0])):
            total += observation_prob[i][j]

        print("POS: " + str(_part_types[i]) + " Sum: " + str(total))

    


    for s in states:
        x = 1


def predict(word):
    print("Not implemented")

if  __name__ == "__main__":

    createTrainingAndDevData()

    calcProbData()

    viterbi()

    generateOutput()
