from collections import Counter
from itertools import dropwhile
import numpy as np
import random
import re

# initialize the data field variables
_train_data_x = []
_train_data_y = []
_dev_data_x = []
_dev_data_y = []

_observation_space = []

# global string values for files and markers
_pos_file_name = "hotelPosT-train.txt"
_neg_file_name = "hotelNegT-train.txt"

_test_file_name = "test.txt"
_test_results_name = "test-results.txt"

_dev_results_name = "dev-results.txt"

_unknown_word_marker = "<UNK>"

_prob_dict = dict()
_dev_classifications = []

# global int values for algorithm params
_dev_partition_ratio = 10
        
# preprocess the data to create a training and dev set
def preprocess():
    global _train_data_x
    global _train_data_y
    global _dev_data_x
    global _dev_data_y

    _train_data = []
    for run in {"POS", "NEG"}:
        # set file name to read
        file_name = None
        if run == "POS":
            file_name = _pos_file_name
        elif run == "NEG":
            file_name = _neg_file_name

        # read in the training data
        cur_file = open(file_name, "r")
        train_lines = []
        for line in cur_file:
            train_lines.append(line)

        # pull out dict of reviews
        train_dict = getReviews(train_lines)

        # create x and y for training
        for review in train_dict.values():
            _train_data_x.append(review)
            _train_data_y.append(run)


    # get a random sampling of lines
    sample = random.sample(range(len(_train_data_x)), int(len(_train_data_x) / _dev_partition_ratio))

    # pull out the sentences into train and dev sets
    count = 0
    for i in sample:
        i -= count
        _dev_data_x.append(_train_data_x.pop(i))
        _dev_data_y.append(_train_data_y.pop(i))        
        count += 1

# calculate all needed probs and counts for hmm/viterbi
def train():

    buildObservationSpace()

    # iterate through training data and count the transitions for pos
    buildCountDict()


    # find the observation likelihood for the pos given words
    _pos_prob_dict, _neg_prob_dict = buildProbDicts(pos_counts_dict, neg_counts_dict)

# pulls out reviews into a dict of IDs and list of words
def getReviews(data_lines):
    
    review_dict = dict()
    for line in data_lines:
        word_list = []
        word = ""
        id_num = ""
        for char in line:
            if len(id_num) == 0:
                if char == "\t":
                    id_num = word
                    word = ""
                else:
                    word += char

            else:
                # finished word
                if re.match(r"[\s\.\?\,\!]", char):
                    # not word
                    if word != "":
                        word_list.append(word.lower())
                        word = ""
                else:
                    word += char

        review_dict[id_num] = word_list

    return review_dict

# scan through data and build the observation and state spaces
def buildObservationSpace():
    global _observation_space

    all_words = []
    for x in _train_data_x:
        all_words += x

    word_counter = Counter(all_words)
    _observation_space = list(set(all_words))

    for word, count in word_counter.items():
        if count == 1:
            _observation_space.remove(word)

    # add unknown word marker <UNK>
    _observation_space.append(_unknown_word_marker)

# build the count dict
def buildCountDict():

    # initialize as ones for laplace smoothing
    count_matrix = np.ones(2, len(_observation_space))
    
    # iterate through training data and count the transitions for pos and emissions for words
    for x, y in zip(_train_data_x, _train_data_y):
        for word in x:
            # increment the count for the emission count
            try:
                count_matrix[word] += 1
            except:
                count_dict[_unknown_word_marker] += 1

    print(count_dict[_unknown_word_marker])

    return count_dict
    
# build the prob dictionary
def buildProbDicts(pos_counts_dict, neg_counts_dict):

    word_sums = dict()
    for word in _observation_space:
        word_sums[word] = pos_counts_dict[word] + neg_counts_dict[word]

    pos_prob_dict = dict()
    for word in _observation_space:
        pos_prob_dict[word] = pos_counts_dict[word] / word_sums[word]

    neg_prob_dict = dict()
    for word in _observation_space:
        neg_prob_dict[word] = neg_counts_dict[word] / word_sums[word]

    return pos_prob_dict, neg_prob_dict

def predict(observations):

    pos_prob_sum = 0
    neg_prob_sum = 0
    for word in observations:
        pos_prob_sum += np.log(_pos_prob_dict[word])
        neg_prob_sum += np.log(_neg_prob_dict[word])

    if pos_prob_sum >= neg_prob_sum:
        return "POS"
    else:
        return "NEG"

#  test using the basic "most frequent tag" technique
def test(run_file_name, results_file_name):
    global _dev_classifications

    # create results file
    results_file = open(results_file_name, "w")

    # retrieve data to run through model
    test_file = open(run_file_name, "r")
    test_lines = []
    for line in test_file:
        test_lines.append(line)
    data = getReviews(test_lines)
            
    # go through data get the most probable class, printing the results
    for key, word_list in data.items():
        observations = []
        for word in word_list:
            # try to get the index of the word, if not found, substitute the unknown word marker
            try:
                _observation_space.index(word)
                observations.append(word)
            except:
                observations.append(_unknown_word_marker)

        classification = predict(observations)
        _dev_classifications.append(classification)

        # write results to the file
        results_file.write(str(key) + "\t" + str(classification))

        # print line between reviews
        results_file.write("\n")

def devAccuracy():
    
    count = 0
    for i in range(0, len(_dev_classifications)):
        if i < 5 and _dev_classifications[i] == "POS":
            count += 1
        elif i >= 5 and _dev_classifications[i] == "NEG":
            count += 1

    acc = np.round(count / len(_dev_classifications) * 100, 2)

    print("Accuracy: " + str(acc) + "%")
    return acc

def restart():
    global _pos_train_data
    global _neg_train_data
    global _observation_space
    global _pos_prob_dict
    global _neg_prob_dict
    global _dev_classifications


    _pos_train_data = []
    _neg_train_data = []
    _observation_space = []
    _pos_prob_dict = dict()
    _neg_prob_dict = dict()
    _dev_classifications = []

# main to run program
if  __name__ == "__main__":

    accuracies = []
    for i in range(0, 100):
        restart()

        # read and process data
        preprocess()

        # train on data
        train()

        # test on the dev set
        test(_pos_dev_name, _pos_dev_results_name)
        test(_neg_dev_name, _neg_dev_results_name)

        accuracies.append(devAccuracy())

    print("Average Accuracy: " + str(np.average(accuracies)) + "%")

    # test on the test set
    # print("Running test data...")
    # test(_test_file_name, _test_results_file_name)

    # finished
    print("Finished.")