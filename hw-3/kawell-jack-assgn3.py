from collections import Counter
import numpy as np
import random
import re

# initialize the data field variables
_pos_train_data = []
_neg_train_data = []
_observation_space = []

# global string values for files and markers
_pos_file_name = "hotelPosT-train.txt"
_pos_train_name = "pos-train.txt"
_pos_dev_name = "pos-dev.txt"

_neg_file_name = "hotelNegT-train.txt"
_neg_train_name = "neg-train.txt"
_neg_dev_name = "neg-dev.txt"


_test_file_name = "test.txt"
_pos_dev_results_name = "pos-dev-results.txt"
_neg_dev_results_name = "neg-dev-results.txt"

_test_results_file_name = "test-results.txt"
_unknown_word_marker = "<UNK>"

# global int values for algorithm params
_dev_partition_ratio = 10
        
# preprocess the data to create a training and dev set
def preprocess(run):
    global _pos_train_data
    global _neg_train_data

    file_name = None
    train_name = None
    dev_name = None
    if run == "pos":
        file_name = _pos_file_name
        train_name = _pos_train_name
        dev_name = _pos_dev_name
    elif run == "neg":
        file_name = _neg_file_name
        train_name = _neg_train_name
        dev_name = _neg_dev_name

    # read in the training data
    cur_file = open(file_name, "r")

    # create subset files for training and dev
    train_file = open(train_name, "w")
    dev_file = open(dev_name, "w")

    train_lines = []
    for line in cur_file:
        train_lines.append(line)

    # get a random sampling of lines
    sample = random.sample(range(len(train_lines)), int(len(train_lines) / _dev_partition_ratio))

    # pull out the sentences into train and dev sets
    dev_lines = []
    count = 0
    for i in sample:
        i -= count
        dev_lines.append(train_lines.pop(i))
        count += 1

    # fill training file
    for line in train_lines:
        train_file.write(line)
        if run == "pos":
            _pos_train_data.append(line)
        elif run == "neg":
            _neg_train_data.append(line)

    # fill dev file
    for line in dev_lines:
        dev_file.write(line)

# calculate all needed probs and counts for hmm/viterbi
def train():

    # get the list of sentences with words and pos for training
    pos_dict = getReviews(_pos_train_data)
    neg_dict = getReviews(_neg_train_data)

    buildObservationSpace(pos_dict)
    buildObservationSpace(neg_dict)

    # iterate through training data and count the transitions for pos
    pos_counts_dict = buildCountDict(pos_dict)
    neg_counts_dict = buildCountDict(neg_dict)


    # find the observation likelihood for the pos given words
    pos_prob, neg_prob = buildProbDicts(pos_counts_dict, neg_counts_dict)

    print("Finished training...")

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
def buildObservationSpace(train_data):
    global _observation_space
    
    # scan through data and find the spaces along with the single counts in the observation space
    _observation_space = []
    single_counts = []
    for word_list in train_data.values():
        for word in word_list:

            # if word doesn't exist yet in observation space, add it
            # else, try to remove it from the single counts list
            if word not in _observation_space:
                _observation_space.append(word)
                single_counts.append(word)
            else:
                # try to remove word if it is in the single counts list
                try:
                    single_counts.remove(word)
                except:
                    pass

    # remove words with only a single count and replace them with <UNK>
    for word in single_counts:
        _observation_space.remove(word)

    # add unknown word marker <UNK>
    _observation_space.append(_unknown_word_marker)

# build the count matrices needed to build the prob matrices
def buildCountDict(train_data):

    # initialize as ones for laplace smoothing
    count_dict = dict()
    for word in _observation_space:
        count_dict[word] = 1
    
    # iterate through training data and count the transitions for pos and emissions for words
    for word_list in train_data.values():
        for word in word_list:

            # increment the count for the emission count
            try:
                count_dict[word] += 1
            except:
                count_dict[_unknown_word_marker] += 1

    return count_dict
    
#  build the prob matrices (both transition and emission)
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

#  test using the basic "most frequent tag" technique
def test(run_file_name, results_file_name):

    # create results file
    results_file = open(results_file_name, "w")

    # retrieve data to run through model
    data = getReviews(run_file_name)
            
    # go through data and run viterbi on each sentence, printing the results
    for key, word_list in data:
        observations = []
        for word in word_list:
            # try to get the index of the word, if not found, substitute the unknown word marker
            try:
                check = _observation_space.index(word)
                observations.append(word)
            except:
                observations.append(_unknown_word_marker)

        # run viterbi on each sentence
        best_path = viterbi(len(_state_space), _transition_prob, _observation_prob, _initial_prob, observations)

        # write results to the file
        count = 0
        for entry in sentence:
            # pull out the num, word, and predicted pos
            num = str(entry[0])
            word = str(entry[1])
            pos = _state_space[best_path[count]]

            # write line
            results_file.write(num + "\t" + word + "\t" + pos + "\n")

            # increment count
            count += 1

        # print line between sentences
        results_file.write("\n")

# implements the viterbi algorithm
def viterbi(num_states, transition, emission, prob, observations):

    # initialize constants for viterbi
    num_observations = len(observations)
    log_transition = np.log(transition)
    log_emission = np.log(emission)
    log_probability = np.log(prob)

    # initialize tracking matrices for viterbi
    path_prob = np.zeros((num_observations, num_states))
    back_pointer = np.zeros((num_observations, num_states))

    # initialize first column of the path prob matrix (first set of states)
    for s in range(0, num_states):
        path_prob[0][s] = log_probability[s] + log_emission[s][observations[0]]

    # scan through remaining observations and states finding the most probable and saving the backpointer
    for o in range(1, num_observations):
        for s in range(0, num_states):
            path_prob[o][s] = np.max(path_prob[o-1] + log_transition[s][:]) + log_emission[s][observations[o]]
            # don't need the emission value for back pointer
            back_pointer[o][s] = np.argmax(path_prob[o-1] + log_transition[s][:])

    # pull out the last saved backpointer
    best_pointer = np.argmax(path_prob[-1])

    # find the remaining backpointers from the last saved
    best_path = np.zeros(num_observations, dtype=np.int32)
    best_path[-1] = best_pointer
    for p in range(num_observations - 2, -1, -1):
        best_path[p] = back_pointer[p+1][best_path[p+1]]

    return best_path

# main to run program
if  __name__ == "__main__":

    # read and process data
    print("Processing data...")
    preprocess("pos")
    preprocess("neg")

    # train on data
    print("Training on data...")
    train()

    # test on the dev set
    print("Running dev data...")
    test(_pos_dev_name, _pos_dev_results_name)

    # test on the test set
    print("Running test data...")
    test(_test_file_name, _test_results_file_name)

    # finished
    print("Finished.")