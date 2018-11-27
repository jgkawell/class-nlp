import numpy as np
import random
import re

# initialize the data field variables
_train_data_x = []
_train_data_y = []
_dev_data_x = []
_dev_data_y = []

_observation_space = []
_count_matrix = []
_prob_matrix = []

# global string values for files and markers
_pos_file_name = "hotelPosT-train.txt"
_neg_file_name = "hotelNegT-train.txt"

_test_file_name = "test.txt"
_test_results_name = "test-results.txt"
_dev_results_name = "dev-results.txt"
_unknown_word_marker = "<UNK>"

_dev_classifications = []

# global int values for algorithm params
_dev_partition_ratio = 20
        
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
        cur_file = open(file_name, "r", encoding="utf8")
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

    buildCountMatrix()

    buildProbMatrix()

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

    _observation_space = list(set(all_words))

# build the count dict
def buildCountMatrix():
    global _count_matrix

    # initialize as ones for laplace smoothing
    _count_matrix = np.zeros((2, len(_observation_space)))
    
    # iterate through training data and count the transitions for pos and emissions for words
    for x, y in zip(_train_data_x, _train_data_y):
        # pull out the y index
        if y == "POS":
            index = 0    
        elif y == "NEG":
            index = 1

        for word in x:
            # increment the count for the emission count
            try:
                _count_matrix[index][_observation_space.index(word)] += 1
            except:
                _count_matrix[index][_observation_space.index(_unknown_word_marker)] += 1
    
# build the prob dictionary
def buildProbMatrix():
    global _prob_matrix

    # scan through and find the sums of the counts on each row (needed for laplace smoothing)
    _prob_matrix = np.zeros((2, len(_observation_space)))

    # find the transition probabilities for the pos
    for row in range(0, 2):
        for col in range(0, len(_observation_space)):
            # add num_rows to denominator for laplace smoothing
            row_sum = _count_matrix[0][col] + _count_matrix[1][col]
            _prob_matrix[row][col] = (_count_matrix[row][col] + 1) / (row_sum + 2)

def predict(observations):

    pos_prob_sum = 0
    neg_prob_sum = 0
    for word in observations:
        word_index = None
        try:
            word_index = _observation_space.index(word)
            pos_prob_sum += np.log(_prob_matrix[0][word_index])
            neg_prob_sum += np.log(_prob_matrix[1][word_index])
        except:
            pass

    dif = pos_prob_sum - neg_prob_sum

    if dif >= 0:
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

def dev():
    correct = 0
    for x, y in zip(_dev_data_x, _dev_data_y):
        y_hat = predict(x)
        if y_hat == y:
            correct += 1

    acc = np.round(correct / len(_dev_data_x) * 100, 2)

    print("Accuracy: " + str(acc) + "%")

    return acc

def restart():
    global _train_data_x
    global _train_data_y
    global _dev_data_x
    global _dev_data_y

    global _observation_space
    global _count_matrix
    global _prob_matrix

    global _dev_classifications


    _train_data_x = []
    _train_data_y = []
    _dev_data_x = []
    _dev_data_y = []

    _observation_space = []
    _count_matrix = []
    _prob_matrix = []

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
        accuracies.append(dev())

    print("Average Accuracy: " + str(np.round(np.average(accuracies), 2)) + "%")

    # test on the test set
    # print("Running test data...")
    # test(_test_file_name, _test_results_file_name)

    # finished
    print("Finished.")