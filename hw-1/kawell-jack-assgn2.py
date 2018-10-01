from collections import Counter
import numpy as np
import random

# initialize the data field variables
_train_data = ([], [], []) # 0=nums, 1=words, 2=parts
_state_space = []
_observation_space = []
_initial_prob = []
_observation_prob = [[]]
_transition_prob = [[]]


_full_file_name = "berp-POS-training.txt"
_train_file_name = "train.txt"
_dev_file_name = "dev.txt"
_test_file_name = "test.txt"
_results_file_name = "results.txt"
_unknown_word = "<UNK>"

        
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
def getLists(file_name, dev = True):

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
            if dev:
                parts.append(fields[2])

    return (nums, words, parts)

# calc all needed probs for hmm/viterbi
def train():
    global _train_data
    global _observation_space
    global _state_space
    global _initial_prob
    global _observation_prob
    global _transition_prob

    _train_data = getLists(_train_file_name)

    # get the list of word types and tokens
    word_counter = Counter(_train_data[1])
    _observation_space = list(word_counter.keys())

    # remove words occuring once and replace them with <UNK>
    words_to_remove = []
    for word, count in word_counter.items():
        if count == 1:
            words_to_remove.append(word)

    for word in words_to_remove:
        _observation_space.remove(word)

    _observation_space.append(_unknown_word)
    num_word_types = len(_observation_space)

    #  get the list of pos types and tokens
    part_counter = Counter(_train_data[2])
    _state_space = list(part_counter.keys())
    num_part_types = len(_state_space)

    # calculate the initial probabilities
    num_part_tokens = 0
    for part in _state_space:
        num_part_tokens += part_counter.get(part)

    for part in _state_space:
        _initial_prob.append(part_counter.get(part) / num_part_tokens)

    # iterate through training data and count the transitions for pos
    parts_transition_count = buildCountMatrix(num_part_types, num_part_types, count_type=1)

    # find the transition probability for pos given previous pos
    _transition_prob = buildProbMatrix(num_part_types, num_part_types, parts_transition_count)

    # calculate the initial probabilities
    # period_index = _state_space.index(".")
    # for s in range(0, len(_state_space)):
    #     _initial_prob.append(_transition_prob[period_index][s])

    # iterate through training data and count the words with corresponding pos
    word_observation_count = buildCountMatrix(num_part_types, num_word_types, count_type=2)

    # find the observation likelihood for the pos given words
    _observation_prob = buildProbMatrix(num_part_types, num_word_types, word_observation_count)

# build the count matrices needed to build the prob matrices
def buildCountMatrix(num_rows, num_cols, count_type):
    # initialize as ones for laplace smoothing
    count_matrix = np.ones((num_rows, num_cols))
    if count_type == 1:
            # iterate through training data and count the transitions for pos
            for row in range(0, len(_train_data[1])):
                prev = "?"
                cur = _state_space.index(_train_data[2][row])
                prev = _state_space.index(_train_data[2][row - 1])

                # increment the count for the transition count
                count_matrix[cur][prev] += 1
    elif count_type == 2:
            # iterate through training data and count the words with corresponding pos
            for i in range(0, len(_train_data[1])):
                # pull out the current word and pos
                try:
                    cur_word = _observation_space.index(_train_data[1][i])
                except:
                    cur_word = _observation_space.index(_unknown_word)

                cur_part = _state_space.index(_train_data[2][i])

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
            # add num_rows to denominator for laplace smoothing
            prob_matrix[row][col] = count_matrix[row][col] / (row_sums[row] + num_rows)

    return prob_matrix

#  test using the basic "most frequent tag" technique
def test():

    # create results file
    results_file = open(_results_file_name, "w")

    # retrieve data to run through model
    dev_data = getLists(_dev_file_name, dev=False)
            
    # write predictions to test file
    sentence_list = []
    sentence = []
    for i in range(0, len(dev_data[1])):
        cur_word = dev_data[1][i]

        # if we're at the beginning of a new sentence, add to list and clear
        if cur_word == ".":
            # add the word to the sentence
            sentence.append(cur_word)
            # add to sentence list
            sentence_list.append(sentence.copy())
            # clear the sentence for a new sentence
            sentence.clear()
        else:
            sentence.append(cur_word)

    position = 0
    for sentence in sentence_list:
        observations = []
        for word in sentence:
            try:
                observations.append(_observation_space.index(word))
            except:
                observations.append(_observation_space.index(_unknown_word))

        best_path, best_prob = viterbi(len(_state_space), _transition_prob, _observation_prob, _initial_prob, observations)

        for i in range(0, len(best_path)):
            num = str(dev_data[0][position])
            word = str(dev_data[1][position])
            pos = _state_space[best_path[i]]

            results_file.write(num + "\t" + word + "\t" + pos + "\n")

            position += 1


        results_file.write("\n")

# implements the viterbi algorithm
def viterbi(num_states, transition, emission, prob, observations):

    num_observations = len(observations)
    log_transition = np.log(transition)
    log_emission = np.log(emission)
    log_probability = np.log(prob)

    path_prob = np.zeros((num_observations, num_states))
    back_pointer = np.zeros((num_observations, num_states))

    for s in range(0, num_states):
        path_prob[0,s] = log_probability[s] + log_emission[s,observations[0]]

    for o in range(1, num_observations):
        for s in range(0, num_states):
            path_prob[o,s] = np.max(path_prob[o-1] + log_transition[:,s]) + log_emission[s,observations[o]]
            # don't need the emission value for back pointer
            back_pointer[o,s] = np.argmax(path_prob[o-1] + log_transition[:,s])

    best_prob = np.max(path_prob[-1])
    best_pointer = np.argmax(path_prob[-1])

    best_path = np.zeros(num_observations, dtype=np.int32)
    best_path[-1] = best_pointer
    for p in range(num_observations - 2, -1, -1):
        best_path[p] = back_pointer[p+1, best_path[p+1]]
        
        # if p == num_observations - 2:
        #     print(_state_space[best_path[p]])

    return (best_path, best_prob)

if  __name__ == "__main__":

    createTrainingAndDevData()

    train()

    test()
