from collections import Counter
import numpy as np
import random
from bpe import Encoder

# initialize the data field variables
_train_data = []
_state_space = []
_tokenized_train_x = []
_tokenized_train_y = []
_observation_space = []
_initial_prob = []
_observation_prob = [[]]
_transition_prob = [[]]

# global string values for files and markers
_full_file_name = "gene-trainF18.txt"
_train_file_name = "train.txt"
_dev_file_name = "dev.txt"
_test_file_name = "test.txt"
_dev_results_file_name = "dev-results.txt"
_test_results_file_name = "test-results.txt"
_sentence_marker = "<s>"

# global int values for algorithm params
_encoder = Encoder(vocab_size=10000, pct_bpe=0.88, ngram_min=1, ngram_max=10)
_dev_partition_ratio = 10
        
# preprocess the data to create a training and dev set
def preprocess():

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
    sample = random.sample(range(len(train_sentences)), int(len(train_sentences) / _dev_partition_ratio))

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

# calculate all needed probs and counts for hmm/viterbi
def train():
    global _train_data
    global _initial_prob
    global _observation_prob
    global _transition_prob

    # get the list of sentences with words and entity for training
    _train_data = getSentences(_train_file_name)

    len_observation_space, len_state_space = buildSpaces()

    # iterate through training data and count the transitions for entity
    transition_count_matrix, observation_count_matrix = buildCountMatrices(len_state_space, len_observation_space)

    # find the observation likelihood for the entity given words
    _observation_prob = buildProbMatrix(len_state_space, len_observation_space, observation_count_matrix)

    # find the transition probability for entity given previous entity
    _transition_prob = buildProbMatrix(len_state_space, len_state_space, transition_count_matrix)

    # calculate the initial probabilities
    sentence_beginning_index = _state_space.index(_sentence_marker)
    for s in range(0, len(_state_space)):
        _initial_prob.append(_transition_prob[sentence_beginning_index][s])

# pulls out words and entity in sentences
def getSentences(file_name, dev=True):
     # read in training data
    lines = open(file_name, "r")
    
    sentence_list = []
    sentence = []
    for line in lines:
         #  pull out the individual fields of the data
        fields = line.rstrip("\n\r").split("\t")
        
        # if the data is not a blank line, add the data sentence
        # else, add sentence to list and clear for new sentence
        if len(fields) > 1:
            # if it is the dev file, we can pull out the entity (2) position for training
            # else, we only have two positions since there is no entity field
            if dev:
                entry = (fields[0], fields[1], fields[2])
            else:
                entry = (fields[0], fields[1])

            # add the entry to the sentence
            sentence.append(entry)
        else:
            # add sentence to list and clear
            sentence_list.append(sentence.copy())
            sentence.clear()

    return sentence_list

# scan through data and build the observation and state spaces
def buildSpaces():
    global _observation_space
    global _state_space
    global _tokenized_train_x
    global _tokenized_train_y
    
    # use the bpe encoder to make the dictionary
    all_words = ""
    sentence_string_list = []
    sentence_string = ""
    for sentence in _train_data:
        for entry in sentence:
            sentence_string += entry[1].lower() + " "
            
        all_words += sentence_string
        sentence_string_list.append(sentence_string)
        sentence_string = ""

    
    _encoder.fit(all_words.split('\n'))


    for sentence in _train_data:
        for entry in sentence:
            word = entry[1].lower()
            entity = entry[2]

            token = _encoder.tokenize(word)

            if len(token) == 1:
                _tokenized_train_x.append(token[0])
                _tokenized_train_y.append(entity)
            else:      
                for t in token:
                    if t != "__sow" and t != "__eow":
                        _tokenized_train_x.append(t)
                        _tokenized_train_y.append(entity)

        _tokenized_train_x.append(_sentence_marker)
        _tokenized_train_y.append(_sentence_marker)


    # count observations and states
    # obs_counter = Counter(_tokenized_train_x)
    state_counter = Counter(_tokenized_train_y)

    # populate spaces from counters
    _observation_space = list(_encoder.bpe_vocab.keys())
    _observation_space += list(_encoder.word_vocab.keys())
    _observation_space.append(_sentence_marker)
    _state_space = list(state_counter.keys())

    #  add sentence marker (<s>)
    _state_space.append(_sentence_marker)

    return (len(_observation_space), len(_state_space))

# build the count matrices needed to build the prob matrices
def buildCountMatrices(len_state_space, len_observation_space):

    # initialize as ones for laplace smoothing
    transition_count_matrix = np.ones((len_state_space, len_state_space))
    emission_count_matrix = np.ones((len_state_space, len_observation_space))
    
    # iterate through training data and count the transitions for entities and emissions for words
    prev_x = _sentence_marker
    for x, y in zip(_tokenized_train_x, _tokenized_train_y):

        # increment the count for the transition count
        transition_count_matrix[_state_space.index(y)][_state_space.index(prev_x)] += 1

        # increment the count for the emission count
        emission_count_matrix[_state_space.index(y)][_observation_space.index(x)] += 1
        
        # set the previous entity
        prev_x = y

    return (transition_count_matrix, emission_count_matrix)

#  build the prob matrices (both transition and emission)
def buildProbMatrix(num_rows, num_cols, count_matrix):

    # scan through and find the sums of the counts on each row (needed for laplace smoothing)
    prob_matrix = np.zeros((num_rows, num_cols))
    row_sums = np.zeros(num_rows)
    for row in range(0, num_rows):
        for col in range(0, num_cols): 
            row_sums[row] += count_matrix[row][col]

    # find the transition probabilities for the entity
    for row in range(0, num_rows):
        for col in range(0, num_cols):
            # add num_rows to denominator for laplace smoothing
            prob_matrix[row][col] = count_matrix[row][col] / (row_sums[row] + num_rows)

    return prob_matrix

#  test using viterbi technique
def test(run_file_name, results_file_name):

    # create results file
    results_file = open(results_file_name, "w")

    # retrieve data to run through model
    data = getSentences(run_file_name, dev=False)
            
    # go through data and run viterbi on each sentence, printing the results
    for sentence in data:
        numbers = []
        observations = []
        for entry in sentence:
            num = entry[0]
            word = entry[1].lower()

            token = _encoder.tokenize(word)

            if len(token) == 1:
                observations.append(_observation_space.index(token[0]))
                numbers.append(num)
            else:      
                for t in token:
                    if t != "__sow" and t != "__eow":
                        observations.append(_observation_space.index(t))
                        numbers.append(num)

        # run viterbi on each sentence
        best_path = viterbi(len(_state_space), _transition_prob, _observation_prob, _initial_prob, observations)

        # write results to the file
        obs_count = 0
        sen_count = 0
        prev_num = '1'
        for num in numbers:
            # pull out the num, word, and predicted entity
            entity = _state_space[best_path[obs_count]]

            if prev_num != num:
                # write line
                results_file.write(prev_num + "\t" + sentence[sen_count][1] + "\t" + entity + "\n")

                # set prev_num and sentence count
                prev_num = num
                sen_count += 1

            # increment count
            obs_count += 1

        # write line
        results_file.write(prev_num + "\t" + sentence[sen_count][1] + "\t" + entity + "\n")

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
    preprocess()

    # train on data
    print("Training on data...")
    train()

    # test on the dev set
    print("Running dev data...")
    test(_dev_file_name, _dev_results_file_name)

    # # test on the test set
    # print("Running test data...")
    # test(_test_file_name, _test_results_file_name)

    # finished
    print("Finished.")