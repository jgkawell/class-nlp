from collections import Counter
import numpy as np

# initialize the data field variables
_train = []
_dev = []
_word_nums = []
_words = []
_parts = []
_unique_parts = []
_unique_words = []
        
# preprocess the data to find unique and separate training from dev data
def preprocess():

    # read in the training data
    _train = open("hw-1/berp-POS-training.txt", "r")

    lines = []
    count = 0
    for line in _train:
        count += 1
        lines.append(line.rstrip("\n\r"))

    # sample from the full training set to make a smaller training and dev set
    _dev = np.random.choice(lines, int(count / 10), False)

    for line in lines:
        print(line)

    print("------------------------------------------------")

    for line in _dev:
        print(line)

    # file_length = 0
    # for line in _train:
    #     # increment the count for the total length of the training data
    #     file_length += 1

    #     #  pull out the individual collumns of the data
    #     fields = line.rstrip("\n\r").split("\t")

    #     #  if the data is not a blank line, add the data to the lists
    #     if len(fields) > 1:
    #         _word_nums.append(fields[0])
    #         _words.append(fields[1])
    #         _parts.append(fields[2])

    
    # print("Number of lines: " + str(file_length))


    # #  get the list of unique parts of speach
    # _unique_parts = list(Counter(_parts).keys())

    # # get the list of unique words
    # _unique_words = list(Counter(_words).keys())

    


# basic "most frequent tag" system to just make a start on the project
def train():

    # count the pos tags associated with words
    _pos_counts = np.zeros((len(_unique_words), len(_unique_parts)))
    for i in range(0, len(_words)):
        word = _words[i]
        part = _parts[i]

        word_index = _unique_words.index(word)
        part_index = _unique_parts.index(part)

        _pos_counts[word_index][part_index] += 1


#  test using the basic "most frequent tag" technique
def test():
    print("not implemented")



if  __name__ == "__main__":
    preprocess()

    # train()
