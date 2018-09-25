# basic "most frequent tag" system to just make a start on the project
def calcStats():
    
    f = open("berp-POS-training.txt", "r")

    file_length = 0
    word_nums = []
    words = []
    parts = []
    for line in f:

        fields = line.rstrip("\n\r").split("\t")

        if len(fields) > 1:
            word_nums.append(fields[0])
            words.append(fields[1])
            parts.append(fields[2])

        file_length += 1

    print("Number of lines: " + str(file_length))    

    


if  __name__ == "__main__":
    calcStats()
