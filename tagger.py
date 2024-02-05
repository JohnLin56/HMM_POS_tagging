# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import numpy as np
import random

def configure_train(training_list):
    """ return three dictionaries and two lists, first one is {word: [[tag1, #], [tag2, #]]},
    second one is {(word1 of sentence1, word2 in sentence1, ...): [word1 tag, word2 tag, ...]}
    third one is {tag: [[word1, #], [word2, #]]}
    forth one is a list that consists of all tag
    fifth one is a list that consists of all words"""
    word_to_tag = {}  # key = word, value = list of lists => [tag, number of occurrence]
    sentence = {}  # key = tuples of words in a sentence, value = list of tags corresponding to the word order
    tag_to_word = {}  # key = tag, value = list of lists => [word, number of occurrence]
    all_tags = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD", "DPS", "DT0", "DTQ", "EX0",
                "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI", "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN",
                "PUQ", "PUR", "TO0", "UNC", "VBB", "VBD", "VBG", "VBI", "VBN", "VBZ", "VDB", "VDD", "VDG", "VDI", "VDN",
                'VDZ', "VHB", "VHD", "VHG", "VHI", "VHN", "VHZ", "VM0", "VVB", "VVD", "VVG", "VVI", "VVN", "VVZ", "XX0",
                "ZZ0", "AJ0-AV0", "AJ0-VVN", "AJ0-VVD", "AJ0-NN1", "AJ0-VVG", "AVP-PRP", "AVQ-CJS", "CJS-PRP",
                "CJT-DT0", "CRD-PNI", "NN1-NP0", "NN1-VVB", "NN1-VVG", "NN2-VVZ", "VVD-VVN", "AV0-AJ0", "VVN-AJ0",
                "VVD-AJ0", "NN1-AJ0", "VVG-AJ0", "PRP-AVP", "CJS-AVQ", "PRP-CJS", "DT0-CJT", "PNI-CRD", "NP0-NN1",
                "VVB-NN1", "VVG-NN1", "VVZ-NN2", "VVN-VVD"]
    all_words = []
    for each_train in training_list:
        f = open(each_train, "r")
        all_lines = f.readlines()  # each item is one line
        sen = tuple()
        tag_seq = list()
        for line in all_lines:  # take each line
            if line.count(":") == 1:
                pair = line.split(":")  # each line become a list, list[0] = word, list[1] = tag
                word = pair[0].strip()  # storing into word_to_tag
                t = pair[1].strip()
            else: # there are two ":"
                word = ":"
                t = "PUN"
            if word in word_to_tag:  # if word is in dictionary
                flag = True
                for each in word_to_tag[word]:  # each[0] = tag, each[1] = number of occurrence
                    if each[0] == t:
                        each[1] += 1  # word exists, and tag exists => occurrence +=1
                        flag = False
                        break
                if flag:  # if true, new tag found, add to corresponding word
                    word_to_tag[word].append([t, 1])
            else:
                word_to_tag[word] = [[t, 1]]  # new word is found

            if t in tag_to_word:  # if tag is in dictionary
                flag = True
                for each in tag_to_word[t]:  # each[0] = word, each[1] = number of occurrence
                    if each[0] == word:
                        each[1] += 1  # tag exists, and word exists => occurrence +=1
                        flag = False
                        break
                if flag:  # if true, new word found, add to corresponding tag
                    tag_to_word[t].append([word, 1])
            else:
                tag_to_word[t] = [[word, 1]]  # new tag is found

            if len(sen) == 0:  # beginning of a sentence
                sen = sen + (word,)
                tag_seq = [t]
            else:
                sen = sen + (word,)
                tag_seq.append(t)
            if t == "PUN" and word != "," and word != ":" and word != ";" and word != "-":
                sentence[sen] = tag_seq  # end of a sentence, record to sentence dictionary and reset variables
                sen = ()
                tag_seq = []

            if word not in all_words:
                all_words.append(word)
    # counter = 0
    # for sen in sentence:
    #     if 20 < len(sen) < 100:
    #         print(sen)
    #         counter += 1
    # print(counter)
    return word_to_tag, tag_to_word, sentence, all_tags, all_words


def position_prob(sentences, all_tags):
    """return the indices of tag with the highest prob at each position """
    longest_sen = 0
    for sen in sentences:  # tuples of strings in a sentence
        if len(sen) > longest_sen:
            longest_sen = len(sen)
    posi_prob = np.zeros((len(all_tags), longest_sen))  # row = each tag, column = position of sen
    for sen in sentences:
        for i in range(0, len(sen)):  # position index
            tag_index = all_tags.index(sentences[sen][i])
            posi_prob[tag_index, i] += 1

    return np.argmax(posi_prob, axis=0)


def configure_test(test_file):
    """ return a list of lists, each list inside is a sentence, or a list of str.
    also return a list of all words"""
    f = open(test_file, "r")
    all_lines = f.readlines()  # each item is one line
    all_test_words = []
    predict_sentences = []
    sentence = []
    for line in all_lines:  # take each line
        word = line.strip()
        all_test_words.append((word))
        sentence.append(word)
        if word == "." or word == "!" or word == "?":
            predict_sentences.append(sentence)
            sentence = []
    return predict_sentences, all_test_words


def initial(all_tags, sentence):
    """ return the initial probabilities of each tag that appears at the beginning of each sentence
    return type is dictionary """
    total = len(sentence)  # total = total number of sentences
    initial_prob = np.zeros(len(all_tags))
    for sen in sentence:
        initial_tag = sentence[sen][0]  # first tag in the tag sequence
        initial_prob[all_tags.index(initial_tag)] += 1
    initial_prob = initial_prob / total
    return initial_prob


def transit(all_tags, sentence):
    """return the transition probabilities
    i.e. return the prob of current tag given the tag of the previous word"""
    transit_mat = np.zeros((len(all_tags), len(all_tags)))
    j = 0
    while j < len(all_tags):  # looping over all the tags
        given = all_tags[j]
        for sen in sentence:  # for each tag sequence
            # if len(sen) <= length:  # only consider the samples that have shorter length than given sentence
            i = 1
            while i < len(sentence[sen]):  # for each tag in each tag sequence
                if given == sentence[sen][i - 1]:  # if previous tag is the "given" tag
                    current = sentence[sen][i]
                    transit_mat[j, all_tags.index(current)] += 1  # add one on that position
                i += 1
        j += 1
    total = np.sum(transit_mat, axis=1) + 1  # plus 1 to avoid division by 0. should not affect much since 0/1 = 0
    return transit_mat / total[:, None]


def emi(all_tags, all_words, tag_word):
    """return the emission probabilities. i.e. return the prob of word given current tag"""
    emi_mat = np.zeros((len(all_tags), len(all_words)))
    j = 0
    while j < len(all_tags):  # looping over all the tags
        given = all_tags[j]
        words = tag_word[given]  # this is a list of [word, # of occurrence] that are associated with the tag
        for word in words:  # word[0] = word, word[1] = # of occurrence
            emi_mat[j, all_words.index(word[0])] += word[1]
        j += 1
    total = np.sum(emi_mat, axis=1) + 1
    return emi_mat / total[:, None]


def check_word_in_list(word, all_words):
    """check if the word has occured in training, if not adjust for it
    return (True, index) in all_words if word exists,
    return (False, tag) if satisfy condition,
    return (False, word) if not found in all_words and not satisfying conditions"""
    if word in all_words:
        return True, all_words.index(word)
    else:
        #print(word)
        if word == "(" or word == "[":
            prob = "PUL"
        elif word == "'" or word == "\"":
            prob = "PUQ"
        elif word == ")" or word == "]":
            prob = "PUR"
        elif word == "about" or word == "at" or word == "in" or word == "on" or word == "on behalf of" or word == "with":
            prob = "PRP"
        elif word[-3:] == "ing":
            prob = "VVG"
        else:  # for other in general, find the highest transition probability.
            prob = word
        return False, prob


def predict(predict_sentences, init_prob, tran_prob, emi_prob, all_tags, all_words, posi_prob):
    """using viterbi algorithm to predict the tag of each word.
     return a list of prediction"""
    prediction = []
    for sentence in predict_sentences:
        prob = np.zeros((len(sentence), tran_prob.shape[1]))

        # for time step 0
        for i in range(0, tran_prob.shape[1]):
            word = check_word_in_list(sentence[0], all_words)
            if word[0]:  # word exists in training files
                prob[0, i] = init_prob[i] * emi_prob[i, word[1]]

            elif word[1] == sentence[0]:  # word is not found in training and cannot be hard coded
                # use position table
                if i != posi_prob[0]:
                    prob[0, i] = 0
                else:
                    prob[0, i] = 1
            else:  # hard coded
                if i != all_tags.index(word[1]):
                    prob[0, i] = 0
                else:
                    prob[0, i] = 1

        # for time steps 1 to last word in sentence:
        for t in range(1, len(sentence)):
            current_word = check_word_in_list(sentence[t], all_words)
            current_tag = ""
            for i in range(0, tran_prob.shape[1]):
                max_prob = 0
                for k in range(0, tran_prob.shape[1]):
                    # find the maximum possibility of reaching current step
                    if current_word[0]:  # word exists in training file
                        option = prob[t - 1, k] * tran_prob[k, i] * emi_prob[i, current_word[1]] + 0.000000000000001
                    elif current_word[1] == sentence[t]:  # word did not occur in training files and
                        # current_word is hard coded with a tag
                        # using the position probability table
                        if t < len(posi_prob):
                            current_tag = posi_prob[t]
                            option = 1
                        else: # new sentence is longer than the longest sentence occurred in training files
                            current_tag = random.choice(posi_prob)
                            option = 1
                    else:  # tag has already hard coded to current_word
                        current_tag = all_tags.index(current_word[1])
                        option = 1
                    if option > max_prob:
                        max_prob = option
                if current_tag != "" and i != current_tag:
                    # if hard coded a tag, probability for other tag is zero
                    # or when it is not possible for this path, assign prob = 0
                    prob[t, i] = 0
                else:
                    prob[t, i] = max_prob
            # normalize:
            total = prob[t].sum()
            for posi in range(0, len(prob[t])):
                prob[t, posi] = prob[t, posi] / total
        max_values = np.argmax(prob, axis=1)  # indices of max values at each step
        prediction = np.append(prediction, max_values)
    return prediction


def write_to_output(result, outputfile, all_tags, all_test_words):
    """ input is a list of all_tag indeices
    need to write to the output file"""
    f = open(outputfile, "w")
    line = ""
    j = 0   # index for all_test_words
    for i in result:
        i = int(i)
        line += " : ".join([all_test_words[j], all_tags[i]]) + "\n"
        j += 1
        f.write(line)
        line = ""


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    word_to_tags, tags_to_word, sentence, all_tags, all_words = configure_train(training_list)

    init_prob = initial(all_tags, sentence)  # initial probabilities
    transition_prob = transit(all_tags, sentence)  # transition probability matrix
    emission_prob = emi(all_tags, all_words, tags_to_word)  # emission probability matrix

    posi_prob = position_prob(sentence, all_tags)  # position prob
    predict_sentences, all_test_words = configure_test(test_file)
    result = predict(predict_sentences, init_prob, transition_prob, emission_prob, all_tags, all_words, posi_prob)
    write_to_output(result, output_file, all_tags, all_test_words)


##### if word in test files did not occur in train file => use position to determine which tag's prob is biggest at
##### that position.


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d") + 1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t") + 1]
    output_file = parameters[parameters.index("-o") + 1]
    print("Training files: " + str(training_list))
    print("Test file: " + test_file)
    print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
