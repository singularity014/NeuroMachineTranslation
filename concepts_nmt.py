from tensorflow.keras.utils import to_categorical

# For feeding the sentence to a Neuro machine translator it needs to be
# converted to a numerical representation, i.e. a vector.
# A simple (old school way) is to represent a word as a one-hot vector.
# In this section we will see how to convert a list of words in to a one hot vector
# takes a list of words and returns the onehot of the words


def compute_onehot(list_words, dict_word2index):
    word_ids = [dict_word2index[word] for word in list_words]
    one_hot_vectors = to_categorical(word_ids, num_classes=5)
    # note that this will be a numpy matrix
    return one_hot_vectors


if __name__ == '__main__':
    l_words = ['I', 'like', 'dogs']
    word2index = {'I': 0, 'like': 1, 'dogs': 2}

    one_hot_results = compute_onehot(l_words, word2index)
    print(one_hot_results)
