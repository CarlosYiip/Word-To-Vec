## Submission.py for COMP6714-Project2
###################################################################################################################
import os
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import collections
import spacy
import pickle
import gensim


def generate_batch(batch_size, num_samples, skip_window, data_index, data): 
    assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span is the width of the sliding window
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span]) # initial buffer content = first sliding window
    
    data_index += span
    for i in range(batch_size // num_samples):
        context_words = [w for w in range(span) if w != skip_window]
        random.shuffle(context_words)
        words_to_use = collections.deque(context_words) # now we obtain a random list of context words
        for j in range(num_samples): # generate the training pairs
            batch[i * num_samples + j] = buffer[skip_window]
            context_word = words_to_use.pop()
            labels[i * num_samples + j, 0] = buffer[context_word] # buffer[context_word] is a random context word
        
        # slide the window to the next position    
        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else: 
            buffer.append(data[data_index]) # note that due to the size limit, the left most word is automatically removed from the buffer.
            data_index += 1
                
    # end-of-for
    data_index = (data_index + len(data) - span) % len(data) # move data_index back by `span`
    return batch, labels, data_index


def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    # Load dataset
    data, dictionary, reversed_dictionary = data_file
        
    # Specification of Training data:
    batch_size = 24
    skip_window = 1
    num_samples = 2
    num_sampled = 24
    vocabulary_size = len(dictionary)

    # Specification of test sample:
    sample_size = 20
    sample_window = 100
    sample_examples = np.random.choice(sample_window, sample_size, replace=False)
    
    # Begin training
    with tf.Session() as session:
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dim], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)            

        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_dim],
                                                          stddev=1.0 / math.sqrt(embedding_dim)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, 
                                                 labels=train_labels, inputs=embed, 
                                                 num_sampled=num_sampled, num_classes=vocabulary_size))
        
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1.0).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

        sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
        similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()

        session.run(init)
        average_loss = 0
        data_index = 0
        for step in range(num_steps):
            batch_inputs, batch_labels, data_index = generate_batch(batch_size, num_samples, skip_window, data_index, data)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)

        final_embeddings = normalized_embeddings.eval()
        
        file = open(embeddings_file_name, 'w')
        for i in range(len(final_embeddings)):
            file.write(reversed_dictionary[i])
            for d in final_embeddings[i]:
                file.write(' ')
                file.write("%.6f" % d)
            file.write('\n')
        file.close()


def process_data(input_dir):
    ## Phase 1: Read files
    data = []
    with zipfile.ZipFile(input_dir) as f:
        file_list = list(filter(lambda x: x[0] != "_" and ".txt" in x, f.namelist()))
        for file_name in file_list:
            data += tf.compat.as_str(f.read(file_name)).split()

    ## Phase 2: Data preprocess
    nlp = spacy.load('en')
    docs = nlp(' '.join(data))
    words = []
    for tok in docs:
        # For each word in the corpus, if it is an adjective, instead of using it's existing context we now take advantage of the dependency
        # parser of Spacy to use all of it's ancestor as it's new context  
        if tok.pos_ == "ADJ":
            context = [anc.lower_ for anc in tok.ancestors]
            context.insert(len(context) // 2, tok.lower_)
            for w in context:
                words.append(w)
    
    ## Phase 3: Build dataset
    vocabulary_size = 2000
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    for word in words:
        index = dictionary.get(word, 0)
        data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    return data, dictionary, reversed_dictionary


def Compute_topk(model_file, input_adjective, top_k):
    gensim.scripts.glove2word2vec.glove2word2vec(model_file, "word2vec.w2v")
    model = gensim.models.KeyedVectors.load_word2vec_format("word2vec.w2v", binary=False)
    nlp = spacy.load('en')

    # In case the word is not in the dictionary
    try:
        possible_words = [input_adjective] + [i[0] for i in model.wv.most_similar(input_adjective, topn=top_k * 100)]
    except KeyError:
        return []
    
    doc = nlp(' '.join(possible_words))

    input_adjective_tag = doc[0].tag_
    if input_adjective_tag in ["JJ", "JJR", "JJS"]:
        target_tag = input_adjective_tag
    else:
        target_tag = "JJ"

    res = []
    i = 0
    while len(res) < top_k and i+1 < len(doc):
        next_word = doc[i+1]
        i += 1
        if (next_word.tag_ == target_tag):
            res.append(next_word.text)
            
    i = 0
    while len(res) < top_k and i+1 < len(doc):
        next_word = doc[i+1]
        i += 1
        if next_word.text not in res and next_word.tag_ in ["JJ", "JJR", "JJS"]:
            res.append(next_word.text)

    return res
    


















    
