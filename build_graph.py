import pickle
import random
import sys
import time

import numpy as np
import scipy.sparse as sp
from math import log, exp
from sklearn.decomposition import TruncatedSVD

from util import read_param


def dump_data(config, data_name, data):
    with open(config + data_name, 'wb') as f:
        pickle.dump(data, f)


def save_graph(config, ls_adj, feature_list, word_freq_list, y, y_hot, train_size, word_vectors, positions_list):
    # dump_data(config, 'ls_adj', ls_adj)
    # dump_data(config, 'feature_list', feature_list)
    # dump_data(config, 'word_freq_list', word_freq_list)
    # dump_data(config, 'y', y)
    # dump_data(config, 'y_hot', y_hot)
    # dump_data(config, 'train_size', train_size)
    # dump_data(config, 'word_vectors', word_vectors)
    dump_data(config, 'all', (ls_adj, feature_list, word_freq_list, y, y_hot, train_size, word_vectors, positions_list))


def read_data(config, dataname):
    f = open(config + dataname, 'rb')
    return pickle.load(f)


def retrieve_graph(config):
    # ls_adj = read_data(config, 'ls_adj')
    # feature_list = read_data(config, 'feature_list')
    # word_freq_list = read_data(config, 'word_freq_list')
    # y = read_data(config, 'y')
    # y_hot = read_data(config, 'y_hot')
    # train_size = read_data(config, 'train_size')
    # word_vectors = read_data(config, 'word_vectors')
    ls_adj, feature_list, word_freq_list, y, y_hot, train_size, word_vectors, positions_list = read_data(config, 'all')
    if isinstance(word_vectors, int):
        word_vectors = np.identity(word_vectors)
    return ls_adj, feature_list, word_freq_list, y, y_hot, train_size, word_vectors, positions_list


def build_graph(config='param'):
    s_t = time.time()
    config_file = 'config/' + config + '.yaml'
    param = read_param(config_file)
    print(param, flush=True)
    dataset = param['dataset']

    if param['retrieve_graph']:
        return retrieve_graph('saved_graphs/data_' + config)

    import fasttext

    doc_name_list = []
    doc_train_list = []
    doc_test_list = []

    # Read meta data
    f = open('data/' + dataset + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split('\t')
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()

    # Read data from clean file
    doc_content_list = []
    f = open('data/corpus/' + dataset + '.clean.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()

    # Find train ids
    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    random.shuffle(train_ids)

    # Find test ids
    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    random.shuffle(test_ids)
    ids = train_ids + test_ids
    train_size = len(train_ids)
    total_size = len(ids)

    # Organize train and test data
    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])

    # Creating labels
    label_set = set()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    y = []
    y_hot = []
    for i in range(total_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for lb in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(label_index)
        y_hot.append(one_hot)
    y_hot = np.array(y_hot)
    y = np.array(y)
    print(label_list)

    # Creating word embeddding
    global_word_to_id = {}
    global_word_set = set()
    idf = {}
    for doc_words in shuffle_doc_words_list:
        tmp_word_set = set()
        words = doc_words.split()
        for word in words:
            global_word_set.add(word)
            if word not in tmp_word_set:
                if word in idf:
                    idf[word] += 1
                else:
                    idf[word] = 1
                tmp_word_set.add(word)

    for word in idf:
        # idf[word] = log(total_size / idf[word])
        idf[word] = log(total_size / (1 + idf[word])) + 1

    global_vocab = ['<pad>', '<unk>'] + list(global_word_set)
    global_vocab_size = len(global_vocab)

    for i in range(global_vocab_size):
        global_word_to_id[global_vocab[i]] = i

    if param['embed_type'] == 'identity':
        word_vectors = global_vocab_size

    elif param['embed_type'] == 'bert':
        from bert_embedding import BertEmbedding
        # bert_embedding = BertEmbedding(model='bert_24_1024_16')
        print('start bert ', int(time.time() - s_t))
        bert_embedding = BertEmbedding()
        result_tmp = bert_embedding(global_vocab)
        word_vectors = []
        for i in range(global_vocab_size):
            word_vectors.append(result_tmp[i][1][0])
        word_vectors = np.array(word_vectors)
        word_vectors[0] = np.zeros(word_vectors.shape[1], dtype='float32')
        print('end bert ', int(time.time() - s_t))

    elif param['embed_type'] == 'fast':
        print('start fast ', int(time.time() - s_t))
        # model = fasttext.train_unsupervised('data/corpus/' + dataset + '.clean.txt', dim=400)
        model = fasttext.load_model('../pretrained_models/model_300')
        word_vectors = []
        for i in range(global_vocab_size):
            word_vectors.append(model.get_word_vector(global_vocab[i]))
        model = None
        word_vectors = np.array(word_vectors)
        word_vectors[0] = np.zeros(word_vectors.shape[1], dtype='float32')
        print('end fast ', int(time.time() - s_t))

    elif param['embed_type'] == 'glove':
        filename = '../glove.840B.300d.txt'
        dim = 300
        count = 0
        word_vectors = np.random.uniform(-0.25, 0.25, (len(global_word_to_id), dim))
        with open(filename, encoding='utf-8') as fp:
            for line in fp:
                elements = line.strip().split()
                word = elements[0]
                if word in global_word_to_id:
                    try:
                        word_vectors[global_word_to_id[word]] = [float(v) for v in elements[1:]]
                        count += 1
                    except ValueError:
                        pass
        print('got embeddings of :', count)
        word_vectors[0] = np.zeros(dim, dtype='float32')

    elif param['embed_type'] == 'global_pmi':
        word_vectors = None

    else:
        print('Invalid word embd type')
        sys.exit()

    print('start adj creation ', int(time.time() - s_t))
    feature_list = []
    word_freq_list = []
    ls_adj = []
    window_size = param['window_size']
    positions_list = []
    # pmi_c = param['pmi_c'] * 1.0
    # pmi_c = (window_size - 1) * 1.0
    pmi_c = 1.0
    index = 0

    n_dropped_edges = 0
    total_edges = 0
    total_possible_edges = 0

    for doc_words in shuffle_doc_words_list:
        windows = []
        words = doc_words.split()

        # Find Word list and frequency
        word_freq = {}
        word_set = set()
        word_to_pos = {}
        for i, word in enumerate(words):
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
            if word in word_to_pos:
                word_to_pos[word].append(i)
            else:
                word_to_pos[word] = [i]

        vocab = list(word_set)
        vocab_size = len(vocab)

        total_possible_edges += vocab_size * (vocab_size - 1)
        features = []
        wf = []
        wf1 = []
        wf2 = []
        positions = []
        for i in range(vocab_size):
            features.append(global_word_to_id[vocab[i]])
            # wf.append(word_freq[vocab[i]] * idf[vocab[i]])
            wf1.append(word_freq[vocab[i]])
            wf2.append(idf[vocab[i]])
            positions.append(word_to_pos[vocab[i]])

        # features = np.array(features)
        feature_list.append(features)
        # word_freq_list.append(wf)
        word_freq_list.append((wf1, wf2))
        positions_list.append(positions)

        # Create map of word to id
        word_id_map = {}
        for i in range(vocab_size):
            word_id_map[vocab[i]] = i

        # # Create node attribute
        # node_size = vocab_size
        # row = []
        # col = []
        # val = []
        #
        # for key in word_freq:
        #     row.append(word_id_map[key])
        #     col.append(0)
        #     val.append(word_freq[key])
        # feat = sp.csr_matrix(
        #     (val, (row, col)), shape=(node_size, 1))
        # word_freq_list.append(feat)

        # Create windows
        # length = len(words)
        # if length <= window_size:
        #     windows.append(words)
        # else:
        #     for j in range(length - window_size + 1):
        #         window = words[j: j + window_size]
        #         windows.append(window)

        length = len(words)
        if length <= 1:
            windows.append(words)
        else:
            for j in range(1 - window_size, length):
                start = j
                end = j + window_size
                if start < 0:
                    start = 0
                if end > length:
                    end = length
                window = words[start: end]
                windows.append(window)

        # Find Word window frequency
        word_window_freq = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])

        # Count word pair
        word_pair_count = {}
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = word_id_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)

                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

        # Create adjacency matrix
        row = []
        col = []
        weight = []
        num_window = len(windows) + 2

        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = word_pair_count[key]
            word_freq_i = word_window_freq[vocab[i]]
            word_freq_j = word_window_freq[vocab[j]]

            pmi = log(.5 + ((pmi_c * count / num_window) /
                      (1.0 * word_freq_i * word_freq_j / (num_window * num_window))))
            if pmi <= 0:
                # print('dropped edge : ', vocab[i], ' ', vocab[j], ' ', exp(pmi))
                n_dropped_edges += 1
                continue

            # pmi = (pmi_c * count / num_window) / (1.0 * word_freq_i * word_freq_j / (num_window * num_window))
            # if pmi <= .7:
            #     # print('dropped edge : ', vocab[i], ' ', vocab[j], ' ', exp(pmi))
            #     n_dropped_edges += 1
            #     continue

            row.append(i)
            col.append(j)
            # weight.append(count)
            weight.append(pmi)
            total_edges += 1

        node_size = vocab_size
        adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

        ls_adj.append(adj)

        # Test
        # print(adj.shape)
        # if index in [0, 3]:
        #     print('Index : ', index)
        #     print(shuffle_doc_name_list[index])
        #     print(doc_words)
        #     print(word_id_map)
        #     print(feat)
        #     print(adj)
        index += 1
    print('end adj creation ', int(time.time() - s_t))
    print('start global adj creation ', int(time.time() - s_t))

    if param['embed_type'] == 'global_pmi':
        # Create global adj matrix
        windows_g = []
        window_size_g = param['window_size_g']
        # pmi_c_g = (window_size_g - 1) * 1.0
        pmi_c_g = 1.0

        for doc_words in shuffle_doc_words_list:
            words = doc_words.split()
            length = len(words)
            # if length <= 1:
            #     continue
            #     # windows_g.append(words)
            if length > 1:
                for j in range(1 - window_size_g, length):
                    start = j
                    end = j + window_size_g
                    if start < 0:
                        start = 0
                    if end > length:
                        end = length
                    window = words[start: end]
                    windows_g.append(window)

        word_window_freq = {}
        for window in windows_g:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])

        word_pair_count = {}
        for window in windows_g:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = global_word_to_id[word_i]
                    word_j = window[j]
                    word_j_id = global_word_to_id[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

        row = []
        col = []
        weight = []

        # pmi as weights
        num_window = len(windows_g) + 2

        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = word_pair_count[key]
            word_freq_i = word_window_freq[global_vocab[i]]
            word_freq_j = word_window_freq[global_vocab[j]]
            pmi = log((pmi_c_g * count / num_window) /
                      (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))

            if pmi <= 0:
                continue
            row.append(i)
            col.append(j)
            weight.append(pmi)

        adj_g = sp.csr_matrix(
            (weight, (row, col)), shape=(global_vocab_size, global_vocab_size))
        print('end global adj creation ', int(time.time() - s_t))
        print('start svd ', int(time.time() - s_t))
        svd = TruncatedSVD(n_components=400, n_iter=7, random_state=42)
        # word_vectors = adj_g + sp.identity(adj_g.shape[0])
        word_vectors = svd.fit_transform(adj_g)
        print('end svd ', int(time.time() - s_t))

    print('total docs : ', len(ls_adj))
    print('total edges : ', total_edges)
    print('total_possible_edges : ', total_possible_edges)
    print('total dropped edges : ', n_dropped_edges)

    if param['save_graph']:
        save_graph('saved_graphs/data_' + config, ls_adj, feature_list, word_freq_list, y,
                   y_hot, train_size, word_vectors, positions_list)

    if isinstance(word_vectors, int):
        word_vectors = np.identity(word_vectors)

    return ls_adj, feature_list, word_freq_list, y, y_hot, train_size, word_vectors, positions_list


def main():
    ls_adj, feature_list, word_freq_list, y, y_hot, train_size = build_graph()
    # Test
    print('Shape of word frequency list index 10 : ', word_freq_list[10].shape)
    print('Shape of adj index 10 : ', ls_adj[10].shape)
    print('Size of features : ', len(word_freq_list))
    print('Size of adjacency : ', len(ls_adj))
    print('Size of feature list : ', len(feature_list))
    print('Shape of features index 10 : ', feature_list[10].shape)


if __name__ == '__main__':
    main()
