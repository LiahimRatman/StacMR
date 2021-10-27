import data
import pickle
import os
from vocab import Vocabulary


# Some experiments with datasets
vocab_path = './vocab/'
workers = 10
data_name = 'f30k'
vocab_name = 'f30k_precomp'
crop_size = 224
batch_size = 128
use_restval = False
data_path = 'data_big'
max_len = 60
text_number = 15
text_dim = 300
vocab = pickle.load(open(os.path.join(
        vocab_path, '%s_vocab.pkl' % vocab_name), 'rb'))
print(vocab('cat'))
t_loader, v_loader = data.get_loaders(data_path, data_name, vocab, crop_size, batch_size, workers, use_restval, max_len, text_number, text_dim)
