# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#Note: Original Source Code from https://github.com/tensorflow/tensorflow.git
#tensorflow/models/rnn/ptb/reader.py

"""Utilities for parsing PTB text files"""
import collections
import tensorflow as tf
import os


def _read_words(filename):
	#Why do we need tf.gfile? Why not simple open()?
	with tf.gfile.GFile(filename, 'r') as f:
		return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
	data = _read_words(filename)
	#now we have data as a biig list of words

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	# [('how', 298), ('Hi', 178), ('about', 10), ('are', 1)...]

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))

	#Only return the dictionary: word -> int_id
	return word_to_id

def _file_to_word_ids(filename, word_to_id):
	data = _read_words(filename)
	#The whole text document in int_ids	
	return [word_to_id[word] for word in data]

def ptb_raw_data(data_path=None):
	"""Load PTB raw data from data dir "data_path".

	Reads PTB text files, converts strings to int ids
	and performs mini-batching of inputs	

	Args:
	    data_path: dir str where simple-examples.tgz was extracted
	Returns:
	     tuple(train_data, valid_data, test_data, vocabulary)
		where each of data objects can be passed to PTBIterator
	"""

	train_path = os.path.join(data_path, "ptb.train.txt")
	valid_path = os.path.join(data_path, "ptb.valid.txt")
	test_path = os.path.join(data_path, "ptb.test.txt")

	word_to_id = _build_vocab(train_path)

	train_data = _file_to_word_ids(train_path, word_to_id)	
	#train_data is whole ptb.train.txt in int_ids

	valid_data = _file_to_word_ids(valid_path, word_to_id)
	#What happens if words not in word_to_id?
	test_data = _file_to_word_ids(test_path, word_to_id)
	vocabulary = len(word_to_id)	#Size of vocabulary
	return train_data, valid_data, test_data, vocabulary

def ptb_iterator(raw_data, batch_size, num_steps):
	"""Iterate on raw PTB data.

	This generates batch_size pointers into raw PTB data, and allowd
	minibatch iteration along these pointers

	Args:
	   raw_data: one of raw data outputs from ptb_raw_data (train_data, valid_data...)
	   batch_size: int   (batch size)
	   num_steps: int, num of unrolls (??? number of LSTM cells ???)


	Outputs:
	  Pairs of batched data, each a matrix of shape [batch_size, num_steps]
	  2nd element of tuple is same data time-shifted to the right by one
	  ?
	  input: raw_data:'the cow jumped over the moon'
		 batch_size: 4
		 num_steps:3
	  output: (4x3 matrix that corresponds to below?)
		 the cow jumped
		 cow jumped over
		 jumped over the
		 over the moon
	Raises:
	  ValueError: if batch_size or num_steps are too high
	"""
	raw_data = np.array(raw_data, dtype=np.int32)

	data_len = len(raw_data)   #Eg: 10,000
	batch_len = data_len // batch_size #Eg: 10,000/4 = 2500  ? 

	data = np.zeros([batch_size, batch_len], dtype=np.int32)
				# 4x2500 ??

	for i in range(batch_size):	#range(4)
		data[i] = raw_data[batch_len*i:batch_len*(i+1)]
				  #2500*0:2500*(1)  = [0:2500]
				  #2500*1:2500*2    = [2500:5000]
				  #                   [5000:7500]
				  #                   [7500:10000]
				  #Why is each batch so big?
				  #possible raw_data is not same size as train_data??

	epoch_size = (batch_len - 1) // num_steps     #2500-1/3 = 833

	if epoch_size == 0:
		raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

	for i in range(epoch_size):
		x = data[:, i*num_steps:(i+1)*num_steps]        # x = data[:, 0:3]
								# x = data[:, 3:6]
								# x = data[:, 6:9]
								# ..
								# x = data[:, 833*3:834*3

		y = data[:, i*num_steps+1:(i+1)*num_steps+1]    # y = data[:, 1:4]
								# y = data[:, 4:6]
								# ..
								# y = data[:, 833*3+1 : 834*3+1
		yield(x, y)
								#Each yeild x of type 4x3, y of type 4x3
								# epoch_size(833) such yeilds

								# Why does y need 1:4? Shouldn't just [4] be enought?
								

