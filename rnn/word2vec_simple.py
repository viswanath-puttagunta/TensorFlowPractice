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
#tensorflow/examples/tutorials/word2vec/word2vec_simple.py

import os
from six.moves import urllib 
from six.moves import xrange
import zipfile
import collections
import numpy as np
import random
import tensorflow as tf
import math

# Step 1: Download the data
url = 'http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
	"""Download file if not present, make sure it is right size"""
	if not os.path.exists(filename):
		filename, _ = urllib.request.urlretrieve(url + filename, filename)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified', filename)
	else:
		print(statinfo.st_size)
		raise Exception(
			'Failed to verify ' + filename + '. Can you get with browser?')
	return filename

filename = maybe_download('text8.zip', 31344016)

#Read the data into list of strings
def read_data(filename):
	"""Extract first file enclosed in zip file as list of words"""
	with zipfile.ZipFile(filename) as f:
		data = f.read(f.namelist()[0]).split()
	return data

words = read_data(filename)
print('Data size', len(words))

#Step 2: Build the directory and replace rare words with UNK token
vocabulary_size = 50000

def build_dataset(words):
	#Get the top 50,000 words and build dictionary
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word,_ in count:
		dictionary[word] = len(dictionary)	
	

	data = list()    #essentially 'words', but has integer index instead of strings
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0	#dictionary['UNK']
			unk_count +=1
		data.append(index)
	count[0][1] = unk_count

	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print ('First few words', words[:10])
del words
print ('Most common words [+UNK]', count[:5])
print ('Sample data', data[:10])

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model
		  #   8          2            1
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window

  batch = np.ndarray(shape=(batch_size), dtype=np.int32)   #8
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32) #8x1
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]  # 3
  buffer = collections.deque(maxlen=span)   # 3 words at a time?

  for _ in range(span):
    buffer.append(data[data_index])         #buffer = ['anarchism', 'originated', 'as']??
    data_index = (data_index + 1) % len(data)

  for i in range(batch_size // num_skips):  #8//2 = 4
    target = skip_window  # target label at the center of the buffer: target = 1   from 0 1 2
    targets_to_avoid = [ skip_window ]    # [ 1 ]
    for j in range(num_skips):   #range(2): 0,1
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)   # pick randint from 0 - 2 (Ex: 0)
      targets_to_avoid.append(target)     # [1 0]
      batch[i * num_skips + j] = buffer[skip_window]   #batch[0] = buffer[1]
							#batch[1] = buffer[1]
      labels[i * num_skips + j, 0] = buffer[target]	#labels[0,0] = buffer[0]
							#labels[1,0] = buffer[0]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])


"""
['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse'
generates batch of 8 below: batch[i] -> label[i,0]

('originated', '->', 'anarchism')
('originated', '->', 'as')

('as', '->', 'a')
('as', '->', 'originated')

('a', '->', 'as')
('a', '->', 'term')

('term', '->', 'of')
('term', '->', 'a')

(target, left_word/right_word)
(target, right_word/left_word) 
"""

#Step 4: Build and train a skip-gram model
batch_size = 128
embedding_size = 128   #128 Dimensions for embedding vector
skip_window = 1		# How many words to consider left/right
num_skips = 2		#How many times to reuse an input to generate a label

#We pick random validations et to sample nearest neightbors.
# Limit validations samples to words that have low numeric ID, which by contruction are
# are also most frequent
valid_size = 16	#Random set of words to evaluate similarity on
valid_window = 100  # Only pick dev samples in the head of distribution
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64     #Number of negative examples to sample
			#This is the K noise words it selects for each batch

graph = tf.Graph()

with graph.as_default():
	#Input data
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])   #Output has to be matrix?
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	#Ops and variables pinned to CPU because of missing GPU implementation
	with tf.device('/cpu:0'):
		#Lkup embeddings for inputs
		embeddings = tf.Variable(
			tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
					# [50000x128] generate random floats between -1 and +1
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)
			#for 128 batch that is pickked, look up vector representations
		
		#Construct variables for NCE loss
		nce_weights = tf.Variable(
				tf.truncated_normal([vocabulary_size, embedding_size],
							stddev=1.0/math.sqrt(embedding_size)))
				#Picking 50,000 weight vectors(128 dimn)
				# with std dev inversely proportional to embedding_size(128)

		nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
				#picking 50,000 biases: single numbers! not vector

	#Compute avg NCE loss for the batch
	# tf.nce_loss auto draws new sample of negative labels each time we evaluate the loss
	loss = tf.reduce_mean(
			tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
					num_sampled, vocabulary_size))
					#num_sampled??? num of -ve examples to sample???
					#Why not converting train_labels into vectors?


	#Construct SGD optimizer using a learning rate of 1.0
	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	#Compute cosine similarity between minibatch examples and all embeddings
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
		#Normalizes each vector.. we'll still have 50,000 vectors
		#[50000x128]

	valid_embeddings = tf.nn.embedding_lookup(
				normalized_embeddings, valid_dataset)
			#gets vectors(now normalized) for 16 words we monitor.
			# [16x128]

	similarity = tf.matmul(
			valid_embeddings, normalized_embeddings, transpose_b=True)

			# [16x128] * [128x50000] = [16x50000]
			# gives how close the 16 words we picked are to the 50,000 words?
			# guess this is doing dot product of each choice word with 50,000 words?
			# and dot product is kind of measuring cosine distances

#Setp 5: Begin training
num_steps = 100001

with tf.Session(graph=graph) as session:
	#We must init all variables before we use them
	tf.initialize_all_variables().run()
	print("Initialized")

	average_loss = 0
	for step in xrange(num_steps):
		batch_inputs, batch_labels = generate_batch(
						batch_size, num_skips, skip_window)
		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

		#We perform one update step by evaluating optimzer op (including it in list
		# of returned values for session.run()
		_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val

		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000
			#The evg loss is an estimate of loss over last 2000 batches
			print("Average loss at step ", step, ": ", average_loss)

		#Note that this is expensive (~20% slowdown if computed every 500 steps)
		if step % 10000 == 0:
			sim = similarity.eval()
				#Why not using session.run()? How does eval work?
				# sim = session.run(similarity)   work the same?
			for i in xrange(valid_size):
				valid_word = reverse_dictionary[valid_examples[i]]
				top_k = 8 # number of nearest neighbors
				nearest = (-sim[i,:]).argsort()[1:top_k+1]
				log_str = "Nearest to %s:," %valid_word
				for k in xrange(top_k):
					close_word = reverse_dictionary[nearest[k]]
					log_str = "%s %s," % (log_str, close_word)
				print(log_str)
	final_embeddings = normalized_embeddings.eval()
