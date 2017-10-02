# Sentiment analysis of 25,000 IMDB  movie reviews
# Dataset obtained from http://ai.stanford.edu/~amaas/data/sentiment/
# Model is a 2 layer neural network implemented in Python

import numpy as np
import time
import sys
from collections import Counter

# Load reviews and labels respectively
fp = open('reviews.txt','r') 
reviews = list(map(lambda a:a[:-1],fp.readlines()))
fp.close()

fp = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda a:a[:-1].upper(),fp.readlines()))
fp.close()

# Intitialize counters
pos_word_count = Counter()
neg_word_count = Counter()
total_count = Counter()


for i in range(len(reviews)):
    lb = labels[i]
    rv = reviews[i]
    sp_rv = rv.split(' ')
    if (lb == 'POSITIVE'):
        for word in sp_rv:
            pos_word_count[word] += 1
            total_count[word] += 1
    else:
        for word in sp_rv:
            neg_word_count[word] += 1
            total_count[word] += 1
            

# Counter for positive-to-negative ratios
pos_neg_ratios = Counter()

# Ratio of most common words in positive and negative reviews
# Set threshold of a "common" word to 100
for word in pos_word_count:
    if pos_word_count[word]>=100:
        pos_neg_ratios[word] = pos_word_count[word] / float(neg_word_count[word] + 1)
        
        
# Convert ratios to logs
for ratio in pos_neg_ratios:
    pos_neg_ratios[ratio] = np.log(pos_neg_ratios[ratio])
    

# Map workds to ino index in a dictionary
vocab = set(total_count)
word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i
    

class sentiment_analysis_nn:
    def __init__(self, reviews, labels, hidden_layer_size = 10, learning_rate = 0.1):
        """
        Neural network class
        """
        np.random.seed(10)
        self.preprocess_data(reviews, labels)
        self.initialize_nn(len(self.review_vocab),hidden_layer_size, 1, learning_rate)

    # Create counters and dictionaries and preprocess input data
    def preprocess_data(self, reviews, labels):
        
        review_vocab = set()
        vocab_cnt = Counter()
        for review in reviews:
            sp_rv = review.split(' ')
            for word in sp_rv:
                vocab_cnt[word] += 1                
        review_vocab = set(vocab_cnt) 
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        label_cnt = Counter()
        for lb in labels:
            label_cnt[lb] += 1            
        label_vocab = set(label_cnt)            
        self.label_vocab = list(label_vocab)
        
        self.review_vocab_size = len(review_vocab)
        print("#unique words in all reviews = ",self.review_vocab_size)
        self.label_vocab_size = len(label_vocab)
        print("#unique labels = ",self.label_vocab_size)
        
        self.word2index = {}
        for i,word in enumerate(self.review_vocab):
            self.word2index[word] = i

        self.label2index = {}
        for i,word in enumerate(self.label_vocab):
            self.label2index[word] = i

    # Set up 2 layer neural network    
    def initialize_nn(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.learning_rate = learning_rate
        self.weights_0_1 = np.zeros((self.input_layer_size,self.hidden_layer_size))
        self.weights_1_2 = np.random.normal(0.0, self.hidden_layer_size**-0.5,(self.hidden_layer_size, self.output_layer_size))
        self.hidden_layer_inputs = np.zeros((1,hidden_layer_size)) 
        
       
    # Encode positive as 1 and negative as 0         
    def encode_label(self,label):
        if (label == "POSITIVE"):
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))
    
    def sigmoid_derivative(self,output):
        return (output*(1-output))
    
    def update_input_layer(self,review):
        self.layer_0 *= 0
        for word in review.split(" "):
            if(word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]] = 1
        
    # Perform forward and backward propagations to train model    
    def train(self, training_reviews_orig, training_labels):
        
        assert(len(training_reviews_orig) == len(training_labels))
        correct_predictions = 0
        
        training_reviews = list()
        for rev in training_reviews_orig:
            sp_rev = rev.split(' ')
            tmp_list = set()
            for word in sp_rev:
                if(word in self.word2index.keys()):
                    tmp_list.add(self.word2index[word])
            training_reviews.append(list(tmp_list))
                   
        start = time.time()

        for i in range(len(training_reviews)):
            indices = training_reviews[i]
            y = self.encode_label(training_labels[i])
         
            # Forward prop
            self.hidden_layer_inputs*=0
            for index in indices:
                self.hidden_layer_inputs += self.weights_0_1[index]
                
            hidden_layer_outputs = self.hidden_layer_inputs    
            final_layer_inputs = np.dot(hidden_layer_outputs,self.weights_1_2)
            final_layer_outputs = self.sigmoid(final_layer_inputs)            
                        
            # Backprop
            error = y - final_layer_outputs
            output_error_term = error*(final_layer_outputs)*(1-final_layer_outputs)
            
            hidden_error = np.dot(output_error_term,self.weights_1_2.T)
            hidden_error_term = hidden_error*1
            
            self.weights_1_2 += self.learning_rate*np.dot(hidden_layer_outputs.T,output_error_term)
                        
            for index in indices:
                self.weights_0_1[index] += self.learning_rate*hidden_error_term[0]
            
            if(final_layer_outputs >= 0.5 and training_labels[i] == 'POSITIVE'):
                correct_predictions += 1
            elif(final_layer_outputs < 0.5 and training_labels[i] == 'NEGATIVE'):
                correct_predictions += 1
                

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_predictions) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_predictions * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    # Predict and test accuracies
    def test(self, test_reviews, test_labels):

        correct = 0
        start = time.time()

        for i in range(len(test_reviews)):
            pred = self.run_nn(test_reviews[i])
            if(pred == test_labels[i]):
                correct += 1
            
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(test_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
            
    def run_nn(self, review):
        
        self.hidden_layer_inputs*=0
        tmp_list = []
        for word in review.lower().split(' '):
                if(word in self.word2index.keys()):
                    tmp_list.append(self.word2index[word])
            
        for index in tmp_list:
            self.hidden_layer_inputs += self.weights_0_1[index]
                
        hidden_layer_outputs = self.hidden_layer_inputs    
        final_layer_inputs = np.dot(hidden_layer_outputs,self.weights_1_2)
        final_layer_outputs = self.sigmoid(final_layer_inputs)            

        if final_layer_outputs >= 0.5:
            return "POSITIVE"
        else:
            return "NEGATIVE"
        
# Create object and train on 24,000 reivews        
sent_obj = sentiment_analysis_nn(reviews[:-1000],labels[:-1000], learning_rate=0.1)
sent_obj.train(reviews[:-1000],labels[:-1000])

# Test on 1000 reviews
sent_obj.test(reviews[-1000:],labels[-1000:])
