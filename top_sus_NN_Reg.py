import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import itertools
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = os.getcwd()

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

class Topological_NN(object):
    def __init__(self, dataF, h_nodes=20, epochs=1000, step_s=1e-4, reg_scale=1e-1):
        self.h_nodes = h_nodes
        self.epochs = epochs
        self.step_s = step_s
        self.dataF = dataF
        self.input_features = 16
        
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)

        self.dir_name = path + '/MetaGraphs/'
        self.fileN = self.dir_name + 'Topological_jump__Hnodes_{:.0f}_Ssize_{:.0e}'.format(h_nodes, step_s)

        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
    
        self.train_X, self.test_X, self.train_y, self.test_y = self.get_data()
        self.x_size = self.train_X.shape[1]
        self.y_size = self.train_y.shape[1]
        
    def init_weights(self, shape):
        weights = tf.random_normal(shape, stddev=0.2)
        return tf.Variable(weights)

    def forwardprop(self, X, w_1, w_2, w_3):
        hid1 = tf.nn.sigmoid(tf.matmul(X, w_1))
        hid2 = tf.nn.sigmoid(tf.matmul(hid1, w_2))
        yhat = tf.matmul(hid2, w_3)
        return yhat

    def get_data(self, frac_test=0.3):
        self.scalar = StandardScaler()
        
        data_file = path + '/data/' + self.dataF
        full_dat = np.loadtxt(data_file)
        np.random.shuffle(full_dat)
        
        input_v = full_dat[:, :self.input_features]
        output_v = np.abs(full_dat[:, self.input_features:])
        
        std_input_v = self.scalar.fit_transform(input_v)
        self.train_size = int((1.-frac_test)*len(input_v))
        self.test_size = int(frac_test*len(input_v))
        
        N, M  = input_v.shape
        all_X = np.ones((N, M + 1))
        all_X[:, 1:] = std_input_v
        return train_test_split(all_X, output_v, test_size=frac_test, random_state=RANDOM_SEED)

    def make_nn(self):
        
        self.X = tf.placeholder("float", shape=[None, self.x_size], name='X')
        self.y = tf.placeholder("float", shape=[None, self.y_size])
        
        self.w_1 = self.init_weights((self.x_size, self.h_nodes))
        self.w_2 = self.init_weights((self.h_nodes, self.h_nodes))
        self.w_3 = self.init_weights((self.h_nodes, self.y_size))

        self.yhat = tf.round(self.forwardprop(self.X, self.w_1, self.w_2, self.w_3)) # Note forcing integer...

        self.cost = tf.reduce_sum(tf.square(self.y - self.yhat)) # Consider alternate cost func
        
        reg_term = tf.contrib.layers.apply_regularization(self.regularizer, [self.w_1, self.w_2, self.w_3])
        self.cost += reg_term

        tf.add_to_collection("activation", self.yhat)
        #self.accuracy = tf.reduce_sum(tf.square(self.y - self.yhat))

        self.updates = tf.train.GradientDescentOptimizer(self.step_s).minimize(self.cost)
        self.saveNN = tf.train.Saver()
        return

    def trainn_NN(self, keep_training=False, btch_sze=20):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if keep_training:
                self.saveNN.restore(sess, self.fileN)
                print 'Model Restored.'
            BATCH_SIZE = btch_sze
            train_count = len(self.train_X)
            for i in range(1, self.epochs + 1):
                for start, end in zip(range(0, train_count, BATCH_SIZE),
                                      range(BATCH_SIZE, train_count + 1,BATCH_SIZE)):

                    sess.run(self.updates, feed_dict={self.X: self.train_X[start:end],
                                                      self.y: self.train_y[start:end]})

                if i % 100 == 0:
                    train_accuracy = sess.run(self.cost, feed_dict={self.X: self.train_X, self.y: self.train_y})
                    test_accuracy = sess.run(self.cost, feed_dict={self.X: self.test_X, self.y: self.test_y})
                    print("Epoch = %d, train accuracy = %.7e, test accuracy = %.7e"
                          % (i + 1, train_accuracy / len(self.train_X), test_accuracy / len(self.test_X)))
                          
            self.saveNN.save(sess, self.fileN)
        return


class ImportGraph():
    def __init__(self, metaFile, dataFile):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(metaFile + '.meta')
            saver.restore(self.sess, metaFile)
            self.activation = tf.get_collection('activation')[0]

        self.scalar = StandardScaler()

        dataIN = np.loadtxt(path + '/data/' + dataFile)
        input_v = dataIN[:, :-1]
        std_input_v = self.scalar.fit_transform(input_v)
        return
    
    def run_yhat(self, data):
        inputV = np.insert(self.scalar.transform(data), 0, 1., axis=1)
        return self.sess.run(self.activation, feed_dict={"X:0": inputV})

