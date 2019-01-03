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
        
        if reg_scale > 1e-10:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
            self.regularize = True
        else:
            self.regularize = False

        self.dir_name = path + '/MetaGraphs/'
        self.fileN = self.dir_name + 'Schwinger_' + self.dataF + '_Hnodes_{:.0f}_Ssize_{:.0e}'.format(h_nodes, step_s)

        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
    
        self.train_X, self.test_X, self.train_y, self.test_y = self.get_data()
        self.x_size = self.train_X.shape[1]
        self.y_size = self.train_y.shape[1]
        
    def init_weights(self, shape):
        weights = tf.random_normal(shape, stddev=self.std_dev_init) # 0.2 works well for topQ
        return tf.Variable(weights)

    def forwardprop(self, X, w_1, w_2, w_3, w_4):
        hid1 = tf.nn.sigmoid(tf.matmul(X, w_1))
        hid2 = tf.nn.sigmoid(tf.matmul(hid1, w_2))
        hid3 = tf.nn.sigmoid(tf.matmul(hid2, w_3))
        yhat = tf.matmul(hid3, w_4)
        return yhat

    def get_data(self, frac_test=0.5):
        print 'Loading Data Files....'
        self.scalar = StandardScaler()
        
        input_data_file = path + '/NN_data/Schwinger_' + self.dataF + '_input_data.dat'
        output_data_file = path + '/NN_data/Schwinger_' + self.dataF + '_output_data.dat'
        
        input_data = np.loadtxt(input_data_file)
        output_data = np.loadtxt(output_data_file)
        
        lattice_dim = 18

        if self.dataF == 'fermion_determinant':
            output_features = 1
            self.round_acc = False
            self.std_dev_init = 200.
        elif self.dataF == 'top_charge':
            output_features = 1
            self.round_acc = True
            self.std_dev_init = 0.05
        elif self.dataF == 'pion_correlator':
            output_features = lattice_dim
            self.round_acc = False
            self.std_dev_init = 0.05
        
        self.input_features = len(input_data) / len(output_data)
        
        input_data = input_data.reshape(len(output_data), self.input_features)
        full_dat = np.column_stack((input_data, output_data))
        
        np.random.shuffle(full_dat)
        
        input_v = full_dat[:, :self.input_features]
        output_v = full_dat[:, self.input_features:]
        
        std_input_v = self.scalar.fit_transform(input_v)
     
        self.train_size = int((1.-frac_test)*len(input_v))
        self.test_size = int(frac_test*len(input_v))
        
        N, M  = input_v.shape
        all_X = np.ones((N, M + 1))
        all_X[:, 1:] = std_input_v
        print 'Finished Preparing Data Files....'
        return train_test_split(all_X, output_v, test_size=frac_test, random_state=RANDOM_SEED)

    def make_nn(self):
        
        self.X = tf.placeholder("float", shape=[None, self.x_size], name='X')
        self.y = tf.placeholder("float", shape=[None, self.y_size])
        
        self.w_1 = self.init_weights((self.x_size, self.h_nodes))
        self.w_2 = self.init_weights((self.h_nodes, self.h_nodes))
        self.w_3 = self.init_weights((self.h_nodes, self.h_nodes))
        self.w_4 = self.init_weights((self.h_nodes, self.y_size))

        self.yhat = self.forwardprop(self.X, self.w_1, self.w_2, self.w_3, self.w_4)

        self.cost = tf.reduce_sum(tf.square(self.y - self.yhat)) # Could consider alternate cost func
        
        
        if self.regularize:
            reg_term = tf.contrib.layers.apply_regularization(self.regularizer, [self.w_1, self.w_2, self.w_3])
            self.cost += reg_term
        
        if self.round_acc:
            self.accur = tf.reduce_sum(tf.cast(tf.equal(self.y, tf.round(self.yhat)), tf.float32))
        else:
            # TEST
            self.accur = tf.reduce_sum(tf.abs(self.y - self.yhat))
        
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

                if i % int(self.epochs / 10.) == 0:
                    train_accuracy = sess.run(self.accur, feed_dict={self.X: self.train_X, self.y: self.train_y})
                    test_accuracy = sess.run(self.accur, feed_dict={self.X: self.test_X, self.y: self.test_y})
                    print("Epoch = %d, train accuracy = %.7e, test accuracy = %.7e"
                          % (i + 1, float(train_accuracy) / len(self.train_X), float(test_accuracy) / len(self.test_X)))
                          
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
        
        input_data_file = path + '/NN_data/Schwinger_' + dataFile + '_input_data.dat'
        output_data_file = path + '/NN_data/Schwinger_' + dataFile + '_output_data.dat'
        input_data = np.loadtxt(input_data_file)
        output_data = np.loadtxt(output_data_file)
        
        lattice_dim = 18

        if dataFile == 'fermion_determinant':
            output_features = 1
        elif dataFile == 'top_charge':
            output_features = 1
        elif dataFile == 'pion_correlator':
            output_features = lattice_dim
        
        input_features = len(input_data) / len(output_data)
        
        
        input_data = input_data.reshape(len(output_data), input_features)
        full_dat = np.column_stack((input_data, output_data))
        
        input_v = full_dat[:, :input_features]
        std_input_v = self.scalar.fit_transform(input_v)
        return
    
    def run_yhat(self, data, round=False):
        inputV = np.insert(self.scalar.transform(data), 0, 1., axis=1)
        val = self.sess.run(self.activation, feed_dict={"X:0": inputV})
        
        return val

