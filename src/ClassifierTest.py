import dna
from TripleGAN import TripleGAN
from utils import *
import time

DEBUG_MODE = False
class ClassifierTest(TripleGAN):
    def __init__(self, sess, epoch, batch_size, unlabel_batch_size, z_dim, dataset_name,
                 nexamples, lr_d, lr_g, lr_c, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.test_set_size = 1000
        self.test_batch_size = 200
        self.model_name = "Classifier"  # name for checkpoint

        if self.dataset_name == 'dna':
            self.categories = np.asarray(['A', 'C', 'T', 'G'])

            self.input_height = 4  # One-hot encoding
            self.input_width = 500  # Input sequence length

            self.y_dim = 2  # Number of labels
            self.c_dim = 1  # "Colour" dimension

            # Learning rates: classifier
            self.lr_c = lr_c

            self.beta1 = 0.9  # C, exponential decay rate for the 1st moment estimates
            self.beta2 = 0.999  # C, exponential decay rate for the 2nd moment estimates
            # C, A small constant for numerical stability. This epsilon is "epsilon hat"
            # in the Kingma and Ba paper (in the formula just before Section 2.1),
            # not the epsilon in Algorithm 1 of the paper.
            self.epsilon = 1e-8
            self.decay_epoch = 100  # Point in epoch we start adding decay. if epoch >= decay_epoch, add decay. Note decay is hard coded.
            self.len_discrete_code = 4  # Think this is one-hot encoding for visualization ?

            self.data_X, self.data_y, self.test_X, self.test_y = self.init_data(nexamples)

            self.num_batches = len(self.data_X) // self.batch_size

        else:
            print("Dataset not supported.")
            raise NotImplementedError

        print("Initializing ClassifierTest with lr_c=%.3g" % (self.lr_c))

    def init_data(self, nexamples):
        print("Running ClassifierTest, so loading both positive and negative samples...")
        print("Train size (each label): %d, test size (total): %d" % (nexamples, self.test_set_size))
        return dna.prepare_data(
            nexamples, self.test_set_size, samples_to_use="both", test_both=True)  # trainX, trainY, testX, testY

    def build_model(self):
        input_dims = [self.input_height, self.input_width, self.c_dim]

        # Placeholders for learning rates for discriminator , generator and classifier
        self.tf_lr_c = tf.placeholder(tf.float32, name='lr_c')

        """ Graph Input """
        # Train sequences, shape: [batch_size, 4, 500, 1]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + input_dims, name='real_sequences')
        # Test sequences, shape: [test_batch_size, 4, 500, 1]
        self.test_inputs = tf.placeholder(tf.float32, [self.test_batch_size] + input_dims, name='test_sequences')

        # Test labels, shape: [test_batch_size, 2], one hot encoded
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        self.test_label = tf.placeholder(tf.float32, [self.test_batch_size, self.y_dim], name='test_label')

        """ Loss Function """
        # output of C for real images
        self.c_real_logits = self.classifier(self.inputs, is_training=True, reuse=False)
        correct_prediction_train = tf.equal(tf.argmax(self.c_real_logits, 1), tf.argmax(self.y, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32))

        c_loss_real = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.y, axis=1), logits=self.c_real_logits))

        # test loss for classify
        self.test_logits = self.classifier(self.test_inputs, is_training=False, reuse=False)
        true_labels = tf.argmax(self.test_label, 1)
        predictions = tf.argmax(self.test_logits, 1)
        correct_prediction = tf.equal(predictions, true_labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # get loss for classifier
        self.c_loss = c_loss_real

        """ Training """

        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()

        c_vars = [var for var in t_vars if 'classifier' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.c_optim = tf.train.AdamOptimizer(self.tf_lr_c, beta1=self.beta1, beta2=self.beta2,
                                                  epsilon=self.epsilon).minimize(self.c_loss, var_list=c_vars)

        """ Summary """
        c_loss_sum = tf.summary.scalar("c_loss", self.c_loss)

        # final summary operations
        self.c_sum = tf.summary.merge([c_loss_sum])

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        lr_c = self.lr_c

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = int(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            with open(self.get_files_location('lr_logs.txt'), 'r') as f:
                line = f.readlines()
                line = line[-1]
                lr_c = float(line.split()[2])
                print("lr_c: %.3g" % lr_c)
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...starting from epoch 0...")

        # loop for epoch
        start_time = time.time()

        for epoch in range(start_epoch, self.epoch):

            # lr_d and lr_g are unimportant - this is only  concerned about classifier
            _, _, lr_c = self.update_learning_rates(epoch, lr_d=1, lr_g=1, lr_c=lr_c)

            # One epoch loop
            for idx in range(start_batch_id, self.num_batches):
                batch_sequences = self.data_X[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_labels = self.data_y[idx * self.batch_size: (idx + 1) * self.batch_size]

                feed_dict = {
                    self.inputs: batch_sequences,
                    self.y: batch_labels,
                    self.tf_lr_c: lr_c,
                    self.test_inputs: np.concatenate([self.test_X[0:100], self.test_X[900:1000]]),
                    self.test_label: np.concatenate([self.test_y[0:100], self.test_y[900:1000]])
                }

                _, summary_str_c, c_loss, train_acc, test_acc = self.sess.run([self.c_optim, self.c_sum, self.c_loss, self.train_accuracy, self.accuracy], feed_dict=feed_dict)
                self.writer.add_summary(summary_str_c, counter)

                if DEBUG_MODE:
                    self.run_debug_statements(feed_dict)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, c_loss: %.8f, train_acc %.2f, test_acc: %.2f"
                      % (epoch, idx, self.num_batches, time.time() - start_time, c_loss, train_acc, test_acc))

            """ Measure accuracy (enhancers vs nonenhancers) of discriminator and save"""
            self.test_and_save_accuracy(epoch=epoch)

            """ Save learning rates to a file in case we wanted to resume later"""
            self.save_learning_rates(1, 1, lr_c)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            if epoch != 0 and epoch % 50 == 0:
                # save model
                self.save(self.checkpoint_dir, counter)

            # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size)

    def run_debug_statements(self, feed_dict):
        logits_r, argmax_logits, argmax_labels = self.sess.run([self.c_real_logits, tf.argmax(self.c_real_logits, 1), tf.argmax(self.y, 1)],
                                                           feed_dict=feed_dict)
        print("Logits for real examples:")
        print(logits_r)
        print("Argmax logits")
        print(argmax_logits)
        print("Argmax true labels")
        print(argmax_labels)
        print("True labels sum")
        print(sum(argmax_labels))

def main():
    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        clf = ClassifierTest(sess, epoch=400, batch_size=100, unlabel_batch_size=-1,
                             z_dim=-1, dataset_name="dna", nexamples=12000, lr_d=-1, lr_g=-1, lr_c=0.001,
                             checkpoint_dir='checkpoint', result_dir='result', log_dir='logs')

        # build graph
        clf.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        clf.train()
        print(" [*] Training finished!")


if __name__ == '__main__':
    main()
