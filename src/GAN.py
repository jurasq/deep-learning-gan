from __future__ import print_function
import dna
from ops import *
from TripleGAN import TripleGAN
from utils import *
import time


class GAN(TripleGAN):
    def __init__(self, sess, epoch, batch_size, unlabel_batch_size, z_dim, dataset_name,
                 nexamples, lr_d, lr_g, lr_c, checkpoint_dir, result_dir, log_dir):
        TripleGAN.__init__(self, sess, epoch, batch_size, unlabel_batch_size, z_dim, dataset_name,
                           nexamples, lr_d, lr_g, lr_c, checkpoint_dir, result_dir, log_dir)
        self.model_name = "dGAN"  # for checkpoint
        self.alpha = 0  # #so that discriminator loss = D_loss_real + D_loss_fake

        print("Initializing GAN with lr_d=%.3g, lr_g=%.3g" % (self.lr_d, self.lr_g))

    # Called in the super class
    def init_data(self, nexamples):
        print("Loading only positive samples...")
        return dna.prepare_data(
            nexamples, self.test_set_size, samples_to_use="pos", test_both=True)  # trainX, trainY, testX, testY

    def discriminator(self, dna_sequence, y=None, scope="discriminator", is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            # convolutional with reverse filters like in paper + pooling #1
            l1 = conv_max_forward_reverse(name_scope="conv1", input_tensor=dna_sequence, num_kernels=20,
                                          kernel_shape=[4, 9], relu=True, is_training=is_training)
            l2 = max_pool_layer(name_scope="pool1", input_tensor=l1, pool_size=[1, 3])

            # convolutional + pooling #2
            l3 = conv_layer(name_scope="conv2", input_tensor=l2, num_kernels=30,
                            kernel_shape=[1, 5], relu=True, is_training=is_training)
            l4 = max_pool_layer(name_scope="pool2", input_tensor=l3, pool_size=[1, 4])

            # convolutional + pooling #3
            l5 = conv_layer(name_scope="conv3", input_tensor=l4, num_kernels=40,
                            kernel_shape=[1, 3], relu=True, is_training=is_training)
            l6 = max_pool_layer(name_scope="pool3", input_tensor=l5, pool_size=[1, 4])

            flat = flatten(l6)
            # fully connected layers
            l7 = tf.layers.dense(inputs=flat, units=90)
            l8 = tf.layers.dense(inputs=l7, units=45)

            logits = tf.layers.dense(inputs=l8, units=2)  # 2 units, ie. probability of each class (fake, real)
            softmax = tf.nn.softmax(logits)

        # previously used only logits, now returning the 3 for compatibility with triple gan
        return l8, logits, softmax

    def generator(self, noise_vector, y=None, scope="generator", is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            batch_size = tf.cast(noise_vector.shape[0], dtype=tf.int32)
            g_dim = 64  # Number of filters of first layer of generator
            c_dim = 1  # dimensionality of the output
            s = 500  # Final length of the sequence

            # We want to slowly upscale the sequence, so these values should help
            # to make that change gradual
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

            width = 4  # because we have 4 letters: ATCG

            # this is a magic number which I'm not sure what means yet
            magic_number = 5

            # output_mlp = mlp('mlp', noise_vector, batch_norm=False, relu=False, is_training=is_training)
            output_mlp = tf.layers.dense(inputs=noise_vector, units=int(width / 4)* (s16 + 1) * magic_number)
            h0 = tf.reshape(output_mlp, [batch_size, int(width / 4), s16 + 1, magic_number])
            h0 = tf.nn.relu(h0)
            # Dimensions of h0 = batch_size x 1 x 31 x magic_number

            # First DeConv Layer
            output1_shape = [batch_size, int(width / 2), s8 + 1, g_dim * 4]
            W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

            b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))

            H_conv1 = tf.nn.conv2d_transpose(value=h0, filter=W_conv1, output_shape=output1_shape,
                                             strides=[1, 2, 2, 1], padding='SAME', name="H_conv1") + b_conv1
            H_conv1 = tf.contrib.layers.batch_norm(inputs=H_conv1, center=True, scale=True, is_training=True,
                                                   scope="g_bn1")
            H_conv1 = tf.nn.relu(H_conv1)
            # Dimensions of H_conv1 = batch_size x 1 x 62 x 256

            # Second DeConv Layer
            output2_shape = [batch_size, int(width / 2), s4, g_dim * 2]
            W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape,
                                             strides=[1, 1, 2, 1], padding='SAME') + b_conv2
            H_conv2 = tf.contrib.layers.batch_norm(inputs=H_conv2, center=True, scale=True, is_training=True,
                                                   scope="g_bn2")
            H_conv2 = tf.nn.relu(H_conv2)
            # Dimensions of H_conv2 = batch_size x 2 x 124 x 128

            # Third DeConv Layer
            output3_shape = [batch_size, int(width), s2, g_dim * 1]
            W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape,
                                             strides=[1, 2, 2, 1], padding='SAME') + b_conv3
            H_conv3 = tf.contrib.layers.batch_norm(inputs=H_conv3, center=True, scale=True, is_training=True,
                                                   scope="g_bn3")
            H_conv3 = tf.nn.relu(H_conv3)
            # Dimensions of H_conv3 = batch_size x 4 x 248 x 64

            # Fourth DeConv Layer
            output4_shape = [batch_size, int(width), s, c_dim]
            W_conv4 = tf.get_variable('g_wconv4', [1, 2, output4_shape[-1], int(H_conv3.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape,
                                             strides=[1, 1, 2, 1], padding='VALID') + b_conv4

            H_conv4 = tf.nn.softmax(H_conv4, axis=1, name="softmax_H_conv4")
            gene = tf.where(tf.equal(tf.reduce_max(H_conv4, axis=1, keep_dims=True), H_conv4),
                            tf.divide(H_conv4, tf.reduce_max(H_conv4, axis=1, keep_dims=True)),
                            tf.multiply(H_conv4, 0.))
            # Dimensions of H_conv4 = batch_size x 4 x 500 x 1
        return H_conv4, gene

    def build_model(self):
        input_dims = [self.input_height, self.input_width, self.c_dim]

        # Placeholders for learning rates for discriminator and generator
        self.tf_lr_d = tf.placeholder(tf.float32, name='lr_d')
        self.tf_lr_g = tf.placeholder(tf.float32, name='lr_g')

        """ Graph Input """
        # Train sequences, shape: [batch_size, 4, 500, 1]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + input_dims, name='real_sequences')
        # Test sequences, shape: [test_batch_size, 4, 500, 1]
        self.test_inputs = tf.placeholder(tf.float32, [self.test_batch_size] + input_dims, name='test_sequences')

        # Test labels, shape: [test_batch_size, 2], one hot encoded
        self.test_label = tf.placeholder(tf.float32, [self.test_batch_size, self.y_dim], name='test_label')

        # Labels for the generated sequences - this is not really needed
        self.visual_y = tf.placeholder(tf.float32, [self.generated_batch_size, self.y_dim], name='visual_y')

        # noises, shape: [batch_size, z_dim]
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.visual_z = tf.placeholder(tf.float32, [self.generated_batch_size, self.z_dim], name='visual_z')

        """ Loss Function """
        # A Game with two Players

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, y=None, is_training=True, reuse=False)

        # output of D for fake images
        G_approx_gene, G_gene = self.generator(self.z, y=None, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G_gene, y=None, is_training=True, reuse=True)

        #

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

        # test loss for classification using the discriminator
        _, _, predicted_test_logits = self.discriminator(self.test_inputs, is_training=False, reuse=True)
        correct_prediction = tf.equal(tf.argmax(predicted_test_logits, 1), tf.argmax(self.test_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        """ Training """

        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # for var in t_vars: print(var.name)
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.tf_lr_d, beta1=self.GAN_beta1).minimize(self.d_loss,
                                                                                            var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.tf_lr_g, beta1=self.GAN_beta1).minimize(self.g_loss,
                                                                                            var_list=g_vars)

        """" Generating sequences """
        self.fake_sequences = self.generator(self.visual_z, y=None, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        lr_d = self.lr_d
        lr_g = self.lr_g

        # inputs for generating testing sequences
        visual_sample_z = np.random.uniform(-1, 1, size=(self.generated_batch_size, self.z_dim))

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
                lr_d = float(line.split()[0])
                lr_g = float(line.split()[1])
                print("lr_d: %.3g" % lr_d)
                print("lr_g: %.3g" % lr_g)
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed, starting from epoch 0...")

        # loop for epoch
        start_time = time.time()

        for epoch in range(start_epoch, self.epoch):
            lr_d, lr_g, _ = self.update_learning_rates(epoch, lr_d=lr_d, lr_g=lr_g)

            # One epoch loop
            for idx in range(start_batch_id, self.num_batches):
                batch_sequences = self.data_X[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

                feed_dict = {
                    self.inputs: batch_sequences,
                    self.z: batch_z,
                    self.tf_lr_d: lr_d, self.tf_lr_g: lr_g,
                }
                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str_g, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict=feed_dict)
                self.writer.add_summary(summary_str_g, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

            """ Save generated samples to a file"""
            if epoch != 0 and epoch % 10 == 0:
                self.generate_and_save_samples(visual_sample_z=visual_sample_z, epoch=epoch);

            """ Measure accuracy (enhancers vs nonenhancers) of discriminator and save"""
            self.test_and_save_accuracy(epoch=epoch)

            """ Save learning rates to a file in case we wanted to resume later"""
            self.save_learning_rates(lr_d, lr_g)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            if epoch != 0 and epoch % 50 == 0:
                # save model
                self.save(self.checkpoint_dir, counter)

            # save model for final step
        self.save(self.checkpoint_dir, counter)
