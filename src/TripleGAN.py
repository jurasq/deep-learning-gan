import dna
from ops import *
from utils import *
import time


class TripleGAN(object):
    def __init__(self, sess, epoch, batch_size, unlabel_batch_size, z_dim, dataset_name,
                 nexamples, lr_d, lr_g, lr_c, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.unlabelled_batch_size = unlabel_batch_size
        self.test_batch_size = 100
        self.model_name = "TripleGAN"  # name for checkpoint

        if self.dataset_name == 'dna':
            self.input_height = 4  # One-hot encoding
            self.input_width = 500  # Input sequence length
            self.output_height = 4  # One-hot encoding
            self.output_width = 500  # Output sequence length

            self.z_dim = z_dim  # Random noise dimension
            self.y_dim = 2  # Number of labels
            self.c_dim = 1  # "Colour" dimension

            # Learning rates: discriminator, generator, classifier
            self.lr_d = lr_d
            self.lr_g = lr_g
            self.lr_c = lr_c

            self.GAN_beta1 = 0.5  # D and G, exponential decay rate for the 1st moment estimates
            self.beta1 = 0.9  # C, exponential decay rate for the 1st moment estimates
            self.beta2 = 0.999  # C, exponential decay rate for the 2nd moment estimates
            # C, A small constant for numerical stability. This epsilon is "epsilon hat"
            # in the Kingma and Ba paper (in the formula just before Section 2.1),
            # not the epsilon in Algorithm 1 of the paper.
            self.epsilon = 1e-8
            self.alpha = 0.5  # discriminator loss: (1-alpha) * D_loss_fake + alpha * D_loss_classifier + D_loss_real
            self.alpha_cla_adv = 0.01
            self.init_alpha_p = 0.0  # 0.1, 0.03
            self.apply_alpha_p = 0.1
            self.apply_epoch = 200  # 200, 300 Point in epoch we change alpha_p from init to apply, if epoch >= self.apply_epoch: alpha_p = self.apply_alpha_p else: alpha_p = self.init_alpha_p
            self.decay_epoch = 50  # Point in epoch we start adding decay. if epoch >= decay_epoch, add decay. Note decay is hard coded.

            self.sample_num = 64  # ? Unused for now
            self.generated_batch_size = 15  # Visualization frame size. We get a floor(sqrt(visual_num)) x floor(sqrt(visual_num)) sample
            self.len_discrete_code = 4  # Think this is one-hot encoding for visualization ?

            self.data_X, self.data_y, self.unlabelled_X, self.unlabelled_y, self.test_X, self.test_y = dna.prepare_data(
                nexamples, self.test_batch_size)  # trainX, trainY, testX, testY

            self.num_batches = len(self.data_X) // self.batch_size

        else:
            print("Dataset not supported.")
            raise NotImplementedError

        print("Initializing TripleGAN with lr_d=%.3g, lr_g=%.3g, lr_c=%.3g" % (self.lr_d, self.lr_g, self.lr_c))

    def discriminator(self, x, y_, scope='discriminator', is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            y = tf.reshape(y_, [-1, 1, 1, self.y_dim])

            x = conv_concat(x, y)
            x = conv_layer(name_scope="convolutional_1", input_tensor=x, num_kernels=20, kernel_shape=[4, 9], relu=True)
            x = max_pool_layer(name_scope="max_pool_1", input_tensor=x, pool_size=[1, 3])

            x = conv_concat(x, y)
            x = conv_layer(name_scope="convolutional_2", input_tensor=x, num_kernels=30, kernel_shape=[1, 5])
            x = max_pool_layer(name_scope="max_pool_2", input_tensor=x, pool_size=[1, 4])

            x = conv_concat(x, y)
            x = conv_layer(name_scope="convolutional_3", input_tensor=x, num_kernels=40, kernel_shape=[1, 3])
            x = max_pool_layer(name_scope="max_pool_3", input_tensor=x, pool_size=[1, 4])

            x = flatten(x)

            x = concat([x, y_])
            x = tf.layers.dense(inputs=x, units=90)
            x = tf.layers.dense(inputs=x, units=45)
            logits = tf.layers.dense(inputs=x, units=2)
            logits_sigmoid = tf.nn.softmax(logits)

            return x, logits, logits_sigmoid

    def generator(self, noise_vector, y_, scope="generator", is_training=True, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            # y = tf.reshape(y_, [-1, 1, 1, self.y_dim])
            # noise_vector = conv_concat(noise_vector,y)
            batch_size = self.batch_size
            g_dim = 64  # Number of filters of first layer of generator
            c_dim = 1  # dimensionality of the output
            s = 500  # Final length of the sequence

            # We want to slowly upscale the sequence, so these values should help
            # to make that change gradual
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

            width = 4  # because we have 4 letters: ATCG

            # this is a magic number which I'm not sure what means yet
            magic_number = 5

            output_mlp = mlp('mlp', noise_vector, batch_norm=False, relu=False)

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
            #         H_conv4 = tf.nn.tanh(H_conv4)

            H_conv4 = tf.nn.softmax(H_conv4, axis=1, name="softmax_H_conv4")
            gene = tf.where(tf.equal(tf.reduce_max(H_conv4, axis=1, keep_dims=True), H_conv4),
                            tf.divide(H_conv4, tf.reduce_max(H_conv4, axis=1, keep_dims=True)),
                            tf.multiply(H_conv4, 0.))
            # Dimensions of H_conv4 = batch_size x 4 x 500 x 1
        return H_conv4, gene

    def classifier(self, x, scope='classifier', is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            if (reuse):
                tf.get_variable_scope().reuse_variables()

            # TODO: make a reverse filter conv layer like in the Enhancer paper 

            # convolutional + pooling #1
            l1 = conv_max_forward_reverse(name_scope="conv1", input_tensor=x, num_kernels=20,
                                          kernel_shape=[4, 9], relu=True)
            l2 = max_pool_layer(name_scope="pool1", input_tensor=l1, pool_size=[1, 3])

            # convolutional + pooling #2
            l3 = conv_max_forward_reverse(name_scope="conv2", input_tensor=l2, num_kernels=30, kernel_shape=[1, 5])
            l4 = max_pool_layer(name_scope="pool2", input_tensor=l3, pool_size=[1, 4])

            # convolutional + pooling #3
            l5 = conv_max_forward_reverse(name_scope="conv3", input_tensor=l4, num_kernels=40, kernel_shape=[1, 3])
            l6 = max_pool_layer(name_scope="pool3", input_tensor=l5, pool_size=[1, 4])

            flat = flatten(l6)
            # fully connected layers
            l7 = tf.layers.dense(inputs=flat, units=90)
            l8 = tf.layers.dense(inputs=l7, units=45)

            logits = tf.layers.dense(inputs=l8, units=2)

        return tf.nn.softmax(logits, axis=1, name="softmax_tensor")

    def build_model(self):
        input_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        test_bs = self.test_batch_size

        self.alpha_p = tf.placeholder(tf.float32, name='alpha_p')
        self.tf_lr_d = tf.placeholder(tf.float32, name='lr_d')
        self.tf_lr_g = tf.placeholder(tf.float32, name='lr_g')
        self.tf_lr_c = tf.placeholder(tf.float32, name='lr_c')

        self.c_beta1 = tf.placeholder(tf.float32, name='c_beta1')

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + input_dims, name='real_sequences')
        self.test_inputs = tf.placeholder(tf.float32, [test_bs] + input_dims, name='test_sequences')

        # labels
        self.y = tf.placeholder(tf.float32, [bs, self.y_dim], name='y')
        self.test_label = tf.placeholder(tf.float32, [test_bs, self.y_dim], name='test_label')
        self.visual_y = tf.placeholder(tf.float32, [self.generated_batch_size, self.y_dim], name='visual_y')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')
        self.visual_z = tf.placeholder(tf.float32, [self.generated_batch_size, self.z_dim], name='visual_z')

        """ Loss Function """
        # A Game with Three Players

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, self.y, is_training=True, reuse=False)

        # output of D for fake images
        G_approx_gene, G_gene = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G_gene, self.y, is_training=True, reuse=True)

        # output of C for real images
        C_real_logits = self.classifier(self.inputs, is_training=True, reuse=False)
        R_L = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=C_real_logits))

        # output of C for fake images
        C_fake_logits = self.classifier(G_gene, is_training=True, reuse=True)
        R_P = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=C_fake_logits))

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

        # test loss for classify
        test_Y = self.classifier(self.test_inputs, is_training=False, reuse=True)
        correct_prediction = tf.equal(tf.argmax(test_Y, 1), tf.argmax(self.test_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # get loss for classify
        # max_c = tf.cast(tf.argmax(Y_c, axis=1), tf.float32)
        # c_loss_dis = tf.reduce_mean(max_c * tf.nn.softmax_cross_entropy_with_logits(logits=D_cla_logits, labels=tf.ones_like(D_cla)))
        # self.c_loss = alpha * c_loss_dis + R_L + self.alpha_p*R_P

        # R_UL = self.unsup_weight * tf.reduce_mean(tf.squared_difference(Y_c, self.unlabelled_inputs_y))
        self.c_loss = R_L + self.alpha_p * R_P

        """ Training """

        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        c_vars = [var for var in t_vars if 'classifier' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.tf_lr_d, beta1=self.GAN_beta1).minimize(self.d_loss,
                                                                                               var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.tf_lr_g, beta1=self.GAN_beta1).minimize(self.g_loss,
                                                                                               var_list=g_vars)
            self.c_optim = tf.train.AdamOptimizer(self.tf_lr_c, beta1=self.beta1, beta2=self.beta2,
                                                  epsilon=self.epsilon).minimize(self.c_loss, var_list=c_vars)

        """" Testing """
        # for test
        self.fake_sequences = self.generator(self.visual_z, self.visual_y, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        c_loss_sum = tf.summary.scalar("c_loss", self.c_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.c_sum = tf.summary.merge([c_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()
        lr_d = self.lr_d
        lr_g = self.lr_g
        lr_c = self.lr_c

        # graph inputs for visualize training results
        self.visual_sample_z = np.random.uniform(-1, 1, size=(self.generated_batch_size, self.z_dim))
        self.test_codes = self.data_y[0:self.generated_batch_size]

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            with open('lr_logs.txt', 'r') as f:
                line = f.readlines()
                line = line[-1]
                lr_d = float(line.split()[0])
                lr_g = float(line.split()[1])
                lr_c = float(line.split()[2])
                print("lr_d: %.3g" % lr_d)
                print("lr_g: %.3g" % lr_g)
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

            if epoch >= self.decay_epoch:
                lr_d *= 0.995
                lr_g *= 0.995
                lr_c *= 0.99
                print("**** learning rate DECAY ****")
                print("lr_d is now: %.3g" % lr_d)
                print("lr_g is now: %.3g" % lr_g)
                print("lr_c is now: %.3g" % lr_c)

            if epoch >= self.apply_epoch:
                alpha_p = self.apply_alpha_p
            else:
                alpha_p = self.init_alpha_p

            # rampup_value = rampup(epoch - 1)
            # unsup_weight = rampup_value * 100.0 if epoch > 1 else 0

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_codes = self.data_y[idx * self.batch_size: (idx + 1) * self.batch_size]

                batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

                feed_dict = {
                    self.inputs: batch_images, self.y: batch_codes,
                    self.z: batch_z, self.alpha_p: alpha_p,
                    self.tf_lr_d: lr_d, self.tf_lr_g: lr_g, self.tf_lr_c: lr_c
                }
                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str_g, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict=feed_dict)
                self.writer.add_summary(summary_str_g, counter)

                # update C network
                _, summary_str_c, c_loss = self.sess.run([self.c_optim, self.c_sum, self.c_loss], feed_dict=feed_dict)
                self.writer.add_summary(summary_str_c, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, c_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, c_loss))

                # save training results for every 100 steps
                """
                if np.mod(counter, 100) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z, self.y: self.test_codes})
                    image_frame_dim = int(np.floor(np.sqrt(self.visual_num)))
                    save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                                './' + check_folder(
                                    self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))
                """

            # classifier test
            test_acc = 0.0

            for idx in range(10):
                test_batch_x = self.test_X[idx * self.test_batch_size: (idx + 1) * self.test_batch_size]
                test_batch_y = self.test_y[idx * self.test_batch_size: (idx + 1) * self.test_batch_size]

                acc_ = self.sess.run(self.accuracy, feed_dict={
                    self.test_inputs: test_batch_x,
                    self.test_label: test_batch_y
                })

                test_acc += acc_
            test_acc /= 10

            summary_test = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
            self.writer.add_summary(summary_test, epoch)

            line = "Epoch: [%2d], test_acc: %.4f\n" % (epoch, test_acc)
            print(line)
            with open('logs.txt', 'a') as f:
                f.write(line)

            lr = "{} {} {}".format(lr_d, lr_g, lr_c)
            with open('lr_logs.txt', 'a') as f:
                f.write(lr + '\n')

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

            # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        # tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(self.generated_batch_size)))
        z_sample = np.random.uniform(-1, 1, size=(self.generated_batch_size, self.z_dim))

        """ random noise, random discrete code, fixed continuous code """
        # y = np.random.choice(self.len_discrete_code, self.visual_num)
        # Generated 10 labels with batch_size
        y_one_hot = np.zeros((self.generated_batch_size, self.y_dim))
        # y_one_hot[np.arange(self.visual_num), y] = 1

        samples = self.sess.run(self.fake_sequences, feed_dict={self.visual_z: z_sample, self.visual_y: y_one_hot})

        print("Generated samples:")
        print(samples)

        return

        # save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
        #             check_folder(
        #                 self.result_dir + '/' + self.model_dir + '/all_classes') + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        """ specified condition, random noise """
        n_styles = 10  # must be less than or equal to self.batch_size

        np.random.seed()
        si = np.random.choice(self.visual_num, n_styles)

        for l in range(self.len_discrete_code):
            y = np.zeros(self.visual_num, dtype=np.int64) + l
            y_one_hot = np.zeros((self.visual_num, self.y_dim))
            y_one_hot[np.arange(self.visual_num), y] = 1

            samples = self.sess.run(self.fake_images, feed_dict={self.visual_z: z_sample, self.visual_y: y_one_hot})
            save_sequences(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                           check_folder(
                            self.result_dir + '/' + self.model_dir + '/class_%d' % l) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_class_%d.png' % l)

            samples = samples[si, :, :, :]

            if l == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """
        canvas = np.zeros_like(all_samples)
        for s in range(n_styles):
            for c in range(self.len_discrete_code):
                canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_sequences(canvas, [n_styles, self.len_discrete_code],
                       check_folder(
                        self.result_dir + '/' + self.model_dir + '/all_classes_style_by_style') + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

    def generate_and_save_samples(self, visual_sample_z, epoch):
        saving_start_time = time.time()
        # save some (15) generated sequences for every epoch
        samples = self.sess.run(self.fake_sequences,
                                feed_dict={self.visual_z: visual_sample_z})
        save_sequences(samples,
                       './' + check_folder(
                           self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_generated_sequences_epoch_{:02d}.txt'.format(
                           epoch))
        saving_end_time = time.time()
        print("Saved %d generated samples to file, took %4.4f" % (
            self.generated_batch_size, saving_end_time - saving_start_time))

    def test_and_save_accuracy(self, epoch):
        # testing the accuracy (enhancers vs nonenhancers) of discriminator
        test_acc = 0.0

        for idx in range(10):
            test_batch_x = self.test_X[idx * self.test_batch_size: (idx + 1) * self.test_batch_size]
            test_batch_y = self.test_y[idx * self.test_batch_size: (idx + 1) * self.test_batch_size]

            acc_ = self.sess.run(self.accuracy, feed_dict={
                self.test_inputs: test_batch_x,
                self.test_label: test_batch_y
            })

            test_acc += acc_
        test_acc /= 10

        summary_test = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
        self.writer.add_summary(summary_test, epoch)

        line = "Epoch: [%2d], test_acc: %.4f\n" % (epoch, test_acc)
        print(line)
        with open('accuracy.txt', 'a') as f:
            f.write(line)

    def save_learning_rates(self, lr_d, lr_g, lr_c=None):
        if lr_c:
            pass
        else:
            lr = "{} {}".format(lr_d, lr_g)
            with open('lr_logs.txt', 'a') as f:
                f.write(lr + '\n')

    def update_learning_rates(self, epoch, lr_d, lr_g, lr_c=None):
        if lr_c:
            if epoch >= self.decay_epoch:
                lr_d *= 0.995
                lr_g *= 0.995
                lr_c *= 0.99
                print("**** learning rate DECAY ****")
                print("lr_d is now: %.3g" % lr_d)
                print("lr_g is now: %.3g" % lr_g)
                print("lr_c is now: %.3g" % lr_c)
        else:
            if epoch >= self.decay_epoch:
                lr_d *= 0.995
                lr_g *= 0.99
                print("**** learning rate DECAY ****")
                print("lr_d lr is now: %.3g" % lr_d)
                print("lr_g lr is now: %.3g" % lr_g)

        return lr_d, lr_g, lr_c

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
