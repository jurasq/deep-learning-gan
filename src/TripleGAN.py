import dna
from ops import *
from utils import *
import time

class TripleGAN(object) :
    def __init__(self, sess, epoch, batch_size, unlabel_batch_size, z_dim, dataset_name, nexamples, gan_lr, cla_lr, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.unlabelled_batch_size = unlabel_batch_size
        self.test_batch_size = 100
        self.test_set_size = 1000
        self.model_name = "TripleGAN"     # name for checkpoint
        if self.dataset_name == 'cifar10' :
            self.input_height = 32
            self.input_width = 32
            self.output_height = 32
            self.output_width = 32

            self.z_dim = z_dim
            self.y_dim = 10
            self.c_dim = 3

            self.learning_rate = gan_lr # 3e-4, 1e-3
            self.cla_learning_rate = cla_lr # 3e-3, 1e-2 ?
            self.GAN_beta1 = 0.5
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.alpha = 0.5
            self.alpha_cla_adv = 0.01
            self.init_alpha_p = 0.0 # 0.1, 0.03
            self.apply_alpha_p = 0.1
            self.apply_epoch = 200 # 200, 300
            self.decay_epoch = 50

            self.sample_num = 64
            self.visual_num = 100
            self.len_discrete_code = 10

            self.data_X, self.data_y, self.unlabelled_X, self.unlabelled_y, self.test_X, self.test_y = cifar10.prepare_data(n) # trainX, trainY, testX, testY

            self.num_batches = len(self.data_X) // self.batch_size
        elif self.dataset_name == 'dna':
            self.categories = np.asarray(['A', 'C', 'T', 'G'])

            self.input_height = 4  # One-hot encoding
            self.input_width = 500  # Input sequence length
            self.output_height = 4  # One-hot encoding
            self.output_width = 500  # Output sequence length

            self.z_dim = z_dim  # Random noise dimension
            self.y_dim = 2  # Number of labels
            self.c_dim = 1  # "Colour" dimension

            self.learning_rate = gan_lr # 3e-4, 1e-3
            self.cla_learning_rate = cla_lr # 3e-3, 1e-2 ?
            self.GAN_beta1 = 0.5
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.alpha = 0.5
            self.alpha_cla_adv = 0.01
            self.init_alpha_p = 0.0 # 0.1, 0.03
            self.apply_alpha_p = 0.1
            self.apply_epoch = 200 # 200, 300
            self.decay_epoch = 50

            self.generated_batch_size = 500

            self.sample_num = 64
            self.visual_num = 100
            self.len_discrete_code = 10

            self.data_X, self.data_y, self.test_X, self.test_y = self.init_data(nexamples)

            self.num_batches = len(self.data_X) // self.batch_size
        else :
            raise NotImplementedError

    def init_data(self, nexamples):
        print("Running TripleGAN, so loading both positive and negative samples...")
        print(nexamples)
        return dna.prepare_data(nexamples, self.test_set_size, samples_to_use="both", test_both=True)  # trainX, trainY, testX, testY

    def discriminator(self, x, y_, scope='discriminator', is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse) :
            #x = dropout(x, rate=0.2, is_training=is_training)
            y = tf.reshape(y_, [-1, 1, 1, self.y_dim])
            x = conv_concat(x,y)

            x = lrelu(conv_layer(x, filter_size=20, kernel=[4,9], layer_name=scope+'_conv1'))
            x = max_pooling(x, kernel=[1,3], stride=1)

            x = lrelu(conv_layer(x, filter_size=30, kernel=[1,5], layer_name=scope+'_conv2'))
            x = max_pooling(x, kernel=[1,4], stride=1)

            x = lrelu(conv_layer(x, filter_size=40, kernel=[1,3], layer_name=scope+'_conv2'))
            x = max_pooling(x, kernel=[1,4], stride=1)

            x = flatten(x)
            x = concat([x,y_]) # mlp_concat
            x = linear(x, unit=90, layer_name=scope+'_linear1')
            #x = relu(x)
            x = linear(x, unit=45, layer_name=scope+'_linear2')
            #x = relu(x)
            #x = dropout(x, rate=0.2, is_training=is_training)
            x_logit = linear(x, unit=1, layer_name=scope+'_linear3')
            out = tf.nn.softmax(x_logit)


            return out, x_logit, x

    def generator(self, noise_vector, y_, scope='generator', is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse) :
            noise_vector = concat([noise_vector, y_])
            y = tf.reshape(y_, [-1, 1, 1, self.y_dim])

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

            # output_mlp = mlp('mlp', noise_vector, batch_norm=False, relu=False)
            output_mlp = tf.layers.dense(inputs=noise_vector, units=int(width / 4) * (s16 + 1) * magic_number)

            h0 = tf.reshape(output_mlp, [batch_size, int(width / 4), s16 + 1, magic_number])
            h0 = lrelu(h0)
            h0 = conv_concat(h0, y)

            output1_shape = [batch_size, int(width / 2), s8 + 1, g_dim * 4]
            W_conv1 = tf.get_variable('g_wconv1',[5,5,output1_shape[-1],int(h0.get_shape()[-1])],initializer=tf.truncated_normal_initializer(stddev=0.1))

            b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))

            H_conv1 = tf.nn.conv2d_transpose(value=h0, filter=W_conv1, output_shape=output1_shape,
                                                 strides=[1, 2, 2, 1], padding='SAME', name="H_conv1") + b_conv1
            H_conv1 = tf.contrib.layers.batch_norm(inputs=H_conv1, center=True, scale=True, is_training=is_training,
                                                       scope="g_bn1")
            H_conv1 = tf.nn.leaky_relu(H_conv1)

            # Dimensions of H_conv1 = batch_size x 1 x 62 x 256
            H_conv1 = conv_concat(H_conv1, y)

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
            H_conv2 = conv_concat(H_conv2, y)

            # Third DeConv Layer
            output3_shape = [batch_size, int(width), s2, g_dim * 1]
            W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape,
                                                 strides=[1, 2, 2, 1], padding='SAME') + b_conv3
            H_conv3 = tf.contrib.layers.batch_norm(inputs=H_conv3, center=True, scale=True, is_training=True,
                                                       scope="g_bn3")
            H_conv3 = tf.nn.leaky_relu(H_conv3)
            # Dimensions of H_conv3 = batch_size x 4 x 248 x 64
            H_conv3 = conv_concat(H_conv3, y)

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


    def classifier(self, x, scope='classifier', is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse) :
            #x = gaussian_noise_layer(x) # default = 0.15
            x = lrelu(conv_layer(x, filter_size=20, kernel=[4,9], layer_name=scope+'_conv1'))
            x = max_pooling(x, kernel=[1,3], stride=1)

            x = lrelu(conv_layer(x, filter_size=30, kernel=[1,5], layer_name=scope+'_conv2'))
            x = max_pooling(x, kernel=[1,4], stride=1)

            x = lrelu(conv_layer(x, filter_size=40, kernel=[1,3], layer_name=scope+'_conv2'))
            x = max_pooling(x, kernel=[1,4], stride=1)

            x = flatten(x)
            x = linear(x, unit=90, layer_name=scope+'_linear1')
            x = relu(x)
            x = linear(x, unit=45, layer_name=scope+'_linear2')
            x = relu(x)
            x = dropout(x, rate=0.2, is_training=is_training)
            x = linear(x, unit=2, layer_name=scope+'_linear3')
            return x

    def build_model(self):
        input_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        unlabel_bs = self.unlabelled_batch_size
        test_bs = self.test_batch_size
        alpha = self.alpha
        alpha_cla_adv = self.alpha_cla_adv
        self.alpha_p = tf.placeholder(tf.float32, name='alpha_p')
        self.gan_lr = tf.placeholder(tf.float32, name='gan_lr')
        self.cla_lr = tf.placeholder(tf.float32, name='cla_lr')
        self.unsup_weight = tf.placeholder(tf.float32, name='unsup_weight')
        self.c_beta1 = tf.placeholder(tf.float32, name='c_beta1')

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + input_dims, name='real_sequences')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        # Batch test sequences + labels, shape: [test_batch_size, 4, 500, 1], [test_batch_size, 2]
        self.test_inputs = tf.placeholder(tf.float32, [self.test_batch_size] + input_dims, name='test_sequences')
        self.test_label = tf.placeholder(tf.float32, [self.test_batch_size, self.y_dim], name='test_label')

        # Full test sequences + labels, shape: [test_set_size, 4, 500, 1], [test_set_size, 2], one hot encoded
        self.full_test_dataset = tf.placeholder(tf.float32, [self.test_set_size] + input_dims, name="all_test_sequences")
        self.full_test_dataset_labels = tf.placeholder(tf.float32, [self.test_set_size, self.y_dim], name="all_test_label")

        # Labels for the generated sequences - this is not really needed
        self.visual_y = tf.placeholder(tf.float32, [self.generated_batch_size, self.y_dim], name='visual_y')

        # noises, shape: [batch_size, z_dim
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
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

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = (1-alpha)*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = (1-alpha)*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        # test loss for classify
        test_Y = self.classifier(self.test_inputs, is_training=False, reuse=True)
        correct_prediction = tf.equal(tf.argmax(test_Y, 1), tf.argmax(self.test_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # self.c_loss = alpha * c_loss_dis + R_L + self.alpha_p*R_P

        # R_UL = self.unsup_weight * tf.reduce_mean(tf.squared_difference(Y_c, self.unlabelled_inputs_y))
        self.c_loss = R_L + self.alpha_p*R_P

        # test AUC on full dataset
        test_logits_full = self.classifier(self.full_test_dataset, is_training=False, reuse=True)
        softmax_logits_full = tf.nn.softmax(test_logits_full, axis=1)
        self.auc_on_test_set = tf.metrics.auc(predictions=softmax_logits_full, labels=self.full_test_dataset_labels)

        """ Training """

        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        c_vars = [var for var in t_vars if 'classifier' in var.name]

        for var in t_vars: print(var.name)
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.gan_lr, beta1=self.GAN_beta1).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.gan_lr, beta1=self.GAN_beta1).minimize(self.g_loss, var_list=g_vars)
            self.c_optim = tf.train.AdamOptimizer(self.cla_lr, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.c_loss, var_list=c_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.visual_z, self.visual_y, is_training=False, reuse=True)

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
        tf.local_variables_initializer().run()
        
        gan_lr = self.learning_rate
        cla_lr = self.cla_learning_rate

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.visual_num, self.z_dim))
        self.test_codes = self.data_y[0:self.visual_num]

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
            with open('lr_logs.txt', 'r') as f :
                line = f.readlines()
                line = line[-1]
                gan_lr = float(line.split()[0])
                cla_lr = float(line.split()[1])
                print("gan_lr : ", gan_lr)
                print("cla_lr : ", cla_lr)
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()

        for epoch in range(start_epoch, self.epoch):

            if epoch >= self.decay_epoch :
                gan_lr *= 0.995
                cla_lr *= 0.99
                print("**** learning rate DECAY ****")
                print(gan_lr)
                print(cla_lr)

            if epoch >= self.apply_epoch :
                alpha_p = self.apply_alpha_p
            else :
                alpha_p = self.init_alpha_p

            rampup_value = rampup(epoch - 1)
            unsup_weight = rampup_value * 100.0 if epoch > 1 else 0

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx * self.batch_size : (idx + 1) * self.batch_size]
                batch_codes = self.data_y[idx * self.batch_size : (idx + 1) * self.batch_size]

                batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

                feed_dict = {
                    self.inputs: batch_images, self.y: batch_codes,
                    self.z: batch_z, self.alpha_p: alpha_p,
                    self.gan_lr: gan_lr, self.cla_lr: cla_lr,
                    self.unsup_weight : unsup_weight
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

            for idx in range(10) :
                test_batch_x = self.test_X[idx * self.test_batch_size : (idx+1) * self.test_batch_size]
                test_batch_y = self.test_y[idx * self.test_batch_size : (idx+1) * self.test_batch_size]

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
            lr = "{} {}".format(gan_lr, cla_lr)
            with open('logs.txt', 'a') as f:
                f.write(line)
            with open('lr_logs.txt', 'a') as f :
                f.write(lr+'\n')

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.test_and_save_accuracy(epoch=epoch)

            # save model for final step
        self.save(self.checkpoint_dir, counter)

    def generate_and_save_samples(self, visual_sample_z, epoch):
        saving_start_time = time.time()
        # save some (15) generated sequences for every epoch
        _, samples = self.sess.run(self.fake_sequences,
                                feed_dict={self.visual_z: visual_sample_z})
        one_hot_decoded = self.one_hot_decode(samples)
        save_sequences(one_hot_decoded,
                       self.get_files_location('_generated_sequences_epoch_{:03d}.txt'.format(
                           epoch)))
        saving_end_time = time.time()
        print("Saved %d generated samples to file, took %4.4f" % (
            self.generated_batch_size, saving_end_time - saving_start_time))

    def one_hot_decode(self, samples):
        decode_indices = tf.argmax(samples, axis=1)
        decoded = self.sess.run(decode_indices)
        seqs = tf.squeeze(tf.reduce_join(self.categories[decoded], 1))
        return self.sess.run(seqs)


    def test_and_save_accuracy(self, epoch):
        # testing the accuracy (enhancers vs nonenhancers) of discriminator
        test_acc = 0.0

        print("==== Accuracy for batches: (first half are positive (enhancers), last half are negative (non-enhancers)) ====")
        for idx in range(int(self.test_set_size/self.test_batch_size)):
            test_batch_x = self.test_X[idx * self.test_batch_size: (idx + 1) * self.test_batch_size]
            test_batch_y = self.test_y[idx * self.test_batch_size: (idx + 1) * self.test_batch_size]
            acc_ = self.sess.run(self.accuracy, feed_dict={
                self.test_inputs: test_batch_x,
                self.test_label: test_batch_y
            })
            print("Batch #%2d: %f" % (idx+1, acc_))

            test_acc += acc_
        test_acc /= int(self.test_set_size/self.test_batch_size)

        if self.auc_on_test_set:
            auc = self.sess.run(self.auc_on_test_set, feed_dict={
                self.full_test_dataset: self.test_X,
                self.full_test_dataset_labels: self.test_y
            })
            print("AUC")
            print(auc)

        summary_test = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
        self.writer.add_summary(summary_test, epoch)

        line = "Epoch: [%2d], test_acc: %.4f\n" % (epoch, test_acc)
        print(line)
        with open(self.get_files_location('accuracy.txt'), 'a') as f:
            f.write(line)

    def save_learning_rates(self, lr_d, lr_g, lr_c=None):
        if lr_c:
            lr = "{} {} {}".format(lr_d, lr_g, lr_c)
            with open(self.get_files_location('lr_logs.txt'), 'a') as f:
                    f.write(lr + '\n')
        else:
            lr = "{} {}".format(lr_d, lr_g)
            with open(self.get_files_location('lr_logs.txt'), 'a') as f:
                f.write(lr + '\n')

    def get_files_location(self, suffix):
        return './' + check_folder(
                           self.result_dir + '/' + self.model_dir) + '/' + self.model_name + suffix;

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
