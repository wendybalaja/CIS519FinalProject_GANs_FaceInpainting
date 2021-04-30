import os
import errno
import numpy as np,os
import scipy
import scipy.misc
import json
import tensorflow as tf

def readMouthJson(category):

    json_cat = category + "/mouth-data.json"
    with open(json_cat, 'r') as f:
        data = json.load(f)

    all_iden_info = []
    all_ref_info = []

    test_all_iden_info = []
    test_all_ref_info = []

    #c: id, k: name of identity, v: details
    for c, (k, v) in enumerate(data.items()):

        identity_info = []

        is_close_id = 0


        if len(v) < 2:
            continue

        for i in range(len(v)):

            str_info = str(v[i]['filename']) + "_"

            if 'mouth_center' in v[i] and v[i]['mouth_center'] != None:
                str_info += str(v[i]['mouth_center']['y']) + "_"
                str_info += str(v[i]['mouth_center']['x']) + "_"
            else:
                str_info += str(0) + "_"
                str_info += str(0) + "_"

            if 'mouth_box' in v[i] and v[i]['mouth_box'] != None: 
                str_info += str(v[i]['mouth_box']['h']) + "_"
                str_info += str(v[i]['mouth_box']['w'])
            else:
                str_info += str(0) + "_"
                str_info += str(0)

            identity_info.append(str_info)


        rando = np.random.randint(0,10)
        if rando < 8:
            is_close = False 
        else: 
            is_close = True

        if is_close == False:

            for j in range(len(v)):

                first_n = np.random.randint(0, len(v), size=1)[0]
                all_iden_info.append(identity_info[first_n])
                middle_value = identity_info[first_n]
                identity_info.remove(middle_value)

                second_n = np.random.randint(0, len(v) - 1, size=1)[0]
                all_ref_info.append(identity_info[second_n])

                identity_info.append(middle_value)

        else:
            middle_value = identity_info[is_close_id]
            test_all_iden_info.append(middle_value)
            identity_info.remove(middle_value)

            second_n = np.random.randint(0, len(v) - 1, size=1)[0]
            test_all_ref_info.append(identity_info[second_n])

            test_all_iden_info.append(middle_value)

            second_n = np.random.randint(0, len(v) - 1, size=1)[0]
            test_all_ref_info.append(identity_info[second_n])

    assert len(all_iden_info) == len(all_ref_info)
    assert len(test_all_iden_info) == len(test_all_ref_info)

    print "train_data", len(all_iden_info)
    print "test_data", len(test_all_iden_info)

    return all_iden_info, all_ref_info, test_all_iden_info, test_all_ref_info

class Mouths(object):

    def __init__(self, image_path):
        self.dataname = "Mouths"
        self.image_size = 256
        self.channel = 3
        self.image_path = image_path
        self.dims = self.image_size*self.image_size
        self.shape = [self.image_size, self.image_size, self.channel]
        self.train_images_name, self.train_mouth_pos_name, self.train_ref_images_name, self.train_ref_pos_name, self.test_images_name, self.test_mouth_pos_name, self.test_ref_images_name, self.test_ref_pos_name = self.load_Mouths(image_path)

    def load_Mouths(self, image_path):

        images_list, images_ref_list, test_images_list, test_images_ref_list = readMouthJson(image_path)

        train_images_name = []
        train_mouth_pos_name = []
        train_ref_images_name = []
        train_ref_pos_name = []

        test_images_name = []
        test_mouth_pos_name = []
        test_ref_images_name = []
        test_ref_pos_name = []

        #train
        for images_info_str in images_list:

            mouth_pos = []
            image_name, mouth_x, mouth_y, mouth_h, mouth_w = images_info_str.split('_', 5)
            mouth_pos.append((int(mouth_x), int(mouth_y), int(mouth_h), int(mouth_w)))
            image_name = os.path.join(self.image_path, image_name)

            train_images_name.append(image_name)
            train_mouth_pos_name.append(mouth_pos)

        for images_info_str in images_ref_list:

            mouth_pos = []
            image_name, mouth_x, mouth_y, mouth_h, mouth_w = images_info_str.split('_', 5)

            mouth_pos.append((int(mouth_x), int(mouth_y), int(mouth_h), int(mouth_w)))

            image_name = os.path.join(self.image_path, image_name)
            train_ref_images_name.append(image_name)
            train_ref_pos_name.append(mouth_pos)

        for images_info_str in test_images_list:

            mouth_pos = []
            image_name, mouth_x, mouth_y, mouth_h, mouth_w = images_info_str.split('_', 5)
            mouth_pos.append((int(mouth_x), int(mouth_y), int(mouth_h), int(mouth_w)))
            image_name = os.path.join(self.image_path, image_name)

            test_images_name.append(image_name)
            test_mouth_pos_name.append(mouth_pos)

        for images_info_str in test_images_ref_list:

            mouth_pos = []
            image_name, mouth_x, mouth_y, mouth_h, mouth_w = images_info_str.split('_', 5)
            mouth_pos.append((int(mouth_x), int(mouth_y), int(mouth_h), int(mouth_w)))
            image_name = os.path.join(self.image_path, image_name)

            test_ref_images_name.append(image_name)
            test_ref_pos_name.append(mouth_pos)

        assert len(train_images_name) == len(train_mouth_pos_name) == len(train_ref_images_name) == len(train_ref_pos_name)
        assert len(test_images_name) == len(test_mouth_pos_name) == len(test_ref_images_name) == len(test_ref_pos_name)

        return train_images_name, train_mouth_pos_name, train_ref_images_name, train_ref_pos_name, \
               test_images_name, test_mouth_pos_name, test_ref_images_name, test_ref_pos_name

    def getShapeForData(self, filenames, is_test=False):

        array = [get_image(batch_file, 108, is_crop=False, resize_w=256,
                           is_grayscale=False, is_test=is_test) for batch_file in filenames]
        sample_images = np.array(array)
        return sample_images

    def getNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):

        ro_num = len(self.train_images_name) / batch_size
        if batch_num % ro_num == 0 and is_shuffle:

            length = len(self.train_images_name)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.train_images_name = np.array(self.train_images_name)
            self.train_images_name = self.train_images_name[perm]

            self.train_mouth_pos_name = np.array(self.train_mouth_pos_name)
            self.train_mouth_pos_name = self.train_mouth_pos_name[perm]

            self.train_ref_images_name = np.array(self.train_ref_images_name)
            self.train_ref_images_name = self.train_ref_images_name[perm]

            self.train_ref_pos_name = np.array(self.train_ref_pos_name)
            self.train_ref_pos_name = self.train_ref_pos_name[perm]

        return self.train_images_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.train_mouth_pos_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.train_ref_images_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
                self.train_ref_pos_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

    def getTestNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):

        ro_num = len(self.test_images_name) / batch_size
        if batch_num == 0 and is_shuffle:

            length = len(self.test_images_name)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.test_images_name = np.array(self.test_images_name)
            self.test_images_name = self.test_images_name[perm]

            self.test_mouth_pos_name = np.array(self.test_mouth_pos_name)
            self.test_mouth_pos_name = self.test_mouth_pos_name[perm]

            self.test_ref_images_name = np.array(self.test_ref_images_name)
            self.test_ref_images_name = self.test_ref_images_name[perm]

            self.test_ref_pos_name = np.array(self.test_ref_pos_name)
            self.test_ref_pos_name = self.test_ref_pos_name[perm]

        return self.test_images_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_mouth_pos_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_ref_images_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_ref_pos_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

def mkDir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class eGAN(object):

    def __init__(self, batch_size, max_iters, model_path, data_ob, sample_path, log_dir, learning_rate, is_load, lam_recon,
                 lam_gp, use_sp, beta1, beta2, n_critic):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.model_path = model_path
        self.data_ob = data_ob
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.log_vars = []
        self.channel = data_ob.channel
        self.shape = data_ob.shape
        self.lam_recon = lam_recon
        self.lam_gp = lam_gp
        self.use_sp = use_sp
        self.is_load = is_load
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_critic = n_critic
        self.output_size = data_ob.image_size
        self.input_img = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.exemplar_images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.img_mask = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.exemplar_mask =  tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.domain_label = tf.placeholder(tf.int32, [batch_size])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

    def build_model_GAN(self):

        self.incomplete_img = self.input_img * (1 - self.img_mask)
        self.local_real_img = self.input_img * self.img_mask

        self.x_tilde = self.encode_decode(self.incomplete_img, self.exemplar_images, 1 - self.img_mask, self.exemplar_mask, reuse=False)
        self.local_fake_img = self.x_tilde * self.img_mask

        self.D_real_gan_logits = self.discriminate(self.input_img, self.exemplar_images, self.local_real_img, spectural_normed=self.use_sp, reuse=False)
        self.D_fake_gan_logits = self.discriminate(self.x_tilde, self.exemplar_images, self.local_fake_img, spectural_normed=self.use_sp, reuse=True)

        self.D_loss = self.loss_dis(self.D_real_gan_logits, self.D_fake_gan_logits)
        self.G_gan_loss = self.loss_gen(self.D_fake_gan_logits)

        self.recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.abs(self.x_tilde - self.input_img), axis=[1, 2, 3]) / (
            self.output_size * self.output_size * self.channel))

        self.G_loss = self.G_gan_loss + self.lam_recon * self.recon_loss

        self.log_vars.append(("D_loss", self.D_loss))
        self.log_vars.append(("G_loss", self.G_loss))

        self.t_vars = tf.trainable_variables()
        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in self.t_vars if 'encode_decode' in var.name]

        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    def loss_dis(self, d_real_logits, d_fake_logits):

        l1 = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
        l2 = tf.reduce_mean(tf.nn.softplus(d_fake_logits))

        return l1 + l2

    def loss_gen(self, d_fake_logits):
        return tf.reduce_mean(tf.nn.softplus(-d_fake_logits))

    def train(self):

        d_trainer = tf.train.AdamOptimizer(self.learning_rate * self.lr_decay, beta1=self.beta1, beta2=self.beta2)
        d_gradients = d_trainer.compute_gradients(self.D_loss, var_list=self.d_vars)
        opti_D = d_trainer.apply_gradients(d_gradients)

        m_trainer = tf.train.AdamOptimizer(self.learning_rate * self.lr_decay, beta1=self.beta1, beta2=self.beta2)
        m_gradients = m_trainer.compute_gradients(self.G_loss, var_list=self.g_vars)
        opti_M = m_trainer.apply_gradients(m_gradients)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            step = 0
            step2 = 0
            lr_decay = 1

            if self.is_load:
                # TODO: need to manually insert ckpt number
                self.saver.restore(sess, os.path.join(self.model_path, 'model_000055.ckpt'.format(step)))

            while step <= self.max_iters:


                if step > 20000 and lr_decay > 0.1:
                    lr_decay = (self.max_iters - step) / float(self.max_iters - 10000)

                for i in range(self.n_critic):

                    train_data_list, batch_mouth_pos, batch_train_ex_list, batch_ex_mouth_pos = self.data_ob.getNextBatch(step2, self.batch_size)
                    batch_images_array = self.data_ob.getShapeForData(train_data_list)
                    batch_exem_array = self.data_ob.getShapeForData(batch_train_ex_list)
                    batch_mouth_pos = np.squeeze(batch_mouth_pos)

                    batch_ex_mouth_pos = np.squeeze(batch_ex_mouth_pos)
                    f_d = {self.input_img: batch_images_array, self.exemplar_images: batch_exem_array,
                           self.img_mask: self.get_Mask(batch_mouth_pos), self.exemplar_mask: self.get_Mask(batch_ex_mouth_pos), self.lr_decay: lr_decay}

                    # optimize D
                    sess.run(opti_D, feed_dict=f_d)
                    step2 += 1

                # optimize M
                sess.run(opti_M, feed_dict=f_d)
                summary_str = sess.run(summary_op, feed_dict=f_d)
                summary_writer.add_summary(summary_str, step)

                # save model
                if step % 5 == 0: 
                    self.saver.save(sess, os.path.join(self.model_path, 'model_{:06d}.ckpt'.format(step)))

                    x_tilde, incomplete_img, local_real, local_fake = sess.run([self.x_tilde, self.incomplete_img, self.local_real_img, self.local_fake_img], feed_dict=f_d)
                    output_concat = np.concatenate([batch_images_array, batch_exem_array, incomplete_img, x_tilde, local_real, local_fake], axis=0)
                    save_images(output_concat, [output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_output.jpg'.format(self.sample_path, step))
                    print("Saved a model!")

                if step % 50 == 0:
                    d_loss,  g_loss = sess.run([self.D_loss, self.G_loss],
                        feed_dict=f_d)
                    print("step %d d_loss = %.4f, g_loss=%.4f" % (step, d_loss, g_loss))

                if np.mod(step, 400) == 0:

                    x_tilde, incomplete_img, local_real, local_fake = sess.run([self.x_tilde, self.incomplete_img, self.local_real_img, self.local_fake_img], feed_dict=f_d)
                    output_concat = np.concatenate([batch_images_array, batch_exem_array, incomplete_img, x_tilde, local_real, local_fake], axis=0)
                    save_images(output_concat, [output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_output.jpg'.format(self.sample_path, step))
                if np.mod(step, 2000) == 0:
                    self.saver.save(sess, os.path.join(self.model_path, 'model_{:06d}.ckpt'.format(step)))

                step += 1

            save_path = self.saver.save(sess, self.model_path)
            print "Model saved in file: %s" % save_path

    def discriminate(self, x_var, x_exemplar, local_x_var, spectural_normed=False, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()

            conv = tf.concat([x_var, x_exemplar], axis=3)
            for i in range(5):
                output_dim = np.minimum(64 * np.power(2, i+1), 512)
                conv = lrelu(conv2d(conv, spectural_normed=spectural_normed, output_dim=output_dim, name='dis_conv_{}'.format(i)))

            conv = tf.reshape(conv, shape=[self.batch_size, conv.shape[1] * conv.shape[2] * conv.shape[3]])
            ful_global = fully_connect(conv, output_size=output_dim, spectural_normed=spectural_normed, scope='dis_fully1')

            conv = local_x_var
            for i in range(5):
                output_dim = np.minimum(64 * np.power(2, i+1), 512)
                conv = lrelu(conv2d(conv, spectural_normed=spectural_normed, output_dim=output_dim, name='dis_conv_2_{}'.format(i)))

            conv = tf.reshape(conv, shape=[self.batch_size, conv.shape[1] * conv.shape[2] * conv.shape[3]])
            ful_local = fully_connect(conv, output_size=output_dim, spectural_normed=spectural_normed, scope='dis_fully2')

            gan_logits = fully_connect(tf.concat([ful_global, ful_local], axis=1), output_size=1, spectural_normed=spectural_normed, scope='dis_fully3')

            return gan_logits

    def encode_decode(self, x_var, x_exemplar, img_mask, exemplar_mask, reuse=False):

        with tf.variable_scope("encode_decode") as scope:

            if reuse == True:
                scope.reuse_variables()

            x_var = tf.concat([x_var, img_mask, x_exemplar, exemplar_mask], axis=3)

            conv1 = tf.nn.relu(
                instance_norm(conv2d(x_var, output_dim=64, k_w=7, k_h=7, d_w=1, d_h=1, name='e_c1'), scope='e_in1'))
            conv2 = tf.nn.relu(
                instance_norm(conv2d(conv1, output_dim=128, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c2'), scope='e_in2'))
            conv3 = tf.nn.relu(
                instance_norm(conv2d(conv2, output_dim=256, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c3'), scope='e_in3'))

            r1 = Residual(conv3, residual_name='re_1')
            r2 = Residual(r1, residual_name='re_2')
            r3 = Residual(r2, residual_name='re_3')
            r4 = Residual(r3, residual_name='re_4')
            r5 = Residual(r4, residual_name='re_5')
            r6 = Residual(r5, residual_name='re_6')

            g_deconv1 = tf.nn.relu(instance_norm(de_conv(r6, output_shape=[self.batch_size,
                                                                           self.output_size/2, self.output_size/2, 128], name='gen_deconv1'), scope="gen_in"))
            # for 1
            g_deconv_1_1 = tf.nn.relu(instance_norm(de_conv(g_deconv1,
                        output_shape=[self.batch_size, self.output_size, self.output_size, 32], name='g_deconv_1_1'), scope='gen_in_1_1'))

            g_deconv_1_1_x = tf.concat([g_deconv_1_1, x_var], axis=3)
            x_tilde1 = conv2d(g_deconv_1_1_x, output_dim=self.channel, k_w=7, k_h=7, d_h=1, d_w=1, name='gen_conv_1_2')

            return tf.nn.tanh(x_tilde1)

    def get_Mask(self, mouth_pos, flag=0):

        mouth_pos = mouth_pos
        print ("**mouth_pos:",mouth_pos)
        batch_mask = []
        for i in range(self.batch_size):

            current_mouth_pos = mouth_pos[i]
            #mouth
            if flag == 0:
                #left mouth, y
                mask = np.zeros(shape=[self.output_size, self.output_size, self.channel])
                scale = current_mouth_pos[0] - 25 #current_mouth_pos[3] / 2
                down_scale = current_mouth_pos[0] + 25 #current_mouth_pos[3] / 2
                l1_1 =int(scale)
                u1_1 =int(down_scale)
                #x
                scale = current_mouth_pos[1] - 35 #current_mouth_pos[2] / 2
                down_scale = current_mouth_pos[1] + 35 #current_mouth_pos[2] / 2
                l1_2 = int(scale)
                u1_2 = int(down_scale)

                mask[l1_1:u1_1, l1_2:u1_2, :] = 1.0


            batch_mask.append(mask)

        return np.array(batch_mask)

def save_images(images, size, image_path, is_ouput=False):
        return imsave(inverseImage(images, is_ouput), size, image_path)

def imsave(images, size, path):
        return scipy.misc.imsave(path, mergeImages(images, size))

def inverseImage(image, is_ouput=False):
        result = ((image + 1) * 127.5).astype(np.uint8)
        return result

def mergeImages(images, size):

    if size[0] + size[1] == 2:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def conv2d(input_, output_dim,k_h=5, k_w=5, d_h= 2, d_w=2, stddev=0.02, spectural_normed=False,name="conv2d", padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],initializer=tf.random_normal_initializer(stddev=stddev))
        if spectural_normed:
            conv = tf.nn.conv2d(input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def lrelu(x, alpha= 0.2, name="LeakyReLU"):
    return tf.maximum(x , alpha*x)


def de_conv(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
            biases = tf.get_variable('biases', [output_shape[-1]], tf.float32, initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

            if with_w:
                return deconv, w, biases

            else:
                return deconv

def instance_norm(input, scope="instance_norm"):

    with tf.variable_scope(scope):

        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)

        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv

        return scale * normalized + offset

def Residual(x, output_dims=256, kernel=3, strides=1, residual_name='resi'):

    with tf.variable_scope('residual_{}'.format(residual_name)) as scope:

        conv1 = instance_norm(conv2d(x, output_dims, k_h=kernel, k_w=kernel, d_h=strides, d_w=strides, name="conv1"), scope='in1')
        conv2 = instance_norm(conv2d(tf.nn.relu(conv1), output_dims, k_h=kernel, k_w=kernel,d_h=strides, d_w=strides, name="conv2"), scope='in2')
        resi = x + conv2

        return tf.nn.relu(resi)

