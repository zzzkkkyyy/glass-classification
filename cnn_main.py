import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc as misc
import os, glob
from random import shuffle
import numpy as np

# classes of datasets
predict_dict = {"hao" : 0, "hua" : 1, "bian" : 2, "jiao" : 3}
predict_dict_reverse = {0 : "hao", 1 : "hua", 2 : "bian", 3 : "jiao"}
image_list = {"training" : [], "validation" : []}

adam_beta1 = 0.9
adam_init_lr = 5e-4
batch_size = 24

def batch_norm_layer(x, train_phase, scope_bn):
    # reshape the original data block to [N(#feature maps), L(#unrolled pixels)]
    shape = x.get_shape().as_list()
    x_unrolled = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape = [x_unrolled.shape[-1]]), name = 'beta', trainable = True)
        gamma = tf.Variable(tf.constant(1.0, shape = [x_unrolled.shape[-1]]), name = 'gamma', trainable = True)
        batch_mean, batch_var = tf.nn.moments(x_unrolled, axes = [0], name = 'moments')
        # create an ExpMovingAver object
        ema = tf.train.ExponentialMovingAverage(decay = 0.5)
        def mean_var_with_update():
            # update the shadow variable for batch_mean, batch_var
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x_unrolled, mean, var, beta, gamma, 1e-3)
        normed = tf.reshape(normed, tf.shape(x))
    return normed

def cnn_model(X, y, is_training):
    train_phase = tf.convert_to_tensor(is_training)

    Wconv1 = tf.get_variable("Wconv1", shape = [3, 3, 1, 64], initializer = tf.truncated_normal_initializer(mean = 0, stddev = 1e-4))
    bconv1 = tf.get_variable("bconv1", shape = [64], initializer = tf.zeros_initializer())
    conv1 = tf.nn.conv2d(X, Wconv1, strides = [1, 1, 1, 1], padding = 'SAME') + bconv1

    scope_bn1 = 'BN1'
    batch_norm1 = batch_norm_layer(conv1, train_phase, scope_bn1)
    del conv1
    relu1 = tf.nn.relu(batch_norm1, 'relu1')
    del batch_norm1

    Wconv2 = tf.get_variable("Wconv2", shape = [3, 3, 64, 64], initializer = tf.truncated_normal_initializer(mean = 0, stddev = 1e-4))
    bconv2 = tf.get_variable("bconv2", shape = [64], initializer = tf.zeros_initializer())
    conv2 = tf.nn.conv2d(relu1, Wconv2, strides = [1, 1, 1, 1], padding = 'SAME') + bconv2
    del relu1

    max_pooling2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')
    del conv2
    max_pooling2 = tf.cast(max_pooling2, tf.float32)

    scope_bn2 = 'BN2'
    batch_norm2 = batch_norm_layer(max_pooling2, train_phase, scope_bn2)
    del max_pooling2
    relu2 = tf.nn.relu(batch_norm2, 'relu2')
    del batch_norm2
    if is_training is True:
        relu2 = tf.layers.dropout(relu2, keep_prob = 0.5)

    Wconv3 = tf.get_variable("Wconv3", shape = [3, 3, 64, 128], initializer = tf.truncated_normal_initializer(mean = 0, stddev = 1e-4))
    bconv3 = tf.get_variable("bconv3", shape = [128], initializer = tf.zeros_initializer())
    conv3 = tf.nn.conv2d(relu2, Wconv3, strides = [1, 2, 2, 1], padding = 'SAME') + bconv3
    del relu2

    scope_bn3 = 'BN3'
    batch_norm3 = batch_norm_layer(conv3, train_phase, scope_bn3)
    del conv3
    relu3 = tf.nn.relu(batch_norm3, 'relu3')
    del batch_norm3
    if is_training is True:
        relu3 = tf.layers.dropout(relu3, keep_prob = 0.5)

    Wconv4 = tf.get_variable("Wconv4", shape = [3, 3, 128, 64], initializer = tf.truncated_normal_initializer(mean = 0, stddev = 1e-4))
    bconv4 = tf.get_variable("bconv4", shape = [64], initializer = tf.zeros_initializer())
    conv4 = tf.nn.conv2d(relu3, Wconv4, strides = [1, 2, 2, 1], padding = 'SAME') + bconv4
    del relu3

    scope_bn4 = 'BN4'
    batch_norm4 = batch_norm_layer(conv4, train_phase, scope_bn4)
    del conv4
    relu4 = tf.nn.relu(batch_norm4, 'relu4')
    del batch_norm4

    full_connected_size = relu4.get_shape().as_list()[1] * relu4.get_shape().as_list()[2] * relu4.get_shape().as_list()[3]
    W5 = tf.get_variable("W5", shape = [full_connected_size, 4], initializer = tf.truncated_normal_initializer(mean = 0, stddev = 1e-4))
    b5 = tf.get_variable("b5", shape = [4], initializer = tf.zeros_initializer())
    relu4 = tf.reshape(relu4, [-1, full_connected_size])

    if is_training is True:
        fc5_dropout = tf.layers.dropout(relu4, dropout_rate = 0.5, training = is_training)
    else:
        fc5_dropout = relu4
    del relu4

    y_out = tf.matmul(fc5_dropout, W5) + b5

    return y_out


def main(argv = None):
    print("construct network...")
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 256, 256, 1])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    global_step = tf.Variable(0, trainable = False)

    y_out = cnn_model(x, y, is_training)
    pred_label = tf.argmax(y_out, 1)
    correct_prediction = tf.equal(pred_label, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    total_loss = tf.nn.softmax_cross_entropy_with_logits(logits = y_out, labels = tf.one_hot(y, 4, dtype = tf.int64))
    mean_loss = tf.reduce_mean(total_loss)
    trainable_var = tf.trainable_variables()
    adam_lr = tf.train.exponential_decay(adam_init_lr, global_step = global_step, decay_steps = 100, decay_rate = 0.5)
    optimizer = tf.train.AdamOptimizer(adam_lr, adam_beta1)
    grads = optimizer.compute_gradients(mean_loss, var_list = trainable_var)
    train_op = optimizer.apply_gradients(grads)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    print("read image..")
    type_list = ["training", "validation"]
    dataset_path = {"training": "/data", "validation": "/data2"}
    for t in type_list:
        file_list = []
        p = os.path.join(dataset_path[t], t, "*.jpg")
        #print(p)
        file_list.extend(glob.glob(p))
        if not file_list:
            print("{} files not found.".format(t))
            return
        #shuffle(file_list)
        for f in file_list:
            image = misc.imread(f)
            img = misc.imresize(image, (256, 256)) * 1.0 / 255
            # name rule: hua_1.bmp, jiao_3.bmp
            category = int(predict_dict[(os.path.splitext(f.split("/")[-1])[0]).split("_")[0]]);
            record = {"name": os.path.splitext(f.split("/")[-1])[0], "image" : img, "category" : category}
            image_list[t].append(record)
        del file_list

    for t in type_list:
        shuffle(image_list[t])

    start, end = 0, 0
    train_num = len(image_list["training"])
    loss = []
    x_axis = []
    name_valid_batch = [item["name"] for item in image_list["validation"][0: batch_size]]
    x_valid_batch = [np.expand_dims(item["image"], axis = 2) for item in image_list["validation"][0: batch_size]]
    y_valid_batch = [item["category"] for item in image_list["validation"][0: batch_size]]
    for i in range(batch_size):
        print("{}, truth: {}".format(name_valid_batch[i], predict_dict_reverse[y_valid_batch[i]]))
        #print("truth: {}".format(predict_dict_reverse[y_valid_batch[i]]))
    for itr in range(1001):
        start = end
        end = start + batch_size
        if end > train_num:
            shuffle(image_list["training"])
            start = 0
            end = batch_size
            print("***********all element is relisted***********")
        x_train_batch = [np.expand_dims(item["image"], axis = 2) for item in image_list["training"][start: end]]
        y_train_batch = [item["category"] for item in image_list["training"][start: end]]
        feed_dict = {x : x_train_batch, y : y_train_batch, is_training: True}
        [train_loss, temp] = sess.run([mean_loss, train_op], feed_dict = feed_dict)
        if itr % 5 == 0:
            print("Epoch: {}, train_loss: {}".format(itr, train_loss))
            loss.append(train_loss)
            x_axis.append(itr)
        if itr % 20 == 0:
            [acc, valid_loss, pred] = sess.run([accuracy, mean_loss, pred_label], feed_dict = {x : x_valid_batch, y : y_valid_batch, is_training: False})
            print("Epoch: {}, valid_loss: {}".format(itr, valid_loss))
            print("Accuracy: {}".format(acc))
            for i in range(batch_size):
                print("prediction: {}".format(predict_dict_reverse[pred[i]]))
    """
    pred = sess.run(pred_label, feed_dict = {x : x_valid_batch, y : y_valid_batch, is_training: False})
    for i in range(batch_size):
        print("{}".format(name_valid_batch[i]))
        print("truth: {}".format(predict_dict_reverse[y_valid_batch[i]]))
        print("prediction: {}".format(predict_dict_reverse[pred[i]]))
        print("********************************")
        #print("{}, prediction: {}, truth: {}".format(name_valid_batch[i], predict_dict_reverse[tf.cast(pred[i], tf.int64)], predict_dict_reverse[y_valid_batch[i]]))
    """
    print(loss)

    
if __name__ == "__main__":
    tf.app.run()