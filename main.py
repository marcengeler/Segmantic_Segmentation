import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # Load session
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # get default graph
    graph = tf.get_default_graph()
    
    # load all the individual layers
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    w2 = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w5 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return w1, w2, w3, w4, w5
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    # 1x1 convolution of last layer
    vgg_layer7_conv = tf.layers.conv2d(vgg_layer7_out,
                                    num_classes,
                                    1,
                                    padding = 'same',
                                    kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001))
                                    
    # upsampling
    vgg_layer_4_in = tf.layers.conv2d_transpose(vgg_layer7_conv,
                                                num_classes,
                                                4, 
                                                strides= (2, 2), 
                                                padding= 'same', 
                                                kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                                kernel_regularizer= tf.contrib.layers.l2_regularizer(0.001))
                                                
    # 1x1 convolution of vgg layer 4
    vgg_layer4_conv = tf.layers.conv2d(vgg_layer4_out,
                                    num_classes,
                                    1, 
                                    padding= 'same', 
                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(0.001))
                                    
    # skip connection (element-wise addition)
    vgg_layer_4 = tf.add(vgg_layer_4_in, vgg_layer4_conv)
    
    # upsampling
    vgg_layer_3_in = tf.layers.conv2d_transpose(vgg_layer_4,
                                                num_classes,
                                                4,  
                                                strides= (2, 2), 
                                                padding= 'same', 
                                                kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                                kernel_regularizer= tf.contrib.layers.l2_regularizer(0.001))
                                             
    # 1x1 convolution of vgg layer 3
    vgg_layer_3_conv = tf.layers.conv2d(vgg_layer3_out,
                                        num_classes,
                                        1, 
                                        padding= 'same', 
                                        kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                        kernel_regularizer= tf.contrib.layers.l2_regularizer(0.01))
                                        
    # skip connection (element-wise addition)
    vgg_layer_3 = tf.add(vgg_layer_3_in, vgg_layer_3_conv)
    
    # return last layer
    return tf.layers.conv2d_transpose(vgg_layer_3,
                                        num_classes,
                                        16,  
                                        strides= (8, 8), 
                                        padding= 'same', 
                                        kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                        kernel_regularizer= tf.contrib.layers.l2_regularizer(0.001))

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # compute logity by reshaping the last layers according to the number of classes
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    
    # get the correct labels in the same dimensions
    correct_label = tf.reshape(correct_label, (-1,num_classes))

    # standard cross entropy loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    
    # get regularization losses
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    # calculate total loss
    loss = cross_entropy_loss + 0.004 * sum(reg_losses)
    
    # AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(loss)
    
    return (logits, train_op, cross_entropy_loss)
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
    
    print("Training...")
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image,
                                            correct_label: label,
                                            keep_prob: 0.5,
                                            learning_rate: 0.0005})
            print("Loss: = {}".format(loss))
        print()
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        epochs = 25
        batch_size = 32
        
        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
