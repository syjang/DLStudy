#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import sys
import select
from IPython import embed
#from tensorflow.python.client import timeline
import resnet

from inputData import inputdata



def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
    lr = initial_lr
    for s in lr_decay_steps:
        if global_step >= s:
            lr *= lr_decay
    return lr


def train(num_input,num_classes,batch_size):

    num_gpus = 1
    momentum =  0.9
    finetune = False
    l2_weight = 0.0001
    num_train_instance = 1281167
    lr_step_epoch = "30.0,60.0"
    ip = inputdata();    
    with tf.Graph().as_default():
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of ImageNet
        with tf.variable_scope('train_image'):
            train_images_tmp, train_labels_tmp = ip.train_data,ip.train_labels
        with tf.variable_scope('test_image'):
            val_images_tmp, val_labels_tmp = ip.test_data,ip.test_labels
        tf.summary.image('images', train_images_tmp[0][:2])        

        #train_images = tf.convert_to_tensor(train_images,np.float32)
        #train_labels = tf.convert_to_tensor(train_labels,np.float32)

        train_images = []
        train_labels = []
        val_images = []
        val_labels = []
        ex = 0
        
        for i in range(batch_size,np.size(train_images_tmp,0),batch_size) :
            test = np.reshape(train_images_tmp[ex:i], (batch_size, 28,28,1))    
            test  = tf.convert_to_tensor(test,np.float32)    
            test.set_shape((batch_size, 28, 28, 1)) 
            train_images.append(test)            
            
            test2 = np.reshape(train_labels_tmp[ex:i], (batch_size,))
            test2 = tf.convert_to_tensor(test2,np.int32)
            test2.set_shape((batch_size, ))
            train_labels.append(test2)

            ex = i

        ex = 0
        for i in range(batch_size,np.size(val_images_tmp,0),batch_size) :
            test = np.reshape(val_images_tmp[ex:i], (batch_size, 28,28,1))    
            test  = tf.convert_to_tensor(test,np.float32)    
            test.set_shape((batch_size, 28, 28, 1)) 
            val_images.append(test)            
            
            test2 = np.reshape(val_labels_tmp[ex:i], (batch_size,))
            test2 = tf.convert_to_tensor(test2,np.int32)
            test2.set_shape((batch_size, ))
            val_labels.append(test2)

            ex = i


        # Build model
       
        lr_decay_steps = map(float,lr_step_epoch.split(','))
        lr_decay_steps = map(int,[s*num_train_instance/batch_size/num_gpus for s in lr_decay_steps])
        hp = resnet.HParams(batch_size=batch_size,
                            num_gpus=num_gpus,
                            num_classes=num_classes,
                            weight_decay=l2_weight,
                            momentum=momentum,
                            finetune=finetune)

        
        network_train = resnet.ResNet(hp, train_images, train_labels, global_step, name="train")        
        network_train.build_model()
        network_train.build_train_op()
        train_summary_op = tf.summary.merge_all()  # Summaries(training)
        network_val = resnet.ResNet(hp, val_images, val_labels, global_step, name="val", reuse_weights=True)
        network_val.build_model()
        print('Number of Weights: %d' % network_train._weights)
        print('FLOPs: %d' % network_train._flops)


        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.

        gpu_fraction =  0.95
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction),
            allow_soft_placement=False,
            # allow_soft_placement=True,
            log_device_placement=False))
        sess.run(init)


        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)

        #if not os.path.exists(FLAGS.train_dir):
        #    os.mkdir(FLAGS.train_dir)
        #summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, str(global_step.eval(session=sess))),
        #                                        sess.graph)

        # Training!
        max_steps = 10000
        val_best_acc = 0.0
        val_interval = 1000
        val_iter = 100
        initial_lr = 0.1
        lr_decay = 0.1
        for step in xrange(init_step, max_steps):
            # val
            if step % val_interval == 0:
                val_loss, val_acc = 0.0, 0.0
                for i in range(val_iter):
                    loss_value, acc_value = sess.run([network_val.loss, network_val.acc],
                                feed_dict={network_val.is_train:False})
                    val_loss += loss_value
                    val_acc += acc_value
                val_loss /= val_iter
                val_acc /= val_iter
                val_best_acc = max(val_best_acc, val_acc)
                format_str = ('%s: (val)     step %d, loss=%.4f, acc=%.4f')
                print (format_str % (datetime.now(), step, val_loss, val_acc))

                val_summary = tf.Summary()
                val_summary.value.add(tag='val/loss', simple_value=val_loss)
                val_summary.value.add(tag='val/acc', simple_value=val_acc)
                val_summary.value.add(tag='val/best_acc', simple_value=val_best_acc)
                #summary_writer.add_summary(val_summary, step)
                #summary_writer.flush()

            # Train            
            lr_value = get_lr(initial_lr, lr_decay, lr_decay_steps, step)
            start_time = time.time()
            _, loss_value, acc_value, train_summary_str = \
                    sess.run([network_train.train_op, network_train.loss, network_train.acc, train_summary_op],
                            feed_dict={network_train.is_train:True, network_train.lr:lr_value})
            duration = time.time() - start_time

            assert not np.isnan(loss_value)

            # Display & Summary(training)
            '''
            if step % FLAGS.display == 0 or step < 10:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (Training) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, acc_value, lr_value,
                                     examples_per_sec, sec_per_batch))
                summary_writer.add_summary(train_summary_str, step)
            '''
            # Save the model checkpoint periodically.
            #if (step > init_step and step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
            #    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            #    saver.save(sess, checkpoint_path, global_step=step)

            #if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            #  char = sys.stdin.read(1)
            #  if char == 'b':
            #    embed()

