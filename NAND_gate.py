import tensorflow as tf 
import numpy as np 

#used Tensor flow to create a one hidden layer neural network 

def add_layer(inputs,in_size,out_size,activation_function=None):
 		Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = "weights")
 		biases = tf.Variable(tf.zeros([1,out_size])+0.1, name = "biases")
 		Wx_plus_b = tf.matmul(inputs, Weights) + biases 
 		if activation_function is None:
 			outputs = Wx_plus_b
 		else:
 			outputs = activation_function(Wx_plus_b) 
 		return outputs			


x_data = np.array([[0,0],[0,1],[1,1],[1,0]]) #test conditions for a 2 input gate
y_data = np.array([[0,1],[0,1],[1,0],[0,1]])	# Outputs of a NAND GATE

xs = tf.placeholder(tf.float32, [None,2])
ys = tf.placeholder(tf.float32, [None,2])

l1 = add_layer(xs,2,10,activation_function=None)
prediction = add_layer(l1, 10, 2, activation_function=tf.nn.softmax)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices= [1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 #saver = tf.train.Saver()


sess = tf.Session()

init = tf.initialize_all_variables()

sess.run(init)

for i in range(1000):
	
 	sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
 	if i% 50 ==0:
 		print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

print(sess.run(prediction, feed_dict={xs: [[0,0]]}))
correct_prediction = tf.equal(tf.argmax(ys,1), tf.argmax(y_data,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accu = (sess.run(accuracy, feed_dict={xs: x_data, ys: y_data}))*100
print("%s %% accurate" % accu)
#  save_path = saver.save(sess, "./save_net.ckpt")
#  print("Save to path: ", save_path)

# W = tf.Variable(np.arange(20).reshape((2, 10)), dtype=tf.float32, name="weights")
# b = tf.Variable(np.arange(10).reshape((1, 10)), dtype=tf.float32, name="biases")

# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, "./save_net.ckpt")
#     print("weights:", sess.run(W))
#     print("biases:", sess.run(b))


