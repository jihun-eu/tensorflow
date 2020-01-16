import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

X= tf.placeholder(tf.float32, shape=[None,2])
Y=tf.placeholder(tf.float32, shape=[None,1])

W=tf.Variable(tf.random_normal([2,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+ (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
predicted = tf.cast(hypothesis >0.5, dtype = tf.float32)
#0.5보다 높을 경우 True를 반환하는데 boolean을 float를 casting시키면 True의 경우 1.0, False의 경우 0.0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#예측값과 실제 값의 정확도를 평균을 통해 체크
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict = {X: x_data, Y: y_data})
        #sess.run을 통해 cost값과 train값을 체크하는데 x_data와 y_data값을 차례로 넣어주면서 체크한다.
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypotheis: ", h,"\nCorrect (Y): ", c,"\nAccuracy: ",a)