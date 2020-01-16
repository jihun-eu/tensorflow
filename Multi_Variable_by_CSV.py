#csv에 있는 데이터 타입이 모두 같아야 하는 단점
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('data-01-test-score.csv',delimiter=',',dtype=np.float32)
x_data = xy[:, 0:-1]
#전체 행에서 처음에서 마지막 전까지의 열을 가져온다.
y_data = xy[:, [-1]]
#전체 행에서 마지막 열을 가져온다.
print(x_data.shape,x_data,len(x_data))
print(y_data.shape,y_data)

X=tf.placeholder(tf.float32,shape=[None,3])
Y=tf.placeholder(tf.float32, shape=[None,1])

W=tf.Variable(tf.random_normal([3,1]),name='weight')
#[x,y]: x=들어오는 값, y=나가는 값
b=tf.Variable(tf.random_normal([1]),name='bias')
#[y]: y=나가는 값
hypothesis=tf.matmul(X,W)+b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],feed_dict={X: x_data, Y: y_data})
    if step%10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\nb",hy_val)

        print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100,70,101]]}))

        print("Other scores will be ", sess.run(hypothesis, feed_dict={X: [[60,70,110],[90,100,80]]}))