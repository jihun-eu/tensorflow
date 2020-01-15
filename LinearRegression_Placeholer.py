import tensorflow as tf


#tf.Variable(): tensorflow 가 자체적으로 변경시키는 값이다./trainable한 값이다.
#Shape 정의하고 이름을 준다.
#tf.random_normal([1]): 값이 하나인 랜덤한 일차원 배열
W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b
#tf.reduce_mean(): 평균값을 구해줌
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#Minimize(최소화)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
#tf실행을 위해 Session 생성
sess = tf.Session()
#tf.global_variables_initializer: Variable을 사용해서 실행하기 전에 초기화
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1,2,3,4,5], Y: [2.1,3.1,4.1,5.1,6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

        #train
        #|
        #cost
        #|
        #hypothesis
        #|   |
        #W   b