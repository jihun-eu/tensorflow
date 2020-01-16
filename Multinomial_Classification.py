import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],#2
          [0, 0, 1],#2
          [0, 0, 1],#2
          [0, 1, 0],#1
          [0, 1, 0],#1
          [0, 1, 0],#1
          [1, 0, 0],#0
          [1, 0, 0]]#0
#여러개의 class가 있기 때문에 one-hot encoding 사용
X = tf.placeholder("float",[None, 4])
Y = tf.placeholder("float", [None, 3])
#Y의 개수 = label의 개수(class의 개수)
nb_classes =3
W = tf.Variable(tf.random_normal([4,nb_classes]), name='weight')
#random_normal[x:들어오는 값, y:나가는 값]
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)#값이 확률로 주어짐
#Score / Logit을 넣는다.
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis),axis=1))
#loss function: reduce_sum = ()안의 값을 더한다. reduce_mean =()안 값의 평균을 구함
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:#세션 열기
    sess.run(tf.global_variables_initializer())#변수 초기화

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})#학습진행
        if step % 200 == 0:
            a,b = sess.run([cost,hypothesis],feed_dict={X: x_data,Y: y_data})
            print(a)
            print(b)

#학습은 그래프형식으로 진행된다.
