import tensorflow as tf
import numpy as np
#Logit/Score->SoftMax->Probabilities

xy = np.loadtxt('data-04-zoo.csv',delimiter=',',dtype=np.float32)
#data를 읽어옴
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

nb_classes = 7

X = tf.placeholder(tf.float32,[None,16])
#X data수는 16개
Y = tf.placeholder(tf.int32, [None,1])#0 ~ 6, shape=(?, 1)
#Y data수는 1개
#우리가 함수로 넘겨줄 때 필요한 것은 one_hot
Y_one_hot = tf.one_hot(Y, nb_classes)# one hot shape =(?, 1, 7)
#one hot은 한차원을 더 높여서 데이터를 출력한다.
#ex) [[0],[3]]--input-->One_hot--output-->[[[1000000]][[0001000]]]
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])# shape = (?, 7)
#한 차원이 더해지는 구조를 reshape
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X,W) +b#Score or Logits
hypothesis = tf.nn.softmax(logits)#SoftMax, 확률

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
#cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))를 간략화
cost = tf.reduce_mean(cost_i)
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)#probability 0 ~ 6
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
#예측값과 실제 값을 비교
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#정확히 예측한 값들을 모아 평균을 낸다 -> 정확도
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        #학습 진행
        if step % 100 ==0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    pred = sess.run(prediction, feed_dict={X: x_data})
    #예측이 잘 됐는지 확인
    for p,y in zip(pred, y_data.flatten()):#.flatten(): [[1],[0]]--flatten()-->[1, 0]
        #zip(): List의 element를 넘겨주기 편하게 하기 위해 묶어준다.
        print("[{}] Prediction: {} True Y: {}".format(p == int(y),p,int(y)))

