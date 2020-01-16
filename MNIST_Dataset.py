from tensorflow.examples.tutorials.mnist import input_data
#그림을 만들어놓은 라이브러리
import tensorflow as tf
import matplotlib.pyplot as plt
#실제 데이터를 출력하기 위한 라이브러리
import random
nb_classes = 10
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#디렉토리를 지정하여 데이터를 읽어옴
#one_hot=True: Y의 값을 원하는대로 one_hot으로 처리한다.
batch_xs, batch_ys = mnist.train.next_batch(100)
#train.next_batch(100): 100개의 x와 y train data가 읽어짐

X = tf.placeholder(tf.float32,[None,784])
#28*28 픽셀로 이루어진 dataset
Y = tf.placeholder(tf.float32,[None, nb_classes])
#0 ~ 9 :10개의 예측 class

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W)+b)
#matrix X와 W의 곱에 bias를 더한 값을 softmax로 넘겨준 값
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
#Y와 hypothesis의 log한 값을 곱하여 더한다음 평균을 낸다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#최소화
is_correct =tf.equal(tf.arg_max(hypothesis, 1),tf.arg_max(Y, 1))
#Y값과 hypothesis의 값이 같으면 true
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#true값을 cast

training_epochs = 15
#전체 데이터 셋을 학습시킨것을 1 epoch라고 하는데
#너무 많은 데이터가 있을 경우 batch_size로 나눠서 연산한다.
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):#몇 번 epoch할 것인가
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):#몇번 반복 할 것인가
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #batch_size만큼 나눠서 데이터를 학습시킨다.
            c, _ = sess.run([cost, optimizer],feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print("Accuracy: ",accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    r= random.randint(0, mnist.test.num_examples - 1)#random 한 숫자를 읽어옴
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    #test할 label의 숫자를 읽음
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
    #hypothesis를 통해 유추
    plt.imshow(mnist.test.images[r:r + 1].reshape(28,28), cmap='Greys', interpolation='nearest')
    #imshow를 통해 이미지가 맞는지 출력
    plt.show()
    #accuacy 평가: sess.run() or accuracy.eval()->값을 하나만 출력할때 사용하곤 함
#test dataset을 사용해서 정확도 측정

#epoch: 전체 데이터셋을 한번 돈 것
#batch size: n개씩 나눠서 돌리는 것
#1000 training examples and 500 batch size = 2 iterations