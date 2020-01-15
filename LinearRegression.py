import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]
#tf.Variable(): tensorflow 가 자체적으로 변경시키는 값이다./trainable한 값이다.
#Shape 정의하고 이름을 준다.
#tf.random_normal([1]): 값이 하나인 랜덤한 일차원 배열
W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesias = x_train * W + b
#tf.reduce_mean(): 평균값을 구해줌
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
#Minimize(최소화)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
#tf실행을 위해 Session 생성
sess = tf.Session()
#tf.global_variables_initializer: Variable을 사용해서 실행하기 전에 초기화
sess.run(tf.global_variables_initializer())

for step in range(2001):
    #sess.run(train): train을 실행
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

        #train
        #|
        #cost
        #|
        #hypothesis
        #|   |
        #W   b