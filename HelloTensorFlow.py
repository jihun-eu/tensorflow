import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")
#node 생성
sess = tf.Session()
#세션 생성
print(sess.run(hello))
#세션 실행

#b'String' => Byte String