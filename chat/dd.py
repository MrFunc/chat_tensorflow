from seq2seq import Seq2seq as seq
# 训练
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
char_inputs = [[2,1],[1,2],[2,3],[3,4],[4,0]]

used = tf.sign(tf.abs(char_inputs))
length = tf.reduce_sum(used, reduction_indices=0)
lengths = tf.cast(length, tf.int32)


sess = tf.Session()
print(sess.run(lengths))

# 训练
seq.train()
# 预测
seq.predict("天气")
# 重新训练
seq.retrain()
