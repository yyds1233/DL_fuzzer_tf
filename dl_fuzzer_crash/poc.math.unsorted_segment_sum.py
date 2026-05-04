import tensorflow as tf

# data: uint16, shape (2, 2) -> 去除第一维后乘积 = 2
data = tf.constant([[1, 2], [3, 4]], dtype=tf.uint16)

# segment_ids: int64, shape (2, 2) 与 data 相同
segment_ids = tf.constant([[0, 0], [1, 1]], dtype=tf.int64)

# num_segments: int64 极大正数
num_segments = tf.constant(9223372036854775807, dtype=tf.int64)

# 直接调用 raw_ops 或 math.unsorted_segment_sum 均可，
# 因为 num_segments >=0 检查会通过，然后溢出分配。
tf.raw_ops.UnsortedSegmentSum(
    data=data,
    segment_ids=segment_ids,
    num_segments=num_segments
)