import tensorflow as tf

# 构造 stfts，最后一维为 1 → 默认 fft_length = 2*(1-1) = 0
stfts = tf.constant([[[1.0+0j], [2.0+0j]]], dtype=tf.complex64)  # shape (1, 2, 1)

# frame_length=0 使得 0 <= 0 检查通过，随后触发 DUCC 断言
tf.signal.inverse_stft(
    stfts,
    frame_length=0,
    frame_step=1,
    fft_length=None   # 推导为 0
)