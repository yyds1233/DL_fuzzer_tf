#!/usr/bin/env python3
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import atheris
import numpy as np
import tensorflow as tf


MAX_ELEMENTS = 1 << 16
PADDINGS = ["SAME", "VALID"]
DATA_FORMATS = ["NHWC", "NCHW"]
DTYPES = [tf.float32, tf.float64]


def disable_gpu():
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def consume_shape(fdp, data_format):
    n = fdp.ConsumeIntInRange(0, 8)
    h = fdp.ConsumeIntInRange(0, 64)
    w = fdp.ConsumeIntInRange(0, 64)
    c = fdp.ConsumeIntInRange(0, 64)

    if data_format == "NHWC":
        return [n, h, w, c]
    return [n, c, h, w]


def num_elements(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def consume_tensor(fdp, shape, dtype):
    total = num_elements(shape)
    if total > MAX_ELEMENTS:
        raise ValueError("too many elements")

    values = fdp.ConsumeRegularFloatList(total)
    arr = np.asarray(
        values,
        dtype=np.float32 if dtype == tf.float32 else np.float64,
    ).reshape(shape)

    return tf.convert_to_tensor(arr, dtype=dtype)


def consume_pool_arg(fdp, data_format):
    mode = fdp.ConsumeIntInRange(0, 2)

    if mode == 0:
        return fdp.ConsumeIntInRange(0, 16)

    a = fdp.ConsumeIntInRange(0, 16)
    b = fdp.ConsumeIntInRange(0, 16)

    if mode == 1:
        return [a, b]

    if data_format == "NHWC":
        return [1, a, b, 1]
    return [1, 1, a, b]


def fuzz_avg_pool2d(data):
    fdp = atheris.FuzzedDataProvider(data)

    data_format = fdp.PickValueInList(DATA_FORMATS)
    padding = fdp.PickValueInList(PADDINGS)
    dtype = fdp.PickValueInList(DTYPES)

    try:
        shape = consume_shape(fdp, data_format)
        x = consume_tensor(fdp, shape, dtype)
        ksize = consume_pool_arg(fdp, data_format)
        strides = consume_pool_arg(fdp, data_format)

        y = tf.nn.avg_pool2d(
            input=x,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )

        # 强制真正执行到底层 kernel
        _ = y.numpy()

    except (
        ValueError,
        TypeError,
        tf.errors.InvalidArgumentError,
        tf.errors.UnimplementedError,
    ):
        # 忽略参数/shape/前端校验类异常
        # 只让更底层的问题继续冒出来
        return


def main():
    disable_gpu()
    atheris.Setup(sys.argv, fuzz_avg_pool2d)
    atheris.Fuzz()


if __name__ == "__main__":
    main()