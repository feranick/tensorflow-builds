import tensorflow as tf, time
print('TF32 default:', tf.config.experimental.tensor_float_32_execution_enabled())
tf.config.experimental.enable_tensor_float_32_execution(True)

N, ITERS = 8192, 10
with tf.device('/GPU:0'):
    x = tf.random.normal([N, N])
    tf.matmul(x, x)
    t = time.perf_counter()
    for _ in range(ITERS): c = tf.matmul(x, x)
    _ = c.numpy()
    dt = time.perf_counter() - t
print(c.device)
print(f'{ITERS * 2 * N**3 / dt / 1e12:.1f} TFLOPS')
