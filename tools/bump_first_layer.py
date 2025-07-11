import numpy as np, onnx
from onnx import numpy_helper

fname_in  = "nnverify/nets/mnist-net_256x2.onnx"
fname_out = "nnverify/nets/mnist-net_260x2.onnx"
delta     = 4                 # neurons to add
rng       = np.random.default_rng(0)

model = onnx.load(fname_in)
graph = model.graph
gemms = [n for n in graph.node if n.op_type == "Gemm"]
assert len(gemms) >= 2, "Expected at least two Gemm layers"

# ---------- 1) First hidden layer: add rows ----------
gemm1 = gemms[0]
W1 = next(t for t in graph.initializer if t.name == gemm1.input[1])
b1 = next(t for t in graph.initializer if t.name == gemm1.input[2])

W1_arr = numpy_helper.to_array(W1).astype(np.float32)     # (256, 784)
b1_arr = numpy_helper.to_array(b1).astype(np.float32)     # (256,)

W1_pad = rng.normal(0, 0.02, size=(delta, W1_arr.shape[1])).astype(np.float32)
b1_pad = rng.normal(0, 0.02, size=delta).astype(np.float32)

W1_new = np.vstack([W1_arr, W1_pad])
b1_new = np.concatenate([b1_arr, b1_pad])

W1.CopyFrom(numpy_helper.from_array(W1_new, W1.name))
b1.CopyFrom(numpy_helper.from_array(b1_new, b1.name))

# ---------- 2) Second hidden layer: add columns ----------
gemm2 = gemms[1]
W2 = next(t for t in graph.initializer if t.name == gemm2.input[1])

W2_arr = numpy_helper.to_array(W2).astype(np.float32)     # (256, 256)
col_pad = rng.normal(0, 0.02, size=(W2_arr.shape[0], delta)).astype(np.float32)
W2_new = np.hstack([W2_arr, col_pad])                     # (256, 260)

W2.CopyFrom(numpy_helper.from_array(W2_new, W2.name))

onnx.save(model, fname_out)
print("Saved", fname_out, "with shapes:",
      W1_new.shape, "and", W2_new.shape)
