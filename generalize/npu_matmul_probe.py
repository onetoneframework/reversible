"""Can the NPU / iGPU do the integer model's GEMM exactly through OpenVINO?

16-bit activations x int8 weights summed over 192 (|sum| < 2**31): exact in
int32/int64 and in float64, NOT exact in float32 (24-bit mantissa) or float16.
"""
import time

import numpy as np
import openvino as ov
from openvino import opset13 as ops

core = ov.Core()
rng = np.random.default_rng(0)
X = rng.integers(-32768, 32767, (512, 192)).astype(np.int64)
W = rng.integers(-128, 127, (192, 384)).astype(np.int64)
ref = X @ W


def run(dev, xdt, wdt, name):
    px = ops.parameter([512, 192], xdt, name="x")
    w = ops.constant(W.astype(wdt))
    y = ops.matmul(px, w, False, False)
    model = ov.Model([y], [px], name)
    try:
        t = time.time()
        cm = core.compile_model(model, dev)
        tc = time.time() - t
        out = cm(X.astype(xdt))[cm.output(0)]
        got = np.rint(out.astype(np.float64)).astype(np.int64)
        exact = bool(np.array_equal(got, ref))
        maxerr = int(np.abs(got - ref).max())
        t = time.time()
        for _ in range(20):
            cm(X.astype(xdt))
        ms = (time.time() - t) / 20 * 1e3
        print("%-4s %-6s in=%-8s w=%-8s exact=%-5s maxerr=%-10d compile %.1fs  %.2f ms/call"
              % (dev, name, np.dtype(xdt).name, np.dtype(wdt).name, exact, maxerr, tc, ms), flush=True)
    except Exception as e:
        print("%-4s %-6s in=%-8s w=%-8s FAILED: %s" % (dev, name, np.dtype(xdt).name, np.dtype(wdt).name,
                                                       str(e).splitlines()[0][:120]), flush=True)


for dev in ("CPU", "GPU", "NPU"):
    run(dev, np.int32, np.int32, "i32mm")
    run(dev, np.float32, np.float32, "f32mm")
    run(dev, np.int64, np.int64, "i64mm")
    run(dev, np.float16, np.float16, "f16mm")
