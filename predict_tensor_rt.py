import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# --- Load TensorRT engine ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

with open("yolov8.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# --- Allocate buffers ---
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

for i in range(engine.num_bindings):
    size = trt.volume(engine.get_binding_shape(i)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(i))

    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    bindings.append(int(device_mem))
    if engine.binding_is_input(i):
        inputs.append({"host": host_mem, "device": device_mem})
    else:
        outputs.append({"host": host_mem, "device": device_mem})

# --- Preprocess frame ---
def preprocess(frame, input_shape=(640,640)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)   # batch
    return img

def do_inference(context, bindings, inputs, outputs, stream):
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_v2(bindings=bindings)
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()
    return outputs[0]["host"]

# --- Webcam loop ---
cap = cv2.VideoCapture(0)  # 0 = default webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = preprocess(frame, (640,640))
    np.copyto(inputs[0]["host"], img.ravel())

    preds = do_inference(context, bindings, inputs, outputs, stream)

    # TODO: postprocess preds into boxes/classes
    # For now, just show the original webcam feed
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
