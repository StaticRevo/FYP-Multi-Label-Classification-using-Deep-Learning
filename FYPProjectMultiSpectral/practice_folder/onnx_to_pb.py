import onnx
from onnx_tf.backend import prepare

# Define paths
onnx_model_path = r"C:\Users\isaac\Desktop\Experiment Folders\converted_model.onnx"
tf_model_path = r"C:\Users\isaac\Desktop\Experiment Folders\tf_model"

# Load ONNX model
onnx_model = onnx.load(onnx_model_path)

# Convert ONNX model to TensorFlow format
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)

print(f"Model successfully converted to TensorFlow format at: {tf_model_path}")
