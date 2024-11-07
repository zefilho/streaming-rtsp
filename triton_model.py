from pathlib import Path

# Define paths
model_name = "yolo"
triton_repo_path = Path("tmp") / "triton_repo"
triton_model_path = triton_repo_path / model_name

# Create directories
(triton_model_path / "1").mkdir(parents=True, exist_ok=True)
Path('yolo11n.onnx').rename(triton_model_path / "1" / "model.onnx")
(triton_model_path / "config.pbtxt").touch()
