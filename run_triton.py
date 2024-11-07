import subprocess
from pathlib import Path
import contextlib
import time
from tritonclient.http import InferenceServerClient
# Define paths
model_name = "yolo"
triton_repo_path = Path("tmp") / "triton_repo"
triton_model_path = triton_repo_path / model_name

print(triton_repo_path)

# Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = "nvcr.io/nvidia/tritonserver:23.09-py3"

#subprocess.call(f"docker pull {tag}", shell=True)

# Run the Triton server and capture the container ID
container_id = (
    subprocess.check_output(
        f"docker run -d --rm -v {triton_repo_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
        shell=True,
    )
    .decode("utf-8")
    .strip()
)

print(container_id)
#docker run -d --rm -v ./tmp/triton_repo:/models -p 8000:8000 nvcr.io/nvidia/tritonserver:23.09-py3 tritonserver --model-repository=/models
triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

for _ in range(10):
    with contextlib.suppress(Exception):
        assert triton_client.is_model_ready(model_name)
        break
    time.sleep(1)