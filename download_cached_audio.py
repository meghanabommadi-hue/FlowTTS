from huggingface_hub import snapshot_download
from huggingface_hub import login
login(token="**")

snapshot_download(
    repo_id="Shubhangi7/bajaj_cached_audio", 
    repo_type="dataset",
    local_dir="/home/ubuntu/FlowTTS/cached_data",
    local_dir_use_symlinks=False,
    resume_download=True
)