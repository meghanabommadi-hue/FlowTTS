from huggingface_hub import create_repo, HfApi
from huggingface_hub import login
login(token="**")

repo_id = "Shubhangi7/bajaj_cached_audio"
folder_path = "/home/ubuntu/FlowTTS/cache_dataset"

api = HfApi()

api.upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=folder_path,
)