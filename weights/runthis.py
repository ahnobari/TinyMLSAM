from huggingface_hub import snapshot_download

snapshot_download("ahn1376/TinySAMWeights", local_dir='.', repo_type='model')