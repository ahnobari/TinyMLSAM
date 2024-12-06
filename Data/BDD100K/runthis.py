from huggingface_hub import snapshot_download

snapshot_download("ahn1376/bdd100K_test", local_dir='.', repo_type='dataset')