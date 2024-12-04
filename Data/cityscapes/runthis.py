from huggingface_hub import snapshot_download

snapshot_download("ahn1376/cityscapes_test", local_dir='.', repo_type='dataset')