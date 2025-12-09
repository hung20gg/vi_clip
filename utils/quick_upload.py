from huggingface_hub import hf_hub_download, HfApi, login

api = HfApi()

path = '../../data/cc12m-clip-b224/merged_output_fixed.parquet'

api.upload_file(
        path_or_fileobj=path,
        path_in_repo='merged_output_fixed.parquet',
        repo_id='hung20gg/cc12m_clip_b224',
        repo_type="dataset",
    )