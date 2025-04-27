from convert_data import RussianOpenSTTPreparer, cleanup_data
from datasets import load_from_disk
from huggingface_hub import login


if __name__ == "__main__":
    arch_prefix = "buriy_audiobooks_2_val"

    dataset_url = (
        "https://azureopendatastorage.blob.core.windows.net/openstt/"
        f"ru_open_stt_opus/archives/{arch_prefix}.tar.gz"
    )
    manifest_url = (
        "https://azureopendatastorage.blob.core.windows.net/openstt/"
        f"ru_open_stt_opus/manifests/{arch_prefix}.csv"
    )

    prep = RussianOpenSTTPreparer(
        dataset_url=dataset_url,
        manifest_url=manifest_url,
        archive_name=arch_prefix + ".tar.gz",
        data_dir="data",
        manifest_local=arch_prefix + ".csv"
    )

    prep.prepare()

    split_folder = f"{arch_prefix}_dataset"
    prep.split_dataset(
        target_base=split_folder,
        test_size=0.001,
        val_size=0.001,
        copy_files=True
    )

    hf_dataset = prep.to_huggingface_dataset(
        split_dir=split_folder,
        save_path=f"{arch_prefix}_dataset",
        sampling_rate=16000
    )

    HF_TOKEN = "PUT TOKEN HERE"

    login(token=HF_TOKEN)

    ds = load_from_disk(f"{arch_prefix}_dataset")

    repo_id = f"Malecc/{arch_prefix}"

    from huggingface_hub import create_repo
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

    ds.push_to_hub(repo_id=repo_id, token=HF_TOKEN)

    cleanup_data(arch_prefix)
