from convert_data import RussianOpenSTTPreparer


if __name__ == "__main__":
    arch_prefix = "asr_public_phone_calls_1"

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

    # Step 1: Download, extract, and re-root paths
    prep.prepare()

    # Step 2: Split the dataset into train/val/test
    split_folder = f"{arch_prefix}_dataset"
    prep.split_dataset(
        target_base=split_folder,
        test_size=0.001,
        val_size=0.001,
        copy_files=True
    )

    # Step 3: Convert splits to Hugging Face dataset and save to disk
    hf_dataset = prep.to_huggingface_dataset(
        split_dir=split_folder,
        save_path=f"{arch_prefix}_dataset",
        sampling_rate=16000
    )
