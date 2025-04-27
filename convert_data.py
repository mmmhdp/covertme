import os
import requests
import tarfile
from tqdm import tqdm
import pandas as pd
import soundfile as sf
from IPython.display import Audio, display
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from pathlib import Path
from datasets import load_dataset, Audio
import subprocess
import glob
import os
import shutil
from typing import Optional


class RussianOpenSTTPreparer:
    def __init__(
        self,
        dataset_url: str,
        manifest_url: str,
        archive_name: str = "asr_calls_2_val.tar.gz",
        data_dir: str = "data",
        manifest_local: str = "asr_calls_2_val.csv"
    ):
        """
        A pipeline class that:
          1) Downloads a chosen subset archive from Russian OpenSTT
          2) Extracts it into `data_dir` with a tqdm progress bar
          3) Reads the manifest and re-roots paths
          4) Converts any .opus files to .wav with parallel, multi-threaded ffmpeg
          5) Provides an interface to read/preview the data
          6) Splits the data into train/val/test with tqdm for file copies
          7) Exports as a HuggingFace dataset
        """
        self.dataset_url = dataset_url
        self.manifest_url = manifest_url
        self.archive_name = archive_name
        self.data_dir = data_dir
        self.manifest_local = manifest_local
        self.df = None

    def download_archive(self):
        if os.path.exists(self.archive_name):
            print(
                f"Archive {self.archive_name} already exists, skipping download.")
            return
        print(f"Downloading {self.dataset_url} → {self.archive_name}")
        resp = requests.get(self.dataset_url, stream=True)
        resp.raise_for_status()
        with open(self.archive_name, "wb") as f:
            for chunk in tqdm(resp.iter_content(8192), desc="Downloading archive", unit="B", unit_scale=True, unit_divisor=1024):
                if chunk:
                    f.write(chunk)

    def extract_archive(self):
        if not os.path.exists(self.archive_name):
            raise FileNotFoundError(
                f"{self.archive_name} not found. Run download_archive() first.")
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Extracting {self.archive_name} → {self.data_dir}/")
        with tarfile.open(self.archive_name, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting archive", unit="file"):
                tar.extract(member, path=self.data_dir)
        print("Extraction complete.")

    def download_manifest(self):
        if os.path.exists(self.manifest_local):
            print(f"Manifest {
                  self.manifest_local} already exists, skipping download.")
            return
        print(f"Downloading manifest {
              self.manifest_url} → {self.manifest_local}")
        r = requests.get(self.manifest_url)
        r.raise_for_status()
        with open(self.manifest_local, "wb") as f:
            f.write(r.content)

    def load_manifest(self):
        if not os.path.exists(self.manifest_local):
            raise FileNotFoundError(
                f"{self.manifest_local} not found. Run download_manifest() first.")
        self.df = pd.read_csv(self.manifest_local, names=[
                              "wav_path", "text_path", "duration"])
        print(f"Loaded manifest: {len(self.df)} entries")

    def reroot_paths(self, old_root: str, new_root: str):
        if self.df is None:
            raise RuntimeError("Call load_manifest() before reroot_paths().")

        def _r(x):
            return x.replace(old_root, new_root) if old_root in x else os.path.join(new_root, x)
        self.df["wav_path"] = self.df["wav_path"].apply(_r)
        self.df["text_path"] = self.df["text_path"].apply(_r)

    @staticmethod
    def _convert_task(args):
        """Top-level task for multiprocessing: converts one opus to wav."""
        idx, opus_path, threads = args
        wav_path = opus_path.with_suffix(".wav")

        threads = min(threads, os.cpu_count() or 1)

        cmd = [
            "ffmpeg", "-y",
            "-threads", str(threads),
            "-i", str(opus_path),
            "-ac", "1", "-ar", "16000",
            str(wav_path)
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"⚠️ Conversion failed for {opus_path}")
        return idx, str(wav_path)

    @staticmethod
    def _copy_file(args):
        """Top-level task for multiprocessing: copies one audio file."""
        src, dst = args
        shutil.copy(src, dst)
        return dst

    def prepare(self, ffmpeg_threads: int = 1, workers: int = os.cpu_count()):
        """Download, extract, load manifest, reroot, and convert any .opus files in parallel."""
        self.download_archive()
        self.extract_archive()
        self.download_manifest()
        self.load_manifest()

        base = self.archive_name.replace(".tar.gz", "")
        new_root = os.path.join(self.data_dir, base)
        self.reroot_paths(old_root=base, new_root=new_root)

        opus_items = [
            (i, Path(row["wav_path"]), ffmpeg_threads)
            for i, row in self.df.iterrows()
            if Path(row["wav_path"]).suffix.lower() == ".opus"
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, wav_str in tqdm(
                executor.map(RussianOpenSTTPreparer._convert_task, opus_items),
                total=len(opus_items),
                desc="Converting OPUS→WAV",
                unit="file"
            ):
                self.df.at[idx, "wav_path"] = wav_str

        print("Dataset ready (all .opus files converted).")

    def preview_sample(self, idx: int = 0):
        """Play one sample and show its transcript (in Jupyter)."""
        if self.df is None:
            raise RuntimeError("Call prepare() first.")
        row = self.df.iloc[idx]
        print("Audio:", row["wav_path"])
        print("Transcript file:", row["text_path"])
        audio, sr = sf.read(row["wav_path"])
        print(f"(shape={audio.shape}, sr={sr})")
        if os.path.exists(row["text_path"]):
            print("Text:", open(row["text_path"],
                  encoding="utf-8").read().strip())
        else:
            print("No transcript found.")
        display(Audio(audio, rate=sr))

    def split_dataset(
        self,
        target_base: str = "split_data",
        test_size: float = 0.15,
        val_size: float = 0.05,
        random_state: int = 42,
        copy_files: bool = True,
        copy_workers: int = 8
    ):
        """Split into train/val/test, optionally copy audio, and write metadata CSVs."""
        if self.df is None:
            raise RuntimeError("Call prepare() first.")

        self.df["wav_path"] = self.df["wav_path"].apply(Path)
        self.df["text_path"] = self.df["text_path"].apply(Path)

        self.df = self.df[self.df["wav_path"].apply(lambda p: p.exists())]
        print(f"Found {len(self.df)} valid audio files.")

        if "text" not in self.df.columns:
            self.df["text"] = self.df["text_path"].apply(
                lambda p: p.read_text(
                    encoding="utf-8").strip() if p.exists() else ""
            )

        train_val, test = train_test_split(
            self.df, test_size=test_size, random_state=random_state
        )
        val_frac_of_tv = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_frac_of_tv, random_state=random_state
        )

        for split_name, subset in [("train", train), ("validation", val), ("test", test)]:
            out_dir = Path(target_base) / split_name / "audio"
            out_dir.mkdir(parents=True, exist_ok=True)

            if copy_files:
                tasks = [
                    (row["wav_path"], out_dir / row["wav_path"].name)
                    for _, row in subset.iterrows()
                ]
                with ThreadPoolExecutor(max_workers=copy_workers) as executor:
                    list(tqdm(
                        executor.map(RussianOpenSTTPreparer._copy_file, tasks),
                        total=len(tasks),
                        desc=f"Copying {split_name} audio",
                        unit="file"
                    ))

            md = subset.copy()
            md["audio_filename"] = md["wav_path"].apply(lambda p: p.name)
            md = md[["audio_filename", "text", "duration"]]
            md.to_csv(Path(target_base)/split_name/"metadata.csv",
                      index=False, encoding="utf-8")
            print(f"{split_name.title()}: {len(subset)} samples → {out_dir}")

        print(f"All splits saved under `{target_base}`.")

    def to_huggingface_dataset(
        self,
        split_dir: str,
        save_path: str = "hf_dataset",
        sampling_rate: int = 16000
    ):
        """Convert split CSVs + audio into a HuggingFace `datasets` and save to disk."""
        print(f"Loading from splits in `{split_dir}`...")
        dataset = load_dataset(
            "csv",
            data_files={
                "train":      f"{split_dir}/train/metadata.csv",
                "validation": f"{split_dir}/validation/metadata.csv",
                "test":       f"{split_dir}/test/metadata.csv",
            }
        )

        def _attach(example, split):
            example["audio"] = f"{
                split_dir}/{split}/audio/{example['audio_filename']}"
            example["transcript"] = example["text"]
            return example

        for split in ["train", "validation", "test"]:
            dataset[split] = dataset[split].map(
                _attach, fn_kwargs={"split": split})

        dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=sampling_rate))
        print(f"Saving HuggingFace dataset to `{save_path}`...")
        dataset.save_to_disk(save_path)
        print("Done.")
        return dataset


def cleanup_data(prefix: str, base_dir: Optional[str] = None) -> None:
    """
    Delete all files and directories whose names start with `prefix`.

    Args:
        prefix: The filename/directory prefix to match (e.g. "prefix" will match "prefix_train", "prefix_tokenized_data", etc.).
        base_dir: Optional directory in which to look (defaults to current working directory).
    """
    # build the glob pattern
    if base_dir:
        pattern = os.path.join(base_dir, f"{prefix}*")
    else:
        pattern = f"{prefix}*"

    # find and delete matches
    matches = glob.glob(pattern)
    for path in matches:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Deleted directory: {path!r}")
        elif os.path.isfile(path):
            os.remove(path)
            print(f"Deleted file:      {path!r}")
        else:
            print(f"Skipped (not found or unknown type): {path!r}")
