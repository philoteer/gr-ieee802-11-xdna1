from pathlib import Path

from .dataset import ChannelCorpus, concatenate_corpora


DEFAULT_DATASET_DIRECTORY = Path(__file__).resolve().parents[3] / "data" / "datasets" / "channel_estimator"


def load_corpus(directory: Path, dataset: str, split: str) -> ChannelCorpus:
    if dataset == "both":
        sources = ("synthetic_original", "synthetic_gnuradio")
    else:
        names = dataset.split(",")
        if any(not name or Path(name).name != name or name in (".", "..") for name in names):
            raise ValueError("dataset must be a source name within the dataset directory")
        source_directories = {"original": "synthetic_original", "gnuradio": "synthetic_gnuradio"}
        sources = tuple(source_directories.get(name, name) for name in names)
    corpora = []
    for source in sources:
        path = directory / source / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"dataset split not found: {path}; prepare it with "
                f"`python3 utils/frame_equalizer_channel_ai/dataset_generator/"
                f"generate_dataset.py --convert <raw_samples.npz> --output {path}`")
        corpora.append(ChannelCorpus.load(path))
    return concatenate_corpora(*corpora)
