from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

try:
    from khmerspeech import (  # type: ignore
        cardinals,
        currency,
        datetime as km_datetime,
        decimals,
        dict_verbalize,
        normalize,
        ordinals,
        parenthesis,
        phone_numbers,
        punctuations,
        repeater,
        urls,
    )
except ImportError as exc:  # pragma: no cover - khmerspeech is required at runtime
    raise ImportError(
        "train_deepspeech2.py requires the 'khmerspeech' package; "
        "install it via `uv add khmerspeech` before running training."
    ) from exc


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingConfig:
    """Collection of hyper-parameters and runtime knobs for DeepSpeech2 training."""

    data_tsv: Path = Path("data/khmerasr-data-v1.txt")
    train_split: float = 0.95
    output_path: Path = Path("logs/ds2_small_clean.pt")
    seed: int = 42

    sample_rate: int = 16_000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80

    conv_channels1: int = 32
    conv_channels2: int = 64
    conv_kernel: Tuple[int, int] = (11, 5)
    conv_stride: Tuple[int, int] = (2, 2)
    conv_padding: Tuple[int, int] = (5, 2)

    rnn_hidden: int = 256
    rnn_layers: int = 3
    rnn_bidirectional: bool = True
    dropout: float = 0.1

    batch_size: int = 512
    lr: float = 1e-3
    epochs: int = 100
    grad_clip: float = 1.0
    log_interval: int = 25

    device: str | None = None
    num_workers: int = 2

    loss_curve_path: Path | None = None


CFG = TrainingConfig(
    data_tsv=Path("data/khmerasr-data-v1.txt"),
    train_split=0.5,
    output_path=Path("logs/ds2_small_clean.pt"),
    seed=42,
    sample_rate=16_000,
    n_fft=400,
    hop_length=160,
    n_mels=80,
    conv_channels1=32,
    conv_channels2=64,
    conv_kernel=(11, 5),
    conv_stride=(2, 2),
    conv_padding=(5, 2),
    rnn_hidden=256,
    rnn_layers=3,
    rnn_bidirectional=True,
    dropout=0.1,
    batch_size=8,
    lr=1e-3,
    epochs=100,
    grad_clip=1.0,
    log_interval=25,
    device=None,
    num_workers=2,
    loss_curve_path=None,
)


CONFIG_FILENAME = "config.json"


def training_config_to_json(cfg: TrainingConfig) -> dict[str, Any]:
    return {
        "data_tsv": str(cfg.data_tsv),
        "train_split": cfg.train_split,
        "output_path": str(cfg.output_path),
        "seed": cfg.seed,
        "sample_rate": cfg.sample_rate,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "n_mels": cfg.n_mels,
        "conv_channels1": cfg.conv_channels1,
        "conv_channels2": cfg.conv_channels2,
        "conv_kernel": list(cfg.conv_kernel),
        "conv_stride": list(cfg.conv_stride),
        "conv_padding": list(cfg.conv_padding),
        "rnn_hidden": cfg.rnn_hidden,
        "rnn_layers": cfg.rnn_layers,
        "rnn_bidirectional": cfg.rnn_bidirectional,
        "dropout": cfg.dropout,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "epochs": cfg.epochs,
        "grad_clip": cfg.grad_clip,
        "log_interval": cfg.log_interval,
        "device": cfg.device,
        "num_workers": cfg.num_workers,
        "loss_curve_path": str(cfg.loss_curve_path) if cfg.loss_curve_path else None,
    }


def training_config_from_json(payload: Mapping[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        data_tsv=Path(payload["data_tsv"]),
        train_split=float(payload["train_split"]),
        output_path=Path(payload["output_path"]),
        seed=int(payload["seed"]),
        sample_rate=int(payload["sample_rate"]),
        n_fft=int(payload["n_fft"]),
        hop_length=int(payload["hop_length"]),
        n_mels=int(payload["n_mels"]),
        conv_channels1=int(payload["conv_channels1"]),
        conv_channels2=int(payload["conv_channels2"]),
        conv_kernel=tuple(int(x) for x in payload["conv_kernel"]),
        conv_stride=tuple(int(x) for x in payload["conv_stride"]),
        conv_padding=tuple(int(x) for x in payload["conv_padding"]),
        rnn_hidden=int(payload["rnn_hidden"]),
        rnn_layers=int(payload["rnn_layers"]),
        rnn_bidirectional=bool(payload["rnn_bidirectional"]),
        dropout=float(payload["dropout"]),
        batch_size=int(payload["batch_size"]),
        lr=float(payload["lr"]),
        epochs=int(payload["epochs"]),
        grad_clip=float(payload["grad_clip"]),
        log_interval=int(payload["log_interval"]),
        device=payload["device"],
        num_workers=int(payload["num_workers"]),
        loss_curve_path=(
            Path(payload["loss_curve_path"]) if payload.get("loss_curve_path") else None
        ),
    )


def write_config_file(cfg: TrainingConfig, vocab: Sequence[str]) -> Path:
    payload = {"config": training_config_to_json(cfg), "vocab": list(vocab)}
    config_path = cfg.output_path.parent / CONFIG_FILENAME
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Persisted config metadata to %s", config_path)
    return config_path


# The training data contains many non-speech artifacts; these characters are removed
# from the vocabulary altogether.
IGNORED_CHARACTERS: Sequence[str] = [
    "!",
    '"',
    "#",
    "$",
    "(",
    ")",
    ",",
    "-",
    ".",
    "/",
    "។",
    "៕",
    "៖",
    "ៗ",
    "០",
    "១",
    "២",
    "៣",
    "៤",
    "៥",
    "៦",
    "៧",
    "៨",
    "៩",
    "​",
    "—",
    "“",
    "”",
    "취",
    "[",
    "]",
    ":",
    "?",
    "…",
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    *list("abcdefghijklmnopqrstuvwxyz"),
]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_khmer(text: str) -> str:
    text = normalize.processor(text)
    text = phone_numbers.processor(text, chunk_size=3)
    text = km_datetime.date_processor(text)
    text = km_datetime.time_processor(text)
    text = urls.processor(text)
    text = repeater.processor(text)
    text = currency.processor(text)
    text = cardinals.processor(text)
    text = decimals.processor(text)
    text = ordinals.processor(text)
    text = punctuations.processor(text)
    text = dict_verbalize(text)
    text = parenthesis.processor(text)
    return re.sub(r"\s+", " ", text.strip())


def load_dataset_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    df = pd.read_csv(path, sep="\t", names=["audio_path", "transcript"], dtype=str)
    df = df.dropna(subset=["audio_path", "transcript"])
    df["audio_path"] = df["audio_path"].map(lambda p: Path(p).expanduser().resolve())
    df["transcript"] = df["transcript"].map(normalize_khmer)

    df = df[(df["audio_path"] != "") & (df["transcript"] != "")]

    exists_mask = df["audio_path"].map(Path.exists)
    missing = int((~exists_mask).sum())
    if missing:
        LOGGER.warning("Skipping %d rows with missing audio files", missing)
    return df.loc[exists_mask].reset_index(drop=True)


def build_vocab(texts: Iterable[str], ignored_chars: Sequence[str]) -> List[str]:
    ignored = set(ignored_chars)
    symbols: set[str] = set()
    for text in texts:
        symbols.update(ch for ch in text if ch not in ignored)

    vocab = ["<pad>", "<blank>", *sorted(symbols)]
    LOGGER.info("Vocab built with %d symbols (ignored %d characters)", len(vocab), len(ignored))
    return vocab


class ASRDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, str, torch.Tensor]]):
    def __init__(self, df: pd.DataFrame, vocab: Sequence[str], sr: int) -> None:
        self.df = df.reset_index(drop=True)
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.sample_rate = sr
        self.allowed_chars = {c for c in vocab if c not in {"<pad>", "<blank>"}}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        row = self.df.iloc[idx]
        path: Path = row["audio_path"]
        text: str = row["transcript"]

        waveform, sr = torchaudio.load(path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        waveform = waveform.clamp(-1.0, 1.0)
        filtered = "".join(c for c in text if c in self.allowed_chars)
        if not filtered:
            raise ValueError(f"Transcript at index {idx} has no supported characters: {text!r}")
        labels = torch.tensor([self.char_to_idx[c] for c in filtered], dtype=torch.long)
        return waveform, filtered, labels


def pad_waveforms(batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([item.numel() for item in batch], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.zeros(len(batch), max_len)
    for i, wav in enumerate(batch):
        padded[i, : wav.numel()] = wav
    return padded, lengths


def collate_batch(batch: Sequence[Tuple[torch.Tensor, str, torch.Tensor]]):
    waveforms, texts, labels = zip(*batch)
    padded, lengths = pad_waveforms(waveforms)
    return padded, list(texts), list(labels), lengths


class LogMelExtractor(nn.Module):
    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
            center=False,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.spectrogram(wav)
        return self.to_db(spec).transpose(1, 2)


def _conv_out(length: torch.Tensor, kernel: int, stride: int, padding: int) -> torch.Tensor:
    return torch.div(
        length + (2 * padding) - (kernel - 1) - 1,
        stride,
        rounding_mode="floor",
    ) + 1


def feature_lengths(
    wav_lengths: torch.Tensor,
    cfg: TrainingConfig,
) -> torch.Tensor:
    spec_lengths = torch.div(
        torch.clamp(wav_lengths - cfg.n_fft, min=0),
        cfg.hop_length,
        rounding_mode="floor",
    ) + 1

    time_kernel = cfg.conv_kernel[0]
    time_stride = cfg.conv_stride[0]
    time_pad = cfg.conv_padding[0]

    conv1 = _conv_out(spec_lengths, time_kernel, time_stride, time_pad)
    conv2 = _conv_out(conv1, time_kernel, time_stride, time_pad)
    return torch.clamp(conv2, min=1)


class DeepSpeech2Small(nn.Module):
    def __init__(self, cfg: TrainingConfig, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=cfg.conv_channels1,
            kernel_size=cfg.conv_kernel,
            stride=cfg.conv_stride,
            padding=cfg.conv_padding,
        )
        self.bn1 = nn.BatchNorm2d(cfg.conv_channels1)
        self.conv2 = nn.Conv2d(
            in_channels=cfg.conv_channels1,
            out_channels=cfg.conv_channels2,
            kernel_size=cfg.conv_kernel,
            stride=cfg.conv_stride,
            padding=cfg.conv_padding,
        )
        self.bn2 = nn.BatchNorm2d(cfg.conv_channels2)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 200, cfg.n_mels)
            out = self.conv2(self.conv1(dummy))
            _, channels, frames, freq = out.size()
            rnn_input = channels * freq

        hidden = cfg.rnn_hidden
        bidir = 2 if cfg.rnn_bidirectional else 1

        self.rnn_stack = nn.ModuleList()
        input_dim = rnn_input
        for _ in range(cfg.rnn_layers):
            gru = nn.GRU(
                input_dim,
                hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=cfg.rnn_bidirectional,
            )
            self.rnn_stack.append(gru)
            input_dim = hidden * bidir

        self.norms = nn.ModuleList(nn.LayerNorm(hidden * bidir) for _ in range(cfg.rnn_layers))
        self.dropout = nn.Dropout(cfg.dropout)
        self.output = nn.Linear(hidden * bidir, vocab_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu_(x)
        x = self.dropout(x)

        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)

        for gru, norm in zip(self.rnn_stack, self.norms):
            x, _ = gru(x)
            x = norm(x)
            x = F.leaky_relu_(x)
            x = self.dropout(x)

        return F.log_softmax(self.output(x), dim=-1)


def greedy_decode(log_probs: torch.Tensor, blank_idx: int) -> List[List[int]]:
    predictions: List[List[int]] = []
    for sequence in log_probs.argmax(dim=-1):
        prev = -1
        decoded: List[int] = []
        for idx in sequence.tolist():
            if idx != prev and idx != blank_idx:
                decoded.append(idx)
            prev = idx
        predictions.append(decoded)
    return predictions


def indices_to_text(indices: List[int], vocab: Sequence[str]) -> str:
    return "".join(vocab[idx] for idx in indices if 0 <= idx < len(vocab))


def levenshtein_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    previous = list(range(len(hyp) + 1))
    current = [0] * (len(hyp) + 1)

    for i, ref_token in enumerate(ref, start=1):
        current[0] = i
        for j, hyp_token in enumerate(hyp, start=1):
            cost = 0 if ref_token == hyp_token else 1
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1
            substitution = previous[j - 1] + cost
            current[j] = min(deletion, insertion, substitution)
        previous, current = current, previous

    return previous[-1]


def compute_error_rates(pairs: Sequence[Tuple[str, str]]) -> Tuple[float, float]:
    word_distance = 0
    word_total = 0
    char_distance = 0
    char_total = 0

    for hyp, ref in pairs:
        ref_words = ref.split()
        hyp_words = hyp.split()
        word_distance += levenshtein_distance(ref_words, hyp_words)
        word_total += len(ref_words)

        ref_chars = list(ref)
        hyp_chars = list(hyp)
        char_distance += levenshtein_distance(ref_chars, hyp_chars)
        char_total += len(ref_chars)

    wer = word_distance / word_total if word_total else 0.0
    cer = char_distance / char_total if char_total else 0.0
    return wer, cer


def append_eval_metrics(path: Path, epoch: int, val_loss: float, wer: float, cer: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"epoch={epoch} val_loss={val_loss:.4f} WER={wer:.4f} CER={cer:.4f}\n"
        )


def pack_targets(labels: Sequence[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([label.numel() for label in labels], dtype=torch.long, device=device)
    if torch.any(lengths == 0):
        raise ValueError("Empty transcription encountered; ensure transcripts contain at least one symbol.")
    packed = torch.cat([label.to(device) for label in labels])
    return packed, lengths


def train_one_epoch(
    model: DeepSpeech2Small,
    feature_extractor: LogMelExtractor,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    cfg: TrainingConfig,
    device: torch.device,
    blank_idx: int,
) -> float:
    model.train()
    feature_extractor.train()
    total_loss = 0.0

    for step, (waveforms, _, labels, lengths) in enumerate(loader, start=1):
        waveforms = waveforms.to(device)
        lengths = lengths.to(device)

        feats = feature_extractor(waveforms)
        input_lengths = feature_lengths(lengths, cfg).to(device)
        targets, target_lengths = pack_targets(labels, device)

        log_probs = model(feats)
        log_probs_tnc = log_probs.transpose(0, 1)

        loss = criterion(log_probs_tnc, targets, input_lengths, target_lengths)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item()

        if step % cfg.log_interval == 0:
            LOGGER.info("Step %d/%d - loss: %.4f", step, len(loader), loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: DeepSpeech2Small,
    feature_extractor: LogMelExtractor,
    loader: torch.utils.data.DataLoader,
    criterion: nn.CTCLoss,
    cfg: TrainingConfig,
    device: torch.device,
    vocab: Sequence[str],
    blank_idx: int,
) -> Tuple[float, List[Tuple[str, str]]]:
    model.eval()
    feature_extractor.eval()

    total_loss = 0.0
    samples: List[Tuple[str, str]] = []

    for waveforms, texts, labels, lengths in loader:
        waveforms = waveforms.to(device)
        lengths = lengths.to(device)

        feats = feature_extractor(waveforms)
        input_lengths = feature_lengths(lengths, cfg).to(device)
        targets, target_lengths = pack_targets(labels, device)

        log_probs = model(feats)
        loss = criterion(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
        total_loss += loss.item()

        for hyp, ref in zip(greedy_decode(log_probs, blank_idx), texts):
            samples.append((indices_to_text(hyp, vocab), ref))

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, samples


@torch.no_grad()
def stream_decode(
    waveform: torch.Tensor,
    model: DeepSpeech2Small,
    feature_extractor: LogMelExtractor,
    cfg: TrainingConfig,
    device: torch.device,
    blank_idx: int,
    vocab: Sequence[str],
    chunk_ms: int = 640,
    hop_ms: int = 320,
) -> str:
    model.eval()
    feature_extractor.eval()

    sample_rate = cfg.sample_rate
    chunk = int(chunk_ms * sample_rate / 1000)
    hop = int(hop_ms * sample_rate / 1000)

    hypothesis: List[int] = []
    prev: int | None = None
    position = 0

    while position < waveform.numel():
        segment = waveform[position : position + chunk]
        if segment.numel() < chunk:
            segment = F.pad(segment, (0, chunk - segment.numel()))

        feats = feature_extractor(segment.unsqueeze(0).to(device))
        log_probs = model(feats)
        decoded = greedy_decode(log_probs, blank_idx)[0]

        if prev is not None and decoded and decoded[0] == prev:
            decoded = decoded[1:]
        if decoded:
            prev = decoded[-1]
            hypothesis.extend(decoded)

        position += hop

    return indices_to_text(hypothesis, vocab)


def save_checkpoint(model: nn.Module, cfg: TrainingConfig) -> None:
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.output_path)
    LOGGER.info("Checkpoint saved to %s", cfg.output_path)


def maybe_write_loss_curve(path: Path | None, train: List[float], val: List[float]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"train": train, "val": val}, handle, indent=2)
    LOGGER.info("Persisted loss curve to %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    cfg = TrainingConfig(**asdict(CFG))

    device = resolve_device(cfg.device)
    set_random_seed(cfg.seed)

    LOGGER.info("Loading dataset from %s", cfg.data_tsv)
    df = load_dataset_tsv(cfg.data_tsv)
    if df.empty:
        raise RuntimeError("Dataset is empty after filtering; check manifest and normalization.")

    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    vocab = build_vocab(df["transcript"].tolist(), IGNORED_CHARACTERS)
    blank_idx = vocab.index("<blank>")

    train_size = int(len(df) * cfg.train_split)
    if train_size == 0 or train_size == len(df):
        raise RuntimeError("Invalid train/validation split; provide more data or adjust --train-split.")

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    train_ds = ASRDataset(train_df, vocab, cfg.sample_rate)
    val_ds = ASRDataset(val_df, vocab, cfg.sample_rate)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    write_config_file(cfg, vocab)

    feature_extractor = LogMelExtractor(cfg).to(device)
    model = DeepSpeech2Small(cfg, vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = math.inf
    best_model_saved = False

    LOGGER.info("Starting training for %d epochs on device %s", cfg.epochs, device)
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model,
            feature_extractor,
            train_loader,
            optimizer,
            criterion,
            cfg,
            device,
            blank_idx,
        )
        val_loss, samples = evaluate(
            model,
            feature_extractor,
            val_loader,
            criterion,
            cfg,
            device,
            vocab,
            blank_idx,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        wer, cer = compute_error_rates(samples)

        LOGGER.info("Epoch %d/%d - train loss: %.4f - val loss: %.4f", epoch, cfg.epochs, train_loss, val_loss)
        for hyp, ref in samples[:3]:
            LOGGER.info("  hyp: %s | ref: %s", hyp, ref)
        LOGGER.info("  WER: %.2f%% | CER: %.2f%%", wer * 100, cer * 100)

        append_eval_metrics(cfg.output_path.parent / "evaluate.txt", epoch, val_loss, wer, cer)

        improved = math.isfinite(val_loss) and (
            not math.isfinite(best_val_loss) or val_loss < best_val_loss
        )
        if improved:
            best_val_loss = val_loss
            save_checkpoint(model, cfg)
            best_model_saved = True
            LOGGER.info("Saved new best model checkpoint at epoch %d (val_loss=%.4f)", epoch, val_loss)
        elif not best_model_saved:
            best_val_loss = val_loss
            save_checkpoint(model, cfg)
            best_model_saved = True
            LOGGER.info("Saved initial model checkpoint after epoch %d", epoch)

    maybe_write_loss_curve(cfg.loss_curve_path, train_losses, val_losses)

    if len(val_ds) > 0:
        wav, ref, _ = val_ds[0]
        decoded = stream_decode(wav, model, feature_extractor, cfg, device, blank_idx, vocab)
        LOGGER.info("Streaming decode example\n  hyp: %s\n  ref: %s", decoded, ref)


if __name__ == "__main__":
    main()


import torch
import torchaudio
import pandas as pd
from pathlib import Path
from torch import nn
from typing import Sequence, Iterable, Tuple
from loguru import logger as LOGGER


def load_dataset_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    df = pd.read_csv(path, sep="\t", names=["audio_path", "transcript"], dtype=str)
    df = df.dropna(subset=["audio_path", "transcript"])
    df["audio_path"] = df["audio_path"].map(lambda p: Path(p).expanduser().resolve())
    df["transcript"] = df["transcript"].map(normalize_khmer)

    df = df[(df["audio_path"] != "") & (df["transcript"] != "")]

    exists_mask = df["audio_path"].map(Path.exists)
    missing = int((~exists_mask).sum())
    if missing:
        LOGGER.warning("Skipping %d rows with missing audio files", missing)
    return df.loc[exists_mask].reset_index(drop=True)


def build_vocab(texts: Iterable[str], ignored_chars: Sequence[str]) -> list[str]:
    ignored = set(ignored_chars)
    symbols: set[str] = set()
    for text in texts:
        symbols.update(ch for ch in text if ch not in ignored)

    vocab = ["<pad>", "<blank>", *sorted(symbols)]
    LOGGER.info("Vocab built with %d symbols (ignored %d characters)", len(vocab), len(ignored))
    return vocab


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Sequence[str], sr: int, n_mels: int = 80) -> None:
        """
        Dataset that loads audio and converts it to log-Mel spectrograms,
        while tokenizing transcripts using your own vocab.
        """
        self.df = df.reset_index(drop=True)
        self.sample_rate = sr
        self.n_mels = n_mels

        # char-to-index mapping
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.allowed_chars = {c for c in vocab if c not in {"<pad>", "<blank>"}}

        # Mel-spectrogram extractor
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            power=2.0,
            center=False,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path: Path = row["audio_path"]
        text: str = row["transcript"]

        # ---- Load audio ----
        waveform, sr = torchaudio.load(path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        waveform = waveform.clamp(-1.0, 1.0)

        # ---- Convert to log-mel spectrogram ----
        mel = self.mel_extractor(waveform)
        mel_db = self.to_db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        mel_db = mel_db.transpose(0, 1)  # (time, n_mels)

        # ---- Tokenize transcript using vocab ----
        filtered = "".join(c for c in text if c in self.allowed_chars)
        if not filtered:
            raise ValueError(f"Transcript at index {idx} has no supported characters: {text!r}")
        label_ids = torch.tensor([self.char_to_idx[c] for c in filtered], dtype=torch.long)

        return mel_db, label_ids


def pad_spectrograms(batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([spec.size(0) for spec in batch], dtype=torch.long)
    max_len = int(lengths.max())
    n_mels = batch[0].size(1)

    padded = torch.zeros(len(batch), max_len, n_mels)
    for i, spec in enumerate(batch):
        padded[i, : spec.size(0)] = spec
    return padded, lengths


def collate_batch(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    specs, labels = zip(*batch)

    padded_specs, spec_lengths = pad_spectrograms(specs)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    max_label_len = int(label_lengths.max())

    padded_labels = torch.zeros(len(labels), max_label_len, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, : len(l)] = l

    return padded_specs, spec_lengths, padded_labels, label_lengths
