"""Shared WavLM speaker-sim cache for pair construction and quality checks."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np


def _fingerprint(audio_path: str) -> tuple[str, int, int]:
    real_path = os.path.realpath(audio_path)
    stat = os.stat(real_path)
    return real_path, stat.st_size, stat.st_mtime_ns


class DiskEmbeddingCache:
    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self._rows: dict[str, dict] | None = None
        self._dirty = False

    def _ensure_loaded(self) -> None:
        if self._rows is not None:
            return
        if self.cache_path.exists():
            with self.cache_path.open("rb") as f:
                self._rows = pickle.load(f)
        else:
            self._rows = {}

    def get(self, audio_path: str) -> np.ndarray | None:
        self._ensure_loaded()
        key, size, mtime_ns = _fingerprint(audio_path)
        row = self._rows.get(key)
        if row is None:
            return None
        if row.get("size") != size or row.get("mtime_ns") != mtime_ns:
            return None
        return np.asarray(row["embedding"], dtype=np.float32)

    def put(self, audio_path: str, embedding: np.ndarray) -> None:
        self._ensure_loaded()
        key, size, mtime_ns = _fingerprint(audio_path)
        self._rows[key] = {
            "size": size,
            "mtime_ns": mtime_ns,
            "embedding": np.asarray(embedding, dtype=np.float32),
        }
        self._dirty = True

    def save(self) -> None:
        self._ensure_loaded()
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(self._rows, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self.cache_path)
        self._dirty = False


class CachedSpeakerSimilarity:
    def __init__(self, speaker_similarity, cache_path: Path):
        self.speaker_similarity = speaker_similarity
        self.cache = DiskEmbeddingCache(cache_path)
        self.cache_hits = 0
        self.cache_misses = 0

    def embed_from_file(self, audio_path: str) -> np.ndarray:
        embedding = self.cache.get(audio_path)
        if embedding is not None:
            self.cache_hits += 1
            return embedding
        embedding = self.speaker_similarity.embed_from_file(audio_path)
        self.cache.put(audio_path, embedding)
        self.cache_misses += 1
        return embedding

    def compute_similarity_files(self, reference_audio: str, target_audio: str) -> float:
        emb_ref = self.embed_from_file(reference_audio)
        emb_tgt = self.embed_from_file(target_audio)
        return self.speaker_similarity.compute_similarity(emb_ref, emb_tgt)

    def save(self) -> None:
        self.cache.save()

    def stats(self) -> dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
