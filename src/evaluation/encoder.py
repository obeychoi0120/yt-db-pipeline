"""BGE text embeddings (CLS token, L2-normalised), as the downstream encoder uses them."""

from __future__ import annotations

import numpy as np


class TextEncoder:
    def __init__(self, model_path, *, max_length=512, batch_size=64, device=None):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("evaluation requires torch and transformers") from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        with self._torch.no_grad():
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True,
                                         max_length=self.max_length, return_tensors="pt")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                hidden = self.model(**encoded).last_hidden_state[:, 0]
                hidden = self._torch.nn.functional.normalize(hidden, p=2, dim=1)
                for text, vector in zip(batch, hidden.cpu().numpy().astype(np.float32)):
                    self._cache[text] = vector
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([self._cache[text] for text in texts])
