"""RadBERT labelling — the same channel the official scorer reads through.

CRG and macro-F1 are not computed on the text; they are computed on 18 labels
that ``reportgen_evaluation`` extracts from the text with
``RadBertClassifier.pth``. The ensemble votes and repairs in that label space,
so it has to run *that* classifier, the same way. A different labeller — or the
same one configured differently — would optimise a proxy of the metric instead
of the metric.

TWO CALLERS, TWO TOKENISATIONS, BOTH DELIBERATE
-----------------------------------------------
``label_documents`` reproduces the scorer's own path
(``reportgen_evaluation/src/infer.py`` -> ``dataset.CTDataset`` ->
``model_trainer.infer``): ``padding='max_length'``, max_length 512, batch 32,
binarised at 0.5. This is what the vote and the verify loop's read-back use,
and it must match the scorer bit for bit.

``label_sentences`` reproduces PETRG-3D ``scripts/label_report_sentences.classify``:
``padding=True`` and probabilities returned unbinarised. RoBERTa derives its
position ids from the padding mask, so padded positions shift nothing and this
is numerically identical to ``max_length`` padding while being ~10x faster on
sentence-length inputs — which matters because the sentence pool is ~20x the
document count.

CAVEAT, stated because it is real: the classifier was trained on whole reports,
so single sentences are out of distribution. The merge uses those labels only
to FILTER candidate sentences, never as ground truth; every decision that
reaches the output is confirmed by ``label_documents``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RADBERT_DIR_NAME = "radbert/radbert_local"
RADBERT_CKPT_NAME = "radbert/RadBertClassifier.pth"

N_CLASSES = 18
MAX_LENGTH = 512
DOC_BATCH = 32          # infer.py's DataLoader batch_size
SENT_BATCH = 128        # label_report_sentences.py's default
THRESHOLD = 0.5         # ModelTrainer's default, used by infer.py


class RadBertClassifier(nn.Module):
    """Verbatim copy of ``reportgen_evaluation/src/classifier.py``.

    Copied rather than imported because that module hard-codes
    ``MODEL_DIR = /opt/app/radbert_local`` at import time. The class shape
    defines the state_dict keys, and ``strict=True`` below verifies the copy.
    """

    def __init__(self, model_dir: str | Path, n_classes: int = N_CLASSES):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.config = AutoConfig.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(str(model_dir), config=self.config)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attn_mask):
        output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        return self.classifier(output.pooler_output)


class RadBertLabeller:
    def __init__(self, weights_dir: str | Path, device: str = "cuda"):
        from transformers import AutoTokenizer

        weights_dir = Path(weights_dir)
        model_dir = weights_dir / RADBERT_DIR_NAME
        ckpt = weights_dir / RADBERT_CKPT_NAME
        for p in (model_dir / "config.json", ckpt):
            if not p.exists():
                raise FileNotFoundError(f"RadBERT asset missing: {p}")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir),
                                                       do_lower_case=True)
        model = RadBertClassifier(model_dir)
        state = torch.load(str(ckpt), map_location="cpu")
        # Older transformers saved `position_ids` as a persistent buffer; newer
        # versions removed it. infer.py strips it the same way.
        state = {k: v for k, v in state.items() if not k.endswith("position_ids")}
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _run(self, texts: list[str], padding, batch_size: int) -> np.ndarray:
        out = np.zeros((len(texts), N_CLASSES), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            chunk = [t if isinstance(t, str) and t.strip() else " "
                     for t in texts[i:i + batch_size]]
            enc = self.tokenizer(chunk, return_tensors="pt",
                                 max_length=MAX_LENGTH, padding=padding,
                                 truncation=True)
            logits = self.model(enc["input_ids"].to(self.device),
                                enc["attention_mask"].to(self.device))
            out[i:i + len(chunk)] = torch.sigmoid(logits.float()).cpu().numpy()
        return out

    def label_documents(self, reports: list[str]) -> np.ndarray:
        """(N, 18) binary — the scorer's own path."""
        probs = self._run(reports, "max_length", DOC_BATCH)
        return (probs >= THRESHOLD).astype(np.int8)

    def label_sentences(self, sentences: list[str]) -> np.ndarray:
        """(N, 18) sigmoid probabilities — the merge's candidate filter."""
        return self._run(sentences, True, SENT_BATCH)
