"""`phrase_align_polar` -- PhraseAlign local-only loss for polarity+anatomy (58-dim) phrase labels.

Nothing in `loss_phrase_align.py` is modified. This subclasses `PhraseAlignLoss` and
overrides exactly the three methods whose assumptions break once phrase labels stop
being sparse presence-only probabilities, so `phrase_align`/`phrase_align_split`/`lapa`
keep running bit-identically off the original file.

WHAT CHANGED AND WHY
--------------------
The 58-dim labels (see `scripts/build_phrase_align_polar_labels.py`) lift the fraction of
phrases carrying usable label evidence from a measured 27.4% to 97.8% of instances,
by adding an explicit ABSENT block for negated findings and an anatomy block for the
regions the 18-class taxonomy cannot represent at all. Three consequences follow, and
each needs a code change; without them the new labels are silently worse than the old
ones.

1. `known_pair_mask` no longer means "rare". At 27% known, filing every
   label-unknown pair as a hard negative was the only way to have negatives at all.
   At 98% known that rule is both unnecessary and actively wrong: it is what would
   turn a deliberately-abstained "uncertain" polarity (an all-zero row, 1.6% of
   instances -- hedged and unevaluable sentences) back into a hard negative against
   everything. `unknown_pair_as_ignore` moves those pairs to the ignore zone, which
   is where this loss already puts every pair it has no evidence about.

2. Boilerplate normals are label-positive to nearly every volume in the window, so
   their negative set empties out. "Normal upper abdominal organs within the field of
   view" recurs in ~every CT-RATE report, so once it is `known` its
   `label_sim_to_image` is 1.0 against almost the whole batch. An empty negative set
   makes `logsumexp` return `finfo.min`, `softplus(finfo.min - pos)` return exactly
   0, and the existing unconditional `.mean()` then averages those zeros into the
   loss. Measured on a synthetic 8-volume/32-phrase batch with three shared
   boilerplate normals: 24/32 rows go dead and source_loss is diluted 1.70x
   (4.0845 vs 6.9276). That is the same failure mode that killed job 27526199, where
   promoted duplicates diluted phrase_pair_loss from 13.2301 to 3.7621.
   `source_valid_row_only` restricts the reduction to rows that actually have a
   negative, matching what `_weighted_simlabel_pair_nce_loss` and
   `_phrase_pair_nce_loss` already do via their `row_valid` guards -- source_loss
   never needed one before because rule (1) guaranteed every row had negatives.

   Note this does not stop boilerplate phrases from dropping out of source_loss, and
   it should not: a sentence present in every report has no evidence for pointing at
   one volume, and forcing it to was teaching noise. The consequence is that
   source_loss narrows from "every phrase" to "discriminative phrases", and the
   clustering of normals is carried by `phrase_pair_loss` instead -- so
   `phrase_pair_weight` is likely to want raising above 0.5. Watch
   `source_valid_row_frac` in the logs.

3. The label discretization has to move upstream. `_prepare_soft_label_map`'s
   filters were built for raw classifier probabilities: `label_prob_threshold=0.4`
   zeroes the ABNORMAL block outright (its weight is 0.25 -- verified: block sum
   1.50 -> 0.00), and `clamp(max=1.0)` would crush any weight above 1.0, collapsing
   the intended NORMAL:ABSENT = 2:1 ratio. The tagger therefore emits already
   discretized values, all <= 1.0, and the three filters default to off here. Do not
   re-enable them for 58-dim labels.

STILL TO CALIBRATE (do not assume the inherited defaults transfer)
    label_pos_threshold / label_neg_threshold (0.8 / 0.4) -- the similarity
    distribution is no longer near-binary; measure a histogram first.
    label_sim_weight / phrase_pair_weight (0.5 / 0.5) -- see (2).
    label_balance_mode -- per-column counts now include the NORMAL block, whose
    columns are both more frequent and more heavily weighted than the rest.
"""

import logging
from typing import Optional

import torch
from torch.nn import functional as F

from .loss_phrase_align import PhraseAlignLoss


class PhraseAlignPolarLoss(PhraseAlignLoss):
    """local-only PhraseAlign loss for 58-dim polarity+anatomy labels.

        total_loss = source_loss
                   + label_sim_weight   * simlabel_pair_loss
                   + phrase_pair_weight * phrase_pair_loss

    Same three terms and same absence of clip_loss as `PhraseAlignLoss` (see its
    docstring for why VL-CABS is computed here rather than in model.forward). Both
    behavioural changes are flagged and default on; turning both off plus restoring
    the three prob filters reproduces `PhraseAlignLoss` exactly, which is how the
    no-op path is verified.
    """

    def __init__(
            self,
            unknown_pair_as_ignore: bool = True,
            source_valid_row_only: bool = True,
            # The 58-dim labels arrive already discretized, so the inherited
            # probability filters are off by default here. See point (3) above.
            label_prob_threshold: float = 0.0,
            label_prob_relative_threshold: float = 0.0,
            label_prob_topk: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(
            label_prob_threshold=label_prob_threshold,
            label_prob_relative_threshold=label_prob_relative_threshold,
            label_prob_topk=label_prob_topk,
            **kwargs,
        )
        self.unknown_pair_as_ignore = bool(unknown_pair_as_ignore)
        self.source_valid_row_only = bool(source_valid_row_only)

        if label_prob_threshold > 0 or label_prob_relative_threshold > 0 or label_prob_topk:
            logging.warning(
                "PhraseAlignPolarLoss: a probability filter is enabled "
                "(threshold=%s, relative=%s, topk=%s). With 58-dim polarity labels the "
                "absolute threshold zeroes the anatomy-ABNORMAL block (weight 0.25) and "
                "the relative/topk filters drop columns the tagger deliberately set. "
                "Only do this to reproduce the pre-polar behaviour.",
                label_prob_threshold, label_prob_relative_threshold, label_prob_topk,
            )

        # Diagnostics filled in per forward pass and surfaced through the output dict.
        self._source_valid_row_frac = None
        self._source_valid_col_frac = None

    # ------------------------------------------------------------------
    # (1) unknown pairs -> ignore instead of negative
    # ------------------------------------------------------------------
    def _build_source_and_label_similarity(self, *args, **kwargs):
        masks = super()._build_source_and_label_similarity(*args, **kwargs)
        if not self.unknown_pair_as_ignore:
            return masks

        # The parent's negative set is
        #   non_source & ~pos & ((~known_pair) | (sim <= neg_thr))
        # so intersecting with known_pair leaves exactly
        #   non_source & ~pos & known_pair & (sim <= neg_thr)
        # i.e. only pairs where both sides carry label evidence AND that evidence says
        # they are dissimilar. Everything else falls through to ignore below.
        known_pair = masks["label_known"].view(-1, 1) & masks["image_known"].view(1, -1)
        masks["label_neg_mask"] = masks["label_neg_mask"] & known_pair
        masks["label_ignore_mask"] = (
            (~masks["source_pos_mask"])
            & (~masks["label_pos_mask"])
            & (~masks["label_neg_mask"])
        )
        return masks

    def _build_phrase_pair_masks(self, *args, **kwargs):
        pair = super()._build_phrase_pair_masks(*args, **kwargs)
        if not self.unknown_pair_as_ignore:
            return pair

        known = pair["phrase_known"]
        known_pair = known.view(-1, 1) & known.view(1, -1)
        pair["phrase_neg_mask"] = pair["phrase_neg_mask"] & known_pair
        n = pair["phrase_neg_mask"].size(0)
        not_self = ~torch.eye(n, dtype=torch.bool, device=pair["phrase_neg_mask"].device)
        pair["phrase_ignore_mask"] = (
            not_self & (~pair["phrase_pos_mask"]) & (~pair["phrase_neg_mask"])
        )
        return pair

    # ------------------------------------------------------------------
    # (2) source_loss: reduce over rows that actually have a negative
    # ------------------------------------------------------------------
    def _masked_source_pair_nce_loss(
            self,
            logits: torch.Tensor,
            group_map: torch.Tensor,
            source_pos_mask: torch.Tensor,
            label_neg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.source_valid_row_only:
            return super()._masked_source_pair_nce_loss(
                logits=logits, group_map=group_map,
                source_pos_mask=source_pos_mask, label_neg_mask=label_neg_mask,
            )

        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got {logits.shape}")

        device = logits.device
        N, B = logits.shape

        if group_map.numel() != N:
            raise ValueError(
                f"group_map length mismatch: group_map={group_map.numel()}, logits rows={N}"
            )
        if group_map.min().item() < 0 or group_map.max().item() >= B:
            raise ValueError(
                f"group_map index out of range: min={group_map.min().item()}, "
                f"max={group_map.max().item()}, B={B}"
            )

        z = logits
        neg_inf = torch.finfo(z.dtype).min
        pos_z = z[torch.arange(N, device=device), group_map]

        if self.except_label_positive_in_source_neg and label_neg_mask is not None:
            neg_mask = label_neg_mask
        else:
            neg_mask = ~source_pos_mask

        neg_z = z.masked_fill(~neg_mask, neg_inf)

        # Row direction: phrase -> image. A row with no negative contributes
        # softplus(finfo.min - pos) == 0; averaging that in is the dilution this guard
        # exists to prevent, so those rows leave the reduction entirely.
        row_valid = neg_mask.any(dim=1)
        row_loss = F.softplus(torch.logsumexp(neg_z, dim=1) - pos_z)
        if row_valid.any():
            row_term = row_loss[row_valid].mean()
        else:
            row_term = z.sum() * 0.0

        # Column direction: image -> phrase. Validity is a property of the image
        # column, so it maps back to phrases through group_map.
        col_valid_image = neg_mask.any(dim=0)
        col_valid = col_valid_image[group_map]
        col_loss = F.softplus(torch.logsumexp(neg_z, dim=0)[group_map] - pos_z)
        if col_valid.any():
            col_term = col_loss[col_valid].mean()
        else:
            col_term = z.sum() * 0.0

        self._source_valid_row_frac = float(row_valid.float().mean().item())
        self._source_valid_col_frac = float(col_valid.float().mean().item())

        return 0.5 * (row_term + col_term)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if isinstance(out, dict):
            # How much of the batch source_loss is actually learning from. Expected to
            # sit well below 1.0 once boilerplate normals become `known` -- that is the
            # narrowing described in point (2), not a bug, but a collapse toward 0 means
            # label_pos_threshold is too permissive and everything has become a positive.
            if self._source_valid_row_frac is not None:
                out["source_valid_row_frac"] = torch.as_tensor(self._source_valid_row_frac)
            if self._source_valid_col_frac is not None:
                out["source_valid_col_frac"] = torch.as_tensor(self._source_valid_col_frac)
        return out
