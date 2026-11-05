import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn  # noqa: F401
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    dist = None
    has_distributed = False

def _all_gather_sizes(local_size, device, world_size=1):
    if world_size == 1:
        return torch.tensor([int(local_size)], device=device, dtype=torch.long)

    size = torch.tensor([int(local_size)], device=device, dtype=torch.long)
    gathered_sizes = [torch.zeros_like(size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, size)

    return torch.cat(gathered_sizes, dim=0)

def _gather_variable_1d(x, sizes, pad_value=0, world_size=1):
    """
    x:     (N_local,)
    sizes: 각 rank의 N_local
    return (sum_rank N_local,)
    """
    if world_size == 1:
        return x

    max_size = int(sizes.max().item())

    x_padded = torch.full((max_size,), fill_value=pad_value, device=x.device, dtype=x.dtype,)
    x_padded[:x.numel()] = x

    gathered = [torch.empty_like(x_padded) for _ in range(world_size)]
    dist.all_gather(gathered, x_padded)

    gathered = [tensor[:int(size.item())] for tensor, size in zip(gathered, sizes)]

    return torch.cat(gathered, dim=0)

def _gather_variable_2d(x, sizes, preserve_grad=True, world_size=1, gather_with_grad=True, rank=0):
    """
    x:     (N_local, D) or (N_local, Seq, D)  # 3D used for local (per-patch) image tokens
    sizes: 각 rank의 N_local
    return (sum_rank N_local, ...)
    """
    if world_size == 1:
        return x

    max_size = int(sizes.max().item())

    if x.size(0) < max_size:
        # *x.shape[1:] (not x.size(1)): the VL-CABS path gathers 3D local token maps
        # (N, Seq, D) through here, where a 2D-only pad would be non-concatenable.
        pad = x.new_zeros(max_size - x.size(0), *x.shape[1:])
        x_padded = torch.cat([x, pad], dim=0)
    else:
        x_padded = x

    if gather_with_grad and preserve_grad:
        gathered = list(torch.distributed.nn.all_gather(x_padded))
    else:
        gathered = [torch.empty_like(x_padded) for _ in range(world_size)]
        dist.all_gather(gathered, x_padded.detach())

        # local rank는 원본 tensor로 교체해서 gradient 유지
        if preserve_grad:
            gathered[rank] = x_padded

    gathered = [tensor[:int(size.item())] for tensor, size in zip(gathered, sizes)]

    return torch.cat(gathered, dim=0)

def gather_features(
        image_features,
        text_features,
        aggregated_image_features,
        key_phrase_features,
        label_map,
        counts,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        volume_label=None,
):
    """
    Args:
        image_features:            (B_local, D)
        text_features:             (B_local, D)
        aggregated_image_features: (B_local, D)
        key_phrase_features:       (N_phrase_local, D)  # variable
        label_map:                 list of phrase-level tensors
                                   after cat: (N_phrase_local, C)
        counts:                    nested list
                                   after flatten: (B_local,)
        volume_label:               optional (B_local, C) whole-scan label, one row per
                                   sample (NOT per-phrase) -- only gathered/used when the
                                   loss's `use_global_label_for_negative` is enabled.

    Returns:
        all_image_features:            (B_global, D)
        all_text_features:             (B_global, D)
        all_aggregated_image_features: (B_global, D)
        all_key_phrase_features:       (N_phrase_global, D)
        all_label_map:                 (N_phrase_global, C)
        all_counts:                    (B_global,)
        all_volume_label:               (B_global, C) or None if volume_label was None
    """

    assert has_distributed, ("torch.distributed did not import correctly, please use a PyTorch version with support.")

    # flatten label_map, counts
    if isinstance(label_map, (list, tuple)):
        label_map = torch.cat(label_map, dim=0) # (N_phrase_local, C)

    if isinstance(counts, (list, tuple)) and len(counts) > 0 and isinstance(counts[0], (list, tuple)):
        counts = [x for group in counts for x in group] # (B_local,)

    device = image_features.device
    label_map = label_map.to(device=device, dtype=torch.float32)
    counts = torch.as_tensor(counts, device=device, dtype=torch.long)

    if isinstance(volume_label, (list, tuple)):
        # Each element is already a (B_i, C) batch (one accum-window micro-batch), matching
        # how label_map is accumulated above -- concatenate, don't stack.
        volume_label = torch.cat(volume_label, dim=0) if len(volume_label) > 0 else None
    if volume_label is not None:
        volume_label = volume_label.to(device=device, dtype=torch.float32)

    B_local = image_features.size(0)
    N_phrase_local = key_phrase_features.size(0)

    if world_size > 1:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("world_size > 1, but torch.distributed is not initialized.")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    image_sizes  = _all_gather_sizes(B_local, device, world_size)
    phrase_sizes = _all_gather_sizes(N_phrase_local, device, world_size)

    # ------------------------------------------------------------
    # 실제 gather
    # ------------------------------------------------------------
    all_image_features = _gather_variable_2d(image_features, image_sizes, preserve_grad=True, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)
    all_text_features = _gather_variable_2d(text_features, image_sizes, preserve_grad=True, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)
    all_key_phrase_features = _gather_variable_2d(key_phrase_features, phrase_sizes, preserve_grad=True, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)
    all_aggregated_image_features = _gather_variable_2d(aggregated_image_features, image_sizes, preserve_grad=True, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)
    all_label_map = _gather_variable_2d(label_map, phrase_sizes, preserve_grad=False, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)
    all_counts = _gather_variable_1d(counts, image_sizes, pad_value=0, world_size=world_size)
    all_volume_label = None
    if volume_label is not None:
        all_volume_label = _gather_variable_2d(volume_label, image_sizes, preserve_grad=False, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)

    return (all_image_features, all_text_features, all_aggregated_image_features, all_key_phrase_features, all_label_map, all_counts, all_volume_label)


def gather_features_vlcabs(
        local_image_features,
        key_phrase_features,
        label_map,
        counts,
        global_image_features=None,
        text_features=None,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        volume_label=None,
        phrase_ids=None,
):
    """Gather for the loss-side VL-CABS variants (PhraseAlignLoss, PhraseAlignSplitLoss).

    Unlike `gather_features` above -- which gathers an ALREADY-aggregated (one row per image)
    feature computed inside model.forward() -- this gathers the raw local token map so the
    phrase<->patch attention can be computed against the FULL cross-rank pool of phrases.

    global_image_features/text_features/volume_label are all optional and exist only for the
    image<->report clip_loss branch: PhraseAlignLoss (local-only) passes none of them and
    gets None back, which also skips their all_gather entirely. Every rank runs the same loss,
    so the collectives stay in lockstep either way.

    Args:
        local_image_features:      (B_local, Seq_img(=32*32*32), D)
        key_phrase_features:       (N_phrase_local, D)  # variable
        label_map:                 list of phrase-level tensors
                                   after cat: (N_phrase_local, C)
        counts:                    nested list
                                   after flatten: (B_local,)
        global_image_features:     optional (B_local, D)
        text_features:             optional (B_local, D)
        volume_label:              optional (B_local, C) whole-scan label, one row per
                                   sample (NOT per-phrase) -- only gathered/used when the
                                   loss's `use_global_label_for_negative` is enabled.

    Returns:
        all_global_image_features:     (B_global, D) or None
        all_local_image_features:      (B_global, Seq_img, D)
        all_text_features:             (B_global, D) or None
        all_key_phrase_features:       (N_phrase_global, D)
        all_label_map:                 (N_phrase_global, C)
        all_counts:                    (B_global,)
        all_volume_label:              (B_global, C) or None if volume_label was None
    """

    assert has_distributed, ("torch.distributed did not import correctly, please use a PyTorch version with support.")

    # flatten label_map, counts
    if isinstance(label_map, (list, tuple)):
        label_map = torch.cat(label_map, dim=0)

    if isinstance(counts, (list, tuple)) and len(counts) > 0 and isinstance(counts[0], (list, tuple)):
        counts = [x for group in counts for x in group]

    device = local_image_features.device
    label_map = label_map.to(device=device, dtype=torch.float32)
    counts = torch.as_tensor(counts, device=device, dtype=torch.long)

    if isinstance(volume_label, (list, tuple)):
        # Each element is already a (B_i, C) batch (one accum-window micro-batch), matching
        # how label_map is accumulated above -- concatenate, don't stack.
        volume_label = torch.cat(volume_label, dim=0) if len(volume_label) > 0 else None
    if volume_label is not None:
        volume_label = volume_label.to(device=device, dtype=torch.float32)

    B_local = local_image_features.size(0)
    N_phrase_local = key_phrase_features.size(0)

    if world_size > 1:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("world_size > 1, but torch.distributed is not initialized.")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    image_sizes  = _all_gather_sizes(B_local, device, world_size)
    phrase_sizes = _all_gather_sizes(N_phrase_local, device, world_size)

    # ------------------------------------------------------------
    # 실제 gather
    # ------------------------------------------------------------
    all_global_image_features = None
    if global_image_features is not None:
        all_global_image_features = _gather_variable_2d(global_image_features, image_sizes, preserve_grad=True, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)
    all_local_image_features = _gather_variable_2d(local_image_features, image_sizes, preserve_grad=True, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)
    all_text_features = None
    if text_features is not None:
        all_text_features = _gather_variable_2d(text_features, image_sizes, preserve_grad=True, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)
    all_key_phrase_features = _gather_variable_2d(key_phrase_features, phrase_sizes, preserve_grad=True, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)
    all_label_map = _gather_variable_2d(label_map, phrase_sizes, preserve_grad=False, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)
    all_counts = _gather_variable_1d(counts, image_sizes, pad_value=0, world_size=world_size)
    all_volume_label = None
    if volume_label is not None:
        all_volume_label = _gather_variable_2d(volume_label, image_sizes, preserve_grad=False, world_size=world_size, gather_with_grad=gather_with_grad, rank=rank)

    # Per-phrase text fingerprint (one int64 per phrase, same layout as key_phrase_features).
    # Rides on phrase_sizes, not image_sizes -- it is per-phrase, not per-image.
    all_phrase_ids = None
    if phrase_ids is not None:
        if isinstance(phrase_ids, (list, tuple)):
            phrase_ids = torch.cat([p.reshape(-1) for p in phrase_ids], dim=0) if len(phrase_ids) > 0 else None
    if phrase_ids is not None:
        phrase_ids = phrase_ids.to(device=device, dtype=torch.long).reshape(-1)
        all_phrase_ids = _gather_variable_1d(phrase_ids, phrase_sizes, pad_value=0, world_size=world_size)

    return (all_global_image_features, all_local_image_features, all_text_features, all_key_phrase_features, all_label_map, all_counts, all_volume_label, all_phrase_ids)



class _PhraseAlignLossBase(nn.Module):
    """
    Shared machinery for the PhraseAlign-family losses (PhraseAlignJepaLoss, PhraseAlignLoss,
    PhraseAlignSplitLoss): soft-label preparation, phrase<->image and phrase<->phrase
    pos/neg/ignore mask construction, label-imbalance reweighting, and the masked NCE loss
    terms built on top of those masks. Subclasses differ only in how they gather distributed
    features and how the image/phrase logits are actually produced (see `gather_features`,
    `get_logits`, `forward` in each subclass).

    Two families of subclass live here:
      * PhraseAlignJepaLoss -- consumes an already-aggregated (one row per image) image feature that
        model.forward() produced; used by the JEPA variant, whose predictor has to run inside
        the model.
      * PhraseAlignLoss / PhraseAlignSplitLoss -- consume the raw local token map and run
        VL-CABS themselves over the full gathered pool (see `_vlcabs_local_terms`), which is
        the only way every (image, phrase) pair gets its own attention-pooled image embedding.
        PhraseAlignSplitLoss adds the image<->report clip_loss on top (`_vlcabs_forward`);
        PhraseAlignLoss is the local-only ablation and has no global branch at all.
    """
    def __init__(
            self,
            gather_with_grad: bool = False,
            cache_labels: bool = False,  # old config compatibility only
            rank: int = 0,
            world_size: int = 1,

            # add
            label_sim_weight: float = 0.5,
            phrase_pair_weight: float = 0.5,
            label_pos_threshold: float = 0.8,
            label_neg_threshold: float = 0.4,
            label_sim_gamma: float = 2.0,
            except_label_positive_in_source_neg: bool = True,
            label_balance_mode: Optional[str] = "sqrt_inverse",
            label_balance_beta: float = 0.999,
            label_prob_threshold: float = 0.4,     # absolute threshold
            label_prob_relative_threshold: float = 0.3, # relative to row max
            label_prob_topk: Optional[int] = 5,

            # Optional: apply the same "don't push away genuinely-similar cases" protection
            # to the main clip_loss term (image<->text), using a whole-scan/image-level label
            # (e.g. CT-RATE's per-volume multi-abnormality CSV) instead of the per-phrase
            # sentence_probs used everywhere else in this class. Off by default -- when off,
            # clip_loss is computed exactly as before (plain N-way cross-entropy, no masking).
            #
            # Thresholds default to values calibrated separately from label_pos/neg_threshold:
            # empirically, whole-scan label cosine similarity is much more continuous than the
            # near-binary phrase-level similarity (most scans carry ~3-4 co-occurring labels vs
            # ~1 dominant category per phrase), so reusing 0.8/0.4 here would make "positive"
            # fire on <2% of pairs -- barely different from doing nothing. 0.5/0.15 instead
            # captures a meaningful ~17% as positive (protected) and ~40% as confidently
            # negative, leaving the ambiguous middle (~42%) untouched either way.
            use_global_label_for_negative: bool = False,
            global_label_pos_threshold: float = 0.5,
            global_label_neg_threshold: float = 0.15,

            # VL-CABS chunking (loss-side VL-CABS variants only -- see _vlcabs_aggregate)
            local_attn_chunk_size: int = 8,

            # Verbatim-duplicate phrase handling (needs `phrase_ids`; no-op without it).
            # CT-RATE reports are templated, so the same sentence recurs across volumes:
            # measured on merged_train_filtered_prob_half.json, 364,852 phrase
            # instances collapse to 78,244 unique strings, and in a 32-volume window
            # (2 GPU x bs 2 x accum 8, ~491 phrases) 48.5% of phrases share their exact text
            # with a DIFFERENT volume in the same window. Of those, 95% are label-unknown,
            # which means `label_neg_mask` -- whose only protection is soft-label similarity
            # -- files them as hard negatives.
            #
            # That is not merely a weak signal, it is self-contradictory: identical strings
            # produce bit-identical text features, so all_logits_per_phrase[k, j] and
            # [k', j] are the SAME scalar, and source_loss pushes it down as k's negative
            # while pushing it up as k''s positive. Per phrase, 9.2% of the 31 negative
            # images carry such a copy.
            #
            # `promote_duplicate_phrases` decides what happens to those pairs once they
            # leave the negative set. The default "none" leaves them in the ignore zone,
            # which is what the rest of this loss does with every pair it has no label
            # evidence for: `label_pos_mask` requires `known_pair_mask`, and its strength
            # is the measured similarity `sim ** label_sim_gamma`, never a flat constant.
            # Promoting a duplicate asserts a maximum-strength (1.0) positive on a pair
            # that is label-unknown 95% of the time, which is the opposite of that rule.
            #
            # Measured (756 phrases / 64 images, real CT-RATE labels), promotion is also
            # empirically bad on the phrase<->phrase side: identical strings have cosine
            # exactly 1.0, i.e. already the row's maximum logit, so the pair is
            # unlearnable, and adding it turned previously inactive rows into valid ones
            # with nothing to learn -- phrase_pair_loss collapsed 13.2301 -> 3.7621.
            #   "image" promotes only phrase<->image, "both" adds phrase<->phrase.
            # Both are experiment-only; leave at "none" unless ablating.
            mask_duplicate_phrases: bool = False,
            promote_duplicate_phrases: str = "none",

            eps: float = 1e-8,
            **kwargs,
    ):
        super().__init__()

        self.gather_with_grad = bool(gather_with_grad)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.cache_labels = cache_labels

        self.label_sim_weight = float(label_sim_weight)
        self.phrase_pair_weight = float(phrase_pair_weight)

        self.label_pos_threshold = float(label_pos_threshold)
        self.label_neg_threshold = float(label_neg_threshold)
        self.label_sim_gamma = float(label_sim_gamma)

        self.except_label_positive_in_source_neg = bool(except_label_positive_in_source_neg)

        self.label_balance_mode = label_balance_mode
        self.label_balance_beta = float(label_balance_beta)

        self.label_prob_threshold = float(label_prob_threshold)
        self.label_prob_relative_threshold = float(label_prob_relative_threshold)
        self.label_prob_topk = label_prob_topk if label_prob_topk is None else int(label_prob_topk)

        self.use_global_label_for_negative = bool(use_global_label_for_negative)
        self.global_label_pos_threshold = float(global_label_pos_threshold)
        self.global_label_neg_threshold = float(global_label_neg_threshold)

        self.local_attn_chunk_size = int(local_attn_chunk_size)

        self.mask_duplicate_phrases = bool(mask_duplicate_phrases)

        # Back-compat: the first version of this option was a bool. True meant "promote on
        # both sides", which is the configuration the measurement above rejected.
        if isinstance(promote_duplicate_phrases, bool):
            promote_duplicate_phrases = "both" if promote_duplicate_phrases else "none"
        if promote_duplicate_phrases not in ("none", "image", "both"):
            raise ValueError(
                "promote_duplicate_phrases must be one of 'none' | 'image' | 'both', "
                f"got {promote_duplicate_phrases!r}."
            )
        self.promote_duplicate_phrases = promote_duplicate_phrases

        if self.global_label_pos_threshold < self.global_label_neg_threshold:
            raise ValueError(
                "global_label_pos_threshold must be >= global_label_neg_threshold."
            )

        self.eps = float(eps)

        if self.label_pos_threshold < self.label_neg_threshold:
            raise ValueError(
                "label_pos_threshold must be >= label_neg_threshold. "
                "For hard thresholding, use the same value for both. "
                "Example: pos=neg=0.5 means sim >= 0.5 is positive and sim < 0.5 is negative."
            )


        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def _build_group_map_from_counts(self, counts, all_num_images: int, all_num_phrases: int) -> torch.Tensor:
        counts = counts.to(dtype=torch.long)

        if len(counts) != all_num_images:
            raise ValueError(f"counts length must match number of all images. ")

        if sum(counts) != all_num_phrases:
            raise ValueError(f"counts length must match number of all phrases. ")

        if all_num_phrases == 0:
            return torch.empty(0, device=counts.device, dtype=torch.long)

        return torch.repeat_interleave(torch.arange(all_num_images, device=counts.device, dtype=torch.long), counts)

    # ------------------------------------------------------------------
    # Label similarity construction
    # ------------------------------------------------------------------
    def _prepare_soft_label_map(self, labels: torch.Tensor):
        if labels.dim() != 2:
            raise ValueError(f"label_map must be 2D, got {labels.shape}")

        labels = labels.to(dtype=torch.float32)
        labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
        labels = labels.clamp(min=0.0, max=1.0)

        # 1) absolute threshold
        if self.label_prob_threshold > 0:
            labels = labels * (labels >= self.label_prob_threshold).to(labels.dtype)

        # 2) relative threshold per phrase
        # keep labels whose prob >= row_max * label_prob_relative_threshold
        if self.label_prob_relative_threshold > 0:
            row_max = labels.max(dim=-1, keepdim=True).values
            rel_thr = row_max * self.label_prob_relative_threshold
            labels = labels * (labels >= rel_thr).to(labels.dtype)

        # 3) optional top-k
        if self.label_prob_topk is not None:
            k = min(int(self.label_prob_topk), labels.size(-1))
            if k > 0 and k < labels.size(-1):
                vals, idx = labels.topk(k, dim=-1)
                keep = torch.zeros_like(labels, dtype=torch.bool)
                keep.scatter_(dim=1, index=idx, src=(vals > 0))
                labels = labels * keep.to(labels.dtype)

        mass = labels.sum(dim=-1)
        known = mass > self.eps

        return labels, known

    def _safe_label_normalize(self, labels: torch.Tensor):
        labels, known = self._prepare_soft_label_map(labels)

        norm = labels.norm(p=2, dim=-1, keepdim=True)
        label_vec = labels / norm.clamp_min(self.eps)

        label_vec = label_vec * known.float().unsqueeze(-1)

        return label_vec, known

    def _build_source_and_label_similarity(self, group_map: torch.Tensor, label_map: torch.Tensor, num_images: int, phrase_sim: torch.Tensor, label_known: torch.Tensor, pos_threshold: Optional[float] = None, neg_threshold: Optional[float] = None):
        pos_threshold = self.label_pos_threshold if pos_threshold is None else pos_threshold
        neg_threshold = self.label_neg_threshold if neg_threshold is None else neg_threshold

        device = group_map.device
        N = group_map.numel()
        B = int(num_images)

        source_pos_mask = torch.zeros(N, B, dtype=torch.bool, device=device)
        source_pos_mask[torch.arange(N, device=device), group_map] = True

        # label_vec, label_known = self._safe_label_normalize(label_map)

        valid_phrase = label_known & (group_map >= 0) & (group_map < B)

        label_sim_to_image = torch.zeros(N, B, device=device, dtype=torch.float32)
        image_known = torch.zeros(B, dtype=torch.bool, device=device)

        if valid_phrase.any():
            # phrase_sim = label_vec @ label_vec.T
            # phrase_sim = phrase_sim.clamp_min(0.0).clamp_max(1.0)

            # unknown phrase와의 similarity는 0으로 제거
            phrase_sim = (phrase_sim * label_known.float().view(-1, 1) * label_known.float().view(1, -1))

            # image j 안에 있는 phrase label들과의 max similarity 사용
            for image_idx in range(B):
                phrase_mask = valid_phrase & (group_map == image_idx)

                if phrase_mask.any():
                    image_known[image_idx] = True
                    label_sim_to_image[:, image_idx] = phrase_sim[:, phrase_mask].max(dim=1).values

            # # 위의 코드 대체.
            # valid_sim = phrase_sim.masked_fill(~valid_phrase.view(1, -1), 0.0)
            # expand_group = group_map.view(1, -1).expand(N, -1)
            # label_sim_to_image.scatter_reduce_(dim=1, index=expand_group, src=valid_sim, reduce="amax", include_self=True)
            # image_known = torch.zeros(B, dtype=torch.bool, device=device)
            # image_known[group_map[valid_phrase]] = True

            # unknown row는 sim 0 유지
            label_sim_to_image = label_sim_to_image * label_known.float().unsqueeze(1)

        non_source_mask = ~source_pos_mask

        known_pair_mask = (label_known.view(-1, 1) & image_known.view(1, -1))

        # similar label image: auxiliary positive
        label_pos_mask = (non_source_mask & known_pair_mask & (label_sim_to_image >= pos_threshold)) # [N_phrase, B_image]

        # dissimilar or unknown label image: main/source negative
        label_neg_mask = (non_source_mask & (~label_pos_mask) & ((~known_pair_mask) | (label_sim_to_image <= neg_threshold)))

        # between neg_threshold and pos_threshold
        label_ignore_mask = (non_source_mask & (~label_pos_mask) & (~label_neg_mask))

        label_sim_weight = torch.zeros_like(label_sim_to_image)

        if label_pos_mask.any():
            label_sim_weight[label_pos_mask] = (label_sim_to_image[label_pos_mask].clamp_min(0.0).clamp_max(1.0).pow(self.label_sim_gamma)) # similarity가 클수록 auxiliary positive strength 증가

        return {
            "source_pos_mask": source_pos_mask,
            "label_sim_to_image": label_sim_to_image,
            "label_pos_mask": label_pos_mask,
            "label_neg_mask": label_neg_mask,
            "label_ignore_mask": label_ignore_mask,
            "label_sim_weight": label_sim_weight,
            "label_known": label_known,
            "image_known": image_known,
        }

    # ------------------------------------------------------------------
    # Verbatim-duplicate phrase masks (see `mask_duplicate_phrases` in __init__)
    # ------------------------------------------------------------------
    def _build_duplicate_masks(self, phrase_ids: torch.Tensor, group_map: torch.Tensor, num_images: int):
        """Returns (dup_to_image (N, B) bool, dup_to_phrase (N, N) bool).

        `phrase_ids` is any integer fingerprint of the phrase text -- equal ids must mean
        equal strings. Built in the dataset from the raw sentence, so it is exact and does
        not depend on any floating-point comparison of the cached embeddings.

        dup_to_image[k, j]  : volume j contains a phrase whose text equals phrase k's,
                              excluding k's own volume (so the true positive is never touched).
        dup_to_phrase[k, l] : phrase l has the same text as phrase k, l != k (same volume
                              included -- a report that repeats itself should not have those
                              two rows pushed apart either).
        """
        device = phrase_ids.device
        N = phrase_ids.numel()

        same = phrase_ids.view(-1, 1) == phrase_ids.view(1, -1)                  # (N, N)
        eye = torch.eye(N, dtype=torch.bool, device=device)
        dup_to_phrase = same & ~eye

        # float32, not int: CUDA has no integer addmm. Counts are bounded by N (a few
        # hundred), well inside float32's exact-integer range.
        vol_onehot = F.one_hot(group_map.long(), int(num_images)).to(torch.float32)  # (N, B)
        dup_to_image = (same.to(torch.float32) @ vol_onehot) > 0                     # (N, B)
        dup_to_image[torch.arange(N, device=device), group_map] = False

        return dup_to_image, dup_to_phrase

    def _apply_duplicate_masks(self, masks: dict, phrase_pair_masks: dict, dup_to_image: torch.Tensor, dup_to_phrase: torch.Tensor):
        """Pull verbatim duplicates out of both negative sets. With the default
        `promote_duplicate_phrases="none"` they land in the ignore zone -- the same place
        this loss puts every other pair it has no label evidence about. Mutates and
        returns the two mask dicts."""
        promote_image = self.promote_duplicate_phrases in ("image", "both")
        promote_phrase = self.promote_duplicate_phrases == "both"

        # ---- phrase <-> image (source_loss / simlabel_pair_loss) ----
        masks["label_neg_mask"] = masks["label_neg_mask"] & ~dup_to_image
        if promote_image:
            new_pos = dup_to_image & (~masks["source_pos_mask"]) & (~masks["label_pos_mask"])
            masks["label_pos_mask"] = masks["label_pos_mask"] | new_pos
            masks["label_sim_weight"] = masks["label_sim_weight"].masked_fill(new_pos, 1.0)
        masks["label_ignore_mask"] = (
            (~masks["source_pos_mask"]) & (~masks["label_pos_mask"]) & (~masks["label_neg_mask"])
        )

        # ---- phrase <-> phrase (phrase_pair_loss) ----
        phrase_pair_masks["phrase_neg_mask"] = phrase_pair_masks["phrase_neg_mask"] & ~dup_to_phrase
        if promote_phrase:
            new_pp = dup_to_phrase & (~phrase_pair_masks["phrase_pos_mask"])
            phrase_pair_masks["phrase_pos_mask"] = phrase_pair_masks["phrase_pos_mask"] | new_pp
            phrase_pair_masks["phrase_sim_weight"] = phrase_pair_masks["phrase_sim_weight"].masked_fill(new_pp, 1.0)
        not_self = ~torch.eye(dup_to_phrase.size(0), dtype=torch.bool, device=dup_to_phrase.device)
        phrase_pair_masks["phrase_ignore_mask"] = (
            not_self & (~phrase_pair_masks["phrase_pos_mask"]) & (~phrase_pair_masks["phrase_neg_mask"])
        )

        return masks, phrase_pair_masks

    def _masked_source_pair_nce_loss(self, logits: torch.Tensor, group_map: torch.Tensor, source_pos_mask: torch.Tensor, label_neg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got {logits.shape}")

        device = logits.device
        N, B = logits.shape

        if group_map.numel() != N:
            raise ValueError(f"group_map length mismatch: group_map={group_map.numel()}, logits rows={N}")

        if group_map.min().item() < 0 or group_map.max().item() >= B:
            raise ValueError(
                f"group_map index out of range: "
                f"min={group_map.min().item()}, "
                f"max={group_map.max().item()}, "
                f"B={B}"
            )

        z = logits
        neg_inf = torch.finfo(z.dtype).min

        pos_z = z[torch.arange(N, device=device), group_map]

        if self.except_label_positive_in_source_neg and label_neg_mask is not None:
            neg_mask = label_neg_mask # Ours
        else:
            neg_mask = ~source_pos_mask # PhraseAlign

        # Row direction: phrase -> image
        row_neg_z = z.masked_fill(~neg_mask, neg_inf)
        row_log_neg_sum = torch.logsumexp(row_neg_z, dim=1)
        row_loss = F.softplus(row_log_neg_sum - pos_z) # F.softplus(x) = log(1 + exp(x))

        # Column direction: image -> phrase
        col_neg_z = z.masked_fill(~neg_mask, neg_inf)
        col_log_neg_sum = torch.logsumexp(col_neg_z, dim=0)
        col_loss = F.softplus(col_log_neg_sum[group_map] - pos_z)

        return 0.5 * (row_loss.mean() + col_loss.mean())

    def _make_label_balance_weights(self, label_map: torch.Tensor) -> torch.Tensor:
        device = label_map.device
        N = label_map.size(0)
        mode = self.label_balance_mode

        weights = torch.ones(N, device=device, dtype=torch.float32)

        if mode is None or mode == "none":
            return weights

        soft_labels, known = self._prepare_soft_label_map(label_map)
        soft_labels = soft_labels.to(device=device, dtype=torch.float32)

        if not known.any():
            return weights

        raw_counts = soft_labels[known].sum(dim=0)

        active_label = raw_counts > self.eps
        counts = raw_counts.clamp_min(self.eps)

        if mode == "inverse":
            label_weights = 1.0 / counts

        elif mode == "sqrt_inverse":
            label_weights = 1.0 / torch.sqrt(counts)

        elif mode == "effective_num":
            beta = torch.tensor(self.label_balance_beta, device=device, dtype=torch.float32)
            label_weights = ((1.0 - beta) / (1.0 - torch.pow(beta, counts) + self.eps))

        else:
            raise ValueError(f"Unknown label_balance_mode: {mode}")

        if active_label.any():
            label_weights = (label_weights / label_weights[active_label].mean().clamp_min(self.eps))

        phrase_mass = soft_labels.sum(dim=1)
        known_phrase = phrase_mass > self.eps

        if known_phrase.any():
            phrase_weights = (soft_labels * label_weights.view(1, -1)).sum(dim=1) / phrase_mass.clamp_min(self.eps)

            weights[known_phrase] = phrase_weights[known_phrase]

        weights = weights / weights.mean().clamp_min(self.eps)

        return weights

    def _weighted_simlabel_pair_nce_loss(self, logits: torch.Tensor, label_sim_weight: torch.Tensor, label_pos_mask: torch.Tensor, label_neg_mask: torch.Tensor, row_balance_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got {logits.shape}")

        if not label_pos_mask.any():
            return logits.new_tensor(0.0)

        z = logits
        N, B = z.shape
        neg_inf = torch.finfo(z.dtype).min

        if row_balance_weights is None:
            row_balance_weights = torch.ones(N, device=z.device, dtype=torch.float32,)
        else:
            row_balance_weights = row_balance_weights.to(device=z.device, dtype=torch.float32,)

        sim_weight = label_sim_weight.to(dtype=z.dtype)
        pos_mask_f = label_pos_mask.float()



        # Row auxiliary: phrase -> similar-label images
        row_neg_z = z.masked_fill(~label_neg_mask, neg_inf)
        row_log_neg_sum = torch.logsumexp(row_neg_z, dim=1, keepdim=True)
        row_pair_loss = F.softplus(row_log_neg_sum - z)

        row_pos_count = pos_mask_f.sum(dim=1)
        row_valid = row_pos_count > 0

        # denominator는 pos 개수 기준.
        # sim_weight가 작으면 해당 positive의 gradient가 약해짐.
        row_loss_per_phrase = (row_pair_loss * sim_weight).sum(dim=1) / row_pos_count.clamp_min(1.0)

        if row_valid.any():
            row_weights = row_balance_weights[row_valid]
            row_aux_loss = (row_loss_per_phrase[row_valid] * row_weights).sum() / row_weights.sum().clamp_min(self.eps)
        else:
            row_aux_loss = logits.new_tensor(0.0)



        # Column auxiliary: image -> similar-label phrases
        col_neg_z = z.masked_fill(~label_neg_mask, neg_inf)
        col_log_neg_sum = torch.logsumexp(col_neg_z, dim=0, keepdim=True)
        col_pair_loss = F.softplus(col_log_neg_sum - z)

        pair_balance = row_balance_weights.view(-1, 1).to(dtype=z.dtype)
        col_den = (pos_mask_f * pair_balance).sum(dim=0)
        col_valid = col_den > 0

        col_loss_per_image = (col_pair_loss * sim_weight * pair_balance).sum(dim=0) / col_den.clamp_min(self.eps)

        if col_valid.any():
            col_aux_loss = col_loss_per_image[col_valid].mean()
        else:
            col_aux_loss = logits.new_tensor(0.0)

        return 0.5 * (row_aux_loss + col_aux_loss)

    def _build_phrase_pair_masks(self, group_map: torch.Tensor, label_map: torch.Tensor, label_sim: torch.Tensor, label_known: torch.Tensor) -> dict:
        """
        phrase-phrase 쌍에 대한 pos/neg/ignore 마스크 및 sim_weight 계산.

        의도:
          Positive:
            - self 제외
            - 양쪽 phrase label이 known
            - label similarity >= self.label_pos_threshold
            - source/image 같고 다름은 상관 없음

          Negative:
            - self 제외
            - 다른 image/source phrase
            - unknown pair 포함 OR label similarity <= self.label_neg_threshold

          Ignore:
            - self 제외
            - positive도 negative도 아닌 pair
            - 예: 같은 image인데 label sim이 낮은 경우
            - 예: 다른 image인데 0.4 < label sim < 0.8인 애매한 경우

        Args:
            group_map: (N,) 각 phrase가 속한 image index
            label_map: (N, C) phrase별 label 확률 벡터

        Returns:
            dict with keys:
                phrase_pos_mask    : (N, N) bool
                phrase_neg_mask    : (N, N) bool
                phrase_ignore_mask : (N, N) bool
                phrase_sim_weight  : (N, N) float
                phrase_known       : (N,)   bool
                phrase_label_sim   : (N, N) float
        """
        if group_map.dim() != 1:
            raise ValueError(f"group_map must be 1D, got {group_map.shape}")

        if label_map.dim() != 2:
            raise ValueError(f"label_map must be 2D, got {label_map.shape}")

        N = group_map.numel()
        device = group_map.device

        if label_map.size(0) != N:
            raise ValueError(
                f"label_map rows must match group_map length: "
                f"{label_map.size(0)} vs {N}"
            )

        label_map = label_map.to(device=device, dtype=torch.float32)

        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        not_self_mask = ~self_mask

        same_source = group_map.view(-1, 1) == group_map.view(1, -1)
        diff_source = (~same_source) & not_self_mask

        # label 유사도 계산
        # label_vec, label_known = self._safe_label_normalize(label_map)
        known_pair = label_known.view(-1, 1) & label_known.view(1, -1)

        # label_sim = label_vec @ label_vec.T
        # label_sim = label_sim.clamp_min(0.0).clamp_max(1.0)

        # unknown이 섞인 pair의 sim은 0으로 제거
        label_sim = label_sim * known_pair.to(dtype=label_sim.dtype)

        # ------------------------------------------------------------
        # Positive:
        # source 상관없이 label similarity만 본다.
        # ------------------------------------------------------------
        phrase_pos_mask = (not_self_mask & known_pair & (label_sim >= self.label_pos_threshold))

        # ------------------------------------------------------------
        # Negative:
        # 다른 image/source phrase 중에서
        # unknown pair 또는 label similarity <= 0.4
        # ------------------------------------------------------------
        phrase_neg_mask = (diff_source & ( (~known_pair) | (label_sim <= self.label_neg_threshold) ))

        # ------------------------------------------------------------
        # Ignore:
        # positive도 negative도 아닌 pair
        # ------------------------------------------------------------
        phrase_ignore_mask = (not_self_mask & (~phrase_pos_mask) & (~phrase_neg_mask))

        # ------------------------------------------------------------
        # Positive strength
        # ------------------------------------------------------------
        phrase_sim_weight = torch.zeros(N, N, device=device, dtype=torch.float32)

        if phrase_pos_mask.any():
            phrase_sim_weight[phrase_pos_mask] = (label_sim[phrase_pos_mask].clamp_min(0.0).clamp_max(1.0).pow(self.label_sim_gamma))

        return {
            "phrase_pos_mask": phrase_pos_mask,
            "phrase_neg_mask": phrase_neg_mask,
            "phrase_ignore_mask": phrase_ignore_mask,
            "phrase_sim_weight": phrase_sim_weight,
            "phrase_known": label_known,
            "phrase_label_sim": label_sim,
        }

    def _phrase_pair_nce_loss(self, logits: torch.Tensor, phrase_pos_mask: torch.Tensor, phrase_neg_mask: torch.Tensor, phrase_sim_weight: torch.Tensor, row_balance_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Phrase-phrase supervised pair-NCE loss.

        의도:
            - anchor phrase i 기준
            - positive j:
                label similarity(i, j) >= label_pos_threshold
            - negative k:
                다른 image/source phrase 중
                unknown pair 또는 label similarity(i, k) <= label_neg_threshold

        logits:
            (N, N)
            logit_scale * normalize(phrase_features) @ normalize(phrase_features).T

        phrase_pos_mask:
            (N, N) bool
            positive phrase pair mask

        phrase_neg_mask:
            (N, N) bool
            negative phrase pair mask

        phrase_sim_weight:
            (N, N) float
            positive strength.
            보통 positive 위치에서 label_sim ** label_sim_gamma,
            나머지 위치는 0.

        row_balance_weights:
            (N,) optional
            phrase별 label balancing weight.

        Loss:
            loss_i,j = log(1 + sum_k exp(s_i,k) / exp(s_i,j))
                     = softplus(logsumexp_k(s_i,k) - s_i,j)

            최종적으로 모든 valid anchor i에 대해 평균.
            여기서 valid anchor는 positive와 negative가 모두 있는 row.
        """
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got {logits.shape}")

        if logits.size(0) != logits.size(1):
            raise ValueError(f"logits must be square, got {logits.shape}")

        N = logits.size(0)
        device = logits.device

        if phrase_pos_mask.shape != (N, N):
            raise ValueError(
                f"phrase_pos_mask must have shape ({N}, {N}), "
                f"got {phrase_pos_mask.shape}"
            )

        if phrase_neg_mask.shape != (N, N):
            raise ValueError(
                f"phrase_neg_mask must have shape ({N}, {N}), "
                f"got {phrase_neg_mask.shape}"
            )

        if phrase_sim_weight.shape != (N, N):
            raise ValueError(
                f"phrase_sim_weight must have shape ({N}, {N}), "
                f"got {phrase_sim_weight.shape}"
            )

        if N <= 1:
            return logits.sum() * 0.0

        not_self_mask = ~torch.eye(N, dtype=torch.bool, device=device)

        pos_mask = phrase_pos_mask.to(device=device, dtype=torch.bool) & not_self_mask
        neg_mask = phrase_neg_mask.to(device=device, dtype=torch.bool) & not_self_mask

        if not pos_mask.any():
            return logits.sum() * 0.0

        sim_weight = phrase_sim_weight.to(device=device, dtype=logits.dtype)

        # positive 위치만 사용하도록 명시적으로 masking
        pos_weight = pos_mask.to(dtype=logits.dtype) * sim_weight

        pos_count = pos_mask.sum(dim=1)          # (N,)
        neg_count = neg_mask.sum(dim=1)          # (N,)
        pos_weight_sum = pos_weight.sum(dim=1)   # (N,)

        # positive와 negative가 둘 다 있고,
        # positive weight도 실제로 존재하는 anchor만 사용
        row_valid = ((pos_count > 0) & (neg_count > 0) & (pos_weight_sum > self.eps))

        if not row_valid.any():
            return logits.sum() * 0.0

        if row_balance_weights is None:
            row_balance_weights = torch.ones(N, device=device, dtype=torch.float32)
        else:
            row_balance_weights = row_balance_weights.to(device=device, dtype=torch.float32)

        neg_inf = torch.finfo(logits.dtype).min

        # ------------------------------------------------------------
        # Row-anchor direction only
        # anchor = phrase i
        # negatives = phrase k where phrase_neg_mask[i, k] == True
        # positives = phrase j where phrase_pos_mask[i, j] == True
        # ------------------------------------------------------------
        neg_logits = logits.masked_fill(~neg_mask, neg_inf)

        # (N, 1)
        log_neg_sum = torch.logsumexp(neg_logits, dim=1, keepdim=True)

        # (N, N)
        # positive j에 대해서:
        # softplus(logsumexp_k negative_logit_i,k - positive_logit_i,j)
        pair_loss = F.softplus(log_neg_sum - logits)

        # positive pair만 loss에 반영.
        # denominator를 pos_count로 두면 sim_weight가 "strength"로 작동함.
        # 즉 label_sim이 높은 positive가 더 강하게 당겨짐.
        loss_per_phrase = (pair_loss * pos_weight).sum(dim=1) / pos_count.clamp_min(1).to(dtype=logits.dtype)

        valid_weights = row_balance_weights[row_valid]

        loss = (loss_per_phrase[row_valid] * valid_weights.to(dtype=logits.dtype)).sum() / valid_weights.sum().clamp_min(self.eps).to(dtype=logits.dtype)

        return loss

    def get_ground_truth_clip(self, logits_per_image, logits_per_text) -> torch.Tensor:
        assert logits_per_image.shape[0] == logits_per_text.shape[0]
        num_logits = logits_per_image.shape[0]
        device = logits_per_image.device

        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def _compute_clip_loss(self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor, volume_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Symmetric image<->text contrastive loss. By default this is plain N-way
        cross-entropy (every non-matching pair in the window is a hard negative) -- identical
        to the original behavior.

        When `use_global_label_for_negative` is on and a whole-scan multi-label vector is
        supplied per sample (e.g. from CT-RATE's per-volume abnormality CSV -- NOT the
        per-phrase `label_map` used elsewhere in this class), pairs whose scan-level labels are
        confidently similar (>= global_label_pos_threshold) are excluded from the negative set,
        mirroring what `_masked_source_pair_nce_loss` already does for phrase<->image source_loss.
        This reuses that same masked-NCE machinery: each image's "positive" is just itself
        (group_map = arange(N)), so a single masked_source_pair_nce_loss call already gives the
        symmetric image->text and text->image directions.
        """
        if not self.use_global_label_for_negative or volume_label is None:
            labels = self.get_ground_truth_clip(logits_per_image, logits_per_text)
            return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        N = logits_per_image.shape[0]
        device = logits_per_image.device

        label_vec, label_known = self._safe_label_normalize(volume_label)
        image_sim = (label_vec @ label_vec.T).clamp(0.0, 1.0)
        group_map = torch.arange(N, device=device, dtype=torch.long)

        masks = self._build_source_and_label_similarity(
            group_map=group_map, label_map=volume_label, num_images=N,
            phrase_sim=image_sim, label_known=label_known,
            pos_threshold=self.global_label_pos_threshold,
            neg_threshold=self.global_label_neg_threshold,
        )

        return self._masked_source_pair_nce_loss(
            logits=logits_per_image, group_map=group_map,
            source_pos_mask=masks["source_pos_mask"], label_neg_mask=masks["label_neg_mask"],
        )

    # ------------------------------------------------------------------
    # VL-CABS (loss-side): phrase <-> patch attention over the FULL gathered pool
    # ------------------------------------------------------------------
    def _global_pair_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        """Plain all-pairs cosine logits: (N_img, D) x (N_txt, D) -> (N_img, N_txt)."""
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features  = F.normalize(text_features, p=2, dim=-1)

        logits_per_image = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits_per_image = logits_per_image + logit_bias

        return logits_per_image, logits_per_image.T

    def _vlcabs_aggregate(self, local_image_features, key_phrase_features, temperature):
        """VL-CABS: one attention-pooled image embedding per (image, phrase) PAIR.

        Every phrase in the pool attends over every image's patch tokens, so the returned
        tensor is (N_images, N_phrases, D) -- the (i, p) row is image i seen through phrase p.
        That cross product is the whole point of VL-CABS: the phrase<->image logit for a
        NON-matching pair must be computed from that pair's own attention map, not from a
        phrase-agnostic (or own-phrases-only) image vector.

        Chunked over the image dim: N_images and N_phrases both scale with the accumulated
        batch (train's accum-freq window, or eval's whole val set), so a single
        [N_images, N_phrases, Seq_local] bmm grows quadratically with accumulation size and
        OOMs once val accumulates dozens of batches (Seq_local is already large on its own,
        e.g. 32768 multiscale local tokens/image). Each image's phrase-attention is independent
        of other images, so chunking here is exact -- it only bounds the bmm's batch dim.
        """
        if temperature is None:
            raise ValueError(
                "temperature is required for loss-side VL-CABS -- pass the model's "
                "`temperature` buffer (e.g. unwrap_model(model).temperature) into loss()."
            )

        num_images = local_image_features.size(0)
        chunk_size = min(num_images, self.local_attn_chunk_size) if self.local_attn_chunk_size > 0 else num_images
        aggregated_chunks = []
        for start in range(0, num_images, chunk_size):
            end = min(start + chunk_size, num_images)
            chunk_local = local_image_features[start:end]
            chunk_phrase = key_phrase_features.unsqueeze(0).repeat(end - start, 1, 1)
            chunk_scores = torch.bmm(chunk_phrase, chunk_local.permute(0, 2, 1)) / temperature.exp() # [Similarity Scores] between query and local tokens
            chunk_weights = F.softmax(chunk_scores, dim=-1) # [Patch-Level Similarity Map]
            aggregated_chunks.append(torch.matmul(chunk_weights, chunk_local)) # aggregated = [Vision Embedding] in paper
        aggregated = torch.cat(aggregated_chunks, dim=0)

        return F.normalize(aggregated, p=2, dim=-1)

    def _vlcabs_pair_logits(self, aggregated_image_features, key_phrase_features, logit_scale, logit_bias=None):
        """Pairwise (NOT all-pairs matmul) logits between the VL-CABS aggregates and phrases.

        aggregated_image_features: (N_img, N_phrase, D)  -- from _vlcabs_aggregate
        key_phrase_features:       (N_phrase, D), repeated to (N_img, N_phrase, D)
        returns (logits_per_image (N_img, N_phrase), logits_per_phrase (N_phrase, N_img))
        """
        aggregated_image_features = F.normalize(aggregated_image_features, p=2, dim=-1)
        key_phrase_features = F.normalize(key_phrase_features, p=2, dim=-1)

        num_images = aggregated_image_features.size(0)
        repeat_key_phrase_features = key_phrase_features.unsqueeze(0).repeat(num_images, 1, 1) # repeat

        # squeeze(-1).squeeze(-1), not a bare squeeze(): the matmul yields
        # (N_img, N_phrase, 1, 1) and a bare squeeze would also drop N_img or N_phrase whenever
        # a window happens to hold exactly one image or one phrase.
        logits_per_image = logit_scale * (aggregated_image_features.unsqueeze(2) @ repeat_key_phrase_features.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        if logit_bias is not None:
            logits_per_image = logits_per_image + logit_bias

        return logits_per_image, logits_per_image.T

    def _vlcabs_local_terms(
            self,
            all_local_image_features,
            all_key_phrase_features,
            all_label_map,
            all_counts,
            logit_scale,
            logit_bias=None,
            temperature=None,
            phrase_ids=None,
    ):
        """The three phrase-level terms, computed on already-gathered features.

        Shared by every loss-side VL-CABS variant, with or without the image<->report
        clip_loss branch on top. Returns (source_loss, simlabel_pair_loss, phrase_pair_loss,
        details) where `details` holds the detached, log-only diagnostics.
        """
        #### Add LOSS
        # phrase_align (VL-CABS)
        all_aggregated_img_feat = self._vlcabs_aggregate(all_local_image_features, all_key_phrase_features, temperature)

        # logits
        all_logits_per_phrase = self._vlcabs_pair_logits(all_aggregated_img_feat, all_key_phrase_features, logit_scale, logit_bias=logit_bias)[1] # [local_num_phrases, local_num_images]
        all_group_map = self._build_group_map_from_counts(counts=all_counts, all_num_images=all_local_image_features.size(0), all_num_phrases=all_key_phrase_features.size(0))

        label_vec, label_known = self._safe_label_normalize(all_label_map)
        phrase_sim = (label_vec @ label_vec.T).clamp(0.0, 1.0)
        masks = self._build_source_and_label_similarity(group_map=all_group_map, label_map=all_label_map, num_images=len(all_counts), phrase_sim=phrase_sim, label_known=label_known)
        phrase_pair_masks = self._build_phrase_pair_masks(group_map=all_group_map, label_map=all_label_map, label_sim=phrase_sim, label_known=label_known)

        # Verbatim-duplicate correction, applied to BOTH negative sets before any loss term
        # reads them (see `mask_duplicate_phrases` in __init__ for the measured motivation).
        dup_rate = None
        if self.mask_duplicate_phrases and phrase_ids is not None and all_key_phrase_features.size(0) > 0:
            dup_to_image, dup_to_phrase = self._build_duplicate_masks(
                phrase_ids=phrase_ids, group_map=all_group_map, num_images=len(all_counts)
            )
            masks, phrase_pair_masks = self._apply_duplicate_masks(
                masks, phrase_pair_masks, dup_to_image, dup_to_phrase
            )
            dup_rate = dup_to_image.any(dim=1).float().mean().detach()

        # 1. Source pair Loss
        source_loss = self._masked_source_pair_nce_loss(logits=all_logits_per_phrase, group_map=all_group_map, source_pos_mask=masks["source_pos_mask"], label_neg_mask=masks["label_neg_mask"])

        # 2. Simlabel pair loss
        phrase_balance_weights = self._make_label_balance_weights(label_map=all_label_map)
        simlabel_pair_loss = self._weighted_simlabel_pair_nce_loss(logits=all_logits_per_phrase, label_sim_weight=masks["label_sim_weight"], label_pos_mask=masks["label_pos_mask"], label_neg_mask=masks["label_neg_mask"], row_balance_weights=phrase_balance_weights)

        # 3. Phrase-Phrase pair loss (similar-label phrase → phrase)
        # logit_bias는 phrase-phrase에 적용하지 않음 (image-text 간 bias이므로)
        logits_per_phrase_phrase = self._global_pair_logits(all_key_phrase_features, all_key_phrase_features, logit_scale, logit_bias=None)[0]  # (N_phrase, N_phrase), 대칭 행렬

        phrase_pair_loss = self._phrase_pair_nce_loss(logits=logits_per_phrase_phrase, phrase_pos_mask=phrase_pair_masks["phrase_pos_mask"], phrase_neg_mask=phrase_pair_masks["phrase_neg_mask"], phrase_sim_weight=phrase_pair_masks["phrase_sim_weight"], row_balance_weights=phrase_balance_weights)

        details = {
            "source_loss": source_loss.detach(),
            "simlabel_pair_loss": simlabel_pair_loss.detach(),
            "phrase_pair_loss": phrase_pair_loss.detach(),
        }
        # add
        details.update({
            "num_label_pos": masks["label_pos_mask"].sum().detach(),
            "num_label_neg": masks["label_neg_mask"].sum().detach(),
            "num_label_ignore": masks["label_ignore_mask"].sum().detach(),
            "known_phrase_ratio": masks["label_known"].float().mean().detach(),
            "known_image_ratio": masks["image_known"].float().mean().detach(),
            "num_phrase_pair_pos": phrase_pair_masks["phrase_pos_mask"].sum().detach(),
            "num_phrase_pair_neg": phrase_pair_masks["phrase_neg_mask"].sum().detach(),
            "num_phrase_pair_ignore": phrase_pair_masks["phrase_ignore_mask"].sum().detach(),
            "known_phrase_pair_ratio": phrase_pair_masks["phrase_known"].float().mean().detach(),
        })
        if dup_rate is not None:
            # fraction of phrases in the window whose exact text also occurs in another volume
            details["duplicate_phrase_rate"] = dup_rate

        return source_loss, simlabel_pair_loss, phrase_pair_loss, details

    def _vlcabs_forward(
            self,
            global_image_features,
            local_image_features,
            text_features,
            key_phrase_features,
            logit_scale=1.0,
            logit_bias=None,
            output_dict: bool = False,
            label_map=None,
            counts=None,
            temperature=None,
            volume_label=None,
    ):
        """Loss-side VL-CABS *with* the image<->report clip_loss branch (PhraseAlignSplitLoss).

        PhraseAlignLoss deliberately does NOT go through here -- it is the local-only
        ablation and has no global/report branch at all (see its docstring).
        """
        all_global_image_features, all_local_image_features, all_text_features, all_key_phrase_features, all_label_map, all_counts, all_volume_label, _ = gather_features_vlcabs(global_image_features=global_image_features,
                                                                                                                    local_image_features=local_image_features,
                                                                                                                    text_features=text_features,
                                                                                                                    key_phrase_features=key_phrase_features,
                                                                                                                    label_map=label_map,
                                                                                                                    counts=counts,
                                                                                                                    gather_with_grad=self.gather_with_grad,
                                                                                                                    rank=self.rank,
                                                                                                                    world_size=self.world_size,
                                                                                                                    volume_label=volume_label)

        # Original CLIP Loss (optionally protected by whole-scan label similarity -- see
        # _compute_clip_loss / use_global_label_for_negative)
        logits_per_image, logits_per_text = self._global_pair_logits(all_global_image_features, all_text_features, logit_scale, logit_bias=logit_bias)
        clip_loss = self._compute_clip_loss(logits_per_image, logits_per_text, volume_label=all_volume_label)

        source_loss, simlabel_pair_loss, phrase_pair_loss, details = self._vlcabs_local_terms(
            all_local_image_features=all_local_image_features,
            all_key_phrase_features=all_key_phrase_features,
            all_label_map=all_label_map,
            all_counts=all_counts,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            temperature=temperature,
        )

        # loss sum
        total_loss = clip_loss + source_loss + self.label_sim_weight*simlabel_pair_loss + self.phrase_pair_weight*phrase_pair_loss

        details = {"clip_loss": clip_loss.detach(), **details}

        if output_dict:
            output = {"total_loss": total_loss}
            output.update(details)
            return output

        return total_loss


class PhraseAlignJepaLoss(_PhraseAlignLossBase):
    """Model-side-aggregation variant, used by the JEPA-PhraseAlign model.

    `aggregated_image_features` arrives with ONE row per image, because
    CustomTextCLIP_JEPA_PhraseAlign runs the phrase attention (and then the JEPA predictor, whose
    final pooling collapses the per-phrase dim) inside model.forward(). That means every
    phrase<->image logit in `logits_per_phrase` below reuses the same image vector regardless
    of which phrase it is scored against, i.e. the cross pairs are not real VL-CABS. Keeping it
    for the JEPA variant only, where the predictor has to live inside the model; the non-JEPA
    phrase_align variant uses PhraseAlignLoss instead.
    """
    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        image_features = F.normalize(image_features,p=2, dim=-1)
        text_features  = F.normalize(text_features, p=2, dim=-1)

        logits_per_image = logit_scale * image_features @ text_features.T

        if logit_bias is not None:
            logits_per_image = logits_per_image + logit_bias

        logits_per_text  = logits_per_image.T
        return logits_per_image, logits_per_text


    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
            self,
            image_features,
            text_features,
            aggregated_image_features,
            key_phrase_features,
            logit_scale=1.0,
            logit_bias=None,
            output_dict: bool = False,
            label_map=None,
            counts=None,
            volume_label=None,
            **kwargs,
    ):

        all_image_features, all_text_features, all_aggregated_image_features, all_key_phrase_features, all_label_map, all_counts, all_volume_label = gather_features(image_features=image_features,
                                                                                                                                                       text_features=text_features,
                                                                                                                                                       aggregated_image_features=aggregated_image_features,
                                                                                                                                                       key_phrase_features=key_phrase_features,
                                                                                                                                                       label_map=label_map,
                                                                                                                                                       counts=counts,
                                                                                                                                                       gather_with_grad=self.gather_with_grad,
                                                                                                                                                       rank=self.rank,
                                                                                                                                                       world_size=self.world_size,
                                                                                                                                                       volume_label=volume_label)

        # Original CLIP Loss (optionally protected by whole-scan label similarity -- see
        # _compute_clip_loss / use_global_label_for_negative)
        logits_per_image, logits_per_text = self.get_logits(all_image_features, all_text_features, logit_scale, logit_bias=logit_bias)
        clip_loss = self._compute_clip_loss(logits_per_image, logits_per_text, volume_label=all_volume_label)

        # Add loss
        logits_per_phrase = self.get_logits(all_aggregated_image_features, all_key_phrase_features, logit_scale, logit_bias=logit_bias)[1] # [local_num_phrases, local_num_images]
        all_group_map = self._build_group_map_from_counts(counts=all_counts, all_num_images=all_image_features.size(0), all_num_phrases=all_key_phrase_features.size(0))

        label_vec, label_known = self._safe_label_normalize(all_label_map)
        phrase_sim = (label_vec @ label_vec.T).clamp(0.0, 1.0)

        masks = self._build_source_and_label_similarity(group_map=all_group_map, label_map=all_label_map, num_images=len(all_counts), phrase_sim=phrase_sim, label_known=label_known)

        # 1. Source pair Loss
        source_loss = self._masked_source_pair_nce_loss(logits=logits_per_phrase, group_map=all_group_map, source_pos_mask=masks["source_pos_mask"], label_neg_mask=masks["label_neg_mask"])

        # 2. Simlabel pair loss
        phrase_balance_weights = self._make_label_balance_weights(label_map=all_label_map)
        simlabel_pair_loss = self._weighted_simlabel_pair_nce_loss(logits=logits_per_phrase, label_sim_weight=masks["label_sim_weight"], label_pos_mask=masks["label_pos_mask"], label_neg_mask=masks["label_neg_mask"], row_balance_weights=phrase_balance_weights)

        # 3. Phrase-Phrase pair loss (similar-label phrase → phrase)
        # logit_bias는 phrase-phrase에 적용하지 않음 (image-text 간 bias이므로)
        logits_per_phrase_phrase = self.get_logits(all_key_phrase_features, all_key_phrase_features, logit_scale, logit_bias=None)[0]  # (N_phrase, N_phrase), 대칭 행렬

        phrase_pair_masks = self._build_phrase_pair_masks(group_map=all_group_map, label_map=all_label_map, label_sim=phrase_sim, label_known=label_known)
        phrase_pair_loss = self._phrase_pair_nce_loss(
            logits=logits_per_phrase_phrase,
            phrase_pos_mask=phrase_pair_masks["phrase_pos_mask"],
            phrase_neg_mask=phrase_pair_masks["phrase_neg_mask"],
            phrase_sim_weight=phrase_pair_masks["phrase_sim_weight"],
            row_balance_weights=phrase_balance_weights,
        )

        # loss sum
        total_loss = clip_loss + source_loss + self.label_sim_weight*simlabel_pair_loss + self.phrase_pair_weight*phrase_pair_loss

        details = {
            "clip_loss": clip_loss.detach(),
            "source_loss": source_loss.detach(),
            "simlabel_pair_loss": simlabel_pair_loss.detach(),
            "phrase_pair_loss": phrase_pair_loss.detach(),
        }
        # add
        details.update({
            "num_label_pos": masks["label_pos_mask"].sum().detach(),
            "num_label_neg": masks["label_neg_mask"].sum().detach(),
            "num_label_ignore": masks["label_ignore_mask"].sum().detach(),
            "known_phrase_ratio": masks["label_known"].float().mean().detach(),
            "known_image_ratio": masks["image_known"].float().mean().detach(),
            "num_phrase_pair_pos": phrase_pair_masks["phrase_pos_mask"].sum().detach(),
            "num_phrase_pair_neg": phrase_pair_masks["phrase_neg_mask"].sum().detach(),
            "num_phrase_pair_ignore": phrase_pair_masks["phrase_ignore_mask"].sum().detach(),
            "known_phrase_pair_ratio": phrase_pair_masks["phrase_known"].float().mean().detach(),
        })

        if output_dict:
            output = {"total_loss": total_loss}
            output.update(details)
            return output

        return total_loss


class PhraseAlignLoss(_PhraseAlignLossBase):
    """Local-only PhraseAlign loss: VL-CABS done properly, and NO image<->report branch at all.

    Two things separate this from PhraseAlignJepaLoss (the JEPA variant's loss):

    1. WHERE VL-CABS happens. PhraseAlignJepaLoss receives one image vector per image, computed in
       model.forward() against that image's OWN phrases only, so every off-diagonal
       phrase<->image logit compares a phrase against an image embedding that was never
       aggregated with respect to that phrase -- source_loss/simlabel_pair_loss then push
       against meaningless negatives. Here the model hands over the raw local token map and
       `_vlcabs_local_terms` builds an (N_images, N_phrases, D) aggregate over the FULL
       cross-rank pool, so every pair is scored from its own attention map.
    2. NO clip_loss. There is no pooled/global image feature and no report-level text feature
       in this variant -- `total_loss = source_loss + label_sim_weight*simlabel_pair_loss +
       phrase_pair_weight*phrase_pair_loss`. That is the whole point of the ablation against
       phrase_align_split, which keeps the global readout and its Original CLIP loss on top of the
       same local terms. Consequently `volume_label` / `use_global_label_for_negative` do
       nothing here: they only ever protected clip_loss's negatives.

    Consumes the output of CustomTextCLIP_PhraseAlign, whose `temperature` must be a buffer (not
    nn.Parameter): it is read here, outside model.forward(), which trips DDP's "Expected to
    mark a variable ready only once" if it is trainable.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.use_global_label_for_negative:
            logging.warning(
                "use_global_label_for_negative has no effect for PhraseAlignLoss: this "
                "variant has no clip_loss term for it to protect. Ignoring."
            )

    def forward(
            self,
            local_image_features,
            key_phrase_features,
            logit_scale=1.0,
            logit_bias=None,
            output_dict: bool = False,
            label_map=None,
            counts=None,
            temperature=None,
            phrase_ids=None,
            **kwargs,
    ):
        # No global_image_features/text_features/volume_label: they exist only for clip_loss,
        # which this variant drops (see class docstring). Passing them is harmless -- they are
        # swallowed by **kwargs and never gathered.
        _, all_local_image_features, _, all_key_phrase_features, all_label_map, all_counts, _, all_phrase_ids = gather_features_vlcabs(
            local_image_features=local_image_features,
            key_phrase_features=key_phrase_features,
            label_map=label_map,
            counts=counts,
            gather_with_grad=self.gather_with_grad,
            rank=self.rank,
            world_size=self.world_size,
            phrase_ids=phrase_ids,
        )

        source_loss, simlabel_pair_loss, phrase_pair_loss, details = self._vlcabs_local_terms(
            all_local_image_features=all_local_image_features,
            all_key_phrase_features=all_key_phrase_features,
            all_label_map=all_label_map,
            all_counts=all_counts,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            temperature=temperature,
            phrase_ids=all_phrase_ids,
        )

        # loss sum -- no clip_loss term
        total_loss = source_loss + self.label_sim_weight*simlabel_pair_loss + self.phrase_pair_weight*phrase_pair_loss

        if output_dict:
            output = {"total_loss": total_loss}
            output.update(details)
            return output

        return total_loss
