from .loss_phrase_align import _PhraseAlignLossBase
# The gather + loss pipeline this class needs (global + local image tokens + phrases +
# labels/counts) is exactly what PhraseAlignLoss needs, so both live in loss_phrase_align.py
# (`gather_features_vlcabs`, `_PhraseAlignLossBase._vlcabs_forward`) instead of being kept here as
# a second copy that would have to be edited twice.


class PhraseAlignSplitLoss(_PhraseAlignLossBase):
    """VL-CABS-in-the-loss variant fed by the *split* vision tower, WITH the CLIP branch.

    Shares its three phrase-level terms with PhraseAlignLoss (both call
    `_vlcabs_local_terms`) and adds the Original CLIP loss on top, computed between the
    encoder's separate global readout and the report-level text feature. That extra branch is
    exactly what the phrase_align (local-only) vs phrase_align_split ablation isolates.
    """

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
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
            **kwargs,
    ):
        # Whole pipeline (gather -> clip_loss -> VL-CABS -> source/simlabel/phrase-pair terms)
        # lives in _PhraseAlignLossBase._vlcabs_forward, shared with PhraseAlignLoss so the two
        # variants cannot drift apart.
        return self._vlcabs_forward(
            global_image_features=global_image_features,
            local_image_features=local_image_features,
            text_features=text_features,
            key_phrase_features=key_phrase_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            output_dict=output_dict,
            label_map=label_map,
            counts=counts,
            temperature=temperature,
            volume_label=volume_label,
        )
