import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .abstract_model import AbstractModel
# detr.py 경로에 맞게 조정하세요 (decoder만 재사용)
from .detr import TransformerDecoder, TransformerDecoderLayer
from .pos_embed import gen_3d_sincos_pos_embed


class Simple_TranSQ(AbstractModel):
    """
    pillar-0의 DETR3D decoder(원본 DETR decoder + 3D sincos pos)를 재사용한 TranSQ.

    - vision encoder가 이미 토큰을 문맥화했으므로 DETR encoder는 사용하지 않고
      decoder만 사용. visual_embed가 곧 decoder의 memory(K/V)가 됨.
    - query_pos는 원본 DETR처럼 decoder 매 층 self/cross-attention에 주입됨.
    """

    def __init__(
        self,
        args,
        input_dim=1152,        # 비전 인코더 출력 채널 (Conv3d proj의 in_channels)
        d_model=1152,          # DETR decoder 내부 차원
        d_target=4096,         # 텍스트 임베딩 차원 (Qwen)
        num_queries=64,        # Semantic Query 개수 (N)
        num_decoder_layers=6,  # 논문 L=6
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        normalize_before=False,
        match_mu=0.5,          # Hungarian matching selection ratio μ (논문 4.3)
        aux_loss=False,        # True면 각 decoder 층 출력 모두 반환
    ):
        super(Simple_TranSQ, self).__init__(args)

        self.num_queries = num_queries
        self.match_mu = match_mu
        self.aux_loss = aux_loss
        self.d_model = d_model

        # 1. 원본 DETR decoder만 구성 (encoder 미사용)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation="relu", normalize_before=normalize_before,)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=aux_loss,)

        # 2. Visual token projection (시각 토큰 채널 → d_model)
        self.input_proj = nn.Conv3d(input_dim, d_model, kernel_size=1)

        # 3. Query positional embedding (DETR object query)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # 4. Heads
        # (1) Semantic embedding: 3-layer FFN + ReLU (논문 3.2.1)
        self.topic_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_target),
        )
        # (2) Selection classifier: 단일 linear layer (논문 3.2.1)
        self.topic_clas = nn.Linear(d_model, 1)

        # 원본 Transformer._reset_parameters와 동일한 초기화 (decoder에 적용)
        self._reset_parameters()

        # gen_3d_sincos_pos_embed(numpy, 32768 position x d_model)는 입력 x가 아니라
        # (h, d, dtype)에만 의존하는 순수 함수인데, 이전엔 매 forward마다 numpy로 다시
        # 계산해서 CPU/전송 오버헤드가 매 스텝 반복됐다. shape이 바뀌지 않는 한(고정 해상도
        # 학습이므로 사실상 항상 동일) 한 번 계산해서 캐시해둔다.
        self._pos_embed_cache = {}

    def _reset_parameters(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _apply_heads(self, x):
        """x: (..., N_q, d_model) -> (sent_feats, topic_preds)"""
        sent_feats = self.topic_proj(x)
        sent_feats = F.normalize(sent_feats, p=2, dim=-1)  # 논문: L2 정규화
        topic_preds = self.topic_clas(x).squeeze(-1)
        return sent_feats, topic_preds

    def forward(self, vis_embeds):
        """
        Args:
            vis_embeds: (B, C=input_dim, D, H, W)  — 비전 인코더 feature map
                        (gen_3d_sincos_pos_embed가 H==W 정사각을 가정함에 유의)
        Returns:
            aux_loss=False: sent_feats (B, N_q, d_target), topic_preds (B, N_q)
            aux_loss=True : sent_feats (L, B, N_q, d_target),
                            topic_preds (L, B, N_q)
        """
        # Conv3d proj (flatten 불필요, 5D 유지)
        features = self.input_proj(vis_embeds)          # (B, d_model, D, H, W)
        bs, c, d, h, w = features.shape

        # sequence-first: (B,C,D,H,W) -> (D*H*W, B, C)
        src = features.flatten(2).permute(2, 0, 1)

        # 3D sincos positional embedding (memory key에 주입) — (h, w, d, dtype)가 같으면 재사용.
        # chest_ct의 activ grid는 정사각이 아니므로(patch_size가 축마다 달라 H!=W) w도 넘긴다.
        cache_key = (h, w, d, features.dtype, features.device)
        pos_base = self._pos_embed_cache.get(cache_key)
        if pos_base is None:
            pos_np = gen_3d_sincos_pos_embed(self.d_model, img_seqlen=h, frame_seqlen=d, w_seqlen=w)
            pos_base = torch.as_tensor(pos_np, dtype=features.dtype, device=features.device).unsqueeze(1)
            self._pos_embed_cache[cache_key] = pos_base
        pos = pos_base.repeat(1, bs, 1)                 # (D*H*W, B, d_model)

        # query_pos & zero content
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (N_q, B, d_model)
        tgt = torch.zeros_like(query_pos)

        # decoder only: visual token(src)이 곧 memory
        hs, _cross_attn = self.decoder(tgt, src, pos=pos, query_pos=query_pos)
        # hs: return_intermediate=False -> (1, N_q, B, d_model)
        #     return_intermediate=True  -> (L, N_q, B, d_model)
        hs = hs.transpose(1, 2)  # (*, B, N_q, d_model)

        if self.aux_loss:
            return self._apply_heads(hs)          # (L, B, N_q, ...)
        return self._apply_heads(hs[-1])          # (B, N_q, ...)

    # =====================================================================
    # [핵심 로직 1] Hungarian Matcher  (논문 식 2: -μ·p̂ - cos_sim)
    # =====================================================================
    @staticmethod
    @torch.no_grad()
    def hungarian_match(sent_feats, topic_preds, gt_embeds, gt_lens, mu=0.5):
        """loss/metric에서 공유하기 위한 staticmethod 구현."""
        bs, num_query, _ = sent_feats.shape
        # AMP(bf16) 대비 float32로 캐스팅 (numpy는 bfloat16 미지원)
        prob = topic_preds.float().sigmoid()
        pred_norm = F.normalize(sent_feats.float(), p=2, dim=-1)

        indices = []
        for i in range(bs):
            n_tgt = int(gt_lens[i])
            if n_tgt == 0:
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue

            gt_norm = F.normalize(gt_embeds[i, :n_tgt].float(), p=2, dim=-1)

            cost_sim = -(pred_norm[i] @ gt_norm.T)                 # -cosine
            cost_pick = -prob[i].unsqueeze(1).expand(-1, n_tgt)    # -prob
            C = cost_sim + mu * cost_pick

            # numpy 변환 전 float32 보장 (bf16 → TypeError 방지)
            r, c = linear_sum_assignment(C.detach().cpu().float().numpy())
            indices.append((torch.as_tensor(r, dtype=torch.long),
                            torch.as_tensor(c, dtype=torch.long)))
        return indices

    @torch.no_grad()
    def matcher(self, sent_feats, topic_preds, gt_embeds, gt_lens):
        """인스턴스 래퍼: self.match_mu를 주입."""
        return self.hungarian_match(sent_feats, topic_preds, gt_embeds, gt_lens,
                                    mu=self.match_mu)

    # =====================================================================
    # NEFTune-style 텍스트 임베딩 노이즈 (논문 3.2.1 Text encoder)
    #   frozen text encoder라 임베딩이 고정 → overfitting 완화용.
    #   ε ~ U[-1, 1] 를 1/√d 로 스케일해 더한 뒤 L2 정규화. 학습 시에만 적용.
    # =====================================================================
    @staticmethod
    def add_neftune_noise(text_embeds, training=True, normalize=True):
        """
        Args:
            text_embeds: (..., d)  텍스트(key phrase) 임베딩
            training: 학습 시에만 노이즈 적용 (eval에선 정규화만)
            normalize: 노이즈 주입 후 L2 정규화 여부 (논문: 정규화함)
        Returns:
            노이즈가 더해지고 L2 정규화된 임베딩 (dtype/shape 보존)
        """
        if not training:
            return F.normalize(text_embeds, p=2, dim=-1) if normalize else text_embeds

        d = text_embeds.size(-1)
        # ε ~ U[-1, 1] 를 1/√d 로 스케일
        noise = (torch.rand_like(text_embeds) * 2.0 - 1.0) / (d ** 0.5)
        out = text_embeds + noise
        if normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    # =====================================================================
    # [핵심 로직 2] Inference (논문 3.3.1, threshold=0.4)
    # =====================================================================
    @torch.no_grad()
    def retrieve(self, sent_feats, topic_preds, gallery_embeds, gallery_texts, prob_threshold=0.4):
        bs = sent_feats.size(0)
        prob = topic_preds.sigmoid()
        pred_norm = F.normalize(sent_feats.float(), p=2, dim=-1)
        gallery_norm = F.normalize(gallery_embeds.to(sent_feats.device).float(), p=2, dim=-1)

        results = []
        for i in range(bs):
            valid_idx = (prob[i] > prob_threshold).nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                valid_idx = torch.argmax(prob[i]).unsqueeze(0)

            valid_feats = pred_norm[i, valid_idx]
            sim = valid_feats @ gallery_norm.T
            max_idx = torch.argmax(sim, dim=-1)
            phrases = [gallery_texts[int(idx)] for idx in max_idx.cpu()]
            phrases = list(dict.fromkeys(phrases))  # 순서를 유지하며 중복 제거
            results.append(phrases)
            
        return results