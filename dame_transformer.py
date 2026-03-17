#!/usr/bin/env python3

from __future__ import annotations

import os, sys, re, json, math, time, random, logging, argparse, sqlite3
import struct, hashlib, pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from collections import Counter, defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, IterableDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[UYARI] PyTorch bulunamadı. Sadece --phase info çalışır.")
    print("  GPU için : pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("  CPU için : pip install torch torchvision torchaudio")

AMP_OK = False
try:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast  as _autocast
    def GradScaler(enabled=True):
        return _GradScaler("cuda", enabled=enabled)
    def amp_autocast():
        return _autocast("cuda")
    AMP_OK = True
except Exception:
    try:
        from torch.cuda.amp import GradScaler, autocast as amp_autocast
        AMP_OK = True
    except Exception:
        class GradScaler:
            def __init__(self, enabled=False): pass
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass
        def amp_autocast():
            import contextlib
            return contextlib.nullcontext()
        AMP_OK = False

CSI = "\x1b["
GREEN = CSI + "32m"
CYAN = CSI + "36m"
YELLOW = CSI + "33m"
RED = CSI + "31m"
BOLD = CSI + "1m"
RESET = CSI + "0m"

DAME_DIR = Path.home() / ".dame"
WIKI_DIR = DAME_DIR / "wiki"
MODEL_DIR = DAME_DIR / "transformer"
BPE_PATH = MODEL_DIR / "bpe_tokenizer.pkl"
MODEL_PATH = MODEL_DIR / "dame_lm.pt"
CONFIG_PATH = MODEL_DIR / "config.json"
LOG_PATH = MODEL_DIR / "train.log"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_logger() -> logging.Logger:
    log = logging.getLogger("DAMETransformer")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", "%H:%M:%S")
    if not log.handlers:
        fh = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(ch)
    return log

log = build_logger()


class DAMEConfig:
    __slots__ = (
        "vocab_size",
        "n_layer",
        "n_head",
        "n_embd",
        "n_ff",
        "block_size",
        "dropout",
        "batch_size",
        "grad_accum",
        "lr",
        "lr_min",
        "warmup_steps",
        "max_steps",
        "clip_grad",
        "weight_decay",
        "train_split",
        "seed",
        "pad_id",
        "bos_id",
        "eos_id",
        "unk_id",
    )

    FIELD_NAMES = {
        "vocab_size",
        "n_layer",
        "n_head",
        "n_embd",
        "n_ff",
        "block_size",
        "dropout",
        "batch_size",
        "grad_accum",
        "lr",
        "lr_min",
        "warmup_steps",
        "max_steps",
        "clip_grad",
        "weight_decay",
        "train_split",
        "seed",
        "pad_id",
        "bos_id",
        "eos_id",
        "unk_id",
    }

    def __init__(
        self,
        vocab_size: int = 16_000,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 512,
        n_ff: int = 2048,
        block_size: int = 512,
        dropout: float = 0.10,
        batch_size: int = 32,
        grad_accum: int = 4,
        lr: float = 3e-4,
        lr_min: float = 1e-5,
        warmup_steps: int = 2_000,
        max_steps: int = 100_000,
        clip_grad: float = 1.0,
        weight_decay: float = 0.01,
        train_split: float = 0.97,
        seed: int = 42,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        unk_id: int = 3,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_ff = n_ff
        self.block_size = block_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.lr = lr
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.clip_grad = clip_grad
        self.weight_decay = weight_decay
        self.train_split = train_split
        self.seed = seed
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    @classmethod
    def field_names(cls):
        return cls.FIELD_NAMES

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "n_ff": self.n_ff,
            "block_size": self.block_size,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "lr": self.lr,
            "lr_min": self.lr_min,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "clip_grad": self.clip_grad,
            "weight_decay": self.weight_decay,
            "train_split": self.train_split,
            "seed": self.seed,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "DAMEConfig":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.field_names()})

    def param_count(self) -> int:
        emb = self.vocab_size * self.n_embd
        pos = self.block_size * self.n_embd
        per_layer = (
            4 * self.n_embd * self.n_embd +
            2 * self.n_embd * self.n_ff +
            4 * self.n_embd
        )
        return emb + pos + self.n_layer * per_layer + self.vocab_size * self.n_embd


class BPETokenizer:
    SPECIAL = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inv: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._trained = False

    def train(
        self,
        texts: Iterator[str],
        vocab_size: int = 16_000,
        min_freq: int   = 2,
        max_words: int  = 300_000,
        show_progress: bool = True
    ):
        import heapq
        log.info("BPE eğitimi başlıyor (hızlı mod)...")
        t0 = time.time()

        word_freqs: Counter = Counter()
        n_texts = 0
        for text in texts:
            for word in re.findall(r"[a-z\u00c0-\u024f\u011e\u011f\u0130\u0131\u015e\u015f\u00fc\u00f6\u00e7\u015f]+", text.lower()):
                if len(word) > 1:
                    word_freqs[word] += 1
            n_texts += 1
            if n_texts % 10_000 == 0 and show_progress:
                print(f"\r  BPE: {n_texts:,} metin, {len(word_freqs):,} kelime",
                      end="", flush=True)

        print(f"\r  BPE: {n_texts:,} metin, {len(word_freqs):,} kelime  ({int(time.time()-t0)}s)")

        word_freqs = Counter({w: c for w, c in word_freqs.items() if c >= min_freq})
        if len(word_freqs) > max_words:
            word_freqs = Counter(dict(word_freqs.most_common(max_words)))
        log.info(f"Çalışma kelime sayısı: {len(word_freqs):,}")

        vocab_seqs: Dict[str, List[str]] = {
            w: list(w) + ["</w>"] for w in word_freqs
        }

        id_map: Dict[str, int] = {}
        for sp in self.SPECIAL:
            id_map[sp] = len(id_map)
        for word in word_freqs:
            for ch in list(word) + ["</w>"]:
                if ch not in id_map:
                    id_map[ch] = len(id_map)

        pair_freq: Dict[Tuple[str,str], int] = defaultdict(int)
        pair_words: Dict[Tuple[str,str], set] = defaultdict(set)

        for word, syms in vocab_seqs.items():
            f = word_freqs[word]
            for a, b in zip(syms, syms[1:]):
                pair_freq[(a, b)] += f
                pair_words[(a, b)].add(word)

        heap = [(-f, p) for p, f in pair_freq.items()]
        heapq.heapify(heap)

        merges: List[Tuple[str, str]] = []
        target = vocab_size - len(id_map)
        log.info(f"Hedef BPE merge sayısı: {target:,}")

        step = 0
        while step < target:
            best = None
            while heap:
                neg_f, cand = heapq.heappop(heap)
                actual = pair_freq.get(cand, 0)
                if actual == -neg_f:
                    best = cand
                    break
                elif actual > 0:
                    heapq.heappush(heap, (-actual, cand))
                    best = cand
                    break
            if best is None or pair_freq.get(best, 0) == 0:
                break

            a0, b0 = best
            merged  = a0 + b0
            merges.append(best)
            if merged not in id_map:
                id_map[merged] = len(id_map)

            affected = list(pair_words.get(best, set()))
            for word in affected:
                syms = vocab_seqs[word]
                f    = word_freqs[word]
                new_syms: List[str] = []
                i = 0
                while i < len(syms):
                    if i < len(syms)-1 and syms[i] == a0 and syms[i+1] == b0:
                        if i > 0:
                            old_p = (syms[i-1], a0)
                            pair_freq[old_p] -= f
                            pair_words[old_p].discard(word)
                        if i+2 < len(syms):
                            old_p = (b0, syms[i+2])
                            pair_freq[old_p] -= f
                            pair_words[old_p].discard(word)
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1

                vocab_seqs[word] = new_syms
                for j, (x, y) in enumerate(zip(new_syms, new_syms[1:])):
                    if x == merged or y == merged:
                        pair_freq[(x, y)] += f
                        pair_words[(x, y)].add(word)
                        heapq.heappush(heap, (-pair_freq[(x, y)], (x, y)))

            pair_freq[best] = 0
            pair_words.pop(best, None)

            step += 1
            if step % 500 == 0 and show_progress:
                elapsed = time.time() - t0
                eta     = elapsed / step * (target - step)
                print(f"\r  BPE merge: {step:>6,}/{target:,} "
                      f"({step/target*100:.1f}%)  "
                      f"vocab={len(id_map):,}  "
                      f"{int(elapsed)}s  ETA~{int(eta)}s",
                      end="", flush=True)

        elapsed = int(time.time() - t0)
        print(f"\r  BPE tamamlandı: {len(id_map):,} token, "
              f"{len(merges):,} merge  ({elapsed}s)          ")

        self.vocab    = id_map
        self.inv      = {v: k for k, v in id_map.items()}
        self.merges   = merges
        self._trained = True
        log.info(f"BPE vocab boyutu: {len(self.vocab):,}")

    def _encode_word(self, word: str) -> List[int]:
        syms = list(word) + ["</w>"]
        for a, b in self.merges:
            i = 0
            new = []
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(syms[i])
                    i += 1
            syms = new
        return [self.vocab.get(s, self.UNK_ID) for s in syms]

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True
    ) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.BOS_ID)
        for word in re.findall(r"\S+", text.lower()):
            ids.extend(self._encode_word(word))
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [
            self.inv.get(i, "[UNK]") for i in ids
            if i not in (self.BOS_ID, self.EOS_ID, self.PAD_ID)
        ]
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "inv": self.inv,
                    "merges": self.merges,
                },
                f,
                protocol=4
            )
        log.info(f"BPE kaydedildi: {path}")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.inv = data["inv"]
        tok.merges = data["merges"]
        tok._trained = True
        log.info(f"BPE yüklendi: {path}  vocab={len(tok.vocab):,}")
        return tok


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DAMEConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd, n_head'e bölünebilmeli"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, cfg: DAMEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_ff, bias=False),
            nn.GELU(),
            nn.Linear(cfg.n_ff, cfg.n_embd, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: DAMEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DAMETransformer(nn.Module):
    def __init__(self, cfg: DAMEConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict({
            "tok_emb": nn.Embedding(cfg.vocab_size, cfg.n_embd),
            "pos_emb": nn.Embedding(cfg.block_size, cfg.n_embd),
            "drop": nn.Dropout(cfg.dropout),
            "blocks": nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)]),
            "ln_f": nn.LayerNorm(cfg.n_embd),
        })
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer["tok_emb"].weight

        self.apply(self._init_weights)
        log.info(f"DAMETransformer: {self.param_count():,} parametre")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Seq len {T} > block_size {self.cfg.block_size}"

        device = idx.device
        pos = torch.arange(T, device=device).unsqueeze(0)

        tok = self.transformer["tok_emb"](idx)
        pe = self.transformer["pos_emb"](pos)
        x = self.transformer["drop"](tok + pe)

        for block in self.transformer["blocks"]:
            x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.92,
        repetition_penalty: float = 1.3,
        eos_id: int = 2
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = idx[b].unique()
                    logits[b, seen] = torch.where(
                        logits[b, seen] > 0,
                        logits[b, seen] / repetition_penalty,
                        logits[b, seen] * repetition_penalty,
                    )

            if top_k > 0:
                k = min(top_k, logits.size(-1))
                thr = logits.topk(k).values[:, -1, None]
                logits = logits.masked_fill(logits < thr, float("-inf"))

            if 0.0 < top_p < 1.0:
                probs_sorted, sorted_idx = torch.sort(
                    F.softmax(logits, dim=-1), dim=-1, descending=True
                )
                cum = probs_sorted.cumsum(dim=-1)
                remove = cum - probs_sorted > top_p
                remove[:, 0] = False
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask.scatter_(1, sorted_idx, remove)
                logits = logits.masked_fill(mask, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_t], dim=1)

            if (next_t == eos_id).all():
                break

        return idx

    def save(self, path: Path, cfg: DAMEConfig, step: int = 0, loss: float = 0.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "step": step,
                "loss": loss,
                "cfg": cfg.to_dict(),
            },
            str(path)
        )
        log.info(f"Model kaydedildi: {path}  step={step}  loss={loss:.4f}")

    @classmethod
    def load(cls, path: Path) -> Tuple["DAMETransformer", DAMEConfig, int]:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = DAMEConfig(**{
            k: v for k, v in ckpt["cfg"].items()
            if k in DAMEConfig.field_names()
        })
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        step = ckpt.get("step", 0)
        return model, cfg, step


class WikiTokenDataset(IterableDataset):
    def __init__(
        self,
        db_paths: List[Path],
        tokenizer: BPETokenizer,
        block_size: int,
        split: str = "train",
        val_split: float = 0.03,
        seed: int = 42
    ):
        self.db_paths = [str(p) for p in db_paths if Path(p).exists()]
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.split = split
        self.val_split = val_split
        self.seed = seed

        if not self.db_paths:
            raise FileNotFoundError(
                "Hiç Wikipedia DB bulunamadı. "
                "Önce wiki_builder.py ile build edin."
            )

    def _get_article_ids(self, db_path: str) -> List[int]:
        conn = sqlite3.connect(db_path)
        ids = [r[0] for r in conn.execute("SELECT id FROM articles ORDER BY id").fetchall()]
        conn.close()
        random.seed(self.seed)
        random.shuffle(ids)
        cut = int(len(ids) * (1 - self.val_split))
        return ids[:cut] if self.split == "train" else ids[cut:]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed)
        buf: List[int] = []

        for db_path in self.db_paths:
            all_ids = self._get_article_ids(db_path)

            if worker_info is not None:
                per_w = math.ceil(len(all_ids) / worker_info.num_workers)
                start = worker_info.id * per_w
                all_ids = all_ids[start:start + per_w]

            rng.shuffle(all_ids)
            conn = sqlite3.connect(db_path)

            for art_id in all_ids:
                row = conn.execute(
                    "SELECT text FROM articles WHERE id=?", (art_id,)
                ).fetchone()
                if not row:
                    continue
                tokens = self.tokenizer.encode(row[0], add_bos=True, add_eos=True)
                buf.extend(tokens)

                while len(buf) >= self.block_size + 1:
                    chunk = buf[:self.block_size + 1]
                    buf = buf[self.block_size:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield {"input_ids": x, "labels": y}

            conn.close()


def get_lr(step: int, cfg: DAMEConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine


class Trainer:
    def __init__(
        self,
        model: DAMETransformer,
        cfg: DAMEConfig,
        tokenizer: BPETokenizer,
        db_paths: List[Path],
        device: torch.device,
        save_every: int = 2_000,
        val_every: int = 500,
        use_amp: bool = True
    ):
        self.model = model
        self.cfg = cfg
        self.tok = tokenizer
        self.device = device
        self.save_every = save_every
        self.val_every = val_every
        self.step = 0

        decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        self.use_amp = use_amp and AMP_OK and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        train_ds = WikiTokenDataset(
            db_paths, tokenizer, cfg.block_size,
            split="train", val_split=1 - cfg.train_split,
            seed=cfg.seed
        )
        val_ds = WikiTokenDataset(
            db_paths, tokenizer, cfg.block_size,
            split="val", val_split=1 - cfg.train_split,
            seed=cfg.seed
        )
        nw = min(4, os.cpu_count() or 1)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            num_workers=nw,
            pin_memory=(device.type == "cuda")
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            num_workers=nw,
            pin_memory=(device.type == "cuda")
        )

    def _val_loss(self, max_batches: int = 20) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= max_batches:
                    break
                x = batch["input_ids"].to(self.device)
                y = batch["labels"].to(self.device)
                if self.use_amp:
                    with amp_autocast():
                        _, loss = self.model(x, y)
                else:
                    _, loss = self.model(x, y)
                if loss is not None:
                    losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses) if losses else float("inf")

    def _checkpoint_path(self, step: int) -> Path:
        return MODEL_DIR / f"ckpt_step_{step:07d}.pt"

    def resume(self) -> bool:
        ckpts = sorted(MODEL_DIR.glob("ckpt_step_*.pt"))
        if not ckpts:
            return False
        latest = ckpts[-1]
        ckpt = torch.load(str(latest), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.opt.load_state_dict(ckpt.get("opt_state", {}))
        self.step = ckpt.get("step", 0)
        log.info(f"Checkpoint yüklendi: {latest}  step={self.step}")
        return True

    def train(self, max_steps: Optional[int] = None):
        max_steps = max_steps or self.cfg.max_steps
        self.model.train()
        self.model.to(self.device)

        accum = self.cfg.grad_accum
        t_start = time.time()
        running_loss = 0.0

        log.info(
            f"Eğitim başlıyor: max_steps={max_steps}  "
            f"device={self.device}  amp={self.use_amp}"
        )
        print(
            f"\n{BOLD}{CYAN}Eğitim başlıyor{RESET}  "
            f"steps={max_steps}  device={self.device}  "
            f"params={self.model.param_count():,}  "
            f"amp={'ON' if self.use_amp else 'OFF'}"
        )

        self.opt.zero_grad()

        for batch in self.train_loader:
            if self.step >= max_steps:
                break

            x = batch["input_ids"].to(self.device)
            y = batch["labels"].to(self.device)

            lr_now = get_lr(self.step, self.cfg)
            for g in self.opt.param_groups:
                g["lr"] = lr_now

            if self.use_amp:
                with amp_autocast():
                    _, loss = self.model(x, y)
                    loss = loss / accum
                self.scaler.scale(loss).backward()
            else:
                _, loss = self.model(x, y)
                loss = loss / accum
                loss.backward()

            running_loss += loss.item() * accum

            if (self.step + 1) % accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
                if self.use_amp:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
                self.opt.zero_grad()

            self.step += 1

            if self.step % 50 == 0:
                elapsed = time.time() - t_start
                avg_loss = running_loss / 50
                ppl = math.exp(min(avg_loss, 20))
                pct = self.step / max_steps * 100
                eta = elapsed / self.step * (max_steps - self.step)
                running_loss = 0.0
                print(
                    f"\r  step={self.step:>7,}/{max_steps}  "
                    f"({pct:.1f}%)  loss={avg_loss:.4f}  "
                    f"ppl={ppl:.1f}  lr={lr_now:.2e}  "
                    f"ETA~{int(eta//60)}dk  ",
                    end="",
                    flush=True
                )

            if self.step % self.val_every == 0:
                val_loss = self._val_loss()
                val_ppl = math.exp(min(val_loss, 20))
                print(
                    f"\n  {YELLOW}[val]{RESET}  "
                    f"step={self.step}  val_loss={val_loss:.4f}  "
                    f"val_ppl={val_ppl:.1f}"
                )
                log.info(f"step={self.step}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}")

            if self.step % self.save_every == 0:
                ckpt_path = self._checkpoint_path(self.step)
                torch.save(
                    {
                        "state_dict": self.model.state_dict(),
                        "opt_state": self.opt.state_dict(),
                        "step": self.step,
                        "loss": running_loss,
                        "cfg": self.cfg.to_dict(),
                    },
                    str(ckpt_path)
                )
                self.model.save(MODEL_PATH, self.cfg, self.step)
                print(f"\n  {GREEN}✓ Checkpoint:{RESET} {ckpt_path.name}")

        self.model.save(MODEL_PATH, self.cfg, self.step)
        total_time = int(time.time() - t_start)
        print(
            f"\n\n{GREEN}✓ Eğitim tamamlandı!{RESET}  "
            f"steps={self.step}  süre={total_time//60}dk {total_time%60}s"
        )
        log.info(f"Eğitim tamamlandı: steps={self.step}  süre={total_time}s")


class DAMEInference:
    SYSTEM_PROMPT = (
        "Aşağıdaki soruyu Türkçe olarak bilgili ve öz biçimde yanıtla.\n\n"
    )

    def __init__(
        self,
        model: DAMETransformer,
        tokenizer: BPETokenizer,
        cfg: DAMEConfig,
        device: torch.device
    ):
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.cfg = cfg
        self.device = device

    @classmethod
    def load(
        cls,
        model_path: Path = MODEL_PATH,
        tokenizer_path: Path = BPE_PATH,
        device_str: str = "auto"
    ) -> "DAMEInference":
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {model_path}\n"
                "Önce eğitim yapın:  "
                "python dame_transformer.py --phase train"
            )
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer bulunamadı: {tokenizer_path}\n"
                "Önce:  python dame_transformer.py --phase tokenizer"
            )

        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        model, cfg, step = DAMETransformer.load(model_path)
        tok = BPETokenizer.load(tokenizer_path)
        log.info(f"DAMEInference yüklendi: step={step}  device={device}")
        return cls(model, tok, cfg, device)

    def available(self) -> bool:
        return MODEL_PATH.exists() and BPE_PATH.exists()

    def generate(
        self,
        prompt: str,
        max_new: int = 200,
        temperature: float = 0.75,
        top_k: int = 50,
        top_p: float = 0.92
    ) -> str:
        ids = self.tok.encode(prompt, add_bos=True, add_eos=False)
        if len(ids) > self.cfg.block_size - max_new:
            ids = ids[-(self.cfg.block_size - max_new):]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                x,
                max_new=max_new,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=self.tok.EOS_ID
            )
        generated = out[0, len(ids):].tolist()
        return self.tok.decode(generated).strip()

    def answer(
        self,
        question: str,
        max_new: int = 300,
        temperature: float = 0.6
    ) -> str:
        prompt = f"{self.SYSTEM_PROMPT}Soru: {question}\nCevap:"
        raw = self.generate(prompt, max_new=max_new, temperature=temperature)
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        if lines:
            return lines[0]
        return raw.strip()

    def perplexity(self, text: str) -> float:
        ids = self.tok.encode(text, add_bos=True, add_eos=True)
        if len(ids) < 2:
            return float("inf")
        ids = ids[:self.cfg.block_size + 1]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, loss = self.model(x, y)
        return math.exp(loss.item()) if loss is not None else float("inf")


def _detect_dbs() -> List[Path]:
    paths = [WIKI_DIR / "wiki_tr.db", WIKI_DIR / "wiki_en.db"]
    return [p for p in paths if p.exists()]


def main():
    parser = argparse.ArgumentParser(
        description="DAME Transformer — BPE + GPT eğitim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Aşamalar:
  Faz A — BPE tokenizer eğit:
    python dame_transformer.py --phase tokenizer --vocab-size 16000

  Faz B — Modeli eğit:
    python dame_transformer.py --phase train --epochs 10 --amp

  Faz C — Test:
    python dame_transformer.py --phase generate --prompt "Türkiye nedir?"

  Faz D — Perplexity ölç:
    python dame_transformer.py --phase eval --text "Ankara Türkiye'nin başkentidir."
        """
    )
    parser.add_argument(
        "--phase",
        choices=["tokenizer", "train", "generate", "eval", "info"],
        default="info"
    )
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = epochs'tan hesapla")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompt", type=str, default="Türkiye hakkında")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--max-new", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cpu")
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"\n{BOLD}{CYAN}DAME Transformer  —  Faz 3{RESET}  device={device}")

    if args.phase == "info":
        dbs_info = _detect_dbs()
        torch_str = ("VAR  v" + torch.__version__) if TORCH_OK else "YOK  → pip install torch"
        print(f"\n  PyTorch : {torch_str}")
        print(f"  Wiki DB : {[str(p) for p in dbs_info] or 'Bulunamadi'}")
        print(f"  BPE     : {BPE_PATH}  ({'VAR' if BPE_PATH.exists() else 'YOK'})")
        print(f"  Model   : {MODEL_PATH}  ({'VAR' if MODEL_PATH.exists() else 'YOK'})")
        if not TORCH_OK:
            print("\n  Kurulum:")
            print("    GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("    CPU: pip install torch torchvision torchaudio")
            return
        cfg = DAMEConfig(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size
        )
        print(f"\n  Model config:")
        print(f"    n_layer    = {cfg.n_layer}")
        print(f"    n_head     = {cfg.n_head}")
        print(f"    n_embd     = {cfg.n_embd}")
        print(f"    n_ff       = {cfg.n_ff}")
        print(f"    block_size = {cfg.block_size}")
        print(f"    vocab_size = {cfg.vocab_size}")
        print(f"    ~parametre = {cfg.param_count():,}")
        dbs = _detect_dbs()
        print(f"\n  Wiki DB'ler: {[str(p) for p in dbs] or 'Bulunamadı'}")
        print(f"  BPE path   : {BPE_PATH}  ({'VAR' if BPE_PATH.exists() else 'YOK'})")
        print(f"  Model path : {MODEL_PATH}  ({'VAR' if MODEL_PATH.exists() else 'YOK'})")
        print(f"\n  Sonraki adım:")
        if not dbs:
            print(f"    {YELLOW}1. python wiki_builder.py --lang tr --limit 200000{RESET}")
        if not BPE_PATH.exists():
            print(f"    {YELLOW}2. python dame_transformer.py --phase tokenizer{RESET}")
        if not MODEL_PATH.exists():
            print(f"    {YELLOW}3. python dame_transformer.py --phase train --amp{RESET}")
        return

    if args.phase in ("tokenizer", "train", "generate", "eval") and not TORCH_OK:
        print(f"HATA: PyTorch gerekli.  pip install torch")
        print(f"GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    if args.phase == "tokenizer":
        dbs = _detect_dbs()
        if not dbs:
            print(f"{RED}Wikipedia DB bulunamadı. Önce: python wiki_builder.py --lang tr{RESET}")
            sys.exit(1)

        def _text_gen():
            for db_path in dbs:
                conn = sqlite3.connect(str(db_path))
                for (text,) in conn.execute("SELECT text FROM articles"):
                    if text:
                        yield text
                conn.close()

        tok = BPETokenizer()
        tok.train(_text_gen(), vocab_size=args.vocab_size, show_progress=True)
        tok.save(BPE_PATH)
        print(f"\n{GREEN}✓ BPE tokenizer kaydedildi:{RESET} {BPE_PATH}")
        print(f"  Vocab boyutu: {tok.vocab_size():,}")

        cfg = DAMEConfig(vocab_size=tok.vocab_size())
        cfg.save(CONFIG_PATH)
        return

    if args.phase == "train":
        dbs = _detect_dbs()
        if not dbs:
            print(f"{RED}Wikipedia DB bulunamadı.{RESET}")
            sys.exit(1)
        if not BPE_PATH.exists():
            print(f"{RED}BPE tokenizer bulunamadı. Önce: --phase tokenizer{RESET}")
            sys.exit(1)

        tok = BPETokenizer.load(BPE_PATH)

        if CONFIG_PATH.exists():
            cfg = DAMEConfig.load(CONFIG_PATH)
            cfg.n_layer = args.n_layer
            cfg.n_head = args.n_head
            cfg.n_embd = args.n_embd
            cfg.n_ff = args.n_embd * 4
            cfg.block_size = args.block_size
            cfg.batch_size = args.batch_size
            cfg.lr = args.lr
            cfg.vocab_size = tok.vocab_size()
        else:
            cfg = DAMEConfig(
                vocab_size=tok.vocab_size(),
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                n_ff=args.n_embd * 4,
                block_size=args.block_size,
                batch_size=args.batch_size,
                lr=args.lr,
            )
        cfg.save(CONFIG_PATH)

        model = DAMETransformer(cfg).to(device)
        print(f"  Parametre sayısı: {model.param_count():,}")

        if args.max_steps > 0:
            cfg.max_steps = args.max_steps
        else:
            try:
                total_tokens = sum(
                    sqlite3.connect(str(p)).execute(
                        "SELECT SUM(tokens) FROM articles"
                    ).fetchone()[0] or 0
                    for p in dbs
                )
                steps_per_epoch = total_tokens // (cfg.block_size * cfg.batch_size)
                cfg.max_steps = steps_per_epoch * args.epochs
                print(
                    f"  Toplam token: {total_tokens:,}  "
                    f"→ {args.epochs} epoch = {cfg.max_steps:,} adım"
                )
            except Exception:
                cfg.max_steps = 50_000 * args.epochs

        trainer = Trainer(model, cfg, tok, dbs, device,
                          use_amp=(args.amp and AMP_OK and device.type == "cuda"))

        if args.resume:
            trainer.resume()

        trainer.train()
        return

    if args.phase == "generate":
        lm = DAMEInference.load(device_str="auto")
        print(f"\n{CYAN}Prompt:{RESET} {args.prompt}")
        result = lm.generate(
            args.prompt,
            max_new=args.max_new,
            temperature=args.temperature
        )
        print(f"\n{GREEN}Çıktı:{RESET}\n{result}")
        return

    if args.phase == "eval":
        lm = DAMEInference.load(device_str="auto")
        text = args.text or "Türkiye Orta Doğu ve Avrupa arasında köprü görevi gören bir ülkedir."
        ppl = lm.perplexity(text)
        ans = lm.answer("Türkiye'nin başkenti neresidir?")
        print(f"\n  Perplexity ({text[:50]}...): {ppl:.2f}")
        print(f"  Test cevabı: {ans}")
        return


if __name__ == "__main__":
    main()