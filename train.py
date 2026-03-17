#!/usr/bin/env python3

import os
import sys
import argparse
import json
import time
from pathlib import Path
import importlib.util
import random

CSI = "\x1b["
GREEN = CSI + "32m"
CYAN = CSI + "36m"
YELLOW = CSI + "33m"
RESET = CSI + "0m"

THIS_DIR = Path(__file__).resolve().parent
DAME_PY = THIS_DIR / "DAME.py"

if not DAME_PY.exists():
    print("ERROR: DAME.py bulunamadı")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("dame_mod", str(DAME_PY))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

Cfg = getattr(mod, "Config")
NLPEngine = getattr(mod, "NLPEngine")
NeuralIntentModel = getattr(mod, "NeuralIntentModel", None)

cfg = Cfg()

parser = argparse.ArgumentParser(description="DAME NeuralIntentModel trainer")
parser.add_argument("--epochs", type=int, default=int(os.environ.get("DAME_NEURAL_EPOCHS", "60")))
parser.add_argument("--lr", type=float, default=float(os.environ.get("DAME_NEURAL_LR", "0.005")))
parser.add_argument("--model-type", choices=["auto", "small", "large"], default=cfg.NEURAL_MODEL_TYPE)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--val-split", type=float, default=0.12)
parser.add_argument("--save-dir", type=str, default=str(cfg.APP_DIR))
parser.add_argument("--amp", action="store_true")
parser.add_argument("--patience", type=int, default=6)
parser.add_argument("--accum-steps", type=int, default=1)
args = parser.parse_args()

print(
    f"{CYAN}DAME trainer{RESET} | "
    f"epochs={args.epochs} lr={args.lr} batch={args.batch_size} model={args.model_type}"
)

nlp = NLPEngine()

TORCH_AVAILABLE = getattr(mod, "TORCH_OK", False)
try:
    if not TORCH_AVAILABLE:
        import torch
        TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

if not TORCH_AVAILABLE:
    print("PyTorch yok. Sadece metadata yazılacak.")
    toks = set()

    for intent, (kwlist, req, neg, base) in nlp.INTENT_PATTERNS.items():
        for kw in kwlist:
            for t in nlp.tokenize(kw):
                toks.add(t)

    for mapping in (nlp.APP_MAP, nlp.FOLDER_MAP, nlp.CURRENCY_MAP):
        for key in mapping.keys():
            for t in nlp.tokenize(key):
                toks.add(t)

    meta = {
        "vocab": sorted(list(toks)),
        "intents": list(nlp.INTENT_PATTERNS.keys()),
        "generated_at": time.asctime(),
        "note": "PyTorch kurulu değil. Bu dosya sadece metadata içerir."
    }

    outp = Path(args.save_dir) / "neural_meta.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Metadata yazıldı: {outp}")
    sys.exit(0)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

device = torch.device("cpu")
if (not args.no_cuda) and torch.cuda.is_available():
    device = torch.device("cuda")

print(f"{GREEN}Cihaz:{RESET} {device}")

if NeuralIntentModel is None:
    print("NeuralIntentModel sınıfı DAME.py içinde bulunamadı")
    sys.exit(1)

model = NeuralIntentModel(nlp, cfg, device=str(device))
setattr(model.cfg, "NEURAL_MODEL_TYPE", args.model_type)

try:
    model._build_model()
except Exception:
    pass

def vectorize_text(text):
    vec = [0.0] * max(1, len(model.vocab))
    for tok in nlp.tokenize(text):
        if tok in model.vocab:
            vec[model.vocab[tok]] += 1.0
    return vec

def augment_text(text):
    toks = nlp.tokenize(text)
    if not toks:
        return text

    out = []
    for w in toks:
        if random.random() < 0.08:
            continue
        out.append(w)

    if not out:
        out = toks[:]

    if len(out) >= 2 and random.random() < 0.12:
        i = random.randint(0, len(out) - 2)
        out[i], out[i + 1] = out[i + 1], out[i]

    if random.random() < 0.08:
        out.insert(0, random.choice(list(nlp.TR_STOPWORDS)))

    return " ".join(out)

X_list = []
Y_list = []
intents = list(model.intents)

for idx, intent in enumerate(intents):
    kwlist, req, neg, base = nlp.INTENT_PATTERNS.get(intent, ([], [], [], 1.0))

    for kw in kwlist:
        X_list.append(vectorize_text(kw))
        Y_list.append(idx)

    for kw in kwlist:
        base_text = " ".join(kwlist[:2]) if len(kwlist) >= 2 else kw
        for _ in range(3):
            aug = augment_text(base_text)
            X_list.append(vectorize_text(aug))
            Y_list.append(idx)

    if not kwlist:
        X_list.append([0.0] * max(1, len(model.vocab)))
        Y_list.append(idx)

if not X_list:
    print("Eğitim verisi üretilemedi.")
    sys.exit(1)

X = torch.tensor(X_list, dtype=torch.float32)
Y = torch.tensor(Y_list, dtype=torch.long)

dataset = TensorDataset(X, Y)

val_size = max(1, int(len(dataset) * args.val_split))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

print(
    f"Dataset hazır | total={len(dataset)} train={train_size} "
    f"val={val_size} classes={len(intents)}"
)

net = getattr(model, "model", None)
if net is None:
    print("Model ağı başlatılamadı.")
    sys.exit(1)

net.to(device)

opt = torch.optim.Adam(net.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

use_amp = args.amp and device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

best_val = float("inf")
best_epoch = -1
patience = args.patience
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

def evaluate(loader):
    net.eval()
    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = net(xb)
            else:
                out = net(xb)

            loss = loss_fn(out, yb)
            running_loss += loss.item() * xb.size(0)

            preds = out.argmax(dim=-1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

print(f"{CYAN}Eğitim başlıyor...{RESET}")

epochs = args.epochs
accum_steps = max(1, args.accum_steps)

try:
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0.0
        seen = 0
        opt.zero_grad()
        t0 = time.time()

        for i, (xb, yb) in enumerate(train_loader, 1):
            xb = xb.to(device)
            yb = yb.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = net(xb)
                    loss = loss_fn(out, yb) / accum_steps
            else:
                out = net(xb)
                loss = loss_fn(out, yb) / accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if i % accum_steps == 0:
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()

            epoch_loss += loss.item() * xb.size(0) * accum_steps
            seen += xb.size(0)

        if seen == 0:
            print("Epoch içinde veri işlenemedi.")
            break

        train_loss = epoch_loss / seen
        val_loss, val_acc = evaluate(val_loader)

        t1 = time.time()
        epoch_time = t1 - t0
        remaining = (epochs - epoch) * epoch_time

        print(
            f"{YELLOW}Epoch {epoch}/{epochs}{RESET}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}  "
            f"time={epoch_time:.1f}s  "
            f"ETA~{int(remaining)}s"
        )

        chk_path = save_dir / f"neural_epoch_{epoch}.pt"
        try:
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "vocab": model.vocab,
                    "intents": model.intents,
                },
                str(chk_path)
            )
        except Exception:
            pass

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_path = save_dir / "neural_best.pt"
            try:
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "vocab": model.vocab,
                        "intents": model.intents,
                    },
                    str(best_path)
                )
                print(f"{GREEN}En iyi model kaydedildi:{RESET} {best_path}")
            except Exception:
                pass
        else:
            if epoch - best_epoch >= patience:
                print(f"{YELLOW}{patience} epoch boyunca gelişme yok. Eğitim durduruluyor.{RESET}")
                break

except KeyboardInterrupt:
    print("\nEğitim kullanıcı tarafından durduruldu. Checkpoint kaydediliyor...")
    try:
        intr_path = save_dir / "neural_interrupted.pt"
        torch.save(
            {
                "state_dict": net.state_dict(),
                "vocab": model.vocab,
                "intents": model.intents,
            },
            str(intr_path)
        )
        print(f"Kesilen checkpoint kaydedildi: {intr_path}")
    except Exception as e:
        print(f"Kesilen checkpoint kaydedilemedi: {e}")

try:
    final_path = Path(cfg.NEURAL_SAVE_PATH)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "vocab": model.vocab,
            "intents": model.intents,
        },
        str(final_path)
    )
    print(f"Final model yazıldı: {final_path}")
except Exception as e:
    print(f"Final model kaydedilemedi: {e}")

print("Eğitim bitti.")