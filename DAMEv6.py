#!/usr/bin/env python3

import tkinter as tk
from tkinter import scrolledtext
import threading, os, sys, re, time, random, math, ast
import datetime, platform, subprocess, webbrowser, socket
import json, html, urllib.request, urllib.parse, urllib.error
import sqlite3, logging, shutil, ctypes
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod

try:
    import pyttsx3;                 TTS_OK = True
except:
    TTS_OK = False

try:
    import speech_recognition as sr; STT_OK = True
except:
    STT_OK = False

try:
    import psutil;                  PSUTIL_OK = True
except:
    PSUTIL_OK = False

try:
    import pyautogui;               PYAUTOGUI_OK = True
except:
    PYAUTOGUI_OK = False

try:
    from bs4 import BeautifulSoup;  BS4_OK = True
except:
    BS4_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_OK = True
except Exception:
    TORCH_OK = False

_THIS_DIR = Path(__file__).resolve().parent

DAME_LM: Optional[Any] = None
try:
    import importlib.util as _ilu

    _dt_path = _THIS_DIR / "dame_transformer.py"
    if _dt_path.exists():
        _spec = _ilu.spec_from_file_location("dame_transformer", str(_dt_path))
        _dt = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_dt)
        if TORCH_OK and _dt.MODEL_PATH.exists() and _dt.BPE_PATH.exists():
            DAME_LM = _dt.DAMEInference.load()
            print(f"[DAME] DAMEInference yüklendi ✓")
        else:
            print(f"[DAME] dame_transformer.py bulundu ama model henüz eğitilmemiş.")
    else:
        print(f"[DAME] dame_transformer.py bulunamadı — LM devre dışı.")
except Exception as _e:
    print(f"[DAME] LM yükleme hatası: {_e}")

WIKI_RETRIEVER: Optional[Any] = None
try:
    _wb_path = _THIS_DIR / "wiki_builder.py"
    if _wb_path.exists():
        _spec2 = _ilu.spec_from_file_location("wiki_builder", str(_wb_path))
        _wb = _ilu.module_from_spec(_spec2)
        _spec2.loader.exec_module(_wb)
        WIKI_RETRIEVER = _wb.get_retriever()
        if WIKI_RETRIEVER:
            stats = WIKI_RETRIEVER.stats()
            total = sum(v.get("articles", 0) for v in stats.values())
            print(f"[DAME] WikiRetriever yüklendi ✓  ({total:,} makale)")
        else:
            print(f"[DAME] wiki_builder.py bulundu ama DB henüz oluşturulmamış.")
    else:
        print(f"[DAME] wiki_builder.py bulunamadı — Wiki DB devre dışı.")
except Exception as _e:
    print(f"[DAME] WikiRetriever yükleme hatası: {_e}")

LM_AVAILABLE = DAME_LM is not None
WIKI_AVAILABLE = WIKI_RETRIEVER is not None


class Config:
    APP_DIR = Path.home() / ".dame"
    DB_PATH = APP_DIR / "memory.db"
    LOG_PATH = APP_DIR / "dame.log"
    SCRAPER_TIMEOUT = 9
    SHORT_MEM_LIMIT = 24
    VOICE_RATE = 158
    VOICE_VOLUME = 0.92
    VERSION = "6.0"

    LM_MAX_NEW = int(os.environ.get("DAME_LM_MAX_NEW", "300"))
    LM_TEMPERATURE = float(os.environ.get("DAME_LM_TEMP", "0.65"))
    LM_TOP_K = int(os.environ.get("DAME_LM_TOP_K", "50"))
    LM_TOP_P = float(os.environ.get("DAME_LM_TOP_P", "0.92"))

    WIKI_CTX_CHARS = int(os.environ.get("DAME_WIKI_CTX", "600"))

    NEURAL_ENABLED = os.environ.get("DAME_NEURAL_ENABLED", "1") == "1"
    NEURAL_EPOCHS = int(os.environ.get("DAME_NEURAL_EPOCHS", "40"))
    NEURAL_LR = float(os.environ.get("DAME_NEURAL_LR", "0.01"))
    NEURAL_HIDDEN1 = int(os.environ.get("DAME_NEURAL_HIDDEN1", "256"))
    NEURAL_HIDDEN2 = int(os.environ.get("DAME_NEURAL_HIDDEN2", "128"))
    NEURAL_TRAIN_BG = os.environ.get("DAME_NEURAL_TRAIN_BG", "1") == "1"
    NEURAL_SAVE_PATH = APP_DIR / "neural_intent.pt"
    NEURAL_MODEL_TYPE = os.environ.get("DAME_NEURAL_MODEL_TYPE", "auto")

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8",
    }

    def __init__(self):
        self.APP_DIR.mkdir(exist_ok=True)


def build_logger(cfg: Config) -> logging.Logger:
    logger = logging.getLogger("DAME")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(cfg.LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


class Memory:
    def __init__(self, cfg: Config, log: logging.Logger):
        self.log = log
        self.db = sqlite3.connect(str(cfg.DB_PATH), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._tables()
        self.short: deque = deque(maxlen=cfg.SHORT_MEM_LIMIT)

    def _tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY, value TEXT,
                updated_at TEXT DEFAULT (datetime('now')));
            CREATE TABLE IF NOT EXISTS tool_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT, args TEXT, result TEXT,
                ts TEXT DEFAULT (datetime('now')));
            CREATE TABLE IF NOT EXISTS long_term (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT, content TEXT, source TEXT,
                ts TEXT DEFAULT (datetime('now')));
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT, ts TEXT DEFAULT (datetime('now')));
            CREATE TABLE IF NOT EXISTS lexicon (
                term TEXT PRIMARY KEY,
                definition TEXT, source TEXT,
                ts TEXT DEFAULT (datetime('now')));
        """)
        self.db.commit()

    def push(self, role: str, text: str):
        self.short.append({"role": role, "text": text,
                           "ts": datetime.datetime.now().isoformat()})

    def recent(self, n=8) -> list:
        return list(self.short)[-n:]

    def context_window(self) -> list:
        return list(self.short)

    def set_pref(self, key, value):
        self.db.execute(
            "INSERT OR REPLACE INTO user_preferences(key,value,updated_at) VALUES(?,?,datetime('now'))",
            (key, value))
        self.db.commit()

    def get_pref(self, key, default=""):
        row = self.db.execute(
            "SELECT value FROM user_preferences WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def log_tool(self, name, args, result):
        self.db.execute(
            "INSERT INTO tool_history(tool_name,args,result) VALUES(?,?,?)",
            (name, json.dumps(args, ensure_ascii=False), result[:600]))
        self.db.commit()

    def last_tools(self, n=5):
        return [dict(r) for r in
                self.db.execute(
                    "SELECT tool_name,args,ts FROM tool_history ORDER BY id DESC LIMIT ?",
                    (n,)).fetchall()]

    def store_fact(self, topic, content, source=""):
        self.db.execute(
            "INSERT INTO long_term(topic,content,source) VALUES(?,?,?)",
            (topic, content[:1000], source))
        self.db.commit()

    def add_lexicon(self, term, definition, source="user"):
        term = term.strip()
        if not term or not definition:
            return False
        self.db.execute(
            "INSERT OR REPLACE INTO lexicon(term,definition,source) VALUES(?,?,?)",
            (term, definition[:2000], source))
        self.db.commit()
        return True

    def get_definition(self, term):
        if not term:
            return None
        row = self.db.execute(
            "SELECT definition FROM lexicon WHERE term=?", (term,)).fetchone()
        return row["definition"] if row else None

    def search_facts(self, query, limit=3):
        return [dict(r) for r in self.db.execute(
            "SELECT topic,content,source FROM long_term "
            "WHERE topic LIKE ? OR content LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit)).fetchall()]

    def get_user_name(self):
        return self.get_pref("user_name", "Efendim")

    def set_user_name(self, name):
        self.set_pref("user_name", name)

    def lm_history(self, n=4) -> str:
        lines = []
        msgs = list(self.short)[-n * 2:]
        for m in msgs:
            role = "Kullanıcı" if m["role"] == "user" else "Dame"
            lines.append(f"{role}: {m['text'][:120]}")
        return "\n".join(lines)


_TOOL_TAG_RE = re.compile(
    r"\[TOOL:(\w+)((?:\s+[\w]+=[^\]]*?)*)\]",
    re.IGNORECASE
)
_KV_RE = re.compile(r'([\w]+)=([^\s\]]+(?:\s+[^\s=\]]+)*?)(?=\s+\w+=|\s*\]|$)')


@dataclass
class ParsedToolCall:
    tool_name: str
    args: Dict[str, str]
    raw_tag: str


def parse_tool_calls(text: str) -> Tuple[List[ParsedToolCall], str]:
    calls = []
    cleaned = text

    for m in _TOOL_TAG_RE.finditer(text):
        tool_name = m.group(1).lower()
        kv_str = m.group(2).strip()
        args: Dict[str, str] = {}

        for kv in _KV_RE.finditer(kv_str):
            args[kv.group(1)] = kv.group(2).strip()

        calls.append(ParsedToolCall(tool_name=tool_name,
                                    args=args,
                                    raw_tag=m.group(0)))
        cleaned = cleaned.replace(m.group(0), "").strip()

    return calls, cleaned


_TOOL_DOCS = """
Kullanılabilir araçlar (gerektiğinde etiketle çağır):
  [TOOL:get_time]                          — Saat sormak için
  [TOOL:get_date]                          — Tarih sormak için
  [TOOL:weather city=ŞEHİR]               — Hava durumu
  [TOOL:currency from=USD to=TRY]         — Döviz kuru
  [TOOL:news topic=KONU]                  — Haberler
  [TOOL:calculate expression=İFADE]       — Matematik
  [TOOL:system_info]                       — Sistem bilgisi
  [TOOL:battery]                           — Pil durumu
  [TOOL:web_search query=SORGU]           — Web'de ara
  [TOOL:open_url url=URL]                 — URL aç
  [TOOL:open_app app=UYGULAMA]            — Uygulama aç
  [TOOL:screenshot]                        — Ekran görüntüsü
  [TOOL:joke]                              — Şaka anlat
  [TOOL:translate text=METİN]             — Çeviri
  [TOOL:remember_name name=İSİM]          — İsim kaydet
  [TOOL:teach term=KELİME definition=TANIM] — Tanım öğren
  [TOOL:define term=KELİME]               — Tanım sorgula
  [TOOL:shutdown]                          — Bilgisayarı kapat (onay gerekir)
  [TOOL:restart]                           — Yeniden başlat (onay gerekir)
  [TOOL:sleep]                             — Uyku modu (onay gerekir)
"""

_SYSTEM_BASE = """Sen Dame, akıllı bir masaüstü asistanısın.
Türkçe ve İngilizce sorulara kısa, doğru, yardımcı yanıtlar verirsin.
Bilmediğin şeyleri uydurmaz, araçları doğru şekilde kullanırsın.
Yanıtların açık, sade ve anlaşılır olsun.
"""


class PromptBuilder:
    def __init__(self, cfg: Config, mem: Memory):
        self.cfg = cfg
        self.mem = mem

    def build(self, user_text: str, wiki_context: str = "") -> str:
        u = self.mem.get_user_name()
        history = self.mem.lm_history(n=3)

        parts = [_SYSTEM_BASE.strip()]
        parts.append(f"Kullanıcı adı: {u}")
        parts.append(_TOOL_DOCS.strip())

        if wiki_context:
            parts.append(
                f"\n--- Bilgi tabanından ilgili bağlam ---\n{wiki_context}\n---")

        if history:
            parts.append(f"\n--- Son konuşma ---\n{history}\n---")

        parts.append(f"\nKullanıcı: {user_text}")
        parts.append("Dame:")

        return "\n".join(parts)

    def build_tool_result(self, user_text: str,
                          tool_name: str, result: str) -> str:
        u = self.mem.get_user_name()
        return (
            f"{_SYSTEM_BASE.strip()}\n"
            f"Kullanıcı adı: {u}\n\n"
            f"Kullanıcı: {user_text}\n"
            f"Araç ({tool_name}) sonucu:\n{result}\n\n"
            f"Bu sonucu {u}'e kısa, doğal bir cümleyle aktar:\nDame:"
        )


class NLPEngine:
    TR_STOPWORDS = {"bir", "bu", "şu", "da", "de", "ve", "ile", "için", "mi", "mu", "mı", "mü",
                    "ne", "ki", "ya", "daha", "çok", "az", "en", "hem", "ya da", "veya", "ama",
                    "ancak", "fakat", "ise", "bile", "kadar", "gibi", "göre", "karşı", "beri"}

    CITIES_TR = {
        "adana", "ankara", "istanbul", "izmir", "bursa", "antalya", "konya", "gaziantep",
        "şanlıurfa", "kocaeli", "mersin", "diyarbakır", "hatay", "manisa", "kayseri",
        "samsun", "balıkesir", "kahramanmaraş", "van", "aydın", "tekirdağ", "sakarya",
        "denizli", "muğla", "eskişehir", "trabzon", "erzurum", "malatya", "batman",
        "london", "paris", "berlin", "madrid", "rome", "tokyo", "beijing", "moscow",
    }
    CURRENCY_MAP = {
        "dolar": ("USD", "TRY"), "usd": ("USD", "TRY"), "$": ("USD", "TRY"),
        "euro": ("EUR", "TRY"), "eur": ("EUR", "TRY"), "€": ("EUR", "TRY"),
        "sterlin": ("GBP", "TRY"), "gbp": ("GBP", "TRY"),
        "japon": ("JPY", "TRY"), "jpy": ("JPY", "TRY"),
    }
    APP_MAP = {
        "tarayıcı": "browser", "browser": "browser", "chrome": "browser",
        "firefox": "browser", "edge": "browser",
        "not defteri": "notepad", "notepad": "notepad",
        "hesap makinesi": "calculator", "calculator": "calculator",
        "görev yöneticisi": "taskmgr", "vscode": "vscode",
        "cmd": "cmd", "komut satırı": "cmd", "powershell": "powershell",
        "explorer": "explorer", "dosya gezgini": "explorer",
        "paint": "paint", "word": "word", "excel": "excel",
    }
    FOLDER_MAP = {
        "masaüstü": "desktop", "desktop": "desktop",
        "indirmeler": "downloads", "downloads": "downloads",
        "belgeler": "documents", "documents": "documents",
        "resimler": "pictures", "müzik": "music", "videolar": "videos",
    }
    VOLUME_MAP = {
        "sesi aç": "up", "sesi yükselt": "up", "ses artır": "up", "volume up": "up",
        "sesi kıs": "down", "sesi azalt": "down", "volume down": "down",
        "sessiz": "mute", "mute": "mute", "ses kapat": "mute",
    }
    NEWS_TOPICS = {
        "spor": "spor haberleri", "futbol": "futbol haberleri",
        "teknoloji": "teknoloji", "ekonomi": "ekonomi", "borsa": "borsa piyasaları",
        "siyaset": "siyaset", "dünya": "dünya haberleri",
        "bilim": "bilim", "uzay": "uzay haberleri", "sağlık": "sağlık",
    }
    INTENT_PATTERNS = {
        "get_time": (["saat", "zaman", "kaçta"], [], [], 8.0),
        "get_date": (["tarih", "bugün", "hangi gün"], [], ["hava"], 8.0),
        "system_info": (["sistem", "cpu", "ram", "bellek", "disk", "performans"], [], [], 8.5),
        "battery": (["pil", "batarya", "şarj"], [], [], 8.5),
        "weather": (["hava", "sıcaklık", "yağmur", "derece", "hava durumu"], [], [], 9.0),
        "currency": (["dolar", "euro", "sterlin", "kur", "döviz", "kaç lira"], [], [], 9.0),
        "news": (["haber", "son dakika", "gündem", "manşet"], [], [], 8.0),
        "screenshot": (["ekran görüntüsü", "screenshot"], [], [], 9.0),
        "joke": (["şaka", "fıkra", "komik", "güldür"], [], [], 9.0),
        "music": (["müzik", "şarkı", "müzik aç"], [], [], 7.5),
        "shutdown": (["bilgisayarı kapat", "pc kapat", "shutdown"], [], [], 10.0),
        "restart": (["yeniden başlat", "restart", "reboot"], [], [], 10.0),
        "sleep": (["uyku modu", "bilgisayarı uyut", "sleep"], [], [], 9.5),
        "lock": (["kilitle", "ekranı kilitle", "lock"], [], [], 9.0),
        "empty_trash": (["çöp", "geri dönüşüm"], [], [], 9.0),
        "disk_clean": (["geçici dosya", "disk temizle", "temp temizle"], [], [], 9.0),
        "wikipedia": (["nedir", "kimdir", "anlat", "açıkla", "bilgi ver"],
                      [], ["hava", "kur", "dolar", "saat"], 5.0),
        "web_search": (["ara ", "google ", "search ", "bul "], [], [], 8.5),
        "open_url": (["https://", "http://", "www."], [], [], 10.0),
        "calculate": (["hesapla", "kaç eder", "toplam", "çarp", "böl"], [], [], 7.0),
        "remember_name": (["adım ", "ismim ", "beni çağır "], [], [], 9.5),
        "greeting": (["merhaba", "selam", "hey", "hello", "günaydın"], [], [], 9.0),
        "how_are_you": (["nasılsın", "nasıl gidiyor", "ne haber"], [], [], 9.0),
        "identity": (["kimsin", "nesin", "dame nedir", "dame kimdir"], [], [], 9.0),
        "help": (["yardım", "help", "ne yapabilirsin", "özellikler"], [], [], 9.0),
        "uptime": (["çalışma süresi", "uptime", "ne zamandır"], [], [], 8.5),
        "thanks": (["teşekkür", "sağ ol", "eyvallah", "harika", "bravo"], [], [], 7.0),
        "goodbye": (["güle güle", "hoşça kal", "bye", "görüşürüz", "çıkış"], [], [], 9.5),
        "translate": (["çevir", "translate", "ingilizce", "türkçe"], [], [], 8.5),
        "define": (["anlamı", "ne demek", "tanım"], [], [], 9.5),
        "teach": (["öğren:", "öğren", "öğret"], [], [], 10.0),
    }

    def __init__(self):
        pass

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[?!.,;:\"'(){}\[\]]", " ", text)
        return re.sub(r"\s+", " ", text)

    def tokenize(self, text: str) -> list:
        return [t for t in self.normalize(text).split() if t not in self.TR_STOPWORDS]

    def extract_entities(self, tl: str, original: str) -> dict:
        ents = {}
        m = re.search(r"(https?://\S+|www\.\S+\.\S+)", tl)
        if m:
            ents["url"] = m.group(1) if m.group(1).startswith("http") else "https://" + m.group(1)
        m = re.search(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/\%\^])\s*(\d+(?:\.\d+)?)", tl)
        if m:
            ents["math_expr"] = m.group(0).replace("^", "**")
        for city in self.CITIES_TR:
            if city in tl:
                ents["city"] = city.title()
                break
        for kw, (fr, to) in self.CURRENCY_MAP.items():
            if kw in tl:
                ents.setdefault("currency_from", fr)
                ents.setdefault("currency_to", to)
        for kw in sorted(self.APP_MAP, key=len, reverse=True):
            if kw in tl:
                ents["app"] = self.APP_MAP[kw]
                break
        for kw in sorted(self.FOLDER_MAP, key=len, reverse=True):
            if kw in tl:
                ents["folder"] = self.FOLDER_MAP[kw]
                break
        for kw, action in sorted(self.VOLUME_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if kw in tl:
                ents["volume_action"] = action
                break
        for kw, topic in self.NEWS_TOPICS.items():
            if kw in tl:
                ents["news_topic"] = topic
                break
        m = re.search(r"(?:adım|ismim|bana de|beni çağır)\s+([a-zA-ZçşğüöıÇŞĞÜÖİ]+)", tl)
        if m:
            ents["name"] = m.group(1).capitalize()
        m = re.match(r"^(?:ara|google|search|bul|araştır)\s+(.+)", tl)
        if m:
            ents["search_query"] = m.group(1).strip()
        wiki_q = re.sub(r"\b(nedir|ne demek|anlat|açıkla|bilgi ver|kimdir)\b", "", tl).strip()
        if wiki_q:
            ents["wiki_query"] = wiki_q
        m = re.search(r"(?:çevir|translate)\s+(.+?)(?:\s+(?:ingilizce|türkçe)(?:ye|ya|e|a)?)?$", tl)
        if m:
            ents["translate_text"] = m.group(1).strip()
        orig = original.strip()
        for pat in [r"(?i)öğren\s*[:]?\s*([^=:\n]+)\s*[=:\-]+\s*(.+)$",
                    r"(?i)öğren\s+([^\s=:\n]+)\s+(.+)$"]:
            mm = re.search(pat, orig)
            if mm:
                ents["teach_term"] = mm.group(1).strip()
                ents["teach_definition"] = mm.group(2).strip()
                break
        m3 = re.search(r"([a-zığüşöç0-9\-]+)\s+(?:anlamı|ne demek)\b", tl)
        if m3:
            ents["term"] = m3.group(1).strip()
        nums = re.findall(r"\d+(?:\.\d+)?", tl)
        if nums:
            ents["numbers"] = [float(n) if "." in n else int(n) for n in nums]
        return ents

    def detect_intent(self, text: str, context: list) -> tuple:
        tl = self.normalize(text)
        tokens = self.tokenize(text)
        ents = self.extract_entities(tl, text)

        if ents.get("teach_term") and ents.get("teach_definition"):
            return "teach", 0.99, ents
        if ents.get("term"):
            return "define", 0.90, ents

        scores = defaultdict(float)
        for intent, (kwlist, req, neg, base) in self.INTENT_PATTERNS.items():
            for kw in kwlist:
                if kw in tl:
                    scores[intent] += base * len(kw.split()) * 1.2
            for nw in neg:
                if nw in tl:
                    scores[intent] -= base * 2

        if re.search(r"\d+\s*[\+\-\*\/\%\^]\s*\d+", tl):
            scores["calculate"] += 15
        if re.search(r"https?://|www\.", tl):
            scores["open_url"] += 20
        if re.search(r"(?:adım|ismim)\s+\w+", tl):
            scores["remember_name"] += 20
        if any(q in tl for q in ["nedir", "kimdir", "anlat"]):
            scores["wikipedia"] += 4

        if not scores:
            return "unknown", 0.0, ents
        best = max(scores, key=scores.get)
        total = sum(abs(v) for v in scores.values()) or 1
        conf = min(scores[best] / total * 2.5, 1.0)
        if scores[best] < 3.0:
            return "unknown", conf, ents
        return best, conf, ents


class ResponseGenerator:
    TEMPLATES = {
        "get_time": ["Şu an {output}"],
        "get_date": ["{output}"],
        "system_info": ["Sistem durumu:\n{output}"],
        "battery": ["Pil: {output}"],
        "calculate": ["Sonuç: {output}"],
        "joke": ["{output}"],
        "screenshot": ["Ekran görüntüsü alındı: {output}"],
        "weather": ["{output}"],
        "currency": ["Kur: {output}"],
        "news": ["{output}"],
        "wikipedia": ["Bulduklarım:\n{output}"],
        "duckduckgo": ["Web'den:\n{output}"],
        "network": ["{output}"],
        "processes": ["{output}"],
        "volume": ["{output}"],
        "music": ["{output}"],
        "remember_name": ["{output}"],
        "open_app": ["{output}"],
        "open_folder": ["{output}"],
        "open_url": ["{output}"],
        "web_search": ["{output}"],
        "translate": ["{output}"],
        "define": ["{output}"],
        "teach": ["Kaydettim: {output}"],
        "lock": ["{output}"],
        "empty_trash": ["{output}"],
        "disk_clean": ["{output}"],
    }
    CHAT_RESPONSES = {
        "greeting": {
            "morning": ["Günaydın {u}! Dame v6.0 hazır."],
            "afternoon": ["Merhaba {u}! Emrinizdeyim."],
            "evening": ["İyi akşamlar {u}!"],
        },
        "how_are_you": ["Harika durumdayım {u}! Size nasıl yardımcı olabilirim?"],
        "identity": [
            "Ben Dame v6.0 — {u}. Sıfırdan eğitilmiş bir transformer dil modeli, "
            "Wikipedia bilgi tabanı ve 29 araçla donanmış offline masaüstü asistanınım."
        ],
        "thanks": ["Ne demek {u}, her zaman emrinizdeyim!", "Rica ederim {u}!"],
        "goodbye": ["Güle gidin {u}! Görüşürüz."],
        "clipboard": ["Pano için Ctrl+C / Ctrl+V kullanabilirsiniz {u}."],
    }

    def __init__(self, mem: Memory):
        self.mem = mem

    def build(self, tool_name: str, raw: str, intent: str, ents: dict) -> str:
        u = self.mem.get_user_name()
        tpls = self.TEMPLATES.get(tool_name) or self.TEMPLATES.get(intent, ["{output}"])
        return random.choice(tpls).replace("{output}", raw).replace("{u}", u)

    def build_confirm(self, tool_name: str, u: str) -> str:
        msgs = {
            "shutdown": f"⚠  Bilgisayarı kapatmak istediğinizden emin misiniz {u}? (evet/hayır)",
            "restart": f"⚠  Yeniden başlatmayı onaylıyor musunuz {u}? (evet/hayır)",
            "sleep": f"⚠  Uyku moduna geçmek istiyor musunuz {u}? (evet/hayır)",
            "empty_trash": f"⚠  Geri dönüşüm kutusunu boşaltmak istiyor musunuz {u}? (evet/hayır)",
            "disk_clean": f"⚠  Geçici dosyaları silmek istiyor musunuz {u}? (evet/hayır)",
        }
        return msgs.get(tool_name, f"⚠  Bu işlemi onaylıyor musunuz {u}? (evet/hayır)")

    def build_chat(self, intent: str, ents: dict, session_start: datetime.datetime) -> str:
        u = self.mem.get_user_name()
        sm = int((datetime.datetime.now() - session_start).total_seconds() // 60)
        if intent == "greeting":
            h = datetime.datetime.now().hour
            k = "morning" if h < 12 else ("afternoon" if h < 18 else "evening")
            return random.choice(self.CHAT_RESPONSES["greeting"][k]).replace("{u}", u)
        if intent == "help":
            return self._help_text(u)
        if intent == "uptime":
            n = sum(1 for m in self.mem.short if m["role"] == "user")
            return f"Bu oturumda {sm} dakikadır çalışıyorum {u}. {n} mesajınızı işledim."
        if intent == "session_info":
            msgs = [m["text"][:40] for m in list(self.mem.short) if m["role"] == "user"][-3:]
            return f"Son mesajlarınız {u}: " + (" · ".join(msgs) if msgs else "Henüz yok.")
        if intent == "goodbye":
            return random.choice(self.CHAT_RESPONSES["goodbye"]).replace("{u}", u)
        pool = self.CHAT_RESPONSES.get(intent, [])
        if pool:
            return random.choice(pool).replace("{u}", u)
        return f"Anladım {u}."

    def _help_text(self, u: str) -> str:
        lm_status = "✓ Aktif" if LM_AVAILABLE else "✗ Henüz eğitilmemiş"
        wk_status = "✓ Aktif" if WIKI_AVAILABLE else "✗ DB oluşturulmamış"
        return (
            f"┌── Dame v6.0 Komut Kılavuzu ────────────────────────────────\n"
            f"│  LM Motoru   : {lm_status}\n"
            f"│  Wiki DB     : {wk_status}\n"
            f"│  ─────────────────────────────────────────────────────────\n"
            f"│  Soru & Bilgi  ▸  Her türlü soru (LM aktifse doğal cevap)\n"
            f"│  Zaman         ▸  'saat kaç?' / 'bugün ne?'\n"
            f"│  Sistem        ▸  'sistem durumu' / 'pil durumu'\n"
            f"│  Matematik     ▸  '125 * 48' / '2^10'\n"
            f"│  Hava          ▸  'hava durumu' / 'adana hava'\n"
            f"│  Döviz         ▸  'dolar kuru' / 'euro kaç TL'\n"
            f"│  Haberler      ▸  'son haberler' / 'teknoloji haberleri'\n"
            f"│  Wikipedia     ▸  'yapay zeka nedir'\n"
            f"│  Uygulama      ▸  'tarayıcı aç' / 'vscode aç'\n"
            f"│  Ses           ▸  'sesi aç' / 'sessiz'\n"
            f"│  Tanım Öğren   ▸  'öğren: kelime = anlam'\n"
            f"│  Kapat         ▸  'bilgisayarı kapat' / 'yeniden başlat'\n"
            f"└──────────────────────────────────────────────────────────────"
        )

    def build_unknown(self, text: str) -> str:
        u = self.mem.get_user_name()
        facts = self.mem.search_facts(text[:30])
        if facts:
            return f"Hafızamda buldum {u}:\n{facts[0]['content']}"
        return random.choice([
            f"'{text[:35]}' ifadesini anlayamadım {u}. 'Yardım' yazın.",
            f"Bu isteği işleyemedim {u}. 'ara {text[:25]}' yazabilirsiniz.",
        ])


@dataclass
class ToolResult:
    success: bool
    output: str
    speak: str = ""

    def __str__(self):
        return self.output


class Tool(ABC):
    name: str = ""
    description: str = ""
    risk_level: str = "safe"

    @abstractmethod
    def run(self, args: dict, mem: Memory) -> ToolResult:
        ...


class ToolTime(Tool):
    name = "get_time"

    def run(self, a, m):
        n = datetime.datetime.now()
        o = f"saat {n.strftime('%H:%M:%S')}"
        return ToolResult(True, o, o)


class ToolDate(Tool):
    name = "get_date"

    def run(self, a, m):
        n = datetime.datetime.now()
        D = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
        M = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım",
             "Aralık"]
        o = f"{D[n.weekday()]}, {n.day} {M[n.month - 1]} {n.year}"
        return ToolResult(True, o, o)


class ToolSystemInfo(Tool):
    name = "system_info"

    def run(self, a, m):
        if not PSUTIL_OK:
            return ToolResult(False, "psutil yüklü değil")
        try:
            cpu = psutil.cpu_percent(interval=0.4)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            def bar(p, w=12):
                f = int(p / 100 * w)
                return "█" * f + "░" * (w - f)

            lines = [
                f"┌── Sistem {datetime.datetime.now().strftime('%H:%M:%S')} ──────────────────────",
                f"│  CPU  {cpu:5.1f}%  [{bar(cpu)}]",
                f"│  RAM  {ram.percent:5.1f}%  [{bar(ram.percent)}]  {ram.used >> 20:,}MB/{ram.total >> 20:,}MB",
                f"│  DISK {disk.percent:5.1f}%  [{bar(disk.percent)}]  {disk.used >> 30:.1f}GB/{disk.total >> 30:.1f}GB",
                f"│  OS   {platform.system()} {platform.release()}",
                f"└────────────────────────────────────────────"]
            return ToolResult(True, "\n".join(lines), f"CPU %{cpu:.0f}, RAM %{ram.percent:.0f}")
        except Exception as e:
            return ToolResult(False, f"Hata: {e}")


class ToolBattery(Tool):
    name = "battery"

    def run(self, a, m):
        if not PSUTIL_OK:
            return ToolResult(False, "psutil yüklü değil")
        try:
            b = psutil.sensors_battery()
            if not b:
                return ToolResult(False, "Pil sensörü yok.")
            st = "Şarj Oluyor" if b.power_plugged else "Pille Çalışıyor"
            bar = "█" * int(b.percent / 10) + "░" * (10 - int(b.percent / 10))
            o = f"[{bar}] %{b.percent:.0f}  {st}"
            return ToolResult(True, o, f"Pil %{b.percent:.0f}")
        except Exception as e:
            return ToolResult(False, f"Hata: {e}")


class ToolCalculate(Tool):
    name = "calculate"

    def run(self, args, m):
        expr = args.get("expression", "")
        if not expr:
            return ToolResult(False, "İfade boş.")
        try:
            tree = ast.parse(expr, mode="eval")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    return ToolResult(False, "Fonksiyon çağrısı yasak.")
            result = eval(compile(tree, "<e>", "eval"), {"__builtins__": {}},
                          {"abs": abs, "round": round, "max": max, "min": min,
                           "pow": pow, "sqrt": math.sqrt, "int": int, "float": float})
            if isinstance(result, float) and result == int(result):
                result = int(result)
            o = f"{expr.replace('**', '^')} = {result}"
            return ToolResult(True, o, f"Sonuç {result}")
        except ZeroDivisionError:
            return ToolResult(False, "Sıfıra bölme!")
        except Exception:
            return ToolResult(False, f"Hesaplanamadı: {expr}")


class ToolOpenApp(Tool):
    name = "open_app"
    MAP = {
        "browser": (lambda: webbrowser.open("https://www.google.com"), "Tarayıcı açılıyor."),
        "notepad": (lambda: subprocess.Popen("notepad.exe"), "Not defteri açılıyor."),
        "calculator": (lambda: subprocess.Popen("calc.exe"), "Hesap makinesi açılıyor."),
        "taskmgr": (lambda: subprocess.Popen("taskmgr.exe"), "Görev yöneticisi açılıyor."),
        "vscode": (lambda: subprocess.Popen(["code"]), "VS Code açılıyor."),
        "cmd": (lambda: subprocess.Popen("cmd"), "Komut satırı açılıyor."),
        "powershell": (lambda: subprocess.Popen("powershell"), "PowerShell açılıyor."),
        "explorer": (lambda: subprocess.Popen("explorer.exe"), "Dosya gezgini açılıyor."),
        "paint": (lambda: subprocess.Popen("mspaint.exe"), "Paint açılıyor."),
    }

    def run(self, args, m):
        app = args.get("app", "browser")
        e = self.MAP.get(app)
        if not e:
            return ToolResult(False, f"Bilinmeyen uygulama: {app}")
        fn, msg = e
        try:
            fn()
            return ToolResult(True, msg, msg)
        except Exception as ex:
            return ToolResult(False, f"{app} açılamadı: {ex}")


class ToolOpenFolder(Tool):
    name = "open_folder"
    FMAP = {"desktop": "Desktop", "downloads": "Downloads", "documents": "Documents",
            "pictures": "Pictures", "music": "Music", "videos": "Videos"}

    def run(self, args, m):
        fn = self.FMAP.get(args.get("folder", "home"))
        path = str(Path.home() / fn) if fn else str(Path.home())
        try:
            if platform.system() == "Windows":
                subprocess.Popen(f'explorer "{path}"')
            else:
                subprocess.Popen(["xdg-open", path])
            return ToolResult(True, path, f"{fn or 'Ana klasör'} açıldı.")
        except Exception as e:
            return ToolResult(False, f"Hata: {e}")


class ToolOpenURL(Tool):
    name = "open_url"

    def run(self, args, m):
        url = args.get("url", "")
        if not url:
            return ToolResult(False, "URL belirtilmedi.")
        if not url.startswith("http"):
            url = "https://" + url
        webbrowser.open(url)
        return ToolResult(True, f"Açılıyor: {url}", "Açılıyor.")


class ToolScreenshot(Tool):
    name = "screenshot"

    def run(self, a, m):
        if not PYAUTOGUI_OK:
            return ToolResult(False, "pip install pyautogui pillow")
        try:
            img = pyautogui.screenshot()
            path = Path.home() / "Pictures" / f"dame_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
            path.parent.mkdir(exist_ok=True)
            img.save(str(path))
            return ToolResult(True, str(path), "Ekran görüntüsü alındı.")
        except Exception as e:
            return ToolResult(False, f"Hata: {e}")


class ToolProcesses(Tool):
    name = "processes"

    def run(self, a, m):
        if not PSUTIL_OK:
            return ToolResult(False, "psutil yüklü değil")
        procs = sorted(
            [p.info for p in psutil.process_iter(["name", "cpu_percent", "memory_percent"])],
            key=lambda x: x.get("cpu_percent") or 0, reverse=True)
        lines = ["┌── En Aktif Süreçler ──────────────────────"]
        for p in procs[:10]:
            n = (p.get("name") or "?")[:28]
            lines.append(f"│  {n:<28} CPU:{p.get('cpu_percent', 0):5.1f}%")
        lines.append("└──────────────────────────────────────────")
        return ToolResult(True, "\n".join(lines), "Süreç listesi hazır.")


class ToolNetwork(Tool):
    name = "network"

    def run(self, a, m):
        try:
            h = socket.gethostname()
            ip = socket.gethostbyname(h)
            return ToolResult(True, f"Bilgisayar: {h}\nYerel IP: {ip}", f"IP: {ip}")
        except Exception as e:
            return ToolResult(False, f"Hata: {e}")


class ToolVolume(Tool):
    name = "volume"

    def run(self, args, m):
        action = args.get("action", "up")
        try:
            if platform.system() == "Windows" and PYAUTOGUI_OK:
                key = {"up": "volumeup", "down": "volumedown", "mute": "volumemute"}.get(action, "volumeup")
                pyautogui.press(key, presses=5 if action in ("up", "down") else 1)
            elif platform.system() == "Linux":
                cmd = {"up": "10%+", "down": "10%-", "mute": "toggle"}.get(action, "10%+")
                subprocess.Popen(["amixer", "-q", "sset", "Master", cmd])
            msg = {"up": "Ses yükseltildi.", "down": "Ses kısıldı.", "mute": "Ses kapatıldı."}.get(action, "Tamam.")
            return ToolResult(True, msg, msg)
        except:
            return ToolResult(False, "Ses kontrolü çalışmıyor.")


class ToolMusic(Tool):
    name = "music"

    def run(self, a, m):
        webbrowser.open("https://music.youtube.com")
        return ToolResult(True, "YouTube Music açılıyor.", "Müzik açılıyor.")


class ToolWeather(Tool):
    name = "weather"

    def run(self, args, m):
        city = args.get("city", "Istanbul")
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        try:
            req = urllib.request.Request(url, headers=Config.HEADERS)
            with urllib.request.urlopen(req, timeout=Config.SCRAPER_TIMEOUT) as r:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
            cur = data["current_condition"][0]
            dl = cur.get("lang_tr") or cur.get("weatherDesc", [{}])
            desc = dl[0].get("value", "") if dl else ""
            temp = cur.get("temp_C", "?")
            hum = cur.get("humidity", "?")
            area = data.get("nearest_area", [{}])[0].get("areaName", [{}])[0].get("value", city)
            o = f"{area}: {desc}, {temp}°C, nem %{hum}"
            return ToolResult(True, o, o)
        except Exception as e:
            return ToolResult(False, f"Hava alınamadı: {e}")


class ToolCurrency(Tool):
    name = "currency"

    def run(self, args, m):
        fr = args.get("from", "USD").upper()
        to = args.get("to", "TRY").upper()
        for url in [f"https://api.exchangerate-api.com/v4/latest/{fr}",
                    f"https://open.er-api.com/v6/latest/{fr}"]:
            try:
                req = urllib.request.Request(url, headers=Config.HEADERS)
                with urllib.request.urlopen(req, timeout=Config.SCRAPER_TIMEOUT) as r:
                    data = json.loads(r.read().decode())
                rate = data.get("rates", {}).get(to)
                if rate:
                    o = f"1 {fr} = {round(rate, 4)} {to}"
                    return ToolResult(True, o, o)
            except:
                pass
        return ToolResult(False, f"Kur alınamadı: {fr}/{to}")


class ToolNews(Tool):
    name = "news"

    def run(self, args, m):
        topic = args.get("topic", "son dakika")
        q = urllib.parse.quote_plus(topic)
        url = f"https://news.google.com/rss/search?q={q}&hl=tr&gl=TR&ceid=TR:tr"
        try:
            req = urllib.request.Request(url, headers=Config.HEADERS)
            with urllib.request.urlopen(req, timeout=Config.SCRAPER_TIMEOUT) as r:
                raw = r.read().decode("utf-8", errors="replace")
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", raw)
            if not titles:
                titles = re.findall(r"<title>(.*?)</title>", raw)
            titles = [t for t in titles if t and "Google" not in t][:6]
            if titles:
                lines = [f"Haberler — {topic}:"]
                for i, t in enumerate(titles, 1):
                    lines.append(f"  {i}. {html.unescape(t)}")
                return ToolResult(True, "\n".join(lines), f"{len(titles)} haber.")
            return ToolResult(False, "Haber bulunamadı.")
        except Exception as e:
            return ToolResult(False, f"Haber hatası: {e}")


class ToolWebSearch(Tool):
    name = "web_search"

    def run(self, args, m):
        q = args.get("query", "")
        if not q:
            return ToolResult(False, "Arama terimi yok.")
        webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote_plus(q)}")
        return ToolResult(True, f"'{q}' aranıyor.", f"{q} aranıyor.")


class ToolWikipedia(Tool):
    name = "wikipedia"

    def _get(self, url):
        try:
            req = urllib.request.Request(url, headers=Config.HEADERS)
            with urllib.request.urlopen(req, timeout=Config.SCRAPER_TIMEOUT) as r:
                return r.read().decode("utf-8", errors="replace")
        except:
            return ""

    def run(self, args, m):
        query = args.get("query", "")
        if not query:
            return ToolResult(False, "Konu belirtilmedi.")
        if WIKI_AVAILABLE:
            try:
                s = WIKI_RETRIEVER.get_summary(query, max_chars=600)
                if s:
                    m.store_fact(query, s, "wiki_db")
                    return ToolResult(True, s, s[:150])
            except:
                pass

        for lang in ["tr", "en"]:
            q = urllib.parse.quote_plus(query)
            raw = self._get(
                f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit=1")
            if not raw:
                continue
            try:
                hits = json.loads(raw).get("query", {}).get("search", [])
                if not hits:
                    continue
                title = urllib.parse.quote(hits[0]["title"].replace(" ", "_"))
                raw2 = self._get(
                    f"https://{lang}.wikipedia.org/w/api.php?action=query&titles={title}&prop=extracts&exintro=true&explaintext=true&format=json")
                for page in json.loads(raw2).get("query", {}).get("pages", {}).values():
                    ex = page.get("extract", "")
                    if ex:
                        sents = re.split(r"(?<=[.!?])\s+", ex.strip())
                        result = " ".join(sents[:5])
                        m.store_fact(query, result, "wikipedia")
                        return ToolResult(True, result, result[:150])
            except:
                pass
        return ToolResult(False, f"'{query}' için sonuç yok.")


class ToolShutdown(Tool):
    name = "shutdown"
    risk_level = "confirm"

    def run(self, a, m):
        os.system("shutdown /s /t 5" if platform.system() == "Windows" else "shutdown -h now")
        return ToolResult(True, f"Sistem kapanıyor. Güle gidin {m.get_user_name()}.", "Kapatılıyor.")


class ToolRestart(Tool):
    name = "restart"
    risk_level = "confirm"

    def run(self, a, m):
        os.system("shutdown /r /t 5" if platform.system() == "Windows" else "reboot")
        return ToolResult(True, "Sistem yeniden başlatılıyor.", "Yeniden başlatılıyor.")


class ToolSleep(Tool):
    name = "sleep"
    risk_level = "confirm"

    def run(self, a, m):
        if platform.system() == "Windows":
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        else:
            os.system("systemctl suspend")
        return ToolResult(True, "Uyku moduna geçildi.", "Uyku modu.")


class ToolLock(Tool):
    name = "lock"

    def run(self, a, m):
        if platform.system() == "Windows":
            ctypes.windll.user32.LockWorkStation()
        elif platform.system() == "Linux":
            os.system("gnome-screensaver-command -l")
        return ToolResult(True, "Ekran kilitleniyor.", "Kilitlendi.")


class ToolEmptyTrash(Tool):
    name = "empty_trash"
    risk_level = "confirm"

    def run(self, a, m):
        try:
            if platform.system() == "Windows":
                ctypes.windll.shell32.SHEmptyRecycleBinW(None, None, 1)
                return ToolResult(True, "Geri dönüşüm kutusu boşaltıldı.", "Boşaltıldı.")
            trash = Path.home() / ".local/share/Trash/files"
            if trash.exists():
                shutil.rmtree(str(trash))
                trash.mkdir()
            return ToolResult(True, "Çöp temizlendi.", "Temizlendi.")
        except Exception as e:
            return ToolResult(False, f"Hata: {e}")


class ToolDiskClean(Tool):
    name = "disk_clean"
    risk_level = "confirm"

    def run(self, a, m):
        if platform.system() != "Windows":
            return ToolResult(False, "Sadece Windows.")
        cleaned = 0
        for tp in [Path(os.environ.get("TEMP", "")), Path("C:/Windows/Temp")]:
            if tp.exists():
                for f in tp.iterdir():
                    try:
                        if f.is_file():
                            f.unlink()
                            cleaned += 1
                    except:
                        pass
        return ToolResult(True, f"{cleaned} geçici dosya silindi.", f"{cleaned} dosya.")


class ToolRememberName(Tool):
    name = "remember_name"

    def run(self, args, m):
        name = args.get("name", "").strip().capitalize()
        if not name:
            return ToolResult(False, "İsim belirtilmedi.")
        m.set_user_name(name)
        return ToolResult(True, f"Artık sizi {name} olarak tanıyacağım!", f"Merhaba {name}!")


class ToolJoke(Tool):
    name = "joke"
    JOKES = [
        "Bir programcı alışverişe gider. Eşi: '10 ekmek al, yumurta varsa 6 tane.' Programcı 6 ekmek alır. Yumurta vardı.",
        "Stack Overflow çökseydi kaç yazılımcı işsiz kalırdı? Söylemeyin, saymaktan korktum.",
        "Git commit -m 'son düzeltme' — yazılım dünyasının en büyük yalanı.",
        "QA testi yapar. Bara 0, -1, 999, NULL bira ister. Müşteri girer ve bar yanar.",
        "DevOps cennete gelir. Kapıda yazar: 'Dağıtım başarısız. Log kontrolü gerekli.'",
    ]

    def run(self, a, m):
        j = random.choice(self.JOKES)
        return ToolResult(True, j, j)


class ToolTranslate(Tool):
    name = "translate"

    def run(self, args, m):
        text = args.get("text", "")
        if text:
            webbrowser.open(f"https://translate.google.com/?sl=auto&tl=en&text={urllib.parse.quote_plus(text)}")
            return ToolResult(True, f"'{text}' çeviriliyor.", "Çeviri açılıyor.")
        return ToolResult(False, "Metin belirtilmedi.")


class ToolTeach(Tool):
    name = "teach"

    def run(self, args, m):
        term = (args.get("term") or "").strip()
        defn = (args.get("definition") or "").strip()
        if not term or not defn:
            return ToolResult(False, "Örnek: öğren: kelime = anlam")
        ok = m.add_lexicon(term, defn, source="user")
        return ToolResult(True, f"{term} = {defn}", f"{term} kaydedildi.") if ok else ToolResult(False,
                                                                                                 "Kaydedilemedi.")


class ToolDefine(Tool):
    name = "define"

    def run(self, args, m):
        term = (args.get("term") or "").strip()
        if not term:
            return ToolResult(False, "Kelime belirtilmedi.")
        defn = m.get_definition(term)
        if defn:
            return ToolResult(True, f"{term}: {defn}", f"{term} bulundu.")
        res = ToolWikipedia().run({"query": term}, m)
        if res.success:
            return ToolResult(True, f"{term}: {res.output}", f"{term} bulundu.")
        return ToolResult(False, f"'{term}' için tanım bulunamadı. Öğretebilirsiniz: öğren: {term} = anlam")


class ToolRegistry:
    def __init__(self):
        self._t: dict = {}

    def register(self, t: Tool):
        self._t[t.name] = t

    def get(self, name) -> Optional[Tool]:
        return self._t.get(name)

    def names(self) -> list:
        return list(self._t.keys())


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for cls in [ToolTime, ToolDate, ToolSystemInfo, ToolBattery, ToolCalculate,
                ToolOpenApp, ToolOpenFolder, ToolOpenURL, ToolScreenshot,
                ToolProcesses, ToolNetwork, ToolVolume, ToolMusic,
                ToolWeather, ToolCurrency, ToolNews, ToolWebSearch,
                ToolWikipedia, ToolShutdown, ToolRestart, ToolSleep, ToolLock,
                ToolEmptyTrash, ToolDiskClean, ToolRememberName, ToolJoke,
                ToolTranslate, ToolTeach, ToolDefine]:
        reg.register(cls())
    return reg


CHAT_INTENTS = {"greeting", "how_are_you", "identity", "help", "uptime", "thanks",
                "goodbye", "session_info", "clipboard"}

INTENT_TO_TOOL = {
    "get_time": ("get_time", lambda e: {}),
    "get_date": ("get_date", lambda e: {}),
    "system_info": ("system_info", lambda e: {}),
    "battery": ("battery", lambda e: {}),
    "screenshot": ("screenshot", lambda e: {}),
    "processes": ("processes", lambda e: {}),
    "network": ("network", lambda e: {}),
    "joke": ("joke", lambda e: {}),
    "music": ("music", lambda e: {}),
    "shutdown": ("shutdown", lambda e: {}),
    "restart": ("restart", lambda e: {}),
    "sleep": ("sleep", lambda e: {}),
    "lock": ("lock", lambda e: {}),
    "empty_trash": ("empty_trash", lambda e: {}),
    "disk_clean": ("disk_clean", lambda e: {}),
    "weather": ("weather", lambda e: {"city": e.get("city", "Istanbul")}),
    "currency": ("currency", lambda e: {"from": e.get("currency_from", "USD"), "to": e.get("currency_to", "TRY")}),
    "news": ("news", lambda e: {"topic": e.get("news_topic", "son dakika")}),
    "calculate": ("calculate", lambda e: {"expression": e.get("math_expr", "")}),
    "open_app": ("open_app", lambda e: {"app": e.get("app", "browser")}),
    "open_folder": ("open_folder", lambda e: {"folder": e.get("folder", "home")}),
    "open_url": ("open_url", lambda e: {"url": e.get("url", "")}),
    "web_search": ("web_search", lambda e: {"query": e.get("search_query", "")}),
    "wikipedia": ("wikipedia", lambda e: {"query": e.get("wiki_query", "")}),
    "remember_name": ("remember_name", lambda e: {"name": e.get("name", "")}),
    "translate": ("translate", lambda e: {"text": e.get("translate_text", "")}),
    "volume": ("volume", lambda e: {"action": e.get("volume_action", "up")}),
    "define": ("define", lambda e: {"term": e.get("term", "")}),
    "teach": ("teach", lambda e: {"term": e.get("teach_term", ""), "definition": e.get("teach_definition", "")}),
}


class DameBrain:
    def __init__(self, gui_callback=None):
        self.cfg = Config()
        self.log = build_logger(self.cfg)
        self.mem = Memory(self.cfg, self.log)
        self.reg = build_registry()
        self.nlp = NLPEngine()
        self.respgen = ResponseGenerator(self.mem)
        self.prompt_builder = PromptBuilder(self.cfg, self.mem)
        self.gui_cb = gui_callback
        self._pending = None
        self.session_start = datetime.datetime.now()

        self.lm = DAME_LM
        self.wiki = WIKI_RETRIEVER

        mode = "LM+Wiki" if (LM_AVAILABLE and WIKI_AVAILABLE) else \
            "LM only" if LM_AVAILABLE else \
                "Wiki+NLP" if WIKI_AVAILABLE else "NLP only"
        self.log.info(f"DAME v{self.cfg.VERSION} başlatıldı — mod: {mode}")

    def _emit(self, event: str, data: str = ""):
        if self.gui_cb:
            self.gui_cb(event, data)

    def _handle_confirm(self, tl: str) -> Optional[str]:
        if not self._pending:
            return None
        pc = self._pending
        self._pending = None
        if re.search(r"\b(evet|tamam|onay|yes|ok|olur)\b", tl):
            tool = self.reg.get(pc["tool"])
            if tool:
                res = tool.run(pc["args"], self.mem)
                self.mem.log_tool(pc["tool"], pc["args"], res.output)
                return self.respgen.build(pc["tool"], res.output, pc["tool"], {})
        return f"İşlem iptal edildi {self.mem.get_user_name()}."

    def _run_tool(self, tool_name: str, args: dict, user_text: str,
                  skip_confirm: bool = False) -> str:
        tool = self.reg.get(tool_name)
        if not tool:
            return f"Bilinmeyen araç: {tool_name}"

        if not skip_confirm and tool.risk_level == "confirm":
            self._pending = {"tool": tool_name, "args": args}
            return self.respgen.build_confirm(tool_name, self.mem.get_user_name())

        self._emit("pipeline", f"Araç: {tool_name}")
        try:
            result = tool.run(args, self.mem)
            self.mem.log_tool(tool_name, args, result.output)

            if LM_AVAILABLE and result.success and tool_name not in {
                "system_info", "processes", "news", "wikipedia"}:
                try:
                    prompt = self.prompt_builder.build_tool_result(
                        user_text, tool_name, result.output)
                    lm_resp = self.lm.generate(
                        prompt,
                        max_new=80,
                        temperature=0.55,
                    )
                    if lm_resp and len(lm_resp) > 10:
                        return lm_resp
                except Exception as e:
                    self.log.debug(f"LM tool-result rewrite hatası: {e}")

            return result.output if result.success else result.output

        except Exception as e:
            self.log.error(f"Araç hatası ({tool_name}): {e}", exc_info=True)
            return f"Araç hatası: {e}"

    def process(self, text: str) -> str:
        self.log.info(f"Kullanıcı: {text}")
        self.mem.push("user", text)
        tl = self.nlp.normalize(text)

        cr = self._handle_confirm(tl)
        if cr is not None:
            self.mem.push("dame", cr)
            return cr

        nlp_obj = self.nlp
        for kw, action in nlp_obj.VOLUME_MAP.items():
            if kw in tl:
                resp = self._run_tool("volume", {"action": action}, text)
                self.mem.push("dame", resp)
                return resp

        self._emit("pipeline", "İşleniyor...")
        context = self.mem.context_window()

        if LM_AVAILABLE:
            self._emit("pipeline", "LM düşünüyor...")
            try:
                wiki_ctx = ""
                if WIKI_AVAILABLE:
                    try:
                        wiki_ctx = WIKI_RETRIEVER.get_summary(
                            text, max_chars=self.cfg.WIKI_CTX_CHARS)
                    except Exception as e:
                        self.log.debug(f"Wiki lookup hatası: {e}")

                prompt = self.prompt_builder.build(text, wiki_ctx)

                raw_lm = self.lm.generate(
                    prompt,
                    max_new=self.cfg.LM_MAX_NEW,
                    temperature=self.cfg.LM_TEMPERATURE,
                    top_k=self.cfg.LM_TOP_K,
                    top_p=self.cfg.LM_TOP_P,
                )

                tool_calls, clean_text = parse_tool_calls(raw_lm)

                if tool_calls:
                    tc = tool_calls[0]
                    self.log.info(f"LM tool call: {tc.tool_name}  args={tc.args}")
                    self._emit("pipeline", f"LM araç: {tc.tool_name}")
                    tool_resp = self._run_tool(tc.tool_name, tc.args, text)

                    if clean_text and len(clean_text) > 15:
                        resp = f"{clean_text}\n\n{tool_resp}"
                    else:
                        resp = tool_resp
                else:
                    resp = clean_text if clean_text else raw_lm

                if len(resp.strip()) < 8:
                    raise ValueError("LM yanıtı çok kısa — fallback")

                self.mem.push("dame", resp)
                return resp

            except Exception as e:
                self.log.warning(f"LM işleme hatası ({e}), NLP fallback'e geçiliyor")
                self._emit("pipeline", "NLP fallback...")

        intent, confidence, entities = self.nlp.detect_intent(text, context)
        self.log.info(f"[NLP fallback] intent={intent}  conf={confidence:.2f}")
        self._emit("pipeline", f"NLP: {intent} ({confidence:.0%})")

        if intent in CHAT_INTENTS:
            resp = self.respgen.build_chat(intent, entities, self.session_start)
            self.mem.push("dame", resp)
            if intent == "goodbye":
                self._emit("goodbye", "")
            return resp

        tool_entry = INTENT_TO_TOOL.get(intent)
        if not tool_entry and entities.get("app"):
            tool_entry = ("open_app", lambda e: {"app": e.get("app", "browser")})
            intent = "open_app"
        if not tool_entry and entities.get("url"):
            tool_entry = ("open_url", lambda e: {"url": e.get("url", "")})
            intent = "open_url"
        if not tool_entry and entities.get("math_expr"):
            tool_entry = ("calculate", lambda e: {"expression": e.get("math_expr", "")})
            intent = "calculate"
        if not tool_entry and entities.get("currency_from"):
            tool_entry = ("currency",
                          lambda e: {"from": e.get("currency_from", "USD"), "to": e.get("currency_to", "TRY")})
            intent = "currency"

        if tool_entry:
            tool_name, args_fn = tool_entry
            args = args_fn(entities)
            resp = self._run_tool(tool_name, args, text)
            if not resp.startswith("⚠"):
                resp = self.respgen.build(tool_name, resp, intent, entities)
            self.mem.push("dame", resp)
            return resp

        if confidence < 0.3 or intent == "unknown":
            self._emit("pipeline", "Araştırılıyor...")
            resp = self._auto_research(text, entities)
            self.mem.push("dame", resp)
            return resp

        resp = self.respgen.build_unknown(text)
        self.mem.push("dame", resp)
        return resp

    def _auto_research(self, text: str, entities: dict) -> str:
        u = self.mem.get_user_name()
        query = entities.get("wiki_query") or entities.get("search_query") or text

        facts = self.mem.search_facts(query[:40])
        if facts:
            return f"Hafızamda buldum {u}:\n{facts[0]['content']}"

        if WIKI_AVAILABLE:
            try:
                s = WIKI_RETRIEVER.get_summary(query, max_chars=700)
                if s:
                    return f"[Wiki DB]\n{s}"
            except:
                pass

        wiki = self.reg.get("wikipedia")
        res = wiki.run({"query": query}, self.mem)
        if res.success:
            return f"Araştırdım {u}:\n{res.output}"

        webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}")
        return f"Bulamadım {u}, Google'da '{query}' araması açılıyor."

    def retrain_neural(self, epochs=None, lr=None) -> str:
        return "Neural model bu versiyonda LM tarafından karşılanıyor."

    def get_status(self) -> dict:
        return {
            "version": self.cfg.VERSION,
            "lm": LM_AVAILABLE,
            "wiki": WIKI_AVAILABLE,
            "user": self.mem.get_user_name(),
            "messages": len(self.mem.short),
            "tools": len(self.reg.names()),
        }


class DameVoice:
    def __init__(self, cfg, log):
        self.log = log
        self.engine = None
        self.recognizer = None
        if TTS_OK:
            try:
                self.engine = pyttsx3.init()
                for v in self.engine.getProperty("voices"):
                    if "turkish" in v.name.lower() or "tr" in v.id.lower():
                        self.engine.setProperty("voice", v.id)
                        break
                self.engine.setProperty("rate", cfg.VOICE_RATE)
                self.engine.setProperty("volume", cfg.VOICE_VOLUME)
            except Exception as e:
                self.engine = None
        if STT_OK:
            self.recognizer = sr.Recognizer()

    def speak(self, text):
        if not self.engine:
            return
        clean = re.sub(r"[│┌┐└┘├┤─║╔╗╚╝█░▸►●◈]", "", text)
        clean = re.sub(r"\n+", ". ", clean).strip()[:500]

        def _go():
            try:
                self.engine.say(clean)
                self.engine.runAndWait()
            except:
                pass

        threading.Thread(target=_go, daemon=True).start()

    def listen(self):
        if not self.recognizer:
            return None
        try:
            with sr.Microphone() as src:
                self.recognizer.adjust_for_ambient_noise(src, duration=0.5)
                audio = self.recognizer.listen(src, timeout=6, phrase_time_limit=10)
            try:
                return self.recognizer.recognize_google(audio, language="tr-TR")
            except:
                return self.recognizer.recognize_google(audio, language="en-US")
        except:
            return None


class DameGUI:
    BG = "#061016"
    PANEL = "#0d1722"
    BORDER = "#142533"
    ACCENT = "#6fb3c8"
    ACCENT2 = "#4b8ca3"
    GREEN = "#39b78a"
    WARN = "#d19a3d"
    RED = "#d04b4b"
    TEXT = "#c9e6f2"
    DIM = "#5b6f7c"
    DAME_C = "#9ccad6"
    USER_C = "#cdeed8"
    SRC_C = "#f6e7c9"
    PIPE_C = "#9fb4c8"

    def __init__(self):
        self.cfg = Config()
        self.log = build_logger(self.cfg)
        self.brain = DameBrain(gui_callback=self._brain_cb)
        self.voice = DameVoice(self.cfg, self.log)
        self.listening = False
        self.cmd_hist = []
        self.hist_idx = -1

        self.root = tk.Tk()
        self.root.title(f"DAME v{self.cfg.VERSION}  —  LM + Wiki + Tool Pipeline")
        self.root.geometry("1100x800")
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)
        self.root.minsize(800, 560)
        self._build_ui()
        self._loops()
        self.root.after(800, self._startup)

    def _brain_cb(self, event, data):
        if event == "pipeline":
            self.root.after(0, self._pipe_status, data)
        elif event == "goodbye":
            self.root.after(2500, self.root.destroy)

    def _pipe_status(self, msg):
        self.lbl_pipeline.configure(text=f"▸ {msg}")
        self._set_status(msg[:24], self.PIPE_C)

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=self.PANEL, height=70)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="◈", font=("Courier New", 28, "bold"), fg=self.ACCENT, bg=self.PANEL).pack(side=tk.LEFT,
                                                                                                      padx=(16, 4))
        tk.Label(hdr, text="DAME", font=("Courier New", 26, "bold"), fg=self.ACCENT, bg=self.PANEL).pack(side=tk.LEFT)
        self.lbl_lm_badge = tk.Label(hdr,
                                     text=f"LM:{'ON' if LM_AVAILABLE else 'OFF'}",
                                     font=("Courier New", 8, "bold"),
                                     fg=self.GREEN if LM_AVAILABLE else self.RED, bg=self.PANEL)
        self.lbl_lm_badge.pack(side=tk.LEFT, padx=6)
        self.lbl_wiki_badge = tk.Label(hdr,
                                       text=f"Wiki:{'ON' if WIKI_AVAILABLE else 'OFF'}",
                                       font=("Courier New", 8, "bold"),
                                       fg=self.GREEN if WIKI_AVAILABLE else self.DIM, bg=self.PANEL)
        self.lbl_wiki_badge.pack(side=tk.LEFT, padx=2)

        tk.Label(hdr, text=f"  v{self.cfg.VERSION}", font=("Courier New", 9), fg=self.DIM, bg=self.PANEL).pack(
            side=tk.LEFT)
        sf = tk.Frame(hdr, bg=self.PANEL)
        sf.pack(side=tk.RIGHT, padx=16)
        self.lbl_online = tk.Label(sf, text="◉OFF", font=("Courier New", 9), fg=self.RED, bg=self.PANEL)
        self.lbl_online.pack(side=tk.LEFT, padx=4)
        self.lbl_dot = tk.Label(sf, text="●", font=("Arial", 14), fg=self.GREEN, bg=self.PANEL)
        self.lbl_dot.pack(side=tk.LEFT)
        self.lbl_status = tk.Label(sf, text="HAZIR", font=("Courier New", 9), fg=self.GREEN, bg=self.PANEL)
        self.lbl_status.pack(side=tk.LEFT, padx=4)

        tk.Frame(self.root, bg=self.ACCENT2, height=1).pack(fill=tk.X)
        pipe_bar = tk.Frame(self.root, bg="#050e20", height=22)
        pipe_bar.pack(fill=tk.X)
        pipe_bar.pack_propagate(False)
        tk.Label(pipe_bar, text="Pipeline:", font=("Courier New", 8), fg=self.DIM, bg="#050e20").pack(side=tk.LEFT,
                                                                                                      padx=8)
        self.lbl_pipeline = tk.Label(pipe_bar, text="▸ Bekleniyor", font=("Courier New", 8), fg=self.PIPE_C,
                                     bg="#050e20")
        self.lbl_pipeline.pack(side=tk.LEFT)

        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        sidebar = tk.Frame(body, bg=self.PANEL, width=220)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 7))
        sidebar.pack_propagate(False)

        def sec(t):
            tk.Label(sidebar, text=f"─ {t} ─", font=("Courier New", 8), fg=self.ACCENT2, bg=self.PANEL).pack(
                pady=(8, 2), fill=tk.X, padx=8)

        sec("SİSTEM")
        self.lbl_cpu = tk.Label(sidebar, text="CPU  : ---%", font=("Courier New", 8), fg=self.TEXT, bg=self.PANEL,
                                anchor="w")
        self.lbl_ram = tk.Label(sidebar, text="RAM  : ---%", font=("Courier New", 8), fg=self.TEXT, bg=self.PANEL,
                                anchor="w")
        self.lbl_disk = tk.Label(sidebar, text="DISK : ---%", font=("Courier New", 8), fg=self.TEXT, bg=self.PANEL,
                                 anchor="w")
        for l in (self.lbl_cpu, self.lbl_ram, self.lbl_disk):
            l.pack(fill=tk.X, padx=12)

        sec("SAAT")
        self.lbl_clock = tk.Label(sidebar, text="--:--:--", font=("Courier New", 20, "bold"), fg=self.ACCENT,
                                  bg=self.PANEL)
        self.lbl_clock.pack()
        self.lbl_date2 = tk.Label(sidebar, text="", font=("Courier New", 7), fg=self.DIM, bg=self.PANEL)
        self.lbl_date2.pack()

        sec("MOD")
        mode_str = ("LM + Wiki" if (LM_AVAILABLE and WIKI_AVAILABLE) else
                    "LM only" if LM_AVAILABLE else
                    "Wiki+NLP" if WIKI_AVAILABLE else "NLP only")
        tk.Label(sidebar, text=mode_str, font=("Courier New", 8, "bold"),
                 fg=self.GREEN if LM_AVAILABLE else self.WARN, bg=self.PANEL).pack(padx=12, anchor="w")

        sec("HAFIZA")
        self.lbl_mem = tk.Label(sidebar, text="", font=("Courier New", 7), fg=self.DIM, bg=self.PANEL, anchor="w",
                                justify=tk.LEFT)
        self.lbl_mem.pack(fill=tk.X, padx=10)

        sec("HIZLI KOMUTLAR")
        cmds = ["saat kaç?", "sistem durumu", "pil durumu", "hava durumu", "dolar kuru", "yardım"]
        for c in cmds:
            b = tk.Button(sidebar, text=f"  {c}", font=("Courier New", 8), fg=self.DIM, bg=self.PANEL,
                          relief=tk.FLAT, anchor="w", cursor="hand2",
                          activebackground=self.BORDER, activeforeground=self.ACCENT,
                          command=lambda x=c: self._quick(x))
            b.pack(fill=tk.X, padx=6, pady=1)
            b.bind("<Enter>", lambda e, btn=b: btn.configure(fg=self.ACCENT))
            b.bind("<Leave>", lambda e, btn=b: btn.configure(fg=self.DIM))

        chat_frame = tk.Frame(body, bg=self.BG)
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat = scrolledtext.ScrolledText(
            chat_frame, bg=self.BG, fg=self.TEXT, font=("Courier New", 11), wrap=tk.WORD,
            state=tk.DISABLED, relief=tk.FLAT, padx=14, pady=10,
            insertbackground=self.ACCENT, selectbackground=self.ACCENT2)
        self.chat.pack(fill=tk.BOTH, expand=True)
        for tag, fg, font in [
            ("dame_hdr", self.ACCENT, ("Courier New", 9, "bold")),
            ("dame_body", self.DAME_C, ("Courier New", 11)),
            ("user_hdr", self.GREEN, ("Courier New", 9, "bold")),
            ("user_body", self.USER_C, ("Courier New", 11)),
            ("sys_msg", self.DIM, ("Courier New", 9, "italic")),
            ("warn_msg", self.WARN, ("Courier New", 11, "bold")),
        ]:
            self.chat.tag_configure(tag, foreground=fg, font=font)

        irow = tk.Frame(chat_frame, bg=self.PANEL, height=54)
        irow.pack(fill=tk.X, pady=(5, 0))
        irow.pack_propagate(False)
        tk.Frame(irow, bg=self.ACCENT2, height=1).pack(fill=tk.X)
        inner = tk.Frame(irow, bg=self.PANEL)
        inner.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        tk.Label(inner, text="▶", fg=self.ACCENT, bg=self.PANEL, font=("Courier New", 11)).pack(side=tk.LEFT,
                                                                                                padx=(4, 0))
        self.var_in = tk.StringVar()
        self.entry = tk.Entry(inner, textvariable=self.var_in, bg="#040810", fg=self.TEXT,
                              font=("Courier New", 12), relief=tk.FLAT,
                              insertbackground=self.ACCENT, selectbackground=self.ACCENT2)
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)
        self.entry.bind("<Return>", lambda _: self._send())
        self.entry.bind("<Up>", self._hist_up)
        self.entry.focus_set()
        self.btn_mic = tk.Button(inner, text="🎤", command=self._toggle_mic,
                                 bg=self.PANEL, fg=self.DIM, font=("Arial", 14), relief=tk.FLAT, cursor="hand2", padx=4)
        self.btn_mic.pack(side=tk.RIGHT, padx=2)
        self.btn_send = tk.Button(inner, text="GÖNDER ▶", command=self._send,
                                  bg=self.ACCENT2, fg="white", font=("Courier New", 9, "bold"), relief=tk.FLAT,
                                  padx=10, cursor="hand2", activebackground=self.ACCENT, activeforeground="white")
        self.btn_send.pack(side=tk.RIGHT, padx=4)

        sbar = tk.Frame(self.root, bg="#030810", height=20)
        sbar.pack(fill=tk.X, side=tk.BOTTOM)
        lm_t = "LM:ON" if LM_AVAILABLE else "LM:OFF"
        wk_t = "Wiki:ON" if WIKI_AVAILABLE else "Wiki:OFF"
        tk.Label(sbar,
                 text=f"◈ DAME v{self.cfg.VERSION}  |  {lm_t}  |  {wk_t}  |  29 Araç  |  SQLite",
                 font=("Courier New", 7), fg=self.DIM, bg="#030810", anchor="w").pack(side=tk.LEFT, padx=10)

    def _quick(self, cmd):
        self.var_in.set(cmd)
        self._send()

    def _hist_up(self, event):
        if self.cmd_hist:
            self.hist_idx = min(self.hist_idx + 1, len(self.cmd_hist) - 1)
            self.var_in.set(self.cmd_hist[-(self.hist_idx + 1)])
            self.entry.icursor(tk.END)

    def _startup(self):
        self._sys_write("Sistemler başlatılıyor...")
        threading.Thread(target=self._async_startup, daemon=True).start()

    def _async_startup(self):
        ok = self._check_online()
        self.root.after(0, self._post_startup, ok)

    def _check_online(self):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            return True
        except:
            return False

    def _post_startup(self, online):
        self.lbl_online.configure(
            fg=self.GREEN if online else self.RED,
            text="◉NET" if online else "◉OFF")
        u = self.brain.mem.get_user_name()
        stat = self.brain.get_status()
        lm_s = "LM motoru aktif." if stat["lm"] else "LM henüz eğitilmemiş (NLP modu)."
        wk_s = "Wiki DB bağlı." if stat["wiki"] else "Wiki DB yok."
        msg = (f"Merhaba {u}! Dame v{stat['version']} hazır. "
               f"{lm_s} {wk_s} {stat['tools']} araç aktif. 'Yardım' yazın.")
        self._dame_write(msg)
        self.voice.speak(msg)

    def _send(self):
        text = self.var_in.get().strip()
        if not text:
            return
        self.var_in.set("")
        self.cmd_hist.append(text)
        self.hist_idx = -1
        self._user_write(text)
        self._set_status("İşleniyor...", self.WARN)
        threading.Thread(target=self._respond, args=(text,), daemon=True).start()

    def _respond(self, text):
        try:
            resp = self.brain.process(text)
        except Exception as e:
            self.brain.log.error(f"Process hatası: {e}", exc_info=True)
            resp = f"Beklenmedik hata: {e}"
        tag = "warn_msg" if resp.startswith("⚠") else "dame_body"
        self.root.after(0, self._dame_write, resp, tag)
        self.root.after(0, self._set_status, "HAZIR", self.GREEN)
        self.root.after(0, self._pipe_status, "Bekleniyor")
        self.voice.speak(resp)

    def _toggle_mic(self):
        if not STT_OK:
            self._sys_write("pip install SpeechRecognition pyaudio")
            return
        if not self.listening:
            self.listening = True
            self.btn_mic.configure(fg=self.GREEN)
            self._set_status("Dinleniyor...", self.GREEN)
            threading.Thread(target=self._listen_thread, daemon=True).start()

    def _listen_thread(self):
        result = self.voice.listen()
        self.listening = False
        self.root.after(0, self.btn_mic.configure, {"fg": self.DIM})
        if result:
            self.root.after(0, self._user_write, f"[🎤] {result}")
            self.root.after(100, lambda: threading.Thread(target=self._respond, args=(result,), daemon=True).start())
        else:
            self.root.after(0, self._set_status, "Ses anlaşılamadı", self.DIM)
            self.root.after(2500, self._set_status, "HAZIR", self.GREEN)

    def _write(self, text, tag):
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, text, tag)
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _dame_write(self, text, body_tag="dame_body"):
        ts = datetime.datetime.now().strftime("%H:%M")
        self._write(f"\n◈ DAME  [{ts}]\n", "dame_hdr")
        self._write(f"  {text}\n", body_tag)

    def _user_write(self, text):
        ts = datetime.datetime.now().strftime("%H:%M")
        self._write(f"\n▶ SİZ  [{ts}]\n", "user_hdr")
        self._write(f"  {text}\n", "user_body")

    def _sys_write(self, text):
        self._write(f"\n  ⟫ {text}\n", "sys_msg")

    def _set_status(self, text, color):
        self.lbl_status.configure(text=text, fg=color)
        self.lbl_dot.configure(fg=color)

    def _loops(self):
        self._sysinfo_loop()
        self._clock_loop()
        self._mem_loop()
        self._online_loop()

    def _sysinfo_loop(self):
        if PSUTIL_OK:
            try:
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
                disk = psutil.disk_usage("/").percent

                def col(v):
                    return self.ACCENT if v < 75 else self.WARN if v < 90 else self.RED

                self.lbl_cpu.configure(text=f"CPU  : {cpu:5.1f}%", fg=col(cpu))
                self.lbl_ram.configure(text=f"RAM  : {ram:5.1f}%", fg=col(ram))
                self.lbl_disk.configure(text=f"DISK : {disk:5.1f}%", fg=col(disk))
            except:
                pass
        self.root.after(2000, self._sysinfo_loop)

    def _clock_loop(self):
        n = datetime.datetime.now()
        self.lbl_clock.configure(text=n.strftime("%H:%M:%S"))
        D = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
        M = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
        self.lbl_date2.configure(text=f"{D[n.weekday()]}  {n.day} {M[n.month - 1]} {n.year}")
        self.root.after(1000, self._clock_loop)

    def _mem_loop(self):
        u = self.brain.mem.get_user_name()
        ctx = len(self.brain.mem.short)
        self.lbl_mem.configure(
            text=f"Kullanıcı : {u}\nBağlam    : {ctx} mesaj\nMod       : {'LM' if LM_AVAILABLE else 'NLP'}")
        self.root.after(5000, self._mem_loop)

    def _online_loop(self):
        def chk():
            ok = self._check_online()
            self.root.after(0, self.lbl_online.configure,
                            {"fg": self.GREEN if ok else self.RED, "text": "◉NET" if ok else "◉OFF"})

        threading.Thread(target=chk, daemon=True).start()
        self.root.after(30000, self._online_loop)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.voice.speak("Güle gidin.")
        self.brain.mem.db.close()
        self.root.destroy()


if __name__ == "__main__":
    print(f"║  DAME v6.0  —  LM + Wiki + Tool Pipeline  ║")
    print(f"Python {sys.version.split()[0]}  |  {platform.system()} {platform.release()}")
    print(f"LM Motor  : {'✓ Aktif' if LM_AVAILABLE else '✗ dame_transformer.py eğitilmemiş'}")
    print(f"Wiki DB   : {'✓ Aktif' if WIKI_AVAILABLE else '✗ wiki_builder.py ile build edin'}")
    print()
    opt = []
    if not TTS_OK:
        opt.append("pyttsx3")
    if not STT_OK:
        opt.append("SpeechRecognition pyaudio")
    if not PSUTIL_OK:
        opt.append("psutil")
    if not PYAUTOGUI_OK:
        opt.append("pyautogui pillow")
    if not TORCH_OK:
        opt.append("torch  (LM için zorunlu)")
    if opt:
        print("Eksik isteğe bağlı kütüphaneler:")
        for m in opt:
            print(f"   pip install {m}")
        print()
    DameGUI().run()