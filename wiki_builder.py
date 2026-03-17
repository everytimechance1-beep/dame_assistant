import os
import sys
import re
import bz2
import gzip
import math
import json
import time
import sqlite3
import hashlib
import logging
import argparse
import threading
import xml.sax
import xml.sax.handler
import urllib.request
import urllib.parse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, Iterator, List, Dict, Tuple
from dataclasses import dataclass, field
import html

CSI    = "\x1b["
GREEN  = CSI + "32m"; CYAN  = CSI + "36m"; YELLOW = CSI + "33m"
RED    = CSI + "31m"; BLUE  = CSI + "34m"; RESET  = CSI + "0m"
BOLD   = CSI + "1m"

WIKI_DIR   = Path.home() / ".dame" / "wiki"
DB_PATH_TR = WIKI_DIR / "wiki_tr.db"
DB_PATH_EN = WIKI_DIR / "wiki_en.db"
DB_PATH_COMBINED = WIKI_DIR / "wiki_combined.db"
DUMP_DIR   = WIKI_DIR / "dumps"
LOG_PATH   = WIKI_DIR / "wiki_builder.log"

DUMP_URL_TMPL = (
    "https://dumps.wikimedia.org/{lang}wiki/latest/"
    "{lang}wiki-latest-pages-articles.xml.bz2"
)

DUMP_INDEX_TMPL = (
    "https://dumps.wikimedia.org/{lang}wiki/latest/"
    "{lang}wiki-latest-pages-articles-multistream-index.txt.bz2"
)

BATCH_SIZE   = 500
MAX_TEXT_LEN = 8000
MIN_TEXT_LEN = 100
MAX_TOKENS_TF= 20000

HEADERS = {
    "User-Agent": "DAME-WikiBuilder/2.0 (https://github.com/dame-assistant; "
                  "contact@dame-assistant.local) Python urllib",
}

def build_logger() -> logging.Logger:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("WikiBuilder")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", "%H:%M:%S")
    fh  = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    ch  = logging.StreamHandler()
    ch.setLevel(logging.INFO);  ch.setFormatter(fmt)
    if not log.handlers:
        log.addHandler(fh); log.addHandler(ch)
    return log

log = build_logger()

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA page_size    = 8192;

CREATE TABLE IF NOT EXISTS meta (
    key  TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS articles (
    id        INTEGER PRIMARY KEY,
    title     TEXT NOT NULL,
    lang      TEXT NOT NULL DEFAULT 'tr',
    text      TEXT NOT NULL,
    tokens    INTEGER DEFAULT 0,
    checksum  TEXT,
    indexed   INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_title ON articles(title);
CREATE INDEX IF NOT EXISTS idx_lang  ON articles(lang);

CREATE TABLE IF NOT EXISTS vocab (
    term_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    term      TEXT UNIQUE NOT NULL,
    df        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS postings (
    term_id   INTEGER NOT NULL REFERENCES vocab(term_id),
    art_id    INTEGER NOT NULL REFERENCES articles(id),
    tf        REAL    NOT NULL,
    PRIMARY KEY (term_id, art_id)
);

CREATE INDEX IF NOT EXISTS idx_post_term ON postings(term_id);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_articles
    USING fts5(title, text, content='articles', content_rowid='id');
"""

_RE_TEMPLATE  = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
_RE_WIKILINK  = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_RE_EXT_LINK  = re.compile(r"\[https?://\S+\s+([^\]]+)\]")
_RE_HTML_TAG  = re.compile(r"<[^>]+>")
_RE_HEADING   = re.compile(r"={2,}[^=]+=+")
_RE_WHITESPACE= re.compile(r"\s{2,}")
_RE_REF        = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_RE_COMMENTS  = re.compile(r"", re.DOTALL)

def clean_wikitext(raw: str) -> str:
    t = html.unescape(raw)
    t = _RE_COMMENTS.sub("", t)
    t = _RE_REF.sub("", t)
    for _ in range(4):
        prev = t
        t = _RE_TEMPLATE.sub("", t)
        if t == prev: break
    t = _RE_WIKILINK.sub(r"\1", t)
    t = _RE_EXT_LINK.sub(r"\1", t)
    t = _RE_HTML_TAG.sub(" ", t)
    t = _RE_HEADING.sub(" ", t)
    t = re.sub(r"^\s*[|!].*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"\{\|.*?\|\}", "", t, flags=re.DOTALL)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"^\s*[\*#:;]+", " ", t, flags=re.MULTILINE)
    t = re.sub(r"https?://\S+", "", t)
    t = _RE_WHITESPACE.sub(" ", t)
    return t.strip()[:MAX_TEXT_LEN]

_TR_STOPWORDS = {
    "bir","bu","şu","da","de","ve","ile","için","mi","mu","mı","mü",
    "ne","ki","ya","daha","çok","az","en","hem","ya","veya","ama",
    "ancak","fakat","ise","bile","kadar","gibi","göre","beri","olan",
    "olarak","o","ben","sen","biz","siz",
    "the","a","an","of","in","to","and","or","is","are","was","were",
    "that","this","it","for","on","with","at","by","from","not","be",
    "have","has","had","do","does","did","will","would","could","should",
}

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9çşğüöıa-zçşğüöı\-]", " ", text)
    tokens = [t for t in text.split() if len(t) > 2 and t not in _TR_STOPWORDS]
    return tokens

@dataclass
class WikiArticle:
    title: str = ""
    text:  str = ""
    ns:    str = "0"

class WikiSAXHandler(xml.sax.handler.ContentHandler):
    def __init__(self, callback, limit: int = 0):
        super().__init__()
        self._callback = callback
        self._limit    = limit
        self._count    = 0
        self._cur_art  = WikiArticle()
        self._in_text  = False
        self._in_title = False
        self._in_ns    = False
        self._buf: List[str] = []

    def startElement(self, name, attrs):
        if   name == "page":    self._cur_art = WikiArticle()
        elif name == "title":   self._in_title = True; self._buf = []
        elif name == "ns":      self._in_ns = True;    self._buf = []
        elif name == "text":    self._in_text = True;  self._buf = []

    def characters(self, content):
        if self._in_title or self._in_ns or self._in_text:
            self._buf.append(content)

    def endElement(self, name):
        if name == "title":
            self._cur_art.title = "".join(self._buf).strip()
            self._in_title = False
        elif name == "ns":
            self._cur_art.ns = "".join(self._buf).strip()
            self._in_ns = False
        elif name == "text":
            self._cur_art.text = "".join(self._buf)
            self._in_text = False
        elif name == "page":
            if self._cur_art.ns == "0" and self._cur_art.title and self._cur_art.text:
                cleaned = clean_wikitext(self._cur_art.text)
                if len(cleaned) >= MIN_TEXT_LEN:
                    self._callback(self._cur_art.title, cleaned)
                    self._count += 1
                    if self._limit and self._count >= self._limit:
                        raise StopIteration(f"Limit reached: {self._limit}")

    @property
    def count(self) -> int:
        return self._count

def _fmt_size(n: int) -> str:
    for unit in ["B","KB","MB","GB"]:
        if n < 1024: return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

def download_dump(lang: str, dest_dir: Path, force: bool = False) -> Optional[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{lang}wiki-latest-pages-articles.xml.bz2"
    out   = dest_dir / fname
    if out.exists() and not force:
        log.info(f"Dump zaten mevcut, atlanıyor: {out}")
        return out
    url = DUMP_URL_TMPL.format(lang=lang)
    log.info(f"İndiriliyor: {url}")
    start = time.time()
    downloaded = [0]
    total_size = [0]
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            total_size[0] = total
            chunk = 1024 * 256
            with open(out, "wb") as f:
                while True:
                    data = resp.read(chunk)
                    if not data: break
                    f.write(data)
                    downloaded[0] += len(data)
                    pct = downloaded[0] / total * 100 if total > 0 else 0
                    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                    elapsed = time.time() - start
                    speed = downloaded[0] / elapsed if elapsed > 0 else 1
                    eta = (total - downloaded[0]) / speed if speed > 0 else 0
                    print(f"\r  [{bar}] {pct:.1f}%  "
                          f"{_fmt_size(downloaded[0])}/{_fmt_size(total)}  "
                          f"{_fmt_size(speed)}/s  ETA:{int(eta)}s   ",
                          end="", flush=True)
        print(f"\n{GREEN}✓ İndirme tamamlandı:{RESET} {_fmt_size(downloaded[0])}")
        return out
    except KeyboardInterrupt:
        print(f"\n{YELLOW}İndirme kullanıcı tarafından iptal edildi.{RESET}")
        return None
    except Exception as e:
        log.error(f"İndirme hatası ({lang}): {e}")
        print(f"\n{RED}İndirme hatası:{RESET} {e}")
        return None

class WikiDBBuilder:
    def __init__(self, db_path: Path, lang: str):
        self.db_path = db_path
        self.lang    = lang
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn    = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
        self._buf: List[Tuple] = []
        self._inserted = 0

    def _flush(self):
        if not self._buf: return
        self.conn.executemany(
            "INSERT OR IGNORE INTO articles(title, lang, text, tokens, checksum) "
            "VALUES (?,?,?,?,?)", self._buf)
        self.conn.commit()
        self._inserted += len(self._buf)
        self._buf.clear()

    def add_article(self, title: str, text: str):
        chk = hashlib.md5(text.encode("utf-8","replace")).hexdigest()[:12]
        tok = len(tokenize(text))
        self._buf.append((title, self.lang, text, tok, chk))
        if len(self._buf) >= BATCH_SIZE:
            self._flush()

    def finish(self):
        self._flush()
        cnt = self.conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('article_count',?)", (str(cnt),))
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('lang',?)",          (self.lang,))
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('built_at',?)",
                          (time.strftime("%Y-%m-%d %H:%M:%S"),))
        self.conn.commit()
        log.info(f"DB hazır: {self.db_path} — {cnt:,} makale")
        return cnt

    def close(self):
        self.conn.close()

class TFIDFIndexer:
    def __init__(self, db_path: Path):
        self.db   = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")

    def build(self, max_vocab: int = MAX_TOKENS_TF):
        log.info("TF-IDF indeksleme başlıyor")
        total_docs = self.db.execute(
            "SELECT COUNT(*) FROM articles WHERE indexed=0").fetchone()[0]
        if total_docs == 0:
            log.info("İndekslenecek makale yok."); return
        df: Counter = Counter()
        t0 = time.time()
        cur = self.db.execute("SELECT id, text FROM articles WHERE indexed=0")
        for i, (art_id, text) in enumerate(cur, 1):
            terms = set(tokenize(text))
            for t in terms:
                df[t] += 1
            if i % 5000 == 0:
                pct = i / total_docs * 100
                print(f"\r  Geçiş 1: {i:>8,}/{total_docs:,}  ({pct:.1f}%)  "
                      f"{int(time.time()-t0)}s", end="", flush=True)
        top_terms = [t for t, _ in df.most_common() if len(t) > 2][:max_vocab]
        self.db.execute("DELETE FROM vocab")
        self.db.executemany(
            "INSERT OR IGNORE INTO vocab(term, df) VALUES (?,?)",
            [(t, df[t]) for t in top_terms])
        self.db.commit()
        rows = self.db.execute("SELECT term, term_id FROM vocab").fetchall()
        term_to_id = {r[0]: r[1] for r in rows}
        self.db.execute("DELETE FROM postings")
        buf: List[Tuple] = []
        t0  = time.time()
        cur2 = self.db.execute("SELECT id, text FROM articles WHERE indexed=0")
        for i, (art_id, text) in enumerate(cur2, 1):
            tokens  = tokenize(text)
            if not tokens: continue
            tf_raw  = Counter(tokens)
            max_tf  = max(tf_raw.values())
            doc_len = len(tokens)
            for term, cnt in tf_raw.items():
                if term not in term_to_id: continue
                tid = term_to_id[term]
                tf_norm = (cnt / max_tf) * math.log(1 + doc_len)
                buf.append((tid, art_id, round(tf_norm, 6)))
            if len(buf) >= 50_000:
                self.db.executemany(
                    "INSERT OR REPLACE INTO postings(term_id,art_id,tf) VALUES(?,?,?)", buf)
                buf.clear()
            if i % 5000 == 0:
                self.db.commit()
                pct = i / total_docs * 100
                print(f"\r  Geçiş 2: {i:>8,}/{total_docs:,}  ({pct:.1f}%)  "
                      f"{int(time.time()-t0)}s", end="", flush=True)
        if buf:
            self.db.executemany(
                "INSERT OR REPLACE INTO postings(term_id,art_id,tf) VALUES(?,?,?)", buf)
        self.db.execute("UPDATE articles SET indexed=1 WHERE indexed=0")
        self.db.commit()
        try:
            self.db.execute(
                "INSERT INTO fts_articles(fts_articles) VALUES('rebuild')")
            self.db.commit()
        except Exception as e:
            log.debug(f"FTS5 rebuild: {e}")
        log.info("TF-IDF indeksleme tamamlandı.")

    def close(self):
        self.db.close()

class WikiRetriever:
    def __init__(self, db_paths: List[Path], top_k: int = 5):
        self._conns: List[sqlite3.Connection] = []
        for p in db_paths:
            if p.exists():
                c = sqlite3.connect(str(p), check_same_thread=False)
                c.row_factory = sqlite3.Row
                self._conns.append(c)
                log.info(f"WikiRetriever: {p.name} yüklendi")
        self.top_k = top_k

    def _tfidf_search(self, conn: sqlite3.Connection,
                      tokens: List[str], top_k: int) -> List[Dict]:
        if not tokens: return []
        placeholders = ",".join("?" * len(tokens))
        try:
            rows = conn.execute(f"""
                SELECT a.id, a.title, a.text,
                       SUM(p.tf * (LOG(1.0 + (
                           SELECT COUNT(*) FROM articles
                       ) / MAX(1.0, v.df)))) AS score
                FROM postings p
                JOIN vocab    v ON v.term_id = p.term_id
                JOIN articles a ON a.id      = p.art_id
                WHERE v.term IN ({placeholders})
                GROUP BY a.id
                ORDER BY score DESC
                LIMIT ?
            """, tokens + [top_k]).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            log.debug(f"TF-IDF search error: {e}")
            return []

    def _fts_search(self, conn: sqlite3.Connection,
                    query: str, top_k: int) -> List[Dict]:
        try:
            rows = conn.execute("""
                SELECT a.id, a.title, a.text, bm25(fts_articles) AS score
                FROM fts_articles f
                JOIN articles a ON a.id = f.rowid
                WHERE fts_articles MATCH ?
                ORDER BY score
                LIMIT ?
            """, (query, top_k)).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            log.debug(f"FTS search error: {e}")
            return []

    def _title_search(self, conn: sqlite3.Connection,
                      query: str, top_k: int) -> List[Dict]:
        try:
            rows = conn.execute(
                "SELECT id, title, text FROM articles "
                "WHERE title LIKE ? LIMIT ?",
                (f"%{query}%", top_k)).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            log.debug(f"Title search error: {e}")
            return []

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        k      = top_k or self.top_k
        tokens = tokenize(query)
        seen   = set()
        merged: List[Dict] = []
        for conn in self._conns:
            exact = conn.execute(
                "SELECT id, title, text FROM articles WHERE title = ? LIMIT 1",
                (query,)).fetchone()
            if exact:
                r = dict(exact)
                if r["id"] not in seen:
                    seen.add(r["id"]); r["score"] = 999.0; merged.append(r)
            for r in self._fts_search(conn, query, k):
                if r["id"] not in seen:
                    seen.add(r["id"]); merged.append(r)
            for r in self._tfidf_search(conn, tokens, k):
                if r["id"] not in seen:
                    seen.add(r["id"]); merged.append(r)
            if len(merged) < 2:
                for r in self._title_search(conn, query, k):
                    if r["id"] not in seen:
                        seen.add(r["id"]); merged.append(r)
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return merged[:k]

    def get_summary(self, query: str, max_chars: int = 800) -> str:
        results = self.search(query, top_k=1)
        if not results:
            return ""
        r    = results[0]
        text = r["text"]
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        summ  = " ".join(sents[:5])
        return f"[{r['title']}]\n{summ[:max_chars]}"

    def stats(self) -> Dict:
        s = {}
        for conn in self._conns:
            try:
                lang  = conn.execute("SELECT value FROM meta WHERE key='lang'").fetchone()
                cnt   = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
                vcnt  = conn.execute("SELECT COUNT(*) FROM vocab").fetchone()[0]
                pcnt  = conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
                l = lang[0] if lang else "?"
                s[l] = {"articles": cnt, "vocab": vcnt, "postings": pcnt}
            except Exception as e:
                s["?"] = {"error": str(e)}
        return s

    def close(self):
        for c in self._conns: c.close()

def build_wiki(lang: str, limit: int, force_download: bool, skip_index: bool):
    db_path = DB_PATH_TR if lang == "tr" else DB_PATH_EN
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    dump_file = download_dump(lang, DUMP_DIR, force=force_download)
    if dump_file is None:
        log.error("Dump indirilemedi, çıkılıyor."); return
    builder = WikiDBBuilder(db_path, lang)
    count   = [0]
    t_start = time.time()
    def _cb(title: str, text: str):
        builder.add_article(title, text)
        count[0] += 1
        if count[0] % 1000 == 0:
            elapsed = time.time() - t_start
            rate    = count[0] / elapsed if elapsed > 0 else 0
            print(f"\r  Makale: {count[0]:>9,}  {rate:,.0f} art/s  "
                  f"{int(elapsed)}s geçti", end="", flush=True)
    handler = WikiSAXHandler(callback=_cb, limit=limit)
    try:
        with bz2.open(str(dump_file), "rb") as f:
            try:
                xml.sax.parse(f, handler)
            except StopIteration:
                pass
    except Exception as e:
        log.error(f"Parse hatası: {e}")
    total = builder.finish()
    builder.close()
    if not skip_index:
        indexer = TFIDFIndexer(db_path)
        indexer.build()
        indexer.close()

def get_retriever() -> Optional[WikiRetriever]:
    paths = [p for p in [DB_PATH_TR, DB_PATH_EN] if p.exists()]
    if not paths:
        return None
    return WikiRetriever(paths)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",          choices=["tr","en","both"], default="tr")
    parser.add_argument("--limit",         type=int, default=200_000)
    parser.add_argument("--force-download",action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-index",    action="store_true")
    parser.add_argument("--query",         type=str, default="")
    parser.add_argument("--stats",         action="store_true")
    parser.add_argument("--reindex",       action="store_true")
    args = parser.parse_args()

    if args.query:
        ret = get_retriever()
        if ret:
            results = ret.search(args.query, top_k=3)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']} [{r.get('score', 0):.3f}]")
            ret.close()
        return

    if args.stats:
        ret = get_retriever()
        if ret:
            print(ret.stats())
            ret.close()
        return

    if args.reindex:
        langs = ["tr","en"] if args.lang == "both" else [args.lang]
        for lang in langs:
            db = DB_PATH_TR if lang == "tr" else DB_PATH_EN
            if db.exists():
                conn = sqlite3.connect(str(db))
                conn.execute("UPDATE articles SET indexed=0")
                conn.execute("DELETE FROM vocab")
                conn.execute("DELETE FROM postings")
                conn.commit(); conn.close()
                idx = TFIDFIndexer(db)
                idx.build(); idx.close()
        return

    langs = ["tr","en"] if args.lang == "both" else [args.lang]
    for lang in langs:
        if args.skip_download:
            db_path = DB_PATH_TR if lang == "tr" else DB_PATH_EN
            if not args.skip_index:
                indexer = TFIDFIndexer(db_path)
                indexer.build(); indexer.close()
        else:
            build_wiki(lang, args.limit, args.force_download, args.skip_index)

if __name__ == "__main__":
    main()