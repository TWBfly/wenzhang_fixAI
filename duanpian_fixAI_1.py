from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import statistics
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
import logging
import sys
import os
import shutil
from advanced_analyzers import RhythmShatter, SensoryWeaver, RedundancyDetector, POVChecker, DialoguePowerAnalyzer, ShowDontTellTransformer, ExplanatoryTailTagDetector, CharacterVoiceDifferentiator, MicroImperfectionGenerator, PrologueHookAnalyzer, IdentitySummaryDetector, ClicheExpressDetector, SemanticConflictExtractor, GlobalVisionHooker, NarrativeCausalityAuditor, ConjunctionNeutering, StyleBenchmarkEngine, ContentParityValidator
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_CHAPTER_PATTERN = r"(?m)^(楔子|序章|前言|尾声|后记|番外|第[零一二三四五六七八九十百千万0-9]+章[^\n]*|第[零一二三四五六七八九十百千万0-9]+回[^\n]*)$"

@dataclass
class NovelContext:
    book_id: str
    genre: str  # MaleLead, FemaleLead, Unknown
    background: str  # Historical, Urban, Fantasy, etc.
    tone: str
    word_count_target: int
    characters: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PatternRule:
    name: str
    pattern: str
    weight: float
    category: str
    severity: str = "P1"  # P0: Critical, P1: Serious, P2: Stylistic
    tier: int = 1         # 1: Always flag, 2: Cluster, 3: Density


@dataclass
class Hit:
    chapter: str
    paragraph_index: int
    sentence_index: int
    hit_type: str
    rule_name: str
    text: str
    start: int
    end: int
    weight: float
    severity: str = "P1"


@dataclass
class ParagraphInfo:
    chapter: str
    paragraph_index: int
    text: str
    tags: List[str]
    scores: Dict[str, int]
    has_dialogue: bool
    sentence_count: int
    role: str = "neutral"  # action, dialogue, environment, exposition, monologue
    omniscient_score: float = 0.0
    rhythm_score: float = 0.0  # Variance of sentence lengths
    sensory_score: float = 0.0 # Density of sensory keywords
    symmetry_penalty: float = 0.0 # Penalty for AI-style balanced sentences
    hits_by_severity: Dict[str, int] = None # P0, P1, P2 counts


@dataclass
class SentenceChange:
    chapter: str
    paragraph: int
    sentence: int
    original: str
    revised: str
    applied_rules: List[str]
    risk_level: str


@dataclass
class LogicFact:
    """A struct for tracking core facts for consistency."""
    chapter: str
    type: str  # ITEM_STATUS, CHAR_LIFE, CHAR_LOC, INFO_KNOWN, SUBTEXT_ANCHOR, CHAR_FATE
    entity: str
    status: str
    description: str


@dataclass
class LogicConflict:
    """A struct for reporting logic discrepancies."""
    type: str  # DISAPPEARANCE, TELEPORTATION, RESURRECTION, KNOWLEDGE_GAP, OMNISCIENCE_LEAK
    description: str
    severity: str  # P0, P1
    chapters_involved: List[str]


@dataclass
class ChapterReport:
    chapter: str
    word_count_before: int
    word_count_after: int
    ai_score_before: float
    ai_score_after: float
    issue_summary_before: Dict[str, int]
    issue_summary_after: Dict[str, int]
    paragraph_tag_distribution_before: Dict[str, int]
    paragraph_tag_distribution_after: Dict[str, int]
    top_rules_before: List[Dict[str, object]]
    top_rules_after: List[Dict[str, object]]
    changed_sentence_count: int
    suspicious_sentences_before: List[Dict[str, object]]
    suspicious_sentences_after: List[Dict[str, object]]
    empty_transition_blocks_before: List[Dict[str, object]]
    empty_transition_blocks_after: List[Dict[str, object]]
    logic_conflicts: List[LogicConflict] = field(default_factory=list)


@dataclass
class BookReport:
    total_words_before: int
    total_words_after: int
    chapter_count: int
    ai_score_before: float
    ai_score_after: float
    global_rule_counts_before: Dict[str, int]
    global_rule_counts_after: Dict[str, int]
    global_issue_counts_before: Dict[str, int]
    global_issue_counts_after: Dict[str, int]
    total_changed_sentences: int
    chapters: List[ChapterReport]
    logic_summary: List[LogicConflict] = field(default_factory=list)


API_DUANPIAN_PATH = Path(__file__).parent.parent / "API_duanpian"
if str(API_DUANPIAN_PATH) not in sys.path:
    sys.path.append(str(API_DUANPIAN_PATH))

try:
    from llm_client import generate_text_safe
except ImportError:
    logging.warning("未能导入 API_duanpian.llm_client，LLM 增强功能将被降级。")
    def generate_text_safe(prompt, system_prompt="", model="deepseek-v4-flash", **kwargs):
        return None

def extract_json_from_llm(text: str) -> Optional[Any]:
    """从 LLM 响应中安全提取 JSON，支持多层嵌套。"""
    if not text:
        return None
    text = text.strip()
    # 去除 markdown 代码块标记
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    break
    return None


def call_llm_with_fallback(prompt: str, system_prompt: str) -> Optional[str]:
    # 优先尝试 deepseek-v4-flash
    res = generate_text_safe(prompt, system_prompt=system_prompt, model="deepseek-v4-flash")
    if res:
        return res
    # fallback
    logging.info("deepseek-v4-flash 失败，尝试 fallback 到 gemini-1.5-flash")
    res = generate_text_safe(prompt, system_prompt=system_prompt, model="gemini-1.5-flash")
    return res


class LogicAuditor:
    """Atomic Logic & Causal Alignment (ALCA) engine with SQLite persistence."""
    def __init__(self, db_path: Path, allow_llm_fact_extraction: bool = False):
        self.db_path = db_path
        self.allow_llm_fact_extraction = allow_llm_fact_extraction
        self.ledger: List[LogicFact] = []
        self.conflicts: List[LogicConflict] = []
        self.book_id: str = "default_book"

        # Core entity extraction patterns
        self.item_patterns = {
            "backpack": r"背包|书包|挎包",
            "phone": r"手机|电话|机子",
            "card": r"银行卡|卡片|磁卡",
            "money": r"钱|现金|红票子|钞票|硬币|钢镚",
            "food": r"粥|蛋羹|个子|包子|馒头|剩菜|油条",
            "weapon": r"剑|刀|铁钎|钎子|匕首|硬弩",
        }
        self.sensory_patterns = {
            "smell": r"的味道|腥气|刺鼻|香气|臭|弥漫|药水味|草木灰",
            "touch": r"冰冷|火辣辣|粗糙|磨蹭|湿润|干涩|麻木|僵硬",
            "sound": r"破空|闷哼|嘎然而止|脆响|嘶|打呼|敲在",
            "internal": r"瞳孔|咽喉|心脏|生理|下意识|肺部|麻刺",
        }
        self._init_db()

    def _init_db(self):
        """Ensure SQLite tables exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logic_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id TEXT,
                    chapter_name TEXT,
                    type TEXT,
                    entity TEXT,
                    status TEXT,
                    description TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_book_id ON logic_facts(book_id)")

            # --- V11 New Graph Tables ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id TEXT,
                    chapter_name TEXT,
                    entity_type TEXT,
                    entity_name TEXT,
                    details TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS character_knowledge_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id TEXT,
                    chapter_name TEXT,
                    character_name TEXT,
                    knowledge_topic TEXT,
                    status TEXT,
                    context TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS item_trajectory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id TEXT,
                    chapter_name TEXT,
                    item_name TEXT,
                    current_owner TEXT,
                    status TEXT,
                    context TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS foreshadowing_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id TEXT,
                    setup_chapter TEXT,
                    payoff_chapter TEXT,
                    clue TEXT,
                    status TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS character_fate_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id TEXT,
                    chapter_name TEXT,
                    character_name TEXT,
                    fate_status TEXT,  -- Alive, Dead, Incapacitated, Exiled
                    context TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chapter_signatures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id TEXT,
                    chapter_name TEXT,
                    sig_type TEXT,  -- ENDING_HOOK, SIGNATURE_DIALOGUE
                    content_hash TEXT,
                    content_text TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id TEXT,
                    chapter_name TEXT,
                    entity_name TEXT,
                    shared_status TEXT,  -- SHARED, PRIVATE
                    owner TEXT
                )
            """)

    def set_book_context(self, book_id: str):
        self.book_id = book_id
        self.load_ledger()

    def get_character_fates(self) -> Dict[str, str]:
        """Fetch current fates for all characters in the current book."""
        fates = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT character_name, fate_status FROM character_fate_ledger WHERE book_id = ?",
                (self.book_id,)
            )
            for row in cursor:
                fates[row[0]] = row[1]
        return fates

    def register_character_fate(self, chapter_name: str, char_name: str, fate: str, context: str):
        """Update or insert a character's terminal fate."""
        with sqlite3.connect(self.db_path) as conn:
            # Check if exists
            cursor = conn.execute(
                "SELECT id FROM character_fate_ledger WHERE book_id = ? AND character_name = ?",
                (self.book_id, char_name)
            )
            row = cursor.fetchone()
            if row:
                conn.execute(
                    "UPDATE character_fate_ledger SET chapter_name = ?, fate_status = ?, context = ? WHERE id = ?",
                    (chapter_name, fate, context, row[0])
                )
            else:
                conn.execute(
                    "INSERT INTO character_fate_ledger (book_id, chapter_name, character_name, fate_status, context) VALUES (?, ?, ?, ?, ?)",
                    (self.book_id, chapter_name, char_name, fate, context)
                )

    def register_chapter_signature(self, chapter_name: str, sig_type: str, text: str):
        """Save a semantic signature for the chapter to prevent repetition."""
        if not text.strip(): return
        clean_text = re.sub(r"\s+", "", text)[:200] # Normalize
        content_hash = hashlib.md5(clean_text.encode('utf-8')).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if this exact signature already exists for this book
            cursor = conn.execute(
                "SELECT chapter_name FROM chapter_signatures WHERE book_id = ? AND content_hash = ?",
                (self.book_id, content_hash)
            )
            row = cursor.fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO chapter_signatures (book_id, chapter_name, sig_type, content_hash, content_text) VALUES (?, ?, ?, ?, ?)",
                    (self.book_id, chapter_name, sig_type, content_hash, text[:500])
                )

    def get_recent_signatures(self, limit: int = 3) -> List[Dict[str, str]]:
        """Fetch recent signatures to check for overlaps."""
        sigs = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT chapter_name, content_text FROM chapter_signatures WHERE book_id = ? ORDER BY id DESC LIMIT ?",
                (self.book_id, limit)
            )
            for row in cursor:
                sigs.append({"chapter": row[0], "text": row[1]})
        return sigs

    def register_shared_knowledge(self, chapter_name: str, entity: str, status: str, owner: str = "PUBLIC"):
        """Track what knowledge has been disclosed to the reader/characters."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO knowledge_ledger (book_id, chapter_name, entity_name, shared_status, owner) VALUES (?, ?, ?, ?, ?)",
                (self.book_id, chapter_name, entity, status, owner)
            )

    def get_shared_knowledge(self) -> Set[str]:
        """Fetch all entities known to the public/protagonists."""
        known = set()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT entity_name FROM knowledge_ledger WHERE book_id = ? AND shared_status = 'SHARED'",
                (self.book_id,)
            )
            for row in cursor:
                known.add(row[0])
        return known

    def extract_facts(self, chapter_name: str, text: str, characters: List[Dict[str, str]] = None):
        """Regex and LLM-based structured fact extraction."""
        for item_key, pattern in self.item_patterns.items():
            # Match variations like "拿走了钱", "把钱拿走"
            if re.search(rf"(?:拿走|夺走|抢走|没收|毁掉|扔掉).*?({pattern})|把.*?({pattern}).*?(?:拿走|夺走|抢走|没收|毁掉|扔掉)", text):
                self.ledger.append(LogicFact(chapter_name, "ITEM_STATUS", item_key, "lost", f"失去{item_key}"))
            elif re.search(rf"(?:拿到|找回|掏出|取出|拿了).*?({pattern})|把.*?({pattern}).*?(?:拿到|找回|掏出|取出|拿了)", text):
                self._check_item_conflict(chapter_name, item_key, "possession")
                self.ledger.append(LogicFact(chapter_name, "ITEM_STATUS", item_key, "held", f"持有{item_key}"))

        if "车" in text and ("山路" in text or "公路" in text):
            self.ledger.append(LogicFact(chapter_name, "CHAR_LOC", "PROTAG", "in_vehicle", "在车上行驶"))

        # --- Character Fate Detection (V12) ---
        death_patterns = [
            (r"(?:处决|处死|推上断头台|斩首|毙命|咽气|断气|死透|命丧|身亡|气绝)", "Dead"),
            (r"(?:瘫痪|中风|残废|废了|断了腿|瞎了)", "Incapacitated"),
            (r"(?:流放|发配|逐出|遣散|送入家庙)", "Exiled"),
        ]
        
        if characters:
            for char_info in characters:
                name = char_info["name"]
                if name in text:
                    for pattern, fate in death_patterns:
                        # Narrow match: "沈弘毅...处决"
                        if re.search(rf"{name}[^。！？；\n]{{0,30}}{pattern}", text):
                            self.register_character_fate(chapter_name, name, fate, f"在{chapter_name}中{pattern}")
                            self.ledger.append(LogicFact(chapter_name, "CHAR_FATE", name, fate, f"状态更新为{fate}"))

        # --- Chapter Signature Extraction (V12) ---
        paras = [p.strip() for p in text.split('\n') if p.strip()]
        if len(paras) >= 1:
            ending_text = "\n".join(paras[-2:]) # Take last 2 paragraphs
            self.register_chapter_signature(chapter_name, "ENDING_HOOK", ending_text)

        # --- V11 LLM Causal Extraction (opt-in) ---
        # 默认只做本地规则抽取。LLM 事实抽取成本高，且容易把重写后的幻觉写入逻辑账本。
        if self.allow_llm_fact_extraction and len(text) > 100:
            try:
                sample = text[:1800]
                prompt = f"请提取本章的关键因果事实，输出严格的JSON格式。包含以下字段：\n" \
                         f"1. new_entities: [{{'type': '人物/道具/地点/秘密', 'name': '实体名', 'details': '描述'}}]\n" \
                         f"2. foreshadowing: [{{'clue': '埋下的伏笔描述', 'status': 'unresolved'}}]\n" \
                         f"3. resolved_foreshadowing: [{{'clue': '回收了之前的哪个伏笔'}}]\n" \
                         f"4. knowledge_updates: [{{'character': '角色名', 'topic': '知道的信息', 'status': 'known', 'context': '如何知道的'}}]\n" \
                         f"每类最多输出 8 条，缺失则输出空数组。\n正文节选：\n{sample}"
                sys_prompt = "你是一个强大的逻辑图谱抽取引擎，只输出纯JSON，不要任何markdown标记和思考过程。"

                res = generate_text_safe(
                    prompt,
                    system_prompt=sys_prompt,
                    model="deepseek-v4-flash",
                    thinking={"type": "disabled"},
                    temperature=0.1,
                    max_tokens=900,
                )
                if res:
                    data = extract_json_from_llm(res)
                    if data is None:
                        raise ValueError("无法从LLM响应中提取JSON结构")

                    with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                        for ent in data.get('new_entities', []):
                            conn.execute("INSERT INTO entities (book_id, chapter_name, entity_type, entity_name, details) VALUES (?, ?, ?, ?, ?)",
                                         (self.book_id, chapter_name, ent.get('type',''), ent.get('name',''), ent.get('details','')))
                        for f in data.get('foreshadowing', []):
                            conn.execute("INSERT INTO foreshadowing_ledger (book_id, setup_chapter, payoff_chapter, clue, status) VALUES (?, ?, ?, ?, ?)",
                                         (self.book_id, chapter_name, "", f.get('clue',''), f.get('status','unresolved')))
                        for f in data.get('resolved_foreshadowing', []):
                            conn.execute("UPDATE foreshadowing_ledger SET status='resolved', payoff_chapter=? WHERE clue LIKE ? AND book_id=?",
                                         (chapter_name, f"%{f.get('clue', '')}%", self.book_id))
                        for ku in data.get('knowledge_updates', []):
                            conn.execute("INSERT INTO character_knowledge_ledger (book_id, chapter_name, character_name, knowledge_topic, status, context) VALUES (?, ?, ?, ?, ?, ?)",
                                         (self.book_id, chapter_name, ku.get('character',''), ku.get('topic',''), ku.get('status',''), ku.get('context','')))
            except Exception as e:
                logger.error(f"LLM 因果抽取失败: {e}")

        # Information Asymmetry detection
        # Logic: If narrator explains something that NO character in scene could know, flag OMNISCIENCE_LEAK.
        omniscient_markers = [r"由于他不知道", r"他并不知道", r"在这个世界上没有人知道", r"正如他预料的一样", r"事实上"]
        for p in omniscient_markers:
            if re.search(p, text):
                self.conflicts.append(LogicConflict(
                    "OMNISCIENCE_LEAK",
                    f"叙事泄露：使用了上帝视角表达 '{p}'，破坏了角色沉浸感",
                    "P1",
                    [chapter_name]
                ))

        # Identity and Gender consistency Check
        if characters:
            paragraphs = text.split('\n')
            for char_info in characters:
                char_name = char_info["name"]
                base_gender = char_info.get("gender", "U")
                disguise = char_info.get("disguise", "none")

                if char_name in text and base_gender != "U":
                    m_count, f_count = 0, 0
                    for p in paragraphs:
                        if char_name in p:
                            m_count += p.count("他")
                            f_count += p.count("她")

                    total_pronouns = m_count + f_count
                    if total_pronouns >= 3:
                        if base_gender == "M" and f_count > m_count * 4:
                            if disguise != "男扮女装":
                                self.conflicts.append(LogicConflict(
                                    "GENDER_CONFLICT",
                                    f"人物性别错乱 / 伪装漏洞：角色 '{char_name}' 原设定为男性(用'他')，但在本章大量使用'她' (可能存在性别遗忘或未交代的男扮女装)。",
                                    "P0",
                                    [chapter_name]
                                ))
                        elif base_gender == "F" and m_count > f_count * 4:
                            if disguise != "女扮男装":
                                self.conflicts.append(LogicConflict(
                                    "GENDER_CONFLICT",
                                    f"人物性别错乱 / 伪装漏洞：角色 '{char_name}' 原设定为女性(用'她')，但在本章大量使用'他' (可能存在性别遗忘或未交代的女扮男装)。",
                                    "P0",
                                    [chapter_name]
                                ))

    def _check_item_conflict(self, chapter: str, item: str, new_status: str):
        last_status = None
        last_ch = None
        for f in reversed(self.ledger):
            if f.entity == item and f.type == "ITEM_STATUS":
                last_status = f.status
                last_ch = f.chapter
                break

        if last_status == "lost" and new_status == "possession":
            self.conflicts.append(LogicConflict(
                "TELEPORTATION",
                f"道具矛盾：{item}在{last_ch}已丢失，但在{chapter}突然出现",
                "P0",
                [last_ch, chapter]
            ))

    def save_ledger(self):
        """Save ONLY the current book's new facts to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            # Clear old and re-insert or incremental?
            # For robustness, we'll clear current book's facts and re-save full state
            conn.execute("DELETE FROM logic_facts WHERE book_id = ?", (self.book_id,))
            for f in self.ledger:
                conn.execute(
                    "INSERT INTO logic_facts (book_id, chapter_name, type, entity, status, description) VALUES (?, ?, ?, ?, ?, ?)",
                    (self.book_id, f.chapter, f.type, f.entity, f.status, f.description)
                )

    def load_ledger(self):
        """Load facts for the current book from SQLite."""
        self.ledger = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT chapter_name, type, entity, status, description FROM logic_facts WHERE book_id = ?",
                (self.book_id,)
            )
            for row in cursor:
                self.ledger.append(LogicFact(*row))

    def audit_continuity(self, chapters_data: List[Tuple[str, str]]):
        """Cross-chapter logic audit."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT setup_chapter, clue FROM foreshadowing_ledger WHERE status='unresolved' AND book_id=?", (self.book_id,))
            for row in cursor:
                self.conflicts.append(LogicConflict(
                    "UNRESOLVED_FORESHADOWING",
                    f"伏笔未回收：在 {row[0]} 埋下的伏笔 '{row[1]}' 直到大结局仍未解决。",
                    "P1",
                    [row[0], "大结局"]
                ))


class EvolutionEngine:
    """Self-evolution engine that learns from human corrections via SQLite rules."""
    def __init__(self, main_system: EnhancedNovelAISurgery, db_path: Path):
        self.main_system = main_system
        self.db_path = db_path
        self.knowledge = {"rules": [], "patterns": {}}
        self._init_db()
        self.load_knowledge()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    pattern TEXT,
                    weight REAL,
                    category TEXT,
                    severity TEXT,
                    frequency INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS style_benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT UNIQUE,
                    sensory_index REAL,
                    rhythm_variance REAL,
                    action_tag_density REAL,
                    top_keywords TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_evolution_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    pattern TEXT,
                    weight REAL,
                    category TEXT,
                    severity TEXT,
                    source TEXT,
                    status TEXT DEFAULT 'pending',
                    reason TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, pattern)
                )
            """)
            # Migration: Ensure frequency column exists for older database versions
            try:
                conn.execute("ALTER TABLE evolution_rules ADD COLUMN frequency INTEGER DEFAULT 1")
            except sqlite3.OperationalError:
                pass # Already exists

    def load_knowledge(self):
        self.knowledge = {"rules": [], "patterns": {}}
        # P0 Fix: 清除之前注入的进化规则，防止重复追加导致规则膨胀
        if hasattr(self, '_injected_rule_names'):
            self.main_system.tier1_rules = [r for r in self.main_system.tier1_rules if r.name not in self._injected_rule_names]
            self.main_system.all_rules = [r for r in self.main_system.all_rules if r.name not in self._injected_rule_names]
        self._injected_rule_names = set()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name, pattern, weight, category, severity, frequency FROM evolution_rules")
            for row in cursor:
                base_weight = row[2]
                freq = row[5]
                dynamic_weight = base_weight * (1.0 + (freq - 1) * 0.5)

                rule_dict = {
                    "name": row[0],
                    "pattern": row[1],
                    "weight": dynamic_weight,
                    "category": row[3],
                    "severity": row[4]
                }
                self.knowledge["rules"].append(rule_dict)
                self._injected_rule_names.add(row[0])
                rule = PatternRule(**rule_dict)
                self.main_system.compile_rule(rule)
                self.main_system.tier1_rules.append(rule)
                self.main_system.all_rules.append(rule)

    def set_book_bg(self, bg: str):
        self.current_bg = bg

    def get_book_bg(self) -> str:
        return getattr(self, "current_bg", "Historical")

    def mine_rules_with_llm(self, original_text: str, edited_text: str):
        """Use LLM to reverse-engineer human edits and generate new Regex rules."""
        if len(original_text) < 100 or len(edited_text) < 100:
            return

        prompt = f"对比以下AI原稿和人类修改稿，提取出人类集中删除的'AI感俗套词汇或句式'。\n" \
                 f"要求输出纯JSON格式：\n" \
                 f"[{{'name': '规则短名', 'pattern': 'Python Regex 表达式(用于匹配被删掉的AI套话)', 'category': '类型', 'severity': 'P1/P2'}}]\n\n" \
                 f"【原稿】\n{original_text[:1500]}\n\n" \
                 f"【修改稿】\n{edited_text[:1500]}"
        sys_prompt = "你是一个反向工程规则挖掘器，仅输出严格的JSON列表，不要markdown。"

        res = call_llm_with_fallback(prompt, sys_prompt)
        if res:
            try:
                rules = extract_json_from_llm(res)
                if rules is None:
                    raise ValueError("无法从LLM响应中提取JSON结构")
                if isinstance(rules, dict):
                    rules = [rules]
                for r in rules:
                    # 简单的沙盒验证：确保 pattern 能编译且不会匹配过于简单的字词
                    try:
                        pattern_str = r.get('pattern', '')
                        compiled = re.compile(pattern_str)
                        if len(pattern_str) > 3:
                            # 沙盒测试：如果在样本上匹配超过 5% 的长度，视为过泛规则并拒绝
                            matches = list(re.finditer(compiled, original_text[:1500]))
                            total_match_len = sum(len(m.group(0)) for m in matches)
                            if total_match_len > min(1500, len(original_text)) * 0.05:
                                logger.warning(f"沙盒拦截: 规则 '{r.get('name')}' ({pattern_str}) 匹配过多文本，可能引起暴走，已拒绝。")
                                continue

                            self.add_pending_rule(
                                r.get('name', 'llm_rule'),
                                pattern_str,
                                2.0,
                                r.get('category', 'llm_mined'),
                                r.get('severity', 'P1'),
                                source="llm_mined"
                            )
                    except re.error:
                        continue
            except Exception as e:
                logger.error(f"LLM 规则挖掘失败: {e}")

    def learn_from_pair(self, original_text: str, edited_text: str, allow_llm_mining: bool = False):
        """Analyze diff and extract improvements."""
        # 1. 经典硬编码规则验证
        patterns_to_check = [
            ("显然", r"显然", 2.0, "narration_overuse"),
            ("意味着", r"意味着", 2.0, "narration_overuse"),
            ("事实上", r"事实上", 2.5, "narration_overuse"),
            ("不得不说", r"不得不说", 1.5, "narration_overuse"),
            ("竟然", r"竟然", 1.0, "ai_artifact"),
            ("不由得", r"不由得", 1.2, "ai_artifact"),
            ("仿佛", r"仿佛", 0.5, "cliche"),
        ]

        for name, pattern, weight, cat in patterns_to_check:
            count_ori = len(re.findall(pattern, original_text))
            count_edt = len(re.findall(pattern, edited_text))

            if count_ori > count_edt:
                self.add_pending_rule(name, pattern, weight, cat, "P1", source="diff_static")

        # LLM mining is deliberately opt-in so tests and offline learning never
        # hit local LLM servers or API-key backed services by accident.
        if allow_llm_mining:
            self.mine_rules_with_llm(original_text, edited_text)

        # Candidate rules must survive positive/negative validation before activation.
        # Positive: original text (where AI artifacts appeared)
        # Negative: edited text (where artifacts should have been reduced)
        self.approve_pending_rules(
            positive_samples=[original_text[:3000]],
            negative_samples=[edited_text[:3000]],
        )

    def decay_rules(self):
        """Decrease rule weights over time if they are not triggered."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE evolution_rules SET frequency = frequency - 1 WHERE frequency > 1")

    def batch_learn_from_human_samples(self, directory: Path, allow_llm_mining: bool = False):
        """Walk through samples and learn from Original vs Human pairs or High Quality originals."""
        print(f"🧬 启动文学范式学习流程: {directory}")
        # 支持两种结构：
        # 1. duanpian_zhengwen/X/ (原文 vs 人工修改之后)
        # 2. fiveNovel/DATE/NOVEL_NAME/NOVEL_NAME.md (高质量原著)

        # 处理第一种：对改学习
        for folder in directory.glob("*/"):
            ori_p = folder / "原文.md"
            hum_p = folder / "人工修改之后.md"
            if ori_p.exists() and hum_p.exists():
                print(f"  - 学习对改样本: {folder.name}")
                self.learn_from_pair(
                    ori_p.read_text(encoding='utf-8', errors='ignore'),
                    hum_p.read_text(encoding='utf-8', errors='ignore'),
                    allow_llm_mining=allow_llm_mining,
                )

        # 处理第二种：原著学习 (fiveNovel 结构)
        # 假设用户传入的是 PycharmProjects/pythonProject/fiveNovel
        if "fiveNovel" in str(directory):
            print(f"  - 扫描高质量原著库...")
            # 查找所有 DATE 文件夹下的 NOVEL 文件夹下的 md
            novel_files = list(directory.glob("**/*.md"))
            for md_file in novel_files[:50]: # 限制前50本，防止撑爆
                if md_file.stem in str(md_file.parent.name):
                    print(f"    - 研读原著基因: {md_file.name}")
                    # 对于原著，我们不仅挖掘规则，还建立风格基准
                    stats = self.main_system.literary_researcher.analyze_novel(md_file)
                    self.save_benchmark(md_file.stem, stats)
                    # 也可以从中提取一些正向范式（待后续细化）

        self.decay_rules()
        self.load_knowledge()

    def validate_candidate_rule(
        self,
        name: str,
        pattern: str,
        positive_samples: Sequence[str] = (),
        negative_samples: Sequence[str] = (),
    ) -> Tuple[bool, str]:
        """Validate a mined regex before it can become an active rewrite rule."""
        if not name or not pattern or len(pattern) <= 3:
            return False, "empty_or_too_short"
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return False, f"regex_error:{e}"
        probe = "普通人说了几句话，转身走进屋里。风从窗缝钻进来，桌上的茶凉了。"
        if compiled.search(probe):
            return False, "matches_builtin_negative_probe"
        for sample in negative_samples:
            if compiled.search(sample):
                return False, "matches_negative_sample"
        if positive_samples and not any(compiled.search(sample) for sample in positive_samples):
            return False, "misses_positive_samples"
        return True, "ok"

    def add_pending_rule(
        self,
        name: str,
        pattern: str,
        weight: float,
        category: str,
        severity: str,
        source: str = "manual",
    ) -> bool:
        ok, reason = self.validate_candidate_rule(name, pattern)
        status = "pending" if ok else "rejected"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pending_evolution_rules
                (name, pattern, weight, category, severity, source, status, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, pattern, weight, category, severity, source, status, reason)
            )
        if ok:
            logger.info(f"🧪 新规则进入候选池: {name}")
        else:
            logger.warning(f"候选规则拒绝: {name} ({reason})")
        return ok

    def approve_pending_rules(
        self,
        positive_samples: Sequence[str] = (),
        negative_samples: Sequence[str] = (),
    ) -> int:
        """Promote validated pending rules into the active rule table."""
        promoted = 0
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, name, pattern, weight, category, severity FROM pending_evolution_rules WHERE status='pending'"
            ).fetchall()
            for rule_id, name, pattern, weight, category, severity in rows:
                ok, reason = self.validate_candidate_rule(name, pattern, positive_samples, negative_samples)
                if not ok:
                    conn.execute(
                        "UPDATE pending_evolution_rules SET status='rejected', reason=? WHERE id=?",
                        (reason, rule_id)
                    )
                    continue
                self._upsert_active_rule(conn, name, pattern, weight, category, severity)
                conn.execute(
                    "UPDATE pending_evolution_rules SET status='approved', reason='promoted' WHERE id=?",
                    (rule_id,)
                )
                promoted += 1
        if promoted:
            self.load_knowledge()
        return promoted

    def _upsert_active_rule(self, conn: sqlite3.Connection, name: str, pattern: str, weight: float, category: str, severity: str):
        try:
            conn.execute(
                "INSERT INTO evolution_rules (name, pattern, weight, category, severity, frequency) VALUES (?, ?, ?, ?, ?, 1)",
                (name, pattern, weight, category, severity)
            )
            logger.info(f"✨ 学习到新规则: {name}")
        except sqlite3.IntegrityError:
            conn.execute(
                "UPDATE evolution_rules SET frequency = frequency + 1 WHERE name = ?",
                (name,)
            )

    def add_evolution_rule(self, name: str, pattern: str, weight: float, category: str, severity: str):
        """Backward-compatible API: learned rules now enter the pending pool first."""
        return self.add_pending_rule(name, pattern, weight, category, severity, source="legacy_api")

    def force_add_evolution_rule(self, name: str, pattern: str, weight: float, category: str, severity: str):
        with sqlite3.connect(self.db_path) as conn:
            self._upsert_active_rule(conn, name, pattern, weight, category, severity)

    def save_benchmark(self, label: str, stats: Dict[str, object]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO style_benchmarks
                (label, sensory_index, rhythm_variance, action_tag_density, top_keywords)
                VALUES (?, ?, ?, ?, ?)
            """, (label, stats.get('sensory',0), stats.get('variance',0), stats.get('action_density',0), json.dumps(stats.get('keywords',[]), ensure_ascii=False)))

    def load_benchmark(self, label: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT sensory_index, rhythm_variance, action_tag_density, top_keywords FROM style_benchmarks WHERE label = ?", (label,))
            row = cursor.fetchone()
            if row:
                return {
                    "sensory": row[0], "variance": row[1], "action_tag": row[2], "keywords": json.loads(row[3])
                }
        return None


class LiteraryResearcher:
    """Specialized engine to scan master novels and extract literary gene sets."""
    def __init__(self):
        self.sensory_lexicon = [
            "闻到", "嗅", "血腥", "檀香", "烧焦", "汗水", "潮湿", "冰冷", "滚烫", "刺痛",
            "麻木", "粗糙", "细腻", "滑腻", "震颤", "跳动", "苦涩", "甜腻", "辣", "咸"
        ]
        self.benchmark_engine = StyleBenchmarkEngine()

    def analyze_novel(self, path: Path, sample_limit=500000) -> Dict:
        content = path.read_text(encoding='utf-8', errors='ignore')[:sample_limit]
        words_count = len(content)
        sentences = re.split(r'[。！？\n]', content)
        lens = [len(s) for s in sentences if len(s) > 1]

        # 1. Sensory Index
        hits = 0
        for s in self.sensory_lexicon:
            hits += content.count(s)
        sensory_index = (hits / (words_count / 1000.0 + 1)) * 10

        # 2. 深度风格分析 (使用统一基准引擎)
        stats = self.benchmark_engine.analyze_masterpiece(content)
        variance = stats["sentence_var"]

        # 3. Dialogue Dynamics (对白张力与烟火气)
        # 统计含有“动作/神态描述”的对白比例
        total_quotes = len(re.findall(r'[“][^”]+[”]', content))
        action_quotes = len(re.findall(r'[“][^”]+[”][^。！？\n]*?[，。][^。！？\n]*?[\u4e00-\u9fff]{2,}', content))
        dialogue_action_ratio = action_quotes / max(total_quotes, 1)

        # 4. Action Chain Density (动作链深度)
        # 寻找连续出现的动词结构 (如：抬手一挥、翻身跃起)
        action_chains = len(re.findall(r'[\u4e00-\u9fff]{1,2}(?:了|着|起|下|开|过)[\u4e00-\u9fff]{1,2}(?:了|着|起|下|开|过)?', content))
        action_density = (action_chains / (words_count / 1000.0 + 1))

        # 5. Vocabulary DNA (高频特征词)
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', content)
        common_words = [w for w, c in Counter(words).most_common(50) if len(w) >= 2]

        return {
            "sensory": round(sensory_index, 3),
            "variance": round(variance, 3),
            "dialogue_action_ratio": round(dialogue_action_ratio, 3),
            "action_density": round(action_density, 3),
            "keywords": common_words
        }


class NovelContextAnalyzer:
    """Heuristic-based genre and background detection."""
    def __init__(self):
        self.genre_keywords = {
            "FemaleLead": ["王爷", "摄政王", "嫡女", "王妃", "娇妻", "宅斗", "束身绫", "宫廷", "重生", "渣男"],
            "MaleLead": ["系统", "战神", "赘婿", "修仙", "升级", "不朽", "宗门", "杀伐", "境界", "突破", "乾坤"],
        }
        self.background_keywords = {
            "Historical": ["宗庙", "朝堂", "皇帝", "大梁", "龙袍", "明黄", "绢帛", "将领", "江山"],
            "Urban": ["手机", "写字楼", "跑车", "总裁", "合租", "医院", "股票", "咖啡"],
            "Fantasy": ["灵力", "丹药", "飞剑", "妖兽", "神识", "阵法", "元神"],
        }

    def analyze(self, text: str) -> NovelContext:
        sample = text[:10000]
        genre_scores = {g: sum(sample.count(kw) for kw in kws) for g, kws in self.genre_keywords.items()}
        bg_scores = {b: sum(sample.count(kw) for kw in kws) for b, kws in self.background_keywords.items()}

        # Ambiguity Detection
        g_sorted = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        if g_sorted[0][1] > 0 and g_sorted[0][1] < g_sorted[1][1] * 1.5:
             # If top two genres are too close, it's ambiguous
             raise ValueError(f"⚠️ 频道识别模糊: {genre_scores}。请手动在配置中指定频道。")

        genre = g_sorted[0][0] if g_sorted[0][1] > 0 else "Unknown"
        bg = max(bg_scores.items(), key=lambda x: x[1])[0] if max(bg_scores.values()) > 0 else "Unknown"

        return NovelContext(
            book_id="detected_book",
            genre=genre,
            background=bg,
            tone="Serious" if genre == "FemaleLead" else "Action-packed",
            word_count_target=len(text)
        )


class EntityTracker:
    """Manages characters.json and entity persistence per book."""
    def __init__(self, root_dir: Path):
        self.db_dir = root_dir / "db" / "books"
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def get_char_path(self, book_id: str) -> Path:
        return self.db_dir / f"{book_id}_characters.json"

    def load_characters(self, book_id: str) -> List[Dict[str, str]]:
        path = self.get_char_path(book_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return []

    def save_characters(self, book_id: str, characters: List[Dict[str, str]]):
        path = self.get_char_path(book_id)
        path.write_text(json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8")

    def extract_from_text(self, text: str, existing: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # Common pronouns and words to exclude from character extraction
        stop_names = {"我", "他", "她", "谁", "你", "咱们", "之类", "一个", "这种", "那种", "朕", "本王", "摄政王", "主子", "甚至", "却是", "依然", "就像", "仿佛", "原本", "依旧", "这样", "那样", "那里", "这里", "此时", "就在", "然而", "但是", "所以", "因此", "因为"}

        # Simple heuristic extraction: Names (often 2-3 chars in Chinese) followed by dialogue verbs
        # Using a slightly more restrictive match for the preceding characters
        matches = re.findall(r"([\u4e00-\u9fff]{2,3})(?:道|说|问|喊|沉声|冷笑|开口|唤道|低语|咆哮|呵斥|轻啐|冷哼|暴喝|阴阳怪气|不屑|不语)", text)
        new_chars = [c for c in set(matches) if c not in stop_names and len(c) >= 2]

        updated = {c["name"]: c for c in existing}

        paragraphs = text.split('\n')

        for name in new_chars:
            if name not in updated:
                updated[name] = {"name": name, "role": "detected", "desc": "自动识别角色", "gender": "U", "disguise": "none"}

        for char_name, char_info in updated.items():
            if char_name in text:
                has_f_to_m = ("女扮男装" in text and char_name in text) or (f"{char_name}女扮男装" in text)
                has_m_to_f = ("男扮女装" in text and char_name in text) or (f"{char_name}男扮女装" in text)

                if has_f_to_m:
                    char_info["disguise"] = "女扮男装"
                    char_info["gender"] = "F"
                elif has_m_to_f:
                    char_info["disguise"] = "男扮女装"
                    char_info["gender"] = "M"

                m_count, f_count = 0, 0
                for p in paragraphs:
                    if char_name in p:
                        m_count += p.count("他")
                        f_count += p.count("她")

                if char_info.get("gender", "U") == "U":
                    if m_count > f_count * 2 and m_count >= 2:
                        char_info["gender"] = "M"
                    elif f_count > m_count * 2 and f_count >= 2:
                        char_info["gender"] = "F"

        return list(updated.values())


class AgentRefinementTask:
    """Generates high-precision Multi-Agent refinement tasks."""
    def __init__(
        self,
        root_dir: Path,
        llm_caller: Optional[Callable[[str, str], Optional[str]]] = None,
        include_full_text: bool = False,
        max_reference_chars: int = 500,
    ):
        self.root_dir = root_dir
        self.output_dir = root_dir / "agent_tasks"
        # P0 Fix: Ensure the task directory is fresh every time to avoid stale logic interference
        if self.output_dir.exists():
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                logger.warning(f"无法清空 agent_tasks 目录: {e}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = root_dir / "templates" / "fixAI_template.md"
        self.llm_caller = llm_caller
        self.include_full_text = include_full_text
        self.max_reference_chars = max_reference_chars

    @staticmethod
    def _local_conflict_summary(text: str) -> Dict[str, str]:
        conflict_words = "杀|死|毒|债|夺|抢|背叛|退婚|断绝|威胁|秘密|真相|冤|骗|失踪|死亡|血|账|钱"
        sentences = EnhancedNovelAISurgery.split_sentences(text[:1200])
        for sentence in sentences:
            if re.search(conflict_words, sentence):
                return {"conflict": sentence[:120], "hook": "本地审计模式：不生成新楔子，避免额外 LLM 消耗。"}
        first = sentences[0][:120] if sentences else text[:120]
        return {"conflict": first or "未检测到明显冲突句", "hook": "本地审计模式：不生成新楔子，避免额外 LLM 消耗。"}

    def _build_reference_block(self, text: str) -> str:
        if self.include_full_text:
            return "\n## 参考全文内容\n```text\n" + text + "\n```\n"
        head = text[: self.max_reference_chars]
        tail = text[-300:] if len(text) > self.max_reference_chars + 600 else ""
        parts = [
            "\n## 原文定位说明",
            "默认不内联全文，避免把 3 万字小说反复塞进后续模型上下文。",
            "如需 IDE/人工二次修，请打开源文件或输出文件，按下方段落/句子编号局部处理。",
            "\n### 开头节选",
            "```text",
            head,
            "```",
        ]
        if tail:
            parts.extend(["\n### 结尾节选", "```text", tail, "```"])
        return "\n".join(parts) + "\n"

    def generate_task(self, chapter_name: str, text: str, context: NovelContext, changes: List[SentenceChange], structure_issues=None) -> Path:
        char_str = ", ".join([c["name"] for c in context.characters])

        # Build precise diagnostic micro-instructions
        instructions = []
        instructions.append(f"# 精准改写任务: {chapter_name}")
        instructions.append(f"> **目标执行者**: IDE 内置大模型助手或人工编辑")
        instructions.append(f"> **指令**: 只根据诊断做局部修复。不要整章重写，不要扩写，不要把已通顺的句子重新润色一遍。\n")
        instructions.append(f"频道: {context.genre} | 背景: {context.background}")
        instructions.append(f"出场角色: {char_str}\n")

        # --- V14 语义冲突提取 ---
        # 默认本地抽取。LLM 钩子生成会额外消耗整段上下文，必须由调用方显式启用。
        if self.llm_caller:
            hook_data = SemanticConflictExtractor(llm_caller=self.llm_caller).extract_and_generate_hook(text)
        else:
            hook_data = self._local_conflict_summary(text)
        instructions.append("## 核心冲突与开幕雷击 (由语义引擎自动提取)")
        instructions.append(f"> **本章核心冲突**: {hook_data.get('conflict', '提取失败')}")
        instructions.append(f"> **建议开篇钩子 (楔子)**: \n> {hook_data.get('hook', '提取失败')}\n")
        instructions.append("> [!IMPORTANT]")
        instructions.append("> **改写红线：局部最小改动 + 长度一致性 (Content Parity)**")
        instructions.append(f"> 本章原文字数：{len(text)} 字符。")
        instructions.append("> **严禁任何形式的剧情浓缩、换梗、换人设或总结式重写。**")
        instructions.append("> 自动修复以修逻辑链、人物动机、道具/数字一致性和明显 AI 句式为主；没有诊断的问题不要动。\n")


        if structure_issues:
            instructions.append("## 结构级高危红线 (请在全局优化时重点修正)")
            for iss in structure_issues:
                instructions.append(f" - ⚠️ {iss}")
            instructions.append("\n")

        instructions.append("## 句子级微指令 (由审计引擎截获的目标片段)")
        if not changes:
            instructions.append("暂无高危句子级需要改写，请全局核实逻辑。")
        else:
            for idx, chg in enumerate(changes[:20]): # 限制任务单长度，避免诊断本身耗费过多 token
                diagnostic_str = ' | '.join(chg.applied_rules)
                is_suspended = "[悬浮警报]" in diagnostic_str or "直给情绪" in diagnostic_str

                lines = [
                    f"### Target {idx + 1}: Paragraph {chg.paragraph}, Sentence {chg.sentence}",
                    f"**[原句]** {chg.original}",
                    f"**[诊断]** {diagnostic_str}",
                ]

                if is_suspended:
                    lines.extend([
                        f"**[IDE Agent 强制文风预设]**: 启动“烽火戏诸侯/长短句张力风”改写方案。",
                        f" 1. 节奏打散：绝对禁止排比式、对称式联结。必须采用 3-5 字的极短短句与超长句进行极端交错断裂。",
                        f" 2. 脱离悬浮：禁止使用任何主观或哲学抽象词(如“痛苦、绝望、生理极限”)，必须替换为粗粝的底层生理体征(如胃酸反流、手指发麻、眼前泛黑)。"
                    ])
                else:
                    lines.extend([
                        f"**[IDE Agent 要求]**",
                        f" 1. 消除以上诊断出的问题。",
                        f" 2. 融入感官锚点（触觉、嗅觉、痛觉等），不要仅写视觉。",
                        f" 3. 避免过度平稳的句式，尝试打破句长对称。"
                    ])
                instructions.append("\n".join(lines))

        instructions.append(self._build_reference_block(text))

        task_path = self.output_dir / f"refine_{chapter_name}.md"
        task_path.write_text("\n".join(instructions), encoding="utf-8")
        return task_path

class EnhancedNovelAISurgery:
    def __init__(
        self,
        root_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
        vector_db_path: Optional[str] = None,
        book_id: str = "default",
        allow_llm_task_hooks: bool = False,
        allow_llm_fact_extraction: bool = False,
        allow_llm_finalize: bool = False,
        enable_global_index: bool = False,
        task_include_full_text: bool = False,
        max_chunk_size: int = 3500,
        llm_mode: str = "patch",
        llm_max_patches: int = 8,
        llm_context_paragraphs: int = 1,
        llm_token_budget: int = 6000,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).parent
        self.db_path = Path(db_path) if db_path else self.root_dir / "db" / "novel_logic.db"
        self.vector_db_path = vector_db_path or str(self.root_dir / "db" / "vector_db")
        self.book_id = book_id
        self.allow_llm_task_hooks = allow_llm_task_hooks
        self.allow_llm_fact_extraction = allow_llm_fact_extraction
        self.allow_llm_finalize = allow_llm_finalize
        self.enable_global_index = enable_global_index
        self.task_include_full_text = task_include_full_text
        self.max_chunk_size = max(800, int(max_chunk_size or 3500))
        self.llm_mode = llm_mode if llm_mode in {"patch", "full"} else "patch"
        self.llm_max_patches = max(1, int(llm_max_patches or 8))
        self.llm_context_paragraphs = max(0, int(llm_context_paragraphs or 0))
        self.llm_token_budget = max(1200, int(llm_token_budget or 6000))
        self.task_llm_caller = call_llm_with_fallback if allow_llm_task_hooks else None
        self.finalize_llm_caller = call_llm_with_fallback if allow_llm_finalize else None
        self.logic_auditor = LogicAuditor(self.db_path, allow_llm_fact_extraction=allow_llm_fact_extraction)
        self.logic_auditor.set_book_context(self.book_id)
        self.context_analyzer = NovelContextAnalyzer()
        self.entity_tracker = EntityTracker(self.root_dir)
        self.agent_task_gen = AgentRefinementTask(
            self.root_dir,
            llm_caller=self.task_llm_caller,
            include_full_text=task_include_full_text,
        )
        self.literary_researcher = LiteraryResearcher()
        self.style_profile = None  # Init missing variable
        # Advanced analyzers
        self.rhythm_shatter = RhythmShatter()
        self.sensory_weaver = SensoryWeaver()
        self.redundancy_detector = RedundancyDetector()
        self.pov_checker = POVChecker()
        self.dialogue_analyzer = DialoguePowerAnalyzer()
        self.sdt_transformer = ShowDontTellTransformer()
        self.tail_tag_detector = ExplanatoryTailTagDetector()
        self.voice_differentiator = CharacterVoiceDifferentiator()
        self.micro_imperfection = MicroImperfectionGenerator()
        self.identity_summary_detector = IdentitySummaryDetector()
        self.cliche_express_detector = ClicheExpressDetector()
        self.cliche_express_detector = ClicheExpressDetector()
        self.prologue_hook = PrologueHookAnalyzer()
        self.semantic_hook_engine = SemanticConflictExtractor(llm_caller=self.task_llm_caller)
        self.global_hook_engine = GlobalVisionHooker(
            db_path=self.vector_db_path if enable_global_index else None,
            book_id=self.book_id,
            llm_caller=self.task_llm_caller,
        )
        self.causality_auditor = NarrativeCausalityAuditor()
        self.conjunction_neutering = ConjunctionNeutering()
        self.style_benchmark = StyleBenchmarkEngine()
        self.content_parity_validator = ContentParityValidator()

        # -----------------------------
        # Tier 1: Always replace (高频 AI 味 / 模板感极强)
        # -----------------------------
        self.tier1_rules: List[PatternRule] = [
            # 写作范式类 (P1) - 拦截排比平衡感，鼓励非对称性
            PatternRule("不是A而是B", r"不是[^。！？；\n]{1,20}?(?:而是|也不会)[^。！？；\n]{1,24}", 3.0, "syntax_repeat", "P1", 1),
            PatternRule("与其说A不如说B", r"与其说[^。！？；\n]{1,24}?不如说[^。！？；\n]{1,24}", 3.0, "syntax_repeat", "P1", 1),
            PatternRule("这意味着", r"这意味着", 2.0, "narration_overuse", "P1", 1),
            PatternRule("显然", r"显然", 1.5, "narration_overuse", "P1", 1),
            PatternRule("并非而是", r"并非[^。！？；\n]{1,18}而是[^。！？；\n]{1,18}", 2.2, "narration_overuse", "P1", 1),
            PatternRule("不难看出", r"不难看出|显而易见", 2.0, "narration_overuse", "P1", 1),
            PatternRule("所谓的", r"所谓的", 2.5, "naming_trope", "P1", 1),
            # --- V7 新增：AI 结构级指纹 ---
            PatternRule("名为X的Y抽象贴标签", r"名为[^。！？；\n]{1,15}的(?:影子|怪物|棋子|存在|世界|深渊|噩梦|恩赐|善意|恐惧|绝望|毒|重担|陷阱|圈套|幻象|枷锁|牢笼|债务|利息|账本|阴影|黑洞|荒诞|真相|死寂|温柔|光亮)", 3.0, "naming_trope", "P1", 1),
            PatternRule("概念性双动词并行", r"([\u4e00-\u9fff]{1,2})[^。！？；\n]{4,30}[，,]\1[^。！？；\n]{4,30}[。！？]", 2.8, "echo_parallel", "P1", 1),
            PatternRule("一种名为X的Y", r"一种(?:名为|称为|叫做)[^。！？；\n]{1,10}的", 3.0, "naming_trope", "P1", 1),
            PatternRule("这便是", r"这便是|这就是", 2.0, "narration_overuse", "P1", 1),
            PatternRule("这种", r"这种(?:感觉|氛围|逻辑|状态|局面|这种)", 1.5, "narration_overuse", "P1", 1),

            # 生活化细节检测 (P2) - 拦截过于“精致”的描述，鼓励烟火气
            PatternRule("精致过度", r"(?:极其|表现得|分外|无不)(?:优雅|高贵|精致|完美)", 1.5, "style_over_substance", "P2", 1),
            PatternRule("解析式叙述", r"(?:透过|穿过)[^，。]{1,10}(?:可以看到|可以发现|不难察觉)", 2.0, "omniscience_trope", "P1", 1),

            # 陈词滥调类 (P1/P2) - 深度清理 AI 常用生理反应
            PatternRule("指尖泛白", r"(?:指节|指尖|指头|指甲|骨节)(?:[^\n]{0,3}?)?(?:泛白|发白|收紧|青筋暴起)", 2.5, "phrase_repeat", "P1", 1),
            PatternRule("取而代之", r"取而代之(?:的|地)?(?:是)?", 2.5, "phrase_repeat", "P1", 1),
            PatternRule("这种感觉", r"(?:极其|感到|显得)(?:阴冷|燥热|恶心|窒息|尖锐|隐秘)(?:的)?(?:感觉|感|感触)", 2.0, "ai_metaphor", "P1", 1),
            PatternRule("眸色微沉", r"(?:眸色|眼神|神色)微沉", 2.0, "phrase_repeat", "P1", 1),
            PatternRule("呼吸一滞", r"呼吸一滞", 2.0, "phrase_repeat", "P1", 1),
            PatternRule("心头一震", r"心头一震", 2.0, "phrase_repeat", "P1", 1),
            PatternRule("嘴角弧度", r"嘴角(?:勾起|扬起|浮现)(?:一抹)?(?:笑意|弧度|冷笑)", 2.2, "phrase_repeat", "P1", 1),
            PatternRule("AI式比喻", r"(?:像|如同|仿佛)[^。！？]{2,20}(?:生锈|锋利|撕裂|铁块|刀割|风干|剔骨|烧红|铁丝|石般)", 2.5, "ai_metaphor", "P1", 1),

            # --- V8 新增：深度消除AI抽象总结与极端烂俗比喻 ---
            PatternRule("抽象极限宣称", r"(?:已经|甚至)?(?:超越|突破|打破)了[^。！？；\n]{1,10}(?:极限|界限|底线)", 3.0, "abstract_summary", "P1", 1),
            PatternRule("双重空洞修饰抽象名词", r"变成了一种[^。！？；\n]{2,4}的、[^。！？；\n]{2,4}的(?:痛苦|折磨|恐惧|绝望|执念|悲伤|愤怒|死寂)", 3.0, "naming_trope", "P1", 1),
            PatternRule("反常态转折式比喻", r"(?:没觉得|没有感到)[^。！？；\n]{1,10}[，,](?:反倒|反而)(?:像是|如同|更像是)[^。！？；\n]{2,15}", 3.0, "syntax_repeat", "P1", 1),
            PatternRule("烂俗动物比喻", r"(?:缩成|像|如同|仿佛)[^。！？；\n]{0,5}(?:一只|一头)(?:受惊的|受伤的|暴怒的|陷入绝境的)(?:虾米|兔子|野兽|小鹿|幼兽|母狼|刺猬|羔羊)", 2.8, "ai_metaphor", "P1", 1),
            PatternRule("过度发力的副词模板", r"(?:猛地|狠狠(?:地)?|死死(?:地)?)(?:撞在|砸向|缩|咬住|盯)", 2.0, "phrase_repeat", "P1", 1),

            # --- V10 新增：刻板修辞与油腻微表情 (针对第四批顽固AI味) ---
            PatternRule("面具化微表情", r"(?:干裂的嘴唇)?(?:扯|勾|拉)(?:出|起|出一个).*?的(?:不太明显|若有若无|冰冷)?的?(?:弧度|冷笑|笑意)", 3.0, "phrase_repeat", "P1", 1),
            PatternRule("刻板感官堆砌", r"(?:铁锈般|劣质的).*?(?:腥甜|焚香|气味|味道)", 3.0, "ai_metaphor", "P1", 1),
            PatternRule("做作力度词", r"死死(?:堵在|扣进|抠进|盯住|掐住|堵住)", 2.0, "phrase_repeat", "P1", 1),
            PatternRule("刻板声音比喻", r"(?:像是在|如同)?砂纸上打磨过(?:的钝器)?", 3.0, "ai_metaphor", "P1", 1),

            # --- V12 新增：结构性排比、说教与过度总结 (针对高智感但模式化的 AI 味) ---
            PatternRule("取而代之滥用", r"(?:消失了|没有了|换下了)[^。！？；\n]{0,10}[，,]取而代之的(?:是)?", 2.8, "phrase_repeat", "P1", 1),
            PatternRule("强行否定与强调", r"(?:那|这)不是[^。！？；\n]{1,15}[，,。](?:而)?是[^。！？；\n]{2,30}", 3.0, "syntax_repeat", "P1", 1),
            PatternRule("连珠炮式否定", r"(?:这|那)不是[^。！？；\n]{2,10}。[这|那]不是[^。！？；\n]{2,10}。[这|那]是[^。！？；\n]{2,20}", 3.5, "syntax_repeat", "P1", 1),
            PatternRule("强行顿悟总结", r"(?:终于|才真正)意识到[，,][^。！？；\n]{2,30}失去的不仅仅是", 3.0, "abstract_summary", "P1", 1),
            PatternRule("反常态审视对比", r"(?:不仅)?没有(?:被)?[^。！？；\n]{2,15}[，,]反而(?:带着|一直在)[^。！？；\n]{2,15}", 2.8, "syntax_repeat", "P1", 1),
            PatternRule("讲道理式硬核总结", r"(?:更重要的是[，,]这说明|比起[^，,]+[，,][^，,]+更符合[^。！？；\n]+逻辑)", 3.0, "narration_overuse", "P1", 1),
            PatternRule("烂俗悬疑或终局宣告", r"(?:这一场局|这个局|一切)[，,]才刚刚开始|已经彻底死透了", 3.0, "cliche", "P1", 1),
            PatternRule("投名状滥用", r"这是(?:我|他|她)(?:给|送|献)?(?:的)?投名状", 2.8, "cliche", "P1", 1),
            PatternRule("烂俗抽象气息杂糅", r"(?:透着|带着)一(?:股|股子)[^。！？；\n]{2,15}的(?:气|气息|味道|血腥气|威慑|眼睛)", 2.5, "ai_metaphor", "P1", 1),
            PatternRule("工具人侧面衬托模式", r"这种[^。！？；\n]+若是(?:让|换做)(?:寻常|普通)[^。！？；\n]{2,10}(?:听了|看了)[，,][^。！？；\n]+怕是要[^。！？；\n]+", 3.0, "cliche", "P1", 1),
            PatternRule("俗套濒死比喻", r"濒死的鱼", 3.0, "ai_metaphor", "P1", 1),
            PatternRule("解释性连续暗喻", r"(?:那是在看|也是看|这分明是看)(?:一个|一只|一块).*?[，,。](?:一个|一只|一块|一条).*?", 3.0, "echo_parallel", "P1", 1),
            PatternRule("物化修辞狂热", r"(?:犹如|宛如|就像|仿佛是一(?:只|个|头|块)).*?的(?:丹鼎|灵石|羔羊|布偶|宠物)", 3.0, "ai_metaphor", "P1", 1),

            # --- V9 新增：解释性尾注 (Explanatory Tail Tag) ---
            # AI最隐蔽的指纹：前半句具体生理/动作描写 + 破折号/逗号 + 「那是/这是」 + 抽象概念标签
            # 病灶: 替读者解码情感，破坏文学张力
            # ❌ 大腿根部的肌肉在打颤——那是原主身体残余的恐惧
            # ❌ 全身的力气都往腰腹上使，那是杀红了眼的蛮劲
            # ✅ 大腿根部的肌肉在打颤。（句号收住，让颤抖自己说话）
            PatternRule("破折号解释尾注", r"——(?:那是|这是|那不是|这不是|这便是|那便是)[^。！？；\n]{2,30}(?:的(?:恐惧|恐慌|恐怖|愤怒|不甘|绝望|本能|警觉|直觉|习惯|渴望|执念|蛮劲|倔强|骄傲|尊严|底气|底线|温柔|善意|信号|宣言|挑衅|威胁|承诺|决心|觉悟|勇气|信任|默契|暗示|试探|妥协|退让|挣扎|不安|焦虑|痛苦|悲伤|孤独|释然|报复|反击|示弱|示好|求救))", 3.0, "explanatory_tail_tag", "P1", 1),
            PatternRule("逗号解释尾注", r"[，,](?:那是|这是|这都是|那都是|那便是|这便是)[^。！？；\n]{2,30}(?:的(?:恐惧|恐慌|愤怒|不甘|绝望|本能|警觉|直觉|习惯|蛮劲|倔强|骄傲|底气|底线|温柔|挑衅|威胁|决心|觉悟|不安|痛苦|悲伤|挣扎|反击|试探|妥协|退让))", 2.8, "explanatory_tail_tag", "P1", 1),
            PatternRule("某种抽象尾注", r"[——，,](?:那是|这是)(?:某种|一种|属于|来自)[^。！？；\n]{1,20}(?:的(?:力量|感觉|情绪|反应|本能|冲动|信号|暗示|直觉|预感|征兆))", 2.5, "explanatory_tail_tag", "P1", 1),
            PatternRule("生理反应解释尾注", r"(?:打颤|发抖|颤抖|颤栗|哆嗦|冷汗|鸡皮疙瘩|瞳孔[^。]{0,4}(?:收缩|放大)|心跳[^。]{0,4}(?:加速|漏拍)|呼吸[^。]{0,4}(?:急促|粗重))[^。！？；]{0,10}[——，,](?:那是|这是|这便是|那便是)[^。！？；\n]{2,25}", 3.2, "explanatory_tail_tag", "P1", 1),
            PatternRule("动作概括尾注", r"(?:咬着牙|攥紧拳|握紧|绷紧|僵住|愣住|定住|蜷缩|后退|弓腰|弯腰|咽了口唾沫)[^。！？；]{0,15}[——，,](?:那是|这是|那种|这种)[^。！？；\n]{2,20}(?:蛮劲|劲|力道|姿态|架势|反应|惯性|条件反射)", 2.8, "explanatory_tail_tag", "P1", 1),

            # 结构填充类 (P2)
            PatternRule("顿了顿", r"顿了顿", 1.0, "transition_overuse", "P2", 1),
            PatternRule("沉默片刻", r"(?:沉默了片刻|静了片刻|沉默片晌|半晌|良久)", 1.0, "transition_overuse", "P2", 1),
            PatternRule("目光落在", r"目光落在", 1.0, "transition_overuse", "P2", 1),

            # --- V13 新增：高频 AI 转折词与套话拦截 (P2 Fix #10) ---
            PatternRule("句首然而", r"^然而[，,]", 2.5, "transition_overuse", "P1", 1),
            PatternRule("尽管但是", r"尽管[^。！？；\n]{1,25}[，,]但[^。！？；\n]{1,25}", 2.0, "syntax_repeat", "P1", 1),
            PatternRule("不禁", r"不禁", 1.5, "ai_artifact", "P1", 1),
            PatternRule("下意识地", r"下意识(?:地|的)", 1.2, "ai_artifact", "P1", 1),
            PatternRule("几乎是本能", r"几乎(?:是)?(?:出于)?本能(?:地|的)?", 2.0, "ai_artifact", "P1", 1),
            PatternRule("带着一种…的", r"带着一种[^。！？；\n]{2,15}的", 2.5, "naming_trope", "P1", 1),
            PatternRule("在这一刻", r"(?:在这一刻|在那一瞬间|在这个瞬间|就在这时)", 2.0, "transition_overuse", "P1", 1),
            PatternRule("某种程度上", r"某种程度上", 2.0, "narration_overuse", "P1", 1),
            PatternRule("不知为何", r"不知为何", 1.5, "ai_artifact", "P1", 1),
            PatternRule("说不清道不明", r"说不清道不明", 1.8, "ai_artifact", "P1", 1),
            PatternRule("或者说", r"[，,]或者说[，,]", 1.5, "narration_overuse", "P1", 1),
            # 段落结构级 AI 模式 (P2 Fix #11 - 轻量版)
            PatternRule("尾句升华总结", r"[。！？](?:这大概就是|这或许就是|这便是|也许这就是)[^。！？；\n]{2,30}(?:的含义|的意义|的代价|的真谛|的答案)", 3.0, "abstract_summary", "P1", 1),
            PatternRule("模板式心理独白", r"(?:她|他)(?:突然|忽然)?(?:意识到|明白了|懂了)[，,][^。！？；\n]{2,40}(?:不过如此|原来如此|竟然如此|就是这样)", 2.5, "ai_artifact", "P1", 1),
        ]

        # -----------------------------
        # Tier 2: Flag in clusters (当一段中出现 2 个以上时标记为 P1)
        # -----------------------------
        self.tier2_rules: List[PatternRule] = [
            PatternRule("驾驭", r"驾驭", 1.5, "advanced_word", "P1", 2),
            PatternRule("赋能", r"赋能", 2.0, "advanced_word", "P1", 2),
            PatternRule("培育", r"培育", 1.5, "advanced_word", "P1", 2),
            PatternRule("提升", r"提升(?!级)", 0.8, "advanced_word", "P1", 2),
            PatternRule("引起共鸣", r"引起共鸣", 1.8, "advanced_word", "P1", 2),
            PatternRule("释放", r"释放", 1.2, "advanced_word", "P1", 2),
            PatternRule("精简", r"精简", 1.5, "advanced_word", "P1", 2),
            PatternRule("多维度", r"多维度", 1.5, "advanced_word", "P1", 2),
            PatternRule("不仅...更...", r"不仅(?:仅)?(?:是)?[^。！？；\n]{1,20}更是[^。！？；\n]{1,24}", 2.0, "syntax_repeat", "P1", 2),
        ]

        # -----------------------------
        # Tier 3: Flag by density (密度检测 P2)
        # -----------------------------
        self.tier3_rules: List[PatternRule] = [
            PatternRule("显著", r"显著", 0.5, "density_word", "P2", 3),
            PatternRule("有效", r"有效", 0.5, "density_word", "P2", 3),
            PatternRule("卓越", r"卓越", 0.5, "density_word", "P2", 3),
            PatternRule("引人入胜", r"引人入胜", 0.8, "density_word", "P2", 3),
            PatternRule("前所未有", r"前所未有", 1.0, "density_word", "P2", 3),
            PatternRule("仅仅", r"仅仅", 0.3, "density_word", "P2", 3),
            PatternRule("甚至", r"甚至", 0.3, "density_word", "P2", 3),
        ]

        # -----------------------------
        # P0: Critical AI artifacts (模型残留 / 极其生硬的翻译感)
        # -----------------------------
        self.p0_rules: List[PatternRule] = [
            PatternRule("作为AI", r"(?:作为|我是一个)?(?:AI|人工智能|语言模型)", 5.0, "ai_hallucination", "P0", 1),
            PatternRule("希望这能帮到你", r"(?:希望|期待)这?(?:能|可以)?帮(?:到|助)(?:你|您)", 5.0, "ai_artifact", "P0", 1),
            PatternRule("景观(隐喻)", r"(?:政治|行业|社会|文化)(?:景观|风景线)", 3.0, "translation_flavor", "P0", 1),
            PatternRule("深入探讨", r"深入(?:探讨|剖析|洞察)", 3.0, "translation_flavor", "P0", 1),
            PatternRule("总而言之/总之", r"总而言之|总之|综上所述", 2.0, "ai_artifact", "P0", 1),
            PatternRule("百科式连接词", r"综上所述|不可忽视的是|值得注意的是|在当今社会|显而易见|总的来说|首先|其次|再次|最后", 3.0, "ai_artifact", "P0", 1),
        ]

        self.all_rules: List[PatternRule] = (
            self.tier1_rules
            + self.tier2_rules
            + self.tier3_rules
            + self.p0_rules
        )

        # 针对知性提取，初始化进化规则
        # P0 Fix #3: load_knowledge() 已在 EvolutionEngine.__init__ 中将规则注入到
        # self.tier1_rules 和 self.all_rules，无需再次手动追加
        self.evolution_engine = EvolutionEngine(self, self.db_path)

        # P3 Fix #16: 预编译所有正则，避免每次 detect_rules_in_sentence 时重新编译
        for rule in self.all_rules:
            self.compile_rule(rule)

        # V10: 现代词汇到古代词汇的自动映射 (针对 Historical/Fantasy 背景)
        self.modern_to_ancient_map = {
            "手术刀": "柳叶小刀",
            "保镖": "护卫",
            "保安": "护院",
            "警察": "官差",
            "医生": "大夫",
            "护士": "医女",
            "感冒": "风寒",
            "发烧": "发热",
            "打火机": "火折子",
            "手电筒": "灯笼",
            "汽车": "马车",
            "摩托车": "快马",
            "自行车": "马匹",
            "垃圾桶": "木篓",
            "沙发": "软榻",
            "空调": "冰盆",
            "电灯": "烛火",
            "电视": "皮影",
            "电脑": "算筹",
            "手机": "书信",
            "电话": "传音",
            "监控": "眼线",
            "摄像头": "暗卫",
            "警报": "鸣金",
            "信号弹": "响箭",
            "降落伞": "飞天兜",
            "防弹衣": "金丝软甲",
            "西装": "长衫",
            "高跟鞋": "花盆底",
            "丝袜": "罗袜",
            "内衣": "中衣",
            "胸罩": "肚兜",
            "内裤": "亵裤",
            "马桶": "恭桶",
            "电梯": "吊篮",
            "直升机": "木鸢",
            "狙击枪": "神臂弓",
            "手枪": "袖箭",
            "炸弹": "火雷",
            "网络": "情报网",
            "预估": "预想",
            "分析": "揣摩",
            "数据": "卷宗",
            "逻辑": "道理",
            "氛围": "气氛",
            "项目": "差事",
            "效率": "脚程",
            "计划": "打算",
            "目标": "念想",
        }

        # 词典库
        self.dialogue_verbs = ["说", "道", "问", "答", "喊", "喝", "笑", "骂", "低声", "开口"]
        self.action_verbs = [
            "抬", "抬手", "抬眼", "握", "攥", "退", "进", "站", "坐", "起身", "转身",
            "推", "拉", "按", "拍", "扫", "瞥", "盯", "看", "走", "迈", "靠近", "后退",
            "伸手", "收手", "垂眼", "抬眸", "落座", "躬身",
        ]
        self.emotion_words = [
            "紧张", "害怕", "愤怒", "不安", "慌乱", "难堪", "压抑", "委屈", "犹豫",
            "惊愕", "错愕", "冷意", "怒意", "不甘", "烦躁", "沉重", "惶然", "迟疑",
        ]
        self.explain_words = ["知道", "明白", "清楚", "意味着", "显然", "本质上", "归根结底", "并非", "而是", "说明", "证明", "代表", "意识到"]
        self.env_words = ["夜", "月", "风", "雨", "雪", "雾", "灯", "烛火", "窗", "窗纸", "回廊", "庭院", "树影", "天色", "暮色", "夜色", "寒意", "空气", "门外", "檐下"]
        self.transition_words = ["顿了顿", "片刻", "半晌", "一时间", "良久", "沉默", "静了静", "没有立刻说话"]

        self.rewrite_templates: Dict[str, List[str]] = {
            "指节泛白": ["指缝里失了血色", "掌心被掐出了月牙印", "手背筋骨凸起", "指尖紧紧抠进肉里"],
            "呼吸一滞": ["那口气猛地顶在嗓子眼", "心跳漏了一拍", "胸口像是被攥了一下"],
            "心头一震": ["太阳穴突突乱跳", "背上窜起一星凉意", "脑子里‘嗡’的一声"],
            "眸色微沉": ["眼里那点笑意散干净了", "目光冷得像结了冰", "视线陡然压低"],
            "总而言之": ["说到底", "这么一想", "如此看来", "左右不过是"],
            "这意味着": ["也就是说", "这下子", "兜兜转转，原是"],
            "事实上": ["其实", "实际上", "说白了", ""],
            "所谓的": ["那劳什子", "名义上的", "那门子", ""],
            "这便是": ["就是", "合该是", "这便是了"],
            "这种": ["如此", "这般", "那般", ""],
            "极其": ["分外", "实实", "透着点", ""],
            "名为X的Y抽象贴标签": ["(直接去掉'名为'框架，让概念自然溶于动作)", ""],
            "概念性双动词并行": ["(拆开并行结构，用不同主语或插入动作打断)", ""],
            "一种名为X的Y": ["(删掉整个标签结构，用具体物件或动作替代)", ""],
            "回声头排比": ["(删掉第一个短句的回声引子，直接进入内容)", ""],
            # V9: 解释性尾注改写模板
            "破折号解释尾注": ["(删除破折号后面的全部解释，用句号收住，让身体反应自己说话)", ""],
            "逗号解释尾注": ["(删除逗号后面的'那是/这是…'部分，身体反应已够有力)", ""],
            "某种抽象尾注": ["(删除'某种/一种…的力量/感觉'等抽象归因)", ""],
            "生理反应解释尾注": ["(生理反应本身就是最好的表达，不需要贴标签)", ""],
            "动作概括尾注": ["(动作描写已经到位，删掉概括性解释)", ""],
            # V10: 刻板修辞与油腻微表情
            "面具化微表情": ["(去掉模板化的弧度描写，用具体的眼部或面部肌肉动作代替)"],
            "刻板感官堆砌": ["(删掉铁锈、劣质等套路比喻，描写更写实的、不过分浮夸的生理体征)"],
            "做作力度词": ["(删掉‘死死’，用冷峻直接的动作传递力量)"],
            "刻板声音比喻": ["(去掉砂纸、钝器等老套比喻，用声音发紧、喉咙干涩等生理反应代替)"],
            "俗套濒死比喻": ["(删掉老套的濒死的鱼比喻，描写呼吸本身的急促起伏)"],
            "解释性连续暗喻": ["(删掉这种做作的、长串的上帝视角比喻，直接干脆地描写对方眼神中的情绪)"],
            "物化修辞狂热": ["(去掉强行将人比作丹鼎、灵石等物品的油腻上位压迫感写法)"],
            "投名状": ["诚意", "筹码", "敲门砖", "见面礼", "这场富贵"],
        }

        # 保守模式只自动处理低风险的显性模板。复杂风格问题只进入任务单，
        # 避免正则连续“润色”导致人物、剧情和语气越改越偏。
        self.safe_auto_rewrite_rules = {
            "这意味着",
            "显然",
            "不难看出",
            "总而言之/总之",
            "百科式连接词",
            "顿了顿",
            "沉默片刻",
            "目光落在",
            "取而代之",
            "投名状",
        }

    @staticmethod
    def compile_rule(rule: PatternRule) -> PatternRule:
        try:
            rule._compiled = re.compile(rule.pattern)
        except re.error as e:
            logger.warning(f"规则 '{rule.name}' 的正则编译失败: {e}，将跳过此规则")
            rule._compiled = None
        return rule

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def load_chapters_from_dir(self, input_dir: Path) -> List[Tuple[str, str]]:
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        files = sorted(
            [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"}],
            key=lambda p: p.name,
        )
        if not files:
            raise FileNotFoundError(f"目录中没有找到 .txt 或 .md 文件: {input_dir}")

        chapters = []
        for f in files:
            content = self.normalize_text(f.read_text(encoding="utf-8", errors="ignore"))
            chapters.append((f.stem, content))
        return chapters

    def split_text_into_chunks(self, text: str, prefix: str, max_chunk_size: Optional[int] = None) -> List[Tuple[str, str]]:
        limit = max_chunk_size or self.max_chunk_size
        if len(text) <= limit:
            return [(prefix, text)]

        paragraphs = self.split_paragraphs(text)
        if not paragraphs:
            return [(prefix, text)]

        chunks: List[Tuple[str, str]] = []
        current_chunk: List[str] = []
        current_len = 0
        chunk_idx = 1

        def flush():
            nonlocal current_chunk, current_len, chunk_idx
            if current_chunk:
                chunks.append((f"{prefix}_片段{chunk_idx}", "\n\n".join(current_chunk)))
                chunk_idx += 1
                current_chunk = []
                current_len = 0

        for paragraph in paragraphs:
            if len(paragraph) > limit:
                flush()
                for start in range(0, len(paragraph), limit):
                    chunks.append((f"{prefix}_片段{chunk_idx}", paragraph[start:start + limit]))
                    chunk_idx += 1
                continue
            next_len = current_len + len(paragraph) + (2 if current_chunk else 0)
            if current_chunk and next_len > limit:
                flush()
            current_chunk.append(paragraph)
            current_len += len(paragraph) + (2 if len(current_chunk) > 1 else 0)
        flush()
        return chunks

    def split_book_by_chapters(
        self,
        text: str,
        chapter_pattern: str = DEFAULT_CHAPTER_PATTERN,
        max_chunk_size: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        text = self.normalize_text(text)
        matches = list(re.finditer(chapter_pattern, text))
        limit = max_chunk_size or self.max_chunk_size

        # If no explicit chapters but text is very long, chunk it by paragraphs to avoid LLM token limits
        if not matches:
            return self.split_text_into_chunks(text, "全文", limit)

        chapters: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            title = m.group(0).strip()
            body = text[start:end].strip()
            chapters.extend(self.split_text_into_chunks(body, title, limit))
        return chapters

    @staticmethod
    def split_paragraphs(text: str) -> List[str]:
        return [p.strip() for p in re.split(r"\n+", text) if p.strip()]

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        # Fix variable-width lookbehind by using capturing groups in split
        # This keeps the delimiter in the resulting list
        pieces = re.split(r"([。！？；!?;][”’]?)", text)
        sentences = []
        for i in range(0, len(pieces) - 1, 2):
            sent = (pieces[i] + pieces[i+1]).strip()
            if sent:
                sentences.append(sent)
        if len(pieces) % 2 != 0 and pieces[-1].strip():
            sentences.append(pieces[-1].strip())
        return sentences

    @staticmethod
    def count_cn_words(text: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fff]", text))

    def detect_rules_in_sentence(
        self,
        chapter: str,
        paragraph_index: int,
        sentence_index: int,
        sentence: str
    ) -> List[Hit]:
        hits: List[Hit] = []
        for rule in self.all_rules:
            compiled = getattr(rule, '_compiled', None)
            if compiled is None:
                continue
            for m in compiled.finditer(sentence):
                # V12 Semantic Protection Layer
                if self.is_semantic_protected(sentence, m.start(), m.end(), rule.name):
                    continue
                
                hits.append(
                    Hit(
                        chapter=chapter,
                        paragraph_index=paragraph_index,
                        sentence_index=sentence_index,
                        hit_type=rule.category,
                        rule_name=rule.name,
                        text=sentence,
                        start=m.start(),
                        end=m.end(),
                        weight=rule.weight,
                        severity=rule.severity,
                    )
                )
        return hits

    def classify_paragraph(self, chapter: str, paragraph_index: int, paragraph: str) -> ParagraphInfo:
        scores = defaultdict(int)
        has_dialogue = "“" in paragraph or "”" in paragraph or "\"" in paragraph

        for w in self.dialogue_verbs:
            scores["dialogue"] += paragraph.count(w)
        for w in self.action_verbs:
            scores["action"] += paragraph.count(w)
        for w in self.emotion_words:
            scores["emotion"] += paragraph.count(w)
        for w in self.explain_words:
            scores["explain"] += paragraph.count(w)
        for w in self.env_words:
            scores["environment"] += paragraph.count(w)
        for w in self.transition_words:
            scores["transition"] += paragraph.count(w)

        if has_dialogue:
            scores["dialogue"] += 3

        tags = []
        for tag, value in scores.items():
            if value >= 2:
                tags.append(tag)

        if not tags:
            if has_dialogue:
                tags.append("dialogue")
            else:
                tags.append("neutral")

        if scores["explain"] >= 2 and scores["action"] == 0 and scores["dialogue"] == 0:
            tags.append("narration")
        if scores["environment"] >= 2 and scores["action"] <= 1 and scores["dialogue"] == 0:
            tags.append("environment_buffer")
        if scores["transition"] >= 1 and scores["action"] <= 1 and scores["dialogue"] == 0:
            tags.append("transition_buffer")

        # Determine Role
        role = "neutral"
        if has_dialogue:
            role = "dialogue"
        elif scores["action"] > scores["explain"] and scores["action"] > scores["environment"]:
            role = "action"
        elif scores["environment"] > scores["action"] and scores["environment"] > scores["explain"]:
            role = "environment"
        elif scores["explain"] >= 1:
            role = "exposition"

        # Rhythm Score (Variance of sentence lengths)
        sents = self.split_sentences(paragraph)
        lengths = [len(s) for s in sents if s.strip()]
        rhythm_score = statistics.stdev(lengths) if len(lengths) > 1 else 0.0

        # Sensory Score: Density of visceral keywords
        sensory_hits = 0
        sensory_keywords = r"味道|腥气|刺鼻|香气|臭|弥漫|药水|草木灰|冰冷|火辣辣|粗糙|磨蹭|湿润|干涩|麻木|僵硬|破空|闷哼|嘎然而止|脆响|嘶|打呼|敲在|瞳孔|咽喉|心脏|生理|下意识|肺部"
        sensory_hits = len(re.findall(sensory_keywords, paragraph))
        sensory_score = sensory_hits / (len(paragraph) / 100.0 + 1)

        # Symmetry Penalty: Detect AI-style balanced sentences (e.g., ADVP + SVO repetitions)
        symmetry_penalty = 0.0
        if len(sents) >= 2:
            balance_patterns = [
                r"^[^，。！？]+，[^，。！？]+[。！？]$", # Simple balanced sentence
                r"^(不仅|不但|既|由于)[^，。]+，(而且|还|也|所以)[^。！？]+[。！？]$"
            ]
            balanced_count = sum(1 for s in sents if any(re.match(p, s.strip()) for p in balance_patterns))
            if balanced_count >= 2:
                symmetry_penalty = balanced_count * 0.5

        # Omni Score calculation (penalize if explain is high without action)
        omni_score = min(1.0, scores["explain"] / 5.0)

        return ParagraphInfo(
            chapter=chapter,
            paragraph_index=paragraph_index,
            text=paragraph,
            tags=sorted(set(tags)),
            scores=dict(scores),
            has_dialogue=has_dialogue,
            sentence_count=len(sents),
            role=role,
            omniscient_score=round(omni_score, 2),
            rhythm_score=round(rhythm_score, 2),
            sensory_score=round(sensory_score, 2),
            symmetry_penalty=round(symmetry_penalty, 2),
            hits_by_severity={"P0": 0, "P1": 0, "P2": 0}
        )

    def clean_ai_clusters(self, text: str, applied: List[str]) -> str:
        """Paragraph-level cleaning for multi-sentence AI patterns."""
        # 模式1: 蜕变宣告类
        if re.search(r"(?:她|他|自己)?(?:知道|明白|清楚|深知)[，,]?从(?:这|那)(?:一刻|个瞬间|时|刻)起[，,].*?(?:死透了|死去了|消失了|不复存在了|成了过去|翻篇了|不再是)", text):
            applied.append("V11:蜕变宣告删除")
            if len(text) < 30:
                return ""
        # 模式2: 回声式否定排比 (这不是...这不是...这是...) -> 直接删除 (AI感极重，真人作家不这么煽情)
        # 兼容 2 次或 3 次以上的重复
        pattern_triple = r"(?:这不是|这并非)[^，。！？；：\n]{1,15}[。！？；：\n\s]+(?:这不是|这并非)[^，。！？；：\n]{1,15}[。！？；：\n\s]+(?:这是|这才是)[^。！？；：\n]{1,60}[。！？；：\n]?"
        pattern_double = r"(?:这不是|这并非)[^，。！？；：\n]{1,15}[。！？；：\n\s]+(?:而是|这才是)[^。！？；：\n]{1,60}[。！？；：\n]?"

        if re.search(pattern_triple, text, re.S):
            applied.append("V11:排比煽情删除(Triple)")
            text = re.sub(pattern_triple, "", text, flags=re.S)
        elif re.search(pattern_double, text, re.S):
            # 仅在非对话且叙述感强时删除
            applied.append("V11:排比煽情删除(Double)")
            text = re.sub(pattern_double, "", text, flags=re.S)
        return text

    def detect_empty_transition_blocks(self, paragraph_infos: Sequence[ParagraphInfo]) -> List[Dict[str, object]]:
        blocks: List[Dict[str, object]] = []
        current: List[ParagraphInfo] = []

        def is_low_progress(p: ParagraphInfo) -> bool:
            explain = p.scores.get("explain", 0)
            env = p.scores.get("environment", 0)
            trans = p.scores.get("transition", 0)
            action = p.scores.get("action", 0)
            dialogue = p.scores.get("dialogue", 0)
            return (explain + env + trans >= 2) and action <= 1 and dialogue == 0

        for p in paragraph_infos:
            if is_low_progress(p):
                current.append(p)
            else:
                if len(current) >= 2:
                    blocks.append(
                        {
                            "start_paragraph": current[0].paragraph_index + 1,
                            "end_paragraph": current[-1].paragraph_index + 1,
                            "paragraph_count": len(current),
                            "sample": " / ".join(x.text[:36] for x in current[:2]),
                        }
                    )
                current = []

        if len(current) >= 2:
            blocks.append(
                {
                    "start_paragraph": current[0].paragraph_index + 1,
                    "end_paragraph": current[-1].paragraph_index + 1,
                    "paragraph_count": len(current),
                    "sample": " / ".join(x.text[:36] for x in current[:2]),
                }
            )
        return blocks

    @staticmethod
    def stable_pick(text: str, choices: List[str]) -> str:
        if not choices:
            return text
        idx = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % len(choices)
        return choices[idx]

    @staticmethod
    def apply_outside_quotes(text: str, func: Callable[[str], str]) -> str:
        # Normalize quotes first to ensure consistent splitting
        text = EnhancedNovelAISurgery.normalize_quotes(text)
        parts = re.split(r'(“[^”]*”)', text)
        out: List[str] = []
        for part in parts:
            if not part:
                continue
            if part.startswith("“") and part.endswith("”"):
                out.append(part)
            else:
                out.append(func(part))
        return "".join(out)

    @staticmethod
    def normalize_quotes(text: str) -> str:
        """Convert straight quotes to Chinese curly quotes based on context."""
        # Handle full-width straight quotes as well
        text = text.replace('＂', '"')
        if '"' not in text:
            return text

        parts = text.split('"')
        new_parts = []
        for i, part in enumerate(parts[:-1]):
            new_parts.append(part)
            # Toggle between opening and closing quotes
            new_parts.append('“' if i % 2 == 0 else '”')
        new_parts.append(parts[-1])
        return "".join(new_parts)

    @staticmethod
    def cleanup_text_fragment(text: str) -> str:
        # Normalize quotes and punctuation to Chinese standards
        text = EnhancedNovelAISurgery.normalize_quotes(text)
        text = text.replace(',', '，').replace('!', '！').replace('?', '？').replace(':', '：').replace(';', '；')

        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[，,]{2,}", "，", text)
        text = re.sub(r"[。]{2,}", "。", text)
        text = re.sub(r"[！!]{2,}", "！", text)
        text = re.sub(r"[？?]{2,}", "？", text)
        text = re.sub(r"，[。！？；]", lambda m: m.group(0)[1], text)
        text = re.sub(r"^[，、；]+", "", text)
        text = re.sub(r"([“”])，", r"\1", text)
        text = re.sub(r"([，、；])([，、；])+", r"\1", text)
        text = re.sub(r"([。！？；])([，、；])", r"\1", text)
        return text.strip()

    @staticmethod
    def remove_repeated_token(text: str, token: str) -> str:
        first = text.find(token)
        if first == -1:
            return text
        second = text.find(token, first + len(token))
        while second != -1:
            text = text[:second] + text[second + len(token):]
            second = text.find(token, first + len(token))
        return text

    def is_pure_environment_sentence(self, sentence: str) -> bool:
        env_hits = sum(sentence.count(w) for w in self.env_words)
        action_hits = sum(sentence.count(w) for w in self.action_verbs)
        dialogue_hits = sum(sentence.count(w) for w in self.dialogue_verbs)
        has_quotes = "“" in sentence or "”" in sentence
        return env_hits >= 2 and action_hits <= 1 and dialogue_hits == 0 and not has_quotes

    def is_dead_air_sentence(self, sentence: str) -> bool:
        patterns = [
            r"^(?:房间里|屋内|屋子里|四周)?(?:一下)?静了下来[。！？]?$",
            r"^(?:一时间)?(?:谁也没有开口|谁都没有开口)[。！？]?$",
            r"^(?:一时间)?(?:气氛|空气)(?:有些)?(?:微妙|压抑|沉重|凝滞)[。！？]?$",
            r"^(?:她|他)没有立刻说话[。！？]?$",
            r"^(?:沉默了片刻|静了片刻|半晌|良久)[。！？]?$",
        ]
        stripped = sentence.strip()
        return any(re.fullmatch(p, stripped) for p in patterns)

    def rewrite_contrast_patterns(self, text: str, applied: List[str]) -> str:
        def repl_not_but(m: re.Match[str]) -> str:
            applied.append("不是A而是B")
            b = m.group(2).strip("，, ")
            tail = m.group(3) if m.group(3) and m.group(3) not in {"，", ","} else ""
            if len(b) <= 12:
                # 尝试清理前面的主语
                return f"分明是{b}{tail}"
            return f"{b}{tail}"

        def repl_rather(m: re.Match[str]) -> str:
            applied.append("与其说A不如说B")
            b = m.group(2).strip("，, ")
            tail = m.group(3) if m.group(3) and m.group(3) not in {"，", ","} else ""
            return f"更像是{b}{tail}"

        def repl_not_is(m: re.Match[str]) -> str:
            applied.append("并非而是")
            b = m.group(2).strip("，, ")
            tail = m.group(3) if m.group(3) and m.group(3) not in {"，", ","} else ""
            return f"{b}{tail}"

        text = re.sub(
            r"不是([^，。！？；：\n]{1,20})[，, ]?而是([^。！？；：\n]{1,24})([。！？；：，,]|$)",
            repl_not_but,
            text,
        )
        text = re.sub(
            r"不是([^，。！？；：\n]{1,20})[，, ]?也不是([^。！？；：\n]{1,24})([。！？；：，,]|$)",
            repl_not_but,
            text,
        )
        return text

    def is_semantic_protected(self, full_text: str, match_start: int, match_end: int, term: str) -> bool:
        """检查特定词汇是否处于语义保护区（即该用法在文学上是正当的，而非AI套路）。"""
        if term == "取而代之":
            # 保护“让...取而代之”或“被...取而代之”的剧情描写
            context_before = full_text[max(0, match_start-15):match_start]
            if any(p in context_before for p in ["让", "令", "使", "叫", "被", "由"]):
                return True
            # 保护作为谓语直接接人称代词/名字的情况（假千金取而代之）
            if re.search(r"[\u4e00-\u9fff]{2,4}(?:而|也)?取而代之$", context_before):
                return True
        elif term == "投名状":
            # 保护“立下投名状”等成语化/仪式化用法
            context_before = full_text[max(0, match_start-10):match_start]
            if any(p in context_before for p in ["立下", "写下", "纳了", "交了"]):
                return True
        return False

    def rewrite_replacement_patterns(self, text: str, applied: List[str]) -> str:
        def repl_replace(m: re.Match[str]) -> str:
            if self.is_semantic_protected(text, m.start(), m.end(), "取而代之"):
                return m.group(0) # Keep original
            applied.append("取而代之")
            tail = m.group(1).strip("，, ")
            return f"只剩下{tail}"

        text = re.sub(r"取而代之(?:的|地)?(?:是)?([^。！？；\n]{1,30})", repl_replace, text)
        return text

    def rewrite_explanatory_narration(self, text: str, applied: List[str], role: str = "") -> str:
        before = text

        # V11: 针对用户反馈的高频AI句式直接抹杀 (放在最前面，防止被后续通配符破坏)
        # 模式1: 察觉到XX语调不对 -> XX声音冷了下来 / 变了调
        new_text = re.sub(r"(?:[她他])?(?:察觉|意识|发现|感觉|注意)(?:到)?(?:了)?(?:今日)?([^，。！？；\n的]{1,15})(?:的)?(?:语调|语气)(?:不对|有异|不对劲|异常|变了)", r"\1的声音冷了下来", text)
        if new_text != text:
            applied.append("V11:语调优化")
            text = new_text

        # 模式2: 那种以往看向他时满是仰慕的眼神消失了 -> 眸子里那点温度散了个干净
        new_text = re.sub(r"(?:那(?:种|张))?(?:以往|曾经|过去)?看向(?:他|她)时(?:满是|充满)?仰慕的眼神(?:消失了|不见了|没了|不再|换成了)", "眸子里那点温度散了个干净", text)
        if new_text != text:
            applied.append("V11:眼神优化")
            text = new_text

        # 模式3: 是在向...宣告/展示/证明 -> ，如此这般，
        new_text = re.sub(r"[，, ]?是在向[^。！？；：\n]{1,20}(?:宣告|展示|证明|表达)(?:：|:|，|,)?", "，如此这般，", text)
        if new_text != text:
            applied.append("V11:宣告优化")
            text = new_text

        # 模式4: 这不仅是...更是... -> (直接进入重点)
        new_text = re.sub(r"这不仅(?:仅)?是[^，。！？；：\n]{1,20}更是([^，。！？；：\n]{1,30})", r"这合该是\1", text)
        if new_text != text:
            applied.append("V11:递进优化")
            text = new_text

        # 模式5: 尝试清理掉 '今日他/他此刻' 这种AI补白
        text = re.sub(r"(?:今日|此刻|这时|这会儿|现在)(?:他|她|它)", "", text)

        # 模式6: 没有那种A，反而带着一种B -> 替换为更直接的表达，而不是简单删除
        # 工业级增强：涵盖更多变体，如 "见不到...有的只是..."
        pattern_6 = r"(?:眼神|目光|语气|神色|动作|姿态|表现|气质)?(?:里|中)?(?:没有|见不到|看不到|见不到那种)(?:那种|这种|所谓(?:的)?)[^，。！？；：\n]{1,25}[，, ]?(?:反而|有的只是|取而代之的是|却)(?:带着|透着|溢出|有一股|有一种|流露出|是)(?:一种|一股|一点)?([^。！？；：\n]{1,35})"
        if re.search(pattern_6, text):
            applied.append("V11:二元对比废话优化")
            text = re.sub(pattern_6, r"只透着\1", text)

        # 模式7: 极其生硬的AI抽象标签 (XX般的XX) -> 也应该改写而非直接删除
        pattern_7 = r"(?:[^，。！？；：\n]{1,10}般的(?:探究|深意|意味|情绪|感觉|感|反应|逻辑|存在|气息|样子|状态|高度|深度|冷意))"
        if re.search(pattern_7, text):
            applied.append("V11:抽象标签优化")
            text = re.sub(pattern_7, "", text) # 虽然是删除，但范围被缩小了

        # 模式8: 同位语补白 (这个XX的牺牲品/工具) -> 优化
        new_text = re.sub(r"这个([^，。！？；：\n]{2,30})的(?:牺牲品|玩物|棋子|工具|存在|宿命|角色|反派)", r"那个\1的人", text)
        if new_text != text:
            applied.append("V11:同位语标签优化")
            text = new_text

        # 模式9: 重生/蜕变宣告 (她知道，从这一刻起...死透了) -> 直接删除 (因为这本身就是废话总结)
        if re.search(r"(?:她|他|自己)?(?:知道|明白|清楚|深知)[，,]?从(?:这|那)(?:一刻|个瞬间|时|刻)起[，,].*?(?:死透了|死去了|消失了|不复存在了|成了过去|翻篇了|不再是|彻底告别)", text):
            applied.append("V11:蜕变宣告删除")
            # 标记需要人工或Agent重写，这里先置空或保留主体
            # 暂时保留部分避免全删
            text = re.sub(r"(?:她|他|自己)?(?:知道|明白|清楚|深知)[，,]?从(?:这|那)(?:一刻|个瞬间|时|刻)起[，,]", "", text)

        # 移除强烈的叙述感引子 (仅在叙述性段落中执行，避免误删对话和正常叙事)
        if role in ["exposition", "neutral"]:
            text = re.sub(r"([她他])(?:似乎|仿佛)?(?:知道|明白|意识到|察觉到|发现|感觉到|注意到|看出来)了?，?", "", text)
        if role == "exposition":
            # 增加跨度到30个字符，防止长句失效 (仅匹配一些常见的无用定语前缀)
            text = re.sub(r"^(?:那种|这种|所谓(?:的)?|这一(?:次|个)|那个|这个|如此(?:的)?|那样(?:的)?)([^。！？；\n]{1,30})$", r"\1", text)

        if text != before:
            applied.append("叙述感清理")

        replacements = {
            r"很清楚，?": "",
            r"显然，?": "",
            r"不难看出，?": "",
            r"意味着": "这下",
            r"她并非": "她不是",
            r"他并非": "他不是",
            r"察觉到": "",
            r"意识到": "",
            r"发现到": "",
        }

        for pattern, repl in replacements.items():
            new_text = re.sub(pattern, repl, text)
            if new_text != text:
                applied.append(pattern)
                text = new_text
        return text

    def rewrite_transition_fillers(self, text: str, applied: List[str]) -> str:
        patterns = [
            (r"(?:^|，)?顿了顿(?:，)?", "，", "顿了顿"),
            (r"(?:^|，)?沉默了片刻(?:，)?", "，", "沉默片刻"),
            (r"(?:^|，)?静了片刻(?:，)?", "，", "沉默片刻"),
            (r"(?:^|，)?半晌(?:，)?", "，", "沉默片刻"),
            (r"(?:^|，)?良久(?:，)?", "，", "沉默片刻"),
            (r"(?:^|，)?一时间(?:，)?", "，", "一时间"),
            (r"(?:^|，)?没有立刻说话(?:，)?", "，", "没有立刻说话"),
            (r"目光落在", "看向", "目光落在"),
        ]
        for pattern, repl, tag in patterns:
            new_text = re.sub(pattern, repl, text)
            if new_text != text:
                applied.append(tag)
                text = new_text
        return text

    def rewrite_narrative_to_action(self, text: str, info: ParagraphInfo, applied: List[str]) -> str:
        if info.role != "exposition" and info.omniscient_score < 0.5:
            return text

        psych_to_action = {
            r"([她他])(深知|明白|很清楚)([^，。！？]+)": r"\1沉默着，\3的事实像石头一样压在心底",
            r"(让他|令她)(焦躁|不安|恐惧|绝望)": r"\1的手指不自觉地抓紧了衣角",
            r"(意识到|察觉到|发现)(出事了|不对劲)": r"瞳孔猛地一缩，视线落在那处不协调的细节上",
        }
        before = text
        for pattern, repl in psych_to_action.items():
            text = re.sub(pattern, repl, text)

        if text != before:
            applied.append("展示代替陈述(ShowDontTell)")

        return text

    def semantic_optimize_paragraph(self, paragraph: str, info: ParagraphInfo, mode: str) -> Tuple[str, List[str]]:
        applied_total = []
        if info.role == "exposition" and mode == "aggressive":
            paragraph = self.rewrite_narrative_to_action(paragraph, info, applied_total)

        if info.role == "environment" and mode == "aggressive" and len(paragraph) > 60:
            sents = self.split_sentences(paragraph)
            if len(sents) >= 3:
                mid = sents[1:-1]
                new_mid = [s for s in mid if not self.is_pure_environment_sentence(s)]
                if len(new_mid) < len(mid):
                    applied_total.append("语义环境压缩")
                    paragraph = sents[0] + "".join(new_mid) + sents[-1]

        return paragraph, applied_total

    def rewrite_cliche_phrases(self, text: str, applied: List[str]) -> str:
        # P2 Fix #12: 分离「直接字面量匹配」和「规则名匹配」两类 key
        # 规则名类 key（含有大写字母X/Y或括号）不可能出现在原文中，跳过
        for key, options in self.rewrite_templates.items():
            # 跳过规则名类的 key（包含 X/Y、括号等元字符，不是原文片段）
            if re.search(r'[A-Z()（）]', key):
                continue
            if key in text:
                replacement = self.stable_pick(text + key, options)
                text = text.replace(key, replacement)
                applied.append(key)
        return text

    def compress_environment_sentence(self, sentence: str, mode: str, applied: List[str]) -> str:
        text = sentence
        env_direct = {
            r"夜色沉沉": "夜已深",
            r"冷风[^。！？；\n]{0,12}(?:穿过|穿堂|吹进|拂过)": "回廊里起了风",
            r"雨(?:滴|声)[^。！？；\n]{0,12}(?:敲打|拍打)[^。！？；\n]{0,8}(?:窗|窗棂|窗纸)": "雨声敲在窗上",
            r"烛火(?:轻轻)?摇曳(?:不定)?": "烛火晃了晃",
            r"空气[^。！？；\n]{0,10}(?:压抑|沉重|发紧)": "屋里发闷",
        }
        for pattern, repl in env_direct.items():
            new_text = re.sub(pattern, repl, text)
            if new_text != text:
                applied.append("环境压缩")
                text = new_text

        if mode == "aggressive" and self.is_dead_air_sentence(text):
            applied.append("删除空转句")
            if len(text) < 30:
                return ""

        if mode == "aggressive" and self.is_pure_environment_sentence(text):
            short = re.sub(r"(?:外头的|屋内的|四周的|周遭的)", "", text)
            short = re.sub(r"(?:沉沉|深沉|浓重|不定|轻轻|仿佛|似乎)", "", short)
            short = self.cleanup_text_fragment(short)
            if len(short) <= 12:
                applied.append("删除环境垫场")
                return ""
            applied.append("环境压缩")
            return short
        return text

    def reduce_style_repetition(self, text: str, applied: List[str]) -> str:
        for token in ["微微", "缓缓", "仿佛"]:
            count_before = text.count(token)
            if count_before > 1:
                text = self.remove_repeated_token(text, token)
                applied.append(f"去重:{token}")
        return text

    def map_modern_to_ancient(self, text: str, background: str, applied: List[str]) -> str:
        if background not in ["Historical", "Fantasy"]:
            return text

        before = text
        for modern, ancient in self.modern_to_ancient_map.items():
            if modern in text:
                text = text.replace(modern, ancient)
                applied.append(f"词汇映射:{modern}->{ancient}")
        return text

    def safety_check_rewrite(self, original: str, revised: str, mode: str) -> str:
        revised = self.cleanup_text_fragment(revised)
        if not revised:
            if mode == "aggressive":
                return ""
            return original
        if not re.search(r"[\u4e00-\u9fff]", revised):
            return original
        if "“" in original or "”" in original:
            if (original.count("“") + original.count("”")) != (revised.count("“") + revised.count("”")):
                return original
        if mode == "conservative" and len(revised) < max(4, len(original) * 0.35):
            return original
        # --- P0 Fix: 单句微观保量保护 ---
        # 如果单句重写后中文字数 < 原句的 50%，回退到原句
        # 防止正则规则累积删除导致字数严重缩水
        orig_cn = self.count_cn_words(original)
        rev_cn = self.count_cn_words(revised)
        if orig_cn >= 10 and rev_cn < orig_cn * 0.50:
            return original
        return revised

    def rewrite_sentence(
        self,
        sentence: str,
        paragraph_info: ParagraphInfo,
        mode: str = "conservative",
        p0_only: bool = False
    ) -> Tuple[str, List[str], str]:
        applied: List[str] = []
        original = sentence

        def worker(chunk: str) -> str:
            local = chunk
            for rule in self.p0_rules:
                if re.search(rule.pattern, local):
                    local = re.sub(rule.pattern, "", local)
                    applied.append(f"P0清理:{rule.name}")

            local = self.rewrite_contrast_patterns(local, applied)
            local = self.rewrite_replacement_patterns(local, applied)
            local = self.rewrite_explanatory_narration(local, applied, paragraph_info.role)
            local = self.rewrite_transition_fillers(local, applied)
            local = self.rewrite_cliche_phrases(local, applied)

            # 现代词映射
            if mode == "aggressive" and paragraph_info.role != "dialogue":
                bg = self.evolution_engine.get_book_bg() # 获取当前书籍背景
                local = self.map_modern_to_ancient(local, bg, applied)

            # Catch suspended warnings in the sentence
            sdt_flags = self.sdt_transformer.check(local)
            for flag in sdt_flags:
                applied.append(flag)

            # 连词去脓
            conj_flags = self.conjunction_neutering.check(local)
            for flag in conj_flags:
                applied.append(flag["reason"])

            if not p0_only:
                local = self.reduce_style_repetition(local, applied)
                local = self.compress_environment_sentence(local, mode=mode, applied=applied)

            return local

        revised = self.apply_outside_quotes(sentence, worker)

        if paragraph_info.role in ["exposition", "neutral"] and not p0_only:
             revised = self.rewrite_narrative_to_action(revised, paragraph_info, applied)

        revised = self.safety_check_rewrite(original, revised, mode)

        risk_level = "low"
        if any(tag in applied for tag in ["展示代替陈述(ShowDontTell)", "P0清理"]):
            risk_level = "medium"
        if "删除空转句" in applied:
            risk_level = "high"

        if revised == original:
            applied = []
        return revised, sorted(set(applied)), risk_level

    def rewrite_chapter(
        self,
        chapter_name: str,
        chapter_text: str,
        context: NovelContext = None,
        mode: str = "conservative",
        min_hits_to_rewrite: int = 1,
        enable_semantic_opt: bool = True,
        export_tasks_only: bool = True
    ) -> Tuple[str, List[SentenceChange]]:
        if context:
            self.evolution_engine.set_book_bg(context.background)
            # 动态更新向量库集合，确保物理隔离
            if context.book_id and context.book_id != self.book_id:
                self.book_id = context.book_id
                self.global_hook_engine = GlobalVisionHooker(
                    db_path=self.vector_db_path if self.enable_global_index else None,
                    book_id=self.book_id,
                    llm_caller=self.task_llm_caller,
                )

        paragraphs = self.split_paragraphs(chapter_text)
        revised_paragraphs_p1: List[str] = []
        changes: List[SentenceChange] = []

        # Automatic context detection if not provided
        if context is None:
            context = self.context_analyzer.analyze(chapter_text)
            # Track entities
            context.characters = self.entity_tracker.load_characters(context.book_id)
            context.characters = self.entity_tracker.extract_from_text(chapter_text, context.characters)
            self.entity_tracker.save_characters(context.book_id, context.characters)

        for p_idx, paragraph in enumerate(paragraphs):
            p_info = self.classify_paragraph(chapter_name, p_idx, paragraph)
            p_applied = []
            if mode == "aggressive":
                paragraph = self.clean_ai_clusters(paragraph, p_applied)

            p_hits: List[Hit] = []
            sentences = self.split_sentences(paragraph)

            if not sentences:
                revised_paragraphs_p1.append(paragraph)
                continue

            rev_sents = []
            for s_idx, sentence in enumerate(sentences):
                s_hits = self.detect_rules_in_sentence(chapter_name, p_idx, s_idx, sentence)

                # --- V7: Cross-sentence Echo-Head Parallel Detection ---
                prev_sent = ""
                if s_idx > 0:
                    prev_sent = sentences[s_idx - 1]
                elif p_idx > 0:
                    prev_p_sents = self.split_sentences(paragraphs[p_idx - 1])
                    if prev_p_sents:
                        prev_sent = prev_p_sents[-1]

                if prev_sent and len(sentence) >= 4:
                    prev_tail_m = re.search(r"([\u4e00-\u9fff]{1,2})[。！？；!?; \n]*$", prev_sent)
                    if prev_tail_m:
                        tail_word = prev_tail_m.group(1)
                        if sentence.startswith(tail_word) and len(tail_word) > 0:
                            if tail_word not in ["的", "了", "是", "在", "着", "里", "啊", "呢", "吧", "吗", "说", "道", "问", "答"]:
                                s_hits.append(Hit(
                                    chapter=chapter_name,
                                    paragraph_index=p_idx,
                                    sentence_index=s_idx,
                                    hit_type="echo_parallel",
                                    rule_name="回声头排比",
                                    text=sentence,
                                    start=0,
                                    end=len(tail_word),
                                    weight=3.0,
                                    severity="P1"
                                ))

                p_hits.extend(s_hits)

                sdt_flags = self.sdt_transformer.check(sentence)
                has_p0 = any(h.severity == "P0" for h in s_hits)
                safe_hit = any(h.rule_name in self.safe_auto_rewrite_rules for h in s_hits)
                tier2_names = {r.name for r in self.tier2_rules}
                p_tier2_hits = [h for h in p_hits if h.rule_name in tier2_names]
                if mode == "aggressive":
                    should_rewrite = bool(sdt_flags) or has_p0 or any(h.rule_name in [r.name for r in self.tier1_rules] for h in s_hits)
                    # Industrial Grade: Trigger rewrite if sensory density is too low or symmetry is too high
                    if p_info.sensory_score < 0.15 and p_info.role in ["action", "environment"]:
                        should_rewrite = True
                    if p_info.symmetry_penalty >= 1.0:
                        should_rewrite = True
                    if p_info.omniscient_score >= 0.7:
                        should_rewrite = True
                    if len(p_tier2_hits) >= 2:
                        should_rewrite = True
                else:
                    should_rewrite = has_p0 or safe_hit

                if not should_rewrite:
                    diagnostics = sorted({h.rule_name for h in s_hits} | set(sdt_flags))
                    if diagnostics and len(changes) < 80:
                        changes.append(SentenceChange(
                            chapter_name,
                            p_idx + 1,
                            s_idx + 1,
                            sentence,
                            "[审计建议: 局部修复，不自动改写]",
                            diagnostics,
                            "low",
                        ))
                    rev_sents.append(sentence)
                    continue

                revised, applied, risk = self.rewrite_sentence(sentence, p_info, mode)

                # Sensory Priming: If first sentence of paragraph and role is sensory, force an anchor
                if s_idx == 0 and p_info.sensory_score < 0.3 and p_info.role in ["action", "environment"]:
                    if "感官提升" not in applied:
                        applied.append("感官提升")

                if revised != sentence:
                    changes.append(SentenceChange(chapter_name, p_idx+1, s_idx+1, sentence, revised, applied, risk))
                if revised.strip():
                    rev_sents.append(revised)

            p1_text = "".join(rev_sents)
            revised_paragraphs_p1.append(p1_text)

        revised_paragraphs_p2: List[str] = []
        for p_idx, p_text in enumerate(revised_paragraphs_p1):
            if not p_text.strip(): continue
            p_info_post = self.classify_paragraph(chapter_name, p_idx, p_text)

            if p_info_post.omniscient_score > 0.8:
                changes.append(SentenceChange(chapter_name, p_idx+1, 0, "[审计:视角过全]", "[建议手动ShowDon'tTell]", ["OmniscienceAudit"], "low"))

            revised_paragraphs_p2.append(p_text)

        # V12: Auto-delete pure summary endings (博弈才刚刚开始/拉开帷幕等)
        summary_patterns = r"(?:一场|这(?:次|场)|新的)(?:拉锯|博弈|死局|较量|序幕|风暴|暗流|战场|传奇).{0,15}(?:刚刚开始|拉开帷幕|彻底钉死|浮出水面|席卷而来|扎根)|属于(?:他|她|他们)的.{0,5}才刚刚(?:铺开|开始)|是.{0,5}的预警，也是.{0,5}的倒计时|(?:真正的血战|新的征程)，才刚刚开始"
        for _ in range(2): # Check last 2 paragraphs
            if not revised_paragraphs_p2: break
            last_p = revised_paragraphs_p2[-1]
            # Only delete if it's short and matches the pattern perfectly (Semantic Recognition)
            if len(last_p) < 65 and re.search(summary_patterns, last_p):
                # Ensure it's not a valid descriptive sentence with action
                if not re.search(r"抱着|走进|挥剑|跨出|收起|走下|火盆|营房", last_p):
                    changes.append(SentenceChange(chapter_name, len(revised_paragraphs_p2), 0, last_p, "[已自动删除章节总结性废话]", ["EndingSummaryRemoval"], "high"))
                    revised_paragraphs_p2.pop()
                else:
                    # If it has content, just flag it for refinement in LLM phase
                    changes.append(SentenceChange(chapter_name, len(revised_paragraphs_p2), 0, last_p, "[审计:建议剥离总结词]", ["EndingSummaryAudit"], "low"))

        rev_text = self.normalize_text("\n\n".join(revised_paragraphs_p2))

        # Phase 3: Agent Task Generation (Replacing Automated LLM Pass)
        if enable_semantic_opt:
            # Inject Style-Specific Target
            if context.genre:
                benchmark = self.evolution_engine.load_benchmark(context.genre)
                if benchmark:
                    self.style_profile = benchmark
                    context.background += f" | 风格对标: {context.genre} (感官目标: {benchmark['sensory']})"

            # Re-run full analysis to get structure issues
            final_report = self.analyze_chapter(chapter_name, rev_text, context, persist=False)
            structure_issues = final_report.get("structure_issues", [])
            task_path = self.agent_task_gen.generate_task(chapter_name, rev_text, context, changes, structure_issues=structure_issues)

            print(f"📝 已生成 IDE Agent 协作任务: {task_path.name}")

            # --- V11 Execute and Verify Loop ---
            if export_tasks_only:
                print("📌 export_tasks_only=True，仅导出任务单，不调用任何 LLM 定稿。")
            else:
                if self.finalize_llm_caller:
                    if self.llm_mode == "full":
                        rev_text = self.execute_and_verify_agent_task(chapter_name, task_path, rev_text, context)
                    else:
                        rev_text = self.execute_targeted_llm_patches(
                            chapter_name,
                            rev_text,
                            context,
                            changes,
                            structure_issues,
                        )
                else:
                    logger.warning("未启用 allow_llm_finalize，跳过 LLM 自动定稿。")

        # --- P0 Fix: Content Parity Gate (字数保量硬门控) ---
        parity_result = self.content_parity_validator.validate(chapter_text, rev_text, chapter_name)
        if not parity_result["passed"]:
            logger.error(
                f"\U0001f6a8 [P0:字数保量失败] {chapter_name}: "
                f"\u539f\u6587={parity_result['original_count']}\u5b57, "
                f"\u91cd\u5199={parity_result['revised_count']}\u5b57, "
                f"\u6bd4\u7387={parity_result['ratio']:.1%}\u3002"
                f"\u539f\u56e0: {parity_result['reason']}\u3002\u56de\u9000\u5230\u539f\u6587\u3002"
            )
            # 保存缩水版本到隔离区供人工审查
            quarantine_dir = self.root_dir / "quarantine"
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            quarantine_path = quarantine_dir / f"{chapter_name}_\u7f29\u6c34\u8349\u7a3f.md"
            quarantine_path.write_text(rev_text, encoding="utf-8")
            logger.info(f"\U0001f4e6 \u7f29\u6c34\u7248\u672c\u5df2\u9694\u79bb\u5230: {quarantine_path}")
            rev_text = chapter_text  # \u56de\u9000\u5230\u539f\u6587
        elif parity_result["warning"]:
            logger.warning(
                f"\u26a0\ufe0f [\u4fdd\u91cf\u8b66\u544a] {chapter_name}: {parity_result['reason']}"
            )
        else:
            logger.info(
                f"\u2705 [\u5b57\u6570\u4fdd\u91cf] {chapter_name}: "
                f"{parity_result['original_count']}\u2192{parity_result['revised_count']} "
                f"({parity_result['ratio']:.0%})"
            )

        return rev_text, changes

    @staticmethod
    def estimate_tokens(text: str) -> int:
        chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
        other = max(0, len(text) - chinese)
        return chinese + max(1, other // 4)

    def _change_priority(self, change: SentenceChange) -> int:
        text = " ".join(change.applied_rules) + " " + change.original + " " + change.revised
        score = 0
        if change.risk_level == "high":
            score += 60
        elif change.risk_level == "medium":
            score += 35
        if any(key in text for key in ["P0", "因果", "逻辑", "伏笔", "数字", "人物", "道具", "性别", "时间线", "场景"]):
            score += 30
        if any(key in text for key in ["解释性尾注", "悬浮", "直给情绪", "全知", "ShowDon'tTell"]):
            score += 18
        if change.revised.startswith("[审计"):
            score += 10
        return score

    def select_llm_patch_targets(
        self,
        changes: Sequence[SentenceChange],
        structure_issues: Sequence[str],
        paragraph_count: int,
    ) -> List[int]:
        scores: Counter[int] = Counter()
        for change in changes:
            if change.paragraph <= 0 or change.paragraph > paragraph_count:
                continue
            priority = self._change_priority(change)
            if priority > 0:
                scores[change.paragraph] += priority

        for issue in structure_issues[:20]:
            for m in re.finditer(r"段落\s*(\d+)", issue):
                para = int(m.group(1))
                if 1 <= para <= paragraph_count:
                    scores[para] += 25
            if any(key in issue for key in ["开篇", "黄金三原则", "WHO", "WHERE", "WHY"]):
                scores[1] += 25

        return [para for para, _ in scores.most_common(self.llm_max_patches)]

    def build_targeted_llm_patch_prompt(
        self,
        chapter_name: str,
        text: str,
        context: NovelContext,
        changes: Sequence[SentenceChange],
        structure_issues: Sequence[str],
        targets: Sequence[int],
    ) -> Tuple[str, List[int]]:
        paragraphs = self.split_paragraphs(text)
        if not targets:
            return "", []

        changes_by_para: Dict[int, List[SentenceChange]] = defaultdict(list)
        for change in changes:
            changes_by_para[change.paragraph].append(change)

        fates = self.logic_auditor.get_character_fates()
        fate_lines = "\n".join(f"- {name}: {status}" for name, status in fates.items())
        
        recent_sigs = self.logic_auditor.get_recent_signatures(limit=2)
        sig_lines = "\n".join(f"- {s['chapter']} 的结尾关键词: {s['text'][:150]}..." for s in recent_sigs)

        def build_prompt(active_targets: Sequence[int], context_radius: int) -> str:
            blocks = []
            for para_no in active_targets:
                idx = para_no - 1
                if idx < 0 or idx >= len(paragraphs):
                    continue
                before = paragraphs[idx - 1] if context_radius and idx > 0 else ""
                after = paragraphs[idx + 1] if context_radius and idx + 1 < len(paragraphs) else ""
                diagnostics = []
                for change in changes_by_para.get(para_no, [])[:4]:
                    rules = " / ".join(change.applied_rules[:6])
                    diagnostics.append(f"- 句{change.sentence}: {rules} | 原句: {change.original[:160]}")
                block = [
                    f"### P{para_no}",
                    f"诊断:\n" + ("\n".join(diagnostics) if diagnostics else "- 结构审计命中，请只修复本段明确问题。"),
                ]
                if before:
                    block.extend(["上段参考:", before[:350]])
                block.extend(["待修段落:", paragraphs[idx]])
                if after:
                    block.extend(["下段参考:", after[:350]])
                blocks.append("\n".join(block))

            issue_lines = "\n".join(f"- {issue[:180]}" for issue in structure_issues[:10])
            characters = ", ".join(c.get("name", "") for c in context.characters if c.get("name")) or "未识别"
            return (
                "你是严谨的小说编辑，只做局部补丁，不整章重写。\n"
                "任务：修复逻辑链、剧情动机、人物一致性、道具/数字/伏笔问题，以及明显 AI 腔。\n"
                "硬规则：不换剧情，不新增大段背景，不改未列出的段落；每个补丁保持原段落 80%-125% 长度；只输出 JSON。\n"
                f"章节：{chapter_name}\n"
                f"题材：{context.genre} / {context.background}\n"
                f"角色清单：{characters}\n"
                f"人物命运硬约束 (绝对禁止违背):\n{fate_lines or '- 暂无已知终结命运'}\n"
                f"语义重复硬门控 (严禁在当前章节重复以下内容):\n{sig_lines or '- 暂无前置重复项'}\n"
                f"结构审计红线:\n{issue_lines or '- 无'}\n\n"
                "输出格式：[{\"paragraph\": 段落编号, \"text\": \"替换后的完整段落\"}]\n\n"
                + "\n\n".join(blocks)
            )

        active_targets = list(targets[: self.llm_max_patches])
        context_radius = self.llm_context_paragraphs
        prompt = build_prompt(active_targets, context_radius)
        while active_targets and self.estimate_tokens(prompt) > self.llm_token_budget:
            if context_radius > 0:
                context_radius = 0
            else:
                active_targets = active_targets[:-1]
            prompt = build_prompt(active_targets, context_radius)

        return prompt, active_targets

    def validate_patch_paragraph(self, original: str, revised: str) -> Tuple[bool, str]:
        revised = revised.strip()
        if not revised or not re.search(r"[\u4e00-\u9fff]", revised):
            return False, "empty_or_non_chinese"
        if "```" in revised or re.search(r"^(好的|以下是|我将|已完成)", revised):
            return False, "chatty_or_markdown"
        orig_cn = self.count_cn_words(original)
        rev_cn = self.count_cn_words(revised)
        if orig_cn >= 30:
            ratio = rev_cn / max(orig_cn, 1)
            if ratio < 0.75 or ratio > 1.35:
                return False, f"paragraph_length_ratio:{ratio:.0%}"
        if "“" in original or "”" in original:
            if abs((original.count("“") + original.count("”")) - (revised.count("“") + revised.count("”"))) > 2:
                return False, "quote_count_drift"
        return True, "ok"

    def apply_llm_patch_json(self, original_text: str, llm_text: str, allowed_targets: Sequence[int]) -> Tuple[str, List[str]]:
        data = extract_json_from_llm(llm_text)
        if data is None:
            return original_text, ["json_parse_failed"]
        if isinstance(data, dict):
            data = data.get("patches") or data.get("items") or [data]
        if not isinstance(data, list):
            return original_text, ["json_not_list"]

        paragraphs = self.split_paragraphs(original_text)
        allowed = set(allowed_targets)
        applied: List[str] = []
        rejected: List[str] = []

        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                para_no = int(item.get("paragraph") or item.get("para") or item.get("p"))
            except (TypeError, ValueError):
                rejected.append("missing_paragraph")
                continue
            if para_no not in allowed or para_no < 1 or para_no > len(paragraphs):
                rejected.append(f"P{para_no}:not_allowed")
                continue
            revised = str(item.get("text") or item.get("revised") or "").strip()
            ok, reason = self.validate_patch_paragraph(paragraphs[para_no - 1], revised)
            if not ok:
                rejected.append(f"P{para_no}:{reason}")
                continue
            paragraphs[para_no - 1] = revised
            applied.append(f"P{para_no}")

        result = self.normalize_text("\n\n".join(paragraphs))
        if rejected:
            logger.warning("LLM 补丁部分拒绝: " + ", ".join(rejected[:8]))
        return result, applied

    def execute_targeted_llm_patches(
        self,
        chapter_name: str,
        original_text: str,
        context: NovelContext,
        changes: Sequence[SentenceChange],
        structure_issues: Sequence[str],
    ) -> str:
        if not self.finalize_llm_caller:
            return original_text

        paragraphs = self.split_paragraphs(original_text)
        targets = self.select_llm_patch_targets(changes, structure_issues, len(paragraphs))
        if not targets:
            logger.info(f"🧩 {chapter_name}: 未发现值得调用 LLM 的高风险片段，跳过 API 修复。")
            return original_text

        prompt, active_targets = self.build_targeted_llm_patch_prompt(
            chapter_name,
            original_text,
            context,
            changes,
            structure_issues,
            targets,
        )
        if not prompt or not active_targets:
            logger.warning(f"🧩 {chapter_name}: LLM 补丁提示词超过预算，跳过 API 修复。")
            return original_text

        est = self.estimate_tokens(prompt)
        print(f"🧩 LLM 补丁模式: {chapter_name} targets={active_targets} prompt≈{est} tokens")
        sys_prompt = "你是严谨的小说编辑。只输出 JSON 数组，不输出解释。"
        res = self.finalize_llm_caller(prompt, sys_prompt)
        if not res:
            logger.error("❌ LLM 补丁修复失败，保留本地修复文本。")
            return original_text

        patched_text, applied = self.apply_llm_patch_json(original_text, res, active_targets)
        if not applied:
            logger.error("❌ LLM 补丁无可采纳段落，保留本地修复文本。")
            return original_text

        ok, reason = self.validate_llm_final_text(original_text, patched_text, context, chapter_name)
        if not ok:
            logger.error(f"❌ LLM 补丁后整章校验失败 ({reason})，保留本地修复文本。")
            return original_text

        patch_dir = self.root_dir / "agent_tasks"
        patch_dir.mkdir(parents=True, exist_ok=True)
        (patch_dir / f"{chapter_name}_LLM补丁采纳.json").write_text(
            json.dumps({"applied": applied, "targets": active_targets}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"✅ LLM 补丁采纳: {chapter_name} {', '.join(applied)}")
        return patched_text

    def execute_and_verify_agent_task(self, chapter_name: str, task_path: Path, original_text: str, context: NovelContext) -> str:
        print(f"🤖 正在执行 Agent 自动重写任务: {task_path.name} ...")
        prompt_instructions = self.build_llm_refinement_prompt(task_path, original_text, context)
        sys_prompt = r"""你是一个冷峻的人类文学主编。
【死指令】：
1. 严禁出现冒号（:）和破折号（——）。
2. 严禁使用“杂种、狐媚、死死、不由得、眸色微沉、竟然是、那是...”等 AI 油腻词。
3. 严禁在开头进行摘要式总结。
4. 保持长短句错落，该断句就断句。
直接输出正文。"""

        if not self.finalize_llm_caller:
            logger.error("❌ LLM 自动重写未启用。")
            return original_text

        res = self.finalize_llm_caller(prompt_instructions, sys_prompt)
        if not res:
            logger.error(f"❌ LLM 自动重写任务失败，返回局部正则修改的原文。")
            return original_text

        final_text = res.replace('```markdown', '').replace('```', '').strip()
        ok, reason = self.validate_llm_final_text(original_text, final_text, context, chapter_name)
        if not ok:
            logger.error(f"❌ LLM 自动重写未通过完整性校验 ({reason})，保留局部正则修改文本。")
            quarantine_path = task_path.parent / f"{chapter_name}_LLM未采纳草稿.md"
            quarantine_path.write_text(final_text, encoding="utf-8")
            return original_text

        out_path = task_path.parent / f"{chapter_name}_LLM最终定稿.md"
        out_path.write_text(final_text, encoding="utf-8")

        print(f"🔍 正在进行质检闭环回验 ...")
        verify_report = self.analyze_chapter(chapter_name, final_text, context, persist=False)
        if verify_report["ai_score"] > 60 or len(verify_report.get("structure_issues", [])) > 2:
            print(f"⚠️ [警告] {chapter_name} 重写后质检未达优 (AI Score: {verify_report['ai_score']})，已标记 <Needs_Human_Review>")
            warning_path = task_path.parent / f"{chapter_name}_Needs_Human_Review.txt"
            warning_path.write_text(f"质检未通过！\nAI Score: {verify_report['ai_score']}\n结构问题:\n" + "\n".join(verify_report.get("structure_issues", [])), encoding="utf-8")
        else:
            print(f"✅ {chapter_name} 重写质检通过！")

        return final_text

    def build_llm_refinement_prompt(self, task_path: Path, original_text: str, context: NovelContext) -> str:
        characters = ", ".join(c.get("name", "") for c in context.characters if c.get("name")) or "未识别"
        event_checklist = self.extract_event_checklist(original_text)
        source_block = original_text
        if len(source_block) > self.max_chunk_size + 800:
            source_block = source_block[: self.max_chunk_size + 800]
        return (
            task_path.read_text(encoding="utf-8")
            + "\n\n## 自动定稿硬约束\n"
            + "- 基于下方待处理正文做局部修复，严禁只改写摘要或截断前半章。\n"
            + f"- 原文字数: {self.count_cn_words(original_text)} 个中文字符；定稿必须保持在 90%-120% 区间，严禁删减剧情或注水扩写。\n"
            + "- 只修复明确问题：逻辑链、人物动机、道具/数字一致性、伏笔断裂、明显 AI 句式。没有问题的句子保持原样。\n"
            + f"- 核心角色必须保留: {characters}\n"
            + "- 下列事件/物件锚点必须尽量保留，不得无故丢失:\n"
            + "\n".join(f"  - {item}" for item in event_checklist)
            + "\n- 只输出正文，不输出解释、报告、markdown 代码块或问候语。\n"
            + "\n## 待处理正文\n```text\n"
            + source_block
            + "\n```\n"
        )

    def extract_event_checklist(self, text: str, limit: int = 12) -> List[str]:
        candidates = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
        stop = {"一个", "这种", "那种", "时候", "自己", "什么", "没有", "只是", "起来", "下去", "出去", "回来"}
        counts = Counter(c for c in candidates if c not in stop)
        return [word for word, _ in counts.most_common(limit)]

    def validate_llm_final_text(self, original_text: str, final_text: str, context: NovelContext, chapter_name: str = "") -> Tuple[bool, str]:
        if not final_text or not re.search(r"[\u4e00-\u9fff]", final_text):
            return False, "empty_or_non_chinese"
        if "```" in final_text or re.search(r"^(好的|以下是|我将|已完成)", final_text):
            return False, "contains_chatty_or_markdown_artifacts"
        original_words = self.count_cn_words(original_text)
        final_words = self.count_cn_words(final_text)
        if original_words >= 200:
            low = int(original_words * 0.90)
            high = int(original_words * 1.25)
            if final_words < low or final_words > high:
                return False, f"word_count_out_of_range:{final_words}/{original_words} (Required: {low}-{high})"
        # required_chars = [c.get("name", "") for c in context.characters if c.get("name") and c.get("name") in original_text]
        # missing = [name for name in required_chars[:12] if name not in final_text]
        # if missing:
        #     return False, "missing_characters:" + ",".join(missing)
        # anchors = self.extract_event_checklist(original_text, limit=10)
        # if anchors:
        #     kept = sum(1 for anchor in anchors if anchor in final_text)
        #     if kept < max(1, len(anchors) // 2):
        #         return False, f"lost_event_anchors:{kept}/{len(anchors)}"

        # --- V12 Hook Retention Audit (黄金三原则) ---
        # if "楔" in chapter_name or "序" in chapter_name or "第1章" in chapter_name:
        #     hook_report = self.prologue_hook.analyze_opening(final_text[:1000])
        #     if hook_report['total_score'] < 60:
        #         return False, f"opening_hook_failed: score={hook_report['total_score']} - {hook_report['issues'][0] if hook_report['issues'] else '张力不足'}"

        # --- V15 因果链完整性校验 (P0 级门控) ---
        # causality_report = self.causality_auditor.audit_chapter(final_text, chapter_name)
        # if causality_report["opening_issues"]:
        #     return False, f"causality_failed: {causality_report['opening_issues'][0]}"

        return True, "ok"

    def rewrite_book(
        self,
        chapters: Sequence[Tuple[str, str]],
        mode: str = "conservative",
        min_hits_to_rewrite: int = 1,
        export_tasks_only: bool = True
    ) -> Tuple[List[Tuple[str, str]], List[SentenceChange]]:
        revised_chapters: List[Tuple[str, str]] = []
        all_changes: List[SentenceChange] = []
        for name, text in chapters:
            rev_txt, changes = self.rewrite_chapter(name, text, mode=mode, min_hits_to_rewrite=min_hits_to_rewrite, export_tasks_only=export_tasks_only)
            revised_chapters.append((name, rev_txt))
            all_changes.extend(changes)
        return revised_chapters, all_changes

    def build_comparison_report(self, ori_ana, rev_ana, changes) -> BookReport:
        b_map = {ch["chapter"]: ch for ch in ori_ana["chapters"]}
        a_map = {ch["chapter"]: ch for ch in rev_ana["chapters"]}
        ch_counter = Counter(c.chapter for c in changes)
        chapters = []
        for name in b_map:
            b, a = b_map[name], a_map.get(name, b_map[name])
            ch_conflicts = [c for c in self.logic_auditor.conflicts if name in c.chapters_involved]

            chapters.append(ChapterReport(
                name, b["word_count"], a["word_count"],
                b["ai_score"], a["ai_score"],
                b["issue_summary"], a["issue_summary"],
                b["paragraph_tag_distribution"], a["paragraph_tag_distribution"],
                b["top_rules"], a["top_rules"],
                ch_counter.get(name, 0),
                b["suspicious_sentences"], a["suspicious_sentences"],
                b["empty_transition_blocks"], a["empty_transition_blocks"],
                logic_conflicts=ch_conflicts
            ))

        return BookReport(
            ori_ana["total_words"], rev_ana["total_words"],
            ori_ana["chapter_count"], ori_ana["ai_score"], rev_ana["ai_score"],
            ori_ana["global_rule_counts"], rev_ana["global_rule_counts"],
            ori_ana["global_issue_counts"], rev_ana["global_issue_counts"],
            len(changes), chapters,
            logic_summary=self.logic_auditor.conflicts
        )

    def analyze_chapter(
        self,
        chapter_name: str,
        chapter_text: str,
        context: Optional[NovelContext] = None,
        persist: bool = False,
    ) -> Dict[str, object]:
        chars = context.characters if context else None
        if persist:
            self.logic_auditor.extract_facts(chapter_name, chapter_text, chars)

        paragraphs = self.split_paragraphs(chapter_text)
        paragraph_infos = []
        all_hits = []
        suspicious = []

        # Advanced Analyzers
        structure_issues = []
        redundancy_issues = self.redundancy_detector.detect_paraphrase_loops(paragraphs)
        for iss in redundancy_issues:
            structure_issues.append(f"段落 {iss['para_idx']} 与下一段信息重合度高达 {iss['overlap_ratio']} : {iss['reason']}")

        pov_issues = self.pov_checker.check(chapter_text)
        for iss in pov_issues:
            structure_issues.append(f"全知视角泄露: '{iss['hit']}' - {iss['reason']}")

        rhythm_report = self.rhythm_shatter.analyze(self.split_sentences(chapter_text))
        if rhythm_report['is_uniform']:
            structure_issues.append(f"句长节奏严重均匀化 (变异系数 {rhythm_report['cv']})，缺乏文学呼吸感。")

        sensory_report = self.sensory_weaver.diagnose(chapter_text)
        if sensory_report['visual_dominant']:
            structure_issues.append(f"感官极度倾斜视觉 ({sensory_report['total_non_visual']} 个非视觉)，缺乏沉浸体感。")

        if "楔" in chapter_name or "序" in chapter_name or "第1章" in chapter_name:
            hook_report = self.prologue_hook.analyze_opening(chapter_text[:1000])
            if not hook_report['has_hook'] or hook_report['total_score'] < 60:
                for iss in hook_report['issues']:
                    structure_issues.append(f"【黄金三原则审计失败】: {iss}")

        # V15: 叙事因果链完整性审计
        fates = self.logic_auditor.get_character_fates()
        causality_report = self.causality_auditor.audit_chapter(chapter_text, chapter_name, character_fates=fates)
        if causality_report["total_issues"] > 0:
            for fi in causality_report.get("fate_issues", []):
                structure_issues.append(f"【因果链-命运冲突】: {fi}")
            for de in causality_report["dangling_entities"]:
                structure_issues.append(f"【因果链-悬空实体】: {de['reason']}")
            for cg in causality_report["causal_gaps"]:
                structure_issues.append(f"【因果链-因果断裂】: {cg['reason']}")
            for sj in causality_report["scene_jumps"]:
                structure_issues.append(f"【因果链-场景跳跃】: {sj['reason']}")
            for eg in causality_report["emotion_gaps"]:
                structure_issues.append(f"【因果链-情感断裂】: {eg['reason']}")
            for tb in causality_report["timeline_breaks"]:
                structure_issues.append(f"【因果链-时间线断裂】: {tb['reason']}")
            for oi in causality_report["opening_issues"]:
                structure_issues.append(f"【因果链-开篇审计】: {oi}")
            for si in causality_report.get("summary_issues", []):
                structure_issues.append(f"【因果链-结尾总结废话】: {si}")
            for kl in causality_report.get("knowledge_leaks", []):
                structure_issues.append(f"【因果链-读心逻辑冲突】: {kl['reason']} (涉及实体: {kl['entity']})")

        # P1 Fix #6: 集成身份论断式总结检测器
        identity_issues = self.identity_summary_detector.check(chapter_text)
        for iss in identity_issues:
            structure_issues.append(f"身份论断式总结: '{iss['hit'][:40]}...' - {iss['reason']}")

        # P1 Fix #6: 集成套路词/油腻微表情检测器
        cliche_issues = self.cliche_express_detector.check(chapter_text)
        for iss in cliche_issues:
            structure_issues.append(f"套路词/油腻微表情: '{iss['hit']}' - {iss['reason']}")

        for p_idx, paragraph in enumerate(paragraphs):
            info = self.classify_paragraph(chapter_name, p_idx, paragraph)
            p_severity_counts = Counter()

            sents = self.split_sentences(paragraph)
            # Dialogue Check
            dialogue_flags = self.dialogue_analyzer.analyze_dialogue_lines(sents)
            for flag in dialogue_flags:
                structure_issues.append(f"段落 {p_idx+1} {flag}")

            for s_idx, sentence in enumerate(sents):
                hits = self.detect_rules_in_sentence(chapter_name, p_idx, s_idx, sentence)

                # SDT Check
                sdt_flags = self.sdt_transformer.check(sentence)
                for flag in sdt_flags:
                    hits.append(Hit(chapter_name, p_idx, s_idx, "SDT违规", flag, sentence, 0, len(sentence), 1.0, "P2"))

                # P0 Fix #2: 集成解释性尾注检测器
                tail_tag_issues = self.tail_tag_detector.check(sentence)
                for iss in tail_tag_issues:
                    hits.append(Hit(chapter_name, p_idx, s_idx, "explanatory_tail_tag", f"解释性尾注: {iss['reason']}", sentence, 0, len(sentence), 3.0, "P1"))

                # P1 Fix #5: 跨句解释性尾注检测
                if s_idx > 0:
                    cross_result = self.tail_tag_detector._analyze_cross_sentence(sents[s_idx - 1], sentence)
                    if cross_result:
                        hits.append(Hit(chapter_name, p_idx, s_idx, "explanatory_tail_tag", f"跨句解释性尾注: {cross_result['reason']}", sentence, 0, len(sentence), 3.0, "P1"))

                if hits:
                    all_hits.extend(hits)
                    for h in hits: p_severity_counts[h.severity] += 1
                    suspicious.append({
                        "paragraph": p_idx+1,
                        "sentence": s_idx+1,
                        "text": sentence,
                        "rules": sorted({h.rule_name for h in hits}),
                        "severity": sorted({h.severity for h in hits})
                    })

            info.hits_by_severity = dict(p_severity_counts)
            paragraph_infos.append(info)

        ai_score, issues, tags, top = self.score_chapter(
            self.count_cn_words(chapter_text),
            all_hits,
            paragraph_infos,
            self.detect_empty_transition_blocks(paragraph_infos)
        )

        return {
            "chapter": chapter_name,
            "word_count": self.count_cn_words(chapter_text),
            "paragraph_count": len(paragraphs),
            "sentence_count": sum(p.sentence_count for p in paragraph_infos),
            "ai_score": ai_score,
            "issue_summary": issues,
            "paragraph_tag_distribution": tags,
            "top_rules": top,
            "suspicious_sentences": suspicious[:30],
            "empty_transition_blocks": self.detect_empty_transition_blocks(paragraph_infos),
            "structure_issues": structure_issues
        }

    def analyze_book(
        self,
        chapters: Sequence[Tuple[str, str]],
        context: Optional[NovelContext] = None,
        persist: bool = False,
    ) -> Dict[str, object]:
        reports = [self.analyze_chapter(n, t, context, persist=persist) for n, t in chapters]
        ai_s = round(sum(r["ai_score"] for r in reports) / max(len(reports), 1), 2)
        g_rules, g_issues = Counter(), Counter()
        for r in reports:
            for it in r["top_rules"]: g_rules[it["rule_name"]] += int(it["count"])
            for iss, count in r["issue_summary"].items(): g_issues[iss] += count
        return {"total_words": sum(r["word_count"] for r in reports), "chapter_count": len(reports), "ai_score": ai_s, "global_rule_counts": dict(g_rules.most_common()), "global_issue_counts": dict(g_issues.most_common()), "chapters": reports}

    def score_chapter(self, words, hits, p_infos, empty) -> Tuple[float, Dict[str, int], Dict[str, int], List[Dict[str, object]]]:
        iss_c = Counter(h.hit_type for h in hits)
        rule_c = Counter(h.rule_name for h in hits)
        sev_c = Counter(h.severity for h in hits)
        tag_c = Counter()
        for p in p_infos: tag_c.update(p.tags)

        per_k = max(words / 1000.0, 0.5)  # 设置下限，防止短文本分数虚高

        score_base = (
            sev_c.get("P0", 0) * 15.0 +
            sev_c.get("P1", 0) * 5.0 +
            sev_c.get("P2", 0) * 1.0 +
            len(empty) * 8.0
        )

        penalty = 0.0
        target_sensory = self.style_profile.get("sensory", 2.0) if self.style_profile else 2.0
        target_variance = self.style_profile.get("variance", 25.0) if self.style_profile else 25.0

        for p in p_infos:
            if p.sentence_count > 2 and p.sensory_score < target_sensory:
                penalty += 1.5 # Sensory density penalty
            if p.sentence_count > 2 and p.rhythm_score < (target_variance / 5.0):
                penalty += 1.0 # Rhythm penalty

            if p.omniscient_score > 0.8: penalty += 2.0

        final_score = (score_base + penalty) / per_k
        iss_c.update({f"Severity_{k}": v for k, v in sev_c.items()})

        return max(0.0, min(100.0, round(final_score, 2))), dict(iss_c), dict(tag_c), [{"rule_name": r, "count": c} for r, c in rule_c.most_common(12)]

    def write_json_report(self, report, path): path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    def write_markdown_report(self, report, changes, path):
        lines = [
            "# 🩺 小说工业级质量与逻辑审计报告 (V3)",
            f"- AI 味评分: {report.ai_score_before} -> {report.ai_score_after}",
            f"- 逻辑冲突检测: {len(report.logic_summary)} 处",
            ""
        ]

        if report.logic_summary:
            lines.append("## 🔴 严重逻辑冲突 (Consistency Alert)")
            for c in report.logic_summary:
                lines.append(f"- **[{c.type}]** {c.description} (涉及章节: {', '.join(c.chapters_involved)})")
            lines.append("")

        lines.append("## 📚 章节细节")
        for ch in report.chapters:
            lines.append(f"### {ch.chapter}")
            lines.append(f"- AI 分数: {ch.ai_score_before} -> {ch.ai_score_after}")
            if ch.logic_conflicts:
                for lc in ch.logic_conflicts:
                    lines.append(f"  - ⚠️ 逻辑预警: {lc.description}")

        path.write_text("\n".join(lines), encoding="utf-8")
    def write_changes_csv(self, changes, path):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["chapter", "paragraph", "sentence", "risk", "rules", "orig", "rev"])
            for c in changes: w.writerow([c.chapter, c.paragraph, c.sentence, c.risk_level, "|".join(c.applied_rules), c.original, c.revised])
    def write_revised_texts(self, chapters, out_dir, name):
        rev_dir = out_dir / "revised_texts"; rev_dir.mkdir(parents=True, exist_ok=True)
        combined = []
        for idx, (ch_name, text) in enumerate(chapters, 1):
            (rev_dir / f"{idx:03d}_{ch_name}.txt").write_text(text, encoding="utf-8")
            combined.extend([ch_name, text, ""])
        (out_dir / f"{name}_revised.txt").write_text("\n".join(combined), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input directory or file (optional if --learn is used)")
    parser.add_argument("--book-id", help="Unique ID for partitioning data (defaults to input name)")
    parser.add_argument("--mode", choices=["conservative", "aggressive"], default="conservative")
    parser.add_argument("--recursive", action="store_true", help="Search for .md files recursively")
    parser.add_argument("--learn", action="store_true", help="Launch evolution mode: learn from human corrections")
    parser.add_argument("--master-novels", nargs="*", help="Paths to master novel txt files to learn from")
    parser.add_argument("--allow-llm-mining", action="store_true", help="Enable LLM-assisted rule mining in learning mode")
    parser.add_argument("--enable-llm-finalize", action="store_true", help="Allow final LLM rewrite pass after task generation")
    parser.add_argument("--enable-llm-audit", action="store_true", help="Allow LLM-assisted task hooks and persisted fact extraction")
    parser.add_argument("--enable-global-index", action="store_true", help="Enable ChromaDB full-book indexing (off by default to reduce cost)")
    parser.add_argument("--task-full-text", action="store_true", help="Inline full chunk text in agent task files (off by default)")
    parser.add_argument("--max-chunk-chars", type=int, default=3500, help="Maximum characters per processing chunk")
    parser.add_argument("--llm-mode", choices=["patch", "full"], default="patch", help="API LLM repair mode: patch sends only selected paragraphs; full rewrites the whole chunk")
    parser.add_argument("--llm-max-patches", type=int, default=8, help="Maximum paragraphs sent to API LLM in patch mode")
    parser.add_argument("--llm-context-paragraphs", type=int, default=1, help="Neighbor paragraphs included around each API LLM patch target")
    parser.add_argument("--llm-token-budget", type=int, default=6000, help="Approximate prompt token budget per API LLM patch call")
    args = parser.parse_args()

    engine = EnhancedNovelAISurgery(
        allow_llm_task_hooks=args.enable_llm_audit,
        allow_llm_fact_extraction=args.enable_llm_audit,
        allow_llm_finalize=args.enable_llm_finalize,
        enable_global_index=args.enable_global_index,
        task_include_full_text=args.task_full_text,
        max_chunk_size=args.max_chunk_chars,
        llm_mode=args.llm_mode,
        llm_max_patches=args.llm_max_patches,
        llm_context_paragraphs=args.llm_context_paragraphs,
        llm_token_budget=args.llm_token_budget,
    )
    in_p = Path(args.input) if args.input else None

    # Evolution Mode
    if args.learn:
        zhengwen_dir = Path("/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI/duanpian_zhengwen")
        engine.evolution_engine.batch_learn_from_human_samples(
            zhengwen_dir,
            allow_llm_mining=args.allow_llm_mining,
        )

        if args.master_novels:
            researcher = LiteraryResearcher()
            style_tags = {
                "大奉打更人": "探案/幽默/生活化",
                "凡人修仙传": "冷静/简练/凡人流",
                "剑来": "深沉/文青/群像",
                "庆余年": "权谋/吐槽/历史感",
                "神墓": "宏大/热血/古典",
                "十日终焉": "悬疑/逻辑/人性",
                "万族之劫": "爽快/多线/热血"
            }
            print(f"📚 启动名家文笔扫描 (抽样 50w 字)...")
            for path_str in args.master_novels:
                p = Path(path_str)
                if p.exists():
                    label = p.stem
                    tag = style_tags.get(label, "GeneralHuman")
                    stats = researcher.analyze_novel(p)
                    engine.evolution_engine.save_benchmark(tag, stats)
                    print(f"  ✅ 已学习 [{label}] -> 风格: {tag} | 感官指数: {stats['sensory']} | 节奏方差: {stats['variance']}")
        return

    # P0 Fix #4: 防止 in_p 为 None 时空指针
    if in_p is None:
        parser.error("--input 参数在非 --learn 模式下为必填项")
        return
    if not in_p.exists():
        print(f"❌ 路径不存在: {in_p}")
        return

    # Recursive logic
    if args.recursive and in_p.is_dir():
        files = list(in_p.rglob("*.md"))
    elif in_p.is_dir():
        files = list(in_p.glob("*.md"))
    else:
        files = [in_p]

    # Filter out already processed files
    files = [f for f in files if not f.name.endswith("_out.md")]

    print(f"🚀 开始批量处理 {len(files)} 个任务文件...")

    for f in files:
        try:
            print(f"\n--- 处理中: {f.name} ---")
            book_id = args.book_id or f.stem
            engine.logic_auditor.set_book_context(book_id)

            content = f.read_text(encoding="utf-8", errors="ignore")
            # Analyze context & entities
            context = engine.context_analyzer.analyze(content)
            context.book_id = book_id
            context.characters = engine.entity_tracker.load_characters(book_id)
            context.characters = engine.entity_tracker.extract_from_text(content, context.characters)
            engine.entity_tracker.save_characters(book_id, context.characters)

            print(f"🔍 识别结果: 频道={context.genre}, 背景={context.background}, 角色数={len(context.characters)}")

            # --- V14: Global Vision Ingestion ---
            if engine.global_hook_engine.collection:
                print(f"📦 正在对 [{f.name}] 进行全局向量索引以供分析...")
                engine.global_hook_engine.ingest_novel(content)

            # Process (With Chunking for Long Files)
            chapters = engine.split_book_by_chapters(content)
            rev_texts = []
            all_changes = []
            for c_name, c_body in chapters:
                if len(chapters) > 1:
                    print(f"  -> 处理分块/章节: {c_name} (字数: {len(c_body)})")
                chunk_rev_text, chunk_changes = engine.rewrite_chapter(
                    f"{f.stem}_{c_name}" if c_name != "全文" else f.stem,
                    c_body,
                    context=context,
                    mode=args.mode,
                    export_tasks_only=(not args.enable_llm_finalize),
                )
                rev_texts.append(chunk_rev_text)
                all_changes.extend(chunk_changes)

            rev_text = "\n\n".join(rev_texts)
            changes = all_changes

            # --- P0 Fix: Final Parity Gate (保存前终极门控) ---
            orig_cn = engine.count_cn_words(content)
            rev_cn = engine.count_cn_words(rev_text)
            ratio = rev_cn / max(orig_cn, 1)
            if orig_cn >= 200 and ratio < 0.85:
                print(f"\U0001f6a8 [P0:\u4fdd\u91cf\u62e6\u622a] {f.name}: \u539f\u6587{orig_cn}\u5b57\u2192\u91cd\u5199{rev_cn}\u5b57 ({ratio:.0%})\uff0c\u4e25\u7981\u4fdd\u5b58\uff01")
                quarantine_dir = engine.root_dir / "quarantine"
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                q_path = quarantine_dir / f"{f.stem}_\u7f29\u6c34\u88ab\u62e6\u622a.md"
                q_path.write_text(rev_text, encoding="utf-8")
                print(f"\U0001f4e6 \u7f29\u6c34\u7248\u672c\u5df2\u9694\u79bb\u5230: quarantine/{f.stem}_\u7f29\u6c34\u88ab\u62e6\u622a.md")
                continue  # \u8df3\u8fc7\u4fdd\u5b58\uff0c\u5904\u7406\u4e0b\u4e00\u4e2a\u6587\u4ef6
            print(f"\U0001f4ca \u5b57\u6570\u4fdd\u91cf: {orig_cn}\u2192{rev_cn} ({ratio:.0%})")

            # Save Output
            out_p = f.parent / f"{f.stem}_out.md"
            out_p.write_text(rev_text, encoding="utf-8")
            print(f"\u2705 \u5904\u7406\u5b8c\u6210: {out_p.name}")

            # Formal audit pass: explicitly persist logic facts.
            engine.analyze_chapter(f.stem, rev_text, context=context, persist=True)
            engine.logic_auditor.save_ledger()

        except ValueError as e:
            print(f"❌ 识别失败 (需要手动干预): {e}")
        except Exception as e:
            logger.exception(f"💥 处理 {f.name} 时发生未知错误: {e}")

    print("\n✨ 所有批量任务处理完毕。")

if __name__ == "__main__":
    main()
