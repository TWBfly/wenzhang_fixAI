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
from advanced_analyzers import RhythmShatter, SensoryWeaver, RedundancyDetector, POVChecker, DialoguePowerAnalyzer, ShowDontTellTransformer, ExplanatoryTailTagDetector
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
    type: str  # ITEM_STATUS, CHAR_LIFE, CHAR_LOC, INFO_KNOWN, SUBTEXT_ANCHOR
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


class LogicAuditor:
    """Atomic Logic & Causal Alignment (ALCA) engine with SQLite persistence."""
    def __init__(self, db_path: Path):
        self.db_path = db_path
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

    def set_book_context(self, book_id: str):
        self.book_id = book_id
        self.load_ledger()

    def extract_facts(self, chapter_name: str, text: str):
        """Regex-based atomic fact extraction."""
        for item_key, pattern in self.item_patterns.items():
            # Match variations like "拿走了钱", "把钱拿走"
            if re.search(rf"(?:拿走|夺走|抢走|没收|毁掉|扔掉).*?({pattern})|把.*?({pattern}).*?(?:拿走|夺走|抢走|没收|毁掉|扔掉)", text):
                self.ledger.append(LogicFact(chapter_name, "ITEM_STATUS", item_key, "lost", f"失去{item_key}"))
            elif re.search(rf"(?:拿到|找回|掏出|取出|拿了).*?({pattern})|把.*?({pattern}).*?(?:拿到|找回|掏出|取出|拿了)", text):
                self._check_item_conflict(chapter_name, item_key, "possession")
                self.ledger.append(LogicFact(chapter_name, "ITEM_STATUS", item_key, "held", f"持有{item_key}"))
        
        if "车" in text and ("山路" in text or "公路" in text):
            self.ledger.append(LogicFact(chapter_name, "CHAR_LOC", "PROTAG", "in_vehicle", "在车上行驶"))

        # Information Asymmetry detection
        # Logic: If text describes character's inner state of a hidden secret NOT yet revealed, track it.
        # If narrator explains something that NO character in scene could know, flag OMNISCIENCE_LEAK.
        omniscient_markers = [r"由于他不知道", r"他并不知道", r"在这个世界上没有人知道", r"正如他预料的一样", r"事实上"]
        for p in omniscient_markers:
            if re.search(p, text):
                self.conflicts.append(LogicConflict(
                    "OMNISCIENCE_LEAK",
                    f"叙事泄露：使用了上帝视角表达 '{p}'，破坏了角色沉浸感",
                    "P1",
                    [chapter_name]
                ))

        # Character Knowledge Tracking (Simple)
        # If a character speaks about a secret item not yet encountered, flag KNOWLEDGE_GAP.
        # (This is a simplified programmatic version of the cognitive engine)

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
        pass


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
            # Migration: Ensure frequency column exists for older database versions
            try:
                conn.execute("ALTER TABLE evolution_rules ADD COLUMN frequency INTEGER DEFAULT 1")
            except sqlite3.OperationalError:
                pass # Already exists

    def load_knowledge(self):
        self.knowledge = {"rules": [], "patterns": {}}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name, pattern, weight, category, severity, frequency FROM evolution_rules")
            for row in cursor:
                # Dynamic weight adjustment based on frequency
                # The more a rule is triggered (and corrected by human), the higher its weight
                base_weight = row[2]
                freq = row[5]
                dynamic_weight = base_weight * (1.0 + (freq - 1) * 0.5) # +50% weight per recurrence
                
                rule_dict = {
                    "name": row[0], "pattern": row[1], "weight": dynamic_weight,
                    "category": row[3], "severity": row[4]
                }
                self.knowledge["rules"].append(rule_dict)
                # Synergize with main system
                self.main_system.tier1_rules.append(PatternRule(**rule_dict))
                self.main_system.all_rules.append(PatternRule(**rule_dict))

    def learn_from_pair(self, original_text: str, edited_text: str):
        """Analyze diff and extract improvements."""
        # Detect patterns that were in original but removed/changed in edited
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
                self.add_evolution_rule(name, pattern, weight, cat, "P1")

    def batch_learn_from_human_samples(self, directory: Path):
        """Walk through duanpian_zhengwen/X/ folders and learn from Original vs Human pairs."""
        print(f"🧬 启动人类经验学习流程: {directory}")
        for folder in directory.glob("*"):
            if folder.is_dir():
                ori_p = folder / "原文.md"
                hum_p = folder / "人工修改之后.md"
                if ori_p.exists() and hum_p.exists():
                    print(f"  - 学习样本: {folder.name}")
                    self.learn_from_pair(ori_p.read_text(encoding='utf-8', errors='ignore'), 
                                       hum_p.read_text(encoding='utf-8', errors='ignore'))
        self.load_knowledge()

    def add_evolution_rule(self, name: str, pattern: str, weight: float, category: str, severity: str):
        with sqlite3.connect(self.db_path) as conn:
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

    def save_benchmark(self, label: str, stats: Dict[str, float]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO style_benchmarks 
                (label, sensory_index, rhythm_variance, action_tag_density, top_keywords)
                VALUES (?, ?, ?, ?, ?)
            """, (label, stats['sensory'], stats['variance'], stats['action_tag'], json.dumps(stats['keywords'])))

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

    def analyze_novel(self, path: Path, sample_limit=500000) -> Dict:
        content = path.read_text(encoding='utf-8', errors='ignore')[:sample_limit]
        words_count = len(content)
        sentences = re.split(r'[。！？]', content)
        lens = [len(s) for s in sentences if len(s) > 1]
        
        # Sensory Index
        hits = 0
        for s in self.sensory_lexicon:
            hits += content.count(s)
        sensory_index = (hits / (words_count / 1000.0)) * 10
        
        # Rhythm Variance
        if len(lens) > 1:
            mean = sum(lens) / len(lens)
            variance = (sum((x - mean)**2 for x in lens) / len(lens))**0.5
        else:
            variance = 0.0

        # Action Tag Density in Dialogue
        # Look for patterns like "说道，..." or "，说道" where action/desc follows
        tags = len(re.findall(r'[“][^”]+[”][^。！？\n]*[，。]', content))
        action_tag_density = (tags / (words_count / 1000.0))

        # Top Keywords (High specificity)
        # Simplified: top chars or unique objects
        return {
            "sensory": round(sensory_index, 3),
            "variance": round(variance, 3),
            "action_tag": round(action_tag_density, 3),
            "keywords": [] # Placeholder for future POS tagging
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
        stop_names = {"我", "他", "她", "谁", "你", "咱们", "之类", "一个", "这种", "那种", "朕", "本王", "摄政王", "主子"}
        
        # Simple heuristic extraction: Names followed by dialogue verbs
        matches = re.findall(r"([\u4e00-\u9fff]{2,4})(?:说|道|问|喊|沉声|冷笑|开口|唤道)", text)
        new_chars = [c for c in set(matches) if c not in stop_names and len(c) >= 2]
        
        updated = {c["name"]: c for c in existing}
        for name in new_chars:
            if name not in updated:
                updated[name] = {"name": name, "role": "detected", "desc": "自动识别角色"}
        
        return list(updated.values())


class AgentRefinementTask:
    """Generates high-precision Multi-Agent refinement tasks."""
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.output_dir = root_dir / "agent_tasks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = root_dir / "templates" / "fixAI_template.md"

    def generate_task(self, chapter_name: str, text: str, context: NovelContext, changes: List[SentenceChange], structure_issues=None) -> Path:
        char_str = ", ".join([c["name"] for c in context.characters])
        
        # Build precise diagnostic micro-instructions
        instructions = []
        instructions.append(f"# 精准改写任务: {chapter_name}")
        instructions.append(f"频道: {context.genre} | 背景: {context.background}")
        instructions.append(f"出场角色: {char_str}\n")
        
        if structure_issues:
            instructions.append("## 结构级高危红线 (请在全局优化时重点修正)")
            for iss in structure_issues:
                instructions.append(f" - ⚠️ {iss}")
            instructions.append("\n")
            
        instructions.append("## 句子级微指令 (由审计引擎截获的目标片段)")
        if not changes:
            instructions.append("暂无高危句子级需要改写，请全局核实逻辑。")
        else:
            for idx, chg in enumerate(changes[:30]): # 取前30个严重问题
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
        
        instructions.append("\n## 参考全文内容\n```text\n" + text[:4000] + "\n...```\n")
        
        task_path = self.output_dir / f"refine_{chapter_name}.md"
        task_path.write_text("\n".join(instructions), encoding="utf-8")
        return task_path


class EnhancedNovelAISurgery:
    def __init__(self) -> None:
        self.root_dir = Path(__file__).parent
        self.db_path = self.root_dir / "db" / "novel_logic.db"
        self.logic_auditor = LogicAuditor(self.db_path)
        self.context_analyzer = NovelContextAnalyzer()
        self.entity_tracker = EntityTracker(self.root_dir)
        self.agent_task_gen = AgentRefinementTask(self.root_dir)
        self.style_profile = None  # Init missing variable
        # Advanced analyzers
        self.rhythm_shatter = RhythmShatter()
        self.sensory_weaver = SensoryWeaver()
        self.redundancy_detector = RedundancyDetector()
        self.pov_checker = POVChecker()
        self.dialogue_analyzer = DialoguePowerAnalyzer()
        self.sdt_transformer = ShowDontTellTransformer()
        self.tail_tag_detector = ExplanatoryTailTagDetector()
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
        self.evolution_engine = EvolutionEngine(self, self.db_path)
        for r_dict in self.evolution_engine.knowledge["rules"]:
            self.all_rules.append(PatternRule(**r_dict))

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
        }


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

    def split_book_by_chapters(
        self,
        text: str,
        chapter_pattern: str = DEFAULT_CHAPTER_PATTERN,
    ) -> List[Tuple[str, str]]:
        text = self.normalize_text(text)
        matches = list(re.finditer(chapter_pattern, text))
        if not matches:
            return [("全文", text)]

        chapters: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            title = m.group(0).strip()
            body = text[start:end].strip()
            chapters.append((title, body))
        return chapters

    @staticmethod
    def split_paragraphs(text: str) -> List[str]:
        return [p.strip() for p in re.split(r"\n+", text) if p.strip()]

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        pieces = re.split(r"(?<=[。！？；!?;])", text)
        sentences = []
        buffer = ""

        for piece in pieces:
            if not piece:
                continue
            buffer += piece
            if re.search(r"[。！？；!?;]$", piece):
                sentence = buffer.strip()
                if sentence:
                    sentences.append(sentence)
                buffer = ""

        if buffer.strip():
            sentences.append(buffer.strip())
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
            for m in re.finditer(rule.pattern, sentence):
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
    def cleanup_text_fragment(text: str) -> str:
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
            if len(b) <= 10:
                return f"分分是{b}{tail}"
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
            r"不是([^，。！？；\n]{1,20})[，, ]?而是([^。！？；\n]{1,24})([。！？；，,]|$)",
            repl_not_but,
            text,
        )
        text = re.sub(
            r"不是([^，。！？；\n]{1,20})[，, ]?也不是([^。！？；\n]{1,24})([。！？；，,]|$)",
            repl_not_but,
            text,
        )
        return text

    def rewrite_replacement_patterns(self, text: str, applied: List[str]) -> str:
        def repl_replace(m: re.Match[str]) -> str:
            applied.append("取而代之")
            tail = m.group(1).strip("，, ")
            return f"只剩下{tail}"

        text = re.sub(r"取而代之(?:的|地)?(?:是)?([^。！？；\n]{1,30})", repl_replace, text)
        return text

    def rewrite_explanatory_narration(self, text: str, applied: List[str]) -> str:
        before = text
        text = re.sub(r"([她他])(?:似乎|仿佛)?(?:知道|明白|意识到)，?", "", text)
        if text != before:
            applied.append("她知道/他知道")

        replacements = {
            r"很清楚，?": "",
            r"显然，?": "",
            r"不难看出，?": "",
            r"所谓的，?": "",
            r"名为[^。！？；\n]{1,15}的": "",
            r"更准确地说，?": "",
            r"本质上，?": "",
            r"这说明": "这下",
            r"这便是": "这也就是",
            r"这下子": "也就是",
            r"这种": "这种",
            r"这意味着": "这下",
            r"她并非": "她不是",
            r"他并非": "他不是",
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
        for key, options in self.rewrite_templates.items():
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
            local = self.rewrite_explanatory_narration(local, applied)
            local = self.rewrite_explanatory_narration(local, applied)
            local = self.rewrite_transition_fillers(local, applied)
            local = self.rewrite_cliche_phrases(local, applied)
            
            # Catch suspended warnings in the sentence
            sdt_flags = self.sdt_transformer.check(local)
            for flag in sdt_flags:
                applied.append(flag)
            
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
    ) -> Tuple[str, List[SentenceChange]]:
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
                
                should_rewrite = False
                if self.sdt_transformer.check(sentence): should_rewrite = True
                if any(h.severity == "P0" for h in s_hits): should_rewrite = True
                if any(h.rule_name in [r.name for r in self.tier1_rules] for h in s_hits): should_rewrite = True
                
                # Industrial Grade: Trigger rewrite if sensory density is too low or symmetry is too high
                if p_info.sensory_score < 0.15 and p_info.role in ["action", "environment"]:
                    should_rewrite = True
                if p_info.symmetry_penalty >= 1.0:
                    should_rewrite = True
                if p_info.omniscient_score >= 0.7:
                    should_rewrite = True

                tier2_names = {r.name for r in self.tier2_rules}
                p_tier2_hits = [h for h in p_hits if h.rule_name in tier2_names]
                if len(p_tier2_hits) >= 2: should_rewrite = True
                
                if not should_rewrite and len(s_hits) < min_hits_to_rewrite:
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
            final_report = self.analyze_chapter(chapter_name, rev_text)
            structure_issues = final_report.get("structure_issues", [])
            task_path = self.agent_task_gen.generate_task(chapter_name, rev_text, context, changes, structure_issues=structure_issues)

            print(f"📝 已生成 IDE Agent 协作任务: {task_path.name}")
            print(f"⚠️ 请 IDE Agent (如 Antigravity) 分别处理该任务。")
            
        return rev_text, changes

    def rewrite_book(
        self,
        chapters: Sequence[Tuple[str, str]],
        mode: str = "conservative",
        min_hits_to_rewrite: int = 1,
    ) -> Tuple[List[Tuple[str, str]], List[SentenceChange]]:
        revised_chapters: List[Tuple[str, str]] = []
        all_changes: List[SentenceChange] = []
        for name, text in chapters:
            rev_txt, changes = self.rewrite_chapter(name, text, mode, min_hits_to_rewrite)
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

    def analyze_chapter(self, chapter_name: str, chapter_text: str) -> Dict[str, object]:
        self.logic_auditor.extract_facts(chapter_name, chapter_text)
        
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

    def analyze_book(self, chapters: Sequence[Tuple[str, str]]) -> Dict[str, object]:
        reports = [self.analyze_chapter(n, t) for n, t in chapters]
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
        
        per_k = max(words / 1000.0, 1e-6)
        
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
    args = parser.parse_args()
    
    engine = EnhancedNovelAISurgery()
    in_p = Path(args.input) if args.input else None

    # Evolution Mode
    if args.learn:
        zhengwen_dir = Path("/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI/duanpian_zhengwen")
        engine.evolution_engine.batch_learn_from_human_samples(zhengwen_dir)
        
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

            # Process
            rev_text, changes = engine.rewrite_chapter(f.stem, content, context=context, mode=args.mode)
            
            # Save Output
            out_p = f.parent / f"{f.stem}_out.md"
            out_p.write_text(rev_text, encoding="utf-8")
            print(f"✅ 处理完成: {out_p.name}")
            
            # Save persistent state
            engine.logic_auditor.save_ledger()
            
        except ValueError as e:
            print(f"❌ 识别失败 (需要手动干预): {e}")
        except Exception as e:
            logger.exception(f"💥 处理 {f.name} 时发生未知错误: {e}")

    print("\n✨ 所有批量任务处理完毕。")

if __name__ == "__main__":
    main()
