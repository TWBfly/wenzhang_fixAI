from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_CHAPTER_PATTERN = r"(?m)^(楔子|序章|前言|尾声|后记|番外|第[零一二三四五六七八九十百千万0-9]+章[^\n]*|第[零一二三四五六七八九十百千万0-9]+回[^\n]*)$"


@dataclass
class PatternRule:
    name: str
    pattern: str
    weight: float
    category: str
    severity: str = "soft"


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


@dataclass
class ParagraphInfo:
    chapter: str
    paragraph_index: int
    text: str
    tags: List[str]
    scores: Dict[str, int]
    has_dialogue: bool
    sentence_count: int


@dataclass
class ChapterReport:
    chapter: str
    word_count: int
    paragraph_count: int
    sentence_count: int
    ai_score: float
    issue_summary: Dict[str, int]
    paragraph_tag_distribution: Dict[str, int]
    top_rules: List[Dict[str, object]]
    suspicious_sentences: List[Dict[str, object]]
    empty_transition_blocks: List[Dict[str, object]]
    rewrite_suggestions: List[Dict[str, object]]


@dataclass
class BookReport:
    total_words: int
    chapter_count: int
    ai_score: float
    global_rule_counts: Dict[str, int]
    global_issue_counts: Dict[str, int]
    chapters: List[ChapterReport]


class NovelAIDetector:
    def __init__(self) -> None:
        # -----------------------------
        # 1) 高频套板 / 强限制表达
        # -----------------------------
        self.hard_phrase_rules: List[PatternRule] = [
            PatternRule("指节泛白", r"指节泛白", 2.5, "phrase_repeat", "hard"),
            PatternRule("取而代之", r"取而代之(?:的|地)?(?:是)?", 2.5, "phrase_repeat", "hard"),
            PatternRule("不是A而是B", r"不是[^。！？；\n]{1,20}?而是[^。！？；\n]{1,24}", 3.0, "syntax_repeat", "hard"),
            PatternRule("与其说A不如说B", r"与其说[^。！？；\n]{1,24}?不如说[^。！？；\n]{1,24}", 3.0, "syntax_repeat", "hard"),
            PatternRule("眸色微沉", r"(?:眸色|眼神|神色)微沉", 2.0, "phrase_repeat", "hard"),
            PatternRule("呼吸一滞", r"呼吸一滞", 2.0, "phrase_repeat", "hard"),
            PatternRule("心头一震", r"心头一震", 2.0, "phrase_repeat", "hard"),
            PatternRule("空气凝滞", r"空气(?:仿佛|似乎)?[^。！？；\n]{0,10}凝滞", 2.0, "phrase_repeat", "hard"),
            PatternRule("周遭死寂", r"周遭[^。！？；\n]{0,10}(?:死寂|寂静|沉寂)", 2.0, "phrase_repeat", "hard"),
        ]

        # -----------------------------
        # 2) 软限制表达 / 过渡垫片
        # -----------------------------
        self.soft_phrase_rules: List[PatternRule] = [
            PatternRule("顿了顿", r"顿了顿", 1.0, "transition_overuse"),
            PatternRule("沉默片刻", r"(?:沉默了片刻|静了片刻|沉默片晌|半晌)", 1.0, "transition_overuse"),
            PatternRule("一时间", r"一时间", 1.0, "transition_overuse"),
            PatternRule("微微", r"微微", 0.4, "style_repeat"),
            PatternRule("缓缓", r"缓缓", 0.4, "style_repeat"),
            PatternRule("仿佛", r"仿佛", 0.4, "style_repeat"),
            PatternRule("目光落在", r"目光落在", 1.0, "transition_overuse"),
            PatternRule("抿唇", r"抿(?:了)?抿唇", 0.8, "style_repeat"),
        ]

        # -----------------------------
        # 3) 解释型旁白
        # -----------------------------
        self.explain_rules: List[PatternRule] = [
            PatternRule("她知道/他知道", r"[她他][似仿]?知道", 1.8, "narration_overuse"),
            PatternRule("她明白/他明白", r"[她他][似仿]?明白", 1.8, "narration_overuse"),
            PatternRule("很清楚", r"很清楚", 1.5, "narration_overuse"),
            PatternRule("这意味着", r"这意味着", 2.0, "narration_overuse"),
            PatternRule("显然", r"显然", 1.5, "narration_overuse"),
            PatternRule("并非而是", r"并非[^。！？；\n]{1,18}而是[^。！？；\n]{1,18}", 2.2, "narration_overuse"),
            PatternRule("更准确地说", r"更准确地说", 1.5, "narration_overuse"),
            PatternRule("本质上", r"本质上", 1.5, "narration_overuse"),
            PatternRule("百科式套话", r"综上所述|不可忽视的是|值得注意的是|在当今社会|显而易见|总的来说|首先|其次|再次|最后", 2.0, "narration_overuse"),
        ]

        # -----------------------------
        # 4) 环境垫场 / 无功能环境句
        # -----------------------------
        self.environment_rules: List[PatternRule] = [
            PatternRule("夜色沉沉", r"夜色(?:沉沉|深沉|浓重)", 1.3, "environment_overuse"),
            PatternRule("冷风穿堂", r"冷风[^。！？；\n]{0,12}(?:穿过|穿堂|吹进|拂过)", 1.3, "environment_overuse"),
            PatternRule("雨滴窗棂", r"雨(?:滴|声)[^。！？；\n]{0,12}(?:敲打|拍打)[^。！？；\n]{0,8}(?:窗|窗棂|窗纸)", 1.3, "environment_overuse"),
            PatternRule("烛火摇曳", r"烛火(?:轻轻)?摇曳", 1.3, "environment_overuse"),
            PatternRule("空气压抑", r"空气[^。！？；\n]{0,10}(?:压抑|沉重|发紧)", 1.3, "environment_overuse"),
        ]

        # -------------------------------------------------------------
        # 5) 解释性尾注 (Explanatory Tail Tag)
        #    AI 最隐蔽的指纹之一：前半句是具体生理/动作描写，
        #    后半句用「——那是/，那是/，这是」贴上抽象概念标签，
        #    替读者进行情感解码。人类作家让身体反应自己说话。
        #
        #    病灶结构:
        #      [具体描写] + [破折号/逗号] + [那是/这是] + [抽象概念]
        #    示例:
        #      ❌ 大腿根部的肌肉在打颤——那是原主身体残余的恐惧
        #      ❌ 全身的力气都往腰腹上使，那是杀红了眼的蛮劲
        #      ❌ 后背浮起一层冷汗——这是本能的警觉
        #      ✅ 大腿根部的肌肉在打颤。（删掉尾注，让颤抖自己说话）
        # -------------------------------------------------------------
        self.tail_tag_rules: List[PatternRule] = [
            # 核心模式：破折号 + 那是/这是 + 抽象情感/状态词
            PatternRule(
                "破折号解释尾注",
                r"——(?:那是|这是|那不是|这不是|这便是|那便是)[^。！？；\n]{2,30}(?:的(?:恐惧|恐慌|恐怖|愤怒|不甘|绝望|本能|警觉|直觉|习惯|渴望|执念|贪婪|占有欲|蛮劲|倔强|骄傲|尊严|底气|底线|温柔|善意|信号|宣言|挑衅|威胁|承诺|决心|觉悟|勇气|信任|默契|暗示|试探|妥协|退让|挣扎|不安|焦虑|痛苦|悲伤|孤独|寂寞|释然|解脱|报复|反击|示弱|示好|求救))",
                3.0, "explanatory_tail_tag", "hard"
            ),
            # 逗号引出的解释尾注
            PatternRule(
                "逗号解释尾注",
                r"[，,](?:那是|这是|这都是|那都是|那便是|这便是)[^。！？；\n]{2,30}(?:的(?:恐惧|恐慌|愤怒|不甘|绝望|本能|警觉|直觉|习惯|蛮劲|倔强|骄傲|底气|底线|温柔|挑衅|威胁|决心|觉悟|不安|痛苦|悲伤|挣扎|反击|试探|妥协|退让))",
                2.8, "explanatory_tail_tag", "hard"
            ),
            # 泛化模式：「那是 + 某种/一种 + 抽象名词」
            PatternRule(
                "某种抽象尾注",
                r"[——，,](?:那是|这是)(?:某种|一种|属于|来自)[^。！？；\n]{1,20}(?:的(?:力量|感觉|情绪|反应|本能|冲动|信号|暗示|直觉|预感|征兆))",
                2.5, "explanatory_tail_tag", "hard"
            ),
            # 身体反应 + 尾注解释模式（更宽泛的捕获）
            PatternRule(
                "生理反应解释尾注",
                r"(?:打颤|发抖|颤抖|颤栗|哆嗦|冷汗|鸡皮疙瘩|瞳孔[^。]{0,4}(?:收缩|放大)|心跳[^。]{0,4}(?:加速|漏拍)|呼吸[^。]{0,4}(?:急促|粗重))[^。！？；]{0,10}[——，,](?:那是|这是|这便是|那便是)[^。！？；\n]{2,25}",
                3.2, "explanatory_tail_tag", "hard"
            ),
            # 动作/姿态 + 解释性概括
            PatternRule(
                "动作概括尾注",
                r"(?:咬着牙|攥紧拳|握紧|绷紧|僵住|愣住|定住|蜷缩|后退|弓腰|弯腰|咽了口唾沫)[^。！？；]{0,15}[——，,](?:那是|这是|那种|这种)[^。！？；\n]{2,20}(?:蛮劲|劲|力道|姿态|架势|反应|惯性|条件反射)",
                2.8, "explanatory_tail_tag", "hard"
            ),
        ]

        self.all_rules: List[PatternRule] = (
            self.hard_phrase_rules
            + self.soft_phrase_rules
            + self.explain_rules
            + self.environment_rules
            + self.tail_tag_rules
        )

        # -----------------------------
        # 段落功能粗分类用词表
        # -----------------------------
        self.dialogue_verbs = ["说", "道", "问", "答", "喊", "喝", "笑", "骂", "低声", "开口"]
        self.action_verbs = [
            "抬", "抬手", "抬眼", "握", "攥", "退", "进", "站", "坐", "起身", "转身",
            "推", "拉", "按", "拍", "扫", "瞥", "盯", "看", "走", "迈", "靠近", "后退",
        ]
        self.emotion_words = [
            "紧张", "害怕", "愤怒", "不安", "慌乱", "难堪", "压抑", "委屈", "犹豫",
            "惊愕", "错愕", "冷意", "怒意", "不甘", "烦躁", "沉重"
        ]
        self.explain_words = [
            "知道", "明白", "清楚", "意味着", "显然", "本质上", "归根结底",
            "并非", "而是", "说明", "证明", "代表"
        ]
        self.env_words = [
            "夜", "月", "风", "雨", "雪", "雾", "灯", "烛火", "窗", "窗纸", "回廊",
            "庭院", "树影", "天色", "暮色", "夜色", "寒意", "空气"
        ]
        self.transition_words = ["顿了顿", "片刻", "半晌", "一时间", "良久", "沉默", "静了静"]

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
        # 近似按中文字符计数，适合中文小说场景
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

        return ParagraphInfo(
            chapter=chapter,
            paragraph_index=paragraph_index,
            text=paragraph,
            tags=sorted(set(tags)),
            scores=dict(scores),
            has_dialogue=has_dialogue,
            sentence_count=len(self.split_sentences(paragraph)),
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

    def generate_rewrite_suggestion(self, hit: Hit) -> Dict[str, object]:
        sentence = hit.text

        if hit.rule_name in {"指节泛白", "呼吸一滞", "心头一震", "眸色微沉"}:
            principle = "把抽象情绪标签改成具体动作或可见反应。"
            suggestions = [
                "改成手部、步伐、衣料褶皱、杯盖轻响等可见动作。",
                "用对白承接情绪，不直接贴‘震惊/紧张’标签。",
                "把一句拆短，减少‘情绪词+解释’叠加。"
            ]
        elif hit.rule_name in {"取而代之", "不是A而是B", "与其说A不如说B", "并非而是"}:
            principle = "避免模板反转句式，直接给出新的动作、判断或对白。"
            suggestions = [
                "删去对比框架，直接陈述人物新的选择。",
                "用具体场景信息取代抽象反转。",
                "优先改成一句更有压迫感的对白。"
            ]
        elif hit.hit_type == "narration_overuse":
            principle = "把解释型旁白改为人物通过看到、听到、触到的信息自行显现。"
            suggestions = [
                "把‘她知道/他明白’改成对方一句带压迫感的对白。",
                "把‘这意味着’改成脚步声、称呼变化、桌上信物等可见信息。",
                "减少替读者下结论，让读者自己推断。"
            ]
        elif hit.hit_type == "environment_overuse":
            principle = "保留环境时，要让环境承担信息或压迫，而不是纯垫场。"
            suggestions = [
                "将风、雨、灯、门、影子与人物动作绑定。",
                "把纯气氛句改成‘门外有人/窗纸被掀/脚步停住’等具体信息。",
                "避免连续两句纯环境。"
            ]
        else:
            principle = "减少过渡缓冲，直接切入动作、冲突或对白。"
            suggestions = [
                "删掉‘顿了顿/一时间/半晌’等垫片式过渡。",
                "让下一句直接发生动作或冲突。",
                "用短句拉节奏，压缩说明。"
            ]

        return {
            "rule_name": hit.rule_name,
            "paragraph": hit.paragraph_index + 1,
            "sentence": hit.sentence_index + 1,
            "original": sentence,
            "principle": principle,
            "suggestions": suggestions,
        }

    def score_chapter(
        self,
        word_count: int,
        rule_hits: List[Hit],
        paragraph_infos: Sequence[ParagraphInfo],
        empty_blocks: Sequence[Dict[str, object]],
    ) -> Tuple[float, Dict[str, int], Dict[str, int], List[Dict[str, object]]]:
        issue_counter = Counter(hit.hit_type for hit in rule_hits)
        rule_counter = Counter(hit.rule_name for hit in rule_hits)
        tag_counter = Counter()

        for p in paragraph_infos:
            tag_counter.update(p.tags)

        per_k = max(word_count / 1000.0, 1e-6)

        phrase_score = (
            issue_counter.get("phrase_repeat", 0) * 4.0
            + issue_counter.get("syntax_repeat", 0) * 4.5
            + issue_counter.get("style_repeat", 0) * 1.2
        ) / per_k

        narration_score = issue_counter.get("narration_overuse", 0) * 3.8 / per_k
        environment_score = issue_counter.get("environment_overuse", 0) * 2.8 / per_k
        transition_score = issue_counter.get("transition_overuse", 0) * 2.2 / per_k
        empty_block_score = len(empty_blocks) * 6.0

        explain_ratio = (
            sum(1 for p in paragraph_infos if "narration" in p.tags or "transition_buffer" in p.tags)
            / max(len(paragraph_infos), 1)
        )
        env_ratio = (
            sum(1 for p in paragraph_infos if "environment_buffer" in p.tags)
            / max(len(paragraph_infos), 1)
        )

        ratio_penalty = 0.0
        if explain_ratio > 0.30:
            ratio_penalty += (explain_ratio - 0.30) * 50
        if env_ratio > 0.12:
            ratio_penalty += (env_ratio - 0.12) * 40

        raw_score = (
            phrase_score
            + narration_score
            + environment_score
            + transition_score
            + empty_block_score
            + ratio_penalty
        )
        ai_score = max(0.0, min(100.0, round(raw_score, 2)))

        top_rules = [
            {"rule_name": rule_name, "count": count}
            for rule_name, count in rule_counter.most_common(12)
        ]
        return ai_score, dict(issue_counter), dict(tag_counter), top_rules

    def analyze_chapter(self, chapter_name: str, chapter_text: str) -> ChapterReport:
        paragraphs = self.split_paragraphs(chapter_text)
        paragraph_infos: List[ParagraphInfo] = []
        all_hits: List[Hit] = []
        suspicious_sentences: List[Dict[str, object]] = []

        for p_idx, paragraph in enumerate(paragraphs):
            info = self.classify_paragraph(chapter_name, p_idx, paragraph)
            paragraph_infos.append(info)

            sentences = self.split_sentences(paragraph)
            for s_idx, sentence in enumerate(sentences):
                hits = self.detect_rules_in_sentence(chapter_name, p_idx, s_idx, sentence)
                if hits:
                    all_hits.extend(hits)
                    suspicious_sentences.append(
                        {
                            "paragraph": p_idx + 1,
                            "sentence": s_idx + 1,
                            "text": sentence,
                            "rules": sorted({h.rule_name for h in hits}),
                            "issue_types": sorted({h.hit_type for h in hits}),
                        }
                    )

        empty_blocks = self.detect_empty_transition_blocks(paragraph_infos)
        word_count = self.count_cn_words(chapter_text)
        ai_score, issue_summary, tag_distribution, top_rules = self.score_chapter(
            word_count=word_count,
            rule_hits=all_hits,
            paragraph_infos=paragraph_infos,
            empty_blocks=empty_blocks,
        )

        unique_suggestions = []
        seen = set()
        for hit in all_hits:
            key = (hit.paragraph_index, hit.sentence_index, hit.rule_name)
            if key in seen:
                continue
            seen.add(key)
            unique_suggestions.append(self.generate_rewrite_suggestion(hit))

        return ChapterReport(
            chapter=chapter_name,
            word_count=word_count,
            paragraph_count=len(paragraphs),
            sentence_count=sum(p.sentence_count for p in paragraph_infos),
            ai_score=ai_score,
            issue_summary=issue_summary,
            paragraph_tag_distribution=tag_distribution,
            top_rules=top_rules,
            suspicious_sentences=suspicious_sentences[:30],
            empty_transition_blocks=empty_blocks,
            rewrite_suggestions=unique_suggestions[:20],
        )

    def analyze_book(self, chapters: Sequence[Tuple[str, str]]) -> BookReport:
        chapter_reports = [self.analyze_chapter(name, text) for name, text in chapters]
        total_words = sum(ch.word_count for ch in chapter_reports)
        chapter_count = len(chapter_reports)
        ai_score = round(sum(ch.ai_score for ch in chapter_reports) / max(chapter_count, 1), 2)

        global_rule_counts = Counter()
        global_issue_counts = Counter()

        for ch in chapter_reports:
            for item in ch.top_rules:
                global_rule_counts[item["rule_name"]] += int(item["count"])
            for issue, count in ch.issue_summary.items():
                global_issue_counts[issue] += count

        return BookReport(
            total_words=total_words,
            chapter_count=chapter_count,
            ai_score=ai_score,
            global_rule_counts=dict(global_rule_counts.most_common()),
            global_issue_counts=dict(global_issue_counts.most_common()),
            chapters=chapter_reports,
        )

    @staticmethod
    def write_json_report(report: BookReport, output_path: Path) -> None:
        output_path.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def write_markdown_report(report: BookReport, output_path: Path) -> None:
        lines: List[str] = []
        lines.append("# 小说降 AI 感分析报告")
        lines.append("")
        lines.append(f"- 总字数：{report.total_words}")
        lines.append(f"- 章节数：{report.chapter_count}")
        lines.append(f"- 全书 AI 味综合分：{report.ai_score}")
        lines.append("")

        lines.append("## 全书高频问题")
        for name, count in list(report.global_rule_counts.items())[:20]:
            lines.append(f"- {name}: {count}")
        lines.append("")

        lines.append("## 全书问题类型统计")
        for name, count in report.global_issue_counts.items():
            lines.append(f"- {name}: {count}")
        lines.append("")

        for ch in report.chapters:
            lines.append(f"## {ch.chapter}")
            lines.append(f"- 字数：{ch.word_count}")
            lines.append(f"- 段落数：{ch.paragraph_count}")
            lines.append(f"- 句子数：{ch.sentence_count}")
            lines.append(f"- AI 味分：{ch.ai_score}")
            lines.append("")

            lines.append("### 问题分布")
            for k, v in ch.issue_summary.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

            lines.append("### 段落标签分布")
            for k, v in ch.paragraph_tag_distribution.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

            lines.append("### 高频规则")
            for item in ch.top_rules[:10]:
                lines.append(f"- {item['rule_name']}: {item['count']}")
            lines.append("")

            if ch.empty_transition_blocks:
                lines.append("### 连续空转段")
                for block in ch.empty_transition_blocks[:8]:
                    lines.append(
                        f"- 第 {block['start_paragraph']} 段 ~ 第 {block['end_paragraph']} 段，"
                        f"共 {block['paragraph_count']} 段；样本：{block['sample']}"
                    )
                lines.append("")

            if ch.suspicious_sentences:
                lines.append("### 可优先修改句子")
                for item in ch.suspicious_sentences[:12]:
                    rules = "、".join(item["rules"])
                    lines.append(
                        f"- 第{item['paragraph']}段 第{item['sentence']}句 [{rules}]：{item['text']}"
                    )
                lines.append("")

            if ch.rewrite_suggestions:
                lines.append("### 改写建议")
                for item in ch.rewrite_suggestions[:8]:
                    lines.append(
                        f"- 位置：第{item['paragraph']}段 第{item['sentence']}句 | 规则：{item['rule_name']}"
                    )
                    lines.append(f"  - 原句：{item['original']}")
                    lines.append(f"  - 原则：{item['principle']}")
                    for s in item["suggestions"]:
                        lines.append(f"  - 建议：{s}")
                lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量检测小说中的 AI 模板化表达、旁白过多、环境空转和过渡废话。"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入路径：可以是单个 txt/md 文件，也可以是包含多章文件的目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="output_reports",
        help="输出目录，默认 output_reports",
    )
    parser.add_argument(
        "--chapter-pattern",
        default=DEFAULT_CHAPTER_PATTERN,
        help="当输入为单个文件时，用于切章的正则。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = NovelAIDetector()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if input_path.is_dir():
            chapters = detector.load_chapters_from_dir(input_path)
            book_name = input_path.name
        elif input_path.is_file():
            text = input_path.read_text(encoding="utf-8", errors="ignore")
            chapters = detector.split_book_by_chapters(text, chapter_pattern=args.chapter_pattern)
            book_name = input_path.stem
        else:
            raise FileNotFoundError(f"输入路径不存在: {input_path}")

        report = detector.analyze_book(chapters)

        json_path = output_dir / f"{book_name}_ai_report.json"
        md_path = output_dir / f"{book_name}_ai_report.md"

        detector.write_json_report(report, json_path)
        detector.write_markdown_report(report, md_path)

        print(f"分析完成：{book_name}")
        print(f"JSON 报告：{json_path}")
        print(f"Markdown 报告：{md_path}")
        print(f"总字数：{report.total_words}")
        print(f"章节数：{report.chapter_count}")
        print(f"全书 AI 味综合分：{report.ai_score}")

    except Exception as exc:
        print(f"运行失败: {exc}")
        raise


if __name__ == "__main__":
    main()