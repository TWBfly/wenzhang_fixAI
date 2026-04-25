import re
import random
import statistics
from typing import List, Dict, Tuple, Set, Optional, Callable
import json
import os


class RhythmShatter:
    """分析并打散过于均匀的句长分布"""

    def analyze(self, sentences: List[str]) -> Dict:
        if not sentences:
            return {"cv": 0.0, "is_uniform": False, "lengths": []}

        lengths = [len(s.strip()) for s in sentences if s.strip()]
        if not lengths:
            return {"cv": 0.0, "is_uniform": False, "lengths": []}

        mean = sum(lengths) / len(lengths)
        stdev = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
        cv = stdev / max(mean, 1)

        return {
            "cv": round(cv, 3),
            "is_uniform": cv < 0.35 and mean > 10,  # Mean > 10 to avoid flagging very short dialogues
            "lengths": lengths
        }


class SensoryWeaver:
    """按场景类型注入恰当的非视觉感官"""

    SENSORY_MAP = {
        "smell": {
            "combat": ["血腥味", "铁锈气", "焦糊味", "汗臭", "硝烟", "腥风", "令人作呕", "刺鼻"],
            "indoor": ["熏香", "炭火味", "霉味", "药香", "旧纸味", "檀香", "脂粉气", "潮湿的木头味", "灰尘味", "陈腐"],
            "outdoor": ["泥土的腥气", "草木灰", "雨后的潮气", "粪便的臭", "花香", "土腥味", "清冽的风", "松脂味"],
            "food": ["热油的香", "甜腻", "馊味", "焦糖色", "酒气", "肉香", "辛辣", "苦涩"],
        },
        "touch": {
            "combat": ["拳头砸在骨头上的钝痛", "刀柄的缠皮摩擦", "铁甲冰冷", "割裂感", "温热的血", "粘稠", "发麻的虎口", "粗糙"],
            "indoor": ["粗糙的木纹", "丝绸的凉滑", "被褥潮湿", "冰凉的瓷壁", "柔软", "僵硬的椅背"],
            "emotion": ["太阳穴突突跳", "胃部抽搐", "肺里烧灼", "手指发麻", "鸡皮疙瘩", "喉结滑动", "掌心出汗", "头皮发麻", "背脊发凉"],
            "weather": ["刺骨的冷", "闷热得喘不过气", "针扎般的寒意", "湿黏的衣服", "火烤般的日头"],
        },
        "internal": {
            "tension": ["嗓子发紧", "牙根发酸", "舌头干得粘住", "后脑勺发冷", "心跳漏了一拍", "胸腔鼓噪", "血往上涌", "呼吸急促"],
            "fatigue": ["眼皮沉重", "腰酸得像被剔骨", "脚底火辣辣", "筋骨欲裂", "虚脱感", "肌肉抽搐", "双腿灌铅"],
        },
        "sound": {
            "combat": ["骨节的脆响", "衣料撕裂声", "闷哼", "刀身嗡鸣", "金铁交击", "倒吸冷气", "骨骼断裂声", "破风声"],
            "indoor": ["门轴的吱呀", "茶杯闷响", "窗纸哗啦啦", "脚步声悉索", "蜡烛燃烧的剥啪声", "更具打在青石板上", "低语", "粗重的喘息"],
        },
    }

    def __init__(self):
        # 预先展平词库以便快速匹配
        self.flat_words = {}
        for sense, categories in self.SENSORY_MAP.items():
            words = []
            for cat, w_list in categories.items():
                words.extend(w_list)
            self.flat_words[sense] = set(words)

    def diagnose(self, text: str) -> Dict:
        distribution = {sense: sum(text.count(w) for w in words) for sense, words in self.flat_words.items()}

        total_non_visual = sum(distribution.values())
        visual_ratio = text.count("看") + text.count("望") + text.count("瞥") + text.count("盯") + text.count("视线") + text.count("目光")
        total_sensory = total_non_visual + visual_ratio

        return {
            "distribution": distribution,
            "visual_dominant": (visual_ratio / max(total_sensory, 1)) > 0.6,
            "total_non_visual": total_non_visual,
            "needs_enrichment": total_non_visual < 1 and visual_ratio > 2
        }


class RedundancyDetector:
    """检测段落级别的信息冗余"""

    def _extract_key_nouns(self, text: str) -> Set[str]:
        # Simple extraction for demo: characters > 1 and common noun indicators
        # In a real NLP setting this would use jieba pos tagging.
        # Here we do a crude approximation for robust IDE usage.
        tokens = re.findall(r"[\u4e00-\u9fff]{2,4}", text)
        return set(tokens)

    def detect_paraphrase_loops(self, paragraphs: List[str]) -> List[Dict]:
        issues = []
        for i in range(len(paragraphs) - 1):
            if not paragraphs[i].strip() or not paragraphs[i+1].strip():
                continue

            nouns_a = self._extract_key_nouns(paragraphs[i])
            nouns_b = self._extract_key_nouns(paragraphs[i + 1])
            if not nouns_a or not nouns_b:
                continue

            intersection = len(nouns_a & nouns_b)
            union = len(nouns_a | nouns_b)
            overlap = intersection / union if union > 0 else 0

            if overlap > 0.45 and len(nouns_a) > 5 and len(nouns_b) > 5:
                issues.append({
                    "para_idx": i,
                    "overlap_ratio": round(overlap, 2),
                    "reason": "上下文大量使用重复词汇，疑似废话循环"
                })
        return issues


class POVChecker:
    """检测视角泄露"""

    OMNISCIENT_LEAKS = [
        r"[她他](?:并)?不知道(?:的是)?，",
        r"如果[她他]回头(?:看|望|瞥)(?:的话)?",
        r"在[她他](?:看不到|不知道|留意不到)的(?:地方|角落)",
        r"此时此刻[，,]远在.{1,15}的",
        r"[她他](?:还)?没(?:有)?意识到",
        r"命运(?:的齿轮|之神)",
        r"冥冥之中",
        r"此时的[她他]并不知道",
        r"遗憾的是",
    ]

    def check(self, text: str) -> List[Dict]:
        issues = []
        for pattern in self.OMNISCIENT_LEAKS:
            for m in re.finditer(pattern, text):
                issues.append({
                    "hit": m.group(),
                    "reason": "全知视角泄露（上帝旁白）"
                })
        return issues


class DialoguePowerAnalyzer:
    """检测对白的长度、密度和非自然说教"""

    def analyze_dialogue_lines(self, lines: List[str]) -> List[str]:
        flags = []

        for line in lines:
            line = line.strip("“”\"' \n")
            if not line:
                continue

            if len(line) > 60:
                flags.append("对白冗长：单条超60字，信息像倒豆子一样没有交互。建议中途打断或加入动作。")

            # Interrogation dump
            if len(re.findall(r"(?:到底|究竟|是不是|为什么)[^。！？]{2,20}[？?]", line)) >= 2:
                flags.append("连续质问：带有较强AI模板感，真人往往回避或沉默。")

            # Exposition dummy
            explain_markers = ["事情是这样的", "你听我解释", "让我来告诉你", "这得从头说起"]
            if any(m in line for m in explain_markers):
                flags.append("剧情解说：角色像旁白一样替作者解释背景，显得生硬。")

        return list(set(flags))


class ShowDontTellTransformer:
    """捕捉主观情感词并标记 SDT，同时深度拦截 AI 悬浮做作比喻和哲学说教"""

    EMOTIONAL_TELLS = [
        r"([她他])(?:感到|觉得|显得|看起来)?(?:极其|十分|格外|有些|非常)?(紧张|害怕|愤怒|不安|慌乱|难堪|压抑|委屈|犹豫|恐惧|悲伤|绝望|痛苦|欣慰|感动)",
        r"一(?:阵|股|种)(紧张|害怕|愤怒|不安|恐惧|绝望|压迫|痛苦|寒意|热流)(?:感|意)?(?:涌上|袭来|充斥|浮现|席卷)"
    ]

    # 生理反馈库：引导 AI 从“写情绪”转为“写反应”
    PHYSIOLOGICAL_FEEDBACK = {
        "紧张/恐惧": ["瞳孔骤缩", "背脊发凉", "指尖微颤", "冷汗沁出", "呼吸滞涩", "喉结滑动", "头皮发麻", "肌肉僵硬"],
        "愤怒": ["额角青筋跳动", "指节攥得发白", "呼吸粗重", "牙根发酸", "血往头上涌", "眼底爬满血丝"],
        "不安": ["心跳漏了一拍", "太阳穴突突跳", "手心见汗", "胃部抽搐"],
    }

    # 针对性拦截系统：哲学病句与做作感官夸张
    SUSPENDED_TELLS = [
        r"(超越了.*?(?:极限|界限|常理)|纯粹的.*?|原始的.*?)(痛苦|恐惧|绝望|本能|压迫|冲动)",
        r"像(?:一个|一把|一只).*?的(捕兽夹|齿轮|利刃|机械|生锈的)",
        r"(胃袋|肚子)?腔里.*?(空响|声声|阵阵)?的回音",
        r"如同实质般的",
        r"仿佛抽干了周围所有的",
        r"一种难言的(?:沉默|寂静|压抑)",
        r"(?:那是一抹|那是一种|这一刻).*?的(?:色彩|音符|光影)",
        r"由于.{2,10}的缘故",
        r"(?:陷入了|沉浸在).{2,10}的(?:海洋|深渊|旋涡)"
    ]

    def check(self, text: str) -> List[str]:
        hits = []
        for p in self.EMOTIONAL_TELLS:
            for m in re.finditer(p, text):
                emotion = m.group(2) if len(m.groups()) >= 2 else "强烈情绪"
                hits.append(f"直给情绪：【{emotion}】。改用生理反应描写。")

        for p in self.SUSPENDED_TELLS:
            for m in re.finditer(p, text):
                hits.append(f"[悬浮警报]：使用抽象大词或做作隐喻('{m.group(0)}')。严格禁止伪高级的哲学痛觉解释。")

        return list(set(hits))



class ClicheExpressDetector:
    """检测并拦截AI高频套路词、油腻微表情及刻板的感官堆砌"""

    CLICHE_PATTERNS = [
        (r"(?:扯|勾|拉)[出起一个]+.*?的(?:不太明显|若有若无|冰冷)?(?:弧度|冷笑|笑意)", "面具化微表情：‘扯出一个不太明显的弧度’是极经典的AI描写，建议用具体的眼部或面部动作代替。"),
        (r"(?:铁锈般|劣质的).*?(?:腥甜|焚香|气味)", "刻板感官堆砌：‘铁锈般的腥甜’、‘劣质的焚香’是AI滥用的味觉比喻，显得廉价油腻，应改换角度。"),
        (r"死死(?:堵在|扣进|抠进|盯住|掐住)", "做作力度词：‘死死XX’在AI文中泛滥，显得浮夸。"),
        (r"(?:像是在|如同)?砂纸上打磨过(?:的钝器)?", "刻板声音比喻：‘砂纸打磨的钝器’常用于AI描写沙哑声音，极其烂俗。"),
        (r"濒死的鱼", "俗套比喻：‘像濒死的鱼一样喘气’是AI高度重复修辞。"),
        (r"(?:那是在看|也是看|这分明是看)(?:一个|一只|一块).*?[，,。](?:一个|一只|一块|一条).*?", "解释性连续暗喻：强行并列的物品暗喻，是AI试图模仿文学感时的翻车现象，啰嗦且做作。"),
        (r"(?:犹如|宛如|就像|仿佛是一只).*?的(?:丹鼎|灵石|羔羊|布偶|宠物)", "修辞狂热：过度强调被看者的‘物品’属性，是AI滥用的上位压迫感写法。")
    ]

    def __init__(self):
        self.compiled_patterns = [(re.compile(p), reason) for p, reason in self.CLICHE_PATTERNS]

    def check(self, text: str) -> List[Dict]:
        issues = []
        for p, reason in self.compiled_patterns:
            m = p.search(text)
            if m:
                issues.append({
                    "hit": m.group(0),
                    "reason": reason
                })
        return issues


class IdentitySummaryDetector:
    """检测在具体描写段落结尾，像旁白一样跳出来给角色下定义的AI通病

    病灶结构:
      (主语) + [是一个/是个/就像个/宛如/仿佛] + [...]的+ [身份词/比喻词] + [，/也是/也是个/还是] + [...]的+ [身份词]

    示例:
      ❌ 他是一个极其出色的猎手，也是个准备割开自己颈动脉的疯子。
      ❌ 她就像一头护崽的母狼，又像一个坠入冰窖的迷路者。
    """

    IDENTITY_PATTERNS = [
        r"(?:他|她)(?:是一个|是个)(?:极其|十分|非常|无比)?.{2,15}的(猎手|野兽|疯子|魔鬼|怪物|神明|机器|木偶|赌徒|主宰|凡人|棋子)，(?:也是|又是)(?:一个|个)?.{2,15}的(疯子|野兽|猎手|魔鬼|怪物|木偶|赌徒|复仇者|棋子|灰尘)",
        r"(?:他|她)(?:就像|仿佛是?|宛如)(?:一头|一个|一只).{2,10}的(野兽|孤狼|猎豹|怪物|雕塑|行尸走肉|幽灵)",
        r"(?:这是|那是)(?:他|她)(?:第一次|最后一次|唯一一次)(?:意识到|感觉到|明白)",
        r"(?:他|她)(?:其实|终究|本质上)(?:不过是|只是)(?:一个|个).{2,10}的"
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p) for p in self.IDENTITY_PATTERNS]

    def check(self, text: str) -> List[Dict]:
        issues = []
        sentences = re.split(r'(?<=[。！？])', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue

            for p in self.compiled_patterns:
                m = p.search(sentence)
                if m:
                    issues.append({
                        "hit": sentence,
                        "reason": f"身份论断式总结：使用高度抽象的比喻给角色下定义（'{m.group(0)}'）。应砍掉这种旁白式的总结，用具体动作和生理细节来展现角色性格和状态。"
                    })
        return issues


        return issues


class ModernVocabularyDetector:
    """检测并修复古代题材中的现代/工程/商业词汇（时空错位词）。

    示例：
    ❌ 把这种幻觉变成杀回商会的路基
    ✅ 把这种幻觉变成杀回商会的基石
    """

    MODERN_TO_ANCIENT = {
        "路基": ["基石", "铺路石", "进身之阶", "根本", "根基"],
        "底层逻辑": ["根本", "命脉", "源头", "天理", "公道"],
        "维度": ["层面", "门道", "角度", "身分", "地位"],
        "战略": ["长远之计", "大局", "布局", "韬略", "谋划"],
        "KPI": ["考核", "业绩", "成效", "功劳"],
        "颗粒度": ["细节", "精细度", "深浅"],
        "抓手": ["契机", "切入点", "凭据", "把柄"],
        "赋能": ["助力", "提拔", "帮扶", "点化"],
        "闭环": ["圆满", "周全", "自洽", "首尾呼应"],
        "落地": ["落实", "成真", "践行"],
        "架构": ["布局", "章法", "格局", "框架"],
        "模块": ["板块", "环节", "部分"],
        "系统性": ["全局", "全面", "周密"],
        "场景": ["场面", "情景", "境地"],
        "痛点": ["软肋", "死穴", "难处", "心头大患"],
        "核心竞争力": ["看家本领", "杀手锏", "独门秘籍", "立身之本"]
    }

    def __init__(self, setting: str = "古代"):
        self.setting = setting
        self.patterns = {word: re.compile(rf"{word}") for word in self.MODERN_TO_ANCIENT.keys()}

    def check(self, text: str) -> List[Dict]:
        issues = []
        if "古代" not in self.setting and "武侠" not in self.setting:
            return issues

        lines = text.split('\n')
        for line_num, line in enumerate(lines, 1):
            for word, pattern in self.patterns.items():
                for m in pattern.finditer(line):
                    suggestions = self.MODERN_TO_ANCIENT[word]
                    issues.append({
                        "hit": word,
                        "line": line_num,
                        "context": line.strip(),
                        "reason": f"时空错位词：在古代题材中使用了现代词汇‘{word}’。应改为更具古风的词汇，如：{', '.join(suggestions)}。",
                        "suggestions": suggestions
                    })
        return issues

    def fix(self, text: str) -> str:
        """自动化初级修复：替换为第一个建议词"""
        if "古代" not in self.setting and "武侠" not in self.setting:
            return text

        new_text = text
        for word, suggestions in self.MODERN_TO_ANCIENT.items():
            new_text = new_text.replace(word, suggestions[0])
        return new_text


class ConjunctionNeutering:
    """连词去脓：拦截 AI 过度使用的逻辑连词，强制改用动作衔接。"""

    AI_CONJUNCTIONS = {
        r"随着": "AI 指纹：‘随着’引导的背景解说往往平淡无奇，建议直接描写动作或环境变化。",
        r"不仅.*?而且": "AI 递进：‘不仅而且’是典型的学生作文结构，建议拆分为两个短句，增加节奏感。",
        r"然而|但是|却": "逻辑转折过多：AI 喜欢用显性连词表达转折。真人作家倾向于用情节本身的反差来体现转折。",
        r"从而|因此|所以": "推论解说：严禁在小说正文中进行因果推论，这是旁白的行为。应让动作自然产生后果。",
        r"与此同时": "AI 并列：典型的时间线切分词，建议改用具体的空间或角色动作切换。",
    }

    def check(self, text: str) -> List[Dict]:
        issues = []
        for pattern, reason in self.AI_CONJUNCTIONS.items():
            for m in re.finditer(pattern, text):
                issues.append({
                    "hit": m.group(0),
                    "reason": reason
                })
        return issues


class StyleBenchmarkEngine:
    """风格基准引擎：通过分析目标作品，提取‘大师基因’作为生成指标。"""

    def __init__(self):
        self.metrics = {
            "sentence_var": 0.0,      # 句长方差（越高越有呼吸感）
            "verb_density": 0.0,      # 动词密度（越高越硬朗）
            "sensory_density": 0.0,   # 感官密度（越高越真人）
            "abstract_ratio": 0.0     # 抽象词比例（越低越好）
        }

    def analyze_masterpiece(self, text: str) -> Dict:
        # 1. 句长方差
        sentences = re.split(r'[。！？\n]', text)
        lengths = [len(s.strip()) for s in sentences if len(s.strip()) > 1]
        if len(lengths) > 5:
            mean = sum(lengths) / len(lengths)
            stdev = statistics.stdev(lengths)
            self.metrics["sentence_var"] = round(stdev / max(mean, 1), 3)

        # 2. 动词/感官词密度 (粗略计算)
        verbs = re.findall(r"按|提|跨|劈|斩|冲|碎|裂|咬|攥|拍|砸|踢", text) # 动作词锚点
        self.metrics["verb_density"] = round(len(verbs) / (len(text) / 100 + 1), 2)

        sensory = re.findall(r"味|气|响|冷|热|痛|麻|涩|腥|颤|抖", text)
        self.metrics["sensory_density"] = round(len(sensory) / (len(text) / 100 + 1), 2)

        return self.metrics


class HookImpactAuditor:
    """审计开篇冲击力。

    进化基因：第一句话严禁‘环境起手’，必须‘冲突/情感/悬念’起手。
    """
    def check(self, text: str) -> List[Dict]:
        issues = []
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return issues

        first_line = lines[0]
        # 如果第一句全是环境描写（雨、风、天色）而无情感/冲突词
        env_words = ["雨", "天", "风", "云", "日", "月", "雾"]
        conflict_words = ["命", "死", "恨", "血", "背叛", "债", "印", "局", "杀"]

        is_env = any(w in first_line for w in env_words)
        has_conflict = any(w in first_line for w in conflict_words)

        if is_env and not has_conflict:
            issues.append({
                "hit": first_line,
                "reason": "开篇冲击力不足：检测到环境起手。根据进化经验，第一句应直接抛出核心矛盾或情感极值（如：七年泣血辅助，等来的却是背叛）。",
                "suggestions": ["将核心冲突提前", "用强烈的生理/心理感受起手"]
            })
        return issues


class SensoryFeedbackAnalyzer:
    """分析生理反馈。

    进化基因：身体特征（如硬茧、伤疤）必须配合‘痛觉/触觉’描写。
    """
    def check(self, text: str) -> List[Dict]:
        issues = []
        # 检测静态描写关键词
        static_traits = ["硬茧", "伤疤", "老茧", "伤痕", "冰冷"]
        # 检测缺失的生理反馈词
        feedback_words = ["痛", "麻", "痒", "蹭", "紧", "颤"]

        lines = text.split('\n')
        for line in lines:
            has_trait = any(t in line for t in static_traits)
            has_feedback = any(f in line for f in feedback_words)

            if has_trait and not has_feedback:
                issues.append({
                    "hit": line,
                    "reason": "缺乏感官锚定：检测到身体特征描写，但缺乏即时的生理反馈。人工进化经验显示，加入‘蹭一下的钝痛’或‘手心的潮汗’能大幅提升真实感。",
                    "suggestions": ["增加痛觉/触觉描写", "增加身体的下意识反应"]
                })
        return issues


class NumericalConsistencyDetector:
    """检测并修复数值漂移错误（如前面五万两，后面四万两）。

    逻辑：在滑动窗口内追踪 [数字]+[量词] 组合，如果量词相同但数字不同且距离过近，则触发预警。
    """

    UNITS = ["两", "万两", "人", "军", "里", "步", "岁", "两银子"]

    def __init__(self):
        # 匹配中文或阿拉伯数字 + 量词
        self.num_pat = r"([0-9一二三四五六七八九十百千万]{1,10})\s*(%s)" % "|".join(self.UNITS)

    def check(self, text: str) -> List[Dict]:
        issues = []
        # 记录最近出现的数值：{量词: (数值, 上下文)}
        history = {}

        lines = text.split('\n')
        for line in lines:
            matches = re.findall(self.num_pat, line)
            for num, unit in matches:
                if unit in history:
                    prev_num, prev_context = history[unit]
                    if num != prev_num:
                        issues.append({
                            "hit": f"数值矛盾：前文是‘{prev_num}{unit}’，此处变为‘{num}{unit}’",
                            "context": line.strip(),
                            "prev_context": prev_context,
                            "reason": f"数值漂移Bug：在短距离内对同一单位‘{unit}’使用了不同的数值。除非中间有明确的变动描写，否则疑似AI幻觉导致的数值失准。"
                        })
                # 更新历史（仅保留最近的一个，避免过久以前的正常变动触发）
                history[unit] = (num, line.strip())
        return issues

    def fix(self, text: str) -> str:
        # 数值修复风险极高，仅报错
        return text


class FamilyNameConsistencyDetector:
    """检测并修复亲属姓氏不一致的逻辑Bug。

    逻辑：识别“A...称谓...B”结构，校验 A 和 B 的姓氏是否一致。
    """

    KINSHIP_TERMS = ["爹", "父亲", "爹爹", "阿爹", "儿子", "犬子", "逆子", "女儿", "千金", "兄", "弟", "姐", "妹"]

    def __init__(self):
        # 匹配中文姓名（2-4字）
        self.name_pat = r"[\u4e00-\u9fff]{2,4}"

    def check(self, text: str) -> List[Dict]:
        issues = []
        lines = text.split('\n')

        # 维护一个简单的上下文姓名池
        names_found = re.findall(rf"({self.name_pat})", text)
        if not names_found:
            return issues

        for line in lines:
            for term in self.KINSHIP_TERMS:
                if term in line:
                    # 尝试在该行及前后寻找姓名
                    potential_names = re.findall(rf"({self.name_pat})", line)
                    if len(potential_names) >= 2:
                        # 简单的姓氏提取（取第一个字，复姓暂不处理，AI出错多在单姓）
                        surnames = [n[0] for n in potential_names]
                        if len(set(surnames)) > 1:
                            issues.append({
                                "hit": f"称谓‘{term}’关联了不同姓氏：{', '.join(potential_names)}",
                                "context": line.strip(),
                                "reason": f"血缘逻辑Bug：在称谓‘{term}’附近发现了不同姓氏的人物。在古代背景下，除非有特殊设定，父子/兄妹应同姓。当前发现‘{potential_names[0]}’与‘{potential_names[1]}’异姓，疑似AI幻觉。"
                            })
        return issues

    def fix(self, text: str) -> str:
        # 姓氏逻辑修复通常需要人工或LLM决策，此处仅报错
        return text


class NamedLabelDetector:
    """检测并修复“名为‘XX’的[力量/深渊]”这类AI解说标签。

    范式：名为[“‘].*?[”’]的(力量|深渊|气息|意志|信念|光芒|阴影)
    """

    PATTERN = r"名为[“‘]([^”’]{2,6})[”’]的(力量|深渊|气息|意志|信念|光芒|阴影|端倪|逻辑|真相)"

    def __init__(self):
        self.prog = re.compile(self.PATTERN)

    def check(self, text: str) -> List[Dict]:
        issues = []
        lines = text.split('\n')
        for line in lines:
            for m in self.prog.finditer(line):
                label = m.group(1)
                container = m.group(2)
                issues.append({
                    "hit": m.group(0),
                    "context": line.strip(),
                    "reason": f"AI定义式标签：使用了‘{m.group(0)}’范式。这种‘名为XX’的写法带有强烈的AI上帝视角和解说感，破坏了文学张力。建议直接描写其表现，或直接简写为‘{label}的{container}’（去掉‘名为’和引号）。"
                })
        return issues

    def fix(self, text: str) -> str:
        """自动化初级修复：去掉‘名为’和引号"""
        return self.prog.sub(r"\1的\2", text)


class ColonSummaryDetector:
    """检测并修复“AI冒号病”（总结式描写）

    范式：[主体] + [判断/感知动词] + [抽象概括名词] + ： + [具体解释]

    示例：
    ❌ 她判断出了这背后的威胁：如果此时她直接撕票离场...
    ✅ 她看穿了这背后的威胁。如果此时她直接撕票离场... (或者改用逗号)
    """

    PATTERNS = [
        # 主体 + 动词 + 抽象词 + ：
        r"([他她它]|[^\s。！？，、]{2,6})(?:判断出|意识到|察觉到|发现|明白|想到|看出了|听出了|分析出|断定|断言|确认|感知到)(?:了|到)?(?:这背后的|其中的|潜在的|致命的|眼前的)?(威胁|真相|后果|端倪|不对劲|深意|局势|逻辑|圈套|陷阱|杀机|意图|目的|计划|安排|算计|恶意|端倪|变化)："
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p) for p in self.PATTERNS]

    def check(self, text: str) -> List[Dict]:
        issues = []
        # 按行或段落检测
        lines = text.split('\n')
        for line in lines:
            if not line.strip():
                continue
            for p in self.compiled_patterns:
                for m in p.finditer(line):
                    issues.append({
                        "hit": m.group(0),
                        "full_context": line,
                        "reason": f"AI冒号总结病：使用了‘{m.group(0)}’结构。这种‘概括名词+冒号’的写法带有强烈的AI解说感，破坏了叙事张力。建议将冒号改为逗号，或直接删掉抽象名词，让后续内容直接展现。"
                    })
        return issues

    def fix(self, text: str) -> str:
        """自动化初级修复：将冒号改为逗号"""
        new_text = text
        for p in self.compiled_patterns:
            new_text = p.sub(lambda m: m.group(0).replace('：', '，'), new_text)
        return new_text


class ExplanatoryTailTagDetector:
    """解释性尾注检测器 (Explanatory Tail Tag Detector)

    检测AI最隐蔽的指纹之一：前半句是具体的生理/动作描写，
    后半句通过破折号或逗号 + 「那是/这是」贴上抽象概念标签。

    病灶结构:
      [具体描写] + [——/，] + [那是/这是] + [抽象概念]

    示例:
      ❌ 大腿根部的肌肉在打颤——那是原主身体残余的恐惧
      ❌ 全身的力气都往腰腹上使，那是杀红了眼的蛮劲
      ❌ 后背浮起一层冷汗——这是本能的警觉
      ✅ 大腿根部的肌肉在打颤。（删掉尾注，让颤抖自己说话）
    """

    # 前半句的「具体描写锚点词」——这些词出现时说明作者已经做了感官描写
    CONCRETE_ANCHORS = [
        # 生理反应
        "打颤", "发抖", "颤抖", "颤栗", "哆嗦", "抽搐", "痉挛",
        "冷汗", "虚汗", "汗珠", "鸡皮疙瘩", "寒毛", "汗毛",
        "瞳孔", "心跳", "呼吸", "脉搏", "血管", "太阳穴",
        "喉结", "喉咙", "嗓子", "胃部", "胸腔", "胸口",
        "肌肉", "筋脉", "骨头", "关节", "指节", "虎口",
        # 动作/姿态
        "咬着牙", "咬牙", "攥紧", "握紧", "绷紧", "僵住",
        "愣住", "定住", "蜷缩", "后退", "弓起", "弯腰",
        "咽了口", "深吸一口气", "屏住呼吸", "闭上眼",
        "抿唇", "抿嘴", "紧闭", "咬住", "攥住",
        # 力量/动能描写
        "使劲", "发力", "用力", "力气", "蛮力", "一股劲",
    ]

    # 后半句的「连接词」——这些词将具体描写与抽象标签粘合
    CONNECTORS = [
        r"——那是", r"——这是", r"——那不是", r"——这不是",
        r"——那便是", r"——这便是", r"——这都是", r"——那都是",
        r"[，,]那是", r"[，,]这是", r"[，,]那便是", r"[，,]这便是",
        r"[，,]那都是", r"[，,]这都是",
        r"[，,]那种", r"[，,]这种",
    ]

    # 后半句的「抽象概念词」——这些词作为尾注的核心解释
    ABSTRACT_TAGS = [
        # 情绪
        "恐惧", "恐慌", "恐怖", "愤怒", "不甘", "绝望", "焦虑", "不安",
        "痛苦", "悲伤", "孤独", "寂寞", "迷茫", "空虚",
        # 本能/心理
        "本能", "警觉", "直觉", "习惯", "条件反射", "应激反应",
        "求生欲", "占有欲", "控制欲", "保护欲", "嫉妒",
        # 性格/态度
        "倔强", "骄傲", "尊严", "底气", "底线", "蛮劲",
        "温柔", "善意", "信任", "默契",
        # 状态/信号
        "信号", "暗示", "宣言", "挑衅", "威胁", "承诺",
        "决心", "觉悟", "勇气", "试探", "妥协", "退让",
        "挣扎", "反击", "示弱", "示好", "求救",
        # 抽象归因
        "力量", "力道", "冲动", "渴望", "执念", "贪婪",
        "惯性", "惰性", "残余", "余韵", "后遗症",
    ]

    def __init__(self):
        # 预编译连接词正则
        self._connector_pattern = re.compile(
            "|".join(self.CONNECTORS)
        )
        # 构建锚点词集合用于快速查询
        self._anchor_set = set(self.CONCRETE_ANCHORS)
        self._abstract_set = set(self.ABSTRACT_TAGS)

    def check(self, text: str) -> List[Dict]:
        """检测文本中的解释性尾注。

        Returns:
            List of dicts with keys: hit, reason, front, tail, severity
        """
        issues = []

        # 按句号/感叹号/问号切分
        sentences = re.split(r'(?<=[。！？])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for i, sentence in enumerate(sentences):
            if not sentence or len(sentence) < 8:
                continue

            # 单句内检测
            result = self._analyze_sentence(sentence)
            if result:
                issues.append(result)

            # P1 Fix #5: 跨句检测 — 上一句有具体描写，当前句贴标签
            if i > 0 and len(sentences[i-1]) >= 8:
                cross_result = self._analyze_cross_sentence(sentences[i-1], sentence)
                if cross_result:
                    issues.append(cross_result)

        return issues

    def _analyze_sentence(self, sentence: str) -> Optional[Dict]:
        """分析单句是否包含解释性尾注结构。"""
        # Step 1: 检查是否包含连接词
        connector_match = self._connector_pattern.search(sentence)
        if not connector_match:
            return None

        split_pos = connector_match.start()
        front = sentence[:split_pos]
        tail = sentence[split_pos:]

        # Step 2: 前半句需有具体描写锚点
        has_anchor = any(anchor in front for anchor in self._anchor_set)
        if not has_anchor:
            return None

        # Step 3: 后半句需有抽象概念词
        has_abstract = any(tag in tail for tag in self._abstract_set)
        if not has_abstract:
            return None

        # 找到匹配的锚点和标签用于报告
        matched_anchors = [a for a in self._anchor_set if a in front]
        matched_tags = [t for t in self._abstract_set if t in tail]

        return {
            "hit": sentence,
            "reason": f"解释性尾注：前半句已有具体描写（{'/'.join(matched_anchors[:3])}），"
                      f"尾巴贴了抽象标签（{'/'.join(matched_tags[:3])}）。"
                      f"应砍掉'{''.join(connector_match.group())}'后的全部解释。",
            "front": front,
            "tail": tail,
            "severity": "P1",
            "matched_anchors": matched_anchors,
            "matched_tags": matched_tags,
        }

    def suggest_fix(self, issue: Dict) -> str:
        """生成修复建议。"""
        front = issue["front"].rstrip("，, ")
        # 基本策略：保留前半句，砍掉尾注，用句号收住
        if not front.endswith(("。", "！", "？")):
            front += "。"
        return front

    def _analyze_cross_sentence(self, prev: str, curr: str) -> Optional[Dict]:
        """分析跨句解释性尾注结构。"""
        connector_pattern = re.compile(r"^[，,]*?(那是|这是|那不是|这不是|那便是|这便是|那都是|这都是|那种|这种)")
        connector_match = connector_pattern.search(curr)
        if not connector_match:
            return None

        has_anchor = any(anchor in prev for anchor in self._anchor_set)
        if not has_anchor:
            return None

        has_abstract = any(tag in curr for tag in self._abstract_set)
        if not has_abstract:
            return None

        matched_anchors = [a for a in self._anchor_set if a in prev]
        matched_tags = [t for t in self._abstract_set if t in curr]

        return {
            "hit": f"{prev} {curr}",
            "reason": f"跨句解释性尾注：上一句已有具体描写（{'/'.join(matched_anchors[:3])}），"
                      f"当前句却贴了抽象标签（{'/'.join(matched_tags[:3])}）。应直接砍掉当前句。",
            "front": prev,
            "tail": curr,
            "severity": "P1",
            "matched_anchors": matched_anchors,
            "matched_tags": matched_tags,
        }


class PrologueHookAnalyzer:
    """楔子开篇钩子检测器 - 黄金三原则版"""

    def analyze_opening(self, first_1000_chars: str) -> Dict:
        """
        根据黄金三原则审计开篇：
        1. 3秒生死线 (痛点/冲突)
        2. 300字身份线 (不可替代性/核按钮)
        3. 好奇心鸿沟 (认知反差)
        """
        issues = []
        scores = {"pain": 0, "identity": 0, "curiosity": 0}

        text_300 = first_1000_chars[:300]
        text_full = first_1000_chars

        # 1. 3秒生死线 (前 100 字)
        pain_markers = r'杀|死|毒|废|退婚|断绝|火|血|没收|夺|抢|债|背叛|绝路|命数|时辰到|利刃|寒光'
        scenery_markers = r'阳光|微风|群山|云雾|宁静|喧嚣|景色|岁月|那是|很久以前|传说'

        if re.search(pain_markers, text_300[:100]):
            scores["pain"] = 100
        elif re.search(scenery_markers, text_300[:100]):
            issues.append("❌ 【3秒生死线故障】：开篇前 100 字陷入写景或背景陈述。读者 3 秒内未感到任何威胁、痛苦或生存危机。")
        else:
            issues.append("⚠️ 【3秒生死线薄弱】：开篇缺乏即时冲突。建议立即抛出极致痛点（如功劳被冒领、利刃压颈）。")

        # 2. 300字身份线 (前 300 字)
        # 审计主角是否展现了独特的底气/技能/核按钮
        identity_markers = r'账|算|医|毒|符|阵|剑意|杀气|冷静|底牌|证据|名单|筹码|后手'
        if re.search(identity_markers, text_300):
            scores["identity"] = 100
        else:
            issues.append("⚠️ 【300字身份线缺失】：主角在前 300 字显得过于被动，未展现出独特的“不可替代性”或反击的“核按钮”。")

        # 3. 好奇心鸿沟 (整体认知反差)
        # 寻找“极强威胁”与“极淡定主角”的并存
        threat_found = re.search(r'威胁|死|杀|逼|迫', text_full)
        calm_found = re.search(r'平淡|冷静|微不可察|笃地|慢条斯理|嘴角', text_full)
        if threat_found and calm_found:
            scores["curiosity"] = 100
        else:
            issues.append("⚠️ 【好奇心鸿沟过窄】：缺乏认知反差。建议制造一个“极端冲突”与“极端冷静主角”的碰撞，拉大读者的好奇心。")

        # 4. AI感常规拦截
        if "那是" in text_300[:50] or "他意识到" in text_300[:50]:
            issues.append("❌ 【AI指纹拦截】：开篇使用了典型的 AI 总结式/全知视角起手，极其赶客。")

        total_score = sum(scores.values()) / 3
        return {
            "total_score": round(total_score, 1),
            "has_hook": total_score >= 60,
            "issues": issues,
            "scores": scores
        }


class SemanticConflictExtractor:
    """语义冲突提取引擎：利用 LLM 识别核心冲突，并生成强留人的开篇钩子。"""

    def __init__(self, llm_caller: Optional[Callable] = None):
        self.llm_caller = llm_caller

    def extract_and_generate_hook(self, text: str, global_context: str = "") -> Dict[str, str]:
        """
        1. 识别全文最核心的冲突点（利益、生命、尊严、情感反转）。
        2. 生成一段 100-200 字的“开幕雷击”式楔子。
        """
        if not self.llm_caller:
            return {"conflict": "未配置 LLM，无法自动提取冲突", "hook": ""}

        context_str = f"参考全书核心逻辑（如可用）：\n{global_context}\n\n" if global_context else ""

        prompt = f"你是一个金牌小说编辑。请阅读以下正文，并执行以下任务：\n" \
                 f"{context_str}" \
                 f"1. 提取出本章中冲突最激烈、最能抓人眼球的‘核心矛盾点’（用一句话概括）。\n" \
                 f"2. 基于这个矛盾点，写一段 150 字左右的‘开幕雷击’式起手（楔子）。\n" \
                 f"要求：极致压抑或极致爆发，充满感官细节（痛感、温度、气味），严禁 AI 总结式废话，严禁写景。\n\n" \
                 f"正文内容：\n{text[:5000]}" # 采样前 5000 字

        sys_prompt = "你只输出 JSON 格式，包含字段: 'conflict', 'hook'。不要输出任何其他解释。"

        try:
            res = self.llm_caller(prompt, sys_prompt)
            if res:
                # 尝试从 markdown 代码块中提取
                json_str = re.search(r'\{.*\}', res, re.S)
                if json_str:
                    data = json.loads(json_str.group())
                    return data
        except Exception as e:
            return {"conflict": f"提取失败: {str(e)}", "hook": ""}

        return {"conflict": "无法解析冲突", "hook": ""}


class GlobalVisionHooker:
    """
    全书视野钩子引擎：利用 ChromaDB 检索全书冲突轨迹。
    支持物理隔离（一书一集合）与性能优化（动态加载）。
    """
    def __init__(self, db_path: Optional[str] = None, book_id: str = "default", llm_caller: Optional[Callable] = None):
        self.db_path = db_path
        self.book_id = book_id
        self.llm_caller = llm_caller
        self.client = None
        self.collection = None
        if db_path:
            try:
                import chromadb
                import hashlib
                # 性能优化：使用持久化客户端
                self.client = chromadb.PersistentClient(path=db_path)

                # 物理隔离：根据 book_id 生成唯一的集合名称（符合 ChromaDB 命名规范）
                safe_book_id = hashlib.md5(book_id.encode()).hexdigest()
                collection_name = f"novel_{safe_book_id}"

                # 动态加载：仅获取当前书籍的集合
                self.collection = self.client.get_or_create_collection(name=collection_name)
                print(f"📦 ChromaDB 集合已挂载: {collection_name} (Source: {book_id})")
            except ImportError:
                print("⚠️ [警告] 未安装 chromadb，GlobalVisionHooker 降级。")
            except Exception as e:
                print(f"⚠️ [错误] ChromaDB 初始化失败: {e}")

    def ingest_novel(self, text: str):
        """
        将整本小说内容写入 ChromaDB。按段落切片。
        """
        if not self.collection:
            print("⚠️ [错误] ChromaDB 集合未挂载，无法写入。")
            return

        # 物理隔离：先清空当前书籍的集合，防止旧数据污染
        try:
            self.collection.delete()
            print(f"🧹 已清空旧集合: {self.collection.name}")
        except Exception:
            pass

        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if not paragraphs:
            return

        import uuid
        ids = [str(uuid.uuid4()) for _ in paragraphs]
        metadatas = [{"book_id": self.book_id, "para_idx": i} for i in range(len(paragraphs))]

        # 批量写入（ChromaDB 建议分批）
        batch_size = 100
        for i in range(0, len(paragraphs), batch_size):
            self.collection.add(
                documents=paragraphs[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print(f"✅ [ChromaDB] 已成功写入 {len(paragraphs)} 个段落到集合 {self.collection.name}。")

    def query_future_conflicts(self, query: str = "全书最核心的秘密、反转或生死冲突是什么？", n_results: int = 5) -> str:
        """
        检索全书中的关键冲突切片。
        """
        if not self.collection:
            return ""

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return "\n---\n".join(results['documents'][0])
        except Exception as e:
            return f"检索失败: {e}"

    def synthesize_global_hook(self, current_chapter_text: str) -> Dict:
        """
        结合当前章节与全书检索结果，生成一个具备“全书视野”的顶级开篇。
        """
        # 1. 检索全书高能冲突
        future_context = self.query_future_conflicts()

        # 2. 调用 LLM 进行跨时空博弈合成
        if not self.llm_caller:
            return {"hook": "LLM 缺失"}

        prompt = f"你是一个顶级的‘倒叙叙事’大师。请根据以下‘全书未来高能冲突切片’和‘当前开篇正文’，" \
                 f"为本小说合成一个充满宿命感、压抑感且‘开幕雷击’的全新序章。\n\n" \
                 f"【全书未来冲突参考】：\n{future_context}\n\n" \
                 f"【当前开篇正文】：\n{current_chapter_text[:2000]}\n\n" \
                 f"要求：\n" \
                 f"1. 提取全书最核心的‘核按钮’（秘密或杀招）作为钩子暗示。\n" \
                 f"2. 极致的感官细节渲染（如：旧纸的味道、朱砂的血色、冷雨的腥气）。\n" \
                 f"3. 严禁 AI 总结感。不要写‘这是一个关于...的故事’。\n" \
                 f"4. 长度在 300-500 字左右。\n\n" \
                 f"输出 JSON 格式: {{'global_conflict': '概括', 'thunderstrike_hook': '重写后的内容'}}"

        sys_prompt = "只输出 JSON 格式。不要废话。"

        try:
            res = self.llm_caller(prompt, sys_prompt)
            json_str = re.search(r'\{.*\}', res, re.S)
            if json_str:
                return json.loads(json_str.group())
        except Exception as e:
            return {"error": str(e)}

        return {"error": "Synthesis failed"}



class CharacterVoiceDifferentiator:
    """角色声轨分离器：检测角色对白是否千人一面"""
    def __init__(self):
        from collections import defaultdict
        self._defaultdict = defaultdict

    def analyze_voices(self, text: str, characters: List[Dict[str, str]]) -> List[Dict]:
        issues = []
        char_names = [c["name"] for c in characters]
        if not char_names:
            return issues

        dialogue_pattern = re.compile(r'([^\n。！？]+)(?:道|说|问|开口|低声).*?["\u201c]([^"\u201d]+)["\u201d]')
        char_dialogues = self._defaultdict(list)

        for match in dialogue_pattern.finditer(text):
            narrative = match.group(1)
            quote = match.group(2)
            speaker = "Unknown"
            for name in char_names:
                if name in narrative:
                    speaker = name
                    break
            if speaker != "Unknown":
                char_dialogues[speaker].append(quote)

        # 同质化检测
        if len(char_dialogues) >= 2:
            avg_lengths = {s: sum(len(q) for q in qs)/len(qs) for s, qs in char_dialogues.items()}
            lengths = list(avg_lengths.values())
            if max(lengths) > 0 and (max(lengths) - min(lengths)) / max(lengths) < 0.15:
                issues.append({
                    "reason": "角色声轨同质化警报：不同角色的台词平均长度和结构过于相似（千人一面），缺乏个性化口吻。建议按身份重写部分对白。",
                    "details": avg_lengths
                })
        return issues


class MicroImperfectionGenerator:
    """微瑕疵模拟器：随机寻找完全无瑕疵的段落并建议植入烟火气锚点"""
    def __init__(self):
        self.anchor_pool = [
            "桌角剥落的红漆", "鞋底黏着的一块碎石子", "袖口脱出的一截线头",
            "空气中细小的浮尘", "门缝里漏进来的一丝冷风", "指甲缝里的泥垢",
            "椅腿摩擦地面的刺耳声", "衣襟上沾的一点油渍", "头发丝上的汗珠",
            "牙齿咬合时的酸涩", "鞋帮子蹭破的一块皮"
        ]
        # P3 Fix #17: 扩展生活锚点判断词库
        self.grounding_nouns = {
            "桌", "椅", "墙", "地", "灰", "风", "杯", "血",
            "土", "泥", "汗", "尘", "碗", "盘", "锅", "碎",
            "线头", "油渍", "裂缝", "指甲", "鞋", "袖", "衣角",
            "蜡", "烛", "炭", "灰烬", "木屑", "铁锈", "绳",
            "石子", "沙砾", "草叶", "树皮", "虫", "鼠", "蛛网",
        }
        # 非视觉感官词（用于综合判断段落是否有烟火气）
        self.sensory_words = {
            "闻到", "嗅", "腥", "臭", "香", "潮", "湿", "干", "冰",
            "烫", "凉", "热", "冷", "痒", "痛", "麻", "酸", "涩",
            "苦", "甜", "咸", "辣", "粗糙", "光滑", "黏", "嘎吱",
        }

    def check(self, paragraphs: List[str]) -> List[Dict]:
        issues = []
        for idx, p in enumerate(paragraphs):
            if len(p) > 100:
                has_grounding = any(noun in p for noun in self.grounding_nouns)
                has_sensory = any(word in p for word in self.sensory_words)
                if not has_grounding and not has_sensory:
                    suggestion = random.choice(self.anchor_pool)
                    issues.append({
                        "para_idx": idx,
                        "hit": p[:20] + "...",
                        "reason": f"过度平滑警报：该段落缺乏现实生活锚点和非视觉感官，显得像悬浮的AI旁白。建议在动作间隙随机植入微瑕疵，如：'{suggestion}'。"
                    })
                    break  # 每章最多建议一个
        return issues


class NarrativeCausalityAuditor:
    """叙事因果链完整性审计器 - 六维审计

    检测AI重写中最常见的逻辑断层问题：
    1. 悬空实体 - 专有名词/道具首次出现时缺乏最小化解释
    2. 因果断裂 - 动作的对象在前文没有引入
    3. 场景跳跃 - 相邻段落空间标记无过渡切换
    4. 情感断裂 - 角色情感状态不合理跳跃
    5. 时间线断裂 - 时间标记不连贯
    6. 开篇因果完整性 - 楔子/第一章前500字核心冲突缺乏WHO/WHERE/WHY

    设计为通用模块，不依赖特定小说配置，对第N本小说自动生效。
    """

    # 常见的身份标签词（用于判断实体是否有最小化解释）
    IDENTITY_MARKERS = [
        "的", "是", "叫", "名叫", "名为", "称为", "人称", "外号",
        "掌柜", "东家", "老板", "会首", "账首", "掌门", "弟子", "徒弟",
        "公子", "小姐", "姑娘", "夫人", "老爷", "大人", "将军",
        "师父", "师兄", "师妹", "父亲", "母亲", "兄长", "妹妹",
        "商会", "银号", "票号", "镖局", "衙门", "宗门",
    ]

    # 场景/空间标记词
    SCENE_MARKERS = [
        "堂", "院", "房", "室", "阁", "楼", "街", "巷", "码头", "城门",
        "山", "林", "河", "湖", "洞", "崖", "庙", "祠", "仓", "库",
        "门口", "门前", "门外", "门内", "窗前", "窗边", "廊下", "阶前",
        "屋内", "屋外", "厅内", "厅外", "堂内", "堂外", "院内", "院外",
    ]

    # 时间标记词
    TIME_MARKERS = {
        "night": ["夜", "深夜", "半夜", "子时", "丑时", "入夜", "夜色", "月", "星"],
        "dawn": ["拂晓", "天亮", "鱼肚白", "晨", "清早", "卯时", "天明"],
        "morning": ["上午", "辰时", "巳时", "晌午前", "早"],
        "noon": ["正午", "午时", "晌午", "中午", "日头"],
        "afternoon": ["下午", "未时", "申时", "日头西斜"],
        "evening": ["傍晚", "黄昏", "日暮", "酉时", "夕阳", "落日", "暮色"],
    }

    # 情感标记词（用于检测情感跳跃）
    EMOTION_MARKERS = {
        "anger": ["愤怒", "暴怒", "怒", "咆哮", "咬牙", "气得", "恼", "火冒"],
        "fear": ["恐惧", "害怕", "惊恐", "发抖", "颤栗", "瑟瑟", "惶恐"],
        "calm": ["平静", "淡然", "从容", "冷静", "微笑", "笑意", "轻笑"],
        "sadness": ["悲伤", "流泪", "泪", "哭", "心酸", "凄然", "哀"],
        "shock": ["惊", "愣", "呆", "震惊", "错愕", "目瞪口呆"],
    }

    # 动作-对象配对词（用于检测因果断裂）
    ACTION_OBJECT_PATTERNS = [
        (r"撕碎(?:了)?(.{1,8})", "撕碎"),
        (r"扔掉(?:了)?(.{1,8})", "扔掉"),
        (r"锁(?:上|死|住)(?:了)?(.{1,8})", "锁"),
        (r"打开(?:了)?(.{1,8})", "打开"),
        (r"抽出(?:了)?(.{1,8})", "抽出"),
        (r"推(?:开|倒)(?:了)?(.{1,8})", "推"),
        (r"拿(?:起|出|走)(?:了)?(.{1,8})", "拿"),
        (r"摊(?:开|在)(?:了)?(.{1,8})", "摊"),
    ]

    def audit_character_fate(self, text: str, character_fates: Dict[str, str]) -> List[str]:
        """检测已终结命运的角色是否在文本中异常出现（诈尸检测）。"""
        issues = []
        for char_name, fate in character_fates.items():
            if fate in ["Dead", "Incapacitated", "Exiled"] and char_name in text:
                # 检查是否只是在对话或回忆中提到
                # 如果有具体的动作描写，则是逻辑错误
                # 模式：角色名 + 动作动词
                action_pattern = rf"{char_name}[^。！？；\n]{{0,10}}(?:说|道|笑|哭|走|站|坐|看|望|拿|动|听|闻|见)"
                if re.search(action_pattern, text):
                    issues.append(f"角色命运冲突：'{char_name}' 已处于 '{fate}' 状态，但在本章有具体的动作描写（疑似诈尸）。")
        return issues

    def audit_chapter(self, chapter_text: str, chapter_name: str,
                      preceding_context: str = "", character_fates: Dict[str, str] = None) -> Dict:
        """审计一个章节的因果链完整性。

        Args:
            chapter_text: 当前章节正文
            chapter_name: 章节名称
            preceding_context: 前序章节文本（用于跨章审计）
            character_fates: 人物命运账本 (V12)

        Returns:
            审计报告字典
        """
        paragraphs = [p.strip() for p in chapter_text.split('\n') if p.strip()]

        report = {
            "chapter": chapter_name,
            "dangling_entities": [],
            "causal_gaps": [],
            "scene_jumps": [],
            "emotion_gaps": [],
            "timeline_breaks": [],
            "opening_issues": [],
            "fate_issues": [],
            "total_issues": 0,
            "severity": "OK",
        }

        # V12 Fate Check
        if character_fates:
            report["fate_issues"] = self.audit_character_fate(chapter_text, character_fates)
            report["total_issues"] += len(report["fate_issues"])
            
        # V12 Ending Summary Check
        report["summary_issues"] = self._detect_ending_summaries(paragraphs)
        report["total_issues"] += len(report["summary_issues"])

        # V12 Knowledge Leak Check
        report["knowledge_leaks"] = self._detect_knowledge_leaks(paragraphs)
        report["total_issues"] += len(report["knowledge_leaks"])

        # 1. 悬空实体检测
        report["dangling_entities"] = self._detect_dangling_entities(
            chapter_text[:1500], preceding_context
        )

        # 2. 因果断裂检测
        report["causal_gaps"] = self._detect_causal_gaps(paragraphs)

        # 3. 场景跳跃检测
        report["scene_jumps"] = self._detect_scene_jumps(paragraphs)

        # 4. 情感断裂检测
        report["emotion_gaps"] = self._detect_emotion_gaps(paragraphs)

        # 5. 时间线断裂检测
        report["timeline_breaks"] = self._detect_timeline_breaks(paragraphs)

        # 6. 开篇因果完整性（仅对楔子/第一章）
        is_opening = any(kw in chapter_name for kw in
                         ["楔子", "序章", "前言", "第一章", "第1章", "Chapter 1"])
        if is_opening:
            report["opening_issues"] = self._audit_opening_causality(
                chapter_text[:1500], chapter_text
            )

        # 汇总
        total = (len(report["dangling_entities"]) +
                 len(report["causal_gaps"]) +
                 len(report["scene_jumps"]) +
                 len(report["emotion_gaps"]) +
                 len(report["timeline_breaks"]) +
                 len(report["opening_issues"]))
        report["total_issues"] = total

        if report["opening_issues"]:
            report["severity"] = "P0"
        elif total >= 3:
            report["severity"] = "P1"
        elif total >= 1:
            report["severity"] = "P2"

        return report

    def _detect_dangling_entities(self, text: str, context: str) -> List[Dict]:
        """检测悬空实体：专有名词首次出现时缺乏最小化解释。"""
        issues = []
        # 提取可能的专有名词（2-4字中文词，出现在前500字）
        first_500 = text[:500]
        # 提取人名模式：X说/X道/X问 等
        name_candidates = set(re.findall(
            r'([\u4e00-\u9fff]{2,4})(?:说|道|问|喊|冷笑|开口|站|坐|走)',
            first_500
        ))
        # 提取道具/地点模式：的X、X里/中/上
        item_candidates = set(re.findall(
            r'(?:那|这|一)(?:叠|封|张|把|枚|块|盏)([\u4e00-\u9fff]{2,4})',
            first_500
        ))

        # 常见功能词排除
        stop_words = {"自己", "什么", "这种", "那种", "一个", "他们", "她们",
                      "所有", "因为", "但是", "虽然", "如果", "这样", "那样"}

        all_candidates = (name_candidates | item_candidates) - stop_words

        for entity in all_candidates:
            if len(entity) < 2:
                continue
            # 在前文或上下文中是否有解释
            if context and entity in context:
                continue  # 前序章节已出现

            # 检查该实体在±200字范围内是否有身份标签
            positions = [m.start() for m in re.finditer(re.escape(entity), first_500)]
            if not positions:
                continue

            first_pos = positions[0]
            window_start = max(0, first_pos - 100)
            window_end = min(len(text), first_pos + len(entity) + 200)
            window = text[window_start:window_end]

            has_explanation = False
            for marker in self.IDENTITY_MARKERS:
                # 检查实体附近是否有身份标签（如"青州商会的会首"）
                if re.search(
                    rf'{re.escape(entity)}.{{0,20}}{re.escape(marker)}|'
                    rf'{re.escape(marker)}.{{0,10}}{re.escape(entity)}',
                    window
                ):
                    has_explanation = True
                    break

            if not has_explanation:
                issues.append({
                    "entity": entity,
                    "position": first_pos,
                    "reason": f"悬空实体：'{entity}'在开篇首次出现，但±200字内"
                              f"缺乏身份/功能的最小化解释。读者无法理解它是什么。"
                })

        return issues[:5]  # 限制输出数量

    def _detect_causal_gaps(self, paragraphs: List[str]) -> List[Dict]:
        """检测因果断裂：动作的对象在前文没有引入。"""
        issues = []
        accumulated_text = ""

        for p_idx, para in enumerate(paragraphs):
            for pattern, action_name in self.ACTION_OBJECT_PATTERNS:
                for m in re.finditer(pattern, para):
                    obj = m.group(1).strip("，。！？；的了着")
                    if len(obj) < 2 or len(obj) > 6:
                        continue
                    # 检查对象是否在之前的文本中出现过
                    if obj not in accumulated_text and p_idx > 0:
                        issues.append({
                            "paragraph": p_idx + 1,
                            "action": action_name,
                            "object": obj,
                            "reason": f"因果断裂：'{action_name}{obj}'，"
                                      f"但'{obj}'在之前的文本中从未出现，"
                                      f"读者不知道这个对象从何而来。"
                        })
            accumulated_text += para

        return issues[:5]

    def _detect_scene_jumps(self, paragraphs: List[str]) -> List[Dict]:
        """检测场景跳跃：相邻段落空间标记无过渡切换。"""
        issues = []

        def extract_scene(text: str) -> Optional[str]:
            for marker in self.SCENE_MARKERS:
                # 匹配如"岁验堂"、"城南盐仓"等
                m = re.search(rf'([\u4e00-\u9fff]{{0,4}}{re.escape(marker)})', text)
                if m:
                    return m.group(1)
            return None

        prev_scene = None
        prev_idx = -1
        transition_words = ["走到", "赶到", "来到", "抵达", "去了", "进了",
                            "踏入", "步入", "推门", "穿过", "折回", "拐进"]

        for p_idx, para in enumerate(paragraphs):
            curr_scene = extract_scene(para)
            if curr_scene and prev_scene:
                if curr_scene != prev_scene:
                    # 检查是否有过渡
                    has_transition = any(tw in para for tw in transition_words)
                    if not has_transition and (p_idx - prev_idx) <= 2:
                        issues.append({
                            "from_para": prev_idx + 1,
                            "to_para": p_idx + 1,
                            "from_scene": prev_scene,
                            "to_scene": curr_scene,
                            "reason": f"场景跳跃：从'{prev_scene}'直接切到"
                                      f"'{curr_scene}'，缺乏过渡动作。"
                        })
            if curr_scene:
                prev_scene = curr_scene
                prev_idx = p_idx

        return issues[:5]

    def _detect_ending_summaries(self, paragraphs: List[str]) -> List[str]:
        """检测每章结尾常见的AI总结性废话。"""
        issues = []
        if not paragraphs: return issues
        
        # 重点检查最后2段
        ending_paras = paragraphs[-2:]
        summary_patterns = [
            r"(?:一场|这(?:次|场)|新的)(?:拉锯|博弈|死局|较量|序幕|风暴|暗流|战场|传奇).{0,15}(?:刚刚开始|拉开帷幕|彻底钉死|浮出水面|席卷而来|扎根)",
            r"属于(?:他|她|他们)的.{0,5}才刚刚(?:铺开|开始)",
            r"是.{0,5}的预警，也是.{0,5}的倒计时",
            r"(?:暗流|危机).{0,10}悄然(?:汇聚|滋生|涌动)",
            r"从未如此(?:自由|坚定|轻松)",
        ]
        
        for idx, para in enumerate(ending_paras):
            para_idx = len(paragraphs) - len(ending_paras) + idx + 1
            for pattern in summary_patterns:
                if re.search(pattern, para):
                    # 语义识别：如果这段话除了总结还有具体的动作/视觉描写，则建议修改而非彻底删除
                    has_concrete_content = len(para) > 40 or re.search(r"抱着|走进|挥剑|跨出|收起|走下|火盆|营房", para)
                    if has_concrete_content:
                        issues.append(f"段落 {para_idx} 包含过度总结的AI套路词（如：{pattern}），建议剥离抽象总结，保留具体动作。")
                    else:
                        issues.append(f"段落 {para_idx} 是典型的AI总结性废话（博弈/开始/死局等），建议直接删除。")
                    break # 一个段落只报一个总结问题
        return issues

    def _detect_knowledge_leaks(self, paragraphs: List[str]) -> List[Dict]:
        """检测知识泄露/提前反应逻辑错误（读心逻辑纠错）。"""
        issues = []
        disclosed_entities = set()
        
        for p_idx, para in enumerate(paragraphs):
            # 1. 记录对话中新披露的信息 (仅限说话，避开读心/心理)
            # 模式：XX说：“...” 或 “...” XX道
            if re.search(r"说|道|问|喊|冷笑|叹道|咬牙|呵斥", para):
                dialogues = re.findall(r"“(.*?)”", para)
                for d in dialogues:
                    entities = re.findall(r"([\u4e00-\u9fff]{2,10}(?:钱庄|账本|门|府|密道|计划|名单|真相|证据|秘密|消息|余孽))", d)
                    for e in entities:
                        disclosed_entities.add(e)
            
            # 2. 检查读心/心理活动是否提到了未披露的信息
            # 模式：读心内容通常包含关键词，且是在某种心理描写之后
            thought_patterns = [
                r"“([^“”]*?(?:竟然知道|为何会知道|怎么知道|居然知道|是从哪知道|也知道|知道|晓得)[^“”]*)”",
                r"心中[^“”]*?[：:](?:“|『)([^“”『』]*?)(?:”|』)",
                r"想(?:道|着)[^“”]*?[：:](?:“|『)([^“”『』]*?)(?:”|』)",
            ]
            
            found_thoughts = []
            for pat in thought_patterns:
                matches = re.findall(pat, para)
                for m in matches:
                    if isinstance(m, tuple):
                        found_thoughts.extend([t for t in m if t])
                    else:
                        found_thoughts.append(m)
            
            for thought in found_thoughts:
                # 寻找专有名词（钱庄、地名等）
                target_entities = re.findall(r"([\u4e00-\u9fff]{2,10}(?:钱庄|账本|门|府|密道|计划|名单|真相|证据|秘密|消息|余孽))", thought)
                for te in target_entities:
                    # 去除前导语气词
                    clean_te = re.sub(r"^(?:这女人|她|他|这家伙|竟然|居然|怎么|为何|到底|居然)?(?:知道|晓得|发现)?", "", te)
                    if clean_te and clean_te not in disclosed_entities:
                        issues.append({
                            "paragraph": p_idx + 1,
                            "entity": clean_te,
                            "reason": f"读心逻辑冲突：角色惊叹对方知道 '{clean_te}'，但此前对话中并未提及该信息（信息差断层）。"
                        })
        return issues

    def _detect_emotion_gaps(self, paragraphs: List[str]) -> List[Dict]:
        """检测情感断裂：角色情感不合理跳跃。"""
        issues = []

        def detect_emotion(text: str) -> Optional[str]:
            for emotion, markers in self.EMOTION_MARKERS.items():
                if any(m in text for m in markers):
                    return emotion
            return None

        # 不兼容的情感对
        incompatible = {
            ("anger", "calm"), ("fear", "calm"),
            ("sadness", "calm"), ("anger", "sadness"),
        }

        prev_emotion = None
        prev_idx = -1

        for p_idx, para in enumerate(paragraphs):
            curr_emotion = detect_emotion(para)
            if curr_emotion and prev_emotion:
                pair = tuple(sorted([prev_emotion, curr_emotion]))
                if pair in incompatible and (p_idx - prev_idx) <= 1:
                    # 检查是否有过渡词
                    transition = any(w in para for w in
                                     ["渐渐", "慢慢", "终于", "深呼吸",
                                      "长出一口气", "收敛", "压下"])
                    if not transition:
                        issues.append({
                            "from_para": prev_idx + 1,
                            "to_para": p_idx + 1,
                            "from_emotion": prev_emotion,
                            "to_emotion": curr_emotion,
                            "reason": f"情感断裂：从'{prev_emotion}'突然跳到"
                                      f"'{curr_emotion}'，缺乏心理过渡。"
                        })
            if curr_emotion:
                prev_emotion = curr_emotion
                prev_idx = p_idx

        return issues[:3]

    def _detect_timeline_breaks(self, paragraphs: List[str]) -> List[Dict]:
        """检测时间线断裂：时间标记不连贯。"""
        issues = []
        TIME_ORDER = ["night", "dawn", "morning", "noon",
                      "afternoon", "evening", "night"]

        def detect_time(text: str) -> Optional[str]:
            for period, markers in self.TIME_MARKERS.items():
                if any(m in text for m in markers):
                    return period
            return None

        prev_time = None
        prev_idx = -1

        for p_idx, para in enumerate(paragraphs):
            curr_time = detect_time(para)
            if curr_time and prev_time and curr_time != prev_time:
                # 检查是否反向跳跃（如从morning直接到night没有过渡）
                try:
                    prev_order = TIME_ORDER.index(prev_time)
                    curr_order = TIME_ORDER.index(curr_time)
                except ValueError:
                    continue

                gap = curr_order - prev_order
                if gap < 0:
                    gap += len(TIME_ORDER)
                # 跳跃超过2个时段且没有过渡
                if gap >= 3 and (p_idx - prev_idx) <= 2:
                    transition = any(w in para for w in
                                     ["过了", "几个时辰", "到了", "等到",
                                      "直到", "天色", "不知过了多久"])
                    if not transition:
                        issues.append({
                            "from_para": prev_idx + 1,
                            "to_para": p_idx + 1,
                            "from_time": prev_time,
                            "to_time": curr_time,
                            "reason": f"时间线断裂：从'{prev_time}'跳到"
                                      f"'{curr_time}'，缺乏时间过渡。"
                        })
            if curr_time:
                prev_time = curr_time
                prev_idx = p_idx

        return issues[:3]

    def _audit_opening_causality(self, opening_text: str,
                                  full_chapter: str) -> List[str]:
        """专门审计开篇的因果完整性（WHO/WHERE/WHY三要素）。

        检查开篇前500字中：
        1. 是否有角色身份的最小化交代（WHO）
        2. 是否有场景/位置的交代（WHERE）
        3. 核心冲突动作是否有因果前提（WHY）
        """
        issues = []
        first_300 = opening_text[:300]
        first_500 = opening_text[:500]

        # WHO检测：前300字是否有角色+身份标签
        names = re.findall(
            r'([\u4e00-\u9fff]{2,4})(?:说|道|问|的|站|坐)',
            first_300
        )
        has_identity = False
        for name in names:
            if len(name) < 2:
                continue
            for marker in ["掌柜", "东家", "会首", "账首", "公子", "姑娘",
                           "大人", "师父", "商会", "银号", "票号"]:
                if marker in first_500:
                    has_identity = True
                    break
            if has_identity:
                break

        if names and not has_identity:
            issues.append(
                "【P0:WHO缺失】开篇出现角色名但缺乏身份交代。"
                "读者不知道主角是谁、做什么的。"
                "必须在角色首次出现的±200字内给出最小化身份标签。"
            )

        # WHERE检测：前300字是否有场景交代
        has_scene = any(marker in first_300 for marker in self.SCENE_MARKERS)
        if not has_scene:
            issues.append(
                "【P1:WHERE缺失】开篇300字内缺乏场景/位置交代。"
                "读者不知道故事发生在哪里。"
            )

        # WHY检测：如果有强动作（撕碎/锁死/攥紧等），检查是否有因果前提
        strong_actions = re.findall(
            r'(撕碎|锁死|攥紧|砸碎|扔掉|摔碎|推翻|掀翻)', first_500
        )
        if strong_actions:
            # 检查动作前是否有原因交代
            for action in strong_actions:
                pos = first_500.find(action)
                if pos < 100:  # 动作出现在前100字
                    # 检查动作前面是否有"因为/由于/发现/看到"等因果词
                    pre_text = first_500[:pos]
                    cause_markers = ["因为", "由于", "发现", "看到",
                                     "察觉", "原来", "竟然", "不对"]
                    has_cause = any(cm in pre_text for cm in cause_markers)
                    if not has_cause:
                        issues.append(
                            f"【P0:WHY缺失-空降高潮】开篇前100字出现强动作"
                            f"'{action}'，但之前缺乏因果前提。"
                            f"读者不知道角色为什么要这样做。"
                            f"严禁将后文高潮片段直接移到开头而不补充因果链。"
                        )

        return issues


class ContentParityValidator:
    """内容保量硬校验器 (P0 级门控)

    确保重写后的文本字数不低于原文的指定比例。
    适用于所有路径（正则重写 + LLM 重写），在保存前强制拦截。
    设计为通用模块，对第 N 本小说自动生效。
    """

    MIN_RATIO = 0.90   # 最低保量比 (90%)：低于此值直接回退原文
    MAX_RATIO = 1.60   # 最高膨胀比 (160%)：高于此值警告可能注水
    WARN_RATIO = 0.95  # 警告线 (95%)：低于此值输出警告但不拦截

    @staticmethod
    def count_cn(text: str) -> int:
        """统计中文字数。"""
        return len(re.findall(r'[\u4e00-\u9fff]', text))

    def validate(self, original: str, revised: str, chapter_name: str = "") -> dict:
        """验证重写后文本的字数保量。

        Returns:
            dict: {
                "passed": bool,      # 是否通过验证
                "warning": bool,     # 是否触发警告
                "original_count": int,
                "revised_count": int,
                "ratio": float,
                "reason": str,
            }
        """
        orig_cn = self.count_cn(original)
        rev_cn = self.count_cn(revised)

        # 短文本（< 200 中文字）不强制校验
        if orig_cn < 200:
            return {
                "passed": True,
                "warning": False,
                "original_count": orig_cn,
                "revised_count": rev_cn,
                "ratio": rev_cn / max(orig_cn, 1),
                "reason": "short_text_bypass",
            }

        ratio = rev_cn / max(orig_cn, 1)

        result = {
            "passed": True,
            "warning": False,
            "original_count": orig_cn,
            "revised_count": rev_cn,
            "ratio": ratio,
            "reason": "ok",
        }

        if ratio < self.MIN_RATIO:
            result["passed"] = False
            result["reason"] = (
                f"字数保量失败: 原文{orig_cn}字→重写{rev_cn}字 "
                f"({ratio:.0%})，低于最低保量线{self.MIN_RATIO:.0%}。"
                f"重写过程中累积删除了过多内容。"
            )
        elif ratio < self.WARN_RATIO:
            result["warning"] = True
            result["reason"] = (
                f"字数保量警告: 原文{orig_cn}字→重写{rev_cn}字 "
                f"({ratio:.0%})，接近保量下限。"
            )
        elif ratio > self.MAX_RATIO:
            result["warning"] = True
            result["reason"] = (
                f"字数膨胀警告: 原文{orig_cn}字→重写{rev_cn}字 "
                f"({ratio:.0%})，超过膨胀上限{self.MAX_RATIO:.0%}，"
                f"可能存在注水。"
            )

        return result

    def enforce(self, original: str, revised: str, chapter_name: str = "") -> tuple:
        """强制校验并在失败时回退原文。

        Returns:
            tuple: (final_text, parity_report)
        """
        report = self.validate(original, revised, chapter_name)

        if not report["passed"]:
            return original, report

        return revised, report
