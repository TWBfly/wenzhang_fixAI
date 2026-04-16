import re
import random
import statistics
from typing import List, Dict, Tuple, Set, Optional


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
        r"([她他])(?:感到|觉得|显得|看起来)?(?:极其|十分|格外|有些|非常)?(紧张|害怕|愤怒|不安|慌乱|难堪|压抑|委屈|犹豫|恐惧|悲伤)",
        r"一(?:阵|股|种)(紧张|害怕|愤怒|不安|恐惧|绝望|压迫)(?:感|意)?(?:涌上|袭来|充斥|浮现)"
    ]
    
    # 针对性拦截系统：哲学病句与做作感官夸张
    SUSPENDED_TELLS = [
        r"(超越了.*?(?:极限|界限|常理)|纯粹的.*?|原始的.*?)(痛苦|恐惧|绝望|本能|压迫|冲动)",
        r"像(?:一个|一把|一只).*?的(捕兽夹|齿轮|利刃|机械|生锈的)",
        r"(胃袋|肚子)?腔里.*?(空响|声声|阵阵)?的回音",
        r"如同实质般的",
        r"仿佛抽干了周围所有的",
        r"一种难言的(?:沉默|寂静|压抑)"
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
        r"(?:他|她)(?:是一个|是个)(?:极其|十分|非常|无比)?.{2,15}的(猎手|野兽|疯子|魔鬼|怪物|神明|机器|木偶|赌徒)，也是(?:一个|个)?.{2,15}的(疯子|野兽|猎手|魔鬼|怪物|木偶|赌徒|复仇者)",
        r"(?:他|她)(?:就像|仿佛是?|宛如)(?:一头|一个|一只).{2,10}的(野兽|孤狼|猎豹|怪物|雕塑)"
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
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 8:
                continue
            
            result = self._analyze_sentence(sentence)
            if result:
                issues.append(result)
        
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

