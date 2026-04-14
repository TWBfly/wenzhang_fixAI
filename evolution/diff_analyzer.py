import difflib
import re
from pathlib import Path
from collections import Counter

class DiffEvolutionAnalyzer:
    def __init__(self):
        # 预定义的一些常见的 AI 痕迹正则
        self.ai_patterns = [
            r"不仅.*?更是",
            r"不是.*?而是",
            r"意味着",
            r"显然",
            r"取而代之",
            r"眸色微沉",
            r"指节泛白",
            r"呼吸一滞",
            r"心头一震",
            r"[\$].*?[\$]|\\\(.*?\\\)", # 数学公式
            r"(?:像|如同|仿佛)[^。！？]{2,20}(?:生锈|锋利|撕裂|铁块)", # AI式比喻
        ]

    def analyze_pair(self, original_path: Path, edited_path: Path):
        """对比原文与人工修改后的文章，分析进化方向"""
        print(f"\n{'='*20} 深度进化分析报告: {original_path.parent.name} {'='*20}")
        
        with open(original_path, 'r', encoding='utf-8') as f:
            original = f.read()
        with open(edited_path, 'r', encoding='utf-8') as f:
            edited = f.read()

        # 按段落对齐分析
        ori_paras = [p.strip() for p in original.split("\n\n") if p.strip()]
        edt_paras = [p.strip() for p in edited.split("\n\n") if p.strip()]
        
        diff = list(difflib.ndiff(ori_paras, edt_paras))
        
        removed = []
        added = []
        
        for line in diff:
            if line.startswith('- '):
                removed.append(line[2:].strip())
            elif line.startswith('+ '):
                added.append(line[2:].strip())

        # 1. 检测被删除的 AI 痕迹
        self._detect_removed_patterns(removed)
        
        # 2. 分析句式简化
        self._analyze_sentence_simplification(removed, added)

        # 3. 提取高频被删词汇 (潜在的新进化规则)
        self._extract_potential_rules(removed, added)

    def _detect_removed_patterns(self, removed):
        found = []
        for line in removed:
            for pattern in self.ai_patterns:
                if re.search(pattern, line):
                    found.append(pattern)
        
        if found:
            counts = Counter(found)
            print("\n[关键 AI 痕迹削减频率]:")
            for p, c in counts.most_common():
                print(f" - {p}: {c} 次")

    def _analyze_sentence_simplification(self, removed, added):
        if not removed or not added: return
        avg_rem = sum(len(s) for s in removed) / len(removed) if removed else 0
        avg_add = sum(len(s) for s in added) / len(added) if added else 0
        
        print(f"\n[句式压缩分析]:")
        print(f" - 原文段落平均长: {avg_rem:.1f} 字")
        print(f" - 修改后段落平均长: {avg_add:.1f} 字")
        print(f" - 整体压缩率: {(1 - avg_add / max(avg_rem, 1))*100:.1f}%")

    def _extract_potential_rules(self, removed, added):
        # 简单提取高频被删词汇
        # 这里使用正则表达式模拟分词
        def simple_tokenize(text_list):
            tokens = []
            for t in text_list:
                # 提取 2-4 字的词组
                tokens.extend(re.findall(r"[\u4e00-\u9fff]{2,4}", t))
            return tokens
            
        ori_tokens = Counter(simple_tokenize(removed))
        edt_tokens = Counter(simple_tokenize(added))
        
        diff_tokens = []
        for token, count in ori_tokens.items():
            if count > 2 and (token not in edt_tokens or count > edt_tokens[token] * 2):
                diff_tokens.append((token, count))
        
        if diff_tokens:
            print("\n[建议新增的进化规则 (高频被删词汇)]:")
            for token, count in sorted(diff_tokens, key=lambda x: x[1], reverse=True)[:10]:
                print(f" - {token}: 触发 {count} 次 (建议纳入负面清单)")

if __name__ == "__main__":
    analyzer = DiffEvolutionAnalyzer()
    # 自动搜索当前目录下的样本
    base_dir = Path("/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI/duanpian_zhengwen")
    for i in range(1, 11):
        ori = base_dir / str(i) / "原文.md"
        edt = base_dir / str(i) / "人工修改之后.md"
        if ori.exists() and edt.exists():
            analyzer.analyze_pair(ori, edt)
