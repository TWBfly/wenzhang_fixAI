import os
import re

file_path = '/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI/duanpian_fixAI_1.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add import statement at top
if 'from advanced_analyzers' not in content:
    content = content.replace('import logging', 'import logging\nfrom advanced_analyzers import RhythmShatter, SensoryWeaver, RedundancyDetector, POVChecker, DialoguePowerAnalyzer, ShowDontTellTransformer\nimport traceback')

# 2. Fix indentation
bad_indentation = """        if enable_semantic_opt:
            # Inject Style-Specific Target
        if context.genre:"""
good_indentation = """        if enable_semantic_opt:
            # Inject Style-Specific Target
            if context.genre:"""
content = content.replace(bad_indentation, good_indentation)

# 3. Add classes instances to __init__
init_str = """        self.agent_task_gen = AgentRefinementTask(self.root_dir)"""
new_init_str = """        self.agent_task_gen = AgentRefinementTask(self.root_dir)
        self.style_profile = None  # Init missing variable
        # Advanced analyzers
        self.rhythm_shatter = RhythmShatter()
        self.sensory_weaver = SensoryWeaver()
        self.redundancy_detector = RedundancyDetector()
        self.pov_checker = POVChecker()
        self.dialogue_analyzer = DialoguePowerAnalyzer()
        self.sdt_transformer = ShowDontTellTransformer()"""
if 'self.rhythm_shatter' not in content:
    content = content.replace(init_str, new_init_str)

# 4. AgentRefinementTask overhaul
old_agent_task = """    def generate_task(self, chapter_name: str, text: str, context: NovelContext, changes: List[SentenceChange]) -> Path:
        if not self.template_path.exists():
            raise FileNotFoundError(f"指令模板缺失: {self.template_path}")
        
        template = self.template_path.read_text(encoding="utf-8")
        
        char_str = ", ".join([c["name"] for c in context.characters])
        word_count = len(text)
        
        task_content = template\\
            .replace("{{GENRE}}", context.genre)\\
            .replace("{{BACKGROUND}}", context.background)\\
            .replace("{{CHARACTERS}}", char_str)\\
            .replace("{{WORD_COUNT}}", str(word_count))\\
            .replace("{{SOURCE_TEXT}}", text)

        task_path = self.output_dir / f"refine_{chapter_name}.md"
        task_path.write_text(task_content, encoding="utf-8")
        return task_path"""

new_agent_task = """    def generate_task(self, chapter_name: str, text: str, context: NovelContext, changes: List[SentenceChange], structure_issues=None) -> Path:
        char_str = ", ".join([c["name"] for c in context.characters])
        
        # Build precise diagnostic micro-instructions
        instructions = []
        instructions.append(f"# 精准改写任务: {chapter_name}")
        instructions.append(f"频道: {context.genre} | 背景: {context.background}")
        instructions.append(f"出场角色: {char_str}\\n")
        
        if structure_issues:
            instructions.append("## 结构级高危红线 (请在全局优化时重点修正)")
            for iss in structure_issues:
                instructions.append(f" - ⚠️ {iss}")
            instructions.append("\\n")
            
        instructions.append("## 句子级微指令 (由审计引擎截获的目标片段)")
        if not changes:
            instructions.append("暂无高危句子级需要改写，请全局核实逻辑。")
        else:
            for idx, chg in enumerate(changes[:30]): # 取前30个严重问题
                lines = [
                    f"### Target {idx + 1}: Paragraph {chg.paragraph}, Sentence {chg.sentence}",
                    f"**[原句]** {chg.original}",
                    f"**[诊断]** {' | '.join(chg.applied_rules)}",
                    f"**[IDE Agent 要求]**",
                    f" 1. 消除以上诊断出的问题。",
                    f" 2. 融入感官锚点（触觉、嗅觉、痛觉等），不要仅写视觉。",
                    f" 3. 避免过度平稳的句式，尝试打破句长对称。"
                ]
                instructions.append("\\n".join(lines))
        
        instructions.append("\\n## 参考全文内容\\n```text\\n" + text[:4000] + "\\n...```\\n")
        
        task_path = self.output_dir / f"refine_{chapter_name}.md"
        task_path.write_text("\\n".join(instructions), encoding="utf-8")
        return task_path"""

content = content.replace(old_agent_task, new_agent_task)

# 5. Injection to analyze_chapter for new metrics
analyze_chpt_def = """    def analyze_chapter(self, chapter_name: str, chapter_text: str) -> Dict[str, object]:"""
if analyze_chpt_def in content:
    pass

# We can replace the contents of analyze_chapter safely
old_analyze_chapter = """    def analyze_chapter(self, chapter_name: str, chapter_text: str) -> Dict[str, object]:
        self.logic_auditor.extract_facts(chapter_name, chapter_text)
        
        paragraphs = self.split_paragraphs(chapter_text)
        paragraph_infos = []
        all_hits = []
        suspicious = []
        
        for p_idx, paragraph in enumerate(paragraphs):
            info = self.classify_paragraph(chapter_name, p_idx, paragraph)
            p_severity_counts = Counter()
            
            for s_idx, sentence in enumerate(self.split_sentences(paragraph)):
                hits = self.detect_rules_in_sentence(chapter_name, p_idx, s_idx, sentence)
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
            "empty_transition_blocks": self.detect_empty_transition_blocks(paragraph_infos)
        }"""

new_analyze_chapter = """    def analyze_chapter(self, chapter_name: str, chapter_text: str) -> Dict[str, object]:
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
                    hits.append(Hit(chapter_name, p_idx, s_idx, "SDT违规", "主观情绪标签", sentence, 0, len(sentence), 1.0, "P2"))
                    
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
        }"""
content = content.replace(old_analyze_chapter, new_analyze_chapter)

# Inject structural issues into generate_task calling
old_gen_call = """task_path = self.agent_task_gen.generate_task(chapter_name, rev_text, context, changes)"""
new_gen_call = """# Re-run full analysis to get structure issues
            final_report = self.analyze_chapter(chapter_name, rev_text)
            structure_issues = final_report.get("structure_issues", [])
            task_path = self.agent_task_gen.generate_task(chapter_name, rev_text, context, changes, structure_issues=structure_issues)"""
content = content.replace(old_gen_call, new_gen_call)


with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied.")
