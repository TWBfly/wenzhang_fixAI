import re

file_path = '/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI/duanpian_fixAI_1.py'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update rewrite_sentence
old_worker = """            local = self.rewrite_cliche_phrases(local, applied)
            
            if not p0_only:"""

new_worker = """            local = self.rewrite_cliche_phrases(local, applied)
            
            # Catch suspended warnings in the sentence
            sdt_flags = self.sdt_transformer.check(local)
            for flag in sdt_flags:
                applied.append(flag)
            
            if not p0_only:"""

text = text.replace(old_worker, new_worker)

# 2. Update analyze_chapter
old_analyze = """                sdt_flags = self.sdt_transformer.check(sentence)
                for flag in sdt_flags:
                    hits.append(Hit(chapter_name, p_idx, s_idx, "SDT违规", "主观情绪标签", sentence, 0, len(sentence), 1.0, "P2"))"""

new_analyze = """                sdt_flags = self.sdt_transformer.check(sentence)
                for flag in sdt_flags:
                    hits.append(Hit(chapter_name, p_idx, s_idx, "SDT违规", flag, sentence, 0, len(sentence), 1.0, "P2"))"""

text = text.replace(old_analyze, new_analyze)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Patch SDT successful.")
