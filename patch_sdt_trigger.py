import re

file_path = '/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI/duanpian_fixAI_1.py'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

old_logic = """                should_rewrite = False
                if any(h.severity == "P0" for h in s_hits): should_rewrite = True
                if any(h.rule_name in [r.name for r in self.tier1_rules] for h in s_hits): should_rewrite = True"""

new_logic = """                should_rewrite = False
                if self.sdt_transformer.check(sentence): should_rewrite = True
                if any(h.severity == "P0" for h in s_hits): should_rewrite = True
                if any(h.rule_name in [r.name for r in self.tier1_rules] for h in s_hits): should_rewrite = True"""

text = text.replace(old_logic, new_logic)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Added SDT trigger to should_rewrite.")
