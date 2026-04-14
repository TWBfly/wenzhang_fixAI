file_path = '/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI/duanpian_fixAI_1.py'
import re

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# remove the one at the end that we added
text = text.replace('        self.evolution_engine = EvolutionEngine(self, self.db_path)\n\n    @staticmethod', '    @staticmethod')

# insert it before the loop
target_loop = '        for r_dict in self.evolution_engine.knowledge["rules"]:'
new_loop = '        self.evolution_engine = EvolutionEngine(self, self.db_path)\n' + target_loop

text = text.replace(target_loop, new_loop)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed init again.")
