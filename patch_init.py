file_path = '/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI/duanpian_fixAI_1.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'self.evolution_engine = EvolutionEngine(self, self.db_path)' in line:
        continue
    if '    @staticmethod' in line and lines.index(line) > 650:
        new_lines.append('        self.evolution_engine = EvolutionEngine(self, self.db_path)\n\n')
        new_lines.append(line)
        pass # append it here
    else:
        new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Init patched.")
