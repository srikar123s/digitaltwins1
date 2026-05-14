import os

with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if 268 <= i <= 285:
        continue # skip 'import pickle' down to 'crs = cache["crs"]'
    if i == 286:
        continue # skip 'else:'
    if 287 <= i <= 433: # the main block
        if line.startswith('    '):
            new_lines.append(line[4:])
        else:
            new_lines.append(line)
        continue
    if 434 <= i <= 446:
        continue # skip saving the cache!
    new_lines.append(line)

with open('main.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
