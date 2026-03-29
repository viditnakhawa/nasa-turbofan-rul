import json
import time

with open(r'c:\Users\Sunil\.vscode\NASA CMaps\notebooks\04_Deep_Learning_LSTM.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        if 'batch_size=256' in source:
            source = source.replace('batch_size=256', 'batch_size=1024')
        if 'batch_size=64' in source:
            source = source.replace('batch_size=64', 'batch_size=1024')
            
        if 'epochs=10' in source:
            source = source.replace('epochs=10', 'epochs=2')
        if 'epochs=30' in source:
            source = source.replace('epochs=30', 'epochs=2')
            
        if 'patience=3' in source:
            source = source.replace('patience=3', 'patience=1')
        if 'patience=10' in source:
            source = source.replace('patience=10', 'patience=1')
            
        cell['source'] = [line + '\n' for line in source.split('\n')]
        cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(r'c:\Users\Sunil\.vscode\NASA CMaps\notebooks\04_Deep_Learning_LSTM.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Super-fast DL Notebook params set.")
