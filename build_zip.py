import zipfile
from pathlib import Path

data_files = [p for p in Path('data_prepared').rglob('*') if p.is_file()]
config_files = [p for p in Path('configs').rglob('*') if p.is_file()]
total = len(data_files) + len(config_files)

print(f'Zipping {len(data_files)} data files and {len(config_files)} config files ({total} total)...')

with zipfile.ZipFile('bdd100k_data.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for i, p in enumerate(data_files, 1):
        zf.write(p, p.as_posix())
        if i % 1000 == 0 or i == len(data_files):
            print(f'  data_prepared: {i}/{len(data_files)}')
    for i, p in enumerate(config_files, 1):
        zf.write(p, 'data_prepared/' + p.as_posix())
    print(f'  configs: {len(config_files)}/{len(config_files)}')

print(f'Done. Output: bdd100k_data.zip')
