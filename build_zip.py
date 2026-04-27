import zipfile                                                                                                                                                                                                                                                                                                                                        
from pathlib import Path                                  

with zipfile.ZipFile('bdd100k_data.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for folder in ['data_prepared', 'configs']:
        for p in Path(folder).rglob('*'):
            if p.is_file():
                zf.write(p, p.as_posix())