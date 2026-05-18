"""Rebuild data/processed_list.csv from all files in data/processed/."""
import pandas as pd
import glob
import os
import re

rows = []
for group in ['AD', 'NC', 'MCI']:
    for fp in sorted(glob.glob(f'data/processed/{group}/*.npy')):
        fname = os.path.basename(fp)
        stem = fname[:-4]
        m = re.match(r'^(.+)_(\d{5})$', stem)
        subject_id = m.group(1) if m else stem
        rows.append({
            'subject_id': subject_id,
            'group': group,
            'file_path': fp,
            'source_path': '',
        })

df = pd.DataFrame(rows)
df.to_csv('data/processed_list.csv', index=False)
print(f"Written {len(df)} rows")
print(df['group'].value_counts().to_string())
print("\nSample:")
print(df[df['group'] == 'AD'].head(3)[['subject_id', 'group', 'file_path']].to_string(index=False))
