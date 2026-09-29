import csv
from collections import defaultdict

groups = defaultdict(set)
details = defaultdict(list)

with open('UrbanSound8K/metadata/UrbanSound8K.csv', 'r', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

for r in rows:
    fsid = r['fsID']
    fold = int(r['fold'])
    groups[fsid].add(fold)
    if fsid in groups and len(groups[fsid]) > 1:
        details[fsid].append(r)
    elif fsid not in details:
        details[fsid] = [r]

print(f"Total rows: {len(rows)}")
print(f"Total unique fsIDs: {len(groups)}")

bad = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
print(f"fsIDs in multiple folds: {len(bad)}\n")

for fsid, folds in bad.items():
    print(f"fsID={fsid}  folds={folds}")
    for r in [row for row in rows if row['fsID'] == fsid]:
        print(f"  {r['slice_file_name']}  class={r['class']}  fold={r['fold']}")
    print()
