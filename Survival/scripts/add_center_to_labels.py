"""
Add a 'center' column to AHM_labels.csv by reading the center field from
clinical_data.xlsx.  Pure-stdlib xlsx parser (no openpyxl needed).

Usage:
    python add_center_to_labels.py \
        --xlsx /path/to/clinical_data.xlsx \
        --labels ./AHM/AHM_labels.csv \
        --case_col case_id
"""
import argparse, csv, re, zipfile, xml.etree.ElementTree as ET
from collections import Counter

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def read_xlsx(path):
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        r = ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in r:
            shared.append(''.join(t.text or '' for t in si.iter(NS + 't')))
    sheet = [n for n in z.namelist() if re.match(r'xl/worksheets/sheet\d+\.xml', n)][0]
    r = ET.fromstring(z.read(sheet))
    rows = list(r.find(NS + 'sheetData'))

    def cv(c):
        t = c.get('t'); v = c.find(NS + 'v')
        if v is None:
            isv = c.find(NS + 'is')
            return ''.join(tt.text or '' for tt in isv.iter(NS + 't')) if isv is not None else ''
        return shared[int(v.text)] if t == 's' else (v.text or '')

    def col(ref): return re.match(r'[A-Z]+', ref).group()
    hcells = {col(c.get('r')): cv(c) for c in rows[0]}
    n2l = {hcells[L]: L for L in hcells}
    recs = []
    for row in rows[1:]:
        d = {col(c.get('r')): cv(c) for c in row}
        recs.append({k: d.get(n2l[k], '') for k in n2l})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--case_col', default='case_id')
    args = ap.parse_args()

    recs = read_xlsx(args.xlsx)
    # build case_id -> center, map distinct centers to C1/C2/C3 by size
    raw = {str(r.get('case_id', '')).strip(): str(r.get('center', '')).strip()
           for r in recs if str(r.get('case_id', '')).strip()}
    ctr_count = Counter(raw.values())
    order = [c for c, _ in ctr_count.most_common()]
    cmap = {c: f'C{i+1}' for i, c in enumerate(order)}
    case2center = {cid: cmap.get(c, 'NA') for cid, c in raw.items()}
    print('center mapping (size):',
          {cmap[c]: ctr_count[c] for c in order})

    # read existing labels, add center, rewrite
    with open(args.labels, newline='') as f:
        rd = list(csv.DictReader(f))
    cols = list(rd[0].keys())
    if 'center' not in cols:
        cols = cols + ['center']
    n_match = 0
    for row in rd:
        cid = str(row[args.case_col]).strip()
        row['center'] = case2center.get(cid, 'NA')
        if row['center'] != 'NA':
            n_match += 1
    with open(args.labels, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rd)
    print(f"updated {args.labels}: {len(rd)} rows, {n_match} matched a center")
    cc = Counter(r['center'] for r in rd)
    print('per-center counts in labels:', dict(cc))


if __name__ == '__main__':
    main()
