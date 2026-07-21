"""
Rename every *.h5 in an h5_files directory to its 12-char TCGA case id + '.h5'
(e.g. TCGA-EP-A3RK-01Z-00-DX1.UUID.h5  ->  TCGA-EP-A3RK.h5).

On collision (several slides of the same patient map to the same case id),
keep exactly one file and resolve the rest according to --on_dup.

Safety: dry-run by default. Nothing is touched until you pass --apply.

Usage:
    # 1) preview the plan
    python scripts/rename_h5_to_caseid.py --dir /path/to/h5_files

    # 2) execute (extras left untouched)
    python scripts/rename_h5_to_caseid.py --dir /path/to/h5_files --apply

    # 3) execute and delete the duplicate slides
    python scripts/rename_h5_to_caseid.py --dir /path/to/h5_files --apply --on_dup delete
"""

import argparse
import os
import glob


def case_id(fname):
    """First 12 chars of the basename = TCGA-XX-XXXX."""
    return os.path.basename(fname)[:12]


def pick_keeper(files, target_name):
    """Choose which file to keep for a given case id.

    Priority:
      1. a file already named exactly target (no rename needed)
      2. the lexicographically smallest name (tends to pick DX1 before DX2)
    """
    for f in sorted(files):
        if os.path.basename(f) == target_name:
            return f
    return sorted(files)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='h5_files directory')
    ap.add_argument('--apply', action='store_true',
                    help='Actually perform the operations (default: dry-run).')
    ap.add_argument('--on_dup', choices=['skip', 'delete'], default='skip',
                    help="What to do with the extra files of a collided case id: "
                         "'skip' leaves them in place (default), "
                         "'delete' removes them.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, '*.h5')))
    if not files:
        raise SystemExit(f"No .h5 files found in {args.dir}")
    print(f"[scan] {len(files)} .h5 files in {args.dir}")

    # group by 12-char case id
    groups = {}
    for f in files:
        cid = case_id(f)
        groups.setdefault(cid, []).append(f)

    n_rename = n_already = n_dup = n_bad = 0
    plan = []           # (action, src, dst)

    for cid, flist in sorted(groups.items()):
        target = f"{cid}.h5"
        target_path = os.path.join(args.dir, target)

        if not cid.startswith('TCGA'):
            print(f"[warn] '{cid}' does not look like a TCGA id "
                  f"({len(flist)} file(s)) -- skipping this group")
            n_bad += len(flist)
            continue

        keeper = pick_keeper(flist, target)

        # the keeper: rename if needed
        if os.path.basename(keeper) == target:
            n_already += 1
        else:
            plan.append(('rename', keeper, target_path))
            n_rename += 1

        # the extras (duplicates)
        for f in flist:
            if f == keeper:
                continue
            n_dup += 1
            if args.on_dup == 'delete':
                plan.append(('delete', f, None))
            else:
                plan.append(('skip-dup', f, None))

    # ---- report ----
    print(f"[plan] unique case ids : {len(groups)}")
    print(f"       to rename       : {n_rename}")
    print(f"       already named   : {n_already}")
    print(f"       duplicates      : {n_dup}  (on_dup={args.on_dup})")
    if n_bad:
        print(f"       non-TCGA skipped: {n_bad}")
    print("-" * 60)

    # preview first 20 actions
    for action, src, dst in plan[:20]:
        if action == 'rename':
            print(f"  RENAME  {os.path.basename(src)}  ->  {os.path.basename(dst)}")
        elif action == 'delete':
            print(f"  DELETE  {os.path.basename(src)}")
        else:
            print(f"  KEEP-AS-IS (dup)  {os.path.basename(src)}")
    if len(plan) > 20:
        print(f"  ... and {len(plan)-20} more")
    print("-" * 60)

    if not args.apply:
        print("[dry-run] nothing changed. Re-run with --apply to execute.")
        return

    # ---- execute ----
    done_rename = done_delete = 0
    for action, src, dst in plan:
        try:
            if action == 'rename':
                # never overwrite an existing different file
                if os.path.exists(dst) and os.path.abspath(dst) != os.path.abspath(src):
                    print(f"[skip] target exists, not overwriting: {os.path.basename(dst)}")
                    continue
                os.rename(src, dst)
                done_rename += 1
            elif action == 'delete':
                os.remove(src)
                done_delete += 1
        except OSError as e:
            print(f"[err] {action} {src}: {e}")
    print(f"[done] renamed {done_rename}, deleted {done_delete}")


if __name__ == '__main__':
    main()
