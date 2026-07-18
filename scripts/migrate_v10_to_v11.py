# migrate_v10_to_v11.py -- run once, from the repo root.
import json, shutil, sys
import os
from src import schema as S
from src.grader import grade, GoldParseError

def migrate(path):
    backup = path + ".v10.bak"
    if os.path.exists(backup):
        print(f"  !! backup exists ({backup}) -- refusing to re-migrate {path}. "
              f"Delete the backup manually if you really mean to.", file=sys.stderr)
        return
    shutil.copy(path, backup)        
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        row["schema_version"] = S.SCHEMA_VERSION
        # drop v1.0 grades entirely -- they predate the status split
        for k in ("parsed_answer", "correct", "box_extraction_status",
                  "parse_ok", "all_boxed_matches", "grader_version", "graded_at"):
            row[k] = None
        try:
            g = grade(row["raw_output"], row["gold"])
        except GoldParseError as e:
            print(f"  !! GOLD BUG {row['row_id']}: {e}", file=sys.stderr)
            S.validate_row(row); out.append(row); continue   # leave ungraded
        row.update(g)
        row["grader_version"] = "grader_v2"
        row["graded_at"] = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()
        S.validate_row(row)                       # v1.1 rules, incl. the invariant
        out.append(row)
    with open(path, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(out)} rows -> v1.1")

if len(sys.argv) < 2:
    sys.exit("usage: python src/migrate_v10_to_v11.py <path.jsonl> [more.jsonl ...]")

for p in sys.argv[1:]:
    migrate(p)