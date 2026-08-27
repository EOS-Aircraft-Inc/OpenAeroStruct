"""Merge the coupled_toc_profile shard files and print the grid table."""
import json
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "out" / "logs"
res = []
for f in sorted(LOGS.glob("coupled_toc_profile_shard*.json")):
    res.extend(json.loads(f.read_text()))
res.sort(key=lambda r: (r["root_toc"], r["ratio"]))
(LOGS / "coupled_toc_profile.json").write_text(json.dumps(res, indent=2))

if not res:
    raise SystemExit("no shard files found")
base = max(res, key=lambda r: r["R_nmi"])
print("=" * 118)
print(f"{'root':>6} {'ratio':>6} {'t/c rt':>7} {'t/c tip':>8} {'t/c ail':>8} {'S_ref ft2':>10} "
      f"{'drag N':>9} {'W_wing':>9} {'m_batt':>9} {'R nmi':>8} {'vs best':>8}")
for r in res:
    print(f"{r['root_toc']:>6.3f} {r['ratio']:>6.2f} {r['toc_root']:>7.4f} "
          f"{r['toc_tip']:>8.4f} {r['toc_ail']:>8.4f} {r['S_ref']*10.7639104:>10.1f} "
          f"{r['drag_N']:>9.1f} {r['w_wing_lb']:>9.1f} {r['m_batt_lb']:>9.1f} "
          f"{r['R_nmi']:>8.1f} {100*(r['R_nmi']/base['R_nmi']-1):>+7.2f}%")
print("=" * 118)
print(f"BEST: root t/c {base['root_toc']:.3f}, tip/root {base['ratio']:.2f} "
      f"(tip {base['toc_tip']:.4f}) -> {base['R_nmi']:.1f} nmi, "
      f"W_wing {base['w_wing_lb']:.1f} lb, drag {base['drag_N']:.1f} N")
print(f"merged {len(res)} points into coupled_toc_profile.json")
