"""
One-off helper: walk data/Transcripts/<topic>/ and emit a single markdown
file per topic that lists every file's summary, keywords, takeaways, and
file type. Used to seed the topic overview summaries in data/.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "data" / "Transcripts"
OUT_DIR = ROOT / "tools_scripts" / "_topic_dumps"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_topic(topic_dir: Path) -> str:
    lines = [f"# Topic: {topic_dir.name}", ""]
    files = sorted(topic_dir.glob("with summary_*.json"))
    lines.append(f"_File count: {len(files)}_\n")
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            lines.append(f"- [error reading {f.name}: {exc}]")
            continue
        s = data.get("summary") or {}
        src = data.get("source_file", f.name)
        stype = data.get("source_type", "?")
        chars = data.get("stats", {}).get("character_count", 0)
        lines.append(f"## {src}")
        lines.append(f"- type: `{stype}`  •  chars: {chars}")
        if s:
            lines.append(f"- topic: **{s.get('topic', '')}**")
            text = (s.get("text") or "").strip().replace("\n", " ")
            lines.append(f"- summary: {text}")
            kws = s.get("keywords") or []
            if kws:
                lines.append(f"- keywords: {', '.join(kws)}")
            takeaways = s.get("takeaways") or []
            if takeaways:
                lines.append("- takeaways:")
                for t in takeaways:
                    lines.append(f"  - {t}")
        lines.append("")
    return "\n".join(lines)


def main():
    if not TRANSCRIPTS.exists():
        raise SystemExit(f"Transcripts dir not found: {TRANSCRIPTS}")
    for topic in sorted(p for p in TRANSCRIPTS.iterdir() if p.is_dir()):
        out = OUT_DIR / f"{topic.name}.md"
        out.write_text(render_topic(topic), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
