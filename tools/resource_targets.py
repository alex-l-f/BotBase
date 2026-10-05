"""
Checks that a resource's delivery target actually exists before a tool
tells the frontend to show it.

A database row only says where a resource *should* be. A moved data file,
a renamed or re-exported SCORM package, or a stale lesson id would
otherwise reach the user as a "Not Found" page in the viewer (or, for a
lesson id Rise doesn't know, a silently different lesson) while the bot
believes the delivery worked. Paths resolve exactly as server.py serves
them.
"""

import base64
import json
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(_ROOT, "data")
SCORM_ROOT = _ROOT

_COURSE_URL = re.compile(
    r"^/?scorm/([^/]+)/scormcontent/index\.html#/lessons/([^/?#]+)$")

# index.html path -> (mtime, lesson ids); re-read when the package changes.
_lesson_ids_cache: dict[str, tuple[float, frozenset[str]]] = {}


def _rise_lesson_ids(pkg_dir: str) -> frozenset[str] | None:
    """Lesson ids in the Rise course JSON embedded in scormcontent/index.html."""
    index_path = os.path.join(pkg_dir, "scormcontent", "index.html")
    try:
        mtime = os.path.getmtime(index_path)
    except OSError:
        return None
    cached = _lesson_ids_cache.get(index_path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(index_path, encoding="utf-8") as f:
            m = re.search(r'deserialize\("([^"]+)"\)', f.read())
        if not m:
            return None
        course = json.loads(base64.b64decode(m.group(1))).get("course") or {}
    except (OSError, ValueError):
        return None
    ids = frozenset(
        lesson["id"] for lesson in course.get("lessons", [])
        if lesson.get("type") != "section" and lesson.get("id")
    )
    _lesson_ids_cache[index_path] = (mtime, ids)
    return ids


def course_page_problem(url: str) -> str | None:
    """Why *url* would not open its lesson in the viewer, or None if it would."""
    m = _COURSE_URL.match(url or "")
    if not m:
        return "its link is not a course-viewer link"
    package, lesson_id = m.groups()
    pkg_dir = os.path.join(SCORM_ROOT, package)
    if not os.path.isfile(os.path.join(pkg_dir, "imsmanifest.xml")):
        return f"its e-learning package ({package}) is not installed"
    lesson_ids = _rise_lesson_ids(pkg_dir)
    if lesson_ids is None:
        return "its e-learning package could not be read"
    if lesson_id not in lesson_ids:
        return "its lesson no longer exists in the e-learning course"
    return None


def file_problem(resource: dict) -> str | None:
    """Why *resource*'s file can't be served, or None if it can."""
    source_path = resource.get("source_path") or ""
    if not source_path:
        return "it has no file on the server"
    data_root = os.path.abspath(DATA_ROOT)
    abs_path = os.path.abspath(os.path.join(data_root, source_path))
    if not abs_path.startswith(data_root + os.sep) or not os.path.isfile(abs_path):
        return "its file is missing on the server"
    return None
