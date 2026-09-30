#!/usr/bin/env python3
"""Build the public Markdown catalog from reviewed, editable skill JSON files."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
FIELDS = (
    "name", "version", "scope", "authority_basis", "description",
    "use_when", "do_not_use_when", "inputs", "procedure",
    "output_contract", "failure", "validation",
)
SECTIONS = (
    ("Use When", "use_when"),
    ("Do Not Use When", "do_not_use_when"),
    ("Inputs", "inputs"),
    ("Procedure", "procedure"),
    ("Output Contract", "output_contract"),
    ("Failure", "failure"),
    ("Validation", "validation"),
)


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate(skill, folder):
    assert isinstance(skill, dict) and set(skill) == set(FIELDS), folder
    assert re.fullmatch(r"[a-z][a-z0-9-]{1,80}", skill["name"]), folder
    assert skill["name"] == folder.name, folder
    assert skill["scope"] in {
        "general", "agent", "coding", "math", "reading", "science", "writing"
    }, folder
    assert skill["authority_basis"] == "authored_task_contract", folder
    for field in FIELDS:
        assert isinstance(skill[field], str) and skill[field].strip(), (folder, field)
    for field in FIELDS:
        assert "\x00" not in skill[field], (folder, field)


def markdown(skill):
    front = ["---"]
    for key in ("name", "version", "scope", "description", "authority_basis"):
        front.append(f"{key}: {json.dumps(skill[key], ensure_ascii=False)}")
    front.append("---")
    body = ["\n".join(front), ""]
    for title, key in SECTIONS:
        body.extend((f"## {title}", "", skill[key].strip(), ""))
    return ("\n".join(body).rstrip() + "\n").encode("utf-8")


def put(path, data, check):
    if check:
        assert path.is_file() and path.read_bytes() == data, f"OUT_OF_DATE: {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def build(check):
    assert SKILLS.is_dir(), "MISSING_SKILLS"
    folders = sorted(p for p in SKILLS.iterdir() if p.is_dir())
    assert folders, "EMPTY_LIBRARY"
    catalog = []
    entries = []
    scopes = Counter()
    for folder in folders:
        source = folder / "skill.json"
        assert source.is_file(), f"MISSING_SOURCE: {folder}"
        skill = json.loads(source.read_text(encoding="utf-8"))
        validate(skill, folder)
        assert source.read_bytes() == encoded(skill), f"NONCANONICAL_JSON: {source}"
        md = markdown(skill)
        put(folder / "SKILL.md", md, check)
        catalog.append(skill)
        scopes[skill["scope"]] += 1
        entries.append({
            "name": skill["name"], "version": skill["version"],
            "scope": skill["scope"],
            "json_path": f"skills/{skill['name']}/skill.json",
            "markdown_path": f"skills/{skill['name']}/SKILL.md",
            "json_sha256": sha(source.read_bytes()),
            "markdown_sha256": sha(md),
        })
    assert len({s["name"] for s in catalog}) == len(catalog)
    public_catalog = {"schema_version": "1", "skills": catalog}
    cat_bytes = encoded(public_catalog)
    put(ROOT / "catalog.json", cat_bytes, check)
    manifest = {
        "schema_version": "1", "skill_count": len(catalog),
        "scope_counts": dict(sorted(scopes.items())),
        "catalog_sha256": sha(cat_bytes), "skills": entries,
    }
    put(ROOT / "manifest.json", encoded(manifest), check)
    print("SKILL_LIBRARY_VALID", len(catalog), "CHECK" if check else "BUILT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    build(parser.parse_args().check)
