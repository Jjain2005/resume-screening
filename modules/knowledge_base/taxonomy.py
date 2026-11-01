import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from .models import SkillEntry, SkillRelationship, RelationshipType

def normalize_text(text: str) -> str:
    """Normalize text for consistent lookup: lowercase, strip punctuation variations."""
    if not text:
        return ""
    text = text.lower().strip()
    # Normalize common symbols like c++ / c# / .net / node.js while cleaning extra punctuation
    text = re.sub(r'[\s\-_]+', ' ', text)
    return text.strip()

class SkillTaxonomy:
    """
    Structured skill taxonomy manager providing canonicalization, alias resolution,
    relationship tracking, and unknown skill logging.
    """
    def __init__(self, json_path: Optional[str] = None):
        self.json_path = json_path
        self.categories: List[str] = []
        self.skills: Dict[str, SkillEntry] = {}  # canonical_name -> SkillEntry
        self._lookup_map: Dict[str, str] = {}    # normalized string -> canonical_name
        self._alias_to_canonical: Dict[str, str] = {} # normalized alias -> canonical_name
        self._relationships_map: Dict[Tuple[str, str], SkillRelationship] = {}
        self.unknown_skills: Set[str] = set()

        if json_path and os.path.exists(json_path):
            self.load(json_path)

    def load(self, json_path: str):
        self.json_path = json_path
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.categories = data.get("categories", [])
        self.skills.clear()
        self._lookup_map.clear()
        self._alias_to_canonical.clear()
        self._relationships_map.clear()

        for s_data in data.get("skills", []):
            entry = SkillEntry.from_dict(s_data)
            self.add_skill(entry)

    def add_skill(self, entry: SkillEntry):
        canon_name = entry.canonical_name
        self.skills[canon_name] = entry
        norm_canon = normalize_text(canon_name)
        self._lookup_map[norm_canon] = canon_name

        for alias in entry.aliases:
            norm_alias = normalize_text(alias)
            self._lookup_map[norm_alias] = canon_name
            self._alias_to_canonical[norm_alias] = canon_name

        for rel in entry.relationships:
            self._relationships_map[(canon_name.lower(), rel.target.lower())] = rel

    def get_skill(self, name_or_alias: str) -> Optional[SkillEntry]:
        canonical = self.resolve_canonical(name_or_alias)
        if canonical:
            return self.skills.get(canonical)
        return None

    def resolve_canonical(self, text: str) -> Optional[str]:
        norm = normalize_text(text)
        return self._lookup_map.get(norm)

    def is_alias_match(self, query: str, canonical_target: str) -> bool:
        """Checks if query is a registered alias for canonical_target."""
        resolved = self.resolve_canonical(query)
        if resolved and resolved.lower() == canonical_target.lower():
            # Check if query itself is an alias (and not the exact canonical name)
            return normalize_text(query) in self._alias_to_canonical
        return False

    def get_relationship(self, skill_a: str, skill_b: str) -> Optional[SkillRelationship]:
        """Returns direct or reciprocal relationship between two skills."""
        canon_a = self.resolve_canonical(skill_a) or skill_a
        canon_b = self.resolve_canonical(skill_b) or skill_b

        key1 = (canon_a.lower(), canon_b.lower())
        if key1 in self._relationships_map:
            return self._relationships_map[key1]

        key2 = (canon_b.lower(), canon_a.lower())
        if key2 in self._relationships_map:
            rel = self._relationships_map[key2]
            # Invert hierarchical relationships
            if rel.type == RelationshipType.PARENT_OF:
                return SkillRelationship(target=canon_a, type=RelationshipType.SUBFIELD_OF)
            elif rel.type == RelationshipType.SUBFIELD_OF:
                return SkillRelationship(target=canon_a, type=RelationshipType.PARENT_OF)
            return rel

        return None

    def are_related_not_equivalent(self, skill_a: str, skill_b: str) -> bool:
        """
        Check if two skills are explicitly marked as related but distinct (e.g., React vs Angular,
        FastAPI vs Flask, MySQL vs PostgreSQL).
        """
        rel = self.get_relationship(skill_a, skill_b)
        if rel and rel.type in (
            RelationshipType.RELATED_NOT_EQUIVALENT,
            RelationshipType.RELATED,
            RelationshipType.SUBFIELD_OF,
            RelationshipType.PARENT_OF,
            RelationshipType.COMPLEMENTARY
        ):
            return True
        return False

    def log_unknown_skill(self, skill_candidate: str):
        cleaned = skill_candidate.strip()
        if cleaned and not self.resolve_canonical(cleaned):
            self.unknown_skills.add(cleaned)

    def get_all_canonical_names(self) -> List[str]:
        return list(self.skills.keys())

    def get_all_lookup_terms(self) -> Dict[str, str]:
        """Returns map of normalized search term -> canonical skill name."""
        return dict(self._lookup_map)

    def save(self, output_path: Optional[str] = None):
        target_path = output_path or self.json_path
        if not target_path:
            raise ValueError("No output path specified to save taxonomy.")
        
        data = {
            "version": "1.0.0",
            "categories": self.categories,
            "skills": [s.to_dict() for s in self.skills.values()]
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
