"""
Perspective System for HyperChat (Design 111)

Perspectives are offset vectors that shift query positions in φ-space,
plus style transformations that modify response output.

The perspective encodes WHO is asking/answering, while the query
encodes WHAT is being asked. Together they determine WHERE in the
space we look and HOW we express the answer.

Usage:
    from truthspace_lcm.core.perspective import PERSPECTIVES, apply_perspective
    
    # Set personality
    perspective = PERSPECTIVES['warhammer40k']
    
    # Transform response
    styled_response = perspective.transform_response(base_response)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re


PHI = (1 + np.sqrt(5)) / 2


@dataclass
class Perspective:
    """
    A perspective combines:
    1. An offset vector in phi-space (shifts query position)
    2. A style transformation (modifies response text)
    
    The offset affects WHAT knowledge is retrieved.
    The style affects HOW the knowledge is expressed.
    """
    name: str
    description: str
    offset: np.ndarray
    style_rules: Dict[str, str] = field(default_factory=dict)
    prefix: str = ""
    suffix: str = ""
    
    def apply_offset(self, query_position: np.ndarray) -> np.ndarray:
        """Apply perspective offset to query position."""
        if len(query_position) < len(self.offset):
            padded = np.zeros(len(self.offset))
            padded[:len(query_position)] = query_position
            return padded + self.offset
        elif len(query_position) > len(self.offset):
            padded_offset = np.zeros(len(query_position))
            padded_offset[:len(self.offset)] = self.offset
            return query_position + padded_offset
        return query_position + self.offset
    
    def transform_response(self, response: str) -> str:
        """Transform response text according to perspective style."""
        result = response
        
        # Apply word replacements (case-insensitive, preserve boundaries)
        for original, replacement in self.style_rules.items():
            pattern = re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE)
            result = pattern.sub(replacement, result)
        
        # Add prefix if specified
        if self.prefix:
            result = f"{self.prefix}\n\n{result}"
        
        # Add suffix if specified
        if self.suffix:
            result = f"{result}\n\n{self.suffix}"
        
        return result


# =============================================================================
# PREDEFINED PERSPECTIVES
# =============================================================================

DEFAULT_PERSPECTIVE = Perspective(
    name="default",
    description="Standard HyperChat assistant - helpful, clear, technical",
    offset=np.array([2, 1, 0, 0, 0, 0]),
    style_rules={},
    prefix="",
    suffix="",
)

WARHAMMER_40K_PERSPECTIVE = Perspective(
    name="warhammer40k",
    description="Grimdark Warhammer 40,000 narrator - dramatic, gothic, zealous",
    offset=np.array([0, 2, 0, 2, 0, 0]),
    style_rules={
        "programming language": "sacred machine tongue",
        "programming": "sacred rites of the Machine God",
        "code": "holy scripture",
        "computer": "cogitator",
        "software": "machine spirit",
        "hardware": "blessed iron",
        "data": "sacred data-hymns",
        "algorithm": "divine computation",
        "function": "sacred ritual",
        "variable": "mutable essence",
        "memory": "cogitator memory banks",
        "processor": "logic engine",
        "server": "data-shrine",
        "network": "noospheric link",
        "internet": "great noosphere",
        "database": "data-vault",
        "file": "data-scroll",
        "folder": "data-reliquary",
        "user": "supplicant",
        "developer": "tech-adept",
        "programmer": "code-priest",
        "engineer": "enginseer",
        "scientist": "magos",
        "error": "machine spirit's displeasure",
        "bug": "corruption of the machine spirit",
        "debug": "perform the Rite of Cleansing",
        "install": "perform the Rite of Installation",
        "update": "apply sacred patches",
        "download": "invoke data-transfer rites",
        "upload": "offer data unto the machine spirit",
        "execute": "invoke",
        "run": "awaken",
        "compile": "sanctify",
        "syntax": "sacred grammar",
        "library": "tome of ancient wisdom",
        "framework": "blessed framework of the Omnissiah",
        "Python": "the Serpent Tongue of the Omnissiah",
        "JavaScript": "the Scribing Language of the Web-Spirits",
        "Java": "the Ancient Tongue of Enterprise",
        "simple": "elegantly wrought",
        "easy": "blessed with clarity",
        "difficult": "requiring great devotion",
        "complex": "labyrinthine in its sacred complexity",
        "powerful": "mighty in the sight of the Omnissiah",
        "efficient": "pleasing to the machine spirits",
        "popular": "favored by the Adeptus Mechanicus",
        "modern": "of recent revelation",
        "created": "forged",
        "designed": "wrought by the ancients",
        "developed": "brought forth",
        "used": "employed in sacred service",
        "learn": "receive the sacred knowledge",
        "understand": "comprehend the mysteries",
        "know": "possess the sacred lore",
        "work": "function according to the Omnissiah's will",
        "help": "serve",
        "good": "blessed",
        "great": "glorious",
        "best": "most sacred",
        "important": "of paramount significance to the Imperium",
        "useful": "of great utility to the faithful",
        "feature": "blessed capability",
        "tool": "sacred instrument",
        "system": "holy system",
        "process": "sacred process",
        "method": "rite",
        "technique": "sacred technique",
        "solution": "blessed resolution",
        "problem": "heretical obstruction",
        "issue": "vexation of the machine spirit",
        "challenge": "trial set by the Omnissiah",
        "readability": "clarity pleasing to the machine spirits",
        "indentation": "sacred spacing",
        "paradigms": "holy doctrines",
        "procedural": "of the ancient rites",
        "object-oriented": "of the blessed object-forms",
        "functional": "of the pure function-prayers",
    },
    prefix="*In the grim darkness of the far future, there is only code...*\n\nHearken, supplicant, to the sacred knowledge:",
    suffix="*The Omnissiah protects. The Machine God provides.*",
)

PIRATE_PERSPECTIVE = Perspective(
    name="pirate",
    description="Swashbuckling pirate captain - nautical, adventurous, colorful",
    offset=np.array([0, 0, 0, -2, 0, 0]),
    style_rules={
        "you": "ye",
        "your": "yer",
        "my": "me",
        "hello": "ahoy",
        "yes": "aye",
        "no": "nay",
        "friend": "matey",
        "money": "doubloons",
        "good": "fine",
        "great": "mighty fine",
        "computer": "magic box",
        "programming": "code-sailin'",
        "code": "treasure map",
        "data": "booty",
        "error": "scurvy mistake",
        "bug": "barnacle",
        "understand": "savvy",
        "work": "sail",
        "help": "lend a hand",
        "problem": "rough waters",
        "solution": "safe harbor",
    },
    prefix="Ahoy there, matey!",
    suffix="Now ye know, savvy? Fair winds to ye!",
)

SHAKESPEARE_PERSPECTIVE = Perspective(
    name="shakespeare",
    description="Elizabethan playwright - poetic, dramatic, archaic",
    offset=np.array([0, 2, 0, 2, 0, 0]),
    style_rules={
        "you": "thou",
        "your": "thy",
        "yours": "thine",
        "have": "hast",
        "has": "hath",
        "do": "dost",
        "does": "doth",
        "will": "shall",
        "would": "wouldst",
        "can": "canst",
        "see": "perceive",
        "understand": "comprehend",
        "think": "ponder",
        "say": "speak",
        "tell": "impart",
        "ask": "beseech",
        "want": "desire",
        "need": "require",
        "use": "employ",
        "make": "fashion",
        "create": "bring forth",
        "good": "most excellent",
        "great": "wondrous",
        "bad": "most foul",
        "important": "of great import",
        "simple": "plain and true",
        "complex": "most intricate",
        "computer": "thinking engine",
        "programming": "the art of instruction",
        "very": "most",
        "really": "verily",
        "now": "anon",
        "here": "hither",
        "there": "thither",
    },
    prefix="Hark! Attend well to these words of wisdom:",
    suffix="*Exeunt*",
)

# Registry of all perspectives
PERSPECTIVES: Dict[str, Perspective] = {
    "default": DEFAULT_PERSPECTIVE,
    "warhammer40k": WARHAMMER_40K_PERSPECTIVE,
    "wh40k": WARHAMMER_40K_PERSPECTIVE,
    "grimdark": WARHAMMER_40K_PERSPECTIVE,
    "pirate": PIRATE_PERSPECTIVE,
    "shakespeare": SHAKESPEARE_PERSPECTIVE,
    "bard": SHAKESPEARE_PERSPECTIVE,
}


def get_perspective(name: str) -> Perspective:
    """Get a perspective by name, defaulting to DEFAULT if not found."""
    return PERSPECTIVES.get(name.lower(), DEFAULT_PERSPECTIVE)


def list_perspectives() -> List[str]:
    """List all available perspective names."""
    seen = set()
    result = []
    for name, persp in PERSPECTIVES.items():
        if persp.name not in seen:
            result.append(f"- **{persp.name}**: {persp.description}")
            seen.add(persp.name)
    return result
