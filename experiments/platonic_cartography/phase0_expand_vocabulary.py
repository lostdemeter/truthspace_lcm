#!/usr/bin/env python3
"""
Platonic Cartography — Phase 0: Expand Concept Vocabulary
==========================================================

Goal: Go from 88 hand-picked concepts to 500+ clean, diverse, single-token
concepts spanning 20+ semantic categories.

This is the foundation for automated platonic ideal discovery. We need enough
concepts to discover ~79 independent truth axes.

Approach:
1. Define candidate words across many semantic categories
2. Filter to single-token words in the model's vocabulary
3. Quality-check: reasonable embedding norms, no degenerate tokens
4. Validate: the 6 known truth axes must still work with the expanded set
5. Save the curated concept vocabulary for Phase 1
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "experiments" / "model_reverse_engineering_v2" / "phi_model"
sys.path.insert(0, str(PROJECT_ROOT))

D_MODEL = 3584
NOTES_PATH = SCRIPT_DIR / "phase0_notes.md"


# =============================================================================
# FIELD NOTES
# =============================================================================

class FieldNotes:
    """Darwin-style field notes — append-only, markdown formatted."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "w")
        self.f.write("# Platonic Cartography Phase 0 — Concept Vocabulary Expansion\n")
        self.f.write("*Building the foundation for full LLM mapping*\n\n")
        self.f.flush()

    def section(self, title):
        self.f.write(f"\n## {title}\n\n")
        self.f.flush()
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")

    def observe(self, text):
        self.f.write(f"{text}\n")
        self.f.flush()
        print(text)

    def data_table(self, headers, rows, title=""):
        if title:
            self.f.write(f"\n**{title}**\n\n")
            print(f"\n  {title}")
        fmt = " | ".join(f"{{:>{max(12, len(h))}}}" for h in headers)
        sep = " | ".join("-" * max(12, len(h)) for h in headers)
        hdr = fmt.format(*headers)
        self.f.write(f"| {hdr} |\n| {sep} |\n")
        print(f"  | {hdr} |")
        for row in rows:
            line = fmt.format(*[str(x) for x in row])
            self.f.write(f"| {line} |\n")
            print(f"  | {line} |")
        self.f.write("\n")
        self.f.flush()

    def finding(self, text):
        self.f.write(f"\n> **FINDING:** {text}\n\n")
        self.f.flush()
        print(f"\n  >>> FINDING: {text}\n")

    def close(self):
        self.f.close()


@dataclass
class Concept:
    name: str
    token_id: int
    token_str: str
    embedding: np.ndarray
    categories: List[str] = field(default_factory=list)


# =============================================================================
# CANDIDATE WORD LISTS — organized by semantic category
# =============================================================================

CONCEPT_CATEGORIES = {
    # ---- Geography ----
    "country_europe": [
        "France", "Germany", "Poland", "Norway", "Sweden", "Italy", "Portugal",
        "Spain", "Greece", "Ireland", "Finland", "Denmark", "Austria", "Belgium",
        "Netherlands", "Switzerland", "Russia", "Romania", "Hungary", "Czech",
        "Croatia", "Serbia", "Bulgaria", "Slovakia", "Lithuania", "Latvia",
        "Estonia", "Slovenia", "Iceland", "Luxembourg", "Malta", "Cyprus",
        "Albania", "Ukraine", "Belarus", "Moldova", "Georgia", "Armenia",
    ],
    "country_asia": [
        "Japan", "China", "Thailand", "India", "Korea", "Vietnam", "Indonesia",
        "Philippines", "Malaysia", "Singapore", "Iran", "Iraq", "Israel",
        "Pakistan", "Bangladesh", "Myanmar", "Cambodia", "Nepal", "Laos",
        "Mongolia", "Taiwan", "Bhutan", "Kuwait", "Qatar", "Oman", "Yemen",
        "Jordan", "Lebanon", "Syria", "Afghanistan",
    ],
    "country_africa": [
        "Egypt", "Nigeria", "Kenya", "Morocco", "Ghana", "Ethiopia", "Tanzania",
        "Uganda", "Senegal", "Mali", "Niger", "Chad", "Sudan", "Libya",
        "Tunisia", "Algeria", "Somalia", "Rwanda", "Cameroon", "Congo",
        "Zimbabwe", "Zambia", "Mozambique", "Madagascar", "Namibia", "Botswana",
    ],
    "country_americas": [
        "Brazil", "Mexico", "Canada", "Argentina", "Chile", "Colombia", "Peru",
        "Venezuela", "Cuba", "Jamaica", "Panama", "Ecuador", "Bolivia",
        "Paraguay", "Uruguay", "Guatemala", "Honduras", "Nicaragua",
        "Haiti", "Trinidad", "Barbados", "Bahamas", "Guyana", "Suriname",
    ],
    "country_oceania": [
        "Australia", "Zealand", "Fiji", "Samoa", "Tonga", "Papua",
    ],
    "capital_city": [
        "Paris", "Berlin", "Tokyo", "Beijing", "Cairo", "Canberra", "Bangkok",
        "Warsaw", "Oslo", "Stockholm", "Delhi", "Seoul", "Rome", "Lisbon",
        "Moscow", "Madrid", "Athens", "Ankara", "Dublin", "Helsinki",
        "Copenhagen", "Vienna", "Brussels", "Amsterdam", "Ottawa", "Lima",
        "Tehran", "Baghdad", "Hanoi", "Jakarta", "Manila", "Havana",
        "Nairobi", "Accra", "Dakar", "Tunis", "Algiers", "Tripoli",
        "Riyadh", "Doha", "Kabul", "Beirut", "Damascus", "Amman",
        "Bucharest", "Budapest", "Prague", "Zagreb", "Belgrade", "Sofia",
        "Bratislava", "Tallinn", "Riga", "Vilnius", "Ljubljana", "Minsk",
        "Tbilisi", "Yerevan", "Baku", "Taipei",
    ],
    "language": [
        "French", "German", "Japanese", "Chinese", "Spanish", "Italian",
        "Portuguese", "Russian", "Arabic", "English", "Korean", "Thai",
        "Polish", "Norwegian", "Swedish", "Dutch", "Greek", "Turkish",
        "Hindi", "Finnish", "Danish", "Irish", "Persian", "Hebrew",
        "Vietnamese", "Indonesian", "Filipino", "Malay", "Romanian",
        "Hungarian", "Czech", "Croatian", "Serbian", "Bulgarian", "Slovak",
        "Lithuanian", "Latvian", "Estonian", "Slovenian", "Icelandic",
        "Albanian", "Ukrainian", "Georgian", "Armenian", "Swahili",
        "Bengali", "Urdu", "Tamil", "Nepali", "Mongolian", "Tibetan",
        "Burmese", "Cambodian", "Somali", "Kurdish", "Pashto",
    ],

    # ---- Gender pairs ----
    "gender_male": [
        "king", "man", "boy", "father", "brother", "son", "husband",
        "uncle", "prince", "actor", "waiter", "hero", "monk", "wizard",
        "gentleman", "lord", "duke", "emperor", "bachelor", "groom",
        "nephew", "grandson", "god", "sir", "lad",
    ],
    "gender_female": [
        "queen", "woman", "girl", "mother", "sister", "daughter", "wife",
        "aunt", "princess", "actress", "waitress", "heroine", "nun", "witch",
        "lady", "duchess", "empress", "bride",
        "niece", "granddaughter", "goddess", "madam", "lass",
    ],

    # ---- Animals ----
    "animal_mammal": [
        "dog", "cat", "lion", "tiger", "bear", "wolf", "fox", "deer",
        "horse", "cow", "pig", "sheep", "goat", "elephant", "monkey",
        "whale", "dolphin", "bat", "rabbit", "mouse", "rat", "otter",
        "beaver", "squirrel", "hedgehog", "camel", "donkey", "zebra",
        "giraffe", "hippo", "rhino", "leopard", "jaguar", "panther",
        "buffalo", "moose", "elk", "antelope", "gorilla", "chimpanzee",
    ],
    "animal_bird": [
        "eagle", "hawk", "falcon", "owl", "crow", "raven", "sparrow",
        "robin", "pigeon", "dove", "swan", "duck", "goose", "penguin",
        "parrot", "flamingo", "heron", "pelican", "stork", "vulture",
        "cardinal", "magpie", "peacock", "rooster", "hen", "turkey",
        "ostrich", "condor", "albatross", "kingfisher",
    ],
    "animal_reptile_amphibian": [
        "snake", "lizard", "turtle", "crocodile", "alligator", "gecko",
        "iguana", "chameleon", "cobra", "python", "viper", "frog", "toad",
        "salamander", "newt", "tortoise", "dragon", "dinosaur",
    ],
    "animal_fish_sea": [
        "fish", "shark", "salmon", "tuna", "trout", "cod", "bass",
        "herring", "sardine", "anchovy", "swordfish", "octopus", "squid",
        "jellyfish", "starfish", "lobster", "crab", "shrimp", "oyster",
        "clam", "mussel", "coral", "seahorse", "eel", "ray",
    ],
    "animal_insect": [
        "ant", "bee", "wasp", "fly", "mosquito", "butterfly", "moth",
        "beetle", "spider", "scorpion", "cricket", "grasshopper",
        "dragonfly", "ladybug", "firefly", "centipede", "tick", "flea",
        "caterpillar", "cockroach", "termite", "mantis", "cicada",
    ],

    # ---- Colors ----
    "color": [
        "red", "blue", "green", "yellow", "orange", "purple", "pink",
        "brown", "black", "white", "gray", "grey", "gold", "silver",
        "bronze", "crimson", "scarlet", "maroon", "navy", "cyan",
        "magenta", "violet", "indigo", "turquoise", "beige", "ivory",
        "tan", "coral", "salmon", "amber", "teal", "lavender",
    ],

    # ---- Professions ----
    "profession": [
        "doctor", "nurse", "teacher", "lawyer", "judge", "soldier",
        "engineer", "scientist", "artist", "musician", "writer", "poet",
        "chef", "baker", "farmer", "pilot", "sailor", "captain",
        "priest", "bishop", "mayor", "senator", "president", "king",
        "detective", "spy", "thief", "pirate", "knight", "warrior",
        "merchant", "banker", "accountant", "architect", "surgeon",
        "dentist", "pharmacist", "librarian", "professor", "student",
        "monk", "nun", "shepherd", "hunter", "fisherman", "miner",
        "carpenter", "plumber", "electrician", "mechanic", "painter",
        "sculptor", "dancer", "singer", "comedian", "magician",
        "astronaut", "athlete", "coach", "referee",
    ],

    # ---- Food & Drink ----
    "food": [
        "bread", "rice", "pasta", "pizza", "cheese", "butter", "milk",
        "egg", "meat", "chicken", "beef", "pork", "lamb", "fish",
        "soup", "salad", "cake", "pie", "cookie", "chocolate",
        "sugar", "salt", "pepper", "honey", "vinegar", "oil",
        "wine", "beer", "coffee", "tea", "juice", "water",
        "apple", "banana", "grape", "lemon", "cherry", "peach",
        "plum", "pear", "mango", "melon", "coconut", "strawberry",
        "tomato", "potato", "onion", "garlic", "carrot", "corn",
        "bean", "pea", "nut", "wheat", "oat", "barley",
    ],

    # ---- Materials & Elements ----
    "material": [
        "wood", "metal", "glass", "stone", "brick", "concrete", "clay",
        "sand", "dirt", "dust", "mud", "ice", "snow", "crystal",
        "diamond", "ruby", "emerald", "sapphire", "pearl", "jade",
        "iron", "steel", "copper", "bronze", "brass", "tin", "lead",
        "zinc", "aluminum", "titanium", "platinum", "uranium",
        "carbon", "silicon", "nitrogen", "oxygen", "hydrogen", "helium",
        "gold", "silver", "mercury", "sulfur", "phosphorus", "neon",
    ],

    # ---- Body parts ----
    "body": [
        "head", "face", "eye", "ear", "nose", "mouth", "lip", "tongue",
        "tooth", "jaw", "chin", "cheek", "forehead", "neck", "throat",
        "shoulder", "arm", "elbow", "wrist", "hand", "finger", "thumb",
        "chest", "back", "spine", "rib", "hip", "waist", "belly",
        "leg", "knee", "ankle", "foot", "toe", "heel",
        "heart", "lung", "brain", "liver", "kidney", "stomach",
        "bone", "muscle", "skin", "blood", "vein", "nerve",
    ],

    # ---- Emotions & States ----
    "emotion": [
        "happy", "sad", "angry", "afraid", "brave", "calm", "nervous",
        "excited", "bored", "tired", "lonely", "proud", "ashamed",
        "jealous", "grateful", "hopeful", "desperate", "confused",
        "curious", "surprised", "disgusted", "anxious", "peaceful",
        "cheerful", "gloomy", "furious", "terrified", "delighted",
        "miserable", "content", "restless", "eager", "reluctant",
    ],

    # ---- Nature & Weather ----
    "nature": [
        "sun", "moon", "star", "sky", "cloud", "rain", "snow", "wind",
        "storm", "thunder", "lightning", "fog", "mist", "hail", "frost",
        "fire", "flame", "smoke", "ash", "lava", "earthquake",
        "river", "lake", "ocean", "sea", "pond", "stream", "waterfall",
        "mountain", "hill", "valley", "cliff", "cave", "desert", "forest",
        "jungle", "meadow", "swamp", "marsh", "island", "volcano",
        "glacier", "canyon", "plateau", "reef", "tide", "wave",
    ],

    # ---- Time ----
    "time": [
        "morning", "noon", "afternoon", "evening", "night", "midnight",
        "dawn", "dusk", "sunrise", "sunset",
        "spring", "summer", "autumn", "winter",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "yesterday", "today", "tomorrow", "forever", "never", "always",
    ],

    # ---- Abstract concepts ----
    "abstract": [
        "truth", "justice", "freedom", "peace", "war", "love", "hate",
        "hope", "fear", "faith", "doubt", "wisdom", "folly", "beauty",
        "ugly", "good", "evil", "life", "death", "soul", "spirit",
        "mind", "body", "power", "weakness", "courage", "honor",
        "glory", "shame", "mercy", "revenge", "fate", "destiny",
        "chaos", "order", "silence", "noise", "darkness", "light",
        "heaven", "hell", "dream", "nightmare", "miracle", "curse",
        "luck", "fortune", "risk", "danger", "safety", "harmony",
    ],

    # ---- Numbers ----
    "number": [
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "twenty",
        "thirty", "forty", "fifty", "hundred", "thousand", "million",
        "billion", "dozen", "pair", "triple", "half", "quarter", "third",
    ],

    # ---- Sizes & Physical Properties ----
    "physical_property": [
        "big", "small", "tall", "short", "long", "wide", "narrow",
        "thick", "thin", "heavy", "light", "fast", "slow", "hard",
        "soft", "hot", "cold", "warm", "cool", "wet", "dry",
        "sharp", "dull", "smooth", "rough", "loud", "quiet",
        "bright", "dark", "deep", "shallow", "steep", "flat",
        "round", "square", "straight", "curved", "hollow", "solid",
    ],

    # ---- Kinship & Social ----
    "social_role": [
        "friend", "enemy", "stranger", "neighbor", "guest", "host",
        "master", "servant", "slave", "leader", "follower", "rebel",
        "citizen", "immigrant", "refugee", "prisoner", "guard",
        "child", "adult", "elder", "baby", "infant", "teenager",
        "orphan", "widow", "veteran", "rookie", "apprentice",
    ],

    # ---- Buildings & Places ----
    "place": [
        "house", "castle", "church", "temple", "mosque", "palace",
        "prison", "hospital", "school", "library", "museum", "theater",
        "market", "shop", "factory", "farm", "garden", "park",
        "bridge", "tower", "wall", "gate", "door", "window",
        "kitchen", "bedroom", "bathroom", "cellar", "attic", "roof",
        "village", "town", "city", "harbor", "port", "airport",
        "cemetery", "monument", "statue", "fountain", "arena", "stadium",
    ],

    # ---- Vehicles & Transport ----
    "vehicle": [
        "car", "truck", "bus", "train", "ship", "boat", "airplane",
        "helicopter", "bicycle", "motorcycle", "submarine", "rocket",
        "wagon", "carriage", "canoe", "yacht", "ferry", "taxi",
        "ambulance", "tank", "chariot", "sled",
    ],

    # ---- Clothing ----
    "clothing": [
        "shirt", "pants", "dress", "skirt", "coat", "jacket", "hat",
        "cap", "shoe", "boot", "sock", "glove", "scarf", "belt",
        "tie", "suit", "uniform", "armor", "crown", "ring",
        "necklace", "bracelet", "mask", "veil", "cloak", "robe",
    ],

    # ---- Weapons & Tools ----
    "weapon_tool": [
        "sword", "knife", "spear", "arrow", "bow", "shield", "axe",
        "hammer", "gun", "rifle", "cannon", "bomb", "missile",
        "rope", "chain", "key", "lock", "needle", "thread",
        "wheel", "lever", "pulley", "compass", "telescope", "microscope",
        "pen", "pencil", "brush", "mirror", "bell", "drum", "horn",
    ],

    # ---- Music & Art ----
    "music_art": [
        "piano", "guitar", "violin", "flute", "trumpet", "harp",
        "drum", "orchestra", "choir", "symphony", "opera", "ballet",
        "poem", "novel", "painting", "sculpture", "photograph",
        "melody", "rhythm", "harmony", "chorus", "verse",
    ],

    # ---- Shapes & Math ----
    "shape_math": [
        "circle", "triangle", "square", "cube", "sphere", "cylinder",
        "cone", "pyramid", "spiral", "helix", "oval", "arc",
        "angle", "parallel", "perpendicular", "diagonal", "radius",
        "diameter", "circumference", "area", "volume", "surface",
    ],

    # ---- Continents & Regions ----
    "region": [
        "Europe", "Asia", "Africa", "America", "Oceania",
        "European", "Asian", "African", "American",
        "Arctic", "Antarctic", "Caribbean", "Mediterranean",
        "Scandinavian", "Baltic", "Pacific", "Atlantic",
    ],
}


# =============================================================================
# LOADING
# =============================================================================

def load_embeddings_and_vocab():
    """Load output embeddings and tokenizer vocabulary."""
    from phi_geometric.inference.phi_types import PhiEncoded

    print("Loading embeddings...", flush=True)
    phi = PhiEncoded.load(str(MODEL_DIR / "embed_tokens.npz"))
    embeddings = phi.decode()

    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"
    )
    snapshots = os.listdir(cache_dir)
    vocab_file = os.path.join(cache_dir, snapshots[0], "tokenizer.json")
    with open(vocab_file, "r") as f:
        tokenizer_data = json.load(f)
    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    token_to_id = {tok: idx for tok, idx in vocab.items()}

    print(f"  {embeddings.shape[0]} tokens, {embeddings.shape[1]} dims", flush=True)
    return embeddings, token_to_id, id_to_token


def find_token_id(word, token_to_id):
    """Try multiple tokenization patterns to find a single-token match."""
    candidates = [
        word, word.lower(), word.capitalize(), word.upper(),
        f"Ġ{word}", f"Ġ{word.lower()}", f"Ġ{word.capitalize()}",
        f"▁{word}", f"▁{word.lower()}", f"▁{word.capitalize()}",
        f" {word}", f" {word.lower()}", f" {word.capitalize()}",
    ]
    for c in candidates:
        if c in token_to_id:
            return token_to_id[c], c
    return None, None


# =============================================================================
# PHASE 0: BUILD CONCEPT VOCABULARY
# =============================================================================

def build_concept_vocabulary(embeddings, token_to_id, id_to_token, notes):
    """Systematically build a large, clean concept vocabulary."""

    notes.section("1. Candidate Collection")

    # Collect all candidates across categories
    all_candidates = {}  # word -> list of categories
    for cat_name, words in CONCEPT_CATEGORIES.items():
        for word in words:
            if word not in all_candidates:
                all_candidates[word] = []
            all_candidates[word].append(cat_name)

    notes.observe(f"Total unique candidate words: {len(all_candidates)}")
    notes.observe(f"Categories: {len(CONCEPT_CATEGORIES)}")
    for cat_name, words in sorted(CONCEPT_CATEGORIES.items()):
        notes.observe(f"  {cat_name}: {len(words)} candidates")

    # Try to find each candidate as a single token
    notes.section("2. Token Resolution")

    found = {}       # word -> (token_id, token_str, categories)
    not_found = []   # words that aren't single tokens

    for word, cats in all_candidates.items():
        tid, tok_str = find_token_id(word, token_to_id)
        if tid is not None:
            found[word] = (tid, tok_str, cats)
        else:
            not_found.append(word)

    notes.observe(f"Found as single tokens: {len(found)} / {len(all_candidates)} "
                  f"({100*len(found)/len(all_candidates):.1f}%)")
    notes.observe(f"Not found (multi-token): {len(not_found)}")

    if not_found:
        # Show a sample of not-found words
        notes.observe(f"\nSample not-found words (first 30):")
        for w in sorted(not_found)[:30]:
            notes.observe(f"  - {w}")

    # Check for duplicate token IDs (different words mapping to same token)
    tid_to_words = {}
    for word, (tid, tok_str, cats) in found.items():
        if tid not in tid_to_words:
            tid_to_words[tid] = []
        tid_to_words[tid].append(word)

    dupes = {tid: words for tid, words in tid_to_words.items() if len(words) > 1}
    if dupes:
        notes.observe(f"\nDuplicate token IDs ({len(dupes)} cases):")
        for tid, words in sorted(dupes.items())[:20]:
            notes.observe(f"  Token {tid}: {words}")

    # Quality filter: embedding norm check
    notes.section("3. Quality Filtering")

    all_norms = np.linalg.norm(embeddings, axis=1)
    norm_mean = np.mean(all_norms)
    norm_std = np.std(all_norms)
    notes.observe(f"Vocab embedding norms: mean={norm_mean:.4f}, std={norm_std:.4f}")
    notes.observe(f"  2σ range: [{norm_mean - 2*norm_std:.4f}, {norm_mean + 2*norm_std:.4f}]")

    # Filter
    concepts = {}
    rejected_norm = []
    rejected_dupe_tid = set()

    # For duplicates, keep the first alphabetically
    for tid, words in dupes.items():
        keep = sorted(words)[0]
        for w in words:
            if w != keep:
                rejected_dupe_tid.add(w)

    for word, (tid, tok_str, cats) in found.items():
        if word in rejected_dupe_tid:
            continue

        emb = embeddings[tid]
        norm = np.linalg.norm(emb)

        if norm < norm_mean - 2*norm_std or norm > norm_mean + 2*norm_std:
            rejected_norm.append((word, norm))
            continue

        concepts[word] = Concept(
            name=word, token_id=tid, token_str=tok_str,
            embedding=emb, categories=cats,
        )

    notes.observe(f"\nRejected (duplicate token ID): {len(rejected_dupe_tid)}")
    if rejected_dupe_tid:
        notes.observe(f"  {sorted(rejected_dupe_tid)[:20]}")

    notes.observe(f"Rejected (norm outlier): {len(rejected_norm)}")
    if rejected_norm:
        for w, n in sorted(rejected_norm, key=lambda x: x[1])[:10]:
            notes.observe(f"  {w}: norm={n:.4f}")

    notes.observe(f"\n**Final concept count: {len(concepts)}**")

    # Category distribution
    notes.section("4. Category Distribution")

    cat_counts = {}
    for c in concepts.values():
        for cat in c.categories:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    rows = []
    for cat_name in sorted(cat_counts.keys()):
        rows.append((cat_name, cat_counts[cat_name]))
    notes.data_table(["Category", "Count"], rows, "Concepts per category")

    # Supercategory summary
    geo_count = sum(v for k, v in cat_counts.items()
                    if k.startswith("country") or k in ["capital_city", "language", "region"])
    animal_count = sum(v for k, v in cat_counts.items() if k.startswith("animal"))
    gender_count = sum(v for k, v in cat_counts.items() if k.startswith("gender"))
    notes.observe(f"\nSuper-categories:")
    notes.observe(f"  Geography (countries + capitals + languages + regions): {geo_count}")
    notes.observe(f"  Animals: {animal_count}")
    notes.observe(f"  Gender: {gender_count}")
    notes.observe(f"  Other: {len(concepts) - geo_count - animal_count - gender_count}")

    return concepts


# =============================================================================
# VALIDATION: Do the 6 known axes still work?
# =============================================================================

def validate_known_axes(concepts, embeddings, notes):
    """Re-test the 6 known truth axes with the expanded concept set."""

    notes.section("5. Validation — Known Truth Axes on Expanded Vocabulary")

    # The 6 known axes from DC 298
    anchors = {
        "is_european_country": {
            "positive": ["France", "Germany", "Poland", "Norway", "Sweden",
                         "Italy", "Portugal", "Spain", "Greece", "Ireland",
                         "Finland", "Denmark", "Austria", "Belgium", "Netherlands",
                         "Switzerland", "Russia", "Romania", "Hungary",
                         "Croatia", "Serbia", "Bulgaria", "Slovakia", "Lithuania",
                         "Latvia", "Estonia", "Slovenia", "Iceland",
                         "Albania", "Ukraine", "Belarus", "Moldova", "Georgia",
                         "Armenia", "Luxembourg", "Malta", "Cyprus"],
            "negative": ["Japan", "China", "Egypt", "Australia", "Thailand",
                         "India", "Brazil", "Korea", "Turkey", "Nigeria",
                         "Kenya", "Morocco", "Israel", "Iran", "Vietnam",
                         "Indonesia", "Philippines", "Mexico", "Canada",
                         "Argentina", "Chile", "Colombia", "Peru",
                         "Pakistan", "Bangladesh", "Myanmar", "Cambodia",
                         "Nepal", "Mongolia", "Taiwan", "Kuwait", "Qatar",
                         "Oman", "Yemen", "Jordan", "Lebanon", "Syria",
                         "Afghanistan", "Ghana", "Ethiopia", "Tanzania",
                         "Uganda", "Senegal", "Sudan", "Libya", "Tunisia",
                         "Algeria", "Somalia", "Rwanda", "Cameroon",
                         "Zimbabwe", "Zambia", "Mozambique", "Madagascar",
                         "Namibia", "Botswana", "Venezuela", "Cuba",
                         "Jamaica", "Panama", "Ecuador", "Bolivia",
                         "Paraguay", "Uruguay", "Guatemala", "Honduras"],
        },
        "is_asian_country": {
            "positive": ["Japan", "China", "Thailand", "India", "Korea",
                         "Vietnam", "Indonesia", "Philippines", "Malaysia",
                         "Singapore", "Iran", "Iraq", "Israel",
                         "Pakistan", "Bangladesh", "Myanmar", "Cambodia",
                         "Nepal", "Laos", "Mongolia", "Taiwan", "Bhutan",
                         "Kuwait", "Qatar", "Oman", "Yemen", "Jordan",
                         "Lebanon", "Syria", "Afghanistan"],
            "negative": ["France", "Germany", "Poland", "Norway", "Sweden",
                         "Italy", "Portugal", "Spain", "Egypt", "Australia",
                         "Brazil", "Nigeria", "Kenya", "Morocco",
                         "Mexico", "Canada", "Argentina", "Chile",
                         "Colombia", "Peru", "Ghana", "Ethiopia",
                         "Tanzania", "Uganda", "Senegal", "Sudan",
                         "Libya", "Tunisia", "Algeria", "Venezuela",
                         "Cuba", "Jamaica", "Panama", "Ecuador"],
        },
        "is_capital_city": {
            "positive": ["Paris", "Berlin", "Tokyo", "Beijing", "Cairo",
                         "Canberra", "Bangkok", "Warsaw", "Oslo", "Stockholm",
                         "Delhi", "Seoul", "Rome", "Lisbon", "Moscow",
                         "Madrid", "Athens", "Ankara", "Dublin", "Helsinki",
                         "Copenhagen", "Vienna", "Brussels", "Amsterdam",
                         "Ottawa", "Lima", "Tehran", "Baghdad", "Hanoi",
                         "Jakarta", "Manila", "Havana", "Nairobi", "Accra",
                         "Dakar", "Tunis", "Algiers", "Tripoli", "Riyadh",
                         "Doha", "Kabul", "Beirut", "Damascus", "Amman",
                         "Bucharest", "Budapest", "Prague", "Zagreb",
                         "Belgrade", "Sofia", "Bratislava", "Tallinn",
                         "Riga", "Vilnius", "Ljubljana", "Minsk",
                         "Tbilisi", "Yerevan", "Baku", "Taipei"],
            "negative": ["France", "Germany", "Japan", "China", "Egypt",
                         "Australia", "Thailand", "Poland", "Norway", "Sweden",
                         "India", "Korea", "Italy", "Portugal", "Russia",
                         "Spain", "Greece", "Turkey", "Ireland",
                         "Brazil", "Mexico", "Canada", "Argentina",
                         "Chile", "Colombia", "Peru", "Nigeria", "Kenya",
                         "Morocco", "Iran", "Iraq", "Vietnam", "Indonesia",
                         "Philippines", "Malaysia"],
        },
        "is_romance_language": {
            "positive": ["French", "Italian", "Portuguese", "Spanish", "Romanian"],
            "negative": ["German", "Japanese", "Chinese", "Arabic", "English",
                         "Korean", "Thai", "Polish", "Norwegian", "Swedish",
                         "Dutch", "Greek", "Turkish", "Hindi", "Finnish",
                         "Russian", "Danish", "Persian", "Vietnamese",
                         "Hungarian", "Czech", "Croatian", "Serbian",
                         "Bulgarian", "Slovak", "Lithuanian", "Latvian",
                         "Estonian", "Slovenian", "Icelandic", "Albanian",
                         "Ukrainian", "Georgian", "Armenian", "Swahili",
                         "Bengali", "Urdu", "Tamil", "Nepali", "Mongolian",
                         "Indonesian"],
        },
        "is_germanic_language": {
            "positive": ["German", "English", "Dutch", "Norwegian", "Swedish",
                         "Danish", "Icelandic"],
            "negative": ["French", "Italian", "Portuguese", "Spanish",
                         "Japanese", "Chinese", "Arabic", "Korean",
                         "Polish", "Greek", "Turkish", "Hindi", "Finnish",
                         "Russian", "Thai", "Persian", "Vietnamese",
                         "Romanian", "Hungarian", "Czech", "Croatian",
                         "Serbian", "Bulgarian", "Slovak", "Lithuanian",
                         "Latvian", "Estonian", "Slovenian", "Albanian",
                         "Ukrainian", "Georgian", "Armenian", "Swahili",
                         "Bengali", "Urdu", "Tamil", "Indonesian"],
        },
        "is_female_gendered": {
            "positive": ["queen", "woman", "girl", "mother", "sister",
                         "daughter", "wife", "aunt", "princess", "actress",
                         "waitress", "heroine", "nun", "witch",
                         "lady", "duchess", "empress", "bride",
                         "niece", "granddaughter", "goddess", "madam", "lass"],
            "negative": ["king", "man", "boy", "father", "brother",
                         "son", "husband", "uncle", "prince", "actor",
                         "waiter", "hero", "monk", "wizard",
                         "gentleman", "lord", "duke", "emperor",
                         "bachelor", "groom", "nephew", "grandson",
                         "god", "sir", "lad"],
        },
    }

    anchor_directions = {}

    for anchor_name, anchor_def in anchors.items():
        # Get token IDs for concepts we actually have
        pos_tids, pos_names = [], []
        for word in anchor_def["positive"]:
            if word in concepts:
                pos_tids.append(concepts[word].token_id)
                pos_names.append(word)

        neg_tids, neg_names = [], []
        for word in anchor_def["negative"]:
            if word in concepts:
                neg_tids.append(concepts[word].token_id)
                neg_names.append(word)

        notes.observe(f"\n### Anchor: {anchor_name}")
        notes.observe(f"  Positive: {len(pos_tids)} concepts")
        notes.observe(f"  Negative: {len(neg_tids)} concepts")

        if len(pos_tids) < 2 or len(neg_tids) < 2:
            notes.observe(f"  SKIP: insufficient examples")
            continue

        # Compute anchor direction
        pos_embs = np.array([embeddings[tid] for tid in pos_tids])
        neg_embs = np.array([embeddings[tid] for tid in neg_tids])
        anchor_dir = np.mean(pos_embs, axis=0) - np.mean(neg_embs, axis=0)
        anchor_dir_norm = anchor_dir / (np.linalg.norm(anchor_dir) + 1e-20)

        # Classification accuracy (full set)
        d_norm = anchor_dir / (np.linalg.norm(anchor_dir) + 1e-20)
        projections = embeddings @ d_norm
        pos_projs = [projections[tid] for tid in pos_tids]
        neg_projs = [projections[tid] for tid in neg_tids]
        pos_mean = np.mean(pos_projs)
        neg_mean = np.mean(neg_projs)
        threshold = (pos_mean + neg_mean) / 2

        correct = sum(1 for p in pos_projs if p > threshold) + \
                  sum(1 for n in neg_projs if n <= threshold)
        total = len(pos_tids) + len(neg_tids)
        accuracy = correct / total

        # LOO cross-validation
        loo_correct = 0
        loo_total = 0
        for i in range(len(pos_tids)):
            train_pos = [t for j, t in enumerate(pos_tids) if j != i]
            loo_dir = (np.mean([embeddings[t] for t in train_pos], axis=0) -
                       np.mean([embeddings[t] for t in neg_tids], axis=0))
            loo_norm = loo_dir / (np.linalg.norm(loo_dir) + 1e-20)
            proj = np.dot(embeddings[pos_tids[i]], loo_norm)
            thresh = (np.mean([np.dot(embeddings[t], loo_norm) for t in train_pos]) +
                      np.mean([np.dot(embeddings[t], loo_norm) for t in neg_tids])) / 2
            if proj > thresh:
                loo_correct += 1
            loo_total += 1

        for i in range(len(neg_tids)):
            train_neg = [t for j, t in enumerate(neg_tids) if j != i]
            loo_dir = (np.mean([embeddings[t] for t in pos_tids], axis=0) -
                       np.mean([embeddings[t] for t in train_neg], axis=0))
            loo_norm = loo_dir / (np.linalg.norm(loo_dir) + 1e-20)
            proj = np.dot(embeddings[neg_tids[i]], loo_norm)
            thresh = (np.mean([np.dot(embeddings[t], loo_norm) for t in pos_tids]) +
                      np.mean([np.dot(embeddings[t], loo_norm) for t in train_neg])) / 2
            if proj <= thresh:
                loo_correct += 1
            loo_total += 1

        loo_acc = loo_correct / loo_total

        notes.observe(f"  **Classification: {accuracy*100:.1f}%** ({correct}/{total})")
        notes.observe(f"  **LOO CV: {loo_acc*100:.1f}%** ({loo_correct}/{loo_total})")

        # Margin
        if pos_mean > neg_mean:
            margin = min(pos_projs) - max(neg_projs)
        else:
            margin = min(neg_projs) - max(pos_projs)
        notes.observe(f"  Margin: {margin:.4f} ({'SEPARABLE' if margin > 0 else 'OVERLAPPING'})")

        anchor_directions[anchor_name] = {
            "direction": anchor_dir_norm,
            "accuracy": accuracy,
            "loo_accuracy": loo_acc,
            "threshold": float(threshold),
            "pos_tids": pos_tids,
            "neg_tids": neg_tids,
            "n_pos": len(pos_tids),
            "n_neg": len(neg_tids),
        }

    # Cross-anchor orthogonality
    notes.observe("\n### Cross-Anchor Orthogonality")
    names = list(anchor_directions.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            cos = float(np.dot(anchor_directions[names[i]]["direction"],
                               anchor_directions[names[j]]["direction"]))
            notes.observe(f"  cos({names[i]}, {names[j]}) = {cos:.4f}")

    return anchor_directions


# =============================================================================
# VARIANCE ANALYSIS on expanded set
# =============================================================================

def variance_analysis(concepts, anchor_directions, notes):
    """How much variance do the 6 known axes explain on the expanded set?"""

    notes.section("6. Variance Analysis — Known Axes on Expanded Vocabulary")

    # Build concept matrix
    concept_names = sorted(concepts.keys())
    C = np.array([concepts[name].embedding for name in concept_names])
    C_centered = C - np.mean(C, axis=0)
    total_var = np.sum(C_centered ** 2)

    notes.observe(f"Concept matrix: {C.shape[0]} concepts × {C.shape[1]} dims")
    notes.observe(f"Total variance: {total_var:.2f}")

    # Project onto known axes
    axis_names = sorted(anchor_directions.keys())
    A = np.array([anchor_directions[name]["direction"] for name in axis_names])  # (6, 3584)

    # Orthogonalize via QR for clean projection
    Q, R = np.linalg.qr(A.T)  # Q: (3584, 6), orthonormal basis
    n_axes = Q.shape[1]

    # Project
    projections = C_centered @ Q  # (N, 6)
    reconstructed = projections @ Q.T  # (N, 3584)
    residuals = C_centered - reconstructed

    proj_var = np.sum(projections ** 2)
    resid_var = np.sum(residuals ** 2)
    explained = proj_var / total_var

    notes.observe(f"\nProjection onto {n_axes} known axes:")
    notes.observe(f"  Variance explained: {explained*100:.2f}%")
    notes.observe(f"  Residual variance: {(1-explained)*100:.2f}%")

    # PCA analysis: how many dims does the expanded set need?
    notes.observe(f"\n### PCA Analysis")
    U, S, Vt = np.linalg.svd(C_centered, full_matrices=False)
    explained_ratio = np.cumsum(S**2) / np.sum(S**2)

    for target in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        n_dims = int(np.searchsorted(explained_ratio, target) + 1)
        notes.observe(f"  {target*100:.0f}% variance: {n_dims} PCA dims")

    # Compare efficiency: our axes vs PCA
    pca_6 = float(explained_ratio[5]) if len(explained_ratio) > 5 else 0
    notes.observe(f"\n  6 PCA dims explain: {pca_6*100:.2f}%")
    notes.observe(f"  6 known axes explain: {explained*100:.2f}%")
    notes.observe(f"  Efficiency ratio: {explained/pca_6:.2f}x" if pca_6 > 0 else "  PCA N/A")

    # Per-category breakdown
    notes.observe(f"\n### Per-Category Variance Explained by Known Axes")
    cat_groups = {}
    for name, concept in concepts.items():
        for cat in concept.categories:
            if cat not in cat_groups:
                cat_groups[cat] = []
            cat_groups[cat].append(name)

    rows = []
    for cat_name in sorted(cat_groups.keys()):
        members = cat_groups[cat_name]
        if len(members) < 3:
            continue
        cat_embs = np.array([concepts[m].embedding for m in members])
        cat_centered = cat_embs - np.mean(C, axis=0)  # center on global mean
        cat_total = np.sum(cat_centered ** 2)
        cat_proj = (cat_centered @ Q) @ Q.T
        cat_resid = cat_centered - cat_proj
        cat_explained = np.sum(cat_proj ** 2) / cat_total if cat_total > 0 else 0
        rows.append((cat_name, len(members), f"{cat_explained*100:.1f}%"))

    notes.data_table(["Category", "N Concepts", "Var Explained"],
                     rows, "Known axes variance by category")

    notes.finding(f"With {len(concepts)} concepts (up from 88), the 6 known axes "
                  f"explain {explained*100:.1f}% of variance. "
                  f"PCA suggests ~{int(np.searchsorted(explained_ratio, 0.95)+1)} "
                  f"axes for 95% coverage.")

    return {
        "n_concepts": len(concepts),
        "n_axes": n_axes,
        "variance_explained": float(explained),
        "pca_95_dims": int(np.searchsorted(explained_ratio, 0.95) + 1),
        "pca_99_dims": int(np.searchsorted(explained_ratio, 0.99) + 1),
    }


# =============================================================================
# SAVE CONCEPT VOCABULARY
# =============================================================================

def save_vocabulary(concepts, anchor_directions, stats, notes):
    """Save the curated concept vocabulary for Phase 1."""

    notes.section("7. Saved Artifacts")

    # Save concept list as JSON
    vocab_data = {
        "metadata": {
            "n_concepts": len(concepts),
            "n_categories": len(set(cat for c in concepts.values() for cat in c.categories)),
            "stats": stats,
        },
        "concepts": {
            name: {
                "token_id": c.token_id,
                "token_str": c.token_str,
                "categories": c.categories,
            }
            for name, c in sorted(concepts.items())
        },
        "anchors": {
            name: {
                "accuracy": data["accuracy"],
                "loo_accuracy": data["loo_accuracy"],
                "threshold": data["threshold"],
                "n_pos": data["n_pos"],
                "n_neg": data["n_neg"],
            }
            for name, data in anchor_directions.items()
        },
    }

    vocab_path = SCRIPT_DIR / "concept_vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab_data, f, indent=2)
    notes.observe(f"Concept vocabulary saved to: {vocab_path}")

    # Save anchor directions as numpy
    anchor_path = SCRIPT_DIR / "anchor_directions.npz"
    anchor_arrays = {}
    for name, data in anchor_directions.items():
        anchor_arrays[name] = data["direction"]
    np.savez(anchor_path, **anchor_arrays)
    notes.observe(f"Anchor directions saved to: {anchor_path}")

    # Save concept embeddings subset (just our concepts, for fast loading in Phase 1)
    concept_names = sorted(concepts.keys())
    concept_ids = np.array([concepts[name].token_id for name in concept_names])
    concept_embs = np.array([concepts[name].embedding for name in concept_names])
    emb_path = SCRIPT_DIR / "concept_embeddings.npz"
    np.savez(emb_path, names=concept_names, token_ids=concept_ids,
             embeddings=concept_embs)
    notes.observe(f"Concept embeddings saved to: {emb_path}")

    notes.observe(f"\n**Phase 0 complete. {len(concepts)} concepts across "
                  f"{len(set(cat for c in concepts.values() for cat in c.categories))} "
                  f"categories ready for Phase 1 axis discovery.**")


# =============================================================================
# MAIN
# =============================================================================

def main():
    notes = FieldNotes(NOTES_PATH)

    try:
        # Load model data
        embeddings, token_to_id, id_to_token = load_embeddings_and_vocab()

        # Build expanded vocabulary
        concepts = build_concept_vocabulary(embeddings, token_to_id, id_to_token, notes)

        # Validate known axes still work
        anchor_directions = validate_known_axes(concepts, embeddings, notes)

        # Variance analysis
        stats = variance_analysis(concepts, anchor_directions, notes)

        # Save everything
        save_vocabulary(concepts, anchor_directions, stats, notes)

    finally:
        notes.close()

    print(f"\nDone. Notes written to {NOTES_PATH}")


if __name__ == "__main__":
    main()
