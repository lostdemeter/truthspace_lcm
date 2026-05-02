"""
Knowledge Base — Entity × Fact Type definitions
=================================================

Defines all entities, fact types, prompt templates, and expected answers
for the geometric knowledge engine.

To add knowledge:
  1. Add entities to ENTITIES
  2. Add fact types to FACT_TYPES
  3. Answers are auto-discovered from the model during extraction
     OR can be specified explicitly in KNOWN_ANSWERS
"""

# ═══════════════════════════════════════════════════════════════
# Fact type definitions
# ═══════════════════════════════════════════════════════════════

FACT_TYPES = {
    'capital': {
        'template': 'The capital of {entity} is',
        'response': 'The capital of {entity} is {answer}.',
        'keywords': ['capital', 'Capital', 'city'],
    },
    'language': {
        'template': 'The primary language of {entity} is',
        'response': 'The primary language of {entity} is {answer}.',
        'keywords': ['language', 'Language', 'speak', 'spoken'],
    },
    'continent': {
        'template': 'The continent of {entity} is',
        'response': '{entity} is located in {answer}.',
        'keywords': ['continent', 'Continent', 'located', 'where'],
    },
    'currency': {
        'template': 'The currency of {entity} is the',
        'response': 'The currency of {entity} is the {answer}.',
        'keywords': ['currency', 'Currency', 'money', 'pay'],
    },
}

# ═══════════════════════════════════════════════════════════════
# Entity definitions
# ═══════════════════════════════════════════════════════════════

ENTITIES = [
    # Europe
    'France', 'Germany', 'Italy', 'Spain', 'Portugal',
    'Poland', 'Greece', 'Sweden', 'Norway', 'Denmark',
    'Finland', 'Austria', 'Belgium', 'Netherlands', 'Switzerland',
    'Ireland', 'Romania', 'Hungary', 'Ukraine', 'Turkey',
    # Asia
    'Japan', 'China', 'India', 'Thailand', 'Vietnam',
    'Indonesia', 'Philippines', 'Pakistan', 'Bangladesh', 'Malaysia',
    'Korea', 'Taiwan',
    # Americas
    'Brazil', 'Mexico', 'Argentina', 'Colombia', 'Chile',
    'Peru', 'Canada', 'Cuba',
    # Africa
    'Egypt', 'Nigeria', 'Kenya', 'Ethiopia', 'Morocco',
    'Ghana',
    # Oceania
    'Australia',
]

# Known answers: (entity, fact_type) → answer token string
# The extraction pipeline will auto-discover answers from the model,
# but these overrides ensure correctness for tricky cases.
# Answer strings must include leading space (tokenizer convention).
KNOWN_ANSWERS = {
    # Capitals
    ('France', 'capital'): ' Paris',
    ('Germany', 'capital'): ' Berlin',
    ('Italy', 'capital'): ' Rome',
    ('Spain', 'capital'): ' Madrid',
    ('Portugal', 'capital'): ' Lisbon',
    ('Poland', 'capital'): ' Warsaw',
    ('Greece', 'capital'): ' Athens',
    ('Sweden', 'capital'): ' Stockholm',
    ('Norway', 'capital'): ' Oslo',
    ('Denmark', 'capital'): ' Copenhagen',
    ('Finland', 'capital'): ' Helsinki',
    ('Austria', 'capital'): ' Vienna',
    ('Belgium', 'capital'): ' Brussels',
    ('Netherlands', 'capital'): ' Amsterdam',
    ('Switzerland', 'capital'): ' Bern',
    ('Ireland', 'capital'): ' Dublin',
    ('Romania', 'capital'): ' Bucharest',
    ('Hungary', 'capital'): ' Budapest',
    ('Ukraine', 'capital'): ' Ky',        # Kiev/Kyiv — first token
    ('Turkey', 'capital'): ' Ankara',
    ('Japan', 'capital'): ' Tokyo',
    ('China', 'capital'): ' Beijing',
    ('India', 'capital'): ' New',          # New Delhi — first token
    ('Thailand', 'capital'): ' Bangkok',
    ('Vietnam', 'capital'): ' Han',        # Hanoi — first token
    ('Indonesia', 'capital'): ' Jakarta',
    ('Philippines', 'capital'): ' Manila',
    ('Pakistan', 'capital'): ' Islam',     # Islamabad — first token
    ('Bangladesh', 'capital'): ' Dhaka',
    ('Malaysia', 'capital'): ' Ku',        # Kuala Lumpur — first token
    ('Korea', 'capital'): ' Seoul',
    ('Taiwan', 'capital'): ' Taipei',
    ('Brazil', 'capital'): ' Bras',        # Brasília — first token
    ('Mexico', 'capital'): ' Mexico',
    ('Argentina', 'capital'): ' Buenos',   # Buenos Aires — first token
    ('Colombia', 'capital'): ' Bog',       # Bogotá — first token
    ('Chile', 'capital'): ' Santiago',
    ('Peru', 'capital'): ' Lima',
    ('Canada', 'capital'): ' Ottawa',
    ('Cuba', 'capital'): ' Havana',
    ('Egypt', 'capital'): ' Cairo',
    ('Nigeria', 'capital'): ' Ab',         # Abuja — first token
    ('Kenya', 'capital'): ' Nair',         # Nairobi — first token
    ('Ethiopia', 'capital'): ' Add',       # Addis Ababa — first token
    ('Morocco', 'capital'): ' Rab',        # Rabat — first token
    ('Ghana', 'capital'): ' Acc',          # Accra — first token
    ('Australia', 'capital'): ' Canberra',

    # Languages
    ('France', 'language'): ' French',
    ('Germany', 'language'): ' German',
    ('Italy', 'language'): ' Italian',
    ('Spain', 'language'): ' Spanish',
    ('Portugal', 'language'): ' Portuguese',
    ('Poland', 'language'): ' Polish',
    ('Greece', 'language'): ' Greek',
    ('Sweden', 'language'): ' Swedish',
    ('Norway', 'language'): ' Norwegian',
    ('Denmark', 'language'): ' Danish',
    ('Finland', 'language'): ' Finnish',
    ('Austria', 'language'): ' German',
    ('Belgium', 'language'): ' French',     # or Dutch, model picks one
    ('Netherlands', 'language'): ' Dutch',
    ('Switzerland', 'language'): ' German',  # or French/Italian
    ('Ireland', 'language'): ' English',
    ('Romania', 'language'): ' Roman',       # Romanian — first token
    ('Hungary', 'language'): ' Hungarian',
    ('Ukraine', 'language'): ' Ukrainian',
    ('Turkey', 'language'): ' Turkish',
    ('Japan', 'language'): ' Japanese',
    ('China', 'language'): ' Mandarin',
    ('India', 'language'): ' Hindi',
    ('Thailand', 'language'): ' Thai',
    ('Vietnam', 'language'): ' Vietnamese',
    ('Indonesia', 'language'): ' Indonesian',
    ('Philippines', 'language'): ' Filipino',
    ('Pakistan', 'language'): ' Ur',         # Urdu — first token
    ('Bangladesh', 'language'): ' Bengal',    # Bengali — first token
    ('Malaysia', 'language'): ' Mal',         # Malay — first token
    ('Korea', 'language'): ' Korean',
    ('Taiwan', 'language'): ' Mandarin',
    ('Brazil', 'language'): ' Portuguese',
    ('Mexico', 'language'): ' Spanish',
    ('Argentina', 'language'): ' Spanish',
    ('Colombia', 'language'): ' Spanish',
    ('Chile', 'language'): ' Spanish',
    ('Peru', 'language'): ' Spanish',
    ('Canada', 'language'): ' English',
    ('Cuba', 'language'): ' Spanish',
    ('Egypt', 'language'): ' Arabic',
    ('Nigeria', 'language'): ' English',
    ('Kenya', 'language'): ' Sw',            # Swahili — first token
    ('Ethiopia', 'language'): ' Am',          # Amharic — first token
    ('Morocco', 'language'): ' Arabic',
    ('Ghana', 'language'): ' English',
    ('Australia', 'language'): ' English',

    # Continents
    ('France', 'continent'): ' Europe',
    ('Germany', 'continent'): ' Europe',
    ('Italy', 'continent'): ' Europe',
    ('Spain', 'continent'): ' Europe',
    ('Portugal', 'continent'): ' Europe',
    ('Poland', 'continent'): ' Europe',
    ('Greece', 'continent'): ' Europe',
    ('Sweden', 'continent'): ' Europe',
    ('Norway', 'continent'): ' Europe',
    ('Denmark', 'continent'): ' Europe',
    ('Finland', 'continent'): ' Europe',
    ('Austria', 'continent'): ' Europe',
    ('Belgium', 'continent'): ' Europe',
    ('Netherlands', 'continent'): ' Europe',
    ('Switzerland', 'continent'): ' Europe',
    ('Ireland', 'continent'): ' Europe',
    ('Romania', 'continent'): ' Europe',
    ('Hungary', 'continent'): ' Europe',
    ('Ukraine', 'continent'): ' Europe',
    ('Turkey', 'continent'): ' Asia',        # technically transcontinental
    ('Japan', 'continent'): ' Asia',
    ('China', 'continent'): ' Asia',
    ('India', 'continent'): ' Asia',
    ('Thailand', 'continent'): ' Asia',
    ('Vietnam', 'continent'): ' Asia',
    ('Indonesia', 'continent'): ' Asia',
    ('Philippines', 'continent'): ' Asia',
    ('Pakistan', 'continent'): ' Asia',
    ('Bangladesh', 'continent'): ' Asia',
    ('Malaysia', 'continent'): ' Asia',
    ('Korea', 'continent'): ' Asia',
    ('Taiwan', 'continent'): ' Asia',
    ('Brazil', 'continent'): ' South',
    ('Mexico', 'continent'): ' North',
    ('Argentina', 'continent'): ' South',
    ('Colombia', 'continent'): ' South',
    ('Chile', 'continent'): ' South',
    ('Peru', 'continent'): ' South',
    ('Canada', 'continent'): ' North',
    ('Cuba', 'continent'): ' North',
    ('Egypt', 'continent'): ' Africa',
    ('Nigeria', 'continent'): ' Africa',
    ('Kenya', 'continent'): ' Africa',
    ('Ethiopia', 'continent'): ' Africa',
    ('Morocco', 'continent'): ' Africa',
    ('Ghana', 'continent'): ' Africa',
    ('Australia', 'continent'): ' Ocean',    # Oceania — first token

    # Currencies
    ('France', 'currency'): ' Euro',
    ('Germany', 'currency'): ' Euro',
    ('Italy', 'currency'): ' Euro',
    ('Spain', 'currency'): ' Euro',
    ('Portugal', 'currency'): ' Euro',
    ('Poland', 'currency'): ' z',            # złoty
    ('Greece', 'currency'): ' Euro',
    ('Sweden', 'currency'): ' Swedish',      # Swedish krona
    ('Norway', 'currency'): ' Norwegian',    # Norwegian krone
    ('Denmark', 'currency'): ' Danish',      # Danish krone
    ('Finland', 'currency'): ' Euro',
    ('Austria', 'currency'): ' Euro',
    ('Belgium', 'currency'): ' Euro',
    ('Netherlands', 'currency'): ' Euro',
    ('Switzerland', 'currency'): ' Swiss',   # Swiss franc
    ('Ireland', 'currency'): ' Euro',
    ('Romania', 'currency'): ' le',          # leu
    ('Hungary', 'currency'): ' Hungarian',   # Hungarian forint
    ('Ukraine', 'currency'): ' hr',          # hryvnia
    ('Turkey', 'currency'): ' Turkish',      # Turkish lira
    ('Japan', 'currency'): ' yen',
    ('China', 'currency'): ' yuan',
    ('India', 'currency'): ' Indian',        # Indian rupee
    ('Thailand', 'currency'): ' ba',         # baht
    ('Vietnam', 'currency'): ' dong',
    ('Indonesia', 'currency'): ' Indonesian', # Indonesian rupiah
    ('Philippines', 'currency'): ' Philippine', # Philippine peso
    ('Pakistan', 'currency'): ' Pakistani',  # Pakistani rupee
    ('Bangladesh', 'currency'): ' Bangladesh', # Bangladeshi taka
    ('Malaysia', 'currency'): ' Malaysian',  # Malaysian ringgit
    ('Korea', 'currency'): ' South',         # South Korean won
    ('Taiwan', 'currency'): ' New',          # New Taiwan dollar
    ('Brazil', 'currency'): ' Brazilian',    # Brazilian real
    ('Mexico', 'currency'): ' Mexican',      # Mexican peso
    ('Argentina', 'currency'): ' Argentine', # Argentine peso
    ('Colombia', 'currency'): ' Colombian',  # Colombian peso
    ('Chile', 'currency'): ' Chilean',       # Chilean peso
    ('Peru', 'currency'): ' Peruvian',       # Peruvian sol
    ('Canada', 'currency'): ' Canadian',     # Canadian dollar
    ('Cuba', 'currency'): ' Cuban',          # Cuban peso
    ('Egypt', 'currency'): ' Egyptian',      # Egyptian pound
    ('Nigeria', 'currency'): ' na',          # naira
    ('Kenya', 'currency'): ' Kenyan',        # Kenyan shilling
    ('Ethiopia', 'currency'): ' Ethiopian',  # Ethiopian birr
    ('Morocco', 'currency'): ' Moroccan',    # Moroccan dirham
    ('Ghana', 'currency'): ' Ghan',          # Ghanaian cedi
    ('Australia', 'currency'): ' Australian', # Australian dollar
}


def get_all_facts():
    """Return list of (entity, fact_type, prompt, expected_answer) tuples."""
    facts = []
    for entity in ENTITIES:
        for fact_type, ft_info in FACT_TYPES.items():
            prompt = ft_info['template'].format(entity=entity)
            answer = KNOWN_ANSWERS.get((entity, fact_type))
            facts.append((entity, fact_type, prompt, answer))
    return facts


def get_fact_count():
    return len(ENTITIES) * len(FACT_TYPES)


def get_intent_keywords():
    """Return {fact_type: [keywords]} for intent detection."""
    return {ft: info['keywords'] for ft, info in FACT_TYPES.items()}


def get_response_template(fact_type):
    return FACT_TYPES[fact_type]['response']


if __name__ == '__main__':
    facts = get_all_facts()
    print(f"Knowledge base: {len(ENTITIES)} entities × {len(FACT_TYPES)} fact types "
          f"= {len(facts)} facts")
    print(f"\nEntities: {', '.join(ENTITIES)}")
    print(f"Fact types: {', '.join(FACT_TYPES.keys())}")

    # Check coverage
    n_known = sum(1 for f in facts if f[3] is not None)
    print(f"\nKnown answers: {n_known}/{len(facts)} ({n_known/len(facts)*100:.0f}%)")
