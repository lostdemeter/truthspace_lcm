# DC299 Phase 3 — Gödel Address Notes


## Semantic Axis Selection

  Quality cliff detected at axis index 214
  Semantic axes selected:  193
  (quality ≥ 0.5, cap ≤ 300)
  Selected axis indices: [0 … 213]

> **FINDING:** Using 193 semantic axes for Gödel addressing.


## Threshold Computation

  threshold_k = median projection of all concept embeddings onto axis_k
  Threshold range: [-0.1695, 0.0140]
  Threshold mean:   -0.0007

## Gödel Address Assignment

  Computing 193-bit addresses for 25671 concepts …
  Done in 10.3s
  Address matrix: (25671, 193)  (concepts × semantic_axes)
  Bit +1 rate: mean=0.500  min=0.500  max=0.500
  (Ideal for information density: ~0.5 per bit)

  Pairwise Hamming (500-concept sample):
    Mean  = 96.5
    Std   = 7.0
    Min   = 53
    Max   = 125

> **FINDING:** Mean pairwise Hamming distance = 96.5 / 193 bits. (Expected for random: 96)


## Relationship Delta Tests

  Seed axis slots: gender=5  capital=2  european=0  romance=3  germanic=4

### Test 1: Gender Analogy (address space)

  king         vs queen         Hamming= 78  gender_bit_flipped=YES
  man          vs woman         Hamming= 76  gender_bit_flipped=YES
  boy          vs girl          Hamming= 62  gender_bit_flipped=YES
  father       vs mother        Hamming= 68  gender_bit_flipped=YES
  brother      vs sister        Hamming= 51  gender_bit_flipped=YES
  actor        vs actress       Hamming= 82  gender_bit_flipped=YES
  prince       vs princess      Hamming= 64  gender_bit_flipped=YES
  hero         vs heroine       Hamming= 83  gender_bit_flipped=YES

  Delta consistency (Hamming between pair-deltas):
    Mean  = 86.8
    Min   = 74
    Max   = 100
    (Lower = more consistent gender transform)

  Analogy: king - man + woman = ?
  Bits flipped in man↔woman delta: 76
  king - man + woman → top neighbours:
    pea                   cos=0.1854
    taxi                  cos=0.1836
    nurse                 cos=0.1823
    knee                  cos=0.1799
    sunrise               cos=0.1765

  Focused test on gender axis (slot 5):
    king        (bit=0, proj=-0.240)  queen       (bit=1, proj=+0.218)  flipped=YES
    man         (bit=0, proj=-0.275)  woman       (bit=1, proj=+0.259)  flipped=YES
    boy         (bit=0, proj=-0.203)  girl        (bit=1, proj=+0.246)  flipped=YES
    father      (bit=0, proj=-0.176)  mother      (bit=1, proj=+0.188)  flipped=YES
    brother     (bit=0, proj=-0.161)  sister      (bit=1, proj=+0.166)  flipped=YES
    actor       (bit=0, proj=-0.228)  actress     (bit=1, proj=+0.293)  flipped=YES
    prince      (bit=0, proj=-0.146)  princess    (bit=1, proj=+0.261)  flipped=YES
    hero        (bit=0, proj=-0.168)  heroine     (bit=1, proj=+0.286)  flipped=YES

### Test 2: Capital-of Transform Consistency

  France       → Paris         Hamming= 78
  Germany      → Berlin        Hamming= 69
  Japan        → Tokyo         Hamming= 71
  China        → Beijing       Hamming= 69
  Italy        → Rome          Hamming= 88
  Spain        → Madrid        Hamming= 80
  Russia       → Moscow        Hamming= 73
  Greece       → Athens        Hamming= 78
  Poland       → Warsaw        Hamming= 62
  Sweden       → Stockholm     Hamming= 69

  Delta consistency (Hamming between country→capital deltas):
    Mean  = 86.6
    Min   = 72
    Max   = 101
    (Lower = capital-of is a single consistent address transform)

  Bits flipped in ALL capital pairs:  1
  Bits flipped in ≥80% of pairs:      1

> **FINDING:** Capital-of transform: 1 invariant bits, 1 bits consistent in ≥80% of pairs. Mean delta Hamming = 86.6.


  Focused test on capital axis (slot 2):
    France      (bit=0, proj=-0.365)  Paris       (bit=1, proj=+0.103)  capital_bit=1
    Germany     (bit=0, proj=-0.341)  Berlin      (bit=1, proj=+0.099)  capital_bit=1
    Japan       (bit=0, proj=-0.297)  Tokyo       (bit=1, proj=+0.147)  capital_bit=1
    China       (bit=0, proj=-0.327)  Beijing     (bit=1, proj=+0.179)  capital_bit=1
    Italy       (bit=0, proj=-0.314)  Rome        (bit=1, proj=+0.048)  capital_bit=1
    Spain       (bit=0, proj=-0.355)  Madrid      (bit=1, proj=+0.105)  capital_bit=1
    Russia      (bit=0, proj=-0.324)  Moscow      (bit=1, proj=+0.155)  capital_bit=1
    Greece      (bit=0, proj=-0.229)  Athens      (bit=1, proj=+0.164)  capital_bit=1
    Poland      (bit=0, proj=-0.224)  Warsaw      (bit=1, proj=+0.201)  capital_bit=1
    Sweden      (bit=0, proj=-0.288)  Stockholm   (bit=1, proj=+0.236)  capital_bit=1

### Test 3: Language Family Clustering


  Focused seed axis projections:
    French         (Romance )  romance_proj=+0.466  germanic_proj=-0.027
    Italian        (Romance )  romance_proj=+0.490  germanic_proj=-0.049
    Spanish        (Romance )  romance_proj=+0.526  germanic_proj=-0.061
    Portuguese     (Romance )  romance_proj=+0.429  germanic_proj=-0.033
    German         (Germanic)  romance_proj=+0.070  germanic_proj=+0.330
    English        (Germanic)  romance_proj=+0.056  germanic_proj=+0.284
    Dutch          (Germanic)  romance_proj=-0.032  germanic_proj=+0.396
    Swedish        (Germanic)  romance_proj=-0.084  germanic_proj=+0.427
    Norwegian      (Germanic)  romance_proj=-0.096  germanic_proj=+0.417
    Japanese       (Asian   )  romance_proj=-0.027  germanic_proj=-0.136
    Chinese        (Asian   )  romance_proj=-0.007  germanic_proj=-0.165
    Korean         (Asian   )  romance_proj=-0.175  germanic_proj=-0.158
    Arabic         (Semitic )  romance_proj=-0.126  germanic_proj=-0.143
    Hebrew         (Semitic )  romance_proj=-0.047  germanic_proj=-0.059
  Romance: 4 words loaded
  Germanic: 5 words loaded
  Asian: 3 words loaded
  Semitic: 2 words loaded

  Romance    within Romance   : mean Hamming = 75.7  (n=6)
  Romance    between Germanic  : mean Hamming = 67.9  (n=20)
  Romance    between Asian     : mean Hamming = 67.8  (n=12)
  Romance    between Semitic   : mean Hamming = 76.5  (n=8)
  Germanic   between Romance   : mean Hamming = 67.9  (n=20)
  Germanic   within Germanic  : mean Hamming = 75.2  (n=10)
  Germanic   between Asian     : mean Hamming = 69.3  (n=15)
  Germanic   between Semitic   : mean Hamming = 76.4  (n=10)
  Asian      between Romance   : mean Hamming = 67.8  (n=12)
  Asian      between Germanic  : mean Hamming = 69.3  (n=15)
  Asian      within Asian     : mean Hamming = 69.3  (n=3)
  Asian      between Semitic   : mean Hamming = 77.7  (n=6)
  Semitic    between Romance   : mean Hamming = 76.5  (n=8)
  Semitic    between Germanic  : mean Hamming = 76.4  (n=10)
  Semitic    between Asian     : mean Hamming = 77.7  (n=6)
  Semitic    within Semitic   : mean Hamming = 60.0  (n=1)

### Test 4: Address Decode (address → nearest vocab token)

  king          →  top=king             self_rank=0
  queen         →  top=queen            self_rank=0
  France        →  top=France           self_rank=0
  Paris         →  top=Paris            self_rank=0
  Tokyo         →  top=Tokyo            self_rank=0
  German        →  top=German           self_rank=0
  French        →  top=French           self_rank=0

## Output

  Output: /home/thorin/truthspace-lcm/experiments/truthspace_v1/dc299_phase3_godel_addresses.json
  Total time: 51.9s
