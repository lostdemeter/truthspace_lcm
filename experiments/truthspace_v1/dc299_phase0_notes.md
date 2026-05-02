# DC299 Phase 0 — Concept Mining Notes


## Configuration

  MIN_LEN=3  MAX_LEN=15
  NORM_PERCENTILE_LO=10  NORM_PERCENTILE_HI=90

## Filter Pipeline

Total vocab tokens: 151643
After space-prefix + alpha + length filter: 39823
Norm band [0.68, 0.81]  (p10–p90)
After norm-band filter: 31857
After dedup: 25671

## Norm Distribution

  min=0.68  max=0.81
  mean=0.76  std=0.03
  p10 = 0.72
  p25 = 0.74
  p50 = 0.77
  p75 = 0.79
  p90 = 0.80

## Random Sample of Mined Concepts

  tid= 15957  norm=0.81  word=relief
  tid= 16262  norm=0.81  word=entering
  tid= 16733  norm=0.81  word=airport
  tid= 19712  norm=0.78  word=avg
  tid= 24114  norm=0.78  word=Objects
  tid= 25519  norm=0.74  word=comprom
  tid= 39064  norm=0.78  word=wonders
  tid= 41623  norm=0.74  word=fis
  tid= 44105  norm=0.71  word=borderWidth
  tid= 44551  norm=0.77  word=morality
  tid= 44989  norm=0.76  word=plat
  tid= 45623  norm=0.75  word=encontrar
  tid= 49817  norm=0.79  word=overlapping
  tid= 50878  norm=0.78  word=sadness
  tid= 52084  norm=0.77  word=accustomed
  tid= 53729  norm=0.77  word=formidable
  tid= 62812  norm=0.73  word=Lenin
  tid= 63755  norm=0.75  word=penetrate
  tid= 67845  norm=0.69  word=gibi
  tid= 69737  norm=0.74  word=binaries
  tid= 71464  norm=0.77  word=Falk
  tid= 73915  norm=0.76  word=Dund
  tid= 75183  norm=0.76  word=reversing
  tid= 76061  norm=0.68  word=utiliza
  tid= 76499  norm=0.74  word=supermarkets
  tid= 80236  norm=0.73  word=ingr
  tid= 81922  norm=0.75  word=Locke
  tid= 83714  norm=0.74  word=procession
  tid= 90804  norm=0.73  word=annoyance
  tid= 96250  norm=0.71  word=delt

> **FINDING:** Mined 25671 clean single-token concepts and saved to dc299_phase0_concepts.json


Output: /home/thorin/truthspace-lcm/experiments/truthspace_v1/dc299_phase0_concepts.json
