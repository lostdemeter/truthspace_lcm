# TruthSpace Delta Readability — Field Notes
*Can we read relationship deltas and anchor concepts to truth?*

Loaded 158 concepts, 152064 vocab tokens
Computed 6 anchor directions (truth axes)
  is_european_country: 17+ / 23-
  is_asian_country: 13+ / 17-
  is_capital_city: 28+ / 23-
  is_romance_language: 4+ / 19-
  is_germanic_language: 6+ / 17-
  is_female_gendered: 12+ / 12-

## 1. Reconstruction Test — How Much Do Truth Axes Explain?

Truth axis basis: 6 directions in R^3584

Gram matrix (should be near-identity if orthogonal):
  asian_coun  1.000  0.144  -0.439  -0.055  -0.133  -0.168
  capital_ci  0.144  1.000  -0.002  -0.020  0.027  -0.096
  european_c  -0.439  -0.002  1.000  0.054  0.263  0.044
  female_gen  -0.055  -0.020  0.054  1.000  -0.028  0.026
  germanic_l  -0.133  0.027  0.263  -0.028  1.000  -0.166
  romance_la  -0.168  -0.096  0.044  0.026  -0.166  1.000

Orthogonalized basis: 6 orthonormal directions

### Countries
  France           ||emb||=0.767  ||recon||=0.336  ||resid||=0.689  var_explained=19.23%
  Germany          ||emb||=0.759  ||recon||=0.330  ||resid||=0.684  var_explained=18.92%
  Japan            ||emb||=0.766  ||recon||=0.306  ||resid||=0.703  var_explained=15.92%
  China            ||emb||=0.810  ||recon||=0.376  ||resid||=0.718  var_explained=21.50%
  Egypt            ||emb||=0.756  ||recon||=0.297  ||resid||=0.696  var_explained=15.42%
  Australia        ||emb||=0.781  ||recon||=0.269  ||resid||=0.733  var_explained=11.89%
  India            ||emb||=0.789  ||recon||=0.319  ||resid||=0.722  var_explained=16.34%
  Brazil           ||emb||=0.755  ||recon||=0.292  ||resid||=0.696  var_explained=14.94%
  Korea            ||emb||=0.800  ||recon||=0.344  ||resid||=0.722  var_explained=18.45%
  Italy            ||emb||=0.743  ||recon||=0.291  ||resid||=0.684  var_explained=15.33%
  Spain            ||emb||=0.739  ||recon||=0.326  ||resid||=0.663  var_explained=19.50%
  Russia           ||emb||=0.755  ||recon||=0.281  ||resid||=0.701  var_explained=13.88%
  Poland           ||emb||=0.792  ||recon||=0.289  ||resid||=0.737  var_explained=13.29%
  Norway           ||emb||=0.781  ||recon||=0.312  ||resid||=0.716  var_explained=15.93%
  Sweden           ||emb||=0.725  ||recon||=0.269  ||resid||=0.674  var_explained=13.79%
  Turkey           ||emb||=0.775  ||recon||=0.262  ||resid||=0.729  var_explained=11.40%
  Greece           ||emb||=0.787  ||recon||=0.317  ||resid||=0.720  var_explained=16.24%
  Ireland          ||emb||=0.809  ||recon||=0.272  ||resid||=0.762  var_explained=11.28%
  Finland          ||emb||=0.779  ||recon||=0.258  ||resid||=0.734  var_explained=11.02%
  Denmark          ||emb||=0.780  ||recon||=0.293  ||resid||=0.723  var_explained=14.11%
  Mexico           ||emb||=0.743  ||recon||=0.295  ||resid||=0.681  var_explained=15.81%
  Canada           ||emb||=0.785  ||recon||=0.309  ||resid||=0.721  var_explained=15.51%
  Argentina        ||emb||=0.708  ||recon||=0.272  ||resid||=0.654  var_explained=14.75%
  Nigeria          ||emb||=0.797  ||recon||=0.234  ||resid||=0.762  var_explained=8.60%
  Kenya            ||emb||=0.785  ||recon||=0.209  ||resid||=0.757  var_explained=7.11%
  **Group mean variance explained: 14.81%**

### Capitals
  Paris            ||emb||=0.772  ||recon||=0.113  ||resid||=0.763  var_explained=2.13%
  Berlin           ||emb||=0.747  ||recon||=0.121  ||resid||=0.737  var_explained=2.62%
  Tokyo            ||emb||=0.796  ||recon||=0.145  ||resid||=0.783  var_explained=3.34%
  Beijing          ||emb||=0.789  ||recon||=0.170  ||resid||=0.771  var_explained=4.65%
  Cairo            ||emb||=0.647  ||recon||=0.101  ||resid||=0.639  var_explained=2.45%
  Canberra         ||emb||=0.768  ||recon||=0.130  ||resid||=0.757  var_explained=2.86%
  Delhi            ||emb||=0.801  ||recon||=0.155  ||resid||=0.786  var_explained=3.73%
  Seoul            ||emb||=0.762  ||recon||=0.157  ||resid||=0.746  var_explained=4.24%
  Rome             ||emb||=0.788  ||recon||=0.081  ||resid||=0.784  var_explained=1.06%
  Lisbon           ||emb||=0.759  ||recon||=0.181  ||resid||=0.738  var_explained=5.65%
  Moscow           ||emb||=0.788  ||recon||=0.153  ||resid||=0.773  var_explained=3.78%
  Madrid           ||emb||=0.782  ||recon||=0.127  ||resid||=0.772  var_explained=2.65%
  Athens           ||emb||=0.781  ||recon||=0.161  ||resid||=0.764  var_explained=4.24%
  Ankara           ||emb||=0.730  ||recon||=0.152  ||resid||=0.714  var_explained=4.33%
  Dublin           ||emb||=0.796  ||recon||=0.175  ||resid||=0.777  var_explained=4.85%
  Helsinki         ||emb||=0.768  ||recon||=0.222  ||resid||=0.735  var_explained=8.38%
  Copenhagen       ||emb||=0.770  ||recon||=0.213  ||resid||=0.740  var_explained=7.69%
  Vienna           ||emb||=0.785  ||recon||=0.180  ||resid||=0.764  var_explained=5.29%
  Warsaw           ||emb||=0.768  ||recon||=0.193  ||resid||=0.743  var_explained=6.31%
  Oslo             ||emb||=0.758  ||recon||=0.196  ||resid||=0.732  var_explained=6.70%
  Stockholm        ||emb||=0.771  ||recon||=0.234  ||resid||=0.734  var_explained=9.19%
  Ottawa           ||emb||=0.779  ||recon||=0.146  ||resid||=0.766  var_explained=3.53%
  Lima             ||emb||=0.764  ||recon||=0.131  ||resid||=0.752  var_explained=2.96%
  **Group mean variance explained: 4.46%**

### Languages
  French           ||emb||=0.783  ||recon||=0.370  ||resid||=0.691  var_explained=22.27%
  German           ||emb||=0.777  ||recon||=0.290  ||resid||=0.721  var_explained=13.90%
  Japanese         ||emb||=0.768  ||recon||=0.135  ||resid||=0.756  var_explained=3.07%
  Chinese          ||emb||=0.790  ||recon||=0.171  ||resid||=0.771  var_explained=4.68%
  Spanish          ||emb||=0.772  ||recon||=0.408  ||resid||=0.656  var_explained=27.90%
  Italian          ||emb||=0.773  ||recon||=0.387  ||resid||=0.669  var_explained=25.02%
  Portuguese       ||emb||=0.781  ||recon||=0.340  ||resid||=0.703  var_explained=18.92%
  Russian          ||emb||=0.773  ||recon||=0.105  ||resid||=0.766  var_explained=1.84%
  Arabic           ||emb||=0.790  ||recon||=0.179  ||resid||=0.769  var_explained=5.14%
  English          ||emb||=0.812  ||recon||=0.257  ||resid||=0.770  var_explained=10.02%
  Korean           ||emb||=0.797  ||recon||=0.249  ||resid||=0.757  var_explained=9.78%
  Thai             ||emb||=0.729  ||recon||=0.142  ||resid||=0.715  var_explained=3.82%
  Polish           ||emb||=0.800  ||recon||=0.091  ||resid||=0.795  var_explained=1.29%
  Norwegian        ||emb||=0.794  ||recon||=0.333  ||resid||=0.721  var_explained=17.60%
  Swedish          ||emb||=0.797  ||recon||=0.346  ||resid||=0.718  var_explained=18.86%
  Dutch            ||emb||=0.807  ||recon||=0.329  ||resid||=0.737  var_explained=16.61%
  Greek            ||emb||=0.759  ||recon||=0.140  ||resid||=0.746  var_explained=3.42%
  Turkish          ||emb||=0.792  ||recon||=0.110  ||resid||=0.784  var_explained=1.92%
  Hindi            ||emb||=0.779  ||recon||=0.175  ||resid||=0.759  var_explained=5.03%
  Finnish          ||emb||=0.787  ||recon||=0.139  ||resid||=0.775  var_explained=3.11%
  **Group mean variance explained: 10.71%**

### Gender (M)
  king             ||emb||=0.818  ||recon||=0.201  ||resid||=0.792  var_explained=6.05%
  man              ||emb||=0.867  ||recon||=0.245  ||resid||=0.832  var_explained=7.96%
  boy              ||emb||=0.806  ||recon||=0.166  ||resid||=0.789  var_explained=4.26%
  father           ||emb||=0.774  ||recon||=0.152  ||resid||=0.759  var_explained=3.87%
  brother          ||emb||=0.813  ||recon||=0.149  ||resid||=0.799  var_explained=3.35%
  son              ||emb||=0.856  ||recon||=0.268  ||resid||=0.813  var_explained=9.81%
  husband          ||emb||=0.810  ||recon||=0.159  ||resid||=0.794  var_explained=3.83%
  uncle            ||emb||=0.775  ||recon||=0.139  ||resid||=0.763  var_explained=3.23%
  prince           ||emb||=0.764  ||recon||=0.121  ||resid||=0.755  var_explained=2.51%
  actor            ||emb||=0.808  ||recon||=0.194  ||resid||=0.784  var_explained=5.75%
  **Group mean variance explained: 5.06%**

### Gender (F)
  queen            ||emb||=0.714  ||recon||=0.166  ||resid||=0.694  var_explained=5.41%
  woman            ||emb||=0.775  ||recon||=0.203  ||resid||=0.747  var_explained=6.89%
  girl             ||emb||=0.768  ||recon||=0.190  ||resid||=0.744  var_explained=6.14%
  mother           ||emb||=0.762  ||recon||=0.164  ||resid||=0.744  var_explained=4.66%
  sister           ||emb||=0.814  ||recon||=0.156  ||resid||=0.799  var_explained=3.68%
  daughter         ||emb||=0.722  ||recon||=0.161  ||resid||=0.704  var_explained=4.98%
  wife             ||emb||=0.760  ||recon||=0.185  ||resid||=0.737  var_explained=5.95%
  aunt             ||emb||=0.774  ||recon||=0.188  ||resid||=0.751  var_explained=5.93%
  princess         ||emb||=0.765  ||recon||=0.213  ||resid||=0.735  var_explained=7.77%
  actress          ||emb||=0.808  ||recon||=0.244  ||resid||=0.770  var_explained=9.12%
  **Group mean variance explained: 6.05%**

### Overall Reconstruction Summary
  Concepts tested: 88
  Mean variance explained by 6 truth axes: 9.069%
  Median: 6.505%
  Min: 1.055%  Max: 27.899%

  Per group:
    Countries      : 14.806%
    Capitals       : 4.462%
    Languages      : 10.709%
    Gender (M)     : 5.063%
    Gender (F)     : 6.051%

> **FINDING:** 6 truth axes explain 9.069% of embedding variance. The remaining 90.931% is in the residual. If concepts are purely platonic compounds, this residual should be unstructured noise or just 'more platonic ideals we haven't found yet'.


## 2. Residual Structure — Is What's Left Structured or Noise?

Residual matrix: (88, 3584)

### 2a. Residual SVD — Effective Dimensionality
  50% cumulative variance in 27 dimensions
  80% cumulative variance in 56 dimensions
  90% cumulative variance in 68 dimensions
  95% cumulative variance in 75 dimensions
  99% cumulative variance in 82 dimensions

  Top 20 singular values:
    S[ 0] = 1.5369  cumvar = 5.50%
    S[ 1] = 1.3187  cumvar = 9.55%
    S[ 2] = 1.0781  cumvar = 12.26%
    S[ 3] = 1.0022  cumvar = 14.60%
    S[ 4] = 0.9692  cumvar = 16.78%
    S[ 5] = 0.9406  cumvar = 18.85%
    S[ 6] = 0.9120  cumvar = 20.78%
    S[ 7] = 0.8978  cumvar = 22.66%
    S[ 8] = 0.8750  cumvar = 24.44%
    S[ 9] = 0.8665  cumvar = 26.19%
    S[10] = 0.8558  cumvar = 27.90%
    S[11] = 0.8512  cumvar = 29.58%
    S[12] = 0.8390  cumvar = 31.22%
    S[13] = 0.8245  cumvar = 32.81%
    S[14] = 0.8126  cumvar = 34.34%
    S[15] = 0.8100  cumvar = 35.87%
    S[16] = 0.8019  cumvar = 37.37%
    S[17] = 0.7977  cumvar = 38.85%
    S[18] = 0.7919  cumvar = 40.31%
    S[19] = 0.7873  cumvar = 41.76%
  50% variance: real=27 dims, random=38 dims
  80% variance: real=56 dims, random=66 dims
  90% variance: real=68 dims, random=76 dims

> **FINDING:** Residuals are LOWER-DIMENSIONAL than random noise — they are STRUCTURED. The residual space is not noise; it contains geometric information that truth axes don't capture.


### 2b. Nearest-Neighbor Test — Can Residuals Identify Concepts?

Top-5 nearest neighbors: residual space vs original space
Concept         | Residual NN                                        | Original NN                                       
------------------------------------------------------------------------------------------------------------------------
France          | Paris(0.271), Beijing(0.204), Tokyo(0.195), Japan(0.194), Italy(0.185) | Italy(0.321), Germany(0.309), Spain(0.304), Greece(0.267), Russia(0.256)
Paris           | France(0.271), Italy(0.262), Germany(0.220), Brazil(0.217), Berlin(0.210) | Berlin(0.223), Italy(0.220), French(0.215), France(0.211), Athens(0.189)
French          | Japanese(0.348), Turkish(0.345), Dutch(0.342), German(0.333), Chinese(0.333) | Spanish(0.347), German(0.326), Italian(0.295), Japanese(0.286), Dutch(0.283)
king            | queen(0.237), woman(0.201), girl(0.147), princess(0.135), Beijing(0.124) | queen(0.172), Beijing(0.130), son(0.128), woman(0.127), man(0.120)
queen           | king(0.237), father(0.171), prince(0.165), boy(0.147), son(0.143) | woman(0.182), daughter(0.180), princess(0.175), king(0.172), Spain(0.147)
Japan           | Tokyo(0.267), Italy(0.256), Germany(0.241), Seoul(0.231), Spain(0.231) | China(0.321), Korea(0.280), Italy(0.260), Russia(0.245), Germany(0.245)
Tokyo           | Korea(0.293), Beijing(0.271), Japan(0.267), Poland(0.250), Seoul(0.242) | Beijing(0.297), Seoul(0.267), Korea(0.248), Moscow(0.238), Dublin(0.221)
Japanese        | French(0.348), Chinese(0.340), German(0.326), Spanish(0.319), Dutch(0.306) | Chinese(0.364), French(0.286), Russian(0.263), Korean(0.257), German(0.256)
man             | woman(0.330), son(0.262), girl(0.228), wife(0.171), actress(0.169) | son(0.326), woman(0.237), boy(0.178), girl(0.146), father(0.135)
woman           | man(0.330), boy(0.255), son(0.211), king(0.201), girl(0.192) | girl(0.244), man(0.237), daughter(0.188), boy(0.188), mother(0.186)
Germany         | Berlin(0.280), Japan(0.241), China(0.239), Italy(0.238), German(0.237) | Italy(0.362), France(0.309), Spain(0.300), Russia(0.293), Denmark(0.283)
Berlin          | Germany(0.280), Brazil(0.236), German(0.229), Japan(0.217), China(0.213) | Germany(0.246), German(0.226), Paris(0.223), Dublin(0.195), Vienna(0.194)
German          | Chinese(0.344), Russian(0.335), French(0.333), Italian(0.327), Japanese(0.326) | French(0.326), Italian(0.318), Spanish(0.297), Russian(0.281), Chinese(0.266)
boy             | girl(0.358), woman(0.255), daughter(0.172), actress(0.154), wife(0.149) | girl(0.289), woman(0.188), man(0.178), son(0.166), Moscow(0.132)
girl            | boy(0.358), man(0.228), woman(0.192), son(0.188), father(0.176) | boy(0.289), woman(0.244), sister(0.172), daughter(0.164), wife(0.154)

### 2c. Category Clustering — Do Semantic Groups Cluster in Residual Space?

  **Residual space:**
    Within-category mean cos: 0.1539 (std 0.0741)
    Between-category mean cos: 0.0885 (std 0.0673)
    Separation (within - between): 0.0654

  **Original space:**
    Within-category mean cos: 0.1788 (std 0.0616)
    Between-category mean cos: 0.0728 (std 0.0555)
    Separation (within - between): 0.1060

### 2d. Residual Relationship Coherence
If France_residual + capital_residual_delta ≈ Paris_residual,
then the residual encodes relational structure beyond truth axes.

  capital: Egypt + delta → Cairo
    Predicted↔Actual residual cos: 0.0810
    Random baseline cos: 0.0868
    Residual delta consistency (mean pairwise cos): -0.0220
  gender: father + delta → mother
    Predicted↔Actual residual cos: 0.2896
    Random baseline cos: 0.0275
    Residual delta consistency (mean pairwise cos): -0.1120
  language: Spain + delta → Spanish
    Predicted↔Actual residual cos: 0.3135
    Random baseline cos: 0.1371
    Residual delta consistency (mean pairwise cos): 0.1712

## 3. Scale Analysis — How Many Platonic Ideals Exist?

Current: 6 axes → 9.069% variance explained

### PCA of Concept Embeddings (optimal K directions)
    1 PCA dims → 4.99% variance explained
    2 PCA dims → 8.82% variance explained
    3 PCA dims → 12.36% variance explained
    6 PCA dims → 19.68% variance explained
   10 PCA dims → 27.24% variance explained
   15 PCA dims → 35.14% variance explained
   20 PCA dims → 42.18% variance explained
   30 PCA dims → 54.44% variance explained
   50 PCA dims → 74.20% variance explained
   80 PCA dims → 96.12% variance explained

  Our 6 truth axes explain 9.069% variance
  Equivalent to 3 PCA dimensions

### Truth Axes vs PCA Components
  is_asian_country:
    PC4: coeff=0.3980 (cumvar=17.4%)
    PC3: coeff=-0.3129 (cumvar=15.0%)
    PC15: coeff=-0.1545 (cumvar=36.6%)
    PC36: coeff=-0.1495 (cumvar=62.0%)
    PC6: coeff=0.1391 (cumvar=21.8%)
  is_capital_city:
    PC2: coeff=0.8689 (cumvar=12.4%)
    PC3: coeff=-0.2508 (cumvar=15.0%)
    PC5: coeff=0.2102 (cumvar=19.7%)
    PC6: coeff=0.1689 (cumvar=21.8%)
    PC1: coeff=-0.1330 (cumvar=8.8%)
  is_european_country:
    PC3: coeff=0.5026 (cumvar=15.0%)
    PC4: coeff=-0.3723 (cumvar=17.4%)
    PC11: coeff=0.2057 (cumvar=30.5%)
    PC2: coeff=0.1929 (cumvar=12.4%)
    PC38: coeff=-0.1915 (cumvar=64.0%)
  is_female_gendered:
    PC10: coeff=0.4922 (cumvar=28.9%)
    PC5: coeff=0.2711 (cumvar=19.7%)
    PC26: coeff=0.2612 (cumvar=51.0%)
    PC7: coeff=-0.2376 (cumvar=23.7%)
    PC6: coeff=-0.2209 (cumvar=21.8%)
  is_germanic_language:
    PC3: coeff=0.3756 (cumvar=15.0%)
    PC5: coeff=-0.3279 (cumvar=19.7%)
    PC39: coeff=0.2678 (cumvar=65.0%)
    PC35: coeff=-0.2228 (cumvar=61.0%)
    PC9: coeff=0.2006 (cumvar=27.2%)
  is_romance_language:
    PC4: coeff=-0.3872 (cumvar=17.4%)
    PC23: coeff=0.2497 (cumvar=47.3%)
    PC2: coeff=-0.2342 (cumvar=12.4%)
    PC34: coeff=-0.2267 (cumvar=59.9%)
    PC16: coeff=-0.2046 (cumvar=38.0%)

### Simulated Scaling: Variance Explained vs Number of Axes
Using PCA as a proxy for 'optimal truth axis discovery':

    6 axes → 19.68% variance
   10 axes → 27.24% variance
   17 axes → 38.04% variance
   30 axes → 54.44% variance
   50 axes → 74.20% variance
   88 axes → 100.00% variance

> **FINDING:** PCA analysis reveals the intrinsic dimensionality of the concept space. If ~K PCA dims explain >99% of variance, then ~K platonic ideals would suffice to fully describe all concepts. The gap between our 6 truth axes and optimal PCA tells us how many more platonic ideals remain to be discovered.


## 4. The Definitive Test — Concept Identity from Residuals Alone

For each concept, we find the closest vocab token to its RESIDUAL vector.
If residuals are noise, the matches should be random.
If structured, matches should be semantically meaningful.

  France       residual → France(0.373), Paris(0.268), Paris(0.249), france(0.223), Singapore(0.217)
  Paris        residual → Paris(0.530), paris(0.315), å·´é»İ(0.251), France(0.243), Italy(0.241)
  French       residual → French(0.498), french(0.382), Japanese(0.343), Turkish(0.342), rench(0.326)
  Japan        residual → Japan(0.420), japan(0.277), Tokyo(0.263), Italy(0.236), Seoul(0.226)
  Tokyo        residual → Japan(0.335), Osaka(0.269), Korea(0.265), Beijing(0.264), yo(0.259)
  Japanese     residual → Japanese(0.515), japanese(0.386), Chinese(0.332), apanese(0.310), French(0.307)
  king         residual → King(0.355), king(0.269), King(0.262), KING(0.254), kings(0.240)
  queen        residual → queen(0.388), Queen(0.350), Queen(0.335), queens(0.263), king(0.259)
  man          residual → MAN(0.397), woman(0.318), mans(0.313), man(0.299), -man(0.290)
  woman        residual → woman(0.402), Woman(0.342), man(0.317), Woman(0.312), women(0.306)
  boy          residual → boy(0.373), Boy(0.353), girl(0.347), boys(0.344), -boy(0.340)
  girl         residual → girl(0.379), boy(0.350), Girl(0.337), girls(0.334), Girl(0.324)
  Germany      residual → Germany(0.391), Berlin(0.277), Berlin(0.276), germany(0.243), å¾·åĽ½(0.236)
  Berlin       residual → Berlin(0.468), berlin(0.302), Germany(0.253), London(0.244), æŁıæŀĹ(0.229)
  German       residual → German(0.478), Chinese(0.335), german(0.333), Russian(0.332), Japanese(0.321)
  India        residual → India(0.414), Delhi(0.274), Mumbai(0.248), india(0.248), Indian(0.232)
  Delhi        residual → Mumbai(0.303), Kerala(0.277), India(0.263), umbai(0.258), India(0.256)

### Self-Retrieval from Residuals
Can the residual alone find the original concept in the full vocabulary?

  Mean rank: 0.0 (out of ~150K vocab tokens)
  Median rank: 0.0
  In top-10: 88/88
  In top-100: 88/88
  In top-1000: 88/88

  Best retrievals (lowest rank = most identifiable from residual):
    Ankara          rank=0
    Arabic          rank=0
    Argentina       rank=0
    Athens          rank=0
    Australia       rank=0
    Beijing         rank=0
    Berlin          rank=0
    Brazil          rank=0
    Cairo           rank=0
    Canada          rank=0

  Worst retrievals (highest rank = least identifiable):
    man             rank=0
    mother          rank=0
    prince          rank=0
    princess        rank=0
    queen           rank=0
    sister          rank=0
    son             rank=0
    uncle           rank=0
    wife            rank=0
    woman           rank=0

> **FINDING:** Residuals can retrieve concepts at median rank 0 out of ~150K tokens. This means the residual is HIGHLY STRUCTURED — it encodes concept identity far beyond what 6 truth axes capture. The residual is NOT noise. It is 'more platonic ideals we haven't named yet'.


## 5. Synthesis — Are Concepts Purely Platonic Compounds?

### Key Numbers
  Truth axes (6): explain 9.069% of embedding variance
  Residual retrieval: median rank 0 / ~150K
  PCA dims for 90% variance: 71
  PCA dims for 95% variance: 79
  PCA dims for 99% variance: 86

### Interpretation
The residual is EXTREMELY structured. After removing the projection
onto 6 truth axes, what remains can still identify the concept
at median rank 0 in a 150K vocabulary.

This means one of two things:
  (a) There are MANY more platonic ideals (truth axes) that we
      haven't discovered yet, and the residual is just the
      compound of those undiscovered ideals. OR
  (b) There is NON-PLATONIC structure — something about concepts
      that cannot be decomposed into binary truth axes.

The PCA analysis suggests ~79 dimensions capture 95% of
concept variance. If each dimension corresponds to a platonic ideal,
then ~79 ideals would nearly fully describe all concepts.
This is still a finite, tractable number.

### The Answer

**6 truth axes explain 9.1% of variance.** The remaining
90.9% is structured (residual retrieval works).
This is consistent with concepts being compounds of ~79 platonic ideals,
of which we've identified 6. The residual is 'undiscovered platonic structure',
not non-platonic noise.


Completed in 41.8s
