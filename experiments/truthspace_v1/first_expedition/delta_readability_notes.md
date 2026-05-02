# TruthSpace Delta Readability — Field Notes
*Can we read relationship deltas and anchor concepts to truth?*

  capital: 5 train, 7 test pairs
  language: 5 train, 3 test pairs
  gender: 4 train, 3 test pairs

## 1. Dimension Semantics — What Do Individual Dims Encode?


### Relationship: capital
Top 10 most informative dims (by |shift|/CV):

**Dim 1486** — shift=0.013528, CV=0.069, score=0.1949
  HIGH: Ð¾Ð±ÑĢÐ°Ð¶(0.0527), Crack(0.0523), "This(0.0521), ä½Ĩä¹Ł(0.0505), é²ľè¡Ģ(0.0503), rosse(0.0490), richer(0.0490), cff(0.0489)
  LOW:  steward(-0.0511), çŃ¹(-0.0507), Sum(-0.0503), _system(-0.0500), éĢĤå®ľ(-0.0498), åįģåĽĽäºĶ(-0.0483), Miguel(-0.0481), milit(-0.0481)
  Entities: mean=-0.0015, Answers: mean=0.0120, Δ=0.0135

**Dim 926** — shift=0.032817, CV=0.258, score=0.1274
  HIGH: Vincent(0.0551), instanceof(0.0537), decoding(0.0505), MF(0.0464), because(0.0445), components(0.0440), Eq(0.0440), Mult(0.0430)
  LOW:  ras(-0.0559), utor(-0.0551), =com(-0.0543), éĬ(-0.0525), ahr(-0.0515), è¿ĳäºĽå¹´(-0.0513), ahl(-0.0513), å¼Ľ(-0.0511)
  Entities: mean=-0.0184, Answers: mean=0.0144, Δ=0.0328

**Dim 2973** — shift=-0.013702, CV=0.124, score=0.1107
  HIGH: åı¯æĥľ(0.0645), displ(0.0507), æĸ°ä¸ļæĢģ(0.0500), éł(0.0489), åĲİæĿ¥(0.0489), è´¢æĶ¿éĥ¨(0.0483), ocab(0.0471), é¢Ĩå¯¼çıŃåŃĲ(0.0469)
  LOW:  é¦ĸ(-0.0670), ay(-0.0621), ãĢĳ(-0.0587), et(-0.0585), æī¾åĪ°(-0.0564), ID(-0.0549), CC(-0.0549), _presence(-0.0537)
  Entities: mean=-0.0020, Answers: mean=-0.0157, Δ=-0.0137

**Dim 2498** — shift=-0.028541, CV=0.279, score=0.1023
  HIGH: decoration(0.0579), irritation(0.0576), instruction(0.0555), teachers(0.0523), vocabulary(0.0511), ç½ĳçº¦è½¦(0.0511), Assistance(0.0507), activity(0.0507)
  LOW:  )}(-0.0543), Ð½Ðµ(-0.0533), æ¯ĭ(-0.0521), "),(-0.0511), ä¸»è¦ģæľī(-0.0503), Coast(-0.0498), sm(-0.0492), Ð¿Ð¾ÑĤ(-0.0492)
  Entities: mean=0.0190, Answers: mean=-0.0095, Δ=-0.0285

**Dim 2273** — shift=0.028165, CV=0.282, score=0.0998
  HIGH: gif(0.0543), è£ħéħįå¼ı(0.0527), dal(0.0521), diamond(0.0517), briefing(0.0513), èģļ(0.0500), pest(0.0498), (a(0.0492)
  LOW:  ÐºÐ°Ð¶Ð´Ð¾Ð³Ð¾(-0.0574), -line(-0.0535), cherche(-0.0505), ("//(-0.0500), uly(-0.0500), Derm(-0.0498), ä¸Ģä½į(-0.0490), æŃ¤æĹ¶(-0.0490)
  Entities: mean=-0.0166, Answers: mean=0.0116, Δ=0.0282

**Dim 873** — shift=-0.025630, CV=0.275, score=0.0932
  HIGH: ä¸įè§ģ(0.0549), ä¸įåĲ«(0.0535), çŁ®(0.0515), EO(0.0507), Prophet(0.0496), Ã¦(0.0490), WE(0.0490), sque(0.0487)
  LOW:  chart(-0.0568), Nov(-0.0547), belum(-0.0531), grand(-0.0483), Broad(-0.0476), priorities(-0.0471), .html(-0.0464), ä¸¤åĳ¨(-0.0462)
  Entities: mean=0.0098, Answers: mean=-0.0159, Δ=-0.0256

**Dim 2661** — shift=0.032040, CV=0.345, score=0.0927
  HIGH: äººä½ĵ(0.0610), .contact(0.0610), stationary(0.0535), Callable(0.0505), (object(0.0500), .datatables(0.0489), ä¹Łè®©(0.0478), _velocity(0.0478)
  LOW:  QB(-0.0579), ä»ģ(-0.0531), å·(-0.0507), Japan(-0.0503), è¡¥(-0.0490), Brow(-0.0490), item(-0.0483), dig(-0.0481)
  Entities: mean=-0.0303, Answers: mean=0.0017, Δ=0.0320

**Dim 782** — shift=0.024909, CV=0.276, score=0.0902
  HIGH: -tabs(0.0631), neural(0.0585), åħ±èµ¢(0.0535), microbial(0.0527), Value(0.0513), è¿ŀèĥľ(0.0511), å®Įæķ´(0.0511), ä½ĵåŀĭ(0.0503)
  LOW:  .Register(-0.0515), .av(-0.0496), .unpack(-0.0489), Jud(-0.0487), ãĤı(-0.0483), ãĥī(-0.0481), åıĳåĩº(-0.0464), _ui(-0.0458)
  Entities: mean=-0.0102, Answers: mean=0.0147, Δ=0.0249

**Dim 2380** — shift=-0.020401, CV=0.228, score=0.0894
  HIGH: ä¹ħäºĨ(0.0527), nhá»¯ng(0.0500), GG(0.0489), Tea(0.0467), zm(0.0455), chocolate(0.0455), Um(0.0451), sd(0.0451)
  LOW:  free(-0.0603), permissions(-0.0564), Converter(-0.0561), Attrib(-0.0537), .validators(-0.0535), see(-0.0523), cont(-0.0513), free(-0.0500)
  Entities: mean=0.0186, Answers: mean=-0.0018, Δ=-0.0204

**Dim 2449** — shift=0.023771, CV=0.296, score=0.0804
  HIGH: /*ĊĊ(0.0566), Â©(0.0555), .reverse(0.0539), å¾Īå¤ļäºĭæĥħ(0.0539), visions(0.0492), =Ċ(0.0487), ä½łä¼ļ(0.0487), TC(0.0487)
  LOW:  _COMPILER(-0.0564), rompt(-0.0527), Ð´Ð¾Ð±(-0.0523), .xpath(-0.0517), Up(-0.0513), scrambling(-0.0507), pad(-0.0507), Tray(-0.0503)
  Entities: mean=-0.0140, Answers: mean=0.0098, Δ=0.0238

---

### Relationship: language
Top 10 most informative dims (by |shift|/CV):

**Dim 2990** — shift=0.019625, CV=0.116, score=0.1694
  HIGH: (int(0.0714), bubble(0.0564), _classes(0.0527), Segoe(0.0523), clientes(0.0521), èĸ°(0.0517), .Decimal(0.0515), (low(0.0507)
  LOW:  Harris(-0.0551), fabrication(-0.0505), PGA(-0.0500), attainment(-0.0487), precipitation(-0.0483), Industries(-0.0476), broadcasting(-0.0474), Wonder(-0.0471)
  Entities: mean=-0.0062, Answers: mean=0.0134, Δ=0.0196

**Dim 3139** — shift=0.026271, CV=0.199, score=0.1318
  HIGH: cale(0.0594), Lin(0.0568), ILE(0.0568), ä¸įåı¯æĪĸç¼º(0.0566), Ðŀ(0.0531), å®Ŀè´µçļĦ(0.0531), (list(0.0525), audi(0.0515)
  LOW:  \Request(-0.0551), Songs(-0.0547), hoje(-0.0547), confusing(-0.0523), Personnel(-0.0500), raq(-0.0492), eyed(-0.0487), ra(-0.0478)
  Entities: mean=-0.0196, Answers: mean=0.0067, Δ=0.0263

**Dim 568** — shift=0.023723, CV=0.195, score=0.1215
  HIGH: narrower(0.0631), reaches(0.0539), _as(0.0490), _sequences(0.0483), è´µéĩĳå±ŀ(0.0481), extension(0.0476), ÑĤÐ¾Ð²Ð°ÑĢÐ¾Ð²(0.0471), Ñģ(0.0462)
  LOW:  connectivity(-0.0543), FX(-0.0537), ä¿¡æģ¯(-0.0535), æľºéģĩ(-0.0503), thÃªm(-0.0500), \Ċ(-0.0500), Technology(-0.0492), speech(-0.0487)
  Entities: mean=-0.0307, Answers: mean=-0.0070, Δ=0.0237

**Dim 2535** — shift=-0.029314, CV=0.244, score=0.1203
  HIGH: å¼ķå¯¼(0.0574), exe(0.0539), Beth(0.0531), izer(0.0527), ä¹ĭéĻħ(0.0525), åĬł(0.0498), Publishing(0.0498), åĢ¡å¯¼(0.0492)
  LOW:  Ã¡veis(-0.0581), Smart(-0.0525), æ¾³(-0.0523), SMART(-0.0517), straight(-0.0496), quences(-0.0490), _ready(-0.0478), S(-0.0476)
  Entities: mean=0.0107, Answers: mean=-0.0186, Δ=-0.0293

**Dim 1000** — shift=0.023226, CV=0.194, score=0.1197
  HIGH: ACT(0.0515), gameplay(0.0478), ä¸¤ä¸ªäºº(0.0474), æľįåĬ¡å¹³åı°(0.0471), sought(0.0471), '.Ċ(0.0464), thrill(0.0464), >čĊ(0.0462)
  LOW:  ilda(-0.0621), charges(-0.0568), compile(-0.0549), conject(-0.0537), æĺĵ(-0.0525), seconds(-0.0517), å¹´çļĦ(-0.0515), Nz(-0.0505)
  Entities: mean=-0.0092, Answers: mean=0.0140, Δ=0.0232

**Dim 1095** — shift=0.028892, CV=0.247, score=0.1168
  HIGH: å¿ħå°Ĩ(0.0557), Empire(0.0551), heights(0.0531), compress(0.0531), Cyc(0.0515), refuge(0.0500), light(0.0492), ÑıÐ¼Ð¸(0.0489)
  LOW:  uda(-0.0626), èµĦè®¯(-0.0594), -margin(-0.0581), uggest(-0.0574), åºĶæľīçļĦ(-0.0535), éĴ®(-0.0533), å§¿(-0.0527), ä¸Ģéģĵ(-0.0517)
  Entities: mean=-0.0081, Answers: mean=0.0208, Δ=0.0289

**Dim 1362** — shift=0.030286, CV=0.261, score=0.1162
  HIGH: gs(0.0503), convertible(0.0478), ranges(0.0471), .bridge(0.0469), Ass(0.0467), Thread(0.0464), é¡¶(0.0464), æİĺ(0.0462)
  LOW:  Fred(-0.0626), cal(-0.0543), Sony(-0.0533), æģķ(-0.0498), calidad(-0.0498), lene(-0.0496), ç»ıéªĮåĴĮ(-0.0492), Cour(-0.0492)
  Entities: mean=-0.0185, Answers: mean=0.0118, Δ=0.0303

**Dim 2236** — shift=0.022580, CV=0.204, score=0.1105
  HIGH: gland(0.0576), æ¶īå«Į(0.0539), è´ŀ(0.0515), Noel(0.0487), åħ³éĶ®è¯į(0.0483), veh(0.0481), substantive(0.0471), Hastings(0.0469)
  LOW:  athom(-0.0513), .Dropout(-0.0511), probl(-0.0505), å¦Ĥä»Ĭ(-0.0492), quist(-0.0464), æıĲæĮ¯(-0.0462), iguous(-0.0458), total(-0.0458)
  Entities: mean=-0.0100, Answers: mean=0.0125, Δ=0.0226

**Dim 1278** — shift=-0.031826, CV=0.322, score=0.0987
  HIGH: Spain(0.0543), thinkers(0.0533), å·¥å¤«(0.0507), Aviation(0.0483), æłĳæľ¨(0.0474), ç¥ŀæĥħ(0.0464), Red(0.0464), Features(0.0464)
  LOW:  supply(-0.0533), (D(-0.0525), æ½ľæ°´(-0.0513), utom(-0.0503), ond(-0.0500), kut(-0.0492), æĢ¥è¯Ĭ(-0.0487), pas(-0.0483)
  Entities: mean=0.0235, Answers: mean=-0.0083, Δ=-0.0318

**Dim 483** — shift=0.026059, CV=0.266, score=0.0978
  HIGH: contempor(0.0539), ä¸Ģåı·(0.0535), èĩªå·±çļĦ(0.0511), ":(0.0496), shouldn(0.0483), éĤ£ä¸ªäºº(0.0478), .CharField(0.0478), Diamond(0.0445)
  LOW:  %(-0.0521), ç»ĥä¹ł(-0.0517), meny(-0.0513), remain(-0.0511), ká»³(-0.0507), åĲĥ(-0.0490), <>Ċ(-0.0487), link(-0.0483)
  Entities: mean=-0.0152, Answers: mean=0.0109, Δ=0.0261

---

### Relationship: gender
Top 10 most informative dims (by |shift|/CV):

**Dim 2834** — shift=0.016259, CV=0.074, score=0.2192
  HIGH: \[(0.0505), å°¾(0.0481), å¹ķ(0.0464), èŀį(0.0446), yet(0.0441), ç»Ļ(0.0440), åĶ®(0.0436), tragedy(0.0432)
  LOW:  contaminated(-0.0592), .sp(-0.0592), cool(-0.0585), perfected(-0.0564), çľŁæĥħ(-0.0547), disabling(-0.0545), ly(-0.0537), æĹ¥åĨĽ(-0.0535)
  Entities: mean=-0.0182, Answers: mean=-0.0019, Δ=0.0163

**Dim 2339** — shift=0.027856, CV=0.159, score=0.1756
  HIGH: Available(0.0535), .L(0.0525), å²ŃåįĹ(0.0521), array(0.0515), System(0.0511), beautiful(0.0505), Al(0.0505), Mart(0.0498)
  LOW:  æĹ¶ä¸įæĹ¶(-0.0557), éĿĴèĽĻ(-0.0521), Ton(-0.0498), çģ«çģ¾(-0.0498), æĹ¶éĹ´æ®µ(-0.0487), tourism(-0.0487), çĸŁ(-0.0487), Wilkinson(-0.0483)
  Entities: mean=-0.0181, Answers: mean=0.0098, Δ=0.0279

**Dim 974** — shift=-0.021145, CV=0.132, score=0.1598
  HIGH: æ¯«(0.0487), æĭĵ(0.0476), inconsistency(0.0474), Construction(0.0474), _annotation(0.0471), Touch(0.0451), ÑģÑĢÐ°Ð²Ð½(0.0446), æĳĩ(0.0446)
  LOW:  les(-0.0688), cos(-0.0557), fecha(-0.0533), à¸¥(-0.0531), fees(-0.0517), ge(-0.0511), omy(-0.0511), į(-0.0503)
  Entities: mean=0.0025, Answers: mean=-0.0187, Δ=-0.0211

**Dim 2695** — shift=-0.012565, CV=0.082, score=0.1525
  HIGH: ìľ¼ë¡ľ(0.0557), =true(0.0543), '],Ċ(0.0507), right(0.0505), reporting(0.0500), Queen(0.0498), Bros(0.0490), æ¨Ļ(0.0489)
  LOW:  Ã£(-0.0496), èĦĨå¼±(-0.0483), IST(-0.0478), æĻļé¥Ń(-0.0476), Hol(-0.0471), åĢĻéĢī(-0.0455), Ã¤(-0.0446), Vulner(-0.0445)
  Entities: mean=0.0107, Answers: mean=-0.0019, Δ=-0.0126

**Dim 289** — shift=-0.023522, CV=0.183, score=0.1288
  HIGH: taking(0.0594), æİĴåĲįç¬¬(0.0585), ä¸Ĭè°ĥ(0.0572), need(0.0572), AD(0.0555), ä»»(0.0551), excluding(0.0549), se(0.0543)
  LOW:  Má»Ĺi(-0.0564), åĺİ(-0.0533), Abs(-0.0507), ISP(-0.0503), _fs(-0.0503), reflux(-0.0496), _success(-0.0492), aphrag(-0.0492)
  Entities: mean=0.0089, Answers: mean=-0.0146, Δ=-0.0235

**Dim 901** — shift=-0.018459, CV=0.144, score=0.1279
  HIGH: because(0.0525), sessions(0.0521), rooms(0.0507), Â°(0.0507), i(0.0505), å¼ºåĮĸ(0.0492), Tong(0.0490), dataset(0.0489)
  LOW:  orda(-0.0521), recounted(-0.0507), hostility(-0.0503), æĹ¥åĩĮæĻ¨(-0.0503), æĽĻ(-0.0496), andscape(-0.0469), èĥ½æī¾åĪ°(-0.0467), ä¸ĵå®¶ç»Ħ(-0.0458)
  Entities: mean=0.0132, Answers: mean=-0.0052, Δ=-0.0185

**Dim 734** — shift=-0.022914, CV=0.195, score=0.1175
  HIGH: sting(0.0564), è¢Ńåĩ»(0.0557), æĹłç¼Ŀ(0.0547), Ľå»º(0.0545), _delay(0.0500), èĥľåĪ©(0.0500), Spike(0.0498), æĹ¥åĩĮæĻ¨(0.0498)
  LOW:  Architecture(-0.0579), orthogonal(-0.0531), poder(-0.0527), Ops(-0.0523), æĭ¥æľī(-0.0511), jsonify(-0.0507), asserts(-0.0500), æĪĲä¸ºäºĨ(-0.0498)
  Entities: mean=0.0054, Answers: mean=-0.0175, Δ=-0.0229

**Dim 1133** — shift=0.018751, CV=0.160, score=0.1169
  HIGH: sql(0.0596), "%(0.0585), configure(0.0581), glow(0.0543), sql(0.0517), encryption(0.0517), Square(0.0500), control(0.0496)
  LOW:  táº¡i(-0.0535), è¿ĳæľŁ(-0.0517), Director(-0.0505), .ct(-0.0498), rotation(-0.0492), opsis(-0.0490), çİ°å®ŀä¸Ń(-0.0487), ìķ¼(-0.0483)
  Entities: mean=-0.0172, Answers: mean=0.0015, Δ=0.0188

**Dim 1209** — shift=0.020657, CV=0.181, score=0.1144
  HIGH: administr(0.0574), ->Ċ(0.0574), ÐŀÑĤ(0.0564), affordability(0.0531), retirement(0.0517), ):Ċ(0.0507), ORM(0.0503), Overse(0.0503)
  LOW:  å¢¨(-0.0527), æ¿Ģåıĳ(-0.0511), é«ĺè¾¾(-0.0503), iking(-0.0500), æ³¢åĬ¨(-0.0498), pencils(-0.0496), iez(-0.0492), æ°ĽåĽ´(-0.0481)
  Entities: mean=-0.0096, Answers: mean=0.0110, Δ=0.0207

**Dim 1070** — shift=-0.019032, CV=0.174, score=0.1094
  HIGH: yal(0.0576), å´©(0.0564), ob(0.0533), Text(0.0525), ob(0.0517), æĭĸ(0.0507), pan(0.0507), åįĩ(0.0503)
  LOW:  _positive(-0.0543), _upper(-0.0521), .username(-0.0487), virtue(-0.0487), ToString(-0.0467), Europa(-0.0458), Brandon(-0.0457), menos(-0.0451)
  Entities: mean=0.0070, Answers: mean=-0.0120, Δ=-0.0190

---

## 2. Delta Direction Interpretation — What Does Each Delta Mean?


### Relationship: capital
Delta norm: 0.5074

**Most aligned with capital delta direction** (these tokens point where the delta goes):
   1. Beijing               proj=0.2520
   2. Tokyo                 proj=0.2340
   3. Paris                 proj=0.2093
   4. Berlin                proj=0.1895
   5. cairo                 proj=0.1627
   6. Paris                 proj=0.1391
   7. Berlin                proj=0.1373
   8. @@                    proj=0.1125
   9. Helsinki              proj=0.1074
  10. Beirut                proj=0.1061
  11. Nairobi               proj=0.1051
  12. violin                proj=0.1035
  13. éĥ½å¸Ĥ                proj=0.1026
  14. Bristol               proj=0.1004
  15. facilitating          proj=0.0982

**Most anti-aligned with capital delta direction** (opposite of where delta goes):
   1. China                 proj=-0.3139
   2. Japan                 proj=-0.2972
   3. France                proj=-0.2946
   4. Egypt                 proj=-0.2920
   5. Germany               proj=-0.2916
   6. China                 proj=-0.1829
   7. France                proj=-0.1792
   8. Russia                proj=-0.1710
   9. India                 proj=-0.1677
  10. Canada                proj=-0.1562
  11. Australia             proj=-0.1434
  12. Turkey                proj=-0.1423
  13. India                 proj=-0.1415
  14. Spain                 proj=-0.1399
  15. Japan                 proj=-0.1370

**Delta SVD spectrum** (how many directions the relationship uses):
  Dir 0: σ=1.1440, var=28.5%, cumvar=28.5%
  Dir 1: σ=0.9503, var=19.6%, cumvar=48.1%
  Dir 2: σ=0.9291, var=18.8%, cumvar=66.9%
  Dir 3: σ=0.8783, var=16.8%, cumvar=83.7%
  Dir 4: σ=0.8668, var=16.3%, cumvar=100.0%
  cos(mean_delta, SVD_dir_1) = 0.9843


### Relationship: language
Delta norm: 0.5647

**Most aligned with language delta direction** (these tokens point where the delta goes):
   1. French                proj=0.3300
   2. Chinese               proj=0.3161
   3. Japanese              proj=0.3028
   4. Spanish               proj=0.2974
   5. German                proj=0.2723
   6. French                proj=0.2213
   7. Chinese               proj=0.2161
   8. Japanese              proj=0.2067
   9. Spanish               proj=0.1945
  10. Italian               proj=0.1792
  11. french                proj=0.1742
  12. English               proj=0.1704
  13. English               proj=0.1686
  14. German                proj=0.1601
  15. Italian               proj=0.1511

**Most anti-aligned with language delta direction** (opposite of where delta goes):
   1. France                proj=-0.3014
   2. China                 proj=-0.2638
   3. Germany               proj=-0.2481
   4. Spain                 proj=-0.2474
   5. Japan                 proj=-0.2444
   6. France                proj=-0.1812
   7. China                 proj=-0.1681
   8. Japan                 proj=-0.1667
   9. Italy                 proj=-0.1591
  10. Britain               proj=-0.1490
  11. Britain               proj=-0.1477
  12. Russia                proj=-0.1443
  13. Italy                 proj=-0.1431
  14. Spain                 proj=-0.1429
  15. Germany               proj=-0.1356

**Delta SVD spectrum** (how many directions the relationship uses):
  Dir 0: σ=1.2680, var=35.3%, cumvar=35.3%
  Dir 1: σ=0.9337, var=19.1%, cumvar=54.4%
  Dir 2: σ=0.8466, var=15.7%, cumvar=70.1%
  Dir 3: σ=0.8353, var=15.3%, cumvar=85.4%
  Dir 4: σ=0.8160, var=14.6%, cumvar=100.0%
  cos(mean_delta, SVD_dir_1) = 0.9968


### Relationship: gender
Delta norm: 0.5097

**Most aligned with gender delta direction** (these tokens point where the delta goes):
   1. woman                 proj=0.2540
   2. mother                proj=0.2434
   3. queen                 proj=0.2385
   4. girl                  proj=0.2303
   5. spokeswoman           proj=0.1145
   6. girls                 proj=0.1107
   7. Female                proj=0.0965
   8. females               proj=0.0963
   9. women                 proj=0.0934
  10. å¥³ç¥ŀ                proj=0.0930
  11. å¥³åıĭ                proj=0.0927
  12. Girls                 proj=0.0915
  13. grandmother           proj=0.0912
  14. Ð´Ð²Ðµ                proj=0.0904
  15. emales                proj=0.0899

**Most anti-aligned with gender delta direction** (opposite of where delta goes):
   1. man                   proj=-0.3475
   2. king                  proj=-0.2983
   3. boy                   proj=-0.2496
   4. father                proj=-0.1771
   5. berg                  proj=-0.1577
   6. son                   proj=-0.1500
   7. ker                   proj=-0.1454
   8. ky                    proj=-0.1440
   9. ard                   proj=-0.1424
  10. ay                    proj=-0.1329
  11. ner                   proj=-0.1244
  12. hal                   proj=-0.1227
  13. ler                   proj=-0.1209
  14. ley                   proj=-0.1207
  15. land                  proj=-0.1197

**Delta SVD spectrum** (how many directions the relationship uses):
  Dir 0: σ=1.0538, var=30.3%, cumvar=30.3%
  Dir 1: σ=0.9680, var=25.6%, cumvar=55.9%
  Dir 2: σ=0.9295, var=23.6%, cumvar=79.5%
  Dir 3: σ=0.8654, var=20.5%, cumvar=100.0%
  cos(mean_delta, SVD_dir_1) = 0.9113


### Cross-Relationship Delta Comparison
  cos(capital, language) = 0.4107
  cos(capital, gender) = -0.0266
  cos(language, gender) = -0.0047

> **FINDING:** If cross-relationship cosines are near 0, each relationship has its own unique direction in ℝ³⁵⁸⁴ — they're orthogonal transforms, not variations of a common 'relationship axis'.


## 3. Anchor Discovery — Verifiable Properties in the Geometry


### Anchor: is_european_country
  Positive examples: 17 (France, Germany, Poland, Norway, Sweden, Italy, Portugal, Spain...)
  Negative examples: 23 (Japan, China, Egypt, Australia, Thailand, India, Brazil, Korea...)
  **Classification accuracy: 100.0%**
  Positive mean projection: 0.1979
  Negative mean projection: -0.0932
  Margin: 0.1369 (SEPARABLE)
  **LOO cross-validation: 92.5%** (37/40)

  Top-10 vocab most aligned with 'is_european_country':
     1. Belgium               proj=0.2539
     2. Switzerland           proj=0.2517
     3. Denmark               proj=0.2501
     4. Netherlands           proj=0.2474
     5. Norway                proj=0.2377
     6. Austria               proj=0.2309
     7. Greece                proj=0.2250
     8. Germany               proj=0.2197
     9. Poland                proj=0.2171
    10. Finland               proj=0.2144
  Top-10 vocab most anti-aligned:
     1. China                 proj=-0.1797
     2. Nigeria               proj=-0.1430
     3. Iran                  proj=-0.1371
     4. Egypt                 proj=-0.1340
     5. Korea                 proj=-0.1301
     6. Kenya                 proj=-0.1298
     7. jin                   proj=-0.1240
     8. Indian                proj=-0.1232
     9. Japan                 proj=-0.1200
    10. Vietnam               proj=-0.1196

### Anchor: is_asian_country
  Positive examples: 13 (Japan, China, Thailand, India, Korea, Vietnam, Indonesia, Philippines...)
  Negative examples: 17 (France, Germany, Poland, Norway, Sweden, Italy, Portugal, Spain...)
  **Classification accuracy: 100.0%**
  Positive mean projection: 0.1759
  Negative mean projection: -0.1257
  Margin: 0.1702 (SEPARABLE)
  **LOO cross-validation: 80.0%** (24/30)

  Top-10 vocab most aligned with 'is_asian_country':
     1. Korea                 proj=0.2257
     2. Vietnam               proj=0.2111
     3. China                 proj=0.2060
     4. Malaysia              proj=0.1952
     5. Thailand              proj=0.1842
     6. Philippines           proj=0.1834
     7. Indonesia             proj=0.1817
     8. China                 proj=0.1649
     9. Japan                 proj=0.1621
    10. Korean                proj=0.1615
  Top-10 vocab most anti-aligned:
     1. Spain                 proj=-0.1734
     2. Germany               proj=-0.1717
     3. Mexico                proj=-0.1676
     4. Norway                proj=-0.1635
     5. Italy                 proj=-0.1592
     6. Sweden                proj=-0.1571
     7. France                proj=-0.1571
     8. Morocco               proj=-0.1493
     9. Argentina             proj=-0.1356
    10. Poland                proj=-0.1293

### Anchor: is_capital_city
  Positive examples: 28 (Paris, Berlin, Tokyo, Beijing, Cairo, Canberra, Bangkok, Warsaw...)
  Negative examples: 23 (France, Germany, Japan, China, Egypt, Australia, Thailand, Poland...)
  **Classification accuracy: 100.0%**
  Positive mean projection: 0.1273
  Negative mean projection: -0.2269
  Margin: 0.1949 (SEPARABLE)
  **LOO cross-validation: 100.0%** (51/51)

  Top-10 vocab most aligned with 'is_capital_city':
     1. Helsinki              proj=0.1996
     2. Stockholm             proj=0.1816
     3. Dublin                proj=0.1596
     4. Copenhagen            proj=0.1566
     5. Warsaw                proj=0.1540
     6. Nairobi               proj=0.1537
     7. Amsterdam             proj=0.1521
     8. Tehran                proj=0.1521
     9. Brussels              proj=0.1486
    10. Oslo                  proj=0.1482
  Top-10 vocab most anti-aligned:
     1. France                proj=-0.2801
     2. China                 proj=-0.2652
     3. Canada                proj=-0.2629
     4. India                 proj=-0.2625
     5. Spain                 proj=-0.2623
     6. Germany               proj=-0.2588
     7. Australia             proj=-0.2514
     8. Turkey                proj=-0.2497
     9. Russia                proj=-0.2451
    10. Brazil                proj=-0.2371

### Anchor: is_romance_language
  Positive examples: 4 (French, Italian, Portuguese, Spanish)
  Negative examples: 19 (German, Japanese, Chinese, Arabic, English, Korean, Thai, Polish...)
  **Classification accuracy: 100.0%**
  Positive mean projection: 0.3713
  Negative mean projection: -0.0491
  Margin: 0.2806 (SEPARABLE)
  **LOO cross-validation: 82.6%** (19/23)

  Top-10 vocab most aligned with 'is_romance_language':
     1. Spanish               proj=0.4062
     2. Italian               proj=0.3786
     3. French                proj=0.3651
     4. Portuguese            proj=0.3352
     5. Spanish               proj=0.1968
     6. French                proj=0.1531
     7. Italian               proj=0.1344
     8. spanish               proj=0.1316
     9. uese                  proj=0.1238
    10. french                proj=0.1218
  Top-10 vocab most anti-aligned:
     1. Korean                proj=-0.1398
     2. Hindi                 proj=-0.1167
     3. Finnish               proj=-0.1038
     4. Qur                   proj=-0.1029
     5. Arabic                proj=-0.0992
     6. nodes                 proj=-0.0979
     7. candidates            proj=-0.0976
     8. Nokia                 proj=-0.0972
     9. oda                   proj=-0.0968
    10. Mongolia              proj=-0.0964

### Anchor: is_germanic_language
  Positive examples: 6 (German, English, Dutch, Norwegian, Swedish, Danish)
  Negative examples: 17 (French, Italian, Portuguese, Spanish, Japanese, Chinese, Arabic, Korean...)
  **Classification accuracy: 100.0%**
  Positive mean projection: 0.3039
  Negative mean projection: -0.0739
  Margin: 0.1862 (SEPARABLE)
  **LOO cross-validation: 78.3%** (18/23)

  Top-10 vocab most aligned with 'is_germanic_language':
     1. Danish                proj=0.3449
     2. Swedish               proj=0.3403
     3. Norwegian             proj=0.3313
     4. Dutch                 proj=0.3196
     5. German                proj=0.2565
     6. English               proj=0.2307
     7. German                proj=0.1852
     8. Netherlands           proj=0.1707
     9. Denmark               proj=0.1602
    10. Norway                proj=0.1582
  Top-10 vocab most anti-aligned:
     1. Chinese               proj=-0.1303
     2. Persian               proj=-0.1266
     3. Korean                proj=-0.1262
     4. Greek                 proj=-0.1178
     5. Arabic                proj=-0.1127
     6. ä¸įä½ı                proj=-0.1079
     7. Thai                  proj=-0.1071
     8. Hindi                 proj=-0.1048
     9. Japanese              proj=-0.1047
    10. Vietnamese            proj=-0.0969

### Anchor: is_female_gendered
  Positive examples: 12 (queen, woman, girl, mother, sister, daughter, wife, aunt...)
  Negative examples: 12 (king, man, boy, father, brother, son, husband, uncle...)
  **Classification accuracy: 100.0%**
  Positive mean projection: 0.1808
  Negative mean projection: -0.1567
  Margin: 0.1993 (SEPARABLE)
  **LOO cross-validation: 79.2%** (19/24)

  Top-10 vocab most aligned with 'is_female_gendered':
     1. actress               proj=0.2366
     2. heroine               proj=0.2149
     3. woman                 proj=0.2002
     4. princess              proj=0.1998
     5. spokeswoman           proj=0.1902
     6. girl                  proj=0.1887
     7. waitress              proj=0.1805
     8. wife                  proj=0.1795
     9. aunt                  proj=0.1779
    10. actresses             proj=0.1664
  Top-10 vocab most anti-aligned:
     1. son                   proj=-0.2649
     2. man                   proj=-0.2387
     3. king                  proj=-0.1959
     4. actor                 proj=-0.1839
     5. boy                   proj=-0.1637
     6. berg                  proj=-0.1579
     7. ker                   proj=-0.1537
     8. ner                   proj=-0.1463
     9. ky                    proj=-0.1405
    10. husband               proj=-0.1395

### Cross-Anchor Orthogonality
  cos(is_european_country, is_asian_country) = -0.4389
  cos(is_european_country, is_capital_city) = -0.0017
  cos(is_european_country, is_romance_language) = 0.0435
  cos(is_european_country, is_germanic_language) = 0.2629
  cos(is_european_country, is_female_gendered) = 0.0536
  cos(is_asian_country, is_capital_city) = 0.1437
  cos(is_asian_country, is_romance_language) = -0.1683
  cos(is_asian_country, is_germanic_language) = -0.1335
  cos(is_asian_country, is_female_gendered) = -0.0552
  cos(is_capital_city, is_romance_language) = -0.0959
  cos(is_capital_city, is_germanic_language) = 0.0274
  cos(is_capital_city, is_female_gendered) = -0.0199
  cos(is_romance_language, is_germanic_language) = -0.1657
  cos(is_romance_language, is_female_gendered) = 0.0259
  cos(is_germanic_language, is_female_gendered) = -0.0278

> **FINDING:** Anchors with high LOO accuracy AND mutual orthogonality are independent verifiable truths — candidate coordinate axes for TruthSpace.


### Anchor-Delta Alignment
  capital ↔ is_capital_city: cos = 0.6901
  language ↔ is_capital_city: cos = 0.3514
  language ↔ is_romance_language: cos = 0.2141
  gender ↔ is_female_gendered: cos = 0.6976

## 4. Gödel Composition — Concepts as Anchor Coordinate Vectors

Using 6 anchors with LOO >= 70%:
  is_asian_country: LOO=80.0%
  is_capital_city: LOO=100.0%
  is_european_country: LOO=92.5%
  is_female_gendered: LOO=79.2%
  is_germanic_language: LOO=78.3%
  is_romance_language: LOO=82.6%

### Anchor Coordinates for Known Concepts

**Countries:**

**Countries anchor coordinates**

|      Concept | asian countr | capital city | european cou | female gende | germanic lan | romance lang |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|       France |            - |            - |            + |            - |            - |            - |
|      Germany |            - |            - |            + |            + |            - |            - |
|        Japan |            + |            - |            - |            - |            - |            - |
|        China |            + |            - |            - |            - |            - |            - |
|        Egypt |            - |            - |            - |            - |            - |            - |
|    Australia |            - |            - |            - |            - |            - |            - |
|        India |            + |            - |            - |            - |            - |            - |
|       Brazil |            - |            - |            - |            + |            - |            - |
|        Korea |            + |            - |            - |            - |            - |            - |
|        Italy |            - |            - |            + |            - |            - |            - |
|        Spain |            - |            - |            + |            + |            - |            - |
|       Russia |            - |            - |            + |            - |            - |            - |
|       Poland |            - |            - |            + |            - |            - |            - |
|       Norway |            - |            - |            + |            - |            + |            - |
|       Sweden |            - |            - |            + |            + |            - |            - |
|       Turkey |            - |            - |            - |            - |            - |            - |
|       Greece |            - |            - |            + |            + |            - |            - |
|      Ireland |            - |            - |            + |            - |            - |            - |
|      Finland |            - |            - |            + |            - |            - |            - |
|      Denmark |            - |            - |            + |            + |            + |            - |
|       Mexico |            - |            - |            - |            - |            - |            - |
|       Canada |            - |            - |            - |            - |            - |            - |
|    Argentina |            - |            - |            - |            - |            - |            - |
|      Nigeria |            - |            - |            - |            + |            - |            - |
|        Kenya |            - |            - |            - |            - |            - |            - |


**Capitals:**

**Capitals anchor coordinates**

|      Concept | asian countr | capital city | european cou | female gende | germanic lan | romance lang |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|        Paris |            - |            + |            - |            - |            - |            - |
|       Berlin |            - |            + |            + |            - |            - |            - |
|        Tokyo |            + |            + |            - |            - |            - |            - |
|      Beijing |            + |            + |            - |            - |            - |            - |
|        Cairo |            - |            + |            - |            - |            - |            - |
|     Canberra |            + |            + |            - |            - |            - |            - |
|        Delhi |            + |            + |            - |            - |            - |            - |
|        Seoul |            + |            + |            - |            - |            - |            - |
|         Rome |            - |            + |            - |            - |            - |            - |
|       Lisbon |            - |            + |            + |            + |            - |            - |
|       Moscow |            - |            + |            + |            - |            - |            - |
|       Madrid |            - |            + |            + |            - |            - |            - |
|       Athens |            - |            + |            + |            - |            - |            - |
|       Ankara |            - |            + |            - |            - |            - |            - |
|       Dublin |            - |            + |            + |            + |            - |            - |
|     Helsinki |            + |            + |            + |            - |            - |            - |
|   Copenhagen |            - |            + |            + |            - |            + |            - |
|       Vienna |            - |            + |            + |            + |            - |            - |
|       Warsaw |            - |            + |            + |            + |            - |            - |
|         Oslo |            - |            + |            + |            - |            - |            - |
|    Stockholm |            - |            + |            + |            - |            - |            - |
|       Ottawa |            - |            + |            - |            - |            - |            - |
|         Lima |            - |            + |            - |            - |            - |            - |


**Languages:**

**Languages anchor coordinates**

|      Concept | asian countr | capital city | european cou | female gende | germanic lan | romance lang |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|       French |            - |            + |            - |            - |            - |            + |
|       German |            - |            - |            + |            - |            + |            - |
|     Japanese |            + |            + |            - |            - |            - |            - |
|      Chinese |            + |            - |            - |            - |            - |            - |
|      Spanish |            - |            - |            - |            - |            - |            + |
|      Italian |            - |            - |            - |            - |            - |            + |
|   Portuguese |            - |            - |            - |            + |            - |            + |
|      Russian |            - |            - |            - |            + |            - |            - |
|       Arabic |            - |            + |            - |            + |            - |            - |
|      English |            - |            - |            - |            - |            + |            - |
|       Korean |            + |            + |            - |            - |            - |            - |
|         Thai |            + |            + |            - |            - |            - |            - |
|       Polish |            - |            + |            - |            - |            - |            - |
|    Norwegian |            - |            + |            + |            - |            + |            - |
|      Swedish |            - |            + |            + |            + |            + |            - |
|        Dutch |            - |            + |            + |            - |            + |            - |
|        Greek |            - |            + |            - |            + |            - |            - |
|      Turkish |            + |            + |            - |            - |            - |            - |
|        Hindi |            + |            + |            - |            - |            - |            - |
|      Finnish |            - |            + |            + |            - |            - |            - |


**Gender (M):**

**Gender (M) anchor coordinates**

|      Concept | asian countr | capital city | european cou | female gende | germanic lan | romance lang |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|         king |            - |            + |            - |            - |            - |            - |
|          man |            - |            + |            - |            - |            - |            - |
|          boy |            - |            + |            - |            - |            - |            - |
|       father |            - |            - |            - |            - |            - |            - |
|      brother |            - |            + |            + |            - |            - |            - |
|          son |            - |            + |            - |            - |            - |            - |
|      husband |            - |            + |            - |            - |            - |            - |
|        uncle |            - |            - |            - |            - |            - |            - |
|       prince |            - |            + |            - |            - |            - |            - |
|        actor |            - |            + |            - |            - |            - |            - |


**Gender (F):**

**Gender (F) anchor coordinates**

|      Concept | asian countr | capital city | european cou | female gende | germanic lan | romance lang |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|        queen |            - |            + |            - |            + |            - |            - |
|        woman |            - |            + |            - |            + |            - |            - |
|         girl |            - |            + |            - |            + |            - |            - |
|       mother |            - |            - |            - |            + |            - |            - |
|       sister |            - |            + |            + |            + |            - |            - |
|     daughter |            - |            + |            - |            + |            - |            - |
|         wife |            - |            + |            - |            + |            - |            - |
|         aunt |            - |            + |            + |            + |            - |            - |
|     princess |            - |            + |            + |            + |            - |            - |
|      actress |            - |            + |            - |            + |            - |            - |


### Uniqueness Test — Do Coordinates Form Unique Addresses?
  Total concepts tested: 88
  Unique addresses: 19
  Concepts with unique address: 6/88 (6.8%)

  Collisions (13 addresses shared by multiple concepts):
    -+----: Paris, Cairo, Rome, Ankara, Ottawa, Lima, Polish, king, man, boy, son, husband, prince, actor
    ++----: Tokyo, Beijing, Canberra, Delhi, Seoul, Japanese, Korean, Thai, Turkish, Hindi
    ------: Egypt, Australia, Turkey, Mexico, Canada, Argentina, Kenya, father, uncle
    -++---: Berlin, Moscow, Madrid, Athens, Oslo, Stockholm, Finnish, brother
    -+-+--: Arabic, Greek, queen, woman, girl, daughter, wife, actress
    -+++--: Lisbon, Dublin, Vienna, Warsaw, sister, aunt, princess
    --+---: France, Italy, Russia, Poland, Ireland, Finland
    +-----: Japan, China, India, Korea, Chinese
    --++--: Germany, Spain, Sweden, Greece
    ---+--: Brazil, Nigeria, Russian, mother
    -++-+-: Copenhagen, Norwegian, Dutch
    --+-+-: Norway, German
    -----+: Spanish, Italian

> **FINDING:** With 6 anchors, 6.8% of concepts have unique addresses. Each additional verified anchor doubles the address space. Need ~log2(N_concepts) anchors for full uniqueness.


### Composition Test — Predicting Coordinates from Relationships
If France→Paris via capital-of, and we know France's coordinates,
can we predict Paris's coordinates?


**capital relationship:**
  Australia → Canberra: 5/6 coordinates match
  Thailand → Bangkok: 6/6 coordinates match
  Poland → Warsaw: 5/6 coordinates match
  Norway → Oslo: 5/6 coordinates match
  Sweden → Stockholm: 6/6 coordinates match
  India → Delhi: 6/6 coordinates match
  Korea → Seoul: 6/6 coordinates match
  Overall: 39/42 (92.9%) coordinate predictions correct

**language relationship:**
  Italy → Italian: 4/6 coordinates match
  Portugal → Portuguese: 3/6 coordinates match
  Russia → Russian: 5/6 coordinates match
  Overall: 12/18 (66.7%) coordinate predictions correct

**gender relationship:**
  brother → sister: 5/6 coordinates match
  son → daughter: 6/6 coordinates match
  husband → wife: 5/6 coordinates match
  Overall: 16/18 (88.9%) coordinate predictions correct

> **FINDING:** If VA preserves anchor coordinates across relationships, then the anchor coordinate system IS compatible with relationship deltas — we can reason about both simultaneously. This is the foundation for verifiable concept composition.


## 5. Reconstruction / Residual Test — Are Concepts Platonic Compounds?

Using 6 anchor directions (LOO >= 70%):
  is_asian_country: LOO=80.0%
  is_capital_city: LOO=100.0%
  is_european_country: LOO=92.5%
  is_female_gendered: LOO=79.2%
  is_germanic_language: LOO=78.3%
  is_romance_language: LOO=82.6%

### Anchor Basis Properties
  Anchor basis shape: (6, 3584)
  Gram matrix diagonal (should be 1.0): [0.9999989  1.0000002  1.         0.9999985  1.0000011  0.99999887]
  Off-diagonal |cos| — mean: 0.1109, max: 0.4389
  Anchors are NOT orthogonal

### Per-Concept Reconstruction

**Countries:**

**Countries reconstruction**

|      Concept |      ||emb|| |    ||recon|| |    ||resid|| |        ratio |  % explained |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|       France |        0.767 |        0.336 |        0.689 |       0.8987 |       19.23% |
|      Germany |        0.759 |        0.330 |        0.684 |       0.9005 |       18.92% |
|        Japan |        0.766 |        0.306 |        0.703 |       0.9169 |       15.92% |
|        China |        0.810 |        0.376 |        0.718 |       0.8860 |       21.50% |
|        Egypt |        0.756 |        0.297 |        0.696 |       0.9197 |       15.42% |
|    Australia |        0.781 |        0.269 |        0.733 |       0.9387 |       11.89% |
|        India |        0.789 |        0.319 |        0.722 |       0.9147 |       16.34% |
|       Brazil |        0.755 |        0.292 |        0.696 |       0.9223 |       14.93% |
|        Korea |        0.800 |        0.344 |        0.722 |       0.9030 |       18.45% |
|        Italy |        0.743 |        0.291 |        0.684 |       0.9201 |       15.33% |
|        Spain |        0.739 |        0.326 |        0.663 |       0.8972 |       19.51% |
|       Russia |        0.755 |        0.281 |        0.701 |       0.9280 |       13.88% |
|       Poland |        0.792 |        0.289 |        0.737 |       0.9312 |       13.29% |
|       Norway |        0.781 |        0.312 |        0.716 |       0.9169 |       15.93% |
|       Sweden |        0.725 |        0.269 |        0.674 |       0.9285 |       13.79% |
|       Turkey |        0.775 |        0.262 |        0.729 |       0.9413 |       11.40% |
|       Greece |        0.787 |        0.317 |        0.720 |       0.9152 |       16.24% |
|      Ireland |        0.809 |        0.272 |        0.762 |       0.9419 |       11.28% |
|      Finland |        0.779 |        0.258 |        0.734 |       0.9433 |       11.02% |
|      Denmark |        0.780 |        0.293 |        0.723 |       0.9268 |       14.11% |
|       Mexico |        0.743 |        0.295 |        0.681 |       0.9175 |       15.81% |
|       Canada |        0.785 |        0.309 |        0.721 |       0.9192 |       15.51% |
|    Argentina |        0.708 |        0.272 |        0.654 |       0.9233 |       14.75% |
|      Nigeria |        0.797 |        0.234 |        0.762 |       0.9561 |        8.60% |
|        Kenya |        0.785 |        0.209 |        0.757 |       0.9638 |        7.11% |

  Group mean ||resid||/||emb||: 0.9228

**Capitals:**

**Capitals reconstruction**

|      Concept |      ||emb|| |    ||recon|| |    ||resid|| |        ratio |  % explained |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|        Paris |        0.772 |        0.113 |        0.763 |       0.9893 |        2.13% |
|       Berlin |        0.747 |        0.121 |        0.737 |       0.9868 |        2.62% |
|        Tokyo |        0.796 |        0.145 |        0.783 |       0.9832 |        3.34% |
|      Beijing |        0.789 |        0.170 |        0.771 |       0.9765 |        4.65% |
|        Cairo |        0.647 |        0.101 |        0.639 |       0.9877 |        2.45% |
|     Canberra |        0.768 |        0.130 |        0.757 |       0.9856 |        2.86% |
|        Delhi |        0.801 |        0.155 |        0.786 |       0.9812 |        3.73% |
|        Seoul |        0.762 |        0.157 |        0.746 |       0.9786 |        4.24% |
|         Rome |        0.788 |        0.081 |        0.784 |       0.9947 |        1.06% |
|       Lisbon |        0.759 |        0.181 |        0.738 |       0.9713 |        5.65% |
|       Moscow |        0.788 |        0.153 |        0.773 |       0.9809 |        3.78% |
|       Madrid |        0.782 |        0.127 |        0.772 |       0.9867 |        2.65% |
|       Athens |        0.781 |        0.161 |        0.764 |       0.9785 |        4.24% |
|       Ankara |        0.730 |        0.152 |        0.714 |       0.9781 |        4.33% |
|       Dublin |        0.796 |        0.175 |        0.777 |       0.9754 |        4.85% |
|     Helsinki |        0.768 |        0.222 |        0.735 |       0.9572 |        8.38% |
|   Copenhagen |        0.770 |        0.213 |        0.740 |       0.9608 |        7.69% |
|       Vienna |        0.785 |        0.180 |        0.764 |       0.9732 |        5.29% |
|       Warsaw |        0.768 |        0.193 |        0.743 |       0.9679 |        6.31% |
|         Oslo |        0.758 |        0.196 |        0.732 |       0.9659 |        6.70% |
|    Stockholm |        0.771 |        0.234 |        0.734 |       0.9530 |        9.19% |
|       Ottawa |        0.779 |        0.146 |        0.766 |       0.9822 |        3.53% |
|         Lima |        0.764 |        0.131 |        0.752 |       0.9851 |        2.96% |

  Group mean ||resid||/||emb||: 0.9774

**Languages:**

**Languages reconstruction**

|      Concept |      ||emb|| |    ||recon|| |    ||resid|| |        ratio |  % explained |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|       French |        0.783 |        0.370 |        0.691 |       0.8816 |       22.27% |
|       German |        0.777 |        0.290 |        0.721 |       0.9279 |       13.90% |
|     Japanese |        0.768 |        0.135 |        0.756 |       0.9845 |        3.07% |
|      Chinese |        0.790 |        0.171 |        0.771 |       0.9763 |        4.68% |
|      Spanish |        0.772 |        0.408 |        0.656 |       0.8491 |       27.90% |
|      Italian |        0.773 |        0.387 |        0.669 |       0.8659 |       25.02% |
|   Portuguese |        0.781 |        0.340 |        0.703 |       0.9004 |       18.92% |
|      Russian |        0.773 |        0.105 |        0.766 |       0.9908 |        1.84% |
|       Arabic |        0.790 |        0.179 |        0.769 |       0.9740 |        5.14% |
|      English |        0.812 |        0.257 |        0.770 |       0.9486 |       10.02% |
|       Korean |        0.797 |        0.249 |        0.757 |       0.9498 |        9.78% |
|         Thai |        0.729 |        0.142 |        0.715 |       0.9807 |        3.82% |
|       Polish |        0.800 |        0.091 |        0.795 |       0.9936 |        1.29% |
|    Norwegian |        0.794 |        0.333 |        0.721 |       0.9078 |       17.60% |
|      Swedish |        0.797 |        0.346 |        0.718 |       0.9008 |       18.86% |
|        Dutch |        0.807 |        0.329 |        0.737 |       0.9132 |       16.61% |
|        Greek |        0.759 |        0.140 |        0.746 |       0.9827 |        3.42% |
|      Turkish |        0.792 |        0.110 |        0.784 |       0.9904 |        1.92% |
|        Hindi |        0.779 |        0.175 |        0.759 |       0.9745 |        5.03% |
|      Finnish |        0.787 |        0.139 |        0.775 |       0.9843 |        3.11% |

  Group mean ||resid||/||emb||: 0.9438

**Gender (M):**

**Gender (M) reconstruction**

|      Concept |      ||emb|| |    ||recon|| |    ||resid|| |        ratio |  % explained |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|         king |        0.818 |        0.201 |        0.792 |       0.9693 |        6.05% |
|          man |        0.867 |        0.245 |        0.832 |       0.9594 |        7.96% |
|          boy |        0.806 |        0.166 |        0.789 |       0.9785 |        4.26% |
|       father |        0.774 |        0.152 |        0.759 |       0.9804 |        3.87% |
|      brother |        0.813 |        0.149 |        0.799 |       0.9831 |        3.35% |
|          son |        0.856 |        0.268 |        0.813 |       0.9497 |        9.81% |
|      husband |        0.810 |        0.159 |        0.794 |       0.9807 |        3.83% |
|        uncle |        0.775 |        0.139 |        0.763 |       0.9837 |        3.23% |
|       prince |        0.764 |        0.121 |        0.755 |       0.9874 |        2.51% |
|        actor |        0.808 |        0.194 |        0.784 |       0.9708 |        5.75% |

  Group mean ||resid||/||emb||: 0.9743

**Gender (F):**

**Gender (F) reconstruction**

|      Concept |      ||emb|| |    ||recon|| |    ||resid|| |        ratio |  % explained |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|        queen |        0.714 |        0.166 |        0.694 |       0.9726 |        5.41% |
|        woman |        0.775 |        0.203 |        0.747 |       0.9649 |        6.89% |
|         girl |        0.768 |        0.190 |        0.744 |       0.9688 |        6.14% |
|       mother |        0.762 |        0.164 |        0.744 |       0.9764 |        4.66% |
|       sister |        0.814 |        0.156 |        0.799 |       0.9814 |        3.68% |
|     daughter |        0.722 |        0.161 |        0.704 |       0.9748 |        4.98% |
|         wife |        0.760 |        0.185 |        0.737 |       0.9698 |        5.95% |
|         aunt |        0.774 |        0.188 |        0.751 |       0.9699 |        5.93% |
|     princess |        0.765 |        0.213 |        0.735 |       0.9604 |        7.77% |
|      actress |        0.808 |        0.244 |        0.770 |       0.9533 |        9.12% |

  Group mean ||resid||/||emb||: 0.9692

### Aggregate Reconstruction Statistics
  Total concepts analyzed: 88
  Embedding dimension: 3584
  Number of anchor axes: 6
  Theoretical max variance explained by 6 axes: 0.167% (if random directions)

  ||residual|| / ||embedding||:
    Mean:   0.9530
    Median: 0.9669
    Std:    0.0335
    Min:    0.8491
    Max:    0.9947

  Variance explained (% of ||emb||^2):
    Mean:   9.07%
    Median: 6.51%
    Min:    1.06%
    Max:    27.90%

### Residual Structure Analysis
If residuals are random noise, they should:
  1. Have no consistent direction (low pairwise cosine)
  2. Not cluster by concept type
  3. Have low rank (spread across many dimensions)

  Pairwise cosine similarity of residuals:
    Mean: 0.1028
    Std:  0.0739
    |cos| mean: 0.1043
    Max:  0.5040
    Min:  -0.0811

  Within-group vs between-group residual similarity:
    Countries      : within=0.1476, between=0.1072, gap=0.0404
    Capitals       : within=0.1270, between=0.1068, gap=0.0201
    Languages      : within=0.2334, between=0.0905, gap=0.1429
    Gender (M)     : within=0.0706, between=0.0492, gap=0.0214
    Gender (F)     : within=0.0951, between=0.0518, gap=0.0433

  SVD of residual matrix ((88, 3584)):
    Total singular values: 88
    Top-10 singular values: [1.536894   1.3186601  1.0781099  1.0021954  0.96924555 0.9405772
 0.91203713 0.89779824 0.8750459  0.86647   ]
    Top-1 explains: 5.50% of residual variance
    Top-3 explain:  12.26%
    Top-5 explain:  16.78%
    Top-10 explain: 26.19%
    Components for 50% variance: 27
    Components for 80% variance: 56
    Components for 90% variance: 68
    Components for 95% variance: 75

  Top residual principal directions:

    PC1 (explains 5.5%):
      Most positive: sister(0.298), brother(0.294), father(0.282), mother(0.266), son(0.248)
      Most negative: Turkish(-0.256), Finnish(-0.244), Korea(-0.232), Helsinki(-0.229), Swedish(-0.225)

    PC2 (explains 4.0%):
      Most positive: Ottawa(0.183), Oslo(0.156), Dublin(0.153), Ireland(0.147), Canberra(0.143)
      Most negative: Chinese(-0.327), Japanese(-0.323), Turkish(-0.295), Russian(-0.292), French(-0.284)

    PC3 (explains 2.7%):
      Most positive: brother(0.355), sister(0.328), uncle(0.303), husband(0.300), aunt(0.210)
      Most negative: man(-0.229), king(-0.224), woman(-0.198), boy(-0.196), girl(-0.178)

### Random Baseline Comparison
How does reconstruction with truth axes compare to random directions?

  Truth axes (6 dims): 9.07% variance explained
  Random axes (6 dims): 0.16% ± 0.02% (mean ± std over 20 trials)
  Ratio (truth/random): 55.16x

> **FINDING:** Truth axes explain 9.07% of concept variance vs 0.16% for random — 55.2x more! The truth axes capture MEANINGFUL structure, not just random subspace.


### Verdict
  With 6 truth axes spanning 6/3584 dimensions:
  - 9.07% of concept variance is explained by truth axes
  - 90.93% lives in the residual (orthogonal to truth axes)

> **FINDING:** Truth axes capture only 9.1% — concepts are much richer than their platonic coordinates. Either we need many more truth axes, or the residual encodes fundamentally different structure.


## 6. Synthesis — The State of Delta Readability

Investigation completed in 13.6s

### What We Now Know

Part 1 tells us what individual dimensions encode — whether they
carry semantic content (geographic, linguistic, gendered) or noise.

Part 2 tells us what relationship delta directions mean — whether
the delta direction itself separates source from target concepts.

Part 3 tells us which binary properties are geometrically verifiable —
these are candidate TruthSpace anchors.

Part 4 tells us whether anchor coordinates uniquely address concepts
and whether relationships preserve coordinates — the Gödel test.
