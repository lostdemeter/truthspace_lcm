# DC299 Phase 1 — IRD Axis Discovery Notes


## Configuration

  MAX_AXES=1500  MAX_VARIANCE=0.95
  PATIENCE=10  ORTH_TOL=0.1
  BINARY_TOP_K=50  MIN_BINARY_ACC=0.75
  MIN_VARIANCE_STEP=0.001

Seed axes loaded: ['is_european_country', 'is_asian_country', 'is_capital_city', 'is_romance_language', 'is_germanic_language', 'is_female_gendered']

## Train / Holdout Split

  Total concepts : 25671
  Train          : 20537
  Holdout        : 5134
  SVD runs on train only; binary_acc measured on holdout.

## Seed Axes

  is_european_country             binary_acc=0.779  gap=0.2891
  is_asian_country                binary_acc=0.813  gap=0.2525
  is_capital_city                 binary_acc=0.831  gap=0.3229
  is_romance_language             binary_acc=0.872  gap=0.2090
  is_germanic_language            binary_acc=0.739  gap=0.2481
  is_female_gendered              binary_acc=0.860  gap=0.2337

Starting IRD with 6 seed axes.
After seeding: cumulative variance explained = 0.0072

## IRD Discovery Loop


  [   1]  axes=6  step_var=0.0025  binary_acc=0.894  gap=0.2851  max_dot=0.0046  (1.9s)
    TOP:  ãĢĭ(0.08) | äººæīį(0.06) | .âĢĿ(0.06) | à¦Ļ(0.06) | edback(0.05) | .âĢĻ(0.05) | ðŁı¾(0.05) | æľĪèĩ³(0.05)
    BOT:  Colombia(-0.24) | metabol(-0.24) | closure(-0.24) | collaborators(-0.24) | scanner(-0.24) | subsystem(-0.24) | manipulation(-0.24) | attacking(-0.24)
    ACCEPTED as axis_006  cumulative_var=0.0377

  [   2]  axes=7  step_var=0.0094  binary_acc=0.858  gap=0.5280  max_dot=0.0084  (2.0s)
    TOP:  Zionist(0.19) | daycare(0.19) | Qur(0.19) | Algeria(0.19) | Guam(0.18) | Eleanor(0.17) | Gandhi(0.17) | Volkswagen(0.17)
    BOT:  represents(-0.30) | supported(-0.30) | provided(-0.29) | fixed(-0.29) | transformed(-0.29) | removed(-0.29) | function(-0.29) | supports(-0.29)
    ACCEPTED as axis_007  cumulative_var=0.0471

  [   3]  axes=8  step_var=0.0091  binary_acc=0.955  gap=0.5067  max_dot=0.0019  (1.9s)
    TOP:  Jacob(0.27) | Clare(0.26) | Ma(0.25) | ãĢĭ(0.25) | Dong(0.24) | Andrew(0.24) | Lawrence(0.23) | Bernard(0.23)
    BOT:  administered(-0.22) | eligibility(-0.21) | discoveries(-0.20) | promotions(-0.20) | interpreting(-0.20) | diagnose(-0.20) | approvals(-0.19) | downloading(-0.19)
    ACCEPTED as axis_008  cumulative_var=0.0559

  [   4]  axes=9  step_var=0.0072  binary_acc=0.980  gap=0.4692  max_dot=0.0058  (1.9s)
    TOP:  neoliberal(0.22) | unconstitutional(0.21) | unsustainable(0.21) | unrecognized(0.20) | horrified(0.20) | reluctance(0.20) | unstoppable(0.20) | unprotected(0.20)
    BOT:  garden(-0.24) | sales(-0.22) | Green(-0.22) | Bell(-0.22) | principles(-0.22) | fluid(-0.22) | ocean(-0.21) | pet(-0.21)
    ACCEPTED as axis_009  cumulative_var=0.0628

  [   5]  axes=10  step_var=0.0065  binary_acc=0.915  gap=0.4026  max_dot=0.0031  (1.9s)
    TOP:  Entertainment(0.22) | International(0.22) | Development(0.22) | Business(0.21) | Research(0.21) | Information(0.21) | Executive(0.21) | Education(0.20)
    BOT:  pin(-0.16) | nod(-0.16) | slap(-0.16) | pac(-0.16) | snaps(-0.15) | sank(-0.15) | lug(-0.15) | dump(-0.15)
    ACCEPTED as axis_010  cumulative_var=0.0689

  [   6]  axes=11  step_var=0.0050  binary_acc=0.936  gap=0.3926  max_dot=0.0031  (2.0s)
    TOP:  runtime(0.20) | config(0.20) | init(0.20) | env(0.19) | idx(0.19) | info(0.19) | setup(0.19) | sys(0.19)
    BOT:  agricultural(-0.18) | surrendered(-0.17) | flourishing(-0.16) | settlement(-0.16) | harvesting(-0.16) | punishment(-0.16) | bail(-0.16) | bathing(-0.16)
    ACCEPTED as axis_011  cumulative_var=0.0736

  [   7]  axes=12  step_var=0.0047  binary_acc=0.981  gap=0.3919  max_dot=0.0033  (1.9s)
    TOP:  institutions(0.21) | organizations(0.20) | entrepreneurs(0.19) | organisations(0.18) | communities(0.18) | phenomena(0.18) | governments(0.18) | jurisdictions(0.18)
    BOT:  Flip(-0.19) | pulls(-0.19) | Kill(-0.19) | Hit(-0.18) | pushes(-0.18) | Turning(-0.18) | Rolling(-0.18) | pushing(-0.18)
    ACCEPTED as axis_012  cumulative_var=0.0779

  [   8]  axes=13  step_var=0.0050  binary_acc=0.963  gap=0.3900  max_dot=0.0027  (1.9s)
    TOP:  refer(0.19) | infr(0.19) | solic(0.17) | substit(0.16) | tyr(0.16) | consult(0.16) | liable(0.16) | incur(0.16)
    BOT:  genomes(-0.17) | Engines(-0.17) | quarterbacks(-0.17) | trays(-0.17) | decks(-0.17) | Rounds(-0.17) | Cheese(-0.16) | melodies(-0.16)
    ACCEPTED as axis_013  cumulative_var=0.0825

  [   9]  axes=14  step_var=0.0037  binary_acc=0.953  gap=0.3456  max_dot=0.0013  (1.9s)
    TOP:  elegant(0.15) | antique(0.15) | diamond(0.15) | exotic(0.14) | rustic(0.14) | pencil(0.14) | cone(0.14) | pixel(0.14)
    BOT:  injuries(-0.18) | deaths(-0.18) | threats(-0.17) | struggles(-0.17) | efforts(-0.17) | violations(-0.17) | lleg(-0.17) | failures(-0.17)
    ACCEPTED as axis_014  cumulative_var=0.0859

  [  10]  axes=15  step_var=0.0039  binary_acc=0.912  gap=0.3426  max_dot=0.0032  (1.9s)
    TOP:  ahora(0.16) | oltre(0.16) | Lorenzo(0.15) | tiene(0.15) | porque(0.15) | Antonio(0.15) | cuando(0.15) | dopo(0.15)
    BOT:  LIB(-0.16) | informational(-0.16) | philosophical(-0.15) | LAB(-0.15) | Lit(-0.14) | lic(-0.14) | bib(-0.14) | phys(-0.14)
    ACCEPTED as axis_015  cumulative_var=0.0895

  [  11]  axes=16  step_var=0.0039  binary_acc=0.889  gap=0.3711  max_dot=0.0011  (2.0s)
    TOP:  absor(0.17) | irrig(0.16) | fibr(0.16) | Mobility(0.16) | acceler(0.16) | stability(0.15) | steril(0.15) | turb(0.15)
    BOT:  annya(-0.18) | seinen(-0.18) | comme(-0.18) | donc(-0.17) | vous(-0.17) | donde(-0.17) | kann(-0.17) | aussi(-0.17)
    ACCEPTED as axis_016  cumulative_var=0.0931

  [  12]  axes=17  step_var=0.0032  binary_acc=0.984  gap=0.3194  max_dot=0.0010  (1.9s)
    TOP:  Sorry(0.17) | iley(0.15) | silly(0.15) | illy(0.15) | å¯¹ä¸įèµ·(0.14) | affiliate(0.14) | sorry(0.14) | appropriately(0.14)
    BOT:  geomet(-0.15) | Deliver(-0.15) | Consum(-0.15) | Recover(-0.15) | Produ(-0.15) | emerges(-0.14) | erupted(-0.14) | Performing(-0.14)
    ACCEPTED as axis_017  cumulative_var=0.0960

  [  13]  axes=18  step_var=0.0033  binary_acc=0.928  gap=0.3247  max_dot=0.0013  (1.9s)
    TOP:  cites(0.17) | Mits(0.16) | erts(0.15) | incentives(0.15) | emphasizes(0.15) | Gins(0.15) | converts(0.15) | udes(0.14)
    BOT:  Machine(-0.14) | mountain(-0.13) | SERVICE(-0.13) | screen(-0.13) | STAR(-0.13) | scanner(-0.13) | pathway(-0.13) | Teacher(-0.13)
    ACCEPTED as axis_018  cumulative_var=0.0990

  [  14]  axes=19  step_var=0.0032  binary_acc=0.887  gap=0.3477  max_dot=0.0031  (1.9s)
    TOP:  foundational(0.18) | åĪļå¥½(0.14) | confidently(0.14) | forwarding(0.14) | starters(0.14) | stepping(0.14) | standpoint(0.14) | fundamentals(0.14)
    BOT:  violence(-0.20) | murder(-0.19) | destroyed(-0.19) | abuse(-0.18) | Holocaust(-0.18) | destroy(-0.18) | Violence(-0.17) | murdered(-0.17)
    ACCEPTED as axis_019  cumulative_var=0.1019

  [  15]  axes=20  step_var=0.0032  binary_acc=0.965  gap=0.3176  max_dot=0.0007  (1.9s)
    TOP:  cultures(0.16) | clarity(0.15) | ç¹ģåįİ(0.15) | chemistry(0.14) | Paris(0.14) | çŁ¿çī©è´¨(0.14) | ç¡¬åº¦(0.14) | culture(0.14)
    BOT:  incon(-0.15) | acqu(-0.15) | invol(-0.15) | occup(-0.15) | indu(-0.14) | Accom(-0.14) | McConnell(-0.14) | Opp(-0.14)
    ACCEPTED as axis_020  cumulative_var=0.1047

  [  16]  axes=21  step_var=0.0030  binary_acc=0.963  gap=0.3180  max_dot=0.0018  (1.9s)
    TOP:  warnings(0.16) | browsers(0.15) | mailing(0.15) | fundraising(0.15) | messages(0.14) | sponsor(0.14) | vendors(0.14) | wearing(0.14)
    BOT:  equilibrium(-0.16) | dependence(-0.15) | complexities(-0.15) | similarity(-0.14) | competence(-0.14) | complexity(-0.14) | Composition(-0.14) | pton(-0.14)
    ACCEPTED as axis_021  cumulative_var=0.1074

  [  17]  axes=22  step_var=0.0034  binary_acc=0.962  gap=0.3355  max_dot=0.0019  (1.8s)
    TOP:  taxation(0.17) | assessed(0.16) | prognosis(0.16) | lawsuit(0.16) | duty(0.16) | taxes(0.16) | levy(0.15) | exemption(0.15)
    BOT:  strands(-0.19) | stretches(-0.16) | halves(-0.16) | stretched(-0.15) | contrasts(-0.15) | surrounds(-0.15) | stretch(-0.15) | swirl(-0.15)
    ACCEPTED as axis_022  cumulative_var=0.1104

  [  18]  axes=23  step_var=0.0028  binary_acc=0.975  gap=0.3097  max_dot=0.0050  (1.8s)
    TOP:  weakness(0.18) | confidence(0.16) | growth(0.16) | efficiency(0.16) | transformation(0.15) | pleasure(0.15) | éĢŁåº¦(0.15) | strength(0.15)
    BOT:  narr(-0.17) | Marty(-0.15) | Cater(-0.14) | baptized(-0.14) | audit(-0.14) | parasite(-0.13) | ancestral(-0.13) | surveyed(-0.13)
    ACCEPTED as axis_023  cumulative_var=0.1129

  [  19]  axes=24  step_var=0.0029  binary_acc=0.981  gap=0.3139  max_dot=0.0019  (1.9s)
    TOP:  fight(0.16) | league(0.16) | flight(0.16) | çĶµè§Ĩåī§(0.15) | strike(0.15) | battle(0.15) | smith(0.15) | Flight(0.15)
    BOT:  abundant(-0.16) | ened(-0.14) | saturated(-0.14) | powdered(-0.14) | painful(-0.14) | porous(-0.13) | ample(-0.13) | avoided(-0.13)
    ACCEPTED as axis_024  cumulative_var=0.1155

  [  20]  axes=25  step_var=0.0029  binary_acc=0.978  gap=0.3181  max_dot=0.0022  (1.9s)
    TOP:  mandate(0.15) | Consent(0.14) | dispens(0.14) | tint(0.14) | blending(0.14) | fragrance(0.14) | preference(0.14) | endment(0.14)
    BOT:  astronomical(-0.15) | athletes(-0.15) | ecstatic(-0.15) | Jake(-0.14) | failed(-0.14) | Rachel(-0.14) | spectacular(-0.14) | accelerated(-0.14)
    ACCEPTED as axis_025  cumulative_var=0.1180

  [  21]  axes=26  step_var=0.0028  binary_acc=0.955  gap=0.3042  max_dot=0.0070  (1.8s)
    TOP:  wooded(0.14) | orderly(0.14) | posted(0.13) | åı£æľį(0.13) | èĪ¹ä¸Ĭ(0.13) | bookstore(0.13) | Doors(0.13) | fuller(0.13)
    BOT:  bang(-0.15) | Bangladesh(-0.15) | abundance(-0.15) | fraud(-0.14) | Hash(-0.14) | ambiguity(-0.14) | splash(-0.14) | hash(-0.14)
    ACCEPTED as axis_026  cumulative_var=0.1205

  [  22]  axes=27  step_var=0.0028  binary_acc=0.985  gap=0.3204  max_dot=0.0058  (1.9s)
    TOP:  impactful(0.15) | innate(0.15) | floral(0.15) | vibrant(0.15) | alumni(0.15) | allies(0.14) | evil(0.14) | vitality(0.14)
    BOT:  Kensington(-0.16) | strapped(-0.15) | Preston(-0.15) | pressed(-0.14) | Queensland(-0.14) | squeezed(-0.14) | Hutchinson(-0.14) | possession(-0.13)
    ACCEPTED as axis_027  cumulative_var=0.1230

  [  23]  axes=28  step_var=0.0026  binary_acc=0.973  gap=0.2996  max_dot=0.0032  (1.9s)
    TOP:  åĹ£(0.14) | courtesy(0.14) | ä¾į(0.14) | reception(0.14) | sight(0.14) | CCTV(0.14) | Reception(0.13) | dessert(0.13)
    BOT:  analyze(-0.20) | analyzing(-0.17) | analy(-0.15) | evaluate(-0.15) | analyzed(-0.15) | evaluating(-0.14) | Analy(-0.14) | Walker(-0.14)
    ACCEPTED as axis_028  cumulative_var=0.1253

  [  24]  axes=29  step_var=0.0026  binary_acc=0.957  gap=0.2979  max_dot=0.0019  (1.8s)
    TOP:  æļ´éĽ¨(0.15) | post(0.14) | Boston(0.14) | Houston(0.14) | Post(0.14) | fostering(0.13) | assaulted(0.13) | pushing(0.13)
    BOT:  sincere(-0.16) | scri(-0.15) | clever(-0.13) | recursive(-0.13) | scrim(-0.12) | divine(-0.12) | sacrifices(-0.12) | sacred(-0.12)
    ACCEPTED as axis_029  cumulative_var=0.1276

  [  25]  axes=30  step_var=0.0029  binary_acc=0.989  gap=0.3166  max_dot=0.0030  (1.8s)
    TOP:  Equality(0.17) | ouncill(0.16) | coun(0.15) | ighbour(0.15) | cyclists(0.14) | behaviour(0.14) | Anthony(0.14) | neighbours(0.14)
    BOT:  mitochondrial(-0.15) | manufactured(-0.15) | attributable(-0.14) | manufactures(-0.14) | manufacturing(-0.14) | reimbursement(-0.14) | manufacture(-0.14) | -Pro(-0.13)
    ACCEPTED as axis_030  cumulative_var=0.1302

  [  26]  axes=31  step_var=0.0028  binary_acc=0.946  gap=0.3059  max_dot=0.0023  (1.9s)
    TOP:  Coupon(0.15) | booked(0.15) | waiver(0.14) | æĭ¼éŁ³(0.14) | pooled(0.14) | Picker(0.14) | Fiji(0.13) | Ville(0.13)
    BOT:  art(-0.15) | insanity(-0.15) | instinct(-0.15) | ass(-0.14) | grotes(-0.14) | embrace(-0.14) | emotion(-0.14) | improv(-0.14)
    ACCEPTED as axis_031  cumulative_var=0.1326

  [  27]  axes=32  step_var=0.0030  binary_acc=0.981  gap=0.3194  max_dot=0.0038  (1.8s)
    TOP:  movies(0.16) | hitting(0.16) | capturing(0.15) | comic(0.15) | storytelling(0.15) | picking(0.15) | functionality(0.14) | manipulated(0.14)
    BOT:  retreat(-0.15) | ceased(-0.15) | freeze(-0.15) | Increase(-0.15) | ascend(-0.15) | esteem(-0.14) | increase(-0.14) | ayne(-0.13)
    ACCEPTED as axis_032  cumulative_var=0.1353

  [  28]  axes=33  step_var=0.0028  binary_acc=0.978  gap=0.3134  max_dot=0.0080  (1.8s)
    TOP:  Howard(0.15) | commons(0.14) | Harvey(0.14) | reordered(0.14) | Parks(0.14) | Heroes(0.14) | Rogers(0.13) | roy(0.13)
    BOT:  Finnish(-0.18) | Brazilian(-0.16) | tissues(-0.15) | bacterial(-0.14) | Slovenia(-0.14) | ibia(-0.14) | skin(-0.14) | abrasive(-0.14)
    ACCEPTED as axis_033  cumulative_var=0.1377

  [  29]  axes=34  step_var=0.0028  binary_acc=0.943  gap=0.3040  max_dot=0.0030  (1.9s)
    TOP:  éĢĶ(0.15) | Plane(0.14) | proced(0.14) | è¶ĭ(0.14) | prox(0.13) | progressing(0.13) | prog(0.13) | successive(0.13)
    BOT:  harvest(-0.16) | Garn(-0.15) | redeem(-0.15) | sharpen(-0.15) | Johann(-0.14) | reaff(-0.14) | donation(-0.14) | refund(-0.14)
    ACCEPTED as axis_034  cumulative_var=0.1401

  [  30]  axes=35  step_var=0.0027  binary_acc=0.974  gap=0.2912  max_dot=0.0019  (1.8s)
    TOP:  glimps(0.14) | endeavors(0.14) | metaph(0.14) | precip(0.13) | cues(0.13) | Veterinary(0.13) | ptive(0.13) | impulses(0.13)
    BOT:  worker(-0.15) | workers(-0.15) | farmer(-0.15) | manager(-0.14) | æľįåĬ¡åĳĺ(-0.14) | farmers(-0.13) | developer(-0.13) | freelancer(-0.13)
    ACCEPTED as axis_035  cumulative_var=0.1424

  [  31]  axes=36  step_var=0.0027  binary_acc=0.930  gap=0.3014  max_dot=0.0027  (2.0s)
    TOP:  Jess(0.15) | messing(0.15) | colonization(0.14) | persist(0.14) | genetic(0.14) | behavioral(0.14) | cheese(0.13) | sequencing(0.13)
    BOT:  stands(-0.16) | Olympus(-0.13) | çªģçł´åı£(-0.13) | Trophy(-0.13) | Kabul(-0.13) | ground(-0.13) | mountains(-0.13) | stand(-0.13)
    ACCEPTED as axis_036  cumulative_var=0.1447

  [  32]  axes=37  step_var=0.0027  binary_acc=0.994  gap=0.2921  max_dot=0.0025  (1.9s)
    TOP:  lawmakers(0.15) | researchers(0.14) | receptors(0.14) | buyers(0.14) | detectors(0.13) | prosecutors(0.13) | regulators(0.13) | Muk(0.12)
    BOT:  Origin(-0.14) | pilgrimage(-0.14) | Origin(-0.14) | origin(-0.13) | Orbit(-0.13) | Crisis(-0.13) | illness(-0.13) | Emergency(-0.12)
    ACCEPTED as axis_037  cumulative_var=0.1470

  [  33]  axes=38  step_var=0.0024  binary_acc=0.954  gap=0.2758  max_dot=0.0012  (1.8s)
    TOP:  drainage(0.14) | åŃĲåŃĻ(0.13) | Bulldogs(0.13) | someday(0.13) | åŃµåĮĸ(0.12) | èŀ¨(0.12) | à¸ªà¸²(0.12) | sustainable(0.12)
    BOT:  yelled(-0.14) | voiced(-0.14) | phone(-0.14) | phone(-0.13) | Ariel(-0.13) | iot(-0.13) | cellphone(-0.13) | yell(-0.13)
    ACCEPTED as axis_038  cumulative_var=0.1491

  [  34]  axes=39  step_var=0.0028  binary_acc=0.998  gap=0.3047  max_dot=0.0014  (1.9s)
    TOP:  sweeping(0.15) | smoothly(0.14) | timely(0.14) | tight(0.14) | systemic(0.14) | roadside(0.14) | steep(0.14) | trips(0.14)
    BOT:  Son(-0.16) | idol(-0.15) | son(-0.15) | adore(-0.15) | åģ¶åĥı(-0.15) | demon(-0.14) | angel(-0.14) | god(-0.14)
    ACCEPTED as axis_039  cumulative_var=0.1514

  [  35]  axes=40  step_var=0.0026  binary_acc=0.987  gap=0.2874  max_dot=0.0033  (1.8s)
    TOP:  WILL(0.14) | SHOULD(0.13) | citizen(0.13) | Muslims(0.13) | motivate(0.13) | british(0.12) | sustainable(0.12) | iger(0.12)
    BOT:  backdrop(-0.14) | rehearsal(-0.14) | meal(-0.13) | ceremony(-0.13) | ending(-0.13) | trou(-0.12) | termin(-0.12) | panorama(-0.12)
    ACCEPTED as axis_040  cumulative_var=0.1536

  [  36]  axes=41  step_var=0.0028  binary_acc=0.981  gap=0.2981  max_dot=0.0012  (1.8s)
    TOP:  æĲŃå»º(0.16) | ä¹ĭæĹħ(0.15) | ä¹ĭéģĵ(0.15) | modification(0.15) | ä¼ĺåĮĸ(0.14) | æŀĦå»º(0.14) | Sullivan(0.14) | æķ´åĲĪ(0.14)
    BOT:  deer(-0.14) | peanut(-0.14) | screaming(-0.14) | dollar(-0.13) | sperm(-0.13) | competitors(-0.13) | screams(-0.13) | Mercedes(-0.13)
    ACCEPTED as axis_041  cumulative_var=0.1560

  [  37]  axes=42  step_var=0.0026  binary_acc=0.965  gap=0.2857  max_dot=0.0064  (1.8s)
    TOP:  Teresa(0.13) | positively(0.13) | Prospect(0.13) | Rose(0.13) | prosper(0.13) | Rose(0.13) | loser(0.13) | -pop(0.13)
    BOT:  intent(-0.16) | oriented(-0.14) | intended(-0.13) | Intent(-0.13) | infr(-0.13) | å¼¯(-0.12) | æľºåºĬ(-0.12) | åĶ±(-0.12)
    ACCEPTED as axis_042  cumulative_var=0.1582

  [  38]  axes=43  step_var=0.0025  binary_acc=0.974  gap=0.2902  max_dot=0.0041  (1.8s)
    TOP:  dissolved(0.16) | sighed(0.15) | Dimit(0.13) | Wait(0.13) | swallowed(0.13) | ailments(0.12) | zombie(0.12) | blinked(0.12)
    BOT:  iconic(-0.16) | recognizable(-0.14) | attractive(-0.13) | decorative(-0.13) | recognition(-0.13) | contrasting(-0.13) | measurable(-0.13) | attraction(-0.13)
    ACCEPTED as axis_043  cumulative_var=0.1603

  [  39]  axes=44  step_var=0.0026  binary_acc=0.997  gap=0.2903  max_dot=0.0030  (1.9s)
    TOP:  referred(0.15) | implied(0.14) | described(0.13) | presumed(0.13) | Webster(0.13) | written(0.13) | preceded(0.13) | transcript(0.13)
    BOT:  comfortable(-0.14) | helm(-0.13) | Nate(-0.13) | tough(-0.13) | å¤ĩæĪĺ(-0.13) | Southampton(-0.13) | opponent(-0.13) | COMP(-0.13)
    ACCEPTED as axis_044  cumulative_var=0.1625

  [  40]  axes=45  step_var=0.0025  binary_acc=0.993  gap=0.2753  max_dot=0.0091  (1.9s)
    TOP:  calm(0.13) | èĲ½ä¸ĭ(0.13) | imits(0.13) | possess(0.13) | ÑģÐ¾Ð¾ÑĤÐ²ÐµÑĤ(0.13) | hats(0.12) | ASAP(0.12) | collar(0.12)
    BOT:  ref(-0.13) | precio(-0.13) | -interest(-0.12) | ....(-0.12) | anti(-0.12) | Interest(-0.12) | Rev(-0.12) | redevelopment(-0.12)
    ACCEPTED as axis_045  cumulative_var=0.1646

  [  41]  axes=46  step_var=0.0027  binary_acc=0.981  gap=0.2973  max_dot=0.0003  (1.8s)
    TOP:  æĸ©èİ·(0.15) | resurgence(0.15) | åĤ¨å¤ĩ(0.14) | increased(0.14) | increase(0.14) | LSU(0.14) | purchase(0.14) | hit(0.14)
    BOT:  poking(-0.14) | worrying(-0.13) | pointing(-0.13) | Mohammed(-0.13) | Orth(-0.13) | ok(-0.13) | worried(-0.13) | Orthodox(-0.13)
    ACCEPTED as axis_046  cumulative_var=0.1669

  [  42]  axes=47  step_var=0.0026  binary_acc=0.947  gap=0.2893  max_dot=0.0025  (1.8s)
    TOP:  inspire(0.13) | inspiring(0.13) | enabling(0.13) | Horizon(0.12) | skiing(0.12) | Taiwan(0.12) | Naomi(0.12) | resorts(0.12)
    BOT:  Patterson(-0.14) | erection(-0.14) | vigorously(-0.13) | markedly(-0.13) | cigarette(-0.13) | severed(-0.13) | ailments(-0.13) | vengeance(-0.13)
    ACCEPTED as axis_047  cumulative_var=0.1690

  [  43]  axes=48  step_var=0.0026  binary_acc=0.982  gap=0.2845  max_dot=0.0020  (1.8s)
    TOP:  throm(0.15) | consensus(0.15) | thermo(0.14) | Chris(0.14) | confirmed(0.14) | Chris(0.13) | complementary(0.13) | comedy(0.13)
    BOT:  desert(-0.16) | delightful(-0.15) | Mountain(-0.14) | peculiar(-0.14) | äºŃ(-0.14) | Mexican(-0.13) | unicorn(-0.13) | å±(-0.13)
    ACCEPTED as axis_048  cumulative_var=0.1712

  [  44]  axes=49  step_var=0.0024  binary_acc=0.998  gap=0.2771  max_dot=0.0026  (1.8s)
    TOP:  competed(0.14) | swept(0.13) | æİĢ(0.13) | conceived(0.12) | Cos(0.12) | cope(0.12) | Sie(0.12) | æĹ¶æķĪ(0.12)
    BOT:  indian(-0.13) | odd(-0.12) | strange(-0.12) | Bermuda(-0.12) | London(-0.12) | adverts(-0.12) | mentoring(-0.11) | london(-0.11)
    ACCEPTED as axis_049  cumulative_var=0.1732

  [  45]  axes=50  step_var=0.0024  binary_acc=0.923  gap=0.2850  max_dot=0.0048  (1.8s)
    TOP:  horrific(0.15) | æģĲæĢĸ(0.15) | incredibly(0.14) | incredible(0.14) | stolen(0.14) | legally(0.14) | integral(0.13) | haul(0.13)
    BOT:  emphasizing(-0.13) | guides(-0.12) | aster(-0.12) | obstacles(-0.12) | discour(-0.12) | deciding(-0.12) | Nicholas(-0.12) | istles(-0.11)
    ACCEPTED as axis_050  cumulative_var=0.1751

  [  46]  axes=51  step_var=0.0025  binary_acc=0.989  gap=0.2797  max_dot=0.0008  (1.8s)
    TOP:  dancer(0.14) | pilot(0.13) | fights(0.13) | drink(0.13) | rent(0.12) | Nigel(0.12) | Pilot(0.12) | bicycle(0.12)
    BOT:  Commerce(-0.15) | Mer(-0.14) | Emp(-0.14) | Com(-0.14) | Sem(-0.13) | mem(-0.13) | Com(-0.13) | ç¾İ(-0.13)
    ACCEPTED as axis_051  cumulative_var=0.1772

  [  47]  axes=52  step_var=0.0024  binary_acc=0.994  gap=0.2871  max_dot=0.0012  (1.8s)
    TOP:  enriched(0.14) | innings(0.14) | fright(0.13) | spo(0.13) | onset(0.13) | çĪ½(0.13) | hit(0.13) | onslaught(0.13)
    BOT:  cardboard(-0.14) | Frederick(-0.14) | agregar(-0.14) | Cedar(-0.14) | regardless(-0.14) | plastic(-0.13) | cedar(-0.13) | recognizes(-0.13)
    ACCEPTED as axis_052  cumulative_var=0.1793

  [  48]  axes=53  step_var=0.0025  binary_acc=0.957  gap=0.2888  max_dot=0.0013  (1.8s)
    TOP:  grew(0.14) | sovereignty(0.14) | surgery(0.14) | Ginny(0.13) | marry(0.13) | grow(0.13) | Surgery(0.13) | synergy(0.13)
    BOT:  unpleasant(-0.16) | plat(-0.15) | gloves(-0.13) | headaches(-0.13) | Ð±ÐµÑģÐ¿Ð»Ð°ÑĤ(-0.13) | trailers(-0.13) | Flash(-0.13) | æłĩé¢ĺ(-0.13)
    ACCEPTED as axis_053  cumulative_var=0.1814

  [  49]  axes=54  step_var=0.0024  binary_acc=0.990  gap=0.2839  max_dot=0.0013  (1.8s)
    TOP:  onset(0.16) | mindset(0.14) | rationale(0.14) | baseline(0.14) | æľºåĪ¶(0.14) | ese(0.13) | åŁºçŁ³(0.13) | consensus(0.13)
    BOT:  smtp(-0.14) | charming(-0.14) | Hampton(-0.14) | tablespoons(-0.14) | smtp(-0.13) | starring(-0.13) | tam(-0.13) | chauff(-0.13)
    ACCEPTED as axis_054  cumulative_var=0.1833

  [  50]  axes=55  step_var=0.0024  binary_acc=0.994  gap=0.2711  max_dot=0.0033  (1.9s)
    TOP:  nutrition(0.14) | Presentation(0.13) | distress(0.13) | differentiation(0.12) | differ(0.12) | tennis(0.12) | bless(0.12) | kettle(0.12)
    BOT:  opted(-0.14) | protagon(-0.14) | symptoms(-0.13) | opting(-0.13) | ä¸»è§Ĵ(-0.12) | statutory(-0.12) | -founder(-0.12) | protagonist(-0.12)
    ACCEPTED as axis_055  cumulative_var=0.1853

  [  51]  axes=56  step_var=0.0025  binary_acc=0.994  gap=0.2804  max_dot=0.0024  (1.8s)
    TOP:  Bug(0.14) | cute(0.13) | insult(0.12) | sadly(0.12) | nickname(0.12) | Drugs(0.12) | DVD(0.12) | supermarket(0.12)
    BOT:  sitting(-0.15) | rising(-0.14) | æ³¢åĬ¨(-0.14) | æĮ£(-0.14) | issuance(-0.14) | ç¦»éĢĢä¼ĳ(-0.13) | possession(-0.13) | gatherings(-0.13)
    ACCEPTED as axis_056  cumulative_var=0.1873

  [  52]  axes=57  step_var=0.0025  binary_acc=0.978  gap=0.2804  max_dot=0.0034  (1.8s)
    TOP:  relied(0.13) | sparkle(0.13) | excel(0.13) | silk(0.13) | silica(0.13) | Excel(0.12) | cyber(0.12) | é¡ºä¸°(0.12)
    BOT:  pregnant(-0.14) | married(-0.14) | teacher(-0.14) | å©ļå§»(-0.13) | dressed(-0.13) | priesthood(-0.13) | marriage(-0.13) | marrying(-0.13)
    ACCEPTED as axis_057  cumulative_var=0.1893

  [  53]  axes=58  step_var=0.0025  binary_acc=0.971  gap=0.2830  max_dot=0.0018  (1.8s)
    TOP:  improvements(0.17) | projections(0.16) | improvement(0.15) | improvis(0.14) | hikes(0.14) | impacts(0.13) | increases(0.13) | impressive(0.12)
    BOT:  outside(-0.12) | batter(-0.12) | redeem(-0.12) | utterly(-0.12) | sede(-0.12) | subject(-0.12) | ä»ĸäºº(-0.12) | eve(-0.12)
    ACCEPTED as axis_058  cumulative_var=0.1913

  [  54]  axes=59  step_var=0.0024  binary_acc=0.982  gap=0.2764  max_dot=0.0016  (1.9s)
    TOP:  revoked(0.14) | wallet(0.13) | Ð¿ÑĢÐ¸Ð½(0.13) | convicted(0.12) | grat(0.12) | excit(0.12) | driveway(0.12) | é»ĳé¾Ļæ±Łçľģ(0.12)
    BOT:  tumble(-0.13) | æ·±èĢķ(-0.13) | abel(-0.13) | bends(-0.13) | Panda(-0.13) | babel(-0.12) | æ¦ľæł·(-0.12) | ela(-0.12)
    ACCEPTED as axis_059  cumulative_var=0.1933

  [  55]  axes=60  step_var=0.0025  binary_acc=0.983  gap=0.2778  max_dot=0.0044  (1.8s)
    TOP:  Tennessee(0.17) | demonstrations(0.14) | testimony(0.14) | flashlight(0.14) | demonstrators(0.13) | .HttpServletRequest(0.13) | stronghold(0.13) | å¿ĹæĦ¿èĢħ(0.13)
    BOT:  multinational(-0.13) | adapted(-0.13) | prag(-0.13) | coupled(-0.13) | Corporation(-0.13) | allied(-0.12) | Ltd(-0.12) | packaged(-0.12)
    ACCEPTED as axis_060  cumulative_var=0.1953

  [  56]  axes=61  step_var=0.0024  binary_acc=0.983  gap=0.2743  max_dot=0.0011  (1.9s)
    TOP:  diary(0.14) | myths(0.13) | åıĳéĢģ(0.13) | æ¯Ķäºļè¿ª(0.12) | paths(0.12) | Byron(0.12) | metabolic(0.12) | Sales(0.12)
    BOT:  pocket(-0.16) | shelter(-0.13) | hug(-0.13) | clinic(-0.12) | imp(-0.12) | incorporate(-0.12) | coleg(-0.12) | ugging(-0.12)
    ACCEPTED as axis_061  cumulative_var=0.1972

  [  57]  axes=62  step_var=0.0025  binary_acc=0.989  gap=0.2768  max_dot=0.0023  (1.8s)
    TOP:  CDC(0.13) | touchdowns(0.13) | CNC(0.13) | TCP(0.12) | deduct(0.12) | dental(0.12) | Dys(0.12) | fiscal(0.12)
    BOT:  preparations(-0.16) | Am(-0.14) | beg(-0.13) | Pig(-0.13) | require(-0.13) | Airways(-0.13) | rig(-0.12) | refinery(-0.12)
    ACCEPTED as axis_062  cumulative_var=0.1992

  [  58]  axes=63  step_var=0.0024  binary_acc=0.935  gap=0.2722  max_dot=0.0019  (1.8s)
    TOP:  adjustable(0.14) | referral(0.13) | Austin(0.13) | turf(0.13) | overlooked(0.12) | enrollment(0.12) | Bristol(0.12) | attendance(0.12)
    BOT:  ani(-0.14) | lover(-0.13) | han(-0.13) | Part(-0.13) | æ¼«çĶ»(-0.13) | pan(-0.12) | pict(-0.12) | Ð¼Ð°Ð½(-0.12)
    ACCEPTED as axis_063  cumulative_var=0.2011

  [  59]  axes=64  step_var=0.0023  binary_acc=0.989  gap=0.2674  max_dot=0.0059  (1.9s)
    TOP:  èį¯å¸Ī(0.13) | å·¡æŁ¥(0.13) | Clark(0.13) | voters(0.12) | alors(0.12) | fascinating(0.12) | å¿ĹæĦ¿èĢħ(0.12) | cybersecurity(0.12)
    BOT:  footprint(-0.13) | behaviour(-0.13) | repertoire(-0.13) | execut(-0.12) | manufacture(-0.12) | é«ĺä½į(-0.12) | movements(-0.11) | åŃĶ(-0.11)
    ACCEPTED as axis_064  cumulative_var=0.2030

  [  60]  axes=65  step_var=0.0022  binary_acc=0.967  gap=0.2615  max_dot=0.0024  (1.8s)
    TOP:  Perkins(0.12) | alien(0.12) | arsenal(0.11) | expectancy(0.11) | æ½(0.11) | æİı(0.11) | Alec(0.11) | MAK(0.11)
    BOT:  unist(-0.13) | nouns(-0.13) | glo(-0.12) | Cron(-0.12) | costume(-0.12) | subdued(-0.12) | Claud(-0.12) | rue(-0.12)
    ACCEPTED as axis_065  cumulative_var=0.2047

  [  61]  axes=66  step_var=0.0023  binary_acc=0.976  gap=0.2664  max_dot=0.0035  (1.8s)
    TOP:  soils(0.13) | dull(0.12) | ados(0.12) | improbable(0.12) | STREET(0.11) | Efficiency(0.11) | manner(0.11) | fool(0.11)
    BOT:  conflic(-0.13) | license(-0.12) | consultation(-0.12) | distra(-0.12) | chemin(-0.12) | Mt(-0.12) | æķĳæı´(-0.12) | çĤ¸å¼¹(-0.12)
    ACCEPTED as axis_066  cumulative_var=0.2065

  [  62]  axes=67  step_var=0.0022  binary_acc=0.992  gap=0.2609  max_dot=0.0033  (1.9s)
    TOP:  ç®¡æİ§(0.14) | èį¼(0.14) | centres(0.13) | incidents(0.13) | instances(0.13) | Sep(0.13) | peter(0.12) | åįķè½¦(0.12)
    BOT:  musical(-0.13) | cursor(-0.12) | æĻļå¹´(-0.12) | laisse(-0.12) | åĲĪæ³ķ(-0.12) | payload(-0.12) | gaze(-0.12) | bypass(-0.12)
    ACCEPTED as axis_067  cumulative_var=0.2083

  [  63]  axes=68  step_var=0.0024  binary_acc=0.992  gap=0.2712  max_dot=0.0033  (1.8s)
    TOP:  audible(0.15) | glean(0.14) | Budget(0.13) | fluorescence(0.13) | revenue(0.13) | æ²¹çĥŁ(0.12) | Aud(0.12) | budgets(0.12)
    BOT:  dagger(-0.12) | frameworks(-0.12) | -Compatible(-0.12) | Platform(-0.12) | snake(-0.12) | shark(-0.12) | éľ¸(-0.12) | èģĶæīĭ(-0.11)
    ACCEPTED as axis_068  cumulative_var=0.2103

  [  64]  axes=69  step_var=0.0024  binary_acc=0.948  gap=0.2658  max_dot=0.0100  (1.8s)
    TOP:  Dance(0.14) | Pang(0.14) | ä½©(0.13) | fly(0.13) | dance(0.12) | dolphin(0.12) | Pell(0.12) | ale(0.12)
    BOT:  çº¦ç¿°(-0.14) | å¸ļ(-0.13) | baum(-0.12) | trium(-0.12) | Brian(-0.12) | idiot(-0.11) | åĭĭ(-0.11) | obstruct(-0.11)
    ACCEPTED as axis_069  cumulative_var=0.2122

  [  65]  axes=70  step_var=0.0025  binary_acc=0.980  gap=0.2710  max_dot=0.0026  (1.8s)
    TOP:  Leon(0.14) | righteousness(0.13) | ä¸¥å¯Ĩ(0.13) | Honor(0.13) | Dios(0.13) | superiority(0.12) | honor(0.12) | Majesty(0.12)
    BOT:  retaining(-0.14) | gut(-0.14) | ä¿¡ç®±(-0.13) | cottage(-0.13) | è®¿è°Ī(-0.13) | Collins(-0.13) | concerns(-0.13) | èī²ç´ł(-0.12)
    ACCEPTED as axis_070  cumulative_var=0.2141

  [  66]  axes=71  step_var=0.0023  binary_acc=0.990  gap=0.2670  max_dot=0.0075  (1.8s)
    TOP:  unders(0.14) | seam(0.14) | consultation(0.13) | ç¯±(0.12) | insulated(0.12) | ethylene(0.12) | hind(0.12) | tight(0.12)
    BOT:  çºłæŃ£(-0.12) | åħ´å¥ĭ(-0.12) | ricks(-0.12) | crackers(-0.12) | suspicious(-0.12) | appreciation(-0.11) | çºª(-0.11) | åħĳçİ°(-0.11)
    ACCEPTED as axis_071  cumulative_var=0.2159

  [  67]  axes=72  step_var=0.0024  binary_acc=0.988  gap=0.2771  max_dot=0.0013  (1.8s)
    TOP:  lens(0.14) | eliminated(0.14) | eliminates(0.14) | embarrassed(0.14) | Ray(0.13) | gay(0.12) | tracted(0.12) | rect(0.12)
    BOT:  adapt(-0.14) | authenticity(-0.14) | estim(-0.13) | aspiring(-0.13) | Maple(-0.13) | çĿ£ä¿ĥ(-0.12) | adverse(-0.12) | algae(-0.12)
    ACCEPTED as axis_072  cumulative_var=0.2178

  [  68]  axes=73  step_var=0.0023  binary_acc=0.994  gap=0.2621  max_dot=0.0032  (1.8s)
    TOP:  prevalence(0.13) | alla(0.12) | occasionally(0.12) | necessity(0.12) | adds(0.12) | stra(0.12) | Stella(0.12) | McK(0.11)
    BOT:  Hum(-0.13) | åĪĨæīĭ(-0.13) | stepped(-0.12) | Computational(-0.12) | romantic(-0.12) | è£´(-0.12) | è¸(-0.12) | Toe(-0.12)
    ACCEPTED as axis_073  cumulative_var=0.2196

  [  69]  axes=74  step_var=0.0023  binary_acc=0.956  gap=0.2654  max_dot=0.0027  (1.8s)
    TOP:  çĽĲåŁİ(0.13) | ì§ģ(0.13) | milk(0.13) | Healthy(0.12) | untary(0.12) | poverty(0.12) | vulnerable(0.12) | é¸¡èĤī(0.12)
    BOT:  Creator(-0.14) | å·¨å¤´(-0.13) | intimidation(-0.13) | operator(-0.13) | moderator(-0.12) | dominate(-0.12) | operators(-0.12) | detr(-0.12)
    ACCEPTED as axis_074  cumulative_var=0.2214

  [  70]  axes=75  step_var=0.0023  binary_acc=0.982  gap=0.2615  max_dot=0.0058  (1.9s)
    TOP:  overcrow(0.12) | built(0.12) | Built(0.12) | ä½ľé£İå»ºè®¾(0.12) | erection(0.12) | ÑħÐ¾Ñĩ(0.11) | busy(0.11) | nicer(0.11)
    BOT:  tang(-0.14) | palm(-0.13) | æĬ¥èŃ¦(-0.12) | thal(-0.12) | bargaining(-0.12) | åĪĨç¦»(-0.12) | discern(-0.12) | Magnum(-0.12)
    ACCEPTED as axis_075  cumulative_var=0.2232

  [  71]  axes=76  step_var=0.0023  binary_acc=0.996  gap=0.2662  max_dot=0.0020  (1.8s)
    TOP:  stops(0.13) | mistake(0.13) | stop(0.13) | departing(0.12) | navigate(0.12) | checkout(0.12) | kah(0.12) | Diane(0.12)
    BOT:  wield(-0.13) | äººæĢ§(-0.12) | intrinsic(-0.12) | Lens(-0.12) | å±ŀæĢ§(-0.12) | æŃ¦ä¾ł(-0.12) | Avengers(-0.12) | Ultra(-0.12)
    ACCEPTED as axis_076  cumulative_var=0.2250

  [  72]  axes=77  step_var=0.0023  binary_acc=0.982  gap=0.2632  max_dot=0.0019  (1.8s)
    TOP:  guests(0.15) | requests(0.14) | pam(0.13) | Giuliani(0.13) | luxurious(0.13) | generously(0.13) | Dem(0.12) | Pamela(0.12)
    BOT:  Mars(-0.14) | Arbor(-0.13) | é¸£(-0.13) | Voices(-0.13) | æĺİ(-0.12) | Bark(-0.12) | Beginning(-0.12) | brighter(-0.11)
    ACCEPTED as axis_077  cumulative_var=0.2267

  [  73]  axes=78  step_var=0.0023  binary_acc=0.986  gap=0.2553  max_dot=0.0057  (1.9s)
    TOP:  Plaintiff(0.13) | remnants(0.12) | plaintiff(0.12) | Hou(0.12) | èĢģæĹ§(0.12) | Phillips(0.12) | fond(0.12) | åŃļ(0.11)
    BOT:  baking(-0.14) | Krak(-0.12) | barcode(-0.12) | å½ķåıĸ(-0.12) | barley(-0.11) | collaborations(-0.11) | knees(-0.11) | arkin(-0.11)
    ACCEPTED as axis_078  cumulative_var=0.2285

  [  74]  axes=79  step_var=0.0022  binary_acc=0.957  gap=0.2523  max_dot=0.0027  (1.9s)
    TOP:  Nice(0.13) | flatt(0.12) | spite(0.12) | Capt(0.12) | Nice(0.12) | Patriots(0.11) | nice(0.11) | nice(0.11)
    BOT:  oversees(-0.12) | Karen(-0.11) | å°ıåŃ©åŃĲ(-0.11) | hierarchical(-0.11) | sembles(-0.11) | circumference(-0.11) | Karen(-0.11) | Eb(-0.11)
    ACCEPTED as axis_079  cumulative_var=0.2302

  [  75]  axes=80  step_var=0.0024  binary_acc=0.980  gap=0.2654  max_dot=0.0003  (1.9s)
    TOP:  failure(0.14) | serial(0.14) | fantasies(0.13) | fanatic(0.12) | ç»įåħ´(0.12) | dÃ©but(0.12) | fantasy(0.12) | asy(0.12)
    BOT:  glow(-0.12) | æľĭ(-0.12) | .classList(-0.12) | Lip(-0.12) | GFP(-0.12) | Hopefully(-0.12) | grounded(-0.11) | hurdle(-0.11)
    ACCEPTED as axis_080  cumulative_var=0.2320

  [  76]  axes=81  step_var=0.0022  binary_acc=0.990  gap=0.2596  max_dot=0.0019  (1.8s)
    TOP:  practitioner(0.13) | åĲŁ(0.13) | æİ¨è¡Į(0.12) | åĴĮå°ļ(0.12) | practicing(0.12) | è¡Įä¸ļåĨħ(0.12) | è¡£çī©(0.12) | movement(0.12)
    BOT:  aston(-0.15) | Patterson(-0.15) | reun(-0.15) | conn(-0.12) | Carlson(-0.12) | reson(-0.11) | curtain(-0.11) | Apart(-0.11)
    ACCEPTED as axis_081  cumulative_var=0.2338

  [  77]  axes=82  step_var=0.0023  binary_acc=0.974  gap=0.2618  max_dot=0.0003  (1.8s)
    TOP:  cript(0.14) | boon(0.13) | craving(0.13) | CRA(0.13) | Worst(0.12) | cv(0.12) | humanitarian(0.12) | çĹĽçĤ¹(0.12)
    BOT:  éº»å°Ĩ(-0.13) | pairs(-0.13) | ilingual(-0.13) | pair(-0.13) | ber(-0.13) | ãģłãģĳãģ§(-0.12) | replaced(-0.12) | æľįé¥°(-0.12)
    ACCEPTED as axis_082  cumulative_var=0.2355

  [  78]  axes=83  step_var=0.0024  binary_acc=0.995  gap=0.2682  max_dot=0.0020  (1.8s)
    TOP:  pur(0.15) | PUR(0.15) | silence(0.13) | Aud(0.13) | suspension(0.12) | schl(0.12) | Pur(0.12) | suede(0.12)
    BOT:  metast(-0.14) | æĬµæĮ¡(-0.13) | ä»£è°¢(-0.13) | lig(-0.12) | toxic(-0.12) | resorts(-0.12) | debt(-0.12) | æĪļ(-0.12)
    ACCEPTED as axis_083  cumulative_var=0.2373

  [  79]  axes=84  step_var=0.0024  binary_acc=0.991  gap=0.2629  max_dot=0.0049  (1.8s)
    TOP:  apparent(0.13) | chassis(0.13) | dwell(0.12) | encompass(0.12) | convoy(0.12) | trak(0.12) | ATV(0.12) | hp(0.11)
    BOT:  ife(-0.12) | è½®èŀįèµĦ(-0.12) | cucumber(-0.12) | çĭ¬è§Ĵåħ½(-0.12) | fundraising(-0.12) | ucumber(-0.12) | kne(-0.12) | eating(-0.11)
    ACCEPTED as axis_084  cumulative_var=0.2391

  [  80]  axes=85  step_var=0.0023  binary_acc=0.992  gap=0.2570  max_dot=0.0008  (1.9s)
    TOP:  èµŀåĲĮ(0.13) | puntos(0.13) | éĴ¨(0.12) | bracket(0.12) | excited(0.12) | chick(0.11) | è¦ģçĤ¹(0.11) | Yankee(0.11)
    BOT:  rein(-0.13) | pel(-0.13) | personal(-0.12) | adel(-0.12) | navigating(-0.12) | Admin(-0.12) | burning(-0.12) | Repository(-0.12)
    ACCEPTED as axis_085  cumulative_var=0.2408

  [  81]  axes=86  step_var=0.0023  binary_acc=0.988  gap=0.2598  max_dot=0.0045  (1.9s)
    TOP:  adolescent(0.13) | stereotypes(0.13) | CON(0.12) | Dad(0.12) | reprodu(0.11) | Lou(0.11) | savage(0.11) | UD(0.11)
    BOT:  infinity(-0.12) | Infinity(-0.12) | Philipp(-0.12) | anz(-0.12) | Fischer(-0.12) | pass(-0.11) | inn(-0.11) | orchestra(-0.11)
    ACCEPTED as axis_086  cumulative_var=0.2426

  [  82]  axes=87  step_var=0.0022  binary_acc=0.984  gap=0.2561  max_dot=0.0014  (1.8s)
    TOP:  lurking(0.13) | gest(0.12) | Burr(0.12) | Montana(0.12) | ge(0.12) | çĴŁ(0.12) | BT(0.12) | looming(0.11)
    BOT:  pedal(-0.12) | æĻ¨æĬ¥(-0.12) | sharply(-0.12) | conference(-0.11) | cereal(-0.11) | piano(-0.11) | Radio(-0.11) | lessen(-0.11)
    ACCEPTED as axis_087  cumulative_var=0.2443

  [  83]  axes=88  step_var=0.0022  binary_acc=0.988  gap=0.2489  max_dot=0.0029  (1.8s)
    TOP:  Companion(0.13) | jen(0.12) | æ¶ĪèĢĹ(0.12) | åĨľæ°ĳ(0.11) | companion(0.11) | %%(0.11) | å®¶çĶµ(0.11) | kill(0.11)
    BOT:  Street(-0.13) | issu(-0.12) | Salman(-0.12) | issuer(-0.12) | statement(-0.12) | Senate(-0.12) | steel(-0.11) | Arr(-0.11)
    ACCEPTED as axis_088  cumulative_var=0.2459

  [  84]  axes=89  step_var=0.0022  binary_acc=0.981  gap=0.2556  max_dot=0.0006  (1.8s)
    TOP:  Granite(0.13) | Polit(0.12) | itzer(0.12) | stained(0.12) | åıĳèĤ²(0.12) | expenditures(0.12) | femin(0.12) | èĥ½å¤Ł(0.12)
    BOT:  refreshing(-0.12) | unsettling(-0.12) | celebration(-0.11) | Arc(-0.11) | Andr(-0.11) | ARC(-0.11) | åħ³èĬĤ(-0.11) | orsi(-0.11)
    ACCEPTED as axis_089  cumulative_var=0.2476

  [  85]  axes=90  step_var=0.0022  binary_acc=0.991  gap=0.2495  max_dot=0.0008  (1.8s)
    TOP:  warehouse(0.13) | hyster(0.13) | çĶŁäº§è½¦éĹ´(0.12) | Humph(0.12) | aftermath(0.12) | chor(0.12) | chez(0.12) | Massachusetts(0.12)
    BOT:  slogan(-0.14) | è¡«(-0.12) | å¼¹(-0.12) | PN(-0.12) | troops(-0.11) | æĿī(-0.11) | æĭĸå»¶(-0.11) | crop(-0.11)
    ACCEPTED as axis_090  cumulative_var=0.2493

  [  86]  axes=91  step_var=0.0022  binary_acc=0.983  gap=0.2534  max_dot=0.0012  (1.9s)
    TOP:  Guth(0.12) | sprite(0.12) | gamers(0.12) | æĬĸ(0.12) | gro(0.12) | youngsters(0.12) | gamer(0.11) | prise(0.11)
    BOT:  toxic(-0.12) | Toxic(-0.12) | Maj(-0.12) | soak(-0.12) | oak(-0.11) | çħİ(-0.11) | linen(-0.11) | ocene(-0.11)
    ACCEPTED as axis_091  cumulative_var=0.2509

  [  87]  axes=92  step_var=0.0022  binary_acc=0.989  gap=0.2541  max_dot=0.0023  (1.9s)
    TOP:  film(0.12) | film(0.12) | Bond(0.11) | Jets(0.11) | revis(0.11) | dise(0.11) | convers(0.11) | equity(0.11)
    BOT:  Claudia(-0.12) | å±ł(-0.12) | strictly(-0.12) | éĤ±(-0.12) | Go(-0.11) | quota(-0.11) | cleanup(-0.11) | cou(-0.11)
    ACCEPTED as axis_092  cumulative_var=0.2526

  [  88]  axes=93  step_var=0.0022  binary_acc=0.984  gap=0.2521  max_dot=0.0022  (1.9s)
    TOP:  Canadian(0.14) | greater(0.13) | impacts(0.12) | affects(0.12) | Carrie(0.12) | hypers(0.11) | Greater(0.11) | å¹¿æ±½(0.11)
    BOT:  -oriented(-0.13) | Klein(-0.12) | kotlin(-0.12) | versatility(-0.12) | obvious(-0.12) | Tribunal(-0.12) | vain(-0.12) | Lobby(-0.11)
    ACCEPTED as axis_093  cumulative_var=0.2543

  [  89]  axes=94  step_var=0.0022  binary_acc=0.994  gap=0.2535  max_dot=0.0010  (1.8s)
    TOP:  bush(0.14) | cheque(0.13) | åħ¬è¯ģ(0.13) | elephants(0.13) | immutable(0.12) | Bush(0.11) | carving(0.11) | å¾®éĩı(0.11)
    BOT:  obligated(-0.13) | awaken(-0.12) | Oakland(-0.12) | reacting(-0.12) | Omni(-0.12) | Columbia(-0.12) | åĵª(-0.11) | Discussions(-0.11)
    ACCEPTED as axis_094  cumulative_var=0.2559

  [  90]  axes=95  step_var=0.0022  binary_acc=0.941  gap=0.2527  max_dot=0.0037  (1.8s)
    TOP:  Monster(0.11) | Bulletin(0.11) | Baptist(0.11) | varchar(0.11) | boobs(0.11) | .ST(0.10) | primal(0.10) | bulletin(0.10)
    BOT:  appreciated(-0.13) | Ð¸ÑģÐ¿Ð¾Ð»ÑĮÐ·Ð¾Ð²(-0.13) | ä¸ľåįĹ(-0.12) | ç»¼åĲĪåĪ©çĶ¨(-0.12) | appreciate(-0.12) | copper(-0.12) | reused(-0.12) | thanks(-0.12)
    ACCEPTED as axis_095  cumulative_var=0.2575

  [  91]  axes=96  step_var=0.0022  binary_acc=0.985  gap=0.2531  max_dot=0.0009  (1.9s)
    TOP:  Philip(0.14) | Susan(0.13) | Suk(0.13) | Susan(0.12) | nerv(0.12) | uncomfortable(0.11) | surface(0.11) | useful(0.11)
    BOT:  Imm(-0.13) | quadr(-0.13) | åľ°è´¨(-0.12) | Legend(-0.12) | dearly(-0.12) | remedy(-0.12) | Colt(-0.12) | breadth(-0.12)
    ACCEPTED as axis_096  cumulative_var=0.2591

  [  92]  axes=97  step_var=0.0022  binary_acc=0.986  gap=0.2527  max_dot=0.0016  (1.8s)
    TOP:  åħ¬çĽĬ(0.13) | é«ĺå°Ķå¤«(0.12) | Guinness(0.12) | gratitude(0.11) | exquisite(0.11) | revolving(0.11) | Kelly(0.11) | explosive(0.11)
    BOT:  usur(-0.13) | bear(-0.12) | Thor(-0.12) | tad(-0.12) | Nim(-0.12) | å¿į(-0.11) | hor(-0.11) | åī¿(-0.11)
    ACCEPTED as axis_097  cumulative_var=0.2608

  [  93]  axes=98  step_var=0.0024  binary_acc=0.965  gap=0.2628  max_dot=0.0009  (1.8s)
    TOP:  stickers(0.14) | sticky(0.13) | å·¥ä¸ļåĮĸ(0.12) | bubble(0.12) | collaborations(0.12) | unf(0.12) | mosaic(0.12) | budget(0.12)
    BOT:  æĿĳå¹²éĥ¨(-0.12) | shrimp(-0.12) | âĢ¦.(-0.12) | dismissed(-0.12) | Definition(-0.11) | shocked(-0.11) | deport(-0.11) | dolphins(-0.11)
    ACCEPTED as axis_098  cumulative_var=0.2625

  [  94]  axes=99  step_var=0.0022  binary_acc=0.987  gap=0.2530  max_dot=0.0045  (1.8s)
    TOP:  gasoline(0.13) | candy(0.12) | generate(0.11) | troubling(0.11) | SET(0.11) | baseball(0.11) | Sprint(0.11) | GT(0.11)
    BOT:  clinging(-0.12) | Clo(-0.12) | lush(-0.12) | à¸Ĭ(-0.12) | èĩ»(-0.12) | éĴ¾(-0.11) | inch(-0.11) | calm(-0.11)
    ACCEPTED as axis_099  cumulative_var=0.2642

  [  95]  axes=100  step_var=0.0023  binary_acc=0.984  gap=0.2568  max_dot=0.0115  (1.8s)
    TOP:  Env(0.12) | Hon(0.12) | severity(0.12) | Hur(0.12) | critical(0.12) | háº¥p(0.12) | deserves(0.12) | deserve(0.12)
    BOT:  PlayStation(-0.13) | everytime(-0.12) | pand(-0.12) | autofocus(-0.12) | æīŃè½¬(-0.12) | tunes(-0.12) | MC(-0.12) | faithful(-0.11)
    ACCEPTED as axis_100  cumulative_var=0.2659

  [  96]  axes=101  step_var=0.0021  binary_acc=0.996  gap=0.2430  max_dot=0.0035  (1.8s)
    TOP:  species(0.12) | Sigma(0.11) | configurations(0.11) | wife(0.11) | Species(0.11) | conceive(0.11) | å¦»åŃĲ(0.11) | ÑĢÐ¾Ð¶(0.11)
    BOT:  à¸ªà¸³à¸Ħ(-0.13) | Ministry(-0.13) | Mall(-0.13) | stom(-0.12) | Ð¼Ð¾ÑĤÑĢ(-0.12) | Zoo(-0.12) | æĭŃ(-0.11) | Mills(-0.11)
    ACCEPTED as axis_101  cumulative_var=0.2675

  [  97]  axes=102  step_var=0.0022  binary_acc=0.991  gap=0.2514  max_dot=0.0013  (1.8s)
    TOP:  gospel(0.13) | spokes(0.12) | Judy(0.12) | jud(0.11) | Lutheran(0.11) | çīĻé½¿(0.11) | Gospel(0.11) | Rud(0.11)
    BOT:  welcoming(-0.13) | onc(-0.13) | aquel(-0.13) | è§ĦåĪĴå»ºè®¾(-0.12) | èĩ´çĻĮ(-0.12) | attractive(-0.12) | çĶµå½±èĬĤ(-0.11) | AU(-0.11)
    ACCEPTED as axis_102  cumulative_var=0.2691

  [  98]  axes=103  step_var=0.0023  binary_acc=0.978  gap=0.2595  max_dot=0.0056  (1.8s)
    TOP:  rid(0.13) | æįķé±¼(0.13) | Bryant(0.12) | avirus(0.12) | cannabinoids(0.12) | productive(0.12) | expressing(0.12) | rotein(0.12)
    BOT:  partnered(-0.12) | è¶ħçº§(-0.12) | åĩºå¸Ń(-0.12) | McC(-0.12) | concert(-0.12) | wearable(-0.11) | maternal(-0.11) | Suzanne(-0.11)
    ACCEPTED as axis_103  cumulative_var=0.2707

  [  99]  axes=104  step_var=0.0022  binary_acc=0.920  gap=0.2569  max_dot=0.0028  (1.9s)
    TOP:  blueprint(0.14) | relying(0.14) | relies(0.13) | rely(0.13) | battleground(0.13) | neatly(0.13) | atop(0.12) | HQ(0.12)
    BOT:  Katrina(-0.12) | Salem(-0.12) | drawbacks(-0.12) | nineteen(-0.12) | add(-0.12) | èŀįèµĦç§Łèµģ(-0.11) | Arkansas(-0.11) | ç¤¾ä¼ļä¿ĿéĻ©(-0.11)
    ACCEPTED as axis_104  cumulative_var=0.2723

  [ 100]  axes=105  step_var=0.0022  binary_acc=0.998  gap=0.2446  max_dot=0.0056  (1.8s)
    TOP:  cont(0.11) | trunc(0.11) | æħĪ(0.11) | GV(0.11) | åĩłåįģå¹´(0.11) | é½Ĳé²ģ(0.11) | obligatory(0.11) | omb(0.11)
    BOT:  hazard(-0.12) | Rew(-0.12) | predis(-0.11) | Unable(-0.11) | ä¼¼ä¹İæĺ¯(-0.11) | sou(-0.11) | Asset(-0.11) | rew(-0.11)
    ACCEPTED as axis_105  cumulative_var=0.2739

  [ 101]  axes=106  step_var=0.0022  binary_acc=0.996  gap=0.2446  max_dot=0.0021  (1.9s)
    TOP:  dazz(0.12) | deals(0.12) | çĢļ(0.12) | dere(0.11) | Robin(0.11) | DataTable(0.11) | descon(0.11) | dramatically(0.11)
    BOT:  suspicious(-0.12) | languages(-0.12) | igious(-0.12) | lingu(-0.11) | æĹ§(-0.11) | Cultural(-0.11) | ourd(-0.11) | longstanding(-0.11)
    ACCEPTED as axis_106  cumulative_var=0.2755

  [ 102]  axes=107  step_var=0.0023  binary_acc=0.980  gap=0.2522  max_dot=0.0013  (1.9s)
    TOP:  Richard(0.13) | relationships(0.12) | connection(0.12) | iam(0.12) | relaciones(0.12) | rhythm(0.12) | credit(0.11) | connection(0.11)
    BOT:  translateY(-0.14) | lev(-0.13) | chuyá»ĥn(-0.12) | temporarily(-0.12) | éĩıäº§(-0.12) | èĦ±è´«æĶ»åĿļ(-0.11) | Superv(-0.11) | env(-0.11)
    ACCEPTED as axis_107  cumulative_var=0.2771

  [ 103]  axes=108  step_var=0.0022  binary_acc=0.975  gap=0.2465  max_dot=0.0013  (1.9s)
    TOP:  ovich(0.13) | copper(0.12) | firmly(0.12) | _READ(0.11) | arithmetic(0.11) | Chrom(0.11) | åŃĹç¬¦ä¸²(0.11) | itch(0.11)
    BOT:  extent(-0.12) | Je(-0.12) | ¦Ĥ(-0.12) | çĽĳçĲĨ(-0.12) | çŃ±(-0.11) | essence(-0.11) | çľ¼çķĮ(-0.11) | ÑĢÐ°Ð¼ÐºÐ°Ñħ(-0.11)
    ACCEPTED as axis_108  cumulative_var=0.2787

  [ 104]  axes=109  step_var=0.0022  binary_acc=0.971  gap=0.2451  max_dot=0.0017  (1.8s)
    TOP:  Solid(0.12) | challenged(0.11) | Tennis(0.11) | affirm(0.10) | Shed(0.10) | simp(0.10) | hitters(0.10) | å¤§åľ°(0.10)
    BOT:  é£İä¿Ĺ(-0.13) | customized(-0.12) | exact(-0.12) | exclus(-0.12) | broadcast(-0.12) | ä¸»ç®¡(-0.12) | meters(-0.12) | ç§ĺä¹¦éķ¿(-0.12)
    ACCEPTED as axis_109  cumulative_var=0.2803

  [ 105]  axes=110  step_var=0.0021  binary_acc=0.984  gap=0.2430  max_dot=0.0020  (1.8s)
    TOP:  ä¸įåĲĮçļĦ(0.12) | offspring(0.12) | NT(0.11) | ä½ľ(0.11) | delic(0.11) | mango(0.11) | é«ĺé¢ĳ(0.11) | SC(0.11)
    BOT:  pile(-0.12) | burdens(-0.11) | å®«æ®¿(-0.11) | disin(-0.11) | stad(-0.11) | warrior(-0.11) | ç©¿æĪ´(-0.11) | Curse(-0.11)
    ACCEPTED as axis_110  cumulative_var=0.2818

  [ 106]  axes=111  step_var=0.0022  binary_acc=0.984  gap=0.2458  max_dot=0.0019  (1.8s)
    TOP:  ä½ĵç§¯(0.13) | drove(0.13) | semif(0.12) | Dell(0.12) | cost(0.12) | .RequestMapping(0.11) | Ð±ÐµÑģÐ¿(0.11) | desk(0.11)
    BOT:  bub(-0.13) | jam(-0.12) | repercussions(-0.12) | decipher(-0.11) | çŃīåİŁåĽł(-0.11) | ç²¾å½©(-0.11) | sinus(-0.11) | lively(-0.11)
    ACCEPTED as axis_111  cumulative_var=0.2834

  [ 107]  axes=112  step_var=0.0021  binary_acc=0.983  gap=0.2444  max_dot=0.0008  (1.9s)
    TOP:  alcoholic(0.13) | æ±Łæ·®(0.13) | heavens(0.12) | movable(0.11) | Hugh(0.11) | çŁ¥åĲį(0.11) | gaming(0.10) | ihm(0.10)
    BOT:  extrapol(-0.12) | newer(-0.12) | yard(-0.12) | subpo(-0.11) | dollars(-0.11) | colonial(-0.11) | Davis(-0.11) | æī¹åĪ¤(-0.11)
    ACCEPTED as axis_112  cumulative_var=0.2849

  [ 108]  axes=113  step_var=0.0022  binary_acc=0.945  gap=0.2480  max_dot=0.0008  (1.8s)
    TOP:  Digest(0.14) | Minister(0.14) | hour(0.14) | Week(0.13) | Month(0.13) | Planner(0.12) | month(0.12) | Hist(0.12)
    BOT:  embodiments(-0.11) | empirical(-0.11) | è´Ŀå£³(-0.11) | leverage(-0.11) | emerging(-0.11) | EB(-0.11) | ROI(-0.11) | æĪ¿åľ°äº§(-0.11)
    ACCEPTED as axis_113  cumulative_var=0.2865

  [ 109]  axes=114  step_var=0.0022  binary_acc=0.996  gap=0.2422  max_dot=0.0010  (1.8s)
    TOP:  elevated(0.12) | irradi(0.11) | depletion(0.11) | hectares(0.11) | Holl(0.11) | Decl(0.10) | à¸Ķà¸³à¹Ģà¸Ļ(0.10) | Snapchat(0.10)
    BOT:  myth(-0.12) | physician(-0.12) | ghetto(-0.12) | Ð¿ÑĢÐ¸Ñĩ(-0.12) | Cyber(-0.12) | Myth(-0.11) | centroid(-0.11) | GH(-0.11)
    ACCEPTED as axis_114  cumulative_var=0.2881

  [ 110]  axes=115  step_var=0.0022  binary_acc=0.983  gap=0.2527  max_dot=0.0050  (1.9s)
    TOP:  Gang(0.12) | ownership(0.12) | Kentucky(0.12) | é¦ĸåħĪ(0.11) | owned(0.11) | eg(0.11) | beg(0.11) | Partnership(0.11)
    BOT:  mirac(-0.12) | ç«ĭéĿ¢(-0.12) | QMessageBox(-0.12) | conced(-0.11) | ucc(-0.11) | ROUND(-0.11) | Vatican(-0.11) | posts(-0.11)
    ACCEPTED as axis_115  cumulative_var=0.2897

  [ 111]  axes=116  step_var=0.0021  binary_acc=0.992  gap=0.2428  max_dot=0.0029  (1.9s)
    TOP:  åľ°æĿ¿(0.11) | jaw(0.11) | Westminster(0.11) | Lawson(0.11) | qualifying(0.11) | å¾ģä¿¡(0.11) | meter(0.11) | scoreboard(0.11)
    BOT:  stir(-0.12) | Burk(-0.11) | ä¸»åĬŀ(-0.11) | Hers(-0.11) | çº¢æĹĹ(-0.10) | Shir(-0.10) | åıĳéĢģ(-0.10) | Costa(-0.10)
    ACCEPTED as axis_116  cumulative_var=0.2912

  [ 112]  axes=117  step_var=0.0022  binary_acc=0.978  gap=0.2430  max_dot=0.0061  (1.8s)
    TOP:  è°Īè¯Ŀ(0.13) | ç´Ĭä¹±(0.12) | fermentation(0.12) | çĵ¯(0.11) | felt(0.11) | tonight(0.11) | channel(0.11) | considering(0.11)
    BOT:  inv(-0.13) | impressive(-0.13) | impressed(-0.12) | taxi(-0.11) | adverts(-0.11) | Detailed(-0.11) | nev(-0.11) | å½¢(-0.11)
    ACCEPTED as axis_117  cumulative_var=0.2927

  [ 113]  axes=118  step_var=0.0022  binary_acc=0.951  gap=0.2461  max_dot=0.0031  (1.8s)
    TOP:  warranties(0.14) | warranty(0.14) | unity(0.12) | WD(0.12) | guarantee(0.11) | autonomy(0.11) | announcement(0.11) | meny(0.11)
    BOT:  Singh(-0.12) | oversight(-0.12) | slag(-0.11) | cement(-0.11) | fossil(-0.11) | Idol(-0.11) | Oversight(-0.11) | basil(-0.11)
    ACCEPTED as axis_118  cumulative_var=0.2943

  [ 114]  axes=119  step_var=0.0021  binary_acc=0.983  gap=0.2373  max_dot=0.0034  (1.8s)
    TOP:  uchar(0.12) | æºĲæ³ī(0.11) | åįİäºº(0.11) | åĩºæīĭ(0.11) | transf(0.10) | å¤įèĭı(0.10) | mentality(0.10) | MagicMock(0.10)
    BOT:  rou(-0.14) | grou(-0.13) | agreements(-0.12) | éĴ¢ç»ĵæŀĦ(-0.12) | inmate(-0.12) | shadows(-0.11) | ä¿¡è®¿(-0.11) | Nex(-0.11)
    ACCEPTED as axis_119  cumulative_var=0.2957

  [ 115]  axes=120  step_var=0.0021  binary_acc=0.991  gap=0.2501  max_dot=0.0016  (1.8s)
    TOP:  lifestyle(0.14) | æ¸(0.13) | Turk(0.12) | çļĦçĶŁæ´»(0.12) | çĶŁæĢģæĸĩæĺİ(0.11) | çľŁçĲĨ(0.11) | enriched(0.11) | Tek(0.11)
    BOT:  åĬłçıŃ(-0.12) | Buffy(-0.12) | bÃłn(-0.12) | twenty(-0.11) | affen(-0.11) | offices(-0.11) | eight(-0.11) | bitter(-0.11)
    ACCEPTED as axis_120  cumulative_var=0.2972

  [ 116]  axes=121  step_var=0.0022  binary_acc=0.972  gap=0.2509  max_dot=0.0013  (1.8s)
    TOP:  Rabbit(0.13) | åĪĺéĤ¦(0.12) | rulers(0.12) | guilt(0.12) | Earl(0.11) | readline(0.11) | Maple(0.11) | åıįæĢĿ(0.11)
    BOT:  åįģä½³(-0.14) | trophy(-0.14) | æĸ°ä¸ļæĢģ(-0.13) | Trophy(-0.12) | Zucker(-0.12) | guy(-0.12) | åīįæīĢæľªæľī(-0.12) | æľªç»ı(-0.12)
    ACCEPTED as axis_121  cumulative_var=0.2988

  [ 117]  axes=122  step_var=0.0022  binary_acc=0.966  gap=0.2444  max_dot=0.0012  (1.9s)
    TOP:  compassion(0.13) | paste(0.13) | cauliflower(0.12) | perfor(0.11) | bus(0.11) | joining(0.11) | PU(0.11) | jal(0.11)
    BOT:  æŁ¥éĺħ(-0.13) | regimes(-0.12) | çĶŁæ®ĸ(-0.12) | freely(-0.11) | dominant(-0.11) | rÃ´le(-0.11) | Reg(-0.11) | åľ°ä½į(-0.11)
    ACCEPTED as axis_122  cumulative_var=0.3003

  [ 118]  axes=123  step_var=0.0021  binary_acc=0.997  gap=0.2511  max_dot=0.0039  (1.8s)
    TOP:  Mock(0.12) | contests(0.12) | contest(0.12) | blank(0.12) | Mock(0.11) | stricter(0.11) | Grant(0.11) | booths(0.10)
    BOT:  priorities(-0.13) | adrenaline(-0.13) | pain(-0.13) | phosphate(-0.12) | éĩįå¿ĥ(-0.12) | head(-0.12) | focuses(-0.12) | belly(-0.11)
    ACCEPTED as axis_123  cumulative_var=0.3018

  [ 119]  axes=124  step_var=0.0021  binary_acc=0.966  gap=0.2393  max_dot=0.0005  (1.8s)
    TOP:  handwriting(0.11) | impression(0.11) | reasonably(0.11) | ibly(0.11) | precisely(0.11) | overcrow(0.11) | åĩłä½ķ(0.11) | exhaust(0.11)
    BOT:  éģµå¾ª(-0.15) | societal(-0.13) | narciss(-0.11) | gigantic(-0.11) | roasted(-0.11) | æłĳç«ĭ(-0.11) | é£İæľº(-0.11) | strategic(-0.11)
    ACCEPTED as axis_124  cumulative_var=0.3033

  [ 120]  axes=125  step_var=0.0021  binary_acc=0.994  gap=0.2431  max_dot=0.0016  (1.8s)
    TOP:  countryside(0.13) | ÐĿÐ¸Ðº(0.13) | naturally(0.12) | magnet(0.12) | Alto(0.12) | counselor(0.11) | cer(0.11) | unp(0.11)
    BOT:  reproductive(-0.12) | groin(-0.11) | deliveries(-0.11) | ë¦°(-0.11) | temperament(-0.11) | ern(-0.11) | åĪĨå¨©(-0.11) | ific(-0.11)
    ACCEPTED as axis_125  cumulative_var=0.3048

  [ 121]  axes=126  step_var=0.0022  binary_acc=0.988  gap=0.2500  max_dot=0.0073  (1.9s)
    TOP:  went(0.14) | ballots(0.12) | prevention(0.12) | bridal(0.12) | 9(0.11) | gastric(0.11) | arrives(0.11) | æĬķç¥¨(0.11)
    BOT:  frustration(-0.14) | åĸĩ(-0.12) | rebuild(-0.12) | intellectual(-0.12) | Derek(-0.11) | é¦¥(-0.11) | frustrating(-0.11) | ãģŁãģı(-0.11)
    ACCEPTED as axis_126  cumulative_var=0.3063

  [ 122]  axes=127  step_var=0.0022  binary_acc=0.981  gap=0.2390  max_dot=0.0018  (1.8s)
    TOP:  Clifford(0.12) | luxury(0.12) | æ¸¯æ¾³(0.11) | æ·¹(0.11) | leftist(0.11) | (tmp(0.11) | Carlson(0.11) | tmp(0.11)
    BOT:  advert(-0.14) | åĽŀå¤į(-0.13) | åĲĪéĢĤ(-0.11) | çĽĳå¯Ł(-0.11) | desk(-0.11) | åĵĦ(-0.11) | fitting(-0.11) | offense(-0.11)
    ACCEPTED as axis_127  cumulative_var=0.3078

  [ 123]  axes=128  step_var=0.0021  binary_acc=0.978  gap=0.2319  max_dot=0.0025  (1.8s)
    TOP:  End(0.13) | ending(0.12) | scramble(0.12) | El(0.12) | ä½ĵåŀĭ(0.11) | incl(0.11) | appealing(0.11) | æľĪåºķ(0.11)
    BOT:  diagrams(-0.12) | discovery(-0.12) | discussed(-0.11) | äºĭæķħ(-0.11) | simult(-0.11) | æ¸ħåĩī(-0.11) | noticed(-0.11) | å¤©çĮ«(-0.10)
    ACCEPTED as axis_128  cumulative_var=0.3092

  [ 124]  axes=129  step_var=0.0022  binary_acc=0.992  gap=0.2373  max_dot=0.0059  (1.8s)
    TOP:  èµ·ä¼ı(0.12) | åīįä¸ĸ(0.12) | imi(0.12) | asks(0.12) | é³ĥ(0.11) | arises(0.11) | ÑģÑĤÐ°ÑĤÑĮ(0.11) | escort(0.11)
    BOT:  Ð¿ÑĢÐ¾Ð³(-0.12) | gadgets(-0.12) | hipp(-0.12) | éĴĪ(-0.11) | ADC(-0.11) | å·¥èīºåĵģ(-0.11) | nap(-0.11) | åıĤåĬłä¼ļè®®(-0.11)
    ACCEPTED as axis_129  cumulative_var=0.3108

  [ 125]  axes=130  step_var=0.0021  binary_acc=0.939  gap=0.2372  max_dot=0.0013  (1.8s)
    TOP:  æĬķäº§(0.11) | Hispanic(0.11) | tobacco(0.11) | è¶ĬæĿ¥è¶Ĭ(0.11) | abet(0.10) | è¦ĨçĽĸ(0.10) | sleep(0.10) | Zimmer(0.10)
    BOT:  photoc(-0.12) | Ferm(-0.12) | MCU(-0.12) | Quick(-0.12) | quadr(-0.12) | Fountain(-0.11) | à¸Ħà¸§(-0.11) | kon(-0.11)
    ACCEPTED as axis_130  cumulative_var=0.3122

  [ 126]  axes=131  step_var=0.0021  binary_acc=0.987  gap=0.2404  max_dot=0.0011  (1.8s)
    TOP:  æĹłäººæľº(0.14) | ä¾ĽåºĶéĵ¾(0.13) | dain(0.13) | Delta(0.12) | Dun(0.12) | deserved(0.12) | uÃŃ(0.11) | augmented(0.11)
    BOT:  veggies(-0.12) | tablespoons(-0.12) | bahwa(-0.11) | æİīèĲ½(-0.11) | beautiful(-0.11) | è¿ŀè½½(-0.11) | Brady(-0.11) | èİī(-0.11)
    ACCEPTED as axis_131  cumulative_var=0.3136

  [ 127]  axes=132  step_var=0.0022  binary_acc=0.979  gap=0.2416  max_dot=0.0021  (1.9s)
    TOP:  gravitational(0.13) | smallest(0.12) | hardest(0.12) | ç¤¾ä¼ļèµĦæľ¬(0.11) | å«¡(0.11) | éĢĹ(0.11) | Compared(0.11) | unnoticed(0.11)
    BOT:  é£İéĻ©(-0.13) | ascend(-0.12) | transcend(-0.12) | Koh(-0.12) | ç¿¡ç¿ł(-0.12) | flood(-0.11) | æĿı(-0.11) | completed(-0.11)
    ACCEPTED as axis_132  cumulative_var=0.3151

  [ 128]  axes=133  step_var=0.0021  binary_acc=0.988  gap=0.2367  max_dot=0.0034  (1.8s)
    TOP:  è¯¢éĹ®(0.12) | ovo(0.12) | è¢ĸ(0.11) | significantly(0.11) | éĿłè¿ĳ(0.11) | ç¼ĵæħ¢(0.11) | decreased(0.11) | æİ§(0.11)
    BOT:  diversion(-0.12) | catch(-0.11) | ä»¤äºº(-0.11) | immersive(-0.11) | Ð´ÐµÑĢ(-0.11) | divert(-0.11) | æ²īæµ¸(-0.11) | holidays(-0.10)
    ACCEPTED as axis_133  cumulative_var=0.3166

  [ 129]  axes=134  step_var=0.0021  binary_acc=0.980  gap=0.2331  max_dot=0.0047  (1.8s)
    TOP:  æĪĲæŀľè½¬åĮĸ(0.12) | ç§ĳçłĶ(0.12) | ä¸Ńå¤ĸ(0.12) | Could(0.11) | å°ıå°ı(0.11) | utch(0.11) | otros(0.11) | outreach(0.11)
    BOT:  ä¸Ĭåĳ¨(-0.12) | çĶĦ(-0.11) | Defendant(-0.11) | dates(-0.11) | hurdle(-0.11) | JR(-0.11) | Stage(-0.11) | LEG(-0.11)
    ACCEPTED as axis_134  cumulative_var=0.3180

  [ 130]  axes=135  step_var=0.0022  binary_acc=0.996  gap=0.2446  max_dot=0.0010  (1.8s)
    TOP:  transcripts(0.12) | boxing(0.11) | äºĭå®ŀ(0.11) | åĲĪæ³ķ(0.11) | lawful(0.11) | çļĦäºĭå®ŀ(0.11) | cellul(0.11) | çļĦå®ŀåĬĽ(0.10)
    BOT:  anxiety(-0.15) | useMemo(-0.13) | æĢ¥è¯Ĭ(-0.12) | çĦ¦èĻĳ(-0.12) | å¿ĥèĦı(-0.12) | Anxiety(-0.12) | emergency(-0.12) | Sustainability(-0.11)
    ACCEPTED as axis_135  cumulative_var=0.3195

  [ 131]  axes=136  step_var=0.0022  binary_acc=0.981  gap=0.2404  max_dot=0.0052  (1.9s)
    TOP:  royalty(0.12) | ä¾Ľæ°´(0.12) | hungry(0.12) | åŃľ(0.12) | styl(0.11) | minutes(0.11) | frozen(0.11) | èĬľæ¹ĸ(0.11)
    BOT:  Obama(-0.14) | paperback(-0.12) | verb(-0.11) | abal(-0.11) | åıĳçĹħ(-0.11) | æ§Ł(-0.11) | ercicio(-0.11) | Auburn(-0.11)
    ACCEPTED as axis_136  cumulative_var=0.3210

  [ 132]  axes=137  step_var=0.0021  binary_acc=0.988  gap=0.2328  max_dot=0.0018  (1.8s)
    TOP:  nar(0.11) | æĪ¿åľ°äº§(0.11) | ç»½(0.11) | Parkinson(0.11) | neuronal(0.11) | adaptations(0.11) | inson(0.11) | praise(0.11)
    BOT:  ä¸ĬåįĪ(-0.13) | èĶ(-0.11) | urger(-0.11) | lizard(-0.11) | uphe(-0.10) | èĻ¾(-0.10) | herb(-0.10) | Hero(-0.10)
    ACCEPTED as axis_137  cumulative_var=0.3224

  [ 133]  axes=138  step_var=0.0021  binary_acc=0.997  gap=0.2338  max_dot=0.0021  (1.8s)
    TOP:  refugees(0.13) | ç¼´è´¹(0.12) | åı£è¯Ń(0.12) | äººåĬĽ(0.11) | è¿Ľä¿®(0.11) | teenager(0.11) | tráº»(0.11) | STEP(0.10)
    BOT:  IMO(-0.12) | tomato(-0.12) | å®ŀéĻħæİ§åĪ¶(-0.12) | ä¸Ģå¦ĤæĹ¢å¾Ģ(-0.11) | market(-0.11) | åı¯æĮģç»Ń(-0.11) | ÑįÐºÐ¾Ð½Ð¾Ð¼(-0.10) | diag(-0.10)
    ACCEPTED as axis_138  cumulative_var=0.3238

  [ 134]  axes=139  step_var=0.0021  binary_acc=0.997  gap=0.2363  max_dot=0.0095  (2.0s)
    TOP:  happened(0.12) | claimed(0.12) | å¤ºåĨł(0.12) | åĪ¤(0.12) | ivia(0.11) | ze(0.11) | çļĦä¼ĺåĬ¿(0.11) | Olivia(0.10)
    BOT:  éĹªåħī(-0.13) | Gil(-0.12) | unto(-0.12) | å°Ĭæķ¬(-0.12) | æĶ¾å°Ħ(-0.11) | Tele(-0.11) | Quentin(-0.11) | salon(-0.10)
    ACCEPTED as axis_139  cumulative_var=0.3252

  [ 135]  axes=140  step_var=0.0021  binary_acc=0.947  gap=0.2314  max_dot=0.0020  (1.9s)
    TOP:  WhatsApp(0.12) | éŃĤ(0.11) | possession(0.11) | melting(0.11) | predominant(0.11) | ruins(0.11) | absorbed(0.11) | lament(0.11)
    BOT:  åħ¬æŃ£(-0.11) | çĶŁæ®ĸ(-0.11) | æģº(-0.11) | çĮ®è¡Ģ(-0.11) | publicly(-0.11) | ãĥĮ(-0.11) | å©ķ(-0.11) | FX(-0.10)
    ACCEPTED as axis_140  cumulative_var=0.3266

  [ 136]  axes=141  step_var=0.0021  binary_acc=0.998  gap=0.2348  max_dot=0.0027  (1.8s)
    TOP:  oub(0.12) | equipped(0.12) | Ð°Ð½Ð°Ð»Ð¸Ð·(0.11) | èģĺ(0.11) | links(0.11) | åŃ©(0.11) | å¨ĥ(0.11) | Sylv(0.11)
    BOT:  circum(-0.12) | CAM(-0.12) | ESC(-0.11) | remainder(-0.11) | åı¯è¡Į(-0.11) | Rem(-0.11) | Smash(-0.11) | STEM(-0.11)
    ACCEPTED as axis_141  cumulative_var=0.3280

  [ 137]  axes=142  step_var=0.0022  binary_acc=0.998  gap=0.2352  max_dot=0.0015  (1.8s)
    TOP:  éĩįæ¸©(0.12) | themed(0.11) | relacion(0.11) | åĮħåĽ´(0.11) | loves(0.11) | decorations(0.11) | bodily(0.11) | treats(0.11)
    BOT:  çĽ¸æľº(-0.12) | è®¸åı¯(-0.12) | åĪĬ(-0.11) | Bre(-0.11) | ÐºÐ°Ð·Ð¸Ð½Ð¾(-0.11) | shot(-0.11) | Symposium(-0.11) | åĲĪä½ľç¤¾(-0.11)
    ACCEPTED as axis_142  cumulative_var=0.3295

  [ 138]  axes=143  step_var=0.0021  binary_acc=0.982  gap=0.2310  max_dot=0.0024  (1.8s)
    TOP:  properly(0.14) | Oktober(0.12) | ç»ıåİĨè¿ĩ(0.12) | ÐŁÑĢÐ¾(0.12) | patients(0.11) | é«ĺåİŁ(0.11) | upgrading(0.11) | åı¯éĿłæĢ§(0.11)
    BOT:  æĢĿç»´(-0.14) | çī¹éķ¿(-0.13) | cstdlib(-0.12) | steering(-0.12) | éĿŀè¦ģ(-0.12) | motivated(-0.11) | scissors(-0.11) | @Autowired(-0.11)
    ACCEPTED as axis_143  cumulative_var=0.3309

  [ 139]  axes=144  step_var=0.0021  binary_acc=0.964  gap=0.2312  max_dot=0.0054  (1.9s)
    TOP:  office(0.12) | department(0.12) | sorte(0.12) | withdraw(0.11) | apologies(0.11) | æ°ĳåĽ½(0.11) | Dept(0.11) | geomet(0.11)
    BOT:  åħ¨åĽ½äººæ°ĳ(-0.12) | troub(-0.12) | phosph(-0.11) | prosperity(-0.11) | chronic(-0.11) | åĸĦäºİ(-0.11) | permalink(-0.11) | Solomon(-0.11)
    ACCEPTED as axis_144  cumulative_var=0.3323

  [ 140]  axes=145  step_var=0.0021  binary_acc=0.992  gap=0.2322  max_dot=0.0011  (1.8s)
    TOP:  Ottawa(0.11) | Bee(0.11) | _xt(0.11) | oe(0.11) | extreme(0.10) | ä¸ĵä¸ļ(0.10) | pets(0.10) | everybody(0.10)
    BOT:  disrupt(-0.12) | invented(-0.12) | lesions(-0.11) | Brig(-0.10) | çģ«è½¦ç«Ļ(-0.10) | ê²½ìļ°(-0.10) | complaint(-0.10) | Complaint(-0.10)
    ACCEPTED as axis_145  cumulative_var=0.3336

  [ 141]  axes=146  step_var=0.0021  binary_acc=0.993  gap=0.2336  max_dot=0.0047  (1.9s)
    TOP:  compelled(0.12) | Resort(0.12) | supposed(0.12) | defaultdict(0.12) | åŁİç®¡(0.11) | vigor(0.11) | blink(0.11) | rightful(0.11)
    BOT:  å¯ĮåĲ«(-0.12) | low(-0.11) | ä¸¥(-0.11) | Brent(-0.11) | unseen(-0.11) | payout(-0.11) | readme(-0.11) | Äĳáº¡t(-0.11)
    ACCEPTED as axis_146  cumulative_var=0.3351

  [ 142]  axes=147  step_var=0.0021  binary_acc=0.978  gap=0.2345  max_dot=0.0047  (1.9s)
    TOP:  Depart(0.11) | Rather(0.11) | shipment(0.11) | parseFloat(0.11) | ä¸ĢåĲĮ(0.10) | farther(0.10) | dismay(0.10) | loot(0.10)
    BOT:  Tory(-0.13) | QC(-0.12) | Grammar(-0.12) | çĹħæ¯Ĵ(-0.11) | åįĪåĲİ(-0.11) | å¤įå·¥(-0.11) | penalty(-0.11) | QB(-0.11)
    ACCEPTED as axis_147  cumulative_var=0.3364

  [ 143]  axes=148  step_var=0.0021  binary_acc=0.997  gap=0.2386  max_dot=0.0042  (1.8s)
    TOP:  recalls(0.12) | Cherry(0.11) | probably(0.11) | æŁ´(0.11) | ä»¿(0.11) | åıĳéħµ(0.11) | ä»¿çľŁ(0.11) | Za(0.11)
    BOT:  MV(-0.13) | -bedroom(-0.13) | dismiss(-0.12) | breach(-0.12) | inth(-0.12) | berth(-0.11) | beneath(-0.11) | Blvd(-0.11)
    ACCEPTED as axis_148  cumulative_var=0.3379

  [ 144]  axes=149  step_var=0.0021  binary_acc=0.991  gap=0.2356  max_dot=0.0048  (1.8s)
    TOP:  Thurs(0.11) | .classList(0.11) | sculpture(0.11) | upstairs(0.11) | phosphory(0.11) | stall(0.10) | technological(0.10) | thus(0.10)
    BOT:  visas(-0.12) | commencement(-0.12) | Rifle(-0.11) | streams(-0.11) | AA(-0.11) | forma(-0.11) | neurop(-0.11) | iao(-0.11)
    ACCEPTED as axis_149  cumulative_var=0.3392

  [ 145]  axes=150  step_var=0.0021  binary_acc=0.972  gap=0.2326  max_dot=0.0032  (1.9s)
    TOP:  æµĩæ°´(0.13) | dependable(0.12) | Darren(0.11) | é¸¦(0.11) | hiring(0.11) | grown(0.11) | misplaced(0.11) | gather(0.11)
    BOT:  ä¸įåĩº(-0.11) | langu(-0.11) | anka(-0.11) | fot(-0.10) | lobbyists(-0.10) | å¨±ä¹Ĳ(-0.10) | åĦĴ(-0.10) | acab(-0.10)
    ACCEPTED as axis_150  cumulative_var=0.3406

  [ 146]  axes=151  step_var=0.0020  binary_acc=0.962  gap=0.2251  max_dot=0.0053  (1.8s)
    TOP:  åĿ³(0.12) | doi(0.11) | çħ¤æ°Ķ(0.11) | æī®æ¼Ķ(0.10) | æ¯Ľä¸»å¸Ń(0.10) | Mt(0.10) | Mt(0.10) | Rao(0.10)
    BOT:  ìŀ¬(-0.12) | ymph(-0.12) | spectacle(-0.11) | wed(-0.11) | æŃĻ(-0.11) | ä¸įæĥ³(-0.11) | ä½İè°ĥ(-0.10) | complet(-0.10)
    ACCEPTED as axis_151  cumulative_var=0.3419

  [ 147]  axes=152  step_var=0.0021  binary_acc=0.992  gap=0.2300  max_dot=0.0042  (1.8s)
    TOP:  crush(0.12) | çª¦(0.11) | Mia(0.11) | vegetarian(0.11) | æľīéĻĲ(0.11) | intern(0.11) | çķªèĮĦ(0.10) | inmates(0.10)
    BOT:  æĳĨèĦ±(-0.13) | herald(-0.12) | dangerous(-0.12) | Santo(-0.12) | pton(-0.11) | lifestyle(-0.11) | QApplication(-0.11) | exterior(-0.11)
    ACCEPTED as axis_152  cumulative_var=0.3433

  [ 148]  axes=153  step_var=0.0020  binary_acc=0.982  gap=0.2252  max_dot=0.0019  (1.9s)
    TOP:  fence(0.12) | é¡¹çĽ®(0.11) | garbage(0.11) | èĦĤèĤª(0.10) | éłħçĽ®(0.10) | location(0.10) | trash(0.10) | stash(0.10)
    BOT:  ä½ĵä¼ļ(-0.13) | grat(-0.12) | æĭīåĬ¨(-0.12) | æĦıæĦ¿(-0.12) | Drive(-0.11) | ä¸Ŀç»¸ä¹ĭè·¯(-0.11) | åĪ¶çº¦(-0.11) | cheaper(-0.11)
    ACCEPTED as axis_153  cumulative_var=0.3446

  [ 149]  axes=154  step_var=0.0021  binary_acc=0.996  gap=0.2263  max_dot=0.0028  (1.8s)
    TOP:  birthday(0.13) | spiral(0.11) | æĹ¥åŃĲ(0.11) | lic(0.11) | weekend(0.11) | Birthday(0.11) | èµĦåĬ©(0.10) | decided(0.10)
    BOT:  æ¾Ħ(-0.12) | omitted(-0.11) | dread(-0.11) | ing(-0.11) | rown(-0.11) | ç»ıéªĮä¸°å¯Į(-0.11) | EQ(-0.10) | overt(-0.10)
    ACCEPTED as axis_154  cumulative_var=0.3460

  [ 150]  axes=155  step_var=0.0019  binary_acc=0.979  gap=0.2178  max_dot=0.0057  (1.9s)
    TOP:  æĬĸ(0.11) | predominant(0.11) | VB(0.10) | vocals(0.10) | vivid(0.10) | dynamics(0.10) | uet(0.10) | å¼ºç¡¬(0.10)
    BOT:  camp(-0.11) | åĿĳ(-0.11) | ç£¨æįŁ(-0.11) | epic(-0.10) | .setOnClickListener(-0.10) | âĤ¬(-0.10) | æĬķèµĦèĢħ(-0.10) | éĩĮç¨ĭ(-0.10)
    ACCEPTED as axis_155  cumulative_var=0.3473

  [ 151]  axes=156  step_var=0.0022  binary_acc=0.989  gap=0.2358  max_dot=0.0006  (1.9s)
    TOP:  jewels(0.12) | è§ģè¯ģ(0.11) | çĸĻ(0.11) | creatures(0.11) | miracle(0.11) | åĨ·æ¼ł(0.11) | çī(0.10) | crow(0.10)
    BOT:  ä¸ĭåİ»(-0.12) | gens(-0.11) | åįĩ(-0.11) | ensuite(-0.11) | atmosphere(-0.11) | sentencing(-0.11) | æģ©æĸ½(-0.11) | åĺĺ(-0.11)
    ACCEPTED as axis_156  cumulative_var=0.3487

  [ 152]  axes=157  step_var=0.0021  binary_acc=0.975  gap=0.2301  max_dot=0.0027  (1.9s)
    TOP:  toll(0.14) | wishes(0.12) | å¯¹ä¼ģä¸ļ(0.11) | would(0.11) | sure(0.11) | é£İæĥħ(0.11) | äº§åĢ¼(0.11) | é¥®æ°´(0.11)
    BOT:  snag(-0.11) | åģľä¸ĭ(-0.11) | dav(-0.11) | naked(-0.11) | Ave(-0.11) | Gaines(-0.11) | NST(-0.10) | bout(-0.10)
    ACCEPTED as axis_157  cumulative_var=0.3501

  [ 153]  axes=158  step_var=0.0020  binary_acc=0.993  gap=0.2272  max_dot=0.0079  (1.9s)
    TOP:  Davis(0.11) | Davis(0.11) | çĩĥçĥ§(0.11) | æĽ´å¿«(0.11) | auer(0.11) | atomic(0.11) | èİ·æī¹(0.11) | Regardless(0.10)
    BOT:  mary(-0.11) | headlines(-0.11) | ç¦¾(-0.11) | shelf(-0.11) | ä¿¡ç®±(-0.11) | åıĳè¨Ģäºº(-0.11) | -front(-0.10) | å¡¾(-0.10)
    ACCEPTED as axis_158  cumulative_var=0.3514

  [ 154]  axes=159  step_var=0.0020  binary_acc=0.974  gap=0.2238  max_dot=0.0046  (1.8s)
    TOP:  æ¯ħçĦ¶(0.12) | nib(0.12) | çĶ³(0.11) | benchmark(0.11) | .bin(0.11) | obedience(0.10) | çĶŁçĮª(0.10) | declining(0.10)
    BOT:  æľĢéĩįè¦ģçļĦ(-0.13) | aways(-0.12) | èį¯(-0.12) | light(-0.11) | rewind(-0.11) | AutoMapper(-0.11) | Re(-0.11) | WD(-0.10)
    ACCEPTED as axis_159  cumulative_var=0.3527

  [ 155]  axes=160  step_var=0.0021  binary_acc=0.978  gap=0.2317  max_dot=0.0005  (1.8s)
    TOP:  Omni(0.12) | å¤ĸæ±ĩ(0.11) | ä¸Ĭåįĥ(0.11) | ãĥ¬(0.11) | pok(0.11) | brill(0.11) | å¤ĸéĥ¨(0.10) | Kre(0.10)
    BOT:  loyal(-0.14) | çĶŁäº§(-0.12) | conducive(-0.11) | è¡·å¿ĥ(-0.11) | dates(-0.11) | boo(-0.11) | sexual(-0.11) | quest(-0.11)
    ACCEPTED as axis_160  cumulative_var=0.3541

  [ 156]  axes=161  step_var=0.0021  binary_acc=0.969  gap=0.2245  max_dot=0.0013  (1.9s)
    TOP:  Sag(0.11) | åĲĦé¡¹(0.10) | åĬłå¤§(0.10) | cerv(0.10) | sag(0.10) | éĤĿ(0.10) | çĶ±äºİ(0.10) | çıłä¸īè§Ĵ(0.10)
    BOT:  mois(-0.11) | student(-0.11) | Mori(-0.11) | æľªæĪĲå¹´(-0.11) | witness(-0.11) | åħ¥åŃ¦(-0.11) | blas(-0.10) | lens(-0.10)
    ACCEPTED as axis_161  cumulative_var=0.3554

  [ 157]  axes=162  step_var=0.0021  binary_acc=0.972  gap=0.2274  max_dot=0.0046  (1.8s)
    TOP:  é¦ĸéĢī(0.12) | obst(0.12) | tears(0.12) | æ£ł(0.11) | lifes(0.11) | dream(0.11) | souls(0.11) | dreams(0.11)
    BOT:  passage(-0.11) | supra(-0.11) | empirical(-0.11) | æ¸¸çİ©(-0.11) | Vermont(-0.11) | warming(-0.11) | æľ«(-0.11) | antagon(-0.10)
    ACCEPTED as axis_162  cumulative_var=0.3567

  [ 158]  axes=163  step_var=0.0020  binary_acc=0.980  gap=0.2230  max_dot=0.0029  (1.8s)
    TOP:  æĢĶ(0.11) | à¹Ģà¸«à¸¥(0.11) | raith(0.10) | indemn(0.10) | setContentView(0.10) | UITableViewCell(0.10) | hyperlink(0.10) | differing(0.10)
    BOT:  hygiene(-0.12) | é«ĺçº§(-0.11) | ä¿®é¥°(-0.11) | mothers(-0.11) | Air(-0.10) | mish(-0.10) | æ¡¶(-0.10) | heating(-0.10)
    ACCEPTED as axis_163  cumulative_var=0.3580

  [ 159]  axes=164  step_var=0.0020  binary_acc=0.953  gap=0.2310  max_dot=0.0008  (1.8s)
    TOP:  Rs(0.12) | Taj(0.11) | Dahl(0.11) | Ð¾Ð±ÑıÐ·Ð°Ð½(0.10) | diary(0.10) | Reflex(0.10) | ä¼łè¾¾(0.10) | æĹ©æĻ¨(0.10)
    BOT:  humility(-0.12) | à¸£à¸§à¸¡(-0.12) | avoided(-0.11) | hard(-0.11) | Again(-0.11) | å®Ĺ(-0.11) | çľĭéĩį(-0.11) | hamstring(-0.11)
    ACCEPTED as axis_164  cumulative_var=0.3593

  [ 160]  axes=165  step_var=0.0020  binary_acc=0.997  gap=0.2203  max_dot=0.0040  (1.9s)
    TOP:  çļĩåĲİ(0.12) | çĸ«æĥħå½±åĵį(0.11) | Impact(0.10) | Crow(0.10) | disciples(0.10) | Crow(0.10) | crÃ©(0.10) | ä¸Ńæĸĩ(0.10)
    BOT:  èµ·çłģ(-0.11) | ulp(-0.11) | continually(-0.11) | ç»Ĩå¾®(-0.10) | bizarre(-0.10) | é²ľæĺİ(-0.10) | dep(-0.10) | Barbara(-0.10)
    ACCEPTED as axis_165  cumulative_var=0.3606

  [ 161]  axes=166  step_var=0.0020  binary_acc=1.000  gap=0.2206  max_dot=0.0021  (1.8s)
    TOP:  applicant(0.13) | èĸ¯(0.11) | ç¾İåĳ³(0.11) | dough(0.10) | candidates(0.10) | ocyte(0.10) | cbd(0.10) | convinc(0.10)
    BOT:  accents(-0.12) | ä¸ĢåĲĮ(-0.11) | converse(-0.10) | å¯¼èĪª(-0.10) | iegel(-0.10) | åŁºè°ĥ(-0.10) | reiterated(-0.10) | ds(-0.10)
    ACCEPTED as axis_166  cumulative_var=0.3619

  [ 162]  axes=167  step_var=0.0020  binary_acc=0.994  gap=0.2274  max_dot=0.0095  (1.8s)
    TOP:  beat(0.12) | Iv(0.12) | reve(0.10) | åħĭæĢĿä¸»ä¹ī(0.10) | evaluated(0.10) | espect(0.10) | æĸ°ç¯ĩç«ł(0.10) | Levin(0.10)
    BOT:  ä¾ĭåŃĲ(-0.12) | Homer(-0.11) | æĽ¹æĵį(-0.11) | EXAMPLE(-0.11) | çĴĭ(-0.11) | æĨİ(-0.11) | é¥®çĶ¨(-0.11) | åģļäºĭ(-0.11)
    ACCEPTED as axis_167  cumulative_var=0.3632

  [ 163]  axes=168  step_var=0.0021  binary_acc=0.998  gap=0.2246  max_dot=0.0056  (1.8s)
    TOP:  pub(0.12) | èĥĮæĻ¯ä¸ĭ(0.11) | covered(0.11) | detainees(0.10) | è©³ç´°(0.10) | intros(0.10) | åĮħåĲ«äºĨ(0.10) | thorough(0.10)
    BOT:  .ff(-0.11) | smell(-0.11) | Orders(-0.11) | ê°Ģìŀ¥(-0.10) | orders(-0.10) | ä»»ä½ķå½¢å¼ı(-0.10) | OrderedDict(-0.10) | unpl(-0.10)
    ACCEPTED as axis_168  cumulative_var=0.3645

  [ 164]  axes=169  step_var=0.0021  binary_acc=0.998  gap=0.2266  max_dot=0.0038  (1.8s)
    TOP:  synthesis(0.11) | æĭ¨æīĵ(0.10) | enhancement(0.10) | cooperation(0.10) | Lil(0.10) | comeback(0.10) | yell(0.10) | WWW(0.10)
    BOT:  steady(-0.11) | ä¸ªæľĪ(-0.11) | Spotify(-0.11) | èĮĹ(-0.11) | Bailey(-0.11) | PI(-0.11) | Ð·Ð°Ð±Ð¾Ð»ÐµÐ²Ð°Ð½Ð¸Ñı(-0.10) | discretionary(-0.10)
    ACCEPTED as axis_169  cumulative_var=0.3658

  [ 165]  axes=170  step_var=0.0020  binary_acc=0.994  gap=0.2216  max_dot=0.0027  (1.8s)
    TOP:  ç¼ĩ(0.11) | thrown(0.11) | çķħéĶĢ(0.11) | è¶Ĭåıĳ(0.10) | exceptions(0.10) | è¯Ŀåī§(0.10) | æ¬ºè¯Ī(0.10) | infiltration(0.10)
    BOT:  åĬĽåº¦(-0.12) | tens(-0.11) | hydrogen(-0.11) | identical(-0.11) | ä¸Ļ(-0.11) | ambitious(-0.11) | Wake(-0.10) | Ark(-0.10)
    ACCEPTED as axis_170  cumulative_var=0.3671

  [ 166]  axes=171  step_var=0.0021  binary_acc=0.994  gap=0.2268  max_dot=0.0027  (1.9s)
    TOP:  ä½łæĢİä¹Ī(0.13) | ä¸ĭåįĬå¹´(0.12) | åĬ¿åĬĽ(0.11) | orda(0.11) | Direct(0.11) | lox(0.11) | çł¸(0.11) | åķĨåŁİ(0.11)
    BOT:  slip(-0.13) | kept(-0.12) | quoi(-0.12) | Whitney(-0.11) | adequate(-0.11) | Highway(-0.10) | åĲ¯è¿ª(-0.10) | å¹²è´§(-0.10)
    ACCEPTED as axis_171  cumulative_var=0.3684

  [ 167]  axes=172  step_var=0.0020  binary_acc=0.996  gap=0.2165  max_dot=0.0046  (1.8s)
    TOP:  LO(0.11) | ä¸įéĻĲ(0.11) | exp(0.11) | Links(0.11) | éĹ®éģĵ(0.11) | lined(0.10) | ASS(0.10) | éĻĲ(0.10)
    BOT:  ÙĥÙĦ(-0.12) | Siber(-0.11) | Git(-0.11) | resembling(-0.11) | rebut(-0.10) | ëĭ¤ë¥¸(-0.10) | ographer(-0.10) | èµ¡(-0.10)
    ACCEPTED as axis_172  cumulative_var=0.3697

  [ 168]  axes=173  step_var=0.0021  binary_acc=0.999  gap=0.2258  max_dot=0.0044  (1.8s)
    TOP:  Adoption(0.11) | nuclear(0.11) | Att(0.10) | gre(0.10) | ÑģÐµÑĤ(0.10) | tolerance(0.10) | brick(0.10) | æī©(0.10)
    BOT:  å©§(-0.12) | reveals(-0.11) | hid(-0.11) | secrets(-0.11) | common(-0.11) | ÑĥÐ·Ð½Ð°ÑĤÑĮ(-0.10) | Learn(-0.10) | æ±ŁéĹ¨(-0.10)
    ACCEPTED as axis_173  cumulative_var=0.3710

  [ 169]  axes=174  step_var=0.0021  binary_acc=0.989  gap=0.2266  max_dot=0.0015  (1.9s)
    TOP:  æĬĵèİ·(0.13) | duties(0.12) | cáº§n(0.12) | determin(0.12) | crate(0.12) | æĪĲæŀľ(0.11) | çľĭæľĽ(0.11) | æł·æľ¬(0.11)
    BOT:  vacations(-0.12) | Olympics(-0.11) | ëĤ´(-0.11) | Pradesh(-0.11) | verbal(-0.10) | âĢĶâĢĶâĢĶâĢĶ(-0.10) | åĬłçĽŁåºĹ(-0.10) | Ð°Ð·Ð²Ð°Ð½Ð¸Ðµ(-0.10)
    ACCEPTED as axis_174  cumulative_var=0.3723

  [ 170]  axes=175  step_var=0.0020  binary_acc=0.983  gap=0.2274  max_dot=0.0063  (1.8s)
    TOP:  turb(0.11) | é¢Ĩåıĸ(0.11) | lifetime(0.10) | Lambert(0.10) | anx(0.10) | abide(0.10) | ä¼ļç»Ļ(0.10) | æµĭç®Ĺ(0.10)
    BOT:  singles(-0.11) | Kai(-0.11) | cuisine(-0.10) | éĤ®æĶ¿(-0.10) | viagra(-0.10) | Ig(-0.10) | Caj(-0.10) | Veg(-0.10)
    ACCEPTED as axis_175  cumulative_var=0.3736

  [ 171]  axes=176  step_var=0.0019  binary_acc=0.985  gap=0.2190  max_dot=0.0023  (1.8s)
    TOP:  Racing(0.12) | racing(0.10) | .sqrt(0.10) | dashed(0.10) | å¤§åŃ¦çĶŁ(0.10) | Beef(0.10) | å°ıç¼ĸ(0.10) | Exam(0.10)
    BOT:  managed(-0.13) | åŁºåĩĨ(-0.10) | .waitFor(-0.10) | å®£åĳĬ(-0.10) | cualquier(-0.10) | åħ³éĹ¨(-0.10) | âĬĻ(-0.10) | Trust(-0.10)
    ACCEPTED as axis_176  cumulative_var=0.3748

  [ 172]  axes=177  step_var=0.0020  binary_acc=0.995  gap=0.2214  max_dot=0.0012  (1.8s)
    TOP:  åĽŀæĬ¥(0.11) | rats(0.10) | burial(0.10) | éºĭ(0.10) | éĿ¢éĥ¨(0.10) | decoder(0.10) | æĹ¥æĻļ(0.10) | é¢ľèī²(0.10)
    BOT:  ="/(-0.11) | èĪªçº¿(-0.11) | çĻ»éĻĨ(-0.11) | overseas(-0.10) | presumably(-0.10) | ãĢľ(-0.10) | Myanmar(-0.10) | ä¸įçĶ¨æĭħå¿ĥ(-0.10)
    ACCEPTED as axis_177  cumulative_var=0.3760

  [ 173]  axes=178  step_var=0.0020  binary_acc=0.997  gap=0.2193  max_dot=0.0018  (1.9s)
    TOP:  Clemson(0.11) | McLaren(0.10) | Population(0.10) | episode(0.10) | Population(0.10) | actionable(0.10) | widow(0.10) | åºŃå®¡(0.10)
    BOT:  æ°Ķæģ¯(-0.11) | RH(-0.11) | stice(-0.11) | RH(-0.11) | å¥½åĿı(-0.10) | commodities(-0.10) | visitors(-0.10) | come(-0.10)
    ACCEPTED as axis_178  cumulative_var=0.3773

  [ 174]  axes=179  step_var=0.0019  binary_acc=0.997  gap=0.2174  max_dot=0.0015  (1.9s)
    TOP:  æł¸éħ¸(0.12) | äº®çľ¼(0.12) | despuÃ©s(0.11) | å¥Ķèµ´(0.11) | sprung(0.10) | NE(0.10) | åī§æľ¬(0.10) | èģĬåŁİ(0.10)
    BOT:  touchdown(-0.12) | akah(-0.11) | æĹĹ(-0.11) | God(-0.10) | çĶ(-0.10) | cÃ¡o(-0.10) | society(-0.10) | ersh(-0.10)
    ACCEPTED as axis_179  cumulative_var=0.3785

  [ 175]  axes=180  step_var=0.0021  binary_acc=0.981  gap=0.2255  max_dot=0.0013  (1.9s)
    TOP:  Snowden(0.11) | å¥¥è¿Ĳä¼ļ(0.11) | medicine(0.11) | acquired(0.11) | nÄĥm(0.11) | å¥½èİ±åĿŀ(0.11) | medicinal(0.10) | è·³åĩº(0.10)
    BOT:  ä½İä»·(-0.12) | Panthers(-0.12) | rog(-0.12) | roid(-0.11) | rog(-0.11) | åĪĨéĶĢ(-0.10) | normalization(-0.10) | å¤§éģĵ(-0.10)
    ACCEPTED as axis_180  cumulative_var=0.3798

  [ 176]  axes=181  step_var=0.0020  binary_acc=0.973  gap=0.2193  max_dot=0.0028  (1.8s)
    TOP:  PRE(0.12) | ciclo(0.11) | inplace(0.11) | JO(0.10) | _re(0.10) | çļĦäºĭæĥħ(0.10) | Pre(0.10) | scissors(0.10)
    BOT:  ä¼łå¯¼(-0.11) | diner(-0.11) | éļĶçĥŃ(-0.11) | found(-0.10) | åĽŀåįĩ(-0.10) | immunity(-0.10) | ä¸ĭä¹¡(-0.10) | åº§è°Ī(-0.10)
    ACCEPTED as axis_181  cumulative_var=0.3810

  [ 177]  axes=182  step_var=0.0020  binary_acc=0.989  gap=0.2182  max_dot=0.0017  (1.9s)
    TOP:  çĽ¸åħ³æĶ¿çŃĸ(0.12) | åħļçļĦåįģä¹Ŀ(0.11) | æĹħæ¸¸åº¦åģĩ(0.11) | çŃīçĽ¸åħ³(0.11) | å¯ĨåĪĩ(0.11) | Mill(0.10) | corrective(0.10) | encryption(0.10)
    BOT:  okay(-0.12) | Glover(-0.11) | mess(-0.10) | æĦ¤æĢĴ(-0.10) | æ··ä¹±(-0.10) | å¼(-0.10) | sh(-0.10) | ok(-0.09)
    ACCEPTED as axis_182  cumulative_var=0.3822

  [ 178]  axes=183  step_var=0.0020  binary_acc=0.988  gap=0.2175  max_dot=0.0044  (1.8s)
    TOP:  multitude(0.11) | æİ¥ä¸ĭæĿ¥(0.11) | renewed(0.10) | throughout(0.10) | âĸ¼(0.10) | Ariel(0.10) | police(0.10) | authorities(0.10)
    BOT:  å·¥æľŁ(-0.12) | espresso(-0.11) | centuries(-0.11) | çĻ¾å¹´(-0.11) | ä¸¤å¹´(-0.10) | åįģåĩłå¹´(-0.10) | hobby(-0.10) | æľŁåĪĬ(-0.10)
    ACCEPTED as axis_183  cumulative_var=0.3835

  [ 179]  axes=184  step_var=0.0020  binary_acc=0.980  gap=0.2206  max_dot=0.0033  (1.9s)
    TOP:  hoe(0.13) | poignant(0.11) | ç´łæĿĲ(0.11) | longitud(0.11) | hr(0.11) | çĮª(0.11) | Jim(0.10) | æ·±åĬłå·¥(0.10)
    BOT:  trá»Ł(-0.11) | realm(-0.11) | ä¿¨(-0.11) | ä½©æĪ´(-0.10) | ä¸įå®ī(-0.10) | èµ°åľ¨(-0.10) | è®¤åı¯(-0.10) | æĦıåĲĳ(-0.10)
    ACCEPTED as axis_184  cumulative_var=0.3847

  [ 180]  axes=185  step_var=0.0021  binary_acc=0.978  gap=0.2265  max_dot=0.0038  (1.8s)
    TOP:  damned(0.11) | equally(0.11) | instructors(0.10) | unnecessarily(0.10) | merely(0.10) | Ð¿ÑĢÐ¾ÑģÑĤÐ¾(0.10) | é©¾(0.10) | å´ĸ(0.10)
    BOT:  æĶ¶æĭ¾(-0.13) | flexibility(-0.12) | ä¼łçĲĥ(-0.11) | flexible(-0.11) | íĺķ(-0.11) | forgotten(-0.11) | å¿«è®¯(-0.11) | å¿(-0.11)
    ACCEPTED as axis_185  cumulative_var=0.3860

  [ 181]  axes=186  step_var=0.0020  binary_acc=0.972  gap=0.2192  max_dot=0.0024  (1.8s)
    TOP:  åħĳ(0.11) | proud(0.11) | æµĴ(0.11) | éĵºè®¾(0.11) | ä¹Łä¸įæķ¢(0.10) | QUI(0.10) | hurst(0.10) | è±(0.10)
    BOT:  ç»¿è±Ĩ(-0.11) | åĤ¨èĵĦ(-0.11) | vector(-0.11) | Morning(-0.10) | Ly(-0.10) | teacher(-0.10) | jeune(-0.10) | spre(-0.09)
    ACCEPTED as axis_186  cumulative_var=0.3872

  [ 182]  axes=187  step_var=0.0020  binary_acc=0.997  gap=0.2200  max_dot=0.0044  (1.8s)
    TOP:  fashionable(0.14) | Often(0.13) | æ©¡èĥ¶(0.11) | æ³¨æĦıåĪ°(0.11) | indexed(0.11) | çĻ»ä¸Ĭ(0.11) | often(0.10) | Often(0.10)
    BOT:  Ø§ÙĦØª(-0.11) | stew(-0.11) | istic(-0.10) | lady(-0.10) | cord(-0.10) | spokesperson(-0.10) | æĴ¤ç¦»(-0.10) | Ð¿ÐµÑĢÐµÐ¼(-0.10)
    ACCEPTED as axis_187  cumulative_var=0.3885

  [ 183]  axes=188  step_var=0.0020  binary_acc=0.956  gap=0.2201  max_dot=0.0036  (1.9s)
    TOP:  experience(0.11) | çİĦ(0.10) | apparatus(0.10) | å¸¸å¸¸(0.10) | azer(0.10) | âĻ¥(0.10) | inflammation(0.10) | Ã¼h(0.10)
    BOT:  twenty(-0.14) | sixteen(-0.11) | twelve(-0.11) | begun(-0.11) | çĶµéĩı(-0.11) | Ð¾Ð±(-0.11) | six(-0.11) | anyhow(-0.10)
    ACCEPTED as axis_188  cumulative_var=0.3897

  [ 184]  axes=189  step_var=0.0020  binary_acc=0.997  gap=0.2178  max_dot=0.0054  (1.9s)
    TOP:  ViewHolder(0.11) | axes(0.10) | constitu(0.10) | Strategic(0.10) | éĻĲåº¦(0.10) | Foley(0.10) | çļĦçĲĨè§£(0.10) | SX(0.10)
    BOT:  æ¸ħæ´ģèĥ½æºĲ(-0.12) | niet(-0.11) | diapers(-0.11) | å¥½è½¬(-0.11) | æº¢(-0.11) | ä¸įåıĬ(-0.11) | æĸ°åŁİ(-0.10) | æİı(-0.10)
    ACCEPTED as axis_189  cumulative_var=0.3909

  [ 185]  axes=190  step_var=0.0020  binary_acc=0.997  gap=0.2215  max_dot=0.0011  (1.9s)
    TOP:  zburg(0.13) | åİĤå®¶(0.12) | adies(0.11) | impres(0.11) | pricey(0.10) | --------------------------------------------------------------------------------(0.10) | iness(0.10) | .hamcrest(0.10)
    BOT:  hopeful(-0.12) | è´ŃæĪ¿(-0.11) | Hope(-0.11) | æ¢¦(-0.11) | Hope(-0.10) | clue(-0.10) | æ¦ķ(-0.10) | Phill(-0.10)
    ACCEPTED as axis_190  cumulative_var=0.3922

  [ 186]  axes=191  step_var=0.0019  binary_acc=0.983  gap=0.2179  max_dot=0.0016  (1.8s)
    TOP:  ,u(0.11) | ski(0.10) | imoto(0.10) | cru(0.10) | amu(0.10) | IL(0.10) | MO(0.10) | LU(0.10)
    BOT:  é£²(-0.11) | å®¶çĶ¨(-0.11) | Trader(-0.10) | ynec(-0.10) | NaN(-0.10) | ayar(-0.10) | åĲ«(-0.09) | girlfriend(-0.09)
    ACCEPTED as axis_191  cumulative_var=0.3933

  [ 187]  axes=192  step_var=0.0021  binary_acc=0.999  gap=0.2273  max_dot=0.0023  (1.9s)
    TOP:  soy(0.10) | Municipal(0.10) | ç²®æ²¹(0.10) | Robert(0.10) | Roger(0.10) | Robbie(0.10) | betrayal(0.10) | ramifications(0.10)
    BOT:  åĭī(-0.11) | åħĭæľį(-0.11) | manic(-0.11) | marathon(-0.11) | çĶŁæŃ»(-0.11) | fearless(-0.10) | Axis(-0.10) | ê²(-0.10)
    ACCEPTED as axis_192  cumulative_var=0.3946

  [ 188]  axes=193  step_var=0.0019  binary_acc=0.975  gap=0.2089  max_dot=0.0028  (1.8s)
    TOP:  nullable(0.11) | Somerset(0.10) | ä¸ĸçºª(0.10) | Ñĸ(0.10) | å®ŀçĶ¨æĢ§(0.10) | åĽ½ä¼ģ(0.10) | æİ¥åı£(0.10) | widely(0.10)
    BOT:  èĪĮå¤´(-0.11) | éĢłè¡Ģ(-0.11) | sauce(-0.10) | Ø¸(-0.10) | Kb(-0.10) | Kyle(-0.10) | gravy(-0.10) | coeff(-0.10)
    ACCEPTED as axis_193  cumulative_var=0.3957

  [ 189]  axes=194  step_var=0.0021  binary_acc=0.994  gap=0.2224  max_dot=0.0030  (1.8s)
    TOP:  Wh(0.14) | crowds(0.12) | terminal(0.10) | wh(0.10) | Wh(0.10) | åįĥéĩĮ(0.10) | sights(0.10) | angi(0.10)
    BOT:  cháº¯c(-0.13) | Petroleum(-0.11) | çķĻåľ¨(-0.11) | lag(-0.10) | èµ·çłģ(-0.10) | dive(-0.10) | dove(-0.10) | strap(-0.10)
    ACCEPTED as axis_194  cumulative_var=0.3970

  [ 190]  axes=195  step_var=0.0020  binary_acc=0.999  gap=0.2191  max_dot=0.0012  (2.0s)
    TOP:  unwanted(0.11) | Photo(0.11) | extensively(0.11) | bravery(0.11) | å¤ĩåıĹ(0.10) | ØªØ¨(0.10) | Ð¼Ð¾ÑĢ(0.10) | çĻ½èī²(0.10)
    BOT:  ä¸ŃåĮ»(-0.11) | æĪ·ç±į(-0.11) | faint(-0.11) | ëįĶ(-0.10) | cá»Ńa(-0.10) | ITS(-0.10) | ServiceImpl(-0.10) | fiscal(-0.10)
    ACCEPTED as axis_195  cumulative_var=0.3982

  [ 191]  axes=196  step_var=0.0021  binary_acc=0.969  gap=0.2260  max_dot=0.0033  (1.9s)
    TOP:  çłĶåŃ¦(0.10) | Kaw(0.10) | kindness(0.10) | Uran(0.10) | skillet(0.10) | ä¸ľé£İ(0.10) | é¦ĸæī¹(0.10) | readiness(0.10)
    BOT:  compliment(-0.11) | servicing(-0.11) | selenium(-0.11) | slight(-0.11) | Sri(-0.11) | elect(-0.11) | colspan(-0.10) | à¹īà¸²à¸ĩ(-0.10)
    ACCEPTED as axis_196  cumulative_var=0.3995

  [ 192]  axes=197  step_var=0.0020  binary_acc=0.994  gap=0.2158  max_dot=0.0004  (1.8s)
    TOP:  pickup(0.12) | pickups(0.11) | -nav(0.11) | èµŀèµı(0.11) | èģĶç³»æĪĳä»¬(0.10) | åİ¨(0.10) | upset(0.10) | pick(0.10)
    BOT:  ÑģÑĤÑĢÐ°Ð½(-0.10) | jections(-0.10) | campaigns(-0.10) | clown(-0.10) | _part(-0.10) | MLB(-0.10) | LastName(-0.09) | Spring(-0.09)
    ACCEPTED as axis_197  cumulative_var=0.4007

  [ 193]  axes=198  step_var=0.0021  binary_acc=0.979  gap=0.2185  max_dot=0.0015  (1.8s)
    TOP:  éĢļçĶ¨(0.12) | sister(0.11) | ç¬¬äºĶ(0.11) | ç¬¬ä¸ī(0.11) | è¿Ħä»Ĭ(0.11) | ç¬¬ä¸ĥ(0.10) | åħŃåįģ(0.10) | ilee(0.10)
    BOT:  æĸ¹å¼ıè¿Ľè¡Į(-0.12) | çº¢èĮ¶(-0.11) | ä¸»åĬŀæĸ¹(-0.11) | flora(-0.11) | ä¸įçĶ±å¾Ĺ(-0.11) | tendency(-0.11) | arrange(-0.11) | ä¼ģä¸ļå®¶(-0.11)
    ACCEPTED as axis_198  cumulative_var=0.4019

  [ 194]  axes=199  step_var=0.0019  binary_acc=0.974  gap=0.2105  max_dot=0.0026  (1.9s)
    TOP:  æİĪäºĪ(0.10) | colourful(0.10) | bp(0.10) | colorful(0.10) | Inc(0.10) | æ¶²æĻ¶(0.10) | åľ°ä¸Ńæµ·(0.10) | sec(0.09)
    BOT:  é¦¥(-0.12) | lign(-0.12) | çªĸ(-0.11) | flaming(-0.11) | Ale(-0.11) | loy(-0.10) | çĻ¾åĪĨ(-0.10) | åĵŃäºĨ(-0.10)
    ACCEPTED as axis_199  cumulative_var=0.4031

  [ 195]  axes=200  step_var=0.0019  binary_acc=0.987  gap=0.2135  max_dot=0.0020  (1.9s)
    TOP:  ges(0.11) | tÃ¹y(0.10) | å§Ŀ(0.10) | ca(0.10) | rit(0.10) | dominant(0.09) | Camera(0.09) | æĢĿ(0.09)
    BOT:  put(-0.11) | åħļçļĦå»ºè®¾(-0.11) | ç®ĢçĽ´(-0.10) | usefulness(-0.10) | jetzt(-0.10) | planners(-0.10) | æİ©(-0.10) | PIO(-0.10)
    ACCEPTED as axis_200  cumulative_var=0.4042

  [ 196]  axes=201  step_var=0.0019  binary_acc=0.994  gap=0.2135  max_dot=0.0049  (1.8s)
    TOP:  ä¸Ńåįİ(0.11) | qued(0.10) | delegates(0.10) | æ¸ħåįķ(0.10) | preg(0.10) | æ¡Į(0.10) | çī¹åĪ«å£°æĺİ(0.10) | silk(0.10)
    BOT:  adjoining(-0.11) | æľīä¸Ģ(-0.11) | éĢīåıĸ(-0.11) | è¿ŀæİ¥(-0.10) | Pra(-0.10) | winger(-0.10) | åĪ¶èį¯(-0.10) | ä¸¤è¾¹(-0.10)
    ACCEPTED as axis_201  cumulative_var=0.4054

  [ 197]  axes=202  step_var=0.0020  binary_acc=0.974  gap=0.2132  max_dot=0.0017  (1.9s)
    TOP:  surveys(0.12) | åĩĢåĢ¼(0.11) | YG(0.11) | Psych(0.11) | employing(0.11) | employed(0.10) | values(0.10) | Interview(0.10)
    BOT:  åħ¨æ°ĳåģ¥èº«(-0.11) | ÑĢÐµÐ¼(-0.10) | .argmax(-0.10) | ç«ŀä»·(-0.10) | æ¹ĸæ³Ĭ(-0.10) | Ð¿Ð¾Ð±(-0.10) | amma(-0.10) | ä¸Ģåı¥è¯Ŀ(-0.10)
    ACCEPTED as axis_202  cumulative_var=0.4066

  [ 198]  axes=203  step_var=0.0020  binary_acc=1.000  gap=0.2128  max_dot=0.0041  (1.8s)
    TOP:  syll(0.12) | åľ¨ä¸Ĭæµ·(0.11) | aspect(0.11) | gem(0.10) | principals(0.10) | NY(0.10) | Having(0.10) | æį®ç»Łè®¡(0.10)
    BOT:  bizarre(-0.12) | ÑģÐ»ÐµÐ´(-0.11) | Ð¿ÑĢÐ¾Ð¸Ð·(-0.11) | èģĶèµĽ(-0.10) | EX(-0.10) | äº®(-0.10) | heartfelt(-0.10) | Health(-0.10)
    ACCEPTED as axis_203  cumulative_var=0.4078

  [ 199]  axes=204  step_var=0.0019  binary_acc=0.981  gap=0.2112  max_dot=0.0038  (1.8s)
    TOP:  åħįè´¹(0.12) | eag(0.12) | æĥħä¾£(0.11) | adverse(0.11) | proposition(0.10) | å®łçī©(0.10) | é¤Ĳåħ·(0.10) | å¸®åĬ©ä¼ģä¸ļ(0.10)
    BOT:  jealous(-0.12) | Ø¨Ø±(-0.11) | jealousy(-0.11) | pm(-0.11) | vas(-0.10) | ãģĦãģı(-0.10) | çĽ¸éģĩ(-0.10) | ãĥ¨(-0.10)
    ACCEPTED as axis_204  cumulative_var=0.4089

  [ 200]  axes=205  step_var=0.0019  binary_acc=0.986  gap=0.2117  max_dot=0.0016  (1.8s)
    TOP:  åĬłä»¥(0.12) | çµ®(0.11) | flea(0.11) | Toolkit(0.11) | åı¯è¾¾(0.10) | asleep(0.10) | festive(0.10) | èĬ±åįī(0.10)
    BOT:  æ¸¤(-0.11) | bc(-0.10) | æģĴå¤§(-0.10) | eer(-0.10) | çĶŁæĢģ(-0.10) | ç¬ĳè¯Ŀ(-0.10) | Official(-0.10) | æĦıå¤ĸ(-0.10)
    ACCEPTED as axis_205  cumulative_var=0.4100

  [ 201]  axes=206  step_var=0.0020  binary_acc=0.995  gap=0.2137  max_dot=0.0027  (1.8s)
    TOP:  appear(0.11) | hobby(0.11) | è¥¿çº¢æŁ¿(0.11) | å¤ĸè§Ĥ(0.10) | Bachelor(0.10) | å°ı(0.10) | transplant(0.10) | billed(0.10)
    BOT:  æī¾åĪ°äºĨ(-0.12) | æľĢåĲİä¸Ģæ¬¡(-0.11) | æľĢæĹ©(-0.11) | ÑĢÑĭÐ²(-0.11) | ÑģÐ¿(-0.10) | çļĦæľīæķĪ(-0.10) | isEqualTo(-0.10) | ×§(-0.10)
    ACCEPTED as axis_206  cumulative_var=0.4112

  [ 202]  axes=207  step_var=0.0020  binary_acc=0.980  gap=0.2113  max_dot=0.0011  (1.9s)
    TOP:  å¤ĸè´¸(0.11) | å¤§å¹ħ(0.10) | çŃ¾çº¦ä»ªå¼ı(0.10) | }[(0.10) | thá»©(0.10) | unky(0.10) | åĩ¶(0.10) | åĩ¹(0.09)
    BOT:  credible(-0.11) | ç²īå°ĺ(-0.11) | æİ¥åĬĽ(-0.11) | caste(-0.11) | ìŀĺëª»(-0.10) | job(-0.10) | å¢©(-0.10) | Muslim(-0.10)
    ACCEPTED as axis_207  cumulative_var=0.4124

  [ 203]  axes=208  step_var=0.0019  binary_acc=0.994  gap=0.2067  max_dot=0.0021  (1.9s)
    TOP:  suffix(0.10) | ifest(0.10) | ê±°(0.10) | devices(0.10) | éĢĹ(0.09) | è¿ĶåĽŀ(0.09) | Manifest(0.09) | chaque(0.09)
    BOT:  æĭħå½ĵ(-0.12) | ç®¡è¾ĸ(-0.11) | åħļä¸Ńå¤®(-0.10) | _env(-0.10) | Jiang(-0.10) | å¿ĥè·³(-0.10) | åįķåįķ(-0.10) | loginUser(-0.10)
    ACCEPTED as axis_208  cumulative_var=0.4135

  [ 204]  axes=209  step_var=0.0019  binary_acc=0.986  gap=0.2100  max_dot=0.0012  (1.9s)
    TOP:  æ··æ²Į(0.10) | utility(0.10) | éĤ¬(0.10) | æĽ(0.10) | å¤ļå½©(0.10) | ailer(0.10) | umper(0.10) | å¤ķ(0.10)
    BOT:  Theater(-0.11) | çĽ¸åºĶ(-0.10) | èĢķ(-0.10) | éĥ¨åĪĨ(-0.10) | ////(-0.10) | oh(-0.10) | refriger(-0.10) | äºĭæķħ(-0.10)
    ACCEPTED as axis_209  cumulative_var=0.4146

  [ 205]  axes=210  step_var=0.0019  binary_acc=0.983  gap=0.2087  max_dot=0.0045  (1.8s)
    TOP:  shoes(0.11) | éĹ¨(0.11) | ÐµÐ²Ð¸Ñĩ(0.10) | Unfortunately(0.10) | dated(0.10) | è½¬åĲĳ(0.10) | åıĮåĲĳ(0.10) | ä¸Ģä¸ªæĸ°çļĦ(0.10)
    BOT:  è¨ĺäºĭ(-0.11) | çŁ¥(-0.11) | amedi(-0.10) | å»ºæĿĲ(-0.10) | æ°ª(-0.10) | ä¸Ĭåĳ¨(-0.10) | Below(-0.10) | è¡ĢèĦĤ(-0.10)
    ACCEPTED as axis_210  cumulative_var=0.4157

  [ 206]  axes=211  step_var=0.0020  binary_acc=1.000  gap=0.2082  max_dot=0.0037  (1.8s)
    TOP:  Ñ(0.12) | åĨľä¸ļå¤§åŃ¦(0.10) | ä¸ĥå¤§(0.10) | åľ°äº§(0.10) | èĬĤåģĩæĹ¥(0.10) | ,parent(0.10) | äº§åľ°(0.10) | azu(0.10)
    BOT:  .yml(-0.10) | æŁ¥æĺİ(-0.10) | ä¸ĭåİ»(-0.10) | æĮ¯åĬ¨(-0.10) | ÐĿÐ°(-0.10) | painters(-0.10) | Shakespeare(-0.10) | specialists(-0.10)
    ACCEPTED as axis_211  cumulative_var=0.4169

  [ 207]  axes=212  step_var=0.0020  binary_acc=0.973  gap=0.2132  max_dot=0.0058  (1.8s)
    TOP:  drums(0.11) | saturation(0.11) | dom(0.10) | zman(0.10) | widespread(0.10) | Err(0.10) | åĽ½å¤ĸ(0.10) | Win(0.10)
    BOT:  éĩįçĶŁ(-0.11) | èĪĴ(-0.11) | invest(-0.10) | Never(-0.10) | travelers(-0.10) | è®²è¿°äºĨ(-0.10) | çı©(-0.10) | éĺĲè¿°(-0.10)
    ACCEPTED as axis_212  cumulative_var=0.4180

  [ 208]  axes=213  step_var=0.0020  binary_acc=0.990  gap=0.2161  max_dot=0.0036  (1.8s)
    TOP:  Source(0.10) | marque(0.10) | planet(0.10) | neighbour(0.10) | Sources(0.10) | ograph(0.10) | å¤ĸåªĴ(0.10) | asteroids(0.10)
    BOT:  stil(-0.11) | af(-0.11) | ABA(-0.11) | FC(-0.10) | convin(-0.10) | .getBoundingClientRect(-0.10) | stood(-0.10) | äºĺ(-0.10)
    ACCEPTED as axis_213  cumulative_var=0.4192

  [ 209]  axes=214  step_var=0.0020  binary_acc=0.994  gap=0.2171  max_dot=0.0026  (1.8s)
    TOP:  alongside(0.12) | FD(0.11) | [^(0.11) | nda(0.11) | hes(0.10) | durante(0.10) | backed(0.10) | HIR(0.10)
    BOT:  æĹħè¡Į(-0.12) | rom(-0.11) | å¸Ĥæ°ĳ(-0.10) | integers(-0.10) | Challenge(-0.10) | Gray(-0.10) | åľºæĻ¯(-0.10) | èģĮåľº(-0.10)
    ACCEPTED as axis_214  cumulative_var=0.4203

  [ 210]  axes=215  step_var=0.0020  binary_acc=0.946  gap=0.2087  max_dot=0.0009  (1.9s)
    TOP:  hardt(0.13) | æĢĿãģĦ(0.12) | catal(0.11) | AA(0.10) | Roosevelt(0.10) | Maui(0.10) | é©¬æ¡¶(0.10) | ãģĬ(0.10)
    BOT:  semantics(-0.12) | mur(-0.11) | traded(-0.11) | é¢ľèī²(-0.10) | ç®¡çĲĨç³»ç»Ł(-0.10) | ä¸įåı¯èĥ½(-0.10) | empowering(-0.10) | åĳĹ(-0.10)
    ACCEPTED as axis_215  cumulative_var=0.4215

  [ 211]  axes=216  step_var=0.0020  binary_acc=0.968  gap=0.2146  max_dot=0.0022  (1.9s)
    TOP:  UV(0.12) | æīĵåĬ¨(0.11) | å®ŀçĶ¨(0.11) | chac(0.10) | opinion(0.10) | LCD(0.10) | illustrated(0.10) | -ray(0.10)
    BOT:  èĬ±éĴ±(-0.11) | æĮ½(-0.11) | adventurous(-0.11) | çľ¼çļ®(-0.10) | Sevent(-0.10) | åĪĽä¸ļèĢħ(-0.10) | éĩĳèŀįæľºæŀĦ(-0.10) | adesh(-0.10)
    ACCEPTED as axis_216  cumulative_var=0.4226

  [ 212]  axes=217  step_var=0.0019  binary_acc=0.968  gap=0.2116  max_dot=0.0017  (1.8s)
    TOP:  ä¸ĵé¡¹æĸĹäºī(0.12) | è¿ĳè·Ŀç¦»(0.11) | åľ¨ä¸Ģèµ·(0.11) | é£İæ°Ķ(0.11) | éĿ¢å¯¹éĿ¢(0.10) | Bron(0.10) | denomination(0.10) | éĹ´éļĻ(0.10)
    BOT:  ìŀħ(-0.10) | regulations(-0.10) | æ¬¢ä¹Ĳ(-0.10) | Hank(-0.10) | éĵ¶æ²³(-0.10) | çļĦäººéĥ½(-0.09) | ä¸įæİĴéĻ¤(-0.09) | åı¯è§ģ(-0.09)
    ACCEPTED as axis_217  cumulative_var=0.4238

  [ 213]  axes=218  step_var=0.0019  binary_acc=0.985  gap=0.2157  max_dot=0.0024  (1.8s)
    TOP:  blogs(0.12) | å¤§åħ¨(0.11) | å¿ħå¤ĩ(0.11) | å¿«æīĭ(0.11) | çº¯æ´ģ(0.11) | åĨ»ç»ĵ(0.11) | æľįåĬ¡åĻ¨(0.10) | èĪªè¿Ĳ(0.10)
    BOT:  æĹ¥ä¸ĬåįĪ(-0.12) | routines(-0.11) | æĻ¯è±¡(-0.11) | åıĸå¾ĹäºĨ(-0.11) | å²ļ(-0.10) | æĹĹ(-0.10) | ÑĥÐ´Ð°Ð»Ð¾ÑģÑĮ(-0.10) | ball(-0.10)
    ACCEPTED as axis_218  cumulative_var=0.4249

  [ 214]  axes=219  step_var=0.0020  binary_acc=0.997  gap=0.2089  max_dot=0.0023  (1.8s)
    TOP:  éĹŃçİ¯(0.11) | çŁŃæĿ¿(0.11) | ç½¡(0.11) | Ð±ÑĭÐ»Ð¾(0.10) | é²į(0.10) | BO(0.10) | intent(0.10) | ificial(0.10)
    BOT:  åıĤå±ķ(-0.12) | win(-0.11) | .show(-0.11) | èĤ¯å®ļ(-0.11) | åī§éĻ¢(-0.10) | å±ķä¼ļ(-0.10) | taxed(-0.10) | åıĤèµĽ(-0.10)
    ACCEPTED as axis_219  cumulative_var=0.4260

  [ 215]  axes=220  step_var=0.0019  binary_acc=0.999  gap=0.2019  max_dot=0.0013  (1.9s)
    TOP:  éĢĴç»Ļ(0.10) | ä¸Ģä»¶(0.10) | succession(0.10) | æ¶Īè´¹åĵģ(0.09) | ç¾¤å²Ľ(0.09) | åħ±åĴĮ(0.09) | ç¬¬ä¸ĢçĻ¾(0.09) | Faculty(0.09)
    BOT:  è§ĦåĪĴè®¾è®¡(-0.11) | åį³ä½¿(-0.10) | åı¯ä»¥çĶ¨(-0.10) | æľĢè¿ĳ(-0.10) | ÑĢÐ°ÑģÐ¿Ð¾Ð»Ð¾Ð¶(-0.10) | çºµæ¨ª(-0.10) | quite(-0.10) | ç«Ļç«ĭ(-0.10)
    ACCEPTED as axis_220  cumulative_var=0.4271

  [ 216]  axes=221  step_var=0.0020  binary_acc=0.974  gap=0.2099  max_dot=0.0018  (1.9s)
    TOP:  åħħ(0.11) | /.(0.11) | åĲĮå¿Ĺä»¬(0.11) | asylum(0.10) | åĲĮå¿Ĺ(0.10) | transported(0.10) | æ»ķ(0.10) | cerpt(0.10)
    BOT:  clinics(-0.11) | */Ċ(-0.11) | Â»(-0.10) | peu(-0.10) | é½¿(-0.10) | Lots(-0.10) | fruit(-0.10) | Lisa(-0.10)
    ACCEPTED as axis_221  cumulative_var=0.4282

  [ 217]  axes=222  step_var=0.0019  binary_acc=0.989  gap=0.2049  max_dot=0.0017  (1.8s)
    TOP:  emoji(0.11) | èīºäºº(0.10) | åľºéĿ¢(0.10) | å¤§ä½¿(0.10) | ç¨³(0.10) | amat(0.09) | smoothly(0.09) | ×ķ×ľ(0.09)
    BOT:  Answer(-0.11) | Group(-0.10) | é¢łè¦Ĩ(-0.10) | answer(-0.10) | ìĦ¸(-0.10) | ÑĤÑĢÐµÑĤÑĮ(-0.10) | urlparse(-0.10) | gu(-0.10)
    ACCEPTED as axis_222  cumulative_var=0.4293

  [ 218]  axes=223  step_var=0.0020  binary_acc=0.994  gap=0.2200  max_dot=0.0038  (1.9s)
    TOP:  åį§å®¤(0.11) | ç§ĭåŃ£(0.11) | å¤ªæ¹ĸ(0.11) | airborne(0.11) | federal(0.10) | åĩºè¡Į(0.10) | æľĿå»·(0.10) | fall(0.10)
    BOT:  ä½İè¿·(-0.14) | yourselves(-0.11) | merits(-0.11) | herself(-0.11) | Madonna(-0.10) | æĳ¸ç´¢(-0.10) | #####(-0.10) | _stdio(-0.10)
    ACCEPTED as axis_223  cumulative_var=0.4305

  [ 219]  axes=224  step_var=0.0020  binary_acc=0.986  gap=0.2090  max_dot=0.0029  (1.9s)
    TOP:  overwhelm(0.10) | delayed(0.10) | à¹Īà¸§à¸Ļ(0.10) | æĻķ(0.09) | .SimpleDateFormat(0.09) | odynamic(0.09) | OutputStream(0.09) | ourn(0.09)
    BOT:  æľ¬ç§ĳ(-0.12) | Ð½Ð¸ÐºÐ¾Ð³Ð´Ð°(-0.11) | elsewhere(-0.10) | æľ¬è´¨(-0.10) | pedestal(-0.10) | ä½©æĪ´(-0.10) | Posted(-0.10) | gu(-0.10)
    ACCEPTED as axis_224  cumulative_var=0.4316

  [ 220]  axes=225  step_var=0.0019  binary_acc=0.998  gap=0.2065  max_dot=0.0039  (1.8s)
    TOP:  ç»ĵåĲĪèµ·æĿ¥(0.11) | å®¶ä¼Ļ(0.11) | æıĲéĨĴ(0.11) | åį±æľº(0.10) | ç§(0.10) | ä½łä»¬(0.10) | çŃīåľ°(0.10) | åı½(0.09)
    BOT:  åį³æĺ¯(-0.11) | ä»ħæľī(-0.10) | å·´èĲ¨(-0.10) | è®¾(-0.10) | discomfort(-0.10) | ä»£è¡¨æĢ§(-0.10) | èĩªå¸¦(-0.10) | ballet(-0.09)
    ACCEPTED as axis_225  cumulative_var=0.4327

  [ 221]  axes=226  step_var=0.0019  binary_acc=0.983  gap=0.2073  max_dot=0.0016  (1.9s)
    TOP:  äº¤éĢļè¿Ĳè¾ĵ(0.13) | é»ĺå¥ĳ(0.11) | acey(0.11) | conjunction(0.11) | ÑģÐºÐ°Ñĩ(0.10) | æĿ¥èĩªäºİ(0.10) | Animal(0.10) | éħįåĲĪ(0.10)
    BOT:  å®ŀè®Ń(-0.12) | scholarships(-0.10) | ç¡ķå£«(-0.10) | unts(-0.10) | FO(-0.10) | )=>(-0.10) | sebuah(-0.10) | è¿Ĳç®Ĺ(-0.10)
    ACCEPTED as axis_226  cumulative_var=0.4338

  [ 222]  axes=227  step_var=0.0019  binary_acc=0.987  gap=0.2038  max_dot=0.0047  (1.8s)
    TOP:  honest(0.11) | ä¸ĵåįĸ(0.10) | gost(0.10) | -Benz(0.10) | Hear(0.09) | èį£èİ·(0.09) | åĽŀäºĭ(0.09) | æ·±å±Ĥæ¬¡(0.09)
    BOT:  Schw(-0.10) | relative(-0.10) | æĪĳä»¬è®¤ä¸º(-0.10) | æīĵåį°æľº(-0.10) | Video(-0.09) | fairly(-0.09) | æ¸¯åı£(-0.09) | è¾ĥå°ı(-0.09)
    ACCEPTED as axis_227  cumulative_var=0.4349

  [ 223]  axes=228  step_var=0.0020  binary_acc=0.995  gap=0.2181  max_dot=0.0037  (1.8s)
    TOP:  éĢįéģ¥(0.13) | æĿ¡ä»¶(0.12) | gear(0.11) | ä»¶(0.11) | ufen(0.11) | suggestions(0.11) | æĹłå¿§(0.11) | Recommended(0.10)
    BOT:  plt(-0.11) | =>{Ċ(-0.10) | ________________________________________________________________(-0.10) | Tuesday(-0.10) | æ´ª(-0.10) | @endsection(-0.10) | navy(-0.10) | Nak(-0.10)
    ACCEPTED as axis_228  cumulative_var=0.4360

  [ 224]  axes=229  step_var=0.0019  binary_acc=0.992  gap=0.2052  max_dot=0.0067  (1.8s)
    TOP:  ¹(0.11) | éĢīç§Ģ(0.11) | ÑĥÑģÐ»ÑĥÐ³Ð¸(0.11) | electronically(0.10) | éĢļè¡Įè¯ģ(0.10) | electronic(0.10) | eli(0.10) | negoci(0.10)
    BOT:  à¦¾à¦(-0.12) | aspirations(-0.11) | storage(-0.10) | reportedly(-0.10) | åĽ°æĥĳ(-0.10) | {}'.(-0.10) | å¯ºéĻ¢(-0.10) | æ´®(-0.10)
    ACCEPTED as axis_229  cumulative_var=0.4371

  [ 225]  axes=230  step_var=0.0020  binary_acc=0.979  gap=0.2138  max_dot=0.0046  (1.8s)
    TOP:  ìŀĲê¸°(0.11) | ikes(0.10) | ÑĤÐ°ÐºÐ¾Ðµ(0.10) | imposition(0.10) | .IsNullOrWhiteSpace(0.10) | ä½ľæĪĺ(0.10) | å¯¹æĸ¹(0.10) | çĲ´(0.10)
    BOT:  å¹´èĸª(-0.11) | æ·¤(-0.11) | èµĬ(-0.10) | å®Łãģ¯(-0.10) | martyr(-0.10) | marrow(-0.10) | çĥĺå¹²(-0.10) | softly(-0.09)
    ACCEPTED as axis_230  cumulative_var=0.4382

  [ 226]  axes=231  step_var=0.0019  binary_acc=0.992  gap=0.2082  max_dot=0.0063  (1.8s)
    TOP:  WooCommerce(0.10) | clientele(0.10) | homeowners(0.10) | (**(0.10) | RNA(0.10) | EEG(0.09) | æŃ(0.09) | éħļ(0.09)
    BOT:  dÃµi(-0.12) | åĪĿæģĭ(-0.11) | childbirth(-0.11) | ë°°(-0.10) | çĽĳè§Ĩ(-0.10) | tender(-0.10) | falsehood(-0.10) | çľŁçĪ±(-0.10)
    ACCEPTED as axis_231  cumulative_var=0.4393

  [ 227]  axes=232  step_var=0.0019  binary_acc=0.965  gap=0.1991  max_dot=0.0025  (1.8s)
    TOP:  å©·(0.10) | åħ¥æĪ·(0.10) | äººèº«(0.10) | rik(0.10) | Uh(0.10) | NBA(0.10) | Verd(0.10) | çīĮ(0.09)
    BOT:  èĸĦèĨľ(-0.11) | íĥ(-0.11) | çĲĨæĢ§(-0.10) | imestone(-0.10) | é³(-0.10) | ÃŃt(-0.10) | é¦Ļèķī(-0.10) | readers(-0.10)
    ACCEPTED as axis_232  cumulative_var=0.4403

  [ 228]  axes=233  step_var=0.0019  binary_acc=0.959  gap=0.2024  max_dot=0.0068  (1.8s)
    TOP:  é¼łæłĩ(0.10) | histoire(0.10) | sums(0.10) | çģŀ(0.10) | ou(0.10) | åį«(0.10) | éĤ£åĦ¿(0.10) | çļĭ(0.10)
    BOT:  å¹¶ä¸İ(-0.12) | Sug(-0.10) | è¿ŀçº¿(-0.10) | çłĶç©¶æīĢ(-0.10) | çħİ(-0.10) | Embassy(-0.10) | ç²¾å¯Ĩ(-0.10) | å¯»(-0.10)
    ACCEPTED as axis_233  cumulative_var=0.4414

  [ 229]  axes=234  step_var=0.0020  binary_acc=0.990  gap=0.2151  max_dot=0.0008  (1.8s)
    TOP:  promote(0.12) | dial(0.11) | ÐºÐ¾Ð¶Ð¸(0.10) | å¼łå®¶åı£(0.10) | vitality(0.10) | aficion(0.10) | pendant(0.10) | å£¶(0.10)
    BOT:  è®°èĢħéĩĩè®¿(-0.11) | è¿ĩäºİ(-0.11) | coax(-0.11) | è¶Ĭæĺ¯(-0.11) | catches(-0.10) | éĻįæ°´(-0.10) | ÑģÐ¾Ð²ÑģÐµÐ¼(-0.10) | ãĢĢãĢĢ(-0.10)
    ACCEPTED as axis_234  cumulative_var=0.4425

  [ 230]  axes=235  step_var=0.0019  binary_acc=0.958  gap=0.1998  max_dot=0.0066  (1.9s)
    TOP:  pri(0.11) | Early(0.10) | à¹ĥà¸ª(0.10) | [-(0.10) | [,(0.10) | regn(0.10) | ä¸»ä½ĵè´£ä»»(0.09) | downloads(0.09)
    BOT:  Hutch(-0.12) | åĳĨ(-0.11) | extends(-0.10) | åħµåĽ¢(-0.10) | æĭĽ(-0.09) | å¹³åĩ¡(-0.09) | Anim(-0.09) | NEC(-0.09)
    ACCEPTED as axis_235  cumulative_var=0.4436

  [ 231]  axes=236  step_var=0.0019  binary_acc=0.970  gap=0.2013  max_dot=0.0006  (1.8s)
    TOP:  ÐºÑĢÑĭ(0.12) | æ¿Ģåħī(0.10) | Ð¿Ð¾Ð´ÑħÐ¾Ð´(0.10) | closet(0.10) | Ð³Ð¾Ð´(0.10) | æĬĺç£¨(0.10) | ä¸Ńåħ³æĿĳ(0.10) | å°ıäºİ(0.10)
    BOT:  patterns(-0.12) | åĲĮçĽŁ(-0.12) | ĉĉ(-0.11) | Recap(-0.10) | æĿ¥çļĦ(-0.10) | pattern(-0.10) | æĶ¶(-0.09) | åĮ»çĸĹåį«çĶŁ(-0.09)
    ACCEPTED as axis_236  cumulative_var=0.4446

  [ 232]  axes=237  step_var=0.0019  binary_acc=0.983  gap=0.2075  max_dot=0.0057  (1.8s)
    TOP:  æ®´(0.11) | åįĬä¸ªæľĪ(0.10) | åº§æ¤ħ(0.10) | æ®¿ä¸ĭ(0.10) | ä¸Ģå®ļä¼ļ(0.10) | globe(0.10) | workbook(0.09) | ÐĳÐµÑģ(0.09)
    BOT:  ç®¡çĲĨä½ĵç³»(-0.12) | Explanation(-0.11) | åĮħåĲ«äºĨ(-0.11) | fÃ¶(-0.11) | èįīåİŁ(-0.10) | cod(-0.10) | å¿łè¯ļ(-0.10) | Very(-0.10)
    ACCEPTED as axis_237  cumulative_var=0.4457

  [ 233]  axes=238  step_var=0.0019  binary_acc=0.971  gap=0.2022  max_dot=0.0052  (1.8s)
    TOP:  morning(0.12) | tonight(0.12) | Sunday(0.11) | Saturday(0.11) | Tonight(0.11) | Replies(0.11) | Tuesday(0.11) | Tonight(0.11)
    BOT:  åĵ®åĸĺ(-0.11) | ÑģÑĤÐµÐ½(-0.10) | ä¸ľçĽŁ(-0.10) | Effect(-0.10) | æµ·å¤ĸå¸Ĥåľº(-0.10) | ÑģÐ´ÐµÐ»(-0.10) | èĪªæµ·(-0.10) | cosas(-0.10)
    ACCEPTED as axis_238  cumulative_var=0.4468

  [ 234]  axes=239  step_var=0.0019  binary_acc=0.991  gap=0.2048  max_dot=0.0017  (1.9s)
    TOP:  GBT(0.11) | å²ļ(0.11) | èĥ¸åīį(0.10) | /hr(0.10) | mland(0.10) | ä¸¤(0.10) | "~(0.10) | CB(0.10)
    BOT:  sender(-0.10) | appa(-0.10) | Ston(-0.10) | calcium(-0.10) | å®Įåħ¨æ²¡æľī(-0.10) | demasi(-0.10) | shook(-0.10) | éĢıè¿ĩ(-0.09)
    ACCEPTED as axis_239  cumulative_var=0.4478

  [ 235]  axes=240  step_var=0.0019  binary_acc=0.988  gap=0.2020  max_dot=0.0060  (1.8s)
    TOP:  çº¸ä¸Ĭ(0.12) | æľ¨æĿĲ(0.11) | .MIN(0.10) | çłº(0.10) | textile(0.10) | åĮĢ(0.10) | .cpp(0.10) | à¸ŀà¸£(0.10)
    BOT:  åĴ±ä»¬(-0.11) | calculus(-0.10) | yses(-0.10) | dialect(-0.10) | irrelevant(-0.09) | Ep(-0.09) | Cowboys(-0.09) | çº³åħ¥(-0.09)
    ACCEPTED as axis_240  cumulative_var=0.4488

  [ 236]  axes=241  step_var=0.0019  binary_acc=0.995  gap=0.2041  max_dot=0.0044  (1.8s)
    TOP:  çĽ¸å½ĵäºİ(0.12) | rightness(0.10) | åħ¨å¹´(0.10) | oh(0.10) | Second(0.09) | éĿŀæ³ķ(0.09) | ketogenic(0.09) | mercury(0.09)
    BOT:  çľī(-0.11) | äººäºº(-0.10) | outward(-0.10) | äººå¿ĥ(-0.10) | à¸±à¸ģ(-0.10) | çĽ¸è¯Ĩ(-0.10) | ç»¿åŁİ(-0.10) | å¤ĸçķĮ(-0.10)
    ACCEPTED as axis_241  cumulative_var=0.4499

  [ 237]  axes=242  step_var=0.0020  binary_acc=0.960  gap=0.2074  max_dot=0.0033  (1.9s)
    TOP:  amongst(0.12) | éĢ¸(0.11) | expelled(0.10) | Scotch(0.10) | deficient(0.10) | ÙĦÙĦ(0.10) | ä¸Ńæŀ¢(0.10) | _report(0.10)
    BOT:  Ø·(-0.11) | cryptography(-0.10) | baff(-0.10) | bre(-0.09) | uelle(-0.09) | thá»Ŀi(-0.09) | Steelers(-0.09) | quire(-0.09)
    ACCEPTED as axis_242  cumulative_var=0.4510

  [ 238]  axes=243  step_var=0.0019  binary_acc=0.985  gap=0.2025  max_dot=0.0045  (1.8s)
    TOP:  manual(0.10) | Couple(0.10) | ä¹¡åľŁ(0.10) | æĳ©æĵ¦(0.10) | éģ¥(0.10) | va(0.10) | thermo(0.10) | .Toolbar(0.10)
    BOT:  æ¸ħ(-0.10) | agn(-0.10) | Ebony(-0.10) | Baptist(-0.10) | å¤§åĬĽæĶ¯æĮģ(-0.10) | sometimes(-0.10) | sizable(-0.10) | yar(-0.09)
    ACCEPTED as axis_243  cumulative_var=0.4520

  [ 239]  axes=244  step_var=0.0020  binary_acc=0.970  gap=0.2055  max_dot=0.0006  (1.9s)
    TOP:  èĻļæĭŁ(0.11) | å®¡è®¡(0.10) | åħ¬æŃ£(0.10) | æĬĹåĩ»(0.10) | inaccur(0.10) | shortened(0.10) | Ã³i(0.10) | pron(0.10)
    BOT:  ks(-0.11) | plan(-0.10) | é¢ĸ(-0.10) | oplan(-0.10) | Holmes(-0.10) | æīĵåį¡(-0.10) | è¯ķç®¡(-0.10) | å¨ģæµ·(-0.10)
    ACCEPTED as axis_244  cumulative_var=0.4531

  [ 240]  axes=245  step_var=0.0019  binary_acc=0.952  gap=0.2007  max_dot=0.0025  (1.8s)
    TOP:  æ¢µ(0.11) | Company(0.10) | Î¼(0.10) | æĶ¾æĿ¾(0.10) | Method(0.10) | è¯·éĹ®(0.10) | serÃ¡(0.10) | ÑĥÑģÐ»Ð¾Ð²(0.09)
    BOT:  Äĳá»ģ(-0.11) | iod(-0.11) | åĪ°æľŁ(-0.10) | åłµ(-0.10) | èº²(-0.10) | éĢĻ(-0.10) | ä¸Ģåĳ¨(-0.10) | æľīå¾Īå¤§çļĦ(-0.10)
    ACCEPTED as axis_245  cumulative_var=0.4542

  [ 241]  axes=246  step_var=0.0018  binary_acc=0.968  gap=0.1995  max_dot=0.0082  (1.9s)
    TOP:  ëıĦ(0.11) | ä¿®é¥°(0.11) | é«ĺ(0.11) | çĽ´(0.11) | åĽ¾çīĩæĿ¥æºĲ(0.10) | å¤ļä½ĻçļĦ(0.10) | è¿Ĳè´¹(0.10) | niveau(0.10)
    BOT:  æĪĲä¸ºä¸Ģä¸ª(-0.11) | çĶŁäº§çĶŁæ´»(-0.10) | algunas(-0.10) | æ©Ļ(-0.10) | ä¸ļåĬ¡(-0.10) | -mini(-0.10) | semble(-0.09) | carving(-0.09)
    ACCEPTED as axis_246  cumulative_var=0.4552

  [ 242]  axes=247  step_var=0.0019  binary_acc=0.961  gap=0.2031  max_dot=0.0016  (1.9s)
    TOP:  NULL(0.11) | Recogn(0.10) | å¹²(0.10) | somebody(0.10) | llev(0.09) | hires(0.09) | oneself(0.09) | ")).(0.09)
    BOT:  äºĶå¤§(-0.10) | speakers(-0.10) | äº¤æĺĵ(-0.10) | åĲĦå¤§(-0.10) | à¸§à¸¢(-0.10) | åĮĿ(-0.10) | Medic(-0.10) | çĻ½æĸĳ(-0.10)
    ACCEPTED as axis_247  cumulative_var=0.4562

  [ 243]  axes=248  step_var=0.0019  binary_acc=1.000  gap=0.1978  max_dot=0.0026  (1.9s)
    TOP:  kotlinx(0.12) | éķ¿å®ī(0.11) | èħ°éĥ¨(0.10) | enerative(0.10) | èĥĮåĮħ(0.10) | çĶŁæ´»æ°´å¹³(0.10) | anything(0.09) | åĲĥçļĦ(0.09)
    BOT:  åĲĮèĥŀ(-0.12) | åľŃ(-0.10) | åĵ²(-0.10) | åĨ³(-0.10) | æķħéļľ(-0.10) | çĲ¦(-0.10) | åŁºæľ¬(-0.09) | atta(-0.09)
    ACCEPTED as axis_248  cumulative_var=0.4573

  [ 244]  axes=249  step_var=0.0020  binary_acc=0.987  gap=0.2019  max_dot=0.0018  (1.9s)
    TOP:  çĪ¶(0.11) | ç¨³(0.11) | æĻĭåįĩ(0.10) | çĶµåŃĲäº§åĵģ(0.10) | é»ĳèī²(0.10) | ä½ıå®¿(0.10) | FRE(0.10) | ãģĹãģ£ãģĭãĤĬ(0.10)
    BOT:  Ð¿Ð¾ÑĩÐµÐ¼Ñĥ(-0.10) | åıĳçĶŁäºĨ(-0.10) | æĢ§èĥ½(-0.10) | æĭīå¼Ģ(-0.10) | ĉcurrent(-0.10) | timer(-0.10) | trium(-0.10) | è°ĥèĬĤ(-0.09)
    ACCEPTED as axis_249  cumulative_var=0.4583

  [ 245]  axes=250  step_var=0.0019  binary_acc=0.986  gap=0.1965  max_dot=0.0018  (1.8s)
    TOP:  æľĪåĪĿ(0.11) | æľįè£ħ(0.11) | ä¸Ģä¾§(0.11) | Ronald(0.10) | åĬ¡å·¥(0.10) | Isle(0.10) | Ä±lÄ±(0.10) | æ°´è´¨(0.10)
    BOT:  ãģ¸ãģ®(-0.10) | èĩ´çĶµ(-0.10) | extr(-0.09) | rxjs(-0.09) | ê·¸(-0.09) | proven(-0.09) | aggress(-0.09) | à¸·à¸Ńà¸Ļ(-0.09)
    ACCEPTED as axis_250  cumulative_var=0.4593

  [ 246]  axes=251  step_var=0.0019  binary_acc=0.999  gap=0.2032  max_dot=0.0013  (1.8s)
    TOP:  èµ°è¿ĩ(0.11) | servant(0.10) | éĵĤ(0.10) | æµģçķħ(0.10) | WX(0.10) | åĳ¼ãģ°(0.10) | cÃ²n(0.10) | Ð¾Ð±ÑĢÐ°Ð¶(0.10)
    BOT:  æ¯ıå¤©éĥ½(-0.11) | nutritious(-0.11) | (-0.11) | join(-0.10) | EU(-0.10) | æ¯ıæ¬¡éĥ½(-0.10) | Synopsis(-0.10) | everywhere(-0.09)
    ACCEPTED as axis_251  cumulative_var=0.4604

  [ 247]  axes=252  step_var=0.0019  binary_acc=0.975  gap=0.1986  max_dot=0.0028  (1.8s)
    TOP:  éĺİ(0.10) | è´µéĩĳå±ŀ(0.10) | ì°¨(0.10) | ä½³(0.10) | äºĶæĺŁ(0.10) | åĽĽæĸ¹(0.10) | attravers(0.10) | tor(0.10)
    BOT:  æŃ£ç¡®çļĦ(-0.12) | æĥ³è±¡åĬĽ(-0.11) | Foto(-0.10) | opol(-0.09) | ÙĪØ¥(-0.09) | æĬĬä½ł(-0.09) | æĸ½å·¥çİ°åľº(-0.09) | æ¼ıæ°´(-0.09)
    ACCEPTED as axis_252  cumulative_var=0.4614

  [ 248]  axes=253  step_var=0.0019  binary_acc=0.991  gap=0.1974  max_dot=0.0038  (1.8s)
    TOP:  ç»ŃèĪª(0.11) | Jenn(0.10) | è®®ä¼ļ(0.10) | solve(0.10) | æĹłåı¯(0.10) | è§£åĨ³éĹ®é¢ĺ(0.10) | Guests(0.09) | calculator(0.09)
    BOT:  æ¯ıæ¬¡(-0.12) | Ð½Ð°Ð·(-0.12) | Normally(-0.11) | éĥ½çŁ¥éģĵ(-0.11) | ]((-0.11) | æ¯ıå¤©(-0.11) | ?151643(-0.10) | battlefield(-0.10)
    ACCEPTED as axis_253  cumulative_var=0.4624

  [ 249]  axes=254  step_var=0.0019  binary_acc=1.000  gap=0.1990  max_dot=0.0019  (1.8s)
    TOP:  ä¹ĭéĹ´(0.12) | å¹´åºķ(0.11) | dysfunctional(0.10) | ä¹ĭéĸĵ(0.10) | å®¶æĹı(0.10) | Ø¨Ø¥(0.10) | moderator(0.10) | ä¸ĵäºº(0.10)
    BOT:  æĢ¥éľĢ(-0.13) | antal(-0.12) | heraus(-0.11) | really(-0.11) | çļĵ(-0.11) | WO(-0.10) | æ·ĩ(-0.10) | ÐºÑĥÐ¿(-0.10)
    ACCEPTED as axis_254  cumulative_var=0.4635

  [ 250]  axes=255  step_var=0.0020  binary_acc=1.000  gap=0.2030  max_dot=0.0027  (1.8s)
    TOP:  herald(0.11) | é¢ĳé¢ĳ(0.11) | omed(0.10) | ×ľ×Ķ(0.10) | æ¹ĺ(0.10) | Could(0.10) | ä¿¡çĶ¨åį¡(0.10) | à¸´à¸ģ(0.10)
    BOT:  versus(-0.11) | åºŁ(-0.11) | ems(-0.10) | notwithstanding(-0.10) | åĮĸçŁ³(-0.10) | èĩªåĪ¶(-0.10) | Voc(-0.10) | three(-0.09)
    ACCEPTED as axis_255  cumulative_var=0.4645

  [ 251]  axes=256  step_var=0.0019  binary_acc=0.983  gap=0.1980  max_dot=0.0022  (1.8s)
    TOP:  bodies(0.11) | PowerPoint(0.10) | vocabulary(0.10) | è¾¾å°Ķ(0.09) | çīĩæ®µ(0.09) | è´µæĹı(0.09) | æĤ¦(0.09) | è¯¾ç¨ĭ(0.09)
    BOT:  äºķ(-0.11) | æľ±(-0.11) | éĿĴå±±(-0.10) | å¸¸åĬ¡åī¯(-0.10) | ç¬¬ä¸ĢçĻ¾(-0.10) | çĪ²(-0.10) | èıĬèĬ±(-0.10) | å¯»æ±Ĥ(-0.10)
    ACCEPTED as axis_256  cumulative_var=0.4655

  [ 252]  axes=257  step_var=0.0020  binary_acc=0.993  gap=0.2078  max_dot=0.0053  (1.9s)
    TOP:  recuper(0.12) | probable(0.10) | broader(0.10) | æī©å±ķ(0.10) | éľģ(0.10) | ä¸»åĬ¨(0.10) | åĬ©æĶ»(0.10) | å¤§è·Į(0.10)
    BOT:  omet(-0.11) | åħ¬å®ī(-0.11) | incarcerated(-0.11) | aven(-0.10) | ì(-0.10) | çĶ¨æīĭ(-0.10) | .beh(-0.09) | enc(-0.09)
    ACCEPTED as axis_257  cumulative_var=0.4666

  [ 253]  axes=258  step_var=0.0020  binary_acc=0.990  gap=0.2044  max_dot=0.0059  (1.8s)
    TOP:  æŀ¯(0.11) | åĸī(0.10) | çĶ¨é¤Ĳ(0.10) | ???(0.10) | ???(0.10) | divine(0.10) | çĥ¹(0.10) | stern(0.10)
    BOT:  ä¾ĿæĹ§(-0.11) | é«ĺè¾¾(-0.11) | Bros(-0.10) | ä¸ºæ°ĳ(-0.10) | morb(-0.10) | æķĻæİĪ(-0.10) | ÑĢÐ¾Ð¼Ð°Ð½(-0.09) | è´¢æĬ¥(-0.09)
    ACCEPTED as axis_258  cumulative_var=0.4676

  [ 254]  axes=259  step_var=0.0019  binary_acc=0.982  gap=0.2004  max_dot=0.0042  (1.9s)
    TOP:  æĸ°éĹ»ç½ĳ(0.10) | overwhelmingly(0.10) | æĿ¿æĿĲ(0.10) | Ð»Ð¸Ð½(0.09) | wp(0.09) | é«ĺäºİ(0.09) | æ²¡æľīæĥ³åĪ°(0.09) | çºº(0.09)
    BOT:  éĢĢå½¹(-0.11) | íĻ(-0.11) | snd(-0.11) | gev(-0.10) | Minor(-0.10) | åĴ½åĸī(-0.10) | åĶ¤éĨĴ(-0.10) | ä¹°(-0.10)
    ACCEPTED as axis_259  cumulative_var=0.4686

  [ 255]  axes=260  step_var=0.0019  binary_acc=0.995  gap=0.1964  max_dot=0.0040  (1.9s)
    TOP:  éĢłçº¸(0.11) | sole(0.10) | ä¸ŃæľĢ(0.10) | éªŀ(0.10) | å°ıé¢Ŀè´·æ¬¾(0.09) | ãĤĴãģĻãĤĭ(0.09) | éļ¾å¿ĺ(0.09) | ìĹĲìĦľ(0.09)
    BOT:  flamm(-0.11) | ä»ĸå¦Ī(-0.10) | æĹłè®ºå¦Ĥä½ķ(-0.10) | ãģĤ(-0.10) | ç¼ĸåī§(-0.10) | åĪ°æĹ¶åĢĻ(-0.10) | æ±Ĳ(-0.10) | ç®±åŃĲ(-0.09)
    ACCEPTED as axis_260  cumulative_var=0.4696

  [ 256]  axes=261  step_var=0.0019  binary_acc=0.989  gap=0.1952  max_dot=0.0023  (1.8s)
    TOP:  åĵģåĳ³(0.11) | harmonic(0.11) | niveau(0.10) | AV(0.10) | íĺ¸(0.10) | iveau(0.10) | Ø¹Ø¯(0.10) | ä¸įä½Ĩ(0.09)
    BOT:  emp(-0.10) | ä»Ģä¹Īéĥ½ä¸į(-0.09) | Emp(-0.09) | RSS(-0.09) | æŃĮè¯į(-0.09) | Documents(-0.09) | æİ¨èįĲ(-0.09) | sidewalk(-0.09)
    ACCEPTED as axis_261  cumulative_var=0.4706

  [ 257]  axes=262  step_var=0.0019  binary_acc=0.993  gap=0.1972  max_dot=0.0031  (1.9s)
    TOP:  ä¸Ńå°ıåŀĭ(0.12) | bs(0.10) | æī¿åıĹ(0.10) | reservoir(0.10) | Intermediate(0.10) | åŃĻæĤŁç©º(0.09) | åħ¥åı£(0.09) | Ð¿ÑĢÑıÐ¼Ð¾(0.09)
    BOT:  çĶµåĬ¨è½¦(-0.11) | à¸Ńà¸ļ(-0.10) | çĸ«(-0.10) | æĻĸ(-0.10) | intent(-0.10) | åħ¬ç§¯éĩĳ(-0.10) | å®īå¿ĥ(-0.10) | perfume(-0.10)
    ACCEPTED as axis_262  cumulative_var=0.4716

  [ 258]  axes=263  step_var=0.0019  binary_acc=0.991  gap=0.1968  max_dot=0.0008  (1.9s)
    TOP:  Syracuse(0.10) | éĢĲæ¸Ĳ(0.10) | ç½ĳç»ľæ¸¸æĪı(0.10) | admits(0.09) | codile(0.09) | Grammy(0.09) | ä½łåı¯ä»¥(0.09) | éĩį(0.09)
    BOT:  æĮ¨(-0.11) | Ð°ÑĨÐ¸Ð¾Ð½(-0.10) | å½ķçĶ¨(-0.10) | åĨĻäºĨ(-0.10) | pictures(-0.10) | éĢłä»·(-0.09) | éĤ£ä¸Ģ(-0.09) | ëĤ(-0.09)
    ACCEPTED as axis_263  cumulative_var=0.4726

  [ 259]  axes=264  step_var=0.0019  binary_acc=0.981  gap=0.1969  max_dot=0.0014  (1.9s)
    TOP:  recurring(0.10) | åĵªåĦ¿(0.10) | Thursday(0.10) | ĉprintf(0.10) | éķĩæĶ¿åºľ(0.10) | pp(0.10) | Ě(0.10) | [f(0.10)
    BOT:  èĢģå®ŀ(-0.11) | åħīéĺ´(-0.10) | åŁĶ(-0.10) | canal(-0.10) | br(-0.10) | éģĵçĲĨ(-0.10) | èĤ¡ä»·(-0.09) | æ²³æ°´(-0.09)
    ACCEPTED as axis_264  cumulative_var=0.4736

  [ 260]  axes=265  step_var=0.0019  binary_acc=0.996  gap=0.1969  max_dot=0.0043  (1.8s)
    TOP:  ÑĩÐµÐ¼(0.10) | uctive(0.10) | surveyed(0.10) | Ð¢ÐµÐ¼(0.09) | europÃ©enne(0.09) | à¹Īà¸Ń(0.09) | æµĩ(0.09) | æ²¡äºĨ(0.09)
    BOT:  [...,(-0.11) | è£ħå¤ĩ(-0.11) | åİ¿å§Ķ(-0.10) | ç¾¤ä¼Ĺ(-0.10) | ä¸įè¯¥(-0.10) | æľĢåħ·(-0.10) | æĲŃè½½(-0.10) | startling(-0.10)
    ACCEPTED as axis_265  cumulative_var=0.4746

  [ 261]  axes=266  step_var=0.0019  binary_acc=0.983  gap=0.1989  max_dot=0.0032  (1.8s)
    TOP:  Num(0.11) | åİŁèĳĹ(0.10) | æ¹Ľ(0.10) | nationalist(0.10) | IRA(0.10) | è®¤è¯Ĩ(0.10) | æł·å¼ı(0.10) | [t(0.10)
    BOT:  andel(-0.11) | éľľ(-0.10) | æİĪæĿĥ(-0.10) | çļĦéĩįçĤ¹(-0.10) | å¤§åĬĽ(-0.10) | actionable(-0.10) | mid(-0.09) | èĬ¬(-0.09)
    ACCEPTED as axis_266  cumulative_var=0.4756

  [ 262]  axes=267  step_var=0.0020  binary_acc=0.965  gap=0.2026  max_dot=0.0025  (1.8s)
    TOP:  reporters(0.11) | historians(0.10) | suc(0.10) | çĺĻ(0.10) | çĽĳäºĭ(0.10) | çĹħçĲĨ(0.10) | æĦŁçŁ¥(0.10) | ä¾µåįł(0.10)
    BOT:  Don(-0.11) | perfect(-0.11) | quotes(-0.10) | åħŃä¸ªæľĪ(-0.10) | -right(-0.10) | muff(-0.10) | åĽ½åĨħå¤ĸ(-0.09) | èį¯åĵģ(-0.09)
    ACCEPTED as axis_267  cumulative_var=0.4766

  [ 263]  axes=268  step_var=0.0018  binary_acc=0.994  gap=0.1920  max_dot=0.0118  (1.9s)
    TOP:  ä¾ĽçĶµ(0.11) | åı¬åĶ¤(0.11) | çİĩä¸º(0.11) | æŃ¤ç±»(0.11) | çķĻåŃĺ(0.11) | åħ¥åľº(0.10) | æł½åŁ¹(0.10) | çºªå¿µ(0.10)
    BOT:  æĸ°æµª(-0.10) | tired(-0.10) | ï¬ģ(-0.10) | fi(-0.10) | å³ª(-0.09) | afternoon(-0.09) | è¯įè¯Ń(-0.09) | without(-0.09)
    ACCEPTED as axis_268  cumulative_var=0.4776

  [ 264]  axes=269  step_var=0.0019  binary_acc=0.986  gap=0.1991  max_dot=0.0031  (1.9s)
    TOP:  éĩįéĩį(0.12) | çĽ¸åĲĮ(0.11) | lain(0.10) | duties(0.09) | æ²ī(0.09) | #================================================================(0.09) | glasses(0.09) | çŃīçŃī(0.09)
    BOT:  åįĩæ¸©(-0.11) | æĬ¥ä»·(-0.10) | cuando(-0.10) | MJ(-0.10) | little(-0.10) | éĽī(-0.10) | å¹¶åıĳ(-0.10) | è¿ĻåĦ¿(-0.10)
    ACCEPTED as axis_269  cumulative_var=0.4785

  [ 265]  axes=270  step_var=0.0019  binary_acc=0.993  gap=0.1990  max_dot=0.0008  (1.9s)
    TOP:  ;&(0.10) | åľĨå½¢(0.10) | prolific(0.10) | åħħå½ĵ(0.10) | ctypes(0.10) | å®ª(0.10) | Ø¢(0.09) | medications(0.09)
    BOT:  least(-0.13) | çĽ¸å¹²(-0.11) | æľīåħ³(-0.10) | çĥŃéĶĢ(-0.10) | matter(-0.10) | åħį(-0.10) | çļ®éĿ©(-0.10) | precisely(-0.10)
    ACCEPTED as axis_270  cumulative_var=0.4795

  [ 266]  axes=271  step_var=0.0019  binary_acc=0.987  gap=0.1960  max_dot=0.0020  (1.9s)
    TOP:  aly(0.11) | fut(0.11) | é¢Ĺç²Ĵ(0.10) | afa(0.10) | æ¼Ĥ(0.09) | å¹´åĲİ(0.09) | arily(0.09) | åħ¬å¼Ģåıĳ(0.09)
    BOT:  ä¸īä»£(-0.11) | quad(-0.10) | æľ¬å±Ĭ(-0.10) | æĸ°ä¸Ģä»£(-0.10) | æķ°çĽ®(-0.10) | whats(-0.10) | èµ°ä¸ĭåİ»(-0.10) | åįģä½³(-0.09)
    ACCEPTED as axis_271  cumulative_var=0.4805

  [ 267]  axes=272  step_var=0.0019  binary_acc=1.000  gap=0.1974  max_dot=0.0026  (1.9s)
    TOP:  xi(0.10) | cool(0.10) | åĢŁ(0.10) | èĩªèº«çļĦ(0.09) | borrowing(0.09) | èĩªèº«(0.09) | registered(0.09) | exits(0.09)
    BOT:  seldom(-0.11) | çıŃç»Ħ(-0.10) | à¥ĩ(-0.10) | today(-0.10) | éĴ¢ç®¡(-0.10) | å¾Ĺèµ·(-0.09) | å¯¼å¼¹(-0.09) | '|(-0.09)
    ACCEPTED as axis_272  cumulative_var=0.4815

  [ 268]  axes=273  step_var=0.0018  binary_acc=0.989  gap=0.1922  max_dot=0.0009  (1.8s)
    TOP:  ä»Ļå¥³(0.11) | æ´¾(0.11) | åĪ©çī©æµ¦(0.10) | washing(0.10) | æĸ°ä¸ĸçºª(0.10) | ="{{(0.10) | æµ£(0.10) | VIEW(0.10)
    BOT:  ãģ§(-0.10) | åºĶçĶ¨åľºæĻ¯(-0.10) | Sus(-0.10) | æĢ»(-0.10) | Ø¹(-0.09) | ãĢįãģ¨(-0.09) | Oct(-0.09) | generates(-0.09)
    ACCEPTED as axis_273  cumulative_var=0.4824

  [ 269]  axes=274  step_var=0.0019  binary_acc=0.985  gap=0.1943  max_dot=0.0010  (1.8s)
    TOP:  åı£ç½©(0.10) | åħ¥åľº(0.10) | çīĽå¸Ĥ(0.10) | æĸ°èĤ¡(0.09) | ç¤¼çī©(0.09) | âĢĶwhich(0.09) | çĸ«èĭĹ(0.09) | believes(0.09)
    BOT:  åį³åı¯(-0.11) | è¿ĺä¸įæĺ¯(-0.10) | æ²¹çĥŁ(-0.10) | å¯ĵæĦı(-0.10) | åįıè°ĥ(-0.10) | æ±(-0.10) | åħ¶å®ŀå°±æĺ¯(-0.10) | æĲŃå»º(-0.09)
    ACCEPTED as axis_274  cumulative_var=0.4834

  [ 270]  axes=275  step_var=0.0019  binary_acc=0.975  gap=0.1978  max_dot=0.0035  (1.8s)
    TOP:  ä¸»è¦ģåĮħæĭ¬(0.11) | Canton(0.11) | è®²ç©¶(0.10) | NFC(0.10) | GM(0.10) | Ø³Øª(0.10) | STL(0.10) | noticed(0.09)
    BOT:  Environmental(-0.10) | æŀ£(-0.10) | åħ¬åı¸åľ¨(-0.10) | sourced(-0.10) | environmental(-0.10) | industry(-0.10) | sequ(-0.10) | weighs(-0.10)
    ACCEPTED as axis_275  cumulative_var=0.4844

  [ 271]  axes=276  step_var=0.0019  binary_acc=0.997  gap=0.1925  max_dot=0.0023  (1.9s)
    TOP:  æĹ©çĤ¹(0.11) | çĶ¨èĩªå·±çļĦ(0.10) | ä»Ĭå¤©å°ıç¼ĸ(0.10) | éĻĲæľŁ(0.10) | camouflage(0.10) | åĩłçĤ¹(0.10) | ç½ĳè´·(0.10) | æºľ(0.09)
    BOT:  èĢ³æľº(-0.10) | èĨĽ(-0.09) | door(-0.09) | quÃ©(-0.09) | æĹģ(-0.09) | _false(-0.09) | aside(-0.09) | -front(-0.09)
    ACCEPTED as axis_276  cumulative_var=0.4854

  [ 272]  axes=277  step_var=0.0018  binary_acc=0.997  gap=0.1897  max_dot=0.0025  (1.8s)
    TOP:  ä¸įäºĪ(0.11) | æĪĳçĪ±ä½ł(0.11) | vom(0.10) | æĹłäºº(0.10) | äºĨä¸Ģåı£(0.09) | æ°¸ä¸į(0.09) | darÃ¼ber(0.09) | counterfeit(0.09)
    BOT:  cá»Ļng(-0.11) | quizÃ¡(-0.10) | metam(-0.09) | åĨ¬å¤©(-0.09) | èµĶä»ĺ(-0.09) | wides(-0.09) | jetzt(-0.09) | èĴľ(-0.09)
    ACCEPTED as axis_277  cumulative_var=0.4863

  [ 273]  axes=278  step_var=0.0019  binary_acc=0.987  gap=0.1936  max_dot=0.0026  (1.9s)
    TOP:  µ(0.11) | markt(0.10) | æĶĺ(0.10) | ÑĩÑĥÐ²(0.10) | rather(0.10) | TRUE(0.10) | heid(0.09) | éĤ£ä»½(0.09)
    BOT:  ä¿¡æģ¯åħ¬å¼Ģ(-0.12) | parchment(-0.11) | onPress(-0.10) | ç½ĳæ°ĳ(-0.10) | iform(-0.10) | impro(-0.09) | à¸«à¸Ļ(-0.09) | à¹ĥà¸ª(-0.09)
    ACCEPTED as axis_278  cumulative_var=0.4873

  [ 274]  axes=279  step_var=0.0019  binary_acc=0.996  gap=0.1930  max_dot=0.0030  (1.9s)
    TOP:  Ã©tÃ©(0.12) | èµĦè´¨(0.11) | åħĪå¤©(0.11) | {/*(0.10) | Como(0.10) | ood(0.10) | åİŁæĿĲæĸĻ(0.10) | Good(0.10)
    BOT:  Ø°(-0.10) | ÄĳÆ°a(-0.10) | æĶ¾çľ¼(-0.10) | äºĴèģĶ(-0.10) | åħ¨éĿ¢æİ¨è¿Ľ(-0.10) | Satoshi(-0.10) | åį(-0.10) | çĽĳè§Ĩ(-0.10)
    ACCEPTED as axis_279  cumulative_var=0.4883

  [ 275]  axes=280  step_var=0.0019  binary_acc=0.980  gap=0.1939  max_dot=0.0063  (1.8s)
    TOP:  å·¥ä½ľå®¤(0.11) | æĪĳçİ°åľ¨(0.11) | xo(0.11) | è½¦åºĵ(0.11) | æĹ¥å¸¸å·¥ä½ľ(0.11) | ç¾¤ä½ĵ(0.10) | æĹ¥åĨħ(0.10) | åħ¶ä¸Ńä¸Ģä¸ª(0.10)
    BOT:  sequencing(-0.11) | Page(-0.10) | escape(-0.09) | Qualcomm(-0.09) | è£ħåį¸(-0.09) | ØµÙģ(-0.09) | comprehend(-0.09) | æĭĽèĤ¡(-0.09)
    ACCEPTED as axis_280  cumulative_var=0.4892

  [ 276]  axes=281  step_var=0.0019  binary_acc=0.995  gap=0.1927  max_dot=0.0016  (1.9s)
    TOP:  stress(0.11) | item(0.10) | ç³»åĪĹäº§åĵģ(0.10) | ä¸ŃåĽ½åĽ½éĻħ(0.09) | ç½ĳå°ıç¼ĸ(0.09) | ********************************************************(0.09) | å½ĵæĪĳä»¬(0.09) | ä¿¡æģ¯ç³»ç»Ł(0.09)
    BOT:  åĪ¶åĨ·(-0.12) | ìĸ(-0.11) | æī¿åĮħ(-0.10) | ä¸ªåĪ«(-0.10) | äºķ(-0.10) | çŀĦåĩĨ(-0.10) | thousand(-0.10) | æ¤į(-0.10)
    ACCEPTED as axis_281  cumulative_var=0.4902

  [ 277]  axes=282  step_var=0.0020  binary_acc=0.989  gap=0.1994  max_dot=0.0050  (1.9s)
    TOP:  &&(0.11) | puppy(0.11) | é©¬ä¸Ĭ(0.10) | ISBN(0.10) | ìľ¼ë©´(0.09) | dime(0.09) | æľīç½ĳåıĭ(0.09) | thá»ĭ(0.09)
    BOT:  Py(-0.11) | ļ(-0.10) | hat(-0.10) | è¾£(-0.10) | ç»Łè®¡(-0.10) | è§ī(-0.10) | åģļåĩº(-0.10) | help(-0.10)
    ACCEPTED as axis_282  cumulative_var=0.4912

  [ 278]  axes=283  step_var=0.0018  binary_acc=0.998  gap=0.1872  max_dot=0.0082  (1.8s)
    TOP:  æĬ½åĩº(0.11) | å§¿æĢģ(0.10) | shops(0.10) | "/(0.10) | åĳ¦(0.09) | ä¹Ī(0.09) | è§Ħæ¨¡åĮĸ(0.09) | éĶĢåĶ®æĶ¶åħ¥(0.09)
    BOT:  cred(-0.10) | æĬĹçĻĮ(-0.10) | ÑīÐ¸ÑĤ(-0.10) | åĽłåľ°(-0.10) | where(-0.10) | ä¸Ĭæ¦ľ(-0.10) | ä»ĸå°±(-0.10) | è¿Ļå¼ł(-0.09)
    ACCEPTED as axis_283  cumulative_var=0.4921

  [ 279]  axes=284  step_var=0.0019  binary_acc=0.979  gap=0.1941  max_dot=0.0017  (1.9s)
    TOP:  isy(0.12) | upkeep(0.11) | Fee(0.11) | Fee(0.10) | footing(0.10) | $this(0.10) | prominence(0.09) | squir(0.09)
    BOT:  ademÃ¡s(-0.11) | ÂŃt(-0.11) | Ĳľ(-0.10) | èĨ»(-0.10) | æħĪåĸĦ(-0.10) | åĸľæĤ¦(-0.10) | ä¾ł(-0.10) | å½±è§Ĩ(-0.09)
    ACCEPTED as axis_284  cumulative_var=0.4931

  [ 280]  axes=285  step_var=0.0018  binary_acc=0.982  gap=0.1897  max_dot=0.0046  (1.9s)
    TOP:  obtaining(0.10) | çİ¯å¢ĥ(0.10) | çĽ¸å£°(0.09) | ä¸»é¡µ(0.09) | è®®ä¼ļ(0.09) | æŀĦéĢł(0.09) | Southampton(0.09) | .-(0.09)
    BOT:  ç´§çĽ¯(-0.11) | è¿ĲçĶ¨(-0.11) | MetaData(-0.11) | çħ§æĸĻ(-0.10) | emploi(-0.10) | å¤ªéĺ³èĥ½(-0.09) | åĿļå®ļä¸įç§»(-0.09) | jabi(-0.09)
    ACCEPTED as axis_285  cumulative_var=0.4940

  [ 281]  axes=286  step_var=0.0019  binary_acc=0.990  gap=0.2008  max_dot=0.0016  (1.9s)
    TOP:  fine(0.11) | åĿĲæłĩ(0.10) | ç¼ķ(0.10) | åĴĮä¸ªäºº(0.10) | åıįæĺł(0.10) | mobil(0.09) | åĿļå®ļ(0.09) | ently(0.09)
    BOT:  @interface(-0.11) | Suitable(-0.10) | axe(-0.09) | licensing(-0.09) | é£²(-0.09) | Ø¹ÙĨØ¯(-0.09) | Comes(-0.09) | airstrikes(-0.09)
    ACCEPTED as axis_286  cumulative_var=0.4950

  [ 282]  axes=287  step_var=0.0019  binary_acc=0.996  gap=0.1889  max_dot=0.0020  (1.8s)
    TOP:  æŀ³(0.09) | ä¾ĭå¤ĸ(0.09) | Ã©tait(0.09) | ä¸įæŃ¢(0.09) | tout(0.09) | éĢĤ(0.09) | èĤ¡æĮĩ(0.09) | SY(0.09)
    BOT:  åĢº(-0.11) | æĪĳä»¬åı¯ä»¥(-0.10) | atl(-0.10) | Canon(-0.10) | æĪĳä»¬ä¼ļ(-0.10) | liters(-0.09) | abs(-0.09) | å®£ä¼łçīĩ(-0.09)
    ACCEPTED as axis_287  cumulative_var=0.4959

  [ 283]  axes=288  step_var=0.0019  binary_acc=0.998  gap=0.1988  max_dot=0.0061  (1.8s)
    TOP:  Chair(0.11) | grund(0.10) | å½ĵçĦ¶(0.10) | éªĳ(0.10) | ä¸ĢçĤ¹ä¹Łä¸į(0.09) | respir(0.09) | score(0.09) | çĪ±å¥ĩèīº(0.09)
    BOT:  riÃ³(-0.11) | {}(-0.10) | ªĮ(-0.10) | actionable(-0.10) | æ±ŀ(-0.10) | idy(-0.10) | ä¸įåŃķ(-0.09) | ®(-0.09)
    ACCEPTED as axis_288  cumulative_var=0.4969

  [ 284]  axes=289  step_var=0.0019  binary_acc=0.982  gap=0.1932  max_dot=0.0036  (1.9s)
    TOP:  æľīå¾ħ(0.11) | Scots(0.11) | ìĦ±(0.10) | ä¸įè¦ģå¤ª(0.10) | .TabIndex(0.10) | èŀįåĲĪåıĳå±ķ(0.10) | evapor(0.10) | æ·»(0.10)
    BOT:  ç¾½æ¯ĽçĲĥ(-0.10) | nil(-0.10) | Ð±ÑĭÐ»Ð¸(-0.10) | è®¿(-0.09) | depart(-0.09) | è¯£(-0.09) | quienes(-0.09) | isot(-0.09)
    ACCEPTED as axis_289  cumulative_var=0.4978

  [ 285]  axes=290  step_var=0.0019  binary_acc=0.993  gap=0.1933  max_dot=0.0040  (1.9s)
    TOP:  åıĳç¥¨(0.10) | éĩįæĸ°(0.10) | stigma(0.09) | ä¾Ħ(0.09) | enamel(0.09) | Indexes(0.09) | ë©(0.09) | çĶµæ°Ķ(0.09)
    BOT:  tailor(-0.10) | twelve(-0.10) | æ¶¨åģľ(-0.09) | âķĲ(-0.09) | -than(-0.09) | åħ¨åŁŁæĹħæ¸¸(-0.09) | çļĦä¿¡ä»»(-0.09) | virt(-0.09)
    ACCEPTED as axis_290  cumulative_var=0.4988

  [ 286]  axes=291  step_var=0.0019  binary_acc=0.988  gap=0.1898  max_dot=0.0040  (1.8s)
    TOP:  cle(0.12) | èĹ¿(0.10) | ä¸Ģé¦ĸ(0.10) | èķŀ(0.09) | Äĳang(0.09) | construed(0.09) | å³¦(0.09) | çĽ®åīįå·²(0.09)
    BOT:  UnityEngine(-0.12) | çĽ´åĪ°(-0.11) | è·ĳåĪ°(-0.10) | å½ĵæĹ¶(-0.10) | ä½łè¯´(-0.10) | ä¸ĭçļĦ(-0.10) | stopping(-0.10) | çº¦ä¸º(-0.10)
    ACCEPTED as axis_291  cumulative_var=0.4997

  [ 287]  axes=292  step_var=0.0019  binary_acc=0.992  gap=0.1925  max_dot=0.0019  (1.8s)
    TOP:  æĹ¥ä¸ĭåįĪ(0.12) | æ¤įçī©(0.11) | èĲ½åı¶(0.11) | utron(0.10) | æĭīä¸ģ(0.10) | æĵĤ(0.10) | Ð¿ÐµÑĢ(0.10) | åľª(0.10)
    BOT:  éŃĶæ³ķ(-0.11) | >)(-0.10) | åı£ä¸Ń(-0.10) | é¸¡èĤī(-0.10) | .Com(-0.10) | >>>(-0.10) | æĥ³è¦ģ(-0.10) | ½Ķ(-0.10)
    ACCEPTED as axis_292  cumulative_var=0.5006

  [ 288]  axes=293  step_var=0.0018  binary_acc=0.999  gap=0.1879  max_dot=0.0018  (1.9s)
    TOP:  ëĵ¤(0.10) | äºĶ(0.10) | urlparse(0.09) | <meta(0.09) | ewater(0.09) | è¾¼(0.09) | èŃ¬å¦Ĥ(0.09) | sadly(0.09)
    BOT:  Ø£ÙĪÙĦ(-0.10) | æīĴ(-0.10) | .aw(-0.10) | å®½(-0.09) | ä¿Ŀè¯ģéĩĳ(-0.09) | ½(-0.09) | rem(-0.09) | critically(-0.09)
    ACCEPTED as axis_293  cumulative_var=0.5016

  [ 289]  axes=294  step_var=0.0019  binary_acc=0.964  gap=0.1948  max_dot=0.0048  (1.9s)
    TOP:  whoever(0.14) | whom(0.13) | overseas(0.11) | staff(0.10) | Whoever(0.10) | who(0.10) | å¸ĮæľĽå¤§å®¶(0.09) | åį³ä½¿(0.09)
    BOT:  æľīè¶£çļĦ(-0.11) | nearly(-0.10) | åħ¨å±Ģ(-0.10) | -=(-0.10) | åħŃä¸ª(-0.10) | ä¸Ģä¸Ģ(-0.09) | çĸĹæ³ķ(-0.09) | ç¬ĳèĦ¸(-0.09)
    ACCEPTED as axis_294  cumulative_var=0.5025

  [ 290]  axes=295  step_var=0.0018  binary_acc=0.992  gap=0.1855  max_dot=0.0014  (1.9s)
    TOP:  éĿł(0.10) | à´(0.10) | ä¾ĿéĿł(0.10) | åıĹ(0.10) | çŁ(0.09) | çĽ¸å·®(0.09) | ç»ĻåĬĽ(0.09) | operates(0.09)
    BOT:  æĪĳä»¬åºĶè¯¥(-0.11) | ä¸Ńåįİ(-0.10) | æ¶Ŀ(-0.10) | ç¬ĳå®¹(-0.10) | least(-0.09) | å¿µå¤´(-0.09) | rotch(-0.09) | åħĭæĢĿä¸»ä¹ī(-0.09)
    ACCEPTED as axis_295  cumulative_var=0.5034

  [ 291]  axes=296  step_var=0.0019  binary_acc=0.984  gap=0.1876  max_dot=0.0042  (1.8s)
    TOP:  ä¼¼ä¹İæĺ¯(0.10) | ..."Ċ(0.09) | åŁ¹åħ»(0.09) | Peyton(0.09) | ...'Ċ(0.09) | >-(0.09) | ÙĪÙħ(0.09) | åī¯æł¡éķ¿(0.09)
    BOT:  Ð¾Ð»ÑĮ(-0.11) | Ð¶ÐµÐ½(-0.10) | æĹ¦(-0.10) | Ð²Ð¾Ð»ÑĮ(-0.10) | èĨĢ(-0.10) | æĸĩæĹħ(-0.10) | èĩªæĪĳ(-0.10) | ë¥´(-0.09)
    ACCEPTED as axis_296  cumulative_var=0.5043

  [ 292]  axes=297  step_var=0.0019  binary_acc=0.996  gap=0.1925  max_dot=0.0021  (1.8s)
    TOP:  çļĦè§ĤçĤ¹(0.09) | ä¹Łèĥ½(0.09) | æĮĩå°ĸ(0.09) | fd(0.09) | nond(0.09) | .SuppressLint(0.09) | owl(0.09) | çľĭæ³ķ(0.09)
    BOT:  æ·±åľ³å¸Ĥ(-0.12) | ][](-0.11) | çºłéĶĻ(-0.10) | åħ¥åľº(-0.10) | ìļĶ(-0.10) | åıĤè°ĭ(-0.10) | è¿Ļä¸¤å¤©(-0.10) | yo(-0.09)
    ACCEPTED as axis_297  cumulative_var=0.5052

  [ 293]  axes=298  step_var=0.0018  binary_acc=0.997  gap=0.1881  max_dot=0.0022  (1.8s)
    TOP:  uation(0.10) | Ð¿Ð¸(0.09) | åĨħå¤ĸ(0.09) | Ank(0.09) | Brass(0.09) | éĻĭ(0.09) | Cottage(0.09) | .skill(0.09)
    BOT:  ç¨ĭå¼ı(-0.11) | å¤©æīį(-0.10) | åİĨæĹ¶(-0.10) | somebody(-0.10) | ä¸į(-0.10) | ]](-0.10) | åĩºåĽ½(-0.10) | ãĥŀ(-0.09)
    ACCEPTED as axis_298  cumulative_var=0.5062

  [ 294]  axes=299  step_var=0.0018  binary_acc=0.989  gap=0.1909  max_dot=0.0032  (1.8s)
    TOP:  Still(0.09) | é«ĺæ¥¼(0.09) | à¸¶(0.09) | taller(0.09) | (it(0.09) | æĢĿç»ª(0.09) | çļĦç²¾ç¥ŀ(0.09) | dies(0.09)
    BOT:  ä¸Ńåħ±(-0.12) | å¼Ģè®¾(-0.11) | çº¿ä¸ĭ(-0.10) | ä¸ĭæ²ī(-0.10) | æİĴéĻ¤(-0.10) | Ø¯(-0.10) | ç¬¬ä¸īæĸ¹(-0.10) | åıĳåĬ¨(-0.10)
    ACCEPTED as axis_299  cumulative_var=0.5071

  [ 295]  axes=300  step_var=0.0019  binary_acc=0.978  gap=0.1915  max_dot=0.0019  (1.8s)
    TOP:  whereby(0.10) | çĶŁäºİ(0.10) | gency(0.10) | å¹´å¤ľ(0.09) | accountability(0.09) | ìĿ¼(0.09) | Yu(0.09) | ä»İåīį(0.09)
    BOT:  èĬ±çº¹(-0.11) | æİ¨éĶĢ(-0.11) | éĥ½éĿŀå¸¸(-0.10) | åħ¬åħ±äº¤éĢļ(-0.10) | é«ĺåİŁ(-0.10) | åķĨéĩı(-0.10) | æ´Ĺæ¾¡(-0.10) | ë§İìĿĢ(-0.10)
    ACCEPTED as axis_300  cumulative_var=0.5080

  [ 296]  axes=301  step_var=0.0018  binary_acc=0.969  gap=0.1887  max_dot=0.0033  (1.8s)
    TOP:  åĩ½(0.11) | å¥¢ä¾Īåĵģ(0.10) | å¥¢åįİ(0.10) | ÂŃ(0.10) | ç¬ĳå£°(0.10) | æĴŃ(0.10) | æ¸´æľĽ(0.09) | exited(0.09)
    BOT:  <title(-0.11) | è½¦ç«Ļ(-0.10) | <header(-0.10) | åĬłæ²¹(-0.10) | åįķä½į(-0.10) | ,ĊĊ(-0.09) | Dash(-0.09) | tiáº¿p(-0.09)
    ACCEPTED as axis_301  cumulative_var=0.5089

  [ 297]  axes=302  step_var=0.0019  binary_acc=0.974  gap=0.1886  max_dot=0.0020  (1.8s)
    TOP:  æĺİå¹´(0.12) | eligible(0.10) | ç¬(0.10) | ä¸ĢçĽ´éĥ½(0.10) | Ð³Ð»Ð°Ð²(0.10) | Monday(0.10) | ain(0.10) | ç»Ł(0.10)
    BOT:  èĦ¯(-0.10) | Ð½ÑıÑĤÑĮ(-0.09) | resse(-0.09) | é©´(-0.09) | èµ°åĩº(-0.09) | anian(-0.09) | Ð´ÑĥÑħÐ¾Ð²(-0.09) | phenotype(-0.09)
    ACCEPTED as axis_302  cumulative_var=0.5098

  [ 298]  axes=303  step_var=0.0019  binary_acc=0.962  gap=0.1925  max_dot=0.0037  (1.8s)
    TOP:  ç»ıçºª(0.11) | ç§Ģ(0.11) | éĺµå®¹(0.10) | corp(0.10) | æ°Ķè´¨(0.09) | å¿«é¤Ĳ(0.09) | court(0.09) | Does(0.09)
    BOT:  ç¬¬äºĮä¸ª(-0.10) | stiffness(-0.10) | aign(-0.10) | åī¥(-0.10) | part(-0.10) | ç¬¬åĽĽå±Ĭ(-0.10) | å¹³æĹ¶(-0.09) | pouch(-0.09)
    ACCEPTED as axis_303  cumulative_var=0.5108

  [ 299]  axes=304  step_var=0.0018  binary_acc=0.993  gap=0.1858  max_dot=0.0095  (1.9s)
    TOP:  COVER(0.10) | åıĽ(0.09) | ØªØ¯(0.09) | GET(0.09) | Ø¹Ø¯Ùħ(0.09) | .fail(0.09) | _once(0.09) | .urls(0.09)
    BOT:  æīİå®ŀæİ¨è¿Ľ(-0.11) | unque(-0.10) | èħº(-0.10) | because(-0.09) | åĬŀäºĭ(-0.09) | ä½İä½į(-0.09) | éĺ²æ±Ľ(-0.09) | æł¹åŁº(-0.09)
    ACCEPTED as axis_304  cumulative_var=0.5116

  [ 300]  axes=305  step_var=0.0018  binary_acc=0.962  gap=0.1858  max_dot=0.0019  (1.8s)
    TOP:  apel(0.10) | Ð¹(0.10) | adge(0.09) | æ³ķå¸Ī(0.09) | çĭ¬èĩª(0.09) | font(0.09) | =None(0.09) | åı¯ä¾Ľ(0.09)
    BOT:  æĤłæĤł(-0.10) | .Collectors(-0.10) | ná»Ļi(-0.10) | è¾¦(-0.10) | æµģçķħ(-0.10) | å°ıé¢Ŀ(-0.09) | rc(-0.09) | ä¸ĩä½Ļ(-0.09)
    ACCEPTED as axis_305  cumulative_var=0.5125

  [ 301]  axes=306  step_var=0.0018  binary_acc=0.995  gap=0.1872  max_dot=0.0048  (1.8s)
    TOP:  å¨¥(0.11) | ç¾İå®¹(0.10) | ando(0.09) | });Ċ(0.09) | æĢ§ä»·æ¯Ķ(0.09) | çľŁåģĩ(0.09) | Ð¾Ð»(0.09) | ually(0.09)
    BOT:  çħ¤æ°Ķ(-0.11) | èİ«è¿ĩäºİ(-0.10) | çķĮéĿ¢(-0.10) | áĢ¡(-0.10) | åı¸é©¬(-0.10) | ä¸įåħģè®¸(-0.09) | .writeInt(-0.09) | çģ«çĥŃ(-0.09)
    ACCEPTED as axis_306  cumulative_var=0.5134

  [ 302]  axes=307  step_var=0.0018  binary_acc=0.994  gap=0.1837  max_dot=0.0039  (1.9s)
    TOP:  æ³¢çī¹(0.10) | ÑģÑĢÐµÐ´(0.10) | ç»¿èī²(0.10) | mentre(0.09) | åĵ¥ä»¬(0.09) | ,%(0.09) | æ³¨å°Ħ(0.09) | .table(0.09)
    BOT:  Provincial(-0.11) | Ð´Ð²Ð°(-0.11) | æĶ¿æĿĥ(-0.09) | é¢Ŀåº¦(-0.09) | åĽŀè´Ń(-0.09) | reserves(-0.09) | éĢīæĭĶ(-0.09) | å®¢è¿Ĳ(-0.09)
    ACCEPTED as axis_307  cumulative_var=0.5143

  [ 303]  axes=308  step_var=0.0019  binary_acc=0.993  gap=0.1841  max_dot=0.0015  (1.8s)
    TOP:  èĬĤçĽ®ä¸Ń(0.11) | ç²ķ(0.10) | åİŁåĽłæĺ¯(0.10) | kh(0.10) | Ð²ÐµÑī(0.10) | è°·çĪ±åĩĮ(0.10) | ÑĦÐµÑģÑģ(0.09) | èµ¶ä¸Ĭ(0.09)
    BOT:  æĢ¥æķĳ(-0.11) | éļıæľº(-0.10) | å®ªæ³ķ(-0.10) | åı(-0.10) | signific(-0.10) | Ð¿ÑĢÐµÐ´ÑģÑĤÐ°Ð²Ð»ÑıÐµÑĤ(-0.09) | ä¸įä¸ĭ(-0.09) | styles(-0.09)
    ACCEPTED as axis_308  cumulative_var=0.5152

  [ 304]  axes=309  step_var=0.0019  binary_acc=0.995  gap=0.1884  max_dot=0.0063  (1.8s)
    TOP:  ÑģÐµÐ¹ÑĩÐ°Ñģ(0.11) | å°±å¥½(0.11) | çĶļèĩ³(0.11) | special(0.10) | ?151643(0.10) | æľĭåıĭåľĪ(0.10) | ĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊ(0.10) | XXXXXXXX(0.09)
    BOT:  éĩİçĶŁ(-0.10) | advisable(-0.10) | necessary(-0.10) | ëĦĺ(-0.10) | æ½ĺ(-0.09) | å½±éŁ³(-0.09) | éĢĥçĶŁ(-0.09) | These(-0.09)
    ACCEPTED as axis_309  cumulative_var=0.5161

  [ 305]  axes=310  step_var=0.0019  binary_acc=0.976  gap=0.1874  max_dot=0.0010  (1.9s)
    TOP:  æľŁæĿĥ(0.10) | ë§Ī(0.10) | èĢĮåİ»(0.10) | å¸ĤåľºéľĢæ±Ĥ(0.10) | Collect(0.10) | mijn(0.10) | åķĥ(0.10) | vrij(0.10)
    BOT:  starred(-0.09) | .int(-0.09) | éļĲçº¦(-0.09) | useState(-0.09) | excellent(-0.09) | ourt(-0.09) | talked(-0.09) | è¿ĩæķı(-0.09)
    ACCEPTED as axis_310  cumulative_var=0.5170

  [ 306]  axes=311  step_var=0.0019  binary_acc=0.996  gap=0.1895  max_dot=0.0029  (1.9s)
    TOP:  ä¸¤æĿ¡(0.11) | æĭįçħ§(0.11) | æĲ¬è¿ģ(0.10) | ListModel(0.10) | çĭ®(0.10) | ä¸Ģé¡¹(0.10) | >>>>(0.10) | çĽĺæ´»(0.10)
    BOT:  å¹´çº§(-0.10) | åĳ¦(-0.10) | ä¸Ģå®ļèĥ½(-0.10) | ä¸Ģå®ļè¦ģ(-0.10) | ('[(-0.10) | ä¸ºä»Ģä¹Īè¦ģ(-0.10) | fixtures(-0.10) | Firewall(-0.09)
    ACCEPTED as axis_311  cumulative_var=0.5179

  [ 307]  axes=312  step_var=0.0018  binary_acc=0.946  gap=0.1857  max_dot=0.0013  (1.8s)
    TOP:  cÃ¡c(0.10) | BEST(0.10) | ä¸ĩäºĭ(0.10) | ä»¥ä¾¿(0.09) | çĶ¨æĪ·æıĲä¾Ľ(0.09) | CÃ¡c(0.09) | åĿĩæľī(0.09) | ä¿¡æģ¯æľįåĬ¡(0.09)
    BOT:  ç¬¬ä¹Ŀ(-0.12) | enough(-0.11) | amount(-0.11) | alist(-0.10) | seventh(-0.10) | coronavirus(-0.10) | ç¬¬äºĶ(-0.10) | ä¸Ģå®ļç¨ĭåº¦(-0.10)
    ACCEPTED as axis_312  cumulative_var=0.5188

  [ 308]  axes=313  step_var=0.0018  binary_acc=0.980  gap=0.1891  max_dot=0.0006  (1.9s)
    TOP:  åĲįå½ķ(0.11) | FROM(0.10) | delight(0.10) | å¤§åħ¨(0.10) | .ts(0.09) | æĢ»é¢Ŀ(0.09) | :=(0.09) | .splice(0.09)
    BOT:  è¾¹éĻħ(-0.10) | è¿Ļç±»(-0.10) | åĨľä½ľçī©(-0.10) | opposite(-0.10) | æ»ļåĬ¨(-0.09) | é«ĺçº§(-0.09) | è·¨çķĮ(-0.09) | ×©×¨(-0.09)
    ACCEPTED as axis_313  cumulative_var=0.5197

  [ 309]  axes=314  step_var=0.0019  binary_acc=0.984  gap=0.1881  max_dot=0.0025  (1.9s)
    TOP:  (new(0.10) | EEG(0.10) | Diff(0.10) | è´ª(0.10) | å¤ļå¹´(0.10) | æĤ²(0.09) | äºĮæ°§åĮĸ(0.09) | (columns(0.09)
    BOT:  .startswith(-0.11) | qw(-0.11) | Costco(-0.11) | normal(-0.10) | Normal(-0.09) | èĴĮ(-0.09) | ìłľ(-0.09) | xáº¿p(-0.09)
    ACCEPTED as axis_314  cumulative_var=0.5206

  [ 310]  axes=315  step_var=0.0018  binary_acc=0.974  gap=0.1872  max_dot=0.0051  (1.8s)
    TOP:  åľºé¦Ĩ(0.10) | äºĭé¡¹(0.10) | These(0.10) | Gay(0.10) | Come(0.10) | æ¿Ģç´ł(0.10) | çļĦéľĢæ±Ĥ(0.09) | æĸ°åħ´äº§ä¸ļ(0.09)
    BOT:  sstream(-0.10) | æĤ£æľī(-0.10) | çľ¼éĩĮ(-0.10) | ################################################(-0.09) | #!(-0.09) | å¥½è½¬(-0.09) | self(-0.09) | doomed(-0.09)
    ACCEPTED as axis_315  cumulative_var=0.5215

  [ 311]  axes=316  step_var=0.0018  binary_acc=0.970  gap=0.1843  max_dot=0.0023  (1.8s)
    TOP:  åĨ³(0.10) | èģĶèµĽ(0.09) | è¸ıå®ŀ(0.09) | ë°±(0.09) | ä¸ĸçķĮä¸Ĭ(0.09) | æ¯ıä¸Ģ(0.09) | entropy(0.09) | ä¸ľè¥¿(0.09)
    BOT:  Į(-0.10) | enc(-0.09) | æ³¼(-0.09) | è´«åĽ°åľ°åĮº(-0.09) | çģ¯åħī(-0.09) | çģ¾åĮº(-0.09) | èĩ´(-0.09) | èĢģå¹´(-0.09)
    ACCEPTED as axis_316  cumulative_var=0.5223

  [ 312]  axes=317  step_var=0.0018  binary_acc=0.979  gap=0.1828  max_dot=0.0030  (1.9s)
    TOP:  ä¸ºæł¸å¿ĥ(0.11) | assay(0.10) | ä¸»è¦ģåĨħå®¹(0.09) | åĿĲçĿĢ(0.09) | åĩŃä»Ģä¹Ī(0.09) | æĹ¥æĬ¥éģĵ(0.09) | æĬ¥åĳĬæĺ¾ç¤º(0.09) | èĢ·(0.09)
    BOT:  äº§çī©(-0.11) | ob(-0.09) | owning(-0.09) | ?.(-0.09) | æĺ¯åĲ¦æľī(-0.09) | é¡ŀ(-0.09) | being(-0.09) | æĺ¯åĲ¦(-0.09)
    ACCEPTED as axis_317  cumulative_var=0.5232

  [ 313]  axes=318  step_var=0.0018  binary_acc=0.987  gap=0.1856  max_dot=0.0017  (1.9s)
    TOP:  åľ°ä¸ĭ(0.11) | åĩºèī²(0.10) | éĤĳ(0.10) | é³ŀ(0.10) | $((0.10) | igure(0.10) | $($(0.09) | åľ¨ä¸ŃåĽ½(0.09)
    BOT:  ŀ(-0.12) | repetitive(-0.10) | é£İåĲ¹(-0.09) | solely(-0.09) | å¼ĢçĽĺ(-0.09) | singled(-0.09) | æµ·åħ³(-0.09) | çĶµå·¥(-0.09)
    ACCEPTED as axis_318  cumulative_var=0.5241

  [ 314]  axes=319  step_var=0.0018  binary_acc=0.964  gap=0.1829  max_dot=0.0023  (1.8s)
    TOP:  leich(0.11) | liken(0.10) | çĶŁæĢģæĸĩæĺİ(0.10) | è§Ħæ¨¡åĮĸ(0.09) | çĽ´èĲ¥(0.09) | .textContent(0.09) | Async(0.09) | .onload(0.09)
    BOT:  å¥½å¥½(-0.10) | é¦ĸåħĪè¦ģ(-0.10) | è¯¸(-0.10) | èĤĿ(-0.10) | åĩºèº«(-0.10) | åĪ°å¤Ħ(-0.09) | ãĥĥ(-0.09) | ä¸ĥ(-0.09)
    ACCEPTED as axis_319  cumulative_var=0.5249

  [ 315]  axes=320  step_var=0.0018  binary_acc=0.998  gap=0.1833  max_dot=0.0051  (1.9s)
    TOP:  å¿«æį·(0.11) | á¹Ľ(0.10) | ç»ĻäºĪäºĨ(0.10) | texting(0.09) | éĽ¶ç¢İ(0.09) | çİ°åľ¨å¾Īå¤ļ(0.09) | ä¾¿æį·(0.09) | blink(0.09)
    BOT:  é¤Ĳ(-0.09) | åģıåģı(-0.09) | çĻĮ(-0.09) | æ¯ĴæĢ§(-0.09) | èĢģå¤§(-0.09) | ç¾İåŃ¦(-0.09) | æ´ĭæ´ĭ(-0.09) | è·¤(-0.09)
    ACCEPTED as axis_320  cumulative_var=0.5258

  [ 316]  axes=321  step_var=0.0018  binary_acc=0.999  gap=0.1807  max_dot=0.0047  (1.9s)
    TOP:  ä¼Ļ(0.10) | Brit(0.09) | ãĢĲ(0.09) | Pence(0.09) | ä¸ºå¤§å®¶(0.09) | ballots(0.09) | ç»Ŀå¯¹æĺ¯(0.09) | Ð²Ð¾Ð·(0.09)
    BOT:  .....(-0.11) | åģĥ(-0.11) | éĵ¾æİ¥(-0.10) | afford(-0.10) | è½½(-0.09) | entar(-0.09) | æ¡ī(-0.09) | åĲ¬çĿĢ(-0.09)
    ACCEPTED as axis_321  cumulative_var=0.5267

  [ 317]  axes=322  step_var=0.0018  binary_acc=0.970  gap=0.1812  max_dot=0.0048  (1.8s)
    TOP:  shack(0.10) | æįī(0.10) | ä½łè¯´(0.09) | unforgettable(0.09) | ä¸¤ä½į(0.09) | ç¨İæĶ¶(0.09) | .groupby(0.09) | ÑģÐ´ÐµÐ»Ð°Ð»(0.09)
    BOT:  ä¸ºä¸»(-0.11) | ðŁĺī(-0.10) | ä»¶(-0.10) | éĥ¨ä»¶(-0.10) | balk(-0.09) | Journalism(-0.09) | inÃŃcio(-0.09) | FIX(-0.09)
    ACCEPTED as axis_322  cumulative_var=0.5275

  [ 318]  axes=323  step_var=0.0019  binary_acc=1.000  gap=0.1875  max_dot=0.0035  (1.8s)
    TOP:  Ľ(0.11) | ä¸Ĭç½ĳ(0.11) | ÙĪØ¶Ø¹(0.10) | ä»·éĴ±(0.10) | éĢĢå½¹(0.09) | Î½(0.09) | çł´äº§(0.09) | æĸ°éĹ»ç½ĳ(0.09)
    BOT:  ãģ¾ãģŁ(-0.11) | hÃ¡(-0.11) | åģļè¿ĩ(-0.10) | Mutex(-0.10) | èĹ¤(-0.10) | ç§°(-0.09) | è¿Ļä»½(-0.09) | èĤĮèĤ¤(-0.09)
    ACCEPTED as axis_323  cumulative_var=0.5284

  [ 319]  axes=324  step_var=0.0018  binary_acc=0.983  gap=0.1813  max_dot=0.0088  (1.8s)
    TOP:  åĪ(0.10) | cf(0.10) | xB(0.09) | å®ŀåľ¨(0.09) | çļĦåħ³éĶ®(0.09) | aucun(0.09) | æ¦Ĩ(0.08) | appealing(0.08)
    BOT:  Liberation(-0.11) | olvency(-0.10) | åħ¥åı£(-0.10) | _goal(-0.09) | Posts(-0.09) | ä¸Ńå¼ı(-0.09) | è£±(-0.09) | ÄĲÃ´ng(-0.09)
    ACCEPTED as axis_324  cumulative_var=0.5293

  [ 320]  axes=325  step_var=0.0018  binary_acc=0.997  gap=0.1827  max_dot=0.0011  (1.9s)
    TOP:  athers(0.09) | ãĥģãĥ£(0.09) | impost(0.09) | bonus(0.09) | ãģłãģĮ(0.09) | à¹ĩà¸ļ(0.09) | èº«å½¢(0.09) | {%(0.09)
    BOT:  é«ĺäºİ(-0.11) | Ã¢t(-0.10) | ç»§(-0.10) | æŀķ(-0.10) | å¾Īå¤§(-0.10) | ÙĬØ§(-0.09) | âī¥(-0.09) | §(-0.09)
    ACCEPTED as axis_325  cumulative_var=0.5301

  [ 321]  axes=326  step_var=0.0018  binary_acc=0.996  gap=0.1864  max_dot=0.0010  (1.8s)
    TOP:  åĲĦçľģ(0.11) | åĩºèĩª(0.10) | æºĲäºİ(0.10) | æĿ¥æºĲäºİ(0.10) | æĺİå¹´(0.10) | è¿ĳæľŁ(0.10) | çŁŃæľŁåĨħ(0.10) | .arange(0.10)
    BOT:  HD(-0.10) | standard(-0.10) | Cobra(-0.10) | çļ±(-0.09) | our(-0.09) | us(-0.09) | examples(-0.09) | å¼ģ(-0.09)
    ACCEPTED as axis_326  cumulative_var=0.5310

  [ 322]  axes=327  step_var=0.0019  binary_acc=0.981  gap=0.1837  max_dot=0.0031  (1.8s)
    TOP:  etz(0.11) | ateau(0.11) | å¤§éĩıçļĦ(0.10) | è¾¼(0.10) | æľīæķĪçļĦ(0.09) | dominant(0.09) | ì(0.09) | åŃ©åŃĲçļĦ(0.09)
    BOT:  æħķå®¹(-0.10) | éĩĳéĵ¶(-0.10) | è®¡ç®Ĺ(-0.10) | ëªħ(-0.10) | NSURL(-0.10) | mm(-0.10) | å®½å®¹(-0.10) | Ease(-0.09)
    ACCEPTED as axis_327  cumulative_var=0.5318

  [ 323]  axes=328  step_var=0.0018  binary_acc=0.954  gap=0.1826  max_dot=0.0015  (1.9s)
    TOP:  çºªå½ķ(0.11) | ç»¼åĲĪåĪ©çĶ¨(0.10) | èŀįåħ¥(0.10) | åĨĻåĩº(0.10) | åıĸä»£(0.10) | è¿ĻæĦıåĳ³çĿĢ(0.10) | éľĩæĥĬ(0.10) | *=(0.10)
    BOT:  èĦĳæµ·(-0.11) | May(-0.10) | åĩ¯(-0.09) | ####(-0.09) | èĢĻ(-0.09) | å¥½åĿı(-0.09) | may(-0.09) | Ð¿ÑĢÐ°Ð²Ð°(-0.09)
    ACCEPTED as axis_328  cumulative_var=0.5327

  [ 324]  axes=329  step_var=0.0018  binary_acc=0.967  gap=0.1817  max_dot=0.0031  (1.8s)
    TOP:  éĿ¢ä¸Ĭ(0.11) | Bernstein(0.10) | ä¸Ĭçľĭ(0.10) | :${(0.09) | nhiá»ģu(0.09) | ä¸Ńæľī(0.09) | trong(0.09) | çľĭè¿ĩ(0.09)
    BOT:  Template(-0.10) | åıĹè®¿(-0.10) | ä½ĵåĪ¶(-0.10) | .baomidou(-0.09) | æ²¡èĥ½(-0.09) | å¾®éĩıåħĥç´ł(-0.09) | ä¸įæĺ¯å¾Ī(-0.09) | åħ·ä½ĵæĥħåĨµ(-0.09)
    ACCEPTED as axis_329  cumulative_var=0.5335

  [ 325]  axes=330  step_var=0.0018  binary_acc=0.990  gap=0.1818  max_dot=0.0046  (1.9s)
    TOP:  Ùī(0.10) | ×Ķ×¤(0.09) | (iter(0.09) | Bros(0.09) | å¹³åı°ä¸Ĭ(0.09) | #endif(0.09) | Ð´ÐµÐ¹ÑģÑĤÐ²Ð¸ÑĤÐµÐ»ÑĮÐ½Ð¾(0.09) | å¾ĭå¸ĪäºĭåĬ¡(0.09)
    BOT:  å¼ĢéĹ¨(-0.11) | éĹ»(-0.10) | å®ŀè¡Į(-0.10) | åħ¨æĿĳ(-0.10) | æłĩçŃ¾(-0.09) | æľīåĲį(-0.09) | alive(-0.09) | conhec(-0.09)
    ACCEPTED as axis_330  cumulative_var=0.5343

  [ 326]  axes=331  step_var=0.0019  binary_acc=0.981  gap=0.1856  max_dot=0.0032  (1.8s)
    TOP:  ä¸ĢåĪĢ(0.10) | âĳ(0.10) | åĺī(0.10) | å¢ŀåĢ¼(0.09) | roman(0.09) | æĹ¥å¸¸(0.09) | analog(0.09) | èµıæŀĲ(0.09)
    BOT:  ÐµÑħ(-0.10) | èģĶç³»(-0.10) | library(-0.10) | isIn(-0.09) | è¿İæĿ¥(-0.09) | .subplot(-0.09) | ç³ľ(-0.09) | wheels(-0.09)
    ACCEPTED as axis_331  cumulative_var=0.5352

  [ 327]  axes=332  step_var=0.0019  binary_acc=0.981  gap=0.1842  max_dot=0.0033  (1.8s)
    TOP:  éĢŁçİĩ(0.09) | éĤ£åıª(0.09) | éĽĨåĽ¢æĹĹä¸ĭ(0.09) | ãģĦãģŁãģł(0.09) | ]{(0.09) | æ¤įåħ¥(0.09) | Mister(0.09) | ä¸ŃåĽ½æĶ¿åºľ(0.09)
    BOT:  å¾ĢåĲİ(-0.10) | æľ½(-0.10) | ç¶(-0.10) | åĩłä¹İæīĢæľī(-0.09) | rarely(-0.09) | é¢Ī(-0.09) | è¾©è®º(-0.09) | Kro(-0.09)
    ACCEPTED as axis_332  cumulative_var=0.5361

  [ 328]  axes=333  step_var=0.0018  binary_acc=0.979  gap=0.1801  max_dot=0.0018  (1.8s)
    TOP:  ëĤ¨(0.10) | è¶³ä»¥(0.10) | éĢĢå½¹(0.09) | âħ(0.09) | OPY(0.09) | even(0.08) | VS(0.08) | Ð¾Ð±ÑĢ(0.08)
    BOT:  ä¸Ģè¶Ł(-0.11) | ä¸ªä¹¡éķĩ(-0.11) | å®ĺåı¸(-0.11) | âĪĪ(-0.10) | >>>(-0.10) | ä¿¡æīĺ(-0.10) | âĤ¬(-0.10) | éĬ®(-0.10)
    ACCEPTED as axis_333  cumulative_var=0.5369

  [ 329]  axes=334  step_var=0.0019  binary_acc=0.995  gap=0.1860  max_dot=0.0026  (1.8s)
    TOP:  åĽŀåĽ½(0.11) | çİ°ä»£åĮĸ(0.11) | dissertation(0.10) | é«ĺçŃīæķĻèĤ²(0.10) | ç¾İåĽ½äºº(0.10) | âĤ¬(0.10) | ãģĿãģĵ(0.10) | Äĳiá»ģu(0.10)
    BOT:  identities(-0.12) | Rights(-0.09) | è´¦åı·(-0.09) | qp(-0.09) | Relative(-0.09) | infection(-0.09) | dimensional(-0.09) | PSD(-0.09)
    ACCEPTED as axis_334  cumulative_var=0.5378

  [ 330]  axes=335  step_var=0.0018  binary_acc=0.981  gap=0.1803  max_dot=0.0034  (1.8s)
    TOP:  åłµ(0.12) | pÅĻ(0.11) | ï¬Ģ(0.10) | å°º(0.10) | æļĸ(0.10) | -Th(0.10) | wurde(0.09) | çĲ¬(0.09)
    BOT:  Î·(-0.11) | ä¿¦(-0.10) | let(-0.10) | junto(-0.09) | idas(-0.09) | æĽ´å¼º(-0.09) | alle(-0.09) | ä½łè¿ĺ(-0.09)
    ACCEPTED as axis_335  cumulative_var=0.5386

  [ 331]  axes=336  step_var=0.0019  binary_acc=0.987  gap=0.1879  max_dot=0.0055  (1.9s)
    TOP:  à¹īà¸Ńà¸Ļ(0.10) | é¡¹çĽ®çļĦ(0.10) | chapter(0.10) | éĩĳä»·(0.10) | Feb(0.09) | æľ¬æĸĩ(0.09) | åĤ©(0.09) | ++)(0.09)
    BOT:  å§ĭç»Īä¿ĿæĮģ(-0.12) | ä¹Łæĺ¯(-0.09) | åıªä¸įè¿ĩæĺ¯(-0.09) | ä¸įåĨįæĺ¯(-0.09) | çİ°å·²(-0.09) | äºĮç»´çłģ(-0.09) | å¿įåıĹ(-0.09) | justices(-0.09)
    ACCEPTED as axis_336  cumulative_var=0.5395

  [ 332]  axes=337  step_var=0.0018  binary_acc=0.996  gap=0.1766  max_dot=0.0022  (1.8s)
    TOP:  çļĦè¯Ŀ(0.10) | è¦ģåİ»(0.10) | ç¥ĸ(0.10) | æĭ¦(0.09) | à¹Īà¸Ń(0.09) | å½Ĵå±ŀäºİ(0.09) | ritic(0.09) | azi(0.09)
    BOT:  åħŃ(-0.10) | Å¡(-0.10) | idget(-0.10) | ä¸Ģä¼ļåĦ¿(-0.09) | å¿«ä¸ī(-0.09) | vier(-0.09) | SR(-0.09) | -two(-0.09)
    ACCEPTED as axis_337  cumulative_var=0.5403

  [ 333]  axes=338  step_var=0.0018  binary_acc=0.966  gap=0.1789  max_dot=0.0019  (1.8s)
    TOP:  recipe(0.10) | éķ¶(0.10) | è®¿(0.09) | Ø§ÙĦÙĪ(0.09) | éĢĶ(0.09) | Esto(0.09) | 'elle(0.09) | "))(0.08)
    BOT:  Based(-0.12) | Based(-0.12) | increasingly(-0.10) | åĲĥäºı(-0.09) | (prob(-0.09) | åħµåĻ¨(-0.09) | éĴ¢æĿĲ(-0.09) | .choices(-0.09)
    ACCEPTED as axis_338  cumulative_var=0.5411

  [ 334]  axes=339  step_var=0.0018  binary_acc=0.971  gap=0.1790  max_dot=0.0008  (1.9s)
    TOP:  Sep(0.13) | otas(0.10) | å®īä¿Ŀ(0.10) | esteemed(0.10) | Mar(0.10) | â½(0.10) | May(0.10) | Magento(0.10)
    BOT:  aboard(-0.09) | âĴ(-0.09) | eddar(-0.09) | æ¢°(-0.09) | ÑĩÐ¸Ð²(-0.09) | itive(-0.08) | .dat(-0.08) | .asList(-0.08)
    ACCEPTED as axis_339  cumulative_var=0.5420

  [ 335]  axes=340  step_var=0.0018  binary_acc=0.987  gap=0.1782  max_dot=0.0048  (1.9s)
    TOP:  &&Ċ(0.10) | !âĢĻ(0.09) | æ§ĺãĢħãģª(0.09) | å¸¸è§Ħ(0.09) | atoire(0.09) | å¸¸è§ģ(0.09) | åı¯è§Ĩ(0.08) | professional(0.08)
    BOT:  åĨĽå·¥(-0.11) | .drawable(-0.10) | å°ļä¹¦(-0.10) | æħİéĩį(-0.09) | çģ¯çģ«(-0.09) | æĮ¨(-0.09) | å¼Ģå·¥(-0.09) | å¼Ģåĩº(-0.09)
    ACCEPTED as axis_340  cumulative_var=0.5428

  [ 336]  axes=341  step_var=0.0018  binary_acc=0.994  gap=0.1796  max_dot=0.0069  (1.9s)
    TOP:  ç»Īæŀģ(0.11) | å¸ĤåľºçļĦ(0.10) | kindness(0.10) | é«ĺä¸ī(0.09) | Though(0.09) | åľ°äº§(0.09) | Ð°ÑĢÑĤ(0.09) | çļĦäºĭ(0.09)
    BOT:  Ø¨Ø´ÙĥÙĦ(-0.10) | Ø¨(-0.09) | (np(-0.09) | å¾¡(-0.09) | ä¸įæĪĲ(-0.09) | ráº¥t(-0.09) | åİ¿éķ¿(-0.09) | æĪĲ(-0.09)
    ACCEPTED as axis_341  cumulative_var=0.5436

  [ 337]  axes=342  step_var=0.0018  binary_acc=0.998  gap=0.1774  max_dot=0.0059  (1.9s)
    TOP:  è¿Ļä¸ªéĹ®é¢ĺ(0.12) | McConnell(0.10) | ãĢĤï¼Ī(0.10) | SKU(0.10) | phen(0.10) | Consult(0.09) | ÑĢÐµÐ±(0.09) | consult(0.09)
    BOT:  babys(-0.10) | åĨħçļĦ(-0.10) | (:(-0.10) | æľīè¿ĩ(-0.10) | ä½Ĳ(-0.09) | ye(-0.09) | xxx(-0.09) | /{}/(-0.09)
    ACCEPTED as axis_342  cumulative_var=0.5444

  [ 338]  axes=343  step_var=0.0018  binary_acc=0.989  gap=0.1788  max_dot=0.0057  (1.9s)
    TOP:  Shelf(0.10) | ÐºÑĥÑĢ(0.10) | å·¡è§Ĩ(0.09) | åıªè§ģ(0.09) | åħīä¼ı(0.09) | à¸¹à¸ģ(0.09) | å®Ŀé©¬(0.09) | æ¬§åħĥ(0.09)
    BOT:  ToString(-0.10) | æĺ¯æ²¡æľī(-0.10) | âīł(-0.09) | æ²¡æľī(-0.09) | strings(-0.09) | ä¹Łæľī(-0.09) | è·ŁæĪĳ(-0.09) | put(-0.09)
    ACCEPTED as axis_343  cumulative_var=0.5452

  [ 339]  axes=344  step_var=0.0018  binary_acc=0.981  gap=0.1776  max_dot=0.0013  (1.9s)
    TOP:  çĲĨæĢ§(0.09) | åıĹä¼¤(0.09) | å°ĳãģĹ(0.09) | æĮĩæĮ¥éĥ¨(0.09) | åģļä¸ĢäºĽ(0.09) | à¹ĥà¸Ļ(0.09) | åı¤åŁİ(0.09) | .for(0.09)
    BOT:  BI(-0.10) | çļĦè¦ģæ±Ĥ(-0.10) | Bet(-0.09) | igit(-0.09) | âĢŀ(-0.09) | anytime(-0.09) | WA(-0.09) | æİĸ(-0.09)
    ACCEPTED as axis_344  cumulative_var=0.5460

  [ 340]  axes=345  step_var=0.0018  binary_acc=0.988  gap=0.1785  max_dot=0.0006  (1.8s)
    TOP:  å¤ıå¤©(0.10) | å¾Ģè¿Ķ(0.10) | geme(0.09) | æ¯Ķä¸Ĭå¹´(0.09) | ç¥ĸåħĪ(0.09) | ÐŃ(0.09) | çĶ¨é¤Ĳ(0.09) | å®ŀåľ¨æĺ¯(0.09)
    BOT:  ä¸įå¯¹(-0.10) | Accordingly(-0.10) | However(-0.09) | Ð¸ÑĩÐµÑģÐºÐ¸(-0.09) | ÐŀÐ´Ð½Ð°ÐºÐ¾(-0.09) | åı¥åŃĲ(-0.09) | uated(-0.09) | å¥½(-0.09)
    ACCEPTED as axis_345  cumulative_var=0.5469

  [ 341]  axes=346  step_var=0.0018  binary_acc=0.960  gap=0.1780  max_dot=0.0028  (2.0s)
    TOP:  boxed(0.12) | dÃŃa(0.10) | åĲįä¸º(0.10) | äºĭé¡¹(0.09) | ï¼īãĢģ(0.09) | æĿĥçĽĬ(0.09) | Additionally(0.09) | competitive(0.09)
    BOT:  ä»İæĿ¥(-0.10) | ç»Ļäºº(-0.10) | ĉĊ(-0.10) | .S(-0.10) | è¯´ä¸įåĩº(-0.09) | æ»ĭçĶŁ(-0.09) | .How(-0.09) | _DECL(-0.09)
    ACCEPTED as axis_346  cumulative_var=0.5476

  [ 342]  axes=347  step_var=0.0018  binary_acc=0.975  gap=0.1742  max_dot=0.0034  (1.8s)
    TOP:  off(0.11) | ä¸Ĭæµ·å¸Ĥ(0.10) | çŁ¥è¯ĨåĪĨåŃĲ(0.10) | ÑĤÐµÐ»(0.09) | æł¡åıĭ(0.09) | çĶµè§Ĩ(0.09) | Ð¾Ð±ÑīÐµÑģÑĤÐ²(0.09) | Non(0.09)
    BOT:  ä¼ļåĩºçİ°(-0.10) | Paperback(-0.10) | å½ĵæĪĲ(-0.09) | çľĭè§ģ(-0.09) | cÃ²n(-0.09) | éĢłæĪĲ(-0.09) | æķ´æķ´(-0.09) | Â·(-0.09)
    ACCEPTED as axis_347  cumulative_var=0.5485

  [ 343]  axes=348  step_var=0.0018  binary_acc=0.977  gap=0.1810  max_dot=0.0052  (1.9s)
    TOP:  åĳ¨æľŁ(0.10) | .base(0.09) | Span(0.09) | IO(0.09) | ä¹ħ(0.09) | åı£æ°´(0.09) | &S(0.09) | -country(0.09)
    BOT:  æĭįåįĸ(-0.10) | Means(-0.10) | fullest(-0.10) | [data(-0.09) | å®ĥ(-0.09) | Prob(-0.09) | -so(-0.09) | [{(-0.09)
    ACCEPTED as axis_348  cumulative_var=0.5493

  [ 344]  axes=349  step_var=0.0018  binary_acc=0.981  gap=0.1793  max_dot=0.0056  (1.9s)
    TOP:  DM(0.10) | cd(0.09) | rapidly(0.09) | eval(0.09) | XS(0.09) | æķ°æį®åºĵ(0.09) | .By(0.09) | ;s(0.09)
    BOT:  nÃ£o(-0.11) | unmistak(-0.10) | three(-0.10) | å®ŀæķĪ(-0.09) | ÅŁ(-0.09) | tres(-0.09) | ä¼łçĲĥ(-0.09) | ìŀ¥(-0.09)
    ACCEPTED as axis_349  cumulative_var=0.5501

  [ 345]  axes=350  step_var=0.0017  binary_acc=0.980  gap=0.1741  max_dot=0.0025  (1.9s)
    TOP:  à¸ľ(0.11) | èħł(0.09) | capacitÃ©(0.09) | jumlah(0.09) | amongst(0.09) | Ð¿ÑĢÐ¸Ð¼ÐµÐ½(0.09) | åħ§å®¹(0.09) | Ð¾ÑĤÐ½Ð¾(0.09)
    BOT:  Ñģ(-0.10) | Motors(-0.10) | èĪĮå°ĸ(-0.10) | Northeast(-0.09) | ä¾ĽåĽ¾(-0.09) | ìĦľ(-0.09) | éĢģæĿ¥(-0.09) | iP(-0.09)
    ACCEPTED as axis_350  cumulative_var=0.5509

  [ 346]  axes=351  step_var=0.0018  binary_acc=0.961  gap=0.1730  max_dot=0.0009  (1.8s)
    TOP:  [âĢ¦](0.10) | ä½Ĩè¿Ļ(0.10) | wÃ¤hrend(0.10) | è¿Ļç§į(0.09) | âĢ¦but(0.09) | èĥĮåĲİçļĦ(0.09) | unb(0.09) | åĮł(0.09)
    BOT:  æı(-0.11) | æĭ(-0.09) | æĭ·(-0.09) | åĲį(-0.09) | æĩ(-0.09) | åıĳéĢģ(-0.09) | éªĹ(-0.09) | Ð¼ÐµÑģÑĤ(-0.09)
    ACCEPTED as axis_351  cumulative_var=0.5517

  [ 347]  axes=352  step_var=0.0017  binary_acc=0.980  gap=0.1732  max_dot=0.0009  (1.8s)
    TOP:  åĻ¢(0.09) | atravÃ©s(0.09) | Ľ(0.09) | åĶĶ(0.08) | crem(0.08) | menos(0.08) | allem(0.08) | å¤§åŃ¦çĶŁ(0.08)
    BOT:  æ¸©å®¤(-0.10) | åıĳçĶŁäºĨ(-0.10) | æī®æ¼Ķ(-0.10) | åĩºåıĳ(-0.09) | æ¶¦æ»ĳ(-0.09) | ä¸°çĶ°(-0.09) | æĪŁ(-0.08) | å¯¼è´Ń(-0.08)
    ACCEPTED as axis_352  cumulative_var=0.5525

  [ 348]  axes=353  step_var=0.0018  binary_acc=0.967  gap=0.1747  max_dot=0.0009  (1.9s)
    TOP:  å¾Ĺä»¥(0.10) | åĭŁéĽĨèµĦéĩĳ(0.10) | åĩºéģĵ(0.09) | æķŀ(0.09) | Ð¸Ð¹(0.09) | dá»ĭch(0.09) | åı¸(0.09) | guilt(0.09)
    BOT:  Lion(-0.12) | ä¸»æµģ(-0.11) | Ð¾ÑģÐ½Ð¾Ð²Ð½Ð¾Ð¼(-0.09) | corner(-0.09) | æľĹè¯µ(-0.09) | alk(-0.09) | å®¡æł¸(-0.09) | Associated(-0.09)
    ACCEPTED as axis_353  cumulative_var=0.5533

  [ 349]  axes=354  step_var=0.0017  binary_acc=0.985  gap=0.1739  max_dot=0.0017  (1.9s)
    TOP:  âĺĨ(0.09) | .RequestMapping(0.09) | ä½ıæĪ·(0.09) | Ð°ÑĤÑĮ(0.09) | ÐµÑĤÑĮ(0.08) | çĿĢåĬĽ(0.08) | Published(0.08) | falta(0.08)
    BOT:  æĪĳä¸į(-0.11) | ä½łä¸į(-0.10) | æĹ¥æĬ¥(-0.10) | éĤ£ä¸Ģ(-0.09) | å¤©èµĭ(-0.09) | +Ċ(-0.09) | å¹´æĬ¥(-0.09) | å¦Ĥæŀľä¸į(-0.09)
    ACCEPTED as axis_354  cumulative_var=0.5540

  [ 350]  axes=355  step_var=0.0017  binary_acc=0.967  gap=0.1765  max_dot=0.0013  (1.8s)
    TOP:  travÃ©s(0.10) | âĢĻint(0.10) | Ð·ÑĥÐ±(0.09) | qui(0.09) | å±±åİ¿(0.09) | âĢĻil(0.09) | çĸĥ(0.09) | æĪĳä¸į(0.09)
    BOT:  æĺ(-0.11) | éľ²åĩº(-0.11) | åĲ¬äºĨ(-0.10) | åįĩèµ·(-0.09) | mk(-0.09) | CS(-0.09) | æŀĦæĪĲäºĨ(-0.09) | æĮĩåĲĳ(-0.09)
    ACCEPTED as axis_355  cumulative_var=0.5548

  [ 351]  axes=356  step_var=0.0018  binary_acc=0.997  gap=0.1742  max_dot=0.0023  (1.8s)
    TOP:  íŀĪ(0.10) | åıĳå£°(0.09) | æĸ¹åĲĳ(0.09) | universe(0.09) | .fill(0.09) | çļĦæĪĲéķ¿(0.09) | ìĹĲëıĦ(0.09) | çİ¯èĬĤ(0.09)
    BOT:  æľ¬æ¡Ī(-0.12) | æ¯ıæ¬¡(-0.10) | {{(-0.10) | ÑĤÐ°Ðº(-0.10) | {{(-0.09) | åľ¨æŃ¤(-0.09) | åĲĮæł·çļĦ(-0.09) | æľīç§į(-0.09)
    ACCEPTED as axis_356  cumulative_var=0.5556

  [ 352]  axes=357  step_var=0.0018  binary_acc=0.985  gap=0.1765  max_dot=0.0019  (1.9s)
    TOP:  _verbose(0.10) | keras(0.09) | åĽ¢ä½ĵ(0.09) | ĉ(0.09) | é¦ĸ(0.09) | ä¸¤é¡¹(0.09) | è¿Ļæľ¬ä¹¦(0.09) | Tech(0.09)
    BOT:  æľįçĶ¨(-0.10) | wh(-0.10) | åģ¥åº·ç®¡çĲĨ(-0.10) | åĲĥä»Ģä¹Ī(-0.09) | |^(-0.09) | cuck(-0.09) | chl(-0.09) | å®¡è®®éĢļè¿ĩ(-0.09)
    ACCEPTED as axis_357  cumulative_var=0.5564

  [ 353]  axes=358  step_var=0.0018  binary_acc=0.970  gap=0.1835  max_dot=0.0047  (1.8s)
    TOP:  insurance(0.10) | å¡(0.10) | ç»´ä¿®(0.09) | ä¼¯(0.09) | çĦķåıĳ(0.09) | Cbd(0.09) | è¿ĳäºĽ(0.09) | atk(0.09)
    BOT:  .loads(-0.09) | æĪĳä¸įæĥ³(-0.09) | requer(-0.09) | ç¿Ĭ(-0.09) | ?,(-0.08) | founding(-0.08) | oss(-0.08) | Uni(-0.08)
    ACCEPTED as axis_358  cumulative_var=0.5572

  [ 354]  axes=359  step_var=0.0018  binary_acc=0.987  gap=0.1747  max_dot=0.0052  (1.9s)
    TOP:  wie(0.09) | Ã©l(0.09) | âĢĻve(0.09) | cho(0.09) | lots(0.09) | åķ¦(0.09) | å¤©çĶŁ(0.09) | æŃ»åĲİ(0.09)
    BOT:  turbulent(-0.10) | ç¤¼(-0.09) | å·¥æľŁ(-0.09) | ç§¤(-0.09) | æĹĹ(-0.09) | æŃĨ(-0.09) | ä¸įè®ºæĺ¯(-0.09) | ä¸Ģç¾¤(-0.09)
    ACCEPTED as axis_359  cumulative_var=0.5580

  [ 355]  axes=360  step_var=0.0018  binary_acc=0.990  gap=0.1740  max_dot=0.0009  (1.9s)
    TOP:  ä¹Ŀå·ŀ(0.09) | oko(0.09) | dearly(0.08) | ŀĭ(0.08) | '.(0.08) | ç¥ļ(0.08) | vh(0.08) | åıªéľĢ(0.08)
    BOT:  æ¹ĸåįĹçľģ(-0.11) | åľ°è¯´(-0.10) | è¿ĳå¹´æĿ¥(-0.10) | argparse(-0.09) | Echo(-0.09) | æĮĩçº¹(-0.09) | æ··(-0.09) | è¯´çļĦæĺ¯(-0.09)
    ACCEPTED as axis_360  cumulative_var=0.5587

  [ 356]  axes=361  step_var=0.0018  binary_acc=0.986  gap=0.1789  max_dot=0.0040  (1.9s)
    TOP:  entitled(0.11) | ä¹Ŀé¾Ļ(0.09) | +"(0.09) | wont(0.09) | .setError(0.09) | å°±è¯´(0.09) | headache(0.09) | heritance(0.09)
    BOT:  å¥¹(-0.09) | æī¾åĪ°(-0.09) | ä»£çłģ(-0.09) | ä¸Ģæ¬¾(-0.09) | çļĦæľºä¼ļ(-0.09) | Ze(-0.09) | æī¾åĪ°äºĨ(-0.09) | You(-0.09)
    ACCEPTED as axis_361  cumulative_var=0.5595

  [ 357]  axes=362  step_var=0.0017  binary_acc=0.972  gap=0.1706  max_dot=0.0002  (1.8s)
    TOP:  æĽ´å®¹æĺĵ(0.11) | ä¸įçĪ±(0.10) | éĢıéľ²(0.10) | InitializeComponent(0.09) | è¦ģä¸įè¦ģ(0.09) | çĹĽçĤ¹(0.09) | çľŁäºº(0.09) | .ArgumentParser(0.09)
    BOT:  implementations(-0.09) | hÃ£y(-0.09) | î(-0.09) | ÐĴÑĭ(-0.09) | âĳ(-0.08) | å°(-0.08) | çºªå½ķ(-0.08) | Ð²Ñĭ(-0.08)
    ACCEPTED as axis_362  cumulative_var=0.5603

  [ 358]  axes=363  step_var=0.0018  binary_acc=0.982  gap=0.1775  max_dot=0.0010  (1.9s)
    TOP:  datab(0.10) | çĶ³ãģĹ(0.10) | æ¯Ķè¼ĥ(0.09) | chÄĥm(0.09) | ritch(0.09) | Imaging(0.09) | Ø´Ø¨(0.09) | tem(0.08)
    BOT:  åĩĢåĪ©æ¶¦(-0.10) | ANC(-0.10) | ç¨İæĶ¶(-0.09) | warts(-0.09) | Â«(-0.09) | æĶ¶çĽĬ(-0.09) | "Not(-0.09) | å¸½(-0.09)
    ACCEPTED as axis_363  cumulative_var=0.5611

  [ 359]  axes=364  step_var=0.0017  binary_acc=0.986  gap=0.1722  max_dot=0.0061  (1.9s)
    TOP:  iosis(0.11) | ç«ĭæ¡Ī(0.09) | Åļ(0.09) | __(Ċ(0.09) | ="{{(0.09) | åĪļ(0.09) | çļĦè®¾è®¡(0.09) | quickly(0.09)
    BOT:  èĢĮéĿŀ(-0.09) | ĵ(-0.09) | ãģĻãģ¹ãģ¦(-0.09) | (one(-0.09) | å¸¸è¯´(-0.09) | ç¬¬åħŃ(-0.09) | äº«çĶ¨(-0.08) | éĵ¶(-0.08)
    ACCEPTED as axis_364  cumulative_var=0.5618

  [ 360]  axes=365  step_var=0.0019  binary_acc=0.976  gap=0.1764  max_dot=0.0064  (2.0s)
    TOP:  iw(0.10) | ä¸ŃåĮ»èį¯(0.10) | Fig(0.09) | åıĹéĻĲ(0.09) | given(0.09) | éĹ»åĲį(0.09) | åĮ»(0.09) | wastes(0.09)
    BOT:  éĹ®é¢ĺæĺ¯(-0.10) | actly(-0.10) | è¿ĺæľīä¸Ģä¸ª(-0.09) | ç¾İèģĶåĤ¨(-0.09) | åħīè¾ī(-0.09) | violent(-0.09) | æĪĳä¸įæĺ¯(-0.09) | è¿Ļæīįæĺ¯(-0.09)
    ACCEPTED as axis_365  cumulative_var=0.5626

  [ 361]  axes=366  step_var=0.0017  binary_acc=0.994  gap=0.1703  max_dot=0.0087  (1.8s)
    TOP:  åįıè°ĥåıĳå±ķ(0.10) | ä¸Ńå¿ĥåŁİå¸Ĥ(0.09) | affection(0.09) | åģ¥åº·æĪĲéķ¿(0.09) | ëĦĺìĸ´(0.09) | å®īå®ģ(0.08) | æĶ¹éĿ©åıĳå±ķ(0.08) | Ð¾ÑģÑĤÐ¾Ñı(0.08)
    BOT:  "__(-0.10) | (--(-0.09) | è¿Ļå®¶(-0.09) | æĽ´(-0.09) | -effect(-0.09) | mong(-0.09) | åĩłå¤©(-0.09) | åľ¨å¤ĸ(-0.09)
    ACCEPTED as axis_366  cumulative_var=0.5634

  [ 362]  axes=367  step_var=0.0017  binary_acc=0.985  gap=0.1747  max_dot=0.0034  (1.8s)
    TOP:  åĪĨè¡Į(0.09) | }{(0.09) | expectation(0.09) | ç»Ļä»ĸä»¬(0.09) | è¯¥æĿĳ(0.09) | universities(0.09) | æľºéģĩ(0.09) | å¾ĭå¸ĪäºĭåĬ¡(0.09)
    BOT:  å®½(-0.09) | &quot(-0.09) | shed(-0.09) | irregular(-0.09) | è¿ĳä»£(-0.09) | Tests(-0.09) | OBJECT(-0.08) | ä¼¼çļĦ(-0.08)
    ACCEPTED as axis_367  cumulative_var=0.5641

  [ 363]  axes=368  step_var=0.0017  binary_acc=0.985  gap=0.1715  max_dot=0.0030  (1.8s)
    TOP:  ç²¾å½©(0.11) | åĩī(0.09) | æĢĿèĢĥ(0.09) | æ²¤(0.09) | ãģĿãĤĮ(0.09) | entitlement(0.09) | .cli(0.09) | å®Ľ(0.09)
    BOT:  ("$(-0.11) | $\(-0.09) | æ·±åıĹ(-0.09) | doesn(-0.09) | è¿Ļä¹Īå¤§(-0.09) | åħ¨å¥Ĺ(-0.09) | ['$(-0.09) | åĩºå±Ģ(-0.09)
    ACCEPTED as axis_368  cumulative_var=0.5649

  [ 364]  axes=369  step_var=0.0018  binary_acc=0.992  gap=0.1740  max_dot=0.0014  (1.8s)
    TOP:  åĮĹæĸ¹(0.10) | åīįéĿ¢(0.10) | åºĹ(0.10) | ISS(0.09) | æį§(0.09) | kers(0.09) | å±±è¥¿çľģ(0.09) | ìľĦ(0.09)
    BOT:  æĹ¥æĬ¥éģĵ(-0.10) | pe(-0.10) | par(-0.10) | ãģŁ(-0.09) | æ³ķåĪĻ(-0.09) | skinny(-0.09) | åĽ½åº¦(-0.09) | _by(-0.09)
    ACCEPTED as axis_369  cumulative_var=0.5657

  [ 365]  axes=370  step_var=0.0018  binary_acc=0.998  gap=0.1687  max_dot=0.0057  (1.9s)
    TOP:  ï»¿namespace(0.10) | å®ĥ(0.10) | Ã¥(0.10) | æĴµ(0.10) | email(0.10) | è¯·ä½ł(0.09) | Ã©g(0.09) | æĬĬå®ĥ(0.09)
    BOT:  khá»ıi(-0.11) | vÃ´(-0.10) | anh(-0.10) | ìĸĳ(-0.10) | çĨŁæĤīçļĦ(-0.09) | .selectAll(-0.09) | å°Ī(-0.09) | ê°ķ(-0.09)
    ACCEPTED as axis_370  cumulative_var=0.5664

  [ 366]  axes=371  step_var=0.0018  binary_acc=0.977  gap=0.1752  max_dot=0.0007  (1.9s)
    TOP:  ê±´(0.10) | authentic(0.09) | ì²(0.09) | ëĦĺ(0.09) | ëĤ(0.09) | éĢģåĩº(0.09) | unh(0.09) | ä¸Ģå¤§(0.09)
    BOT:  .I(-0.09) | é£İæĻ¯(-0.09) | opl(-0.09) | sep(-0.09) | .Br(-0.09) | -fold(-0.09) | é©¬äºĳ(-0.09) | >I(-0.09)
    ACCEPTED as axis_371  cumulative_var=0.5672

  [ 367]  axes=372  step_var=0.0017  binary_acc=0.994  gap=0.1680  max_dot=0.0061  (1.8s)
    TOP:  ÐµÐ½ÑĮ(0.09) | TypeError(0.09) | .google(0.08) | .nih(0.08) | Statistics(0.08) | Ð¸ÑĢÐ¾Ð²(0.08) | ContextMenu(0.08) | .Query(0.08)
    BOT:  âĬ(-0.11) | dr(-0.10) | Um(-0.10) | dá»±(-0.10) | -Ch(-0.09) | Xt(-0.09) | éª§(-0.08) | Tre(-0.08)
    ACCEPTED as axis_372  cumulative_var=0.5679

  [ 368]  axes=373  step_var=0.0017  binary_acc=0.984  gap=0.1722  max_dot=0.0026  (1.8s)
    TOP:  èŃ¬å¦Ĥ(0.09) | onCreateView(0.09) | ä¸įå¦Ĥ(0.09) | .assert(0.09) | Ã²(0.09) | éĵĻ(0.09) | çī¹æ´¾(0.09) | ?,?,(0.09)
    BOT:  bf(-0.10) | à¹ĥà¸Ļ(-0.10) | åĪ°æĿ¥(-0.10) | <\/(-0.09) | dealing(-0.09) | åħ³æĢĢ(-0.09) | phantom(-0.09) | lain(-0.09)
    ACCEPTED as axis_373  cumulative_var=0.5686

  [ 369]  axes=374  step_var=0.0017  binary_acc=0.986  gap=0.1719  max_dot=0.0064  (1.9s)
    TOP:  Ð¾ÑĤ(0.09) | Hybrid(0.09) | çĲĨäºĭéķ¿(0.09) | unanimous(0.09) | effortless(0.09) | ë°(0.08) | }.(0.08) | èĳ£(0.08)
    BOT:  SEQU(-0.10) | èª(-0.09) | baking(-0.09) | sharply(-0.09) | ç´§ç¼º(-0.09) | careers(-0.09) | Êĥ(-0.08) | äº¨(-0.08)
    ACCEPTED as axis_374  cumulative_var=0.5694

  [ 370]  axes=375  step_var=0.0017  binary_acc=0.994  gap=0.1713  max_dot=0.0027  (1.9s)
    TOP:  ential(0.10) | enuity(0.10) | ä¸įåĨį(0.09) | where(0.09) | \t(0.09) | ="$(0.09) | åĲĳä¸ĭ(0.09) | ê¸°(0.09)
    BOT:  æŃ£å¤Ħäºİ(-0.10) | æŃ¦èŃ¦(-0.10) | Ŀ(-0.09) | æķ£æĸĩ(-0.09) | æ°ĳåĬŀ(-0.09) | Iron(-0.09) | à¸Ńà¸²à¸«à¸²à¸£(-0.09) | âĹı(-0.08)
    ACCEPTED as axis_375  cumulative_var=0.5701

  [ 371]  axes=376  step_var=0.0017  binary_acc=0.981  gap=0.1689  max_dot=0.0023  (1.9s)
    TOP:  -it(0.11) | åľ¨æľªæĿ¥(0.10) | Game(0.09) | [][](0.09) | Out(0.09) | åĨ¶éĩĳ(0.09) | åķ¸(0.09) | ä»ķ(0.09)
    BOT:  ä½Ĩä»į(-0.10) | ä½ĵç§¯(-0.10) | ä¾¿(-0.10) | ä»įå°Ĩ(-0.09) | ä¸Ķ(-0.09) | Ą(-0.09) | Äĳá»ģu(-0.09) | åĤ¨éĩı(-0.09)
    ACCEPTED as axis_376  cumulative_var=0.5708

  [ 372]  axes=377  step_var=0.0018  binary_acc=0.988  gap=0.1721  max_dot=0.0025  (1.8s)
    TOP:  e(0.10) | ä¸ĥå¤§(0.10) | ïĤ(0.10) | (this(0.09) | åĲĦ(0.09) | y(0.09) | o(0.09) | ichever(0.09)
    BOT:  Dimensions(-0.10) | «(-0.10) | ä¹ĭå®¶(-0.09) | ouser(-0.09) | éĹ®åį·(-0.09) | Ð´Ð¾Ð¼Ð°(-0.09) | çģ°å°ĺ(-0.09) | Ð¾Ñı(-0.09)
    ACCEPTED as axis_377  cumulative_var=0.5716

  [ 373]  axes=378  step_var=0.0017  binary_acc=0.995  gap=0.1722  max_dot=0.0083  (1.9s)
    TOP:  é©(0.10) | åĪ°ä½į(0.09) | username(0.09) | ä¸įåĲĪ(0.09) | ĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊ(0.09) | ---ĊĊ(0.09) | ĊĊĊĊĊ(0.09) | WebDriverWait(0.09)
    BOT:  åĮĹäº¬æĹ¶éĹ´(-0.10) | æ¬§(-0.09) | è¶Ł(-0.09) | rugged(-0.09) | ".$(-0.09) | ëįĺ(-0.09) | à¥įà¤(-0.09) | alors(-0.09)
    ACCEPTED as axis_378  cumulative_var=0.5724

  [ 374]  axes=379  step_var=0.0016  binary_acc=0.986  gap=0.1660  max_dot=0.0022  (1.9s)
    TOP:  æĪĳæīį(0.09) | ç»ĻåŃ©åŃĲ(0.09) | åĬŀäºĭ(0.09) | -care(0.09) | åģļäºº(0.08) | ]{(0.08) | /*Ċ(0.08) | æīĵçĲĥ(0.08)
    BOT:  Ex(-0.09) | åĲ(-0.09) | high(-0.08) | NA(-0.08) | âĹ(-0.08) | prestigious(-0.08) | Shape(-0.08) | UPC(-0.08)
    ACCEPTED as axis_379  cumulative_var=0.5731

  [ 375]  axes=380  step_var=0.0017  binary_acc=0.995  gap=0.1688  max_dot=0.0065  (1.9s)
    TOP:  ìĤ(0.10) | ä¸įéĶĪéĴ¢(0.09) | Has(0.09) | _list(0.09) | extern(0.09) | å¾ĭå¸Ī(0.09) | ä¸ŃåĽ½æĸĩåĮĸ(0.09) | ä¸įå®¹æĺĵ(0.09)
    BOT:  çŃīæĸ¹éĿ¢çļĦ(-0.10) | ä½ľåĩº(-0.09) | ä¿ĥè¿ĽäºĨ(-0.09) | åĪĽéĢłåĩº(-0.09) | åĩºä»»(-0.09) | ä½ľäºĨ(-0.09) | km(-0.09) | MN(-0.09)
    ACCEPTED as axis_380  cumulative_var=0.5738

  [ 376]  axes=381  step_var=0.0017  binary_acc=0.976  gap=0.1658  max_dot=0.0057  (1.8s)
    TOP:  çĸĿ(0.10) | Cannot(0.09) | ranking(0.09) | æĵįä½ľ(0.09) | è§Ĩ(0.09) | ç»´æĬ¤(0.09) | ç¡ħè°·(0.09) | Ð²Ð¿Ð¾Ð»Ð½Ðµ(0.08)
    BOT:  å°±æŃ¤(-0.10) | åĨįåº¦(-0.09) | çİĭçĪ·(-0.09) | ying(-0.08) | atom(-0.08) | ties(-0.08) | éĺ¿(-0.08) | %(-0.08)
    ACCEPTED as axis_381  cumulative_var=0.5745

  [ 377]  axes=382  step_var=0.0017  binary_acc=0.992  gap=0.1718  max_dot=0.0040  (1.9s)
    TOP:  fare(0.10) | Ð´Ð¾ÑĢ(0.10) | çīĮåŃĲ(0.09) | åĬŁåĬĽ(0.09) | fw(0.09) | ambitious(0.08) | @@(0.08) | '.Ċ(0.08)
    BOT:  çº½çº¦(-0.10) | cite(-0.10) | åĲĮæĹ¶(-0.09) | å¨ħ(-0.09) | #include(-0.09) | lya(-0.09) | ï»¿namespace(-0.09) | standpoint(-0.09)
    ACCEPTED as axis_382  cumulative_var=0.5752

  [ 378]  axes=383  step_var=0.0016  binary_acc=0.994  gap=0.1625  max_dot=0.0141  (1.9s)
    TOP:  èħ°(0.12) | ç»ŃèĪª(0.10) | åı¯ä¾Ľ(0.10) | à¹ĥà¸«(0.10) | ":ĊĊ(0.09) | feel(0.09) | ä½łèĩªå·±(0.09) | à¸Ĳà¸²à¸Ļ(0.09)
    BOT:  æķĪæŀľåĽ¾(-0.08) | æįŁèĢĹ(-0.08) | è¿Ľåĩºåı£(-0.08) | æ®µæĹ¶éĹ´(-0.08) | Lamb(-0.08) | åħ§å®¹(-0.08) | æĢ(-0.08) | è¯´æĺİ(-0.08)
    ACCEPTED as axis_383  cumulative_var=0.5759

  [ 379]  axes=384  step_var=0.0017  binary_acc=0.987  gap=0.1677  max_dot=0.0025  (1.8s)
    TOP:  å¾ĺå¾Ĭ(0.09) | ä¼ģåĽ¾(0.09) | ÑģÑı(0.08) | [:,:(0.08) | ienza(0.08) | çĽ´èĩ³(0.08) | _=(0.08) | parliament(0.08)
    BOT:  å®¶è£ħ(-0.10) | åĪļæīį(-0.10) | åĵªäºĽ(-0.09) | æĭīå¼Ģ(-0.09) | çĽ®åīį(-0.09) | æĺĶæĹ¥(-0.09) | åİ¨æĪ¿(-0.09) | æĪĳå®¶(-0.09)
    ACCEPTED as axis_384  cumulative_var=0.5766

  [ 380]  axes=385  step_var=0.0017  binary_acc=0.998  gap=0.1701  max_dot=0.0019  (1.9s)
    TOP:  DX(0.10) | åģļå¥½(0.09) | basketball(0.09) | å»·(0.09) | ı(0.09) | reckon(0.09) | ä¹ĭéģĵ(0.08) | mph(0.08)
    BOT:  ä¸ĢåĪĩ(-0.10) | ...((-0.10) | .should(-0.10) | #{(-0.10) | ##(-0.09) | ãĤ¤(-0.09) | ä¸Ģè¡Į(-0.09) | ç½ĳç»ľ(-0.09)
    ACCEPTED as axis_385  cumulative_var=0.5773

  [ 381]  axes=386  step_var=0.0018  binary_acc=0.994  gap=0.1785  max_dot=0.0051  (1.8s)
    TOP:  è¦ģçĶ¨(0.10) | å°±æľī(0.10) | cline(0.09) | çĻ«çĹ«çĹħ(0.09) | ILED(0.09) | TABLE(0.09) | æĺ¯æľī(0.09) | âĢĶwith(0.09)
    BOT:  į(-0.10) | çłĮ(-0.10) | _errno(-0.09) | predators(-0.09) | è¤¶(-0.09) | dow(-0.09) | equipped(-0.08) | éĩįå¤į(-0.08)
    ACCEPTED as axis_386  cumulative_var=0.5781

  [ 382]  axes=387  step_var=0.0017  binary_acc=0.993  gap=0.1708  max_dot=0.0016  (1.9s)
    TOP:  .AttributeSet(0.09) | ALL(0.09) | æ¦Ĥå¿µèĤ¡(0.09) | .getS(0.09) | æ¾İ(0.09) | è¿Ļä¸Ģ(0.09) | åĪĿè¡·(0.08) | å¹¼ç¨ļ(0.08)
    BOT:  rogram(-0.10) | programa(-0.09) | æĭ¿çĿĢ(-0.09) | æĸĩä½ĵ(-0.09) | ilig(-0.09) | æ¸ħæ¥ļ(-0.09) | ä¸Ģåī¯(-0.09) | à¤ķ(-0.08)
    ACCEPTED as axis_387  cumulative_var=0.5788

  [ 383]  axes=388  step_var=0.0017  binary_acc=0.990  gap=0.1689  max_dot=0.0106  (1.9s)
    TOP:  agi(0.10) | ç½ĳçº¦(0.09) | âĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶ(0.09) | å¤§åİħ(0.09) | ------------------------------------------------------------------------------------------------(0.09) | éĺģ(0.09) | youre(0.09) | )`(0.09)
    BOT:  ÑıÑī(-0.10) | åıįå¼¹(-0.09) | ospels(-0.09) | segÃºn(-0.09) | æµģè¡Į(-0.08) | æĹ¶é«¦(-0.08) | à¹īà¸Ńà¸Ļ(-0.08) | atravÃ©s(-0.08)
    ACCEPTED as axis_388  cumulative_var=0.5796

  [ 384]  axes=389  step_var=0.0017  binary_acc=0.997  gap=0.1689  max_dot=0.0027  (1.9s)
    TOP:  Ø§ÛĮ(0.11) | ÛĮ(0.09) | âĢĶI(0.09) | å®Į(0.09) | ÐĿÐĺ(0.09) | æį®æŃ¤(0.09) | ONE(0.09) | (I(0.09)
    BOT:  ä»Ģä¹Īæĺ¯(-0.10) | å°ıåŀĭ(-0.09) | Blog(-0.09) | Cir(-0.09) | compliant(-0.08) | æľ¬å¸Ĥ(-0.08) | een(-0.08) | å°ĳå¥³(-0.08)
    ACCEPTED as axis_389  cumulative_var=0.5803

  [ 385]  axes=390  step_var=0.0017  binary_acc=0.982  gap=0.1661  max_dot=0.0076  (1.8s)
    TOP:  çĲı(0.09) | ATS(0.09) | Freedom(0.09) | :[(0.09) | orestation(0.09) | æµ·æĭĶ(0.08) | ä»£è¡¨(0.08) | æīĵçĿĢ(0.08)
    BOT:  æľīéĻĲ(-0.10) | å¸¦æĿ¥æĽ´å¤ļ(-0.09) | ä¾ĽéĶĢ(-0.09) | ser(-0.09) | ä¸ºçĽ®æłĩ(-0.08) | è¿ĺæĺ¯å¾Ī(-0.08) | è¿ĺ(-0.08) | model(-0.08)
    ACCEPTED as axis_390  cumulative_var=0.5810

  [ 386]  axes=391  step_var=0.0018  binary_acc=0.995  gap=0.1683  max_dot=0.0064  (1.8s)
    TOP:  çŃı(0.11) | kr(0.09) | Fragment(0.09) | "{(0.09) | '%(0.09) | æŀģåħ¶(0.09) | åĽº(0.09) | {(0.09)
    BOT:  è°ģçŁ¥éģĵ(-0.09) | Ð¶ÐµÐ½(-0.08) | afternoon(-0.08) | åįĸå®¶(-0.08) | OA(-0.08) | hÆ¡n(-0.08) | é¤Ĳ(-0.08) | çŃīäºİ(-0.08)
    ACCEPTED as axis_391  cumulative_var=0.5817

  [ 387]  axes=392  step_var=0.0017  binary_acc=0.954  gap=0.1674  max_dot=0.0033  (1.9s)
    TOP:  ¯(0.10) | åĲİåı°(0.09) | åĹħ(0.09) | ---(0.09) | ---(0.09) | ìļ°(0.09) | ãĤĪãģĨ(0.09) | plum(0.08)
    BOT:  SU(-0.09) | åĩºè¡Į(-0.09) | æľĢå¼º(-0.09) | ä¿¡è´·(-0.09) | åĲī(-0.09) | è¾ĥå·®(-0.09) | SU(-0.08) | æ³¨æĺİæĿ¥æºĲ(-0.08)
    ACCEPTED as axis_392  cumulative_var=0.5824

  [ 388]  axes=393  step_var=0.0017  binary_acc=0.978  gap=0.1670  max_dot=0.0077  (1.8s)
    TOP:  ÑĢÑıÐ´(0.10) | èªī(0.10) | ç¬Ļ(0.09) | é£ŁåłĤ(0.09) | entrÃ©e(0.09) | äºĭåħĪ(0.09) | system(0.09) | åĦĦ(0.09)
    BOT:  /)(-0.10) | çļĦéĩįè¦ģ(-0.10) | ×¤×¨(-0.10) | äººä»¬å¯¹(-0.10) | åı¯æĢķ(-0.09) | ÙĦÙĩØ°Ø§(-0.09) | lapse(-0.09) | andid(-0.09)
    ACCEPTED as axis_393  cumulative_var=0.5831

  [ 389]  axes=394  step_var=0.0017  binary_acc=0.980  gap=0.1635  max_dot=0.0109  (1.9s)
    TOP:  å½ĵåīįä½įç½®(0.10) | åıĹè®¿èĢħ(0.09) | ä¸¥æł¼èĲ½å®ŀ(0.09) | è´¸æĺĵ(0.09) | åıĬä»¥ä¸Ĭ(0.09) | æĬĢèĥ½(0.09) | å¯¹çħ§(0.09) | èĢĮä¸į(0.08)
    BOT:  åħ(-0.09) | çŃ±(-0.09) | .After(-0.09) | æ¸(-0.08) | .--(-0.08) | ä»ĬæĻļ(-0.08) | ä¸°(-0.08) | call(-0.08)
    ACCEPTED as axis_394  cumulative_var=0.5838

  [ 390]  axes=395  step_var=0.0016  binary_acc=0.983  gap=0.1593  max_dot=0.0037  (1.8s)
    TOP:  always(0.10) | è§¦åıĬ(0.09) | æ¯«ç±³(0.09) | å±(0.09) | ç¼ĿéļĻ(0.08) | å¹(0.08) | ded(0.08) | å¦Ĥ(0.08)
    BOT:  æĦŁå®ĺ(-0.09) | ä½łåĸľæ¬¢(-0.09) | CMP(-0.09) | Vincent(-0.09) | æĸ°æµª(-0.09) | ĊĊĊĊĊĊĊĊĊĊĊĊ(-0.09) | @SpringBootTest(-0.09) | ĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊĊ(-0.08)
    ACCEPTED as axis_395  cumulative_var=0.5845

  [ 391]  axes=396  step_var=0.0017  binary_acc=0.985  gap=0.1649  max_dot=0.0008  (1.8s)
    TOP:  baugh(0.10) | rather(0.09) | ^^(0.09) | ãĤ¤ãĥ¤(0.08) | (âĪ(0.08) | justified(0.08) | compared(0.08) | ï¿ł(0.08)
    BOT:  »(-0.11) | æĪĳåľ¨(-0.09) | åĲ¬è¯´(-0.09) | cwd(-0.09) | åĲ¬å¾Ĺ(-0.08) | å°±ç®Ĺ(-0.08) | ÑģÐ¿Ð¸ÑģÐ¾Ðº(-0.08) | æĶ¾åĪ°(-0.08)
    ACCEPTED as axis_396  cumulative_var=0.5852

  [ 392]  axes=397  step_var=0.0017  binary_acc=0.968  gap=0.1683  max_dot=0.0010  (1.8s)
    TOP:  éĥ½æĹłæ³ķ(0.11) | æķ¢(0.11) | æľŁæľĽ(0.11) | åı¯(0.11) | æĪĪ(0.10) | omedical(0.10) | èİ«(0.09) | è¿(0.09)
    BOT:  éĢłèĪ¹(-0.09) | åįģä¹Ŀ(-0.09) | Äĳá»§(-0.09) | {-(-0.09) | minist(-0.08) | IN(-0.08) | éĻķè¥¿(-0.08) | Under(-0.08)
    ACCEPTED as axis_397  cumulative_var=0.5859

  [ 393]  axes=398  step_var=0.0018  binary_acc=0.998  gap=0.1709  max_dot=0.0026  (1.8s)
    TOP:  åĩıéĢŁ(0.10) | saw(0.09) | Âł(0.09) | çĶ²(0.08) | åĶ¯ç¾İ(0.08) | ql(0.08) | ÄĹ(0.08) | tons(0.08)
    BOT:  indent(-0.10) | CUR(-0.10) | æĢ¼(-0.09) | zoals(-0.09) | mate(-0.09) | ä¸ĢçĻ¾(-0.09) | ä¸ĳ(-0.08) | èĮĥçķ´(-0.08)
    ACCEPTED as axis_398  cumulative_var=0.5866

  [ 394]  axes=399  step_var=0.0017  binary_acc=0.986  gap=0.1629  max_dot=0.0026  (1.8s)
    TOP:  empresas(0.09) | mq(0.09) | èĢĥèĻĳ(0.08) | enfrent(0.08) | è§¦åıĬ(0.08) | âĢ¦(0.08) | æĿĳ(0.08) | ÑħÐ¾Ð·Ñı(0.08)
    BOT:  æī°ä¹±(-0.10) | ><(-0.09) | orth(-0.09) | å¦¹åŃĲ(-0.09) | éĽĨç»ĵ(-0.09) | ä¸Ńåħ±ä¸Ńå¤®(-0.08) | /her(-0.08) | ä½¿æĪĳ(-0.08)
    ACCEPTED as axis_399  cumulative_var=0.5873

  [ 395]  axes=400  step_var=0.0017  binary_acc=0.987  gap=0.1650  max_dot=0.0074  (1.8s)
    TOP:  æ¤Ĵ(0.10) | -%(0.09) | ï»¿using(0.09) | Ð¾ÑģÑĮ(0.09) | æĺ¯çľŁçļĦ(0.08) | è§Ħæ¨¡æľĢå¤§(0.08) | åĲĮå¿ĥ(0.08) | éļ¶å±ŀ(0.08)
    BOT:  courses(-0.10) | alli(-0.09) | lst(-0.09) | Ð»Ð¸ÑĨ(-0.09) | âĢĶand(-0.09) | accessor(-0.08) | Ð³Ð¸(-0.08) | âĨ(-0.08)
    ACCEPTED as axis_400  cumulative_var=0.5880

  [ 396]  axes=401  step_var=0.0016  binary_acc=0.977  gap=0.1606  max_dot=0.0026  (1.8s)
    TOP:  ./(0.09) | .P(0.09) | ãĤ¤(0.09) | .Equal(0.09) | à¹īà¸§(0.08) | é¬ĥ(0.08) | tribute(0.08) | Ð»(0.08)
    BOT:  atura(-0.09) | æĳĶ(-0.09) | ĉtry(-0.09) | rather(-0.09) | ::(-0.09) | %(-0.08) | tend(-0.08) | rather(-0.08)
    ACCEPTED as axis_401  cumulative_var=0.5887

  [ 397]  axes=402  step_var=0.0017  binary_acc=0.990  gap=0.1631  max_dot=0.0015  (1.9s)
    TOP:  Ø¯(0.09) | \xb(0.09) | Ãģ(0.08) | ä»ĺ(0.08) | MG(0.08) | ãĥ¡(0.08) | OU(0.08) | ãĥİ(0.08)
    BOT:  å®½(-0.09) | ä¸¤çº§(-0.09) | ä¸ĬéĹ¨(-0.09) | ç©ºè°ĥ(-0.09) | ',)Ċ(-0.09) | è¯´è¿ĩ(-0.09) | refuse(-0.09) | åĲĪéĢĤçļĦ(-0.08)
    ACCEPTED as axis_402  cumulative_var=0.5893

  [ 398]  axes=403  step_var=0.0016  binary_acc=0.998  gap=0.1617  max_dot=0.0020  (1.8s)
    TOP:  ç´ł(0.09) | æĺ¯ä¸ºäºĨ(0.09) | æĿĥ(0.08) | (document(0.08) | åŁºéĩĳä¼ļ(0.08) | ç³»ç»ŁçļĦ(0.08) | åŁºåĽł(0.08) | Corn(0.08)
    BOT:  åľ¨æĪĳ(-0.10) | è¿(-0.10) | ÃŃf(-0.09) | ä¸Ĭ(-0.09) | ä¸įä¸Ĭ(-0.09) | ÑıÑī(-0.09) | å½ĵå¹´(-0.09) | åĽ(-0.09)
    ACCEPTED as axis_403  cumulative_var=0.5900

  [ 399]  axes=404  step_var=0.0017  binary_acc=0.999  gap=0.1641  max_dot=0.0029  (1.9s)
    TOP:  åįģå¤§(0.11) | æĬĬ(0.10) | æĹ©æĹ¥(0.09) | é«ĺ(0.09) | åį°åıĳ(0.09) | []((0.09) | æ¸(0.09) | When(0.09)
    BOT:  Ĥ¨(-0.10) | è¿Ĳ(-0.09) | çĿ¡çľł(-0.09) | ìŀ¬(-0.08) | chronic(-0.08) | åĲĪä½ľä¼Ļä¼´(-0.08) | èĤº(-0.08) | HuffPost(-0.08)
    ACCEPTED as axis_404  cumulative_var=0.5907

  [ 400]  axes=405  step_var=0.0017  binary_acc=0.973  gap=0.1651  max_dot=0.0052  (1.9s)
    TOP:  é©¼(0.09) | è¯«(0.09) | ä»²è£ģ(0.09) | .reflect(0.09) | éĺ²èħĲ(0.09) | @d(0.09) | æĸ°é²ľ(0.09) | éª¨éª¼(0.09)
    BOT:  ogen(-0.09) | ìĹ¬(-0.09) | -right(-0.09) | emp(-0.09) | æ²ĵ(-0.09) | idea(-0.09) | mÃ£i(-0.09) | èįī(-0.08)
    ACCEPTED as axis_405  cumulative_var=0.5914

  [ 401]  axes=406  step_var=0.0016  binary_acc=0.983  gap=0.1613  max_dot=0.0040  (1.8s)
    TOP:  å¼¹(0.10) | -f(0.09) | las(0.09) | siÄĻ(0.09) | ä¹ħ(0.09) | åĲ¬è¯´(0.08) | nag(0.08) | el(0.08)
    BOT:  triá»ĩu(-0.09) | èĭ¥å¹²(-0.08) | æ³¨(-0.08) | çħ¤çŁ¿(-0.08) | è¿ĻäºĽ(-0.08) | è¿Ļå¥Ĺ(-0.08) | dedicated(-0.08) | è¿Ļç§į(-0.08)
    ACCEPTED as axis_406  cumulative_var=0.5920

  [ 402]  axes=407  step_var=0.0016  binary_acc=0.977  gap=0.1614  max_dot=0.0013  (1.9s)
    TOP:  cpp(0.09) | çļĦæĸ¹å¼ı(0.09) | .ComponentModel(0.09) | IT(0.08) | ITS(0.08) | entirely(0.08) | ç«ĭåį³(0.08) | è¿İåĲĪ(0.08)
    BOT:  ç¤¾ä¿Ŀ(-0.09) | åī¯ä¸»å¸Ń(-0.09) | _empty(-0.08) | ç©ºè°ĥ(-0.08) | çķ¥(-0.08) | å®¶åħ·(-0.08) | ä¼¸(-0.08) | èºħ(-0.08)
    ACCEPTED as axis_407  cumulative_var=0.5927

  [ 403]  axes=408  step_var=0.0016  binary_acc=0.996  gap=0.1578  max_dot=0.0052  (1.9s)
    TOP:  ç¼º(0.09) | æķ´ä¸ª(0.09) | æīĵåħ¥(0.09) | uje(0.08) | åıĥ(0.08) | ÑĶ(0.08) | åıªæĺ¯ä¸Ģä¸ª(0.08) | cross(0.08)
    BOT:  éĿ¢èĨľ(-0.09) | }//(-0.09) | weniger(-0.09) | iges(-0.09) | åĹĵåŃĲ(-0.09) | gÃ¬(-0.09) | less(-0.09) | ï¼ľ(-0.08)
    ACCEPTED as axis_408  cumulative_var=0.5933

  [ 404]  axes=409  step_var=0.0016  binary_acc=0.969  gap=0.1628  max_dot=0.0060  (1.9s)
    TOP:  Summer(0.09) | Summer(0.08) | power(0.08) | board(0.08) | Biology(0.08) | ä¸ªåĪ«(0.08) | bah(0.08) | .aut(0.08)
    BOT:  platforms(-0.09) | Usa(-0.09) | èĩª(-0.09) | PRE(-0.08) | åĮºåĪĨ(-0.08) | earning(-0.08) | ç©ºæ°Ķè´¨éĩı(-0.08) | åİ»(-0.08)
    ACCEPTED as axis_409  cumulative_var=0.5940

  [ 405]  axes=410  step_var=0.0016  binary_acc=0.994  gap=0.1606  max_dot=0.0008  (1.9s)
    TOP:  ŀ(0.10) | Bye(0.09) | ë³¸(0.09) | æĪĳè¦ģ(0.09) | æĿĥçĽĬ(0.09) | æ°¸(0.09) | æĪĸæĺ¯(0.08) | éĥ¨ä¸»ä»»(0.08)
    BOT:  èĢħçļĦ(-0.10) | .How(-0.09) | ä¹ĭ(-0.09) | åĲĳå¤ĸ(-0.09) | å½ĴæĿ¥(-0.09) | èĢħ(-0.09) | .pack(-0.09) | ç½ĳçº¦è½¦(-0.08)
    ACCEPTED as axis_410  cumulative_var=0.5947

  [ 406]  axes=411  step_var=0.0016  binary_acc=0.963  gap=0.1625  max_dot=0.0025  (1.9s)
    TOP:  hiá»ĩn(0.10) | äº§ä¸ļ(0.09) | olut(0.09) | {čĊ(0.09) | icy(0.09) | ÐµÑĤ(0.09) | claro(0.09) | Phar(0.08)
    BOT:  åĲİåĨį(-0.09) | æĹ¶ä¸į(-0.09) | å¤§éģĵ(-0.09) | å®ŀæĻ¯(-0.09) | è½»æĿ¾(-0.09) | "@(-0.09) | ä¸ĵçıŃ(-0.08) | "></(-0.08)
    ACCEPTED as axis_411  cumulative_var=0.5953

  [ 407]  axes=412  step_var=0.0016  binary_acc=0.979  gap=0.1577  max_dot=0.0025  (1.9s)
    TOP:  æ´Ľ(0.09) | åĽŃ(0.08) | èĴĤ(0.08) | âĢĿãĢĤ(0.08) | those(0.08) | æ·±è¿ľ(0.08) | Ð¼Ð¾(0.08) | "This(0.08)
    BOT:  ¦(-0.10) | çķĻæĦı(-0.09) | åĨįæĿ¥(-0.09) | åľ¨(-0.09) | å°ıå¿ĥ(-0.09) | ÃĲ(-0.09) | ä¸ĢçĽ´åľ¨(-0.09) | è¿ĩæĿ¥(-0.09)
    ACCEPTED as axis_412  cumulative_var=0.5960

  [ 408]  axes=413  step_var=0.0016  binary_acc=0.975  gap=0.1619  max_dot=0.0044  (1.8s)
    TOP:  è¿Ľåĩºåı£(0.09) | æĳĦå½±(0.09) | cs(0.09) | çŁ³çģ°(0.08) | æ¢ħè¥¿(0.08) | spins(0.08) | çĿ¡åīį(0.08) | ç«¥(0.08)
    BOT:  ukt(-0.10) | .O(-0.09) | expressed(-0.08) | Internet(-0.08) | ä»İä¸Ń(-0.08) | orge(-0.08) | Internet(-0.08) | .De(-0.08)
    ACCEPTED as axis_413  cumulative_var=0.5966

  [ 409]  axes=414  step_var=0.0016  binary_acc=0.990  gap=0.1628  max_dot=0.0047  (1.8s)
    TOP:  å¤ļä¹ħ(0.10) | åĲĿ(0.10) | çģ¯(0.10) | å½»(0.09) | Proto(0.09) | çģ¯åħ·(0.08) | å±±åĿ¡(0.08) | éĴ±(0.08)
    BOT:  righteous(-0.09) | droits(-0.09) | Miss(-0.08) | .volley(-0.08) | æįŁåĿı(-0.08) | èĭ¥è¦ģ(-0.08) | å¯ĨåĪĩ(-0.08) | Write(-0.08)
    ACCEPTED as axis_414  cumulative_var=0.5973

  [ 410]  axes=415  step_var=0.0017  binary_acc=0.962  gap=0.1633  max_dot=0.0014  (1.8s)
    TOP:  Turkish(0.11) | Reviews(0.09) | çłĶåĪ¤(0.09) | inyin(0.09) | æĸĩæ¡Ī(0.09) | ä½³(0.09) | éĺ³åı°(0.09) | ãģĶ(0.09)
    BOT:  ì¦(-0.10) | çľĭåĩº(-0.09) | è®©ä»ĸä»¬(-0.09) | å°Ĩç»§ç»Ń(-0.08) | gentlemen(-0.08) | åķ¤(-0.08) | æ¿Ģæĥħ(-0.08) | Kamp(-0.08)
    ACCEPTED as axis_415  cumulative_var=0.5980

  [ 411]  axes=416  step_var=0.0016  binary_acc=0.996  gap=0.1570  max_dot=0.0039  (1.9s)
    TOP:  ä¼ĺè´¨(0.10) | aren(0.09) | votre(0.09) | Â(0.09) | ä»ĭåħ¥(0.08) | Ð¾ÐºÐ¾Ð»Ð¾(0.08) | Materials(0.08) | åį(0.08)
    BOT:  actually(-0.09) | æĶ»åĿļæĪĺ(-0.09) | ä»İä¸¥æ²»åħļ(-0.08) | è´§å¸ģæĶ¿çŃĸ(-0.08) | dumpsters(-0.08) | è¡ĮæĶ¿éĥ¨éĹ¨(-0.08) | åį«åģ¥å§Ķ(-0.08) | enthusiast(-0.08)
    ACCEPTED as axis_416  cumulative_var=0.5986

  [ 412]  axes=417  step_var=0.0016  binary_acc=0.992  gap=0.1584  max_dot=0.0030  (1.8s)
    TOP:  äº²èº«(0.09) | imilar(0.09) | chang(0.09) | undis(0.09) | .ServletException(0.08) | æĪĳéĥ½(0.08) | benefits(0.08) | isme(0.08)
    BOT:  æ»¿(-0.09) | å®ŀè¡Į(-0.09) | æĮīçħ§(-0.09) | ~~~~(-0.09) | Matcher(-0.09) | ä½ľä¸º(-0.08) | ĊĊ(-0.08) | gni(-0.08)
    ACCEPTED as axis_417  cumulative_var=0.5992

  [ 413]  axes=418  step_var=0.0016  binary_acc=0.999  gap=0.1583  max_dot=0.0011  (1.9s)
    TOP:  went(0.09) | æĪĳçİ°åľ¨(0.09) | å®ŀä¸ļ(0.09) | ='-(0.08) | æľīäºĽäºº(0.08) | å»ºè®¾(0.08) | æĺ¼(0.08) | ç»ıè´¸(0.08)
    BOT:  ÃĬ(-0.09) | Whether(-0.09) | âĢĭ(-0.09) | ä¸Ģè¶Ł(-0.09) | .it(-0.09) | Ã¡s(-0.09) | éĢłæĪĲçļĦ(-0.08) | .<(-0.08)
    ACCEPTED as axis_418  cumulative_var=0.5999

  [ 414]  axes=419  step_var=0.0016  binary_acc=0.995  gap=0.1605  max_dot=0.0049  (1.8s)
    TOP:  Portug(0.09) | ãĤ¤(0.09) | faut(0.09) | åĽ¾(0.08) | æĦĽ(0.08) | ç§ģèĲ¥(0.08) | Ã´(0.08) | çºªå§ĶçĽĳå§Ķ(0.08)
    BOT:  ãĢĳãĢĲ(-0.10) | é«ĺéĽĦ(-0.10) | ê²½ìļ°(-0.09) | åı¯èĥ½æĺ¯(-0.09) | ]['(-0.09) | _q(-0.09) | Aid(-0.08) | å·²ç»ıæĺ¯(-0.08)
    ACCEPTED as axis_419  cumulative_var=0.6005

  [ 415]  axes=420  step_var=0.0015  binary_acc=0.977  gap=0.1558  max_dot=0.0003  (1.8s)
    TOP:  've(0.10) | Ø§ÙĦÙĩ(0.10) | á(0.09) | ä¿®è®¢(0.09) | æĪİ(0.08) | æĢ»è£ģ(0.08) | èĩªèĲ¥(0.08) | åĮĨåĮĨ(0.08)
    BOT:  !*(-0.09) | ///(-0.09) | "((-0.09) | KDE(-0.08) | Src(-0.08) | Radio(-0.08) | butter(-0.08) | è°ĥæķ´(-0.08)
    ACCEPTED as axis_420  cumulative_var=0.6011

  [ 416]  axes=421  step_var=0.0016  binary_acc=0.980  gap=0.1608  max_dot=0.0052  (1.8s)
    TOP:  è±Ĩ(0.10) | æĪĳä»¬(0.08) | bor(0.08) | é¢Ĭ(0.08) | ÑĪÐºÐ¸(0.08) | _FUNC(0.08) | ç§Ģ(0.08) | Expo(0.08)
    BOT:  profession(-0.10) | Ú©(-0.09) | Explanation(-0.09) | ä¸įåĲĪ(-0.09) | GEN(-0.09) | :,(-0.08) | Craw(-0.08) | Ä°(-0.08)
    ACCEPTED as axis_421  cumulative_var=0.6017

  [ 417]  axes=422  step_var=0.0016  binary_acc=0.971  gap=0.1529  max_dot=0.0046  (1.8s)
    TOP:  ä¸įåĲĮ(0.09) | flower(0.09) | æľīæĹ¶(0.09) | ä½¿åĳ½(0.09) | äºĮåįģåĽĽ(0.08) | .Scanner(0.08) | ä¸įæĽ¾(0.08) | æĭĴä¸į(0.08)
    BOT:  çĵ·çłĸ(-0.08) | Being(-0.08) | dealership(-0.08) | çĶµåĬ¨è½¦(-0.08) | ab(-0.08) | ullah(-0.08) | è¿ŀå¿Ļ(-0.08) | wallpaper(-0.07)
    ACCEPTED as axis_422  cumulative_var=0.6024

  [ 418]  axes=423  step_var=0.0016  binary_acc=0.968  gap=0.1605  max_dot=0.0078  (1.8s)
    TOP:  macOS(0.09) | áº«u(0.09) | etch(0.09) | há»Ļi(0.09) | _assignment(0.09) | Ñģ(0.09) | router(0.08) | íļĮ(0.08)
    BOT:  éĺħè§Ī(-0.09) | which(-0.09) | Highlights(-0.08) | grind(-0.08) | çĥŃè¡Ģ(-0.08) | åĲĦåİ¿(-0.08) | åµĮ(-0.08) | Carnegie(-0.08)
    ACCEPTED as axis_423  cumulative_var=0.6030

  [ 419]  axes=424  step_var=0.0015  binary_acc=0.979  gap=0.1571  max_dot=0.0021  (1.9s)
    TOP:  GL(0.09) | sf(0.09) | Job(0.09) | Degree(0.08) | z(0.08) | Req(0.08) | Sz(0.08) | ä¿®æŃ£(0.08)
    BOT:  ç¼ĸåĨĻ(-0.11) | (my(-0.09) | è®°ä½ı(-0.09) | Gee(-0.08) | äºĶ(-0.08) | æĦıå¢ĥ(-0.08) | Every(-0.08) | Ã·(-0.08)
    ACCEPTED as axis_424  cumulative_var=0.6036

  [ 420]  axes=425  step_var=0.0016  binary_acc=0.992  gap=0.1591  max_dot=0.0106  (1.9s)
    TOP:  amongst(0.09) | Volley(0.08) | ä»£è¡¨åĽ¢(0.08) | ×Ļ×Ķ(0.08) | Em(0.08) | åĽ½åĢº(0.08) | å¤ªåŃĲ(0.08) | Learned(0.08)
    BOT:  æĥ³åĪ°(-0.08) | åįµå·¢(-0.08) | å¸Ń(-0.08) | .Minimum(-0.08) | lectric(-0.08) | åıĳçĶŁ(-0.08) | æĹħæ¸¸èµĦæºĲ(-0.08) | å®ļ(-0.08)
    ACCEPTED as axis_425  cumulative_var=0.6043

  [ 421]  axes=426  step_var=0.0017  binary_acc=0.970  gap=0.1643  max_dot=0.0030  (1.9s)
    TOP:  ä¸ºéĩįçĤ¹(0.10) | Ð´Ð¾(0.09) | Selector(0.09) | ÐºÐ»Ð¸(0.09) | ãģŁãģı(0.08) | è¦ģä¸įè¦ģ(0.08) | æĪĲ(0.08) | åĳ½åĲįä¸º(0.08)
    BOT:  Äĵ(-0.12) | ibus(-0.11) | Ĭ(-0.09) | yas(-0.09) | é¸¡æ±¤(-0.08) | ä¸ŃåĮ»(-0.08) | ioni(-0.08) | åĩłä¹İ(-0.08)
    ACCEPTED as axis_426  cumulative_var=0.6049

  [ 422]  axes=427  step_var=0.0016  binary_acc=0.990  gap=0.1558  max_dot=0.0053  (1.8s)
    TOP:  nl(0.09) | "\"(0.09) | åħ¶(0.08) | ç²¾ç¡®(0.08) | æķ¦(0.08) | endent(0.08) | ç¬¦(0.07) | æŁ¥è©¢(0.07)
    BOT:  èĦ¸ä¸Ĭ(-0.10) | èµ·äºĨ(-0.09) | å¹²(-0.09) | congen(-0.08) | trading(-0.08) | æĥ³åĪ°(-0.08) | ä½Ľå±±(-0.08) | à¸¡(-0.08)
    ACCEPTED as axis_427  cumulative_var=0.6055

  [ 423]  axes=428  step_var=0.0016  binary_acc=0.978  gap=0.1547  max_dot=0.0014  (1.8s)
    TOP:  æīŃè½¬(0.09) | Â»(0.09) | åĪĨäº«(0.08) | organisms(0.08) | khÃ³a(0.08) | ç¼Ŀ(0.08) | belum(0.08) | ç»´æĮģ(0.08)
    BOT:  Sport(-0.09) | ç«ŀ(-0.09) | èĬĿéº»(-0.08) | .do(-0.08) | å¤ļå°ĳ(-0.08) | èº«é«ĺ(-0.08) | çľĭåĩº(-0.08) | ï¼ļ(-0.08)
    ACCEPTED as axis_428  cumulative_var=0.6061

  [ 424]  axes=429  step_var=0.0016  binary_acc=0.974  gap=0.1566  max_dot=0.0073  (1.8s)
    TOP:  åīµ(0.09) | çªĹå¤ĸ(0.08) | (The(0.08) | "-(0.08) | åĳ¨åħŃ(0.08) | táº¥t(0.08) | Hon(0.08) | .weights(0.08)
    BOT:  ä¸įåģľ(-0.09) | itis(-0.09) | æĺ¾å¾Ĺ(-0.09) | is(-0.09) | å±ŀäºİ(-0.09) | å¯¹çĿĢ(-0.08) | å¾Īæĺ¯(-0.08) | .<(-0.08)
    ACCEPTED as axis_429  cumulative_var=0.6068

  [ 425]  axes=430  step_var=0.0016  binary_acc=0.984  gap=0.1565  max_dot=0.0077  (1.9s)
    TOP:  äººæµģ(0.09) | ÐºÐ°ÐºÐ¸Ðµ(0.09) | vector(0.09) | åĨħåĪĨæ³Į(0.09) | weight(0.09) | natural(0.08) | æĪĴ(0.08) | è¯¯å·®(0.08)
    BOT:  .âĢĵ(-0.10) | åľĥ(-0.10) | &P(-0.09) | (--(-0.09) | .in(-0.09) | constantly(-0.09) | FO(-0.09) | æĬ¤å£«(-0.08)
    ACCEPTED as axis_430  cumulative_var=0.6074

  [ 426]  axes=431  step_var=0.0016  binary_acc=0.973  gap=0.1629  max_dot=0.0053  (1.9s)
    TOP:  å¦Ĥæŀľä¸į(0.09) | à¹ģà¸ķ(0.09) | ç©¿æĲŃ(0.09) | æĭ¿(0.09) | çĶ·åŃĲ(0.09) | includes(0.09) | Owner(0.09) | è¿Ļä¸Ģ(0.09)
    BOT:  é¢Ĩè¡Ķ(-0.10) | ÃŃnh(-0.09) | åīĬåĩı(-0.09) | æ§İ(-0.09) | è¯ĺ(-0.08) | ë¡Ģ(-0.08) | à¸·à¸Ļ(-0.08) | å®īå®ļ(-0.08)
    ACCEPTED as axis_431  cumulative_var=0.6080

  [ 427]  axes=432  step_var=0.0016  binary_acc=0.996  gap=0.1565  max_dot=0.0115  (1.9s)
    TOP:  worksheets(0.10) | [:(0.10) | Zealand(0.10) | å±ŀ(0.09) | éĩįçİ°(0.09) | æĹ¶æĹ¶(0.09) | åģļåĩº(0.08) | ä¸įä¸Ĭ(0.08)
    BOT:  Wi(-0.09) | Î(-0.09) | Ñ(-0.09) | åŁºå±Ĥ(-0.08) | BD(-0.08) | store(-0.08) | $\(-0.08) | Lim(-0.08)
    ACCEPTED as axis_432  cumulative_var=0.6087

  [ 428]  axes=433  step_var=0.0016  binary_acc=0.985  gap=0.1595  max_dot=0.0031  (1.9s)
    TOP:  å¹³å¸¸(0.08) | Ð½Ð¾ÑģÑĤ(0.08) | è½¬è¿Ĳ(0.08) | è´Ńçī©(0.08) | ä¸Ńéĵģ(0.08) | +](0.08) | itere(0.08) | å®(0.08)
    BOT:  their(-0.09) | (v(-0.09) | ç«ĭåĪ»(-0.09) | (class(-0.09) | .This(-0.09) | ('\\(-0.09) | ãĤµ(-0.08) | .preventDefault(-0.08)
    ACCEPTED as axis_433  cumulative_var=0.6093

  [ 429]  axes=434  step_var=0.0016  binary_acc=0.998  gap=0.1566  max_dot=0.0019  (1.8s)
    TOP:  WebDriver(0.09) | å¼Ħ(0.09) | >"(0.09) | yg(0.09) | ä¸Ĭ(0.08) | è²´(0.08) | ï¼Ĥ(0.08) | Living(0.08)
    BOT:  Introduced(-0.08) | é¦ĸåħĪ(-0.08) | âĩĴ(-0.08) | Experience(-0.08) | ::{(-0.08) | ä¼ĺåħĪ(-0.08) | ]+)/(-0.08) | heed(-0.08)
    ACCEPTED as axis_434  cumulative_var=0.6099

  [ 430]  axes=435  step_var=0.0016  binary_acc=0.980  gap=0.1595  max_dot=0.0044  (1.8s)
    TOP:  é»ĺ(0.10) | åŁ¹(0.09) | ä¹ŁæŃ£æĺ¯(0.09) | å¹³(0.09) | âĢľWhen(0.08) | æŁĶ(0.08) | those(0.08) | æŃĮ(0.08)
    BOT:  ä¿¡éĺ³(-0.08) | çĤķ(-0.08) | ä¿©(-0.08) | ãĥįãĥĥãĥĪ(-0.08) | Äĩ(-0.08) | çĩĬ(-0.08) | inn(-0.08) | GED(-0.08)
    ACCEPTED as axis_435  cumulative_var=0.6105

  [ 431]  axes=436  step_var=0.0016  binary_acc=0.977  gap=0.1583  max_dot=0.0020  (1.8s)
    TOP:  æĸ(0.09) | Ft(0.09) | æ§½(0.08) | æĥ¯ä¾ĭ(0.08) | q(0.08) | ê(0.08) | Î·(0.08) | è§Ĵ(0.08)
    BOT:  said(-0.10) | èªª(-0.10) | é«ĺçŃī(-0.09) | çŃīåĨħå®¹(-0.09) | è®²(-0.09) | æĪĳè¯´(-0.09) | åĲĮæĻĤ(-0.08) | ---(-0.08)
    ACCEPTED as axis_436  cumulative_var=0.6111

  [ 432]  axes=437  step_var=0.0015  binary_acc=0.994  gap=0.1573  max_dot=0.0102  (1.9s)
    TOP:  éķ¿æķĪ(0.09) | ç»ŁçŃ¹æİ¨è¿Ľ(0.09) | dismal(0.09) | Impact(0.09) | ],[(0.09) | .grid(0.08) | ä¸Ģäºº(0.08) | å®ŀéĻħ(0.08)
    BOT:  /(-0.08) | ×Ķ(-0.08) | Ñķ(-0.08) | techn(-0.08) | åĸľæ¬¢(-0.08) | æĬ¹(-0.08) | Ð³(-0.08) | âĦĵ(-0.07)
    ACCEPTED as axis_437  cumulative_var=0.6117

  [ 433]  axes=438  step_var=0.0016  binary_acc=0.977  gap=0.1562  max_dot=0.0077  (1.9s)
    TOP:  çģ¾(0.09) | æİ¨çĲĨ(0.09) | å®£åĳĬ(0.09) | æ°´è´¨(0.09) | (čĊ(0.09) | ï¬Ģ(0.09) | æĦıåĽ¾(0.08) | åı¯éĿł(0.08)
    BOT:  rails(-0.08) | accents(-0.08) | -str(-0.08) | æĢİéº¼(-0.08) | ä¸įå¯¹(-0.08) | ä¸ĸçķĮç¬¬ä¸Ģ(-0.07) | áº¡c(-0.07) | -vers(-0.07)
    ACCEPTED as axis_438  cumulative_var=0.6123

  [ 434]  axes=439  step_var=0.0015  binary_acc=0.982  gap=0.1516  max_dot=0.0070  (1.9s)
    TOP:  .Gen(0.10) | Gen(0.10) | åı¯è§Ĥ(0.09) | ä¸ĵä¸ļçļĦ(0.08) | æī§ä¸ļ(0.08) | rowth(0.08) | (State(0.08) | åģ¥åº·(0.08)
    BOT:  èī³(-0.08) | æĥ©(-0.08) | ÑģÑı(-0.08) | ÅĤa(-0.08) | /*(-0.08) | æĥ¹(-0.08) | not(-0.08) | ä»ĸ(-0.08)
    ACCEPTED as axis_439  cumulative_var=0.6129

  [ 435]  axes=440  step_var=0.0015  binary_acc=0.980  gap=0.1513  max_dot=0.0075  (1.9s)
    TOP:  ãģ«ãĤĤ(0.10) | é¢Ĳ(0.09) | éĹ«(0.08) | dapat(0.08) | åºĶæĶ¶(0.08) | /article(0.08) | []((0.08) | \[(0.08)
    BOT:  berg(-0.08) | ÑģÐ¾Ð½(-0.08) | etti(-0.08) | nett(-0.08) | .set(-0.08) | Lesser(-0.08) | âĢĻT(-0.08) | aland(-0.08)
    ACCEPTED as axis_440  cumulative_var=0.6135

  [ 436]  axes=441  step_var=0.0015  binary_acc=0.993  gap=0.1487  max_dot=0.0009  (1.8s)
    TOP:  \'(0.10) | ìķ(0.10) | tu(0.09) | éĴ¦(0.09) | PARTICULAR(0.08) | .a(0.08) | 's(0.08) | âĢĻ(0.08)
    BOT:  }((-0.09) | ä¼ª(-0.09) | ä½¿(-0.08) | ×ķ(-0.08) | than(-0.08) | pseudo(-0.08) | åį³(-0.08) | æĦŁåıĹåĪ°(-0.08)
    ACCEPTED as axis_441  cumulative_var=0.6141

  [ 437]  axes=442  step_var=0.0015  binary_acc=0.989  gap=0.1515  max_dot=0.0047  (1.8s)
    TOP:  "")(0.09) | çĶ«(0.09) | )\(0.09) | é¦ĸåħĪè¦ģ(0.08) | +"\(0.08) | ?a(0.08) | markets(0.08) | market(0.08)
    BOT:  Though(-0.08) | meant(-0.08) | æ¸į(-0.08) | -widget(-0.08) | handed(-0.08) | -id(-0.08) | Ð´ÐµÐºÐ°Ð±ÑĢÑı(-0.08) | paper(-0.08)
    ACCEPTED as axis_442  cumulative_var=0.6146

  [ 438]  axes=443  step_var=0.0016  binary_acc=0.959  gap=0.1556  max_dot=0.0112  (1.9s)
    TOP:  è¦ģåľ¨(0.10) | ('.(0.10) | chá»ī(0.09) | ><?(0.09) | Ø¹ÙĦÙī(0.09) | ']},Ċ(0.08) | -\(0.08) | prÃ¡(0.08)
    BOT:  ä¸ŃæĸŃ(-0.09) | hi(-0.08) | åİ¥(-0.08) | bene(-0.08) | SV(-0.08) | alcohol(-0.07) | requirements(-0.07) | pigeon(-0.07)
    ACCEPTED as axis_443  cumulative_var=0.6153

  [ 439]  axes=444  step_var=0.0015  binary_acc=1.000  gap=0.1511  max_dot=0.0063  (1.9s)
    TOP:  à¹ĥà¸Ļà¸ģà¸²à¸£(0.08) | äºĭè¿¹(0.08) | ãģ¾ãģļãģ¯(0.08) | .HttpSession(0.08) | Â«(0.08) | éĩĳèŀįæľºæŀĦ(0.08) | FX(0.08) | getValue(0.08)
    BOT:  åħļçļĦåįģä¹Ŀ(-0.09) | O(-0.08) | isi(-0.08) | 'u(-0.08) | add(-0.07) | END(-0.07) | ä»¥ä¸ĭåĩłä¸ª(-0.07) | '\(-0.07)
    ACCEPTED as axis_444  cumulative_var=0.6158

  [ 440]  axes=445  step_var=0.0016  binary_acc=0.999  gap=0.1541  max_dot=0.0039  (1.9s)
    TOP:  åĲ¾(0.10) | \'(0.09) | .\(0.09) | .'(0.08) | åĴ±(0.08) | olo(0.08) | ê°Ļ(0.08) | ------------(0.08)
    BOT:  è·¯æ¡¥(-0.09) | å½¢æĢģ(-0.09) | ãĤ·(-0.09) | #!(-0.08) | èĲ½å®ŀ(-0.08) | è·¯çº¿(-0.08) | Via(-0.08) | æ¥¼å±Ĥ(-0.08)
    ACCEPTED as axis_445  cumulative_var=0.6164

  [ 441]  axes=446  step_var=0.0015  binary_acc=0.964  gap=0.1527  max_dot=0.0043  (1.8s)
    TOP:  GET(0.08) | Has(0.08) | Pages(0.08) | gs(0.08) | .gov(0.08) | iliation(0.08) | .au(0.08) | _par(0.08)
    BOT:  should(-0.09) | è¿Ľåħ¥(-0.09) | ä¸įåı¯(-0.08) | è¿Ľè¡Į(-0.08) | ÐĴ(-0.08) | )}Ċ(-0.08) | },(-0.08) | èĥ½åĲ¦(-0.08)
    ACCEPTED as axis_446  cumulative_var=0.6170

  [ 442]  axes=447  step_var=0.0015  binary_acc=0.994  gap=0.1531  max_dot=0.0023  (1.8s)
    TOP:  æĪĳæ²¡(0.11) | ;</(0.09) | å°±ä¸į(0.09) | æĺ¯æ²¡æľī(0.09) | ****************************************************************(0.09) | é«ĺåº¦éĩįè§Ĩ(0.08) | è¦ģåĬłå¼º(0.08) | æ¯Ķè¾ĥå¥½(0.08)
    BOT:  y(-0.09) | el(-0.08) | Results(-0.08) | éĢĲå¹´(-0.08) | æģ¤(-0.08) | äººç¾¤ä¸Ń(-0.08) | -like(-0.08) | Projects(-0.08)
    ACCEPTED as axis_447  cumulative_var=0.6176

  [ 443]  axes=448  step_var=0.0015  binary_acc=0.986  gap=0.1489  max_dot=0.0060  (1.8s)
    TOP:  å½(0.09) | "<(0.08) | æĺ¯æľĢ(0.08) | Loc(0.08) | bf(0.08) | æĺ¯(0.08) | __(0.08) | Â±(0.08)
    BOT:  ãĤªãĥ³(-0.09) | è½¬è½½è¯·(-0.08) | _ON(-0.08) | Electronic(-0.08) | YY(-0.08) | é¢ģåıĳ(-0.08) | è¿ĳæľŁ(-0.08) | /*Ċ(-0.08)
    ACCEPTED as axis_448  cumulative_var=0.6182

  [ 444]  axes=449  step_var=0.0015  binary_acc=0.986  gap=0.1526  max_dot=0.0013  (1.9s)
    TOP:  çĤºäºĨ(0.08) | è¿Ľ(0.08) | è®(0.08) | didn(0.08) | /d(0.07) | ÃĦ(0.07) | à¹īà¸Ń(0.07) | ÙĭØ§(0.07)
    BOT:  èĦ²(-0.08) | (tree(-0.08) | èįĨ(-0.08) | ertype(-0.08) | åºµ(-0.08) | ç½®äºİ(-0.08) | é«ĺçŃīæķĻèĤ²(-0.08) | .css(-0.08)
    ACCEPTED as axis_449  cumulative_var=0.6188

  [ 445]  axes=450  step_var=0.0015  binary_acc=0.995  gap=0.1498  max_dot=0.0015  (1.9s)
    TOP:  è®®(0.09) | éĹ®åį·(0.08) | alias(0.08) | ç½ĳé¡µ(0.08) | ahan(0.08) | Apps(0.08) | ×Ļ×ĵ(0.08) | _validation(0.08)
    BOT:  è¿ĻéĩĮæĺ¯(-0.10) | å»ºæĿĲ(-0.08) | .up(-0.08) | -Za(-0.08) | OUTH(-0.08) | æĿ¡ä»¶(-0.08) | Li(-0.08) | ê²ł(-0.08)
    ACCEPTED as axis_450  cumulative_var=0.6193

  [ 446]  axes=451  step_var=0.0016  binary_acc=0.989  gap=0.1533  max_dot=0.0046  (1.8s)
    TOP:  å·²ç»ıåľ¨(0.10) | å·²(0.09) | å¿ħé¡»(0.09) | åĬ¡å¿ħ(0.09) | æĬ±æŃī(0.09) | æĽ¾ç»ı(0.08) | cÃ²n(0.08) | å¤Ħäºİ(0.08)
    BOT:  (-0.09) | ç¤¾ä¼ļç»Ħç»ĩ(-0.09) | (-0.08) | å¤ļæĸ¹(-0.08) | æ³Ħ(-0.08) | ÑĳÑĢ(-0.08) | (-0.08) | Org(-0.08)
    ACCEPTED as axis_451  cumulative_var=0.6199

  [ 447]  axes=452  step_var=0.0015  binary_acc=0.959  gap=0.1508  max_dot=0.0043  (1.8s)
    TOP:  PTR(0.10) | #ĊĊ(0.09) | è¿ĻéĩĮæĺ¯(0.08) | ÄĲ(0.08) | ç»ıèĲ¥æ´»åĬ¨(0.08) | æķ´ä¸ª(0.08) | collect(0.08) | vl(0.08)
    BOT:  ä½ľåĩº(-0.09) | ä¸įäºĨ(-0.08) | æĮī(-0.08) | _by(-0.08) | .softmax(-0.08) | fd(-0.07) | ulls(-0.07) | æķ°çĽ®(-0.07)
    ACCEPTED as axis_452  cumulative_var=0.6205

  [ 448]  axes=453  step_var=0.0015  binary_acc=0.999  gap=0.1484  max_dot=0.0017  (1.9s)
    TOP:  æĪĳä»¬è¦ģ(0.09) | åĶ¯(0.09) | åıªèĥ½(0.09) | âĢĶto(0.09) | lem(0.09) | åįĪåĲİ(0.08) | âĢĶthe(0.08) | åıªä¸º(0.08)
    BOT:  enie(-0.08) | Add(-0.08) | é©¬åħĭ(-0.08) | Má»Ļt(-0.08) | subset(-0.07) | äºĨå¤ļå°ĳ(-0.07) | mismo(-0.07) | Ð¿Ð°Ð»(-0.07)
    ACCEPTED as axis_453  cumulative_var=0.6211

  [ 449]  axes=454  step_var=0.0015  binary_acc=0.974  gap=0.1496  max_dot=0.0057  (1.8s)
    TOP:  çķĻå®Ī(0.08) | âĢĶif(0.08) | åĵªç§į(0.08) | âĢĵ(0.08) | different(0.08) | æ¼Ĥäº®(0.07) | ĵ(0.07) | âĢ¢(0.07)
    BOT:  _dims(-0.08) | çĪĨåıĳ(-0.08) | perl(-0.08) | (py(-0.08) | */ĊĊ(-0.08) | ///Ċ(-0.08) | ";ĊĊ(-0.08) | æĪĳåĴĮ(-0.07)
    ACCEPTED as axis_454  cumulative_var=0.6216

  [ 450]  axes=455  step_var=0.0015  binary_acc=1.000  gap=0.1526  max_dot=0.0026  (1.8s)
    TOP:  ÅĤo(0.10) | ãģł(0.09) | /K(0.09) | .k(0.09) | æĻ®éĢļ(0.09) | åį´æĺ¯(0.09) | ãģĻãģİ(0.09) | ofs(0.09)
    BOT:  æĮīçħ§(-0.09) | è¢«è¿«(-0.08) | anytime(-0.08) | åĪ©çĶ¨(-0.08) | áº¬(-0.08) | ;"Ċ(-0.08) | home(-0.08) | "{(-0.08)
    ACCEPTED as axis_455  cumulative_var=0.6222

  [ 451]  axes=456  step_var=0.0015  binary_acc=0.995  gap=0.1498  max_dot=0.0017  (1.8s)
    TOP:  vá»ģ(0.09) | æĤ£åĦ¿(0.09) | à¸§(0.09) | ÐµÐ¼(0.08) | åįµ(0.08) | readline(0.08) | Ø±(0.08) | assigns(0.08)
    BOT:  æħĪåĸĦ(-0.10) | Aboriginal(-0.09) | Kind(-0.09) | "((-0.08) | ãĥĳãĥ³(-0.08) | Gui(-0.08) | ä¸įæķ¢(-0.08) | æĪ¾(-0.07)
    ACCEPTED as axis_456  cumulative_var=0.6228

  [ 452]  axes=457  step_var=0.0015  binary_acc=0.946  gap=0.1476  max_dot=0.0019  (1.9s)
    TOP:  _training(0.10) | -induced(0.09) | .pt(0.08) | _timestamp(0.08) | à¸Ĥà¸Ńà¸ĩ(0.08) | _steps(0.08) | _until(0.08) | çº³ç±³(0.08)
    BOT:  ĉG(-0.09) | -----------(-0.08) | (-0.08) | [:(-0.08) | )">(-0.08) | è´«(-0.08) | esco(-0.08) | æĥł(-0.08)
    ACCEPTED as axis_457  cumulative_var=0.6234

  [ 453]  axes=458  step_var=0.0015  binary_acc=0.986  gap=0.1473  max_dot=0.0084  (1.8s)
    TOP:  )[-(0.09) | è¡¨éĿ¢(0.08) | Ã¥(0.08) | åĨįè¯´(0.08) | çĦ¶æĺ¯(0.08) | ä¸Ģéģĵ(0.08) | -Sep(0.08) | Î¦(0.08)
    BOT:  æ¢³(-0.08) | ä½łçļĦ(-0.08) | ä¼ĳéĹ²(-0.08) | _ipv(-0.08) | _RE(-0.07) | çĨĦ(-0.07) | ãģĺãĤĥãģªãģĦ(-0.07) | _cfg(-0.07)
    ACCEPTED as axis_458  cumulative_var=0.6239

  [ 454]  axes=459  step_var=0.0015  binary_acc=0.985  gap=0.1477  max_dot=0.0050  (1.9s)
    TOP:  åľ¨çº¿(0.09) | memb(0.09) | Ð¾Ð»Ð¾Ñģ(0.08) | Fut(0.08) | Â®(0.08) | ä¹ĥ(0.08) | .force(0.08) | Ð¾Ð»ÑĮÐ·(0.08)
    BOT:  nhÃł(-0.08) | ãĤ¢(-0.08) | Ù¾(-0.08) | .annotation(-0.08) | Ù¾(-0.07) | ÙĪÙħ(-0.07) | !.(-0.07) | ÙĪ(-0.07)
    ACCEPTED as axis_459  cumulative_var=0.6245

  [ 455]  axes=460  step_var=0.0015  binary_acc=0.979  gap=0.1507  max_dot=0.0027  (1.9s)
    TOP:  (0.10) | (0.10) | (0.10) | ĉĉĉĉ(0.09) | (0.09) | (0.09) | (0.09) | å¸ĥèİ±(0.09)
    BOT:  -sk(-0.09) | Libert(-0.09) | Ð±Ðµ(-0.09) | è¯ļä¿¡(-0.08) | åĲĮåŃ¦(-0.08) | åĲĦåįķä½į(-0.08) | åľ°éľĩ(-0.08) | æ°´èĤ¿(-0.08)
    ACCEPTED as axis_460  cumulative_var=0.6250

  [ 456]  axes=461  step_var=0.0015  binary_acc=0.970  gap=0.1482  max_dot=0.0031  (1.8s)
    TOP:  UV(0.10) | FY(0.08) | ê°Ģ(0.08) | .to(0.08) | à¦¬(0.08) | ivamente(0.08) | ä¿®çĲĨ(0.07) | occurred(0.07)
    BOT:  æĸĩæĺİ(-0.09) | |--(-0.09) | pointers(-0.09) | Ĺ(-0.08) | é«ĺèĢĥ(-0.08) | frozen(-0.08) | æİĪ(-0.08) | å¯¹æŃ¤(-0.08)
    ACCEPTED as axis_461  cumulative_var=0.6256

  [ 457]  axes=462  step_var=0.0016  binary_acc=0.998  gap=0.1515  max_dot=0.0016  (1.9s)
    TOP:  ï¼Ī(0.10) | __((0.09) | Â·(0.09) | Math(0.08) | Ð±Ð¾Ð»ÑĮÑĪÐµ(0.08) | ç²ĴåŃĲ(0.08) | /dis(0.08) | ()((0.08)
    BOT:  .apps(-0.10) | æĬĵ(-0.09) | ç«¯æŃ£(-0.09) | æľīä¸Ģä½į(-0.08) | åĲĥé¥Ń(-0.08) | <<"(-0.08) | æĲŀ(-0.08) | Â(-0.08)
    ACCEPTED as axis_462  cumulative_var=0.6262

  [ 458]  axes=463  step_var=0.0015  binary_acc=0.971  gap=0.1540  max_dot=0.0015  (1.9s)
    TOP:  ON(0.10) | ais(0.09) | .register(0.08) | Image(0.08) | .,(0.08) | up(0.08) | Ma(0.08) | }/{(0.08)
    BOT:  éĥ½(-0.09) | vs(-0.08) | éģĳ(-0.08) | Animated(-0.08) | ä¸ĢèĪ¬éĥ½æĺ¯(-0.08) | è®¸(-0.08) | Things(-0.08) | .setVisible(-0.08)
    ACCEPTED as axis_463  cumulative_var=0.6267

  [ 459]  axes=464  step_var=0.0015  binary_acc=0.993  gap=0.1507  max_dot=0.0011  (1.8s)
    TOP:  åĪĽæĸ°åıĳå±ķ(0.09) | podemos(0.09) | MEM(0.08) | .View(0.08) | personal(0.08) | (int(0.08) | (max(0.08) | (Login(0.08)
    BOT:  ľ(-0.08) | NSW(-0.08) | between(-0.08) | å°¼äºļ(-0.08) | -box(-0.08) | Ļ(-0.08) | oz(-0.08) | æįŁä¼¤(-0.08)
    ACCEPTED as axis_464  cumulative_var=0.6273

  [ 460]  axes=465  step_var=0.0015  binary_acc=0.987  gap=0.1486  max_dot=0.0049  (1.9s)
    TOP:  axios(0.09) | Ã¸(0.09) | æĤ²(0.09) | ç¢°(0.09) | [](0.08) | Ãª(0.08) | éģįåıĬ(0.08) | é¥¼(0.08)
    BOT:  è¿ŀç»Ń(-0.09) | rg(-0.08) | ya(-0.08) | In(-0.08) | ãĢĢãĢĢ(-0.08) | omed(-0.08) | ÑĢÑĥÑģ(-0.08) | _c(-0.07)
    ACCEPTED as axis_465  cumulative_var=0.6278

  [ 461]  axes=466  step_var=0.0015  binary_acc=0.969  gap=0.1501  max_dot=0.0024  (1.9s)
    TOP:  #:(0.09) | ĥ½(0.09) | åŃ¦çĶŁçļĦ(0.09) | æľªçŁ¥(0.09) | á»©t(0.08) | esson(0.08) | ]):Ċ(0.08) | ]:ĊĊĊ(0.08)
    BOT:  æĽ¾ä»»(-0.08) | jas(-0.08) | å¯¹äºİ(-0.08) | '"(-0.07) | link(-0.07) | /run(-0.07) | uk(-0.07) | Hu(-0.07)
    ACCEPTED as axis_466  cumulative_var=0.6284

  [ 462]  axes=467  step_var=0.0015  binary_acc=0.961  gap=0.1485  max_dot=0.0031  (1.9s)
    TOP:  ï¼ı(0.10) | one(0.09) | Catch(0.09) | /(0.08) | ROOT(0.08) | éĳ«(0.08) | SID(0.08) | æ½®æ¹¿(0.08)
    BOT:  Renew(-0.09) | æĿ(-0.09) | çĶŁæŃ»(-0.09) | æĸĩçĮ®(-0.08) | ä»²(-0.08) | .âĢĿĊĊ(-0.08) | ÐºÐ¾(-0.08) | .ĊĊĊĊĊĊ(-0.08)
    ACCEPTED as axis_467  cumulative_var=0.6290

  [ 463]  axes=468  step_var=0.0015  binary_acc=0.977  gap=0.1478  max_dot=0.0122  (1.9s)
    TOP:  ("/")Ċ(0.08) | åĲĦèĩª(0.08) | (tmp(0.08) | tmp(0.08) | Exec(0.07) | Title(0.07) | æłıçĽ®(0.07) | éĤ£æł·çļĦ(0.07)
    BOT:  Fr(-0.09) | -second(-0.08) | æĸ°ä¸īæĿ¿(-0.08) | ä¸¥ç¦ģ(-0.08) | æĸ¤(-0.08) | com(-0.08) | Spar(-0.08) | FDA(-0.08)
    ACCEPTED as axis_468  cumulative_var=0.6295

  [ 464]  axes=469  step_var=0.0014  binary_acc=0.958  gap=0.1501  max_dot=0.0097  (1.8s)
    TOP:  async(0.09) | çİ°åľ¨(0.09) | 'use(0.09) | é©¬åħĭ(0.09) | åľ¨ç½ĳä¸Ĭ(0.08) | DETAILS(0.08) | ope(0.08) | news(0.08)
    BOT:  èĵĦ(-0.08) | Ùı(-0.08) | Â´(-0.08) | Ð¾Ð²Ð¸Ñĩ(-0.08) | aaS(-0.08) | ä¸Ľä¹¦(-0.07) | Ð½Ð¸Ñĩ(-0.07) | pond(-0.07)
    ACCEPTED as axis_469  cumulative_var=0.6301

  [ 465]  axes=470  step_var=0.0015  binary_acc=0.991  gap=0.1481  max_dot=0.0050  (1.8s)
    TOP:  åĽŀå½Ĵ(0.09) | æĵħéķ¿(0.09) | _running(0.08) | post(0.08) | _at(0.08) | ç»Ħå»º(0.08) | éľ²å¤©(0.08) | åĨ²çªģ(0.08)
    BOT:  äºĽ(-0.09) | èĪ¬çļĦ(-0.09) | âĢŀ(-0.09) | .hasNext(-0.08) | ä¸ª(-0.08) | Im(-0.08) | Â®(-0.07) | aussi(-0.07)
    ACCEPTED as axis_470  cumulative_var=0.6306

  [ 466]  axes=471  step_var=0.0015  binary_acc=0.991  gap=0.1465  max_dot=0.0031  (1.9s)
    TOP:  Ð±(0.09) | ed(0.09) | St(0.09) | sql(0.09) | _d(0.09) | \r(0.08) | ãĤ°(0.08) | dvd(0.08)
    BOT:  æ·±åĬłå·¥(-0.09) | Fig(-0.08) | hÃ©(-0.08) | ä¸ĢèĤ¡(-0.08) | å½ĵæĹ¥(-0.08) | EXPECTED(-0.08) | .Act(-0.07) | .boot(-0.07)
    ACCEPTED as axis_471  cumulative_var=0.6312

  [ 467]  axes=472  step_var=0.0015  binary_acc=0.994  gap=0.1455  max_dot=0.0039  (1.9s)
    TOP:  ÛĮØ±(0.09) | ÛĮØ¯(0.09) | Stat(0.08) | Vous(0.08) | osto(0.07) | éĽĨä¸Ńåľ¨(0.07) | vous(0.07) | OLDER(0.07)
    BOT:  })(-0.10) | })(-0.09) | accordance(-0.08) | cÄĥn(-0.08) | ÑģÑĥÐ´(-0.08) | protein(-0.08) | useless(-0.08) | (p(-0.08)
    ACCEPTED as axis_472  cumulative_var=0.6317

  [ 468]  axes=473  step_var=0.0015  binary_acc=0.987  gap=0.1473  max_dot=0.0040  (1.8s)
    TOP:  çĤĴ(0.09) | batch(0.09) | Entity(0.09) | åŃ(0.08) | å¢¨(0.08) | Symfony(0.08) | ematic(0.08) | çļĦä¸Ģ(0.08)
    BOT:  Ð·(-0.10) | Mi(-0.09) | tal(-0.09) | âĢĻ(-0.08) | field(-0.08) | åıĸ(-0.08) | åı¯æł¹æį®(-0.08) | .filter(-0.07)
    ACCEPTED as axis_473  cumulative_var=0.6323

  [ 469]  axes=474  step_var=0.0015  binary_acc=0.993  gap=0.1509  max_dot=0.0027  (1.9s)
    TOP:  å³°åĢ¼(0.09) | Ð²ÐµÐ½(0.09) | _program(0.08) | å®ĥä»¬(0.08) | alex(0.08) | Ð¿Ð¾Ð²(0.08) | WS(0.08) | pylint(0.07)
    BOT:  æĺ¯éĿŀ(-0.09) | .{(-0.08) | zu(-0.08) | åį³(-0.08) | äººä¸º(-0.08) | à¹ģà¸¥à¸°(-0.08) | loc(-0.08) | éĿŀ(-0.08)
    ACCEPTED as axis_474  cumulative_var=0.6328

  [ 470]  axes=475  step_var=0.0015  binary_acc=1.000  gap=0.1452  max_dot=0.0031  (1.8s)
    TOP:  kw(0.08) | ,_(0.07) | æİī(0.07) | ä¸įè¿ĩ(0.07) | igenous(0.07) | entity(0.07) | OTA(0.07) | \",\"(0.07)
    BOT:  zz(-0.08) | Figure(-0.08) | pick(-0.08) | Ã¥(-0.08) | Card(-0.08) | .getTime(-0.08) | éĢłåģĩ(-0.08) | åĩºè¡Ģ(-0.08)
    ACCEPTED as axis_475  cumulative_var=0.6333

  [ 471]  axes=476  step_var=0.0015  binary_acc=0.979  gap=0.1455  max_dot=0.0064  (1.8s)
    TOP:  escape(0.08) | June(0.08) | Mahmoud(0.08) | >>>(0.08) | å¼ĢæĶ¾(0.08) | Ang(0.08) | Ð¿Ð¾Ð»(0.08) | .setdefault(0.08)
    BOT:  ä¹±è±¡(-0.08) | çĪ¶(-0.08) | åħ±(-0.08) | è¿ĳ(-0.08) | .handlers(-0.08) | éĢłç¦ı(-0.08) | éĩĳå±±(-0.08) | èī°éļ¾(-0.08)
    ACCEPTED as axis_476  cumulative_var=0.6339

  [ 472]  axes=477  step_var=0.0015  binary_acc=0.996  gap=0.1461  max_dot=0.0005  (1.8s)
    TOP:  }}{{(0.10) | å¿ħå°Ĩ(0.09) | ç«ĭåį³(0.09) | æ®´(0.08) | æ¬¡è¦ģ(0.08) | canceled(0.08) | ÑĤÐµÐ¼(0.08) | .assert(0.08)
    BOT:  /C(-0.09) | :.(-0.08) | ^.(-0.08) | ĭ(-0.08) | cá»§a(-0.08) | L(-0.08) | è¶ħè¿ĩ(-0.08) | By(-0.07)
    ACCEPTED as axis_477  cumulative_var=0.6344

  [ 473]  axes=478  step_var=0.0014  binary_acc=0.967  gap=0.1435  max_dot=0.0019  (1.8s)
    TOP:  è°ĥæŁ¥(0.08) | äººæīį(0.08) | çĲĥåĳĺ(0.08) | percussion(0.07) | èĤ¡æľ¬(0.07) | sizeof(0.07) | date(0.07) | åľ¨æĪĳçļĦ(0.07)
    BOT:  elt(-0.08) | âĸ¶(-0.08) | 'RE(-0.08) | chn(-0.08) | .DataSource(-0.08) | 've(-0.08) | çĽı(-0.08) | åľ°æ®µ(-0.07)
    ACCEPTED as axis_478  cumulative_var=0.6349

  [ 474]  axes=479  step_var=0.0014  binary_acc=0.984  gap=0.1423  max_dot=0.0049  (1.8s)
    TOP:  éĢ¢(0.09) | unal(0.08) | _bad(0.07) | *)(0.07) | -An(0.07) | Ð¿Ð»(0.07) | ä¼¼(0.07) | -new(0.07)
    BOT:  mars(-0.08) | =========Ċ(-0.08) | dim(-0.08) | SECTION(-0.08) | ========Ċ(-0.07) | .my(-0.07) | )]ĊĊ(-0.07) | Ċ(-0.07)
    ACCEPTED as axis_479  cumulative_var=0.6354

  [ 475]  axes=480  step_var=0.0015  binary_acc=0.986  gap=0.1479  max_dot=0.0010  (1.8s)
    TOP:  -page(0.09) | _z(0.08) | å¤©ç©º(0.08) | AE(0.08) | _fn(0.08) | Items(0.08) | Ne(0.07) | _statement(0.07)
    BOT:  &gt(-0.08) | Maher(-0.08) | '),(-0.08) | ÙĦÙħØ§(-0.08) | åľ¨å®¶(-0.08) | agr(-0.08) | ]](-0.08) | aden(-0.08)
    ACCEPTED as axis_480  cumulative_var=0.6360

  [ 476]  axes=481  step_var=0.0014  binary_acc=0.974  gap=0.1463  max_dot=0.0097  (1.9s)
    TOP:  å½ĵæĪĳ(0.09) | nearest(0.09) | ç¬¦åı·(0.08) | èĳĹåĲįçļĦ(0.08) | ****************************************************************************(0.08) | æĺİæĺŁ(0.07) | æ²ĥå°Ķ(0.07) | __))Ċ(0.07)
    BOT:  ä¸Ģæĸ¹éĿ¢(-0.09) | ãĢĳãĢĲ(-0.08) | Des(-0.08) | åĬŀåħ¬å®¤(-0.07) | è¡Įè½¦(-0.07) | çİĭèĢħ(-0.07) | ÑĢÐ°Ð·(-0.07) | if(-0.07)
    ACCEPTED as axis_481  cumulative_var=0.6365

  [ 477]  axes=482  step_var=0.0014  binary_acc=0.988  gap=0.1443  max_dot=0.0048  (1.9s)
    TOP:  å°ļæľª(0.09) | schema(0.09) | rel(0.08) | Princess(0.08) | çĤ³(0.08) | åı·ç§°(0.07) | iss(0.07) | æĪ¿åľ°äº§(0.07)
    BOT:  stat(-0.09) | æĥħ(-0.08) | .commands(-0.08) | Flower(-0.07) | ]->(-0.07) | trÃ¡i(-0.07) | Unt(-0.07) | ="#(-0.07)
    ACCEPTED as axis_482  cumulative_var=0.6370

  [ 478]  axes=483  step_var=0.0015  binary_acc=0.989  gap=0.1431  max_dot=0.0134  (1.8s)
    TOP:  Markets(0.09) | .tr(0.08) | çļĦåľŁåľ°(0.08) | .import(0.07) | Hol(0.07) | pygame(0.07) | ç¡«(0.07) | .ver(0.07)
    BOT:  æĮ¡(-0.08) | ĊĊ(-0.08) | _modal(-0.08) | Allow(-0.08) | *(-0.08) | éĩĮ(-0.08) | èµĮ(-0.08) | åĦª(-0.08)
    ACCEPTED as axis_483  cumulative_var=0.6375

  [ 479]  axes=484  step_var=0.0014  binary_acc=0.998  gap=0.1427  max_dot=0.0043  (1.8s)
    TOP:  é«ĺåħ´(0.08) | ç¥Ŀ(0.08) | ]['(0.08) | ert(0.07) | æ¨¡æĿ¿(0.07) | ìĿ´ë¦Ħ(0.07) | _channels(0.07) | çĭ±(0.07)
    BOT:  all(-0.09) | à¹Ħà¸¡(-0.09) | _controller(-0.08) | Merry(-0.08) | encial(-0.08) | .fre(-0.07) | `s(-0.07) | LIMITED(-0.07)
    ACCEPTED as axis_484  cumulative_var=0.6380

  [ 480]  axes=485  step_var=0.0015  binary_acc=0.971  gap=0.1473  max_dot=0.0032  (1.8s)
    TOP:  urning(0.08) | []((0.08) | FILE(0.08) | sth(0.07) | .""(0.07) | (fileName(0.07) | (filename(0.07) | (K(0.07)
    BOT:  å½ĵéĢī(-0.08) | åĪ¶åº¦æĶ¹éĿ©(-0.08) | å¼ĢåĲ¯(-0.08) | åĨ²æ´Ĺ(-0.08) | åıĳ(-0.08) | ROADCAST(-0.08) | vic(-0.08) | åĲĳä¸ĭ(-0.07)
    ACCEPTED as axis_485  cumulative_var=0.6386

  [ 481]  axes=486  step_var=0.0014  binary_acc=0.999  gap=0.1427  max_dot=0.0141  (1.8s)
    TOP:  å°ĳ(0.08) | ________________(0.08) | åŁŁåĲį(0.08) | _embedding(0.08) | ,O(0.08) | .drawable(0.08) | _template(0.08) | _PK(0.08)
    BOT:  don(-0.09) | à§įà¦(-0.09) | oz(-0.08) | æĢ»æĬķèµĦ(-0.08) | Two(-0.08) | gh(-0.08) | àµįà´(-0.08) | type(-0.08)
    ACCEPTED as axis_486  cumulative_var=0.6391

  [ 482]  axes=487  step_var=0.0015  binary_acc=0.994  gap=0.1425  max_dot=0.0031  (1.8s)
    TOP:  æ¨¡çī¹(0.08) | phis(0.08) | sov(0.08) | Ð¿ÑĥÐ±Ð»Ð¸(0.08) | åºĵéĩĮ(0.08) | Ð¼ÑĭÑĪ(0.08) | $($(0.08) | BIT(0.08)
    BOT:  En(-0.09) | ARD(-0.08) | .Al(-0.08) | VALUES(-0.08) | ÐµÐ´(-0.08) | çŀ¬(-0.07) | igital(-0.07) | å§Ķæīĺ(-0.07)
    ACCEPTED as axis_487  cumulative_var=0.6396

  [ 483]  axes=488  step_var=0.0014  binary_acc=1.000  gap=0.1432  max_dot=0.0052  (1.8s)
    TOP:  Ø¥(0.08) | /re(0.08) | ki(0.08) | (map(0.08) | Ser(0.08) | Ø£(0.08) | éĢłçº¸(0.08) | æ±ĩçİĩ(0.07)
    BOT:  .getP(-0.08) | LV(-0.07) | ho(-0.07) | .named(-0.07) | åıĶåıĶ(-0.07) | uchar(-0.07) | ï¼Ŀ(-0.07) | Cookie(-0.07)
    ACCEPTED as axis_488  cumulative_var=0.6402

  [ 484]  axes=489  step_var=0.0015  binary_acc=0.978  gap=0.1466  max_dot=0.0043  (1.8s)
    TOP:  );ĊĊ(0.08) | æī«(0.08) | æĲŀ(0.08) | ìķĮ(0.08) | ISS(0.08) | });ĊĊ(0.07) | Ð²Ð°Ð¼(0.07) | ("./(0.07)
    BOT:  output(-0.09) | FG(-0.09) | å¤ĸåĽ´(-0.08) | hra(-0.08) | åı¤äºº(-0.08) | }{(-0.08) | Software(-0.08) | ///(-0.08)
    ACCEPTED as axis_489  cumulative_var=0.6407

  [ 485]  axes=490  step_var=0.0014  binary_acc=0.989  gap=0.1431  max_dot=0.0031  (1.9s)
    TOP:  |\(0.09) | ĉa(0.08) | serve(0.08) | q(0.08) | -car(0.08) | cls(0.07) | ovie(0.07) | j(0.07)
    BOT:  Ðŀ(-0.08) | ä¹Łä¸įèĥ½(-0.08) | å¯Įè´µ(-0.08) | Mc(-0.08) | ãĥĸ(-0.08) | ÑĥÐ¿(-0.08) | éĨ(-0.08) | Segment(-0.08)
    ACCEPTED as axis_490  cumulative_var=0.6412

  [ 486]  axes=491  step_var=0.0014  binary_acc=0.987  gap=0.1397  max_dot=0.0025  (1.9s)
    TOP:  ï¬ģ(0.08) | _urls(0.08) | wa(0.08) | åĿĲèĲ½åľ¨(0.08) | ÑĤÐ¾(0.08) | ãĤ³ãĥŁ(0.08) | å°±åľ¨(0.08) | OC(0.08)
    BOT:  ä½łä»¬(-0.08) | ..(-0.08) | èĨĪ(-0.07) | iya(-0.07) | ÑĭÐ¼Ð¸(-0.07) | Borg(-0.07) | Heather(-0.07) | è·³(-0.07)
    ACCEPTED as axis_491  cumulative_var=0.6417

  [ 487]  axes=492  step_var=0.0014  binary_acc=0.993  gap=0.1418  max_dot=0.0052  (1.8s)
    TOP:  Mis(0.08) | toutes(0.08) | ett(0.08) | An(0.07) | ÃĮ(0.07) | Soph(0.07) | à¤¿(0.07) | ROUND(0.07)
    BOT:  }\(-0.08) | sui(-0.08) | du(-0.07) | paragraph(-0.07) | '))(-0.07) | uf(-0.07) | die(-0.07) | -nav(-0.07)
    ACCEPTED as axis_492  cumulative_var=0.6422

  [ 488]  axes=493  step_var=0.0015  binary_acc=0.993  gap=0.1418  max_dot=0.0049  (1.8s)
    TOP:  åĸĦ(0.09) | è°ĥèĬĤ(0.09) | &=(0.08) | â¼Ģ(0.08) | è£ĺ(0.08) | Delete(0.08) | ãĤĪãģı(0.07) | delete(0.07)
    BOT:  Files(-0.08) | population(-0.08) | leading(-0.08) | eig(-0.07) | _proba(-0.07) | éĺĢéĹ¨(-0.07) | /event(-0.07) | gown(-0.07)
    ACCEPTED as axis_493  cumulative_var=0.6427

  [ 489]  axes=494  step_var=0.0015  binary_acc=0.983  gap=0.1462  max_dot=0.0008  (1.9s)
    TOP:  tn(0.09) | ",",(0.08) | moz(0.08) | èĴĭ(0.08) | åľ°ä½į(0.08) | .mark(0.08) | å´Ķ(0.08) | lá»Ľn(0.08)
    BOT:  (obj(-0.09) | thee(-0.08) | ç¦»(-0.08) | Mid(-0.08) | ä¾Ľæ°´(-0.08) | -R(-0.08) | Franz(-0.08) | ogh(-0.08)
    ACCEPTED as axis_494  cumulative_var=0.6433

  [ 490]  axes=495  step_var=0.0014  binary_acc=0.969  gap=0.1419  max_dot=0.0015  (1.8s)
    TOP:  ²(0.11) | ráº¥t(0.08) | à¹ģ(0.08) | foot(0.08) | Ð¼Ð¾Ð¶ÐµÑĤÐµ(0.08) | èĤĸ(0.08) | <!--(0.08) | åĨ¼(0.08)
    BOT:  Ð°Ð(-0.09) | åįķä½į(-0.08) | .it(-0.08) | Words(-0.08) | /(?(-0.07) | ìĪĺ(-0.07) | .Ct(-0.07) | advertised(-0.07)
    ACCEPTED as axis_495  cumulative_var=0.6438

  [ 491]  axes=496  step_var=0.0015  binary_acc=0.999  gap=0.1419  max_dot=0.0072  (1.9s)
    TOP:  kor(0.09) | tha(0.08) | çļĦæĸ¹å¼ı(0.08) | this(0.08) | b(0.08) | æĺ¯åľ¨(0.08) | faithful(0.08) | '',(0.07)
    BOT:  é¾Ľ(-0.09) | .combine(-0.08) | åºµ(-0.08) | æĹ±(-0.08) | åħĦ(-0.08) | åĽ½æ°ĳç»ıæµİ(-0.08) | dÃ¼(-0.08) | .collect(-0.07)
    ACCEPTED as axis_496  cumulative_var=0.6443

  [ 492]  axes=497  step_var=0.0014  binary_acc=0.991  gap=0.1412  max_dot=0.0045  (1.9s)
    TOP:  features(0.08) | åĨľæ°ĳ(0.08) | gt(0.08) | Ù¾(0.07) | Seasons(0.07) | acer(0.07) | ëĪ(0.07) | []((0.07)
    BOT:  -pre(-0.08) | Rock(-0.08) | Next(-0.08) | .Get(-0.07) | .call(-0.07) | Eastern(-0.07) | .split(-0.07) | è¿ĶåĽŀ(-0.07)
    ACCEPTED as axis_497  cumulative_var=0.6448

  [ 493]  axes=498  step_var=0.0014  binary_acc=0.990  gap=0.1409  max_dot=0.0099  (1.8s)
    TOP:  Camb(0.08) | &D(0.08) | w(0.07) | ACA(0.07) | /DTD(0.07) | ÙĪØª(0.07) | conn(0.07) | Bl(0.07)
    BOT:  your(-0.08) | .hash(-0.08) | Ð°Ð²ÑĤÐ¾(-0.08) | Â(-0.08) | âĢĭ(-0.08) | åįĥä¸ĩ(-0.08) | _class(-0.08) | âĢĭâĢĭ(-0.08)
    ACCEPTED as axis_498  cumulative_var=0.6453

  [ 494]  axes=499  step_var=0.0015  binary_acc=0.983  gap=0.1435  max_dot=0.0036  (1.8s)
    TOP:  _pts(0.08) | _factor(0.08) | _add(0.08) | specials(0.08) | èĩªçĦ¶æĺ¯(0.07) | æŀľå®ŀ(0.07) | âĢľ(0.07) | å¯Įè´µ(0.07)
    BOT:  åĲ(-0.09) | æİ(-0.09) | (*)(-0.08) | Camp(-0.08) | èĲ§(-0.08) | äºĨåĲ§(-0.08) | æĿ(-0.08) | -Type(-0.08)
    ACCEPTED as axis_499  cumulative_var=0.6458

  [ 495]  axes=500  step_var=0.0014  binary_acc=0.961  gap=0.1429  max_dot=0.0004  (1.8s)
    TOP:  ti(0.08) | Method(0.07) | ãĥķ(0.07) | :pk(0.07) | åľ¨åħ¶(0.07) | å¹´éĻĲ(0.07) | iod(0.07) | çīĪæľ¬(0.07)
    BOT:  best(-0.09) | Script(-0.08) | O(-0.08) | Ts(-0.08) | é¡µ(-0.08) | ä½¿çĶ¨æĿĥ(-0.08) | $\(-0.08) | Nie(-0.08)
    ACCEPTED as axis_500  cumulative_var=0.6463

  [ 496]  axes=501  step_var=0.0014  binary_acc=0.988  gap=0.1459  max_dot=0.0077  (1.8s)
    TOP:  simply(0.08) | æ¶ĵ(0.08) | Vote(0.08) | -one(0.07) | æ¶Īæ¯Ĵ(0.07) | \\.(0.07) | thunk(0.07) | åĬłçĽŁ(0.07)
    BOT:  _by(-0.09) | æķħäºĭ(-0.08) | Porsche(-0.08) | è¾¾(-0.08) | /go(-0.08) | ="/(-0.08) | ä¸įäºĨ(-0.08) | Galaxy(-0.07)
    ACCEPTED as axis_501  cumulative_var=0.6468

  [ 497]  axes=502  step_var=0.0014  binary_acc=0.953  gap=0.1377  max_dot=0.0122  (1.8s)
    TOP:  è¾¹(0.08) | ÙĢ(0.08) | SEN(0.08) | æīī(0.08) | los(0.07) | è®°å¾Ĺ(0.07) | Thomson(0.07) | æĹłå½¢(0.07)
    BOT:  illus(-0.09) | .schema(-0.09) | Representation(-0.08) | /g(-0.08) | ."""ĊĊ(-0.08) | .authentication(-0.08) | :=(-0.08) | (obs(-0.08)
    ACCEPTED as axis_502  cumulative_var=0.6473

  [ 498]  axes=503  step_var=0.0015  binary_acc=0.983  gap=0.1422  max_dot=0.0021  (1.9s)
    TOP:  [l(0.08) | FO(0.08) | subset(0.08) | å°ıåŃ¦(0.08) | $.(0.08) | split(0.07) | Kra(0.07) | Ð·(0.07)
    BOT:  å¤§åŀĭ(-0.08) | './(-0.08) | /commons(-0.08) | _"(-0.08) | .modules(-0.07) | Tong(-0.07) | çĽĳçĿ£ç®¡çĲĨ(-0.07) | é¸¯(-0.07)
    ACCEPTED as axis_503  cumulative_var=0.6478

  [ 499]  axes=504  step_var=0.0014  binary_acc=0.993  gap=0.1387  max_dot=0.0026  (1.9s)
    TOP:  HttpResponse(0.08) | .depend(0.08) | åıĪç§°(0.07) | BLACK(0.07) | æŃ£ç¡®çļĦ(0.07) | items(0.07) | ëĵ±ìĿĺ(0.07) | caract(0.07)
    BOT:  ä¸ľåįĹ(-0.08) | Ma(-0.08) | æ¸¯(-0.08) | éģ¥(-0.07) | //Ċ(-0.07) | åħ¬(-0.07) | åĲ¯(-0.07) | compiler(-0.07)
    ACCEPTED as axis_504  cumulative_var=0.6483

  [ 500]  axes=505  step_var=0.0014  binary_acc=1.000  gap=0.1429  max_dot=0.0058  (2.0s)
    TOP:  (ct(0.08) | "The(0.08) | (q(0.07) | "You(0.07) | oa(0.07) | /*(0.07) | .phase(0.07) | ÏĨ(0.07)
    BOT:  åĪĨåĪ«(-0.08) | æľ¬å¸Ĥ(-0.08) | Ivan(-0.08) | .AppCompatActivity(-0.07) | .runtime(-0.07) | besar(-0.07) | Drive(-0.07) | unsubscribe(-0.07)
    ACCEPTED as axis_505  cumulative_var=0.6488

  [ 501]  axes=506  step_var=0.0014  binary_acc=0.978  gap=0.1450  max_dot=0.0081  (1.9s)
    TOP:  äºĨåĲĹ(0.09) | éĢļçŁ¥ä¹¦(0.08) | _after(0.08) | æ²¡æľīä»Ģä¹Ī(0.08) | _-(0.08) | .mp(0.08) | å¥½åĲĹ(0.08) | ackage(0.08)
    BOT:  anne(-0.09) | Players(-0.08) | å§¿åĬ¿(-0.08) | åŀĦ(-0.08) | âĶ(-0.07) | um(-0.07) | é¡µéĿ¢(-0.07) | åĶ¯ä¸Ģ(-0.07)
    ACCEPTED as axis_506  cumulative_var=0.6493

  [ 502]  axes=507  step_var=0.0014  binary_acc=0.989  gap=0.1450  max_dot=0.0005  (1.9s)
    TOP:  .colors(0.08) | .sleep(0.08) | crire(0.07) | Convert(0.07) | çħ§(0.07) | __()Ċ(0.07) | .MenuItem(0.07) | (file(0.07)
    BOT:  Match(-0.10) | Injectable(-0.09) | äºĮåįģåĽĽ(-0.08) | Bet(-0.07) | éĿĴæĺ¥(-0.07) | und(-0.07) | assistant(-0.07) | ä¸Ģä½ĵåĮĸ(-0.07)
    ACCEPTED as axis_507  cumulative_var=0.6498

  [ 503]  axes=508  step_var=0.0015  binary_acc=0.991  gap=0.1391  max_dot=0.0052  (1.8s)
    TOP:  .M(0.09) | ä¸ŃåĽ½äººæ°ĳ(0.09) | [:,(0.08) | hm(0.08) | oux(0.08) | ukkan(0.08) | Feature(0.07) | directors(0.07)
    BOT:  etc(-0.08) | Ã¶(-0.08) | Ð°Ð½(-0.08) | åĲĮãģĺ(-0.08) | à¹Ħ(-0.08) | BET(-0.08) | Ă(-0.08) | èĭ(-0.07)
    ACCEPTED as axis_508  cumulative_var=0.6503

  [ 504]  axes=509  step_var=0.0014  binary_acc=0.991  gap=0.1365  max_dot=0.0074  (1.8s)
    TOP:  çļĦå°ı(0.08) | æĢ»çļĦ(0.08) | DOCTYPE(0.08) | 'al(0.08) | .In(0.08) | segunda(0.07) | çļĦåľ°(0.07) | straÃŁe(0.07)
    BOT:  çĶ¨å¿ĥ(-0.09) | q(-0.08) | atal(-0.08) | f(-0.08) | âĪĴ(-0.08) | å¾ģæĶ¶(-0.08) | Q(-0.08) | á¹ĩ(-0.07)
    ACCEPTED as axis_509  cumulative_var=0.6508

  [ 505]  axes=510  step_var=0.0014  binary_acc=0.979  gap=0.1432  max_dot=0.0049  (1.9s)
    TOP:  è¥¿èĹı(0.07) | ÑģÐ»(0.07) | _N(0.07) | .&(0.07) | ÃĹ(0.07) | åĽ¾ä¹¦(0.07) | .jackson(0.07) | ltk(0.07)
    BOT:  ëª©(-0.09) | Tay(-0.09) | è¿Ļæī¹(-0.08) | lass(-0.08) | journal(-0.08) | .setup(-0.08) | announced(-0.07) | åĪĨ(-0.07)
    ACCEPTED as axis_510  cumulative_var=0.6513

  [ 506]  axes=511  step_var=0.0014  binary_acc=0.973  gap=0.1428  max_dot=0.0010  (1.8s)
    TOP:  by(0.08) | ad(0.07) | och(0.07) | å§ĵ(0.07) | æĽ´åĬł(0.07) | _in(0.07) | Ð¾ÑģÐ¾Ð±ÐµÐ½Ð½Ð¾(0.07) | åıĳå¸ĥ(0.07)
    BOT:  åĽº(-0.08) | EDURE(-0.08) | ()),(-0.07) | (card(-0.07) | ="#">(-0.07) | .center(-0.07) | Ø¨ÙĦ(-0.07) | ;"><(-0.07)
    ACCEPTED as axis_511  cumulative_var=0.6518

  [ 507]  axes=512  step_var=0.0014  binary_acc=0.980  gap=0.1406  max_dot=0.0056  (1.8s)
    TOP:  server(0.08) | Monthly(0.08) | module(0.08) | lang(0.08) | _block(0.07) | li(0.07) | Confidential(0.07) | mq(0.07)
    BOT:  Ãĸ(-0.09) | armed(-0.08) | oluciÃ³n(-0.08) | æļ´(-0.08) | åľ¨è¿Ļ(-0.08) | Ãłs(-0.08) | ç»ıæµİåŃ¦(-0.08) | androidx(-0.08)
    ACCEPTED as axis_512  cumulative_var=0.6523

  [ 508]  axes=513  step_var=0.0014  binary_acc=0.987  gap=0.1385  max_dot=0.0020  (1.8s)
    TOP:  Ð£(0.08) | .randn(0.08) | Ð¶(0.08) | Le(0.08) | Nor(0.07) | Ver(0.07) | Fixture(0.07) | AN(0.07)
    BOT:  ]+)/(-0.09) | CRE(-0.08) | ĉfor(-0.08) | >>>(-0.08) | sher(-0.08) | ctl(-0.08) | this(-0.07) | should(-0.07)
    ACCEPTED as axis_513  cumulative_var=0.6528

  [ 509]  axes=514  step_var=0.0014  binary_acc=0.976  gap=0.1377  max_dot=0.0091  (1.8s)
    TOP:  adians(0.07) | Run(0.07) | éľ(0.07) | èľ¡(0.07) | ('.')[(0.07) | çĻ»éĻĨ(0.07) | æİ§(0.07) | municipality(0.07)
    BOT:  )"(-0.10) | ],Ċ(-0.09) | )',Ċ(-0.08) | )+"(-0.08) | ¤(-0.08) | mb(-0.08) | dan(-0.08) | .cm(-0.08)
    ACCEPTED as axis_514  cumulative_var=0.6533

  [ 510]  axes=515  step_var=0.0014  binary_acc=0.980  gap=0.1380  max_dot=0.0022  (1.8s)
    TOP:  pose(0.09) | .service(0.08) | WP(0.08) | .minecraftforge(0.08) | å«©(0.08) | ä¾Ŀèµĸ(0.08) | è½¦åŀĭ(0.07) | modules(0.07)
    BOT:  API(-0.09) | {\(-0.08) | [\(-0.08) | Ðĸ(-0.08) | (\(-0.08) | UN(-0.08) | ÙĪØª(-0.08) | _decay(-0.08)
    ACCEPTED as axis_515  cumulative_var=0.6538

  [ 511]  axes=516  step_var=0.0014  binary_acc=0.987  gap=0.1366  max_dot=0.0028  (1.8s)
    TOP:  animation(0.07) | Horn(0.07) | è®«(0.07) | .it(0.07) | Here(0.07) | fs(0.07) | åĸľçĪ±(0.07) | ect(0.07)
    BOT:  *)(-0.08) | åº(-0.08) | (`(-0.08) | indices(-0.08) | ")),Ċ(-0.07) | (player(-0.07) | åĿĩä¸º(-0.07) | Ð±ÐµÑģ(-0.07)
    ACCEPTED as axis_516  cumulative_var=0.6543

  [ 512]  axes=517  step_var=0.0014  binary_acc=0.995  gap=0.1369  max_dot=0.0025  (1.9s)
    TOP:  Ð±ÑĭÐ»Ð¸(0.07) | å½ĵæĹ¶(0.07) | ÑĤÐµ(0.07) | Strat(0.07) | ç»´å°Ķ(0.07) | _ge(0.07) | catch(0.07) | æ¬§ç¾İ(0.07)
    BOT:  -M(-0.09) | -p(-0.09) | -d(-0.09) | å±(-0.08) | *b(-0.08) | çĽĴåŃĲ(-0.08) | çĶ¨æĪ·(-0.08) | -c(-0.08)
    ACCEPTED as axis_517  cumulative_var=0.6547

  [ 513]  axes=518  step_var=0.0015  binary_acc=0.964  gap=0.1425  max_dot=0.0039  (1.9s)
    TOP:  åı¯æĢķ(0.09) | '''(0.08) | éļį(0.08) | Disclaimer(0.07) | Crab(0.07) | æ·Ŀ(0.07) | Overview(0.07) | fire(0.07)
    BOT:  .j(-0.09) | NE(-0.08) | çĽ¸å½ĵäºİ(-0.08) | _links(-0.08) | (num(-0.08) | è¡ĮæĶ¿éĥ¨éĹ¨(-0.07) | normal(-0.07) | Äĳá»Ļ(-0.07)
    ACCEPTED as axis_518  cumulative_var=0.6553

  [ 514]  axes=519  step_var=0.0014  binary_acc=0.989  gap=0.1410  max_dot=0.0007  (1.9s)
    TOP:  Ð¼ÐµÑģÑĤ(0.08) | .CodeAnalysis(0.08) | page(0.08) | information(0.07) | vious(0.07) | æľºæŀĦ(0.07) | book(0.07) | price(0.07)
    BOT:  èĮĥçķ´(-0.09) | åĬŀåŃ¦(-0.08) | åľ¨ä¸ŃåĽ½(-0.08) | HE(-0.08) | THEN(-0.07) | à¹Ħà¸ĭ(-0.07) | Ã¡l(-0.07) | âĢĶat(-0.07)
    ACCEPTED as axis_519  cumulative_var=0.6557

  [ 515]  axes=520  step_var=0.0014  binary_acc=0.982  gap=0.1402  max_dot=0.0096  (1.9s)
    TOP:  ]'(0.09) | åĩº(0.08) | match(0.08) | ],'(0.08) | =url(0.07) | camb(0.07) | /bash(0.07) | æĺ¯ä¸ª(0.07)
    BOT:  Äĳá»ĵng(-0.08) | _py(-0.08) | å°ĨåĨĽ(-0.08) | school(-0.07) | è¾ĸ(-0.07) | å±±ä¸ľ(-0.07) | ãģ¡ãĤĥ(-0.07) | OO(-0.07)
    ACCEPTED as axis_520  cumulative_var=0.6562

  [ 516]  axes=521  step_var=0.0013  binary_acc=0.991  gap=0.1341  max_dot=0.0015  (1.8s)
    TOP:  -dom(0.08) | .fetch(0.07) | Ð¦(0.07) | èĪį(0.07) | _framework(0.07) | .set(0.07) | æĩĤ(0.07) | Ð²Ð¼ÐµÑģÑĤ(0.07)
    BOT:  p(-0.09) | um(-0.08) | æľĢå°ı(-0.08) | æľīæķĪçļĦ(-0.08) | il(-0.08) | éĢŁçİĩ(-0.07) | ç¼ĸçłģ(-0.07) | é¥±åĴĮ(-0.07)
    ACCEPTED as axis_521  cumulative_var=0.6567

  [ 517]  axes=522  step_var=0.0014  binary_acc=0.997  gap=0.1374  max_dot=0.0032  (1.8s)
    TOP:  IS(0.08) | hall(0.08) | SK(0.08) | ...)(0.08) | McL(0.07) | åĶ¤(0.07) | çķľçī§(0.07) | .dao(0.07)
    BOT:  >::(-0.08) | {'(-0.08) | çĥŃ(-0.08) | dde(-0.07) | ä¾Ŀæį®(-0.07) | Daniel(-0.07) | instant(-0.07) | permit(-0.07)
    ACCEPTED as axis_522  cumulative_var=0.6571

  [ 518]  axes=523  step_var=0.0014  binary_acc=0.965  gap=0.1376  max_dot=0.0066  (1.8s)
    TOP:  _eff(0.08) | _real(0.07) | -child(0.07) | Remove(0.07) | _dimensions(0.07) | -half(0.07) | (parts(0.07) | Ø§ÙĦØ¢ÙĨ(0.07)
    BOT:  ))(-0.09) | å®(-0.09) | IS(-0.09) | One(-0.09) | QS(-0.08) | å¹´åº¦(-0.08) | tm(-0.08) | No(-0.08)
    ACCEPTED as axis_523  cumulative_var=0.6576

  [ 519]  axes=524  step_var=0.0014  binary_acc=0.973  gap=0.1365  max_dot=0.0049  (1.8s)
    TOP:  ÐºÐ»(0.08) | è¯¾ç¨ĭ(0.08) | Ã©tÃ©(0.08) | Ð¿ÐµÑĢ(0.08) | Ð½Ð¸Ñĩ(0.08) | matter(0.07) | ë©´(0.07) | engl(0.07)
    BOT:  AI(-0.08) | .spi(-0.08) | âĺĨ(-0.08) | æľīä¸Ģå®ļ(-0.07) | ÑĭÐ²(-0.07) | çŃīåľ°(-0.07) | åĵģçīĮçļĦ(-0.07) | LIMITED(-0.07)
    ACCEPTED as axis_524  cumulative_var=0.6581

  [ 520]  axes=525  step_var=0.0014  binary_acc=0.992  gap=0.1364  max_dot=0.0054  (1.8s)
    TOP:  Dense(0.09) | '.(0.08) | .next(0.08) | ];(0.08) | cannot(0.08) | "."(0.07) | ography(0.07) | Ã²(0.07)
    BOT:  Î¼(-0.09) | èĩ(-0.08) | Ïĥ(-0.08) | æ°¸ä¹ħ(-0.07) | Discussion(-0.07) | isValid(-0.07) | Âµ(-0.07) | color(-0.07)
    ACCEPTED as axis_525  cumulative_var=0.6586

  [ 521]  axes=526  step_var=0.0014  binary_acc=0.998  gap=0.1415  max_dot=0.0047  (1.9s)
    TOP:  _v(0.09) | we(0.08) | ).(0.08) | ÑĢÑĥÐ±(0.08) | ç¹(0.08) | Ú©(0.08) | ]).(0.08) | .).(0.08)
    BOT:  ({(-0.10) | (T(-0.08) | .minecraftforge(-0.08) | (R(-0.08) | disk(-0.08) | `s(-0.08) | regs(-0.08) | <span(-0.08)
    ACCEPTED as axis_526  cumulative_var=0.6590

  [ 522]  axes=527  step_var=0.0014  binary_acc=0.998  gap=0.1417  max_dot=0.0009  (1.8s)
    TOP:  reverse(0.08) | projects(0.08) | '^(0.07) | Christina(0.07) | æķ°çĻ¾(0.07) | åĴĮ(0.07) | Proof(0.07) | åıįèĢĮ(0.07)
    BOT:  _array(-0.09) | __(-0.09) | [](-0.09) | _min(-0.08) | async(-0.08) | èģĶåĬ¨(-0.08) | Boca(-0.08) | _channels(-0.07)
    ACCEPTED as axis_527  cumulative_var=0.6595

  [ 523]  axes=528  step_var=0.0014  binary_acc=0.986  gap=0.1370  max_dot=0.0018  (1.9s)
    TOP:  âĢĻ(0.08) | Bottom(0.08) | Cache(0.08) | son(0.08) | parts(0.08) | ds(0.08) | '/(0.07) | ){(0.07)
    BOT:  When(-0.09) | ä¸Ńæľī(-0.09) | A(-0.09) | ç½ĳä¸Ĭ(-0.08) | åĵªéĩĮ(-0.08) | Technician(-0.08) | åħĪåĲİ(-0.08) | include(-0.07)
    ACCEPTED as axis_528  cumulative_var=0.6600

  [ 524]  axes=529  step_var=0.0014  binary_acc=0.999  gap=0.1380  max_dot=0.0010  (1.9s)
    TOP:  Ð¸Ð»Ð¸(0.09) | è§Ħå®ļ(0.08) | ĉmsg(0.08) | äº§åĵģ(0.07) | èĬĤ(0.07) | phil(0.07) | ÐºÑĤÐ¾(0.07) | èĮ¶åı¶(0.07)
    BOT:  ãģ«ãģĻãĤĭ(-0.09) | åºĦ(-0.09) | \F(-0.08) | åĳ¦(-0.08) | çĿ¡è§ī(-0.07) | ';(-0.07) | Het(-0.07) | Helper(-0.07)
    ACCEPTED as axis_529  cumulative_var=0.6605

  [ 525]  axes=530  step_var=0.0014  binary_acc=0.997  gap=0.1357  max_dot=0.0017  (1.9s)
    TOP:  ,F(0.09) | _L(0.09) | ,v(0.08) | ,N(0.08) | /svg(0.08) | /A(0.08) | _d(0.08) | -the(0.08)
    BOT:  à¸±à¸§(-0.08) | ible(-0.08) | .call(-0.08) | åľ¨ä¸Ģä¸ª(-0.08) | Äģ(-0.08) | Å«(-0.07) | è®(-0.07) | æľīæĹ¶åĢĻ(-0.07)
    ACCEPTED as axis_530  cumulative_var=0.6610

  [ 526]  axes=531  step_var=0.0014  binary_acc=0.969  gap=0.1384  max_dot=0.0063  (1.8s)
    TOP:  ÑĪÐ¸(0.09) | back(0.08) | -index(0.08) | .neo(0.08) | ÑģÐµÐ»(0.07) | nop(0.07) | sd(0.07) | CL(0.07)
    BOT:  åŃĲåħ¬åı¸(-0.08) | .exports(-0.07) | _year(-0.07) | .mongodb(-0.07) | XML(-0.07) | Ð´(-0.07) | æĺŁ(-0.07) | åıĹå®³èĢħ(-0.07)
    ACCEPTED as axis_531  cumulative_var=0.6615

  [ 527]  axes=532  step_var=0.0014  binary_acc=0.978  gap=0.1347  max_dot=0.0089  (1.9s)
    TOP:  ç§ģ(0.08) | Ð¸Ð»ÑĮ(0.08) | Ñİ(0.07) | either(0.07) | Conc(0.07) | Ð¼Ð½Ð¾Ð³(0.07) | una(0.07) | Ãªs(0.07)
    BOT:  .the(-0.08) | Color(-0.08) | Date(-0.08) | ABOUT(-0.08) | å½ĵåĪĿ(-0.08) | ä¼ļå¯¹(-0.07) | tube(-0.07) | patch(-0.07)
    ACCEPTED as axis_532  cumulative_var=0.6619

  [ 528]  axes=533  step_var=0.0014  binary_acc=0.992  gap=0.1415  max_dot=0.0030  (1.8s)
    TOP:  /)Ċ(0.09) | )"Ċ(0.09) | !)Ċ(0.09) | åįģä¸ĥ(0.08) | ,)Ċ(0.08) | éĤ£æł·çļĦ(0.08) | ?)Ċ(0.08) | '"Ċ(0.08)
    BOT:  Scale(-0.10) | ÄĹ(-0.09) | Ñĭ(-0.08) | Î·(-0.08) | çĥŁ(-0.08) | èĻļ(-0.07) | à¤¤(-0.07) | æĬĢ(-0.07)
    ACCEPTED as axis_533  cumulative_var=0.6624

  [ 529]  axes=534  step_var=0.0014  binary_acc=1.000  gap=0.1359  max_dot=0.0055  (1.8s)
    TOP:  cb(0.09) | ak(0.08) | Ð²Ð¾(0.08) | Im(0.08) | _WIDTH(0.07) | åįļè§Ī(0.07) | _register(0.07) | esar(0.07)
    BOT:  åĳ¨å¹´(-0.07) | -community(-0.07) | åħ³éĶ®è¯į(-0.07) | site(-0.07) | éģŃ(-0.07) | .Key(-0.07) | Tik(-0.07) | -code(-0.07)
    ACCEPTED as axis_534  cumulative_var=0.6629

  [ 530]  axes=535  step_var=0.0014  binary_acc=0.987  gap=0.1338  max_dot=0.0038  (1.8s)
    TOP:  scam(0.08) | asin(0.07) | Total(0.07) | EL(0.07) | abis(0.07) | aler(0.07) | åħĦ(0.07) | Summary(0.07)
    BOT:  .im(-0.10) | .new(-0.09) | æł¸å®ŀ(-0.08) | .User(-0.08) | .wh(-0.07) | IF(-0.07) | By(-0.07) | .image(-0.07)
    ACCEPTED as axis_535  cumulative_var=0.6633

  [ 531]  axes=536  step_var=0.0014  binary_acc=0.972  gap=0.1371  max_dot=0.0121  (1.8s)
    TOP:  Style(0.08) | .plugins(0.08) | ä¸Ģçº§(0.08) | åįģ(0.08) | ê¸Ģ(0.07) | implicit(0.07) | Tyson(0.07) | Ø¯(0.07)
    BOT:  æ³(-0.08) | solve(-0.07) | _SEND(-0.07) | .annotation(-0.07) | Solve(-0.07) | æĬ(-0.07) | [...(-0.07) | æĬ¼ãģĹ(-0.07)
    ACCEPTED as axis_536  cumulative_var=0.6638

  [ 532]  axes=537  step_var=0.0013  binary_acc=0.996  gap=0.1349  max_dot=0.0096  (1.9s)
    TOP:  ÃĹĊĊ(0.08) | ÂłĊ(0.07) | ds(0.07) | peer(0.07) | ĺ(0.07) | hic(0.07) | _center(0.07) | /Getty(0.07)
    BOT:  (((-0.09) | ä¸įä»ħ(-0.07) | è¿Ľè¡Į(-0.07) | ($_(-0.07) | è¯´(-0.07) | .Linq(-0.07) | INSERT(-0.07) | ìĹĲ(-0.07)
    ACCEPTED as axis_537  cumulative_var=0.6642

  [ 533]  axes=538  step_var=0.0013  binary_acc=0.981  gap=0.1349  max_dot=0.0009  (1.9s)
    TOP:  payable(0.08) | èĤĨ(0.07) | ä¿Ł(0.07) | éĥ½åľ¨(0.07) | numpy(0.07) | ÏĢ(0.07) | èĭį(0.07) | åıĤ(0.07)
    BOT:  consider(-0.08) | \\\(-0.07) | âĢľThey(-0.07) | Results(-0.07) | contains(-0.07) | Discussions(-0.07) | åĽ½æĹĹ(-0.07) | '\(-0.07)
    ACCEPTED as axis_538  cumulative_var=0.6647

  [ 534]  axes=539  step_var=0.0013  binary_acc=1.000  gap=0.1346  max_dot=0.0103  (1.8s)
    TOP:  submit(0.10) | -ci(0.09) | è¯ķé¢ĺ(0.08) | formed(0.08) | invest(0.08) | mag(0.08) | _release(0.08) | template(0.07)
    BOT:  igh(-0.08) | ge(-0.08) | _R(-0.07) | ìĨ(-0.07) | oy(-0.07) | æ©±(-0.07) | ä»ĳ(-0.07) | çľĭ(-0.07)
    ACCEPTED as axis_539  cumulative_var=0.6651

  [ 535]  axes=540  step_var=0.0014  binary_acc=0.994  gap=0.1382  max_dot=0.0029  (2.1s)
    TOP:  com(0.09) | One(0.08) | stry(0.08) | .One(0.08) | Ba(0.08) | ÐĹ(0.07) | conn(0.07) | Ð·(0.07)
    BOT:  Layer(-0.08) | å¸½(-0.07) | ~=(-0.07) | Ð½Ð¾Ð³(-0.07) | Attr(-0.07) | æĢ§çļĦ(-0.07) | å¾®åįļ(-0.07) | }=(-0.07)
    ACCEPTED as axis_540  cumulative_var=0.6656

  [ 536]  axes=541  step_var=0.0014  binary_acc=0.983  gap=0.1360  max_dot=0.0017  (1.8s)
    TOP:  rt(0.07) | hu(0.07) | cdn(0.07) | posting(0.07) | sig(0.07) | -y(0.07) | hl(0.07) | cls(0.07)
    BOT:  æĸĩåĮĸçļĦ(-0.08) | ÑĭÐ²(-0.08) | Ð¼Ð¾ÑĤ(-0.08) | css(-0.08) | /input(-0.08) | ç®¡çĲĨäººåĳĺ(-0.08) | .GetMapping(-0.07) | ä¸ºä»Ģä¹Ī(-0.07)
    ACCEPTED as axis_541  cumulative_var=0.6661

  [ 537]  axes=542  step_var=0.0014  binary_acc=0.964  gap=0.1379  max_dot=0.0054  (1.8s)
    TOP:  å¹´çļĦ(0.08) | è¡£(0.08) | Packers(0.08) | åĲ¦åĪĻ(0.08) | æ¡£(0.07) | çĳľ(0.07) | _house(0.07) | ä¸ĢæĹ¦(0.07)
    BOT:  };Ċ(-0.08) | ¤(-0.07) | Assistant(-0.07) | ãĥĥ(-0.07) | *Ċ(-0.07) | cÃ¡i(-0.07) | è¿ĺç®Ĺ(-0.07) | _MINOR(-0.07)
    ACCEPTED as axis_542  cumulative_var=0.6665

  [ 538]  axes=543  step_var=0.0014  binary_acc=0.994  gap=0.1344  max_dot=0.0039  (1.8s)
    TOP:  ('',(0.08) | &Ċ(0.08) | åĪĨæĪĲ(0.07) | ULATION(0.07) | -by(0.07) | goto(0.07) | .In(0.07) | æ²¿(0.07)
    BOT:  here(-0.09) | è¦ģç´§(-0.07) | quotient(-0.07) | pages(-0.07) | æķıæĦŁ(-0.07) | adher(-0.07) | å®ŀè·µ(-0.07) | came(-0.07)
    ACCEPTED as axis_543  cumulative_var=0.6670

  [ 539]  axes=544  step_var=0.0014  binary_acc=0.959  gap=0.1347  max_dot=0.0106  (1.9s)
    TOP:  >Title(0.09) | Pl(0.08) | __':Ċ(0.08) | Má»Ļt(0.08) | w(0.08) | >.Ċ(0.07) | .webkit(0.07) | (\"(0.07)
    BOT:  .Shared(-0.08) | æī§(-0.08) | obby(-0.08) | Average(-0.08) | \Core(-0.07) | /default(-0.07) | _AND(-0.07) | ifr(-0.07)
    ACCEPTED as axis_544  cumulative_var=0.6674

  [ 540]  axes=545  step_var=0.0014  binary_acc=0.984  gap=0.1326  max_dot=0.0052  (1.8s)
    TOP:  Ur(0.09) | assets(0.08) | .kernel(0.08) | è·¯éĿ¢(0.08) | .ml(0.08) | cup(0.08) | å¸Ī(0.07) | logger(0.07)
    BOT:  æľĢæĸ°(-0.07) | lington(-0.07) | ---(-0.07) | Spotlight(-0.07) | Bur(-0.07) | Rehabilitation(-0.07) | uga(-0.07) | åħ¨ä½ĵ(-0.07)
    ACCEPTED as axis_545  cumulative_var=0.6679

  [ 541]  axes=546  step_var=0.0014  binary_acc=0.999  gap=0.1343  max_dot=0.0110  (1.8s)
    TOP:  .Select(0.08) | Ð¿Ñĥ(0.08) | dark(0.07) | subst(0.07) | abic(0.07) | https(0.07) | .block(0.07) | Just(0.07)
    BOT:  NOTE(-0.08) | vvm(-0.07) | RUN(-0.07) | without(-0.07) | Framework(-0.07) | Hello(-0.07) | ä»ªåĻ¨(-0.07) | .Job(-0.07)
    ACCEPTED as axis_546  cumulative_var=0.6684

  [ 542]  axes=547  step_var=0.0013  binary_acc=0.996  gap=0.1350  max_dot=0.0066  (1.9s)
    TOP:  "),(0.09) | >',(0.09) | ];Ċ(0.09) | Recipes(0.08) | ()),(0.08) | ']),(0.08) | %),(0.08) | )],Ċ(0.08)
    BOT:  ç»Ļ(-0.09) | aÃ§Ã£o(-0.08) | _where(-0.08) | /questions(-0.08) | ied(-0.07) | .txt(-0.07) | ä¼¼(-0.07) | çº¿ä¸ĭ(-0.07)
    ACCEPTED as axis_547  cumulative_var=0.6688

  [ 543]  axes=548  step_var=0.0013  binary_acc=0.933  gap=0.1340  max_dot=0.0091  (1.8s)
    TOP:  Document(0.09) | CV(0.08) | .se(0.08) | texte(0.08) | km(0.07) | and(0.07) | éªĮæĶ¶(0.07) | post(0.07)
    BOT:  Toys(-0.08) | æ°´åĪ©(-0.08) | .ops(-0.07) | ä¸ĭä¸Ģä¸ª(-0.07) | ä¸ĢåĢĭ(-0.07) | ©(-0.07) | %@(-0.07) | Cater(-0.07)
    ACCEPTED as axis_548  cumulative_var=0.6692

  [ 544]  axes=549  step_var=0.0014  binary_acc=0.986  gap=0.1361  max_dot=0.0043  (1.8s)
    TOP:  env(0.08) | iv(0.07) | __('(0.07) | Decl(0.07) | lon(0.07) | )[(0.07) | (list(0.07) | modelo(0.07)
    BOT:  /**Ċ(-0.11) | å»ºè®¾é¡¹çĽ®(-0.10) | _vol(-0.09) | acz(-0.08) | 'm(-0.08) | _coeff(-0.08) | /**Ċ(-0.07) | Err(-0.07)
    ACCEPTED as axis_549  cumulative_var=0.6697

  [ 545]  axes=550  step_var=0.0014  binary_acc=0.996  gap=0.1364  max_dot=0.0026  (1.8s)
    TOP:  DIS(0.08) | _board(0.08) | Facts(0.08) | Specifications(0.07) | etter(0.07) | func(0.07) | Accessor(0.07) | results(0.07)
    BOT:  æĶ¯ä»ĺ(-0.08) | _go(-0.08) | dess(-0.08) | ï¼į(-0.08) | à¸¹(-0.08) | æĪĳä»¬çļĦ(-0.07) | ãģ§ãģĤãģ£ãģŁ(-0.07) | .nn(-0.07)
    ACCEPTED as axis_550  cumulative_var=0.6702

  [ 546]  axes=551  step_var=0.0014  binary_acc=0.989  gap=0.1333  max_dot=0.0077  (1.9s)
    TOP:  .E(0.09) | }).(0.09) | ()).(0.08) | ()*(0.08) | .cl(0.08) | ").(0.08) | (**(0.08) | ].(0.07)
    BOT:  ç½ĳåĿĢ(-0.09) | åĲķ(-0.08) | College(-0.07) | Ð¾Ð»ÑĮÐ·Ð¾Ð²(-0.07) | ìłķ(-0.07) | off(-0.07) | çĴ§(-0.07) | Om(-0.07)
    ACCEPTED as axis_551  cumulative_var=0.6706

  [ 547]  axes=552  step_var=0.0014  binary_acc=0.974  gap=0.1350  max_dot=0.0024  (1.9s)
    TOP:  ro(0.07) | Ex(0.07) | çĶµç½ĳ(0.07) | åıĳåĬĽ(0.07) | Po(0.07) | Th(0.07) | (0.07) | æ³»(0.07)
    BOT:  /win(-0.08) | .structure(-0.07) | *r(-0.07) | make(-0.07) | åĨįåĬłä¸Ĭ(-0.07) | ãĥĴ(-0.07) | Â¹(-0.07) | âĢº(-0.07)
    ACCEPTED as axis_552  cumulative_var=0.6711

  [ 548]  axes=553  step_var=0.0014  binary_acc=0.982  gap=0.1363  max_dot=0.0018  (1.9s)
    TOP:  ç³»çµ±(0.07) | ä¸Ģ(0.07) | uzione(0.07) | Ki(0.07) | opot(0.07) | )"Ċ(0.07) | ];ĊĊ(0.07) | download(0.07)
    BOT:  çĹ°(-0.08) | equations(-0.08) | .Account(-0.07) | ices(-0.07) | .row(-0.07) | tc(-0.07) | dispatch(-0.07) | ä¸ºäºº(-0.07)
    ACCEPTED as axis_553  cumulative_var=0.6715

  [ 549]  axes=554  step_var=0.0014  binary_acc=0.958  gap=0.1327  max_dot=0.0062  (1.9s)
    TOP:  .Generated(0.08) | æ´»åĬĽ(0.07) | rÃ©s(0.07) | áº¥m(0.07) | annel(0.07) | .githubusercontent(0.07) | æĿ¥æºĲ(0.07) | æķĻèĤ²(0.07)
    BOT:  .URL(-0.09) | Navigation(-0.08) | ä¾§(-0.08) | ="@(-0.08) | ä¸ĭçļĦ(-0.08) | ÑĦ(-0.08) | Ð·Ð½Ð°ÑĩÐµÐ½Ð¸Ðµ(-0.08) | Ãħ(-0.07)
    ACCEPTED as axis_554  cumulative_var=0.6720

  [ 550]  axes=555  step_var=0.0013  binary_acc=1.000  gap=0.1302  max_dot=0.0076  (1.8s)
    TOP:  âĢĳ(0.08) | DB(0.08) | (f(0.08) | [data(0.08) | /H(0.07) | ĭ(0.07) | ĉD(0.07) | todos(0.07)
    BOT:  |\(-0.10) | ?\(-0.08) | éĺ²ç©º(-0.07) | sqlite(-0.07) | #[(-0.07) | -\(-0.07) | Æ¡(-0.07) | ä¹ĥ(-0.07)
    ACCEPTED as axis_555  cumulative_var=0.6724

  [ 551]  axes=556  step_var=0.0013  binary_acc=0.987  gap=0.1356  max_dot=0.0028  (1.8s)
    TOP:  .githubusercontent(0.08) | ine(0.08) | =[[(0.08) | ithub(0.08) | [],(0.07) | ()`(0.07) | .forward(0.07) | _doc(0.07)
    BOT:  MR(-0.08) | seen(-0.08) | æŃ¢(-0.08) | chk(-0.08) | st(-0.08) | MR(-0.07) | qu(-0.07) | æķĻ(-0.07)
    ACCEPTED as axis_556  cumulative_var=0.6728

  [ 552]  axes=557  step_var=0.0013  binary_acc=0.998  gap=0.1288  max_dot=0.0050  (1.8s)
    TOP:  fd(0.07) | image(0.07) | ģ(0.07) | ä¸ĸçķĮ(0.07) | .g(0.07) | addon(0.07) | åĽ½æľī(0.07) | .J(0.07)
    BOT:  åĬĿ(-0.08) | çĶ±äºİ(-0.08) | Frame(-0.07) | éĤ£æĺ¯(-0.07) | åĵįåºĶ(-0.07) | discord(-0.07) | æİĲ(-0.07) | ile(-0.07)
    ACCEPTED as axis_557  cumulative_var=0.6733

  [ 553]  axes=558  step_var=0.0013  binary_acc=0.974  gap=0.1309  max_dot=0.0094  (1.8s)
    TOP:  ä¸įè¦ģ(0.08) | "][(0.08) | either(0.07) | å¹³åĿĩ(0.07) | ä¸»è¦ģæĺ¯(0.07) | ä¸»è¦ģ(0.07) | èĹ¤(0.07) | åį³å°Ĩ(0.07)
    BOT:  åĿĹ(-0.08) | éķ¿å®ī(-0.08) | .res(-0.07) | åŃ¦éĻ¢(-0.07) | #endif(-0.07) | DER(-0.07) | permission(-0.07) | .CSS(-0.07)
    ACCEPTED as axis_558  cumulative_var=0.6737

  [ 554]  axes=559  step_var=0.0014  binary_acc=0.968  gap=0.1352  max_dot=0.0030  (1.8s)
    TOP:  <!--(0.09) | æĶ¿åºľéĩĩè´Ń(0.08) | Tip(0.07) | #!(0.07) | <!--(0.07) | Download(0.07) | cl(0.07) | æī¬å·ŀ(0.07)
    BOT:  /user(-0.07) | /static(-0.07) | '.$(-0.07) | Score(-0.07) | >();ĊĊ(-0.07) | "]Ċ(-0.07) | _url(-0.07) | actories(-0.07)
    ACCEPTED as axis_559  cumulative_var=0.6742

  [ 555]  axes=560  step_var=0.0013  binary_acc=0.966  gap=0.1328  max_dot=0.0025  (1.9s)
    TOP:  \@(0.08) | up(0.08) | è´£ä»»ç¼ĸè¾ĳ(0.07) | _fit(0.07) | æĦ¿æĦı(0.07) | fa(0.07) | -ci(0.07) | _pipeline(0.07)
    BOT:  æĸĩ(-0.08) | .variables(-0.08) | ÐļÐ¾ÑĢ(-0.07) | .For(-0.07) | .i(-0.07) | S(-0.07) | ,D(-0.07) | Che(-0.07)
    ACCEPTED as axis_560  cumulative_var=0.6746

  [ 556]  axes=561  step_var=0.0013  binary_acc=0.971  gap=0.1310  max_dot=0.0034  (1.8s)
    TOP:  .fr(0.08) | .mesh(0.08) | Energy(0.07) | tuple(0.07) | [word(0.07) | nid(0.07) | historically(0.06) | Gorgeous(0.06)
    BOT:  Init(-0.08) | =&(-0.08) | âĨĵ(-0.07) | ]</(-0.07) | Binary(-0.07) | )**(-0.07) | ####(-0.07) | ]^(-0.07)
    ACCEPTED as axis_561  cumulative_var=0.6750

  [ 557]  axes=562  step_var=0.0014  binary_acc=1.000  gap=0.1324  max_dot=0.0018  (1.9s)
    TOP:  çĶµåİĭ(0.09) | Ł³(0.08) | dem(0.08) | seq(0.08) | _sheet(0.07) | west(0.07) | ĵįä½ľ(0.07) | /license(0.07)
    BOT:  /kubernetes(-0.08) | +=(-0.08) | PMID(-0.07) | (kernel(-0.07) | Ð°Ð²(-0.07) | <((-0.07) | async(-0.07) | æ¯Ķ(-0.06)
    ACCEPTED as axis_562  cumulative_var=0.6755

  [ 558]  axes=563  step_var=0.0013  binary_acc=0.980  gap=0.1313  max_dot=0.0013  (1.9s)
    TOP:  _P(0.08) | èĦ±(0.07) | æĿ¾(0.07) | ì¤ĳ(0.07) | Ø³Ø¨(0.07) | adian(0.07) | Âł(0.07) | Carrier(0.07)
    BOT:  },{(-0.08) | .qq(-0.07) | ;margin(-0.07) | -purpose(-0.07) | ä¸Ńåľĭ(-0.07) | ÑģÑĤÐ°(-0.07) | What(-0.07) | Stripe(-0.07)
    ACCEPTED as axis_563  cumulative_var=0.6759

  [ 559]  axes=564  step_var=0.0013  binary_acc=0.974  gap=0.1310  max_dot=0.0080  (1.9s)
    TOP:  /cal(0.11) | å¯¼(0.08) | âĬ¥(0.08) | tests(0.08) | ı(0.07) | .entity(0.07) | æĹłéĻĲ(0.07) | _in(0.07)
    BOT:  X(-0.08) | ResultSet(-0.07) | à¹ĥà¸«(-0.07) | atisf(-0.07) | VR(-0.07) | èĥ½è®©(-0.06) | disclosure(-0.06) | æĪĳæĬĬ(-0.06)
    ACCEPTED as axis_564  cumulative_var=0.6763

  [ 560]  axes=565  step_var=0.0013  binary_acc=0.960  gap=0.1307  max_dot=0.0040  (1.8s)
    TOP:  je(0.08) | _of(0.08) | ros(0.08) | æĮģèĤ¡(0.08) | isme(0.08) | Attached(0.08) | éļ¾åıĹ(0.07) | _nodes(0.07)
    BOT:  function(-0.08) | BEGIN(-0.07) | open(-0.07) | ocomplete(-0.07) | .JSON(-0.07) | åģĩè®¾(-0.07) | Many(-0.07) | .factory(-0.06)
    ACCEPTED as axis_565  cumulative_var=0.6768

  [ 561]  axes=566  step_var=0.0013  binary_acc=0.978  gap=0.1314  max_dot=0.0114  (1.8s)
    TOP:  æĸ°åŀĭ(0.09) | à¹Īà¸Ń(0.07) | Jeff(0.07) | Jeff(0.07) | DESCRIPTION(0.07) | ophe(0.07) | ä½©æľį(0.06) | cludes(0.06)
    BOT:  -L(-0.09) | nh(-0.08) | .ly(-0.08) | :-(-0.08) | <-(-0.08) | uset(-0.07) | .u(-0.07) | ':(-0.07)
    ACCEPTED as axis_566  cumulative_var=0.6772

  [ 562]  axes=567  step_var=0.0013  binary_acc=0.958  gap=0.1291  max_dot=0.0013  (1.8s)
    TOP:  ÑĤÑĥ(0.09) | User(0.08) | Ã©c(0.08) | éĵ¾æĿ¡(0.07) | è¯º(0.07) | Ð³Ð°(0.07) | Ð´ÐµÐ½(0.07) | .track(0.07)
    BOT:  _G(-0.09) | æĽ´(-0.07) | .exceptions(-0.07) | .proxy(-0.07) | ç«¯(-0.07) | _src(-0.07) | .image(-0.07) | rad(-0.07)
    ACCEPTED as axis_567  cumulative_var=0.6776

  [ 563]  axes=568  step_var=0.0013  binary_acc=0.995  gap=0.1346  max_dot=0.0029  (1.8s)
    TOP:  exp(0.08) | _next(0.08) | .ru(0.07) | à¹ĥà¸ļ(0.07) | .sim(0.07) | .Start(0.07) | .services(0.07) | options(0.07)
    BOT:  å¤ľ(-0.07) | ï»¿//(-0.07) | ãĢĳãĢĲ(-0.07) | ÐºÑģ(-0.07) | .png(-0.07) | ãĥĢ(-0.06) | ooke(-0.06) | åĶĨ(-0.06)
    ACCEPTED as axis_568  cumulative_var=0.6780

  [ 564]  axes=569  step_var=0.0013  binary_acc=0.989  gap=0.1295  max_dot=0.0021  (1.8s)
    TOP:  âĢĶthe(0.09) | meet(0.08) | resolve(0.08) | _point(0.07) | _temp(0.07) | .plot(0.07) | ãģ®ãģ¿(0.07) | ìķĦ(0.07)
    BOT:  åºĶåľ¨(-0.09) | ä¼¯(-0.08) | èĬ¦(-0.08) | åĮ»(-0.08) | è½¨(-0.07) | eq(-0.07) | ç§ģ(-0.07) | ÐµÐ¶(-0.07)
    ACCEPTED as axis_569  cumulative_var=0.6785

  [ 565]  axes=570  step_var=0.0013  binary_acc=0.971  gap=0.1315  max_dot=0.0023  (1.8s)
    TOP:  ç¥¨(0.08) | ÙģÙĬÙħØ§(0.07) | _steps(0.07) | ibo(0.07) | åĺİ(0.07) | OL(0.07) | ä¸ĢæĹ¶(0.07) | Ì(0.07)
    BOT:  os(-0.08) | ],Ċ(-0.08) | éĤĵå°ıå¹³(-0.08) | /",Ċ(-0.08) | åĮĹéĥ¨(-0.08) | inspect(-0.08) | };ĊĊ(-0.07) | tom(-0.07)
    ACCEPTED as axis_570  cumulative_var=0.6789

  [ 566]  axes=571  step_var=0.0013  binary_acc=0.996  gap=0.1310  max_dot=0.0039  (1.9s)
    TOP:  _QU(0.08) | _INPUT(0.08) | å½©(0.08) | loat(0.08) | While(0.08) | _mock(0.08) | sa(0.07) | ann(0.07)
    BOT:  edition(-0.08) | =True(-0.07) | çļĦä¸Ģ(-0.07) | à¹Īà¸Ļ(-0.07) | December(-0.07) | aussi(-0.07) | renderer(-0.07) | Qui(-0.07)
    ACCEPTED as axis_571  cumulative_var=0.6793

  [ 567]  axes=572  step_var=0.0013  binary_acc=0.989  gap=0.1302  max_dot=0.0046  (1.9s)
    TOP:  ss(0.08) | '--(0.07) | æĪĳä¸į(0.07) | ä»Ĭå¤©æĪĳä»¬(0.07) | wrap(0.07) | å®ĥ(0.07) | ç±³(0.07) | oro(0.07)
    BOT:  _sql(-0.08) | Logged(-0.07) | æľºæ¢°(-0.07) | .copy(-0.07) | Typography(-0.07) | _output(-0.07) | æ»¤(-0.07) | Stub(-0.07)
    ACCEPTED as axis_572  cumulative_var=0.6797

  [ 568]  axes=573  step_var=0.0013  binary_acc=0.990  gap=0.1334  max_dot=0.0043  (1.8s)
    TOP:  -back(0.07) | .m(0.07) | çļĦæĿĥåĪ©(0.07) | _wp(0.07) | -property(0.07) | elial(0.07) | .emp(0.07) | nam(0.07)
    BOT:  Ãł(-0.08) | æĺĤ(-0.08) | åİĨåı²ä¸Ĭ(-0.07) | åĲĥ(-0.07) | _ctx(-0.07) | ANN(-0.07) | block(-0.07) | åĨĻ(-0.07)
    ACCEPTED as axis_573  cumulative_var=0.6802

  [ 569]  axes=574  step_var=0.0014  binary_acc=0.962  gap=0.1294  max_dot=0.0073  (1.8s)
    TOP:  ager(0.08) | -H(0.08) | adult(0.07) | Native(0.07) | >b(0.07) | ÙĨ(0.07) | çĶ°(0.07) | Zar(0.07)
    BOT:  .pre(-0.09) | %)(-0.08) | ?)(-0.08) | NOTE(-0.07) | cÃ²n(-0.07) | arat(-0.07) | \"(-0.07) | Args(-0.07)
    ACCEPTED as axis_574  cumulative_var=0.6806

  [ 570]  axes=575  step_var=0.0013  binary_acc=0.995  gap=0.1301  max_dot=0.0022  (1.9s)
    TOP:  Integrity(0.08) | åĲĮæĦı(0.07) | .in(0.07) | éĺ´éĺ³(0.07) | egrity(0.07) | dang(0.07) | ungkin(0.07) | aur(0.07)
    BOT:  +\(-0.09) | åįĥ(-0.08) | ;čĊ(-0.07) | '&(-0.07) | ------------------------------------------------(-0.07) | çī¹åĪ«æĺ¯(-0.07) | }-(-0.07) | Tiles(-0.07)
    ACCEPTED as axis_575  cumulative_var=0.6810

  [ 571]  axes=576  step_var=0.0013  binary_acc=0.992  gap=0.1311  max_dot=0.0034  (1.8s)
    TOP:  UN(0.08) | ,'(0.08) | Bat(0.08) | books(0.07) | .is(0.07) | Hotel(0.07) | Additional(0.07) | _G(0.07)
    BOT:  exclus(-0.07) | ied(-0.07) | ãģĤãģªãģŁ(-0.07) | Ð¼ÐµÐ½ÑĮÑĪÐµ(-0.07) | Augustine(-0.07) | å³»(-0.07) | Ã¤(-0.07) | æİ¥åľ°(-0.07)
    ACCEPTED as axis_576  cumulative_var=0.6814

  [ 572]  axes=577  step_var=0.0013  binary_acc=1.000  gap=0.1283  max_dot=0.0043  (1.9s)
    TOP:  .My(0.08) | .objects(0.07) | @author(0.07) | .Key(0.07) | _L(0.07) | çļĦèĢģ(0.07) | .I(0.07) | (Int(0.07)
    BOT:  éĽį(-0.09) | ```(-0.08) | åıĤçħ§(-0.08) | ##(-0.08) | conn(-0.08) | éŀį(-0.07) | æĭĴç»Ŀ(-0.07) | sp(-0.07)
    ACCEPTED as axis_577  cumulative_var=0.6819

  [ 573]  axes=578  step_var=0.0013  binary_acc=0.985  gap=0.1308  max_dot=0.0020  (1.8s)
    TOP:  <int(0.07) | .internal(0.07) | ì¤ĳ(0.07) | ãģ«å¯¾ãģĻãĤĭ(0.07) | åĽłä¸ºåľ¨(0.07) | åıĪç§°(0.07) | åĨ³å®ļ(0.07) | âĨĳ(0.07)
    BOT:  h(-0.08) | sole(-0.07) | Bus(-0.07) | (a(-0.07) | ]}(-0.07) | })(-0.07) | æ²®(-0.07) | ç»ıæµİç¤¾ä¼ļ(-0.07)
    ACCEPTED as axis_578  cumulative_var=0.6823

  [ 574]  axes=579  step_var=0.0013  binary_acc=0.988  gap=0.1272  max_dot=0.0041  (1.8s)
    TOP:  {!(0.09) | try(0.08) | å¤§å®¶åı¯ä»¥(0.07) | TOCOL(0.07) | ÑģÐ¾(0.07) | èªŀ(0.07) | }},Ċ(0.07) | Â®,(0.07)
    BOT:  HH(-0.08) | /w(-0.07) | ãĤĵ(-0.07) | NODE(-0.07) | ä¹ĭéĹ´çļĦ(-0.07) | -the(-0.07) | tend(-0.07) | \E(-0.07)
    ACCEPTED as axis_579  cumulative_var=0.6827

  [ 575]  axes=580  step_var=0.0013  binary_acc=0.996  gap=0.1293  max_dot=0.0093  (1.8s)
    TOP:  (out(0.08) | Over(0.07) | override(0.07) | .transform(0.07) | .play(0.07) | (system(0.07) | =>(0.07) | usr(0.07)
    BOT:  _ARG(-0.08) | .pp(-0.07) | çľŁåģĩ(-0.07) | Desire(-0.07) | .gr(-0.07) | -width(-0.07) | Usage(-0.07) | ÐĳÑĢ(-0.06)
    ACCEPTED as axis_580  cumulative_var=0.6831

  [ 576]  axes=581  step_var=0.0013  binary_acc=0.990  gap=0.1249  max_dot=0.0048  (1.9s)
    TOP:  Car(0.07) | ea(0.07) | .vn(0.07) | udson(0.07) | è¿Ľåĩºåı£(0.07) | æĶ»åħĭ(0.07) | .cls(0.06) | .g(0.06)
    BOT:  æľ½(-0.08) | task(-0.08) | ÑĢÑı(-0.08) | mb(-0.07) | èº«(-0.07) | -color(-0.07) | è¿½(-0.07) | .soft(-0.07)
    ACCEPTED as axis_581  cumulative_var=0.6835

  [ 577]  axes=582  step_var=0.0013  binary_acc=0.998  gap=0.1300  max_dot=0.0070  (1.9s)
    TOP:  çİĭ(0.09) | Ð¿Ð°(0.08) | ãĥĵ(0.08) | çĵ´(0.07) | kre(0.07) | .oracle(0.07) | Cindy(0.07) | ple(0.07)
    BOT:  System(-0.08) | åĬłå¼º(-0.07) | 'm(-0.07) | Linear(-0.07) | onia(-0.07) | '#(-0.07) | prime(-0.07) | edi(-0.06)
    ACCEPTED as axis_582  cumulative_var=0.6839

  [ 578]  axes=583  step_var=0.0013  binary_acc=0.993  gap=0.1281  max_dot=0.0013  (1.8s)
    TOP:  TO(0.07) | è´µ(0.07) | _)Ċ(0.07) | ISSN(0.06) | UL(0.06) | GT(0.06) | Investing(0.06) | ØªØ¬(0.06)
    BOT:  }"(-0.08) | .extract(-0.08) | (ns(-0.07) | .w(-0.07) | she(-0.07) | .get(-0.07) | am(-0.07) | ],(-0.07)
    ACCEPTED as axis_583  cumulative_var=0.6843

  [ 579]  axes=584  step_var=0.0013  binary_acc=0.994  gap=0.1273  max_dot=0.0009  (1.8s)
    TOP:  ÃĤ(0.08) | æ±Łæ¹ĸ(0.07) | Am(0.07) | Pablo(0.07) | æıĲç¤º(0.07) | åįģä¸ª(0.07) | infra(0.07) | è®(0.07)
    BOT:  à¸¢(-0.08) | _sym(-0.08) | -time(-0.08) | n(-0.07) | erto(-0.07) | _padding(-0.07) | é¡¹(-0.07) | TAB(-0.07)
    ACCEPTED as axis_584  cumulative_var=0.6847

  [ 580]  axes=585  step_var=0.0013  binary_acc=0.965  gap=0.1303  max_dot=0.0028  (1.9s)
    TOP:  such(0.07) | æ²¸(0.07) | /**Ċ(0.07) | elling(0.07) | EDITOR(0.07) | .preventDefault(0.07) | è§Ĩ(0.07) | out(0.07)
    BOT:  æĺ¯ä»¥(-0.07) | .make(-0.07) | ç¤¾ä¼ļ(-0.07) | breadcrumb(-0.07) | /top(-0.07) | <M(-0.07) | ä¸ĭä¸Ģä¸ª(-0.07) | /database(-0.07)
    ACCEPTED as axis_585  cumulative_var=0.6852

  [ 581]  axes=586  step_var=0.0013  binary_acc=0.993  gap=0.1262  max_dot=0.0041  (1.9s)
    TOP:  _sql(0.08) | .java(0.07) | æģĭ(0.07) | copy(0.07) | è¿Ķ(0.07) | ä¸ĢåŃĹ(0.07) | äº¡(0.07) | [$(0.07)
    BOT:  stra(-0.08) | _calls(-0.07) | _SC(-0.07) | _score(-0.07) | Ci(-0.07) | ru(-0.07) | iw(-0.07) | _G(-0.07)
    ACCEPTED as axis_586  cumulative_var=0.6855

  [ 582]  axes=587  step_var=0.0013  binary_acc=0.973  gap=0.1246  max_dot=0.0006  (1.9s)
    TOP:  f(0.08) | #region(0.08) | ch(0.07) | [F(0.07) | =R(0.07) | frame(0.07) | .X(0.07) | çļĦçĹĩçĬ¶(0.07)
    BOT:  ],(-0.10) | Ŀ(-0.08) | .,(-0.08) | Note(-0.08) | ØĮ(-0.07) | <<(-0.07) | ..(-0.07) | sent(-0.07)
    ACCEPTED as axis_587  cumulative_var=0.6860

  [ 583]  axes=588  step_var=0.0013  binary_acc=0.973  gap=0.1283  max_dot=0.0008  (1.8s)
    TOP:  _factor(0.08) | ns(0.07) | éĩĩåıĸ(0.07) | metric(0.07) | .[(0.07) | ×Ļ×Ŀ(0.07) | ,[(0.07) | Ãī(0.07)
    BOT:  pred(-0.07) | é»(-0.07) | a(-0.07) | ĉa(-0.07) | Ð®(-0.07) | æĢ»ç»Ł(-0.07) | noc(-0.06) | å®¢è½¦(-0.06)
    ACCEPTED as axis_588  cumulative_var=0.6864

  [ 584]  axes=589  step_var=0.0013  binary_acc=0.961  gap=0.1243  max_dot=0.0079  (1.9s)
    TOP:  .Int(0.08) | Fs(0.08) | ,S(0.08) | .S(0.08) | _skip(0.08) | Fs(0.07) | M(0.07) | know(0.07)
    BOT:  proxy(-0.08) | Ø¨Ø§(-0.08) | abc(-0.08) | .application(-0.07) | /the(-0.07) | ÑģÑĤÑĢÐ¾Ðº(-0.07) | ká»³(-0.07) | .avg(-0.07)
    ACCEPTED as axis_589  cumulative_var=0.6868

  [ 585]  axes=590  step_var=0.0013  binary_acc=0.988  gap=0.1308  max_dot=0.0024  (1.9s)
    TOP:  ç«ŀäºī(0.09) | æĹ¨åľ¨(0.08) | /user(0.08) | å¹¿å·ŀ(0.07) | hek(0.07) | æĸ¹æ³ķ(0.07) | (F(0.07) | vidia(0.07)
    BOT:  .put(-0.08) | ano(-0.08) | æĻº(-0.08) | tk(-0.08) | å»ī(-0.07) | am(-0.07) | IS(-0.07) | ris(-0.07)
    ACCEPTED as axis_590  cumulative_var=0.6872

  [ 586]  axes=591  step_var=0.0013  binary_acc=0.996  gap=0.1258  max_dot=0.0107  (1.8s)
    TOP:  æī(0.08) | ++,(0.07) | ®(0.07) | El(0.07) | ,T(0.07) | El(0.07) | å¤ĦçĲĨ(0.07) | Parameter(0.07)
    BOT:  _rand(-0.07) | ä¸įäºĨ(-0.07) | çļĩ(-0.07) | Last(-0.07) | xa(-0.07) | åħĳ(-0.07) | nested(-0.07) | æĭĵå±ķ(-0.06)
    ACCEPTED as axis_591  cumulative_var=0.6876

  [ 587]  axes=592  step_var=0.0013  binary_acc=0.987  gap=0.1237  max_dot=0.0029  (1.8s)
    TOP:  .static(0.07) | å¹´éĹ´(0.07) | *)(0.07) | *ĊĊĊ(0.07) | (Conv(0.07) | æĪĺçķ¥æĢ§(0.07) | åı¯çŁ¥(0.07) | ï¼ļ</(0.06)
    BOT:  fn(-0.09) | .m(-0.08) | ÑĬ(-0.08) | z(-0.08) | æ³¥(-0.07) | Z(-0.07) | .A(-0.07) | è°ģ(-0.07)
    ACCEPTED as axis_592  cumulative_var=0.6880

  [ 588]  axes=593  step_var=0.0013  binary_acc=0.946  gap=0.1298  max_dot=0.0018  (1.9s)
    TOP:  <!(0.08) | ä¸Ĭä¸ĭ(0.07) | (âĢľ(0.07) | shop(0.07) | org(0.07) | oc(0.07) | env(0.07) | rock(0.07)
    BOT:  -E(-0.08) | ç£¨(-0.08) | ģ(-0.08) | es(-0.07) | æ¤į(-0.07) | mad(-0.07) | mouth(-0.07) | Im(-0.07)
    ACCEPTED as axis_593  cumulative_var=0.6884

  [ 589]  axes=594  step_var=0.0013  binary_acc=0.997  gap=0.1314  max_dot=0.0034  (1.9s)
    TOP:  py(0.08) | ç¼ħ(0.08) | _MAP(0.07) | example(0.07) | proto(0.07) | .cmd(0.07) | éĢł(0.07) | /include(0.07)
    BOT:  (K(-0.09) | (W(-0.08) | (Response(-0.08) | //(-0.08) | .((-0.08) | ap(-0.07) | ï¼Ī(-0.07) | scanner(-0.07)
    ACCEPTED as axis_594  cumulative_var=0.6888

  [ 590]  axes=595  step_var=0.0013  binary_acc=0.991  gap=0.1284  max_dot=0.0034  (1.9s)
    TOP:  °(0.08) | Behavior(0.07) | bil(0.07) | "]Ċ(0.07) | Ð½Ð¸Ð·(0.07) | èİ·å¾Ĺ(0.07) | _rights(0.07) | ä»ĺè´¹(0.06)
    BOT:  -up(-0.07) | (s(-0.07) | toolbox(-0.07) | Ð»ÑıÑĢ(-0.07) | -builder(-0.07) | à¸±à¸ª(-0.07) | æ¸¸(-0.07) | inesis(-0.07)
    ACCEPTED as axis_595  cumulative_var=0.6892

  [ 591]  axes=596  step_var=0.0013  binary_acc=0.965  gap=0.1236  max_dot=0.0037  (1.9s)
    TOP:  ni(0.07) | |[(0.07) | _decay(0.07) | [in(0.07) | dark(0.06) | .sc(0.06) | }</(0.06) | (device(0.06)
    BOT:  à¹ĩà¸Ļ(-0.07) | æĬĹæ°§åĮĸ(-0.07) | åı«(-0.07) | Mer(-0.07) | æº¢(-0.06) | Sean(-0.06) | èĲ¥(-0.06) | å®ŀæĪĺ(-0.06)
    ACCEPTED as axis_596  cumulative_var=0.6896

  [ 592]  axes=597  step_var=0.0013  binary_acc=0.985  gap=0.1248  max_dot=0.0089  (1.8s)
    TOP:  .int(0.07) | ãĤī(0.07) | ØŃØ©(0.07) | Class(0.07) | Most(0.07) | éĻ¢(0.07) | .Utils(0.07) | (Item(0.07)
    BOT:  {\(-0.08) | (limit(-0.07) | {\(-0.07) | (torch(-0.07) | æĢ»äº§åĢ¼(-0.06) | æĿ¥äºĨ(-0.06) | ones(-0.06) | Preferences(-0.06)
    ACCEPTED as axis_597  cumulative_var=0.6900

  [ 593]  axes=598  step_var=0.0012  binary_acc=0.988  gap=0.1213  max_dot=0.0024  (1.8s)
    TOP:  .from(0.08) | /b(0.08) | /O(0.07) | erv(0.07) | (false(0.07) | sn(0.07) | Db(0.07) | sentences(0.07)
    BOT:  ä¸įåĬ¨äº§(-0.07) | ä¸Ģä»½(-0.07) | vendor(-0.07) | ^[(-0.07) | ä¸Ģæĸ¹(-0.07) | `'(-0.06) | Cha(-0.06) | No(-0.06)
    ACCEPTED as axis_598  cumulative_var=0.6904

  [ 594]  axes=599  step_var=0.0013  binary_acc=0.993  gap=0.1262  max_dot=0.0056  (1.9s)
    TOP:  ITE(0.08) | çĽĲ(0.07) | -point(0.07) | çī©(0.07) | åĮĸ(0.07) | "@(0.07) | Connection(0.07) | ubble(0.07)
    BOT:  ..(-0.07) | _principal(-0.07) | db(-0.07) | %.ĊĊ(-0.07) | ,P(-0.06) | ch(-0.06) | Focus(-0.06) | nda(-0.06)
    ACCEPTED as axis_599  cumulative_var=0.6908

  [ 595]  axes=600  step_var=0.0013  binary_acc=0.971  gap=0.1265  max_dot=0.0012  (1.8s)
    TOP:  ÃĦ(0.08) | asa(0.07) | `ĊĊ(0.07) | builtin(0.07) | encoder(0.07) | OUT(0.06) | è²·(0.06) | åı¤åħ¸(0.06)
    BOT:  -K(-0.08) | .rest(-0.07) | éļ¶å±ŀ(-0.07) | /w(-0.07) | -y(-0.07) | Am(-0.07) | /file(-0.07) | governing(-0.07)
    ACCEPTED as axis_600  cumulative_var=0.6912

  [ 596]  axes=601  step_var=0.0013  binary_acc=0.979  gap=0.1241  max_dot=0.0052  (1.9s)
    TOP:  TZ(0.08) | ody(0.08) | âĤ¬(0.07) | èķĬ(0.07) | åĮª(0.07) | iet(0.07) | ung(0.07) | éģĹ(0.07)
    BOT:  _n(-0.07) | .m(-0.07) | -z(-0.07) | .wikipedia(-0.07) | _s(-0.07) | _gener(-0.07) | -S(-0.07) | _matrices(-0.07)
    ACCEPTED as axis_601  cumulative_var=0.6916

  [ 597]  axes=602  step_var=0.0013  binary_acc=0.981  gap=0.1277  max_dot=0.0089  (1.9s)
    TOP:  à¸ŀà¸¥(0.07) | Ø±Ø¨(0.07) | Ð¿ÑĢ(0.07) | è´Ń(0.07) | @(0.07) | NUnit(0.06) | åĽ½éĻħ(0.06) | ç®¡çĲĨæĿ¡ä¾ĭ(0.06)
    BOT:  _arch(-0.08) | eren(-0.07) | []ĊĊ(-0.07) | -inter(-0.07) | tc(-0.07) | =c(-0.07) | ,y(-0.07) | :j(-0.07)
    ACCEPTED as axis_602  cumulative_var=0.6920

  [ 598]  axes=603  step_var=0.0013  binary_acc=0.995  gap=0.1254  max_dot=0.0020  (1.8s)
    TOP:  pg(0.08) | Ð¸ÑĦ(0.07) | #if(0.06) | èĩ¾(0.06) | eh(0.06) | coe(0.06) | afc(0.06) | .contains(0.06)
    BOT:  Michael(-0.09) | %.(-0.08) | LIB(-0.08) | *.(-0.08) | ".(-0.07) | >/(-0.07) | ',"(-0.07) | å®ļäºĨ(-0.07)
    ACCEPTED as axis_603  cumulative_var=0.6924

  [ 599]  axes=604  step_var=0.0013  binary_acc=0.995  gap=0.1242  max_dot=0.0025  (1.9s)
    TOP:  ä¸ĭåĪĹ(0.08) | .toString(0.07) | Ĩ(0.07) | Ð¿Ð¾(0.07) | Dump(0.07) | å¢ŀåĢ¼ç¨İ(0.07) | äºĮç»´çłģ(0.07) | æŃ(0.07)
    BOT:  /l(-0.08) | ÐºÑĢ(-0.07) | [],Ċ(-0.06) | ional(-0.06) | amino(-0.06) | It(-0.06) | çİ¯èĬĤ(-0.06) | Ã¤n(-0.06)
    ACCEPTED as axis_604  cumulative_var=0.6928

  [ 600]  axes=605  step_var=0.0012  binary_acc=0.975  gap=0.1275  max_dot=0.0028  (1.8s)
    TOP:  ]Ċ(0.07) | Dar(0.07) | SESSION(0.07) | ):(0.07) | >::(0.07) | Fit(0.07) | ...Ċ(0.07) | unlimited(0.07)
    BOT:  å¾·åĽ½(-0.08) | .web(-0.07) | .parse(-0.07) | Z(-0.07) | boost(-0.07) | pre(-0.07) | bal(-0.07) | -K(-0.07)
    ACCEPTED as axis_605  cumulative_var=0.6931

  [ 601]  axes=606  step_var=0.0012  binary_acc=0.992  gap=0.1243  max_dot=0.0160  (1.9s)
    TOP:  åįĬ(0.08) | Dans(0.07) | .To(0.07) | >,Ċ(0.07) | è¿ĩ(0.07) | (0.07) | />.ĊĊ(0.07) | .)Ċ(0.07)
    BOT:  .im(-0.07) | OPEN(-0.06) | travel(-0.06) | Script(-0.06) | æķ°åŃĹ(-0.06) | _board(-0.06) | Ce(-0.06) | .Print(-0.06)
    ACCEPTED as axis_606  cumulative_var=0.6935

  [ 602]  axes=607  step_var=0.0013  binary_acc=0.957  gap=0.1262  max_dot=0.0055  (1.8s)
    TOP:  -and(0.09) | And(0.08) | /ap(0.07) | (0.07) | _entry(0.07) | _field(0.07) | so(0.06) | Employer(0.06)
    BOT:  èĭ±æĸĩ(-0.08) | æĸĩ(-0.07) | .Field(-0.07) | å»ºæĪĲ(-0.07) | Expo(-0.07) | cursor(-0.07) | Players(-0.07) | Ø¬(-0.07)
    ACCEPTED as axis_607  cumulative_var=0.6939

  [ 603]  axes=608  step_var=0.0013  binary_acc=0.993  gap=0.1231  max_dot=0.0038  (1.8s)
    TOP:  -A(0.08) | .import(0.07) | .support(0.07) | reference(0.07) | éł(0.07) | _E(0.07) | /pages(0.06) | _STR(0.06)
    BOT:  ×ĵ×¨(-0.07) | èĥ±(-0.07) | Actions(-0.07) | bidi(-0.07) | ime(-0.07) | Remote(-0.07) | ulo(-0.07) | >/(-0.07)
    ACCEPTED as axis_608  cumulative_var=0.6943

  [ 604]  axes=609  step_var=0.0012  binary_acc=0.978  gap=0.1233  max_dot=0.0022  (1.9s)
    TOP:  =l(0.07) | (L(0.07) | éĤ¬(0.07) | .collect(0.07) | /the(0.07) | (this(0.07) | +y(0.07) | åĲįä¹ī(0.07)
    BOT:  HF(-0.07) | Element(-0.07) | bian(-0.07) | _html(-0.07) | åĦ¿(-0.07) | intentional(-0.06) | mise(-0.06) | äººå·¥æĻºèĥ½(-0.06)
    ACCEPTED as axis_609  cumulative_var=0.6947

  [ 605]  axes=610  step_var=0.0013  binary_acc=1.000  gap=0.1241  max_dot=0.0117  (1.8s)
    TOP:  nts(0.07) | ãģķãĤĵ(0.07) | q(0.07) | For(0.07) | _K(0.07) | _it(0.07) | .keys(0.06) | Bridge(0.06)
    BOT:  FirstName(-0.08) | fb(-0.07) | ä¸įè¦ģ(-0.07) | fol(-0.07) | cy(-0.07) | rame(-0.07) | Gar(-0.06) | .Message(-0.06)
    ACCEPTED as axis_610  cumulative_var=0.6951

  [ 606]  axes=611  step_var=0.0012  binary_acc=0.974  gap=0.1203  max_dot=0.0027  (1.8s)
    TOP:  .microsoft(0.07) | Pavel(0.07) | ÑĳÐ¼(0.07) | æĺ¥èĬĤ(0.06) | ype(0.06) | enced(0.06) | çļĦåħ·ä½ĵ(0.06) | .es(0.06)
    BOT:  quence(-0.08) | åıĮ(-0.08) | ------Ċ(-0.07) | Wa(-0.07) | tree(-0.07) | ib(-0.07) | MISS(-0.07) | Solutions(-0.07)
    ACCEPTED as axis_611  cumulative_var=0.6954

  [ 607]  axes=612  step_var=0.0013  binary_acc=0.998  gap=0.1232  max_dot=0.0021  (1.8s)
    TOP:  (metadata(0.07) | Ð²Ð¾Ð¿ÑĢÐ¾Ñģ(0.07) | Ð½Ñĭ(0.07) | ä¸ĩ(0.07) | -y(0.06) | von(0.06) | æľ¨(0.06) | items(0.06)
    BOT:  ç»ĻæĪĳ(-0.07) | ä¸Ģç§į(-0.07) | bool(-0.07) | Selected(-0.07) | æŃ¥éª¤(-0.07) | edge(-0.07) | .format(-0.06) | å¾®ç¬ĳ(-0.06)
    ACCEPTED as axis_612  cumulative_var=0.6958

  [ 608]  axes=613  step_var=0.0013  binary_acc=0.979  gap=0.1263  max_dot=0.0045  (1.8s)
    TOP:  à¸Ħ(0.08) | Si(0.07) | .ent(0.07) | /wiki(0.07) | :\\(0.07) | cel(0.07) | .TH(0.07) | .interface(0.07)
    BOT:  .kernel(-0.07) | ÑĥÑĪ(-0.07) | Awareness(-0.07) | anderen(-0.07) | unce(-0.07) | ela(-0.07) | Ð°ÑĤÐ°(-0.07) | .binary(-0.07)
    ACCEPTED as axis_613  cumulative_var=0.6962

  [ 609]  axes=614  step_var=0.0012  binary_acc=0.999  gap=0.1215  max_dot=0.0023  (1.8s)
    TOP:  }&(0.08) | è¯·æ±Ĥ(0.08) | ->(0.07) | :.(0.07) | è®¸åı¯(0.07) | éĻĲåĪ¶(0.07) | .push(0.07) | _folder(0.07)
    BOT:  Henrik(-0.07) | urllib(-0.06) | æ¢ħ(-0.06) | .Stage(-0.06) | NASDAQ(-0.06) | DE(-0.06) | groupBy(-0.06) | yi(-0.06)
    ACCEPTED as axis_614  cumulative_var=0.6966

  [ 610]  axes=615  step_var=0.0012  binary_acc=0.950  gap=0.1232  max_dot=0.0046  (1.9s)
    TOP:  /on(0.07) | æĶ¹å»º(0.07) | .business(0.06) | Meta(0.06) | æ»(0.06) | èĥİ(0.06) | çªĹåı£(0.06) | Validator(0.06)
    BOT:  ()))Ċ(-0.08) | tra(-0.08) | ç²īä¸Ŀ(-0.07) | åĨħé¥°(-0.07) | ]]Ċ(-0.07) | -inv(-0.07) | oda(-0.07) | åħĪè¿Ľ(-0.07)
    ACCEPTED as axis_615  cumulative_var=0.6970

  [ 611]  axes=616  step_var=0.0012  binary_acc=0.989  gap=0.1222  max_dot=0.0017  (1.8s)
    TOP:  èĪªçıŃ(0.08) | 'm(0.07) | æĤ¨(0.07) | //ĊĊ(0.07) | ,J(0.06) | å½ĵä½ł(0.06) | ëĭ¤ë¥¸(0.06) | Quick(0.06)
    BOT:  åĳ¨(-0.07) | Playlist(-0.06) | _utils(-0.06) | -details(-0.06) | Rel(-0.06) | _USE(-0.06) | Resolver(-0.06) | Ãº(-0.06)
    ACCEPTED as axis_616  cumulative_var=0.6973

  [ 612]  axes=617  step_var=0.0012  binary_acc=0.965  gap=0.1228  max_dot=0.0113  (1.9s)
    TOP:  [...(0.08) | ãģł(0.07) | ?:(0.07) | <\/(0.07) | Regression(0.06) | Tomorrow(0.06) | Miss(0.06) | _core(0.06)
    BOT:  code(-0.09) | share(-0.07) | çĶµåĬĽ(-0.07) | DO(-0.07) | Redistribution(-0.07) | åĩĢ(-0.07) | WARE(-0.07) | Or(-0.07)
    ACCEPTED as axis_617  cumulative_var=0.6977

  [ 613]  axes=618  step_var=0.0013  binary_acc=0.979  gap=0.1281  max_dot=0.0078  (1.8s)
    TOP:  (0.07) | ./(0.07) | Ĉ(0.07) | æ³µ(0.07) | Loss(0.06) | gather(0.06) | Laura(0.06) | .standard(0.06)
    BOT:  -y(-0.07) | breadcrumbs(-0.07) | DON(-0.07) | ÑĥÐº(-0.07) | å¦¨(-0.06) | ancer(-0.06) | cho(-0.06) | å¾Ĺå¾Ī(-0.06)
    ACCEPTED as axis_618  cumulative_var=0.6981

  [ 614]  axes=619  step_var=0.0013  binary_acc=0.967  gap=0.1268  max_dot=0.0050  (1.9s)
    TOP:  photo(0.07) | export(0.07) | ä¸ºä¸Ńå¿ĥ(0.07) | _reward(0.07) | statistics(0.07) | handle(0.07) | monitor(0.07) | ata(0.06)
    BOT:  ÙĦÙĬÙĦ(-0.07) | Arch(-0.07) | rait(-0.07) | khÃ´ng(-0.07) | seud(-0.06) | Ðĳ(-0.06) | åħļçļĦ(-0.06) | QUEUE(-0.06)
    ACCEPTED as axis_619  cumulative_var=0.6985

  [ 615]  axes=620  step_var=0.0013  binary_acc=0.993  gap=0.1220  max_dot=0.0026  (1.8s)
    TOP:  ÑĢÐ°(0.08) | æĪĳè§īå¾Ĺ(0.07) | .cr(0.07) | -python(0.07) | (plugin(0.07) | ios(0.07) | (("(0.07) | nton(0.07)
    BOT:  Indexed(-0.08) | target(-0.07) | stable(-0.07) | Configure(-0.07) | âī¥(-0.07) | archivo(-0.07) | channel(-0.06) | Ø¯Ø±(-0.06)
    ACCEPTED as axis_620  cumulative_var=0.6989

  [ 616]  axes=621  step_var=0.0013  binary_acc=0.996  gap=0.1223  max_dot=0.0019  (1.9s)
    TOP:  _tag(0.07) | -dev(0.07) | WRITE(0.07) | _View(0.07) | ony(0.07) | Ø¨Ùĩ(0.07) | Top(0.07) | ipt(0.06)
    BOT:  erry(-0.07) | Tra(-0.07) | base(-0.07) | \":(-0.06) | Pis(-0.06) | qr(-0.06) | ìĿĦ(-0.06) | ãģ³(-0.06)
    ACCEPTED as axis_621  cumulative_var=0.6992

  [ 617]  axes=622  step_var=0.0012  binary_acc=0.990  gap=0.1219  max_dot=0.0041  (1.8s)
    TOP:  ï¼Ļ(0.07) | 6(0.07) | Icon(0.07) | ]/(0.07) | new(0.07) | StyleSheet(0.07) | 3(0.06) | w(0.06)
    BOT:  aman(-0.08) | .testng(-0.07) | ystems(-0.07) | min(-0.07) | Special(-0.07) | Library(-0.07) | )$(-0.07) | ä¸Ńæĸ°ç½ĳ(-0.07)
    ACCEPTED as axis_622  cumulative_var=0.6996

  [ 618]  axes=623  step_var=0.0012  binary_acc=0.988  gap=0.1247  max_dot=0.0033  (1.8s)
    TOP:  ãĥª(0.08) | utf(0.07) | ######(0.07) | behind(0.07) | cole(0.07) | n(0.06) | Browser(0.06) | à¦(0.06)
    BOT:  åľ°çĲĥ(-0.08) | _func(-0.08) | Their(-0.07) | åıĸãĤĬ(-0.07) | base(-0.07) | .open(-0.07) | .invoke(-0.07) | .effect(-0.07)
    ACCEPTED as axis_623  cumulative_var=0.7000

  [ 619]  axes=624  step_var=0.0013  binary_acc=0.996  gap=0.1220  max_dot=0.0024  (1.8s)
    TOP:  o(0.07) | Charles(0.07) | /help(0.07) | hÃ©(0.07) | (Exception(0.07) | (utils(0.07) | /login(0.06) | :e(0.06)
    BOT:  oretical(-0.08) | -date(-0.08) | attles(-0.07) | Network(-0.07) | Ð²ÐµÑĢ(-0.07) | Maritime(-0.07) | "]),Ċ(-0.07) | åĲĳ(-0.07)
    ACCEPTED as axis_624  cumulative_var=0.7004

  [ 620]  axes=625  step_var=0.0012  binary_acc=0.991  gap=0.1190  max_dot=0.0015  (1.8s)
    TOP:  Location(0.08) | Ðĳ(0.07) | sum(0.07) | AYOUT(0.07) | è°ı(0.06) | éĤ£æł·(0.06) | æĬ¥åĳĬ(0.06) | Beacon(0.06)
    BOT:  me(-0.08) | èĲ¥éĶĢ(-0.07) | ARR(-0.07) | èº«ä»½(-0.07) | æ³¨æĺİåĩºå¤Ħ(-0.06) | -g(-0.06) | èĢģæĿ¿(-0.06) | to(-0.06)
    ACCEPTED as axis_625  cumulative_var=0.7007

  [ 621]  axes=626  step_var=0.0013  binary_acc=0.965  gap=0.1229  max_dot=0.0078  (1.9s)
    TOP:  _wrapper(0.08) | sw(0.07) | lying(0.07) | jh(0.07) | test(0.07) | _cost(0.07) | mathrm(0.07) | sci(0.07)
    BOT:  Antonio(-0.08) | åĢºåĬ¡(-0.07) | ä¼Ĭæĸ¯(-0.07) | é¸£(-0.07) | çļĦæĸĩåŃĹ(-0.07) | ç½ĳåĿĢ(-0.07) | Ð¸Ð½Ð³(-0.07) | åħĭæĢĿ(-0.07)
    ACCEPTED as axis_626  cumulative_var=0.7011

  [ 622]  axes=627  step_var=0.0013  binary_acc=0.971  gap=0.1272  max_dot=0.0010  (1.8s)
    TOP:  åı¦æľī(0.07) | Field(0.07) | Developers(0.07) | fun(0.07) | .Basic(0.06) | åıªè¦ģ(0.06) | æł¹æľ¬(0.06) | è®¾(0.06)
    BOT:  eh(-0.09) | Ñĸ(-0.08) | .uc(-0.08) | Ð¾Ð¿(-0.08) | =image(-0.08) | /ip(-0.07) | Timestamp(-0.07) | ologists(-0.07)
    ACCEPTED as axis_627  cumulative_var=0.7015

  [ 623]  axes=628  step_var=0.0012  binary_acc=0.997  gap=0.1196  max_dot=0.0012  (1.8s)
    TOP:  .intellij(0.08) | hash(0.07) | éĹ´(0.07) | ÑģÑĤÐ¸(0.06) | Â¢(0.06) | Penalty(0.06) | ----------------------------(0.06) | rÃ©(0.06)
    BOT:  ])(-0.07) | Prop(-0.07) | /js(-0.06) | çļĦè®¤è¯Ĩ(-0.06) | ĉif(-0.06) | EN(-0.06) | INC(-0.06) | )",Ċ(-0.06)
    ACCEPTED as axis_628  cumulative_var=0.7019

  [ 624]  axes=629  step_var=0.0012  binary_acc=0.966  gap=0.1196  max_dot=0.0043  (1.9s)
    TOP:  çĲĨè§£(0.07) | _CHECK(0.07) | adget(0.07) | *ĊĊ(0.07) | Add(0.06) | ÑĥÐ¶(0.06) | à¤ĸ(0.06) | Natural(0.06)
    BOT:  Card(-0.08) | pez(-0.07) | _desc(-0.07) | .stream(-0.07) | Prev(-0.06) | reaction(-0.06) | ç®¡çĲĨæĿ¡ä¾ĭ(-0.06) | /filter(-0.06)
    ACCEPTED as axis_629  cumulative_var=0.7022

  [ 625]  axes=630  step_var=0.0012  binary_acc=0.989  gap=0.1225  max_dot=0.0051  (1.9s)
    TOP:  AG(0.07) | Ð°Ð·(0.06) | Ð¿ÑĢÐ¸(0.06) | .int(0.06) | _template(0.06) | scriber(0.06) | Smooth(0.06) | up(0.06)
    BOT:  ãģĽ(-0.07) | .(*(-0.07) | "].(-0.07) | èĩ´(-0.07) | ç½®(-0.07) | Ð±Ñĭ(-0.07) | pad(-0.07) | .V(-0.07)
    ACCEPTED as axis_630  cumulative_var=0.7026

  [ 626]  axes=631  step_var=0.0012  binary_acc=0.981  gap=0.1212  max_dot=0.0072  (1.9s)
    TOP:  -Ċ(0.07) | tant(0.07) | å¯Ĩåº¦(0.07) | -g(0.07) | CN(0.07) | '/(0.06) | RF(0.06) | èĳĹåĲįçļĦ(0.06)
    BOT:  à¹ģà¸ķ(-0.07) | ampus(-0.07) | _DIALOG(-0.06) | output(-0.06) | ç¼Ł(-0.06) | lm(-0.06) | ])),(-0.06) | Bruce(-0.06)
    ACCEPTED as axis_631  cumulative_var=0.7030

  [ 627]  axes=632  step_var=0.0012  binary_acc=0.998  gap=0.1202  max_dot=0.0092  (1.8s)
    TOP:  Log(0.08) | æ²Ļæ»©(0.07) | Ð¼Ð¸Ð½(0.07) | -header(0.07) | ivi(0.07) | .interface(0.07) | çİī(0.07) | .make(0.06)
    BOT:  filename(-0.06) | digit(-0.06) | _serial(-0.06) | éĽ»è©±(-0.06) | ä»İä¸ļäººåĳĺ(-0.06) | intelligence(-0.06) | SPDX(-0.06) | Animation(-0.06)
    ACCEPTED as axis_632  cumulative_var=0.7033

  [ 628]  axes=633  step_var=0.0012  binary_acc=0.972  gap=0.1213  max_dot=0.0016  (1.9s)
    TOP:  }/(0.09) | Builder(0.08) | åįķä½į(0.08) | éĽĨ(0.07) | /public(0.07) | }\(0.07) | è¡¥åħħ(0.07) | "}(0.07)
    BOT:  Chem(-0.08) | agit(-0.06) | EM(-0.06) | AC(-0.06) | itch(-0.06) | á»ĩ(-0.06) | Solutions(-0.06) | UI(-0.06)
    ACCEPTED as axis_633  cumulative_var=0.7037

  [ 629]  axes=634  step_var=0.0012  binary_acc=0.992  gap=0.1224  max_dot=0.0111  (1.9s)
    TOP:  Payload(0.08) | Be(0.07) | è½®(0.07) | å°ıãģķãģĦ(0.07) | ä¸įåĲĪçĲĨ(0.07) | /common(0.07) | .Global(0.06) | ingroup(0.06)
    BOT:  Lew(-0.07) | usa(-0.07) | Ann(-0.07) | ARGE(-0.06) | AB(-0.06) | offs(-0.06) | .columns(-0.06) | Bruce(-0.06)
    ACCEPTED as axis_634  cumulative_var=0.7041

  [ 630]  axes=635  step_var=0.0013  binary_acc=0.984  gap=0.1246  max_dot=0.0053  (1.8s)
    TOP:  Figure(0.08) | face(0.07) | access(0.07) | OUTPUT(0.07) | they(0.07) | plus(0.07) | åĽłä¸ºæĪĳ(0.07) | Episode(0.07)
    BOT:  eti(-0.08) | .num(-0.08) | å¸¦æĿ¥äºĨ(-0.07) | loss(-0.07) | ÙĦÙī(-0.07) | related(-0.07) | åĨľæ°ĳ(-0.07) | Render(-0.07)
    ACCEPTED as axis_635  cumulative_var=0.7044

  [ 631]  axes=636  step_var=0.0012  binary_acc=0.999  gap=0.1178  max_dot=0.0078  (1.8s)
    TOP:  DateTime(0.08) | _on(0.07) | ListItem(0.07) | cat(0.07) | Material(0.07) | dr(0.07) | SDK(0.06) | cw(0.06)
    BOT:  (struct(-0.07) | Community(-0.07) | XI(-0.07) | Inc(-0.06) | Classroom(-0.06) | Matthew(-0.06) | .valid(-0.06) | .types(-0.06)
    ACCEPTED as axis_636  cumulative_var=0.7048

  [ 632]  axes=637  step_var=0.0012  binary_acc=0.987  gap=0.1206  max_dot=0.0052  (1.8s)
    TOP:  èµ¢å¾Ĺ(0.07) | å½ĵä»Ĭ(0.07) | weighs(0.07) | article(0.06) | \f(0.06) | iom(0.06) | è§(0.06) | .score(0.06)
    BOT:  Ð¾ÑģÑĤ(-0.07) | æİī(-0.07) | Ùĳ(-0.07) | à§(-0.07) | çģ¼(-0.07) | ¾(-0.07) | .application(-0.06) | ì¹ĺ(-0.06)
    ACCEPTED as axis_637  cumulative_var=0.7052

  [ 633]  axes=638  step_var=0.0012  binary_acc=0.985  gap=0.1203  max_dot=0.0020  (1.9s)
    TOP:  (X(0.08) | mm(0.08) | ï¼Ŀ(0.07) | st(0.07) | /my(0.07) | ç©´(0.07) | æĢ§(0.07) | ez(0.06)
    BOT:  |\(-0.07) | è§ī(-0.07) | <!--(-0.07) | inn(-0.06) | .exception(-0.06) | /svg(-0.06) | åĲĦä¸ªçİ¯èĬĤ(-0.06) | éĥ½åľ¨(-0.06)
    ACCEPTED as axis_638  cumulative_var=0.7055

  [ 634]  axes=639  step_var=0.0012  binary_acc=0.994  gap=0.1231  max_dot=0.0031  (1.9s)
    TOP:  /out(0.08) | æķ´ä¸ª(0.08) | _ELEMENT(0.07) | comment(0.07) | Urb(0.06) | åħļåĴĮåĽ½å®¶(0.06) | Ruiz(0.06) | åı¹(0.06)
    BOT:  ropa(-0.07) | ä¹ĭå®¶(-0.07) | Īĺ(-0.07) | ĉprivate(-0.06) | item(-0.06) | çº¢æĹĹ(-0.06) | .first(-0.06) | anco(-0.06)
    ACCEPTED as axis_639  cumulative_var=0.7059

  [ 635]  axes=640  step_var=0.0012  binary_acc=0.979  gap=0.1222  max_dot=0.0072  (1.8s)
    TOP:  (y(0.08) | å¥ĩ(0.07) | .auto(0.07) | ]);ĊĊ(0.07) | .dep(0.07) | æŁ¥éĺħ(0.07) | factory(0.07) | åīį(0.06)
    BOT:  9(-0.07) | _RGB(-0.07) | çļĦç»ĵæŀľ(-0.07) | Ã²n(-0.07) | åĴĮç¤¾ä¼ļ(-0.07) | Ãĥ(-0.07) | ãģķ(-0.07) | äºĨä¸Ģä¸ª(-0.07)
    ACCEPTED as axis_640  cumulative_var=0.7063

  [ 636]  axes=641  step_var=0.0012  binary_acc=0.990  gap=0.1198  max_dot=0.0061  (1.9s)
    TOP:  +=(0.08) | ä¸ª(0.07) | +)\(0.07) | -.(0.07) | law(0.07) | $((0.07) | -b(0.07) | Ð´Ð°Ð²Ð½Ð¾(0.07)
    BOT:  imize(-0.07) | afe(-0.07) | Lie(-0.06) | ame(-0.06) | asleep(-0.06) | meanings(-0.06) | donc(-0.06) | orb(-0.06)
    ACCEPTED as axis_641  cumulative_var=0.7066

  [ 637]  axes=642  step_var=0.0012  binary_acc=0.971  gap=0.1209  max_dot=0.0018  (1.9s)
    TOP:  èĩª(0.07) | è¡Ŀ(0.06) | íķĺê³ł(0.06) | æıĲ(0.06) | ìĿ´ëĿ¼(0.06) | è¾ĥ(0.06) | (.)(0.06) | <body(0.06)
    BOT:  å¹¿å¤§(-0.07) | cls(-0.07) | gressive(-0.07) | ä¸ĵå®¶(-0.07) | lastic(-0.07) | ih(-0.07) | åħ¬æĸ¤(-0.06) | eb(-0.06)
    ACCEPTED as axis_642  cumulative_var=0.7070

  [ 638]  axes=643  step_var=0.0012  binary_acc=0.991  gap=0.1208  max_dot=0.0140  (1.9s)
    TOP:  Parameters(0.07) | Man(0.07) | ĉprint(0.07) | options(0.06) | MAT(0.06) | Sale(0.06) | random(0.06) | PreparedStatement(0.06)
    BOT:  ^(-0.07) | çĻ¾(-0.07) | *(-0.07) | åĽĽ(-0.07) | à¸¥(-0.07) | 6(-0.06) | Ð¡(-0.06) | eb(-0.06)
    ACCEPTED as axis_643  cumulative_var=0.7073

  [ 639]  axes=644  step_var=0.0012  binary_acc=0.993  gap=0.1209  max_dot=0.0089  (2.0s)
    TOP:  )).(0.07) | ä¼ĺ(0.07) | reset(0.06) | ured(0.06) | isser(0.06) | seen(0.06) | metadata(0.06) | å¸®å¿Ļ(0.06)
    BOT:  =int(-0.07) | (script(-0.07) | .__(-0.07) | (use(-0.07) | '<(-0.07) | Combined(-0.06) | Preview(-0.06) | /*******************************************************************************Ċ(-0.06)
    ACCEPTED as axis_644  cumulative_var=0.7077

  [ 640]  axes=645  step_var=0.0012  binary_acc=0.997  gap=0.1198  max_dot=0.0021  (1.9s)
    TOP:  {'(0.07) | -in(0.07) | ä¸ĩéĩĮ(0.07) | Ð¼Ð°Ð³Ð°Ð·Ð¸Ð½(0.06) | ¿(0.06) | Ð²Ð¾(0.06) | åĶ¤(0.06) | .translation(0.06)
    BOT:  def(-0.08) | akt(-0.07) | æĪĲç«ĭ(-0.07) | reduce(-0.07) | -\(-0.07) | \[(-0.07) | line(-0.07) | l(-0.07)
    ACCEPTED as axis_645  cumulative_var=0.7080

  [ 641]  axes=646  step_var=0.0012  binary_acc=0.973  gap=0.1162  max_dot=0.0053  (1.8s)
    TOP:  ãĤ·(0.08) | .sw(0.07) | _engine(0.07) | åĨĻåĩº(0.07) | /blob(0.07) | Ø¨ÙĬ(0.07) | _P(0.07) | enumerate(0.07)
    BOT:  lo(-0.06) | Workflow(-0.06) | min(-0.06) | olicies(-0.06) | gá»Ńi(-0.06) | ica(-0.06) | ids(-0.06) | IEEE(-0.06)
    ACCEPTED as axis_646  cumulative_var=0.7084

  [ 642]  axes=647  step_var=0.0012  binary_acc=0.992  gap=0.1190  max_dot=0.0086  (1.8s)
    TOP:  _axis(0.07) | Regulation(0.07) | nce(0.06) | .apply(0.06) | Ð±ÑĭÐ»Ð¾(0.06) | Strings(0.06) | ä¸ĢäºĮ(0.06) | âĢĻ.(0.06)
    BOT:  (reply(-0.07) | ìŀĦ(-0.07) | -d(-0.06) | encing(-0.06) | Gut(-0.06) | sonra(-0.06) | "["(-0.06) | ip(-0.06)
    ACCEPTED as axis_647  cumulative_var=0.7087

  [ 643]  axes=648  step_var=0.0012  binary_acc=0.959  gap=0.1202  max_dot=0.0100  (1.9s)
    TOP:  -address(0.07) | DAMAGE(0.07) | ategorie(0.06) | /test(0.06) | fer(0.06) | .ru(0.06) | ëĤ¨(0.06) | inds(0.06)
    BOT:  App(-0.07) | Foundation(-0.07) | çŃīäºİ(-0.07) | -menu(-0.07) | DU(-0.07) | Prices(-0.07) | Ð½Ð¾ÑģÑĤÑĮ(-0.07) | æĶ¿æĿĥ(-0.06)
    ACCEPTED as axis_648  cumulative_var=0.7091

  [ 644]  axes=649  step_var=0.0012  binary_acc=0.998  gap=0.1186  max_dot=0.0022  (1.8s)
    TOP:  çĽ®çļĦ(0.07) | æĸĩ(0.07) | ore(0.06) | ÑħÐ¾(0.06) | Closing(0.06) | Weight(0.06) | VAT(0.06) | speed(0.06)
    BOT:  \-(-0.07) | äºĶåĽĽ(-0.07) | éĺħ(-0.07) | uo(-0.07) | èµĦæľ¬å¸Ĥåľº(-0.07) | äºīéľ¸(-0.07) | _struct(-0.07) | çĽŁ(-0.06)
    ACCEPTED as axis_649  cumulative_var=0.7095

  [ 645]  axes=650  step_var=0.0012  binary_acc=0.998  gap=0.1187  max_dot=0.0063  (1.8s)
    TOP:  Il(0.08) | æŁ³(0.07) | Project(0.07) | pr(0.06) | panied(0.06) | æĬķèµĦ(0.06) | çĬ¶æĢģä¸ĭ(0.06) | åªĴä»ĭ(0.06)
    BOT:  categories(-0.07) | "(-0.07) | .intellij(-0.07) | lag(-0.07) | ä¸įæĺ¯(-0.06) | -us(-0.06) | .now(-0.06) | und(-0.06)
    ACCEPTED as axis_650  cumulative_var=0.7098

  [ 646]  axes=651  step_var=0.0012  binary_acc=0.985  gap=0.1172  max_dot=0.0082  (1.8s)
    TOP:  åºĶçĶ¨(0.07) | /?(0.06) | Vendor(0.06) | thumbnail(0.06) | å¯¹å¤ĸ(0.06) | Å¯(0.06) | ieux(0.06) | Click(0.06)
    BOT:  .services(-0.07) | é£¨(-0.07) | åĻ¨(-0.07) | .typ(-0.07) | -package(-0.07) | Anderson(-0.06) | tick(-0.06) | .baidu(-0.06)
    ACCEPTED as axis_651  cumulative_var=0.7102

  [ 647]  axes=652  step_var=0.0012  binary_acc=0.975  gap=0.1178  max_dot=0.0026  (1.9s)
    TOP:  please(0.06) | igger(0.06) | ']);ĊĊ(0.06) | .user(0.06) | Doesn(0.06) | ==(0.06) | ]],(0.06) | _,(0.06)
    BOT:  _items(-0.07) | pháº©m(-0.07) | /stream(-0.07) | _location(-0.06) | -co(-0.06) | _RUN(-0.06) | RE(-0.06) | .jackson(-0.06)
    ACCEPTED as axis_652  cumulative_var=0.7105

  [ 648]  axes=653  step_var=0.0012  binary_acc=0.966  gap=0.1207  max_dot=0.0053  (1.9s)
    TOP:  åħ³(0.08) | çĽ´(0.07) | /sites(0.07) | ÃĢ(0.07) | æīĭ(0.07) | /start(0.07) | èĬ±(0.07) | æºĲ(0.06)
    BOT:  ."[(-0.07) | .arch(-0.07) | ãĢĤâĢĿĊĊ(-0.07) | ãĢĮ(-0.07) | .[(-0.07) | .Register(-0.07) | "))(-0.07) | (n(-0.06)
    ACCEPTED as axis_653  cumulative_var=0.7108

  [ 649]  axes=654  step_var=0.0012  binary_acc=0.974  gap=0.1198  max_dot=0.0091  (1.9s)
    TOP:  [\(0.07) | ^{(0.07) | .center(0.07) | Condition(0.07) | COLOR(0.07) | ]',(0.07) | Page(0.07) | (+(0.06)
    BOT:  Am(-0.08) | ef(-0.08) | Ł(-0.07) | isi(-0.07) | OC(-0.07) | Ä°(-0.07) | }}Ċ(-0.07) | .libs(-0.06)
    ACCEPTED as axis_654  cumulative_var=0.7112

  [ 650]  axes=655  step_var=0.0012  binary_acc=0.983  gap=0.1175  max_dot=0.0020  (1.8s)
    TOP:  /pkg(0.08) | 'S(0.07) | ercises(0.07) | =v(0.07) | ap(0.07) | h(0.07) | Cou(0.06) | Continue(0.06)
    BOT:  Errors(-0.06) | Bg(-0.06) | Validate(-0.06) | previous(-0.06) | Allocator(-0.06) | sel(-0.06) | _heap(-0.06) | Chef(-0.06)
    ACCEPTED as axis_655  cumulative_var=0.7116

  [ 651]  axes=656  step_var=0.0012  binary_acc=0.947  gap=0.1207  max_dot=0.0068  (1.8s)
    TOP:  /master(0.07) | .connection(0.07) | rpc(0.06) | DEALINGS(0.06) | Sdk(0.06) | Ïĥ(0.06) | à¥ĩà¤(0.06) | å°½éĩı(0.06)
    BOT:  __(-0.08) | æĪ·(-0.07) | amil(-0.07) | æĬ«(-0.07) | ,A(-0.07) | ç²®é£Ł(-0.07) | æ¬Ĭ(-0.07) | Ð¾Ð±Ñĭ(-0.07)
    ACCEPTED as axis_656  cumulative_var=0.7119

  [ 652]  axes=657  step_var=0.0012  binary_acc=0.964  gap=0.1189  max_dot=0.0060  (1.8s)
    TOP:  ."""ĊĊ(0.07) | %-(0.07) | _type(0.07) | âĢ²(0.06) | ##(0.06) | +Ċ(0.06) | ãģłãģ¨(0.06) | âĢĻ(0.06)
    BOT:  era(-0.07) | è¦ĸ(-0.07) | ive(-0.07) | åħĪ(-0.07) | Ã¡(-0.07) | =D(-0.07) | EX(-0.06) | add(-0.06)
    ACCEPTED as axis_657  cumulative_var=0.7122

  [ 653]  axes=658  step_var=0.0012  binary_acc=0.987  gap=0.1197  max_dot=0.0011  (1.8s)
    TOP:  (description(0.08) | _main(0.07) | IClient(0.07) | Fashion(0.07) | -net(0.06) | TABLE(0.06) | phys(0.06) | è¿ĻäºĽå¹´(0.06)
    BOT:  Audio(-0.06) | Ð±(-0.06) | æĪ´(-0.06) | Disability(-0.06) | ale(-0.06) | atus(-0.06) | ire(-0.06) | rieved(-0.06)
    ACCEPTED as axis_658  cumulative_var=0.7126

  [ 654]  axes=659  step_var=0.0012  binary_acc=0.981  gap=0.1174  max_dot=0.0036  (1.9s)
    TOP:  æĬĬ(0.08) | deve(0.07) | cliente(0.07) | -server(0.07) | _the(0.07) | ç¢§(0.07) | the(0.07) | -*-ĊĊ(0.07)
    BOT:  cas(-0.07) | Bluetooth(-0.06) | natur(-0.06) | Requirement(-0.06) | Met(-0.06) | xs(-0.06) | åŃ©åŃĲçļĦ(-0.06) | ('''(-0.06)
    ACCEPTED as axis_659  cumulative_var=0.7129

  [ 655]  axes=660  step_var=0.0011  binary_acc=0.993  gap=0.1165  max_dot=0.0023  (1.8s)
    TOP:  OWN(0.07) | .reply(0.07) | .image(0.07) | VALUES(0.07) | Idea(0.06) | software(0.06) | åĲįç§°(0.06) | Edge(0.06)
    BOT:  çł¥(-0.07) | ãĥ¼ãĤ¿(-0.07) | sr(-0.07) | äº²èĩª(-0.07) | esp(-0.07) | åŁ¹èĤ²(-0.06) | /support(-0.06) | åıĺåİĭåĻ¨(-0.06)
    ACCEPTED as axis_660  cumulative_var=0.7133

  [ 656]  axes=661  step_var=0.0011  binary_acc=0.984  gap=0.1175  max_dot=0.0050  (1.9s)
    TOP:  ÐµÐ½Ð½Ð¾Ð³Ð¾(0.07) | ãģĸ(0.06) | github(0.06) | .with(0.06) | åħ³äºİ(0.06) | Sc(0.06) | /{}/(0.06) | "=>(0.06)
    BOT:  .Instance(-0.07) | rieved(-0.07) | ä¸°å¯Į(-0.06) | trace(-0.06) | Continue(-0.06) | qe(-0.06) | éķ¶åµĮ(-0.06) | èĥİåĦ¿(-0.06)
    ACCEPTED as axis_661  cumulative_var=0.7136

  [ 657]  axes=662  step_var=0.0012  binary_acc=0.981  gap=0.1162  max_dot=0.0057  (1.9s)
    TOP:  ads(0.08) | episode(0.07) | -*-Ċ(0.07) | .github(0.07) | more(0.07) | hashlib(0.07) | ermo(0.06) | -six(0.06)
    BOT:  down(-0.07) | à¸²à¸¢(-0.07) | å±(-0.07) | åĨ(-0.07) | å¥¥(-0.07) | ÃĮ(-0.07) | .Column(-0.06) | .Set(-0.06)
    ACCEPTED as axis_662  cumulative_var=0.7139

  [ 658]  axes=663  step_var=0.0012  binary_acc=0.999  gap=0.1159  max_dot=0.0095  (1.9s)
    TOP:  /B(0.09) | .from(0.08) | _by(0.07) | _state(0.07) | Ð°Ð½(0.06) | .pro(0.06) | è±ĨèħĲ(0.06) | Ð°ÑĢ(0.06)
    BOT:  twelve(-0.06) | Qualified(-0.06) | ,,(-0.06) | termination(-0.06) | å¿ħ(-0.06) | Load(-0.06) | storms(-0.06) | ãģ§(-0.06)
    ACCEPTED as axis_663  cumulative_var=0.7143

  [ 659]  axes=664  step_var=0.0012  binary_acc=0.997  gap=0.1180  max_dot=0.0078  (1.8s)
    TOP:  }%(0.07) | ))/(0.07) | åĥı(0.07) | _q(0.07) | Public(0.06) | é¢ħ(0.06) | Vertex(0.06) | ÂŃ(0.06)
    BOT:  å¹¿ä¸ľçľģ(-0.07) | books(-0.07) | æİĴéĻ¤(-0.07) | Api(-0.06) | temps(-0.06) | çī¹åĪ«æĺ¯(-0.06) | éģĵçĲĨ(-0.06) | _IGNORE(-0.06)
    ACCEPTED as axis_664  cumulative_var=0.7146

  [ 660]  axes=665  step_var=0.0012  binary_acc=0.964  gap=0.1174  max_dot=0.0121  (1.9s)
    TOP:  ('(0.10) | ");Ċ(0.07) | åı¯ä»¥å¸®åĬ©(0.06) | _Z(0.06) | o(0.06) | _DESC(0.06) | ps(0.06) | aten(0.06)
    BOT:  -click(-0.08) | preprocess(-0.07) | Club(-0.07) | éĻ½(-0.06) | .es(-0.06) | Sequence(-0.06) | ham(-0.06) | gorithms(-0.06)
    ACCEPTED as axis_665  cumulative_var=0.7150

  [ 661]  axes=666  step_var=0.0012  binary_acc=0.994  gap=0.1172  max_dot=0.0025  (1.8s)
    TOP:  Ap(0.07) | é»ĺ(0.07) | ä¸įåĪ°(0.06) | rupted(0.06) | ÐµÐ±(0.06) | åĬł(0.06) | _shadow(0.06) | çļ®(0.06)
    BOT:  LICENSE(-0.07) | (ValueError(-0.06) | .navigation(-0.06) | .e(-0.06) | PD(-0.06) | /trans(-0.06) | _appro(-0.06) | km(-0.06)
    ACCEPTED as axis_666  cumulative_var=0.7153

  [ 662]  axes=667  step_var=0.0012  binary_acc=0.991  gap=0.1169  max_dot=0.0041  (1.9s)
    TOP:  As(0.07) | essment(0.07) | æīĢæıĲä¾ĽçļĦ(0.06) | _ver(0.06) | /node(0.06) | prite(0.06) | åħ¬ç§¯éĩĳ(0.06) | Ð¼Ð¾Ð¹(0.06)
    BOT:  Ch(-0.07) | Filter(-0.07) | (req(-0.07) | åĪĺ(-0.07) | åįİ(-0.06) | æĸ¡(-0.06) | Household(-0.06) | simples(-0.06)
    ACCEPTED as axis_667  cumulative_var=0.7156

  [ 663]  axes=668  step_var=0.0012  binary_acc=0.977  gap=0.1169  max_dot=0.0017  (1.9s)
    TOP:  _valid(0.07) | _zone(0.07) | æº¢(0.07) | .attach(0.07) | çĪ¾(0.07) | .serialization(0.07) | .gnu(0.06) | Professional(0.06)
    BOT:  __(-0.08) | ä¼ģä¸ļ(-0.07) | iating(-0.07) | Pr(-0.07) | lÃ¡(-0.06) | '''(-0.06) | findBy(-0.06) | ãģĹãģĭ(-0.06)
    ACCEPTED as axis_668  cumulative_var=0.7160

  [ 664]  axes=669  step_var=0.0012  binary_acc=0.985  gap=0.1165  max_dot=0.0077  (1.9s)
    TOP:  _style(0.07) | reader(0.07) | .Add(0.06) | (level(0.06) | placeholder(0.06) | -this(0.06) | Fonts(0.06) | .diff(0.06)
    BOT:  åľ°åĽ¾(-0.07) | ech(-0.07) | Primitive(-0.06) | è§£æĶ¾(-0.06) | Ut(-0.06) | .web(-0.06) | off(-0.06) | berger(-0.06)
    ACCEPTED as axis_669  cumulative_var=0.7163

  [ 665]  axes=670  step_var=0.0012  binary_acc=0.983  gap=0.1189  max_dot=0.0071  (1.8s)
    TOP:  noqa(0.07) | dc(0.07) | uary(0.06) | BUY(0.06) | kom(0.06) | format(0.06) | APP(0.06) | Cell(0.06)
    BOT:  ysql(-0.07) | åľĨæ»¡(-0.07) | æĹ¶å°ļ(-0.07) | åŁ¹è®Ń(-0.06) | Browser(-0.06) | (!((-0.06) | ä¸įåĪ©(-0.06) | ãģ®ãģ§ãģĻ(-0.06)
    ACCEPTED as axis_670  cumulative_var=0.7167

  [ 666]  axes=671  step_var=0.0012  binary_acc=0.999  gap=0.1148  max_dot=0.0038  (1.9s)
    TOP:  ÐĴÐ°Ð¼(0.06) | cdn(0.06) | og(0.06) | boolean(0.06) | Gins(0.06) | Histogram(0.06) | åĪĨåĮĸ(0.06) | tokenize(0.06)
    BOT:  .Num(-0.07) | Ru(-0.06) | exp(-0.06) | Thomas(-0.06) | .âĢ¢(-0.06) | +A(-0.06) | .t(-0.06) | .Parameter(-0.06)
    ACCEPTED as axis_671  cumulative_var=0.7170

  [ 667]  axes=672  step_var=0.0012  binary_acc=0.991  gap=0.1189  max_dot=0.0035  (1.9s)
    TOP:  .os(0.08) | XML(0.06) | _AS(0.06) | cloud(0.06) | _cmp(0.06) | =((0.06) | BSD(0.06) | mob(0.06)
    BOT:  .shortcuts(-0.07) | Authorization(-0.07) | Errors(-0.06) | roducing(-0.06) | _grade(-0.06) | Red(-0.06) | erta(-0.06) | MG(-0.06)
    ACCEPTED as axis_672  cumulative_var=0.7173

  [ 668]  axes=673  step_var=0.0012  binary_acc=0.995  gap=0.1177  max_dot=0.0059  (1.9s)
    TOP:  -plugin(0.07) | /key(0.06) | (base(0.06) | Ð¸Ñİ(0.06) | PG(0.06) | (upload(0.06) | (props(0.06) | ubbles(0.06)
    BOT:  Âº(-0.08) | Ãµ(-0.07) | U(-0.06) | Infinite(-0.06) | å¼º(-0.06) | .cl(-0.06) | .web(-0.06) | ys(-0.06)
    ACCEPTED as axis_673  cumulative_var=0.7177

  [ 669]  axes=674  step_var=0.0012  binary_acc=0.990  gap=0.1170  max_dot=0.0056  (1.8s)
    TOP:  Anchor(0.07) | [^(0.06) | notas(0.06) | illustrates(0.06) | Syndrome(0.06) | )}(0.06) | ='')(0.06) | ),$(0.06)
    BOT:  (z(-0.07) | akt(-0.07) | .url(-0.06) | _object(-0.06) | ul(-0.06) | STOP(-0.06) | Michel(-0.06) | ({Ċ(-0.06)
    ACCEPTED as axis_674  cumulative_var=0.7180

  [ 670]  axes=675  step_var=0.0012  binary_acc=1.000  gap=0.1156  max_dot=0.0019  (1.9s)
    TOP:  _params(0.07) | +F(0.07) | /view(0.06) | -max(0.06) | _driver(0.06) | åŃĲåħ¬åı¸(0.06) | aat(0.06) | åħ¶ä»ĸçļĦ(0.06)
    BOT:  æĥ³èµ·(-0.06) | ÑĢÐ°Ð±Ð¾ÑĤÐ°(-0.06) | ä»ĸä»¬(-0.06) | hil(-0.06) | .Convert(-0.06) | .parsers(-0.06) | ABLE(-0.06) | æĹ¶(-0.06)
    ACCEPTED as axis_675  cumulative_var=0.7183

  [ 671]  axes=676  step_var=0.0012  binary_acc=0.998  gap=0.1150  max_dot=0.0096  (1.8s)
    TOP:  %(0.07) | He(0.07) | åıĳåĬ¨(0.06) | ä»£è¡¨(0.06) | åİ¥(0.06) | _OT(0.06) | ç»Ħç»ĩ(0.06) | è¿·ä¿¡(0.06)
    BOT:  .window(-0.07) | count(-0.07) | TY(-0.06) | .version(-0.06) | ino(-0.06) | Ho(-0.06) | yp(-0.06) | uk(-0.06)
    ACCEPTED as axis_676  cumulative_var=0.7187

  [ 672]  axes=677  step_var=0.0012  binary_acc=0.996  gap=0.1174  max_dot=0.0042  (1.9s)
    TOP:  .sub(0.07) | /results(0.06) | Market(0.06) | -distance(0.06) | (U(0.06) | _B(0.06) | ificador(0.06) | /blog(0.06)
    BOT:  æĸ½å·¥(-0.07) | ect(-0.06) | ']):Ċ(-0.06) | æĬ¤èº«ç¬¦(-0.06) | XVI(-0.06) | ora(-0.06) | âĪĤ(-0.06) | '},Ċ(-0.06)
    ACCEPTED as axis_677  cumulative_var=0.7190

  [ 673]  axes=678  step_var=0.0012  binary_acc=0.998  gap=0.1171  max_dot=0.0038  (1.8s)
    TOP:  record(0.07) | Bo(0.06) | C(0.06) | FragmentManager(0.06) | Consider(0.06) | ä¹³(0.06) | sy(0.06) | Failed(0.06)
    BOT:  é¹ĥ(-0.07) | /downloads(-0.06) | éķĩ(-0.06) | ade(-0.06) | Ð°Ð½Ð°(-0.06) | åļı(-0.06) | mer(-0.06) | /us(-0.06)
    ACCEPTED as axis_678  cumulative_var=0.7193

  [ 674]  axes=679  step_var=0.0012  binary_acc=0.987  gap=0.1162  max_dot=0.0029  (1.8s)
    TOP:  UA(0.07) | .api(0.07) | AX(0.06) | _loop(0.06) | Results(0.06) | Fan(0.06) | Lima(0.06) | micro(0.06)
    BOT:  all(-0.07) | æĢ»(-0.06) | Ð¸Ñģ(-0.06) | ÑģÑĤÐ¾Ð¸ÑĤ(-0.06) | åĲĦé¡¹å·¥ä½ľ(-0.06) | åħ¨éĿ¢å»ºæĪĲ(-0.06) | éģĩ(-0.06) | day(-0.06)
    ACCEPTED as axis_679  cumulative_var=0.7197

  [ 675]  axes=680  step_var=0.0012  binary_acc=0.999  gap=0.1128  max_dot=0.0051  (1.9s)
    TOP:  /.(0.07) | ä½ł(0.06) | mx(0.06) | mime(0.06) | Pr(0.06) | primes(0.06) | nd(0.06) | Capture(0.06)
    BOT:  (var(-0.07) | Ø³ÙĦ(-0.07) | äºķ(-0.07) | åĪ¸(-0.06) | æ´ŀ(-0.06) | paginate(-0.06) | (exc(-0.06) | -com(-0.06)
    ACCEPTED as axis_680  cumulative_var=0.7200

  [ 676]  axes=681  step_var=0.0012  binary_acc=0.973  gap=0.1140  max_dot=0.0054  (1.8s)
    TOP:  åłµå¡ŀ(0.08) | æķĻå®¤(0.06) | websocket(0.06) | æĤłæĤł(0.06) | æĸ¯åĿ¦(0.06) | é«ĺæĸ°(0.06) | åĽŀçŃĶ(0.06) | atrix(0.06)
    BOT:  (c(-0.07) | (M(-0.07) | (i(-0.07) | [N(-0.07) | ç¼©(-0.07) | à¸Ń(-0.06) | (S(-0.06) | (l(-0.06)
    ACCEPTED as axis_681  cumulative_var=0.7203

  [ 677]  axes=682  step_var=0.0012  binary_acc=0.994  gap=0.1152  max_dot=0.0038  (1.9s)
    TOP:  structor(0.06) | collapse(0.06) | reference(0.06) | rect(0.06) | odes(0.06) | DESC(0.06) | æĢ»æĺ¯(0.06) | Ð½Ð¸Ðµ(0.06)
    BOT:  .l(-0.08) | -i(-0.07) | (P(-0.07) | -m(-0.07) | (E(-0.07) | ĉurl(-0.06) | :c(-0.06) | -a(-0.06)
    ACCEPTED as axis_682  cumulative_var=0.7207

  [ 678]  axes=683  step_var=0.0012  binary_acc=0.976  gap=0.1149  max_dot=0.0026  (1.9s)
    TOP:  Browser(0.07) | _search(0.07) | Dark(0.06) | _access(0.06) | Provider(0.06) | Space(0.06) | Ø³ÙĬ(0.06) | _array(0.06)
    BOT:  .New(-0.08) | âĢļ(-0.08) | .parser(-0.07) | (std(-0.07) | ('*',(-0.07) | .author(-0.06) | '$(-0.06) | âĳ(-0.06)
    ACCEPTED as axis_683  cumulative_var=0.7210

  [ 679]  axes=684  step_var=0.0012  binary_acc=0.971  gap=0.1159  max_dot=0.0015  (1.9s)
    TOP:  /b(0.07) | Ð¥(0.07) | .me(0.07) | /i(0.07) | Â¿(0.07) | Ð³Ð¾(0.06) | }/(0.06) | [Ċ(0.06)
    BOT:  éĤĳ(-0.07) | -manager(-0.07) | let(-0.07) | Leo(-0.07) | Configuration(-0.06) | upgrade(-0.06) | anol(-0.06) | pac(-0.06)
    ACCEPTED as axis_684  cumulative_var=0.7213

  [ 680]  axes=685  step_var=0.0012  binary_acc=0.989  gap=0.1141  max_dot=0.0040  (1.8s)
    TOP:  æĹ¢(0.07) | \M(0.07) | /legal(0.07) | ç«Ł(0.07) | ä¸¥(0.07) | [h(0.06) | çģĮ(0.06) | æ³Ĺ(0.06)
    BOT:  //(-0.06) | end(-0.06) | ina(-0.06) | at(-0.06) | %ĊĊ(-0.06) | _assignment(-0.06) | enced(-0.06) | elen(-0.06)
    ACCEPTED as axis_685  cumulative_var=0.7217

  [ 681]  axes=686  step_var=0.0012  binary_acc=0.987  gap=0.1140  max_dot=0.0013  (1.9s)
    TOP:  (description(0.08) | )ĊĊ(0.07) | ).ĊĊ(0.07) | ãĢĤĊĊ(0.07) | .ĊĊ(0.07) | .[(0.06) | æķĻæİĪ(0.06) | =Ċ(0.06)
    BOT:  ,V(-0.06) | Seq(-0.06) | Dis(-0.06) | VA(-0.06) | _B(-0.06) | New(-0.06) | `,(-0.06) | dis(-0.06)
    ACCEPTED as axis_686  cumulative_var=0.7220

  [ 682]  axes=687  step_var=0.0012  binary_acc=0.978  gap=0.1152  max_dot=0.0021  (1.8s)
    TOP:  _controller(0.06) | Sly(0.06) | ÑģÐºÐ°ÑĩÐ°ÑĤÑĮ(0.06) | DUCTION(0.06) | abbrev(0.06) | (View(0.06) | .Application(0.06) | exponent(0.06)
    BOT:  æĿ°(-0.07) | term(-0.07) | n(-0.07) | åģļä¸Ģä¸ª(-0.07) | Up(-0.07) | (a(-0.07) | }((-0.07) | -sp(-0.07)
    ACCEPTED as axis_687  cumulative_var=0.7223

  [ 683]  axes=688  step_var=0.0011  binary_acc=0.970  gap=0.1123  max_dot=0.0117  (1.9s)
    TOP:  .event(0.07) | èµĶåģ¿(0.07) | "_(0.06) | riority(0.06) | >.ĊĊ(0.06) | Projects(0.06) | [("(0.06) | output(0.06)
    BOT:  å°½ç®¡(-0.06) | ä¹ĭåľ°(-0.06) | èĬ±(-0.06) | )ï¼Į(-0.06) | Äĳá»ĭnh(-0.06) | °(-0.06) | _IMAGE(-0.06) | militar(-0.06)
    ACCEPTED as axis_688  cumulative_var=0.7226

  [ 684]  axes=689  step_var=0.0012  binary_acc=0.993  gap=0.1176  max_dot=0.0043  (1.9s)
    TOP:  ç½ĳç«Ļ(0.07) | k(0.06) | American(0.06) | è¿Ŀæ³ķè¡Įä¸º(0.06) | Ø®(0.06) | dont(0.06) | _coords(0.06) | ensemble(0.06)
    BOT:  _support(-0.07) | (String(-0.07) | pf(-0.06) | Ã©e(-0.06) | pherd(-0.06) | (min(-0.06) | .travel(-0.06) | /javascript(-0.06)
    ACCEPTED as axis_689  cumulative_var=0.7230

  [ 685]  axes=690  step_var=0.0011  binary_acc=0.993  gap=0.1129  max_dot=0.0028  (1.9s)
    TOP:  éŁ³(0.07) | ("@(0.06) | Error(0.06) | æķ°åŃĹåĮĸ(0.06) | æĢ¥(0.06) | å®Ĺ(0.06) | ãģĦãģ¦(0.06) | ë¹Ħ(0.06)
    BOT:  ]](-0.08) | ')).(-0.07) | ä¸Ģæł·(-0.06) | _ssl(-0.06) | Inspector(-0.06) | x(-0.06) | gather(-0.06) | _replace(-0.06)
    ACCEPTED as axis_690  cumulative_var=0.7233

  [ 686]  axes=691  step_var=0.0012  binary_acc=0.981  gap=0.1134  max_dot=0.0017  (1.9s)
    TOP:  .Feature(0.07) | Relationships(0.06) | '')Ċ(0.06) | `Ċ(0.06) | *)ĊĊ(0.06) | ***Ċ(0.06) | ###ĊĊ(0.06) | =ĊĊ(0.06)
    BOT:  .C(-0.08) | _C(-0.07) | -G(-0.07) | (q(-0.06) | (k(-0.06) | o(-0.06) | æĬĢæľ¯åĪĽæĸ°(-0.06) | è»Ĭ(-0.06)
    ACCEPTED as axis_691  cumulative_var=0.7236

  [ 687]  axes=692  step_var=0.0012  binary_acc=0.992  gap=0.1118  max_dot=0.0012  (1.8s)
    TOP:  æ¹¿(0.06) | åŃĶ(0.06) | èĻļ(0.06) | Licensed(0.06) | åħ¹(0.06) | (Method(0.06) | ç´ł(0.06) | èĭŁ(0.06)
    BOT:  bootstrap(-0.07) | _AND(-0.07) | .last(-0.06) | _provider(-0.06) | avras(-0.06) | poi(-0.06) | _USE(-0.06) | />ĊĊ(-0.06)
    ACCEPTED as axis_692  cumulative_var=0.7239

  [ 688]  axes=693  step_var=0.0012  binary_acc=0.971  gap=0.1139  max_dot=0.0047  (1.9s)
    TOP:  aller(0.07) | H(0.06) | [#(0.06) | âĢĿçļĦ(0.06) | .wait(0.06) | .readFile(0.06) | ':(0.06) | æŁ¥éĺħ(0.06)
    BOT:  Final(-0.06) | æĮĩå®ļ(-0.06) | ä¼ĺç§ĢçļĦ(-0.06) | Foot(-0.06) | holder(-0.06) | expand(-0.06) | verte(-0.06) | Experimental(-0.06)
    ACCEPTED as axis_693  cumulative_var=0.7242

  [ 689]  axes=694  step_var=0.0012  binary_acc=0.960  gap=0.1142  max_dot=0.0098  (1.8s)
    TOP:  /pdf(0.07) | ident(0.07) | config(0.06) | ailer(0.06) | (args(0.06) | cast(0.06) | waiting(0.06) | inherits(0.06)
    BOT:  ãĥ¼(-0.08) | _MESSAGE(-0.07) | Depth(-0.06) | ÐµÑģÐ»Ð¸(-0.06) | Î¼(-0.06) | çĶ»çĶ»(-0.06) | å±ķåĩº(-0.06) | æĤ¨çļĦ(-0.06)
    ACCEPTED as axis_694  cumulative_var=0.7246

  [ 690]  axes=695  step_var=0.0012  binary_acc=0.999  gap=0.1144  max_dot=0.0015  (1.9s)
    TOP:  ='',(0.07) | '{}'(0.07) | Ø§(0.07) | åħ¨åªĴä½ĵ(0.07) | ()),(0.07) | [['(0.06) | åī¥(0.06) | .di(0.06)
    BOT:  divisor(-0.06) | z(-0.06) | /vector(-0.06) | inbox(-0.06) | Material(-0.06) | Method(-0.05) | âĶĢâĶĢ(-0.05) | tÃ©(-0.05)
    ACCEPTED as axis_695  cumulative_var=0.7249

  [ 691]  axes=696  step_var=0.0012  binary_acc=0.997  gap=0.1131  max_dot=0.0149  (1.8s)
    TOP:  **Ċ(0.07) | .proto(0.07) | =$(0.06) | Mo(0.06) | .getLogger(0.06) | ãģ¨ãģĦãģĨãģ®ãģĮ(0.06) | _ĊĊ(0.06) | =model(0.06)
    BOT:  æĸ°(-0.07) | imize(-0.06) | æŁ¥(-0.06) | Story(-0.06) | .about(-0.06) | IN(-0.06) | _DIR(-0.06) | èº«è¾¹(-0.06)
    ACCEPTED as axis_696  cumulative_var=0.7252

  [ 692]  axes=697  step_var=0.0012  binary_acc=0.979  gap=0.1139  max_dot=0.0006  (1.8s)
    TOP:  that(0.08) | Who(0.07) | AT(0.07) | åĪĻ(0.06) | itations(0.06) | Ãħ(0.06) | .frame(0.06) | existe(0.06)
    BOT:  Southern(-0.07) | Disclaimer(-0.07) | Inv(-0.06) | Extra(-0.06) | Anim(-0.06) | .Interfaces(-0.06) | mr(-0.06) | .Work(-0.06)
    ACCEPTED as axis_697  cumulative_var=0.7255

  [ 693]  axes=698  step_var=0.0011  binary_acc=0.994  gap=0.1098  max_dot=0.0040  (1.9s)
    TOP:  .java(0.07) | Ð°Ðº(0.06) | ccak(0.06) | /schema(0.06) | specs(0.06) | (IS(0.06) | ÑĨ(0.06) | .re(0.06)
    BOT:  åĩĿ(-0.07) | Tick(-0.06) | æĮ¯åħ´(-0.06) | OURCES(-0.06) | æīĢåľ¨(-0.06) | -toggle(-0.06) | åĿĲåľ¨(-0.06) | æ³ķå¾ĭ(-0.06)
    ACCEPTED as axis_698  cumulative_var=0.7258

  [ 694]  axes=699  step_var=0.0012  binary_acc=0.985  gap=0.1117  max_dot=0.0076  (1.8s)
    TOP:  .make(0.09) | Disclaimer(0.08) | éĢĢ(0.06) | .render(0.06) | à¸Ńà¸ĩ(0.06) | tous(0.06) | .Generic(0.06) | Ent(0.06)
    BOT:  (args(-0.06) | com(-0.06) | Mik(-0.06) | phans(-0.06) | æ°´åĩĨ(-0.06) | ;i(-0.06) | urse(-0.06) | ags(-0.06)
    ACCEPTED as axis_699  cumulative_var=0.7262

  [ 695]  axes=700  step_var=0.0012  binary_acc=0.976  gap=0.1148  max_dot=0.0031  (1.8s)
    TOP:  åŃ¤(0.07) | ulsive(0.07) | UFC(0.07) | (text(0.06) | DOWNLOAD(0.06) | äºīåĪĽ(0.06) | âĢ(0.06) | "\(0.06)
    BOT:  _object(-0.07) | .exists(-0.07) | /TR(-0.07) | Type(-0.07) | ML(-0.06) | <body(-0.06) | Builder(-0.06) | -exp(-0.06)
    ACCEPTED as axis_700  cumulative_var=0.7265

  [ 696]  axes=701  step_var=0.0012  binary_acc=0.978  gap=0.1141  max_dot=0.0015  (1.8s)
    TOP:  Ł³(0.07) | çŁ£(0.07) | .q(0.06) | quÃ©(0.06) | _servers(0.06) | _JSON(0.06) | _INFO(0.06) | .gpu(0.06)
    BOT:  .cloud(-0.06) | è¯´çĿĢ(-0.06) | Choose(-0.06) | Ùħ(-0.06) | Ð´Ð¾Ð³Ð¾Ð²Ð¾ÑĢ(-0.06) | css(-0.05) | å¿ĺè®°(-0.05) | CSV(-0.05)
    ACCEPTED as axis_701  cumulative_var=0.7268

  [ 697]  axes=702  step_var=0.0011  binary_acc=0.961  gap=0.1126  max_dot=0.0030  (1.9s)
    TOP:  "name(0.07) | te(0.06) | -center(0.06) | Pooling(0.06) | -s(0.06) | Storage(0.06) | (counter(0.06) | icate(0.06)
    BOT:  RA(-0.07) | ":(-0.07) | _logger(-0.07) | é¢Ħéĺ²(-0.06) | Mary(-0.06) | Hu(-0.06) | æĮĸæİĺ(-0.06) | Cheng(-0.06)
    ACCEPTED as axis_702  cumulative_var=0.7271

  [ 698]  axes=703  step_var=0.0012  binary_acc=0.997  gap=0.1116  max_dot=0.0043  (1.9s)
    TOP:  },Ċ(0.07) | Scheme(0.07) | Hugo(0.07) | .Query(0.06) | webpack(0.06) | ç§ĳåŃ¦(0.06) | >čĊčĊ(0.06) | [],Ċ(0.06)
    BOT:  _ann(-0.07) | .par(-0.07) | Ãºng(-0.07) | .St(-0.07) | \Event(-0.07) | dataset(-0.06) | Entire(-0.06) | ("*(-0.06)
    ACCEPTED as axis_703  cumulative_var=0.7274

  [ 699]  axes=704  step_var=0.0012  binary_acc=0.981  gap=0.1135  max_dot=0.0026  (1.8s)
    TOP:  '|(0.06) | åĲĮå¿Ĺä»¬(0.06) | .pre(0.06) | .findall(0.06) | _iv(0.06) | ]],(0.06) | ICE(0.06) | )t(0.06)
    BOT:  _random(-0.07) | logo(-0.06) | ides(-0.06) | ï¼į(-0.06) | quiring(-0.06) | Write(-0.06) | ãĥķãĤ§(-0.06) | --Ċ(-0.06)
    ACCEPTED as axis_704  cumulative_var=0.7277

  [ 700]  axes=705  step_var=0.0012  binary_acc=0.960  gap=0.1091  max_dot=0.0002  (1.8s)
    TOP:  ongs(0.06) | olia(0.06) | (gt(0.06) | Visible(0.06) | neither(0.06) | .state(0.06) | fatto(0.06) | Ã¡n(0.05)
    BOT:  ä»»åĬ¡(-0.07) | .wall(-0.07) | .thumbnail(-0.07) | åĪĳ(-0.07) | Fer(-0.07) | listing(-0.06) | èľĤèľľ(-0.06) | .clear(-0.06)
    ACCEPTED as axis_705  cumulative_var=0.7281

  [ 701]  axes=706  step_var=0.0012  binary_acc=0.976  gap=0.1111  max_dot=0.0020  (2.0s)
    TOP:  Feature(0.07) | /oauth(0.07) | _LIST(0.06) | éĩĳèŀįå¸Ĥåľº(0.06) | /or(0.06) | .api(0.06) | rote(0.06) | .fast(0.06)
    BOT:  _SAVE(-0.07) | ç¬Ķ(-0.07) | ww(-0.06) | å¢¨(-0.06) | æµªè´¹(-0.06) | buttons(-0.06) | North(-0.06) | åĪĨåĪ«æĺ¯(-0.06)
    ACCEPTED as axis_706  cumulative_var=0.7284

  [ 702]  axes=707  step_var=0.0012  binary_acc=0.992  gap=0.1150  max_dot=0.0094  (1.8s)
    TOP:  æ®(0.07) | Ð¾Ð½(0.07) | çĨĬ(0.07) | ls(0.06) | lake(0.06) | age(0.06) | æİ£(0.06) | å¼Ĭ(0.06)
    BOT:  json(-0.07) | genders(-0.06) | /The(-0.06) | /shared(-0.06) | {Ċ(-0.06) | ãģĮ(-0.06) | .start(-0.06) | trim(-0.06)
    ACCEPTED as axis_707  cumulative_var=0.7287

  [ 703]  axes=708  step_var=0.0012  binary_acc=0.958  gap=0.1122  max_dot=0.0024  (1.8s)
    TOP:  Eng(0.06) | .sort(0.06) | ĉnew(0.06) | à¹ģà¸¥à¸°(0.06) | Comm(0.06) | à¹ģà¸¥à¸°(0.06) | .View(0.06) | Fields(0.06)
    BOT:  achine(-0.07) | åŃ¤(-0.07) | å²Ľ(-0.07) | æ´ĭ(-0.07) | can(-0.07) | ä½į(-0.07) | æ¸ļ(-0.06) | one(-0.06)
    ACCEPTED as axis_708  cumulative_var=0.7290

  [ 704]  axes=709  step_var=0.0012  binary_acc=0.978  gap=0.1163  max_dot=0.0041  (1.9s)
    TOP:  -N(0.07) | ference(0.07) | Ùĩ(0.07) | K(0.06) | /sp(0.06) | D(0.06) | anders(0.06) | ken(0.06)
    BOT:  äºĴ(-0.06) | Gravity(-0.06) | Qualified(-0.06) | éħ¥(-0.06) | Url(-0.06) | .Excel(-0.06) | $class(-0.06) | Sau(-0.06)
    ACCEPTED as axis_709  cumulative_var=0.7293

  [ 705]  axes=710  step_var=0.0012  binary_acc=0.987  gap=0.1142  max_dot=0.0008  (1.8s)
    TOP:  .lang(0.07) | (module(0.06) | ä¸»è¦ģçĶ¨äºİ(0.06) | OFFSET(0.06) | Script(0.06) | _commands(0.06) | .Build(0.06) | .flags(0.06)
    BOT:  .body(-0.07) | **(-0.07) | /s(-0.07) | UNT(-0.06) | Ñħ(-0.06) | frey(-0.06) | å¥ĸåĬ±(-0.06) | Terms(-0.06)
    ACCEPTED as axis_710  cumulative_var=0.7297

  [ 706]  axes=711  step_var=0.0011  binary_acc=0.985  gap=0.1107  max_dot=0.0112  (1.9s)
    TOP:  Me(0.06) | èºº(0.06) | ç§»(0.06) | åĩŃ(0.06) | æĢ¥è¯Ĭ(0.06) | ç¬¦åĲĪ(0.06) | é©¬(0.05) | ,start(0.05)
    BOT:  ect(-0.07) | -header(-0.07) | NBC(-0.06) | éĿ¢å¯¹éĿ¢(-0.06) | _html(-0.06) | $data(-0.06) | Ð¢(-0.06) | ''(-0.06)
    ACCEPTED as axis_711  cumulative_var=0.7300

  [ 707]  axes=712  step_var=0.0012  binary_acc=0.992  gap=0.1137  max_dot=0.0052  (1.9s)
    TOP:  kw(0.08) | """(0.07) | ç¬ĳ(0.07) | ï»¿//(0.07) | /uploads(0.07) | å½ĵå¹´(0.07) | æĺ¯ä¸ŃåĽ½(0.07) | æĳĩ(0.07)
    BOT:  _path(-0.07) | Global(-0.07) | Zhang(-0.06) | .lib(-0.06) | Sex(-0.06) | est(-0.06) | Furthermore(-0.06) | mill(-0.06)
    ACCEPTED as axis_712  cumulative_var=0.7303

  [ 708]  axes=713  step_var=0.0012  binary_acc=0.971  gap=0.1116  max_dot=0.0041  (1.8s)
    TOP:  ç©º(0.08) | .backends(0.06) | Handler(0.06) | Ã¡c(0.06) | _bit(0.06) | Are(0.06) | -control(0.06) | çĻ»è®°(0.06)
    BOT:  "((-0.06) | .(-0.06) | !(-0.06) | REL(-0.06) | .sample(-0.06) | (?:(-0.06) | .).ĊĊ(-0.06) | ìĤŃìłľ(-0.06)
    ACCEPTED as axis_713  cumulative_var=0.7306

  [ 709]  axes=714  step_var=0.0012  binary_acc=0.991  gap=0.1145  max_dot=0.0026  (1.9s)
    TOP:  Ð¿Ð¾Ð´(0.07) | ÐºÐ¾Ð¼Ð¿Ð°Ð½Ð¸Ð¹(0.07) | compiler(0.07) | à¸Ħ(0.06) | Ùĥ(0.06) | pr(0.06) | .exp(0.06) | æĸ¯(0.06)
    BOT:  å¾®(-0.07) | æĤ²(-0.07) | Clar(-0.06) | gyro(-0.06) | loi(-0.06) | Colors(-0.06) | Full(-0.06) | !ĊĊ(-0.06)
    ACCEPTED as axis_714  cumulative_var=0.7309

  [ 710]  axes=715  step_var=0.0012  binary_acc=0.969  gap=0.1122  max_dot=0.0027  (1.8s)
    TOP:  ios(0.07) | OST(0.06) | Pier(0.06) | -svg(0.06) | stone(0.06) | ubernetes(0.06) | èĦ¾(0.06) | ause(0.06)
    BOT:  .'Ċ(-0.08) | -btn(-0.07) | Http(-0.06) | -Y(-0.06) | ëł¬(-0.06) | Sly(-0.06) | .variable(-0.06) | Alan(-0.06)
    ACCEPTED as axis_715  cumulative_var=0.7312

  [ 711]  axes=716  step_var=0.0011  binary_acc=1.000  gap=0.1097  max_dot=0.0040  (1.8s)
    TOP:  Brands(0.06) | .protocol(0.06) | mit(0.06) | ]))Ċ(0.06) | Mode(0.06) | _picture(0.06) | /object(0.06) | Ð¿Ð¾Ð»Ñı(0.06)
    BOT:  {(-0.06) | Mapper(-0.06) | _cli(-0.06) | ä¿ĥ(-0.06) | åĩĨ(-0.06) | Seminar(-0.06) | 'M(-0.06) | ,H(-0.06)
    ACCEPTED as axis_716  cumulative_var=0.7315

  [ 712]  axes=717  step_var=0.0011  binary_acc=0.996  gap=0.1113  max_dot=0.0005  (1.8s)
    TOP:  /current(0.07) | '))Ċ(0.06) | Text(0.06) | )));(0.06) | TABLE(0.06) | AST(0.06) | .exit(0.06) | """(0.06)
    BOT:  æ³ķå¾ĭ(-0.07) | å¸ģ(-0.06) | gal(-0.06) | ä¸Ģä¾§(-0.06) | ãģ°(-0.06) | çĵ¦(-0.06) | éĩį(-0.06) | âĢĳ(-0.06)
    ACCEPTED as axis_717  cumulative_var=0.7318

  [ 713]  axes=718  step_var=0.0012  binary_acc=0.986  gap=0.1110  max_dot=0.0010  (1.8s)
    TOP:  '],(0.07) | _,(0.07) | -offset(0.07) | },(0.07) | ]](0.07) | İ(0.06) | uba(0.06) | ÐµÑĢÐ°(0.06)
    BOT:  Trigger(-0.08) | oted(-0.07) | æĺ¯è¦ģ(-0.06) | éĩĳèŀįæľºæŀĦ(-0.06) | hom(-0.06) | BP(-0.06) | Kon(-0.06) | è®¸(-0.06)
    ACCEPTED as axis_718  cumulative_var=0.7322

  [ 714]  axes=719  step_var=0.0012  binary_acc=0.990  gap=0.1158  max_dot=0.0040  (1.9s)
    TOP:  éĻĩ(0.06) | è·³(0.06) | ZF(0.06) | éŁµ(0.06) | Angle(0.06) | ROM(0.06) | agement(0.06) | .uniform(0.06)
    BOT:  æµ·æ·ĢåĮº(-0.07) | _per(-0.06) | èµĶåģ¿(-0.06) | \Client(-0.06) | .features(-0.06) | .register(-0.06) | .exception(-0.06) | .fromString(-0.06)
    ACCEPTED as axis_719  cumulative_var=0.7325

  [ 715]  axes=720  step_var=0.0011  binary_acc=0.999  gap=0.1098  max_dot=0.0047  (1.9s)
    TOP:  ler(0.07) | init(0.06) | ar(0.06) | å°±æĺ¯(0.06) | Un(0.06) | ned(0.06) | src(0.06) | h(0.06)
    BOT:  kwargs(-0.07) | .get(-0.06) | Breakfast(-0.06) | _split(-0.06) | çļĦæĶ¿æ²»(-0.06) | Ð»ÑİÐ±(-0.06) | .links(-0.06) | éĩįè¦ģè®²è¯Ŀ(-0.06)
    ACCEPTED as axis_720  cumulative_var=0.7328

  [ 716]  axes=721  step_var=0.0012  binary_acc=0.982  gap=0.1124  max_dot=0.0111  (1.8s)
    TOP:  Ð»Ð¾(0.08) | à¸ªà¸¡(0.07) | ":{"(0.07) | Ð¿ÑĢ(0.07) | ':Ċ(0.06) | :disable(0.06) | ëªħ(0.06) | ):ĊĊ(0.06)
    BOT:  _order(-0.07) | .mm(-0.07) | usto(-0.06) | Rebellion(-0.06) | ä¸įåĲ«(-0.06) | _data(-0.06) | essions(-0.06) | Techniques(-0.06)
    ACCEPTED as axis_721  cumulative_var=0.7331

  [ 717]  axes=722  step_var=0.0012  binary_acc=0.987  gap=0.1103  max_dot=0.0012  (1.9s)
    TOP:  .named(0.07) | URI(0.07) | çĶ¨åĵģ(0.06) | å¸(0.06) | Pattern(0.06) | _xml(0.06) | leases(0.06) | clock(0.06)
    BOT:  /at(-0.08) | æĪĺ(-0.07) | .protobuf(-0.06) | -for(-0.06) | åħ±äº§(-0.06) | never(-0.06) | cr(-0.06) | _target(-0.06)
    ACCEPTED as axis_722  cumulative_var=0.7334

  [ 718]  axes=723  step_var=0.0011  binary_acc=0.976  gap=0.1090  max_dot=0.0077  (2.0s)
    TOP:  ("\(0.07) | èī¯å¥½çļĦ(0.06) | linux(0.06) | ural(0.06) | ships(0.06) | .category(0.06) | national(0.06) | ress(0.06)
    BOT:  ï¼¡(-0.07) | chos(-0.06) | ä¸ī(-0.06) | Ð¥(-0.06) | Electric(-0.06) | èĴĭä»ĭçŁ³(-0.06) | j(-0.06) | ä¸ºä¸»(-0.06)
    ACCEPTED as axis_723  cumulative_var=0.7337

  [ 719]  axes=724  step_var=0.0012  binary_acc=0.993  gap=0.1148  max_dot=0.0015  (1.9s)
    TOP:  _font(0.06) | -pol(0.06) | %ĊĊ(0.06) | .Author(0.06) | ÑĢÐµ(0.06) | /library(0.06) | templateUrl(0.06) | ç§ĭ(0.06)
    BOT:  tp(-0.07) | ython(-0.06) | hi(-0.06) | size(-0.06) | ip(-0.06) | nous(-0.06) | è¯Ħè®º(-0.06) | æİ§åĪ¶(-0.06)
    ACCEPTED as axis_724  cumulative_var=0.7340

  [ 720]  axes=725  step_var=0.0012  binary_acc=0.983  gap=0.1127  max_dot=0.0042  (1.9s)
    TOP:  Decimal(0.07) | Boston(0.06) | é¡ºåºĶ(0.06) | opacity(0.06) | å¤ĸåªĴ(0.06) | æ¯ıæĹ¥(0.06) | çİ°ä»Ĭ(0.06) | Sym(0.06)
    BOT:  letters(-0.07) | _log(-0.06) | .shared(-0.06) | .check(-0.06) | _grid(-0.06) | .Core(-0.06) | -out(-0.06) | users(-0.06)
    ACCEPTED as axis_725  cumulative_var=0.7343

  [ 721]  axes=726  step_var=0.0011  binary_acc=0.990  gap=0.1107  max_dot=0.0042  (1.9s)
    TOP:  requests(0.06) | CMS(0.06) | adients(0.06) | bol(0.05) | =config(0.05) | -twitter(0.05) | olume(0.05) | Ħ(0.05)
    BOT:  Testing(-0.07) | aki(-0.06) | Launch(-0.06) | .Identity(-0.06) | AW(-0.06) | _TO(-0.06) | å¤´(-0.06) | BER(-0.06)
    ACCEPTED as axis_726  cumulative_var=0.7346

  [ 722]  axes=727  step_var=0.0012  binary_acc=0.984  gap=0.1095  max_dot=0.0022  (1.9s)
    TOP:  _pending(0.06) | ],Ċ(0.06) | });Ċ(0.06) | ;(0.06) | .Send(0.06) | .count(0.06) | ãĢĤãĢĤ(0.06) | =>(0.06)
    BOT:  ar(-0.06) | agh(-0.06) | ASC(-0.06) | II(-0.06) | ilion(-0.06) | Ñĭ(-0.06) | Amazon(-0.06) | Ñı(-0.06)
    ACCEPTED as axis_727  cumulative_var=0.7349

  [ 723]  axes=728  step_var=0.0012  binary_acc=0.998  gap=0.1107  max_dot=0.0029  (1.8s)
    TOP:  Decay(0.07) | shared(0.06) | ÐºÐ¾Ð²(0.06) | iej(0.06) | function(0.06) | BITS(0.06) | PRIMARY(0.06) | health(0.06)
    BOT:  not(-0.07) | .schema(-0.07) | .setObjectName(-0.06) | _chars(-0.06) | ._(-0.06) | Ĩ(-0.06) | .client(-0.06) | _n(-0.06)
    ACCEPTED as axis_728  cumulative_var=0.7352

  [ 724]  axes=729  step_var=0.0011  binary_acc=0.996  gap=0.1084  max_dot=0.0081  (1.8s)
    TOP:  ier(0.06) | à¤¸(0.06) | Ð¸Ðµ(0.06) | Colors(0.06) | éĹŃçİ¯(0.06) | amped(0.06) | Ju(0.06) | Dispatch(0.06)
    BOT:  ienne(-0.06) | records(-0.06) | .text(-0.06) | _with(-0.06) | ä¸ĢéĿ¢(-0.06) | hrs(-0.06) | _DEV(-0.06) | _order(-0.06)
    ACCEPTED as axis_729  cumulative_var=0.7355

  [ 725]  axes=730  step_var=0.0011  binary_acc=0.996  gap=0.1063  max_dot=0.0029  (1.8s)
    TOP:  */ĊĊ(0.07) | .weights(0.06) | down(0.06) | nÃły(0.06) | },(0.06) | -id(0.05) | '')Ċ(0.05) | |(0.05)
    BOT:  .Grid(-0.06) | V(-0.06) | åľºä¸Ĭ(-0.06) | -V(-0.06) | -D(-0.06) | Body(-0.06) | åŁİ(-0.06) | XF(-0.06)
    ACCEPTED as axis_730  cumulative_var=0.7358

  [ 726]  axes=731  step_var=0.0011  binary_acc=0.986  gap=0.1102  max_dot=0.0080  (1.8s)
    TOP:  ä½ľä¸º(0.07) | /node(0.06) | Ð½Ð¾Ð¹(0.06) | graph(0.06) | Registered(0.06) | fcc(0.06) | äººæ°ĳ(0.06) | }"Ċ(0.06)
    BOT:  _DIS(-0.07) | .style(-0.06) | .mac(-0.06) | -item(-0.06) | _o(-0.06) | _CONFIG(-0.06) | Ð§(-0.06) | UCH(-0.06)
    ACCEPTED as axis_731  cumulative_var=0.7361

  [ 727]  axes=732  step_var=0.0011  binary_acc=0.985  gap=0.1090  max_dot=0.0055  (2.0s)
    TOP:  ophys(0.06) | .rpc(0.06) | å½ĵå¤©(0.06) | #define(0.06) | ç¨¼(0.06) | çļĦå®ŀéĻħ(0.06) | èĢĥçĤ¹(0.06) | èµ¢(0.06)
    BOT:  ai(-0.07) | Ð°ÑĢ(-0.07) | IONS(-0.06) | .android(-0.06) | Ø±(-0.06) | .output(-0.06) | these(-0.06) | ITIES(-0.06)
    ACCEPTED as axis_732  cumulative_var=0.7364

  [ 728]  axes=733  step_var=0.0011  binary_acc=0.997  gap=0.1081  max_dot=0.0016  (1.8s)
    TOP:  èµĦæľ¬(0.07) | AIL(0.06) | ï¼ļ(0.06) | gid(0.06) | Text(0.06) | æĭŁ(0.06) | Ã©ration(0.06) | INTERNATIONAL(0.06)
    BOT:  ools(-0.07) | _line(-0.07) | Hex(-0.06) | -K(-0.06) | .dirname(-0.06) | .device(-0.06) | script(-0.06) | -n(-0.06)
    ACCEPTED as axis_733  cumulative_var=0.7367

  [ 729]  axes=734  step_var=0.0011  binary_acc=0.984  gap=0.1089  max_dot=0.0026  (1.8s)
    TOP:  .components(0.07) | åłĳ(0.07) | _l(0.06) | _col(0.06) | _adapter(0.06) | cs(0.06) | Tennis(0.06) | .Un(0.06)
    BOT:  """Ċ(-0.06) | ä½İä¿Ŀ(-0.06) | åı¯æĺ¯(-0.06) | Apollo(-0.06) | Ð°ÑĢÑĤ(-0.06) | åįģåĽĽ(-0.06) | Validation(-0.06) | urls(-0.06)
    ACCEPTED as axis_734  cumulative_var=0.7370

  [ 730]  axes=735  step_var=0.0012  binary_acc=0.997  gap=0.1104  max_dot=0.0019  (1.8s)
    TOP:  Palette(0.07) | _script(0.06) | .wrap(0.06) | ä¸įæŃ¢(0.06) | _users(0.06) | ($"(0.06) | enter(0.06) | oting(0.06)
    BOT:  zek(-0.07) | Stories(-0.06) | è°Ľ(-0.06) | );ĊĊ(-0.06) | .arm(-0.06) | Sai(-0.06) | æ±Ĺæ°´(-0.06) | token(-0.06)
    ACCEPTED as axis_735  cumulative_var=0.7373

  [ 731]  axes=736  step_var=0.0011  binary_acc=0.978  gap=0.1087  max_dot=0.0141  (1.9s)
    TOP:  è§Ħæ¨¡ä»¥ä¸Ĭ(0.06) | .Work(0.06) | async(0.06) | æĿİ(0.06) | _CREATE(0.06) | percent(0.06) | ctest(0.06) | ener(0.05)
    BOT:  uptools(-0.07) | _OBJECT(-0.06) | page(-0.06) | Ø¨(-0.06) | åĲĮ(-0.06) | torch(-0.06) | åıĮéĩį(-0.06) | Top(-0.06)
    ACCEPTED as axis_736  cumulative_var=0.7376

  [ 732]  axes=737  step_var=0.0011  binary_acc=0.980  gap=0.1113  max_dot=0.0004  (1.8s)
    TOP:  plugins(0.08) | "),(0.08) | .public(0.08) | ();(0.07) | ],(0.07) | å¯¹æŃ¤(0.07) | /cloud(0.07) | _author(0.06)
    BOT:  def(-0.06) | Ð»Ñĥ(-0.06) | Y(-0.06) | ÑĥÑĤÑĮ(-0.06) | YOU(-0.05) | Store(-0.05) | _INTERFACE(-0.05) | é¡»(-0.05)
    ACCEPTED as axis_737  cumulative_var=0.7379

  [ 733]  axes=738  step_var=0.0012  binary_acc=0.985  gap=0.1121  max_dot=0.0116  (1.8s)
    TOP:  .schema(0.06) | _port(0.06) | assets(0.06) | =\"(0.06) | Legacy(0.06) | {k(0.06) | ctx(0.06) | up(0.06)
    BOT:  æ´Ĺ(-0.06) | Camera(-0.06) | .energy(-0.06) | esa(-0.06) | ä»İæŃ¤(-0.06) | itical(-0.06) | çĹ°(-0.06) | ervlet(-0.05)
    ACCEPTED as axis_738  cumulative_var=0.7382

  [ 734]  axes=739  step_var=0.0012  binary_acc=0.984  gap=0.1087  max_dot=0.0038  (1.9s)
    TOP:  cd(0.08) | çĶ±(0.07) | at(0.07) | å¿Į(0.07) | _graph(0.07) | sp(0.07) | å½ĵ(0.07) | dc(0.06)
    BOT:  Print(-0.08) | Tensor(-0.07) | /audio(-0.06) | Favorites(-0.06) | Education(-0.06) | Assert(-0.06) | Tutorial(-0.06) | Am(-0.06)
    ACCEPTED as axis_739  cumulative_var=0.7385

  [ 735]  axes=740  step_var=0.0011  binary_acc=0.994  gap=0.1074  max_dot=0.0101  (1.8s)
    TOP:  .build(0.07) | accur(0.06) | .settings(0.06) | Retrieved(0.06) | çĽ¸çĪ±(0.06) | FW(0.06) | =c(0.06) | mar(0.05)
    BOT:  We(-0.07) | an(-0.07) | SERVER(-0.06) | |^(-0.06) | lh(-0.06) | Telecom(-0.06) | ìĿĢ(-0.06) | RYPT(-0.06)
    ACCEPTED as axis_740  cumulative_var=0.7388

  [ 736]  axes=741  step_var=0.0012  binary_acc=0.991  gap=0.1112  max_dot=0.0029  (1.8s)
    TOP:  )])Ċ(0.07) | Ĵ(0.07) | if(0.07) | _quality(0.06) | åĽ(0.06) | weight(0.06) | pton(0.06) | ods(0.06)
    BOT:  pol(-0.06) | åįĥéĩĮ(-0.06) | ãĥĸ(-0.06) | è®¯(-0.06) | Slide(-0.06) | slice(-0.06) | TY(-0.06) | remaining(-0.06)
    ACCEPTED as axis_741  cumulative_var=0.7391

  [ 737]  axes=742  step_var=0.0011  binary_acc=0.986  gap=0.1115  max_dot=0.0070  (1.8s)
    TOP:  .panel(0.06) | ãĤıãģĳ(0.06) | .cleaned(0.06) | .IS(0.06) | ">čĊ(0.06) | ì¦Ŀ(0.06) | Classic(0.05) | é¡º(0.05)
    BOT:  Y(-0.07) | an(-0.07) | OR(-0.06) | _unique(-0.06) | F(-0.06) | é«ĺæ¸©(-0.06) | cin(-0.06) | è¿ĺæľīå¾Īå¤ļ(-0.06)
    ACCEPTED as axis_742  cumulative_var=0.7394

  [ 738]  axes=743  step_var=0.0012  binary_acc=0.972  gap=0.1134  max_dot=0.0047  (1.8s)
    TOP:  åİŁ(0.06) | âĢľ(0.06) | "(0.06) | éĩĮ(0.06) | ï¿£(0.06) | {(0.06) | /W(0.06) | (token(0.06)
    BOT:  ar(-0.06) | u(-0.06) | Ret(-0.06) | operand(-0.06) | disp(-0.06) | ä¹īåĬ¡(-0.06) | _skill(-0.06) | symbol(-0.06)
    ACCEPTED as axis_743  cumulative_var=0.7397

  [ 739]  axes=744  step_var=0.0011  binary_acc=0.977  gap=0.1101  max_dot=0.0059  (1.9s)
    TOP:  dÃ©p(0.06) | å½ĵæĪĳä»¬(0.06) | `,(0.06) | Ð¾Ð»ÑĮ(0.06) | .errors(0.06) | g(0.06) | Lands(0.05) | normalize(0.05)
    BOT:  _corners(-0.06) | ä½¿(-0.06) | supply(-0.06) | åĮħæĭ¬(-0.06) | (dim(-0.06) | Kennedy(-0.06) | ian(-0.06) | éĻĲ(-0.06)
    ACCEPTED as axis_744  cumulative_var=0.7400

  [ 740]  axes=745  step_var=0.0011  binary_acc=0.973  gap=0.1115  max_dot=0.0036  (1.8s)
    TOP:  ACC(0.06) | fd(0.06) | periment(0.06) | (time(0.06) | iom(0.06) | ')ĊĊĊ(0.05) | æĻĸ(0.05) | ovies(0.05)
    BOT:  /bash(-0.07) | DataLoader(-0.06) | _n(-0.06) | éĽ·(-0.06) | é¡¿(-0.06) | Ð¿Ð¾Ð»Ð½Ð¾ÑģÑĤÑĮÑİ(-0.06) | ts(-0.06) | ä¸ĩäº¿(-0.06)
    ACCEPTED as axis_745  cumulative_var=0.7403

  [ 741]  axes=746  step_var=0.0011  binary_acc=0.999  gap=0.1086  max_dot=0.0032  (1.9s)
    TOP:  -Con(0.08) | atten(0.06) | _tok(0.06) | .Client(0.06) | å·¢(0.06) | astes(0.06) | äºĶå¹´(0.06) | Hotel(0.06)
    BOT:  dx(-0.06) | .cloud(-0.06) | æķ°(-0.06) | '];Ċ(-0.06) | Element(-0.06) | ÑĨ(-0.06) | éºĭ(-0.06) | Multip(-0.06)
    ACCEPTED as axis_746  cumulative_var=0.7406

  [ 742]  axes=747  step_var=0.0011  binary_acc=0.958  gap=0.1116  max_dot=0.0046  (1.8s)
    TOP:  -->Ċ(0.07) | Product(0.07) | />Ċ(0.06) | .Build(0.06) | ]);Ċ(0.06) | éĶļ(0.06) | ]]Ċ(0.06) | */ĊĊ(0.06)
    BOT:  .All(-0.07) | -winning(-0.06) | èĩª(-0.06) | Equality(-0.06) | èĩªæĦ¿(-0.06) | _DOMAIN(-0.06) | ViewSet(-0.06) | åºĶå½ĵ(-0.06)
    ACCEPTED as axis_747  cumulative_var=0.7409

  [ 743]  axes=748  step_var=0.0012  binary_acc=0.973  gap=0.1114  max_dot=0.0058  (1.9s)
    TOP:  ãĢĸ(0.07) | Ñī(0.07) | è¿ĩ(0.06) | .gui(0.06) | /download(0.06) | (Y(0.06) | Ð¾ÑģÑĤÐ¸(0.06) | ram(0.06)
    BOT:  pytest(-0.07) | Calculator(-0.06) | white(-0.06) | their(-0.06) | BUILD(-0.06) | æīĢ(-0.06) | analysis(-0.06) | sta(-0.06)
    ACCEPTED as axis_748  cumulative_var=0.7412

  [ 744]  axes=749  step_var=0.0011  binary_acc=0.968  gap=0.1077  max_dot=0.0022  (1.8s)
    TOP:  JIT(0.07) | allocated(0.06) | um(0.06) | \(0.06) | Ø£(0.06) | _path(0.06) | Investig(0.06) | [Ċ(0.06)
    BOT:  .manual(-0.06) | _replace(-0.06) | ities(-0.06) | aper(-0.06) | -provider(-0.06) | enabled(-0.05) | ito(-0.05) | Stage(-0.05)
    ACCEPTED as axis_749  cumulative_var=0.7415

  [ 745]  axes=750  step_var=0.0011  binary_acc=0.994  gap=0.1116  max_dot=0.0023  (1.8s)
    TOP:  iny(0.07) | Y(0.06) | y(0.06) | æĢª(0.06) | å°Ĩ(0.06) | r(0.06) | å®¹(0.06) | AM(0.06)
    BOT:  .Table(-0.06) | .Session(-0.06) | ];ĊĊ(-0.06) | .Stop(-0.06) | roman(-0.06) | _ITEM(-0.06) | _views(-0.06) | .Module(-0.06)
    ACCEPTED as axis_750  cumulative_var=0.7418

  [ 746]  axes=751  step_var=0.0011  binary_acc=0.992  gap=0.1107  max_dot=0.0045  (1.9s)
    TOP:  ä½ľåĩº(0.06) | æĶ¾æĿ¾(0.06) | ç»¼åĲĪæĢ§(0.06) | USER(0.06) | hÃ³a(0.06) | announced(0.06) | _columns(0.06) | stride(0.06)
    BOT:  CE(-0.07) | us(-0.06) | _NS(-0.06) | Attribute(-0.06) | NAME(-0.06) | Factory(-0.06) | Problem(-0.06) | DEF(-0.06)
    ACCEPTED as axis_751  cumulative_var=0.7421

  [ 747]  axes=752  step_var=0.0011  binary_acc=0.995  gap=0.1085  max_dot=0.0050  (2.0s)
    TOP:  Callbacks(0.06) | .spi(0.06) | Shell(0.06) | onde(0.06) | -xs(0.06) | Collections(0.06) | roach(0.06) | _ROW(0.06)
    BOT:  è¯ļ(-0.06) | æİªæĸ½(-0.06) | system(-0.06) | #[(-0.06) | Specify(-0.06) | "\(-0.06) | '][(-0.06) | ÐĶÐ»Ñı(-0.06)
    ACCEPTED as axis_752  cumulative_var=0.7424

  [ 748]  axes=753  step_var=0.0012  binary_acc=0.999  gap=0.1086  max_dot=0.0047  (1.9s)
    TOP:  /w(0.06) | /C(0.06) | mod(0.06) | through(0.06) | roid(0.06) | [m(0.06) | ÐµÑģÑĤÐ¸(0.06) | -drop(0.06)
    BOT:  è¾¹(-0.07) | -->čĊ(-0.06) | Completed(-0.06) | Good(-0.06) | UNIT(-0.06) | ators(-0.06) | elements(-0.06) | _)(-0.06)
    ACCEPTED as axis_753  cumulative_var=0.7427

  [ 749]  axes=754  step_var=0.0012  binary_acc=0.990  gap=0.1097  max_dot=0.0100  (1.9s)
    TOP:  Accessor(0.06) | _fill(0.06) | cipher(0.06) | Illegal(0.06) | æĶ¶åıĸ(0.06) | åľ¨å¤ĸ(0.06) | Ð¨(0.05) | Articles(0.05)
    BOT:  åºķä¸ĭ(-0.07) | .V(-0.07) | .int(-0.06) | /post(-0.06) | How(-0.06) | âĸ¶(-0.06) | /repos(-0.06) | io(-0.06)
    ACCEPTED as axis_754  cumulative_var=0.7430

  [ 750]  axes=755  step_var=0.0012  binary_acc=0.997  gap=0.1094  max_dot=0.0042  (1.8s)
    TOP:  ->(0.07) | else(0.07) | çļĦèģĮä¸ļ(0.07) | _info(0.06) | Ø©(0.06) | .Parse(0.06) | took(0.06) | -shared(0.06)
    BOT:  Big(-0.07) | ().Ċ(-0.07) | isors(-0.07) | J(-0.06) | Driver(-0.06) | èĪ¹(-0.06) | Ð½Ð°Ð»(-0.06) | Usage(-0.06)
    ACCEPTED as axis_755  cumulative_var=0.7433

  [ 751]  axes=756  step_var=0.0011  binary_acc=0.980  gap=0.1062  max_dot=0.0042  (1.8s)
    TOP:  ](0.07) | åĲĪçĲĨæĢ§(0.07) | peer(0.06) | .converter(0.06) | é¸Ń(0.06) | ?).(0.06) | ë¶Ģ(0.06) | }",(0.06)
    BOT:  _NAMESPACE(-0.06) | sv(-0.06) | yp(-0.06) | .DB(-0.06) | ure(-0.06) | RET(-0.06) | iname(-0.06) | ichten(-0.05)
    ACCEPTED as axis_756  cumulative_var=0.7436

  [ 752]  axes=757  step_var=0.0011  binary_acc=0.984  gap=0.1065  max_dot=0.0060  (1.8s)
    TOP:  _TYPES(0.07) | WARRANTY(0.06) | Defined(0.06) | .yahoo(0.06) | æķ¢(0.05) | å¿Ļ(0.05) | Qty(0.05) | .request(0.05)
    BOT:  _selected(-0.07) | Ã¨s(-0.07) | getic(-0.06) | èĮĥåĽ´(-0.06) | .compress(-0.06) | į(-0.06) | produce(-0.06) | _ASSERT(-0.06)
    ACCEPTED as axis_757  cumulative_var=0.7439

  [ 753]  axes=758  step_var=0.0012  binary_acc=0.983  gap=0.1090  max_dot=0.0063  (1.8s)
    TOP:  è¯ģåĪ¸æĬķèµĦ(0.08) | /wiki(0.07) | çī¹(0.07) | ÑĤÐ°Ðº(0.07) | ç´ł(0.06) | CT(0.06) | æĺĵ(0.06) | ĩ(0.06)
    BOT:  Judy(-0.06) | {(-0.06) | edom(-0.06) | AND(-0.06) | (){(-0.06) | .Bundle(-0.06) | \">(-0.06) | ÑģÑĢÐµÐ´Ð¸(-0.06)
    ACCEPTED as axis_758  cumulative_var=0.7442

  [ 754]  axes=759  step_var=0.0012  binary_acc=0.992  gap=0.1081  max_dot=0.0028  (1.8s)
    TOP:  iot(0.07) | ("(0.06) | ä¸¤æ¬¾(0.06) | _solution(0.06) | lopen(0.06) | $ĊĊ(0.06) | }$(0.06) | ãĥ¼ãĥł(0.06)
    BOT:  .br(-0.07) | .de(-0.06) | UD(-0.06) | (new(-0.06) | bil(-0.06) | spectral(-0.06) | Interrupt(-0.06) | _host(-0.06)
    ACCEPTED as axis_759  cumulative_var=0.7445

  [ 755]  axes=760  step_var=0.0012  binary_acc=0.998  gap=0.1098  max_dot=0.0050  (1.8s)
    TOP:  .media(0.07) | mark(0.07) | alle(0.06) | time(0.06) | ye(0.06) | sworth(0.06) | åºŃ(0.06) | Usage(0.06)
    BOT:  .Server(-0.07) | çĦ¡(-0.06) | è½¬è½½(-0.06) | ç¬Ķ(-0.06) | æĪĺèĥľ(-0.06) | æĸ¹(-0.06) | ç¨ĭåºıåĳĺ(-0.06) | æĶ¶çĽĬ(-0.06)
    ACCEPTED as axis_760  cumulative_var=0.7448

  [ 756]  axes=761  step_var=0.0012  binary_acc=0.988  gap=0.1084  max_dot=0.0048  (1.9s)
    TOP:  cket(0.07) | .scalar(0.06) | .tools(0.06) | Con(0.06) | ä¹Łæĺ¯(0.06) | Mel(0.06) | .Extensions(0.06) | eri(0.06)
    BOT:  'Ċ(-0.06) | })(-0.06) | ?](-0.06) | Data(-0.06) | kids(-0.06) | ;)(-0.06) | )(-0.06) | ]{(-0.06)
    ACCEPTED as axis_761  cumulative_var=0.7450

  [ 757]  axes=762  step_var=0.0011  binary_acc=0.947  gap=0.1072  max_dot=0.0017  (1.9s)
    TOP:  /login(0.06) | Somerset(0.06) | åĬŁæķĪ(0.06) | .fields(0.06) | ä½ı(0.06) | ÑĥÐ¿(0.06) | reamble(0.06) | æķĻå¸Ī(0.06)
    BOT:  .auto(-0.07) | /select(-0.07) | =target(-0.06) | _author(-0.06) | uning(-0.06) | _Item(-0.06) | _deg(-0.06) | å¸¸(-0.06)
    ACCEPTED as axis_762  cumulative_var=0.7453

  [ 758]  axes=763  step_var=0.0012  binary_acc=0.982  gap=0.1100  max_dot=0.0084  (1.8s)
    TOP:  docs(0.08) | Card(0.06) | cow(0.06) | ':[(0.06) | =_(0.06) | croll(0.06) | Address(0.06) | .subject(0.06)
    BOT:  åĽ¢(-0.06) | Advances(-0.06) | .message(-0.06) | projects(-0.06) | Pil(-0.06) | .ct(-0.05) | CHECK(-0.05) | æ±Łè¥¿çľģ(-0.05)
    ACCEPTED as axis_763  cumulative_var=0.7456

  [ 759]  axes=764  step_var=0.0012  binary_acc=0.987  gap=0.1061  max_dot=0.0028  (1.8s)
    TOP:  ynomial(0.07) | .ic(0.06) | U(0.06) | Provider(0.06) | ä¸ºåĩĨ(0.06) | _SCROLL(0.06) | Holt(0.06) | been(0.06)
    BOT:  -ch(-0.08) | (type(-0.07) | ives(-0.07) | .Cell(-0.07) | ...)Ċ(-0.06) | Hard(-0.06) | entr(-0.06) | _web(-0.06)
    ACCEPTED as axis_764  cumulative_var=0.7459

  [ 760]  axes=765  step_var=0.0012  binary_acc=0.966  gap=0.1112  max_dot=0.0060  (1.9s)
    TOP:  queryset(0.06) | /files(0.06) | _file(0.06) | .custom(0.06) | fence(0.06) | sk(0.05) | _select(0.05) | Golf(0.05)
    BOT:  """Ċ(-0.07) | on(-0.07) | ")ĊĊĊ(-0.06) | USE(-0.06) | oud(-0.06) | [(-0.06) | =img(-0.06) | ----Ċ(-0.05)
    ACCEPTED as axis_765  cumulative_var=0.7462

  [ 761]  axes=766  step_var=0.0011  binary_acc=0.998  gap=0.1078  max_dot=0.0055  (1.8s)
    TOP:  Ð¾ÑĢÐ¼(0.08) | >");Ċ(0.06) | Ð¾Ð¼(0.06) | Ð¸Ð½Ð³(0.06) | rier(0.06) | åľŃ(0.06) | SENT(0.06) | ).ĊĊ(0.05)
    BOT:  /master(-0.07) | /static(-0.06) | .com(-0.06) | åĴĮç¤¾ä¼ļ(-0.06) | buying(-0.06) | course(-0.06) | éļĶç¦»(-0.06) | Are(-0.06)
    ACCEPTED as axis_766  cumulative_var=0.7465

  [ 762]  axes=767  step_var=0.0012  binary_acc=0.989  gap=0.1083  max_dot=0.0114  (1.9s)
    TOP:  ÐŁ(0.06) | éĩį(0.06) | Unexpected(0.06) | P(0.06) | .Q(0.06) | .forEach(0.06) | Zh(0.06) | çĽĽ(0.06)
    BOT:  ellaneous(-0.07) | igrated(-0.06) | bf(-0.06) | }\(-0.06) | que(-0.06) | ience(-0.06) | isky(-0.06) | æĸ°é²ľ(-0.06)
    ACCEPTED as axis_767  cumulative_var=0.7468

  [ 763]  axes=768  step_var=0.0011  binary_acc=0.974  gap=0.1071  max_dot=0.0021  (1.9s)
    TOP:  ä¸īå¹´çº§(0.07) | -u(0.07) | ROSS(0.07) | .el(0.06) | çµ¦(0.06) | éĽ·(0.06) | _col(0.06) | CHECK(0.06)
    BOT:  ist(-0.07) | events(-0.07) | Frame(-0.06) | ä¸ĭä¸Ģ(-0.06) | month(-0.06) | æĢ§(-0.06) | Value(-0.06) | (real(-0.06)
    ACCEPTED as axis_768  cumulative_var=0.7471

  [ 764]  axes=769  step_var=0.0011  binary_acc=0.972  gap=0.1099  max_dot=0.0015  (1.8s)
    TOP:  -Ċ(0.08) | çĶµ(0.07) | -interface(0.06) | ç¥¨(0.06) | ssh(0.06) | ollection(0.06) | (cos(0.06) | (theta(0.06)
    BOT:  Signs(-0.06) | .Strings(-0.06) | Tags(-0.06) | minor(-0.06) | .pem(-0.05) | ().(-0.05) | /run(-0.05) | .plugin(-0.05)
    ACCEPTED as axis_769  cumulative_var=0.7474

  [ 765]  axes=770  step_var=0.0011  binary_acc=0.985  gap=0.1067  max_dot=0.0007  (1.9s)
    TOP:  _version(0.06) | Calendar(0.06) | /doc(0.06) | ÐµÑĤ(0.06) | piar(0.06) | ())ĊĊ(0.06) | åĬŁæķĪ(0.06) | Credentials(0.05)
    BOT:  Various(-0.06) | /public(-0.06) | Ð¾Ðº(-0.06) | {(-0.06) | Ne(-0.06) | ä»¥(-0.06) | EB(-0.06) | ç»ĺ(-0.06)
    ACCEPTED as axis_770  cumulative_var=0.7477

  [ 766]  axes=771  step_var=0.0012  binary_acc=0.996  gap=0.1088  max_dot=0.0116  (1.9s)
    TOP:  __((0.07) | .Protocol(0.06) | å¡ŀ(0.06) | erte(0.06) | Â¿(0.06) | xia(0.06) | ">Ċ(0.06) | In(0.06)
    BOT:  connected(-0.06) | .es(-0.06) | -art(-0.06) | Multiple(-0.06) | CheckBox(-0.06) | Rot(-0.06) | ä¹ĥ(-0.06) | _SHADOW(-0.06)
    ACCEPTED as axis_771  cumulative_var=0.7480

  [ 767]  axes=772  step_var=0.0011  binary_acc=0.994  gap=0.1080  max_dot=0.0059  (1.9s)
    TOP:  Impact(0.06) | /en(0.06) | æ¶Īéĺ²(0.06) | ä»¥ä¸º(0.06) | swagger(0.06) | åĲĪè®¡(0.06) | å°ģ(0.06) | /st(0.06)
    BOT:  /cc(-0.07) | positive(-0.06) | _buckets(-0.06) | .message(-0.06) | .annotation(-0.06) | wd(-0.06) | andra(-0.06) | .security(-0.06)
    ACCEPTED as axis_772  cumulative_var=0.7482

  [ 768]  axes=773  step_var=0.0011  binary_acc=0.989  gap=0.1094  max_dot=0.0044  (1.8s)
    TOP:  .Data(0.08) | ÐµÑĢÐ¸(0.07) | _AREA(0.07) | End(0.07) | PORT(0.06) | åĮºåŁŁ(0.06) | Ret(0.06) | æĺ¯ä¸Ģä¸ª(0.06)
    BOT:  é«ĺ(-0.07) | indent(-0.07) | æ¸¯(-0.06) | Know(-0.06) | .ib(-0.06) | tasks(-0.06) | åĮ»(-0.06) | isha(-0.06)
    ACCEPTED as axis_773  cumulative_var=0.7485

  [ 769]  axes=774  step_var=0.0011  binary_acc=0.992  gap=0.1086  max_dot=0.0046  (1.9s)
    TOP:  ÑĥÑİ(0.07) | 's(0.06) | .eval(0.06) | Dict(0.06) | ãģĳãĤĭ(0.06) | ä¸įåľ¨(0.06) | .stdin(0.06) | Draw(0.06)
    BOT:  vá»ģ(-0.07) | äºļ(-0.07) | """ĊĊ(-0.06) | (Qt(-0.06) | èģĶ(-0.06) | SS(-0.06) | ,[(-0.06) | _FLAGS(-0.06)
    ACCEPTED as axis_774  cumulative_var=0.7488

  [ 770]  axes=775  step_var=0.0011  binary_acc=0.999  gap=0.1089  max_dot=0.0024  (1.8s)
    TOP:  .Generic(0.06) | Logged(0.06) | .users(0.06) | .AF(0.06) | _lang(0.06) | .Render(0.05) | Cmd(0.05) | en(0.05)
    BOT:  åį¿(-0.07) | _ARCH(-0.07) | Modifier(-0.07) | æĸ¹æ³ķ(-0.06) | åŀĭ(-0.06) | Group(-0.06) | conf(-0.06) | .interfaces(-0.06)
    ACCEPTED as axis_775  cumulative_var=0.7491

  [ 771]  axes=776  step_var=0.0011  binary_acc=1.000  gap=0.1087  max_dot=0.0037  (1.8s)
    TOP:  .now(0.06) | .GO(0.06) | ..Ċ(0.06) | Dynamic(0.06) | æĬĢ(0.05) | cherche(0.05) | ãģķ(0.05) | æĶ¸(0.05)
    BOT:  .getInstance(-0.07) | ].(-0.07) | ule(-0.06) | Result(-0.06) | _df(-0.06) | åĤ¨åŃĺ(-0.06) | MS(-0.06) | """Ċ(-0.06)
    ACCEPTED as axis_776  cumulative_var=0.7494

  [ 772]  axes=777  step_var=0.0011  binary_acc=0.996  gap=0.1082  max_dot=0.0053  (1.8s)
    TOP:  _DATE(0.07) | .apps(0.06) | be(0.06) | _RUNTIME(0.06) | åĲĦé¡¹å·¥ä½ľ(0.06) | _pipeline(0.06) | å¦Ĥæľīä¾µæĿĥ(0.06) | zn(0.06)
    BOT:  çĤ®(-0.06) | è¯ģåĪ¸(-0.06) | å±ķå¼Ģ(-0.06) | /api(-0.06) | [self(-0.05) | atan(-0.05) | æ³¢(-0.05) | acht(-0.05)
    ACCEPTED as axis_777  cumulative_var=0.7497

  [ 773]  axes=778  step_var=0.0011  binary_acc=0.971  gap=0.1075  max_dot=0.0030  (1.8s)
    TOP:  çº¦å®ļ(0.07) | Finite(0.07) | .extensions(0.06) | occus(0.06) | /us(0.06) | ½æķ°(0.06) | loading(0.06) | Coverage(0.06)
    BOT:  è¯ķåį·(-0.06) | Ð±Ð¸(-0.06) | _credentials(-0.06) | Festival(-0.05) | ###Ċ(-0.05) | _,(-0.05) | ë¬¸(-0.05) | :=(-0.05)
    ACCEPTED as axis_778  cumulative_var=0.7500

  [ 774]  axes=779  step_var=0.0012  binary_acc=0.988  gap=0.1082  max_dot=0.0046  (1.8s)
    TOP:  metadata(0.07) | th(0.06) | city(0.06) | *a(0.06) | Com(0.06) | wer(0.06) | resize(0.06) | SN(0.05)
    BOT:  _replace(-0.07) | .parse(-0.07) | åĩ½æķ°(-0.06) | .im(-0.06) | åĢļ(-0.06) | Ã¹(-0.06) | .SET(-0.06) | _VIEW(-0.06)
    ACCEPTED as axis_779  cumulative_var=0.7503

  [ 775]  axes=780  step_var=0.0011  binary_acc=0.996  gap=0.1069  max_dot=0.0013  (1.9s)
    TOP:  æĬĹæ°§åĮĸ(0.07) | .state(0.06) | Tan(0.06) | _mb(0.06) | snippet(0.06) | PH(0.06) | W(0.06) | iss(0.06)
    BOT:  æľ¨(-0.06) | Kim(-0.06) | emia(-0.06) | describe(-0.05) | å·¥ä½ľä¸Ń(-0.05) | æľīæĿĥ(-0.05) | ÑĤÐµÐ¼(-0.05) | site(-0.05)
    ACCEPTED as axis_780  cumulative_var=0.7505

  [ 776]  axes=781  step_var=0.0012  binary_acc=0.986  gap=0.1083  max_dot=0.0112  (1.9s)
    TOP:  à¸µà¹Ī(0.06) | AMS(0.06) | .K(0.06) | ÐĿ(0.06) | ª(0.06) | ('^(0.06) | .Schema(0.05) | .Cloud(0.05)
    BOT:  åŁİéķĩ(-0.06) | UDP(-0.06) | dog(-0.06) | æĪ·ç±į(-0.06) | ipt(-0.06) | MAIL(-0.06) | .Layout(-0.06) | .selected(-0.06)
    ACCEPTED as axis_781  cumulative_var=0.7508

  [ 777]  axes=782  step_var=0.0012  binary_acc=0.984  gap=0.1056  max_dot=0.0073  (1.8s)
    TOP:  edit(0.07) | um(0.07) | Sponsor(0.06) | er(0.06) | !Ċ(0.06) | '})Ċ(0.06) | markets(0.06) | ef(0.06)
    BOT:  ãģĹãģ¦(-0.07) | ä¸Ń(-0.06) | äºĶåįģ(-0.06) | (@(-0.06) | .get(-0.06) | /tr(-0.06) | chi(-0.06) | .old(-0.06)
    ACCEPTED as axis_782  cumulative_var=0.7511

  [ 778]  axes=783  step_var=0.0012  binary_acc=0.981  gap=0.1074  max_dot=0.0069  (1.9s)
    TOP:  http(0.06) | make(0.06) | æ¯ı(0.06) | =A(0.06) | å¡«(0.06) | èĤº(0.06) | ATH(0.06) | .flex(0.06)
    BOT:  Callback(-0.06) | token(-0.06) | èº«ä¸º(-0.06) | ä»ŀ(-0.06) | tra(-0.06) | âĳ(-0.06) | biased(-0.06) | åĬŁæķĪ(-0.06)
    ACCEPTED as axis_783  cumulative_var=0.7514

  [ 779]  axes=784  step_var=0.0011  binary_acc=0.961  gap=0.1101  max_dot=0.0051  (1.9s)
    TOP:  æľŁ(0.06) | å¾Ĺ(0.06) | ä½ĵ(0.06) | _source(0.06) | /form(0.06) | æĥħ(0.06) | self(0.06) | å°ı(0.06)
    BOT:  Rav(-0.06) | transformations(-0.06) | uch(-0.06) | .merge(-0.06) | '^(-0.05) | ,n(-0.05) | cn(-0.05) | åľ¨ç½ĳä¸Ĭ(-0.05)
    ACCEPTED as axis_784  cumulative_var=0.7517

  [ 780]  axes=785  step_var=0.0012  binary_acc=0.964  gap=0.1092  max_dot=0.0062  (1.8s)
    TOP:  ç®Ģåįķ(0.07) | crypt(0.07) | Bundle(0.06) | developer(0.06) | ÑĢÐ¸(0.06) | .plugins(0.06) | _module(0.06) | æĦı(0.06)
    BOT:  verages(-0.07) | è¸ı(-0.07) | filename(-0.06) | _tag(-0.06) | _title(-0.06) | >>((-0.06) | )ĊĊ(-0.06) | _py(-0.06)
    ACCEPTED as axis_785  cumulative_var=0.7520

  [ 781]  axes=786  step_var=0.0011  binary_acc=0.987  gap=0.1059  max_dot=0.0030  (1.9s)
    TOP:  çłĶç©¶æīĢ(0.07) | _A(0.06) | .step(0.06) | è¦ģ(0.06) | je(0.06) | _ITEM(0.06) | ÙĤØ§ÙĦ(0.06) | åŁºéĩĳä¼ļ(0.06)
    BOT:  .name(-0.07) | ube(-0.06) | .session(-0.06) | Vertex(-0.06) | Show(-0.06) | tg(-0.05) | Privacy(-0.05) | arg(-0.05)
    ACCEPTED as axis_786  cumulative_var=0.7523

  [ 782]  axes=787  step_var=0.0012  binary_acc=0.966  gap=0.1074  max_dot=0.0031  (1.8s)
    TOP:  _Get(0.06) | Ð½Ð¸ÑĨÐ°(0.06) | _static(0.06) | _execute(0.06) | Suffix(0.06) | YELLOW(0.06) | normalization(0.05) | HÃ¶(0.05)
    BOT:  _modules(-0.06) | xing(-0.06) | _/(-0.06) | BT(-0.06) | bs(-0.06) | å¥³äºº(-0.06) | Position(-0.06) | åĢŁ(-0.06)
    ACCEPTED as axis_787  cumulative_var=0.7526

  [ 783]  axes=788  step_var=0.0011  binary_acc=0.983  gap=0.1082  max_dot=0.0046  (1.8s)
    TOP:  .Font(0.08) | _configure(0.07) | issen(0.06) | _vol(0.06) | |"(0.06) | .count(0.06) | "/>Ċ(0.06) | Node(0.06)
    BOT:  bootstrap(-0.06) | å¤©(-0.06) | Mind(-0.06) | v(-0.06) | default(-0.06) | J(-0.06) | é¦Ļ(-0.06) | .IN(-0.05)
    ACCEPTED as axis_788  cumulative_var=0.7528

  [ 784]  axes=789  step_var=0.0011  binary_acc=0.993  gap=0.1078  max_dot=0.0024  (2.0s)
    TOP:  ÐŁÐ¾(0.06) | å®ļ(0.06) | Vict(0.06) | ,âĢĻ(0.06) | added(0.06) | Convention(0.06) | akte(0.05) | breaker(0.05)
    BOT:  .d(-0.08) | E(-0.06) | N(-0.06) | ve(-0.06) | .Create(-0.06) | Ð¾(-0.06) | Â·(-0.06) | èĤĺ(-0.06)
    ACCEPTED as axis_789  cumulative_var=0.7531

  [ 785]  axes=790  step_var=0.0012  binary_acc=0.981  gap=0.1042  max_dot=0.0054  (1.9s)
    TOP:  /tags(0.07) | .entity(0.07) | .schedule(0.07) | .call(0.06) | /send(0.06) | /category(0.06) | Kyle(0.05) | -template(0.05)
    BOT:  _S(-0.07) | ãĢİ(-0.06) | (F(-0.06) | ].[(-0.06) | ($(-0.06) | .f(-0.06) | /assets(-0.06) | ĉT(-0.06)
    ACCEPTED as axis_790  cumulative_var=0.7534

  [ 786]  axes=791  step_var=0.0012  binary_acc=0.973  gap=0.1098  max_dot=0.0007  (1.9s)
    TOP:  èº«ä½ĵ(0.06) | çľģçº§(0.06) | åīįè¿Ľ(0.06) | ä¹ĭåĬ¿(0.06) | $.(0.06) | *.(0.06) | æĹłéĻĲ(0.06) | Ð´Ð°Ð½Ð½ÑĭÑħ(0.06)
    BOT:  Worksheets(-0.07) | æĺİ(-0.06) | transform(-0.06) | Ñĩ(-0.06) | çº¯æ´ģ(-0.06) | _sh(-0.05) | .concatenate(-0.05) | Mappings(-0.05)
    ACCEPTED as axis_791  cumulative_var=0.7537

  [ 787]  axes=792  step_var=0.0012  binary_acc=0.984  gap=0.1077  max_dot=0.0021  (1.8s)
    TOP:  _l(0.07) | ivo(0.06) | apeutics(0.06) | ,p(0.06) | Ø§ÛĮ(0.06) | .Qt(0.06) | .Bar(0.06) | )[(0.06)
    BOT:  à¸ģà¸£(-0.06) | Cors(-0.06) | ï»¿namespace(-0.06) | .b(-0.06) | åıĬ(-0.06) | Def(-0.06) | /extensions(-0.06) | ÐºÐ¾Ð¼(-0.05)
    ACCEPTED as axis_792  cumulative_var=0.7540

  [ 788]  axes=793  step_var=0.0012  binary_acc=0.996  gap=0.1076  max_dot=0.0041  (1.8s)
    TOP:  .com(0.07) | H(0.06) | Object(0.06) | -best(0.06) | Z(0.06) | åĽ¢(0.06) | .log(0.06) | K(0.06)
    BOT:  .Binding(-0.06) | Production(-0.06) | è®©(-0.06) | .tests(-0.06) | .Flow(-0.06) | has(-0.05) | indice(-0.05) | ÑĢÐ¾Ð²(-0.05)
    ACCEPTED as axis_793  cumulative_var=0.7543

  [ 789]  axes=794  step_var=0.0012  binary_acc=0.982  gap=0.1035  max_dot=0.0044  (1.8s)
    TOP:  ìķ¼(0.06) | Ð¾ÐºÐ°(0.06) | logout(0.06) | .mod(0.06) | _ref(0.06) | -direction(0.06) | Latest(0.05) | çļĦæł¹æľ¬(0.05)
    BOT:  }čĊčĊ(-0.07) | BE(-0.06) | .dk(-0.06) | erve(-0.06) | }ĊĊ(-0.06) | }ĊĊ(-0.06) | Bad(-0.06) | ãĢĤ",Ċ(-0.06)
    ACCEPTED as axis_794  cumulative_var=0.7545

  [ 790]  axes=795  step_var=0.0011  binary_acc=0.990  gap=0.1071  max_dot=0.0055  (1.8s)
    TOP:  /pdf(0.06) | Col(0.06) | éĽĨä½ĵ(0.06) | .users(0.06) | è¾ĥå¤ļ(0.06) | Configure(0.06) | /xml(0.06) | åĲ¯åıĳ(0.06)
    BOT:  :Ċ(-0.07) | -collapse(-0.07) | ï¼ļĊ(-0.06) | åĦĴ(-0.06) | ()Ċ(-0.06) | .cm(-0.06) | (Math(-0.06) | (),Ċ(-0.06)
    ACCEPTED as axis_795  cumulative_var=0.7548

  [ 791]  axes=796  step_var=0.0012  binary_acc=0.970  gap=0.1084  max_dot=0.0012  (1.9s)
    TOP:  åĵ¨(0.06) | layer(0.06) | _VERTEX(0.06) | ],Ċ(0.06) | Contributions(0.06) | SER(0.06) | /ĊĊ(0.05) | bytes(0.05)
    BOT:  å¯¹èĩªå·±(-0.07) | å¤§(-0.06) | d(-0.06) | Ïĥ(-0.06) | ell(-0.06) | _icons(-0.06) | Ø³(-0.05) | V(-0.05)
    ACCEPTED as axis_796  cumulative_var=0.7551

  [ 792]  axes=797  step_var=0.0012  binary_acc=0.983  gap=0.1056  max_dot=0.0011  (1.8s)
    TOP:  .detect(0.07) | /dashboard(0.06) | .Network(0.06) | .grid(0.06) | (binary(0.06) | .br(0.06) | umps(0.06) | ummy(0.06)
    BOT:  ANSW(-0.06) | |ĊĊ(-0.06) | ãĤ»ãĥĥãĥĪ(-0.06) | -column(-0.06) | v(-0.06) | çĬ¹(-0.06) | åĨįçİ°(-0.06) | SIZE(-0.06)
    ACCEPTED as axis_797  cumulative_var=0.7554

  [ 793]  axes=798  step_var=0.0011  binary_acc=0.995  gap=0.1059  max_dot=0.0018  (2.0s)
    TOP:  index(0.06) | Den(0.06) | ouncil(0.06) | âĢĿ)(0.05) | ilarity(0.05) | .retry(0.05) | ));Ċ(0.05) | ç»¼åĲĪç´łè´¨(0.05)
    BOT:  Ws(-0.06) | {@(-0.06) | ÑĥÑĤ(-0.06) | æĺ¯åľ¨(-0.06) | .dispatch(-0.06) | Scene(-0.06) | bff(-0.06) | Advice(-0.06)
    ACCEPTED as axis_798  cumulative_var=0.7557

  [ 794]  axes=799  step_var=0.0012  binary_acc=0.992  gap=0.1057  max_dot=0.0080  (1.8s)
    TOP:  ')(0.07) | é¡µ(0.06) | /channel(0.06) | _str(0.06) | _ob(0.06) | .Plugin(0.06) | _section(0.06) | .Bold(0.06)
    BOT:  ves(-0.07) | Un(-0.07) | ONE(-0.06) | NE(-0.06) | .launch(-0.06) | .q(-0.06) | le(-0.06) | on(-0.05)
    ACCEPTED as axis_799  cumulative_var=0.7560

  [ 795]  axes=800  step_var=0.0012  binary_acc=0.988  gap=0.1079  max_dot=0.0029  (1.8s)
    TOP:  Bot(0.07) | .Rest(0.07) | Orientation(0.06) | Voice(0.06) | ï¼ļĊ(0.06) | CHO(0.06) | ãģ§ãģĻãģĹ(0.06) | Ð°Ð²ÑĤÐ¾ÑĢ(0.06)
    BOT:  ÐµÐ¼(-0.06) | å½ĵ(-0.06) | pn(-0.06) | lin(-0.06) | ISTR(-0.06) | ä»»(-0.06) | automation(-0.06) | .graph(-0.06)
    ACCEPTED as axis_800  cumulative_var=0.7562

  [ 796]  axes=801  step_var=0.0012  binary_acc=0.988  gap=0.1066  max_dot=0.0044  (1.9s)
    TOP:  _m(0.07) | Validation(0.06) | job(0.06) | Multip(0.06) | å¸ĺ(0.06) | 't(0.06) | æ¶ĪéĻ¤(0.06) | èĢħ(0.06)
    BOT:  "+(-0.07) | è®¾(-0.06) | ref(-0.06) | .require(-0.06) | Null(-0.06) | !(-0.06) | );čĊčĊčĊ(-0.06) | .use(-0.06)
    ACCEPTED as axis_801  cumulative_var=0.7565

  [ 797]  axes=802  step_var=0.0012  binary_acc=0.996  gap=0.1067  max_dot=0.0088  (1.9s)
    TOP:  æľŁ(0.07) | learnt(0.06) | .item(0.06) | éĽ¨(0.06) | alchemy(0.06) | AMB(0.06) | Animal(0.06) | Broker(0.06)
    BOT:  er(-0.08) | (block(-0.07) | .step(-0.07) | .googleapis(-0.07) | Symphony(-0.07) | /sp(-0.06) | _j(-0.06) | .testing(-0.06)
    ACCEPTED as axis_802  cumulative_var=0.7568

  [ 798]  axes=803  step_var=0.0012  binary_acc=0.998  gap=0.1065  max_dot=0.0003  (1.8s)
    TOP:  åģļ(0.06) | Your(0.06) | _EXCEPTION(0.06) | bind(0.06) | æ³¨åĨĮèµĦæľ¬(0.05) | minimized(0.05) | à¸±(0.05) | rt(0.05)
    BOT:  èī²(-0.07) | .prepare(-0.07) | /post(-0.07) | .page(-0.06) | -w(-0.06) | -icon(-0.06) | bine(-0.06) | _build(-0.06)
    ACCEPTED as axis_803  cumulative_var=0.7571

  [ 799]  axes=804  step_var=0.0012  binary_acc=0.988  gap=0.1092  max_dot=0.0010  (1.8s)
    TOP:  Duplicate(0.06) | :(0.06) | readable(0.06) | Ð¸ÑĢÐ¾Ð²(0.05) | yd(0.05) | Equal(0.05) | _high(0.05) | Invoice(0.05)
    BOT:  [self(-0.07) | +',(-0.07) | éĢĶ(-0.07) | íķĺê²Į(-0.06) | plt(-0.06) | åĸª(-0.06) | ("{(-0.06) | ãģ¾ãģ¾(-0.06)
    ACCEPTED as axis_804  cumulative_var=0.7574

  [ 800]  axes=805  step_var=0.0012  binary_acc=0.988  gap=0.1065  max_dot=0.0040  (1.9s)
    TOP:  å°Ĩåľ¨(0.06) | å½ĵåľ°(0.06) | .Object(0.06) | akah(0.06) | Strip(0.06) | _speed(0.06) | session(0.05) | (x(0.05)
    BOT:  Ð°Ð½Ð¸(-0.06) | _status(-0.06) | (top(-0.06) | æŃĩ(-0.05) | _response(-0.05) | .Size(-0.05) | ä¹Ł(-0.05) | _probability(-0.05)
    ACCEPTED as axis_805  cumulative_var=0.7576

  [ 801]  axes=806  step_var=0.0011  binary_acc=0.975  gap=0.1026  max_dot=0.0012  (1.8s)
    TOP:  \/(0.06) | -primary(0.06) | iq(0.06) | Ð£(0.06) | ToInt(0.05) | æĥħ(0.05) | /sp(0.05) | íļĮ(0.05)
    BOT:  (Duration(-0.07) | åĽĬ(-0.06) | ().(-0.06) | åĲ¸çĥŁ(-0.06) | çĶµæ¢¯(-0.06) | äº§çī©(-0.06) | _template(-0.05) | go(-0.05)
    ACCEPTED as axis_806  cumulative_var=0.7579

  [ 802]  axes=807  step_var=0.0012  binary_acc=0.985  gap=0.1074  max_dot=0.0014  (1.9s)
    TOP:  Ð±Ð¾ÑĢ(0.06) | å±±ä¸ľçľģ(0.06) | <!--(0.06) | åĲĪèĤ¥(0.06) | =("(0.06) | Quarter(0.06) | Curr(0.06) | âĢĺ(0.06)
    BOT:  arios(-0.07) | Heap(-0.06) | -class(-0.06) | waters(-0.06) | .har(-0.06) | between(-0.06) | _sub(-0.06) | SOC(-0.06)
    ACCEPTED as axis_807  cumulative_var=0.7582

  [ 803]  axes=808  step_var=0.0012  binary_acc=0.956  gap=0.1058  max_dot=0.0034  (1.8s)
    TOP:  au(0.06) | imagen(0.06) | .auto(0.06) | .load(0.06) | Details(0.06) | .layer(0.06) | -select(0.06) | AMPLE(0.06)
    BOT:  .disconnect(-0.06) | ,N(-0.06) | ä¼ĺè¶Ĭ(-0.06) | åĴ¸(-0.06) | é¡ºåºĶ(-0.06) | url(-0.06) | å¤ªå¤§(-0.06) | bor(-0.06)
    ACCEPTED as axis_808  cumulative_var=0.7585

  [ 804]  axes=809  step_var=0.0012  binary_acc=0.980  gap=0.1062  max_dot=0.0097  (1.8s)
    TOP:  _py(0.07) | .tensor(0.06) | .Attribute(0.06) | _MEM(0.06) | Ùĩ(0.06) | .external(0.06) | AGR(0.06) | æ¶µ(0.06)
    BOT:  Mock(-0.09) | Failed(-0.06) | uninsured(-0.06) | ipelines(-0.06) | äº¤æ±ĩ(-0.06) | ï¼Ĳ(-0.05) | å·¡è§Ĩ(-0.05) | äºĭ(-0.05)
    ACCEPTED as axis_809  cumulative_var=0.7588

  [ 805]  axes=810  step_var=0.0012  binary_acc=0.994  gap=0.1066  max_dot=0.0007  (1.9s)
    TOP:  æĪı(0.07) | ç¥ĸ(0.06) | convolution(0.06) | .remove(0.06) | agination(0.06) | åĲĦç±»(0.06) | ÑĨÑĭ(0.06) | /Ċ(0.06)
    BOT:  warnings(-0.07) | .n(-0.06) | is(-0.06) | .crm(-0.06) | Ùĩ(-0.06) | å¿ĺ(-0.06) | t(-0.06) | .Content(-0.05)
    ACCEPTED as axis_810  cumulative_var=0.7591

  [ 806]  axes=811  step_var=0.0012  binary_acc=0.975  gap=0.1065  max_dot=0.0036  (1.8s)
    TOP:  åĳĬè¯ī(0.07) | .Property(0.06) | .DEBUG(0.06) | .strptime(0.06) | _container(0.06) | _images(0.06) | off(0.06) | _tools(0.06)
    BOT:  [string(-0.06) | se(-0.06) | ale(-0.06) | skin(-0.06) | .quit(-0.06) | æ·±æ·±(-0.06) | _H(-0.05) | ,and(-0.05)
    ACCEPTED as axis_811  cumulative_var=0.7593

  [ 807]  axes=812  step_var=0.0012  binary_acc=0.992  gap=0.1074  max_dot=0.0008  (1.9s)
    TOP:  (sys(0.06) | fixed(0.06) | auss(0.06) | SDK(0.06) | Sergei(0.06) | .typ(0.06) | clusions(0.05) | ulumi(0.05)
    BOT:  _prefix(-0.06) | _conditions(-0.06) | Registry(-0.06) | .net(-0.06) | Pattern(-0.06) | ãĥ»(-0.06) | From(-0.06) | "",Ċ(-0.06)
    ACCEPTED as axis_812  cumulative_var=0.7596

  [ 808]  axes=813  step_var=0.0012  binary_acc=0.966  gap=0.1070  max_dot=0.0025  (1.8s)
    TOP:  V(0.07) | å¤ªç©º(0.06) | ÐĿ(0.06) | /web(0.06) | Documentation(0.06) | matrices(0.06) | å®ŀéĻħæĥħåĨµ(0.06) | æĬ±(0.05)
    BOT:  .Parameters(-0.06) | \"(-0.06) | ä¹ĭéģĵ(-0.06) | gp(-0.06) | hidden(-0.06) | datetime(-0.06) | _test(-0.05) | /gr(-0.05)
    ACCEPTED as axis_813  cumulative_var=0.7599

  [ 809]  axes=814  step_var=0.0012  binary_acc=0.998  gap=0.1059  max_dot=0.0025  (1.8s)
    TOP:  (base(0.06) | .wait(0.06) | _IMAGES(0.06) | ,z(0.06) | gems(0.06) | .fileName(0.06) | ä¸¤æīĭ(0.06) | çī§(0.06)
    BOT:  Instagram(-0.06) | /news(-0.06) | Hay(-0.06) | emption(-0.06) | Extract(-0.06) | .rotate(-0.06) | vez(-0.06) | .general(-0.05)
    ACCEPTED as axis_814  cumulative_var=0.7602

  [ 810]  axes=815  step_var=0.0011  binary_acc=0.984  gap=0.1051  max_dot=0.0067  (1.8s)
    TOP:  æ²Ľ(0.07) | +i(0.06) | ÑĢÐ¾Ð²ÐµÑĢ(0.06) | .ui(0.06) | .security(0.06) | .Content(0.06) | /auth(0.06) | [t(0.06)
    BOT:  _layout(-0.06) | æĶ¹èī¯(-0.06) | Water(-0.06) | _NORMAL(-0.06) | ä¸įè®ºæĺ¯(-0.05) | à¹Ģà¸Ļ(-0.05) | .Information(-0.05) | .DATA(-0.05)
    ACCEPTED as axis_815  cumulative_var=0.7604

  [ 811]  axes=816  step_var=0.0012  binary_acc=0.989  gap=0.1049  max_dot=0.0012  (1.8s)
    TOP:  IN(0.06) | version(0.05) | è½¬è¿Ĳ(0.05) | _out(0.05) | çĽĺæ´»(0.05) | \Database(0.05) | Who(0.05) | FOR(0.05)
    BOT:  sem(-0.07) | tb(-0.06) | èĭ¦(-0.06) | .shape(-0.06) | EXT(-0.06) | tÃłu(-0.06) | Become(-0.06) | .tech(-0.06)
    ACCEPTED as axis_816  cumulative_var=0.7607

  [ 812]  axes=817  step_var=0.0012  binary_acc=0.963  gap=0.1054  max_dot=0.0025  (1.9s)
    TOP:  /console(0.06) | /c(0.06) | è½¬ç§»(0.06) | Service(0.06) | å¸¸åĬ¡(0.06) | "><(0.06) | Next(0.06) | -area(0.06)
    BOT:  FHA(-0.06) | B(-0.06) | .web(-0.05) | EP(-0.05) | torch(-0.05) | æ¶Īéĺ²å®īåħ¨(-0.05) | Matchers(-0.05) | æī¹è¯Ħ(-0.05)
    ACCEPTED as axis_817  cumulative_var=0.7610

  [ 813]  axes=818  step_var=0.0012  binary_acc=0.981  gap=0.1053  max_dot=0.0004  (1.8s)
    TOP:  ç²¾(0.06) | ÑĨÐ¾Ð²(0.06) | æ¬¾(0.06) | ,is(0.06) | Version(0.06) | Agency(0.06) | Compute(0.06) | Dictionary(0.06)
    BOT:  *d(-0.06) | _CHECK(-0.06) | Ð¿Ð¾Ð»ÑĥÑĩÐ¸ÑĤÑĮ(-0.06) | «(-0.06) | !)Ċ(-0.05) | !ĊĊ(-0.05) | æ¸ħåĩī(-0.05) | tr(-0.05)
    ACCEPTED as axis_818  cumulative_var=0.7613

  [ 814]  axes=819  step_var=0.0011  binary_acc=0.990  gap=0.1042  max_dot=0.0046  (1.8s)
    TOP:  ãĢĳ(0.07) | æĺ¾(0.06) | åģĩ(0.06) | çĪĨ(0.06) | xt(0.06) | trim(0.06) | viewport(0.06) | .try(0.05)
    BOT:  _attribute(-0.07) | .support(-0.06) | dell(-0.06) | æľªç»ı(-0.06) | uda(-0.06) | vf(-0.05) | _ITEM(-0.05) | HTTP(-0.05)
    ACCEPTED as axis_819  cumulative_var=0.7616

  [ 815]  axes=820  step_var=0.0012  binary_acc=0.960  gap=0.1036  max_dot=0.0140  (1.8s)
    TOP:  gons(0.06) | -det(0.06) | USE(0.06) | /image(0.06) | éģĩ(0.06) | Disease(0.06) | grams(0.06) | ball(0.06)
    BOT:  æ²Īéĺ³(-0.07) | _ph(-0.06) | .recipe(-0.06) | mess(-0.06) | mixins(-0.05) | æŀĹ(-0.05) | æŃ¦è£ħ(-0.05) | -r(-0.05)
    ACCEPTED as axis_820  cumulative_var=0.7618

  [ 816]  axes=821  step_var=0.0012  binary_acc=0.978  gap=0.1046  max_dot=0.0073  (1.9s)
    TOP:  å±Ģ(0.07) | å±ħ(0.06) | å¥ĭ(0.06) | projection(0.06) | ICH(0.06) | Platt(0.05) | å¸Ĥ(0.05) | import(0.05)
    BOT:  .Domain(-0.07) | ros(-0.06) | .disable(-0.06) | .policy(-0.06) | _absolute(-0.06) | music(-0.06) | ARY(-0.06) | .controller(-0.06)
    ACCEPTED as axis_821  cumulative_var=0.7621

  [ 817]  axes=822  step_var=0.0012  binary_acc=0.965  gap=0.1081  max_dot=0.0041  (1.9s)
    TOP:  iers(0.07) | Fixture(0.06) | Kernel(0.06) | istence(0.06) | CED(0.06) | Wrap(0.06) | AOL(0.05) | block(0.05)
    BOT:  },(-0.07) | ),(-0.06) | ',(-0.06) | âĢĿ(-0.06) | examinations(-0.06) | ä¸į(-0.06) | æŃ¢(-0.06) | Ø§ÙĦÙħ(-0.06)
    ACCEPTED as axis_822  cumulative_var=0.7624

  [ 818]  axes=823  step_var=0.0012  binary_acc=0.993  gap=0.1060  max_dot=0.0021  (1.8s)
    TOP:  ersistence(0.06) | æĽ¹æĵį(0.06) | _ON(0.06) | ç»ĻäºĨæĪĳ(0.05) | ÐŀÐ±(0.05) | transforms(0.05) | èªī(0.05) | JOB(0.05)
    BOT:  .ch(-0.06) | _block(-0.06) | _n(-0.06) | Implementation(-0.06) | Delete(-0.06) | .decoder(-0.06) | Repair(-0.06) | æĬĢæľ¯æ°´å¹³(-0.06)
    ACCEPTED as axis_823  cumulative_var=0.7627

  [ 819]  axes=824  step_var=0.0012  binary_acc=0.985  gap=0.1041  max_dot=0.0040  (1.9s)
    TOP:  thed(0.07) | Edwards(0.06) | pw(0.06) | rgb(0.06) | æĺ¯ä½ł(0.06) | emos(0.06) | èģĺçĶ¨(0.06) | .radius(0.06)
    BOT:  use(-0.08) | out(-0.06) | æľīåºı(-0.06) | /service(-0.06) | ("+(-0.06) | /sample(-0.06) | /node(-0.06) | ani(-0.06)
    ACCEPTED as axis_824  cumulative_var=0.7629

  [ 820]  axes=825  step_var=0.0012  binary_acc=0.978  gap=0.1071  max_dot=0.0126  (1.8s)
    TOP:  .Framework(0.08) | .prepare(0.07) | ä¸Ĭè¿°(0.07) | ç¢º(0.06) | .Format(0.06) | _else(0.06) | ationale(0.06) | Bur(0.06)
    BOT:  _create(-0.06) | .functional(-0.06) | .Enable(-0.06) | Meh(-0.06) | _FOR(-0.05) | igner(-0.05) | ipi(-0.05) | çľŁ(-0.05)
    ACCEPTED as axis_825  cumulative_var=0.7632

  [ 821]  axes=826  step_var=0.0012  binary_acc=0.985  gap=0.1069  max_dot=0.0010  (1.9s)
    TOP:  Product(0.06) | æĲľ(0.06) | (ans(0.06) | æĸŃ(0.06) | èįĲ(0.06) | ÐºÑĢÐ°Ñģ(0.05) | _FLOAT(0.05) | åĪĺ(0.05)
    BOT:  /default(-0.07) | .Counter(-0.07) | æĸĹäºī(-0.06) | \Type(-0.06) | updating(-0.06) | Registry(-0.06) | .E(-0.06) | Â±(-0.06)
    ACCEPTED as axis_826  cumulative_var=0.7635

  [ 822]  axes=827  step_var=0.0012  binary_acc=0.993  gap=0.1056  max_dot=0.0061  (1.8s)
    TOP:  æİ¨(0.07) | èĽĭ(0.06) | ublish(0.06) | reduce(0.06) | çĽ´æİ¥(0.06) | çĪĨ(0.06) | åħ±äº§(0.06) | _gain(0.06)
    BOT:  }},(-0.08) | ].(-0.07) | ÐĴ(-0.06) | /web(-0.06) | By(-0.06) | .ACTION(-0.06) | More(-0.06) | D(-0.06)
    ACCEPTED as axis_827  cumulative_var=0.7638

  [ 823]  axes=828  step_var=0.0011  binary_acc=0.970  gap=0.1049  max_dot=0.0058  (1.9s)
    TOP:  _al(0.06) | see(0.06) | PROCESS(0.06) | .head(0.06) | _ACTION(0.06) | datetime(0.06) | boolean(0.05) | Äĥng(0.05)
    BOT:  COMM(-0.06) | }{(-0.06) | .fc(-0.06) | .us(-0.06) | .att(-0.05) | )ĊĊ(-0.05) | ()])Ċ(-0.05) | -bordered(-0.05)
    ACCEPTED as axis_828  cumulative_var=0.7640

  [ 824]  axes=829  step_var=0.0012  binary_acc=0.979  gap=0.1052  max_dot=0.0111  (1.8s)
    TOP:  èĽĭçĻ½(0.06) | è¾Ĭ(0.06) | ç©¹(0.06) | PM(0.06) | èľĢ(0.05) | fÃ¶r(0.05) | />Ċ(0.05) | urniture(0.05)
    BOT:  ics(-0.08) | (string(-0.07) | _mask(-0.06) | .R(-0.06) | $c(-0.06) | <link(-0.06) | queue(-0.06) | (-0.06)
    ACCEPTED as axis_829  cumulative_var=0.7643

  [ 825]  axes=830  step_var=0.0012  binary_acc=0.974  gap=0.1035  max_dot=0.0113  (1.8s)
    TOP:  ]{(0.07) | )`Ċ(0.07) | {}".(0.06) | _extra(0.06) | -->Ċ(0.06) | average(0.06) | ä¸¾è¡Į(0.06) | ></(0.06)
    BOT:  åĮĹ(-0.07) | H(-0.06) | Dep(-0.06) | f(-0.06) | frm(-0.06) | .Data(-0.06) | Dah(-0.06) | -registration(-0.06)
    ACCEPTED as axis_830  cumulative_var=0.7646

  [ 826]  axes=831  step_var=0.0011  binary_acc=0.973  gap=0.1016  max_dot=0.0022  (1.9s)
    TOP:  _controller(0.07) | resource(0.06) | =============Ċ(0.06) | _transaction(0.06) | .args(0.06) | .S(0.06) | .True(0.06) | è¯ģä¹¦(0.06)
    BOT:  æķĻå¸Ī(-0.06) | è¡Įä¸ļ(-0.06) | _target(-0.06) | partie(-0.06) | socket(-0.06) | Mem(-0.06) | HK(-0.06) | ä¸Ģçº§(-0.06)
    ACCEPTED as axis_831  cumulative_var=0.7648

  [ 827]  axes=832  step_var=0.0012  binary_acc=0.993  gap=0.1041  max_dot=0.0013  (1.8s)
    TOP:  åıĸ(0.07) | é¡¿æĹ¶(0.06) | Ground(0.06) | Resolver(0.06) | pay(0.06) | æĪ¿(0.06) | æĺ¯ä¸ºäºĨ(0.06) | ke(0.06)
    BOT:  æł¸åĩĨ(-0.07) | .Http(-0.06) | çĦ¡(-0.06) | R(-0.05) | Mall(-0.05) | _DISPLAY(-0.05) | (args(-0.05) | ī(-0.05)
    ACCEPTED as axis_832  cumulative_var=0.7651

  [ 828]  axes=833  step_var=0.0012  binary_acc=0.996  gap=0.1034  max_dot=0.0098  (1.9s)
    TOP:  proxy(0.07) | æİĮ(0.06) | (s(0.06) | å®¶(0.06) | Ð»Ð¸Ð½(0.06) | PER(0.06) | éĢīè´Ń(0.06) | .Base(0.06)
    BOT:  Api(-0.06) | .status(-0.06) | iat(-0.05) | agrams(-0.05) | rnn(-0.05) | .visual(-0.05) | Se(-0.05) | _service(-0.05)
    ACCEPTED as axis_833  cumulative_var=0.7654

  [ 829]  axes=834  step_var=0.0012  binary_acc=0.974  gap=0.1044  max_dot=0.0061  (1.9s)
    TOP:  /sn(0.07) | à¹īà¸²à¸ĩ(0.06) | _GEN(0.06) | _an(0.06) | our(0.06) | Allowed(0.06) | Wand(0.06) | /K(0.06)
    BOT:  ice(-0.07) | rus(-0.06) | cs(-0.06) | Greeks(-0.06) | Semantic(-0.06) | ia(-0.06) | term(-0.05) | Mock(-0.05)
    ACCEPTED as axis_834  cumulative_var=0.7657

  [ 830]  axes=835  step_var=0.0012  binary_acc=0.979  gap=0.1048  max_dot=0.0053  (1.8s)
    TOP:  form(0.06) | (h(0.06) | å¦Ĥ(0.06) | disclosing(0.06) | .aut(0.06) | them(0.05) | æ®Ĭ(0.05) | /cal(0.05)
    BOT:  _bit(-0.07) | Dialog(-0.06) | .currency(-0.06) | è´µ(-0.06) | å©Ĩ(-0.06) | ="")Ċ(-0.06) | Login(-0.06) | ÑĢÑĥ(-0.06)
    ACCEPTED as axis_835  cumulative_var=0.7659

  [ 831]  axes=836  step_var=0.0012  binary_acc=0.948  gap=0.1064  max_dot=0.0055  (1.8s)
    TOP:  SPDX(0.07) | èĢħ(0.05) | Dark(0.05) | éĦĻ(0.05) | (['(0.05) | Merge(0.05) | Yu(0.05) | Db(0.05)
    BOT:  net(-0.07) | _version(-0.06) | bb(-0.06) | ÑĦÐ¸(-0.06) | "_"(-0.06) | her(-0.06) | MAN(-0.06) | _r(-0.06)
    ACCEPTED as axis_836  cumulative_var=0.7662

  [ 832]  axes=837  step_var=0.0012  binary_acc=0.979  gap=0.1044  max_dot=0.0006  (1.8s)
    TOP:  terraform(0.07) | 'l(0.06) | quette(0.06) | caster(0.06) | Session(0.06) | TV(0.06) | _extent(0.06) | load(0.06)
    BOT:  restore(-0.07) | ç¾½(-0.07) | .ReLU(-0.06) | æĹ¶ä»£(-0.06) | ÑģÑģÑĭÐ»(-0.06) | çł´åĿı(-0.06) | äºĶåĽĽ(-0.06) | _OBJECT(-0.06)
    ACCEPTED as axis_837  cumulative_var=0.7665

  [ 833]  axes=838  step_var=0.0012  binary_acc=0.993  gap=0.1043  max_dot=0.0029  (1.8s)
    TOP:  _bold(0.06) | (component(0.06) | .J(0.06) | -Fi(0.06) | \R(0.06) | ><?(0.05) | ÂŃ(0.05) | Ð¸Ð»Ð¸(0.05)
    BOT:  Params(-0.06) | idence(-0.06) | ATCH(-0.06) | isors(-0.06) | åŃ¦(-0.06) | Ø§Ø¡(-0.06) | åºĲ(-0.06) | åı£ç½©(-0.06)
    ACCEPTED as axis_838  cumulative_var=0.7668

  [ 834]  axes=839  step_var=0.0012  binary_acc=0.983  gap=0.1062  max_dot=0.0019  (1.8s)
    TOP:  format(0.07) | Section(0.06) | _close(0.06) | .seed(0.06) | Callback(0.06) | SAVE(0.06) | CMP(0.06) | há»įc(0.06)
    BOT:  .SUCCESS(-0.06) | Land(-0.06) | -M(-0.06) | Plot(-0.06) | /alert(-0.06) | .jsp(-0.06) | ÑĨ(-0.05) | lex(-0.05)
    ACCEPTED as axis_839  cumulative_var=0.7670

  [ 835]  axes=840  step_var=0.0012  binary_acc=0.997  gap=0.1059  max_dot=0.0033  (1.8s)
    TOP:  /p(0.07) | ("[(0.06) | _en(0.06) | _wrapper(0.06) | èµĦæĸĻ(0.06) | .Logger(0.06) | çº²(0.06) | [(((0.06)
    BOT:  (The(-0.07) | ullet(-0.07) | ors(-0.07) | UTES(-0.06) | Lead(-0.06) | Auto(-0.06) | u(-0.06) | .Plugin(-0.06)
    ACCEPTED as axis_840  cumulative_var=0.7673

  [ 836]  axes=841  step_var=0.0012  binary_acc=0.973  gap=0.1075  max_dot=0.0040  (1.9s)
    TOP:  .identity(0.06) | Cal(0.06) | -hook(0.06) | /">(0.06) | /chat(0.06) | ras(0.06) | /me(0.05) | éĴĹ(0.05)
    BOT:  Failure(-0.06) | Ã¤r(-0.06) | _ENV(-0.06) | _REQUEST(-0.06) | bcrypt(-0.05) | ri(-0.05) | ula(-0.05) | Creature(-0.05)
    ACCEPTED as axis_841  cumulative_var=0.7676

  [ 837]  axes=842  step_var=0.0012  binary_acc=0.973  gap=0.1015  max_dot=0.0113  (1.8s)
    TOP:  me(0.07) | are(0.06) | ('(0.06) | boto(0.06) | p(0.05) | r(0.05) | .weight(0.05) | æĢİèĥ½(0.05)
    BOT:  Declared(-0.07) | .Connection(-0.06) | .items(-0.06) | Resize(-0.06) | çļĦå½¢è±¡(-0.06) | .email(-0.06) | -The(-0.06) | _PROPERTY(-0.06)
    ACCEPTED as axis_842  cumulative_var=0.7679

  [ 838]  axes=843  step_var=0.0012  binary_acc=0.976  gap=0.1056  max_dot=0.0040  (1.8s)
    TOP:  /code(0.06) | ymes(0.06) | :S(0.06) | éĢĤå®ľ(0.06) | ']);Ċ(0.06) | pause(0.05) | æĪ¿ä»·(0.05) | èĬ±(0.05)
    BOT:  Enumerable(-0.07) | article(-0.06) | /ag(-0.06) | è®¸åı¯è¯ģ(-0.06) | Ð´Ð°Ð½(-0.06) | replace(-0.06) | Definition(-0.06) | Checking(-0.06)
    ACCEPTED as axis_843  cumulative_var=0.7681

  [ 839]  axes=844  step_var=0.0012  binary_acc=0.987  gap=0.1039  max_dot=0.0005  (1.8s)
    TOP:  -python(0.07) | cdot(0.07) | ising(0.06) | .documents(0.06) | .BOLD(0.06) | çªģåĩº(0.06) | _button(0.06) | _rate(0.06)
    BOT:  è´¤(-0.06) | PLICATION(-0.06) | SPDX(-0.06) | .DEFAULT(-0.06) | LC(-0.06) | form(-0.05) | èıľé¸Ł(-0.05) | åİ¿çº§(-0.05)
    ACCEPTED as axis_844  cumulative_var=0.7684

  [ 840]  axes=845  step_var=0.0012  binary_acc=1.000  gap=0.1042  max_dot=0.0107  (1.9s)
    TOP:  content(0.06) | run(0.06) | dirname(0.06) | ippet(0.06) | key(0.05) | kim(0.05) | block(0.05) | tabel(0.05)
    BOT:  _SHA(-0.07) | .body(-0.06) | _or(-0.06) | _for(-0.06) | character(-0.05) | Enum(-0.05) | ç¨İ(-0.05) | -md(-0.05)
    ACCEPTED as axis_845  cumulative_var=0.7687

  [ 841]  axes=846  step_var=0.0012  binary_acc=0.999  gap=0.1045  max_dot=0.0044  (1.8s)
    TOP:  >',(0.07) | /media(0.06) | ãģĵãģ¨(0.06) | options(0.06) | jango(0.06) | bol(0.06) | ("(0.06) | (C(0.06)
    BOT:  Ð°Ð¼(-0.07) | -meta(-0.06) | cales(-0.06) | ponential(-0.06) | åĽ¾çĶ»(-0.05) | çļĦå®īåħ¨(-0.05) | amage(-0.05) | TT(-0.05)
    ACCEPTED as axis_846  cumulative_var=0.7689

  [ 842]  axes=847  step_var=0.0012  binary_acc=1.000  gap=0.1036  max_dot=0.0104  (1.8s)
    TOP:  /platform(0.06) | ÑįÐºÑĢÐ°Ð½(0.06) | .channels(0.05) | .count(0.05) | ç§ĳåĪĽ(0.05) | ä¸įåĪ©(0.05) | èĥĥèĤł(0.05) | _sprite(0.05)
    BOT:  .testing(-0.08) | imate(-0.07) | Operations(-0.07) | /ex(-0.07) | Mem(-0.06) | iki(-0.06) | .Color(-0.06) | wy(-0.06)
    ACCEPTED as axis_847  cumulative_var=0.7692

  [ 843]  axes=848  step_var=0.0012  binary_acc=0.992  gap=0.1073  max_dot=0.0085  (1.8s)
    TOP:  icon(0.06) | _action(0.06) | ctic(0.06) | Ã³m(0.05) | .detect(0.05) | (ui(0.05) | .field(0.05) | Providers(0.05)
    BOT:  IL(-0.07) | (labels(-0.06) | _PAGE(-0.06) | Import(-0.06) | op(-0.06) | Leigh(-0.06) | .node(-0.06) | Usuario(-0.06)
    ACCEPTED as axis_848  cumulative_var=0.7695

  [ 844]  axes=849  step_var=0.0012  binary_acc=0.974  gap=0.1063  max_dot=0.0037  (2.0s)
    TOP:  )]Ċ(0.06) | /_(0.06) | }}Ċ(0.06) | Am(0.06) | '),Ċ(0.06) | _ID(0.06) | Query(0.06) | proto(0.06)
    BOT:  STM(-0.06) | rÃ©(-0.06) | 've(-0.06) | lock(-0.06) | (self(-0.06) | å¾Ĺ(-0.06) | åŁĶ(-0.06) | å¥½çļĦ(-0.06)
    ACCEPTED as axis_849  cumulative_var=0.7697

  [ 845]  axes=850  step_var=0.0012  binary_acc=0.991  gap=0.1025  max_dot=0.0006  (1.9s)
    TOP:  ÑĢÐ°Ð²(0.07) | .Bl(0.06) | (stop(0.06) | PORT(0.06) | seven(0.06) | okemon(0.05) | aud(0.05) | é¥²(0.05)
    BOT:  çļĦç»ĵæŀľ(-0.06) | on(-0.06) | .auth(-0.06) | ìłķë³´(-0.06) | /docs(-0.06) | Support(-0.06) | .cn(-0.06) | _U(-0.06)
    ACCEPTED as axis_850  cumulative_var=0.7700

  [ 846]  axes=851  step_var=0.0012  binary_acc=0.992  gap=0.1050  max_dot=0.0026  (1.9s)
    TOP:  )))Ċ(0.06) | ()))Ċ(0.06) | bases(0.06) | à¦¸(0.06) | )]Ċ(0.05) | Temporary(0.05) | opaque(0.05) | Sender(0.05)
    BOT:  Matcher(-0.06) | CODE(-0.06) | /block(-0.06) | Ðĳ(-0.06) | False(-0.06) | Fuk(-0.06) | Ð½ÑĭÐ¼Ð¸(-0.06) | izaÃ§Ã£o(-0.06)
    ACCEPTED as axis_851  cumulative_var=0.7703

  [ 847]  axes=852  step_var=0.0012  binary_acc=0.989  gap=0.1055  max_dot=0.0058  (1.8s)
    TOP:  Context(0.06) | ='(0.06) | .commands(0.06) | Ã«(0.06) | docker(0.06) | R(0.06) | å¯Ŀ(0.06) | ¢åįķ(0.05)
    BOT:  å·¨å¤§(-0.06) | ÑĭÐ²(-0.06) | credits(-0.06) | ĊĊ(-0.06) | "))ĊĊ(-0.06) | patible(-0.06) | /classes(-0.06) | angular(-0.06)
    ACCEPTED as axis_852  cumulative_var=0.7705

  [ 848]  axes=853  step_var=0.0012  binary_acc=0.978  gap=0.1015  max_dot=0.0028  (1.9s)
    TOP:  åħħ(0.06) | åģ¿(0.06) | -request(0.06) | /r(0.06) | OM(0.06) | ÑĢÐ°Ð²(0.06) | ]ĊĊ(0.06) | .Search(0.05)
    BOT:  (player(-0.06) | :"(-0.06) | (f(-0.06) | (tab(-0.06) | }s(-0.05) | istogram(-0.05) | .ms(-0.05) | ('__(-0.05)
    ACCEPTED as axis_853  cumulative_var=0.7708

  [ 849]  axes=854  step_var=0.0012  binary_acc=0.991  gap=0.1035  max_dot=0.0015  (1.8s)
    TOP:  å¡Į(0.07) | .save(0.06) | global(0.06) | &t(0.06) | _decoder(0.06) | çĵ¦(0.06) | _CONTENT(0.06) | oft(0.06)
    BOT:  éĢĤçĶ¨(-0.06) | _disable(-0.06) | gs(-0.06) | Json(-0.06) | æĸŃ(-0.06) | åĢĴåľ¨(-0.06) | On(-0.05) | åħ³å¿ĥ(-0.05)
    ACCEPTED as axis_854  cumulative_var=0.7711

  [ 850]  axes=855  step_var=0.0012  binary_acc=0.951  gap=0.1032  max_dot=0.0035  (1.8s)
    TOP:  oute(0.06) | ested(0.06) | çĶ¨èĩªå·±çļĦ(0.06) | ÑģÑĤÐ²Ñĥ(0.06) | /tools(0.06) | ning(0.06) | /G(0.06) | .domain(0.05)
    BOT:  ,R(-0.07) | [^(-0.06) | (),(-0.06) | radians(-0.06) | \R(-0.06) | åįķä½į(-0.06) | åħ´(-0.06) | replace(-0.06)
    ACCEPTED as axis_855  cumulative_var=0.7713

  [ 851]  axes=856  step_var=0.0012  binary_acc=0.960  gap=0.1028  max_dot=0.0092  (1.9s)
    TOP:  _iterations(0.06) | _documents(0.05) | AB(0.05) | spir(0.05) | relude(0.05) | edy(0.05) | ean(0.05) | Sol(0.05)
    BOT:  t(-0.07) | Ð²Ð°ÑĪ(-0.06) | av(-0.06) | _one(-0.06) | Ð½Ð¸Ñı(-0.06) | ()čĊ(-0.05) | =================================================(-0.05) | .session(-0.05)
    ACCEPTED as axis_856  cumulative_var=0.7716

  [ 852]  axes=857  step_var=0.0012  binary_acc=0.974  gap=0.1055  max_dot=0.0040  (1.8s)
    TOP:  è¿°(0.06) | é¡¿(0.06) | å¹ħ(0.06) | package(0.06) | ç»ĵæŀĦæĢ§(0.05) | /cc(0.05) | Ðł(0.05) | Ð»Ð¸ÑĨ(0.05)
    BOT:  anne(-0.07) | urement(-0.07) | ];(-0.06) | ib(-0.06) | urer(-0.06) | ik(-0.06) | į(-0.06) | å¤§è±¡(-0.06)
    ACCEPTED as axis_857  cumulative_var=0.7719

  [ 853]  axes=858  step_var=0.0012  binary_acc=0.995  gap=0.1050  max_dot=0.0107  (1.9s)
    TOP:  Span(0.06) | .Compute(0.06) | Die(0.06) | layered(0.06) | _groups(0.06) | Ð´(0.06) | Plugin(0.06) | Cake(0.06)
    BOT:  mont(-0.07) | on(-0.07) | romium(-0.06) | ang(-0.06) | -json(-0.06) | compte(-0.06) | Feld(-0.06) | acy(-0.06)
    ACCEPTED as axis_858  cumulative_var=0.7722

  [ 854]  axes=859  step_var=0.0012  binary_acc=0.994  gap=0.1026  max_dot=0.0021  (1.8s)
    TOP:  ä¼łç»Ł(0.06) | Ð±(0.06) | èĭ±åĽ½(0.05) | .stem(0.05) | uiltin(0.05) | qs(0.05) | Affero(0.05) | æķĪçĽĬ(0.05)
    BOT:  ante(-0.07) | çķ¥(-0.07) | _joint(-0.06) | ä¾µå®³(-0.06) | iert(-0.06) | .h(-0.06) | ÑĨÐ¸(-0.06) | âĢĻll(-0.06)
    ACCEPTED as axis_859  cumulative_var=0.7724

  [ 855]  axes=860  step_var=0.0012  binary_acc=0.951  gap=0.1034  max_dot=0.0054  (1.9s)
    TOP:  .al(0.06) | Query(0.06) | .reason(0.05) | _USER(0.05) | ()],(0.05) | ä¸Ń(0.05) | anches(0.05) | .wav(0.05)
    BOT:  å¬ī(-0.07) | roi(-0.06) | obe(-0.06) | å¯Ĩçłģ(-0.06) | .math(-0.06) | axios(-0.06) | ÅĽcie(-0.06) | åħįè´£å£°æĺİ(-0.06)
    ACCEPTED as axis_860  cumulative_var=0.7727

  [ 856]  axes=861  step_var=0.0011  binary_acc=0.981  gap=0.1032  max_dot=0.0040  (1.8s)
    TOP:  Comics(0.07) | (day(0.06) | ogenesis(0.06) | ÐµÐ½Ð¾(0.06) | ÑįÑĤÐ¸Ð¼(0.06) | learning(0.05) | Ã¤lt(0.05) | ocols(0.05)
    BOT:  æĢĿç»´(-0.06) | çļĦç¾İå¥½(-0.06) | ĩ(-0.06) | Ø£(-0.06) | .step(-0.06) | éº»(-0.06) | ×ĺ(-0.06) | ÑĢ(-0.06)
    ACCEPTED as axis_861  cumulative_var=0.7729

  [ 857]  axes=862  step_var=0.0012  binary_acc=0.968  gap=0.1054  max_dot=0.0066  (1.8s)
    TOP:  è¿Ŀ(0.06) | é¢Ħå®ļ(0.06) | Contract(0.06) | è®¨(0.06) | rical(0.06) | Ð½ÑĥÑĤÑĮ(0.06) | display(0.06) | (original(0.06)
    BOT:  ac(-0.07) | æĸ¹å¼ı(-0.06) | ("/",(-0.06) | CC(-0.06) | ph(-0.06) | _author(-0.06) | Barber(-0.06) | Ec(-0.06)
    ACCEPTED as axis_862  cumulative_var=0.7732

  [ 858]  axes=863  step_var=0.0012  binary_acc=0.964  gap=0.1033  max_dot=0.0077  (1.8s)
    TOP:  _found(0.06) | [l(0.06) | _OP(0.06) | _al(0.06) | /A(0.06) | Quant(0.06) | èĭ¥æľī(0.06) | (encoding(0.06)
    BOT:  ople(-0.06) | enheim(-0.06) | beth(-0.06) | False(-0.06) | Ãľ(-0.06) | supported(-0.06) | Distribution(-0.06) | åįıè°ĥåıĳå±ķ(-0.05)
    ACCEPTED as axis_863  cumulative_var=0.7735

  [ 859]  axes=864  step_var=0.0012  binary_acc=0.990  gap=0.1032  max_dot=0.0021  (1.8s)
    TOP:  _estimators(0.07) | .j(0.06) | çļĦåľŁåľ°(0.06) | filename(0.06) | :S(0.06) | åĿļåĨ³(0.06) | (comp(0.06) | æ¹ĺ(0.06)
    BOT:  ation(-0.06) | .Property(-0.06) | Ø±(-0.06) | ä»ĸçļĦ(-0.06) | ãģŁ(-0.06) | us(-0.06) | arel(-0.06) | ãģ¦(-0.06)
    ACCEPTED as axis_864  cumulative_var=0.7737

  [ 860]  axes=865  step_var=0.0012  binary_acc=0.980  gap=0.1008  max_dot=0.0045  (1.8s)
    TOP:  straints(0.07) | ÑĤÐµ(0.06) | ets(0.06) | .entity(0.06) | .Identity(0.06) | ialis(0.06) | mer(0.05) | wnd(0.05)
    BOT:  çļĦäº§åĵģ(-0.06) | dots(-0.06) | sj(-0.06) | points(-0.05) | battery(-0.05) | _asset(-0.05) | net(-0.05) | ipp(-0.05)
    ACCEPTED as axis_865  cumulative_var=0.7740

  [ 861]  axes=866  step_var=0.0012  binary_acc=0.975  gap=0.1042  max_dot=0.0089  (1.9s)
    TOP:  Heading(0.06) | .Views(0.06) | =s(0.06) | .email(0.06) | æķĳ(0.06) | uch(0.06) | j(0.05) | ENTER(0.05)
    BOT:  P(-0.08) | by(-0.07) | (rate(-0.06) | Event(-0.06) | src(-0.06) | or(-0.06) | .lp(-0.06) | .Value(-0.06)
    ACCEPTED as axis_866  cumulative_var=0.7743

  [ 862]  axes=867  step_var=0.0012  binary_acc=0.990  gap=0.1048  max_dot=0.0018  (1.8s)
    TOP:  _begin(0.07) | specified(0.06) | (content(0.05) | Models(0.05) | Script(0.05) | Cy(0.05) | values(0.05) | it(0.05)
    BOT:  Ð½ÐµÐµ(-0.07) | rupt(-0.06) | Input(-0.06) | [](-0.06) | Comment(-0.06) | Henri(-0.06) | ]]Ċ(-0.05) | oration(-0.05)
    ACCEPTED as axis_867  cumulative_var=0.7745

  [ 863]  axes=868  step_var=0.0012  binary_acc=0.998  gap=0.1045  max_dot=0.0009  (1.8s)
    TOP:  instance(0.07) | .when(0.06) | exposure(0.06) | /code(0.06) | -web(0.06) | .Parameter(0.06) | Transactional(0.06) | åĮħæĭ¬(0.06)
    BOT:  Welcome(-0.07) | From(-0.06) | ty(-0.06) | be(-0.06) | gen(-0.06) | _utc(-0.06) | Long(-0.06) | å®ľ(-0.06)
    ACCEPTED as axis_868  cumulative_var=0.7748

  [ 864]  axes=869  step_var=0.0012  binary_acc=0.990  gap=0.1033  max_dot=0.0035  (1.8s)
    TOP:  keywords(0.06) | MED(0.06) | éħĴç²¾(0.06) | cling(0.06) | -an(0.06) | suggest(0.06) | _top(0.06) | Arbitrary(0.05)
    BOT:  æľĢåıĹ(-0.06) | ..ĊĊ(-0.06) | "((-0.06) | GED(-0.06) | Take(-0.06) | ==Ċ(-0.06) | åħ¬çĶ¨(-0.05) | ')čĊ(-0.05)
    ACCEPTED as axis_869  cumulative_var=0.7751

  [ 865]  axes=870  step_var=0.0012  binary_acc=1.000  gap=0.1012  max_dot=0.0007  (1.9s)
    TOP:  ("(0.06) | fas(0.06) | .uri(0.06) | obe(0.06) | _with(0.06) | yr(0.06) | Ø±Ø¨(0.06) | ble(0.05)
    BOT:  '@(-0.06) | èİ«(-0.06) | At(-0.06) | .open(-0.06) | .query(-0.06) | written(-0.05) | .P(-0.05) | åı¯èĥ½åıĳçĶŁ(-0.05)
    ACCEPTED as axis_870  cumulative_var=0.7753

  [ 866]  axes=871  step_var=0.0012  binary_acc=0.998  gap=0.1032  max_dot=0.0037  (1.8s)
    TOP:  -L(0.06) | uez(0.06) | range(0.06) | ives(0.06) | uire(0.06) | å¤©æ¶¯(0.06) | side(0.05) | oten(0.05)
    BOT:  _password(-0.06) | ='(-0.06) | .Change(-0.06) | (action(-0.06) | ä¸»è¦ģåĨħå®¹(-0.06) | -table(-0.06) | babel(-0.06) | .forRoot(-0.06)
    ACCEPTED as axis_871  cumulative_var=0.7756

  [ 867]  axes=872  step_var=0.0012  binary_acc=0.989  gap=0.1028  max_dot=0.0102  (1.9s)
    TOP:  GED(0.07) | æĺ¯æĪĳåĽ½(0.07) | .Simple(0.06) | æºĲäºİ(0.06) | from(0.06) | é¢Ĭ(0.06) | _SWITCH(0.06) | åĨ¬(0.06)
    BOT:  #(-0.06) | ).ĊĊ(-0.06) | Â¦(-0.06) | cc(-0.06) | Apply(-0.06) | i(-0.06) | -time(-0.06) | ution(-0.06)
    ACCEPTED as axis_872  cumulative_var=0.7759

  [ 868]  axes=873  step_var=0.0012  binary_acc=0.983  gap=0.1021  max_dot=0.0030  (1.8s)
    TOP:  .auth(0.06) | dtype(0.06) | _t(0.06) | èº¯(0.06) | ãģĴ(0.06) | Offer(0.06) | -gray(0.06) | ("<(0.06)
    BOT:  OWER(-0.06) | .feature(-0.06) | éļ¾é¢ĺ(-0.06) | ival(-0.06) | \Table(-0.06) | -api(-0.05) | åĽºä½ĵ(-0.05) | ains(-0.05)
    ACCEPTED as axis_873  cumulative_var=0.7761

  [ 869]  axes=874  step_var=0.0012  binary_acc=0.975  gap=0.1057  max_dot=0.0036  (1.9s)
    TOP:  éĹ¨(0.06) | å¡ĺ(0.05) | åı¯çĶ¨äºİ(0.05) | elapsed(0.05) | None(0.05) | aes(0.05) | CC(0.05) | Upper(0.05)
    BOT:  Char(-0.06) | _file(-0.06) | .pol(-0.06) | LICENSE(-0.06) | kt(-0.06) | .Basic(-0.05) | Rehabilitation(-0.05) | çĲĨæĥ³ä¿¡å¿µ(-0.05)
    ACCEPTED as axis_874  cumulative_var=0.7764

  [ 870]  axes=875  step_var=0.0012  binary_acc=0.961  gap=0.1035  max_dot=0.0027  (1.9s)
    TOP:  _os(0.07) | .files(0.06) | (len(0.06) | (max(0.06) | åĸľ(0.06) | éĹ¯(0.06) | Loading(0.06) | :h(0.06)
    BOT:  /store(-0.06) | ä¸īåĽĽ(-0.06) | _extract(-0.06) | .parallel(-0.06) | mond(-0.06) | à¸ķ(-0.06) | ç»Ł(-0.06) | urther(-0.06)
    ACCEPTED as axis_875  cumulative_var=0.7767

  [ 871]  axes=876  step_var=0.0012  binary_acc=0.984  gap=0.1012  max_dot=0.0053  (1.8s)
    TOP:  If(0.06) | inverse(0.06) | Ãī(0.06) | ave(0.06) | Exporter(0.05) | orp(0.05) | yn(0.05) | (),(0.05)
    BOT:  son(-0.06) | ~/(-0.06) | å®¢è½¦(-0.06) | æ°ĳéĹ´(-0.06) | _SOURCE(-0.06) | music(-0.06) | time(-0.06) | ."(-0.06)
    ACCEPTED as axis_876  cumulative_var=0.7769

  [ 872]  axes=877  step_var=0.0012  binary_acc=0.985  gap=0.1011  max_dot=0.0110  (1.9s)
    TOP:  pc(0.06) | èĪį(0.05) | .pr(0.05) | resolve(0.05) | .da(0.05) | thon(0.05) | .File(0.05) | _categorical(0.05)
    BOT:  Util(-0.07) | åºĶ(-0.06) | è¦ĸ(-0.06) | {((-0.06) | (labels(-0.06) | íĮĲ(-0.05) | .generic(-0.05) | Learning(-0.05)
    ACCEPTED as axis_877  cumulative_var=0.7772

  [ 873]  axes=878  step_var=0.0012  binary_acc=0.972  gap=0.1036  max_dot=0.0016  (1.9s)
    TOP:  .trace(0.07) | Ð¿ÐµÑģ(0.06) | _git(0.06) | ãģĵãģĨ(0.06) | Bits(0.06) | åľĸ(0.06) | ìĦ¸(0.06) | Resize(0.06)
    BOT:  err(-0.07) | .com(-0.06) | Writer(-0.06) | çļĦæĸ¹æ³ķ(-0.06) | (mode(-0.06) | åĨ¶éĩĳ(-0.06) | (logging(-0.06) | /memory(-0.06)
    ACCEPTED as axis_878  cumulative_var=0.7775

  [ 874]  axes=879  step_var=0.0012  binary_acc=0.970  gap=0.1011  max_dot=0.0034  (2.0s)
    TOP:  Global(0.07) | Object(0.06) | .)ĊĊ(0.06) | consistent(0.06) | REGISTER(0.06) | Rectangle(0.06) | Community(0.05) | .support(0.05)
    BOT:  =B(-0.07) | é£İéĻ©ç®¡çĲĨ(-0.07) | (conv(-0.06) | _extension(-0.06) | ied(-0.06) | _http(-0.06) | /add(-0.06) | .transfer(-0.06)
    ACCEPTED as axis_879  cumulative_var=0.7777

  [ 875]  axes=880  step_var=0.0012  binary_acc=0.975  gap=0.1040  max_dot=0.0045  (1.8s)
    TOP:  ~~~~(0.06) | [](0.06) | Conway(0.06) | Worker(0.06) | å¾ģ(0.05) | twitter(0.05) | Compiler(0.05) | ç¬ĳ(0.05)
    BOT:  .Local(-0.07) | Math(-0.06) | Ð½ÑĭÑħ(-0.06) | çĶ±(-0.06) | Johnson(-0.06) | _iterations(-0.06) | IENCE(-0.06) | ÑİÑīÐ¸Ñħ(-0.06)
    ACCEPTED as axis_880  cumulative_var=0.7780

  [ 876]  axes=881  step_var=0.0012  binary_acc=0.983  gap=0.1026  max_dot=0.0085  (1.8s)
    TOP:  .comp(0.08) | _OUT(0.07) | _NOT(0.07) | '))(0.06) | inated(0.06) | Footer(0.06) | embers(0.06) | _max(0.06)
    BOT:  tÆ°(-0.06) | ł(-0.06) | âĢŀ(-0.06) | cia(-0.06) | æŁĲ(-0.06) | e(-0.05) | =["(-0.05) | åŃ¦çĶŁ(-0.05)
    ACCEPTED as axis_881  cumulative_var=0.7782

  [ 877]  axes=882  step_var=0.0012  binary_acc=0.977  gap=0.1010  max_dot=0.0024  (1.8s)
    TOP:  DIST(0.06) | Al(0.06) | Ø¨(0.06) | äºĴ(0.06) | =default(0.06) | .endswith(0.06) | Ten(0.06) | ç»ĵ(0.06)
    BOT:  ADMIN(-0.06) | oo(-0.06) | .extract(-0.06) | /top(-0.06) | ()Ċ(-0.06) | .com(-0.06) | èĤ¢(-0.05) | ISE(-0.05)
    ACCEPTED as axis_882  cumulative_var=0.7785

  [ 878]  axes=883  step_var=0.0012  binary_acc=0.983  gap=0.1019  max_dot=0.0084  (1.8s)
    TOP:  (tt(0.06) | Greenwich(0.06) | :'(0.06) | Ð¾ÑĢ(0.05) | Appointment(0.05) | ffer(0.05) | .device(0.05) | /rs(0.05)
    BOT:  _q(-0.06) | Calculate(-0.06) | sl(-0.06) | -properties(-0.06) | .dd(-0.06) | -Cal(-0.06) | æĲŀå¥½(-0.06) | å²Ĺä½į(-0.06)
    ACCEPTED as axis_883  cumulative_var=0.7788

  [ 879]  axes=884  step_var=0.0012  binary_acc=0.982  gap=0.1023  max_dot=0.0097  (1.9s)
    TOP:  Emitter(0.07) | .take(0.07) | .edu(0.06) | .setState(0.05) | Ring(0.05) | Ð±ÑĭÐ»(0.05) | .Json(0.05) | Sultan(0.05)
    BOT:  ase(-0.07) | ou(-0.06) | /gr(-0.06) | ä¸Ń(-0.06) | -align(-0.06) | nie(-0.06) | _FINAL(-0.06) | chemas(-0.06)
    ACCEPTED as axis_884  cumulative_var=0.7790

  [ 880]  axes=885  step_var=0.0012  binary_acc=0.993  gap=0.1029  max_dot=0.0046  (1.8s)
    TOP:  /qu(0.06) | Ð¼(0.06) | .monitor(0.06) | -ste(0.06) | ;$(0.06) | .action(0.05) | que(0.05) | Ð½ÐµÑĤ(0.05)
    BOT:  Post(-0.06) | Delta(-0.06) | (token(-0.06) | _D(-0.05) | Episode(-0.05) | b(-0.05) | pipe(-0.05) | _leaf(-0.05)
    ACCEPTED as axis_885  cumulative_var=0.7793

  [ 881]  axes=886  step_var=0.0012  binary_acc=0.993  gap=0.1029  max_dot=0.0012  (1.8s)
    TOP:  ](0.06) | private(0.06) | );Ċ(0.06) | åĩ¯(0.06) | pend(0.06) | GNU(0.06) | "]),Ċ(0.06) | `ĊĊ(0.06)
    BOT:  (Response(-0.06) | /dev(-0.06) | Ùĩ(-0.06) | Window(-0.06) | allegations(-0.06) | .cert(-0.06) | _plus(-0.06) | åıĳå±ķ(-0.05)
    ACCEPTED as axis_886  cumulative_var=0.7796

  [ 882]  axes=887  step_var=0.0012  binary_acc=0.993  gap=0.1018  max_dot=0.0032  (1.8s)
    TOP:  (ref(0.06) | >=(0.06) | ISM(0.06) | "__(0.06) | å¹³åı°(0.06) | ":(0.06) | èģĺè¯·(0.06) | `[(0.05)
    BOT:  degrees(-0.06) | commands(-0.06) | meta(-0.06) | header(-0.06) | nets(-0.06) | cd(-0.05) | Warning(-0.05) | ê°ľ(-0.05)
    ACCEPTED as axis_887  cumulative_var=0.7798

  [ 883]  axes=888  step_var=0.0012  binary_acc=0.980  gap=0.1020  max_dot=0.0022  (1.9s)
    TOP:  _;Ċ(0.06) | Z(0.06) | (sqrt(0.06) | .iter(0.06) | _content(0.06) | Move(0.06) | )):Ċ(0.05) | Ã©(0.05)
    BOT:  Max(-0.06) | Digit(-0.06) | CÃ´ng(-0.06) | creativecommons(-0.06) | SECRET(-0.06) | setting(-0.06) | åĨ¬(-0.06) | è¯´çĿĢ(-0.06)
    ACCEPTED as axis_888  cumulative_var=0.7801

  [ 884]  axes=889  step_var=0.0012  binary_acc=0.965  gap=0.1009  max_dot=0.0073  (1.8s)
    TOP:  .cal(0.06) | b(0.06) | prise(0.06) | NP(0.06) | Save(0.06) | codec(0.06) | .b(0.05) | (mask(0.05)
    BOT:  ãģĭãĤī(-0.07) | è´£ä»¤(-0.07) | _curve(-0.06) | ç¾İåŃ¦(-0.06) | æķĻèĤ²(-0.06) | åıĪæľī(-0.05) | .INSTANCE(-0.05) | ä¿ĺ(-0.05)
    ACCEPTED as axis_889  cumulative_var=0.7803

  [ 885]  axes=890  step_var=0.0012  binary_acc=0.988  gap=0.1031  max_dot=0.0103  (1.9s)
    TOP:  _gpu(0.07) | èĭį(0.07) | æ¯ıä¸ª(0.06) | åŃ¤(0.06) | åĬ©(0.06) | EL(0.06) | _POST(0.06) | -light(0.06)
    BOT:  chemas(-0.06) | rospy(-0.06) | agens(-0.05) | Using(-0.05) | Ð¼Ð¸Ð½(-0.05) | sqrt(-0.05) | auf(-0.05) | å®ŀçī©(-0.05)
    ACCEPTED as axis_890  cumulative_var=0.7806

  [ 886]  axes=891  step_var=0.0012  binary_acc=0.991  gap=0.1023  max_dot=0.0090  (1.9s)
    TOP:  åħļ(0.06) | considerations(0.05) | videos(0.05) | _ack(0.05) | -y(0.05) | -K(0.05) | åĬłä¸Ĭ(0.05) | åī§æĥħ(0.05)
    BOT:  _HEADER(-0.07) | .tests(-0.07) | .data(-0.06) | .count(-0.06) | Annotation(-0.06) | Adapter(-0.06) | Reviewed(-0.06) | ®(-0.06)
    ACCEPTED as axis_891  cumulative_var=0.7809

  [ 887]  axes=892  step_var=0.0012  binary_acc=0.987  gap=0.1038  max_dot=0.0013  (1.8s)
    TOP:  Ð°Ð·(0.06) | !!(0.06) | è¿Ŀ(0.06) | åįĥéĩĮ(0.06) | (indices(0.06) | æķ´(0.05) | åħį(0.05) | .all(0.05)
    BOT:  Element(-0.07) | _CONTENT(-0.06) | (pool(-0.06) | _W(-0.06) | Provider(-0.06) | Short(-0.06) | Disable(-0.06) | Crypto(-0.06)
    ACCEPTED as axis_892  cumulative_var=0.7811

  [ 888]  axes=893  step_var=0.0012  binary_acc=0.987  gap=0.1011  max_dot=0.0083  (1.8s)
    TOP:  .M(0.06) | Ð°(0.06) | ()[(0.06) | /reset(0.06) | nÃŃ(0.06) | Ã¶m(0.05) | untuk(0.05) | (matrix(0.05)
    BOT:  Context(-0.06) | /blog(-0.06) | atio(-0.06) | æĸŃ(-0.06) | wr(-0.06) | Flatten(-0.06) | cat(-0.05) | Settings(-0.05)
    ACCEPTED as axis_893  cumulative_var=0.7814

  [ 889]  axes=894  step_var=0.0012  binary_acc=0.982  gap=0.1003  max_dot=0.0029  (1.9s)
    TOP:  _fl(0.06) | Ð¾Ð±ÑĢÐ°Ñī(0.05) | Schema(0.05) | ="${(0.05) | [Ċ(0.05) | Units(0.05) | dialect(0.05) | Stub(0.05)
    BOT:  æĤ£èĢħ(-0.06) | Cl(-0.06) | quence(-0.06) | expand(-0.06) | mu(-0.06) | Main(-0.06) | val(-0.06) | UP(-0.06)
    ACCEPTED as axis_894  cumulative_var=0.7816

  [ 890]  axes=895  step_var=0.0012  binary_acc=0.996  gap=0.1024  max_dot=0.0043  (1.8s)
    TOP:  king(0.07) | ^{(0.07) | fe(0.06) | _STATUS(0.06) | (status(0.06) | _password(0.06) | _percent(0.06) | (get(0.06)
    BOT:  .channel(-0.07) | .full(-0.06) | İ(-0.05) | Ã¼m(-0.05) | ares(-0.05) | éģ¿å¼Ģ(-0.05) | rophic(-0.05) | .Canvas(-0.05)
    ACCEPTED as axis_895  cumulative_var=0.7819

  [ 891]  axes=896  step_var=0.0012  binary_acc=1.000  gap=0.1034  max_dot=0.0072  (1.8s)
    TOP:  Serialization(0.07) | _ip(0.07) | IS(0.07) | è¾Ħ(0.07) | mb(0.06) | Arguments(0.06) | æĹ¥(0.06) | *a(0.06)
    BOT:  .stream(-0.06) | _AS(-0.06) | _he(-0.06) | /forum(-0.06) | .sign(-0.05) | _sets(-0.05) | (tb(-0.05) | files(-0.05)
    ACCEPTED as axis_896  cumulative_var=0.7822

  [ 892]  axes=897  step_var=0.0012  binary_acc=0.995  gap=0.1011  max_dot=0.0172  (1.8s)
    TOP:  .generic(0.06) | clip(0.06) | En(0.06) | /sw(0.06) | .value(0.06) | ongoose(0.06) | ...,(0.06) | Still(0.05)
    BOT:  äººæł¼(-0.07) | /debug(-0.06) | åľ°åĽ¾(-0.06) | Toolbar(-0.06) | artin(-0.06) | âĺħ(-0.06) | {...(-0.05) | è¾Ľèĭ¦(-0.05)
    ACCEPTED as axis_897  cumulative_var=0.7824

  [ 893]  axes=898  step_var=0.0012  binary_acc=0.965  gap=0.1038  max_dot=0.0023  (1.8s)
    TOP:  ]):Ċ(0.07) | åĬ¨èĦī(0.06) | åľ°çĲĥ(0.06) | FFT(0.06) | æķ´æĶ¹(0.06) | .facebook(0.06) | ester(0.06) | .messages(0.06)
    BOT:  (bg(-0.06) | .runtime(-0.06) | \-(-0.06) | H(-0.06) | æīĢä»¥æĪĳä»¬(-0.05) | -xl(-0.05) | Prop(-0.05) | (state(-0.05)
    ACCEPTED as axis_898  cumulative_var=0.7827

  [ 894]  axes=899  step_var=0.0012  binary_acc=0.989  gap=0.1057  max_dot=0.0031  (1.8s)
    TOP:  .Notification(0.06) | Exchange(0.05) | Execution(0.05) | converted(0.05) | /version(0.05) | deserialize(0.05) | Shr(0.05) | /render(0.05)
    BOT:  éľĢè¦ģ(-0.07) | arges(-0.07) | .Schema(-0.06) | /news(-0.06) | .App(-0.06) | div(-0.06) | inator(-0.06) | çļĦç»ıåİĨ(-0.06)
    ACCEPTED as axis_899  cumulative_var=0.7829

  [ 895]  axes=900  step_var=0.0012  binary_acc=0.994  gap=0.1037  max_dot=0.0011  (1.8s)
    TOP:  generator(0.06) | çķĻåŃĺ(0.06) | caso(0.05) | supported(0.05) | pct(0.05) | .extract(0.05) | .modify(0.05) | LY(0.05)
    BOT:  æľĢ(-0.07) | _line(-0.07) | =None(-0.07) | èĪŀ(-0.06) | /ext(-0.06) | <p(-0.06) | Animal(-0.06) | inations(-0.06)
    ACCEPTED as axis_900  cumulative_var=0.7832

  [ 896]  axes=901  step_var=0.0012  binary_acc=0.983  gap=0.1021  max_dot=0.0024  (2.0s)
    TOP:  .Dict(0.06) | _script(0.06) | PRO(0.06) | Center(0.06) | .u(0.06) | (Ċ(0.05) | _async(0.05) | +b(0.05)
    BOT:  Bou(-0.06) | éĢĥç¦»(-0.06) | ä»²(-0.06) | Spirits(-0.06) | At(-0.06) | à¸Ľà¸£à¸°à¸ģà¸²à¸¨(-0.05) | åľĥ(-0.05) | ast(-0.05)
    ACCEPTED as axis_901  cumulative_var=0.7835

  [ 897]  axes=902  step_var=0.0012  binary_acc=0.998  gap=0.1028  max_dot=0.0131  (1.8s)
    TOP:  entities(0.06) | /random(0.05) | .action(0.05) | Pipeline(0.05) | build(0.05) | ä¸ļ(0.05) | .metadata(0.05) | Ð´Ð¾Ð»Ð¶Ð½Ð¾(0.05)
    BOT:  .ok(-0.06) | -p(-0.06) | .listen(-0.06) | (filters(-0.06) | .utils(-0.06) | /cms(-0.06) | .ceil(-0.06) | .txt(-0.06)
    ACCEPTED as axis_902  cumulative_var=0.7837

  [ 898]  axes=903  step_var=0.0012  binary_acc=0.980  gap=0.1012  max_dot=0.0077  (1.9s)
    TOP:  /help(0.08) | etic(0.07) | "",(0.06) | .run(0.06) | true(0.06) | .ng(0.06) | çĤ¹åĩ»(0.05) | Adds(0.05)
    BOT:  èµĦ(-0.06) | .em(-0.06) | concerns(-0.06) | numpy(-0.06) | è®¿(-0.06) | .AppCompatActivity(-0.06) | Ð»Ðµ(-0.06) | å´ĸ(-0.06)
    ACCEPTED as axis_903  cumulative_var=0.7840

  [ 899]  axes=904  step_var=0.0012  binary_acc=1.000  gap=0.1018  max_dot=0.0081  (1.9s)
    TOP:  =/(0.06) | ok(0.06) | -keys(0.06) | aking(0.06) | ability(0.06) | (lock(0.05) | {};ĊĊ(0.05) | ([],(0.05)
    BOT:  /z(-0.06) | .Diagnostics(-0.06) | )],(-0.06) | å¾(-0.06) | æ¥¼(-0.05) | =db(-0.05) | æļ´(-0.05) | ç½ĳåĿĢ(-0.05)
    ACCEPTED as axis_904  cumulative_var=0.7842

  [ 900]  axes=905  step_var=0.0012  binary_acc=0.993  gap=0.1015  max_dot=0.0010  (1.8s)
    TOP:  æł·çļĦ(0.06) | _xyz(0.06) | Education(0.06) | èº²(0.06) | ç»ĵæŀĦè°ĥæķ´(0.06) | licate(0.05) | /oauth(0.05) | Tales(0.05)
    BOT:  Ð½Ð¸(-0.06) | ())(-0.05) | '],(-0.05) | .classes(-0.05) | Ã£o(-0.05) | ategorized(-0.05) | Â§(-0.05) | (-0.05)
    ACCEPTED as axis_905  cumulative_var=0.7845

  [ 901]  axes=906  step_var=0.0012  binary_acc=0.983  gap=0.1005  max_dot=0.0015  (1.9s)
    TOP:  .sp(0.06) | _g(0.06) | calc(0.06) | .cart(0.06) | Global(0.06) | .before(0.06) | è§Ħæ¨¡ä»¥ä¸Ĭ(0.06) | \M(0.06)
    BOT:  çĲ¥(-0.07) | job(-0.06) | firebase(-0.05) | _map(-0.05) | initialized(-0.05) | _estimator(-0.05) | ries(-0.05) | Den(-0.05)
    ACCEPTED as axis_906  cumulative_var=0.7847

  [ 902]  axes=907  step_var=0.0012  binary_acc=0.980  gap=0.1039  max_dot=0.0063  (1.8s)
    TOP:  ìłĢ(0.06) | ç¾İæĸ¹(0.06) | .pref(0.06) | solo(0.05) | Beginner(0.05) | Äĳá»Ļ(0.05) | ogeneous(0.05) | .un(0.05)
    BOT:  })Ċ(-0.06) | _feature(-0.06) | DATABASE(-0.06) | Ð¸ÑĤÐµÐ»ÑĮ(-0.05) | ADM(-0.05) | BASE(-0.05) | ä½įç½®(-0.05) | è¿½(-0.05)
    ACCEPTED as axis_907  cumulative_var=0.7850

  [ 903]  axes=908  step_var=0.0012  binary_acc=0.971  gap=0.1014  max_dot=0.0009  (1.9s)
    TOP:  /users(0.07) | .R(0.06) | Sand(0.06) | _IF(0.06) | .array(0.06) | æīĢåľ¨åľ°(0.05) | .channel(0.05) | /C(0.05)
    BOT:  zend(-0.06) | çĥĽ(-0.06) | _ACCESS(-0.06) | .duration(-0.06) | ker(-0.06) | }-(-0.06) | ROP(-0.05) | Foot(-0.05)
    ACCEPTED as axis_908  cumulative_var=0.7853

  [ 904]  axes=909  step_var=0.0012  binary_acc=0.983  gap=0.1016  max_dot=0.0019  (1.9s)
    TOP:  .exceptions(0.07) | chen(0.07) | autor(0.06) | Ack(0.06) | .openapi(0.06) | .document(0.06) | HEADER(0.05) | MOD(0.05)
    BOT:  Ð¾ÑĢÐ³Ð°Ð½Ð¸Ð·Ð°ÑĨÐ¸Ð¸(-0.06) | ç¬¬ä¸Ģä¸ª(-0.06) | rho(-0.06) | Prop(-0.06) | /src(-0.06) | .react(-0.05) | ä½ľçĶ¨(-0.05) | èĩª(-0.05)
    ACCEPTED as axis_909  cumulative_var=0.7855

  [ 905]  axes=910  step_var=0.0012  binary_acc=0.983  gap=0.1009  max_dot=0.0052  (1.9s)
    TOP:  #include(0.07) | ================================================================(0.07) | æİ¥è¿ĳ(0.06) | æĺ¯(0.06) | ================================(0.06) | è¶Ĭé«ĺ(0.06) | >();ĊĊ(0.06) | åį«çĶŁåģ¥åº·(0.06)
    BOT:  erview(-0.06) | -license(-0.06) | controller(-0.06) | è°§(-0.05) | (g(-0.05) | clients(-0.05) | (table(-0.05) | ç»ĵ(-0.05)
    ACCEPTED as axis_910  cumulative_var=0.7858

  [ 906]  axes=911  step_var=0.0012  binary_acc=0.979  gap=0.1015  max_dot=0.0004  (1.9s)
    TOP:  )")Ċ(0.06) | .Order(0.06) | _array(0.06) | .init(0.06) | ):ĊĊ(0.06) | .PO(0.06) | ])ĊĊ(0.06) | !,(0.06)
    BOT:  ager(-0.06) | de(-0.06) | Feature(-0.06) | icularly(-0.06) | "T(-0.06) | _sum(-0.06) | By(-0.06) | æł¹æį®(-0.05)
    ACCEPTED as axis_911  cumulative_var=0.7860

  [ 907]  axes=912  step_var=0.0012  binary_acc=0.995  gap=0.1015  max_dot=0.0029  (1.9s)
    TOP:  guarantee(0.06) | Datos(0.05) | Population(0.05) | Nel(0.05) | Hiring(0.05) | Verb(0.05) | deprecated(0.05) | Experimental(0.05)
    BOT:  éķľ(-0.07) | ],(-0.07) | .cat(-0.06) | lig(-0.06) | /i(-0.06) | /j(-0.06) | _ai(-0.06) | ]Ċ(-0.06)
    ACCEPTED as axis_912  cumulative_var=0.7863

  [ 908]  axes=913  step_var=0.0012  binary_acc=0.958  gap=0.1005  max_dot=0.0100  (1.8s)
    TOP:  RPC(0.06) | ERING(0.06) | Api(0.05) | èĥ«(0.05) | .est(0.05) | Include(0.05) | Marine(0.05) | Alias(0.05)
    BOT:  fait(-0.07) | tf(-0.06) | çĽ®åīį(-0.06) | nan(-0.06) | .token(-0.06) | Communications(-0.06) | _user(-0.06) | çłĶ(-0.06)
    ACCEPTED as axis_913  cumulative_var=0.7865

  [ 909]  axes=914  step_var=0.0012  binary_acc=0.998  gap=0.0993  max_dot=0.0095  (1.9s)
    TOP:  Â©(0.06) | ÑĦÐ¾ÑĢÐ¼Ñĥ(0.06) | Bitmap(0.06) | --(0.05) | Assessment(0.05) | æĶ»åĿļ(0.05) | >*(0.05) | èģĺ(0.05)
    BOT:  ÐºÐ¾(-0.06) | ulus(-0.06) | åģ¥åº·åıĳå±ķ(-0.06) | ella(-0.06) | ances(-0.06) | ry(-0.06) | .annotation(-0.06) | ];Ċ(-0.06)
    ACCEPTED as axis_914  cumulative_var=0.7868

  [ 910]  axes=915  step_var=0.0012  binary_acc=0.954  gap=0.0992  max_dot=0.0028  (1.8s)
    TOP:  C(0.07) | der(0.07) | ">Ċ(0.06) | -backend(0.06) | includes(0.06) | SPDX(0.06) | Ron(0.06) | -p(0.06)
    BOT:  plain(-0.06) | Style(-0.06) | .decode(-0.06) | .label(-0.06) | _START(-0.06) | _ep(-0.06) | _plugin(-0.06) | tree(-0.06)
    ACCEPTED as axis_915  cumulative_var=0.7871

  [ 911]  axes=916  step_var=0.0012  binary_acc=0.996  gap=0.0994  max_dot=0.0010  (1.8s)
    TOP:  å¯¹å¤ĸå¼ĢæĶ¾(0.07) | -toggle(0.06) | æłı(0.06) | Services(0.06) | .Text(0.06) | opt(0.06) | ç©Ĺ(0.05) | demo(0.05)
    BOT:  ."ĊĊ(-0.07) | ,n(-0.06) | /tree(-0.06) | /how(-0.06) | division(-0.06) | ãĤįãģĨ(-0.06) | .test(-0.05) | ,Ċ(-0.05)
    ACCEPTED as axis_916  cumulative_var=0.7873

  [ 912]  axes=917  step_var=0.0012  binary_acc=0.990  gap=0.1010  max_dot=0.0035  (1.9s)
    TOP:  Actual(0.06) | Axis(0.06) | iar(0.06) | "#(0.06) | DataTypes(0.06) | å°Ĭéĩį(0.06) | On(0.06) | .lr(0.06)
    BOT:  _shape(-0.08) | General(-0.06) | die(-0.06) | _time(-0.06) | æİ¢(-0.06) | .astype(-0.06) | ä¸ľæĸ¹(-0.05) | .cpu(-0.05)
    ACCEPTED as axis_917  cumulative_var=0.7876

  [ 913]  axes=918  step_var=0.0012  binary_acc=0.975  gap=0.1017  max_dot=0.0047  (1.8s)
    TOP:  /api(0.06) | \M(0.06) | çŃīæĥħåĨµ(0.06) | /search(0.06) | -logo(0.06) | anton(0.06) | Writer(0.05) | Browser(0.05)
    BOT:  .blit(-0.06) | .ep(-0.06) | payment(-0.06) | .error(-0.06) | (keys(-0.05) | .no(-0.05) | jdbc(-0.05) | Davis(-0.05)
    ACCEPTED as axis_918  cumulative_var=0.7878

  [ 914]  axes=919  step_var=0.0012  binary_acc=0.973  gap=0.1001  max_dot=0.0010  (2.0s)
    TOP:  .filters(0.07) | >;(0.06) | Step(0.06) | _k(0.06) | .t(0.05) | å®¶ä¹¡(0.05) | */(0.05) | ...)(0.05)
    BOT:  èĢķ(-0.06) | äºĶ(-0.06) | READ(-0.06) | ÑĩÐ¸Ðº(-0.06) | .Connection(-0.06) | QtWidgets(-0.06) | bw(-0.06) | beta(-0.06)
    ACCEPTED as axis_919  cumulative_var=0.7881

  [ 915]  axes=920  step_var=0.0012  binary_acc=0.970  gap=0.1018  max_dot=0.0050  (1.8s)
    TOP:  æīĭç»Ń(0.06) | _surface(0.06) | initial(0.06) | .release(0.06) | Id(0.06) | íı¬(0.06) | .kernel(0.06) | bá»Ļ(0.05)
    BOT:  AIN(-0.06) | et(-0.06) | mr(-0.06) | åı¥(-0.06) | æĬķèº«(-0.05) | .future(-0.05) | ((-0.05) | ä¼ĺè´¨(-0.05)
    ACCEPTED as axis_920  cumulative_var=0.7883

  [ 916]  axes=921  step_var=0.0012  binary_acc=0.991  gap=0.1007  max_dot=0.0033  (1.8s)
    TOP:  _clear(0.07) | Ø¯(0.06) | (func(0.06) | (dataset(0.06) | FilePath(0.06) | Comment(0.06) | åĶĲä»£(0.06) | End(0.06)
    BOT:  èĢĮå¯¼èĩ´(-0.07) | uke(-0.06) | å·²ç¶ĵ(-0.05) | æľ¨æĿĲ(-0.05) | THE(-0.05) | ÑĩÑĤÐ¾(-0.05) | Builder(-0.05) | (y(-0.05)
    ACCEPTED as axis_921  cumulative_var=0.7886

  [ 917]  axes=922  step_var=0.0012  binary_acc=0.999  gap=0.0997  max_dot=0.0036  (1.8s)
    TOP:  ad(0.07) | ãĤ¢(0.06) | è¿ª(0.06) | *_(0.06) | ").Ċ(0.06) | ä»½é¢Ŀ(0.06) | ST(0.06) | .tasks(0.06)
    BOT:  Louise(-0.06) | .Assert(-0.06) | åįļå£«(-0.05) | \Module(-0.05) | .image(-0.05) | ç«łç¨ĭ(-0.05) | mission(-0.05) | .Resize(-0.05)
    ACCEPTED as axis_922  cumulative_var=0.7888

  [ 918]  axes=923  step_var=0.0012  binary_acc=0.972  gap=0.1006  max_dot=0.0058  (1.9s)
    TOP:  ogeneity(0.06) | Generic(0.06) | request(0.05) | é¢§(0.05) | _location(0.05) | ä»Ģä¹ĪæĹ¶åĢĻ(0.05) | r(0.05) | change(0.05)
    BOT:  Arts(-0.06) | Operator(-0.06) | .open(-0.06) | _is(-0.06) | -links(-0.06) | æ±ĩ(-0.06) | Reg(-0.06) | æ¹ĸåĮº(-0.06)
    ACCEPTED as axis_923  cumulative_var=0.7891

  [ 919]  axes=924  step_var=0.0012  binary_acc=0.987  gap=0.0995  max_dot=0.0109  (1.8s)
    TOP:  _CONTENT(0.06) | .Required(0.06) | Template(0.05) | è®°èĢħ(0.05) | Listing(0.05) | STRING(0.05) | ).čĊ(0.05) | reconciliation(0.05)
    BOT:  à¸µ(-0.06) | _LIBRARY(-0.06) | ç²¹(-0.06) | os(-0.06) | iciency(-0.06) | exp(-0.06) | osen(-0.06) | .test(-0.06)
    ACCEPTED as axis_924  cumulative_var=0.7893

  [ 920]  axes=925  step_var=0.0012  binary_acc=0.999  gap=0.1035  max_dot=0.0037  (1.9s)
    TOP:  /license(0.08) | _simple(0.06) | è¾ħåĬ©(0.05) | .geometry(0.05) | adapter(0.05) | ValidationError(0.05) | ç»ıèĲ¥èĮĥåĽ´(0.05) | ä¿¡æģ¯(0.05)
    BOT:  _session(-0.07) | viz(-0.06) | TestCase(-0.06) | (c(-0.06) | Kernel(-0.06) | ãĢĳĊĊ(-0.05) | TensorFlow(-0.05) | âĢĿãĢĤ(-0.05)
    ACCEPTED as axis_925  cumulative_var=0.7896

  [ 921]  axes=926  step_var=0.0012  binary_acc=0.964  gap=0.1012  max_dot=0.0039  (1.9s)
    TOP:  /testing(0.06) | .side(0.06) | .service(0.06) | Ã¤t(0.06) | id(0.05) | ãģĭ(0.05) | èµ·åĪĿ(0.05) | dx(0.05)
    BOT:  recherche(-0.06) | é£İæĥħ(-0.06) | Brief(-0.06) | jim(-0.06) | ç£¨(-0.05) | Delivery(-0.05) | .beh(-0.05) | Visibility(-0.05)
    ACCEPTED as axis_926  cumulative_var=0.7898

  [ 922]  axes=927  step_var=0.0012  binary_acc=0.981  gap=0.1015  max_dot=0.0040  (1.9s)
    TOP:  -auto(0.06) | It(0.06) | /[(0.06) | educational(0.06) | .Type(0.05) | FA(0.05) | argv(0.05) | session(0.05)
    BOT:  ires(-0.07) | _name(-0.07) | Graphics(-0.06) | )\(-0.06) | Trace(-0.06) | Î½(-0.06) | .http(-0.06) | Q(-0.06)
    ACCEPTED as axis_927  cumulative_var=0.7901

  [ 923]  axes=928  step_var=0.0012  binary_acc=0.992  gap=0.1010  max_dot=0.0021  (1.9s)
    TOP:  C(0.08) | it(0.07) | .dtype(0.06) | y(0.06) | <(0.06) | emplate(0.06) | åĵ¦(0.06) | åĪ¶(0.06)
    BOT:  (db(-0.06) | æ¡ĥ(-0.06) | (sqrt(-0.06) | Communications(-0.05) | exhaustive(-0.05) | .apps(-0.05) | .Button(-0.05) | (Graphics(-0.05)
    ACCEPTED as axis_928  cumulative_var=0.7904

  [ 924]  axes=929  step_var=0.0012  binary_acc=0.988  gap=0.1026  max_dot=0.0057  (2.0s)
    TOP:  _raw(0.06) | Required(0.06) | hos(0.06) | ')čĊčĊ(0.05) | /material(0.05) | /f(0.05) | .eng(0.05) | åŁİå¸Ĥ(0.05)
    BOT:  asha(-0.07) | -heading(-0.06) | N(-0.06) | ='(-0.06) | {\(-0.06) | >$(-0.06) | å¹¶(-0.06) | {(-0.06)
    ACCEPTED as axis_929  cumulative_var=0.7906

  [ 925]  axes=930  step_var=0.0012  binary_acc=0.997  gap=0.0997  max_dot=0.0067  (1.9s)
    TOP:  æľĢ(0.06) | é«ĭ(0.06) | çŃĸ(0.06) | /cms(0.06) | -cont(0.06) | akash(0.06) | æİ¥è§¦(0.06) | çĿ¹(0.06)
    BOT:  (Object(-0.06) | Ver(-0.06) | Phone(-0.06) | Ca(-0.06) | _labels(-0.06) | Secondary(-0.06) | ä¸ºé¦ĸ(-0.06) | /light(-0.06)
    ACCEPTED as axis_930  cumulative_var=0.7909

  [ 926]  axes=931  step_var=0.0012  binary_acc=0.975  gap=0.0991  max_dot=0.0009  (1.9s)
    TOP:  .profile(0.06) | eur(0.06) | ik(0.06) | olk(0.06) | .m(0.06) | art(0.06) | (0.06) | branch(0.06)
    BOT:  _extra(-0.06) | Politics(-0.06) | "*.(-0.05) | _DISABLED(-0.05) | .measure(-0.05) | åĲĳçĿĢ(-0.05) | r(-0.05) | .Parameters(-0.05)
    ACCEPTED as axis_931  cumulative_var=0.7911

  [ 927]  axes=932  step_var=0.0012  binary_acc=0.984  gap=0.1026  max_dot=0.0104  (1.8s)
    TOP:  .Event(0.06) | .material(0.06) | scheme(0.06) | .Hosting(0.06) | ElementType(0.06) | .ml(0.06) | empo(0.06) | Br(0.05)
    BOT:  çŁ¥è¯Ĩ(-0.06) | å¤´(-0.06) | Path(-0.06) | _DATA(-0.06) | ê°Ĵ(-0.06) | ä¸ŃçļĦ(-0.06) | æİªæĸ½(-0.06) | (real(-0.06)
    ACCEPTED as axis_932  cumulative_var=0.7914

  [ 928]  axes=933  step_var=0.0012  binary_acc=0.983  gap=0.0993  max_dot=0.0053  (1.8s)
    TOP:  ia(0.07) | (seed(0.06) | -radius(0.06) | -google(0.06) | (region(0.06) | .ContentType(0.06) | DIR(0.06) | AS(0.06)
    BOT:  feeds(-0.07) | èĤ¡ä¸ľ(-0.06) | Material(-0.06) | notify(-0.06) | .write(-0.06) | Housing(-0.06) | List(-0.06) | validator(-0.06)
    ACCEPTED as axis_933  cumulative_var=0.7916

  [ 929]  axes=934  step_var=0.0012  binary_acc=0.989  gap=0.1012  max_dot=0.0119  (1.9s)
    TOP:  Ð¾Ð´Ð½Ð¾(0.06) | -links(0.06) | Cou(0.06) | .context(0.06) | l(0.06) | IND(0.06) | RES(0.05) | IR(0.05)
    BOT:  /ex(-0.06) | _MODE(-0.06) | _detector(-0.06) | Loop(-0.06) | rait(-0.06) | âĢĿ;(-0.06) | å¼Ģåıĳ(-0.05) | ois(-0.05)
    ACCEPTED as axis_934  cumulative_var=0.7919

  [ 930]  axes=935  step_var=0.0012  binary_acc=0.982  gap=0.1011  max_dot=0.0036  (1.9s)
    TOP:  å·§(0.06) | sa(0.06) | Activation(0.06) | Crypto(0.06) | åħ·(0.06) | çļĦçĶŁåĳ½(0.06) | ä¸įè§ģ(0.06) | .Line(0.05)
    BOT:  :a(-0.07) | _split(-0.06) | column(-0.06) | cpu(-0.06) | [n(-0.06) | _exe(-0.06) | };ĊĊ(-0.05) | -bottom(-0.05)
    ACCEPTED as axis_935  cumulative_var=0.7921

  [ 931]  axes=936  step_var=0.0012  binary_acc=0.995  gap=0.1003  max_dot=0.0035  (1.8s)
    TOP:  .per(0.06) | '):Ċ(0.06) | /@(0.06) | '])(0.06) | ([]);Ċ(0.06) | /?(0.06) | ']].(0.06) | +\(0.06)
    BOT:  ty(-0.06) | sd(-0.06) | grid(-0.06) | Payment(-0.06) | æ¬§(-0.06) | .translate(-0.06) | dre(-0.05) | .integration(-0.05)
    ACCEPTED as axis_936  cumulative_var=0.7924

  [ 932]  axes=937  step_var=0.0012  binary_acc=0.994  gap=0.1021  max_dot=0.0086  (1.8s)
    TOP:  ç»Ń(0.06) | æİĮ(0.06) | _CHECK(0.06) | .O(0.06) | .inverse(0.06) | åģ¥(0.05) | å¤§å°ı(0.05) | çĶ»(0.05)
    BOT:  Swagger(-0.06) | Style(-0.06) | Constants(-0.06) | ikal(-0.06) | #ĊĊ(-0.06) | -I(-0.06) | metadata(-0.06) | '''ĊĊ(-0.05)
    ACCEPTED as axis_937  cumulative_var=0.7926

  [ 933]  axes=938  step_var=0.0012  binary_acc=0.988  gap=0.1006  max_dot=0.0072  (1.8s)
    TOP:  ãģ¶(0.06) | åİŁåĪĻ(0.06) | æ¼ı(0.05) | COPYING(0.05) | è¿ª(0.05) | ä¸įå®ī(0.05) | ases(0.05) | Zah(0.05)
    BOT:  /W(-0.06) | region(-0.06) | =my(-0.06) | OA(-0.05) | extended(-0.05) | .Input(-0.05) | _visual(-0.05) | /inc(-0.05)
    ACCEPTED as axis_938  cumulative_var=0.7929

  [ 934]  axes=939  step_var=0.0012  binary_acc=0.976  gap=0.1026  max_dot=0.0039  (1.8s)
    TOP:  ))ĊĊ(0.06) | .core(0.06) | ÙĪÙĨ(0.06) | .gen(0.06) | .l(0.06) | _INLINE(0.06) | con(0.06) | sm(0.06)
    BOT:  lider(-0.06) | alias(-0.06) | Ning(-0.06) | Import(-0.05) | Ð²ÐµÐ´(-0.05) | Print(-0.05) | _sparse(-0.05) | _SINGLE(-0.05)
    ACCEPTED as axis_939  cumulative_var=0.7931

  [ 935]  axes=940  step_var=0.0012  binary_acc=0.975  gap=0.1009  max_dot=0.0019  (1.8s)
    TOP:  all(0.07) | Pages(0.06) | Director(0.06) | å¥½åĲĥ(0.06) | .Meta(0.06) | XML(0.06) | pod(0.06) | '*(0.06)
    BOT:  attributes(-0.06) | -auto(-0.06) | hift(-0.06) | Geometry(-0.05) | æ¥·(-0.05) | è¿Ĳç®Ĺ(-0.05) | _html(-0.05) | æĳ¸(-0.05)
    ACCEPTED as axis_940  cumulative_var=0.7934

  [ 936]  axes=941  step_var=0.0012  binary_acc=0.999  gap=0.0989  max_dot=0.0021  (1.8s)
    TOP:  |ĊĊ(0.06) | çļĦä½ľçĶ¨(0.06) | åĩºèº«(0.05) | PAGE(0.05) | imates(0.05) | ÑĤÐ¾(0.05) | argument(0.05) | /win(0.05)
    BOT:  åĽ¢ç»ĵ(-0.06) | .entities(-0.06) | ãĥĸãĥ©(-0.06) | Res(-0.06) | secutive(-0.05) | _async(-0.05) | Border(-0.05) | .module(-0.05)
    ACCEPTED as axis_941  cumulative_var=0.7936

  [ 937]  axes=942  step_var=0.0012  binary_acc=0.984  gap=0.1000  max_dot=0.0036  (1.8s)
    TOP:  /image(0.06) | bounds(0.06) | è¦ģæ³¨æĦı(0.06) | åĽ½å®¶çº§(0.06) | Education(0.05) | gren(0.05) | Cloud(0.05) | åīĳ(0.05)
    BOT:  ected(-0.07) | _CURRENT(-0.06) | ä¸º(-0.06) | -mod(-0.06) | Stub(-0.06) | [str(-0.06) | (~(-0.06) | CW(-0.06)
    ACCEPTED as axis_942  cumulative_var=0.7939

  [ 938]  axes=943  step_var=0.0012  binary_acc=0.985  gap=0.0985  max_dot=0.0129  (1.9s)
    TOP:  ç¥Ŀ(0.07) | Assert(0.06) | .Auth(0.06) | Scope(0.06) | æĢĴ(0.06) | _artist(0.06) | ICH(0.06) | æĪĳä¼ļ(0.06)
    BOT:  Velocity(-0.06) | ari(-0.06) | Monitor(-0.06) | è¿Ĳç®Ĺ(-0.06) | combined(-0.06) | åħįè´£å£°æĺİ(-0.05) | lib(-0.05) | _helpers(-0.05)
    ACCEPTED as axis_943  cumulative_var=0.7941

  [ 939]  axes=944  step_var=0.0012  binary_acc=0.961  gap=0.1008  max_dot=0.0070  (1.8s)
    TOP:  aning(0.07) | /icon(0.06) | .relative(0.06) | _shape(0.06) | resume(0.05) | =>(0.05) | ¥(0.05) | _levels(0.05)
    BOT:  legacy(-0.06) | /login(-0.06) | .chat(-0.06) | .d(-0.05) | ALLY(-0.05) | .list(-0.05) | If(-0.05) | Apt(-0.05)
    ACCEPTED as axis_944  cumulative_var=0.7944

  [ 940]  axes=945  step_var=0.0012  binary_acc=0.969  gap=0.1007  max_dot=0.0026  (1.8s)
    TOP:  _command(0.06) | -nav(0.05) | _do(0.05) | _fields(0.05) | à¸ĩ(0.05) | Class(0.05) | not(0.05) | PURE(0.05)
    BOT:  èªĵ(-0.07) | .CLIENT(-0.06) | åħ¬å¼ı(-0.06) | .OS(-0.06) | __.(-0.06) | å®ĥä»¬(-0.05) | /blob(-0.05) | ä¸įåħ·å¤ĩ(-0.05)
    ACCEPTED as axis_945  cumulative_var=0.7946

  [ 941]  axes=946  step_var=0.0012  binary_acc=0.995  gap=0.0974  max_dot=0.0144  (1.9s)
    TOP:  Metadata(0.06) | è¯¬(0.06) | .Object(0.06) | /c(0.06) | Threshold(0.05) | eg(0.05) | _Ċ(0.05) | è¢«åĳĬ(0.05)
    BOT:  .plugins(-0.06) | éī´(-0.06) | çļĦçģµéŃĤ(-0.06) | Dess(-0.06) | Alpha(-0.06) | Faces(-0.05) | æľī(-0.05) | Variables(-0.05)
    ACCEPTED as axis_946  cumulative_var=0.7949

  [ 942]  axes=947  step_var=0.0012  binary_acc=0.992  gap=0.0996  max_dot=0.0054  (1.9s)
    TOP:  ÑĨ(0.07) | _correct(0.06) | top(0.06) | çķĮçļĦ(0.06) | ').(0.06) | def(0.06) | .cmd(0.06) | .Error(0.06)
    BOT:  _LINK(-0.06) | (kind(-0.05) | ,is(-0.05) | íĦ°(-0.05) | fb(-0.05) | .dis(-0.05) | è¾¾äºº(-0.05) | _period(-0.05)
    ACCEPTED as axis_947  cumulative_var=0.7951

  [ 943]  axes=948  step_var=0.0012  binary_acc=1.000  gap=0.0996  max_dot=0.0099  (1.9s)
    TOP:  Tales(0.05) | author(0.05) | ä¼Ĭ(0.05) | ouis(0.05) | ver(0.05) | sample(0.05) | omens(0.05) | .proc(0.05)
    BOT:  */Ċ(-0.06) | Router(-0.06) | _tokens(-0.06) | Argentina(-0.05) | .Positive(-0.05) | .exception(-0.05) | stock(-0.05) | /products(-0.05)
    ACCEPTED as axis_948  cumulative_var=0.7954

  [ 944]  axes=949  step_var=0.0012  binary_acc=0.978  gap=0.0997  max_dot=0.0144  (1.9s)
    TOP:  ãĤ¤ãĥ³(0.06) | _import(0.06) | leaning(0.06) | -p(0.06) | onto(0.06) | _exists(0.06) | '})Ċ(0.05) | IRD(0.05)
    BOT:  .contract(-0.06) | Fr(-0.05) | .Line(-0.05) | services(-0.05) | Dialog(-0.05) | GEN(-0.05) | _unit(-0.05) | lÃŃnea(-0.05)
    ACCEPTED as axis_949  cumulative_var=0.7956

  [ 945]  axes=950  step_var=0.0012  binary_acc=0.978  gap=0.0971  max_dot=0.0024  (1.9s)
    TOP:  _component(0.07) | å®ŀçİ°(0.06) | /h(0.06) | ãģ«(0.06) | å¾®(0.06) | Chapman(0.06) | _hint(0.06) | åįģ(0.06)
    BOT:  ZZ(-0.06) | FALSE(-0.06) | Async(-0.06) | .âĢľ(-0.06) | PE(-0.05) | _oauth(-0.05) | Chat(-0.05) | Vid(-0.05)
    ACCEPTED as axis_950  cumulative_var=0.7959

  [ 946]  axes=951  step_var=0.0012  binary_acc=0.977  gap=0.0987  max_dot=0.0047  (1.8s)
    TOP:  ł(0.06) | /book(0.06) | _equal(0.06) | .proj(0.06) | _destroy(0.06) | abetic(0.06) | /files(0.06) | .mask(0.06)
    BOT:  Bind(-0.06) | Modify(-0.06) | By(-0.06) | _base(-0.06) | èĪħ(-0.05) | Daily(-0.05) | Ð±Ð°Ð»(-0.05) | Streaming(-0.05)
    ACCEPTED as axis_951  cumulative_var=0.7961

  [ 947]  axes=952  step_var=0.0012  binary_acc=0.977  gap=0.0997  max_dot=0.0103  (1.9s)
    TOP:  è¿ĩ(0.06) | á»ĭ(0.06) | enerate(0.06) | _original(0.05) | Display(0.05) | printing(0.05) | -tests(0.05) | æĥħ(0.05)
    BOT:  ():čĊ(-0.06) | .Duration(-0.06) | Topic(-0.06) | RESOURCE(-0.06) | .frame(-0.06) | used(-0.06) | namespace(-0.06) | ateway(-0.06)
    ACCEPTED as axis_952  cumulative_var=0.7964

  [ 948]  axes=953  step_var=0.0012  binary_acc=0.995  gap=0.0993  max_dot=0.0081  (1.9s)
    TOP:  .J(0.06) | .misc(0.06) | .reshape(0.06) | engine(0.06) | _tokens(0.06) | SAVE(0.06) | +w(0.05) | a(0.05)
    BOT:  .analysis(-0.06) | kip(-0.06) | ãĢı(-0.06) | çłģ(-0.05) | BBB(-0.05) | (err(-0.05) | Scheme(-0.05) | alpha(-0.05)
    ACCEPTED as axis_953  cumulative_var=0.7966

  [ 949]  axes=954  step_var=0.0012  binary_acc=0.967  gap=0.0993  max_dot=0.0066  (1.8s)
    TOP:  Ali(0.06) | Unknown(0.06) | =edge(0.05) | (rand(0.05) | .ContentType(0.05) | adores(0.05) | Science(0.05) | .try(0.05)
    BOT:  m(-0.06) | Ðµ(-0.06) | ä¾µçĬ¯(-0.06) | Ð¥(-0.06) | Ð¼(-0.06) | at(-0.06) | ATED(-0.06) | K(-0.06)
    ACCEPTED as axis_954  cumulative_var=0.7969

  [ 950]  axes=955  step_var=0.0012  binary_acc=0.998  gap=0.1004  max_dot=0.0072  (1.8s)
    TOP:  _resource(0.07) | _copy(0.06) | _and(0.06) | .cluster(0.06) | egral(0.06) | BB(0.06) | ud(0.06) | Ð°Ñģ(0.06)
    BOT:  (tk(-0.06) | åħ³å¿ĥ(-0.06) | åĲĽåŃĲ(-0.06) | ine(-0.06) | abies(-0.05) | \Exceptions(-0.05) | ÙħØ©(-0.05) | èĬ±å¼Ģ(-0.05)
    ACCEPTED as axis_955  cumulative_var=0.7971

  [ 951]  axes=956  step_var=0.0012  binary_acc=0.988  gap=0.0995  max_dot=0.0195  (1.8s)
    TOP:  Ø³ÙĬ(0.06) | Procedures(0.06) | ({Ċ(0.06) | wares(0.06) | ãĥĥãĥĪ(0.05) | (r(0.05) | Peace(0.05) | finds(0.05)
    BOT:  wb(-0.07) | .Scene(-0.06) | epy(-0.06) | .fl(-0.05) | Popup(-0.05) | ä¹Ĳ(-0.05) | ISK(-0.05) | {}'.(-0.05)
    ACCEPTED as axis_956  cumulative_var=0.7974

  [ 952]  axes=957  step_var=0.0012  binary_acc=0.985  gap=0.1015  max_dot=0.0073  (2.0s)
    TOP:  ent(0.08) | .collections(0.07) | .gl(0.06) | Ð°ÑĤÑĮ(0.06) | inct(0.06) | Stadium(0.06) | osing(0.05) | ÐµÑĤ(0.05)
    BOT:  _CHECK(-0.06) | ç©ºæ°Ķä¸Ń(-0.06) | ("$(-0.05) | -config(-0.05) | package(-0.05) | /helpers(-0.05) | disk(-0.05) | xmlns(-0.05)
    ACCEPTED as axis_957  cumulative_var=0.7976

  [ 953]  axes=958  step_var=0.0012  binary_acc=0.979  gap=0.0997  max_dot=0.0113  (1.8s)
    TOP:  .V(0.06) | note(0.06) | .Theme(0.06) | _DAT(0.06) | (scale(0.06) | ãģ³(0.06) | _black(0.06) | .Unit(0.06)
    BOT:  Search(-0.06) | åħ¬åı¸ç«łç¨ĭ(-0.06) | _location(-0.05) | ge(-0.05) | æµģåĩº(-0.05) | LN(-0.05) | è·ª(-0.05) | onic(-0.05)
    ACCEPTED as axis_958  cumulative_var=0.7979

  [ 954]  axes=959  step_var=0.0012  binary_acc=0.968  gap=0.1010  max_dot=0.0028  (1.8s)
    TOP:  format(0.06) | si(0.06) | ):Ċ(0.06) | .notifications(0.06) | ():čĊ(0.06) | OTHER(0.06) | ifes(0.06) | `čĊ(0.06)
    BOT:  .ver(-0.06) | les(-0.06) | ëħĦ(-0.06) | _window(-0.06) | _ind(-0.05) | .Key(-0.05) | /AP(-0.05) | .abs(-0.05)
    ACCEPTED as axis_959  cumulative_var=0.7981

  [ 955]  axes=960  step_var=0.0012  binary_acc=0.988  gap=0.0983  max_dot=0.0096  (1.9s)
    TOP:  .isEnabled(0.06) | .opt(0.06) | Sch(0.06) | feature(0.06) | essa(0.06) | categories(0.06) | .describe(0.05) | éĨĴ(0.05)
    BOT:  ìĿĦ(-0.06) | å°±æĺ¯(-0.06) | ÑĥÐ´(-0.06) | ä¸Ģçº§(-0.06) | oothing(-0.06) | j(-0.06) | º(-0.05) | .email(-0.05)
    ACCEPTED as axis_960  cumulative_var=0.7984

  [ 956]  axes=961  step_var=0.0012  binary_acc=1.000  gap=0.1007  max_dot=0.0017  (1.9s)
    TOP:  ä¸ĭ(0.06) | Tools(0.06) | åķĨäºº(0.06) | Apply(0.06) | Unique(0.06) | ä¼¤äº¡(0.05) | å§ĭç»Īä¿ĿæĮģ(0.05) | æĢ§(0.05)
    BOT:  '.(-0.06) | listeners(-0.06) | .Content(-0.06) | æĮĤçīĮ(-0.06) | omm(-0.06) | :utf(-0.05) | .?(-0.05) | å®ĺæĸ¹ç½ĳç«Ļ(-0.05)
    ACCEPTED as axis_961  cumulative_var=0.7986

  [ 957]  axes=962  step_var=0.0012  binary_acc=0.996  gap=0.1024  max_dot=0.0063  (1.9s)
    TOP:  ç½Ĺæĸ¯(0.06) | -find(0.06) | _lazy(0.06) | ATERIAL(0.06) | gio(0.06) | ì¶Ķ(0.05) | Ð½Ð¸(0.05) | seconds(0.05)
    BOT:  u(-0.06) | -messages(-0.06) | (S(-0.06) | Hor(-0.05) | æĶ¹éĿ©åĪĽæĸ°(-0.05) | Provides(-0.05) | KI(-0.05) | Newtonsoft(-0.05)
    ACCEPTED as axis_962  cumulative_var=0.7989

  [ 958]  axes=963  step_var=0.0012  binary_acc=0.977  gap=0.1022  max_dot=0.0094  (1.8s)
    TOP:  ley(0.07) | =c(0.07) | âĢĻ(0.07) | Staff(0.06) | (0.06) | !!!ĊĊ(0.06) | */ĊĊ(0.06) | æī§è¡Į(0.06)
    BOT:  ÐµÐº(-0.06) | .Listener(-0.06) | .log(-0.06) | /cc(-0.06) | Pag(-0.06) | urar(-0.05) | Ð¾ÑĤ(-0.05) | funktion(-0.05)
    ACCEPTED as axis_963  cumulative_var=0.7991

  [ 959]  axes=964  step_var=0.0012  binary_acc=0.966  gap=0.1013  max_dot=0.0100  (1.9s)
    TOP:  /L(0.06) | ä¹ĥ(0.06) | ¢(0.06) | .callbacks(0.06) | _STORAGE(0.05) | Recent(0.05) | .icon(0.05) | .adapter(0.05)
    BOT:  Test(-0.06) | ko(-0.06) | liers(-0.06) | bed(-0.06) | han(-0.05) | qt(-0.05) | èıĬ(-0.05) | åĩĿ(-0.05)
    ACCEPTED as axis_964  cumulative_var=0.7994

  [ 960]  axes=965  step_var=0.0012  binary_acc=0.999  gap=0.1016  max_dot=0.0015  (1.8s)
    TOP:  (segment(0.06) | -analytics(0.06) | çŃīè¡Įä¸ļ(0.06) | -L(0.05) | -provider(0.05) | Repository(0.05) | ãĤĮãĤĭ(0.05) | éķ¿(0.05)
    BOT:  \Contracts(-0.06) | æ°¸(-0.06) | æķ·(-0.06) | crets(-0.06) | egative(-0.06) | _BLOCK(-0.06) | itar(-0.06) | deprecated(-0.05)
    ACCEPTED as axis_965  cumulative_var=0.7996

  [ 961]  axes=966  step_var=0.0012  binary_acc=0.993  gap=0.1030  max_dot=0.0024  (1.8s)
    TOP:  ells(0.07) | .Type(0.06) | anford(0.06) | inition(0.06) | .AUTH(0.06) | +k(0.05) | .feature(0.05) | ();Ċ(0.05)
    BOT:  E(-0.06) | donner(-0.06) | éĴ¢(-0.06) | Panel(-0.06) | ï¸ı(-0.05) | /database(-0.05) | _points(-0.05) | RA(-0.05)
    ACCEPTED as axis_966  cumulative_var=0.7998

  [ 962]  axes=967  step_var=0.0012  binary_acc=0.993  gap=0.0995  max_dot=0.0156  (1.9s)
    TOP:  LAN(0.05) | Stable(0.05) | è¯ģä»¶(0.05) | /delete(0.05) | çļĦçľ¼çĿĽ(0.05) | heet(0.05) | sqlite(0.05) | Ð»Ð¸(0.05)
    BOT:  -x(-0.06) | à¸²à¸ģ(-0.06) | );ĊĊĊ(-0.06) | $Ċ(-0.06) | å®ĥ(-0.06) | existence(-0.06) | @(-0.06) | ä¸Ī(-0.06)
    ACCEPTED as axis_967  cumulative_var=0.8001

  [ 963]  axes=968  step_var=0.0013  binary_acc=0.985  gap=0.1010  max_dot=0.0007  (1.9s)
    TOP:  sh(0.07) | atch(0.07) | _F(0.06) | çĶŁåĳ½(0.06) | ISBN(0.06) | çĸıéĢļ(0.06) | -collapse(0.06) | identifying(0.06)
    BOT:  å¸¸è§ģ(-0.06) | ä¸ĢæĶ¯(-0.06) | ischen(-0.06) | (),(-0.06) | ierce(-0.06) | Generic(-0.05) | -description(-0.05) | _HEADERS(-0.05)
    ACCEPTED as axis_968  cumulative_var=0.8003

  [ 964]  axes=969  step_var=0.0012  binary_acc=0.994  gap=0.0987  max_dot=0.0071  (1.8s)
    TOP:  .Web(0.07) | ç¤¼(0.06) | _ct(0.06) | ä¸Ģä¸ª(0.06) | ä¸»æĮģ(0.06) | çħ¦(0.06) | æĿĲæĸĻ(0.06) | datatype(0.06)
    BOT:  .for(-0.06) | .Insert(-0.06) | ?>(-0.06) | ãĢĳĊĊ(-0.05) | ÐµÐ»Ð°(-0.05) | Square(-0.05) | ValidationError(-0.05) | ADO(-0.05)
    ACCEPTED as axis_969  cumulative_var=0.8006

  [ 965]  axes=970  step_var=0.0012  binary_acc=0.984  gap=0.1002  max_dot=0.0075  (1.8s)
    TOP:  -expand(0.07) | C(0.06) | $/(0.06) | -content(0.06) | You(0.06) | ined(0.06) | J(0.06) | æĢ§(0.06)
    BOT:  _nat(-0.06) | ï¼Į(-0.06) | .method(-0.06) | /theme(-0.05) | /release(-0.05) | /ss(-0.05) | åĿļåĽº(-0.05) | ìľ¼ë¡ľ(-0.05)
    ACCEPTED as axis_970  cumulative_var=0.8008

  [ 966]  axes=971  step_var=0.0012  binary_acc=0.981  gap=0.1005  max_dot=0.0004  (2.0s)
    TOP:  :Ċ(0.06) | _flow(0.06) | "),(0.06) | dur(0.06) | :{}(0.05) | _boxes(0.05) | å¾ĹæĦı(0.05) | _retry(0.05)
    BOT:  æĸĩåĮĸéģĹäº§(-0.07) | ãĤģ(-0.06) | rap(-0.06) | alchemy(-0.06) | longer(-0.06) | çłĶç©¶äººåĳĺ(-0.06) | itation(-0.06) | atype(-0.06)
    ACCEPTED as axis_971  cumulative_var=0.8011

  [ 967]  axes=972  step_var=0.0012  binary_acc=0.998  gap=0.0975  max_dot=0.0097  (1.9s)
    TOP:  .p(0.06) | åı·(0.06) | .l(0.06) | -from(0.06) | Ðľ(0.05) | Integration(0.05) | èĩ³ä¸Ĭ(0.05) | ç©´ä½į(0.05)
    BOT:  (blocks(-0.06) | que(-0.06) | Ã¨(-0.05) | TT(-0.05) | _content(-0.05) | mongoose(-0.05) | wax(-0.05) | /stream(-0.05)
    ACCEPTED as axis_972  cumulative_var=0.8013

  [ 968]  axes=973  step_var=0.0012  binary_acc=0.970  gap=0.0979  max_dot=0.0081  (1.8s)
    TOP:  (logits(0.07) | Thin(0.06) | .rules(0.06) | _lo(0.06) | url(0.06) | /de(0.06) | -transparent(0.06) | .color(0.05)
    BOT:  ):Ċ(-0.06) | ):čĊ(-0.06) | éĹ»(-0.06) | .search(-0.05) | Concepts(-0.05) | à¸ģ(-0.05) | åħ³ç³»(-0.05) | ä¸Ńæĸ¹(-0.05)
    ACCEPTED as axis_973  cumulative_var=0.8016

  [ 969]  axes=974  step_var=0.0012  binary_acc=0.993  gap=0.0988  max_dot=0.0013  (1.8s)
    TOP:  ide(0.06) | dirname(0.06) | _rest(0.06) | -et(0.06) | ['(0.05) | åĽ¾æ¡Ī(0.05) | _and(0.05) | ï¼ł(0.05)
    BOT:  _gradient(-0.06) | _var(-0.06) | åĩºç§Ł(-0.06) | USA(-0.05) | Facade(-0.05) | ucks(-0.05) | _fore(-0.05) | .summary(-0.05)
    ACCEPTED as axis_974  cumulative_var=0.8018

  [ 970]  axes=975  step_var=0.0013  binary_acc=0.999  gap=0.0990  max_dot=0.0004  (1.9s)
    TOP:  èħº(0.07) | çº¿(0.06) | çľģ(0.06) | upgrade(0.06) | Åĵ(0.06) | api(0.06) | Attribute(0.06) | La(0.06)
    BOT:  (count(-0.06) | .structure(-0.05) | .Enums(-0.05) | .annotate(-0.05) | -framework(-0.05) | .Max(-0.05) | åĩĨç¡®æĢ§(-0.05) | Holden(-0.05)
    ACCEPTED as axis_975  cumulative_var=0.8021

  [ 971]  axes=976  step_var=0.0012  binary_acc=0.979  gap=0.1006  max_dot=0.0081  (1.8s)
    TOP:  Ð¸Ðµ(0.06) | ,is(0.06) | disturbed(0.06) | _CURRENT(0.06) | Ð°ÑĤÑĮÑģÑı(0.06) | ervo(0.06) | our(0.06) | /g(0.06)
    BOT:  .params(-0.06) | è®¿(-0.06) | åı³ä¾§(-0.06) | ROUND(-0.06) | Ðľ(-0.06) | hold(-0.05) | _cart(-0.05) | å¼ķèµ·(-0.05)
    ACCEPTED as axis_976  cumulative_var=0.8023

  [ 972]  axes=977  step_var=0.0012  binary_acc=0.998  gap=0.0980  max_dot=0.0033  (1.9s)
    TOP:  IRTH(0.06) | (self(0.05) | ï¼īĊĊ(0.05) | END(0.05) | Md(0.05) | æľĢæĹ©çļĦ(0.05) | .recipe(0.05) | :Ċ(0.05)
    BOT:  .original(-0.06) | .presentation(-0.05) | rib(-0.05) | ÄĻ(-0.05) | Ð±Ð¾Ð»ÑĮÑĪÐµ(-0.05) | é«ĺåº¦(-0.05) | éĻĲåº¦(-0.05) | ("#(-0.05)
    ACCEPTED as axis_977  cumulative_var=0.8025

  [ 973]  axes=978  step_var=0.0012  binary_acc=0.958  gap=0.0992  max_dot=0.0082  (2.0s)
    TOP:  oci(0.07) | .img(0.06) | åı¦æľī(0.06) | èĮ¬(0.06) | /not(0.06) | settings(0.05) | ãĤīãĤĮãĤĭ(0.05) | íķĺìĦ¸ìļĶ(0.05)
    BOT:  çĶľ(-0.06) | _template(-0.06) | }((-0.06) | .imread(-0.06) | ("(-0.06) | /detail(-0.05) | åıĳ(-0.05) | cloud(-0.05)
    ACCEPTED as axis_978  cumulative_var=0.8028

  [ 974]  axes=979  step_var=0.0012  binary_acc=0.990  gap=0.0986  max_dot=0.0086  (1.9s)
    TOP:  pad(0.06) | _states(0.06) | .setAuto(0.06) | .set(0.06) | ftware(0.05) | .ones(0.05) | operators(0.05) | Pt(0.05)
    BOT:  ä¸¹(-0.07) | .Serialization(-0.06) | Plain(-0.06) | b(-0.06) | /functions(-0.06) | iang(-0.06) | äºĨä¸ª(-0.06) | à¸±à¸ļ(-0.06)
    ACCEPTED as axis_979  cumulative_var=0.8030

  [ 975]  axes=980  step_var=0.0012  binary_acc=0.986  gap=0.0979  max_dot=0.0018  (1.9s)
    TOP:  ï¼ļ(0.07) | _initializer(0.06) | /(0.06) | asis(0.06) | èįļ(0.06) | pk(0.06) | çĽĳæİ§(0.05) | .pub(0.05)
    BOT:  ãĥ³ãĥĪ(-0.06) | Ð¼Ð°(-0.06) | æĬ±(-0.06) | _cond(-0.06) | iens(-0.05) | ÑĢÐ°(-0.05) | ÐµÐ´(-0.05) | olves(-0.05)
    ACCEPTED as axis_980  cumulative_var=0.8033

  [ 976]  axes=981  step_var=0.0012  binary_acc=0.995  gap=0.0963  max_dot=0.0070  (1.8s)
    TOP:  ena(0.06) | &s(0.06) | uhan(0.06) | GPIO(0.06) | oping(0.05) | _stdio(0.05) | ulture(0.05) | _SERVICE(0.05)
    BOT:  ("(-0.07) | æĶ¾åħ¥(-0.06) | _property(-0.06) | =True(-0.06) | å±ŀäºİ(-0.06) | ä»£(-0.06) | Of(-0.06) | Popup(-0.06)
    ACCEPTED as axis_981  cumulative_var=0.8035

  [ 977]  axes=982  step_var=0.0013  binary_acc=0.977  gap=0.1005  max_dot=0.0022  (1.9s)
    TOP:  -email(0.06) | .metrics(0.06) | Ùħ(0.06) | Ã½(0.06) | documentation(0.05) | months(0.05) | è¯Ĭ(0.05) | OPS(0.05)
    BOT:  });Ċ(-0.07) | ={Ċ(-0.07) | Ċ(-0.07) | )ĊĊĊ(-0.07) | ())ĊĊ(-0.06) | åŃ¦ä¸ļ(-0.06) | Glen(-0.06) | (embed(-0.06)
    ACCEPTED as axis_982  cumulative_var=0.8038

  [ 978]  axes=983  step_var=0.0013  binary_acc=0.980  gap=0.1001  max_dot=0.0126  (1.9s)
    TOP:  å½©èī²(0.07) | utoff(0.06) | Ð½ÐµÑĤ(0.06) | tracking(0.06) | >\(0.06) | Proxy(0.05) | attrs(0.05) | .Integer(0.05)
    BOT:  IS(-0.06) | _field(-0.06) | insert(-0.05) | ç¾İä¸½(-0.05) | (-0.05) | Rule(-0.05) | âĦ¢(-0.05) | nock(-0.05)
    ACCEPTED as axis_983  cumulative_var=0.8040

  [ 979]  axes=984  step_var=0.0012  binary_acc=0.988  gap=0.1001  max_dot=0.0116  (1.9s)
    TOP:  Xi(0.06) | era(0.06) | -toggle(0.06) | çĶ±(0.06) | ãģ§(0.06) | Helper(0.06) | (config(0.06) | Queens(0.06)
    BOT:  -detail(-0.07) | .Endpoint(-0.06) | _groups(-0.06) | iggs(-0.06) | Rath(-0.06) | Cod(-0.06) | æĪĲåĬŁ(-0.06) | /software(-0.06)
    ACCEPTED as axis_984  cumulative_var=0.8043

  [ 980]  axes=985  step_var=0.0013  binary_acc=0.992  gap=0.1014  max_dot=0.0041  (1.8s)
    TOP:  .activation(0.06) | (bl(0.06) | Ðŀ(0.06) | DB(0.06) | kingdom(0.06) | /go(0.05) | useum(0.05) | (map(0.05)
    BOT:  ),Ċ(-0.07) | ifying(-0.06) | Fin(-0.05) | mult(-0.05) | of(-0.05) | çİ°(-0.05) | åį´ä¸į(-0.05) | HG(-0.05)
    ACCEPTED as axis_985  cumulative_var=0.8045

  [ 981]  axes=986  step_var=0.0012  binary_acc=0.995  gap=0.0991  max_dot=0.0064  (1.8s)
    TOP:  ÑĢÐµÐ·ÑĥÐ»ÑĮÑĤÐ°ÑĤ(0.06) | æ¬¢(0.06) | quipe(0.06) | select(0.05) | }*(0.05) | sa(0.05) | ç³ľ(0.05) | ä¸ĭ(0.05)
    BOT:  /)(-0.06) | .Search(-0.06) | IGNAL(-0.05) | /questions(-0.05) | iff(-0.05) | ===(-0.05) | :{(-0.05) | ANGUAGE(-0.05)
    ACCEPTED as axis_986  cumulative_var=0.8047

  [ 982]  axes=987  step_var=0.0012  binary_acc=0.997  gap=0.1004  max_dot=0.0129  (1.8s)
    TOP:  ledger(0.07) | aptop(0.06) | çĮ¾(0.06) | åıĹ(0.06) | >,(0.06) | andbox(0.06) | AMPLE(0.06) | bool(0.06)
    BOT:  .crypto(-0.06) | .*;Ċ(-0.06) | åĴĮè°Ĳ(-0.06) | pada(-0.06) | .cmd(-0.06) | å·¥ä½ľä¼ļè®®(-0.06) | ================================================================(-0.05) | dump(-0.05)
    ACCEPTED as axis_987  cumulative_var=0.8050

  [ 983]  axes=988  step_var=0.0012  binary_acc=0.999  gap=0.1003  max_dot=0.0021  (1.9s)
    TOP:  /cgi(0.07) | çİ°ä»»(0.06) | Interviews(0.06) | >Ċ(0.05) | }}">Ċ(0.05) | .Validation(0.05) | ãģĹãģŁ(0.05) | à¸Ń(0.05)
    BOT:  [x(-0.07) | .cum(-0.06) | k(-0.06) | etadata(-0.06) | Arn(-0.06) | Incorrect(-0.06) | colors(-0.06) | _event(-0.06)
    ACCEPTED as axis_988  cumulative_var=0.8052

  [ 984]  axes=989  step_var=0.0012  binary_acc=0.982  gap=0.0983  max_dot=0.0066  (1.9s)
    TOP:  Entropy(0.07) | (map(0.06) | Prices(0.06) | ç±»(0.06) | Fallback(0.06) | nb(0.05) | Car(0.05) | éĩıåĮĸ(0.05)
    BOT:  _created(-0.06) | Headers(-0.06) | ken(-0.06) | Golden(-0.06) | Dir(-0.06) | ï¼Ķ(-0.06) | Owner(-0.05) | /business(-0.05)
    ACCEPTED as axis_989  cumulative_var=0.8055

  [ 985]  axes=990  step_var=0.0013  binary_acc=0.996  gap=0.0984  max_dot=0.0015  (1.9s)
    TOP:  }Ċ(0.07) | HB(0.07) | .Security(0.06) | /login(0.06) | /cc(0.05) | Field(0.05) | å¦Ĥæŀľæ²¡æľī(0.05) | .append(0.05)
    BOT:  django(-0.06) | åĽĽ(-0.06) | _check(-0.06) | Github(-0.06) | /image(-0.06) | _http(-0.06) | ĉtry(-0.06) | Kim(-0.05)
    ACCEPTED as axis_990  cumulative_var=0.8057

  [ 986]  axes=991  step_var=0.0013  binary_acc=0.976  gap=0.0976  max_dot=0.0021  (1.9s)
    TOP:  .compare(0.07) | mys(0.06) | ([(0.06) | ([(0.06) | _e(0.06) | Recognition(0.05) | SEND(0.05) | .getName(0.05)
    BOT:  ileo(-0.06) | Credential(-0.06) | Ø§Ø¨(-0.06) | lo(-0.06) | Äĳáº§u(-0.06) | _out(-0.06) | Ð»ÑĮ(-0.06) | Usage(-0.06)
    ACCEPTED as axis_991  cumulative_var=0.8060

  [ 987]  axes=992  step_var=0.0012  binary_acc=0.976  gap=0.0973  max_dot=0.0158  (1.8s)
    TOP:  ä¸ĥ(0.06) | .po(0.06) | .en(0.05) | éĩĮ(0.05) | Ð¿Ð¾Ð¸ÑģÐº(0.05) | ç£ĭåķĨ(0.05) | Ð°ÐµÑĤÑģÑı(0.05) | script(0.05)
    BOT:  /j(-0.08) | )(-0.07) | è´¨(-0.06) | ,f(-0.06) | oni(-0.06) | (B(-0.06) | .stream(-0.06) | .System(-0.06)
    ACCEPTED as axis_992  cumulative_var=0.8062

  [ 988]  axes=993  step_var=0.0013  binary_acc=0.987  gap=0.0978  max_dot=0.0081  (1.8s)
    TOP:  opping(0.07) | Real(0.06) | .record(0.06) | NS(0.06) | /api(0.06) | .parameters(0.06) | ä»¥(0.06) | tl(0.06)
    BOT:  èĥ¡(-0.06) | _placement(-0.06) | /mark(-0.05) | Kosten(-0.05) | .loads(-0.05) | èĸĽ(-0.05) | ÐŃ(-0.05) | olin(-0.05)
    ACCEPTED as axis_993  cumulative_var=0.8064

  [ 989]  axes=994  step_var=0.0012  binary_acc=0.978  gap=0.0973  max_dot=0.0045  (1.8s)
    TOP:  .basename(0.06) | èģĶ(0.05) | PLICATE(0.05) | -or(0.05) | ber(0.05) | è¿ľè¿ľ(0.05) | element(0.05) | Market(0.05)
    BOT:  à¸Ĺ(-0.07) | ":(-0.06) | (((-0.06) | ']),(-0.06) | ãģĹãģ¦(-0.06) | }"(-0.06) | .GL(-0.05) | ")Ċ(-0.05)
    ACCEPTED as axis_994  cumulative_var=0.8067

  [ 990]  axes=995  step_var=0.0012  binary_acc=0.989  gap=0.0973  max_dot=0.0055  (1.8s)
    TOP:  /St(0.06) | èŀįåĲĪåıĳå±ķ(0.06) | args(0.06) | ãĥīãĥ¬ãĤ¹(0.06) | ATORS(0.06) | -decoration(0.05) | -height(0.05) | auge(0.05)
    BOT:  Che(-0.06) | _retry(-0.06) | ãģ°(-0.06) | _builder(-0.06) | '):Ċ(-0.06) | ç¦Ħ(-0.06) | ç¼º(-0.06) | ä¼ļ(-0.05)
    ACCEPTED as axis_995  cumulative_var=0.8069

  [ 991]  axes=996  step_var=0.0013  binary_acc=0.992  gap=0.0997  max_dot=0.0007  (1.8s)
    TOP:  -auto(0.07) | ä¼ļæľī(0.06) | .et(0.06) | -Line(0.06) | :ĊĊ(0.05) | Air(0.05) | /install(0.05) | å¹´åĨħ(0.05)
    BOT:  DAQ(-0.06) | /status(-0.06) | (node(-0.06) | CR(-0.06) | à¹Ħà¸Ł(-0.05) | éĽ¢(-0.05) | ç¬¨(-0.05) | +d(-0.05)
    ACCEPTED as axis_996  cumulative_var=0.8072

  [ 992]  axes=997  step_var=0.0012  binary_acc=0.989  gap=0.0969  max_dot=0.0106  (1.8s)
    TOP:  ift(0.06) | (start(0.06) | COMPLETE(0.06) | Temperature(0.06) | Mapping(0.06) | .Single(0.05) | (email(0.05) | Ð°Ð»Ð°(0.05)
    BOT:  -m(-0.07) | "",Ċ(-0.06) | =view(-0.06) | _res(-0.06) | ä¿Ĭ(-0.06) | Grid(-0.05) | Secondary(-0.05) | /build(-0.05)
    ACCEPTED as axis_997  cumulative_var=0.8074

  [ 993]  axes=998  step_var=0.0012  binary_acc=0.991  gap=0.0961  max_dot=0.0042  (1.9s)
    TOP:  _id(0.07) | s(0.06) | Of(0.06) | also(0.06) | ä¸į(0.06) | China(0.05) | any(0.05) | _seed(0.05)
    BOT:  å²Ń(-0.07) | .document(-0.06) | çļĦåŃ¦ä¹ł(-0.06) | ÑĦÐ¾ÑĢ(-0.06) | PATH(-0.06) | .Services(-0.06) | ism(-0.06) | fg(-0.05)
    ACCEPTED as axis_998  cumulative_var=0.8076

  [ 994]  axes=999  step_var=0.0012  binary_acc=0.960  gap=0.0991  max_dot=0.0038  (1.9s)
    TOP:  ç«¯æŃ£(0.06) | Properties(0.06) | nombres(0.06) | Optional(0.06) | .io(0.06) | .transfer(0.05) | .Compute(0.05) | èī°éļ¾(0.05)
    BOT:  .description(-0.06) | aub(-0.06) | _arguments(-0.06) | THE(-0.06) | ä½ľäºĨ(-0.06) | iques(-0.05) | ichen(-0.05) | ä¸Ńåħ±ä¸Ńå¤®(-0.05)
    ACCEPTED as axis_999  cumulative_var=0.8079

  [ 995]  axes=1000  step_var=0.0012  binary_acc=0.979  gap=0.0977  max_dot=0.0056  (1.8s)
    TOP:  -->Ċ(0.06) | ')])Ċ(0.06) | ¹(0.06) | Ð¸ÑĤÑģÑı(0.05) | service(0.05) | img(0.05) | .roles(0.05) | eye(0.05)
    BOT:  (D(-0.06) | (--(-0.06) | ÑĦÐ¾ÑĢÐ¼Ñĥ(-0.05) | .Player(-0.05) | /am(-0.05) | .cleanup(-0.05) | n(-0.05) | Incorrect(-0.05)
    ACCEPTED as axis_1000  cumulative_var=0.8081

  [ 996]  axes=1001  step_var=0.0013  binary_acc=0.985  gap=0.0983  max_dot=0.0018  (1.8s)
    TOP:  .api(0.06) | è´¼(0.06) | block(0.05) | /auth(0.05) | article(0.05) | >",(0.05) | Rotation(0.05) | èĳĹåĲįçļĦ(0.05)
    BOT:  ç·ļ(-0.05) | .enterprise(-0.05) | íķľ(-0.05) | .resources(-0.05) | satisfies(-0.05) | æ¬£èµı(-0.05) | --------(-0.05) | ä»¥(-0.05)
    ACCEPTED as axis_1001  cumulative_var=0.8084

  [ 997]  axes=1002  step_var=0.0013  binary_acc=0.999  gap=0.0965  max_dot=0.0019  (1.9s)
    TOP:  /link(0.07) | _mail(0.06) | å®ŀçİ°äºĨ(0.06) | called(0.05) | safely(0.05) | å·¥ä½ľä½ľé£İ(0.05) | çļĦæľįåĬ¡(0.05) | neh(0.05)
    BOT:  .App(-0.06) | à¹Ģà¸ģ(-0.06) | Formats(-0.06) | By(-0.05) | Ð½Ðµ(-0.05) | nez(-0.05) | .term(-0.05) | (My(-0.05)
    ACCEPTED as axis_1002  cumulative_var=0.8086

  [ 998]  axes=1003  step_var=0.0013  binary_acc=0.983  gap=0.0968  max_dot=0.0090  (1.8s)
    TOP:  itch(0.06) | (0.06) | æĭħ(0.06) | asics(0.06) | Abs(0.06) | .project(0.06) | N(0.05) | finite(0.05)
    BOT:  ä¸Ĭè¿°(-0.07) | åıĳåĬ¨æľº(-0.07) | Â«(-0.06) | å¤ľæĻļ(-0.06) | about(-0.06) | .de(-0.06) | =true(-0.06) | /render(-0.06)
    ACCEPTED as axis_1003  cumulative_var=0.8088

  [ 999]  axes=1004  step_var=0.0013  binary_acc=0.979  gap=0.0991  max_dot=0.0115  (1.8s)
    TOP:  æĪ¿(0.06) | _frame(0.06) | ÐµÐ½Ð°(0.06) | iliated(0.06) | ä¸ĭ(0.06) | antics(0.06) | .request(0.05) | irmingham(0.05)
    BOT:  _:(-0.06) | Sold(-0.06) | E(-0.06) | Summary(-0.06) | _dtype(-0.06) | ';ĊĊ(-0.05) | Pages(-0.05) | .tech(-0.05)
    ACCEPTED as axis_1004  cumulative_var=0.8091

  [1000]  axes=1005  step_var=0.0013  binary_acc=0.994  gap=0.0985  max_dot=0.0175  (1.8s)
    TOP:  ä¸įä¸Ĭ(0.06) | æĶ¶(0.06) | -core(0.06) | èĥĥ(0.06) | =========Ċ(0.06) | closest(0.06) | ç´ł(0.06) | è¯´æĺİ(0.05)
    BOT:  =c(-0.06) | /embed(-0.06) | UBLISH(-0.05) | _CPP(-0.05) | æĪ¿äº§(-0.05) | .users(-0.05) | Oz(-0.05) | _parameters(-0.05)
    ACCEPTED as axis_1005  cumulative_var=0.8093

  [1001]  axes=1006  step_var=0.0013  binary_acc=0.995  gap=0.0984  max_dot=0.0181  (1.8s)
    TOP:  /en(0.06) | steroid(0.05) | Ĥ¹(0.05) | GitHub(0.05) | ï¼īï¼Į(0.05) | _notes(0.05) | Ð¸(0.05) | _review(0.05)
    BOT:  .level(-0.07) | çĻ»(-0.06) | åºĶæĶ¶(-0.06) | æī¶(-0.06) | æĶ¿åºľ(-0.06) | -US(-0.06) | put(-0.06) | .dd(-0.06)
    ACCEPTED as axis_1006  cumulative_var=0.8096

  [1002]  axes=1007  step_var=0.0013  binary_acc=0.961  gap=0.0961  max_dot=0.0022  (1.9s)
    TOP:  User(0.07) | Thu(0.06) | Order(0.06) | _AL(0.05) | .set(0.05) | è¯ģåĪ¸(0.05) | è®Ńç»ĥ(0.05) | URES(0.05)
    BOT:  ãĤ¸(-0.06) | _config(-0.06) | ÑĢÐµÐ¼(-0.06) | -app(-0.06) | .cluster(-0.06) | ity(-0.05) | "čĊ(-0.05) | .size(-0.05)
    ACCEPTED as axis_1007  cumulative_var=0.8098

  [1003]  axes=1008  step_var=0.0012  binary_acc=0.990  gap=0.0957  max_dot=0.0035  (1.8s)
    TOP:  _distribution(0.06) | Ð¾Ð´(0.06) | Pooling(0.06) | "="(0.06) | Insurance(0.05) | ubb(0.05) | _LIST(0.05) | å¼Ģ(0.05)
    BOT:  ç»ĦæĪĲçļĦ(-0.06) | æĭ³(-0.06) | max(-0.06) | Little(-0.05) | unpack(-0.05) | zur(-0.05) | Distribution(-0.05) | *e(-0.05)
    ACCEPTED as axis_1008  cumulative_var=0.8100

  [1004]  axes=1009  step_var=0.0013  binary_acc=0.973  gap=0.0999  max_dot=0.0038  (2.0s)
    TOP:  Reduction(0.06) | gate(0.06) | _inter(0.06) | (font(0.06) | Â»ĊĊ(0.06) | Institute(0.06) | æįŁ(0.06) | ç¨ĭ(0.05)
    BOT:  ork(-0.07) | .poll(-0.06) | clock(-0.06) | Objects(-0.06) | elastic(-0.05) | _LEFT(-0.05) | izon(-0.05) | -video(-0.05)
    ACCEPTED as axis_1009  cumulative_var=0.8103

  [1005]  axes=1010  step_var=0.0012  binary_acc=0.983  gap=0.0966  max_dot=0.0026  (1.9s)
    TOP:  èĥĢ(0.06) | Session(0.06) | ÑĩÐ¸Ð²(0.06) | Font(0.06) | enchmark(0.05) | .storage(0.05) | _alpha(0.05) | .Show(0.05)
    BOT:  ]$(-0.07) | =['(-0.07) | ach(-0.06) | _Create(-0.06) | <div(-0.06) | ://(-0.06) | _plugin(-0.06) | imations(-0.06)
    ACCEPTED as axis_1010  cumulative_var=0.8105

  [1006]  axes=1011  step_var=0.0013  binary_acc=0.970  gap=0.0979  max_dot=0.0071  (1.8s)
    TOP:  ABLE(0.07) | Conditional(0.06) | _exception(0.06) | -screen(0.06) | ÑıÐ¼Ð¸(0.06) | _st(0.06) | ient(0.06) | .A(0.06)
    BOT:  Week(-0.06) | _BODY(-0.06) | .analysis(-0.06) | è¶Ĭé«ĺ(-0.06) | ÐŁÐ¾(-0.06) | Creative(-0.05) | aprÃ¨s(-0.05) | heard(-0.05)
    ACCEPTED as axis_1011  cumulative_var=0.8108

  [1007]  axes=1012  step_var=0.0013  binary_acc=0.962  gap=0.0974  max_dot=0.0032  (1.8s)
    TOP:  .OS(0.06) | .Test(0.05) | Ð»(0.05) | åħ¬åĳĬ(0.05) | ä¾ĭè¡Į(0.05) | Handlers(0.05) | rationale(0.05) | alert(0.05)
    BOT:  (cc(-0.06) | LANG(-0.06) | .js(-0.06) | é½Ĳ(-0.05) | VV(-0.05) | evening(-0.05) | _options(-0.05) | _OPTIONS(-0.05)
    ACCEPTED as axis_1012  cumulative_var=0.8110

  [1008]  axes=1013  step_var=0.0013  binary_acc=0.978  gap=0.0969  max_dot=0.0060  (1.8s)
    TOP:  _basic(0.06) | /K(0.06) | ì²ĺë¦¬(0.06) | _comb(0.06) | çĽĽ(0.06) | ipro(0.05) | WHETHER(0.05) | äºĽä»Ģä¹Ī(0.05)
    BOT:  Ð¾(-0.06) | æĺĮ(-0.06) | Ð°ÑĨÐ¸Ñı(-0.06) | from(-0.06) | gabe(-0.06) | .jpg(-0.06) | IN(-0.05) | ices(-0.05)
    ACCEPTED as axis_1013  cumulative_var=0.8112

  [1009]  axes=1014  step_var=0.0013  binary_acc=0.990  gap=0.0989  max_dot=0.0077  (2.0s)
    TOP:  Browse(0.05) | _BLOCK(0.05) | .sex(0.05) | Pass(0.05) | éĻ¤äºĨ(0.05) | SED(0.05) | .Name(0.05) | ibs(0.05)
    BOT:  {Ċ(-0.07) | (((-0.06) | {čĊ(-0.06) | "čĊ(-0.06) | */Ċ(-0.06) | """čĊ(-0.06) | Elementary(-0.06) | .project(-0.06)
    ACCEPTED as axis_1014  cumulative_var=0.8115

  [1010]  axes=1015  step_var=0.0013  binary_acc=0.980  gap=0.0970  max_dot=0.0020  (1.8s)
    TOP:  çĶ¨(0.06) | docker(0.06) | Aboriginal(0.06) | .analysis(0.06) | .office(0.06) | èĤ²äºº(0.05) | expand(0.05) | Le(0.05)
    BOT:  ylum(-0.06) | .sha(-0.06) | Commercial(-0.06) | é©¬(-0.05) | ---------------(-0.05) | Forest(-0.05) | Comput(-0.05) | _LOG(-0.05)
    ACCEPTED as axis_1015  cumulative_var=0.8117

  [1011]  axes=1016  step_var=0.0013  binary_acc=0.999  gap=0.0957  max_dot=0.0047  (1.9s)
    TOP:  for(0.07) | For(0.06) | professional(0.06) | 'll(0.06) | åħ¨éĿ¢åıĳå±ķ(0.06) | (/(0.05) | Ke(0.05) | Jan(0.05)
    BOT:  Node(-0.07) | åĹŁ(-0.06) | .axes(-0.06) | ä½ĵçİ°(-0.06) | éĩĩçĶ¨(-0.05) | pprint(-0.05) | elenium(-0.05) | <pre(-0.05)
    ACCEPTED as axis_1016  cumulative_var=0.8120

  [1012]  axes=1017  step_var=0.0013  binary_acc=0.992  gap=0.0988  max_dot=0.0083  (1.9s)
    TOP:  phas(0.07) | Ð»(0.07) | _trace(0.06) | ':(0.06) | åıĳ(0.06) | MSG(0.06) | _CENTER(0.06) | t(0.06)
    BOT:  ining(-0.06) | Ð¼Ð¸(-0.06) | /init(-0.06) | yn(-0.06) | .upload(-0.06) | putation(-0.05) | ender(-0.05) | dashboard(-0.05)
    ACCEPTED as axis_1017  cumulative_var=0.8122

  [1013]  axes=1018  step_var=0.0013  binary_acc=0.984  gap=0.0970  max_dot=0.0084  (1.8s)
    TOP:  Guarantee(0.06) | udo(0.05) | (post(0.05) | (one(0.05) | éĿ¢(0.05) | ìĿĢ(0.05) | è¥¿ä¾§(0.05) | between(0.05)
    BOT:  åħ´ä¸ļ(-0.06) | ication(-0.06) | è®¤å®ļ(-0.05) | ULATE(-0.05) | ARS(-0.05) | "<(-0.05) | ';Ċ(-0.05) | Anchor(-0.05)
    ACCEPTED as axis_1018  cumulative_var=0.8124

  [1014]  axes=1019  step_var=0.0013  binary_acc=0.971  gap=0.0980  max_dot=0.0079  (1.9s)
    TOP:  architects(0.06) | .Admin(0.06) | .port(0.05) | _INPUT(0.05) | AMPLE(0.05) | /features(0.05) | ]+(0.05) | Scrap(0.05)
    BOT:  ([[(-0.06) | k(-0.06) | air(-0.06) | /P(-0.06) | ÑıÐ²Ð»ÑıÐµÑĤÑģÑı(-0.06) | è¯¬(-0.06) | éļı(-0.05) | /*(-0.05)
    ACCEPTED as axis_1019  cumulative_var=0.8127

  [1015]  axes=1020  step_var=0.0013  binary_acc=0.969  gap=0.0972  max_dot=0.0007  (1.9s)
    TOP:  å¦Ĥ(0.06) | ;(0.06) | -the(0.05) | .name(0.05) | extra(0.05) | ;</(0.05) | .prod(0.05) | justify(0.05)
    BOT:  't(-0.07) | ](-0.06) | SF(-0.05) | 'Ã©(-0.05) | statuses(-0.05) | èĤīç±»(-0.05) | +t(-0.05) | Fuck(-0.05)
    ACCEPTED as axis_1020  cumulative_var=0.8129

  [1016]  axes=1021  step_var=0.0013  binary_acc=0.971  gap=0.0997  max_dot=0.0080  (1.8s)
    TOP:  _segment(0.06) | (obj(0.05) | .token(0.05) | USTER(0.05) | /x(0.05) | .encrypt(0.05) | .created(0.05) | åĽĽåŃ£(0.05)
    BOT:  Trigger(-0.07) | setting(-0.05) | å¤ª(-0.05) | .Open(-0.05) | .sql(-0.05) | instr(-0.05) | .Make(-0.05) | é¤Ĭ(-0.05)
    ACCEPTED as axis_1021  cumulative_var=0.8131

  [1017]  axes=1022  step_var=0.0013  binary_acc=0.993  gap=0.0958  max_dot=0.0070  (1.9s)
    TOP:  å¸Ŀçİĭ(0.07) | æ³¢(0.07) | .datetime(0.06) | .define(0.06) | åĲįä¹ī(0.06) | _AV(0.06) | Brandon(0.06) | Retry(0.06)
    BOT:  âĢĿĊ(-0.06) | ())Ċ(-0.06) | _voice(-0.06) | hope(-0.06) | ';Ċ(-0.06) | ~/(-0.06) | ãĢĳĊĊ(-0.06) | ")ĊĊ(-0.06)
    ACCEPTED as axis_1022  cumulative_var=0.8134

  [1018]  axes=1023  step_var=0.0013  binary_acc=0.982  gap=0.0957  max_dot=0.0160  (1.8s)
    TOP:  .Module(0.06) | åŃĺæ¬¾(0.06) | Ã¤(0.06) | Ðº(0.06) | _issues(0.05) | generator(0.05) | sex(0.05) | æĬĬæı¡(0.05)
    BOT:  chemas(-0.07) | _t(-0.07) | ->(-0.06) | Hidden(-0.06) | âĢĿçļĦ(-0.05) | æ¶Ī(-0.05) | API(-0.05) | Dialog(-0.05)
    ACCEPTED as axis_1023  cumulative_var=0.8136

  [1019]  axes=1024  step_var=0.0013  binary_acc=0.985  gap=0.0973  max_dot=0.0020  (1.8s)
    TOP:  ouse(0.06) | fa(0.06) | cken(0.06) | ikt(0.06) | ç¤¾(0.06) | æ»©(0.05) | ovies(0.05) | valid(0.05)
    BOT:  æľīéĻĲåħ¬åı¸(-0.07) | Struct(-0.06) | .so(-0.06) | .Mapper(-0.06) | .this(-0.06) | _SER(-0.05) | Rest(-0.05) | _packet(-0.05)
    ACCEPTED as axis_1024  cumulative_var=0.8139

  [1020]  axes=1025  step_var=0.0013  binary_acc=0.976  gap=0.0957  max_dot=0.0091  (1.8s)
    TOP:  é¡¿æĹ¶(0.05) | emp(0.05) | ç¬¬(0.05) | åį«çĶŁ(0.05) | affiliate(0.05) | needed(0.05) | PAD(0.05) | åľ°å¤Ħ(0.05)
    BOT:  ext(-0.06) | <ul(-0.06) | irmware(-0.06) | classifier(-0.06) | _H(-0.06) | Window(-0.06) | af(-0.06) | .n(-0.06)
    ACCEPTED as axis_1025  cumulative_var=0.8141

  [1021]  axes=1026  step_var=0.0013  binary_acc=0.990  gap=0.0958  max_dot=0.0070  (1.9s)
    TOP:  <Object(0.06) | -black(0.06) | ::=(0.06) | å¿«(0.06) | \Request(0.06) | =pk(0.05) | /man(0.05) | .Material(0.05)
    BOT:  OF(-0.06) | vy(-0.06) | ME(-0.05) | [::-(-0.05) | orden(-0.05) | .git(-0.05) | ÑĥÐ´(-0.05) | à¸£(-0.05)
    ACCEPTED as axis_1026  cumulative_var=0.8143

  [1022]  axes=1027  step_var=0.0013  binary_acc=0.995  gap=0.0972  max_dot=0.0034  (1.8s)
    TOP:  _components(0.07) | éĺ¿(0.06) | [(0.06) | æĹłè®ºæĺ¯(0.06) | (inputs(0.06) | ä¸ºä¸»ä½ĵ(0.06) | _quantity(0.06) | PARTMENT(0.05)
    BOT:  anche(-0.06) | Les(-0.06) | DIV(-0.05) | anged(-0.05) | Demand(-0.05) | Money(-0.05) | Diff(-0.05) | .,Ċ(-0.05)
    ACCEPTED as axis_1027  cumulative_var=0.8146

  [1023]  axes=1028  step_var=0.0013  binary_acc=0.984  gap=0.0963  max_dot=0.0060  (1.9s)
    TOP:  it(0.06) | ÙĤ(0.06) | çĶŁèĤ²(0.06) | Tables(0.06) | pwd(0.06) | .Node(0.06) | (cost(0.06) | _range(0.05)
    BOT:  %ĊĊ(-0.06) | ĉpublic(-0.06) | ==Ċ(-0.06) | []Ċ(-0.06) | ')])Ċ(-0.06) | projects(-0.06) | crease(-0.06) | %Ċ(-0.06)
    ACCEPTED as axis_1028  cumulative_var=0.8148

  [1024]  axes=1029  step_var=0.0013  binary_acc=0.984  gap=0.0962  max_dot=0.0033  (1.8s)
    TOP:  .Base(0.06) | .second(0.06) | Condition(0.06) | Joined(0.05) | .sendMessage(0.05) | æľ¯(0.05) | _np(0.05) | .route(0.05)
    BOT:  Ċ(-0.05) | Lion(-0.05) | Vk(-0.05) | ,w(-0.05) | Bus(-0.05) | `,Ċ(-0.05) | /graphql(-0.05) | ':(-0.05)
    ACCEPTED as axis_1029  cumulative_var=0.8150

  [1025]  axes=1030  step_var=0.0013  binary_acc=0.986  gap=0.0966  max_dot=0.0009  (1.8s)
    TOP:  _paths(0.06) | multiply(0.06) | .getAttribute(0.06) | ä¼ĺè´¨(0.06) | TITLE(0.05) | Ð¿Ð¾ÐºÐ°(0.05) | .Assert(0.05) | åĪĹ(0.05)
    BOT:  Â».(-0.07) | /filter(-0.06) | message(-0.06) | Illegal(-0.06) | inform(-0.06) | Usage(-0.06) | nel(-0.06) | º(-0.06)
    ACCEPTED as axis_1030  cumulative_var=0.8153

  [1026]  axes=1031  step_var=0.0013  binary_acc=0.990  gap=0.0973  max_dot=0.0061  (1.9s)
    TOP:  _flag(0.06) | .Expression(0.06) | .c(0.06) | _same(0.06) | äºİ(0.06) | .form(0.06) | å·¥èµĦ(0.05) | äº«åıĹ(0.05)
    BOT:  Name(-0.06) | .program(-0.06) | Attempt(-0.06) | dev(-0.06) | Adds(-0.06) | .environment(-0.06) | <User(-0.06) | Ð´Ð°(-0.06)
    ACCEPTED as axis_1031  cumulative_var=0.8155

  [1027]  axes=1032  step_var=0.0013  binary_acc=0.995  gap=0.0953  max_dot=0.0023  (1.8s)
    TOP:  Dispose(0.06) | Constructor(0.06) | itim(0.06) | _fill(0.06) | Vocabulary(0.06) | iq(0.05) | ced(0.05) | cho(0.05)
    BOT:  )})(-0.06) | Ð°ÑģÑģ(-0.05) | æ¸Ķä¸ļ(-0.05) | >();ĊĊ(-0.05) | .Handle(-0.05) | l(-0.05) | SO(-0.05) | qu(-0.05)
    ACCEPTED as axis_1032  cumulative_var=0.8157

  [1028]  axes=1033  step_var=0.0013  binary_acc=0.987  gap=0.0960  max_dot=0.0035  (2.0s)
    TOP:  dev(0.06) | ({(0.06) | disk(0.06) | opa(0.06) | /simple(0.06) | .rnn(0.06) | .te(0.06) | _media(0.06)
    BOT:  æĪĲåĳĺåįķä½į(-0.06) | }>Ċ(-0.06) | =username(-0.05) | _movies(-0.05) | æ´ĭæ´ĭ(-0.05) | .cn(-0.05) | Dependency(-0.05) | grabbed(-0.05)
    ACCEPTED as axis_1033  cumulative_var=0.8160

  [1029]  axes=1034  step_var=0.0013  binary_acc=0.998  gap=0.0955  max_dot=0.0074  (1.9s)
    TOP:  dee(0.06) | bservice(0.06) | or(0.05) | cot(0.05) | pdf(0.05) | Ð²Ð°(0.05) | å°Ķ(0.05) | psilon(0.05)
    BOT:  _flow(-0.06) | OS(-0.06) | ÑĥÐ±(-0.06) | .draw(-0.06) | MX(-0.06) | {((-0.06) | "ĊĊ(-0.06) | unist(-0.06)
    ACCEPTED as axis_1034  cumulative_var=0.8162

  [1030]  axes=1035  step_var=0.0013  binary_acc=0.996  gap=0.0992  max_dot=0.0023  (1.9s)
    TOP:  å¤ĸå¥Ĺ(0.06) | .Load(0.06) | (NULL(0.05) | ENGINE(0.05) | Doctors(0.05) | éĽ¾(0.05) | GNU(0.05) | Molecular(0.05)
    BOT:  ypes(-0.06) | .standard(-0.06) | .async(-0.06) | doc(-0.06) | ational(-0.06) | logout(-0.06) | B(-0.06) | ëĶĶ(-0.06)
    ACCEPTED as axis_1035  cumulative_var=0.8165

  [1031]  axes=1036  step_var=0.0013  binary_acc=0.999  gap=0.0961  max_dot=0.0004  (1.9s)
    TOP:  Ń(0.08) | Ð¾Ñĩ(0.06) | =max(0.06) | =[],(0.06) | .nl(0.06) | slt(0.06) | ({'(0.05) | ëħĦ(0.05)
    BOT:  .fc(-0.06) | Family(-0.06) | optimized(-0.06) | åīįå¤ķ(-0.05) | éĺ¿æĭīä¼¯(-0.05) | .register(-0.05) | ç¡¬åĮĸ(-0.05) | èº«æĿĲ(-0.05)
    ACCEPTED as axis_1036  cumulative_var=0.8167

  [1032]  axes=1037  step_var=0.0013  binary_acc=0.976  gap=0.0969  max_dot=0.0098  (1.8s)
    TOP:  èıľ(0.06) | å¤ĸéĿ¢(0.06) | éª¨(0.06) | åĩĢåĢ¼(0.06) | .label(0.06) | _lane(0.05) | Ret(0.05) | .one(0.05)
    BOT:  Geometry(-0.06) | getDefault(-0.06) | è¯ĳ(-0.06) | ç§į(-0.06) | Assets(-0.05) | berry(-0.05) | _WARNING(-0.05) | ãĥ¼(-0.05)
    ACCEPTED as axis_1037  cumulative_var=0.8169

  [1033]  axes=1038  step_var=0.0013  binary_acc=0.971  gap=0.0980  max_dot=0.0033  (1.8s)
    TOP:  ä¸Ń(0.06) | .hardware(0.06) | .external(0.06) | (images(0.05) | /aws(0.05) | .Source(0.05) | ITU(0.05) | .Address(0.05)
    BOT:  gresql(-0.06) | ogs(-0.06) | å¦Ĥæŀľä½ł(-0.06) | ASE(-0.06) | ãĢĤâĢĿ(-0.06) | work(-0.06) | äº¤æµģ(-0.06) | Movie(-0.06)
    ACCEPTED as axis_1038  cumulative_var=0.8172

  [1034]  axes=1039  step_var=0.0013  binary_acc=0.997  gap=0.0964  max_dot=0.0093  (1.8s)
    TOP:  .tasks(0.06) | Initialize(0.06) | Wag(0.06) | è§Ħç¨ĭ(0.06) | ources(0.06) | Joined(0.06) | _usage(0.06) | :')Ċ(0.05)
    BOT:  M(-0.05) | DIN(-0.05) | /Math(-0.05) | .Contracts(-0.05) | Diff(-0.05) | ES(-0.05) | Tournament(-0.05) | éĢŁåº¦(-0.05)
    ACCEPTED as axis_1039  cumulative_var=0.8174

  [1035]  axes=1040  step_var=0.0013  binary_acc=0.989  gap=0.0965  max_dot=0.0079  (1.8s)
    TOP:  ton(0.06) | p(0.06) | .strip(0.06) | Builder(0.06) | Interfaces(0.06) | Ð½Ð¾ÑģÑĤÑĮÑİ(0.06) | .contract(0.06) | .plugins(0.05)
    BOT:  /doc(-0.07) | .pb(-0.06) | èĥİ(-0.06) | Excel(-0.05) | æĬĦ(-0.05) | UT(-0.05) | .react(-0.05) | æ·»åĬł(-0.05)
    ACCEPTED as axis_1040  cumulative_var=0.8176

  [1036]  axes=1041  step_var=0.0013  binary_acc=0.999  gap=0.0948  max_dot=0.0013  (1.9s)
    TOP:  æŃ£åĵģ(0.06) | Col(0.06) | è´Ł(0.06) | Ð½Ð°(0.06) | Ð¸ÑģÑĤÐ¾ÑĢÐ¸Ð¸(0.06) | .CONTENT(0.06) | uid(0.06) | py(0.06)
    BOT:  ãĥ¼ãĤ(-0.06) | MER(-0.06) | down(-0.06) | (lang(-0.06) | ec(-0.06) | -md(-0.06) | (batch(-0.06) | .pl(-0.05)
    ACCEPTED as axis_1041  cumulative_var=0.8179

  [1037]  axes=1042  step_var=0.0013  binary_acc=0.988  gap=0.0971  max_dot=0.0024  (1.8s)
    TOP:  acao(0.06) | ught(0.06) | ands(0.06) | Ð´Ð°Ð½Ð½ÑĭÑħ(0.06) | able(0.06) | /gen(0.06) | VIEW(0.06) | MENT(0.06)
    BOT:  [Ċ(-0.06) | /weather(-0.06) | åĨ¶éĩĳ(-0.05) | -----Ċ(-0.05) | Receive(-0.05) | (str(-0.05) | ä½ľå®¶(-0.05) | Repair(-0.05)
    ACCEPTED as axis_1042  cumulative_var=0.8181

  [1038]  axes=1043  step_var=0.0013  binary_acc=0.993  gap=0.0948  max_dot=0.0069  (1.8s)
    TOP:  Toolkit(0.06) | æĿĲæĸĻ(0.06) | Height(0.05) | äº¤(0.05) | å¸ĿåĽ½(0.05) | ANG(0.05) | \D(0.05) | -source(0.05)
    BOT:  be(-0.08) | v(-0.07) | THE(-0.06) | Anne(-0.06) | .Path(-0.06) | Equip(-0.06) | .cs(-0.06) | Ñģ(-0.06)
    ACCEPTED as axis_1043  cumulative_var=0.8183

  [1039]  axes=1044  step_var=0.0013  binary_acc=0.996  gap=0.0966  max_dot=0.0025  (1.9s)
    TOP:  Blog(0.06) | Next(0.06) | ìĭľ(0.05) | .getProperty(0.05) | Discrim(0.05) | ãĤģ(0.05) | '))ĊĊ(0.05) | Ter(0.05)
    BOT:  _EMAIL(-0.06) | ä¹³æĪ¿(-0.06) | Waiting(-0.06) | ç¥ŀç»ı(-0.06) | details(-0.05) | .numpy(-0.05) | Network(-0.05) | /big(-0.05)
    ACCEPTED as axis_1044  cumulative_var=0.8185

  [1040]  axes=1045  step_var=0.0013  binary_acc=0.979  gap=0.0968  max_dot=0.0044  (1.9s)
    TOP:  Type(0.07) | */Ċ(0.06) | åĬ¨ä½ľ(0.06) | Ð¸Ð½(0.06) | æī¿åıĹ(0.06) | dm(0.06) | ÙĪØª(0.06) | åı¯åĪĨä¸º(0.06)
    BOT:  ":"(-0.06) | ()"Ċ(-0.05) | ians(-0.05) | multiplying(-0.05) | ('=(-0.05) | -icon(-0.05) | -text(-0.05) | ic(-0.05)
    ACCEPTED as axis_1045  cumulative_var=0.8188

  [1041]  axes=1046  step_var=0.0013  binary_acc=0.975  gap=0.0970  max_dot=0.0011  (1.8s)
    TOP:  Press(0.07) | irim(0.07) | ank(0.06) | B(0.06) | ptide(0.06) | uin(0.05) | on(0.05) | .sl(0.05)
    BOT:  Flow(-0.06) | """ĊĊ(-0.06) | Find(-0.06) | ãģĹãģ¦ãģĦãģŁ(-0.06) | Accessor(-0.05) | Ordinal(-0.05) | Prev(-0.05) | åı·æ¥¼(-0.05)
    ACCEPTED as axis_1046  cumulative_var=0.8190

  [1042]  axes=1047  step_var=0.0013  binary_acc=0.988  gap=0.0955  max_dot=0.0037  (1.9s)
    TOP:  .files(0.06) | JAVA(0.05) | >"Ċ(0.05) | /topics(0.05) | LOGIN(0.05) | Logical(0.05) | ,np(0.05) | RK(0.05)
    BOT:  /B(-0.07) | oco(-0.07) | =config(-0.06) | èº«(-0.06) | .exec(-0.06) | DSA(-0.06) | æĦ¿(-0.06) | ŀ(-0.06)
    ACCEPTED as axis_1047  cumulative_var=0.8192

  [1043]  axes=1048  step_var=0.0013  binary_acc=0.971  gap=0.0978  max_dot=0.0108  (2.0s)
    TOP:  åŃĲå¼Ł(0.06) | .Client(0.06) | txt(0.05) | ("../(0.05) | ===============Ċ(0.05) | .Builder(0.05) | ç»§æī¿(0.05) | ']);ĊĊ(0.05)
    BOT:  (K(-0.05) | å½ĴæĿ¥(-0.05) | .functions(-0.05) | .statusCode(-0.05) | ãĤ·ãĥ£(-0.05) | è¯Ŀ(-0.05) | _PIN(-0.05) | _P(-0.05)
    ACCEPTED as axis_1048  cumulative_var=0.8195

  [1044]  axes=1049  step_var=0.0013  binary_acc=0.975  gap=0.0966  max_dot=0.0090  (1.9s)
    TOP:  [column(0.07) | .F(0.06) | "--(0.06) | Unit(0.06) | Token(0.05) | Ð½Ð¸Ñħ(0.05) | package(0.05) | Send(0.05)
    BOT:  ÑĤÐµÐ»Ð¸(-0.05) | igation(-0.05) | .binary(-0.05) | apis(-0.05) | ä¹ĥ(-0.05) | /styles(-0.05) | ads(-0.05) | -init(-0.05)
    ACCEPTED as axis_1049  cumulative_var=0.8197

  [1045]  axes=1050  step_var=0.0013  binary_acc=0.992  gap=0.0968  max_dot=0.0036  (1.8s)
    TOP:  ok(0.07) | -set(0.07) | c(0.06) | then(0.06) | work(0.06) | kn(0.06) | ä¸ªæľĪ(0.06) | .charset(0.06)
    BOT:  [],Ċ(-0.05) | åħ¨éĿ¢æİ¨è¿Ľ(-0.05) | æĮģç»Ńæİ¨è¿Ľ(-0.05) | .scheduler(-0.05) | .clear(-0.05) | è´£ä»»æĦŁ(-0.05) | Quá»ĳc(-0.05) | phans(-0.05)
    ACCEPTED as axis_1050  cumulative_var=0.8199

  [1046]  axes=1051  step_var=0.0013  binary_acc=0.992  gap=0.0970  max_dot=0.0042  (1.8s)
    TOP:  .word(0.06) | Ðº(0.06) | ç§ĳæĬĢå¤§åŃ¦(0.06) | çľĭ(0.06) | ={Ċ(0.06) | BÃ¬nh(0.06) | ctr(0.06) | /general(0.06)
    BOT:  .message(-0.06) | _MODULES(-0.05) | .s(-0.05) | expanded(-0.05) | Tags(-0.05) | æ²¡è§ģè¿ĩ(-0.05) | _extractor(-0.05) | .button(-0.05)
    ACCEPTED as axis_1051  cumulative_var=0.8202

  [1047]  axes=1052  step_var=0.0013  binary_acc=0.979  gap=0.0942  max_dot=0.0170  (1.9s)
    TOP:  .lib(0.06) | HOST(0.06) | compares(0.06) | obre(0.06) | ftp(0.05) | uba(0.05) | '%(0.05) | å£«(0.05)
    BOT:  _modules(-0.06) | ãĤº(-0.06) | Delete(-0.06) | import(-0.06) | in(-0.05) | .Number(-0.05) | åŃ¦å®¶(-0.05) | /os(-0.05)
    ACCEPTED as axis_1052  cumulative_var=0.8204

  [1048]  axes=1053  step_var=0.0013  binary_acc=0.984  gap=0.0959  max_dot=0.0037  (1.8s)
    TOP:  _ACCESS(0.07) | çī¹äº§(0.06) | take(0.06) | ç¾İå¥½çļĦ(0.06) | sole(0.05) | .nickname(0.05) | "?(0.05) | ])*(0.05)
    BOT:  Delayed(-0.06) | .Queue(-0.06) | implify(-0.06) | è¡ĮæĶ¿å¤Ħç½ļ(-0.06) | iteral(-0.06) | day(-0.05) | é¢ĺ(-0.05) | At(-0.05)
    ACCEPTED as axis_1053  cumulative_var=0.8206

  [1049]  axes=1054  step_var=0.0013  binary_acc=0.988  gap=0.0957  max_dot=0.0050  (1.9s)
    TOP:  '):Ċ(0.07) | .")Ċ(0.06) | åĩŃ(0.06) | åįĵè¶Ĭ(0.06) | ç»µ(0.06) | _manager(0.06) | ile(0.06) | /shared(0.06)
    BOT:  Align(-0.06) | =item(-0.06) | char(-0.05) | .FAIL(-0.05) | alytics(-0.05) | Sheet(-0.05) | Finding(-0.05) | unicode(-0.05)
    ACCEPTED as axis_1054  cumulative_var=0.8209

  [1050]  axes=1055  step_var=0.0013  binary_acc=0.995  gap=0.0981  max_dot=0.0106  (1.9s)
    TOP:  innerHTML(0.06) | .Home(0.05) | ffective(0.05) | .Window(0.05) | localhost(0.05) | è°¢è°¢(0.05) | User(0.05) | Derived(0.05)
    BOT:  æºª(-0.07) | (Ċ(-0.06) | å¯Ĩ(-0.06) | è¯į(-0.06) | (pair(-0.06) | (T(-0.06) | )ĊĊ(-0.06) | (čĊ(-0.06)
    ACCEPTED as axis_1055  cumulative_var=0.8211

  [1051]  axes=1056  step_var=0.0013  binary_acc=0.985  gap=0.0981  max_dot=0.0134  (1.8s)
    TOP:  /D(0.06) | è¿ĳå¹´(0.06) | (_(0.06) | /id(0.06) | hashed(0.05) | Version(0.05) | (((0.05) | (db(0.05)
    BOT:  ),(-0.06) | Ð½Ð°Ñı(-0.06) | Testing(-0.06) | ãĥ«(-0.05) | -*-(-0.05) | æ£Ģ(-0.05) | ç½ª(-0.05) | .Unique(-0.05)
    ACCEPTED as axis_1056  cumulative_var=0.8213

  [1052]  axes=1057  step_var=0.0013  binary_acc=0.995  gap=0.0933  max_dot=0.0123  (1.8s)
    TOP:  ={Ċ(0.06) | éĢ¾(0.06) | -(0.06) | E(0.06) | _Event(0.06) | (q(0.06) | appearance(0.05) | .open(0.05)
    BOT:  .update(-0.07) | error(-0.05) | MAT(-0.05) | ircles(-0.05) | Ð°Ð¿(-0.05) | _tests(-0.05) | .mult(-0.05) | æŃ£å¸¸(-0.05)
    ACCEPTED as axis_1057  cumulative_var=0.8216

  [1053]  axes=1058  step_var=0.0013  binary_acc=0.983  gap=0.0954  max_dot=0.0037  (1.9s)
    TOP:  éĢŁ(0.06) | }}ĊĊ(0.06) | Regular(0.06) | _IGNORE(0.05) | ersistent(0.05) | å®ĺç½ĳ(0.05) | Ð¸Ðº(0.05) | override(0.05)
    BOT:  éĸĭ(-0.06) | _directory(-0.05) | èĭı(-0.05) | ï¼Į(-0.05) | HIP(-0.05) | unload(-0.05) | .pth(-0.05) | ocket(-0.05)
    ACCEPTED as axis_1058  cumulative_var=0.8218

  [1054]  axes=1059  step_var=0.0013  binary_acc=0.989  gap=0.0982  max_dot=0.0033  (1.8s)
    TOP:  .d(0.06) | 'čĊ(0.06) | spir(0.06) | ]ĊĊ(0.06) | ';Ċ(0.06) | -between(0.06) | Formatted(0.06) | +(0.06)
    BOT:  ductor(-0.07) | _scroll(-0.06) | ç¡®ç«ĭ(-0.06) | .Authentication(-0.06) | /photos(-0.06) | ä¸ĵèģĮ(-0.06) | .accounts(-0.05) | ological(-0.05)
    ACCEPTED as axis_1059  cumulative_var=0.8220

  [1055]  axes=1060  step_var=0.0013  binary_acc=0.971  gap=0.0973  max_dot=0.0006  (1.9s)
    TOP:  .mid(0.06) | âĢ¢(0.06) | This(0.06) | MLS(0.06) | '))Ċ(0.06) | ,{(0.06) | ubernetes(0.06) | -"(0.06)
    BOT:  .permissions(-0.06) | TABLE(-0.05) | .Save(-0.05) | _payment(-0.05) | _list(-0.05) | æ½Ń(-0.05) | è¥¿åĮ»(-0.05) | Raw(-0.05)
    ACCEPTED as axis_1060  cumulative_var=0.8222

  [1056]  axes=1061  step_var=0.0013  binary_acc=0.997  gap=0.0967  max_dot=0.0014  (1.8s)
    TOP:  å½¢æĪĲçļĦ(0.06) | thought(0.06) | .date(0.06) | .Native(0.05) | æĶ¯ä»ĺ(0.05) | Rud(0.05) | èĩªçĦ¶ç§ĳåŃ¦(0.05) | UID(0.05)
    BOT:  Bart(-0.07) | cross(-0.06) | Ð±(-0.06) | .Return(-0.06) | __.(-0.06) | }/(-0.06) | çĶ¨(-0.06) | Shape(-0.06)
    ACCEPTED as axis_1061  cumulative_var=0.8225

  [1057]  axes=1062  step_var=0.0013  binary_acc=0.954  gap=0.0959  max_dot=0.0144  (1.8s)
    TOP:  .Y(0.06) | _python(0.06) | Dmitry(0.06) | çĦī(0.06) | èĤĨ(0.06) | .Logger(0.06) | éģĵ(0.06) | event(0.06)
    BOT:  email(-0.06) | strument(-0.06) | .category(-0.06) | .Result(-0.05) | Participant(-0.05) | çĶŁæĢģç³»ç»Ł(-0.05) | _DELETE(-0.05) | ÑĪÐ¸Ð¹(-0.05)
    ACCEPTED as axis_1062  cumulative_var=0.8227

  [1058]  axes=1063  step_var=0.0013  binary_acc=0.991  gap=0.0965  max_dot=0.0045  (1.9s)
    TOP:  å½»(0.06) | è¿ĳ(0.06) | éĽĨ(0.06) | .AP(0.06) | ç»Īäºİ(0.06) | éĿ¢(0.06) | Ð½ÐµÐ¼(0.06) | /react(0.06)
    BOT:  Ð¸Ð¹(-0.06) | _range(-0.06) | Tutorial(-0.06) | ley(-0.06) | sequel(-0.05) | ç§©åºı(-0.05) | ************************************************************************(-0.05) | ...ĊĊ(-0.05)
    ACCEPTED as axis_1063  cumulative_var=0.8229

  [1059]  axes=1064  step_var=0.0013  binary_acc=0.986  gap=0.0968  max_dot=0.0101  (1.8s)
    TOP:  env(0.06) | .chat(0.06) | wrapper(0.06) | VertexArray(0.06) | Heading(0.06) | -U(0.05) | ÐĴ(0.05) | ç»ĵæŀĦ(0.05)
    BOT:  RE(-0.07) | )))Ċ(-0.07) | .Response(-0.06) | _tree(-0.06) | modify(-0.06) | äº§çĶŁäºĨ(-0.06) | $Ċ(-0.06) | URL(-0.06)
    ACCEPTED as axis_1064  cumulative_var=0.8232

  [1060]  axes=1065  step_var=0.0013  binary_acc=0.992  gap=0.0940  max_dot=0.0112  (1.8s)
    TOP:  urer(0.06) | Tk(0.06) | Boxes(0.06) | )(0.06) | .valid(0.06) | è¯£(0.06) | èĢħ(0.06) | elia(0.05)
    BOT:  .visible(-0.06) | .ops(-0.05) | .State(-0.05) | Ð½ÑĭÐ¼(-0.05) | ViewSet(-0.05) | .choose(-0.05) | .actor(-0.05) | Portable(-0.05)
    ACCEPTED as axis_1065  cumulative_var=0.8234

  [1061]  axes=1066  step_var=0.0013  binary_acc=0.993  gap=0.0938  max_dot=0.0020  (1.8s)
    TOP:  --Ċ(0.07) | my(0.06) | -alert(0.06) | _WRITE(0.06) | or(0.06) | .Raw(0.06) | .softmax(0.06) | .flatten(0.06)
    BOT:  çĿ¡(-0.06) | ADDRESS(-0.05) | Doctor(-0.05) | ç¾½æ¯Ľ(-0.05) | rgb(-0.05) | based(-0.05) | å²Ĺ(-0.05) | Details(-0.05)
    ACCEPTED as axis_1066  cumulative_var=0.8236

  [1062]  axes=1067  step_var=0.0013  binary_acc=0.980  gap=0.0963  max_dot=0.0005  (1.8s)
    TOP:  {čĊ(0.06) | å¥½äºº(0.06) | Farm(0.05) | finish(0.05) | StartTime(0.05) | tiáº¿p(0.05) | wh(0.05) | es(0.05)
    BOT:  ç¨Ģ(-0.07) | _sources(-0.07) | ]/(-0.06) | _ORDER(-0.06) | Ð¼Ð°Ð»Ð¾(-0.06) | /pkg(-0.06) | ,E(-0.06) | delays(-0.05)
    ACCEPTED as axis_1067  cumulative_var=0.8239

  [1063]  axes=1068  step_var=0.0013  binary_acc=0.995  gap=0.0945  max_dot=0.0167  (1.8s)
    TOP:  ä¼ļ(0.06) | åĩºèµĦ(0.06) | æķ°(0.05) | Ð°Ð»ÑĮÐ½ÑĭÐµ(0.05) | åĮĸ(0.05) | LogLevel(0.05) | Ð°ÑĤÑĮÑģÑı(0.05) | æ¡ĮåŃĲä¸Ĭ(0.05)
    BOT:  _END(-0.08) | ec(-0.06) | -toggle(-0.06) | Ec(-0.06) | Fishing(-0.05) | _servers(-0.05) | åħ³çĪ±(-0.05) | Member(-0.05)
    ACCEPTED as axis_1068  cumulative_var=0.8241

  [1064]  axes=1069  step_var=0.0013  binary_acc=1.000  gap=0.0983  max_dot=0.0051  (1.9s)
    TOP:  èĩªå·±(0.07) | from(0.07) | ä»¥(0.06) | _int(0.06) | ç»ı(0.06) | æķ°åŃĹ(0.06) | __(0.06) | .Error(0.05)
    BOT:  ======Ċ(-0.06) | SECTION(-0.05) | keine(-0.05) | Image(-0.05) | tpl(-0.05) | âĸ¼(-0.05) | umber(-0.05) | decision(-0.05)
    ACCEPTED as axis_1069  cumulative_var=0.8243

  [1065]  axes=1070  step_var=0.0013  binary_acc=0.980  gap=0.0963  max_dot=0.0083  (1.9s)
    TOP:  GR(0.06) | å¸½(0.05) | ly(0.05) | PATH(0.05) | skip(0.05) | Ð§ÑĤÐ¾(0.05) | åĨħå®¹(0.05) | config(0.05)
    BOT:  ig(-0.06) | ç¼©(-0.06) | _accuracy(-0.05) | ÑĭÐµ(-0.05) | many(-0.05) | /platform(-0.05) | results(-0.05) | .har(-0.05)
    ACCEPTED as axis_1070  cumulative_var=0.8245

  [1066]  axes=1071  step_var=0.0013  binary_acc=0.995  gap=0.0971  max_dot=0.0029  (1.9s)
    TOP:  aul(0.06) | -application(0.06) | gÃ©nÃ©ral(0.06) | Embed(0.06) | AGE(0.06) | æł¸çĶµ(0.06) | email(0.05) | Presentation(0.05)
    BOT:  Ð´Ð»Ñı(-0.06) | ÑĥÐºÐ°Ð·(-0.06) | www(-0.05) | /interfaces(-0.05) | /C(-0.05) | /sm(-0.05) | _app(-0.05) | citations(-0.05)
    ACCEPTED as axis_1071  cumulative_var=0.8248

  [1067]  axes=1072  step_var=0.0013  binary_acc=0.975  gap=0.0973  max_dot=0.0031  (1.8s)
    TOP:  _file(0.06) | çģ«çģ¾(0.06) | .Operation(0.06) | ä¸Ģæŀļ(0.06) | `).(0.06) | Î³(0.05) | Article(0.05) | ###(0.05)
    BOT:  numpy(-0.06) | (order(-0.05) | Living(-0.05) | FK(-0.05) | .Column(-0.05) | _driver(-0.05) | vasive(-0.05) | NEWS(-0.05)
    ACCEPTED as axis_1072  cumulative_var=0.8250

  [1068]  axes=1073  step_var=0.0013  binary_acc=0.984  gap=0.0949  max_dot=0.0080  (1.9s)
    TOP:  _",(0.06) | ä¸īå¹´(0.06) | erior(0.05) | */ĊĊ(0.05) | Indexed(0.05) | []Ċ(0.05) | aran(0.05) | åŁºç¡Ģä¸Ĭ(0.05)
    BOT:  Loader(-0.07) | application(-0.06) | da(-0.06) | .exists(-0.06) | æĺİæľĿ(-0.06) | ca(-0.06) | _np(-0.06) | _ch(-0.06)
    ACCEPTED as axis_1073  cumulative_var=0.8252

  [1069]  axes=1074  step_var=0.0013  binary_acc=0.988  gap=0.0935  max_dot=0.0023  (1.8s)
    TOP:  æľīä¸įåĲĮçļĦ(0.06) | æľ¬æ¬¡äº¤æĺĵ(0.05) | gpio(0.05) | .broadcast(0.05) | D(0.05) | 8(0.05) | Jess(0.05) | /en(0.05)
    BOT:  Show(-0.06) | çĹ¢(-0.05) | quelle(-0.05) | .z(-0.05) | Roles(-0.05) | _null(-0.05) | ì£¼(-0.05) | .about(-0.05)
    ACCEPTED as axis_1074  cumulative_var=0.8255

  [1070]  axes=1075  step_var=0.0013  binary_acc=0.968  gap=0.0947  max_dot=0.0097  (1.8s)
    TOP:  .w(0.06) | _mt(0.06) | çĶŁåĳ½çļĦ(0.06) | .Location(0.06) | _params(0.06) | therapist(0.05) | ãģ©ãģĨ(0.05) | -word(0.05)
    BOT:  res(-0.06) | Mismatch(-0.06) | sum(-0.06) | izer(-0.06) | in(-0.06) | ir(-0.05) | _VERSION(-0.05) | _rows(-0.05)
    ACCEPTED as axis_1075  cumulative_var=0.8257

  [1071]  axes=1076  step_var=0.0013  binary_acc=0.993  gap=0.0952  max_dot=0.0068  (1.9s)
    TOP:  ations(0.06) | ascade(0.06) | note(0.06) | Ps(0.05) | .ir(0.05) | .uni(0.05) | éī´(0.05) | ä¿ĿæĬ¤(0.05)
    BOT:  (Login(-0.06) | Networks(-0.06) | package(-0.06) | BÃłi(-0.06) | off(-0.05) | =false(-0.05) | .example(-0.05) | ç»´çĶŁç´ł(-0.05)
    ACCEPTED as axis_1076  cumulative_var=0.8259

  [1072]  axes=1077  step_var=0.0013  binary_acc=0.997  gap=0.0952  max_dot=0.0166  (1.8s)
    TOP:  .r(0.06) | sWith(0.06) | ovation(0.06) | think(0.06) | asin(0.06) | ¹(0.05) | ols(0.05) | sid(0.05)
    BOT:  Ã¹(-0.06) | å¯¹(-0.06) | .Map(-0.05) | ycler(-0.05) | DEFAULT(-0.05) | ç³»ç»Ł(-0.05) | æºĥçĸ¡(-0.05) | Ð¼ÐµÐ´Ð¸(-0.05)
    ACCEPTED as axis_1077  cumulative_var=0.8261

  [1073]  axes=1078  step_var=0.0013  binary_acc=0.991  gap=0.0939  max_dot=0.0052  (1.9s)
    TOP:  .username(0.06) | .sources(0.06) | _new(0.06) | _Function(0.05) | Ð¼Ð°Ð½(0.05) | ÑģÐ¸(0.05) | -the(0.05) | ek(0.05)
    BOT:  cart(-0.06) | _ITER(-0.06) | Txt(-0.06) | .linear(-0.06) | .use(-0.06) | .CodeAnalysis(-0.06) | Movie(-0.05) | Object(-0.05)
    ACCEPTED as axis_1078  cumulative_var=0.8264

  [1074]  axes=1079  step_var=0.0013  binary_acc=0.997  gap=0.0947  max_dot=0.0084  (1.8s)
    TOP:  faces(0.07) | _NEXT(0.06) | ä¹ĭæīĢ(0.06) | Tip(0.05) | -*-(0.05) | à¸Ķ(0.05) | aver(0.05) | æ³¨æĺİæĿ¥æºĲ(0.05)
    BOT:  Sl(-0.06) | categories(-0.06) | Authentication(-0.06) | -base(-0.06) | >.(-0.06) | extension(-0.06) | No(-0.06) | .Format(-0.05)
    ACCEPTED as axis_1079  cumulative_var=0.8266

  [1075]  axes=1080  step_var=0.0013  binary_acc=0.999  gap=0.0959  max_dot=0.0037  (1.9s)
    TOP:  (default(0.07) | Persistent(0.06) | ifton(0.05) | IT(0.05) | Random(0.05) | Portions(0.05) | .square(0.05) | Principal(0.05)
    BOT:  é¦Ī(-0.06) | æ»ĳ(-0.06) | Ð½ÐµÑĤ(-0.06) | unknown(-0.05) | dee(-0.05) | .fp(-0.05) | amentals(-0.05) | èĩªçĦ¶(-0.05)
    ACCEPTED as axis_1080  cumulative_var=0.8268

  [1076]  axes=1081  step_var=0.0013  binary_acc=0.978  gap=0.0948  max_dot=0.0017  (1.9s)
    TOP:  (group(0.06) | Author(0.06) | .Default(0.06) | -years(0.06) | .meta(0.06) | ,T(0.06) | -edit(0.06) | (original(0.06)
    BOT:  a(-0.06) | æĸ¯(-0.06) | Ø¯(-0.05) | At(-0.05) | uw(-0.05) | unresolved(-0.05) | ¿(-0.05) | Licensed(-0.05)
    ACCEPTED as axis_1081  cumulative_var=0.8270

  [1077]  axes=1082  step_var=0.0013  binary_acc=0.995  gap=0.0943  max_dot=0.0053  (1.8s)
    TOP:  }Ċ(0.07) | connected(0.06) | éĺ³åİ¿(0.06) | Council(0.06) | Dataset(0.06) | æĶ¾åľ¨(0.06) | åħļå·¥å§Ķ(0.06) | =['(0.05)
    BOT:  scribers(-0.06) | /ms(-0.06) | Upload(-0.06) | _large(-0.05) | .serialization(-0.05) | /apt(-0.05) | .initialize(-0.05) | acie(-0.05)
    ACCEPTED as axis_1082  cumulative_var=0.8273

  [1078]  axes=1083  step_var=0.0013  binary_acc=0.988  gap=0.0932  max_dot=0.0028  (1.9s)
    TOP:  .begin(0.06) | urtle(0.06) | -operator(0.06) | .pet(0.05) | _library(0.05) | .schema(0.05) | -plugin(0.05) | unities(0.05)
    BOT:  èĿ´èĿ¶(-0.06) | ç¨³(-0.06) | uuid(-0.06) | Payment(-0.05) | .Change(-0.05) | è¿ŀ(-0.05) | connection(-0.05) | å½©ç¥¨(-0.05)
    ACCEPTED as axis_1083  cumulative_var=0.8275

  [1079]  axes=1084  step_var=0.0013  binary_acc=0.984  gap=0.0943  max_dot=0.0045  (1.9s)
    TOP:  (start(0.06) | /map(0.05) | onomies(0.05) | çº½å¸¦(0.05) | /__(0.05) | æµ·(0.05) | -twitter(0.05) | å·¥ä½ľçļĦ(0.05)
    BOT:  am(-0.06) | Ð°Ð½ÑĤ(-0.06) | v(-0.06) | out(-0.06) | bs(-0.06) | job(-0.06) | Include(-0.06) | iej(-0.06)
    ACCEPTED as axis_1084  cumulative_var=0.8277

  [1080]  axes=1085  step_var=0.0013  binary_acc=0.981  gap=0.0954  max_dot=0.0071  (1.9s)
    TOP:  IES(0.06) | LES(0.06) | aw(0.06) | _base(0.06) | ä¸įåĲĮçļĦ(0.05) | mini(0.05) | ance(0.05) | /platform(0.05)
    BOT:  .hu(-0.06) | (network(-0.06) | _pe(-0.06) | .Game(-0.06) | åıĳå¸ĥåħ¬åĳĬ(-0.06) | network(-0.05) | .property(-0.05) | Vel(-0.05)
    ACCEPTED as axis_1085  cumulative_var=0.8280

  [1081]  axes=1086  step_var=0.0013  binary_acc=0.993  gap=0.0956  max_dot=0.0056  (1.9s)
    TOP:  è¿ľ(0.06) | .ws(0.06) | .LINE(0.06) | LOW(0.05) | .mapper(0.05) | igration(0.05) | (dec(0.05) | _COUNT(0.05)
    BOT:  (client(-0.06) | -h(-0.06) | .exe(-0.05) | straight(-0.05) | ];Ċ(-0.05) | .boost(-0.05) | Cav(-0.05) | _IR(-0.05)
    ACCEPTED as axis_1086  cumulative_var=0.8282

  [1082]  axes=1087  step_var=0.0013  binary_acc=0.968  gap=0.0938  max_dot=0.0028  (1.9s)
    TOP:  H(0.06) | Callback(0.06) | _An(0.06) | (length(0.05) | çĻ½(0.05) | AZ(0.05) | inqu(0.05) | remely(0.05)
    BOT:  sgi(-0.06) | ros(-0.06) | _css(-0.06) | (settings(-0.06) | TEX(-0.06) | Powers(-0.06) | f(-0.05) | posts(-0.05)
    ACCEPTED as axis_1087  cumulative_var=0.8284

  [1083]  axes=1088  step_var=0.0013  binary_acc=0.974  gap=0.0936  max_dot=0.0073  (1.9s)
    TOP:  /command(0.06) | .audio(0.06) | Maxim(0.06) | ä¾ĽåºĶéĵ¾(0.06) | -table(0.05) | .Message(0.05) | -ins(0.05) | _locations(0.05)
    BOT:  Regards(-0.06) | Release(-0.06) | Helpers(-0.06) | à¸ļ(-0.06) | _authenticated(-0.05) | /uploads(-0.05) | bian(-0.05) | com(-0.05)
    ACCEPTED as axis_1088  cumulative_var=0.8286

  [1084]  axes=1089  step_var=0.0013  binary_acc=0.968  gap=0.0968  max_dot=0.0025  (1.8s)
    TOP:  html(0.07) | studio(0.06) | salary(0.06) | /fast(0.06) | Recent(0.06) | .experimental(0.06) | SR(0.05) | trigger(0.05)
    BOT:  itet(-0.06) | POSITORY(-0.06) | Lands(-0.06) | Ð½ÐµÐ¼(-0.05) | VED(-0.05) | (debug(-0.05) | {id(-0.05) | RuntimeError(-0.05)
    ACCEPTED as axis_1089  cumulative_var=0.8289

  [1085]  axes=1090  step_var=0.0013  binary_acc=0.989  gap=0.0944  max_dot=0.0012  (1.9s)
    TOP:  Java(0.06) | m(0.06) | Context(0.06) | Ø£(0.06) | æĢĿèĢĥ(0.06) | TOP(0.05) | OOK(0.05) | mission(0.05)
    BOT:  -transparent(-0.06) | _PROXY(-0.05) | ä½ł(-0.05) | (char(-0.05) | Ð·(-0.05) | æİ¨(-0.05) | Shop(-0.05) | trinsic(-0.05)
    ACCEPTED as axis_1090  cumulative_var=0.8291

  [1086]  axes=1091  step_var=0.0013  binary_acc=0.986  gap=0.0953  max_dot=0.0104  (1.8s)
    TOP:  );čĊ(0.07) | Ð¿(0.06) | PRE(0.06) | .)ĊĊ(0.06) | Ð¾(0.06) | to(0.06) | Probe(0.06) | T(0.06)
    BOT:  >The(-0.06) | limitations(-0.06) | çŃ¹èµĦ(-0.05) | Very(-0.05) | encode(-0.05) | (l(-0.05) | b(-0.05) | ä¹ĭéĻħ(-0.05)
    ACCEPTED as axis_1091  cumulative_var=0.8293

  [1087]  axes=1092  step_var=0.0013  binary_acc=0.986  gap=0.0961  max_dot=0.0021  (1.9s)
    TOP:  åı·(0.06) | bm(0.06) | å¦Ĭå¨ł(0.06) | (format(0.06) | 'app(0.06) | Unauthorized(0.05) | Fir(0.05) | æ³ī(0.05)
    BOT:  .Send(-0.05) | .tokenize(-0.05) | Chi(-0.05) | /random(-0.05) | licing(-0.05) | .XML(-0.05) | .exe(-0.05) | result(-0.05)
    ACCEPTED as axis_1092  cumulative_var=0.8295

  [1088]  axes=1093  step_var=0.0013  binary_acc=0.987  gap=0.0969  max_dot=0.0026  (1.8s)
    TOP:  è¿Ľ(0.06) | çļĦ(0.05) | her(0.05) | ç»ı(0.05) | ÐµÑģÑĤÐ²Ð¾(0.05) | /wp(0.05) | People(0.05) | NY(0.05)
    BOT:  _has(-0.06) | /legal(-0.06) | }ĊĊĊ(-0.05) | éĶĢæ¯ģ(-0.05) | mill(-0.05) | Em(-0.05) | pow(-0.05) | -->čĊčĊ(-0.05)
    ACCEPTED as axis_1093  cumulative_var=0.8298

  [1089]  axes=1094  step_var=0.0013  binary_acc=0.984  gap=0.0945  max_dot=0.0024  (1.8s)
    TOP:  Sub(0.06) | -dom(0.05) | çĶ¨æĪ·(0.05) | ():Ċ(0.05) | Wild(0.05) | ALL(0.05) | Unicode(0.05) | é¢Ĩå¯¼èĢħ(0.05)
    BOT:  local(-0.07) | /storage(-0.07) | id(-0.06) | _mark(-0.06) | æĦıæĦ¿(-0.06) | edia(-0.06) | af(-0.06) | _txt(-0.06)
    ACCEPTED as axis_1094  cumulative_var=0.8300

  [1090]  axes=1095  step_var=0.0013  binary_acc=0.975  gap=0.0924  max_dot=0.0048  (1.8s)
    TOP:  sec(0.06) | .back(0.06) | /github(0.06) | ffects(0.06) | das(0.06) | .err(0.05) | bies(0.05) | .minute(0.05)
    BOT:  O(-0.08) | Human(-0.06) | .target(-0.06) | John(-0.06) | Construct(-0.06) | æĬĢæľ¯çłĶåıĳ(-0.06) | æµ·(-0.05) | (prediction(-0.05)
    ACCEPTED as axis_1095  cumulative_var=0.8302

  [1091]  axes=1096  step_var=0.0013  binary_acc=0.992  gap=0.0947  max_dot=0.0055  (1.8s)
    TOP:  onia(0.07) | .Time(0.07) | (out(0.07) | AME(0.06) | é£İ(0.06) | æī¿(0.06) | /function(0.06) | .Condition(0.06)
    BOT:  _blue(-0.05) | èĩªä¿¡(-0.05) | çº·(-0.05) | IGHT(-0.05) | ui(-0.05) | LES(-0.05) | ä¸´è¿ĳ(-0.05) | route(-0.05)
    ACCEPTED as axis_1096  cumulative_var=0.8304

  [1092]  axes=1097  step_var=0.0013  binary_acc=0.994  gap=0.0925  max_dot=0.0043  (1.8s)
    TOP:  gp(0.06) | it(0.06) | ä½ıæīĢ(0.06) | mj(0.06) | rike(0.06) | oke(0.06) | ):Ċ(0.06) | Weight(0.06)
    BOT:  .exec(-0.06) | Marker(-0.06) | strips(-0.05) | .Highlight(-0.05) | annels(-0.05) | æĹłæĥħ(-0.05) | Strings(-0.05) | è¾¾åĪ°(-0.05)
    ACCEPTED as axis_1097  cumulative_var=0.8306

  [1093]  axes=1098  step_var=0.0013  binary_acc=0.998  gap=0.0957  max_dot=0.0106  (1.9s)
    TOP:  .čĊ(0.06) | ifacts(0.06) | âĢ¢(0.06) | ";ĊĊ(0.06) | .k(0.06) | .",Ċ(0.06) | Â·(0.06) | ',Ċ(0.06)
    BOT:  _variable(-0.05) | æ»ĳ(-0.05) | Revolution(-0.05) | ä¼łç»ŁçļĦ(-0.05) | èĮĥåĽ´(-0.05) | çĴ§(-0.05) | Guidance(-0.05) | çħ§(-0.05)
    ACCEPTED as axis_1098  cumulative_var=0.8309

  [1094]  axes=1099  step_var=0.0013  binary_acc=0.968  gap=0.0946  max_dot=0.0068  (1.8s)
    TOP:  çĽĳæµĭ(0.06) | .TEST(0.06) | è¿Ŀæ³ķ(0.06) | /ap(0.05) | Factory(0.05) | @@(0.05) | ic(0.05) | ä¸įåİ»(0.05)
    BOT:  ]ĊĊ(-0.06) | à¸ª(-0.06) | reserve(-0.06) | .Non(-0.06) | members(-0.05) | %}Ċ(-0.05) | ."""(-0.05) | ro(-0.05)
    ACCEPTED as axis_1099  cumulative_var=0.8311

  [1095]  axes=1100  step_var=0.0013  binary_acc=0.998  gap=0.0957  max_dot=0.0115  (1.8s)
    TOP:  _F(0.06) | -Q(0.06) | ìĺ¨(0.06) | äººä»¬çļĦ(0.06) | æĦĪ(0.05) | acious(0.05) | ÑĭÐµ(0.05) | .runner(0.05)
    BOT:  {n(-0.06) | Proto(-0.06) | '"Ċ(-0.06) | {Ċ(-0.06) | Errors(-0.05) | Cas(-0.05) | };ĊĊ(-0.05) | Is(-0.05)
    ACCEPTED as axis_1100  cumulative_var=0.8313

  [1096]  axes=1101  step_var=0.0013  binary_acc=0.989  gap=0.0959  max_dot=0.0099  (1.8s)
    TOP:  With(0.07) | Json(0.06) | _forward(0.06) | Park(0.06) | -data(0.05) | _control(0.05) | check(0.05) | next(0.05)
    BOT:  Jobs(-0.06) | INGLE(-0.06) | _CENTER(-0.06) | ÑĤ(-0.05) | .Option(-0.05) | èĢĮ(-0.05) | Ĭ(-0.05) | uset(-0.05)
    ACCEPTED as axis_1101  cumulative_var=0.8315

  [1097]  axes=1102  step_var=0.0013  binary_acc=0.981  gap=0.0952  max_dot=0.0037  (1.9s)
    TOP:  .datasets(0.07) | .verify(0.06) | æĦģ(0.06) | Surface(0.06) | /media(0.06) | /@(0.06) | .lib(0.06) | _method(0.06)
    BOT:  From(-0.06) | åĲĮ(-0.06) | /cs(-0.06) | åıĭæĥħ(-0.05) | Browser(-0.05) | æĦıè§ģ(-0.05) | Sex(-0.05) | .ignore(-0.05)
    ACCEPTED as axis_1102  cumulative_var=0.8318

  [1098]  axes=1103  step_var=0.0013  binary_acc=0.995  gap=0.0948  max_dot=0.0082  (1.8s)
    TOP:  .google(0.06) | xe(0.05) | ').Ċ(0.05) | .up(0.05) | Wiki(0.05) | ,L(0.05) | Mongo(0.05) | _find(0.05)
    BOT:  ä¹ĭç§°(-0.05) | ãģĹãģ¾ãģĻ(-0.05) | èĮĥæĸĩ(-0.05) | .T(-0.05) | _service(-0.05) | _DEBUG(-0.05) | (dist(-0.05) | Sy(-0.05)
    ACCEPTED as axis_1103  cumulative_var=0.8320

  [1099]  axes=1104  step_var=0.0013  binary_acc=0.971  gap=0.0965  max_dot=0.0006  (1.8s)
    TOP:  PushButton(0.06) | Limit(0.06) | -dev(0.05) | MQTT(0.05) | _f(0.05) | Button(0.05) | .Class(0.05) | License(0.05)
    BOT:  èĩªå·±çļĦ(-0.06) | Count(-0.06) | .ylabel(-0.06) | Constraint(-0.05) | \čĊ(-0.05) | \Ċ(-0.05) | Ðº(-0.05) | /article(-0.05)
    ACCEPTED as axis_1104  cumulative_var=0.8322

  [1100]  axes=1105  step_var=0.0013  binary_acc=0.994  gap=0.0950  max_dot=0.0039  (1.9s)
    TOP:  _CENTER(0.05) | Ð½Ð°Ñı(0.05) | æ¡Ĥ(0.05) | .Runtime(0.05) | åĮºåŁŁåĨħ(0.05) | signal(0.05) | Calculate(0.05) | rear(0.05)
    BOT:  .cn(-0.06) | .valid(-0.06) | Exception(-0.06) | validation(-0.05) | ()Ċ(-0.05) | __.(-0.05) | /apis(-0.05) | ']]Ċ(-0.05)
    ACCEPTED as axis_1105  cumulative_var=0.8324

  [1101]  axes=1106  step_var=0.0013  binary_acc=0.982  gap=0.0946  max_dot=0.0167  (1.8s)
    TOP:  .client(0.06) | .check(0.06) | èĤ¥(0.06) | åŁºäºİ(0.06) | ãģĮãģĤãĤĬãģ¾ãģĻ(0.06) | .display(0.05) | é£İæł¼(0.05) | -May(0.05)
    BOT:  Abstract(-0.07) | Bes(-0.05) | ech(-0.05) | Logical(-0.05) | react(-0.05) | serial(-0.05) | Double(-0.05) | Skip(-0.05)
    ACCEPTED as axis_1106  cumulative_var=0.8327

  [1102]  axes=1107  step_var=0.0013  binary_acc=0.997  gap=0.0940  max_dot=0.0260  (1.9s)
    TOP:  Co(0.06) | entr(0.06) | /article(0.06) | ÆĴ(0.05) | print(0.05) | .setMessage(0.05) | .No(0.05) | .it(0.05)
    BOT:  çĶŁåĳ½(-0.07) | def(-0.07) | .parse(-0.06) | åħ¨ä½ĵ(-0.06) | .time(-0.06) | .encode(-0.05) | ');ĊĊĊ(-0.05) | icons(-0.05)
    ACCEPTED as axis_1107  cumulative_var=0.8329

  [1103]  axes=1108  step_var=0.0013  binary_acc=0.984  gap=0.0929  max_dot=0.0023  (1.9s)
    TOP:  _NOTIFICATION(0.06) | ames(0.06) | Repository(0.05) | _attribute(0.05) | hu(0.05) | .changed(0.05) | .static(0.05) | éĶ»çĤ¼(0.05)
    BOT:  area(-0.07) | Ð¸(-0.06) | does(-0.06) | ä¸įä½İäºİ(-0.05) | current(-0.05) | GD(-0.05) | duce(-0.05) | ìĹĲìĦľ(-0.05)
    ACCEPTED as axis_1108  cumulative_var=0.8331

  [1104]  axes=1109  step_var=0.0013  binary_acc=0.995  gap=0.0919  max_dot=0.0050  (1.9s)
    TOP:  isoft(0.06) | (O(0.06) | _operation(0.05) | ìĿ¼(0.05) | mediate(0.05) | =}(0.05) | -gradient(0.05) | Dem(0.05)
    BOT:  è´¤(-0.07) | æĬķ(-0.06) | ness(-0.06) | btc(-0.05) | crire(-0.05) | Layer(-0.05) | ")ĊĊ(-0.05) | complex(-0.05)
    ACCEPTED as axis_1109  cumulative_var=0.8333

  [1105]  axes=1110  step_var=0.0013  binary_acc=0.989  gap=0.0924  max_dot=0.0086  (1.8s)
    TOP:  ZE(0.06) | ||((0.06) | .Init(0.06) | _tr(0.06) | _code(0.05) | (f(0.05) | .)ĊĊ(0.05) | (nb(0.05)
    BOT:  Name(-0.06) | lastic(-0.06) | äº§çī©(-0.06) | pk(-0.06) | Base(-0.06) | his(-0.06) | detach(-0.06) | below(-0.06)
    ACCEPTED as axis_1110  cumulative_var=0.8336

  [1106]  axes=1111  step_var=0.0013  binary_acc=0.984  gap=0.0924  max_dot=0.0089  (1.8s)
    TOP:  .build(0.06) | ','(0.06) | _dict(0.05) | _DOWN(0.05) | (user(0.05) | matplotlib(0.05) | ä¹īåĬ¡(0.05) | -google(0.05)
    BOT:  .multi(-0.06) | .fx(-0.06) | _cid(-0.06) | else(-0.05) | .sleep(-0.05) | ay(-0.05) | ±(-0.05) | gis(-0.05)
    ACCEPTED as axis_1111  cumulative_var=0.8338

  [1107]  axes=1112  step_var=0.0013  binary_acc=0.985  gap=0.0936  max_dot=0.0131  (1.9s)
    TOP:  _ip(0.06) | =device(0.06) | Â©(0.06) | _wait(0.06) | "])Ċ(0.06) | ç»Ħç»ĩå®ŀæĸ½(0.06) | .$(0.05) | pal(0.05)
    BOT:  /source(-0.05) | lem(-0.05) | igan(-0.05) | hex(-0.05) | Forgot(-0.05) | _variable(-0.05) | Burton(-0.05) | mp(-0.05)
    ACCEPTED as axis_1112  cumulative_var=0.8340

  [1108]  axes=1113  step_var=0.0013  binary_acc=0.991  gap=0.0953  max_dot=0.0072  (1.9s)
    TOP:  .invoke(0.06) | çľ¼(0.06) | O(0.06) | Ð¢(0.05) | çº¢(0.05) | indexed(0.05) | .transform(0.05) | T(0.05)
    BOT:  primary(-0.06) | å½¢æĪĲäºĨ(-0.06) | identifier(-0.05) | __)Ċ(-0.05) | ();Ċ(-0.05) | /new(-0.05) | }čĊ(-0.05) | ogeneous(-0.05)
    ACCEPTED as axis_1113  cumulative_var=0.8342

  [1109]  axes=1114  step_var=0.0013  binary_acc=0.988  gap=0.0929  max_dot=0.0160  (1.8s)
    TOP:  åįİ(0.07) | å°Ĩç»§ç»Ń(0.06) | Middleware(0.06) | -danger(0.06) | arrow(0.06) | ç¬º(0.05) | .conv(0.05) | åı¦æľī(0.05)
    BOT:  (Ċ(-0.06) | ='(-0.06) | Outline(-0.06) | import(-0.06) | Include(-0.06) | .api(-0.06) | Voltage(-0.06) | izable(-0.05)
    ACCEPTED as axis_1114  cumulative_var=0.8344

  [1110]  axes=1115  step_var=0.0013  binary_acc=0.983  gap=0.0943  max_dot=0.0050  (1.9s)
    TOP:  /query(0.07) | (ind(0.06) | ("|(0.06) | ".ĊĊ(0.06) | .IO(0.06) | _filter(0.05) | conson(0.05) | Transfer(0.05)
    BOT:  å®ŀæĸ½ç»ĨåĪĻ(-0.06) | _select(-0.05) | ÑĪÐ¸Ñħ(-0.05) | ÃŁ(-0.05) | Compared(-0.05) | _specs(-0.05) | æ·±åħ¥(-0.05) | é¢ĩæľī(-0.05)
    ACCEPTED as axis_1115  cumulative_var=0.8347

  [1111]  axes=1116  step_var=0.0013  binary_acc=0.987  gap=0.0919  max_dot=0.0035  (1.9s)
    TOP:  Ċ(0.06) | Alerts(0.05) | None(0.05) | ({Ċ(0.05) | Msg(0.05) | ĉvoid(0.05) | ]ĊĊĊ(0.05) | åħ±åĲĮ(0.05)
    BOT:  .exception(-0.07) | serializers(-0.06) | Time(-0.05) | ckeditor(-0.05) | æ·»åĬłåīĤ(-0.05) | Oi(-0.05) | .take(-0.05) | output(-0.05)
    ACCEPTED as axis_1116  cumulative_var=0.8349

  [1112]  axes=1117  step_var=0.0013  binary_acc=0.986  gap=0.0923  max_dot=0.0145  (1.8s)
    TOP:  import(0.07) | /view(0.06) | describe(0.06) | .full(0.06) | (filter(0.05) | .Method(0.05) | (context(0.05) | Ã¡veis(0.05)
    BOT:  S(-0.07) | èĤĸ(-0.06) | ãģĿ(-0.06) | æ±ī(-0.06) | aac(-0.06) | æīĭ(-0.06) | ochastic(-0.06) | find(-0.06)
    ACCEPTED as axis_1117  cumulative_var=0.8351

  [1113]  axes=1118  step_var=0.0013  binary_acc=0.996  gap=0.0928  max_dot=0.0155  (1.8s)
    TOP:  .driver(0.07) | Connection(0.06) | Of(0.06) | _after(0.06) | Semester(0.05) | è¾Ł(0.05) | Serious(0.05) | ))ĊĊ(0.05)
    BOT:  .S(-0.07) | _only(-0.07) | Sdk(-0.06) | private(-0.06) | csv(-0.06) | ç»ĵæĿŁåĲİ(-0.05) | Limit(-0.05) | PR(-0.05)
    ACCEPTED as axis_1118  cumulative_var=0.8353

  [1114]  axes=1119  step_var=0.0013  binary_acc=0.963  gap=0.0929  max_dot=0.0052  (1.9s)
    TOP:  typing(0.06) | æŃ¤æĸĩ(0.05) | .',(0.05) | SD(0.05) | -java(0.05) | éĢĶå¾Ħ(0.05) | (rgb(0.05) | )ĊĊ(0.05)
    BOT:  ometrics(-0.06) | .load(-0.06) | *(-0.06) | presso(-0.05) | äºĪ(-0.05) | .POST(-0.05) | .devices(-0.05) | ¹(-0.05)
    ACCEPTED as axis_1119  cumulative_var=0.8355

  [1115]  axes=1120  step_var=0.0013  binary_acc=0.980  gap=0.0936  max_dot=0.0069  (1.8s)
    TOP:  çŃĶæ¡Ī(0.07) | Ð½Ð¾(0.06) | çĬ¶æĢģ(0.06) | à¸´à¸Ļ(0.06) | onte(0.06) | .uint(0.06) | c(0.05) | _visible(0.05)
    BOT:  City(-0.06) | {Ċ(-0.06) | Moral(-0.06) | (error(-0.05) | likewise(-0.05) | _compat(-0.05) | ç§įç±»(-0.05) | (fid(-0.05)
    ACCEPTED as axis_1120  cumulative_var=0.8357

  [1116]  axes=1121  step_var=0.0013  binary_acc=0.986  gap=0.0941  max_dot=0.0079  (1.9s)
    TOP:  æ²¡èĥ½(0.06) | runtime(0.06) | ³(0.05) | _stdout(0.05) | _FILES(0.05) | çłĶç©¶(0.05) | function(0.05) | MÃ¼ller(0.05)
    BOT:  Conv(-0.06) | _zero(-0.06) | Ðľ(-0.06) | Pool(-0.06) | åį«çĶŁåģ¥åº·(-0.06) | .annotation(-0.05) | SHORT(-0.05) | AL(-0.05)
    ACCEPTED as axis_1121  cumulative_var=0.8360

  [1117]  axes=1122  step_var=0.0013  binary_acc=0.992  gap=0.0942  max_dot=0.0092  (1.8s)
    TOP:  (document(0.06) | SYN(0.06) | /embed(0.05) | .em(0.05) | _logo(0.05) | å°¾(0.05) | .epoch(0.05) | Identifier(0.05)
    BOT:  å¹´(-0.06) | ');(-0.06) | ucceed(-0.06) | .refresh(-0.06) | Ð¾(-0.05) | .exec(-0.05) | Followers(-0.05) | è¦ĭ(-0.05)
    ACCEPTED as axis_1122  cumulative_var=0.8362

  [1118]  axes=1123  step_var=0.0013  binary_acc=0.998  gap=0.0944  max_dot=0.0161  (1.8s)
    TOP:  æĹłåħ³(0.06) | _pages(0.06) | ç»Ŀç¼ĺ(0.06) | (request(0.06) | _population(0.06) | backup(0.05) | _readable(0.05) | æ¦Ĥå¿µ(0.05)
    BOT:  format(-0.06) | _Config(-0.06) | romium(-0.06) | ><(-0.06) | Description(-0.05) | schema(-0.05) | æķĮäºº(-0.05) | Frame(-0.05)
    ACCEPTED as axis_1123  cumulative_var=0.8364

  [1119]  axes=1124  step_var=0.0013  binary_acc=0.958  gap=0.0934  max_dot=0.0097  (1.8s)
    TOP:  )."(0.06) | .keyboard(0.06) | .IO(0.05) | urers(0.05) | ').Ċ(0.05) | Roman(0.05) | åħĶåŃĲ(0.05) | _interval(0.05)
    BOT:  è¿ĩ(-0.06) | .Language(-0.06) | iger(-0.06) | .port(-0.06) | æĬ¢(-0.06) | _train(-0.06) | åĽ½æ°ĳç»ıæµİ(-0.06) | æĿ¿(-0.06)
    ACCEPTED as axis_1124  cumulative_var=0.8366

  [1120]  axes=1125  step_var=0.0013  binary_acc=0.958  gap=0.0946  max_dot=0.0096  (1.8s)
    TOP:  JP(0.06) | _media(0.06) | /shared(0.06) | SPORT(0.05) | æĬ½æŁ¥(0.05) | /nginx(0.05) | Mount(0.05) | _control(0.05)
    BOT:  /api(-0.07) | .lang(-0.06) | _DO(-0.06) | Provides(-0.06) | å°±æĺ¯(-0.05) | postgresql(-0.05) | timings(-0.05) | _runtime(-0.05)
    ACCEPTED as axis_1125  cumulative_var=0.8368

  [1121]  axes=1126  step_var=0.0013  binary_acc=0.993  gap=0.0936  max_dot=0.0061  (1.9s)
    TOP:  elic(0.06) | W(0.06) | acoes(0.06) | Ð½Ð¸ÑĨÐ°(0.06) | à¸´à¸ĩ(0.05) | Ø¢(0.05) | feld(0.05) | -bg(0.05)
    BOT:  (worker(-0.06) | _edge(-0.06) | å¯¹åħ¶(-0.06) | Input(-0.06) | =e(-0.05) | çĲĨå·¥å¤§åŃ¦(-0.05) | Nordic(-0.05) | -com(-0.05)
    ACCEPTED as axis_1126  cumulative_var=0.8371

  [1122]  axes=1127  step_var=0.0013  binary_acc=0.989  gap=0.0948  max_dot=0.0015  (1.9s)
    TOP:  /en(0.07) | .Standard(0.07) | .Window(0.06) | .loss(0.06) | ."Ċ(0.06) | _configuration(0.05) | (last(0.05) | .It(0.05)
    BOT:  æ¯ħ(-0.06) | /documents(-0.06) | mr(-0.06) | èĲ¥åĪ©(-0.06) | ind(-0.05) | state(-0.05) | protecting(-0.05) | cost(-0.05)
    ACCEPTED as axis_1127  cumulative_var=0.8373

  [1123]  axes=1128  step_var=0.0013  binary_acc=0.995  gap=0.0932  max_dot=0.0102  (1.9s)
    TOP:  (body(0.05) | (shape(0.05) | .no(0.05) | Cycling(0.05) | Normalized(0.05) | (load(0.05) | æĮĩåįĹ(0.05) | divide(0.05)
    BOT:  ä¹Ļ(-0.06) | ?q(-0.06) | .Ex(-0.06) | è°±(-0.06) | edList(-0.06) | nome(-0.06) | .`(-0.06) | .next(-0.06)
    ACCEPTED as axis_1128  cumulative_var=0.8375

  [1124]  axes=1129  step_var=0.0013  binary_acc=0.999  gap=0.0946  max_dot=0.0011  (1.8s)
    TOP:  æĭ¬(0.05) | Function(0.05) | -Ċ(0.05) | /blog(0.05) | /api(0.05) | .ForeignKey(0.05) | Alloc(0.05) | ÄŁÄ±(0.05)
    BOT:  .model(-0.07) | Spanish(-0.05) | med(-0.05) | were(-0.05) | ef(-0.05) | æķ°æį®(-0.05) | Validator(-0.05) | /math(-0.05)
    ACCEPTED as axis_1129  cumulative_var=0.8377

  [1125]  axes=1130  step_var=0.0013  binary_acc=0.997  gap=0.0943  max_dot=0.0234  (1.9s)
    TOP:  .IO(0.07) | èĢħ(0.06) | getContent(0.06) | (),Ċ(0.06) | ist(0.06) | Ð½Ð¾ÑģÑĤÑĮ(0.06) | Encoded(0.05) | Saved(0.05)
    BOT:  UBE(-0.06) | rd(-0.06) | u(-0.06) | .Usage(-0.06) | Test(-0.05) | .Msg(-0.05) | ##(-0.05) | å±±(-0.05)
    ACCEPTED as axis_1130  cumulative_var=0.8379

  [1126]  axes=1131  step_var=0.0013  binary_acc=0.991  gap=0.0938  max_dot=0.0132  (1.8s)
    TOP:  .as(0.06) | './(0.06) | off(0.06) | there(0.06) | .handler(0.06) | datasets(0.05) | (config(0.05) | Vari(0.05)
    BOT:  _neighbors(-0.06) | UA(-0.06) | _WIDTH(-0.06) | .about(-0.06) | éĥ½æľīèĩªå·±(-0.06) | .No(-0.06) | Ð»ÐµÐ½Ð¸Ðµ(-0.06) | allback(-0.05)
    ACCEPTED as axis_1131  cumulative_var=0.8381

  [1127]  axes=1132  step_var=0.0014  binary_acc=0.996  gap=0.0953  max_dot=0.0120  (1.8s)
    TOP:  ithub(0.06) | intersects(0.06) | trib(0.06) | _files(0.06) | .Core(0.06) | icient(0.05) | ä¸įæĸŃæī©å¤§(0.05) | ä¹Łåıªæĺ¯(0.05)
    BOT:  J(-0.06) | (code(-0.06) | _iter(-0.06) | -util(-0.05) | '}Ċ(-0.05) | å¹¶ä¸İ(-0.05) | _share(-0.05) | å°±åı¯ä»¥(-0.05)
    ACCEPTED as axis_1132  cumulative_var=0.8384

  [1128]  axes=1133  step_var=0.0014  binary_acc=0.982  gap=0.0945  max_dot=0.0113  (1.9s)
    TOP:  .Entity(0.05) | æīĵå¼Ģ(0.05) | ation(0.05) | on(0.05) | License(0.05) | Pen(0.05) | textarea(0.05) | ãģĭ(0.05)
    BOT:  adas(-0.06) | Works(-0.06) | aire(-0.06) | theorem(-0.05) | :test(-0.05) | Singer(-0.05) | larÄ±n(-0.05) | rollers(-0.05)
    ACCEPTED as axis_1133  cumulative_var=0.8386

  [1129]  axes=1134  step_var=0.0013  binary_acc=0.990  gap=0.0963  max_dot=0.0123  (1.8s)
    TOP:  ateg(0.07) | é©¾è½¦(0.06) | permalink(0.06) | right(0.06) | éĩįè§Ĩ(0.05) | ?>Ċ(0.05) | ankind(0.05) | ä½ľäºĨ(0.05)
    BOT:  prises(-0.07) | ITION(-0.06) | ark(-0.06) | KEY(-0.06) | age(-0.06) | _frame(-0.06) | /s(-0.06) | actions(-0.06)
    ACCEPTED as axis_1134  cumulative_var=0.8388

  [1130]  axes=1135  step_var=0.0013  binary_acc=0.988  gap=0.0917  max_dot=0.0066  (1.9s)
    TOP:  '.Ċ(0.06) | .Rectangle(0.05) | Oops(0.05) | signed(0.05) | _MORE(0.05) | Leaders(0.05) | .writer(0.05) | roc(0.05)
    BOT:  .register(-0.06) | _supported(-0.06) | que(-0.06) | inux(-0.06) | ats(-0.06) | .ul(-0.06) | Show(-0.06) | u(-0.05)
    ACCEPTED as axis_1135  cumulative_var=0.8390

  [1131]  axes=1136  step_var=0.0014  binary_acc=0.998  gap=0.0936  max_dot=0.0054  (1.9s)
    TOP:  ["(0.06) | room(0.06) | .user(0.06) | ÑĢÑĥ(0.06) | Generic(0.06) | ï¼īĊ(0.06) | .T(0.06) | stack(0.06)
    BOT:  newsletter(-0.06) | G(-0.05) | -slider(-0.05) | èĥĮåĲİ(-0.05) | void(-0.05) | .se(-0.05) | subscribe(-0.05) | Stan(-0.05)
    ACCEPTED as axis_1136  cumulative_var=0.8392

  [1132]  axes=1137  step_var=0.0014  binary_acc=0.996  gap=0.0938  max_dot=0.0047  (1.8s)
    TOP:  Channel(0.06) | view(0.06) | RM(0.06) | IFF(0.06) | æīĭ(0.05) | dd(0.05) | åİŁæłĩé¢ĺ(0.05) | é«ĺå±±(0.05)
    BOT:  import(-0.07) | By(-0.06) | prÃ©(-0.06) | In(-0.06) | Ð¼ÐµÑĤ(-0.05) | ("\(-0.05) | ãĢĬ(-0.05) | .dropout(-0.05)
    ACCEPTED as axis_1137  cumulative_var=0.8395

  [1133]  axes=1138  step_var=0.0013  binary_acc=0.998  gap=0.0924  max_dot=0.0027  (1.8s)
    TOP:  éĩİ(0.07) | å¯¹æĪĳä»¬(0.06) | å¤·(0.06) | Culture(0.06) | è¶Ĭ(0.06) | éļı(0.06) | æľĽçĿĢ(0.05) | è£Ĥ(0.05)
    BOT:  -header(-0.06) | V(-0.06) | cheduling(-0.05) | fly(-0.05) | .Named(-0.05) | _ng(-0.05) | ints(-0.05) | their(-0.05)
    ACCEPTED as axis_1138  cumulative_var=0.8397

  [1134]  axes=1139  step_var=0.0013  binary_acc=0.980  gap=0.0917  max_dot=0.0069  (1.8s)
    TOP:  /gui(0.06) | ")ĊĊ(0.06) | ä¸»åĬ¨æĢ§(0.05) | >ĊĊĊ(0.05) | solution(0.05) | item(0.05) | åĩºå¢ĥ(0.05) | .NotFound(0.05)
    BOT:  fan(-0.06) | ìŀ¥(-0.06) | =utf(-0.05) | .log(-0.05) | .win(-0.05) | /products(-0.05) | .ag(-0.05) | .ask(-0.05)
    ACCEPTED as axis_1139  cumulative_var=0.8399

  [1135]  axes=1140  step_var=0.0013  binary_acc=0.986  gap=0.0929  max_dot=0.0070  (1.9s)
    TOP:  -W(0.06) | å¤ĦçĲĨ(0.06) | _db(0.05) | -logo(0.05) | estimated(0.05) | äº(0.05) | DAMAGE(0.05) | Scope(0.05)
    BOT:  _color(-0.06) | .placeholder(-0.06) | dependencies(-0.06) | .nn(-0.05) | ç¼Ŀ(-0.05) | .perform(-0.05) | Serializer(-0.05) | }Ċ(-0.05)
    ACCEPTED as axis_1140  cumulative_var=0.8401

  [1136]  axes=1141  step_var=0.0013  binary_acc=0.950  gap=0.0919  max_dot=0.0102  (1.9s)
    TOP:  .linear(0.06) | .attach(0.05) | _status(0.05) | .pl(0.05) | (xx(0.05) | _symbols(0.05) | _args(0.05) | /xml(0.05)
    BOT:  æĤ£èĢħ(-0.07) | âĢĿ(-0.06) | &&(-0.06) | åĩºæĿ¥çļĦ(-0.06) | /Public(-0.06) | _cli(-0.06) | å¯ĨåĪĩ(-0.06) | å¯Ĥå¯ŀ(-0.05)
    ACCEPTED as axis_1141  cumulative_var=0.8403

  [1137]  axes=1142  step_var=0.0014  binary_acc=0.984  gap=0.0937  max_dot=0.0026  (1.8s)
    TOP:  çĶŁçī©(0.06) | (chunk(0.06) | _missing(0.05) | ä¸ºéĩįçĤ¹(0.05) | IT(0.05) | ervice(0.05) | document(0.05) | Sustainability(0.05)
    BOT:  åħ¬ç«ĭ(-0.06) | .Button(-0.06) | plement(-0.06) | filter(-0.06) | SPI(-0.06) | .Card(-0.06) | vent(-0.05) | Theory(-0.05)
    ACCEPTED as axis_1142  cumulative_var=0.8405

  [1138]  axes=1143  step_var=0.0013  binary_acc=0.994  gap=0.0914  max_dot=0.0141  (1.8s)
    TOP:  ##(0.06) | Hand(0.06) | cm(0.06) | encrypted(0.06) | motor(0.05) | ova(0.05) | .Power(0.05) | Son(0.05)
    BOT:  ä»¿ä½Ľ(-0.06) | numerator(-0.06) | basename(-0.06) | .es(-0.06) | PV(-0.06) | RuntimeError(-0.06) | Stay(-0.06) | compose(-0.05)
    ACCEPTED as axis_1143  cumulative_var=0.8408

  [1139]  axes=1144  step_var=0.0014  binary_acc=0.994  gap=0.0933  max_dot=0.0087  (1.8s)
    TOP:  ///(0.07) | Handler(0.06) | #ifndef(0.05) | >/<(0.05) | Example(0.05) | Military(0.05) | è¡ĮæĶ¿æī§æ³ķ(0.05) | Business(0.05)
    BOT:  .Plugin(-0.07) | zug(-0.06) | _creation(-0.06) | ÐµÐ½Ð¸Ñı(-0.05) | .Transaction(-0.05) | .pub(-0.05) | _states(-0.05) | [word(-0.05)
    ACCEPTED as axis_1144  cumulative_var=0.8410

  [1140]  axes=1145  step_var=0.0014  binary_acc=0.949  gap=0.0932  max_dot=0.0065  (1.8s)
    TOP:  ']=(0.06) | Proto(0.06) | .Simple(0.06) | infinite(0.06) | .pg(0.05) | reset(0.05) | ROUT(0.05) | event(0.05)
    BOT:  Word(-0.06) | Ã¤(-0.06) | TAG(-0.06) | use(-0.06) | iman(-0.06) | kip(-0.05) | IFT(-0.05) | le(-0.05)
    ACCEPTED as axis_1145  cumulative_var=0.8412

  [1141]  axes=1146  step_var=0.0014  binary_acc=0.990  gap=0.0923  max_dot=0.0007  (1.8s)
    TOP:  åĲįåŃĹ(0.07) | .services(0.07) | _NODE(0.06) | /F(0.06) | åı¤ä»£(0.06) | .me(0.06) | Without(0.06) | W(0.06)
    BOT:  set(-0.07) | ÐµÐ½Ð°(-0.06) | åĩĭ(-0.06) | Ð»Ð¾Ð²(-0.05) | _connect(-0.05) | Basis(-0.05) | interrupt(-0.05) | cale(-0.05)
    ACCEPTED as axis_1146  cumulative_var=0.8414

  [1142]  axes=1147  step_var=0.0014  binary_acc=0.992  gap=0.0958  max_dot=0.0027  (1.8s)
    TOP:  è´Ń(0.07) | about(0.07) | ilinear(0.06) | .map(0.06) | serialize(0.06) | ìłĲ(0.06) | æ¸¸(0.06) | economic(0.06)
    BOT:  kb(-0.06) | Gamma(-0.05) | _abs(-0.05) | fox(-0.05) | referred(-0.05) | (),(-0.05) | f(-0.05) | .,Ċ(-0.05)
    ACCEPTED as axis_1147  cumulative_var=0.8416

  [1143]  axes=1148  step_var=0.0014  binary_acc=0.997  gap=0.0932  max_dot=0.0011  (1.8s)
    TOP:  Tracker(0.06) | agation(0.06) | _depend(0.06) | oo(0.05) | baseUrl(0.05) | (in(0.05) | Ch(0.05) | -meta(0.05)
    BOT:  å¼¹(-0.06) | ")Ċ(-0.06) | .util(-0.06) | _factor(-0.06) | browser(-0.05) | æľįåĬ¡å¹³åı°(-0.05) | è½½(-0.05) | _sizes(-0.05)
    ACCEPTED as axis_1148  cumulative_var=0.8418

  [1144]  axes=1149  step_var=0.0014  binary_acc=0.985  gap=0.0948  max_dot=0.0015  (1.8s)
    TOP:  olated(0.06) | For(0.05) | '):Ċ(0.05) | LICENSE(0.05) | ILITIES(0.05) | )ĊĊĊ(0.05) | Aggregate(0.05) | ...(0.05)
    BOT:  du(-0.06) | versation(-0.06) | _conv(-0.06) | failed(-0.06) | =db(-0.05) | è°Ĵ(-0.05) | _patient(-0.05) | DESCRIPTION(-0.05)
    ACCEPTED as axis_1149  cumulative_var=0.8421

  [1145]  axes=1150  step_var=0.0014  binary_acc=0.986  gap=0.0928  max_dot=0.0097  (1.9s)
    TOP:  ');Ċ(0.06) | L(0.06) | ']))(0.05) | ignty(0.05) | âĢĿãĢĤ(0.05) | ILLA(0.05) | .Domain(0.05) | Binding(0.05)
    BOT:  tog(-0.06) | >ĊĊ(-0.06) | éĢ²(-0.06) | æī¾åĪ°(-0.05) | åĪĨåĪ«ä¸º(-0.05) | äºĨä¸Ģä¸ª(-0.05) | imonial(-0.05) | _remain(-0.05)
    ACCEPTED as axis_1150  cumulative_var=0.8423

  [1146]  axes=1151  step_var=0.0014  binary_acc=0.988  gap=0.0928  max_dot=0.0082  (1.9s)
    TOP:  R(0.05) | .db(0.05) | Managed(0.05) | _directory(0.05) | ocks(0.05) | factors(0.05) | )".(0.05) | éĩĳèŀįæľºæŀĦ(0.05)
    BOT:  é£ŀè¡Į(-0.06) | :n(-0.06) | /python(-0.06) | p(-0.06) | Bootstrap(-0.06) | åĲ§(-0.05) | '",Ċ(-0.05) | (use(-0.05)
    ACCEPTED as axis_1151  cumulative_var=0.8425

  [1147]  axes=1152  step_var=0.0014  binary_acc=0.993  gap=0.0898  max_dot=0.0104  (1.8s)
    TOP:  ro(0.07) | Clean(0.06) | Selector(0.06) | _no(0.06) | Repository(0.05) | å½Ĵå±ŀäºİ(0.05) | Admin(0.05) | rss(0.05)
    BOT:  çŁ³(-0.06) | å¤±(-0.06) | .base(-0.06) | éĢłæĪĲ(-0.05) | Problem(-0.05) | èĩĤ(-0.05) | åı²(-0.05) | MING(-0.05)
    ACCEPTED as axis_1152  cumulative_var=0.8427

  [1148]  axes=1153  step_var=0.0014  binary_acc=0.996  gap=0.0932  max_dot=0.0065  (1.8s)
    TOP:  ['(0.06) | Plot(0.06) | &(0.06) | è®¸å¤ļ(0.06) | =d(0.06) | acker(0.06) | ]);ĊĊ(0.06) | |ĊĊ(0.06)
    BOT:  Logo(-0.06) | .notifications(-0.06) | Port(-0.06) | _log(-0.06) | resource(-0.05) | Bra(-0.05) | napshot(-0.05) | Ð°Ðº(-0.05)
    ACCEPTED as axis_1153  cumulative_var=0.8429

  [1149]  axes=1154  step_var=0.0014  binary_acc=0.993  gap=0.0956  max_dot=0.0063  (1.8s)
    TOP:  /id(0.05) | _devices(0.05) | OTHERWISE(0.05) | (style(0.05) | _validation(0.05) | Encoding(0.05) | restoration(0.05) | _ex(0.05)
    BOT:  animals(-0.07) | åºĻ(-0.07) | æĦŁåĨĴ(-0.06) | èĢħ(-0.06) | èīº(-0.06) | ä¸İ(-0.06) | åĮ»æĬ¤äººåĳĺ(-0.05) | å¤©èĬ±(-0.05)
    ACCEPTED as axis_1154  cumulative_var=0.8431

  [1150]  axes=1155  step_var=0.0014  binary_acc=0.998  gap=0.0919  max_dot=0.0051  (1.9s)
    TOP:  (page(0.07) | _metadata(0.06) | ÙĪÙĦ(0.06) | ()"Ċ(0.06) | Ð»ÐµÐ¼ÐµÐ½ÑĤ(0.06) | _username(0.05) | çĪ·(0.05) | Ã¡s(0.05)
    BOT:  =False(-0.07) | vest(-0.06) | had(-0.06) | ä¸Ģåľº(-0.06) | æĸ°æĬĢæľ¯(-0.06) | _numpy(-0.05) | mysql(-0.05) | æķ°é¢Ŀ(-0.05)
    ACCEPTED as axis_1155  cumulative_var=0.8433

  [1151]  axes=1156  step_var=0.0014  binary_acc=0.989  gap=0.0916  max_dot=0.0079  (1.9s)
    TOP:  .Play(0.06) | _TARGET(0.06) | _escape(0.06) | .ep(0.06) | åİĤåķĨ(0.06) | Review(0.06) | _cuda(0.05) | çİĭåºľ(0.05)
    BOT:  _epsilon(-0.06) | é¢ĳ(-0.06) | _property(-0.06) | IFO(-0.05) | keyword(-0.05) | -answer(-0.05) | lam(-0.05) | [index(-0.05)
    ACCEPTED as axis_1156  cumulative_var=0.8436

  [1152]  axes=1157  step_var=0.0014  binary_acc=0.993  gap=0.0924  max_dot=0.0144  (1.9s)
    TOP:  -*-ĊĊ(0.06) | Under(0.06) | AH(0.06) | latin(0.06) | cache(0.06) | carousel(0.06) | æ´Ľ(0.05) | :.(0.05)
    BOT:  _BUCKET(-0.06) | (dtype(-0.06) | æĢĢ(-0.06) | _hook(-0.06) | v(-0.06) | .domain(-0.05) | Status(-0.05) | '(-0.05)
    ACCEPTED as axis_1157  cumulative_var=0.8438

  [1153]  axes=1158  step_var=0.0014  binary_acc=0.997  gap=0.0928  max_dot=0.0023  (1.9s)
    TOP:  Probability(0.06) | (scale(0.06) | _attributes(0.06) | cciÃ³n(0.06) | =find(0.05) | Comments(0.05) | metric(0.05) | {}ĊĊ(0.05)
    BOT:  .reshape(-0.06) | .bootstrap(-0.06) | æ³¨(-0.05) | Paginator(-0.05) | _windows(-0.05) | ITCH(-0.05) | tour(-0.05) | config(-0.05)
    ACCEPTED as axis_1158  cumulative_var=0.8440

  [1154]  axes=1159  step_var=0.0014  binary_acc=0.998  gap=0.0904  max_dot=0.0102  (1.9s)
    TOP:  åĽ½(0.06) | .evaluate(0.05) | here(0.05) | éŁ³ä¹Ĳ(0.05) | -download(0.05) | )(0.05) | åĬ£(0.05) | _stack(0.05)
    BOT:  Via(-0.06) | Part(-0.06) | åŃķæľŁ(-0.06) | _metrics(-0.06) | (z(-0.06) | "P(-0.06) | query(-0.06) | So(-0.06)
    ACCEPTED as axis_1159  cumulative_var=0.8442

  [1155]  axes=1160  step_var=0.0014  binary_acc=0.988  gap=0.0928  max_dot=0.0180  (1.9s)
    TOP:  .em(0.06) | focus(0.06) | NAS(0.06) | /resource(0.06) | _con(0.06) | .h(0.05) | /welcome(0.05) | Adapter(0.05)
    BOT:  Ã´(-0.05) | åĽ¾(-0.05) | PCR(-0.05) | Outside(-0.05) | urons(-0.05) | åįİ(-0.05) | æĬĬæĪĳ(-0.05) | åĮº(-0.05)
    ACCEPTED as axis_1160  cumulative_var=0.8444

  [1156]  axes=1161  step_var=0.0014  binary_acc=0.979  gap=0.0904  max_dot=0.0072  (1.9s)
    TOP:  å¥¹ä»¬(0.06) | OLER(0.06) | tok(0.06) | .querySelector(0.05) | vÃŃdeo(0.05) | Side(0.05) | _wr(0.05) | _INPUT(0.05)
    BOT:  .parameters(-0.06) | (String(-0.06) | èĦļ(-0.06) | No(-0.06) | (indent(-0.06) | .element(-0.06) | .seed(-0.05) | éĽĩ(-0.05)
    ACCEPTED as axis_1161  cumulative_var=0.8446

  [1157]  axes=1162  step_var=0.0014  binary_acc=0.984  gap=0.0913  max_dot=0.0187  (1.8s)
    TOP:  Module(0.07) | Attack(0.06) | ={"(0.06) | hardware(0.06) | weak(0.06) | çº¿(0.05) | ä¹ĭæĦı(0.05) | /global(0.05)
    BOT:  versions(-0.07) | datasets(-0.06) | gregate(-0.05) | inecraft(-0.05) | itat(-0.05) | ely(-0.05) | (dead(-0.05) | ÙģÙĬ(-0.05)
    ACCEPTED as axis_1162  cumulative_var=0.8448

  [1158]  axes=1163  step_var=0.0014  binary_acc=0.968  gap=0.0908  max_dot=0.0083  (1.8s)
    TOP:  ()ĊĊĊ(0.06) | å§Ķ(0.06) | ä¸ĢæĹ¥(0.06) | ])Ċ(0.06) | /api(0.06) | .type(0.06) | )).Ċ(0.06) | .out(0.05)
    BOT:  =${(-0.05) | =User(-0.05) | _ADD(-0.05) | .Function(-0.05) | æĦŁæŁĵ(-0.05) | _K(-0.05) | .N(-0.05) | dpi(-0.05)
    ACCEPTED as axis_1163  cumulative_var=0.8450

  [1159]  axes=1164  step_var=0.0014  binary_acc=0.991  gap=0.0937  max_dot=0.0092  (1.9s)
    TOP:  è¿Ļä¸ªäºº(0.05) | Adapter(0.05) | èµ·è¯ī(0.05) | elmet(0.05) | åħīæĺİ(0.05) | .field(0.05) | HN(0.05) | vis(0.05)
    BOT:  æ°ĳ(-0.07) | Values(-0.07) | `(-0.06) | Media(-0.06) | ËĪ(-0.06) | èħ¿(-0.06) | Bravo(-0.06) | angling(-0.05)
    ACCEPTED as axis_1164  cumulative_var=0.8453

  [1160]  axes=1165  step_var=0.0014  binary_acc=0.964  gap=0.0891  max_dot=0.0069  (1.9s)
    TOP:  (PORT(0.06) | /software(0.06) | ãĥķãĤ¡ãĤ¤ãĥ«(0.05) | annotate(0.05) | Ã©c(0.05) | .request(0.05) | .ts(0.05) | -item(0.05)
    BOT:  .t(-0.07) | g(-0.06) | åŁŁåĲį(-0.06) | ,g(-0.06) | Minimum(-0.06) | çļĦä¿¡ä»»(-0.05) | })Ċ(-0.05) | .scope(-0.05)
    ACCEPTED as axis_1165  cumulative_var=0.8455

  [1161]  axes=1166  step_var=0.0014  binary_acc=0.988  gap=0.0930  max_dot=0.0008  (1.8s)
    TOP:  ÑĢ(0.06) | ä»¥å¤ĸ(0.06) | .socket(0.06) | .x(0.05) | l(0.05) | ckett(0.05) | /common(0.05) | ===(0.05)
    BOT:  Separator(-0.06) | olut(-0.05) | Applied(-0.05) | æĺ¯ä¸Ģåº§(-0.05) | logged(-0.05) | :m(-0.05) | Ð¾Ð±ÑĬ(-0.05) | Pods(-0.05)
    ACCEPTED as axis_1166  cumulative_var=0.8457

  [1162]  axes=1167  step_var=0.0014  binary_acc=0.981  gap=0.0919  max_dot=0.0193  (1.9s)
    TOP:  +a(0.06) | "E(0.06) | hlen(0.06) | .J(0.06) | peats(0.06) | .Pro(0.06) | .chrome(0.06) | -id(0.06)
    BOT:  _save(-0.06) | At(-0.05) | Script(-0.05) | He(-0.05) | .Token(-0.05) | _deploy(-0.05) | getFile(-0.05) | .Plugin(-0.05)
    ACCEPTED as axis_1167  cumulative_var=0.8459

  [1163]  axes=1168  step_var=0.0014  binary_acc=0.991  gap=0.0917  max_dot=0.0062  (1.8s)
    TOP:  çĪ±(0.06) | å±Ģ(0.06) | (width(0.06) | .team(0.06) | Ð¿Ð¾Ð»Ð¸ÑĤÐ¸Ðº(0.05) | çľĭæ³ķ(0.05) | factors(0.05) | from(0.05)
    BOT:  :čĊ(-0.06) | (Ċ(-0.06) | u(-0.06) | _initial(-0.06) | .čĊčĊ(-0.05) | uv(-0.05) | æ(-0.05) | Th(-0.05)
    ACCEPTED as axis_1168  cumulative_var=0.8461

  [1164]  axes=1169  step_var=0.0014  binary_acc=0.952  gap=0.0920  max_dot=0.0146  (1.9s)
    TOP:  OF(0.07) | .gray(0.06) | """ĊĊ(0.06) | -validation(0.05) | .pipeline(0.05) | .control(0.05) | åĿł(0.05) | ROW(0.05)
    BOT:  wd(-0.07) | _target(-0.06) | .Service(-0.06) | kt(-0.05) | Metro(-0.05) | ÑĢ(-0.05) | jb(-0.05) | Boxes(-0.05)
    ACCEPTED as axis_1169  cumulative_var=0.8463

  [1165]  axes=1170  step_var=0.0014  binary_acc=0.972  gap=0.0903  max_dot=0.0109  (1.9s)
    TOP:  äº(0.06) | -,(0.06) | åŀĴ(0.06) | pis(0.06) | _LIBRARY(0.05) | .TYPE(0.05) | _normal(0.05) | åī§(0.05)
    BOT:  expression(-0.06) | _flags(-0.06) | ents(-0.06) | _freq(-0.05) | ÐºÐ¾Ð¼Ð¿Ð°Ð½Ð¸Ð¸(-0.05) | -theme(-0.05) | submit(-0.05) | .lib(-0.05)
    ACCEPTED as axis_1170  cumulative_var=0.8465

  [1166]  axes=1171  step_var=0.0014  binary_acc=0.979  gap=0.0914  max_dot=0.0018  (1.9s)
    TOP:  Evalu(0.06) | Pr(0.06) | .Network(0.05) | .local(0.05) | ä½Ĩä¸įéĻĲäºİ(0.05) | ĉĉ(0.05) | nx(0.05) | Agent(0.05)
    BOT:  çļĦæĺ¯(-0.06) | ration(-0.06) | å¤ľéĹ´(-0.06) | -ups(-0.06) | ipped(-0.06) | iability(-0.05) | gal(-0.05) | etag(-0.05)
    ACCEPTED as axis_1171  cumulative_var=0.8467

  [1167]  axes=1172  step_var=0.0014  binary_acc=0.994  gap=0.0913  max_dot=0.0102  (1.9s)
    TOP:  .testing(0.06) | (ab(0.05) | limits(0.05) | _static(0.05) | _ship(0.05) | andoned(0.05) | atie(0.05) | .getContent(0.05)
    BOT:  Are(-0.06) | ['-(-0.06) | åħŃ(-0.06) | '),ĊĊ(-0.06) | :X(-0.06) | ÑıÑĤÑģÑı(-0.06) | å¿Ĺ(-0.05) | _cli(-0.05)
    ACCEPTED as axis_1172  cumulative_var=0.8470

  [1168]  axes=1173  step_var=0.0014  binary_acc=0.983  gap=0.0909  max_dot=0.0021  (2.0s)
    TOP:  Â»(0.07) | omentum(0.06) | ?</(0.06) | AUT(0.06) | '}Ċ(0.06) | coln(0.06) | );ĊĊ(0.06) | utils(0.06)
    BOT:  .theme(-0.05) | Sizer(-0.05) | _normalize(-0.05) | .test(-0.05) | .setdefault(-0.05) | åİ¢(-0.05) | ...,(-0.05) | éĢĤå½ĵçļĦ(-0.05)
    ACCEPTED as axis_1173  cumulative_var=0.8472

  [1169]  axes=1174  step_var=0.0014  binary_acc=0.986  gap=0.0928  max_dot=0.0047  (1.8s)
    TOP:  dt(0.06) | /go(0.06) | known(0.05) | etections(0.05) | è¯»(0.05) | Primitive(0.05) | sparse(0.05) | -dev(0.05)
    BOT:  ALE(-0.06) | èĮĥ(-0.06) | urray(-0.05) | çļĦçī¹çĤ¹(-0.05) | æ½®(-0.05) | .tech(-0.05) | éłħ(-0.05) | _AL(-0.05)
    ACCEPTED as axis_1174  cumulative_var=0.8474

  [1170]  axes=1175  step_var=0.0014  binary_acc=0.996  gap=0.0902  max_dot=0.0042  (1.8s)
    TOP:  IFIED(0.06) | _by(0.05) | .a(0.05) | id(0.05) | web(0.05) | bing(0.05) | æĹ·(0.05) | èĢĮ(0.05)
    BOT:  _LOCATION(-0.07) | Policy(-0.06) | è¯¦è§£(-0.06) | .pt(-0.05) | -Out(-0.05) | _layer(-0.05) | +r(-0.05) | èĤ¡ç¥¨(-0.05)
    ACCEPTED as axis_1175  cumulative_var=0.8476

  [1171]  axes=1176  step_var=0.0014  binary_acc=0.985  gap=0.0915  max_dot=0.0014  (1.9s)
    TOP:  .sequence(0.05) | Signal(0.05) | TU(0.05) | .padding(0.05) | cannot(0.05) | ius(0.05) | ishes(0.05) | iste(0.05)
    BOT:  æĮĩå¯¼ä¸ĭ(-0.05) | .Main(-0.05) | è¦ĸ(-0.05) | /archive(-0.05) | Lic(-0.05) | Ans(-0.05) | (epoch(-0.05) | IGH(-0.05)
    ACCEPTED as axis_1176  cumulative_var=0.8478

  [1172]  axes=1177  step_var=0.0014  binary_acc=0.995  gap=0.0908  max_dot=0.0281  (1.8s)
    TOP:  (op(0.06) | .code(0.06) | .exceptions(0.06) | ende(0.05) | ĉf(0.05) | .l(0.05) | .json(0.05) | .config(0.05)
    BOT:  (username(-0.06) | ä»į(-0.06) | ÑĤÑĭ(-0.06) | not(-0.06) | iframe(-0.06) | ma(-0.06) | iles(-0.05) | ll(-0.05)
    ACCEPTED as axis_1177  cumulative_var=0.8480

  [1173]  axes=1178  step_var=0.0014  binary_acc=0.999  gap=0.0921  max_dot=0.0117  (1.9s)
    TOP:  >](0.06) | _filt(0.06) | pheres(0.06) | ie(0.06) | ply(0.06) | æ·¡(0.06) | .el(0.05) | .HTTP(0.05)
    BOT:  $f(-0.06) | (-0.06) | invention(-0.05) | ":"(-0.05) | };(-0.05) | #[(-0.05) | });ĊĊ(-0.05) | -hover(-0.05)
    ACCEPTED as axis_1178  cumulative_var=0.8482

  [1174]  axes=1179  step_var=0.0014  binary_acc=0.996  gap=0.0914  max_dot=0.0246  (1.9s)
    TOP:  Check(0.06) | Real(0.06) | ç§»åĬ¨(0.06) | æĸĩçī©(0.06) | .description(0.05) | .adapter(0.05) | _embed(0.05) | sj(0.05)
    BOT:  /port(-0.06) | æĪĲéķ¿ä¸º(-0.05) | DIST(-0.05) | Value(-0.05) | .finish(-0.05) | å¯¹å¾ħ(-0.05) | -to(-0.05) | ä»»åĬ¡(-0.05)
    ACCEPTED as axis_1179  cumulative_var=0.8484

  [1175]  axes=1180  step_var=0.0014  binary_acc=0.982  gap=0.0911  max_dot=0.0065  (1.8s)
    TOP:  Ã¶n(0.06) | _OBJECT(0.06) | _CONTROL(0.06) | ç»¼åĲĪ(0.06) | .ident(0.05) | oid(0.05) | ä¸Ģå¤§(0.05) | éĵĿ(0.05)
    BOT:  List(-0.05) | THE(-0.05) | ãĤĤ(-0.05) | squares(-0.05) | å¯¹(-0.05) | ä»»ä½ķ(-0.05) | æĪĳå®¶(-0.05) | åĩ¡(-0.05)
    ACCEPTED as axis_1180  cumulative_var=0.8486

  [1176]  axes=1181  step_var=0.0014  binary_acc=0.966  gap=0.0925  max_dot=0.0097  (2.0s)
    TOP:  .Data(0.08) | On(0.06) | /comment(0.05) | è®¡åĪĴ(0.05) | .Button(0.05) | _address(0.05) | .face(0.05) | By(0.05)
    BOT:  :block(-0.06) | events(-0.05) | way(-0.05) | Sym(-0.05) | fact(-0.05) | -phone(-0.05) | .mongodb(-0.05) | åºı(-0.05)
    ACCEPTED as axis_1181  cumulative_var=0.8489

  [1177]  axes=1182  step_var=0.0014  binary_acc=0.995  gap=0.0896  max_dot=0.0120  (1.8s)
    TOP:  ):čĊ(0.06) | Fi(0.06) | directory(0.05) | }&(0.05) | [](0.05) | Authenticated(0.05) | .Cl(0.05) | .findAll(0.05)
    BOT:  inz(-0.06) | .kernel(-0.06) | _mult(-0.06) | .loader(-0.06) | isel(-0.06) | lication(-0.06) | Cookies(-0.06) | æ¢¦æĥ³(-0.06)
    ACCEPTED as axis_1182  cumulative_var=0.8491

  [1178]  axes=1183  step_var=0.0014  binary_acc=0.994  gap=0.0906  max_dot=0.0138  (1.9s)
    TOP:  '<(0.06) | .features(0.06) | find(0.06) | ::(0.05) | .Tasks(0.05) | -disable(0.05) | .logging(0.05) | .exists(0.05)
    BOT:  wi(-0.07) | poses(-0.06) | SM(-0.06) | Lang(-0.06) | oy(-0.06) | ensing(-0.06) | æĬķç¨¿(-0.05) | Division(-0.05)
    ACCEPTED as axis_1183  cumulative_var=0.8493

  [1179]  axes=1184  step_var=0.0014  binary_acc=0.978  gap=0.0930  max_dot=0.0104  (1.8s)
    TOP:  è¿Ľè¡ĮäºĨ(0.06) | C(0.06) | ä¸ŃåĽ½çļĦ(0.06) | è´¹çĶ¨(0.05) | ){Ċ(0.05) | âĢĿ(0.05) | Erd(0.05) | vable(0.05)
    BOT:  nowrap(-0.06) | uler(-0.05) | _STRING(-0.05) | Tokens(-0.05) | ¦(-0.05) | metadata(-0.05) | LEM(-0.05) | DEF(-0.05)
    ACCEPTED as axis_1184  cumulative_var=0.8495

  [1180]  axes=1185  step_var=0.0014  binary_acc=0.996  gap=0.0909  max_dot=0.0050  (1.9s)
    TOP:  /plugin(0.06) | =json(0.05) | Acc(0.05) | Upgrade(0.05) | å¤©(0.05) | Mas(0.05) | Terminate(0.05) | session(0.05)
    BOT:  Bad(-0.06) | /auto(-0.06) | igua(-0.05) | LF(-0.05) | _event(-0.05) | metadata(-0.05) | (send(-0.05) | ())Ċ(-0.05)
    ACCEPTED as axis_1185  cumulative_var=0.8497

  [1181]  axes=1186  step_var=0.0014  binary_acc=0.988  gap=0.0906  max_dot=0.0044  (1.9s)
    TOP:  =lambda(0.06) | [j(0.05) | P(0.05) | åĪĺæŁĲ(0.05) | _cpu(0.05) | ===(0.05) | [i(0.05) | Helper(0.05)
    BOT:  Agr(-0.05) | oded(-0.05) | manda(-0.05) | åĩ½æķ°(-0.05) | zone(-0.05) | å¼ĢæľĹ(-0.05) | mina(-0.05) | have(-0.05)
    ACCEPTED as axis_1186  cumulative_var=0.8499

  [1182]  axes=1187  step_var=0.0014  binary_acc=0.986  gap=0.0904  max_dot=0.0071  (1.9s)
    TOP:  æĿ¡ä»¶(0.06) | peating(0.06) | ].ĊĊ(0.06) | _batch(0.06) | _AND(0.05) | è¿Ļä¸ĢçĤ¹(0.05) | If(0.05) | }`}(0.05)
    BOT:  (.(-0.06) | TN(-0.05) | etween(-0.05) | .protocol(-0.05) | Airport(-0.05) | Sdk(-0.05) | osen(-0.05) | itize(-0.05)
    ACCEPTED as axis_1187  cumulative_var=0.8501

  [1183]  axes=1188  step_var=0.0014  binary_acc=0.992  gap=0.0939  max_dot=0.0009  (1.9s)
    TOP:  ...Ċ(0.06) | æĶ¯è¡Į(0.05) | _ctrl(0.05) | .tensor(0.05) | ati(0.05) | âĢĿ:(0.05) | æ²³åĮĹçľģ(0.05) | çĿĢæĢ¥(0.05)
    BOT:  /types(-0.06) | mma(-0.06) | vibrations(-0.05) | _DEFAULT(-0.05) | _diff(-0.05) | xmin(-0.05) | freq(-0.05) | Per(-0.05)
    ACCEPTED as axis_1188  cumulative_var=0.8503

  [1184]  axes=1189  step_var=0.0014  binary_acc=0.983  gap=0.0898  max_dot=0.0101  (1.9s)
    TOP:  ald(0.06) | Rule(0.06) | mt(0.06) | oh(0.06) | /code(0.05) | ws(0.05) | Ti(0.05) | _URI(0.05)
    BOT:  .cal(-0.06) | è´ŀ(-0.06) | Ð·Ð°Ð¿Ð¸ÑģÐ¸(-0.05) | .Delete(-0.05) | `(-0.05) | [label(-0.05) | ",(-0.05) | Tuple(-0.05)
    ACCEPTED as axis_1189  cumulative_var=0.8505

  [1185]  axes=1190  step_var=0.0014  binary_acc=0.980  gap=0.0918  max_dot=0.0060  (1.8s)
    TOP:  im(0.07) | ä¸ºäºº(0.06) | /common(0.06) | ve(0.06) | ãĥĨ(0.06) | artment(0.06) | èĽĭçĻ½(0.05) | èĲ¥è¿Ĳ(0.05)
    BOT:  /re(-0.06) | Skip(-0.06) | '];ĊĊ(-0.06) | !ĊĊ(-0.06) | odes(-0.06) | ();(-0.06) | ä»Ģä¹Ī(-0.05) | ä¸ľ(-0.05)
    ACCEPTED as axis_1190  cumulative_var=0.8507

  [1186]  axes=1191  step_var=0.0014  binary_acc=0.998  gap=0.0894  max_dot=0.0032  (1.9s)
    TOP:  f(0.06) | IST(0.06) | \/(0.05) | mo(0.05) | non(0.05) | P(0.05) | m(0.05) | _AND(0.05)
    BOT:  çĽ¸äºĴ(-0.06) | .manager(-0.06) | MBER(-0.06) | .image(-0.06) | (User(-0.06) | ä½İåİĭ(-0.06) | è½¬åĬ¨(-0.05) | .connect(-0.05)
    ACCEPTED as axis_1191  cumulative_var=0.8509

  [1187]  axes=1192  step_var=0.0014  binary_acc=0.990  gap=0.0919  max_dot=0.0179  (1.9s)
    TOP:  (sql(0.06) | (len(0.06) | (window(0.06) | been(0.05) | null(0.05) | _system(0.05) | /py(0.05) | .one(0.05)
    BOT:  oose(-0.06) | IOR(-0.06) | å·¥ç¨ĭ(-0.05) | åĮĹ(-0.05) | èµ·æĿ¥(-0.05) | Ð½Ð¸(-0.05) | locals(-0.05) | aar(-0.05)
    ACCEPTED as axis_1192  cumulative_var=0.8512

  [1188]  axes=1193  step_var=0.0014  binary_acc=0.994  gap=0.0904  max_dot=0.0152  (1.8s)
    TOP:  J(0.07) | Ho(0.06) | .secret(0.06) | å·²æĺ¯(0.06) | _single(0.06) | .Response(0.05) | /LICENSE(0.05) | /design(0.05)
    BOT:  Ð²ÐµÑĢ(-0.06) | tensorflow(-0.06) | _box(-0.06) | ocoa(-0.06) | english(-0.05) | _helper(-0.05) | also(-0.05) | op(-0.05)
    ACCEPTED as axis_1193  cumulative_var=0.8514

  [1189]  axes=1194  step_var=0.0014  binary_acc=0.992  gap=0.0911  max_dot=0.0131  (2.0s)
    TOP:  Analyzer(0.05) | isors(0.05) | Requires(0.05) | trace(0.05) | è¾ĥéķ¿(0.05) | SPA(0.05) | -properties(0.05) | Disabled(0.05)
    BOT:  });Ċ(-0.06) | åİ¿å§Ķ(-0.06) | fr(-0.06) | ;Ċ(-0.06) | olls(-0.06) | ãĢĬ(-0.05) | (batch(-0.05) | _SETTINGS(-0.05)
    ACCEPTED as axis_1194  cumulative_var=0.8516

  [1190]  axes=1195  step_var=0.0014  binary_acc=0.993  gap=0.0914  max_dot=0.0164  (1.9s)
    TOP:  Âł(0.08) | .br(0.06) | .k(0.06) | args(0.06) | ={(0.06) | æ°ĳä¸»åħļ(0.05) | .q(0.05) | [K(0.05)
    BOT:  ëĬĶ(-0.06) | auth(-0.06) | æĭ¥æĬ±(-0.05) | ìłĢ(-0.05) | Shape(-0.05) | :</(-0.05) | ÑģÐºÐ¾Ð³Ð¾(-0.05) | Workshop(-0.05)
    ACCEPTED as axis_1195  cumulative_var=0.8518

  [1191]  axes=1196  step_var=0.0014  binary_acc=0.975  gap=0.0912  max_dot=0.0023  (1.9s)
    TOP:  ä»¥(0.08) | .result(0.06) | ating(0.06) | ç¦ı(0.06) | Point(0.06) | Gener(0.05) | .Controller(0.05) | çļĦæĺ¯(0.05)
    BOT:  .client(-0.07) | Sit(-0.06) | istical(-0.06) | _world(-0.05) | .global(-0.05) | .integration(-0.05) | -text(-0.05) | Contact(-0.05)
    ACCEPTED as axis_1196  cumulative_var=0.8520

  [1192]  axes=1197  step_var=0.0014  binary_acc=0.960  gap=0.0911  max_dot=0.0040  (1.9s)
    TOP:  api(0.06) | my(0.05) | Requirements(0.05) | My(0.05) | <a(0.05) | .ap(0.05) | .github(0.05) | .ylabel(0.05)
    BOT:  >")Ċ(-0.07) | ØĮ(-0.06) | '))ĊĊ(-0.06) | >()Ċ(-0.05) | >')Ċ(-0.05) | comfort(-0.05) | _stride(-0.05) | Cas(-0.05)
    ACCEPTED as axis_1197  cumulative_var=0.8522

  [1193]  axes=1198  step_var=0.0014  binary_acc=0.994  gap=0.0924  max_dot=0.0352  (1.8s)
    TOP:  ÐµÑĢÑĭ(0.06) | gn(0.06) | Separ(0.05) | ymin(0.05) | (\(0.05) | Phrase(0.05) | .extract(0.05) | _D(0.05)
    BOT:  ret(-0.06) | å¼ķ(-0.06) | Formatter(-0.06) | ATCH(-0.06) | (area(-0.06) | edia(-0.05) | éĨĴäºĨ(-0.05) | .Com(-0.05)
    ACCEPTED as axis_1198  cumulative_var=0.8524

  [1194]  axes=1199  step_var=0.0014  binary_acc=0.998  gap=0.0918  max_dot=0.0037  (1.9s)
    TOP:  æĪĲè¯Ń(0.06) | _outputs(0.06) | åĽ¾(0.06) | /template(0.05) | sv(0.05) | ,w(0.05) | pipe(0.05) | P(0.05)
    BOT:  .getRoot(-0.06) | æŃ£(-0.05) | å°Ĩè¿İæĿ¥(-0.05) | Bear(-0.05) | -file(-0.05) | .builders(-0.05) | .url(-0.05) | Ð¿Ð¾Ð»Ð¸ÑĤ(-0.05)
    ACCEPTED as axis_1199  cumulative_var=0.8526

  [1195]  axes=1200  step_var=0.0014  binary_acc=0.976  gap=0.0903  max_dot=0.0054  (1.8s)
    TOP:  Sp(0.06) | _upper(0.06) | .Argument(0.05) | calculated(0.05) | comm(0.05) | (lat(0.05) | _LOG(0.05) | iciencies(0.05)
    BOT:  åį´æĺ¯(-0.06) | âĢĿĊĊ(-0.06) | 'd(-0.06) | !ĊĊ(-0.06) | /src(-0.06) | /dist(-0.05) | /*Ċ(-0.05) | _selection(-0.05)
    ACCEPTED as axis_1200  cumulative_var=0.8528

  [1196]  axes=1201  step_var=0.0014  binary_acc=0.987  gap=0.0902  max_dot=0.0155  (1.8s)
    TOP:  '#'(0.06) | Standard(0.06) | .course(0.06) | MarÃŃa(0.05) | blocks(0.05) | Update(0.05) | ";(0.05) | filename(0.05)
    BOT:  owner(-0.06) | LOW(-0.06) | ==Ċ(-0.06) | ulated(-0.06) | attachment(-0.05) | /com(-0.05) | .visual(-0.05) | åº¦(-0.05)
    ACCEPTED as axis_1201  cumulative_var=0.8530

  [1197]  axes=1202  step_var=0.0014  binary_acc=0.970  gap=0.0895  max_dot=0.0033  (1.8s)
    TOP:  ï¼Ī(0.07) | %ï¼Į(0.06) | ((0.06) | Comput(0.06) | -direction(0.05) | ãģĭãģª(0.05) | J(0.05) | .tools(0.05)
    BOT:  .size(-0.07) | ä»¥æŃ¤(-0.06) | _HAS(-0.06) | æĢ¥è¯Ĭ(-0.05) | ^n(-0.05) | AIM(-0.05) | izards(-0.05) | _global(-0.05)
    ACCEPTED as axis_1202  cumulative_var=0.8532

  [1198]  axes=1203  step_var=0.0014  binary_acc=0.977  gap=0.0915  max_dot=0.0103  (1.9s)
    TOP:  ("<(0.06) | _price(0.05) | Tur(0.05) | _SINGLE(0.05) | Linear(0.05) | -License(0.05) | ç»Ļ(0.05) | personal(0.05)
    BOT:  fft(-0.06) | ito(-0.06) | IFIED(-0.06) | hcp(-0.06) | .internal(-0.06) | ences(-0.06) | Inputs(-0.06) | illy(-0.06)
    ACCEPTED as axis_1203  cumulative_var=0.8534

  [1199]  axes=1204  step_var=0.0014  binary_acc=0.988  gap=0.0912  max_dot=0.0018  (1.8s)
    TOP:  flation(0.06) | Ð¸Ð´(0.06) | .state(0.06) | æĿĥéĻĲ(0.06) | XT(0.05) | .Notification(0.05) | Borg(0.05) | çĶ¨äºĨ(0.05)
    BOT:  .Platform(-0.07) | .k(-0.07) | éĻĦ(-0.06) | Source(-0.06) | ä¸įç®¡(-0.06) | ists(-0.06) | bas(-0.06) | ","(-0.06)
    ACCEPTED as axis_1204  cumulative_var=0.8536

  [1200]  axes=1205  step_var=0.0014  binary_acc=0.989  gap=0.0888  max_dot=0.0043  (1.9s)
    TOP:  _CREATE(0.06) | MAIL(0.06) | helper(0.05) | imports(0.05) | #ifndef(0.05) | å¤©(0.05) | Formation(0.05) | æĪ¿ç§Ł(0.05)
    BOT:  _common(-0.06) | åħįéĻ¤(-0.06) | .content(-0.06) | lib(-0.06) | .mark(-0.06) | .weights(-0.06) | example(-0.05) | Ref(-0.05)
    ACCEPTED as axis_1205  cumulative_var=0.8538

  [1201]  axes=1206  step_var=0.0014  binary_acc=0.983  gap=0.0912  max_dot=0.0051  (2.0s)
    TOP:  P(0.07) | .build(0.06) | Ð¡(0.06) | fos(0.06) | .voice(0.05) | tk(0.05) | Optim(0.05) | .linalg(0.05)
    BOT:  _CREATE(-0.06) | æĢĿæĥ³(-0.06) | -Time(-0.06) | _exchange(-0.05) | _projection(-0.05) | _prefix(-0.05) | _PERMISSION(-0.05) | _predict(-0.05)
    ACCEPTED as axis_1206  cumulative_var=0.8540

  [1202]  axes=1207  step_var=0.0014  binary_acc=0.974  gap=0.0908  max_dot=0.0066  (1.9s)
    TOP:  _cross(0.06) | author(0.06) | RAR(0.06) | =edge(0.06) | original(0.06) | conomy(0.05) | Address(0.05) | /profile(0.05)
    BOT:  js(-0.06) | Dynam(-0.05) | çļĦæĺ¯(-0.05) | ä¸įä¾¿(-0.05) | .query(-0.05) | ti(-0.05) | ibrary(-0.05) | _constraints(-0.05)
    ACCEPTED as axis_1207  cumulative_var=0.8542

  [1203]  axes=1208  step_var=0.0014  binary_acc=0.971  gap=0.0910  max_dot=0.0173  (1.8s)
    TOP:  -readable(0.05) | Aj(0.05) | Genetics(0.05) | _vel(0.05) | Model(0.05) | )(0.05) | Ãģ(0.05) | .commands(0.05)
    BOT:  Also(-0.06) | S(-0.06) | ::(-0.06) | /tr(-0.06) | lda(-0.06) | åħ¬å¸ĥ(-0.06) | .Task(-0.05) | ITAL(-0.05)
    ACCEPTED as axis_1208  cumulative_var=0.8545

  [1204]  axes=1209  step_var=0.0014  binary_acc=0.991  gap=0.0867  max_dot=0.0027  (1.8s)
    TOP:  ":Ċ(0.06) | double(0.06) | /single(0.06) | Thirty(0.05) | å£°åĵį(0.05) | ï¼ĮâĢľ(0.05) | ella(0.05) | td(0.05)
    BOT:  C(-0.06) | .Tool(-0.05) | å·²(-0.05) | _OBJECT(-0.05) | atest(-0.05) | Description(-0.05) | -sw(-0.05) | æĢ§åĴĮ(-0.05)
    ACCEPTED as axis_1209  cumulative_var=0.8547

  [1205]  axes=1210  step_var=0.0014  binary_acc=0.978  gap=0.0915  max_dot=0.0170  (1.9s)
    TOP:  (sentence(0.07) | interactive(0.06) | _period(0.05) | (question(0.05) | _by(0.05) | _encoder(0.05) | .state(0.05) | container(0.05)
    BOT:  ")Ċ(-0.06) | #ĊĊ(-0.06) | F(-0.06) | _nodes(-0.06) | ())ĊĊĊ(-0.05) | ]))ĊĊ(-0.05) | ãĥĨãĤ£(-0.05) | __Ċ(-0.05)
    ACCEPTED as axis_1210  cumulative_var=0.8549

  [1206]  axes=1211  step_var=0.0014  binary_acc=0.984  gap=0.0911  max_dot=0.0123  (1.9s)
    TOP:  -o(0.06) | .de(0.06) | E(0.06) | g(0.06) | (d(0.06) | _frame(0.05) | åı°(0.05) | æĸ¯(0.05)
    BOT:  .Cookie(-0.06) | __,(-0.05) | plementary(-0.05) | ================================(-0.05) | (prediction(-0.05) | nx(-0.05) | undra(-0.05) | .row(-0.05)
    ACCEPTED as axis_1211  cumulative_var=0.8551

  [1207]  axes=1212  step_var=0.0014  binary_acc=0.995  gap=0.0908  max_dot=0.0301  (1.8s)
    TOP:  RES(0.06) | çĻ¾è´§(0.05) | _asm(0.05) | Accessibility(0.05) | _structure(0.05) | COMMENT(0.05) | OPERATION(0.05) | .word(0.05)
    BOT:  management(-0.06) | _zero(-0.06) | Opts(-0.06) | eros(-0.05) | .screen(-0.05) | _DATA(-0.05) | AR(-0.05) | _gpu(-0.05)
    ACCEPTED as axis_1212  cumulative_var=0.8553

  [1208]  axes=1213  step_var=0.0014  binary_acc=0.989  gap=0.0899  max_dot=0.0022  (1.8s)
    TOP:  .float(0.05) | _ENCODING(0.05) | Sid(0.05) | gorithms(0.05) | en(0.05) | Limit(0.05) | viders(0.05) | levision(0.05)
    BOT:  å¤§(-0.07) | å¼ķå¯¼(-0.06) | data(-0.05) | _depth(-0.05) | å®«(-0.05) | ØŃ(-0.05) | Editor(-0.05) | Date(-0.05)
    ACCEPTED as axis_1213  cumulative_var=0.8555

  [1209]  axes=1214  step_var=0.0014  binary_acc=0.985  gap=0.0905  max_dot=0.0152  (1.8s)
    TOP:  es(0.07) | iguous(0.07) | Com(0.06) | _VERIFY(0.06) | .messages(0.06) | åİĤåķĨ(0.06) | _frame(0.06) | days(0.06)
    BOT:  åĲįå½ķ(-0.06) | mod(-0.06) | pv(-0.06) | ylim(-0.06) | mi(-0.06) | gle(-0.06) | æ²Ł(-0.05) | create(-0.05)
    ACCEPTED as axis_1214  cumulative_var=0.8557

  [1210]  axes=1215  step_var=0.0014  binary_acc=0.999  gap=0.0907  max_dot=0.0300  (1.9s)
    TOP:  æĺ¯ä»Ģä¹Ī(0.06) | ,cv(0.05) | World(0.05) | Collision(0.05) | UNDER(0.05) | standen(0.05) | .configuration(0.05) | .sent(0.05)
    BOT:  f(-0.07) | ###(-0.06) | åıĬ(-0.06) | _key(-0.06) | var(-0.06) | u(-0.06) | __(-0.06) | www(-0.06)
    ACCEPTED as axis_1215  cumulative_var=0.8559

  [1211]  axes=1216  step_var=0.0014  binary_acc=0.981  gap=0.0891  max_dot=0.0012  (1.8s)
    TOP:  .args(0.06) | -large(0.06) | Message(0.06) | .state(0.06) | /plain(0.05) | R(0.05) | '));ĊĊ(0.05) | loaded(0.05)
    BOT:  /home(-0.06) | /user(-0.06) | ãĤ¹ãĥĪ(-0.06) | (fe(-0.06) | IN(-0.05) | anchor(-0.05) | emia(-0.05) | viders(-0.05)
    ACCEPTED as axis_1216  cumulative_var=0.8561

  [1212]  axes=1217  step_var=0.0015  binary_acc=0.986  gap=0.0923  max_dot=0.0209  (1.9s)
    TOP:  V(0.06) | [(0.06) | å®¶åħ¬åı¸(0.06) | gments(0.06) | é«ĺæłĩåĩĨ(0.05) | start(0.05) | çļĦæĦıè§ģ(0.05) | ãģĹãģ¾ãģĹãģŁ(0.05)
    BOT:  .GetString(-0.06) | Current(-0.06) | aque(-0.06) | Col(-0.06) | Bob(-0.05) | Factors(-0.05) | Target(-0.05) | .alpha(-0.05)
    ACCEPTED as axis_1217  cumulative_var=0.8563

  [1213]  axes=1218  step_var=0.0014  binary_acc=0.969  gap=0.0917  max_dot=0.0075  (1.9s)
    TOP:  /pub(0.07) | umb(0.06) | va(0.06) | osa(0.06) | _are(0.06) | Algorithms(0.06) | ä¸ĩäºĭ(0.06) | jwt(0.05)
    BOT:  ç§»åĪ°(-0.06) | {Ċ(-0.06) | (paths(-0.06) | à¸¡(-0.05) | Stroke(-0.05) | out(-0.05) | å¢ŀè®¾(-0.05) | å°Ĩè¿Ľä¸ĢæŃ¥(-0.05)
    ACCEPTED as axis_1218  cumulative_var=0.8565

  [1214]  axes=1219  step_var=0.0014  binary_acc=0.994  gap=0.0897  max_dot=0.0116  (1.9s)
    TOP:  ¾(0.06) | ¶(0.06) | è·¯(0.06) | Cross(0.06) | à¤®(0.06) | ±(0.06) | _metric(0.05) | auc(0.05)
    BOT:  /features(-0.06) | Scale(-0.05) | by(-0.05) | son(-0.05) | äºĭçī©(-0.05) | padding(-0.05) | _macros(-0.05) | arguments(-0.05)
    ACCEPTED as axis_1219  cumulative_var=0.8567

  [1215]  axes=1220  step_var=0.0014  binary_acc=0.985  gap=0.0900  max_dot=0.0088  (1.9s)
    TOP:  le(0.07) | ative(0.06) | {Ċ(0.06) | ç¦ı(0.05) | )ĊĊĊĊ(0.05) | alar(0.05) | .')Ċ(0.05) | :ĊĊ(0.05)
    BOT:  ä»ĸä»¬çļĦ(-0.07) | æ²»çĲĨ(-0.06) | è¿ŁåĪ°(-0.05) | Comments(-0.05) | wedge(-0.05) | å¤ļä¸ªåĽ½å®¶(-0.05) | _bytes(-0.05) | (url(-0.05)
    ACCEPTED as axis_1220  cumulative_var=0.8569

  [1216]  axes=1221  step_var=0.0014  binary_acc=0.995  gap=0.0900  max_dot=0.0087  (1.9s)
    TOP:  fu(0.06) | Te(0.05) | }-(0.05) | -review(0.05) | ìĹĲìĦľ(0.05) | [](0.05) | å·ŀ(0.05) | Providers(0.05)
    BOT:  ressing(-0.06) | æŃ¥è¡Į(-0.06) | }'.(-0.06) | .runtime(-0.05) | reib(-0.05) | çļĦçĽ®æłĩ(-0.05) | Du(-0.05) | Argument(-0.05)
    ACCEPTED as axis_1221  cumulative_var=0.8571

  [1217]  axes=1222  step_var=0.0014  binary_acc=0.958  gap=0.0891  max_dot=0.0273  (1.9s)
    TOP:  _Model(0.06) | çļĦè¦ģæ±Ĥ(0.06) | .endswith(0.06) | çļĦæĸ¹åĲĳ(0.06) | alt(0.06) | æĪĳ(0.05) | _d(0.05) | (process(0.05)
    BOT:  gem(-0.05) | ez(-0.05) | _application(-0.05) | Girl(-0.05) | ),čĊ(-0.05) | papers(-0.05) | postgres(-0.05) | .public(-0.05)
    ACCEPTED as axis_1222  cumulative_var=0.8573

  [1218]  axes=1223  step_var=0.0014  binary_acc=0.987  gap=0.0895  max_dot=0.0071  (1.9s)
    TOP:  .Attributes(0.05) | Yet(0.05) | çİ°åľ¨(0.05) | .full(0.05) | speed(0.05) | ###Ċ(0.05) | .events(0.05) | _pan(0.05)
    BOT:  ook(-0.06) | -api(-0.06) | åħ·(-0.06) | utron(-0.06) | åľĨ(-0.05) | stime(-0.05) | à¹ĩ(-0.05) | _since(-0.05)
    ACCEPTED as axis_1223  cumulative_var=0.8575

  [1219]  axes=1224  step_var=0.0014  binary_acc=0.988  gap=0.0873  max_dot=0.0033  (1.9s)
    TOP:  -and(0.06) | aida(0.05) | éĤ»(0.05) | embre(0.05) | å¥İ(0.05) | AX(0.05) | XY(0.05) | Wireless(0.05)
    BOT:  .find(-0.06) | com(-0.06) | Der(-0.05) | sortable(-0.05) | .org(-0.05) | (return(-0.05) | èĲ½å®ŀ(-0.05) | æİĮæı¡(-0.05)
    ACCEPTED as axis_1224  cumulative_var=0.8577

  [1220]  axes=1225  step_var=0.0014  binary_acc=0.992  gap=0.0907  max_dot=0.0136  (1.8s)
    TOP:  str(0.07) | currency(0.07) | cho(0.06) | .elements(0.06) | }Ċ(0.06) | astype(0.06) | ãĢĳ(0.05) | on(0.05)
    BOT:  _pro(-0.07) | DATABASE(-0.06) | æ°ı(-0.06) | /articles(-0.06) | (status(-0.06) | åĲĳå¾Ģ(-0.05) | arbeit(-0.05) | è¡Ĺ(-0.05)
    ACCEPTED as axis_1225  cumulative_var=0.8579

  [1221]  axes=1226  step_var=0.0014  binary_acc=0.971  gap=0.0885  max_dot=0.0215  (1.8s)
    TOP:  ms(0.07) | (cos(0.07) | åŁºéĩĳä»½é¢Ŀ(0.06) | ated(0.06) | knowledge(0.06) | MP(0.06) | urniture(0.06) | infinitely(0.06)
    BOT:  ç£¨æįŁ(-0.05) | .Properties(-0.05) | .Response(-0.05) | _decay(-0.05) | _tolerance(-0.05) | çĽ¸åºĶçļĦ(-0.05) | _LEVEL(-0.05) | ACS(-0.05)
    ACCEPTED as axis_1226  cumulative_var=0.8581

  [1222]  axes=1227  step_var=0.0014  binary_acc=0.976  gap=0.0914  max_dot=0.0108  (1.8s)
    TOP:  strar(0.06) | _endpoint(0.06) | Rem(0.06) | \View(0.06) | Leave(0.05) | /h(0.05) | tere(0.05) | Outstanding(0.05)
    BOT:  AR(-0.06) | _doc(-0.06) | /*Ċ(-0.06) | removed(-0.06) | _STATE(-0.06) | Add(-0.05) | ")Ċ(-0.05) | .Response(-0.05)
    ACCEPTED as axis_1227  cumulative_var=0.8583

  [1223]  axes=1228  step_var=0.0014  binary_acc=0.982  gap=0.0888  max_dot=0.0044  (1.8s)
    TOP:  login(0.05) | âĢĿ,(0.05) | NOTICE(0.05) | _insert(0.05) | Ð±ÐµÐ·Ð¾Ð¿Ð°ÑģÐ½Ð¾ÑģÑĤÐ¸(0.05) | zu(0.05) | %((0.05) | åłª(0.05)
    BOT:  rawing(-0.06) | contract(-0.06) | æ»´æ»´(-0.06) | erg(-0.06) | è·¯çĶ±(-0.05) | ANG(-0.05) | _RECT(-0.05) | Experimental(-0.05)
    ACCEPTED as axis_1228  cumulative_var=0.8585

  [1224]  axes=1229  step_var=0.0014  binary_acc=0.992  gap=0.0900  max_dot=0.0131  (1.8s)
    TOP:  Exclusive(0.06) | (engine(0.06) | Markup(0.06) | ',Ċ(0.05) | Represent(0.05) | Load(0.05) | .annotations(0.05) | Fields(0.05)
    BOT:  åľº(-0.06) | åĲĪæ³ķæĢ§(-0.06) | çļĦéĥ¨åĪĨ(-0.06) | ject(-0.06) | _next(-0.05) | ernational(-0.05) | Ð¼Ð°(-0.05) | å²¸(-0.05)
    ACCEPTED as axis_1229  cumulative_var=0.8587

  [1225]  axes=1230  step_var=0.0014  binary_acc=0.981  gap=0.0882  max_dot=0.0040  (1.8s)
    TOP:  .Message(0.07) | nav(0.06) | into(0.06) | ort(0.06) | kills(0.06) | _SYSTEM(0.06) | messages(0.06) | .Parameter(0.06)
    BOT:  ___(-0.06) | "):Ċ(-0.06) | happens(-0.05) | .USER(-0.05) | "";Ċ(-0.05) | ÐµÑĤÑĮ(-0.05) | annotations(-0.05) | _arguments(-0.05)
    ACCEPTED as axis_1230  cumulative_var=0.8589

  [1226]  axes=1231  step_var=0.0014  binary_acc=0.992  gap=0.0892  max_dot=0.0056  (1.8s)
    TOP:  (sl(0.06) | ()((0.06) | à¸Ńà¸ĩ(0.06) | vm(0.05) | calculator(0.05) | .if(0.05) | é£İæł¼(0.05) | .base(0.05)
    BOT:  ç»Īç»ĵ(-0.06) | /light(-0.05) | -generator(-0.05) | .controllers(-0.05) | Bu(-0.05) | parent(-0.05) | -variable(-0.05) | Net(-0.05)
    ACCEPTED as axis_1231  cumulative_var=0.8591

  [1227]  axes=1232  step_var=0.0014  binary_acc=0.979  gap=0.0897  max_dot=0.0097  (1.8s)
    TOP:  .Job(0.06) | æĿ¥è¯´(0.06) | Rational(0.06) | asta(0.05) | ç©ºæ°Ķ(0.05) | éĺħè¯»(0.05) | Education(0.05) | condition(0.05)
    BOT:  _right(-0.06) | ëĵľ(-0.06) | [row(-0.05) | .pool(-0.05) | Non(-0.05) | (Image(-0.05) | _classification(-0.05) | AS(-0.05)
    ACCEPTED as axis_1232  cumulative_var=0.8593

  [1228]  axes=1233  step_var=0.0014  binary_acc=0.993  gap=0.0919  max_dot=0.0081  (1.9s)
    TOP:  translate(0.06) | /store(0.05) | Ped(0.05) | deny(0.05) | osta(0.05) | req(0.05) | (policy(0.05) | _vector(0.05)
    BOT:  )$(-0.06) | æ¡Į(-0.06) | gn(-0.05) | Q(-0.05) | çļĦéĩįè¦ģ(-0.05) | GN(-0.05) | íķ©ëĭĪëĭ¤(-0.05) | æĶ¹è¿Ľ(-0.05)
    ACCEPTED as axis_1233  cumulative_var=0.8595

  [1229]  axes=1234  step_var=0.0014  binary_acc=0.942  gap=0.0890  max_dot=0.0195  (1.8s)
    TOP:  åĪĨäº«(0.07) | .pad(0.07) | è¯·(0.06) | è¡ĮæĶ¿(0.06) | _depth(0.06) | Tag(0.06) | åı¯æĢľ(0.06) | çŃĶå¤į(0.06)
    BOT:  .Domain(-0.06) | /pages(-0.05) | Listing(-0.05) | .ylim(-0.05) | Te(-0.05) | ules(-0.05) | Ð¾Ð»Ð¸ÑĩÐµÑģÑĤÐ²Ð¾(-0.05) | Regular(-0.05)
    ACCEPTED as axis_1234  cumulative_var=0.8597

  [1230]  axes=1235  step_var=0.0014  binary_acc=0.998  gap=0.0898  max_dot=0.0023  (1.8s)
    TOP:  .resource(0.07) | Alexander(0.06) | Material(0.06) | ä¹¡(0.05) | .backward(0.05) | /n(0.05) | .error(0.05) | /controller(0.05)
    BOT:  /input(-0.06) | =m(-0.06) | ating(-0.06) | æľīæľº(-0.06) | (inst(-0.05) | ç¾¤ä½ĵ(-0.05) | Ð½Ñı(-0.05) | å°ĨåĨĽ(-0.05)
    ACCEPTED as axis_1235  cumulative_var=0.8599

  [1231]  axes=1236  step_var=0.0014  binary_acc=0.998  gap=0.0904  max_dot=0.0088  (1.9s)
    TOP:  Quant(0.06) | iid(0.05) | tera(0.05) | =os(0.05) | nx(0.05) | .translation(0.05) | à¹Ĩ(0.05) | .hist(0.05)
    BOT:  ](-0.06) | Ċ(-0.06) | çº¦åĲĪ(-0.06) | otten(-0.06) | GIN(-0.05) | Ð¼(-0.05) | una(-0.05) | reg(-0.05)
    ACCEPTED as axis_1236  cumulative_var=0.8601

  [1232]  axes=1237  step_var=0.0014  binary_acc=0.994  gap=0.0894  max_dot=0.0113  (1.9s)
    TOP:  ĉobj(0.06) | .parse(0.06) | ä¹¦éĿ¢(0.06) | sgi(0.05) | èĶ¡(0.05) | SP(0.05) | Handler(0.05) | Publication(0.05)
    BOT:  _ui(-0.06) | ley(-0.06) | its(-0.06) | åĪĨçº¢(-0.06) | .gl(-0.06) | G(-0.05) | Canvas(-0.05) | .Web(-0.05)
    ACCEPTED as axis_1237  cumulative_var=0.8603

  [1233]  axes=1238  step_var=0.0015  binary_acc=0.979  gap=0.0910  max_dot=0.0108  (1.9s)
    TOP:  Conversion(0.05) | ÑģÐµÑĢÐ¸Ð°Ð»(0.05) | Hint(0.05) | .com(0.05) | çĽ¸ç»§(0.05) | .Application(0.05) | variables(0.05) | Ph(0.05)
    BOT:  ID(-0.06) | ',Ċ(-0.06) | å¥½(-0.06) | .support(-0.06) | inst(-0.06) | -container(-0.06) | ochen(-0.05) | Ø§Ùģ(-0.05)
    ACCEPTED as axis_1238  cumulative_var=0.8605

  [1234]  axes=1239  step_var=0.0014  binary_acc=0.977  gap=0.0897  max_dot=0.0235  (1.8s)
    TOP:  polynomial(0.06) | å®¢æĪ·(0.06) | kernel(0.05) | (round(0.05) | .rect(0.05) | ä¸įçķĻ(0.05) | æī¹æ¬¡(0.05) | ç¤´(0.05)
    BOT:  /work(-0.06) | à¹Ģà¸Ĺ(-0.05) | _account(-0.05) | çļĦçĽ¸åħ³(-0.05) | qx(-0.05) | osition(-0.05) | (u(-0.05) | -between(-0.05)
    ACCEPTED as axis_1239  cumulative_var=0.8608

  [1235]  axes=1240  step_var=0.0014  binary_acc=0.978  gap=0.0885  max_dot=0.0141  (1.9s)
    TOP:  _place(0.05) | ==Ċ(0.05) | .key(0.05) | è°µ(0.05) | no(0.05) | ãĢį(0.05) | é¦ĸä»ĺ(0.05) | \Request(0.05)
    BOT:  itchen(-0.06) | (parameters(-0.06) | (project(-0.06) | Ð¾ÑģÐ¾Ð±(-0.06) | ette(-0.06) | åħįè´£å£°æĺİ(-0.06) | avan(-0.06) | .*ĊĊ(-0.05)
    ACCEPTED as axis_1240  cumulative_var=0.8610

  [1236]  axes=1241  step_var=0.0014  binary_acc=0.998  gap=0.0876  max_dot=0.0164  (1.9s)
    TOP:  .No(0.06) | Match(0.05) | .OUT(0.05) | .simple(0.05) | _wrapper(0.05) | van(0.05) | Replace(0.05) | _layers(0.05)
    BOT:  ä¸ĭ(-0.06) | sol(-0.05) | _velocity(-0.05) | _python(-0.05) | ini(-0.05) | UUID(-0.05) | /download(-0.05) | ķ(-0.05)
    ACCEPTED as axis_1241  cumulative_var=0.8612

  [1237]  axes=1242  step_var=0.0014  binary_acc=0.981  gap=0.0888  max_dot=0.0028  (1.8s)
    TOP:  Bounds(0.06) | *ĊĊ(0.05) | Broadway(0.05) | _FONT(0.05) | FAST(0.05) | .Enable(0.05) | Waters(0.05) | /{{(0.05)
    BOT:  å¹¶(-0.06) | /material(-0.06) | _split(-0.06) | form(-0.05) | path(-0.05) | å½ĵä»£(-0.05) | ):čĊ(-0.05) | nÃºmero(-0.05)
    ACCEPTED as axis_1242  cumulative_var=0.8614

  [1238]  axes=1243  step_var=0.0015  binary_acc=0.976  gap=0.0892  max_dot=0.0213  (1.9s)
    TOP:  -INF(0.05) | åı¯çĸĳ(0.05) | /cli(0.05) | _T(0.05) | Microsoft(0.05) | \Config(0.05) | (proxy(0.05) | Hex(0.05)
    BOT:  CONF(-0.06) | aa(-0.06) | ior(-0.06) | />ĊĊ(-0.06) | create(-0.06) | Period(-0.05) | ):ĊĊ(-0.05) | ãĥ¬ãĤ¹(-0.05)
    ACCEPTED as axis_1243  cumulative_var=0.8616

  [1239]  axes=1244  step_var=0.0014  binary_acc=0.988  gap=0.0871  max_dot=0.0139  (1.8s)
    TOP:  (events(0.06) | /task(0.06) | It(0.06) | å·¥(0.06) | _LOCK(0.06) | f(0.06) | Tests(0.05) | ÐµÐ½(0.05)
    BOT:  Ball(-0.05) | ();(-0.05) | iÃ§in(-0.05) | //Ċ(-0.05) | .ds(-0.05) | ÑĩÑĤÐ¾(-0.05) | specified(-0.05) | entity(-0.05)
    ACCEPTED as axis_1244  cumulative_var=0.8618

  [1240]  axes=1245  step_var=0.0015  binary_acc=0.989  gap=0.0876  max_dot=0.0011  (1.9s)
    TOP:  /post(0.05) | .bn(0.05) | mes(0.05) | /object(0.05) | ussion(0.05) | _controller(0.05) | .ylabel(0.05) | åĽ¾(0.05)
    BOT:  <(-0.06) | =((-0.06) | SAT(-0.06) | /int(-0.06) | Power(-0.06) | Personal(-0.06) | Transcript(-0.05) | åĵŃæ³£(-0.05)
    ACCEPTED as axis_1245  cumulative_var=0.8620

  [1241]  axes=1246  step_var=0.0014  binary_acc=0.995  gap=0.0886  max_dot=0.0223  (1.9s)
    TOP:  _embedding(0.06) | ulation(0.05) | IMAL(0.05) | IMPLIED(0.05) | Ð¸(0.05) | enter(0.05) | Owned(0.05) | lb(0.05)
    BOT:  å°ıå°ı(-0.06) | ->(-0.06) | è§Ħç«łåĪ¶åº¦(-0.06) | say(-0.05) | æ¶Ĥ(-0.05) | val(-0.05) | xm(-0.05) | ""(-0.05)
    ACCEPTED as axis_1246  cumulative_var=0.8622

  [1242]  axes=1247  step_var=0.0015  binary_acc=0.994  gap=0.0878  max_dot=0.0180  (1.8s)
    TOP:  ">Ċ(0.09) | Ãº(0.06) | Publishers(0.06) | DNA(0.06) | Python(0.05) | Excellent(0.05) | .DB(0.05) | fi(0.05)
    BOT:  .iso(-0.06) | æł¸åĩĨ(-0.05) | /block(-0.05) | GET(-0.05) | =params(-0.05) | ehicles(-0.05) | _scope(-0.05) | çĽĲ(-0.05)
    ACCEPTED as axis_1247  cumulative_var=0.8624

  [1243]  axes=1248  step_var=0.0015  binary_acc=0.987  gap=0.0892  max_dot=0.0015  (1.9s)
    TOP:  ç¾İ(0.06) | /api(0.06) | .bg(0.06) | äºİ(0.05) | èĲĥ(0.05) | Mask(0.05) | å±ŀ(0.05) | ç®¡(0.05)
    BOT:  mind(-0.06) | _row(-0.06) | _evaluation(-0.05) | posite(-0.05) | .print(-0.05) | olesale(-0.05) | STREAM(-0.05) | CONVERT(-0.05)
    ACCEPTED as axis_1248  cumulative_var=0.8626

  [1244]  axes=1249  step_var=0.0015  binary_acc=1.000  gap=0.0894  max_dot=0.0115  (1.8s)
    TOP:  çıĳ(0.06) | ä¸įæĸŃåľ°(0.06) | ä¸įå¾Ĺ(0.06) | Use(0.05) | You(0.05) | .instance(0.05) | Json(0.05) | éĶ°(0.05)
    BOT:  @g(-0.06) | _files(-0.05) | aily(-0.05) | Tags(-0.05) | èıĮ(-0.05) | sWith(-0.05) | type(-0.05) | Added(-0.05)
    ACCEPTED as axis_1249  cumulative_var=0.8628

  [1245]  axes=1250  step_var=0.0014  binary_acc=0.982  gap=0.0855  max_dot=0.0181  (1.8s)
    TOP:  Cancelled(0.06) | .mode(0.06) | çº¦å®ļ(0.06) | ä¸Ńèį¯(0.05) | onitor(0.05) | æľºåĬ¨è½¦(0.05) | éĺ²æİ§(0.05) | åĪĨåĪ«æĺ¯(0.05)
    BOT:  .des(-0.06) | ate(-0.06) | IS(-0.06) | _r(-0.06) | _admin(-0.06) | P(-0.06) | from(-0.06) | alchemy(-0.05)
    ACCEPTED as axis_1250  cumulative_var=0.8629

  [1246]  axes=1251  step_var=0.0015  binary_acc=0.991  gap=0.0884  max_dot=0.0138  (1.8s)
    TOP:  ä¸ľéĥ¨(0.06) | æī©å±ķ(0.05) | åĮ»(0.05) | JSON(0.05) | ç¥ĸåĽ½(0.05) | æĹ¶éĹ´åĴĮ(0.05) | çīĽ(0.05) | ä¸Ģé¢Ĺ(0.05)
    BOT:  _exists(-0.06) | =re(-0.06) | _CPP(-0.06) | =all(-0.05) | .stem(-0.05) | -Z(-0.05) | open(-0.05) | (B(-0.05)
    ACCEPTED as axis_1251  cumulative_var=0.8631

  [1247]  axes=1252  step_var=0.0014  binary_acc=0.996  gap=0.0880  max_dot=0.0096  (1.9s)
    TOP:  TestCase(0.06) | .python(0.05) | __,(0.05) | -show(0.05) | _callback(0.05) | æĺŁåħī(0.05) | ewing(0.05) | Seek(0.05)
    BOT:  (local(-0.06) | atives(-0.05) | åŃĲ(-0.05) | /Ċ(-0.05) | =True(-0.05) | .Order(-0.05) | .trace(-0.05) | Liberty(-0.05)
    ACCEPTED as axis_1252  cumulative_var=0.8633

  [1248]  axes=1253  step_var=0.0014  binary_acc=0.999  gap=0.0879  max_dot=0.0161  (1.9s)
    TOP:  _w(0.05) | .constants(0.05) | Plane(0.05) | æĪĳå®¶(0.05) | åĲĦä¸ªæĸ¹éĿ¢(0.05) | jes(0.05) | THON(0.05) | æľ¬åĽ½(0.05)
    BOT:  .net(-0.06) | _content(-0.06) | +'(-0.06) | DEBUG(-0.06) | .metrics(-0.06) | load(-0.06) | Details(-0.05) | W(-0.05)
    ACCEPTED as axis_1253  cumulative_var=0.8635

  [1249]  axes=1254  step_var=0.0015  binary_acc=0.995  gap=0.0882  max_dot=0.0128  (1.8s)
    TOP:  width(0.06) | iram(0.06) | els(0.06) | at(0.06) | UE(0.05) | in(0.05) | rying(0.05) | çĶµéĺ»(0.05)
    BOT:  .pen(-0.06) | èŀįåªĴä½ĵ(-0.06) | .{(-0.06) | (abs(-0.05) | _registry(-0.05) | .smart(-0.05) | separator(-0.05) | "(-0.05)
    ACCEPTED as axis_1254  cumulative_var=0.8637

  [1250]  axes=1255  step_var=0.0014  binary_acc=0.985  gap=0.0877  max_dot=0.0093  (1.8s)
    TOP:  Ċ            Ċ(0.05) | _dims(0.05) | -ĊĊ(0.05) | _NT(0.05) | æ¯½(0.05) | -bold(0.05) | _text(0.05) | ources(0.05)
    BOT:  create(-0.06) | åħ¨åĽ½äººæ°ĳ(-0.05) | ='./(-0.05) | æ´¾äºº(-0.05) | Am(-0.05) | ĻĤ(-0.05) | Broad(-0.05) | movie(-0.05)
    ACCEPTED as axis_1255  cumulative_var=0.8639

  [1251]  axes=1256  step_var=0.0015  binary_acc=0.998  gap=0.0876  max_dot=0.0253  (1.8s)
    TOP:  ar(0.07) | t(0.06) | da(0.06) | m(0.06) | encia(0.06) | length(0.06) | at(0.05) | were(0.05)
    BOT:  Notice(-0.06) | .Sample(-0.06) | -script(-0.05) | ä¸ľåĮĹ(-0.05) | æĻĥ(-0.05) | loading(-0.05) | Resource(-0.05) | /password(-0.05)
    ACCEPTED as axis_1256  cumulative_var=0.8641

  [1252]  axes=1257  step_var=0.0014  binary_acc=0.999  gap=0.0877  max_dot=0.0122  (1.9s)
    TOP:  è¾¾æłĩ(0.06) | æĺŁ(0.05) | .sec(0.05) | .topic(0.05) | .Response(0.05) | uition(0.05) | teacher(0.05) | Ð²ÐµÐºÐ°(0.05)
    BOT:  de(-0.07) | Streaming(-0.06) | æĸ°(-0.06) | ä¸Ńæĸ¹(-0.05) | åħ¬ç¤¾(-0.05) | .Status(-0.05) | dat(-0.05) | å©·(-0.05)
    ACCEPTED as axis_1257  cumulative_var=0.8643

  [1253]  axes=1258  step_var=0.0014  binary_acc=0.982  gap=0.0875  max_dot=0.0056  (1.9s)
    TOP:  -error(0.06) | App(0.06) | in(0.05) | =torch(0.05) | =========(0.05) | åıĺéĩı(0.05) | æĹ¶éĻĲ(0.05) | _EXPORT(0.05)
    BOT:  rib(-0.06) | Topic(-0.06) | .chat(-0.06) | ervlet(-0.05) | lean(-0.05) | years(-0.05) | /device(-0.05) | .title(-0.05)
    ACCEPTED as axis_1258  cumulative_var=0.8645

  [1254]  axes=1259  step_var=0.0014  binary_acc=0.982  gap=0.0880  max_dot=0.0028  (1.8s)
    TOP:  ÐĶ(0.06) | env(0.06) | BL(0.06) | å½ĵæĪĳä»¬(0.06) | .cls(0.06) | ar(0.06) | Z(0.06) | ").ĊĊ(0.06)
    BOT:  +(-0.06) | ä¸ŃåĽ½(-0.06) | .Format(-0.05) | -Based(-0.05) | Flip(-0.05) | omba(-0.05) | _OUT(-0.05) | _RIGHT(-0.05)
    ACCEPTED as axis_1259  cumulative_var=0.8647

  [1255]  axes=1260  step_var=0.0014  binary_acc=0.973  gap=0.0896  max_dot=0.0163  (1.9s)
    TOP:  =plt(0.07) | èĬ±(0.06) | redentials(0.06) | src(0.06) | Ð¸Ð½Ð°(0.06) | embed(0.06) | wert(0.06) | shire(0.06)
    BOT:  _eval(-0.06) | _package(-0.06) | ÐºÐ¾Ð´(-0.05) | '''Ċ(-0.05) | .token(-0.05) | /pr(-0.05) | (sigma(-0.05) | (Common(-0.05)
    ACCEPTED as axis_1260  cumulative_var=0.8649

  [1256]  axes=1261  step_var=0.0015  binary_acc=0.998  gap=0.0879  max_dot=0.0157  (1.9s)
    TOP:  media(0.06) | è½¦(0.06) | culator(0.05) | æĹİ(0.05) | .enable(0.05) | å¯¼èĩ´(0.05) | Ag(0.05) | Up(0.05)
    BOT:  ND(-0.06) | elog(-0.06) | é¡·(-0.06) | .ie(-0.05) | @class(-0.05) | artifacts(-0.05) | Package(-0.05) | */}Ċ(-0.05)
    ACCEPTED as axis_1261  cumulative_var=0.8651

  [1257]  axes=1262  step_var=0.0015  binary_acc=0.967  gap=0.0881  max_dot=0.0201  (1.8s)
    TOP:  ition(0.07) | /sc(0.06) | That(0.06) | ors(0.06) | Paginator(0.06) | >{(0.05) | æĺİä»£(0.05) | cd(0.05)
    BOT:  Defaults(-0.07) | pages(-0.07) | .Packet(-0.06) | .Color(-0.05) | ä»ĵä½į(-0.05) | æĦļ(-0.05) | åºŁçī©(-0.05) | Align(-0.05)
    ACCEPTED as axis_1262  cumulative_var=0.8653

  [1258]  axes=1263  step_var=0.0014  binary_acc=0.979  gap=0.0875  max_dot=0.0104  (1.8s)
    TOP:  ={'(0.06) | High(0.06) | the(0.06) | Ð²(0.06) | .structure(0.05) | TE(0.05) | r(0.05) | Ðł(0.05)
    BOT:  js(-0.05) | -summary(-0.05) | åĨħéĥ¨(-0.05) | Capture(-0.05) | .Label(-0.05) | {{{(-0.05) | .Login(-0.05) | -reference(-0.05)
    ACCEPTED as axis_1263  cumulative_var=0.8655

  [1259]  axes=1264  step_var=0.0015  binary_acc=0.977  gap=0.0888  max_dot=0.0166  (1.8s)
    TOP:  _sum(0.06) | middlewares(0.06) | .ui(0.05) | .I(0.05) | ATION(0.05) | Print(0.05) | JPEG(0.05) | OOK(0.05)
    BOT:  stream(-0.06) | Zip(-0.06) | åĪĬ(-0.05) | _blocks(-0.05) | .ex(-0.05) | åİ»å¹´åĲĮæľŁ(-0.05) | jo(-0.05) | ä¸¥åİīæīĵåĩ»(-0.05)
    ACCEPTED as axis_1264  cumulative_var=0.8657

  [1260]  axes=1265  step_var=0.0014  binary_acc=0.993  gap=0.0885  max_dot=0.0301  (1.9s)
    TOP:  âĢĶ(0.06) | (config(0.06) | _off(0.06) | _artist(0.05) | '../(0.05) | /models(0.05) | why(0.05) | _block(0.05)
    BOT:  _range(-0.06) | Open(-0.06) | .Socket(-0.06) | åħ³å¿ĥ(-0.05) | utar(-0.05) | Extensions(-0.05) | Ð¸(-0.05) | èĢĲå¿ĥ(-0.05)
    ACCEPTED as axis_1265  cumulative_var=0.8659

  [1261]  axes=1266  step_var=0.0015  binary_acc=0.989  gap=0.0880  max_dot=0.0103  (1.8s)
    TOP:  èĭ¥(0.05) | validity(0.05) | .Iter(0.05) | .max(0.05) | %}Ċ(0.05) | imilar(0.05) | çľĭä¸Ģçľĭ(0.05) | ..(0.05)
    BOT:  Headers(-0.06) | (icon(-0.06) | (E(-0.06) | .ViewModel(-0.06) | yun(-0.06) | Pending(-0.06) | å½ĵæĹ¥(-0.06) | Complete(-0.05)
    ACCEPTED as axis_1266  cumulative_var=0.8661

  [1262]  axes=1267  step_var=0.0015  binary_acc=0.986  gap=0.0894  max_dot=0.0102  (1.9s)
    TOP:  è½®(0.07) | Mapper(0.06) | /job(0.05) | çĽĳ(0.05) | /install(0.05) | (sender(0.05) | Ð¾Ð½Ð»Ð°Ð¹Ð½(0.05) | ('#(0.05)
    BOT:  For(-0.06) | oux(-0.06) | posed(-0.06) | (track(-0.06) | =ĊĊ(-0.06) | Expenses(-0.05) | ...(-0.05) | connector(-0.05)
    ACCEPTED as axis_1267  cumulative_var=0.8663

  [1263]  axes=1268  step_var=0.0015  binary_acc=0.990  gap=0.0887  max_dot=0.0187  (1.8s)
    TOP:  (category(0.06) | ph(0.06) | is(0.05) | qrt(0.05) | Duration(0.05) | isans(0.05) | Values(0.05) | ="@(0.05)
    BOT:  Ð¢(-0.06) | /media(-0.06) | Write(-0.05) | ä»ĵ(-0.05) | æİĪæĿĥ(-0.05) | ä¸½(-0.05) | çĽ¸åĲĮ(-0.05) | /services(-0.05)
    ACCEPTED as axis_1268  cumulative_var=0.8665

  [1264]  axes=1269  step_var=0.0014  binary_acc=0.997  gap=0.0883  max_dot=0.0125  (1.8s)
    TOP:  OS(0.07) | p(0.06) | zf(0.06) | åĩºå¸Ń(0.06) | no(0.06) | ague(0.06) | Ã©n(0.06) | æł¼(0.06)
    BOT:  (dataset(-0.05) | Superintendent(-0.05) | Ðĵ(-0.05) | "Ċ(-0.05) | );Ċ(-0.05) | Leeds(-0.05) | apist(-0.05) | .Transform(-0.05)
    ACCEPTED as axis_1269  cumulative_var=0.8667

  [1265]  axes=1270  step_var=0.0015  binary_acc=0.989  gap=0.0850  max_dot=0.0124  (1.9s)
    TOP:  Ñģ(0.06) | p(0.06) | Init(0.06) | bib(0.05) | åĮºå§Ķ(0.05) | ="">Ċ(0.05) | S(0.05) | by(0.05)
    BOT:  .tests(-0.06) | increment(-0.05) | .photos(-0.05) | ìŀĪëĬĶ(-0.05) | /my(-0.05) | _setup(-0.05) | èĩªå·±(-0.05) | .location(-0.05)
    ACCEPTED as axis_1270  cumulative_var=0.8669

  [1266]  axes=1271  step_var=0.0015  binary_acc=0.991  gap=0.0868  max_dot=0.0236  (2.0s)
    TOP:  ave(0.06) | .internet(0.05) | ieve(0.05) | Smile(0.05) | ferences(0.05) | astr(0.05) | Loader(0.05) | _secret(0.05)
    BOT:  $\(-0.06) | Blur(-0.06) | Large(-0.06) | _Z(-0.06) | Defaults(-0.06) | _extra(-0.06) | /py(-0.05) | äºĭçī©(-0.05)
    ACCEPTED as axis_1271  cumulative_var=0.8671

  [1267]  axes=1272  step_var=0.0015  binary_acc=0.977  gap=0.0901  max_dot=0.0070  (1.9s)
    TOP:  _len(0.06) | even(0.06) | _PREFIX(0.06) | Off(0.05) | _a(0.05) | çļĦåİĨåı²(0.05) | ÙĪÙĬ(0.05) | çĪ¶åŃĲ(0.05)
    BOT:  _bucket(-0.06) | ku(-0.05) | Revenue(-0.05) | ãĤĭ(-0.05) | è¯¿(-0.05) | )ĊĊ(-0.05) | å®£ä¼łæķĻèĤ²(-0.05) | message(-0.05)
    ACCEPTED as axis_1272  cumulative_var=0.8673

  [1268]  axes=1273  step_var=0.0015  binary_acc=0.984  gap=0.0855  max_dot=0.0114  (1.8s)
    TOP:  /export(0.06) | /=(0.06) | _scal(0.05) | dependencies(0.05) | 'une(0.05) | /User(0.05) | PORT(0.05) | .Local(0.05)
    BOT:  .exec(-0.07) | Net(-0.06) | liv(-0.06) | Update(-0.06) | åĳ¨äºĶ(-0.05) | z(-0.05) | at(-0.05) | Äĳá»ģ(-0.05)
    ACCEPTED as axis_1273  cumulative_var=0.8675

  [1269]  axes=1274  step_var=0.0015  binary_acc=0.998  gap=0.0869  max_dot=0.0129  (1.9s)
    TOP:  (map(0.07) | l(0.06) | ÙĤØ©(0.05) | -testing(0.05) | ivil(0.05) | marked(0.05) | clear(0.05) | =url(0.05)
    BOT:  >[(-0.06) | å¦Ĥæŀľ(-0.05) | ä¸ĢèĪ¬(-0.05) | åħ±å»º(-0.05) | Helper(-0.05) | APP(-0.05) | èĢģå¤§(-0.05) | +=(-0.05)
    ACCEPTED as axis_1274  cumulative_var=0.8676

  [1270]  axes=1275  step_var=0.0015  binary_acc=0.989  gap=0.0880  max_dot=0.0025  (1.8s)
    TOP:  (host(0.06) | ç²¾ç¥ŀæĸĩæĺİ(0.05) | State(0.05) | abs(0.05) | .Active(0.05) | .float(0.05) | AB(0.05) | N(0.05)
    BOT:  .'(-0.06) | .c(-0.05) | _common(-0.05) | >](-0.05) | _MULTI(-0.05) | Â»(-0.05) | Utils(-0.05) | ircuit(-0.05)
    ACCEPTED as axis_1275  cumulative_var=0.8678

  [1271]  axes=1276  step_var=0.0015  binary_acc=0.967  gap=0.0904  max_dot=0.0019  (1.9s)
    TOP:  NH(0.06) | services(0.06) | Res(0.05) | VER(0.05) | charge(0.05) | asu(0.05) | merged(0.05) | Ùħ(0.05)
    BOT:  æ¹¿(-0.06) | æ±Ł(-0.06) | Ð°ÑĪ(-0.06) | éĺħè§Ī(-0.06) | starts(-0.06) | .has(-0.06) | }(-0.05) | /css(-0.05)
    ACCEPTED as axis_1276  cumulative_var=0.8680

  [1272]  axes=1277  step_var=0.0015  binary_acc=0.962  gap=0.0894  max_dot=0.0220  (2.0s)
    TOP:  It(0.06) | _multiplier(0.06) | AYS(0.06) | /post(0.05) | cr(0.05) | atches(0.05) | inhibition(0.05) | variant(0.05)
    BOT:  (Object(-0.06) | pq(-0.05) | .object(-0.05) | .export(-0.05) | _timestamp(-0.05) | _receipt(-0.05) | åĵī(-0.05) | .MAIN(-0.05)
    ACCEPTED as axis_1277  cumulative_var=0.8682

  [1273]  axes=1278  step_var=0.0015  binary_acc=0.991  gap=0.0891  max_dot=0.0372  (1.9s)
    TOP:  ai(0.07) | .apple(0.06) | İ(0.05) | _init(0.05) | çļĦåĬĽéĩı(0.05) | æĿĲæĸĻ(0.05) | Ø±Ø¨(0.05) | .transport(0.05)
    BOT:  ed(-0.05) | Reddit(-0.05) | )]Ċ(-0.05) | Drag(-0.05) | _rs(-0.05) | atisch(-0.05) | Editing(-0.05) | _SESSION(-0.05)
    ACCEPTED as axis_1278  cumulative_var=0.8684

  [1274]  axes=1279  step_var=0.0015  binary_acc=0.995  gap=0.0890  max_dot=0.0282  (1.8s)
    TOP:  /Ċ(0.06) | åĽŀçŃĶ(0.06) | è§£(0.06) | åĩ¡(0.05) | elligence(0.05) | Copyright(0.05) | Android(0.05) | .ui(0.05)
    BOT:  from(-0.06) | been(-0.06) | Average(-0.06) | .Private(-0.05) | é¢ľèī²(-0.05) | from(-0.05) | Enhanced(-0.05) | Friend(-0.05)
    ACCEPTED as axis_1279  cumulative_var=0.8686

  [1275]  axes=1280  step_var=0.0015  binary_acc=0.981  gap=0.0890  max_dot=0.0193  (1.9s)
    TOP:  (loss(0.05) | æĪĳå°±(0.05) | _atom(0.05) | fft(0.05) | .mesh(0.05) | Validate(0.05) | fÃ¶r(0.05) | ç¾İåĽ½(0.05)
    BOT:  ale(-0.06) | éĴĪå¯¹(-0.06) | ways(-0.06) | redirect(-0.06) | Edges(-0.05) | Tensor(-0.05) | yo(-0.05) | _extra(-0.05)
    ACCEPTED as axis_1280  cumulative_var=0.8688

  [1276]  axes=1281  step_var=0.0015  binary_acc=0.977  gap=0.0875  max_dot=0.0074  (1.8s)
    TOP:  Institute(0.05) | ac(0.05) | Det(0.05) | .helpers(0.05) | odic(0.05) | _ES(0.05) | AN(0.05) | acle(0.05)
    BOT:  Developed(-0.05) | å¯¹æİ¥(-0.05) | _training(-0.05) | .policy(-0.05) | _using(-0.05) | .backend(-0.05) | RATION(-0.05) | olang(-0.05)
    ACCEPTED as axis_1281  cumulative_var=0.8690

  [1277]  axes=1282  step_var=0.0015  binary_acc=0.999  gap=0.0880  max_dot=0.0326  (1.9s)
    TOP:  .,(0.05) | .),(0.05) | .mask(0.05) | +d(0.05) | _auth(0.05) | .Schema(0.05) | :description(0.05) | .use(0.05)
    BOT:  ÐµÐ»(-0.06) | Based(-0.06) | edBy(-0.06) | å°İ(-0.06) | re(-0.06) | ER(-0.06) | çĲĨ(-0.05) | t(-0.05)
    ACCEPTED as axis_1282  cumulative_var=0.8692

  [1278]  axes=1283  step_var=0.0015  binary_acc=0.979  gap=0.0886  max_dot=0.0176  (1.8s)
    TOP:  .Fixed(0.06) | åĪ¤åĨ³(0.05) | åĩºçİ°(0.05) | .P(0.05) | _api(0.05) | æĬĹè®®(0.05) | .java(0.05) | interfaces(0.05)
    BOT:  Process(-0.07) | _words(-0.06) | ulate(-0.06) | org(-0.06) | éħĴ(-0.05) | å®ĺ(-0.05) | é£İ(-0.05) | æŃ£(-0.05)
    ACCEPTED as axis_1283  cumulative_var=0.8694

  [1279]  axes=1284  step_var=0.0015  binary_acc=0.987  gap=0.0878  max_dot=0.0103  (1.8s)
    TOP:  marshal(0.06) | èŀįåĲĪåıĳå±ķ(0.05) | _ENTER(0.05) | areth(0.05) | /background(0.05) | <string(0.05) | Newsletter(0.05) | æ¶Īå¤±(0.05)
    BOT:  å¤ļ(-0.06) | M(-0.06) | up(-0.06) | al(-0.06) | v(-0.06) | ar(-0.06) | éĹ´(-0.06) | NE(-0.05)
    ACCEPTED as axis_1284  cumulative_var=0.8696

  [1280]  axes=1285  step_var=0.0015  binary_acc=0.994  gap=0.0872  max_dot=0.0419  (1.8s)
    TOP:  ](0.06) | _dropout(0.05) | Ø¯Ùĩ(0.05) | .Contact(0.05) | other(0.05) | also(0.05) | ä¸Ģå®ļ(0.05) | Ð¸ÑĤÐµ(0.05)
    BOT:  éĤ»(-0.06) | ç¼º(-0.06) | creativecommons(-0.06) | at(-0.06) | .wh(-0.05) | .uid(-0.05) | .weight(-0.05) | èµ·åĪ°(-0.05)
    ACCEPTED as axis_1285  cumulative_var=0.8698

  [1281]  axes=1286  step_var=0.0015  binary_acc=0.998  gap=0.0877  max_dot=0.0025  (1.9s)
    TOP:  .Environment(0.06) | [y(0.05) | æ¡ĮéĿ¢(0.05) | encies(0.05) | []Ċ(0.05) | å½©èĻ¹(0.05) | åĽłæŃ¤(0.05) | Blocks(0.05)
    BOT:  views(-0.06) | ">(-0.05) | _test(-0.05) | /packages(-0.05) | _comments(-0.05) | .source(-0.05) | Seeing(-0.05) | _for(-0.05)
    ACCEPTED as axis_1286  cumulative_var=0.8700

  [1282]  axes=1287  step_var=0.0015  binary_acc=0.988  gap=0.0871  max_dot=0.0015  (1.9s)
    TOP:  åħ®(0.06) | _response(0.05) | General(0.05) | (Http(0.05) | /k(0.05) | G(0.05) | share(0.05) | çļĦåĪ©çĽĬ(0.05)
    BOT:  é¡¿(-0.06) | Attached(-0.06) | Serial(-0.05) | Foundation(-0.05) | FS(-0.05) | Te(-0.05) | _safe(-0.05) | .getResource(-0.05)
    ACCEPTED as axis_1287  cumulative_var=0.8702

  [1283]  axes=1288  step_var=0.0015  binary_acc=0.988  gap=0.0872  max_dot=0.0150  (1.8s)
    TOP:  /sp(0.06) | .-(0.06) | (day(0.06) | -plugin(0.06) | (eval(0.05) | Ð»Ð¾(0.05) | ama(0.05) | çĤº(0.05)
    BOT:  struct(-0.05) | èĬĤæĹ¥(-0.05) | Credentials(-0.05) | EEK(-0.05) | Communication(-0.05) | loses(-0.05) | ä»ĸäºº(-0.05) | cam(-0.05)
    ACCEPTED as axis_1288  cumulative_var=0.8704

  [1284]  axes=1289  step_var=0.0015  binary_acc=0.992  gap=0.0886  max_dot=0.0172  (1.9s)
    TOP:  Tk(0.06) | é¢Ĩ(0.06) | ç¦ıå»ºçľģ(0.05) | /platform(0.05) | /latest(0.05) | ä¸Ģä¸ª(0.05) | ('/(0.05) | ource(0.05)
    BOT:  _exc(-0.06) | _scores(-0.06) | loop(-0.06) | ap(-0.06) | _sampler(-0.06) | åĶĲå±±(-0.05) | functor(-0.05) | _callback(-0.05)
    ACCEPTED as axis_1289  cumulative_var=0.8705

  [1285]  axes=1290  step_var=0.0015  binary_acc=0.988  gap=0.0894  max_dot=0.0042  (1.9s)
    TOP:  ;">Ċ(0.06) | _util(0.05) | University(0.05) | MC(0.05) | åł´(0.05) | si(0.05) | è®º(0.05) | to(0.05)
    BOT:  /ns(-0.06) | å¼ķèµ·(-0.05) | _hash(-0.05) | _ACCOUNT(-0.05) | Ãī(-0.05) | éĵ¬(-0.05) | .addEventListener(-0.05) | .Manager(-0.05)
    ACCEPTED as axis_1290  cumulative_var=0.8707

  [1286]  axes=1291  step_var=0.0015  binary_acc=0.987  gap=0.0885  max_dot=0.0238  (1.8s)
    TOP:  -->Ċ(0.06) | ')čĊ(0.05) | æ¸¸(0.05) | _PY(0.05) | .activ(0.05) | =Ċ(0.05) | çļĦèĦ¸(0.05) | .Bottom(0.05)
    BOT:  arna(-0.07) | ola(-0.06) | ur(-0.06) | -analysis(-0.06) | 4(-0.05) | _SHARED(-0.05) | (module(-0.05) | vals(-0.05)
    ACCEPTED as axis_1291  cumulative_var=0.8709

  [1287]  axes=1292  step_var=0.0015  binary_acc=0.972  gap=0.0870  max_dot=0.0051  (1.8s)
    TOP:  .ref(0.06) | -message(0.05) | Dec(0.05) | ifers(0.05) | ANTS(0.05) | >,(0.05) | æ¤ħ(0.05) | ìĦĿ(0.05)
    BOT:  åĢºåĬ¡(-0.06) | çº¤ç»´(-0.06) | ÑģÐºÐ¾Ð¹(-0.05) | Ð¸ÑĤÐµÑģÑĮ(-0.05) | ÑĪÐ¸Ñħ(-0.05) | v(-0.05) | å°Ĩ(-0.05) | æĶ¶åıĸ(-0.05)
    ACCEPTED as axis_1292  cumulative_var=0.8711

  [1288]  axes=1293  step_var=0.0015  binary_acc=0.986  gap=0.0871  max_dot=0.0111  (1.9s)
    TOP:  å¯¹è¯Ŀ(0.05) | Messaging(0.05) | éĩİå¿ĥ(0.05) | Ð¼Ð½Ð¾Ð³(0.05) | preg(0.05) | tbody(0.05) | Digite(0.05) | Recording(0.05)
    BOT:  .l(-0.07) | [-(-0.07) | =f(-0.06) | /be(-0.06) | (cli(-0.06) | der(-0.06) | OL(-0.06) | _xlim(-0.06)
    ACCEPTED as axis_1293  cumulative_var=0.8713

  [1289]  axes=1294  step_var=0.0015  binary_acc=0.978  gap=0.0874  max_dot=0.0115  (1.9s)
    TOP:  å¯¹(0.06) | .primary(0.06) | PrototypeOf(0.05) | '+'(0.05) | é¢ĦçķĻ(0.05) | iples(0.05) | _BY(0.05) | BLE(0.05)
    BOT:  XML(-0.06) | seconds(-0.06) | Links(-0.06) | Tr(-0.06) | an(-0.06) | _timestamp(-0.05) | TRA(-0.05) | More(-0.05)
    ACCEPTED as axis_1294  cumulative_var=0.8715

  [1290]  axes=1295  step_var=0.0015  binary_acc=0.994  gap=0.0893  max_dot=0.0151  (1.8s)
    TOP:  .Conv(0.06) | `(0.05) | reshold(0.05) | uniform(0.05) | WIDTH(0.05) | _container(0.05) | .Key(0.05) | _style(0.05)
    BOT:  ulsive(-0.06) | SE(-0.06) | imension(-0.06) | tip(-0.06) | orem(-0.05) | çªģåĩºéĹ®é¢ĺ(-0.05) | /mobile(-0.05) | IN(-0.05)
    ACCEPTED as axis_1295  cumulative_var=0.8717

  [1291]  axes=1296  step_var=0.0015  binary_acc=0.976  gap=0.0875  max_dot=0.0020  (1.8s)
    TOP:  is(0.06) | éĢĤåĲĪ(0.06) | aff(0.05) | çĬ¯ç½ª(0.05) | chy(0.05) | Moving(0.05) | processor(0.05) | accepted(0.05)
    BOT:  _hook(-0.06) | locks(-0.05) | ENTRY(-0.05) | Sign(-0.05) | _threshold(-0.05) | ession(-0.05) | Address(-0.05) | /fixtures(-0.05)
    ACCEPTED as axis_1296  cumulative_var=0.8719

  [1292]  axes=1297  step_var=0.0015  binary_acc=0.968  gap=0.0862  max_dot=0.0128  (1.9s)
    TOP:  .bot(0.05) | ,id(0.05) | Ã¤ll(0.05) | _idx(0.05) | _available(0.05) | .footer(0.05) | cmap(0.05) | :Ċ(0.05)
    BOT:  verbose(-0.06) | _cost(-0.06) | _models(-0.06) | affe(-0.05) | _logits(-0.05) | overview(-0.05) | æĹ¥ä¸ĭåįĪ(-0.05) | _arg(-0.05)
    ACCEPTED as axis_1297  cumulative_var=0.8721

  [1293]  axes=1298  step_var=0.0015  binary_acc=0.988  gap=0.0867  max_dot=0.0170  (1.9s)
    TOP:  Property(0.05) | Uniform(0.05) | IFEST(0.05) | Editable(0.05) | Tip(0.05) | Merc(0.05) | _numbers(0.05) | Ok(0.05)
    BOT:  .tight(-0.05) | Â£(-0.05) | .urls(-0.05) | urn(-0.05) | -D(-0.05) | who(-0.05) | è°ĥæķ´(-0.05) | Lore(-0.05)
    ACCEPTED as axis_1298  cumulative_var=0.8723

  [1294]  axes=1299  step_var=0.0015  binary_acc=0.996  gap=0.0872  max_dot=0.0016  (1.9s)
    TOP:  Ð¸Ð½Ð°(0.05) | .xlabel(0.05) | ices(0.05) | rates(0.05) | Bin(0.05) | Licence(0.05) | Mint(0.05) | .Basic(0.05)
    BOT:  Âł(-0.06) | âĢĶâĢĶ(-0.06) | ç´§æĢ¥(-0.06) | LIST(-0.06) | -star(-0.06) | leg(-0.06) | .st(-0.06) | è¯¸ä¾¯(-0.05)
    ACCEPTED as axis_1299  cumulative_var=0.8725

  [1295]  axes=1300  step_var=0.0015  binary_acc=0.999  gap=0.0868  max_dot=0.0055  (1.9s)
    TOP:  _follow(0.06) | _threads(0.06) | /Public(0.05) | .menu(0.05) | _us(0.05) | ent(0.05) | ())ĊĊ(0.05) | on(0.05)
    BOT:  istributions(-0.06) | .v(-0.05) | .linear(-0.05) | Dec(-0.05) | tensor(-0.05) | .typ(-0.05) | éĶĪ(-0.05) | Ã¡veis(-0.05)
    ACCEPTED as axis_1300  cumulative_var=0.8727

  [1296]  axes=1301  step_var=0.0015  binary_acc=0.984  gap=0.0858  max_dot=0.0176  (1.9s)
    TOP:  _prot(0.06) | -up(0.05) | _USER(0.05) | å½±çīĩ(0.05) | å¾Ģä¸ĭ(0.05) | _worker(0.05) | ç§ĳå¹»(0.05) | _ac(0.05)
    BOT:  By(-0.06) | ef(-0.06) | in(-0.06) | .make(-0.05) | /of(-0.05) | Detect(-0.05) | be(-0.05) | åįģå¹´(-0.05)
    ACCEPTED as axis_1301  cumulative_var=0.8729

  [1297]  axes=1302  step_var=0.0015  binary_acc=0.981  gap=0.0877  max_dot=0.0145  (1.8s)
    TOP:  åĪĻ(0.06) | aker(0.06) | widget(0.06) | å·©åĽº(0.06) | Ð¾Ð²ÑĭÐµ(0.06) | arg(0.05) | add(0.05) | MENTS(0.05)
    BOT:  pective(-0.05) | .prot(-0.05) | .subscribe(-0.05) | =}(-0.05) | Featuring(-0.05) | é»į(-0.05) | rypto(-0.05) | Animation(-0.05)
    ACCEPTED as axis_1302  cumulative_var=0.8730

  [1298]  axes=1303  step_var=0.0015  binary_acc=0.988  gap=0.0854  max_dot=0.0036  (1.9s)
    TOP:  */ĊĊ(0.06) | /testing(0.05) | ç¦ģ(0.05) | GER(0.05) | (skip(0.05) | State(0.05) | æ´¥(0.05) | ards(0.05)
    BOT:  Framework(-0.06) | _pattern(-0.05) | Continue(-0.05) | AND(-0.05) | æĶ¶åĪ°(-0.05) | ".ĊĊ(-0.05) | -expanded(-0.05) | Encoding(-0.05)
    ACCEPTED as axis_1303  cumulative_var=0.8732

  [1299]  axes=1304  step_var=0.0015  binary_acc=0.982  gap=0.0845  max_dot=0.0056  (1.8s)
    TOP:  -In(0.05) | (params(0.05) | _Init(0.05) | .ch(0.05) | .common(0.05) | .secret(0.05) | Batch(0.05) | etermine(0.05)
    BOT:  .spawn(-0.06) | ä¸Ģ(-0.06) | __)ĊĊ(-0.06) | ants(-0.05) | Server(-0.05) | _path(-0.05) | Mark(-0.05) | regularization(-0.05)
    ACCEPTED as axis_1304  cumulative_var=0.8734

  [1300]  axes=1305  step_var=0.0015  binary_acc=0.991  gap=0.0844  max_dot=0.0087  (1.8s)
    TOP:  ãĢĤčĊ(0.07) | /?(0.07) | /**Ċ(0.06) | å¸¸è¯Ĩ(0.06) | ));čĊ(0.06) | $.(0.06) | ))))ĊĊ(0.06) | ,(0.06)
    BOT:  Ð¾ÐºÐ°(-0.05) | (m(-0.05) | _CLIENT(-0.05) | _SHA(-0.05) | ulation(-0.05) | Desk(-0.05) | .create(-0.05) | uation(-0.05)
    ACCEPTED as axis_1305  cumulative_var=0.8736

  [1301]  axes=1306  step_var=0.0015  binary_acc=0.983  gap=0.0883  max_dot=0.0033  (1.8s)
    TOP:  ochastic(0.07) | _num(0.05) | idity(0.05) | minecraft(0.05) | arily(0.05) | è½¬åŁºåĽł(0.05) | Else(0.05) | relu(0.05)
    BOT:  =/(-0.05) | von(-0.05) | Pure(-0.05) | LICENSE(-0.05) | Level(-0.05) | Could(-0.05) | -stock(-0.05) | Contains(-0.05)
    ACCEPTED as axis_1306  cumulative_var=0.8738

  [1302]  axes=1307  step_var=0.0015  binary_acc=0.999  gap=0.0876  max_dot=0.0052  (1.9s)
    TOP:  (K(0.06) | .N(0.06) | (re(0.05) | .com(0.05) | (configuration(0.05) | çŃīå¤ļä¸ª(0.05) | panels(0.05) | pe(0.05)
    BOT:  ('--(-0.05) | åĪĨäº«(-0.05) | ç»ıèĲ¥æ´»åĬ¨(-0.05) | as(-0.05) | å¥½åĥı(-0.05) | loaf(-0.05) | çĬ¯ç½ª(-0.05) | Rectangle(-0.05)
    ACCEPTED as axis_1307  cumulative_var=0.8740

  [1303]  axes=1308  step_var=0.0015  binary_acc=0.974  gap=0.0854  max_dot=0.0113  (1.8s)
    TOP:  -n(0.06) | (all(0.06) | äº«èªī(0.05) | (container(0.05) | )}(0.05) | .command(0.05) | ide(0.05) | REQUIRED(0.05)
    BOT:  _task(-0.07) | .persistence(-0.06) | ä½ĵ(-0.06) | .Common(-0.06) | ä½¿çĶ¨çļĦ(-0.06) | .Attribute(-0.06) | way(-0.05) | _PASSWORD(-0.05)
    ACCEPTED as axis_1308  cumulative_var=0.8742

  [1304]  axes=1309  step_var=0.0015  binary_acc=0.999  gap=0.0863  max_dot=0.0029  (1.9s)
    TOP:  .Project(0.05) | Prompt(0.05) | ADER(0.05) | .add(0.05) | Michael(0.05) | /main(0.05) | Read(0.05) | ä¸ĢèĪ¬(0.05)
    BOT:  à¸¸(-0.06) | .tensor(-0.06) | IBLE(-0.06) | o(-0.06) | igned(-0.06) | _span(-0.06) | .b(-0.05) | viewBox(-0.05)
    ACCEPTED as axis_1309  cumulative_var=0.8744

  [1305]  axes=1310  step_var=0.0015  binary_acc=0.981  gap=0.0876  max_dot=0.0132  (1.9s)
    TOP:  _save(0.05) | mathematics(0.05) | .gr(0.05) | UPDATE(0.05) | Coordinates(0.05) | /~(0.05) | http(0.05) | Cipher(0.05)
    BOT:  _selected(-0.06) | *n(-0.05) | ller(-0.05) | geometry(-0.05) | Ana(-0.05) | R(-0.05) | -too(-0.05) | èĮĥåĽ´åĨħ(-0.05)
    ACCEPTED as axis_1310  cumulative_var=0.8746

  [1306]  axes=1311  step_var=0.0015  binary_acc=0.998  gap=0.0887  max_dot=0.0116  (1.8s)
    TOP:  UF(0.06) | icle(0.05) | since(0.05) | dong(0.05) | LR(0.05) | .text(0.05) | /docs(0.05) | .Linear(0.05)
    BOT:  omat(-0.06) | K(-0.06) | Ĥ¨(-0.05) | Visualization(-0.05) | å¡«åĨĻ(-0.05) | papers(-0.05) | .encoder(-0.05) | protocols(-0.05)
    ACCEPTED as axis_1311  cumulative_var=0.8748

  [1307]  axes=1312  step_var=0.0015  binary_acc=0.974  gap=0.0861  max_dot=0.0112  (1.8s)
    TOP:  bd(0.06) | selection(0.05) | æłĳ(0.05) | .opt(0.05) | ake(0.05) | =file(0.05) | çĪ±ä¸Ĭ(0.05) | ä¼ĺéĽħ(0.05)
    BOT:  ra(-0.06) | .dispatch(-0.06) | .value(-0.06) | CLK(-0.06) | '((-0.05) | .Th(-0.05) | åĳĬ(-0.05) | .nn(-0.05)
    ACCEPTED as axis_1312  cumulative_var=0.8749

  [1308]  axes=1313  step_var=0.0015  binary_acc=0.989  gap=0.0880  max_dot=0.0185  (1.8s)
    TOP:  _lib(0.05) | .Compose(0.05) | levision(0.05) | :")Ċ(0.05) | _pkt(0.05) | çļĦéĥ¨åĪĨ(0.05) | _condition(0.05) | /list(0.05)
    BOT:  Error(-0.07) | G(-0.07) | .d(-0.06) | oot(-0.06) | åĽ¾æĸĩ(-0.06) | ango(-0.06) | æŁ³(-0.06) | ³(-0.06)
    ACCEPTED as axis_1313  cumulative_var=0.8751

  [1309]  axes=1314  step_var=0.0015  binary_acc=0.996  gap=0.0890  max_dot=0.0012  (1.8s)
    TOP:  R(0.06) | ,a(0.06) | è´¢äº§(0.05) | od(0.05) | _detector(0.05) | _embed(0.05) | ig(0.05) | ch(0.05)
    BOT:  .can(-0.06) | }(-0.05) | "]Ċ(-0.05) | .common(-0.05) | execute(-0.05) | /play(-0.05) | .Mode(-0.05) | (...)(-0.05)
    ACCEPTED as axis_1314  cumulative_var=0.8753

  [1310]  axes=1315  step_var=0.0015  binary_acc=0.990  gap=0.0874  max_dot=0.0055  (1.8s)
    TOP:  /common(0.06) | disable(0.05) | from(0.05) | Limits(0.05) | FROM(0.05) | é«ĺå³°(0.05) | _collision(0.05) | åįĥä¸ĩä¸įè¦ģ(0.05)
    BOT:  abi(-0.07) | Gre(-0.06) | ÙĤ(-0.06) | ly(-0.06) | ore(-0.06) | Consum(-0.06) | Id(-0.05) | (The(-0.05)
    ACCEPTED as axis_1315  cumulative_var=0.8755

  [1311]  axes=1316  step_var=0.0015  binary_acc=0.985  gap=0.0890  max_dot=0.0202  (1.8s)
    TOP:  (se(0.06) | false(0.05) | Ðĵ(0.05) | çº¦(0.05) | Hugh(0.05) | ãĥĥãĤ·ãĥ¥(0.05) | anie(0.05) | ÑĦ(0.05)
    BOT:  [:,(-0.06) | r(-0.06) | Copyright(-0.06) | .Line(-0.06) | Script(-0.05) | ][:(-0.05) | "_(-0.05) | ;n(-0.05)
    ACCEPTED as axis_1316  cumulative_var=0.8757

  [1312]  axes=1317  step_var=0.0015  binary_acc=0.989  gap=0.0865  max_dot=0.0316  (1.9s)
    TOP:  rm(0.06) | =true(0.06) | .âĢĿ(0.06) | ãĢĤâĢĿ(0.06) | Im(0.05) | wc(0.05) | ']Ċ(0.05) | _Out(0.05)
    BOT:  ä¿¡ä»°(-0.06) | Category(-0.05) | åĪĨæĶ¯æľºæŀĦ(-0.05) | FA(-0.05) | que(-0.05) | .pc(-0.05) | (sr(-0.05) | ucumber(-0.05)
    ACCEPTED as axis_1317  cumulative_var=0.8759

  [1313]  axes=1318  step_var=0.0015  binary_acc=0.971  gap=0.0871  max_dot=0.0545  (1.8s)
    TOP:  Fraction(0.06) | Block(0.05) | Con(0.05) | (bytes(0.05) | .(0.05) | Successfully(0.05) | æİ¨(0.05) | HttpContext(0.05)
    BOT:  ={"(-0.06) | emy(-0.06) | ess(-0.06) | spread(-0.06) | Fant(-0.05) | irs(-0.05) | uous(-0.05) | (Date(-0.05)
    ACCEPTED as axis_1318  cumulative_var=0.8761

  [1314]  axes=1319  step_var=0.0015  binary_acc=0.991  gap=0.0852  max_dot=0.0149  (1.8s)
    TOP:  SENT(0.05) | egment(0.05) | atic(0.05) | warz(0.05) | CTIONS(0.05) | Paint(0.05) | Groups(0.05) | Kaz(0.05)
    BOT:  éĢļçŁ¥ä¹¦(-0.05) | In(-0.05) | é»ĺ(-0.05) | ,b(-0.05) | .model(-0.05) | .Body(-0.05) | (create(-0.05) | }'(-0.05)
    ACCEPTED as axis_1319  cumulative_var=0.8763

  [1315]  axes=1320  step_var=0.0015  binary_acc=0.988  gap=0.0855  max_dot=0.0159  (1.9s)
    TOP:  Ðº(0.07) | ä½Ĩæĺ¯åľ¨(0.06) | empty(0.06) | .Target(0.06) | /content(0.05) | ="Ċ(0.05) | ournals(0.05) | Physics(0.05)
    BOT:  publication(-0.06) | .Create(-0.06) | answers(-0.06) | entication(-0.05) | è®¡(-0.05) | (sh(-0.05) | from(-0.05) | eps(-0.05)
    ACCEPTED as axis_1320  cumulative_var=0.8764

  [1316]  axes=1321  step_var=0.0015  binary_acc=0.987  gap=0.0858  max_dot=0.0197  (1.8s)
    TOP:  lw(0.05) | Ð¼ÐµÐ½ÑĤ(0.05) | One(0.05) | ISTICS(0.05) | ANCEL(0.05) | å¼ºçĥĪçļĦ(0.05) | .mouse(0.05) | Une(0.05)
    BOT:  ï¼ļĊĊ(-0.06) | .Server(-0.06) | ä¸ºæł¸å¿ĥ(-0.06) | ä¸įäºĪ(-0.05) | æºĲäºİ(-0.05) | å¸¦å¤´(-0.05) | _address(-0.05) | (ll(-0.05)
    ACCEPTED as axis_1321  cumulative_var=0.8766

  [1317]  axes=1322  step_var=0.0015  binary_acc=0.989  gap=0.0854  max_dot=0.0194  (1.8s)
    TOP:  /Public(0.06) | .Standard(0.06) | Annotation(0.06) | Inline(0.05) | ä¸ĭ(0.05) | Layers(0.05) | HA(0.05) | /manage(0.05)
    BOT:  .Runtime(-0.07) | Chart(-0.06) | _std(-0.05) | Tax(-0.05) | (rank(-0.05) | atched(-0.05) | .Tasks(-0.05) | \Controller(-0.05)
    ACCEPTED as axis_1322  cumulative_var=0.8768

  [1318]  axes=1323  step_var=0.0015  binary_acc=0.980  gap=0.0854  max_dot=0.0163  (1.8s)
    TOP:  åĪĽ(0.05) | æĶ¿æ²»(0.05) | ursos(0.05) | å¤ĸ(0.05) | (scale(0.05) | ÑģÐ°Ð¹ÑĤ(0.05) | é¼»(0.05) | Apple(0.05)
    BOT:  .github(-0.06) | !=(-0.06) | -input(-0.05) | _plot(-0.05) | Surv(-0.05) | Cont(-0.05) | []);Ċ(-0.05) | ber(-0.05)
    ACCEPTED as axis_1323  cumulative_var=0.8770

  [1319]  axes=1324  step_var=0.0015  binary_acc=0.991  gap=0.0853  max_dot=0.0283  (1.8s)
    TOP:  æĸ°æĹ¶ä»£(0.06) | Attr(0.05) | .Column(0.05) | =y(0.05) | Publisher(0.05) | è·ĳåĪ°(0.05) | æľīå¤ļä¹Ī(0.05) | åıĳè¾¾åĽ½å®¶(0.05)
    BOT:  Natural(-0.05) | ling(-0.05) | _xy(-0.05) | select(-0.05) | self(-0.05) | table(-0.05) | _INF(-0.05) | ottle(-0.05)
    ACCEPTED as axis_1324  cumulative_var=0.8772

  [1320]  axes=1325  step_var=0.0015  binary_acc=0.995  gap=0.0859  max_dot=0.0099  (1.8s)
    TOP:  ython(0.06) | Around(0.05) | éĺ´(0.05) | é«ĺè¡Ģåİĭ(0.05) | çĿ£ä¿ĥ(0.05) | Cookies(0.05) | (lp(0.05) | Blast(0.05)
    BOT:  out(-0.06) | ./(-0.06) | _item(-0.06) | _speed(-0.06) | ic(-0.05) | éģĩåĪ°äºĨ(-0.05) | J(-0.05) | more(-0.05)
    ACCEPTED as axis_1325  cumulative_var=0.8774

  [1321]  axes=1326  step_var=0.0015  binary_acc=0.996  gap=0.0851  max_dot=0.0230  (1.8s)
    TOP:  And(0.06) | ä»ĸ(0.06) | //(0.06) | åĽ½åĨħ(0.06) | You(0.06) | Ele(0.05) | -T(0.05) | Ts(0.05)
    BOT:  .pix(-0.05) | ,data(-0.05) | Center(-0.05) | entertainment(-0.05) | Topic(-0.05) | }(-0.05) | uito(-0.05) | Scan(-0.05)
    ACCEPTED as axis_1326  cumulative_var=0.8776

  [1322]  axes=1327  step_var=0.0016  binary_acc=0.989  gap=0.0862  max_dot=0.0200  (1.9s)
    TOP:  /E(0.06) | H(0.06) | M(0.06) | ,W(0.06) | Mer(0.06) | ille(0.05) | _to(0.05) | ĉurl(0.05)
    BOT:  åĳ¼åĲ¸(-0.06) | æĮĳæĪĺ(-0.06) | .support(-0.05) | åİŁåĪĻ(-0.05) | å¹³åı°(-0.05) | =wx(-0.05) | ä¸»ç®¡éĥ¨éĹ¨(-0.05) | _pan(-0.05)
    ACCEPTED as axis_1327  cumulative_var=0.8778

  [1323]  axes=1328  step_var=0.0015  binary_acc=0.986  gap=0.0839  max_dot=0.0165  (1.8s)
    TOP:  .label(0.05) | TEST(0.05) | balls(0.05) | BLOCK(0.05) | -heading(0.05) | .J(0.05) | Queue(0.05) | /dist(0.05)
    BOT:  .spring(-0.07) | "The(-0.06) | .Models(-0.06) | printer(-0.06) | å°ºå¯¸(-0.06) | Sort(-0.05) | 'It(-0.05) | Ð´(-0.05)
    ACCEPTED as axis_1328  cumulative_var=0.8779

  [1324]  axes=1329  step_var=0.0015  binary_acc=0.977  gap=0.0877  max_dot=0.0112  (2.0s)
    TOP:  _form(0.07) | de(0.06) | ŀ(0.05) | ìľ¼ë¡ľ(0.05) | ,((0.05) | _zero(0.05) | oxide(0.05) | *"(0.05)
    BOT:  =M(-0.06) | how(-0.05) | IEWS(-0.05) | .AD(-0.05) | .n(-0.05) | Integration(-0.05) | !')Ċ(-0.05) | :x(-0.05)
    ACCEPTED as axis_1329  cumulative_var=0.8781

  [1325]  axes=1330  step_var=0.0015  binary_acc=0.996  gap=0.0865  max_dot=0.0037  (1.9s)
    TOP:  from(0.06) | _detection(0.06) | utils(0.05) | Nation(0.05) | æ²®ä¸§(0.05) | è§£(0.05) | /list(0.05) | .nih(0.05)
    BOT:  ìĿ¼(-0.05) | }=(-0.05) | è®¤åı¯(-0.05) | /send(-0.05) | FTA(-0.05) | .activity(-0.05) | (r(-0.05) | library(-0.05)
    ACCEPTED as axis_1330  cumulative_var=0.8783

  [1326]  axes=1331  step_var=0.0015  binary_acc=0.994  gap=0.0856  max_dot=0.0298  (1.9s)
    TOP:  _workers(0.06) | ories(0.05) | versions(0.05) | into(0.05) | attributes(0.05) | items(0.05) | och(0.05) | .tools(0.05)
    BOT:  çĦ¦(-0.06) | child(-0.06) | ä¸įè¦ģ(-0.05) | æķħéļľ(-0.05) | åĲ«ä¹ī(-0.05) | æŀĦ(-0.05) | çľĭå¥½(-0.05) | tasks(-0.05)
    ACCEPTED as axis_1331  cumulative_var=0.8785

  [1327]  axes=1332  step_var=0.0015  binary_acc=0.993  gap=0.0831  max_dot=0.0054  (1.9s)
    TOP:  _dot(0.06) | ],Ċ(0.06) | _s(0.05) | ase(0.05) | import(0.05) | lang(0.05) | }ĊĊ(0.05) | _edge(0.05)
    BOT:  quests(-0.06) | Ð±Ñĭ(-0.06) | admin(-0.06) | .Byte(-0.06) | .Screen(-0.05) | forum(-0.05) | enemy(-0.05) | YY(-0.05)
    ACCEPTED as axis_1332  cumulative_var=0.8787

  [1328]  axes=1333  step_var=0.0015  binary_acc=0.979  gap=0.0841  max_dot=0.0103  (1.9s)
    TOP:  TYPE(0.05) | åĪĽåĬŀ(0.05) | çĶ³è¯·äºº(0.05) | relation(0.05) | not(0.05) | èĲ¥åħ»(0.05) | :(0.05) | additional(0.05)
    BOT:  (frame(-0.06) | snake(-0.06) | invoke(-0.05) | .platform(-0.05) | çģ«(-0.05) | .open(-0.05) | Connected(-0.05) | Law(-0.05)
    ACCEPTED as axis_1333  cumulative_var=0.8789

  [1329]  axes=1334  step_var=0.0016  binary_acc=0.996  gap=0.0872  max_dot=0.0012  (1.9s)
    TOP:  iance(0.07) | Ð¡(0.07) | SI(0.06) | _exact(0.06) | pace(0.06) | rost(0.06) | ume(0.06) | çĥŃ(0.06)
    BOT:  Probe(-0.05) | Â»(-0.05) | _as(-0.05) | Store(-0.05) | "In(-0.05) | y(-0.05) | "W(-0.05) | #"(-0.05)
    ACCEPTED as axis_1334  cumulative_var=0.8790

  [1330]  axes=1335  step_var=0.0015  binary_acc=0.986  gap=0.0865  max_dot=0.0099  (1.8s)
    TOP:  ,True(0.06) | job(0.06) | Now(0.05) | Bootstrap(0.05) | Over(0.05) | .preprocessing(0.05) | å¤©(0.05) | è¯ķåīĤ(0.05)
    BOT:  dim(-0.06) | (sender(-0.06) | åĽ´(-0.06) | ]))Ċ(-0.05) | .warn(-0.05) | .special(-0.05) | _after(-0.05) | __)Ċ(-0.05)
    ACCEPTED as axis_1335  cumulative_var=0.8792

  [1331]  axes=1336  step_var=0.0015  binary_acc=0.995  gap=0.0855  max_dot=0.0289  (1.9s)
    TOP:  è§£éĻ¤(0.06) | çīµæĮĤ(0.06) | Ð»Ð°(0.06) | çļĦä¼ĺçĤ¹(0.05) | åĿĲèĲ½åľ¨(0.05) | çŃīæĥħåĨµ(0.05) | ä¸Ģä»½(0.05) | attrs(0.05)
    BOT:  https(-0.06) | paring(-0.05) | Med(-0.05) | shares(-0.05) | _odd(-0.05) | .Enum(-0.05) | .Model(-0.05) | Ð°ÑģÑģ(-0.05)
    ACCEPTED as axis_1336  cumulative_var=0.8794

  [1332]  axes=1337  step_var=0.0015  binary_acc=0.994  gap=0.0855  max_dot=0.0406  (1.9s)
    TOP:  ç«ŀäºīå¯¹æīĭ(0.06) | ning(0.05) | im(0.05) | -Type(0.05) | /op(0.05) | .elements(0.05) | ä¿ĥè¿Ľ(0.05) | _average(0.05)
    BOT:  import(-0.06) | ours(-0.06) | Indices(-0.06) | æĢħ(-0.06) | Ã¡(-0.05) | Invest(-0.05) | ICAL(-0.05) | ÑĤÐ¾ÑĢ(-0.05)
    ACCEPTED as axis_1337  cumulative_var=0.8796

  [1333]  axes=1338  step_var=0.0015  binary_acc=0.999  gap=0.0841  max_dot=0.0156  (1.9s)
    TOP:  .scatter(0.06) | chai(0.05) | voluntarily(0.05) | Des(0.05) | æ¯Ľ(0.05) | _ratio(0.05) | Payment(0.05) | ,label(0.05)
    BOT:  .metadata(-0.06) | us(-0.06) | midt(-0.05) | phere(-0.05) | ange(-0.05) | äº«(-0.05) | q(-0.05) | ited(-0.05)
    ACCEPTED as axis_1338  cumulative_var=0.8798

  [1334]  axes=1339  step_var=0.0015  binary_acc=0.995  gap=0.0847  max_dot=0.0127  (1.8s)
    TOP:  PHP(0.06) | æĦıå¤§åĪ©(0.05) | .prompt(0.05) | /sdk(0.05) | .relu(0.05) | _lib(0.05) | (layers(0.05) | å¤Ħå¤Ħ(0.05)
    BOT:  amble(-0.06) | Route(-0.06) | .generate(-0.06) | .timedelta(-0.05) | ä¸¤æ¬¡(-0.05) | _grid(-0.05) | Adam(-0.05) | ich(-0.05)
    ACCEPTED as axis_1339  cumulative_var=0.8800

  [1335]  axes=1340  step_var=0.0015  binary_acc=0.979  gap=0.0843  max_dot=0.0146  (1.8s)
    TOP:  {Ċ(0.06) | !ĊĊ(0.06) | Extended(0.06) | ={{Ċ(0.06) | """Ċ(0.05) | """čĊ(0.05) | =ax(0.05) | èĢĥè¯ķ(0.05)
    BOT:  LE(-0.05) | able(-0.05) | Ð½ÑĭÐµ(-0.05) | -out(-0.05) | _usage(-0.05) | theses(-0.05) | Sch(-0.05) | .sa(-0.05)
    ACCEPTED as axis_1340  cumulative_var=0.8801

  [1336]  axes=1341  step_var=0.0015  binary_acc=0.974  gap=0.0842  max_dot=0.0184  (1.9s)
    TOP:  Content(0.05) | cribe(0.05) | è¢«(0.05) | -twitter(0.05) | F(0.05) | _income(0.05) | ET(0.05) | çĬ¯ç½ª(0.05)
    BOT:  .apps(-0.06) | .No(-0.06) | .scale(-0.06) | /models(-0.05) | roker(-0.05) | è¿ĽäºĨ(-0.05) | Mod(-0.05) | warnings(-0.05)
    ACCEPTED as axis_1341  cumulative_var=0.8803

  [1337]  axes=1342  step_var=0.0015  binary_acc=0.990  gap=0.0845  max_dot=0.0054  (1.9s)
    TOP:  RequiredMixin(0.06) | .active(0.06) | .gr(0.05) | clare(0.05) | almost(0.05) | Days(0.05) | World(0.05) | Results(0.05)
    BOT:  æĺ¯ä¸Ģä¸ª(-0.06) | åĽłèĢĮ(-0.06) | Win(-0.05) | .oracle(-0.05) | _window(-0.05) | .callback(-0.05) | (code(-0.05) | etic(-0.05)
    ACCEPTED as axis_1342  cumulative_var=0.8805

  [1338]  axes=1343  step_var=0.0015  binary_acc=0.994  gap=0.0833  max_dot=0.0106  (1.9s)
    TOP:  .author(0.06) | &(0.06) | _connect(0.05) | .metadata(0.05) | Well(0.05) | .I(0.05) | Â»(0.05) | //Ċ(0.05)
    BOT:  è¾ĥå¼ºçļĦ(-0.05) | Remote(-0.05) | Ð¾(-0.05) | ç¬¬ä¸Ģ(-0.05) | .tree(-0.05) | registry(-0.05) | èĳī(-0.05) | imento(-0.05)
    ACCEPTED as axis_1343  cumulative_var=0.8807

  [1339]  axes=1344  step_var=0.0015  binary_acc=0.989  gap=0.0879  max_dot=0.0447  (1.9s)
    TOP:  M(0.08) | m(0.06) | _m(0.06) | B(0.06) | V(0.06) | å³Ń(0.06) | t(0.06) | åı³æīĭ(0.06)
    BOT:  communications(-0.05) | putation(-0.05) | :(-0.05) | setup(-0.05) | Mathematical(-0.05) | Demo(-0.05) | ertility(-0.05) | Kal(-0.04)
    ACCEPTED as axis_1344  cumulative_var=0.8809

  [1340]  axes=1345  step_var=0.0015  binary_acc=0.999  gap=0.0844  max_dot=0.0066  (1.8s)
    TOP:  å¤´(0.06) | .training(0.06) | .am(0.05) | kt(0.05) | founding(0.05) | odynamics(0.05) | åĩĨå¤ĩ(0.05) | igits(0.05)
    BOT:  =model(-0.06) | (real(-0.05) | _hosts(-0.05) | )]ĊĊ(-0.05) | .role(-0.05) | _STREAM(-0.05) | (result(-0.05) | /Ċ(-0.05)
    ACCEPTED as axis_1345  cumulative_var=0.8811

  [1341]  axes=1346  step_var=0.0015  binary_acc=0.991  gap=0.0844  max_dot=0.0113  (1.8s)
    TOP:  criterion(0.05) | -query(0.05) | Natural(0.05) | Encode(0.05) | continued(0.05) | .getImage(0.05) | phone(0.05) | ene(0.05)
    BOT:  (inter(-0.06) | _site(-0.06) | Enumeration(-0.06) | Price(-0.05) | -main(-0.05) | ii(-0.05) | æķĻè®Ń(-0.05) | æ(-0.05)
    ACCEPTED as axis_1346  cumulative_var=0.8812

  [1342]  axes=1347  step_var=0.0015  binary_acc=0.984  gap=0.0854  max_dot=0.0130  (1.8s)
    TOP:  æ²¡(0.06) | ists(0.06) | subscribe(0.06) | åıįæĬĹ(0.05) | lg(0.05) | è¿ĳ(0.05) | azard(0.05) | rs(0.05)
    BOT:  "]Ċ(-0.06) | }".(-0.06) | (-0.06) | "].(-0.05) | .)Ċ(-0.05) | OLUTION(-0.05) | mM(-0.05) | Ċ(-0.05)
    ACCEPTED as axis_1347  cumulative_var=0.8814

  [1343]  axes=1348  step_var=0.0015  binary_acc=0.972  gap=0.0872  max_dot=0.0027  (1.8s)
    TOP:  å¯¹äºİ(0.05) | paper(0.05) | Ok(0.05) | makes(0.05) | .text(0.05) | /re(0.05) | æľĹè¯µ(0.05) | [](0.05)
    BOT:  :],(-0.05) | .Task(-0.05) | ologÃŃa(-0.05) | ä¹¡(-0.05) | ie(-0.05) | AVAILABLE(-0.05) | !');Ċ(-0.05) | boards(-0.05)
    ACCEPTED as axis_1348  cumulative_var=0.8816

  [1344]  axes=1349  step_var=0.0015  binary_acc=0.994  gap=0.0861  max_dot=0.0036  (1.9s)
    TOP:  ³(0.07) | agle(0.06) | Ð°ÐµÑĤÑģÑı(0.06) | resources(0.06) | .Content(0.05) | sequence(0.05) | åı¯ä»¥(0.05) | quat(0.05)
    BOT:  python(-0.06) | _square(-0.06) | Registered(-0.05) | _program(-0.05) | _weather(-0.05) | modules(-0.05) | u(-0.05) | Irish(-0.05)
    ACCEPTED as axis_1349  cumulative_var=0.8818

  [1345]  axes=1350  step_var=0.0015  binary_acc=0.992  gap=0.0862  max_dot=0.0119  (1.8s)
    TOP:  .gui(0.05) | _stack(0.05) | .AutoField(0.05) | (i(0.05) | formats(0.05) | (random(0.05) | _iterations(0.05) | organic(0.05)
    BOT:  .prev(-0.06) | ï¼ĮĊ(-0.06) | ga(-0.05) | wp(-0.05) | olated(-0.05) | ">#(-0.05) | ï¼ŁâĢĿ(-0.05) | */(-0.05)
    ACCEPTED as axis_1350  cumulative_var=0.8820

  [1346]  axes=1351  step_var=0.0015  binary_acc=0.967  gap=0.0838  max_dot=0.0055  (1.8s)
    TOP:  edge(0.05) | Numbers(0.05) | æľ¬è´¨(0.05) | å·²(0.05) | _definition(0.05) | æī¹åĩĨ(0.05) | Year(0.05) | .channel(0.05)
    BOT:  .USER(-0.06) | (TAG(-0.05) | _model(-0.05) | }(-0.05) | .download(-0.05) | ref(-0.05) | onas(-0.05) | >(-0.05)
    ACCEPTED as axis_1351  cumulative_var=0.8821

  [1347]  axes=1352  step_var=0.0015  binary_acc=0.994  gap=0.0863  max_dot=0.0301  (1.9s)
    TOP:  -font(0.05) | /src(0.05) | .q(0.05) | .output(0.05) | /mat(0.05) | .functions(0.05) | _NUMBER(0.05) | .Remove(0.05)
    BOT:  ts(-0.06) | Extras(-0.05) | æ¡£(-0.05) | Weight(-0.05) | æłĩæĺİ(-0.05) | (kwargs(-0.05) | (([(-0.05) | .exceptions(-0.05)
    ACCEPTED as axis_1352  cumulative_var=0.8823

  [1348]  axes=1353  step_var=0.0016  binary_acc=0.982  gap=0.0852  max_dot=0.0397  (1.9s)
    TOP:  '}(0.06) | çŁŃæľŁ(0.06) | cache(0.06) | (props(0.05) | -Y(0.05) | orphism(0.05) | (address(0.05) | (dtype(0.05)
    BOT:  Ð½ÑĭÐµ(-0.06) | à¸´(-0.06) | Fluid(-0.05) | .blocks(-0.05) | åįļ(-0.05) | ä½ł(-0.05) | UK(-0.05) | à¹ĩà¸ģ(-0.05)
    ACCEPTED as axis_1353  cumulative_var=0.8825

  [1349]  axes=1354  step_var=0.0016  binary_acc=0.990  gap=0.0832  max_dot=0.0063  (1.9s)
    TOP:  r(0.06) | _group(0.05) | ."(0.05) | åĨ³å¿ĥ(0.05) | ãĢİ(0.05) | /mark(0.05) | -src(0.05) | expand(0.05)
    BOT:  ä¹ī(-0.07) | pre(-0.06) | Since(-0.06) | éķ¿(-0.06) | åĪ¤å®ļ(-0.06) | æ³ķå¾ĭæ³ķè§Ħ(-0.06) | _backend(-0.06) | _US(-0.06)
    ACCEPTED as axis_1354  cumulative_var=0.8827

  [1350]  axes=1355  step_var=0.0015  binary_acc=0.991  gap=0.0844  max_dot=0.0177  (1.9s)
    TOP:  Ð½Ð¾Ð³Ð¾(0.06) | """ĊĊĊ(0.06) | ackage(0.06) | åĩºå¢ĥ(0.06) | ä½İ(0.06) | urn(0.05) | =dict(0.05) | /en(0.05)
    BOT:  emporary(-0.05) | ____(-0.05) | fa(-0.05) | å¸Ĥåľºä»½é¢Ŀ(-0.05) | .allow(-0.05) | Added(-0.05) | ĉ(-0.05) | suis(-0.05)
    ACCEPTED as axis_1355  cumulative_var=0.8829

  [1351]  axes=1356  step_var=0.0015  binary_acc=0.993  gap=0.0843  max_dot=0.0484  (1.9s)
    TOP:  be(0.06) | ãĢĳ(0.06) | ç§ĭåŃ£(0.06) | _rotation(0.06) | INES(0.05) | stry(0.05) | :]Ċ(0.05) | Z(0.05)
    BOT:  _controller(-0.05) | .mean(-0.05) | vis(-0.05) | buttons(-0.05) | Democracy(-0.05) | -syntax(-0.05) | Ð¿(-0.05) | =Y(-0.05)
    ACCEPTED as axis_1356  cumulative_var=0.8830

  [1352]  axes=1357  step_var=0.0015  binary_acc=0.979  gap=0.0855  max_dot=0.0699  (1.8s)
    TOP:  ominator(0.06) | æıĲæĹ©(0.06) | .long(0.05) | _sensor(0.05) | OULD(0.05) | _session(0.05) | .pkg(0.05) | _NODE(0.05)
    BOT:  /K(-0.06) | /legal(-0.06) | ,t(-0.05) | ://(-0.05) | x(-0.05) | .",Ċ(-0.05) | amin(-0.05) | ×Ļ(-0.05)
    ACCEPTED as axis_1357  cumulative_var=0.8832

  [1353]  axes=1358  step_var=0.0016  binary_acc=0.997  gap=0.0846  max_dot=0.0356  (1.9s)
    TOP:  mente(0.06) | !!Ċ(0.06) | Ð½ÑĭÑħ(0.06) | ales(0.05) | .ie(0.05) | èĤ¡æĿĥæĬķèµĦ(0.05) | vs(0.05) | Ð½ÑĥÑİ(0.05)
    BOT:  S(-0.06) | èĦĬ(-0.05) | æķĻèĤ²(-0.05) | head(-0.05) | (self(-0.05) | r(-0.05) | æµĨ(-0.05) | âĢİ(-0.05)
    ACCEPTED as axis_1358  cumulative_var=0.8834

  [1354]  axes=1359  step_var=0.0015  binary_acc=0.993  gap=0.0856  max_dot=0.0514  (1.8s)
    TOP:  aders(0.07) | ers(0.06) | iÃ³n(0.06) | |(0.06) | ati(0.06) | X(0.05) | LL(0.05) | es(0.05)
    BOT:  }}ĊĊ(-0.06) | -bit(-0.05) | .).ĊĊ(-0.05) | Function(-0.05) | _entity(-0.05) | As(-0.05) | });ĊĊ(-0.05) | ]Ċ(-0.05)
    ACCEPTED as axis_1359  cumulative_var=0.8836

  [1355]  axes=1360  step_var=0.0015  binary_acc=1.000  gap=0.0836  max_dot=0.0222  (1.8s)
    TOP:  è´¦æĪ·(0.05) | ossa(0.05) | Gra(0.05) | Hide(0.05) | ä¸ĳ(0.05) | StartTime(0.05) | /users(0.05) | OM(0.05)
    BOT:  (var(-0.05) | ÑħÐ°ÑĢÐ°ÐºÑĤÐµÑĢÐ¸ÑģÑĤ(-0.05) | -image(-0.05) | iÃ³n(-0.05) | Anyway(-0.05) | Øª(-0.05) | ognition(-0.05) | That(-0.05)
    ACCEPTED as axis_1360  cumulative_var=0.8838

  [1356]  axes=1361  step_var=0.0015  binary_acc=0.976  gap=0.0844  max_dot=0.0094  (1.8s)
    TOP:  sb(0.06) | parent(0.06) | (\(0.05) | fathers(0.05) | èĩªåĪ¶(0.05) | .Transport(0.05) | ___(0.05) | ourced(0.05)
    BOT:  apply(-0.06) | æĸ¹(-0.06) | /html(-0.05) | limit(-0.05) | å¼Ł(-0.05) | Individual(-0.05) | çľ¨(-0.05) | PLICATION(-0.05)
    ACCEPTED as axis_1361  cumulative_var=0.8839

  [1357]  axes=1362  step_var=0.0015  binary_acc=0.990  gap=0.0838  max_dot=0.0168  (1.9s)
    TOP:  um(0.06) | é¾Ħ(0.06) | Ðµ(0.06) | ata(0.06) | /template(0.06) | ic(0.06) | bour(0.06) | adir(0.06)
    BOT:  Our(-0.05) | If(-0.05) | .next(-0.05) | Language(-0.05) | ÐºÐ°Ñħ(-0.05) | (stream(-0.05) | '},Ċ(-0.05) | Marina(-0.05)
    ACCEPTED as axis_1362  cumulative_var=0.8841

  [1358]  axes=1363  step_var=0.0015  binary_acc=0.997  gap=0.0846  max_dot=0.0333  (1.8s)
    TOP:  .swing(0.05) | pper(0.05) | Array(0.05) | /py(0.05) | su(0.05) | .Scene(0.05) | ÑĤÐµ(0.05) | å¸ĤåĮº(0.05)
    BOT:  !(-0.06) | æĢģåº¦(-0.05) | >Ċ(-0.05) | '**(-0.05) | .ĊĊ(-0.05) | ;ĊĊĊĊ(-0.05) | æĪĺ(-0.05) | .Ċ(-0.05)
    ACCEPTED as axis_1363  cumulative_var=0.8843

  [1359]  axes=1364  step_var=0.0015  binary_acc=0.981  gap=0.0845  max_dot=0.0130  (1.9s)
    TOP:  _flat(0.06) | _,(0.06) | igation(0.05) | .low(0.05) | .Skip(0.05) | -danger(0.05) | ç¬¬ä¸Ģ(0.05) | """Ċ(0.05)
    BOT:  âĢĻt(-0.06) | 't(-0.06) | ÑĢÐ°Ð±Ð¾ÑĤÑĭ(-0.05) | Print(-0.05) | llvm(-0.05) | CS(-0.05) | íĬ¸(-0.05) | Av(-0.05)
    ACCEPTED as axis_1364  cumulative_var=0.8845

  [1360]  axes=1365  step_var=0.0016  binary_acc=0.995  gap=0.0835  max_dot=0.0053  (1.8s)
    TOP:  L(0.06) | Ð´Ð°Ð½Ð½ÑĭÑħ(0.06) | .static(0.06) | Ðļ(0.05) | èģĮæĿĥ(0.05) | éĶĻè¯¯(0.05) | Promise(0.05) | ';ĊĊ(0.05)
    BOT:  )$(-0.07) | olding(-0.05) | Fox(-0.05) | avra(-0.05) | Ċ    Ċ(-0.05) | on(-0.05) | odo(-0.05) | {(-0.05)
    ACCEPTED as axis_1365  cumulative_var=0.8847

  [1361]  axes=1366  step_var=0.0016  binary_acc=0.999  gap=0.0850  max_dot=0.0130  (1.8s)
    TOP:  PDF(0.06) | Compat(0.06) | Choice(0.05) | .base(0.05) | ksi(0.05) | å¾ĢæĿ¥(0.05) | /n(0.05) | (User(0.05)
    BOT:  itors(-0.06) | èĦ¾(-0.06) | nia(-0.05) | éĽ¨(-0.05) | çļĦåľŁåľ°(-0.05) | æĺıè¿·(-0.05) | `)(-0.05) | depth(-0.05)
    ACCEPTED as axis_1366  cumulative_var=0.8848

  [1362]  axes=1367  step_var=0.0016  binary_acc=0.989  gap=0.0845  max_dot=0.0253  (1.9s)
    TOP:  B(0.06) | âĢĻ,(0.05) | ];Ċ(0.05) | ",(0.05) | .CL(0.05) | has(0.05) | ),Ċ(0.05) | Tur(0.05)
    BOT:  binations(-0.06) | Comments(-0.06) | ä»¥ä¸ĬçļĦ(-0.06) | we(-0.06) | gd(-0.06) | è¿Ĳè¾ĵ(-0.06) | arsing(-0.06) | ect(-0.05)
    ACCEPTED as axis_1367  cumulative_var=0.8850

  [1363]  axes=1368  step_var=0.0015  binary_acc=0.998  gap=0.0822  max_dot=0.0105  (1.9s)
    TOP:  _TABLE(0.06) | Sampling(0.05) | uir(0.05) | AD(0.05) | alyze(0.05) | ÐµÑģÑĤÐ¸(0.05) | éķ¿(0.05) | ä¸Ŀ(0.05)
    BOT:  (DB(-0.06) | {});Ċ(-0.05) | _pagination(-0.05) | _functions(-0.05) | ,ĊĊ(-0.05) | }>Ċ(-0.05) | ").ĊĊ(-0.05) | NULL(-0.05)
    ACCEPTED as axis_1368  cumulative_var=0.8852

  [1364]  axes=1369  step_var=0.0015  binary_acc=0.978  gap=0.0827  max_dot=0.0265  (1.9s)
    TOP:  /download(0.05) | ifacts(0.05) | He(0.05) | _counter(0.05) | except(0.05) | Foreground(0.05) | ll(0.05) | Normalize(0.05)
    BOT:  fon(-0.06) | }.Ċ(-0.06) | os(-0.06) | åĪ©çİĩ(-0.05) | âĢĿï¼Į(-0.05) | );Ċ(-0.05) | ",(-0.05) | æ¶īå«Į(-0.05)
    ACCEPTED as axis_1369  cumulative_var=0.8854

  [1365]  axes=1370  step_var=0.0016  binary_acc=0.991  gap=0.0845  max_dot=0.0063  (1.9s)
    TOP:  arrow(0.05) | utilities(0.05) | TO(0.05) | ç»ĵæĿŁäºĨ(0.05) | åıªæľī(0.05) | self(0.05) | ryan(0.05) | cast(0.05)
    BOT:  /js(-0.06) | .sf(-0.06) | _img(-0.05) | hape(-0.05) | ")]Ċ(-0.05) | _movies(-0.05) | -library(-0.05) | ,is(-0.05)
    ACCEPTED as axis_1370  cumulative_var=0.8856

  [1366]  axes=1371  step_var=0.0016  binary_acc=0.981  gap=0.0835  max_dot=0.0169  (1.9s)
    TOP:  ...(0.06) | ...(0.06) | ï¼ļĊ(0.06) | ><(0.05) | =os(0.05) | axis(0.05) | '(0.05) | å¤įå·¥å¤įäº§(0.05)
    BOT:  sdk(-0.05) | Devices(-0.05) | .functions(-0.05) | Doug(-0.05) | (Path(-0.05) | MR(-0.05) | _stats(-0.05) | -custom(-0.05)
    ACCEPTED as axis_1371  cumulative_var=0.8857

  [1367]  axes=1372  step_var=0.0015  binary_acc=0.974  gap=0.0855  max_dot=0.0594  (1.8s)
    TOP:  /con(0.05) | FT(0.05) | (plot(0.05) | Mathematical(0.05) | "These(0.05) | Transpose(0.05) | dark(0.05) | property(0.05)
    BOT:  }Ċ(-0.06) | (loss(-0.06) | *******Ċ(-0.06) | annes(-0.06) | _ITEM(-0.06) | }čĊ(-0.05) | bold(-0.05) | ä»ª(-0.05)
    ACCEPTED as axis_1372  cumulative_var=0.8859

  [1368]  axes=1373  step_var=0.0015  binary_acc=0.990  gap=0.0840  max_dot=0.0466  (1.9s)
    TOP:  Tang(0.05) | .rb(0.05) | C(0.05) | Default(0.05) | .Graph(0.05) | (),(0.05) | .spawn(0.05) | åıĺéĿ©(0.05)
    BOT:  /lib(-0.07) | /con(-0.06) | (key(-0.05) | backend(-0.05) | "./(-0.05) | ä¸įçŁ¥éģĵ(-0.05) | .action(-0.05) | lambda(-0.05)
    ACCEPTED as axis_1373  cumulative_var=0.8861

  [1369]  axes=1374  step_var=0.0015  binary_acc=0.966  gap=0.0815  max_dot=0.0324  (1.8s)
    TOP:  éħįæĸ¹(0.05) | Days(0.05) | æĸĩæĺİ(0.05) | äººåĿĩ(0.05) | ewolf(0.05) | =form(0.05) | he(0.05) | _the(0.05)
    BOT:  el(-0.05) | æ²¡æľī(-0.05) | navbar(-0.05) | ?'(-0.05) | ope(-0.05) | _sources(-0.05) | éĻµ(-0.05) | obj(-0.05)
    ACCEPTED as axis_1374  cumulative_var=0.8863

  [1370]  axes=1375  step_var=0.0016  binary_acc=0.973  gap=0.0826  max_dot=0.0085  (1.8s)
    TOP:  styles(0.06) | _auto(0.05) | De(0.05) | è¯Ń(0.05) | (0.05) | éĺ¶å±Ĥ(0.05) | Timestamp(0.05) | _return(0.05)
    BOT:  ä¸Ģé¡¹(-0.07) | )]Ċ(-0.06) | æĹ¶éĹ´(-0.05) | ddd(-0.05) | .)ĊĊ(-0.05) | åĮĸ(-0.05) | ï¼İ(-0.05) | .bn(-0.05)
    ACCEPTED as axis_1375  cumulative_var=0.8864

  [1371]  axes=1376  step_var=0.0016  binary_acc=0.986  gap=0.0845  max_dot=0.0028  (1.8s)
    TOP:  åħļå§Ķ(0.05) | åģ¥èº«(0.05) | $',(0.05) | .querySelector(0.05) | å·¥èīº(0.05) | èİ·å¾Ĺ(0.05) | forme(0.05) | è¯ķéªĮåĮº(0.05)
    BOT:  structures(-0.06) | At(-0.06) | E(-0.05) | è¯ĨåĪ«(-0.05) | _fast(-0.05) | KS(-0.05) | =false(-0.05) | .valid(-0.05)
    ACCEPTED as axis_1376  cumulative_var=0.8866

  [1372]  axes=1377  step_var=0.0016  binary_acc=0.978  gap=0.0849  max_dot=0.0082  (1.9s)
    TOP:  El(0.05) | ï¼ļ(0.05) | AND(0.05) | Online(0.05) | =obj(0.05) | .pad(0.05) | _DEVICE(0.05) | acts(0.05)
    BOT:  Hours(-0.06) | æīį(-0.06) | Family(-0.06) | =data(-0.05) | <(-0.05) | Repository(-0.05) | Ð°Ð½Ð¸Ð¸(-0.05) | .Load(-0.05)
    ACCEPTED as axis_1377  cumulative_var=0.8868

  [1373]  axes=1378  step_var=0.0016  binary_acc=0.998  gap=0.0863  max_dot=0.0236  (1.9s)
    TOP:  "//(0.05) | .Str(0.05) | ...(0.05) | }",Ċ(0.05) | ]</(0.05) | VICES(0.05) | _news(0.05) | /Documents(0.05)
    BOT:  èĥ¸(-0.06) | _type(-0.05) | LOB(-0.05) | .User(-0.05) | åħ´(-0.05) | Appro(-0.05) | .Chrome(-0.05) | Exc(-0.05)
    ACCEPTED as axis_1378  cumulative_var=0.8870

  [1374]  axes=1379  step_var=0.0016  binary_acc=0.994  gap=0.0854  max_dot=0.0073  (1.9s)
    TOP:  æł¼(0.06) | SW(0.06) | so(0.06) | ä½łè¿ĺ(0.05) | dr(0.05) | _meta(0.05) | .crypto(0.05) | Desc(0.05)
    BOT:  turtle(-0.06) | Ji(-0.05) | ?q(-0.05) | varies(-0.05) | /view(-0.05) | Genetics(-0.05) | /*Ċ(-0.05) | as(-0.05)
    ACCEPTED as axis_1379  cumulative_var=0.8871

  [1375]  axes=1380  step_var=0.0016  binary_acc=0.983  gap=0.0826  max_dot=0.0087  (2.0s)
    TOP:  èĩ³(0.06) | (theta(0.06) | .:(0.06) | operations(0.05) | ï¼ĮâĢľ(0.05) | ìķ½(0.05) | .Parameters(0.05) | jango(0.05)
    BOT:  .Global(-0.05) | Ðł(-0.05) | _genre(-0.05) | .chapter(-0.05) | .Generic(-0.05) | lements(-0.05) | Outer(-0.05) | å¼Ģå°ģ(-0.05)
    ACCEPTED as axis_1380  cumulative_var=0.8873

  [1376]  axes=1381  step_var=0.0016  binary_acc=0.994  gap=0.0840  max_dot=0.0279  (1.9s)
    TOP:  -License(0.07) | Transfer(0.05) | carousel(0.05) | atoire(0.05) | AFP(0.05) | implementation(0.05) | ('(0.05) | .selector(0.05)
    BOT:  Support(-0.05) | ç®±(-0.05) | èĦī(-0.05) | aren(-0.05) | All(-0.05) | Books(-0.05) | ON(-0.05) | chewing(-0.04)
    ACCEPTED as axis_1381  cumulative_var=0.8875

  [1377]  axes=1382  step_var=0.0015  binary_acc=0.996  gap=0.0837  max_dot=0.0423  (1.9s)
    TOP:  ,and(0.06) | _save(0.06) | ;%(0.05) | .samples(0.05) | \x(0.05) | _boxes(0.05) | rates(0.05) | Points(0.05)
    BOT:  éĺ²(-0.07) | -build(-0.05) | e(-0.05) | åĪ¹è½¦(-0.05) | çĨŁæĤī(-0.05) | /ap(-0.05) | .material(-0.05) | icky(-0.05)
    ACCEPTED as axis_1382  cumulative_var=0.8877

  [1378]  axes=1383  step_var=0.0015  binary_acc=0.996  gap=0.0816  max_dot=0.0035  (1.8s)
    TOP:  .visual(0.05) | -motion(0.05) | Mill(0.05) | .desktop(0.05) | Ð³Ð¾Ð´Ð°(0.05) | å·¦åı³(0.05) | (loss(0.05) | .svg(0.05)
    BOT:  ħ(-0.06) | ä»·(-0.06) | èĩªæŁ¥(-0.06) | æ¯įäº²(-0.06) | ';Ċ(-0.06) | swap(-0.06) | éĩı(-0.05) | Finite(-0.05)
    ACCEPTED as axis_1383  cumulative_var=0.8878

  [1379]  axes=1384  step_var=0.0015  binary_acc=0.996  gap=0.0825  max_dot=0.0033  (1.9s)
    TOP:  ins(0.06) | ÙĬØ©(0.06) | .Please(0.06) | antes(0.06) | Least(0.05) | Prop(0.05) | Time(0.05) | Evaluator(0.05)
    BOT:  æ¯ıä¸Ģä½į(-0.05) | Differences(-0.05) | EFFECT(-0.05) | _props(-0.05) | "A(-0.05) | '],(-0.05) | y(-0.04) | _dist(-0.04)
    ACCEPTED as axis_1384  cumulative_var=0.8880

  [1380]  axes=1385  step_var=0.0016  binary_acc=0.998  gap=0.0832  max_dot=0.0132  (1.9s)
    TOP:  (),(0.06) | (0.06) | ='(0.05) | Reject(0.05) | class(0.05) | new(0.05) | çľł(0.05) | /open(0.05)
    BOT:  æĬĺ(-0.07) | av(-0.06) | sen(-0.06) | acid(-0.06) | letter(-0.06) | yu(-0.05) | ch(-0.05) | ysics(-0.05)
    ACCEPTED as axis_1385  cumulative_var=0.8882

  [1381]  axes=1386  step_var=0.0016  binary_acc=0.970  gap=0.0849  max_dot=0.0185  (1.8s)
    TOP:  (0.06) | ĉ(0.05) | _frame(0.05) | _padding(0.05) | "))ĊĊ(0.05) | Flutter(0.05) | .ĊĊĊ(0.05) | ]/(0.05)
    BOT:  amin(-0.06) | æĺ¯ä¸Ģç§į(-0.06) | ylum(-0.06) | no(-0.06) | aya(-0.06) | Ã¥(-0.06) | ivered(-0.06) | ä½¿(-0.06)
    ACCEPTED as axis_1386  cumulative_var=0.8884

  [1382]  axes=1387  step_var=0.0016  binary_acc=0.990  gap=0.0837  max_dot=0.0021  (1.8s)
    TOP:  _entries(0.06) | pygame(0.06) | Recipes(0.05) | _WIDTH(0.05) | ###ĊĊ(0.05) | programs(0.05) | Iterations(0.05) | ipsoid(0.05)
    BOT:  fill(-0.06) | (hidden(-0.06) | .auto(-0.06) | ot(-0.05) | -Za(-0.05) | ["(-0.05) | /me(-0.05) | Gender(-0.05)
    ACCEPTED as axis_1387  cumulative_var=0.8885

  [1383]  axes=1388  step_var=0.0016  binary_acc=0.995  gap=0.0836  max_dot=0.0133  (1.8s)
    TOP:  åİŁåĳĬ(0.06) | .Generic(0.06) | fill(0.06) | Plus(0.05) | stmt(0.05) | Throw(0.05) | æĻ¯(0.05) | .cpu(0.05)
    BOT:  -N(-0.06) | æĭīèĲ¨(-0.05) | b(-0.05) | attach(-0.05) | .spatial(-0.05) | ling(-0.05) | represent(-0.05) | .security(-0.05)
    ACCEPTED as axis_1388  cumulative_var=0.8887

  [1384]  axes=1389  step_var=0.0015  binary_acc=0.989  gap=0.0827  max_dot=0.0015  (1.9s)
    TOP:  (angle(0.06) | al(0.05) | nÃºmero(0.05) | Invest(0.05) | ogonal(0.05) | Selector(0.05) | finite(0.05) | ly(0.05)
    BOT:  ');Ċ(-0.06) | ä½ĵéĩį(-0.06) | .c(-0.05) | .tom(-0.05) | */Ċ(-0.05) | èº«å¿ĥåģ¥åº·(-0.05) | ':(-0.05) | -license(-0.05)
    ACCEPTED as axis_1389  cumulative_var=0.8889

  [1385]  axes=1390  step_var=0.0015  binary_acc=0.988  gap=0.0824  max_dot=0.0507  (1.9s)
    TOP:  æľµ(0.06) | å®¶(0.05) | ä¹ĭå£°(0.05) | Discrim(0.05) | ressive(0.05) | à¸±à¸ļ(0.05) | summaries(0.05) | ham(0.05)
    BOT:  /b(-0.05) | pad(-0.05) | (__(-0.05) | cli(-0.05) | default(-0.05) | .pro(-0.05) | (units(-0.05) | meta(-0.05)
    ACCEPTED as axis_1390  cumulative_var=0.8891

  [1386]  axes=1391  step_var=0.0016  binary_acc=0.993  gap=0.0828  max_dot=0.0066  (1.8s)
    TOP:  Most(0.06) | å®Īä½ı(0.05) | é«ĺèģĮ(0.05) | Main(0.05) | æĸ¼(0.05) | çķĻåľ¨(0.05) | æĶ¿åºľéĩĩè´Ń(0.05) | equal(0.05)
    BOT:  Master(-0.05) | Forex(-0.05) | ogen(-0.05) | atis(-0.05) | Rate(-0.05) | Lost(-0.05) | ÑĳÐ¼(-0.05) | roach(-0.05)
    ACCEPTED as axis_1391  cumulative_var=0.8892

  [1387]  axes=1392  step_var=0.0015  binary_acc=1.000  gap=0.0821  max_dot=0.0032  (1.8s)
    TOP:  ¸(0.06) | at(0.05) | was(0.05) | {Ċ(0.05) | %}Ċ(0.05) | Foundation(0.05) | Hot(0.05) | are(0.05)
    BOT:  .reset(-0.06) | _exceptions(-0.05) | -dark(-0.05) | -buttons(-0.05) | anga(-0.05) | _gold(-0.05) | =batch(-0.05) | outube(-0.05)
    ACCEPTED as axis_1392  cumulative_var=0.8894

  [1388]  axes=1393  step_var=0.0016  binary_acc=0.994  gap=0.0830  max_dot=0.0068  (1.8s)
    TOP:  Inverse(0.05) | Ðµ(0.05) | åįķä½į(0.05) | æĬĽ(0.05) | .Device(0.05) | ä¸Ĭ(0.05) | .extension(0.05) | functional(0.05)
    BOT:  install(-0.05) | _sqrt(-0.05) | Socket(-0.05) | itre(-0.05) | .Connection(-0.05) | /json(-0.05) | tbody(-0.05) | sold(-0.05)
    ACCEPTED as axis_1393  cumulative_var=0.8896

  [1389]  axes=1394  step_var=0.0016  binary_acc=0.982  gap=0.0841  max_dot=0.0040  (1.8s)
    TOP:  ìĤ°(0.05) | era(0.05) | TS(0.05) | Failed(0.05) | æ¯«åįĩ(0.05) | -api(0.05) | æļĸ(0.05) | åĿļåĽº(0.05)
    BOT:  ,ĊĊ(-0.06) | }},(-0.05) | .Set(-0.05) | ....ĊĊ(-0.05) | ä¸º(-0.05) | å¿«ä¹Ĳ(-0.05) | Initialization(-0.05) | ìĭł(-0.05)
    ACCEPTED as axis_1394  cumulative_var=0.8898

  [1390]  axes=1395  step_var=0.0016  binary_acc=0.996  gap=0.0835  max_dot=0.0430  (1.8s)
    TOP:  -errors(0.05) | /met(0.05) | ä¸įåıĺ(0.05) | _stream(0.05) | åĲį(0.05) | :list(0.05) | steps(0.05) | another(0.05)
    BOT:  çıŃåŃĲæĪĲåĳĺ(-0.06) | ÐĽ(-0.05) | (rgb(-0.05) | *=(-0.05) | æ±īæĹı(-0.05) | âĳ(-0.05) | æĢ»è®¡(-0.05) | ([],(-0.05)
    ACCEPTED as axis_1395  cumulative_var=0.8899

  [1391]  axes=1396  step_var=0.0016  binary_acc=0.971  gap=0.0819  max_dot=0.0234  (1.8s)
    TOP:  .org(0.07) | ALL(0.05) | \Data(0.05) | äº²çĪ±çļĦ(0.05) | ÑĨÐ¸Ñİ(0.05) | County(0.05) | .Dis(0.05) | Lecture(0.05)
    BOT:  ;a(-0.06) | coding(-0.05) | ä¸īå³¡(-0.05) | OURSE(-0.05) | QT(-0.05) | ä½ľåĵģ(-0.05) | ä¼´(-0.05) | Tcp(-0.05)
    ACCEPTED as axis_1396  cumulative_var=0.8901

  [1392]  axes=1397  step_var=0.0016  binary_acc=0.989  gap=0.0827  max_dot=0.0410  (1.8s)
    TOP:  off(0.06) | /ui(0.06) | inya(0.05) | crop(0.05) | undo(0.05) | ya(0.05) | .Integer(0.05) | YGON(0.05)
    BOT:  ';(-0.06) | .c(-0.06) | èĢĮ(-0.05) | |-(-0.05) | --ĊĊ(-0.05) | å£°åĵį(-0.05) | !=(-0.05) | .".(-0.05)
    ACCEPTED as axis_1397  cumulative_var=0.8903

  [1393]  axes=1398  step_var=0.0016  binary_acc=0.966  gap=0.0815  max_dot=0.0154  (1.9s)
    TOP:  çĶ±(0.06) | çĶ¨(0.06) | on(0.05) | vn(0.05) | msg(0.05) | .Task(0.05) | (required(0.05) | ('(0.05)
    BOT:  Combined(-0.05) | theses(-0.05) | -tools(-0.05) | assessed(-0.05) | .M(-0.04) | Citizen(-0.04) | ä¸Ńåįİæ°ĳæĹı(-0.04) | File(-0.04)
    ACCEPTED as axis_1398  cumulative_var=0.8905

  [1394]  axes=1399  step_var=0.0016  binary_acc=0.998  gap=0.0801  max_dot=0.0176  (1.9s)
    TOP:  Y(0.05) | coefficients(0.05) | Callback(0.05) | _AL(0.05) | teaching(0.05) | [node(0.05) | Polar(0.05) | /object(0.05)
    BOT:  åī¯ä¹¦è®°(-0.06) | æĶ¹éĿ©å§Ķ(-0.06) | .client(-0.06) | %,(-0.06) | åĿĿ(-0.06) | ftp(-0.06) | upal(-0.05) | {},(-0.05)
    ACCEPTED as axis_1399  cumulative_var=0.8906

  [1395]  axes=1400  step_var=0.0015  binary_acc=0.982  gap=0.0809  max_dot=0.0288  (1.8s)
    TOP:  ãĢĲ(0.06) | ops(0.05) | ed(0.05) | åĮĸå¦Ĩ(0.05) | modules(0.05) | èº«åĲİ(0.05) | âĢľ(0.05) | æĺ¾ç¤ºåĩº(0.05)
    BOT:  .prop(-0.05) | _type(-0.05) | (co(-0.05) | than(-0.05) | qd(-0.05) | .Process(-0.05) | Â»(-0.05) | importing(-0.05)
    ACCEPTED as axis_1400  cumulative_var=0.8908

  [1396]  axes=1401  step_var=0.0016  binary_acc=0.992  gap=0.0826  max_dot=0.0034  (1.8s)
    TOP:  EST(0.06) | andal(0.06) | íĶĦ(0.06) | Ð¸(0.06) | ff(0.05) | å·ŀå¸Ĥ(0.05) | oster(0.05) | æĸ¯(0.05)
    BOT:  si(-0.06) | .!(-0.05) | åĪĽéĢłæĢ§(-0.05) | ä¸ŃçļĦ(-0.05) | Speed(-0.05) | Any(-0.05) | -slider(-0.05) | .c(-0.04)
    ACCEPTED as axis_1401  cumulative_var=0.8910

  [1397]  axes=1402  step_var=0.0016  binary_acc=0.998  gap=0.0839  max_dot=0.0231  (1.8s)
    TOP:  .concatenate(0.05) | .Security(0.05) | çļĦä»»åĬ¡(0.05) | .AR(0.05) | é«ĺä¸Ń(0.05) | æŁ¥èİ·(0.05) | .resolve(0.05) | è°ĥåīĤ(0.05)
    BOT:  has(-0.06) | _of(-0.06) | å¤©ä¸Ĭ(-0.06) | run(-0.05) | of(-0.05) | èĥĨ(-0.05) | OF(-0.05) | ==(-0.05)
    ACCEPTED as axis_1402  cumulative_var=0.8911

  [1398]  axes=1403  step_var=0.0016  binary_acc=0.976  gap=0.0839  max_dot=0.0508  (1.8s)
    TOP:  .execute(0.05) | Del(0.05) | ä½¿åĳ½æĦŁ(0.05) | å¯¹(0.05) | ------------------------------------------------(0.05) | inputs(0.05) | Introduction(0.05) | Ð¶ÐµÐ½(0.05)
    BOT:  .Book(-0.06) | /R(-0.06) | _match(-0.05) | _eta(-0.05) | _ERR(-0.05) | discard(-0.05) | esh(-0.05) | :id(-0.05)
    ACCEPTED as axis_1403  cumulative_var=0.8913

  [1399]  axes=1404  step_var=0.0015  binary_acc=0.979  gap=0.0811  max_dot=0.0523  (1.8s)
    TOP:  éħ¸(0.06) | åĩı(0.05) | ç¨İ(0.05) | å½¢(0.05) | <(0.05) | æľ¯(0.05) | å¹´é¾Ħ(0.05) | åĩºéĻ¢(0.05)
    BOT:  .logging(-0.06) | _topics(-0.05) | ousand(-0.05) | Ha(-0.05) | In(-0.05) | return(-0.05) | rik(-0.05) | _clusters(-0.05)
    ACCEPTED as axis_1404  cumulative_var=0.8915

  [1400]  axes=1405  step_var=0.0016  binary_acc=0.968  gap=0.0806  max_dot=0.0099  (1.9s)
    TOP:  temp(0.06) | ìļ©(0.06) | Select(0.05) | ç£¨æįŁ(0.05) | (C(0.05) | test(0.05) | kd(0.05) | session(0.05)
    BOT:  uses(-0.05) | .file(-0.05) | these(-0.05) | -DD(-0.05) | ----------Ċ(-0.05) | Ð²(-0.05) | ãĥ»(-0.05) | ]Ċ(-0.05)
    ACCEPTED as axis_1405  cumulative_var=0.8916

  [1401]  axes=1406  step_var=0.0016  binary_acc=0.997  gap=0.0832  max_dot=0.0648  (1.8s)
    TOP:  è¿Ļä¸¤ç§į(0.06) | wording(0.05) | heit(0.05) | it(0.05) | antic(0.05) | Estimate(0.05) | exit(0.05) | _pixel(0.05)
    BOT:  ("(-0.06) | escription(-0.05) | ä¸ŃåĽ½(-0.05) | izing(-0.05) | çłģ(-0.05) | ework(-0.05) | predict(-0.05) | MI(-0.05)
    ACCEPTED as axis_1406  cumulative_var=0.8918

  [1402]  axes=1407  step_var=0.0016  binary_acc=0.991  gap=0.0828  max_dot=0.0326  (1.9s)
    TOP:  mode(0.05) | .thread(0.05) | (BASE(0.05) | look(0.05) | Ń(0.05) | è¯¦(0.05) | changed(0.05) | ä»¥åıĬ(0.05)
    BOT:  Ð´Ð°(-0.05) | LOSS(-0.05) | >,(-0.05) | Dai(-0.05) | .site(-0.05) | >\(-0.05) | åħ³èģĶ(-0.05) | vest(-0.05)
    ACCEPTED as axis_1407  cumulative_var=0.8920

  [1403]  axes=1408  step_var=0.0016  binary_acc=0.970  gap=0.0827  max_dot=0.0160  (1.9s)
    TOP:  çĩĥæ²¹(0.07) | amera(0.05) | ç´«å¤ĸçº¿(0.05) | çº¯(0.05) | da(0.05) | emodel(0.05) | le(0.05) | ä¸ĬçıŃ(0.05)
    BOT:  Task(-0.05) | "))Ċ(-0.05) | [k(-0.05) | -g(-0.05) | .Http(-0.05) | .End(-0.05) | _length(-0.05) | (log(-0.05)
    ACCEPTED as axis_1408  cumulative_var=0.8922

  [1404]  axes=1409  step_var=0.0015  binary_acc=0.994  gap=0.0817  max_dot=0.0106  (1.8s)
    TOP:  fers(0.07) | ul(0.06) | ouses(0.05) | èµ°åĲĳ(0.05) | aging(0.05) | odge(0.05) | iking(0.05) | levation(0.05)
    BOT:  [data(-0.06) | <=(-0.05) | interior(-0.05) | Proc(-0.05) | ï¼īĊ(-0.05) | Tau(-0.05) | itself(-0.05) | (),(-0.05)
    ACCEPTED as axis_1409  cumulative_var=0.8923

  [1405]  axes=1410  step_var=0.0016  binary_acc=0.985  gap=0.0829  max_dot=0.0572  (1.9s)
    TOP:  ä¼ĹæīĢåĳ¨çŁ¥(0.06) | anus(0.06) | iev(0.05) | closed(0.05) | on(0.05) | ERS(0.05) | ives(0.05) | service(0.05)
    BOT:  .script(-0.05) | Admin(-0.05) | );Ċ(-0.05) | ãĢį(-0.05) | read(-0.05) | -logo(-0.05) | past(-0.05) | æľªèĥ½(-0.05)
    ACCEPTED as axis_1410  cumulative_var=0.8925

  [1406]  axes=1411  step_var=0.0016  binary_acc=0.987  gap=0.0839  max_dot=0.0505  (1.8s)
    TOP:  âĢľ(0.07) | ï¼Ī(0.06) | //(0.05) | package(0.05) | âĢĻ(0.05) | ĉ(0.05) | .gif(0.05) | ;ĊĊ(0.05)
    BOT:  -present(-0.06) | My(-0.06) | /reference(-0.05) | /page(-0.05) | (**(-0.05) | .access(-0.05) | days(-0.05) | Params(-0.05)
    ACCEPTED as axis_1411  cumulative_var=0.8927

  [1407]  axes=1412  step_var=0.0016  binary_acc=0.998  gap=0.0812  max_dot=0.0146  (1.9s)
    TOP:  bad(0.05) | -flex(0.05) | nc(0.05) | semble(0.05) | processes(0.05) | .fc(0.05) | rea(0.05) | il(0.05)
    BOT:  ä¸ĢåĪĨ(-0.05) | zilla(-0.05) | .layers(-0.05) | æĬ¥(-0.05) | :description(-0.05) | _namespace(-0.05) | _DEPTH(-0.05) | Entre(-0.05)
    ACCEPTED as axis_1412  cumulative_var=0.8928

  [1408]  axes=1413  step_var=0.0016  binary_acc=0.997  gap=0.0838  max_dot=0.0746  (1.8s)
    TOP:  /software(0.07) | ç»©(0.05) | åŀĭåı·(0.05) | ÐºÐ»(0.05) | æ²ĵ(0.05) | _R(0.05) | _css(0.05) | æīįçŁ¥éģĵ(0.05)
    BOT:  ircuit(-0.06) | ido(-0.05) | ao(-0.05) | éĨĴ(-0.05) | user(-0.05) | ider(-0.05) | /Getty(-0.05) | yan(-0.05)
    ACCEPTED as axis_1413  cumulative_var=0.8930

  [1409]  axes=1414  step_var=0.0016  binary_acc=0.979  gap=0.0806  max_dot=0.0078  (1.9s)
    TOP:  R(0.05) | Code(0.05) | Repeat(0.05) | (e(0.05) | åŃĶåŃĲ(0.05) | lar(0.05) | .compose(0.05) | ulate(0.05)
    BOT:  LENGTH(-0.06) | ("""Ċ(-0.06) | º(-0.06) | about(-0.06) | Ùĥ(-0.06) | },Ċ(-0.05) | }((-0.05) | >čĊ(-0.05)
    ACCEPTED as axis_1414  cumulative_var=0.8932

  [1410]  axes=1415  step_var=0.0016  binary_acc=0.999  gap=0.0812  max_dot=0.0538  (1.8s)
    TOP:  _Custom(0.05) | implify(0.05) | .Normalize(0.05) | Construct(0.05) | (ts(0.05) | (bucket(0.05) | Ð¾ÑĢÐ´Ð¸Ð½(0.05) | .ss(0.05)
    BOT:  ture(-0.06) | .Transform(-0.06) | è°ľ(-0.05) | Word(-0.05) | âĢĻ(-0.05) | (cls(-0.05) | under(-0.05) | _blocks(-0.05)
    ACCEPTED as axis_1415  cumulative_var=0.8933

  [1411]  axes=1416  step_var=0.0016  binary_acc=0.966  gap=0.0822  max_dot=0.0091  (1.9s)
    TOP:  where(0.05) | å®ĹæĹ¨(0.05) | path(0.05) | calculus(0.05) | ud(0.05) | åħīæ»ĳ(0.05) | Further(0.05) | INUE(0.05)
    BOT:  doctor(-0.06) | .files(-0.05) | è®°èĢħ(-0.05) | Names(-0.05) | Commercial(-0.05) | Black(-0.05) | Decoder(-0.05) | gregation(-0.05)
    ACCEPTED as axis_1416  cumulative_var=0.8935

  [1412]  axes=1417  step_var=0.0016  binary_acc=0.956  gap=0.0823  max_dot=0.0028  (1.9s)
    TOP:  âĢ(0.06) | ï¼Ł(0.05) | ...Ċ(0.05) | Ñĸ(0.05) | >>Ċ(0.05) | ?Ċ(0.05) | gue(0.05) | ).Ċ(0.05)
    BOT:  ted(-0.07) | theory(-0.06) | æĺİæľĿ(-0.06) | ä½ıäºĨ(-0.05) | -al(-0.05) | zing(-0.05) | inspect(-0.05) | å¼Ģåĩº(-0.05)
    ACCEPTED as axis_1417  cumulative_var=0.8937

  [1413]  axes=1418  step_var=0.0016  binary_acc=0.971  gap=0.0842  max_dot=0.0528  (1.9s)
    TOP:  å·¥ç¨ĭ(0.07) | _plus(0.05) | theta(0.05) | æ¶Īæ¯Ĵ(0.05) | ä¸Ģç»ı(0.05) | png(0.05) | Urls(0.05) | ç»³(0.05)
    BOT:  {(-0.06) | (-0.06) | ĉw(-0.05) | #(-0.05) | {Ċ(-0.05) | R(-0.05) | {{(-0.05) | âĢ¦(-0.05)
    ACCEPTED as axis_1418  cumulative_var=0.8938

  [1414]  axes=1419  step_var=0.0016  binary_acc=0.994  gap=0.0838  max_dot=0.0820  (1.8s)
    TOP:  .w(0.06) | ,ĊĊ(0.05) | /error(0.05) | [T(0.05) | .Number(0.05) | Th(0.05) | çłĶç©¶çĶŁ(0.05) | Ð³Ðµ(0.05)
    BOT:  local(-0.06) | æ¬§(-0.05) | forth(-0.05) | .sig(-0.05) | [float(-0.05) | .monitor(-0.05) | /maps(-0.05) | City(-0.05)
    ACCEPTED as axis_1419  cumulative_var=0.8940

  [1415]  axes=1420  step_var=0.0016  binary_acc=0.973  gap=0.0821  max_dot=0.0223  (1.9s)
    TOP:  derive(0.05) | "<(0.05) | _job(0.05) | "-(0.05) | Dim(0.05) | izi(0.05) | None(0.05) | .policy(0.05)
    BOT:  æł·(-0.05) | è¯Ħå®¡(-0.05) | âĪŀ(-0.05) | åĨł(-0.05) | æī©åħħ(-0.05) | åħ¨çĲĥ(-0.05) | åľº(-0.05) | åĪĬ(-0.05)
    ACCEPTED as axis_1420  cumulative_var=0.8942

  [1416]  axes=1421  step_var=0.0016  binary_acc=0.995  gap=0.0819  max_dot=0.0714  (1.9s)
    TOP:  Once(0.06) | IR(0.05) | ==Ċ(0.05) | Also(0.05) | APT(0.05) | Tag(0.05) | composer(0.05) | kb(0.05)
    BOT:  .argv(-0.06) | åĪĽå»º(-0.06) | èī²(-0.05) | vomiting(-0.05) | _SYMBOL(-0.05) | <body(-0.05) | _logger(-0.05) | Creating(-0.05)
    ACCEPTED as axis_1421  cumulative_var=0.8944

  [1417]  axes=1422  step_var=0.0016  binary_acc=0.996  gap=0.0788  max_dot=0.0064  (1.9s)
    TOP:  ãĢĮ(0.07) | -runtime(0.05) | *(0.05) | éĢļåĳĬ(0.05) | "~(0.05) | _freq(0.05) | -components(0.05) | Recognition(0.04)
    BOT:  .result(-0.06) | required(-0.06) | ist(-0.06) | az(-0.05) | italic(-0.05) | ä¸Ģèµ·(-0.05) | amily(-0.05) | æĶ¾åĪ°(-0.05)
    ACCEPTED as axis_1422  cumulative_var=0.8945

  [1418]  axes=1423  step_var=0.0016  binary_acc=0.999  gap=0.0809  max_dot=0.0094  (1.9s)
    TOP:  integral(0.05) | æĽ´(0.05) | regarding(0.05) | Extended(0.05) | èµĦæºĲæķ´åĲĪ(0.05) | namespace(0.05) | request(0.05) | years(0.04)
    BOT:  i(-0.06) | .path(-0.05) | roles(-0.05) | Details(-0.05) | èĤ¡(-0.05) | Hak(-0.05) | BA(-0.05) | uide(-0.05)
    ACCEPTED as axis_1423  cumulative_var=0.8947

  [1419]  axes=1424  step_var=0.0016  binary_acc=0.981  gap=0.0816  max_dot=0.0090  (1.9s)
    TOP:  Bob(0.06) | æĿ¡(0.05) | Nich(0.05) | /models(0.05) | ylation(0.05) | NotFound(0.05) | phi(0.05) | Ti(0.05)
    BOT:  )ĊĊĊ(-0.06) | :ĊĊ(-0.06) | ):ĊĊ(-0.06) | (N(-0.06) | ĊĊĊ(-0.06) | )ĊĊ(-0.06) | çĶ±(-0.05) | *(-0.05)
    ACCEPTED as axis_1424  cumulative_var=0.8949

  [1420]  axes=1425  step_var=0.0016  binary_acc=0.987  gap=0.0814  max_dot=0.0260  (1.8s)
    TOP:  ()),(0.06) | (server(0.05) | lections(0.05) | icons(0.05) | stdout(0.05) | )(((0.05) | ãĥ»(0.05) | è¯·æ±Ĥ(0.05)
    BOT:  çĩĥ(-0.05) | éħįä»¶(-0.05) | camera(-0.05) | ained(-0.05) | ÐµÐ½Ð¸Ð¸(-0.05) | iggs(-0.05) | er(-0.05) | æ¬¢è¿İ(-0.05)
    ACCEPTED as axis_1425  cumulative_var=0.8950

  [1421]  axes=1426  step_var=0.0016  binary_acc=0.996  gap=0.0817  max_dot=0.0197  (2.0s)
    TOP:  âĢĵ(0.05) | trá»£(0.05) | Professor(0.05) | Prior(0.05) | .set(0.05) | .roll(0.05) | Gradient(0.04) | Compare(0.04)
    BOT:  izable(-0.06) | HD(-0.06) | ('(-0.06) | 't(-0.06) | _mark(-0.05) | ¬(-0.05) | Ð´Ñĥ(-0.05) | ATES(-0.05)
    ACCEPTED as axis_1426  cumulative_var=0.8952

  [1422]  axes=1427  step_var=0.0016  binary_acc=0.999  gap=0.0800  max_dot=0.0081  (1.8s)
    TOP:  -many(0.05) | è´¢åĬĽ(0.05) | disabled(0.05) | insure(0.05) | å¤ļä¸ªåĽ½å®¶(0.05) | footer(0.05) | wish(0.04) | imento(0.04)
    BOT:  å®ŀåľ¨(-0.06) | iste(-0.06) | èº«(-0.06) | (-0.06) | "]:Ċ(-0.06) | åĪļåĪļ(-0.05) | .figure(-0.05) | æĹłåĬĽ(-0.05)
    ACCEPTED as axis_1427  cumulative_var=0.8954

  [1423]  axes=1428  step_var=0.0016  binary_acc=0.994  gap=0.0824  max_dot=0.0078  (1.8s)
    TOP:  Ð´Ð°(0.05) | >Ċ(0.05) | reduce(0.05) | ce(0.05) | ';Ċ(0.05) | -----Ċ(0.05) | termin(0.05) | polate(0.05)
    BOT:  Application(-0.06) | .b(-0.06) | _EMAIL(-0.06) | Chapter(-0.06) | .order(-0.06) | Watch(-0.06) | .o(-0.05) | ç½ĳ(-0.05)
    ACCEPTED as axis_1428  cumulative_var=0.8955

  [1424]  axes=1429  step_var=0.0016  binary_acc=0.971  gap=0.0811  max_dot=0.0275  (1.8s)
    TOP:  UP(0.06) | /live(0.05) | _util(0.05) | _iterations(0.05) | Disclaimer(0.05) | ch(0.05) | _can(0.04) | Up(0.04)
    BOT:  :The(-0.06) | åĲĦä½į(-0.06) | olders(-0.05) | Ð°ÑĤÑĮ(-0.05) | _ac(-0.05) | ew(-0.05) | ï¼ļ(-0.05) | è§ĦåĪĻ(-0.05)
    ACCEPTED as axis_1429  cumulative_var=0.8957

  [1425]  axes=1430  step_var=0.0016  binary_acc=0.978  gap=0.0830  max_dot=0.0209  (1.9s)
    TOP:  äº¤æĺĵ(0.06) | .Content(0.06) | re(0.06) | calendar(0.06) | /user(0.05) | Documentation(0.05) | Project(0.05) | ä¸ĩäºº(0.05)
    BOT:  .fragment(-0.05) | removeClass(-0.05) | Trades(-0.05) | .train(-0.05) | eliminates(-0.05) | _relative(-0.05) | prising(-0.05) | .Process(-0.05)
    ACCEPTED as axis_1430  cumulative_var=0.8959

  [1426]  axes=1431  step_var=0.0016  binary_acc=0.996  gap=0.0809  max_dot=0.0329  (1.8s)
    TOP:  HL(0.05) | ãģĹãģŁ(0.05) | .clip(0.05) | hints(0.05) | _EXTENSION(0.05) | pton(0.05) | uting(0.05) | indices(0.04)
    BOT:  in(-0.06) | ä½Ľ(-0.05) | ">(-0.05) | ä¸Ĭ(-0.05) | Python(-0.05) | åĪ°(-0.05) | èĵĿèī²(-0.05) | å°Ķ(-0.05)
    ACCEPTED as axis_1431  cumulative_var=0.8960

  [1427]  axes=1432  step_var=0.0016  binary_acc=0.986  gap=0.0834  max_dot=0.0588  (1.9s)
    TOP:  Ð½ÑĭÐµ(0.06) | emann(0.05) | e(0.05) | ioms(0.05) | oly(0.05) | ä¹ĭè·¯(0.05) | With(0.05) | (predict(0.05)
    BOT:  .bg(-0.05) | "]]Ċ(-0.05) | èīºæľ¯å®¶(-0.05) | Grad(-0.05) | åıĪæĺ¯(-0.05) | Dear(-0.04) | Keep(-0.04) | _manage(-0.04)
    ACCEPTED as axis_1432  cumulative_var=0.8962

  [1428]  axes=1433  step_var=0.0017  binary_acc=0.993  gap=0.0846  max_dot=0.0223  (1.9s)
    TOP:  M(0.07) | (y(0.05) | _EPS(0.05) | /cloud(0.05) | âĢĿĊ(0.05) | .COLOR(0.05) | -on(0.05) | g(0.05)
    BOT:  .Resource(-0.06) | Symbols(-0.05) | gebra(-0.05) | /doc(-0.05) | Could(-0.05) | avascript(-0.05) | modern(-0.05) | iam(-0.05)
    ACCEPTED as axis_1433  cumulative_var=0.8964

  [1429]  axes=1434  step_var=0.0016  binary_acc=0.987  gap=0.0810  max_dot=0.0297  (1.8s)
    TOP:  (res(0.05) | Producto(0.05) | dis(0.05) | Picture(0.05) | Sh(0.05) | ''(0.05) | Assoc(0.05) | mon(0.05)
    BOT:  =edge(-0.05) | _sem(-0.05) | /server(-0.05) | Van(-0.05) | Blogger(-0.05) | endorsement(-0.05) | -logo(-0.05) | inson(-0.05)
    ACCEPTED as axis_1434  cumulative_var=0.8965

  [1430]  axes=1435  step_var=0.0016  binary_acc=0.986  gap=0.0805  max_dot=0.0236  (1.8s)
    TOP:  uality(0.06) | vm(0.05) | è¯¯(0.05) | ,label(0.05) | atform(0.05) | (token(0.05) | .ss(0.05) | (record(0.05)
    BOT:  ãĢĤĊĊ(-0.06) | ï¼ļ(-0.05) | To(-0.05) | ')Ċ(-0.05) | ]Ċ(-0.05) | she(-0.05) | åŀ«(-0.05) | >.(-0.05)
    ACCEPTED as axis_1435  cumulative_var=0.8967

  [1431]  axes=1436  step_var=0.0016  binary_acc=0.973  gap=0.0800  max_dot=0.0209  (1.8s)
    TOP:  .generated(0.05) | çĽ¸è¿ŀ(0.05) | çī¢åĽº(0.05) | Disney(0.04) | square(0.04) | Link(0.04) | ,"(0.04) | (Context(0.04)
    BOT:  å°ı(-0.06) | It(-0.05) | Our(-0.05) | ua(-0.05) | Ø®(-0.05) | (position(-0.05) | weathermap(-0.05) | .Item(-0.05)
    ACCEPTED as axis_1436  cumulative_var=0.8969

  [1432]  axes=1437  step_var=0.0016  binary_acc=0.996  gap=0.0822  max_dot=0.0572  (1.8s)
    TOP:  .ReLU(0.05) | .Reflection(0.05) | ]),(0.05) | _TIMESTAMP(0.05) | .power(0.05) | Restore(0.05) | Subscribe(0.05) | à¸´à¸ķ(0.04)
    BOT:  pper(-0.06) | ÐºÐµÑĢ(-0.06) | asan(-0.06) | variable(-0.05) | xf(-0.05) | itions(-0.05) | auth(-0.05) | itter(-0.05)
    ACCEPTED as axis_1437  cumulative_var=0.8970

  [1433]  axes=1438  step_var=0.0016  binary_acc=0.997  gap=0.0815  max_dot=0.0623  (1.9s)
    TOP:  ua(0.06) | ()Ċ(0.06) | ]Ċ(0.06) | Python(0.05) | #čĊ(0.05) | install(0.05) | ```(0.05) | $$(0.05)
    BOT:  çĽĽ(-0.05) | æı¡(-0.05) | è·¯çĶ±åĻ¨(-0.05) | æ¼Ĥäº®(-0.05) | çĨŁ(-0.05) | çļĦè¶ĭåĬ¿(-0.05) | Founder(-0.05) | -rest(-0.05)
    ACCEPTED as axis_1438  cumulative_var=0.8972

  [1434]  axes=1439  step_var=0.0016  binary_acc=0.986  gap=0.0788  max_dot=0.0026  (2.0s)
    TOP:  E(0.05) | _ATTACK(0.05) | G(0.05) | HT(0.04) | MD(0.04) | ç»ĻæĤ¨(0.04) | ct(0.04) | hm(0.04)
    BOT:  ADING(-0.05) | ä»¥ä¸Ĭ(-0.05) | spotify(-0.05) | TLS(-0.05) | rypted(-0.05) | obs(-0.05) | oting(-0.05) | æĢĿå¿µ(-0.05)
    ACCEPTED as axis_1439  cumulative_var=0.8973

  [1435]  axes=1440  step_var=0.0016  binary_acc=0.986  gap=0.0792  max_dot=0.0357  (1.9s)
    TOP:  Ele(0.05) | Ind(0.05) | .Dec(0.05) | _bn(0.04) | _TOP(0.04) | -R(0.04) | (trigger(0.04) | .prot(0.04)
    BOT:  inter(-0.06) | '))Ċ(-0.06) | xy(-0.05) | åĬĽ(-0.05) | by(-0.05) | }",Ċ(-0.05) | çıį(-0.05) | âĢľThe(-0.05)
    ACCEPTED as axis_1440  cumulative_var=0.8975

  [1436]  axes=1441  step_var=0.0016  binary_acc=0.964  gap=0.0783  max_dot=0.0059  (1.8s)
    TOP:  Telephone(0.05) | .update(0.05) | Register(0.05) | Doing(0.05) | ç§ĳæķĻ(0.05) | _tag(0.05) | Unlike(0.04) | fection(0.04)
    BOT:  _SAMPLE(-0.05) | Cl(-0.05) | (L(-0.05) | UUID(-0.05) | çİ°ä»»(-0.05) | éĻªä½ł(-0.05) | hl(-0.05) | ^^(-0.05)
    ACCEPTED as axis_1441  cumulative_var=0.8977
  [1437] max |dot| = 0.1040 > 0.1 — not orthogonal enough. (patience 1/10)

  [1438]  axes=1442  step_var=0.0016  binary_acc=0.995  gap=0.0796  max_dot=0.0075  (1.9s)
    TOP:  ):Ċ(0.06) | };(0.06) | }/(0.05) | (optimizer(0.05) | V(0.05) | ):čĊ(0.05) | .handlers(0.05) | (l(0.05)
    BOT:  dist(-0.05) | STATE(-0.05) | Regardless(-0.05) | éĨī(-0.05) | Statement(-0.05) | texture(-0.05) | Load(-0.05) | ISTIC(-0.05)
    ACCEPTED as axis_1442  cumulative_var=0.8978

  [1439]  axes=1443  step_var=0.0016  binary_acc=0.988  gap=0.0811  max_dot=0.0451  (1.8s)
    TOP:  remote(0.05) | SM(0.05) | èĨĪ(0.05) | also(0.05) | _xlabel(0.05) | -default(0.04) | .check(0.04) | /md(0.04)
    BOT:  .init(-0.05) | _r(-0.05) | Co(-0.05) | Math(-0.05) | _k(-0.05) | failed(-0.05) | Progress(-0.05) | urv(-0.05)
    ACCEPTED as axis_1443  cumulative_var=0.8980

  [1440]  axes=1444  step_var=0.0016  binary_acc=0.974  gap=0.0807  max_dot=0.0544  (1.9s)
    TOP:  ĉpublic(0.05) | metro(0.05) | be(0.05) | ish(0.05) | åĵģ(0.05) | Blog(0.05) | ĉif(0.05) | odings(0.05)
    BOT:  .dst(-0.05) | è¦ģåģļåĪ°(-0.05) | Activation(-0.05) | -fe(-0.05) | Buying(-0.05) | -(-0.05) | _family(-0.05) | .Host(-0.05)
    ACCEPTED as axis_1444  cumulative_var=0.8982

  [1441]  axes=1445  step_var=0.0016  binary_acc=0.992  gap=0.0786  max_dot=0.0512  (1.9s)
    TOP:  Â«(0.06) | amon(0.06) | Problem(0.05) | periment(0.05) | discontinued(0.05) | rawl(0.05) | events(0.05) | Foundation(0.05)
    BOT:  /query(-0.06) | ĊĊ(-0.05) | çĿĢæīĭ(-0.05) | ]ĊĊĊ(-0.05) | contents(-0.05) | ')ĊĊ(-0.05) | ä¸»é¢ĺæķĻèĤ²(-0.05) | è®¤(-0.05)
    ACCEPTED as axis_1445  cumulative_var=0.8983

  [1442]  axes=1446  step_var=0.0016  binary_acc=0.995  gap=0.0805  max_dot=0.0264  (1.9s)
    TOP:  /a(0.06) | (location(0.06) | æĪĳçļĦ(0.05) | ULL(0.05) | ç´Ħ(0.05) | =>(0.05) | _count(0.05) | not(0.05)
    BOT:  icle(-0.06) | /ĊĊ(-0.05) | ...Ċ(-0.05) | Install(-0.05) | âĢĿĊ(-0.05) | In(-0.05) | }`);Ċ(-0.05) | Gl(-0.05)
    ACCEPTED as axis_1446  cumulative_var=0.8985

  [1443]  axes=1447  step_var=0.0016  binary_acc=0.994  gap=0.0808  max_dot=0.0448  (1.8s)
    TOP:  Hyper(0.05) | ACTER(0.05) | Got(0.05) | /in(0.05) | USE(0.05) | .Product(0.05) | extract(0.05) | èĥĥ(0.05)
    BOT:  *.(-0.06) | åĮ»çĸĹæľºæŀĦ(-0.06) | /__(-0.05) | /share(-0.05) | $,(-0.05) | idx(-0.05) | ]$(-0.05) | >Ċ(-0.05)
    ACCEPTED as axis_1447  cumulative_var=0.8986

  [1444]  axes=1448  step_var=0.0016  binary_acc=0.974  gap=0.0779  max_dot=0.0101  (1.8s)
    TOP:  )Ċ(0.07) | at(0.06) | çĶ·åŃĲ(0.05) | "))ĊĊ(0.05) | ")ĊĊ(0.05) | loader(0.05) | gmail(0.05) | åĬ¨åĬĽçĶµæ±ł(0.05)
    BOT:  .properties(-0.06) | game(-0.06) | ÙĪ(-0.06) | anchors(-0.05) | åĬ¨(-0.05) | Ø§Øª(-0.05) | olumes(-0.05) | Ð°Ð»ÑĮÐ½Ð¾Ðµ(-0.05)
    ACCEPTED as axis_1448  cumulative_var=0.8988

  [1445]  axes=1449  step_var=0.0016  binary_acc=0.999  gap=0.0789  max_dot=0.0079  (1.9s)
    TOP:  .Abstract(0.05) | /demo(0.05) | IN(0.05) | _ATTRIBUTES(0.05) | all(0.05) | Len(0.05) | Status(0.05) | reibung(0.05)
    BOT:  èµİåĽŀ(-0.05) | åĨ²çªģ(-0.05) | .copy(-0.05) | .registration(-0.05) | -action(-0.05) | |"(-0.05) | åľºåľ°(-0.05) | _decoder(-0.05)
    ACCEPTED as axis_1449  cumulative_var=0.8990

  [1446]  axes=1450  step_var=0.0016  binary_acc=0.991  gap=0.0789  max_dot=0.0099  (1.9s)
    TOP:  Employee(0.05) | .forms(0.05) | Professional(0.05) | cp(0.05) | Extract(0.05) | _literals(0.05) | turtle(0.05) | termination(0.05)
    BOT:  {čĊ(-0.06) | time(-0.05) | /en(-0.05) | º(-0.05) | _MULT(-0.05) | çļĦä¸Ģç§į(-0.04) | -is(-0.04) | F(-0.04)
    ACCEPTED as axis_1450  cumulative_var=0.8991

  [1447]  axes=1451  step_var=0.0016  binary_acc=0.978  gap=0.0807  max_dot=0.0210  (1.9s)
    TOP:  Hill(0.05) | sp(0.05) | iche(0.05) | ELLOW(0.05) | To(0.05) | case(0.05) | IENT(0.05) | qr(0.05)
    BOT:  lawyers(-0.05) | port(-0.05) | Functional(-0.05) | Recursive(-0.05) | Creative(-0.04) | .resource(-0.04) | .event(-0.04) | /Documents(-0.04)
    ACCEPTED as axis_1451  cumulative_var=0.8993

  [1448]  axes=1452  step_var=0.0016  binary_acc=0.997  gap=0.0776  max_dot=0.0184  (2.0s)
    TOP:  Minimal(0.06) | .logical(0.05) | ired(0.05) | _ref(0.05) | Expected(0.05) | čĊčĊ(0.05) | ç»ĻæĪĳ(0.05) | çŃīåİŁåĽł(0.05)
    BOT:  _model(-0.07) | (kwargs(-0.05) | /w(-0.05) | problem(-0.05) | Refs(-0.05) | _COUNT(-0.05) | Cube(-0.04) | OLL(-0.04)
    ACCEPTED as axis_1452  cumulative_var=0.8994

  [1449]  axes=1453  step_var=0.0016  binary_acc=0.987  gap=0.0822  max_dot=0.0051  (1.8s)
    TOP:  .export(0.06) | ])(0.06) | (to(0.05) | )):(0.05) | ---(0.05) | (0.05) | .Struct(0.05) | smallest(0.04)
    BOT:  å¾ªçİ¯(-0.06) | ç¾İ(-0.05) | v(-0.05) | éĵģè·¯(-0.05) | >y(-0.05) | igation(-0.05) | }{(-0.05) | "-"(-0.05)
    ACCEPTED as axis_1453  cumulative_var=0.8996

  [1450]  axes=1454  step_var=0.0016  binary_acc=0.972  gap=0.0785  max_dot=0.0544  (1.9s)
    TOP:  Date(0.05) | åĩĢåĪ©æ¶¦(0.05) | motor(0.05) | .train(0.05) | GRA(0.04) | Practice(0.04) | (Type(0.04) | Initialization(0.04)
    BOT:  ':(-0.06) | }ĊĊ(-0.05) | ignore(-0.05) | Bonus(-0.05) | **ĊĊ(-0.05) | OR(-0.05) | å£ģ(-0.05) | ford(-0.05)
    ACCEPTED as axis_1454  cumulative_var=0.8998

  [1451]  axes=1455  step_var=0.0016  binary_acc=0.988  gap=0.0801  max_dot=0.0392  (1.8s)
    TOP:  (labels(0.06) | Algebra(0.06) | Funeral(0.06) | Mode(0.05) | irsch(0.05) | èĢĥæł¸(0.05) | æĬ¥(0.05) | vary(0.05)
    BOT:  :Ċ(-0.06) | -of(-0.06) | ==Ċ(-0.06) | ±(-0.05) | -CN(-0.05) | :ĊĊ(-0.05) | Ċ(-0.05) | .'ĊĊ(-0.05)
    ACCEPTED as axis_1455  cumulative_var=0.8999

  [1452]  axes=1456  step_var=0.0016  binary_acc=0.994  gap=0.0783  max_dot=0.0108  (1.9s)
    TOP:  ons(0.06) | enda(0.06) | scan(0.06) | ÐµÐ»Ð°(0.06) | æĢ§(0.05) | aire(0.05) | vh(0.05) | Toolbar(0.05)
    BOT:  "(-0.07) | âĢľ(-0.07) | ?Ċ(-0.06) | .(-0.06) | ?(-0.06) | ç¼ºå°ĳ(-0.06) | !ĊĊ(-0.06) | !(-0.06)
    ACCEPTED as axis_1456  cumulative_var=0.9001

  [1453]  axes=1457  step_var=0.0016  binary_acc=0.997  gap=0.0783  max_dot=0.0260  (1.9s)
    TOP:  ],Ċ(0.06) | ished(0.05) | Edition(0.05) | -->(0.05) | Option(0.05) | =((0.05) | -known(0.05) | ensity(0.05)
    BOT:  D(-0.06) | of(-0.05) | _unicode(-0.05) | d(-0.05) | prefer(-0.05) | du(-0.04) | å°ĳè§ģ(-0.04) | ä¹ĭ(-0.04)
    ACCEPTED as axis_1457  cumulative_var=0.9002

  [1454]  axes=1458  step_var=0.0016  binary_acc=0.998  gap=0.0787  max_dot=0.0159  (1.8s)
    TOP:  /site(0.05) | .char(0.05) | Fuse(0.05) | èĥĨ(0.05) | .domain(0.05) | layer(0.05) | Village(0.05) | æĶ¯ä»ĺå®Ŀ(0.04)
    BOT:  +(-0.05) | åħĪ(-0.05) | Import(-0.05) | Field(-0.05) | _TIMER(-0.05) | .count(-0.05) | ï¼Į(-0.05) | ECT(-0.05)
    ACCEPTED as axis_1458  cumulative_var=0.9004

  [1455]  axes=1459  step_var=0.0016  binary_acc=0.950  gap=0.0807  max_dot=0.0015  (1.9s)
    TOP:  /book(0.05) | å¾Ħ(0.05) | STATE(0.05) | _gt(0.05) | into(0.05) | urrence(0.05) | LETTER(0.05) | bits(0.05)
    BOT:  [T(-0.05) | ãĤĵãģ§ãģĻãģĳãģ©(-0.05) | Ð°ÑĤÐµÐ»ÑĮ(-0.05) | grew(-0.05) | ());ĊĊ(-0.05) | =c(-0.04) | ">čĊ(-0.04) | ekt(-0.04)
    ACCEPTED as axis_1459  cumulative_var=0.9006

  [1456]  axes=1460  step_var=0.0016  binary_acc=0.997  gap=0.0806  max_dot=0.0614  (1.9s)
    TOP:  `(0.05) | </(0.05) | >&(0.05) | }(0.05) | :',(0.05) | ()(0.05) | mining(0.05) | ),$(0.05)
    BOT:  mann(-0.06) | ec(-0.05) | ac(-0.05) | am(-0.05) | Print(-0.05) | aches(-0.05) | c(-0.05) | abolic(-0.05)
    ACCEPTED as axis_1460  cumulative_var=0.9007

  [1457]  axes=1461  step_var=0.0016  binary_acc=0.996  gap=0.0801  max_dot=0.0240  (1.9s)
    TOP:  protocol(0.05) | å®īåįĵ(0.05) | conomy(0.05) | å¼ĢéĺĶ(0.05) | BG(0.05) | imit(0.05) | =sum(0.05) | ÐŀÐ±(0.04)
    BOT:  _FILES(-0.06) | ago(-0.06) | ÙĦØ§(-0.05) | ÃŃa(-0.05) | ä¸Ńå¿ĥ(-0.05) | ]],(-0.05) | header(-0.05) | Project(-0.05)
    ACCEPTED as axis_1461  cumulative_var=0.9009

  [1458]  axes=1462  step_var=0.0016  binary_acc=0.989  gap=0.0789  max_dot=0.0241  (1.8s)
    TOP:  Python(0.06) | (load(0.05) | }>Ċ(0.05) | è£¨(0.05) | yclic(0.05) | èĥ¸(0.05) | Ò¯(0.05) | é³ĸ(0.04)
    BOT:  github(-0.06) | åłµå¡ŀ(-0.05) | å¸«(-0.05) | _dd(-0.05) | friends(-0.05) | LA(-0.05) | _preferences(-0.05) | ams(-0.05)
    ACCEPTED as axis_1462  cumulative_var=0.9010

  [1459]  axes=1463  step_var=0.0016  binary_acc=1.000  gap=0.0785  max_dot=0.0709  (1.9s)
    TOP:  .Named(0.06) | !ĊĊĊ(0.06) | ::(0.05) | ":ĊĊ(0.05) | _exc(0.05) | ;$(0.05) | ;Ċ(0.05) | .append(0.05)
    BOT:  ëł¥(-0.06) | ä¸Ģå¤Ħ(-0.06) | çĮª(-0.06) | ences(-0.05) | åĳ³(-0.05) | lege(-0.05) | Ã©(-0.05) | ulkan(-0.05)
    ACCEPTED as axis_1463  cumulative_var=0.9012

  [1460]  axes=1464  step_var=0.0016  binary_acc=0.998  gap=0.0799  max_dot=0.0353  (1.9s)
    TOP:  .segment(0.05) | åĬłå¿«(0.05) | _ERR(0.05) | Time(0.05) | Leaf(0.05) | \\(0.05) | -circle(0.05) | '));Ċ(0.05)
    BOT:  æķĻèĤ²(-0.05) | provides(-0.05) | æĻ®éĢļè¯Ŀ(-0.05) | by(-0.05) | roles(-0.05) | çĶ±(-0.05) | W(-0.05) | argued(-0.04)
    ACCEPTED as axis_1464  cumulative_var=0.9013

  [1461]  axes=1465  step_var=0.0016  binary_acc=0.974  gap=0.0802  max_dot=0.0278  (1.8s)
    TOP:  __.(0.06) | iversal(0.05) | xa(0.05) | gt(0.05) | inator(0.05) | .nav(0.05) | ><(0.05) | /D(0.05)
    BOT:  .path(-0.05) | /hooks(-0.05) | .package(-0.05) | Apply(-0.05) | ç¨ĭåºı(-0.05) | Supplementary(-0.04) | .sep(-0.04) | .object(-0.04)
    ACCEPTED as axis_1465  cumulative_var=0.9015

  [1462]  axes=1466  step_var=0.0016  binary_acc=0.978  gap=0.0811  max_dot=0.0289  (1.8s)
    TOP:  -linux(0.05) | =out(0.05) | -template(0.04) | checkboxes(0.04) | medicine(0.04) | Title(0.04) | .forms(0.04) | ).Ċ(0.04)
    BOT:  Ð¾Ð½(-0.06) | booking(-0.06) | OLE(-0.06) | å¹´(-0.06) | Ap(-0.06) | rypto(-0.06) | aac(-0.06) | AQ(-0.06)
    ACCEPTED as axis_1466  cumulative_var=0.9017

  [1463]  axes=1467  step_var=0.0016  binary_acc=0.976  gap=0.0780  max_dot=0.0029  (1.9s)
    TOP:  _data(0.06) | Profile(0.05) | graph(0.05) | who(0.05) | åľ¨ç½ĳç»ľ(0.05) | P(0.05) | be(0.05) | def(0.05)
    BOT:  pk(-0.05) | assuming(-0.05) | max(-0.05) | ç¬º(-0.05) | .'ĊĊ(-0.05) | remote(-0.05) | latent(-0.04) | tracks(-0.04)
    ACCEPTED as axis_1467  cumulative_var=0.9018

  [1464]  axes=1468  step_var=0.0016  binary_acc=0.981  gap=0.0806  max_dot=0.0585  (1.9s)
    TOP:  ä¸Ĭ(0.05) | ",(0.05) | .error(0.05) | ive(0.05) | .tabs(0.05) | standard(0.05) | ";Ċ(0.05) | .scroll(0.05)
    BOT:  _title(-0.06) | by(-0.05) | _D(-0.05) | Medal(-0.05) | Sweden(-0.05) | .DOM(-0.05) | awesome(-0.05) | åħ±äº§(-0.05)
    ACCEPTED as axis_1468  cumulative_var=0.9020

  [1465]  axes=1469  step_var=0.0016  binary_acc=0.980  gap=0.0779  max_dot=0.0060  (1.9s)
    TOP:  /ns(0.05) | ç«Ń(0.05) | ated(0.05) | ORY(0.05) | .agent(0.05) | Two(0.05) | ÐµÐ½Ð¸Ñı(0.05) | .login(0.05)
    BOT:  âĢ¢(-0.06) | Find(-0.05) | -b(-0.05) | For(-0.05) | ãĥ»(-0.05) | Wireless(-0.05) | æĺ¯ä¸Ģå®¶(-0.05) | school(-0.05)
    ACCEPTED as axis_1469  cumulative_var=0.9021

  [1466]  axes=1470  step_var=0.0016  binary_acc=0.977  gap=0.0803  max_dot=0.0065  (1.8s)
    TOP:  posts(0.05) | =edge(0.05) | strategy(0.05) | /messages(0.04) | (detail(0.04) | Critical(0.04) | Adj(0.04) | Graduate(0.04)
    BOT:  c(-0.06) | ()Ċ(-0.06) | This(-0.05) | copy(-0.05) | 'ĊĊ(-0.05) | /sec(-0.05) | _attribute(-0.05) | );čĊčĊ(-0.05)
    ACCEPTED as axis_1470  cumulative_var=0.9023

  [1467]  axes=1471  step_var=0.0016  binary_acc=0.982  gap=0.0796  max_dot=0.0531  (1.9s)
    TOP:  )čĊčĊ(0.06) | ,(0.06) | (original(0.05) | æľīæĦıä¹ī(0.05) | Us(0.05) | å¤§åĬĽæĶ¯æĮģ(0.05) | å¤§åŃ¦(0.05) | });Ċ(0.05)
    BOT:  (reg(-0.05) | Javascript(-0.05) | Water(-0.04) | Median(-0.04) | _initializer(-0.04) | Other(-0.04) | .module(-0.04) | vol(-0.04)
    ACCEPTED as axis_1471  cumulative_var=0.9024

  [1468]  axes=1472  step_var=0.0016  binary_acc=0.999  gap=0.0796  max_dot=0.0349  (1.8s)
    TOP:  Henry(0.05) | Dynamics(0.05) | can(0.05) | -type(0.05) | _eq(0.05) | äºº(0.05) | Adjusted(0.05) | ¡(0.04)
    BOT:  gress(-0.05) | æĮĩå°ĸ(-0.05) | åħ³æĢĢ(-0.05) | instance(-0.05) | basket(-0.05) | LV(-0.05) | (api(-0.05) | cial(-0.05)
    ACCEPTED as axis_1472  cumulative_var=0.9026

  [1469]  axes=1473  step_var=0.0016  binary_acc=0.993  gap=0.0779  max_dot=0.0157  (1.8s)
    TOP:  Launch(0.05) | ertificate(0.05) | Content(0.05) | æĬ¤(0.05) | åĽ½æľī(0.05) | è´¥(0.05) | Toolbar(0.05) | they(0.05)
    BOT:  Hot(-0.05) | Reason(-0.05) | åĪĴå®ļ(-0.05) | åĢĴæĺ¯(-0.05) | _margin(-0.05) | Typed(-0.05) | "H(-0.05) | ÐĿ(-0.04)
    ACCEPTED as axis_1473  cumulative_var=0.9028

  [1470]  axes=1474  step_var=0.0016  binary_acc=0.978  gap=0.0772  max_dot=0.0323  (1.9s)
    TOP:  Corporate(0.06) | Square(0.06) | "(0.06) | Take(0.05) | Of(0.05) | _kernel(0.05) | ''(0.05) | `ĊĊ(0.05)
    BOT:  Ã©(-0.06) | simplify(-0.05) | c(-0.04) | relations(-0.04) | .value(-0.04) | ÑĥÑģÐ»ÑĥÐ³(-0.04) | _commands(-0.04) | table(-0.04)
    ACCEPTED as axis_1474  cumulative_var=0.9029

  [1471]  axes=1475  step_var=0.0015  binary_acc=0.991  gap=0.0763  max_dot=0.0086  (1.9s)
    TOP:  '))ĊĊ(0.05) | ?ĊĊ(0.05) | }](0.05) | /pre(0.05) | (size(0.05) | ):Ċ(0.05) | edium(0.05) | Estimated(0.05)
    BOT:  one(-0.08) | Generic(-0.06) | val(-0.06) | ran(-0.05) | ACCOUNT(-0.05) | iar(-0.05) | ld(-0.05) | vc(-0.05)
    ACCEPTED as axis_1475  cumulative_var=0.9031

  [1472]  axes=1476  step_var=0.0015  binary_acc=0.982  gap=0.0784  max_dot=0.0238  (1.9s)
    TOP:  to(0.05) | attach(0.05) | INFO(0.05) | though(0.05) | ^n(0.05) | Toolbar(0.05) | .generate(0.05) | _d(0.05)
    BOT:  ismic(-0.06) | ("/")Ċ(-0.05) | __Ċ(-0.05) | (input(-0.05) | --Ċ(-0.04) | Ċ(-0.04) | Type(-0.04) | supplying(-0.04)
    ACCEPTED as axis_1476  cumulative_var=0.9032

  [1473]  axes=1477  step_var=0.0016  binary_acc=0.995  gap=0.0779  max_dot=0.0279  (1.9s)
    TOP:  èĭ±(0.06) | room(0.05) | å¹³(0.05) | .Method(0.05) | put(0.05) | åŁºæľ¬(0.05) | AMA(0.05) | JI(0.05)
    BOT:  HttpResponse(-0.05) | _pending(-0.05) | _tables(-0.05) | .mul(-0.05) | æķĻæĿĲ(-0.04) | female(-0.04) | (options(-0.04) | _(-0.04)
    ACCEPTED as axis_1477  cumulative_var=0.9034

  [1474]  axes=1478  step_var=0.0016  binary_acc=0.957  gap=0.0778  max_dot=0.0263  (1.9s)
    TOP:  äº¤æĺĵä¸Ńå¿ĥ(0.04) | Intelligent(0.04) | _pages(0.04) | è®¯(0.04) | _part(0.04) | Module(0.04) | Multip(0.04) | _quality(0.04)
    BOT:  ï¼Ľ(-0.08) | ;(-0.07) | der(-0.06) | ï¼İ(-0.06) | ï¼ī(-0.06) | g(-0.06) | ur(-0.06) | uti(-0.06)
    ACCEPTED as axis_1478  cumulative_var=0.9035

  [1475]  axes=1479  step_var=0.0015  binary_acc=0.997  gap=0.0783  max_dot=0.0173  (1.9s)
    TOP:  éªĽ(0.05) | .Collections(0.05) | Combine(0.05) | creed(0.04) | Qi(0.04) | Recovered(0.04) | pw(0.04) | .generator(0.04)
    BOT:  åıĳå±ķ(-0.05) | æ³ķ(-0.05) | !Ċ(-0.05) | _df(-0.04) | embedding(-0.04) | Flatten(-0.04) | /category(-0.04) | _vector(-0.04)
    ACCEPTED as axis_1479  cumulative_var=0.9037

  [1476]  axes=1480  step_var=0.0016  binary_acc=0.972  gap=0.0789  max_dot=0.0398  (1.8s)
    TOP:  ado(0.07) | ÑĭÐ¹(0.06) | n(0.05) | Under(0.05) | .Response(0.05) | ishing(0.05) | ATION(0.05) | To(0.05)
    BOT:  .endswith(-0.06) | æŁłæª¬(-0.05) | ç½ĳæ°ĳ(-0.05) | .fac(-0.05) | Absolute(-0.05) | ROUND(-0.05) | Into(-0.05) | æł¹æľ¬(-0.05)
    ACCEPTED as axis_1480  cumulative_var=0.9038

  [1477]  axes=1481  step_var=0.0016  binary_acc=0.975  gap=0.0784  max_dot=0.0312  (1.8s)
    TOP:  éĶĢæ¯ģ(0.05) | between(0.05) | _MODULE(0.05) | .player(0.05) | agination(0.05) | _remote(0.05) | æĿĥ(0.05) | èĬ¬åħ°(0.05)
    BOT:  :Ċ(-0.06) | æ¯ķä¸ļçĶŁ(-0.06) | :čĊ(-0.05) | èĩ³æŃ¤(-0.05) | :(-0.05) | ä¸ĸä»£(-0.05) | ."""Ċ(-0.05) | å¾Īå°ı(-0.05)
    ACCEPTED as axis_1481  cumulative_var=0.9040

  [1478]  axes=1482  step_var=0.0016  binary_acc=0.991  gap=0.0768  max_dot=0.0177  (1.8s)
    TOP:  Col(0.05) | Encoder(0.05) | schools(0.05) | .directory(0.05) | During(0.04) | with(0.04) | OURNAL(0.04) | deviceId(0.04)
    BOT:  _theme(-0.06) | _module(-0.05) | .em(-0.05) | k(-0.05) | _user(-0.05) | çĥŃ(-0.05) | åĳķåĲĲ(-0.05) | ]$(-0.05)
    ACCEPTED as axis_1482  cumulative_var=0.9041

  [1479]  axes=1483  step_var=0.0015  binary_acc=0.983  gap=0.0783  max_dot=0.0221  (1.9s)
    TOP:  è(0.05) | _req(0.05) | on(0.05) | å®ŀè·µæ´»åĬ¨(0.05) | &(0.05) | -known(0.05) | ven(0.05) | ms(0.05)
    BOT:  elist(-0.05) | #define(-0.05) | (vars(-0.05) | .Port(-0.05) | Uncomment(-0.05) | .background(-0.05) | #import(-0.04) | .pause(-0.04)
    ACCEPTED as axis_1483  cumulative_var=0.9043

  [1480]  axes=1484  step_var=0.0016  binary_acc=0.997  gap=0.0776  max_dot=0.0218  (1.9s)
    TOP:  latest(0.05) | .Http(0.05) | scal(0.05) | mj(0.05) | ä¿Ĭ(0.04) | åĩłä¸ª(0.04) | .one(0.04) | é»Ħéĩĳ(0.04)
    BOT:  kit(-0.05) | _pl(-0.05) | ###(-0.05) | EN(-0.05) | {(-0.05) | fort(-0.05) | If(-0.05) | A(-0.05)
    ACCEPTED as axis_1484  cumulative_var=0.9044

  [1481]  axes=1485  step_var=0.0016  binary_acc=0.987  gap=0.0779  max_dot=0.0224  (2.0s)
    TOP:  ov(0.05) | teil(0.05) | _beam(0.05) | -blue(0.05) | ä¼ļè®©(0.05) | Viv(0.05) | athi(0.05) | conda(0.05)
    BOT:  consideration(-0.06) | ));Ċ(-0.06) | )Ċ(-0.05) | Ð¡(-0.05) | .org(-0.05) | OF(-0.05) | double(-0.05) | YNAMIC(-0.05)
    ACCEPTED as axis_1485  cumulative_var=0.9046

  [1482]  axes=1486  step_var=0.0016  binary_acc=0.952  gap=0.0778  max_dot=0.0170  (1.8s)
    TOP:  enerate(0.05) | ONG(0.05) | Enumeration(0.04) | .et(0.04) | OMATIC(0.04) | iating(0.04) | Department(0.04) | "}Ċ(0.04)
    BOT:  Tab(-0.05) | é£İä¿Ĺ(-0.05) | (-0.05) | åı¸(-0.05) | '',Ċ(-0.05) | On(-0.05) | XP(-0.05) | press(-0.05)
    ACCEPTED as axis_1486  cumulative_var=0.9047

  [1483]  axes=1487  step_var=0.0016  binary_acc=0.955  gap=0.0780  max_dot=0.0106  (1.9s)
    TOP:  ä»Ģä¹Ī(0.05) | ãĤĵ(0.05) | (input(0.05) | æĺ¯ä¸Ģä¸ª(0.05) | _int(0.05) | _requests(0.05) | bits(0.05) | è°±åĨĻ(0.05)
    BOT:  '.(-0.05) | {}](-0.05) | yle(-0.05) | ÐµÑģÑĤ(-0.05) | ibus(-0.05) | WARD(-0.05) | du(-0.05) | sequently(-0.05)
    ACCEPTED as axis_1487  cumulative_var=0.9049

  [1484]  axes=1488  step_var=0.0016  binary_acc=0.994  gap=0.0809  max_dot=0.0821  (1.9s)
    TOP:  âĢĻm(0.05) | imagine(0.05) | _splits(0.05) | ä¾¿ç§ĺ(0.05) | MAV(0.05) | '../(0.05) | -controlled(0.04) | _delete(0.04)
    BOT:  color(-0.05) | åĬ¨(-0.05) | May(-0.05) | v(-0.05) | Hook(-0.05) | ,h(-0.05) | ÐĿÐ°(-0.05) | åħ¨çĲĥ(-0.05)
    ACCEPTED as axis_1488  cumulative_var=0.9050

  [1485]  axes=1489  step_var=0.0016  binary_acc=0.997  gap=0.0780  max_dot=0.0333  (1.9s)
    TOP:  æľº(0.06) | .Rule(0.06) | ä¸ĭ(0.05) | world(0.05) | .channel(0.05) | xb(0.05) | Garage(0.05) | into(0.05)
    BOT:  actic(-0.05) | binary(-0.05) | Games(-0.05) | (server(-0.05) | company(-0.04) | Inter(-0.04) | Future(-0.04) | .getClass(-0.04)
    ACCEPTED as axis_1489  cumulative_var=0.9052

  [1486]  axes=1490  step_var=0.0016  binary_acc=0.985  gap=0.0784  max_dot=0.0285  (1.8s)
    TOP:  =='(0.06) | ""Ċ(0.05) | }ĊĊ(0.05) | '"(0.05) | }čĊ(0.05) | !(0.05) | }Ċ(0.05) | Ð½Ð¾ÑģÑĤÑĮÑİ(0.05)
    BOT:  ara(-0.06) | aks(-0.06) | ena(-0.06) | UN(-0.06) | Ã©e(-0.06) | ja(-0.06) | ism(-0.05) | On(-0.05)
    ACCEPTED as axis_1490  cumulative_var=0.9053

  [1487]  axes=1491  step_var=0.0016  binary_acc=0.956  gap=0.0779  max_dot=0.0310  (2.0s)
    TOP:  Code(0.05) | MainWindow(0.05) | This(0.05) | There(0.05) | It(0.05) | Col(0.05) | .subplots(0.04) | Invalid(0.04)
    BOT:  ç³ĸå°¿çĹħ(-0.06) | xd(-0.05) | é¢¤(-0.05) | ixed(-0.05) | Forums(-0.05) | "./(-0.05) | çĭ¬æľīçļĦ(-0.05) | /py(-0.05)
    ACCEPTED as axis_1491  cumulative_var=0.9055

  [1488]  axes=1492  step_var=0.0016  binary_acc=0.985  gap=0.0773  max_dot=0.0051  (1.9s)
    TOP:  åĲİ(0.06) | p(0.06) | Øµ(0.05) | ï¼Ī(0.05) | /docs(0.05) | ĳ(0.05) | world(0.05) | å·®(0.05)
    BOT:  itations(-0.05) | NING(-0.05) | _me(-0.05) | (Component(-0.05) | igion(-0.05) | ç¼ĵåĨ²(-0.04) | /man(-0.04) | Creator(-0.04)
    ACCEPTED as axis_1492  cumulative_var=0.9056

  [1489]  axes=1493  step_var=0.0016  binary_acc=0.994  gap=0.0755  max_dot=0.0352  (1.9s)
    TOP:  from(0.05) | After(0.05) | its(0.05) | from(0.05) | phere(0.05) | ){Ċ(0.05) | }ĊĊĊ(0.05) | _form(0.05)
    BOT:  .Frame(-0.06) | Send(-0.05) | (rgb(-0.05) | .bp(-0.05) | iqu(-0.05) | Knowing(-0.05) | å¦Īå¦Ī(-0.05) | full(-0.05)
    ACCEPTED as axis_1493  cumulative_var=0.9058

  [1490]  axes=1494  step_var=0.0016  binary_acc=0.980  gap=0.0771  max_dot=0.0051  (1.9s)
    TOP:  time(0.05) | (The(0.05) | ,request(0.04) | carries(0.04) | head(0.04) | -success(0.04) | MAT(0.04) | us(0.04)
    BOT:  ">(-0.06) | RV(-0.05) | /**Ċ(-0.05) | tery(-0.05) | kin(-0.05) | ÂłĊ(-0.05) | en(-0.05) | Ī(-0.05)
    ACCEPTED as axis_1494  cumulative_var=0.9059

  [1491]  axes=1495  step_var=0.0016  binary_acc=0.998  gap=0.0772  max_dot=0.0025  (1.8s)
    TOP:  .extensions(0.05) | _size(0.05) | .decode(0.05) | èĭ±å¯¸(0.05) | vertex(0.05) | .b(0.04) | .browser(0.04) | ĊĊ(0.04)
    BOT:  _number(-0.05) | ÐºÐ°ÑĩÐµÑģÑĤÐ²Ð¾(-0.05) | ipt(-0.05) | Refer(-0.05) | Python(-0.05) | an(-0.05) | åĽ¾(-0.05) | rad(-0.05)
    ACCEPTED as axis_1495  cumulative_var=0.9061

  [1492]  axes=1496  step_var=0.0016  binary_acc=0.974  gap=0.0754  max_dot=0.0088  (1.9s)
    TOP:  faces(0.06) | inge(0.06) | OUT(0.06) | eds(0.06) | etz(0.06) | ash(0.05) | Ð°Ð¼Ð¸(0.05) | rm(0.05)
    BOT:  _ĊĊ(-0.07) | **ĊĊ(-0.06) | +ĊĊ(-0.06) | although(-0.05) | `čĊ(-0.05) | ).ĊĊ(-0.05) | (raw(-0.05) | ].ĊĊ(-0.05)
    ACCEPTED as axis_1496  cumulative_var=0.9062

  [1493]  axes=1497  step_var=0.0016  binary_acc=0.979  gap=0.0760  max_dot=0.0344  (1.8s)
    TOP:  by(0.06) | iner(0.06) | req(0.05) | time(0.05) | idy(0.05) | Title(0.05) | esting(0.05) | ents(0.05)
    BOT:  æģª(-0.04) | âĢĿ.(-0.04) | Extract(-0.04) | .Ex(-0.04) | Happ(-0.04) | token(-0.04) | distinct(-0.04) | ä¸įç»ıæĦı(-0.04)
    ACCEPTED as axis_1497  cumulative_var=0.9064

  [1494]  axes=1498  step_var=0.0016  binary_acc=0.999  gap=0.0754  max_dot=0.0185  (1.9s)
    TOP:  äº§(0.05) | Your(0.05) | functools(0.05) | ä¸İæŃ¤åĲĮæĹ¶(0.04) | Deleting(0.04) | å¤©çĦ¶(0.04) | Common(0.04) | ä¸įèī¯(0.04)
    BOT:  .contrib(-0.05) | /Ċ(-0.05) | spaces(-0.05) | olvers(-0.05) | "";Ċ(-0.05) | Names(-0.05) | requencies(-0.05) | Greatest(-0.05)
    ACCEPTED as axis_1498  cumulative_var=0.9065

  [1495]  axes=1499  step_var=0.0016  binary_acc=0.987  gap=0.0783  max_dot=0.0219  (1.8s)
    TOP:  """Ċ(0.05) | /?(0.04) | GENERAL(0.04) | MF(0.04) | Fan(0.04) | è´¡çĮ®(0.04) | eco(0.04) | ä¸įåĲĪæł¼(0.04)
    BOT:  Ð°(-0.06) | ar(-0.05) | ï¼Ī(-0.05) | e(-0.05) | d(-0.05) | Ð¾ÐºÐ°(-0.05) | _filter(-0.05) | ä¸Ģå®ļä¼ļ(-0.05)
    ACCEPTED as axis_1499  cumulative_var=0.9067

> **FINDING:** Stopped: reached MAX_AXES=1500.


## Summary

  Total axes in basis:     1500
  Seed axes:               6
  Discovered axes:         1494

> **FINDING:** Discovered 1494 new binary truth axes beyond the 6 seeds.


Output: /home/thorin/truthspace-lcm/experiments/truthspace_v1/dc299_phase1_axes.json
