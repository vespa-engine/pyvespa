# Querying Vespa[¶](#querying-vespa)

This guide goes through how to query a Vespa instance using the Query API and <https://search.vespa.ai/> app as an example.

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

You can run this tutorial in Google Colab:

In \[ \]:

Copied!

```
!pip3 install pyvespa
```

!pip3 install pyvespa

Let us first just deploy and get a connection to a Vespa instance.

In \[2\]:

Copied!

```
from vespa.application import Vespa
from vespa.deployment import VespaDocker
from vespa.io import VespaQueryResponse
from vespa.exceptions import VespaError
from vespa.package import sample_package

vespa_docker = VespaDocker()
app: Vespa = vespa_docker.deploy(sample_package)
```

from vespa.application import Vespa from vespa.deployment import VespaDocker from vespa.io import VespaQueryResponse from vespa.exceptions import VespaError from vespa.package import sample_package vespa_docker = VespaDocker() app: Vespa = vespa_docker.deploy(sample_package)

```
Waiting for configuration server, 0/60 seconds...
Waiting for configuration server, 5/60 seconds...
Waiting for application to come up, 0/300 seconds.
Waiting for application to come up, 5/300 seconds.
Application is up!
Finished deployment.
```

In \[3\]:

Copied!

```
from datasets import load_dataset

dataset = load_dataset("BeIR/nfcorpus", "corpus", split="corpus", streaming=True)
vespa_feed = dataset.map(
    lambda x: {
        "id": x["_id"],
        "fields": {"title": x["title"], "body": x["text"], "id": x["_id"]},
    }
).take(100)
```

from datasets import load_dataset dataset = load_dataset("BeIR/nfcorpus", "corpus", split="corpus", streaming=True) vespa_feed = dataset.map( lambda x: { "id": x["\_id"], "fields": {"title": x["title"], "body": x["text"], "id": x["\_id"]}, } ).take(100)

In \[4\]:

Copied!

```
from vespa.io import VespaResponse


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(f"Error when feeding document {id}: {response.get_json()}")


app.feed_iterable(vespa_feed, schema="doc", namespace="tutorial", callback=callback)
```

from vespa.io import VespaResponse def callback(response: VespaResponse, id: str): if not response.is_successful(): print(f"Error when feeding document {id}: {response.get_json()}") app.feed_iterable(vespa_feed, schema="doc", namespace="tutorial", callback=callback)

See the [Vespa query language](https://docs.vespa.ai/en/reference/query-api-reference.html) for Vespa query api request parameters.

The YQL [userQuery()](https://docs.vespa.ai/en/reference/query-language-reference.html#userquery) operator uses the query read from `query`. The query also specifies to use the app-specific [documentation rank profile](https://docs.vespa.ai/en/reference/bm25.html). The code uses [context manager](https://realpython.com/python-with-statement/) `with session` statement to make sure that connection pools are released. If you attempt to make multiple queries, this is important as each query will not have to set up new connections.

In \[5\]:

Copied!

```
with app.syncio() as session:
    response: VespaQueryResponse = session.query(
        yql="select title, body from doc where userQuery()",
        hits=1,
        query="Is statin use connected to breast cancer?",
        ranking="bm25",
    )
    print(response.is_successful())
    print(response.url)
```

with app.syncio() as session: response: VespaQueryResponse = session.query( yql="select title, body from doc where userQuery()", hits=1, query="Is statin use connected to breast cancer?", ranking="bm25", ) print(response.is_successful()) print(response.url)

```
True
http://localhost:8080/search/?yql=select+title%2C+body+from+doc+where+userQuery%28%29&hits=1&query=Is+statin+use+connected+to+breast+cancer%3F&ranking=bm25
```

Alternatively, if the native [Vespa query parameter](https://docs.vespa.ai/en/reference/query-api-reference.html) contains ".", which cannot be used as a `kwarg`, the parameters can be sent as HTTP POST with the `body` argument. In this case, `ranking` is an alias of `ranking.profile`, but using `ranking.profile` as a `**kwargs` argument is not allowed in python. This will combine HTTP parameters with an HTTP POST body.

In \[6\]:

Copied!

```
with app.syncio() as session:
    response: VespaQueryResponse = session.query(
        body={
            "yql": "select title, body from doc where userQuery()",
            "query": "Is statin use connected to breast cancer?",
            "ranking": "bm25",
            "presentation.timing": True,
        },
    )
    print(response.is_successful())
```

with app.syncio() as session: response: VespaQueryResponse = session.query( body={ "yql": "select title, body from doc where userQuery()", "query": "Is statin use connected to breast cancer?", "ranking": "bm25", "presentation.timing": True, }, ) print(response.is_successful())

```
True
```

The query specified that we wanted one hit:

In \[7\]:

Copied!

```
response.hits
```

response.hits

Out\[7\]:

```
[{'id': 'index:sample_content/0/2deca9d7029f3a77c092dfeb',
  'relevance': 21.850306796449487,
  'source': 'sample_content',
  'fields': {'body': 'Recent studies have suggested that <hi>statins</hi>, an established drug group in the prevention of cardiovascular mortality, could delay or prevent <hi>breast</hi> <hi>cancer</hi> recurrence but the effect on disease-specific mortality remains unclear. We evaluated risk of <hi>breast</hi> <hi>cancer</hi> death among <hi>statin</hi> users in a population-based cohort of <hi>breast</hi> <hi>cancer</hi> patients. The study cohort included all newly diagnosed <hi>breast</hi> <hi>cancer</hi> patients in Finland during 1995–2003 (31,236 cases), identified from the Finnish <hi>Cancer</hi> Registry. Information on <hi>statin</hi> <hi>use</hi> before and after the diagnosis was obtained from a national prescription database. We <hi>used</hi> the Cox proportional hazards regression method <hi>to</hi> estimate mortality among <hi>statin</hi> users with <hi>statin</hi> <hi>use</hi> as time-dependent variable. A total of 4,151 participants had <hi>used</hi> <hi>statins</hi>. During the median follow-up of 3.25 years after the diagnosis (range 0.08–9.0 years) 6,011 participants died, of which 3,619 (60.2%) was due <hi>to</hi> <hi>breast</hi> <hi>cancer</hi>. After adjustment for age, tumor characteristics, and treatment selection, both post-diagnostic and pre-diagnostic <hi>statin</hi> <hi>use</hi> were associated with lowered risk of <hi>breast</hi> <hi>cancer</hi> death (HR 0.46, 95% CI 0.38–0.55 and HR 0.54, 95% CI 0.44–0.67, respectively). The risk decrease by post-diagnostic <hi>statin</hi> <hi>use</hi> was likely affected by healthy adherer bias; that <hi>is</hi>, the greater likelihood of dying <hi>cancer</hi> patients <hi>to</hi> discontinue <hi>statin</hi> <hi>use</hi> as the association was not clearly dose-dependent and observed already at low-dose/short-term <hi>use</hi>. The dose- and time-dependence of the survival benefit among pre-diagnostic <hi>statin</hi> users suggests a possible causal effect that should be evaluated further in a clinical trial testing <hi>statins</hi>’ effect on survival in <hi>breast</hi> <hi>cancer</hi> patients.',
   'title': 'Statin Use and Breast Cancer Survival: A Nationwide Cohort Study from Finland'}},
 {'id': 'index:sample_content/0/d2f48bedc26e3838b2fc40d8',
  'relevance': 20.71331154820049,
  'source': 'sample_content',
  'fields': {'body': 'BACKGROUND: Preclinical studies have shown that <hi>statins</hi>, particularly simvastatin, can prevent growth in <hi>breast</hi> <hi>cancer</hi> cell lines and animal models. We investigated whether <hi>statins</hi> <hi>used</hi> after <hi>breast</hi> <hi>cancer</hi> diagnosis reduced the risk of <hi>breast</hi> <hi>cancer</hi>-specific, or all-cause, mortality in a large cohort of <hi>breast</hi> <hi>cancer</hi> patients. METHODS: A cohort of 17,880 <hi>breast</hi> <hi>cancer</hi> patients, newly diagnosed between 1998 and 2009, was identified from English <hi>cancer</hi> registries (from the National <hi>Cancer</hi> Data Repository). This cohort was linked <hi>to</hi> the UK Clinical Practice Research Datalink, providing prescription records, and <hi>to</hi> the Office of National Statistics mortality data (up <hi>to</hi> 2013), identifying 3694 deaths, including 1469 deaths attributable <hi>to</hi> <hi>breast</hi> <hi>cancer</hi>. Unadjusted and adjusted hazard ratios (HRs) for <hi>breast</hi> <hi>cancer</hi>-specific, and all-cause, mortality in <hi>statin</hi> users after <hi>breast</hi> <hi>cancer</hi> diagnosis were calculated <hi>using</hi> time-dependent Cox regression models. Sensitivity analyses were conducted <hi>using</hi> multiple imputation methods, propensity score methods and a case-control approach. RESULTS: There was some evidence that <hi>statin</hi> <hi>use</hi> after a diagnosis of <hi>breast</hi> <hi>cancer</hi> had reduced mortality due <hi>to</hi> <hi>breast</hi> <hi>cancer</hi> and all causes (fully adjusted HR = 0.84 [95% confidence interval = 0.68-1.04] and 0.84 [0.72-0.97], respectively). These associations were more marked for simvastatin 0.79 (0.63-1.00) and 0.81 (0.70-0.95), respectively. CONCLUSIONS: In this large population-based <hi>breast</hi> <hi>cancer</hi> cohort, there was some evidence of reduced mortality in <hi>statin</hi> users after <hi>breast</hi> <hi>cancer</hi> diagnosis. However, these associations were weak in magnitude and were attenuated in some sensitivity analyses.',
   'title': 'Statin use after diagnosis of breast cancer and survival: a population-based cohort study.'}},
 {'id': 'index:sample_content/0/abb8c59b326f35dc406e914d',
  'relevance': 10.546049129391914,
  'source': 'sample_content',
  'fields': {'body': 'BACKGROUND: Although high soy consumption may be associated with lower <hi>breast</hi> <hi>cancer</hi> risk in Asian populations, findings from epidemiological studies have been inconsistent. OBJECTIVE: We investigated the effects of soy intake on <hi>breast</hi> <hi>cancer</hi> risk among Korean women according <hi>to</hi> their menopausal and hormone receptor status. METHODS: We conducted a case-control study with 358 incident <hi>breast</hi> <hi>cancer</hi> patients and 360 age-matched controls with no history of malignant neoplasm. Dietary consumption of soy products was examined <hi>using</hi> a 103-item food frequency questionnaire. RESULTS: The estimated mean intakes of total soy and isoflavones from this study population were 76.5 g per day and 15.0 mg per day, respectively. <hi>Using</hi> a multivariate logistic regression model, we found a significant inverse association between soy intake and <hi>breast</hi> <hi>cancer</hi> risk, with a dose-response relationship (odds ratios (OR) (95% confidence interval (CI)) for the highest vs the lowest intake quartile: 0.36 (0.20-0.64)). When the data were stratified by menopausal status, the protective effect was observed only among postmenopausal women (OR (95% CI) for the highest vs the lowest intake quartile: 0.08 (0.03-0.22)). The association between soy and <hi>breast</hi> <hi>cancer</hi> risk did not differ according <hi>to</hi> estrogen receptor (ER)/progesterone receptor (PR) status, but the estimated intake of soy isoflavones showed an inverse association only among postmenopausal women with ER+/PR+ tumors. CONCLUSIONS: Our findings suggest that high consumption of soy might be related <hi>to</hi> lower risk of <hi>breast</hi> <hi>cancer</hi> and that the effect of soy intake could vary depending on several factors.',
   'title': 'Effect of dietary soy intake on breast cancer risk according to menopause and hormone receptor status.'}},
 {'id': 'index:sample_content/0/3b8b700c0db54b8272a2da54',
  'relevance': 7.360259724997032,
  'source': 'sample_content',
  'fields': {'body': 'Docosahexaenoic acid (DHA) <hi>is</hi> an omega-3 fatty acid that comprises 22 carbons and 6 alternative double bonds in its hydrocarbon chain (22:6omega3). Previous studies have shown that DHA from fish oil controls the growth and development of different <hi>cancers</hi>; however, safety issues have been raised repeatedly about contamination of toxins in fish oil that makes it no longer a clean and safe source of the fatty acid. We investigated the cell growth inhibition of DHA from the cultured microalga Crypthecodinium cohnii (algal DHA [aDHA]) in human <hi>breast</hi> carcinoma MCF-7 cells. aDHA exhibited growth inhibition on <hi>breast</hi> <hi>cancer</hi> cells dose-dependently by 16.0% <hi>to</hi> 59.0% of the control level after 72-h incubations with 40 <hi>to</hi> 160 microM of the fatty acid. DNA flow cytometry shows that aDHA induced sub-G(1) cells, or apoptotic cells, by 64.4% <hi>to</hi> 171.3% of the control levels after incubations with 80 mM of the fatty acid for 24, 48, and 72 h. Western blot studies further show that aDHA did not modulate the expression of proapoptotic Bax protein but induced the downregulation of anti-apoptotic Bcl-2 expression time-dependently, causing increases of Bax/Bcl-2 ratio by 303.4% and 386.5% after 48- and 72-h incubations respectively with the fatty acid. Results from this study suggest that DHA from the cultured microalga <hi>is</hi> also effective in controlling <hi>cancer</hi> cell growth and that downregulation of antiapoptotic Bcl-2 <hi>is</hi> an important step in the induced apoptosis.',
   'title': 'Docosahexaenoic acid from a cultured microalga inhibits cell growth and induces apoptosis by upregulating Bax/Bcl-2 ratio in human breast carcinoma...'}},
 {'id': 'index:sample_content/0/9c2d39bb63ce85fcda9bfe6c',
  'relevance': 5.441906201913548,
  'source': 'sample_content',
  'fields': {'body': 'Background Based on the hypothesized protective effect, we examined the effect of soy foods on estrogens in nipple aspirate fluid (NAF) and serum, possible indicators of <hi>breast</hi> <hi>cancer</hi> risk. Methods In a cross-over design, we randomized 96 women who produced ≥10 μL NAF <hi>to</hi> a high- or low-soy diet for 6-months. During the high-soy diet, participants consumed 2 soy servings of soy milk, tofu, or soy nuts (approximately 50 mg of isoflavones/day); during the low-soy diet, they maintained their usual diet. Six NAF samples were obtained <hi>using</hi> a FirstCyte© Aspirator. Estradiol (E2) and estrone sulfate (E1S) were assessed in NAF and estrone (E1) in serum only <hi>using</hi> highly sensitive radioimmunoassays. Mixed-effects regression models accounting for repeated measures and left-censoring limits were applied. Results Mean E2 and E1S were lower during the high-soy than the low-soy diet (113 vs. 313 pg/mL and 46 vs. 68 ng/mL, respectively) without reaching significance (p=0.07); the interaction between group and diet and was not significant. There was no effect of the soy treatment on serum E2 (p=0.76), E1 (p=0.86), or E1S (p=0.56). Within individuals, NAF and serum levels of E2 (rs=0.37; p<0.001) but not E1S (rs=0.004; p=0.97) were correlated. E2 and E1S in NAF and serum were strongly associated (rs=0.78 and rs=0.48; p<0.001). Conclusions Soy foods in amounts consumed by Asians did not significantly modify estrogen levels in NAF and serum. Impact The trend towards lower estrogens in NAF during the high-soy diet counters concerns about adverse effects of soy foods on <hi>breast</hi> <hi>cancer</hi> risk.',
   'title': 'Estrogen levels in nipple aspirate fluid and serum during a randomized soy trial'}},
 {'id': 'index:sample_content/0/449eccc1b30615316ab136bc',
  'relevance': 5.241472721415711,
  'source': 'sample_content',
  'fields': {'body': 'The relation between various types of fiber and oral, pharyngeal and esophageal <hi>cancer</hi> was investigated <hi>using</hi> data from a case-control study conducted between 1992 and 1997 in Italy. Cases were 271 hospital patients with incident, histologically confirmed oral <hi>cancer</hi>, 327 with pharyngeal <hi>cancer</hi> and 304 with esophageal <hi>cancer</hi>. Controls were 1,950 subjects admitted <hi>to</hi> the same network of hospitals as the cases for acute, nonneoplastic diseases. Cases and controls were interviewed during their hospital stay <hi>using</hi> a validated food frequency questionnaire. Odds ratios (OR) were computed after allowance for age, sex, and other potential confounding factors, including alcohol, tobacco consumption, and energy intake. The ORs for the highest vs. the lowest quintile of intake of oral, pharyngeal and esophageal <hi>cancer</hi> combined were 0.40 for total (Englyst) fiber, 0.37 for soluble fiber, 0.52 for cellulose, 0.48 for insoluble non cellulose polysaccharide, 0.33 for total insoluble fiber and 0.38 for lignin. The inverse relation were similar for vegetable fiber (OR = 0.51), fruit fiber (OR = 0.60) and grain fiber (OR = 0.56), and were somewhat stronger for oral and pharyngeal <hi>cancer</hi> than for esophageal <hi>cancer</hi>. The ORs were similar for the two sexes and strata of age, education, alcohol and tobacco consumption, and total non-alcohol energy intake. Our study indicates that fiber intake may have a protective role on oral, pharyngeal and esophageal <hi>cancer</hi>.',
   'title': 'Fiber intake and the risk of oral, pharyngeal and esophageal cancer.'}},
 {'id': 'index:sample_content/0/c4cb3b969a89b81a3da71e9d',
  'relevance': 5.0658599969730735,
  'source': 'sample_content',
  'fields': {'body': 'BACKGROUND & AIMS: Increasing evidence suggests that a low folate intake and impaired folate metabolism may be implicated in the development of gastrointestinal <hi>cancers</hi>. We conducted a systematic review with meta-analysis of epidemiologic studies evaluating the association of folate intake or genetic polymorphisms in 5,10-methylenetetrahydrofolate reductase (MTHFR), a central enzyme in folate metabolism, with risk of esophageal, gastric, or pancreatic <hi>cancer</hi>. METHODS: A literature search was performed <hi>using</hi> MEDLINE for studies published through March 2006. Study-specific relative risks were weighted by the inverse of their variance <hi>to</hi> obtain random-effects summary estimates. RESULTS: The summary relative risks for the highest versus the lowest category of dietary folate intake were 0.66 (95% confidence interval [CI], 0.53-0.83) for esophageal squamous cell carcinoma (4 case-control), 0.50 (95% CI, 0.39-0.65) for esophageal adenocarcinoma (3 case-control), and 0.49 (95% CI, 0.35-0.67) for pancreatic <hi>cancer</hi> (1 case-control, 4 cohort); there was no heterogeneity among studies. Results on dietary folate intake and risk of gastric <hi>cancer</hi> (9 case-control, 2 cohort) were inconsistent. In most studies, the MTHFR 677TT (variant) genotype, which <hi>is</hi> associated with reduced enzyme activity, was associated with an increased risk of esophageal squamous cell carcinoma, gastric cardia adenocarcinoma, noncardia gastric <hi>cancer</hi>, gastric <hi>cancer</hi> (all subsites), and pancreatic <hi>cancer</hi>; all but one of 22 odds ratios were >1, of which 13 estimates were statistically significant. Studies of the MTHFR A1298C polymorphism were limited and inconsistent. CONCLUSIONS: These findings support the hypothesis that folate may play a role in carcinogenesis of the esophagus, stomach, and pancreas.',
   'title': 'Folate intake, MTHFR polymorphisms, and risk of esophageal, gastric, and pancreatic cancer: a meta-analysis.'}},
 {'id': 'index:sample_content/0/bb0fe2bd511527ef78587e95',
  'relevance': 4.780565525377517,
  'source': 'sample_content',
  'fields': {'body': 'Individual-based studies that investigated the relation between dietary alpha-linolenic acid (ALA) intake and prostate <hi>cancer</hi> risk have shown inconsistent results. We carried out a meta-analysis of prospective studies <hi>to</hi> examine this association. We systematically searched studies published up <hi>to</hi> December 2008. Log relative risks (RRs) were weighted by the inverse of their variances <hi>to</hi> obtain a pooled estimate with its 95% confidence interval (CI). We identified five prospective studies that met our inclusion criteria and reported risk estimates by categories of ALA intake. Comparing the highest <hi>to</hi> the lowest ALA intake category, the pooled RR was 0.97 (95% CI:0.86-1.10) but the association was heterogeneous. <hi>Using</hi> the reported numbers of cases and non-cases in each category of ALA intake, we found that subjects who consumed more than 1.5 g/day of ALA compared with subjects who consumed less than 1.5 g/day had a significant decreased risk of prostate <hi>cancer</hi>: RR = 0.95 (95% CI:0.91-0.99). Divergences in results could partly be explained by differences in sample sizes and adjustment but they also highlight limits in dietary ALA assessment in such prospective studies. Our findings support a weak protective association between dietary ALA intake and prostate <hi>cancer</hi> risk but further research <hi>is</hi> needed <hi>to</hi> conclude on this question.',
   'title': 'Prospective studies of dietary alpha-linolenic acid intake and prostate cancer risk: a meta-analysis.'}},
 {'id': 'index:sample_content/0/90efd2c6652f323a8244690d',
  'relevance': 4.7044749035958535,
  'source': 'sample_content',
  'fields': {'body': 'High serum levels of testosterone and estradiol, the bioavailability of which may be increased by Western dietary habits, seem <hi>to</hi> be important risk factors for postmenopausal <hi>breast</hi> <hi>cancer</hi>. We hypothesized that an ad libitum diet low in animal fat and refined carbohydrates and rich in low-glycemic-index foods, monounsaturated and n-3 polyunsaturated fatty acids, and phytoestrogens, might favorably modify the hormonal profile of postmenopausal women. One hundred and four postmenopausal women selected from 312 healthy volunteers on the basis of high serum testosterone levels were randomized <hi>to</hi> dietary intervention or control. The intervention included intensive dietary counseling and specially prepared group meals twice a week over 4.5 months. Changes in serum levels of testosterone, estradiol, and sex hormone-binding globulin were the main outcome measures. In the intervention group, sex hormone-binding globulin increased significantly (from 36.0 <hi>to</hi> 45.1 nmol/liter) compared with the control group (25 versus 4%,; P < 0.0001) and serum testosterone decreased (from 0.41 <hi>to</hi> 0.33 ng/ml; -20 versus -7% in control group; P = 0.0038). Serum estradiol also decreased, but the change was not significant. The dietary intervention group also significantly decreased body weight (4.06 kg versus 0.54 kg in the control group), waist:hip ratio, total cholesterol, fasting glucose level, and area under insulin curve after oral glucose tolerance test. A radical modification in diet designed <hi>to</hi> reduce insulin resistance and also involving increased phytoestrogen intake decreases the bioavailability of serum sex hormones in hyperandrogenic postmenopausal women. Additional studies are needed <hi>to</hi> determine whether such effects can reduce the risk of developing <hi>breast</hi> <hi>cancer</hi>.',
   'title': 'Reducing bioavailable sex hormones through a comprehensive change in diet: the diet and androgens (DIANA) randomized trial.'}},
 {'id': 'index:sample_content/0/9b56be58163850a7b2ee2425',
  'relevance': 3.896398317302996,
  'source': 'sample_content',
  'fields': {'body': '<hi>Breast</hi> pain <hi>is</hi> a common condition affecting most women at some stage in their reproductive life. Mastalgia <hi>is</hi> resistant <hi>to</hi> treatment in 6% of cyclical and 26% non-cyclical patients. Surgery <hi>is</hi> not widely <hi>used</hi> <hi>to</hi> treat this condition and only considered in patients with severe mastalgia resistant <hi>to</hi> medication. The aims of this study were <hi>to</hi> audit the efficacy of surgery in severe treatment resistant mastalgia and <hi>to</hi> assess patient satisfaction following surgery. This <hi>is</hi> a retrospective review of the medical records of all patients seen in mastalgia clinic in the University Hospital of Wales, Cardiff since 1973. A postal questionnaire was distributed <hi>to</hi> all patients who had undergone surgery. Results showed that of the 1054 patients seen in mastalgia clinic, 12 (1.2%) had undergone surgery. Surgery included 8 subcutaneous mastectomies with implants (3 bilateral, 5 unilateral), 1 bilateral simple mastectomy and 3 quadrantectomies (1 having a further simple mastectomy). The median duration of symptoms was 6.5 years (range 2-16 years). Five patients (50%) were pain free following surgery, 3 developed capsular contractures and 2 wound infections with dehiscence. Pain persisted in both patients undergoing quadrantectomy. We conclude that surgery for mastalgia should only be considered in a minority of patients. Patients should be informed of possible complications inherent of reconstructive surgery and warned that in 50% cases their pain will not be improved.',
   'title': 'Is there a role for surgery in the treatment of mastalgia?'}}]
```

Example of iterating over the returned hits obtained from `response.hits`, extracting the `title` field:

In \[8\]:

Copied!

```
[hit["fields"]["title"] for hit in response.hits]
```

\[hit["fields"]["title"] for hit in response.hits\]

Out\[8\]:

```
['Statin Use and Breast Cancer Survival: A Nationwide Cohort Study from Finland',
 'Statin use after diagnosis of breast cancer and survival: a population-based cohort study.',
 'Effect of dietary soy intake on breast cancer risk according to menopause and hormone receptor status.',
 'Docosahexaenoic acid from a cultured microalga inhibits cell growth and induces apoptosis by upregulating Bax/Bcl-2 ratio in human breast carcinoma...',
 'Estrogen levels in nipple aspirate fluid and serum during a randomized soy trial',
 'Fiber intake and the risk of oral, pharyngeal and esophageal cancer.',
 'Folate intake, MTHFR polymorphisms, and risk of esophageal, gastric, and pancreatic cancer: a meta-analysis.',
 'Prospective studies of dietary alpha-linolenic acid intake and prostate cancer risk: a meta-analysis.',
 'Reducing bioavailable sex hormones through a comprehensive change in diet: the diet and androgens (DIANA) randomized trial.',
 'Is there a role for surgery in the treatment of mastalgia?']
```

Access the full JSON response in the Vespa [default JSON result format](https://docs.vespa.ai/en/reference/default-result-format.html):

In \[9\]:

Copied!

```
response.json
```

response.json

Out\[9\]:

```
{'timing': {'querytime': 0.004,
  'summaryfetchtime': 0.005,
  'searchtime': 0.011},
 'root': {'id': 'toplevel',
  'relevance': 1.0,
  'fields': {'totalCount': 97},
  'coverage': {'coverage': 100,
   'documents': 100,
   'full': True,
   'nodes': 1,
   'results': 1,
   'resultsFull': 1},
  'children': [{'id': 'index:sample_content/0/2deca9d7029f3a77c092dfeb',
    'relevance': 21.850306796449487,
    'source': 'sample_content',
    'fields': {'body': 'Recent studies have suggested that <hi>statins</hi>, an established drug group in the prevention of cardiovascular mortality, could delay or prevent <hi>breast</hi> <hi>cancer</hi> recurrence but the effect on disease-specific mortality remains unclear. We evaluated risk of <hi>breast</hi> <hi>cancer</hi> death among <hi>statin</hi> users in a population-based cohort of <hi>breast</hi> <hi>cancer</hi> patients. The study cohort included all newly diagnosed <hi>breast</hi> <hi>cancer</hi> patients in Finland during 1995–2003 (31,236 cases), identified from the Finnish <hi>Cancer</hi> Registry. Information on <hi>statin</hi> <hi>use</hi> before and after the diagnosis was obtained from a national prescription database. We <hi>used</hi> the Cox proportional hazards regression method <hi>to</hi> estimate mortality among <hi>statin</hi> users with <hi>statin</hi> <hi>use</hi> as time-dependent variable. A total of 4,151 participants had <hi>used</hi> <hi>statins</hi>. During the median follow-up of 3.25 years after the diagnosis (range 0.08–9.0 years) 6,011 participants died, of which 3,619 (60.2%) was due <hi>to</hi> <hi>breast</hi> <hi>cancer</hi>. After adjustment for age, tumor characteristics, and treatment selection, both post-diagnostic and pre-diagnostic <hi>statin</hi> <hi>use</hi> were associated with lowered risk of <hi>breast</hi> <hi>cancer</hi> death (HR 0.46, 95% CI 0.38–0.55 and HR 0.54, 95% CI 0.44–0.67, respectively). The risk decrease by post-diagnostic <hi>statin</hi> <hi>use</hi> was likely affected by healthy adherer bias; that <hi>is</hi>, the greater likelihood of dying <hi>cancer</hi> patients <hi>to</hi> discontinue <hi>statin</hi> <hi>use</hi> as the association was not clearly dose-dependent and observed already at low-dose/short-term <hi>use</hi>. The dose- and time-dependence of the survival benefit among pre-diagnostic <hi>statin</hi> users suggests a possible causal effect that should be evaluated further in a clinical trial testing <hi>statins</hi>’ effect on survival in <hi>breast</hi> <hi>cancer</hi> patients.',
     'title': 'Statin Use and Breast Cancer Survival: A Nationwide Cohort Study from Finland'}},
   {'id': 'index:sample_content/0/d2f48bedc26e3838b2fc40d8',
    'relevance': 20.71331154820049,
    'source': 'sample_content',
    'fields': {'body': 'BACKGROUND: Preclinical studies have shown that <hi>statins</hi>, particularly simvastatin, can prevent growth in <hi>breast</hi> <hi>cancer</hi> cell lines and animal models. We investigated whether <hi>statins</hi> <hi>used</hi> after <hi>breast</hi> <hi>cancer</hi> diagnosis reduced the risk of <hi>breast</hi> <hi>cancer</hi>-specific, or all-cause, mortality in a large cohort of <hi>breast</hi> <hi>cancer</hi> patients. METHODS: A cohort of 17,880 <hi>breast</hi> <hi>cancer</hi> patients, newly diagnosed between 1998 and 2009, was identified from English <hi>cancer</hi> registries (from the National <hi>Cancer</hi> Data Repository). This cohort was linked <hi>to</hi> the UK Clinical Practice Research Datalink, providing prescription records, and <hi>to</hi> the Office of National Statistics mortality data (up <hi>to</hi> 2013), identifying 3694 deaths, including 1469 deaths attributable <hi>to</hi> <hi>breast</hi> <hi>cancer</hi>. Unadjusted and adjusted hazard ratios (HRs) for <hi>breast</hi> <hi>cancer</hi>-specific, and all-cause, mortality in <hi>statin</hi> users after <hi>breast</hi> <hi>cancer</hi> diagnosis were calculated <hi>using</hi> time-dependent Cox regression models. Sensitivity analyses were conducted <hi>using</hi> multiple imputation methods, propensity score methods and a case-control approach. RESULTS: There was some evidence that <hi>statin</hi> <hi>use</hi> after a diagnosis of <hi>breast</hi> <hi>cancer</hi> had reduced mortality due <hi>to</hi> <hi>breast</hi> <hi>cancer</hi> and all causes (fully adjusted HR = 0.84 [95% confidence interval = 0.68-1.04] and 0.84 [0.72-0.97], respectively). These associations were more marked for simvastatin 0.79 (0.63-1.00) and 0.81 (0.70-0.95), respectively. CONCLUSIONS: In this large population-based <hi>breast</hi> <hi>cancer</hi> cohort, there was some evidence of reduced mortality in <hi>statin</hi> users after <hi>breast</hi> <hi>cancer</hi> diagnosis. However, these associations were weak in magnitude and were attenuated in some sensitivity analyses.',
     'title': 'Statin use after diagnosis of breast cancer and survival: a population-based cohort study.'}},
   {'id': 'index:sample_content/0/abb8c59b326f35dc406e914d',
    'relevance': 10.546049129391914,
    'source': 'sample_content',
    'fields': {'body': 'BACKGROUND: Although high soy consumption may be associated with lower <hi>breast</hi> <hi>cancer</hi> risk in Asian populations, findings from epidemiological studies have been inconsistent. OBJECTIVE: We investigated the effects of soy intake on <hi>breast</hi> <hi>cancer</hi> risk among Korean women according <hi>to</hi> their menopausal and hormone receptor status. METHODS: We conducted a case-control study with 358 incident <hi>breast</hi> <hi>cancer</hi> patients and 360 age-matched controls with no history of malignant neoplasm. Dietary consumption of soy products was examined <hi>using</hi> a 103-item food frequency questionnaire. RESULTS: The estimated mean intakes of total soy and isoflavones from this study population were 76.5 g per day and 15.0 mg per day, respectively. <hi>Using</hi> a multivariate logistic regression model, we found a significant inverse association between soy intake and <hi>breast</hi> <hi>cancer</hi> risk, with a dose-response relationship (odds ratios (OR) (95% confidence interval (CI)) for the highest vs the lowest intake quartile: 0.36 (0.20-0.64)). When the data were stratified by menopausal status, the protective effect was observed only among postmenopausal women (OR (95% CI) for the highest vs the lowest intake quartile: 0.08 (0.03-0.22)). The association between soy and <hi>breast</hi> <hi>cancer</hi> risk did not differ according <hi>to</hi> estrogen receptor (ER)/progesterone receptor (PR) status, but the estimated intake of soy isoflavones showed an inverse association only among postmenopausal women with ER+/PR+ tumors. CONCLUSIONS: Our findings suggest that high consumption of soy might be related <hi>to</hi> lower risk of <hi>breast</hi> <hi>cancer</hi> and that the effect of soy intake could vary depending on several factors.',
     'title': 'Effect of dietary soy intake on breast cancer risk according to menopause and hormone receptor status.'}},
   {'id': 'index:sample_content/0/3b8b700c0db54b8272a2da54',
    'relevance': 7.360259724997032,
    'source': 'sample_content',
    'fields': {'body': 'Docosahexaenoic acid (DHA) <hi>is</hi> an omega-3 fatty acid that comprises 22 carbons and 6 alternative double bonds in its hydrocarbon chain (22:6omega3). Previous studies have shown that DHA from fish oil controls the growth and development of different <hi>cancers</hi>; however, safety issues have been raised repeatedly about contamination of toxins in fish oil that makes it no longer a clean and safe source of the fatty acid. We investigated the cell growth inhibition of DHA from the cultured microalga Crypthecodinium cohnii (algal DHA [aDHA]) in human <hi>breast</hi> carcinoma MCF-7 cells. aDHA exhibited growth inhibition on <hi>breast</hi> <hi>cancer</hi> cells dose-dependently by 16.0% <hi>to</hi> 59.0% of the control level after 72-h incubations with 40 <hi>to</hi> 160 microM of the fatty acid. DNA flow cytometry shows that aDHA induced sub-G(1) cells, or apoptotic cells, by 64.4% <hi>to</hi> 171.3% of the control levels after incubations with 80 mM of the fatty acid for 24, 48, and 72 h. Western blot studies further show that aDHA did not modulate the expression of proapoptotic Bax protein but induced the downregulation of anti-apoptotic Bcl-2 expression time-dependently, causing increases of Bax/Bcl-2 ratio by 303.4% and 386.5% after 48- and 72-h incubations respectively with the fatty acid. Results from this study suggest that DHA from the cultured microalga <hi>is</hi> also effective in controlling <hi>cancer</hi> cell growth and that downregulation of antiapoptotic Bcl-2 <hi>is</hi> an important step in the induced apoptosis.',
     'title': 'Docosahexaenoic acid from a cultured microalga inhibits cell growth and induces apoptosis by upregulating Bax/Bcl-2 ratio in human breast carcinoma...'}},
   {'id': 'index:sample_content/0/9c2d39bb63ce85fcda9bfe6c',
    'relevance': 5.441906201913548,
    'source': 'sample_content',
    'fields': {'body': 'Background Based on the hypothesized protective effect, we examined the effect of soy foods on estrogens in nipple aspirate fluid (NAF) and serum, possible indicators of <hi>breast</hi> <hi>cancer</hi> risk. Methods In a cross-over design, we randomized 96 women who produced ≥10 μL NAF <hi>to</hi> a high- or low-soy diet for 6-months. During the high-soy diet, participants consumed 2 soy servings of soy milk, tofu, or soy nuts (approximately 50 mg of isoflavones/day); during the low-soy diet, they maintained their usual diet. Six NAF samples were obtained <hi>using</hi> a FirstCyte© Aspirator. Estradiol (E2) and estrone sulfate (E1S) were assessed in NAF and estrone (E1) in serum only <hi>using</hi> highly sensitive radioimmunoassays. Mixed-effects regression models accounting for repeated measures and left-censoring limits were applied. Results Mean E2 and E1S were lower during the high-soy than the low-soy diet (113 vs. 313 pg/mL and 46 vs. 68 ng/mL, respectively) without reaching significance (p=0.07); the interaction between group and diet and was not significant. There was no effect of the soy treatment on serum E2 (p=0.76), E1 (p=0.86), or E1S (p=0.56). Within individuals, NAF and serum levels of E2 (rs=0.37; p<0.001) but not E1S (rs=0.004; p=0.97) were correlated. E2 and E1S in NAF and serum were strongly associated (rs=0.78 and rs=0.48; p<0.001). Conclusions Soy foods in amounts consumed by Asians did not significantly modify estrogen levels in NAF and serum. Impact The trend towards lower estrogens in NAF during the high-soy diet counters concerns about adverse effects of soy foods on <hi>breast</hi> <hi>cancer</hi> risk.',
     'title': 'Estrogen levels in nipple aspirate fluid and serum during a randomized soy trial'}},
   {'id': 'index:sample_content/0/449eccc1b30615316ab136bc',
    'relevance': 5.241472721415711,
    'source': 'sample_content',
    'fields': {'body': 'The relation between various types of fiber and oral, pharyngeal and esophageal <hi>cancer</hi> was investigated <hi>using</hi> data from a case-control study conducted between 1992 and 1997 in Italy. Cases were 271 hospital patients with incident, histologically confirmed oral <hi>cancer</hi>, 327 with pharyngeal <hi>cancer</hi> and 304 with esophageal <hi>cancer</hi>. Controls were 1,950 subjects admitted <hi>to</hi> the same network of hospitals as the cases for acute, nonneoplastic diseases. Cases and controls were interviewed during their hospital stay <hi>using</hi> a validated food frequency questionnaire. Odds ratios (OR) were computed after allowance for age, sex, and other potential confounding factors, including alcohol, tobacco consumption, and energy intake. The ORs for the highest vs. the lowest quintile of intake of oral, pharyngeal and esophageal <hi>cancer</hi> combined were 0.40 for total (Englyst) fiber, 0.37 for soluble fiber, 0.52 for cellulose, 0.48 for insoluble non cellulose polysaccharide, 0.33 for total insoluble fiber and 0.38 for lignin. The inverse relation were similar for vegetable fiber (OR = 0.51), fruit fiber (OR = 0.60) and grain fiber (OR = 0.56), and were somewhat stronger for oral and pharyngeal <hi>cancer</hi> than for esophageal <hi>cancer</hi>. The ORs were similar for the two sexes and strata of age, education, alcohol and tobacco consumption, and total non-alcohol energy intake. Our study indicates that fiber intake may have a protective role on oral, pharyngeal and esophageal <hi>cancer</hi>.',
     'title': 'Fiber intake and the risk of oral, pharyngeal and esophageal cancer.'}},
   {'id': 'index:sample_content/0/c4cb3b969a89b81a3da71e9d',
    'relevance': 5.0658599969730735,
    'source': 'sample_content',
    'fields': {'body': 'BACKGROUND & AIMS: Increasing evidence suggests that a low folate intake and impaired folate metabolism may be implicated in the development of gastrointestinal <hi>cancers</hi>. We conducted a systematic review with meta-analysis of epidemiologic studies evaluating the association of folate intake or genetic polymorphisms in 5,10-methylenetetrahydrofolate reductase (MTHFR), a central enzyme in folate metabolism, with risk of esophageal, gastric, or pancreatic <hi>cancer</hi>. METHODS: A literature search was performed <hi>using</hi> MEDLINE for studies published through March 2006. Study-specific relative risks were weighted by the inverse of their variance <hi>to</hi> obtain random-effects summary estimates. RESULTS: The summary relative risks for the highest versus the lowest category of dietary folate intake were 0.66 (95% confidence interval [CI], 0.53-0.83) for esophageal squamous cell carcinoma (4 case-control), 0.50 (95% CI, 0.39-0.65) for esophageal adenocarcinoma (3 case-control), and 0.49 (95% CI, 0.35-0.67) for pancreatic <hi>cancer</hi> (1 case-control, 4 cohort); there was no heterogeneity among studies. Results on dietary folate intake and risk of gastric <hi>cancer</hi> (9 case-control, 2 cohort) were inconsistent. In most studies, the MTHFR 677TT (variant) genotype, which <hi>is</hi> associated with reduced enzyme activity, was associated with an increased risk of esophageal squamous cell carcinoma, gastric cardia adenocarcinoma, noncardia gastric <hi>cancer</hi>, gastric <hi>cancer</hi> (all subsites), and pancreatic <hi>cancer</hi>; all but one of 22 odds ratios were >1, of which 13 estimates were statistically significant. Studies of the MTHFR A1298C polymorphism were limited and inconsistent. CONCLUSIONS: These findings support the hypothesis that folate may play a role in carcinogenesis of the esophagus, stomach, and pancreas.',
     'title': 'Folate intake, MTHFR polymorphisms, and risk of esophageal, gastric, and pancreatic cancer: a meta-analysis.'}},
   {'id': 'index:sample_content/0/bb0fe2bd511527ef78587e95',
    'relevance': 4.780565525377517,
    'source': 'sample_content',
    'fields': {'body': 'Individual-based studies that investigated the relation between dietary alpha-linolenic acid (ALA) intake and prostate <hi>cancer</hi> risk have shown inconsistent results. We carried out a meta-analysis of prospective studies <hi>to</hi> examine this association. We systematically searched studies published up <hi>to</hi> December 2008. Log relative risks (RRs) were weighted by the inverse of their variances <hi>to</hi> obtain a pooled estimate with its 95% confidence interval (CI). We identified five prospective studies that met our inclusion criteria and reported risk estimates by categories of ALA intake. Comparing the highest <hi>to</hi> the lowest ALA intake category, the pooled RR was 0.97 (95% CI:0.86-1.10) but the association was heterogeneous. <hi>Using</hi> the reported numbers of cases and non-cases in each category of ALA intake, we found that subjects who consumed more than 1.5 g/day of ALA compared with subjects who consumed less than 1.5 g/day had a significant decreased risk of prostate <hi>cancer</hi>: RR = 0.95 (95% CI:0.91-0.99). Divergences in results could partly be explained by differences in sample sizes and adjustment but they also highlight limits in dietary ALA assessment in such prospective studies. Our findings support a weak protective association between dietary ALA intake and prostate <hi>cancer</hi> risk but further research <hi>is</hi> needed <hi>to</hi> conclude on this question.',
     'title': 'Prospective studies of dietary alpha-linolenic acid intake and prostate cancer risk: a meta-analysis.'}},
   {'id': 'index:sample_content/0/90efd2c6652f323a8244690d',
    'relevance': 4.7044749035958535,
    'source': 'sample_content',
    'fields': {'body': 'High serum levels of testosterone and estradiol, the bioavailability of which may be increased by Western dietary habits, seem <hi>to</hi> be important risk factors for postmenopausal <hi>breast</hi> <hi>cancer</hi>. We hypothesized that an ad libitum diet low in animal fat and refined carbohydrates and rich in low-glycemic-index foods, monounsaturated and n-3 polyunsaturated fatty acids, and phytoestrogens, might favorably modify the hormonal profile of postmenopausal women. One hundred and four postmenopausal women selected from 312 healthy volunteers on the basis of high serum testosterone levels were randomized <hi>to</hi> dietary intervention or control. The intervention included intensive dietary counseling and specially prepared group meals twice a week over 4.5 months. Changes in serum levels of testosterone, estradiol, and sex hormone-binding globulin were the main outcome measures. In the intervention group, sex hormone-binding globulin increased significantly (from 36.0 <hi>to</hi> 45.1 nmol/liter) compared with the control group (25 versus 4%,; P < 0.0001) and serum testosterone decreased (from 0.41 <hi>to</hi> 0.33 ng/ml; -20 versus -7% in control group; P = 0.0038). Serum estradiol also decreased, but the change was not significant. The dietary intervention group also significantly decreased body weight (4.06 kg versus 0.54 kg in the control group), waist:hip ratio, total cholesterol, fasting glucose level, and area under insulin curve after oral glucose tolerance test. A radical modification in diet designed <hi>to</hi> reduce insulin resistance and also involving increased phytoestrogen intake decreases the bioavailability of serum sex hormones in hyperandrogenic postmenopausal women. Additional studies are needed <hi>to</hi> determine whether such effects can reduce the risk of developing <hi>breast</hi> <hi>cancer</hi>.',
     'title': 'Reducing bioavailable sex hormones through a comprehensive change in diet: the diet and androgens (DIANA) randomized trial.'}},
   {'id': 'index:sample_content/0/9b56be58163850a7b2ee2425',
    'relevance': 3.896398317302996,
    'source': 'sample_content',
    'fields': {'body': '<hi>Breast</hi> pain <hi>is</hi> a common condition affecting most women at some stage in their reproductive life. Mastalgia <hi>is</hi> resistant <hi>to</hi> treatment in 6% of cyclical and 26% non-cyclical patients. Surgery <hi>is</hi> not widely <hi>used</hi> <hi>to</hi> treat this condition and only considered in patients with severe mastalgia resistant <hi>to</hi> medication. The aims of this study were <hi>to</hi> audit the efficacy of surgery in severe treatment resistant mastalgia and <hi>to</hi> assess patient satisfaction following surgery. This <hi>is</hi> a retrospective review of the medical records of all patients seen in mastalgia clinic in the University Hospital of Wales, Cardiff since 1973. A postal questionnaire was distributed <hi>to</hi> all patients who had undergone surgery. Results showed that of the 1054 patients seen in mastalgia clinic, 12 (1.2%) had undergone surgery. Surgery included 8 subcutaneous mastectomies with implants (3 bilateral, 5 unilateral), 1 bilateral simple mastectomy and 3 quadrantectomies (1 having a further simple mastectomy). The median duration of symptoms was 6.5 years (range 2-16 years). Five patients (50%) were pain free following surgery, 3 developed capsular contractures and 2 wound infections with dehiscence. Pain persisted in both patients undergoing quadrantectomy. We conclude that surgery for mastalgia should only be considered in a minority of patients. Patients should be informed of possible complications inherent of reconstructive surgery and warned that in 50% cases their pain will not be improved.',
     'title': 'Is there a role for surgery in the treatment of mastalgia?'}}]}}
```

## Query Performance[¶](#query-performance)

There are several things that impact end-to-end query performance:

- HTTP layer performance, connecting handling, mututal TLS handshake and network round-trip latency
  - Make sure to re-use connections using context manager `with vespa.app.syncio():` to avoid setting up new connections for every unique query. See [http best practises](https://cloud.vespa.ai/en/http-best-practices)
  - The size of the fields and the number of hits requested also greatly impact network performance; a larger payload means higher latency.
  - By adding `"presentation.timing": True` as a request parameter, the Vespa response includes the server-side processing (also including reading the query from the network, but not delivering the result over the network). This can be handy for debugging latency.
- Vespa performance, the features used inside the Vespa instance.

In \[10\]:

Copied!

```
with app.syncio(connections=12) as session:
    response: VespaQueryResponse = session.query(
        hits=1,
        body={
            "yql": "select title, body from doc where userQuery()",
            "query": "Is statin use connected to breast cancer?",
            "ranking": "bm25",
            "presentation.timing": True,
        },
    )
    print(response.is_successful())
```

with app.syncio(connections=12) as session: response: VespaQueryResponse = session.query( hits=1, body={ "yql": "select title, body from doc where userQuery()", "query": "Is statin use connected to breast cancer?", "ranking": "bm25", "presentation.timing": True, }, ) print(response.is_successful())

```
True
```

## Compressing queries[¶](#compressing-queries)

The `VespaSync` class has a `compress` argument that can be used to compress the query before sending it to Vespa. This can be useful when the query is large and/or the network is slow. The compression is done using `gzip`, and is supported by Vespa.

By default, the `compress` argument is set to `"auto"`, which means that the query will be compressed if the size of the query is larger than 1024 bytes. The `compress` argument can also be set to `True` or `False` to force the query to be compressed or not, respectively.

The compression will be applied to both queries and feed operations. (HTTP POST or PUT requests).

In \[11\]:

Copied!

```
import time

# Will not compress the request, as body is less than 1024 bytes
with app.syncio(connections=1, compress="auto") as session:
    response: VespaQueryResponse = session.query(
        hits=1,
        body={
            "yql": "select title, body from doc where userQuery()",
            "query": "Is statin use connected to breast cancer?",
            "ranking": "bm25",
            "presentation.timing": True,
        },
    )
    print(response.is_successful())

# Will compress, as the size of the body exceeds 1024 bytes.
large_body = {
    "yql": "select title, body from doc where userQuery()",
    "query": "Is statin use connected to breast cancer?",
    "input.query(q)": "asdf" * 10000,
    "ranking": "bm25",
    "presentation.timing": True,
}
compress_time = {}

with app.syncio(connections=1, compress=True) as session:
    start_time = time.time()
    response: VespaQueryResponse = session.query(
        hits=1,
        body=large_body,
    )
    end_time = time.time()
    compress_time["force_compression"] = end_time - start_time
    print(response.is_successful())

with app.syncio(connections=1, compress="auto") as session:
    start_time = time.time()
    response: VespaQueryResponse = session.query(
        hits=1,
        body=large_body,
    )
    end_time = time.time()
    compress_time["auto"] = end_time - start_time
    print(response.is_successful())

# Force no compression
with app.syncio(compress=False) as session:
    start_time = time.time()
    response: VespaQueryResponse = session.query(
        hits=1,
        body=large_body,
        timeout="5s",
    )
    end_time = time.time()
    compress_time["no_compression"] = end_time - start_time
    print(response.is_successful())
```

import time

# Will not compress the request, as body is less than 1024 bytes

with app.syncio(connections=1, compress="auto") as session: response: VespaQueryResponse = session.query( hits=1, body={ "yql": "select title, body from doc where userQuery()", "query": "Is statin use connected to breast cancer?", "ranking": "bm25", "presentation.timing": True, }, ) print(response.is_successful())

# Will compress, as the size of the body exceeds 1024 bytes.

large_body = { "yql": "select title, body from doc where userQuery()", "query": "Is statin use connected to breast cancer?", "input.query(q)": "asdf" * 10000, "ranking": "bm25", "presentation.timing": True, } compress_time = {} with app.syncio(connections=1, compress=True) as session: start_time = time.time() response: VespaQueryResponse = session.query( hits=1, body=large_body, ) end_time = time.time() compress_time["force_compression"] = end_time - start_time print(response.is_successful()) with app.syncio(connections=1, compress="auto") as session: start_time = time.time() response: VespaQueryResponse = session.query( hits=1, body=large_body, ) end_time = time.time() compress_time["auto"] = end_time - start_time print(response.is_successful())

# Force no compression

with app.syncio(compress=False) as session: start_time = time.time() response: VespaQueryResponse = session.query( hits=1, body=large_body, timeout="5s", ) end_time = time.time() compress_time["no_compression"] = end_time - start_time print(response.is_successful())

```
True
True
True
True
```

In \[12\]:

Copied!

```
compress_time
```

compress_time

Out\[12\]:

```
{'force_compression': 0.02625894546508789,
 'auto': 0.013608932495117188,
 'no_compression': 0.009457826614379883}
```

The differences will be more significant the larger the size of the body, and the slower the network. It might be beneficial to perform a proper benchmarking if performance is critical for your application.

## Running Queries asynchronously[¶](#running-queries-asynchronously)

If you want to benchmark the capacity of a Vespa application, we suggest using [vespa-fbench](https://docs.vespa.ai/en/performance/vespa-benchmarking.html#vespa-fbench), which is a load generator tool that lets you measure throughput and latency with a predefined number of clients. Vespa-fbench is not Vespa-specific, and can be used to benchmark any HTTP service.

Another option is to use the Open Source [k6](https://k6.io/) load testing tool.

If you want to run multiple queries from pyvespa, we suggest using the convenience method `Vespa.query_many_async()`, which allows you to run multiple queries in parallel using the async client. Below, we will demonstrate a simple example of running 100 queries in parallel, and capture the server-reported times and the client-reported time (including network latency).

In \[13\]:

Copied!

```
# This cell is necessary when running async code in Jupyter Notebooks, as it already runs an event loop
import nest_asyncio

nest_asyncio.apply()
```

# This cell is necessary when running async code in Jupyter Notebooks, as it already runs an event loop

import nest_asyncio nest_asyncio.apply()

In \[14\]:

Copied!

```
import time


query = {
    "yql": "select title, body from doc where userQuery()",
    "query": "Is statin use connected to breast cancer?",
    "ranking": "bm25",
    "presentation.timing": True,
}

# List of queries with hits from 1 to 100
queries = [{**query, "hits": hits} for hits in range(1, 51)]

# Run the queries concurrently
start_time = time.time()
responses = await app.query_many_async(queries=queries)
end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")
# Print QPS
print(f"QPS: {len(queries) / (end_time - start_time):.2f}")
```

import time query = { "yql": "select title, body from doc where userQuery()", "query": "Is statin use connected to breast cancer?", "ranking": "bm25", "presentation.timing": True, }

# List of queries with hits from 1 to 100

queries = [{\*\*query, "hits": hits} for hits in range(1, 51)]

# Run the queries concurrently

start_time = time.time() responses = await app.query_many_async(queries=queries) end_time = time.time() print(f"Total time: {end_time - start_time:.2f} seconds")

# Print QPS

print(f"QPS: {len(queries) / (end_time - start_time):.2f}")

```
Total time: 0.68 seconds
QPS: 73.49
```

In \[15\]:

Copied!

```
dict_responses = [response.json for response in responses]
```

dict_responses = [response.json for response in responses]

In \[16\]:

Copied!

```
# Create a pandas DataFrame with the responses
import pandas as pd

df = pd.DataFrame(
    [
        {
            "hits": len(
                response.get("root", {}).get("children", [])
            ),  # Some responses may not have 'children'
            "search_time": response["timing"]["searchtime"],
            "query_time": response["timing"]["querytime"],
            "summary_time": response["timing"]["summaryfetchtime"],
        }
        for response in dict_responses
    ]
)
df
```

# Create a pandas DataFrame with the responses

import pandas as pd df = pd.DataFrame( \[ { "hits": len( response.get("root", {}).get("children", []) ), # Some responses may not have 'children' "search_time": response["timing"]["searchtime"], "query_time": response["timing"]["querytime"], "summary_time": response["timing"]["summaryfetchtime"], } for response in dict_responses \] ) df

Out\[16\]:

|     | hits | search_time | query_time | summary_time |
| --- | ---- | ----------- | ---------- | ------------ |
| 0   | 1    | 0.006       | 0.003      | 0.002        |
| 1   | 2    | 0.014       | 0.006      | 0.006        |
| 2   | 3    | 0.046       | 0.024      | 0.019        |
| 3   | 4    | 0.037       | 0.015      | 0.010        |
| 4   | 5    | 0.468       | 0.035      | 0.422        |
| 5   | 6    | 0.199       | 0.014      | 0.177        |
| 6   | 7    | 0.018       | 0.008      | 0.009        |
| 7   | 8    | 0.041       | 0.012      | 0.025        |
| 8   | 9    | 0.103       | 0.018      | 0.082        |
| 9   | 10   | 0.288       | 0.022      | 0.265        |
| 10  | 11   | 0.568       | 0.015      | 0.544        |
| 11  | 12   | 0.507       | 0.026      | 0.480        |
| 12  | 13   | 0.470       | 0.012      | 0.457        |
| 13  | 14   | 0.566       | 0.025      | 0.535        |
| 14  | 15   | 0.566       | 0.027      | 0.534        |
| 15  | 16   | 0.213       | 0.018      | 0.194        |
| 16  | 17   | 0.564       | 0.010      | 0.549        |
| 17  | 18   | 0.543       | 0.025      | 0.516        |
| 18  | 19   | 0.545       | 0.016      | 0.520        |
| 19  | 20   | 0.329       | 0.017      | 0.308        |
| 20  | 21   | 0.413       | 0.010      | 0.396        |
| 21  | 22   | 0.088       | 0.010      | 0.078        |
| 22  | 23   | 0.418       | 0.019      | 0.382        |
| 23  | 24   | 0.401       | 0.021      | 0.379        |
| 24  | 25   | 0.348       | 0.013      | 0.334        |
| 25  | 26   | 0.554       | 0.020      | 0.527        |
| 26  | 27   | 0.532       | 0.204      | 0.322        |
| 27  | 28   | 0.550       | 0.023      | 0.524        |
| 28  | 29   | 0.211       | 0.005      | 0.202        |
| 29  | 30   | 0.524       | 0.312      | 0.208        |
| 30  | 31   | 0.440       | 0.016      | 0.422        |
| 31  | 32   | 0.537       | 0.459      | 0.075        |
| 32  | 33   | 0.532       | 0.285      | 0.232        |
| 33  | 34   | 0.397       | 0.024      | 0.371        |
| 34  | 35   | 0.398       | 0.046      | 0.345        |
| 35  | 36   | 0.555       | 0.036      | 0.512        |
| 36  | 37   | 0.545       | 0.009      | 0.525        |
| 37  | 38   | 0.145       | 0.018      | 0.116        |
| 38  | 39   | 0.418       | 0.022      | 0.394        |
| 39  | 40   | 0.373       | 0.013      | 0.359        |
| 40  | 41   | 0.426       | 0.044      | 0.381        |
| 41  | 0    | 0.446       | 0.446      | 0.000        |
| 42  | 43   | 0.292       | 0.014      | 0.267        |
| 43  | 44   | 0.383       | 0.027      | 0.344        |
| 44  | 45   | 0.422       | 0.012      | 0.409        |
| 45  | 46   | 0.515       | 0.034      | 0.475        |
| 46  | 47   | 0.518       | 0.039      | 0.475        |
| 47  | 48   | 0.504       | 0.007      | 0.493        |
| 48  | 49   | 0.505       | 0.012      | 0.488        |
| 49  | 50   | 0.517       | 0.007      | 0.500        |

## Error handling[¶](#error-handling)

Vespa's default query timeout is 500ms; Pyvespa will by default retry up to 3 times for queries that return response codes like 429, 500,503 and 504. A `VespaError` is raised if retries did not end up with success. In the following example, we set a very low [timeout](https://docs.vespa.ai/en/reference/query-api-reference.html#timeout) of `1ms` which will cause Vespa to time out the request, and it returns a 504 http error code. The underlying error is wrapped in a `VespaError` with the payload error message returned from Vespa:

In \[17\]:

Copied!

```
with app.syncio(connections=12) as session:
    try:
        response: VespaQueryResponse = session.query(
            hits=1,
            body={
                "yql": "select * from doc where userQuery()",
                "query": "Is statin use connected to breast cancer?",
                "timeout": "1ms",
            },
        )
        print(response.is_successful())
    except VespaError as e:
        print(str(e))
```

with app.syncio(connections=12) as session: try: response: VespaQueryResponse = session.query( hits=1, body={ "yql": "select * from doc where userQuery()", "query": "Is statin use connected to breast cancer?", "timeout": "1ms", }, ) print(response.is_successful()) except VespaError as e: print(str(e))

```
[{'code': 12, 'summary': 'Timed out', 'message': 'No time left after waiting for 1ms to execute query'}]
```

In the following example, we forgot to include the `query` parameter but still reference it in the yql. This causes a bad client request response (400):

In \[18\]:

Copied!

```
with app.syncio(connections=12) as session:
    try:
        response: VespaQueryResponse = session.query(
            hits=1, body={"yql": "select * from doc where userQuery()"}
        )
        print(response.is_successful())
    except VespaError as e:
        print(str(e))
```

with app.syncio(connections=12) as session: try: response: VespaQueryResponse = session.query( hits=1, body={"yql": "select * from doc where userQuery()"} ) print(response.is_successful()) except VespaError as e: print(str(e))

```
[{'code': 3, 'summary': 'Illegal query', 'source': 'sample_content', 'message': 'No query'}]
```

## Using the Querybuilder DSL API[¶](#using-the-querybuilder-dsl-api)

From `pyvespa>=0.52.0`, we provide a Domain Specific Language (DSL) that allows you to build queries programmatically in the `vespa.querybuilder`-module. See [reference](https://vespa-engine.github.io/pyvespa/api/vespa/querybuilder/builder/builder.md) for full details. There are also many examples in our tests:

- <https://github.com/vespa-engine/pyvespa/blob/master/tests/unit/test_grouping.py>
- <https://github.com/vespa-engine/pyvespa/blob/master/tests/unit/test_qb.py>
- <https://github.com/vespa-engine/pyvespa/blob/master/tests/integration/test_integration_grouping.py>
- <https://github.com/vespa-engine/pyvespa/blob/master/tests/integration/test_integration_queries.py>

This section demonstrates common query patterns using the querybuilder DSL. All features of the Vespa Query Language are supported by the querybuilder DSL.

Using the Querybuilder DSL is completely optional, and you can always use the Vespa Query Language directly by passing the query as a string, which might be more convenient for simple queries.

We will use the Vespa documentation search app for some advanced examples that require specific schemas. For basic examples, our local sample app works well.

In \[20\]:

Copied!

```
# Example using QueryBuilder with our sample app
import vespa.querybuilder as qb
from vespa.querybuilder import QueryField

title = QueryField("title")
body = QueryField("body")

# Build a query to find documents containing "asthma" in title or body
q = (
    qb.select(["title", "body"])
    .from_("doc")
    .where(title.contains("asthma") | body.contains("asthma"))
    .set_limit(5)
)

print(f"Query: {q}")
with app.syncio() as session:
    resp = session.query(yql=q, ranking="bm25")
    print(f"Found {len(resp.hits)} documents")
    for hit in resp.hits:
        print(f"- {hit['fields']['title']}")
```

# Example using QueryBuilder with our sample app

import vespa.querybuilder as qb from vespa.querybuilder import QueryField title = QueryField("title") body = QueryField("body")

# Build a query to find documents containing "asthma" in title or body

q = ( qb.select(["title", "body"]) .from\_("doc") .where(title.contains("asthma") | body.contains("asthma")) .set_limit(5) ) print(f"Query: {q}") with app.syncio() as session: resp = session.query(yql=q, ranking="bm25") print(f"Found {len(resp.hits)} documents") for hit in resp.hits: print(f"- {hit['fields']['title']}")

```
Query: select title, body from doc where title contains "asthma" or body contains "asthma" limit 5
Found 0 documents
```

In \[24\]:

Copied!

```
app = Vespa(url="https://api.search.vespa.ai")
```

app = Vespa(url="https://api.search.vespa.ai")

### Advanced QueryBuilder Examples[¶](#advanced-querybuilder-examples)

For the following advanced examples, we'll switch to using Vespa's documentation search app which has more complex schemas. First, let's clean up our sample app:

### Example 1 - matches, order by and limit[¶](#example-1-matches-order-by-and-limit)

We want to find the 10 documents with the most terms in the 'pyvespa'-namespace (the documentation search has a 'namespace'-field, which refers to the source of the documentation). Note that the documentation search operates on the 'paragraph'-schema, but for demo purposes, we will use the 'document'-schema.

In \[25\]:

Copied!

```
import vespa.querybuilder as qb
from vespa.querybuilder import QueryField

namespace = QueryField("namespace")
q = (
    qb.select(["title", "path", "term_count"])
    .from_("doc")
    .where(
        namespace.matches("pyvespa")
    )  # matches is regex-match, see https://docs.vespa.ai/en/reference/query-language-reference.html#matches
    .order_by("term_count", ascending=False)
    .set_limit(10)
)
print(f"Query: {q}")
resp = app.query(yql=q)
results = [hit["fields"] for hit in resp.hits]
df = pd.DataFrame(results)
df
```

import vespa.querybuilder as qb from vespa.querybuilder import QueryField namespace = QueryField("namespace") q = ( qb.select(["title", "path", "term_count"]) .from\_("doc") .where( namespace.matches("pyvespa") ) # matches is regex-match, see https://docs.vespa.ai/en/reference/query-language-reference.html#matches .order_by("term_count", ascending=False) .set_limit(10) ) print(f"Query: {q}") resp = app.query(yql=q) results = \[hit["fields"] for hit in resp.hits\] df = pd.DataFrame(results) df

```
Query: select title, path, term_count from doc where namespace matches "pyvespa" order by term_count desc limit 10
```

Out\[25\]:

|     | path                                              | title                                              | term_count |
| --- | ------------------------------------------------- | -------------------------------------------------- | ---------- |
| 0   | /examples/feed_performance.html                   | Feeding performance¶                               | 76669      |
| 1   | /examples/simplified-retrieval-with-colpali-vl... | Scaling ColPALI (VLM) Retrieval¶                   | 14393      |
| 2   | /examples/pdf-retrieval-with-ColQwen2-vlm_Vesp... | PDF-Retrieval using ColQWen2 (ColPali) with Ve...  | 14309      |
| 3   | /examples/colpali-document-retrieval-vision-la... | Vespa 🤝 ColPali: Efficient Document Retrieval ... | 13996      |
| 4   | /examples/colpali-benchmark-vqa-vlm_Vespa-clou... | ColPali Ranking Experiments on DocVQA¶             | 13692      |
| 5   | /examples/visual_pdf_rag_with_vespa_colpali_cl... | Visual PDF RAG with Vespa - ColPali demo appli...  | 8237       |
| 6   | /examples/billion-scale-vector-search-with-coh... | Billion-scale vector search with Cohere binary...  | 7880       |
| 7   | /examples/video_search_twelvelabs_cloud.html      | Video Search and Retrieval with Vespa and Twel...  | 7605       |
| 8   | /examples/chat_with_your_pdfs_using_colbert_la... | Chat with your pdfs with ColBERT, langchain, a...  | 7501       |
| 9   | /api/vespa/package.html                           | Package                                            | 6059       |

### Example 2 - timestamp range, contains[¶](#example-2-timestamp-range-contains)

We want to find the documents where one of the indexed fields contains the query term `embedding`,is updated after Jan 1st 2024 and the current timestamp, and have the documents ranked the 'documentation' rank profile. See <https://github.com/vespa-cloud/vespa-documentation-search/blob/main/src/main/application/schemas/doc.sd>.

In \[26\]:

Copied!

```
import vespa.querybuilder as qb
from vespa.querybuilder import QueryField
from datetime import datetime

queryterm = "embedding"

# We need to instantiate a QueryField for fields that we want to call methods on
last_updated = QueryField("last_updated")
title = QueryField("title")
headers = QueryField("headers")
path = QueryField("path")
namespace = QueryField("namespace")
content = QueryField("content")

from_ts = int(datetime(2024, 1, 1).timestamp())
to_ts = int(datetime.now().timestamp())
print(f"From: {from_ts}, To: {to_ts}")
q = (
    qb.select(
        [title, last_updated, content]
    )  # Select takes either a list of QueryField or strings, (or '*' for all fields)
    .from_("doc")
    .where(
        namespace.matches("op.*")
        & last_updated.in_range(from_ts, to_ts)  # could also use > and <
        & qb.weakAnd(
            title.contains(queryterm),
            content.contains(queryterm),
            headers.contains(queryterm),
            path.contains(queryterm),
        )
    )
    .set_limit(3)
)
print(f"Query: {q}")
resp = app.query(yql=q, ranking="documentation")
```

import vespa.querybuilder as qb from vespa.querybuilder import QueryField from datetime import datetime queryterm = "embedding"

# We need to instantiate a QueryField for fields that we want to call methods on

last_updated = QueryField("last_updated") title = QueryField("title") headers = QueryField("headers") path = QueryField("path") namespace = QueryField("namespace") content = QueryField("content") from_ts = int(datetime(2024, 1, 1).timestamp()) to_ts = int(datetime.now().timestamp()) print(f"From: {from_ts}, To: {to_ts}") q = ( qb.select( [title, last_updated, content] ) # Select takes either a list of QueryField or strings, (or '\*' for all fields) .from\_("doc") .where( namespace.matches("op.\*") & last_updated.in_range(from_ts, to_ts) # could also use > and < & qb.weakAnd( title.contains(queryterm), content.contains(queryterm), headers.contains(queryterm), path.contains(queryterm), ) ) .set_limit(3) ) print(f"Query: {q}") resp = app.query(yql=q, ranking="documentation")

```
From: 1704063600, To: 1749803562
Query: select title, last_updated, content from doc where namespace matches "op.*" and range(last_updated, 1704063600, 1749803562) and weakAnd(title contains "embedding", content contains "embedding", headers contains "embedding", path contains "embedding") limit 3
```

In \[27\]:

Copied!

```
df = pd.DataFrame([hit["fields"] | hit for hit in resp.hits])
df = pd.concat(
    [
        df.drop(["matchfeatures", "fields"], axis=1),
        pd.json_normalize(df["matchfeatures"]),
    ],
    axis=1,
)
df.T
```

df = pd.DataFrame(\[hit["fields"] | hit for hit in resp.hits\]) df = pd.concat( \[ df.drop(["matchfeatures", "fields"], axis=1), pd.json_normalize(df["matchfeatures"]), \], axis=1, ) df.T

Out\[27\]:

|                             | 0                                                 | 1                                                 | 2                                                 |
| --------------------------- | ------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| content                     | <sep />similar data by finding nearby points i... | Reference configuration for <hi>embedders</hi>... | <sep /> basic news search application - applic... |
| title                       | Embedding                                         | Embedding Reference                               | News search and recommendation tutorial - embe... |
| last_updated                | 1749727838                                        | 1749727838                                        | 1749727839                                        |
| id                          | index:documentation/0/5d6e77ca20d4e8ee29716747    | index:documentation/1/a03c4aef22fcde916804d3d9    | index:documentation/1/ad44f35cbd7b8214f88963e3    |
| relevance                   | 23.259617                                         | 22.075122                                         | 16.505077                                         |
| source                      | documentation                                     | documentation                                     | documentation                                     |
| bm25(content)               | 2.385057                                          | 2.352575                                          | 2.384316                                          |
| bm25(headers)               | 7.476571                                          | 8.106656                                          | 5.46136                                           |
| bm25(keywords)              | 0.0                                               | 0.0                                               | 0.0                                               |
| bm25(path)                  | 3.990027                                          | 3.349312                                          | 3.100325                                          |
| bm25(title)                 | 4.703981                                          | 4.133289                                          | 2.779538                                          |
| fieldLength(content)        | 3825.0                                            | 2031.0                                            | 3273.0                                            |
| fieldLength(title)          | 1.0                                               | 2.0                                               | 6.0                                               |
| fieldMatch(content)         | 0.915753                                          | 0.892113                                          | 0.915871                                          |
| fieldMatch(content).matches | 1.0                                               | 1.0                                               | 1.0                                               |
| fieldMatch(title)           | 1.0                                               | 0.933869                                          | 0.842758                                          |
| query(contentWeight)        | 1.0                                               | 1.0                                               | 1.0                                               |
| query(headersWeight)        | 1.0                                               | 1.0                                               | 1.0                                               |
| query(pathWeight)           | 1.0                                               | 1.0                                               | 1.0                                               |
| query(titleWeight)          | 2.0                                               | 2.0                                               | 2.0                                               |

### Example 3 - Basic grouping[¶](#example-3-basic-grouping)

Vespa supports grouping and aggregation of matches through the Vespa grouping language. For an introduction to grouping, see <https://docs.vespa.ai/en/grouping.html>.

We will use [purchase schema](https://github.com/vespa-cloud/vespa-documentation-search/blob/main/src/main/application/schemas/purchase.sd) that is also deployed in the documentation search app.

In \[28\]:

Copied!

```
from vespa.querybuilder import Grouping as G

grouping = G.all(
    G.group("customer"),
    G.each(G.output(G.sum("price"))),
)
q = qb.select("*").from_("purchase").where(True).set_limit(0).groupby(grouping)
print(f"Query: {q}")
resp = app.query(yql=q)
group = resp.hits[0]["children"][0]["children"]
# get value and sum(price) into a DataFrame
df = pd.DataFrame([hit["fields"] | hit for hit in group])
df = df.loc[:, ["value", "sum(price)"]]
df
```

from vespa.querybuilder import Grouping as G grouping = G.all( G.group("customer"), G.each(G.output(G.sum("price"))), ) q = qb.select("\*").from\_("purchase").where(True).set_limit(0).groupby(grouping) print(f"Query: {q}") resp = app.query(yql=q) group = resp.hits[0]["children"][0]["children"]

# get value and sum(price) into a DataFrame

df = pd.DataFrame(\[hit["fields"] | hit for hit in group\]) df = df.loc\[:, ["value", "sum(price)"]\] df

```
Query: select * from purchase where true limit 0 | all(group(customer) each(output(sum(price))))
```

Out\[28\]:

|     | value | sum(price) |
| --- | ----- | ---------- |
| 0   | Brown | 20537      |
| 1   | Jones | 39816      |
| 2   | Smith | 19484      |

### Example 4 - Nested grouping[¶](#example-4-nested-grouping)

Let's find out how much each customer has spent per day by grouping on customer, then date:

In \[29\]:

Copied!

```
from vespa.querybuilder import Grouping as G

# First, we construct the grouping expression:
grouping = G.all(
    G.group("customer"),
    G.each(
        G.group(G.time_date("date")),
        G.each(
            G.output(G.sum("price")),
        ),
    ),
)
# Then, we construct the query:
q = qb.select("*").from_("purchase").where(True).groupby(grouping)
print(f"Query: {q}")
resp = app.query(yql=q)
group_data = resp.hits[0]["children"][0]["children"]
records = [
    {
        "GroupId": group["value"],
        "Date": date_entry["value"],
        "Sum(price)": date_entry["fields"].get("sum(price)", 0),
    }
    for group in group_data
    for date_group in group.get("children", [])
    for date_entry in date_group.get("children", [])
]

# Create DataFrame
df = pd.DataFrame(records)
df
```

from vespa.querybuilder import Grouping as G

# First, we construct the grouping expression:

grouping = G.all( G.group("customer"), G.each( G.group(G.time_date("date")), G.each( G.output(G.sum("price")), ), ), )

# Then, we construct the query:

q = qb.select("\*").from\_("purchase").where(True).groupby(grouping) print(f"Query: {q}") resp = app.query(yql=q) group_data = resp.hits[0]["children"][0]["children"] records = \[ { "GroupId": group["value"], "Date": date_entry["value"], "Sum(price)": date_entry["fields"].get("sum(price)", 0), } for group in group_data for date_group in group.get("children", []) for date_entry in date_group.get("children", []) \]

# Create DataFrame

df = pd.DataFrame(records) df

```
Query: select * from purchase where true | all(group(customer) each(group(time.date(date)) each(output(sum(price)))))
```

Out\[29\]:

|     | GroupId | Date      | Sum(price) |
| --- | ------- | --------- | ---------- |
| 0   | Brown   | 2006-9-10 | 7540       |
| 1   | Brown   | 2006-9-11 | 1597       |
| 2   | Brown   | 2006-9-8  | 8000       |
| 3   | Brown   | 2006-9-9  | 3400       |
| 4   | Jones   | 2006-9-10 | 8900       |
| 5   | Jones   | 2006-9-11 | 20816      |
| 6   | Jones   | 2006-9-8  | 8000       |
| 7   | Jones   | 2006-9-9  | 2100       |
| 8   | Smith   | 2006-9-10 | 6100       |
| 9   | Smith   | 2006-9-11 | 2584       |
| 10  | Smith   | 2006-9-6  | 1000       |
| 11  | Smith   | 2006-9-7  | 3000       |
| 12  | Smith   | 2006-9-9  | 6800       |

### Example 5 - Grouping with expressions[¶](#example-5-grouping-with-expressions)

Instead of just grouping on some attribute value, the group clause may contain arbitrarily complex expressions - see [Grouping reference](https://vespa-engine.github.io/pyvespa/api/vespa/querybuilder/grouping/grouping.md#vespa.querybuilder.Grouping) for exhaustive list.

Examples:

- Select the minimum or maximum of sub-expressions
- Addition, subtraction, multiplication, division, and even modulo of - sub-expressions
- Bitwise operations on sub-expressions
- Concatenation of the results of sub-expressions

Let's use some of these expressions to get the sum the prices of purchases on a per-hour-of-day basis.

In \[30\]:

Copied!

```
from vespa.querybuilder import Grouping as G

grouping = G.all(
    G.group(G.mod(G.div("date", G.mul(60, 60)), 24)),
    G.order(-G.sum("price")),
    G.each(G.output(G.sum("price"))),
)
q = qb.select("*").from_("purchase").where(True).groupby(grouping)
print(f"Query: {q}")
resp = app.query(yql=q)
group_data = resp.hits[0]["children"][0]["children"]
df = pd.DataFrame([hit["fields"] | hit for hit in group_data])
df = df.loc[:, ["value", "sum(price)"]]
df
```

from vespa.querybuilder import Grouping as G grouping = G.all( G.group(G.mod(G.div("date", G.mul(60, 60)), 24)), G.order(-G.sum("price")), G.each(G.output(G.sum("price"))), ) q = qb.select("\*").from\_("purchase").where(True).groupby(grouping) print(f"Query: {q}") resp = app.query(yql=q) group_data = resp.hits[0]["children"][0]["children"] df = pd.DataFrame(\[hit["fields"] | hit for hit in group_data\]) df = df.loc\[:, ["value", "sum(price)"]\] df

```
Query: select * from purchase where true | all(group(mod(div(date, mul(60, 60)),24)) order(-sum(price)) each(output(sum(price))))
```

Out\[30\]:

|     | value | sum(price) |
| --- | ----- | ---------- |
| 0   | 10    | 26181      |
| 1   | 9     | 23524      |
| 2   | 8     | 22367      |
| 3   | 11    | 6765       |
| 4   | 7     | 1000       |
