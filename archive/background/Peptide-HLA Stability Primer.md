Iterate AI × Science Hackathon · Track 3, Serova

# What a peptide-HLA complex is, and why its half-life is worth predicting

A ground-up primer for the Serova track: the immunology, the reason anyone pays for this, what the Karthikeyan et al. preprint actually did, and what each of the four candidate approaches commits you to.

**Written** 2 Oct 2026 **Hack** 3–4 Oct, Paris Garden SE1 **Core paper** [SPEARMINT (bioRxiv)](https://www.biorxiv.org/content/10.64898/2026.06.28.735023) **Terms** dotted underline = tap for a definition

[Orientation](#orient) [1. The biology](#biology) [2. Why it matters](#stakes) [3. The paper](#paper) [4. The benchmark trap](#trap) [5. The four angles](#angles) [Glossary](#glossary)

Contents

1. [Orientation](#orient)
2. [1 · The biology](#biology)
3. [2 · Why it matters](#stakes)
4. [3 · The paper](#paper)
5. [4 · The benchmark trap](#trap)
6. [5 · The four angles](#angles)
7. [Glossary](#glossary)

The whole thing in six lines

Every cell in your body continuously shreds its own proteins and displays the fragments on its surface, clipped into a molecule called **HLA**. Killer T cells patrol past and inspect what is on display. If a fragment looks wrong — viral, or mutated — the cell gets killed.

For that to work, the fragment has to *still be there* when a T cell arrives. Some peptide-HLA complexes fall apart in minutes; others last half a day. That lifetime is the **stability**, measured as a half-life in hours.

Cancer vaccines are built by picking a handful of mutated peptides out of thousands of candidates. Picking the ones that will actually stay on display is the problem. The field’s standard tool for it, `NetMHCstabpan`, is a small 2016 neural net trained on about 27,000 measurements. Your track asks whether protein foundation models — trained on billions of sequences — can do better.

Two things are worth knowing before you read anything else, because almost every confusion downstream traces back to one of them.

1. **Affinity and stability are different quantities.** Affinity asks how tightly a peptide binds at equilibrium. Stability asks how long the complex survives once it exists. They correlate, but not enough — and the immunology cares more about the second one.
2. **The headline number on the leaderboard is contaminated.** `NetMHCstabpan` scores 0.88 on the benchmark everyone quotes because it was trained on that data. Beating it honestly means building a clean benchmark first. More on this in §4 — it is the single highest-leverage thing your team can get right.

PART 01

## HLA, peptides, and the complex they form

What the molecule is, what it is doing there, and why the number we want to predict is a half-life rather than a binding constant.

### The surveillance system

Your immune system has a hard problem: a virus replicating *inside* a cell is invisible from the outside. Antibodies can’t reach it. So vertebrates evolved a reporting mechanism — every nucleated cell is obliged to publish a sample of its own internal protein contents on its outer surface, continuously, whether or not anything is wrong.

The publishing rack is **MHC class I**. In humans these molecules are called **HLA** (human leukocyte antigen) — same thing, human-specific name. There are three class I genes that matter: `HLA-A`, `HLA-B` and `HLA-C`. You inherit one copy of each from each parent, so you walk around with up to six different class I molecules, each displaying its own slice of your proteome.

**Peptide binding to MHC-I is the rate-limiting step** of the whole chain, which is why computational effort concentrates on that one box. The branch is the part this hackathon is about: a complex that forms and then falls apart quickly never gets inspected, no matter how well it bound.

### What the molecule physically looks like

An HLA class I molecule is a heavy chain of about 365 amino acids plus a small partner protein, β2-microglobulin, that props it up. The top of the heavy chain folds into a groove: a flat sheet for the floor and two long helices for the walls. The peptide lies in that groove like a cable in a channel.

The groove is **closed at both ends**. That is the single most consequential structural fact about class I, because it fixes the peptide length: 8 to 11 residues, and overwhelmingly . Class II grooves are open-ended and take 13–25mers, which is why class II is a separate modelling problem.

Within the groove, a couple of positions do most of the work. Their side chains drop into deep pockets in the floor — these are the , usually position 2 and the last position (written PΩ). The residues in between arch upward, out of the groove, where a T-cell receptor can touch them.

**GILGFVFTL is a real epitope** — influenza matrix protein 58–66, presented on HLA-A\*02:01, probably the single most-studied peptide in human immunology. Its isoleucine at P2 and leucine at PΩ are exactly the side chains that pocket B and pocket F of A\*02:01 prefer. Swap either one for a charged residue and the complex falls apart.

### Why there are so many HLAs, and what the names mean

A trick this good invites counter-attack: a virus that evolved to avoid one person’s display rack would wipe out everyone if we all had the same rack. So HLA is the most variable region in the human genome. As of the April 2026 database release there are **30,511 known class I alleles** — 9,175 at `HLA-A`, 11,110 at `HLA-B`, 9,288 at `HLA-C`.

Allele names decode left to right. In `HLA-A*02:01`: the gene is `A`, the first field `02` is the broad protein family, the second field `01` is the specific protein variant. Two fields is the resolution everyone models at. `A*02:01` is the most common class I allele in European populations, which is why it dominates every dataset you will touch.

#### The , and why the preprint drops it

Most of those 365 residues never touch the peptide. The `NetMHCpan` family exploits this: it feeds the model only the **34 residues that line the groove**, a fixed-position extract called the pseudosequence. Compact, and it makes the model pan-specific — it can handle an allele it has never seen as long as you can write down those 34 letters.

Karthikeyan et al. deliberately feed the **whole heavy chain** instead. Their argument: residues outside the groove measurably affect how long the assembled complex survives, even though they do not contact the peptide, and a full-sequence model works for any allele with a sequence in the database — including engineered mutants that have no pseudosequence entry. Whether that trade is worth it is, conveniently, a cheap experiment you could run.

### The distinction everything hinges on

Here is where most people’s intuition goes wrong. “Binds well” is not one property. Binding is a reversible reaction with two separate rates.

Because affinity is a **ratio** and stability depends on **one of the two rates**, two peptides can have identical Kd and completely different half-lives — one associates slowly and lets go slowly, the other does both fast. Affinity predictors cannot see that difference by construction.

Decay is exponential: a fixed fraction leaves per unit time, so **the fraction remaining is 2−t/t½**. The gap between a 0.8-hour and a 10-hour complex is not 12× — at the six-hour mark it is a factor of about 110.

#### How half-lives are actually measured — and why it bites later

You assemble the complex, then watch it fall apart. Three families of assay do this differently:

- — radiolabelled β2-microglobulin; the signal dies as the complex dissociates. Nearly all of the `NetMHCstabpan` training data is SPA at 37 °C, from one lab.
- **Purified fluorescence** — fluorescently labelled antibodies against the intact complex, in a tube.
- **Cellular fluorescence** — the same readout, but on complexes sitting on real cells. Closest to the biology, noisiest.

Hold onto this. In §3 it turns out these three assays do not even *rank* the same peptides the same way, and dealing with that is the preprint’s main contribution.

PART 02

## Who pays for this, and what they do with the number

The honest answer is cancer vaccines. Serova — the sponsor of your track — is a London company building a personalised cancer vaccine platform, which is the exact workflow this prediction sits inside.

### Neoantigens

A tumour accumulates mutations. Some change a protein’s amino-acid sequence. The cell shreds that mutated protein like any other and displays the fragments — and now the surface carries a peptide that exists *nowhere else in the body*. That is a , and it is close to an ideal drug target: the immune system never learned to tolerate it, so there is nothing to break.

The therapeutic move is to take a biopsy, sequence it, work out which neoantigens that patient’s own HLA alleles can present, and manufacture something that trains their immune system onto those peptides — a synthetic peptide vaccine, an mRNA vaccine, or T cells expanded against them in a dish. All three modalities appear in the preprint’s clinical cohorts.

The constraint is numerical. Sequencing turns up hundreds of mutations. Only a slice produce peptides that bind the patient’s alleles at all. And a vaccine can only carry a handful. So the whole enterprise reduces to a ranking problem over a shortlist, run once per patient, where you get one shot.

**Affinity gates; stability ranks inside the gate.** That sentence is the preprint’s central empirical claim and it falls straight out of this picture: by the last step everything left is already a good binder, so affinity has stopped discriminating. Something else has to break the tie.

Illustrative numbers, real mechanism. The affinity percentile rank is **lower is better**; the half-life is **higher is better**. Two peptides swap across the cut, so two of the five slots in this patient’s vaccine change hands. That swap is the entire product of a stability model.

### Why stability specifically, rather than affinity

This is not a hunch. Harndahl and colleagues showed in 2012 that pMHC class I stability predicts whether a peptide triggers a T-cell response *better than affinity does*. Van der Burg had reported the same direction in 1996. More recently, stability has been shown to predict **immunodominance** — which of several competing epitopes a response actually converges on.

The mechanistic story is compensatory. A peptide can bind tightly and still dissociate quickly at the cell surface, in which case the complex is gone before anyone looks. A peptide produced at low abundance can still raise a response if each complex it forms lasts a long time. Abundance, affinity and persistence trade off against each other, and affinity is only one of the three.

### So why hasn’t this been solved

Data. Affinity measurements are cheap and have accumulated for thirty years. Stability means watching dissociation in real time — and the more stable the complex, the longer you have to watch. The result is a field with two public predictors and a training corpus two orders of magnitude smaller than the one affinity models enjoy.

Each gridline is a **tenfold** increase, so the gaps are far larger than they look. Against this, remember there are **30,511 known class I alleles** and the stability corpus covers 72 of them. Every design decision in the preprint — and every one you will make this weekend — is a response to this picture.

PART 03

## The preprint: SPEARMINT

Karthikeyan, Vincent & Rubinsteyn, UNC Chapel Hill, June 2026. “Peptide:MHC Binding Stability Prediction Using Protein Language Models.” The first protein language model applied to this task.

The paper does three separable things. It is worth holding them apart, because your team can take any one of them and leave the others.

1. **It rebuilds the benchmark.** Existing reported numbers are inflated by . The paper constructs splits where no test peptide shares ≥80% sequence identity with any training peptide — in *either* the stability data or the affinity data used for pre-training.
2. **It applies a pLM with transfer learning.** A backbone called MINT, pre-trained on affinity, then fine-tuned on stability.
3. **It conditions on the assay.** Different assays disagree about the same complex, so the model is told which assay and what temperature it is predicting for.

The resulting model is named SPEARMINT — Stability Prediction of Epitopes with Assay Recalibration using MINT.

### The backbone: why MINT and not plain ESM-2

ESM-2 is a transformer trained to fill in masked amino acids across hundreds of millions of protein sequences. To learn that, it has to internalise which residues co-vary, which positions tolerate what, and implicitly a lot of structure. The vector it produces for each residue — its — carries that knowledge, which is why you can bolt a small on top and get a working predictor from a few thousand labels.

The catch is that ESM-2 reads one chain at a time. Your problem has two chains that only matter because they touch. **MINT** (Multimeric INteraction Transformer, Ullanat et al., *Nature Communications* 2026) is ESM-2 650M rebuilt with a second attention path: at every layer, each position attends both within its own chain and across to the other chain. It was pre-trained on about 96 million protein–protein interactions from STRING.

The paper isolates this one difference cleanly: its **ESM-2 single-chain baseline is MINT with the cross-chain attention logits masked to −∞** — same backbone, same tokenisation, same head, only the cross-talk removed. Any performance gap is attributable to the mechanism and nothing else. That is the shape of a good , and worth copying.

### The three-stage curriculum

With 21,626 stability examples you cannot responsibly train 650 million parameters from scratch. The paper’s answer is to approach the target task through a sequence of progressively scarcer, progressively more specific datasets, freezing more of the model at each step.

The most quietly useful finding here: **keeping the Stage 1 head rather than re-initialising it beats re-initialising**. The output mapping learned for affinity is already a decent prior for stability. The direction of the whole curriculum — abundant to scarce, loose to frozen — is the standard answer to small-data problems and the thing to steal if you steal one idea.

### The results, with the asterisks attached

On the held-out slice of the `NetMHCstabpan` corpus (n = 2,700), filtered so no test peptide shares 80% identity with anything in training:

| Model | Spearman ρ | Pearson r | Lin’s CCC | Status |
| --- | --- | --- | --- | --- |
| NetMHCstabpan | 0.88 | — | 0.786 | trained on this data |
| **MINT Transfer** | 0.79 | 0.76 | 0.788 | best clean model |
| ESM-2 Transfer | 0.75 | 0.71 | — | no cross-chain attention |
| MINT Direct | 0.73 | 0.68 | — | no affinity pre-training |
| ESM-2 Direct | 0.57 | 0.52 | — | neither |

Read the bottom two rows against the top two and the 2×2 design pays off. **Transfer learning is worth more than the architecture**: adding the affinity pre-training stage moves ESM-2 from 0.57 to 0.75 and MINT from 0.73 to 0.79. Cross-chain attention is worth a real but smaller amount, about +0.04–0.16 depending on where you look. If you only have time for one of the two ideas, take the curriculum.

Note also that MINT Transfer’s Pearson r beats `NetMHCstabpan`, and its CCC matches it. only measures ordering; and ask whether the actual number of hours is right. The pLM is better at the harder question even where it loses the headline.

### Then everything falls over on new data

Moved onto an independent holdout assembled from IEDB — same task, different labs, different assays — **every model drops below ρ = 0.4**, including on the SPA subset they were all trained on. That is the most important result in the paper and it is a negative one.

The cause turns out to be measurement, not modelling. The same peptide–HLA pair measured by two different assays does not just give a different number — it gives a different *ordering*.

The premise is that a complex has **one real stability**, and each assay observes it through its own distortion. Rather than retrain on pooled, mutually inconsistent labels, the paper freezes the model and learns the distortions. The payoff is large and specific: on purified-fluorescence measurements ρ goes from 0.259 to **0.567**, while SPA performance is untouched.

| Model, external IEDB holdout | SPA ρ n = 217 | Purified fluorescence ρ n = 758 |
| --- | --- | --- |
| **SPEARMINT** (MINT + FiLM conditioning) | 0.359 | 0.567 |
| MINT Transfer (no conditioning) | 0.346 | 0.259 |
| TLStab | — | 0.369 |
| ESM-2 Transfer | — | 0.274 |

Cellular fluorescence stays weak for everyone (SPEARMINT ρ = 0.220) — only 105 training measurements exist for it. On 9-mers SPEARMINT reaches 0.675. Re-tested on the Stage 2 test set afterwards it drops only 0.79 → 0.76, so the recalibration did not destroy what came before.

### Does any of this reach the clinic

The last third of the paper asks whether a better half-life number improves decisions downstream. Two findings, and they point in opposite directions, which is what makes them credible.

#### On eluted ligands, affinity wins

Predicting whether a peptide was actually found on a real cell by mass spectrometry: binding affinity beats every stability model at every threshold. Stability still adds a little on top of affinity, but it is not the primary signal.

#### On immunogenicity, stability wins

Predicting whether a peptide provoked a real CD8 T-cell response: the purified-fluorescence head reaches AUROC 0.59 against 0.55–0.57 for affinity, 0.51–0.53 for `NetMHCstabpan` and 0.53 for TLStab.

The sharpest number is at the hardest operating point. Restricted to the top 2% of predicted binders — a shortlist that is already all good binders — recall at a 5% false-positive rate is **0.16 for SPEARMINT against 0.03 for `NetMHCstabpan` and 0.08 for TLStab**. And precision@10 reaches 0.90, an improvement of +0.60 over using affinity alone.

Across four real patient cohorts (two peptide vaccines, one mRNA vaccine, one T-cell therapy), SPEARMINT’s heads lead on three of four for per-patient ranking. In one cohort plain `NetMHCpan` affinity is still the single best predictor. No one assay head dominates everywhere.

What the authors say is still missing

This list is the most valuable paragraph in the paper for you, because each item is a defensible hackathon project that the field’s own experts have publicly flagged as open.

- **Uncertainty quantification** — stated outright: it “would have immediate benefit by allowing researchers to act on a particular predicted stability in accordance with its associated confidence score.”
- **No structure** — the models are sequence-only, by their own admission.
- **Eluted-ligand data was deliberately held back** from pre-training, so nobody has tried a pLM trained on affinity *and* eluted ligands *and* stability.
- **Rare alleles** — performance degrades where data is thin, and 72 of 30,511 alleles are covered.
- **Only assay type and temperature** are modelled as experimental context; study-level effects are visible but unmodelled.
- **Class II is untouched**, despite CD4 responses mattering in the neoantigen setting.

PART 04

## The benchmark trap

Your track brief says “better than narrow models such as NetMHCstabpan.” Getting that comparison right is the difference between a result and a press release.

`NetMHCstabpan`’s production model was trained by cross-validation and then re-trained on the entire dataset before release. So on the benchmark everyone quotes, it has seen every test peptide. Its 0.88 is an upper bound on its own performance, not a target.

This produces two failure modes, and they are opposite. You can build a sloppy benchmark, beat 0.88, and be wrong. Or you can build an honest benchmark, score 0.79, and look like you lost — when 0.79 on clean data is the better model.

**SLYNTVATL** is the HIV gag epitope on HLA-A\*02:01; **SLFNTVATL** is a real escape variant of it. A random split can put one in train and the other in test, and the model then scores well by near-recall. SPEARMINT clusters peptides at 80% identity by normalised Levenshtein distance and assigns whole clusters, then filters the stability test set against the affinity training set as well.

Practical implication for Saturday morning

Build the clean split **before** you build anything else, and make it the first artefact in the repo. Three rules:

1. Cluster peptides by sequence identity and assign whole clusters, never individual peptides.
2. Filter your stability test set against every dataset you pre-train on, not just the stability training set.
3. Report `NetMHCstabpan` twice — on the contaminated benchmark and on a holdout it has not seen — and say which is which on the slide.

Do this and you can make a strong, defensible claim in the pitch. Skip it and the first judge who knows the field will ask the question you cannot answer.

PART 05

## The four angles, decoded

All four sit on the same pipeline. Three are alternatives to each other; the fourth composes with any of them.

Worth settling early as a team: you are choosing **one** of angles 1–3 as the spine of the project, and then deciding whether angle 4 rides on top. Treating all four as parallel options is the mistake that produces four half-finished things by Sunday.

ANGLE 1

### Fine-tuned protein language model

Run both sequences through ESM-2 or ESM-C, squash the output into one vector, train a small head to turn that vector into a half-life.

What it actually involves

1. **Tokenise and concatenate.** The peptide (9 letters) and the HLA (either the 34-residue pseudosequence or the full \~365-residue chain) become one token sequence, with a parallel chain-ID tensor marking which positions belong to which molecule.
2. **One forward pass.** The pLM returns a vector per residue — 1,280 numbers for ESM-2 650M, 1,152 for ESM-C 600M. These encode what the model inferred about that position from everything around it.
3. **.** Drop the special and padding tokens, average the rest into one fixed-length vector.
4. **Head.** Two linear layers mapping that vector to a single number, trained with mean squared error against log(1 + t½).

Leave the backbone frozen and only the head learns — fast, and hard to overfit. Unfreeze the top layers and the backbone adapts to pMHC too — stronger, and much easier to overfit on 21k labels.

The honest problem

This is the paper’s ESM-2 baseline. It is already published, at ρ = 0.57 direct and 0.75 with transfer. Rebuilding it is reproduction, not contribution — which costs you most of the 20 points for creativity.

Four twists that are genuinely untried

- **Swap ESM-2 for ESM-C.** ESM-C 600M matches ESM-2 3B on representation benchmarks at a fraction of the compute, and the preprint never tests it. A near-one-line swap that produces a real ablation.
- **Change where you pool.** Averaging 9 peptide residues together with 365 MHC residues dilutes the peptide to about 2% of the signal. Pool the two chains separately and concatenate; or attention-pool; or pool only the peptide plus the 34 groove-lining positions. This is the most obviously promising idea the paper leaves on the table.
- **Pseudosequence versus full chain, head to head.** The authors argue for the full chain but never show the direct comparison. Cheap to run, and a negative result is still a result.
- **Which layer.** The final layer is not always the best for transfer; middle layers frequently win. Free to sweep once embeddings are cached.

How to spend the time

Cache the embeddings for all 27k pairs *once*. That is the only expensive step. After it, every head variant trains in seconds, and the weekend becomes dozens of experiments rather than three. Do this on Saturday morning whichever angle you eventually pick — the baseline is your comparator either way.

**Technicality** fair **Creativity** weak without a twist **Usefulness** fair **Demo** fair **Track fit** strong

ANGLE 2

### Multi-task transfer

Learn the representation from the abundant data, then spend the scarce stability labels only on the final adjustment.

What it actually involves

Three kinds of label exist, at wildly different scales: (around a million, binary), binding affinity (170k, continuous), and stability (27k, continuous). Two ways to exploit that:

- **Sequential curriculum** — train fully on task A, carry the weights over, train on task B. What SPEARMINT does, affinity then stability.
- **Joint multi-task** — one shared trunk with several output heads, trained on all tasks at once with a weighted combined loss. Needs the loss weights tuned, because the tasks differ in scale and in how much data each has.

Either way the scarce task borrows representation quality it could never afford to learn itself.

Why this is the strongest claim available

SPEARMINT *deliberately* excluded eluted-ligand data from pre-training. Their reasoning is sound — they wanted eluted ligands as a clean evaluation endpoint, and a composite readout like that blurs the line between kinetic stability and overall presentation likelihood. TLStab does use eluted ligands, but with a small `NetMHC`-style network rather than a pLM.

So: **a protein language model pre-trained on eluted ligands *and* affinity *and* then fine-tuned on stability is a combination nobody in this paper’s own comparison has run.** It attacks the stated root cause — data scarcity — directly, and it is on the authors’ own list of what is missing. That is about as close to a free, defensible novelty claim as a hackathon ever offers.

A second open question you could answer in the same breath: does the sequential curriculum beat joint multi-task here? Nobody has compared them in this setting.

Where the time actually goes

Data engineering, and more of it than you expect. Allele names have to be normalised to two-field nomenclature and mapped to full sequences through IPD-IMGT/HLA. Eluted-ligand data is about 5.4% positive, so imbalance needs handling. And the decontamination has to run across *all three* datasets, not just the stability one. Budget most of Saturday for the pipeline and treat it as a deliverable in its own right — a clean, leakage-audited multi-task dataset is itself a contribution worth showing.

The catch

Once eluted ligands are in training you can no longer use them as a clean evaluation endpoint. Decide up front what you will evaluate on instead, and say so before a judge asks.

**Technicality** strong **Creativity** strong **Usefulness** strong **Demo** fair **Track fit** strong

ANGLE 3

### Structure-aware

Predict the 3D complex, measure the interface, and let those measurements predict the half-life.

What it actually involves

1. **the complex.** Hand a model the peptide sequence, the HLA heavy chain and β2-microglobulin, and it returns 3D coordinates for the assembled thing. Boltz-2, AlphaFold3 and Chai-1 all do this.
2. **Featurise the interface.** From those coordinates, compute numbers: buried surface area between peptide and groove, hydrogen bonds and salt bridges across the interface, how deeply the P2 and PΩ side chains sit in their pockets, per-residue contact counts, and the model’s own confidence ( per residue, interface pTM for the complex).
3. **Regress on them.** Those numbers go into a head, on their own or concatenated with sequence embeddings.

The alternative framing in your brief — using Boltz-2’s affinity head — needs a warning first.

Check this before 10am Saturday

Boltz-2’s affinity head was trained on **protein–small molecule** data: SMILES ligands, pIC50 labels. A 9-mer peptide is a polymer chain, not a small molecule. Before you build a plan on “run the affinity head zero-shot on pMHC”, confirm in the first hour that it accepts a peptide chain as the ligand at all and returns something non-degenerate. Finding out at 2pm on Sunday is how a project dies.

Also check which Boltz you are getting. The original Boltz-2 is open under MIT; as of mid-2026 the newer Boltz 2.1 is closed-source and API-only, with its own rate limits. The co-folding half works on peptides regardless — it is specifically the affinity output that was built for a different kind of ligand.

The throughput problem, and the way around it

Co-folding takes seconds to minutes per complex on a GPU. Twenty-seven thousand complexes will not happen in thirty hours. You have to subsample — so subsample *properly*: stratify by allele and by half-life decile, take 1,000 to 3,000, and put the sample size on the slide. A well-chosen stratified subset with visible error bars is better science than a convenience sample twice the size.

The shortcut worth knowing: MHC class I structures are extraordinarily conserved. The groove barely moves between related alleles, and the PDB holds hundreds of solved pMHC structures. Folding once per allele and threading peptides onto that frame — or starting from a crystal structure — can turn an impossible compute budget into a feasible one. Decide which of these you are doing before you queue a single job.

Why it scores

The paper’s own limitations say its models are sequence-based and encode no structure. Doing the thing the authors said they did not do is the cleanest novelty story available, and predicted structures are the one thing in this whole problem that looks like something on a screen. For a 90-second demo slot, that matters more than it should.

**Technicality** strong **Creativity** strongest **Usefulness** fair **Demo** strong **Track fit** strong

ANGLE 4

### Uncertainty-aware ranking

Output a range instead of a number, prove the range is honest, and let a clinician act on it.

The methods, cheapest first

- **Ensembles.** Train five heads with different seeds; the spread of their predictions is your uncertainty. Almost free once embeddings are cached, and it works.
- **MC dropout.** Leave dropout switched on at inference, run N passes, take the variance. A one-line change.
- **Quantile regression.** Train the head to emit the 10th, 50th and 90th percentile directly, using the pinball loss. An interval falls straight out.
- **Gaussian likelihood head.** Output a mean and a σ, train with the Gaussian negative log-likelihood, and the model learns its own error bar.
- **.** Post-hoc, model-agnostic, distribution-free. A held-out calibration set turns any score into an interval with a *guaranteed* coverage rate. The easiest to bolt onto something that already works, and the only one that comes with a hard guarantee.

What “calibrated” means, and how you prove it on a slide

A model is when its stated confidence matches what happens. For a regression you check **coverage**: of all the 90% intervals you produced, do 90% of them actually contain the true half-life? Plot nominal coverage against observed coverage and the diagonal is perfect. Over-confidence shows up as a curve sagging below the line. One chart, instantly legible, no immunology required to read it.

Why this hits usefulness harder than anything else on the list

- **It changes what someone does.** The clinical act is picking fifteen peptides for one patient. A ranked list with no confidence is a guess. A ranked list that says “these five are solid, these ten are close to coin-flips, send them to the assay” changes Monday morning.
- **Out-of-distribution detection comes free.** The paper’s most damning finding is that performance collapses on unfamiliar assays and thin alleles. A well-calibrated model should *announce* that rather than confidently predicting nonsense. Showing that your uncertainty rises exactly where the paper shows accuracy falling is a genuinely strong slide — and it is a result you can get from their published numbers.
- **The authors asked for it in print.** Uncertainty quantification is named in the future-work paragraph as something that would have immediate benefit.

The catch, and the demo

It is a wrapper, not a model. Alone the pitch is thin, so it has to ride on one of the first three. That is also exactly why it is the cheapest differentiator you can buy.

Demo worth building: one patient’s candidate list, each peptide with an interval, and a slider for how much risk the clinician will accept — which reorders the shortlist live. One screen, ninety seconds, legible to a judge who has never heard of an HLA.

**Technicality** fair **Creativity** fair **Usefulness** strongest **Demo** strongest **Track fit** strong

### Putting it together

The judging is 100 points across five equal criteria: technicality, creativity, usefulness, demo, and track / sponsor alignment. That last one is worth reading literally. **Serova builds a personalised cancer vaccine platform.** Frame every result as per-patient ranking rather than as a correlation coefficient — precision@5 and precision@10 on one patient’s candidate list is the number they will care about, and it is the number the preprint’s final section reports too.

| Angle | What it buys you | What it costs | Fails if… |
| --- | --- | --- | --- |
| **1 · Fine-tuned pLM** | A working baseline in a few hours, and a comparator for everything else | Mostly GPU time for one embedding pass | You ship it without a twist and it reads as reproduction |
| **2 · Multi-task transfer** | The strongest defensible scientific claim; attacks the stated root cause | Most of Saturday on the data pipeline | The pipeline eats the weekend and nothing trains |
| **3 · Structure-aware** | Highest creativity ceiling; the only option that looks like something | Co-folding throughput; forces a subsample | The affinity head does not take peptides and you find out late |
| **4 · Uncertainty** | Usefulness and demo, cheaply; the authors requested it | Hours, not days — especially with ensembles or conformal | You ship it alone, with no model underneath |

A plan, if it helps

1. **Hours 0–4, everyone.** Clean leakage-controlled splits, cached embeddings, and the Angle 1 baseline. You need all three regardless of what you choose, and having them removes most of the risk from the rest of the weekend.
2. **Then pick one spine.** Angle 2 if you want the strongest science. Angle 3 if someone on the team has co-folded something before — and only then.
3. **Wrap it in Angle 4 either way.** It is cheap, the authors asked for it, and it is the only one of the four that scores on usefulness the way a clinician reads the word.
4. **Reserve Sunday morning for the demo.** Round one gives you 90 seconds of live demo and 2 minutes of Q&A. A working interface beats a better model you cannot show.

#### Two free prizes sitting next to your track

**Best use of Devin** is described as: pick a published paper, have Devin reproduce a key result, then push past it. That is a word-for-word description of what this track already is. Reproducing SPEARMINT’s ESM-2 baseline and then beating it is your Angle 1 morning anyway — you would be entering for free.

**Best use of Modal** is open to any team using Modal for anything: training, inference, batch jobs. If you are co-folding thousands of complexes or sweeping heads, you want parallel GPU workers regardless. Same work, second entry.

### Worth settling in the first thirty minutes

- Which single angle is the spine, and who owns the data pipeline. These two decisions determine everything else.
- Full HLA chain or 34-residue pseudosequence — and are you testing both, or just picking one?
- What exactly are you claiming to beat? Name the benchmark and the comparator now, in the README, before any numbers exist.
- What is the demo? Describe the final screen out loud. If nobody can, you do not have one yet.
- Who has actually used a GPU cluster this week, and what are the real limits on the API keys the organisers hand out?

REFERENCE

## Jargon decoder

Everything in this document that could reasonably stop you mid-sentence, in plain language.

### Immunology

Allele

One of the alternative versions of a gene present in the population. Your `HLA-A` gene is one of 9,175 known versions; you carry two. Written as `HLA-A*02:01` — gene, then protein family, then specific variant.

9-mer

A peptide of nine amino acids. Class I grooves are closed at both ends, so they take 8–11 residues and overwhelmingly 9. Class II grooves are open-ended and take 13–25, which makes it a separate modelling problem.

Anchor residue

The peptide positions whose side chains drop into deep pockets in the groove floor — usually position 2 and the final position, written PΩ. They do most of the binding. The residues between them arch upward where a T-cell receptor can reach them.

Pseudosequence

The 34 amino acids that line the binding groove and differ between alleles. The `NetMHCpan` family feeds the model only these rather than the full \~365-residue chain. Compact, and it generalises to unseen alleles — but it throws away everything outside the groove.

Epitope

The exact thing a receptor recognises. For a T cell that means the peptide plus the surrounding HLA surface it touches, not the peptide alone.

Neoantigen

A peptide that exists only in tumour cells, because a mutation changed the protein it came from. The immune system never learned to tolerate it, which makes it an unusually clean drug target.

Immunogenicity

Whether a peptide actually provokes a T-cell response in a real person. The endpoint that binding, presentation and stability are all proxies for.

Eluted ligand

A peptide physically stripped off the surface of real cells and identified by mass spectrometry. Evidence that something *was* presented, rather than that it *could* bind. Abundant, but a composite readout — it folds together processing, binding, stability, abundance and how detectable the peptide is.

### Biophysics and measurement

Kd — affinity

The equilibrium dissociation constant, koff / kon. Lower means tighter binding. Because it is a ratio, two peptides can share a Kd and behave completely differently once bound.

koff

The dissociation rate constant — how often per second a bound peptide lets go. Stability depends on this one rate alone.

Half-life (t½)

How long until half the complexes in a population have fallen apart. t½ = ln2 / koff. For pMHC it is measured in hours and is the quantity every model in this document is trying to predict.

Scintillation proximity assay (SPA)

Radiolabelled β2-microglobulin; the signal decays as the complex dissociates. Nearly all of the `NetMHCstabpan` training data is SPA at 37 °C from a single lab, which is why models trained on it generalise poorly to anything else.

### Machine learning

Embedding

The vector of numbers a language model produces for each residue, summarising what it inferred about that position from the whole sequence. Typically 1,152–1,280 numbers per residue. All the model’s knowledge of chemistry and structure is encoded in there somewhere.

Pooling

Collapsing a variable-length set of per-residue vectors into one fixed-length vector, usually by averaging. Mean-pooling a 9-residue peptide together with a 365-residue chain dilutes the peptide to about 2% of the signal — which is why *where* you pool is a real design decision.

Head

A small network bolted onto a large pre-trained model, trained to turn its embedding into the quantity you want. Here, two linear layers mapping a 1,280-vector to one half-life.

Fine-tuning vs probing

Fine-tuning continues training the pre-trained model’s own weights on your task. Probing freezes the model and trains only the head. Fine-tuning has more capacity and overfits far faster on small data.

Freezing

Holding some layers’ weights fixed during training. SPEARMINT freezes 50–90% of its 33 layers when fine-tuning on stability, on the reasoning that 21,626 labels cannot safely move 650 million parameters.

Transfer / curriculum learning

Training on an abundant related task first and carrying the weights over to the scarce target task. Curriculum is the sequential version; multi-task trains on everything at once with separate heads and a weighted loss.

FiLM

Feature-wise Linear Modulation. A small module that scales and shifts a frozen model’s internal features based on side information — here, which assay and what temperature. It changes how the answer is read out without retraining anything underneath.

Ablation

Deliberately removing one component to measure what it contributed. SPEARMINT’s cross-attention mask is a textbook one: identical model, cross-chain attention switched off, so the gap is attributable to that and nothing else.

Co-folding

Handing a model the sequences of two or more chains and having it predict the structure of the assembled complex, rather than each piece separately. AlphaFold3, Boltz-2, Chai-1.

Zero-shot

Using a trained model on a task it was never trained for, without any further training.

pLDDT

A structure predictor’s own per-residue confidence, 0–100. Useful both as a feature and as a filter for throwing out predictions you should not trust.

### Evaluation

Data leakage

When something in the test set also appeared in training, so the model recognises rather than predicts. Two flavours matter here: near-identical peptides on opposite sides of the split, and peptides from the affinity pre-training set reappearing in the stability test set.

Spearman ρ

Rank correlation. Measures only whether you ordered things correctly, not whether the numbers are right. 1.0 is a perfect ordering, 0 is random. The primary metric here precisely because assays disagree about absolute values.

Pearson r

Straight linear correlation between predicted and measured values. Unlike Spearman it punishes you for getting magnitudes wrong, not just ordering.

Lin’s CCC

Concordance correlation coefficient — correlation *and* agreement. It penalises a prediction that tracks the truth perfectly but sits systematically too high or too low. Pearson would not notice that; CCC does.

AUROC

The probability that a randomly chosen positive scores above a randomly chosen negative. 0.5 is a coin flip. Largely insensitive to class imbalance, which is why AUPRC usually sits next to it.

AUPRC

Area under the precision–recall curve. The one to read when positives are rare — about 5.4% for eluted ligands here.

Calibration

A model is calibrated when its stated confidence matches reality: of the predictions it calls 90% likely, 90% come true. Entirely separate from accuracy — a model can be accurate and badly overconfident at the same time.

Conformal prediction

A post-hoc wrapper that converts any model’s output into an interval with a guaranteed coverage rate, using a held-out calibration set. Distribution-free: it assumes nothing about the shape of the errors.

---

Sources: Karthikeyan, Vincent & Rubinsteyn, *Peptide:MHC Binding Stability Prediction Using Protein Language Models*, bioRxiv 2026 · Rasmussen et al., *J Immunol* 2016 (NetMHCstabpan) · Fasoulis et al., *ImmunoInformatics* 2024 (TLStab) · Ullanat et al., *Nat Commun* 2026 (MINT) · Reynisson et al., *NAR* 2020 (NetMHCpan-4.1) · Harndahl et al., *Eur J Immunol* 2012 · IPD-IMGT/HLA release 3.64, April 2026 · Wohlwend et al., Boltz-2, 2025 · EvolutionaryScale, ESM Cambrian.