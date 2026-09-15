# Related work, and where this paper sits

Drafted as the source for the paper's Section 2. Every entry was read rather
than cited from an abstract, except where marked **[abstract only]**. Two
sourcing notes carried forward:

* Cvitić et al.'s survey cites the multi-path/XOR work as *"N. K. Kutzkov and
  E. O. Kiktenko"*. That attribution is **wrong**: the paper is by Kiktenko,
  Tayduganov and Fedorov, and no author named Kutzkov appears in the record.
  Verified against PMC11675764 and arXiv:2411.07810v2. Do not propagate it.
* Luo & Li's Section 3.3 and Section 4 could not be read verbatim (MDPI returns
  403; PMC truncates). Equations 8–17 and the simulation numbers below come from
  a secondary extraction of the same PMC page and are marked **[verify]**.
  Everything else from that paper is verbatim.

---

## 1. The gap this paper addresses

QKD networks beyond a single link are built on **trusted relays**: the key is
decrypted and re-encrypted at every intermediate node, so a compromised relay
reads every key that crosses it. This is the field's acknowledged structural
weakness — Cvitić et al. open their open-problems section with *"most routing
strategies rely on trusted-node assumptions, which limit end-to-end security
guarantees"* — and the literature answers it in exactly two ways:

1. **Remove the trust structurally** — quantum repeaters, or post-quantum
   hybrids, both future work in every survey that names them.
2. **Tolerate compromise by redundancy** — split the key over multiple disjoint
   paths so no single relay sees all of it.

What the literature does **not** do is *detect that a particular relay has been
compromised, from the telemetry a QKD control plane already collects, and
respond in proportion to the evidence*. Cvitić et al.'s Table 3 classifies the
field's routing strategies into eight families — shortest-path, greedy,
residual-key, SDN-dynamic, reinforcement-learning, multi-path XOR/secret
sharing, QoS-aware, multi-tenant — and **none of the eight is a detection
method**. That absence is the gap.

One recent paper is the exception and it is treated separately below.

---

## 2. The closest prior art: Luo & Li 2025

> Yi Luo and Qiong Li, "Trust-Aware Causal Consistency Routing for Quantum Key
> Distribution Networks Against Malicious Nodes," *Entropy* 27(11):1100, 2025.
> DOI 10.3390/e27111100. Harbin Institute of Technology.

The only published work that scores relay trustworthiness from network telemetry
and responds in a graded way. It must be positioned carefully, because a
careless reading makes this paper look derivative and a careful one shows the
two are complementary.

**What they do.** A causal-consistency layer (vector clocks, witness
acknowledgements with f+1 / n−f thresholds, rotating leader) keeps every honest
node's view of the key state consistent under Byzantine control-plane
misbehaviour. On top of it, an ILP multi-commodity flow with node-splitting
routes key over p ≥ f+1 internally node-disjoint paths, where each link's usable
capacity is multiplied by a trust level **[verify]**

```
C_e = (min(W_e^A, W_e^B) − f)/(n − f) · exp(−β·((G_e^A + S_e^A) − (G_e^B + S_e^B))²)
```

**Threat model:** three canonical Byzantine behaviours — false link-state
advertisement, forwarding deviation, demand forgery — by up to 30 % colluding,
static, non-adaptive malicious nodes, under n ≥ 3f+1.

**Metrics:** demand completion ratio (a satisfied-*volume* fraction, not a count)
and key utilisation (keys consumed per delivered demand; **lower is better**,
despite the name). Reported **[verify]**: proposed DCR 0.90 / KUE 16.6 against
multi-path-without-trust 0.48 / 30.8 and OSPF ≤ 0.12 / 296.

### How this paper differs

| | Luo & Li 2025 | this paper |
|---|---|---|
| what the attacker does | lies to the control plane | **steals key material** |
| what is measured | service delivery (DCR, KUE) | **exposed key and data (`D_eff`)** |
| evidence | endpoint report disagreement | that, **plus key accounting and the quantum layer** |
| trust memory | memoryless, per round, per **edge** | EWMA, per **node** |
| response | capacity derating | derating, routing weight and isolation, under a partition guard |
| detection reported | none — no ROC, no AUC | AUC, FPR, detection delay, per feature and per profile |
| volume vs targeting | not separated | separated against a blind-throttle control |

The two threat models are disjoint: a node that lies about its buffer does not
thereby learn any key, and a node that reads the key it relays need not lie
about anything. **Profile L in this paper is the one that overlaps**, and it is
precisely where their mechanism works — which is why their trust rule is
reimplemented here as baseline **B5** rather than merely cited.

The substantive finding is that their trust rule is *narrow, not weak*: on the
liar it reaches DRR 0.44, and on the passive, greedy and tapping profiles it
correctly does nothing, because a single observable carries evidence about a
single attack. That is the argument for `x1` and `x3`, made with their method
instead of against it.

**Limitations they state:** only three canonical attack modes; the strategy
space is unbounded; adaptive, stateful adversaries are out of scope. Unstated
but relevant: ILP scalability, synthetic topologies only, no measured control
overhead, and β, λ, μ, p₀, K, S_min all unreported — which is why B5 is a port
of the mechanism, not a reproduction of their numbers.

---

## 3. Multi-path key relay — the source of B2

> E. O. Kiktenko, A. Tayduganov, A. K. Fedorov, "Routing Algorithm Within the
> Multiple Non-Overlapping Paths' Approach for Quantum Key Distribution
> Networks," *Entropy* 26(12):1102, 2024. DOI 10.3390/e26121102.

`K_ij = ⊕_{P} K^P_ij` over M internally node-disjoint paths; an adversary must
hold a relay on **every** path, so the compromise probability is bounded by
`ε^M`. **This is the scheme our B2 implements and it should be credited to
them.** What is theirs alone is the path-selection algorithm: an iterative
greedy water-filling that minimises the worst link's key-rate deficiency and
debits every link it uses, so successive demands spread across the key supply.

Two facts matter for our reimplementation (baseline **B6**) and both are worth
reporting: their algorithm enumerates *every* simple path and then every
M-subset, which is exponential — their own evaluation runs at N = 6 and N = 10
— and their deficiency is defined against an offline target-rate matrix. At
N = 50 neither is available, so B6 selects over the same min-cost-flow machinery
with the link cost replaced by a key-rate deficiency surrogate. They report no
baseline comparison at all, and state topology dependence as their main
limitation: M non-overlapping paths simply may not exist.

Related: Zhou et al. (*IEEE/ACM ToN* 30:1328, 2022) on disjoint multipath against
compromised trusted nodes; Lo, Montagna & von Willich's DSKE (arXiv 2022); Yu et
al. (*JLT* 40:3530, 2022) on collaborative routing in partially-trusted-relay
networks. **[abstract only]** for all three.

---

## 4. Key-aware routing and SDN control — the source of B7

> L. Bi, M. Miao, X. Di, "A Dynamic-Routing Algorithm Based on a Virtual Quantum
> Key Distribution Network," *Applied Sciences* 13(15):8690, 2023.
> DOI 10.3390/app13158690.

A four-layer SDN architecture (application / control / data / quantum) in which
the controller collects link state and weights each link by

```
W_ij = α·K_ij^t + β·P_ij ,   α + β = 1 ,   P = 1 − exp(−λ)
```

with `K^t` the remaining key in the link's pool and `P` a Poisson link-blocking
probability. Compared against OSPF and a residual-key-pool scheme on key
utilisation, blocking rate and delay.

Two cautions carried into **B7**. Their Eq. 12 as printed *adds* the blocking
probability and then maximises, which would prefer the more congested link; read
as a benefit score it must be the availability `1 − P` that enters, and that is
what we implement. And no numeric α, β are given, so `alpha_key` is swept.

B7 exists to answer one specific objection — *could a good key-aware routing
algorithm make a detector unnecessary?* — and it carries no detection at all.

Also in this family: Wang, Zhao & Nag (*Appl. Sci.* 9:2081, 2019) on SDN-enabled
QKD; Aguado et al. (*JOCN* 10:421, 2018) on VNF-based quantum encryption;
**Martin et al. (*npj Quantum Information* 10:80, 2024), MadQCI** — a
heterogeneous SDN-QKD network in production facilities, the strongest evidence
that the control architecture assumed here is the one being deployed;
Mehic et al. (*IEEE/ACM ToN* 28:168, 2020) on QoS in trusted-relay QKD, the
QKDNetSim lineage. **[abstract only]** for all four.

---

## 5. Deployments the parameters are grounded in

| network | scale | link lengths | reported key rate |
|---|---|---|---|
| SECOQC Vienna (Peev et al., *NJP* 11:075001, 2009) | 6 nodes | metro | ~kbit/s class |
| Tokyo QKD (Sasaki et al., *Opt. Express* 19:10387, 2011) | 6 nodes | 1–45 km | 300 kbit/s @ 45 km; 0.25 kbit/s on the entanglement link |
| Cambridge (Dynes et al., *npj QI* 5:101, 2019) | metro | metro | — |
| Hefei (Chen et al., *npj QI* 7:134, 2021) | **46 nodes** | metro | — |
| MadQCI Madrid (Martin et al., *npj QI* 10:80, 2024) | multi-site, SDN | metro | — |

These fix two modelling choices. **Scale**: a 50-node metropolitan QKD network is
not hypothetical — Hefei is 46 nodes. **Link length**: 8.7–52 km is the
deployed metro regime, which is what the corrected `L_0` puts our topologies in.

The absolute key rate spans three orders of magnitude across these systems
(0.25 kbit/s to 300 kbit/s at comparable distance), so no single value is "the"
realistic one. This is why the paper argues the point differently: every metric
reported here is dimensionless or a ratio, and `tests/test_baselines.py`
verifies numerically that scaling key supply and key demand together leaves all
of them bit-identical. Only the supply/demand ratio is a modelling choice, and
it is stated.

---

## 6. Standards

ITU-T **Y.3800** (Oct 2019), *Overview on networks supporting quantum key
distribution*, is the framework the control-plane separation used here follows.
Cvitić et al. name standardised key-management interfaces as an open problem;
ETSI's QKD group specifications are referenced generically in that literature
but no specific GS number was verified for this draft — **to be checked before
submission**.

---

## 7. Surveys to cite for positioning

* Cvitić, Peraković & Pinto, "Overview of Routing Approaches in QKD Networks,"
  arXiv:2511.15465, 2025 — 26 routing strategies, 2013–2024. **Preprint, no
  venue**; check for a published version before submitting.
* Cao et al., "The Evolution of QKD Networks: On the Road to the Qinternet,"
  *IEEE COMST* 24:839, 2022. **[abstract only]**
* Dervisevic et al., "QKD Networks — Key Management: A Survey," *ACM Computing
  Surveys* 57, 2025. **[abstract only]**
* Dervisevic, Voznak & Mehic, "Large-scale QKD network simulator," *JOCN*
  16(4):449, 2024 — the natural target for the cross-validation this paper
  leaves open. **[abstract only]**

Two numbers from Cvitić et al. are useful anchors because they are the field's
own summary of the trade-offs this paper measures directly: key-aware routing
reduces service rejection by **25–40 %** over static shortest path, and
multi-path costs **30–60 %** more key.

---

## 8. Open items before submission

1. Verify Luo & Li Eqs. 8–17 and Section 4 against the publisher PDF.
2. Read the four **[abstract only]** surveys properly, and Zhou et al. 2022 in
   full — it is the closest work on disjoint multipath against compromised
   trusted nodes and deserves more than an abstract.
3. Find the ETSI GS QKD number for key-management interfaces.
4. Check whether the Cvitić survey has been published in a venue.
5. Decide whether to cite Curty et al. 2019 and Zapatero et al. 2021 (malicious
   *devices*, a different layer from malicious *relays*) — probably one sentence
   distinguishing the two, no more.
