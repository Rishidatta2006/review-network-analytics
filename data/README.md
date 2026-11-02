# Synthetic Fake-Review-Network Dataset

Dataset for the project **"Detection of Coordinated Fake Product Review Networks Using Spectral Clustering."**

> The ground-truth labels describe synthetic benchmark behavior; they are **not** real-world fraud labels. The generator described in the original dataset notes is not included in this repository, so the data-generation process cannot be rerun here.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m src.preprocessing --data-dir data --output-dir outputs
python -m src.similarity_graph --input-dir outputs --output-dir outputs/graph
python -m src.spectral_clustering --graph-dir outputs/graph --features-dir outputs --output-dir outputs/clustering
```

Run these commands from the repository root. See the root README for environment setup, dashboard launch, and tests.

## Files

| File | Purpose |
|---|---|
| `reviews.csv` | 6,173 review rows; one row per review |
| `ground_truth.csv` | Labels for 1,000 reviewers; evaluation only |
| `products.csv` | 100-product catalogue |
| `generation_summary.json` | Dataset metadata and category/campaign summary |
| `validation_report.txt` | Dataset validation summary |
| `sanity_check_spectral.py` | Optional spectral-clustering feasibility script |

`reviews.csv` spans March 1 through August 31, 2026 and contains 1,000 distinct reviewer IDs, 100 product IDs, and six product categories. The supplied IDs are synthetic-style identifiers; the CSV schema contains no name, email, phone, or address fields. Review text was also scanned for email, URL, and phone-like patterns with no matches.

## Schema

**reviews.csv**: `review_id` (REV000001…), `reviewer_id` (R0001…), `product_id` (P0001…), `rating` (int 1–5), `timestamp` (`YYYY-MM-DD HH:MM:SS`, 2026-03-01 → 2026-08-31), `review_text`, `verified_purchase` (True/False), `product_category` (6 categories), `campaign_id` (`C_POS_01`…`C_NEG_08`, empty when not a campaign review).

**ground_truth.csv**: `reviewer_id`, `ground_truth_category` (`normal` | `positive_manipulation` | `negative_manipulation` | `ambiguous`), `campaign_id`, `campaign_direction` (`positive` | `negative` | `none`), `is_coordinated` (True/False).

**Missing values:** only `campaign_id` is ever empty (in both files), meaning "no campaign". No other column has nulls.

### Label conventions
- `normal`: no campaign, direction `none`, `is_coordinated=False`.
- `positive_manipulation` / `negative_manipulation`: member of exactly one `C_POS_xx` / `C_NEG_xx` campaign, matching direction, `is_coordinated=True`.
- `ambiguous`: **never** a campaign member, so `campaign_id` is null, direction `none`, `is_coordinated=False`. They are produced by separate mimic processes (below). Their reviews carry no `campaign_id`.
- Reviews with a `campaign_id` are exactly the campaign-injection reviews of coordinated reviewers. Their camouflage reviews have `campaign_id` empty, so `campaign_id` in `reviews.csv` is also a label. **Do not use it as a feature.**

## Generation methodology

The following methodology notes were retained with the dataset metadata, but the generator itself is absent. Treat these notes as descriptive rather than as a reproducible generation procedure.

1. **Products (100):** 6 categories; hidden latent quality (mean rating), heavy-tailed popularity, and for ~20% of products an organic "launch buzz" day.
2. **Reviewers (1,000):** group sizes start at 60/15/15/10% with random jitter. IDs are assigned to groups in random order. Campaign sizes vary (minimum 8).
3. **Normal:** heavy-tailed review counts (1 + negative binomial), category preferences, personal rating bias, own verified-purchase rate, active window, verbosity. Ratings = product quality + bias + noise, plus a 7% mixture of extreme ratings. They share popular products and also cluster around sale events and launch buzz.
4. **Campaigns (8 positive + 8 negative):** each has 3–6 target products (positive campaigns favour weaker products, negative favour high-quality ones; some products are hit by several campaigns), a main activity window of 5–16 days (40% have a second burst), a preferred hour of day, a rating profile blended with the natural distribution, a lower verified-purchase rate, and shared wording themes with paraphrase variants. Members review a random subset (at least 2) of targets with timing noise and personal lag, draw ratings from a member-specific Dirichlet, and also write organic *camouflage* reviews on other products.
5. **Ambiguous:** `bandwagon` (2–3 reviews on a campaign's targets near its window, leaning in its direction, plus ordinary reviews), `polarized` (many 1/5-star reviews on diverse products), `burst` (many reviews within 2–4 days), `low_verified` (mostly unverified, one favoured category).
6. **Text:** template-based, sentiment follows rating, length varies. All review texts are globally unique, so campaign wording overlaps only through shared themes.

## Dataset characteristics reported by the supplied notes

The following generator-specific figures were included in the original dataset notes. Because the generator is absent, they have not been independently regenerated as part of this repository review.

| Item | Value |
|---|---|
| Reviews / reviewers / products | 6,173 / 1,000 / 100 (all products reviewed) |
| Reviews per reviewer | mean 6.17, median 6, range 1–24 |
| normal | 625 (62.5%) |
| positive_manipulation | 141 (14.1%) |
| negative_manipulation | 138 (13.8%) |
| ambiguous | 96 (9.6%), subtypes: bandwagon 44, polarized 18, burst 24, low_verified 10 |
| Campaigns | 8 positive (11–30 reviewers), 8 negative (9–37 reviewers) |
| Rating mix | normal 4.8% 1★ / 31.6% 5★ (mean 3.73); positive campaigns 83% 4–5★; negative campaigns 70% 1–2★ |
| Product-set overlap (mean Jaccard) | normal pairs 0.040; within a campaign ≈ 0.29–0.30; positive vs negative campaigns 0.056 |
| Timing | campaign reviews of the same target product: median std 1.8 days |
| Max single-feature AUC | 0.834 (no aggregate feature gives the answer away) |

Full output, including per-campaign statistics, is in `data/validation_report.txt`. **All 20 checks passed.** The checks cover:
- reviewer, review, rating, timestamp and product integrity;
- campaign consistency and label conventions;
- missing values and duplicates;
- campaign sizes;
- extra checks on dataset size, normal 1★/5★ ratings, both manipulation directions, unique text within campaigns, and single-feature separability.

## Why it suits Spectral Clustering
- **Graph structure:** campaign members share target products and time windows, so cosine similarity on product/time/rating-deviation features gives denser within-campaign edges (Jaccard ≈ 0.3 vs 0.04), while cross-group edges exist as well.
- **Both directions:** positive and negative campaigns differ in rating deviation but share the same behavioural mechanics. Low or high ratings alone do not identify members (camouflage reviews, blended profiles, polarized genuine users).
- **Hard cases:** ambiguous reviewers and organic buzz days create incidental similarity, so errors are expected.
- **Enough signal per node:** about 6 reviews per reviewer, repeated products and a 184-day timeline.

**Feasibility check (not a result of your project).** `sanity_check_spectral.py` used generic aggregate + product-deviation + weekly features and got weak recovery (ARI ≈ 0.03–0.07, NMI ≈ 0.13–0.19, silhouette ≈ −0.02). A quick probe with product × polarity features and a k-NN cosine graph reached NMI ≈ 0.4–0.46 against campaign membership. The data therefore contains learnable but imperfect structure, and **feature engineering matters**. If you want an easier benchmark, raise `participation`, lower the camouflage rate or reduce ambiguous reviewers in `generate_dataset.py`.

## Evaluating clusters

The generator-tuning suggestion in the feasibility note above cannot be followed from this repository: `generate_dataset.py` is not included.

- **ARI** (chance-corrected pair agreement) and **NMI** (shared information) compare the predicted partition with the reference partition. Both are invariant to cluster numbering.
- **Silhouette** measures compactness/separation in the feature space and uses no labels.
- These metrics show agreement with *synthetic* labels. They do not prove real-world fraud detection.
- **Never assume cluster 0 = normal or cluster 1 = fraud.** Build a contingency table (`pd.crosstab(cluster, ground_truth_category)`), then either label each cluster by its majority category or use the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) when the number of clusters equals the number of categories. Also report purity per cluster. Choosing `k` > 4 (up to ~17 campaigns + background) is reasonable, since campaigns are separate communities.
- **Leakage rule:** `ground_truth.csv`, `products.csv` internals, `generation_summary.json` and `reviews.campaign_id` are for scoring only. Drop `campaign_id` from `reviews.csv` before building features.

## Limitations
- Texts are template-based and much simpler than real language, so text similarity is a weak signal.
- Campaign behaviour reflects the author's assumptions (windows, participation, camouflage); real fraud rings differ and adapt.
- The platform is idealised: one review per reviewer-product pair, no review edits, deletions, helpfulness votes or IP/device data.
- Ambiguous behaviour is only four hand-designed patterns.
- Results on this dataset do not transfer to real data without validation.
