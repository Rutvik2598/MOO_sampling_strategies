"""
Literature analysis - combines MOOT citations + targeted paper lookups.
Uses DOI-based lookups (better rate limits) for known key papers.
"""

import requests
import time
import json
import csv
import math

API_BASE = "https://api.semanticscholar.org/graph/v1/paper"
FIELDS = "title,year,citationCount,venue,authors,externalIds,openAccessPdf,url"
DELAY = 5  # seconds between individual lookups

# ============================================================
# KNOWN KEY PAPERS - from MOOT citations + domain knowledge
# Format: (DOI or S2 search title, expected_title)
# ============================================================
KNOWN_PAPERS = [
    # From MOOT cited_by.md - SE papers
    ("DOI:10.1109/TSE.2018.2790925", "Sampling as a Baseline Optimizer for SBSE"),
    ("DOI:10.1109/TSE.2014.2372785", "GALE: Geometric Active Learning for SBSE"),
    ("DOI:10.1007/s10515-017-0225-2", "Faster Discovery of Faster System Configurations with Spectral Learning"),
    ("DOI:10.1016/j.infsof.2017.10.012", "Beyond Evolutionary Algorithms for SBSE"),
    ("DOI:10.1007/s10664-022-10230-y", "VEER: Enhancing Interpretability of Model-based Optimizations"),
    ("DOI:10.1109/ACCESS.2024.3473089", "iSNEAK: Partial Ordering as Heuristics"),
    ("DOI:10.1145/3236024.3236074", "Learning to Sample Configurable Systems"),  # Jamshidi FSE 2018
    ("DOI:10.1145/3236024.3236047", "Data-Driven Search-based Software Engineering"),  # Nair MSR 2018
    ("DOI:10.1145/3106237.3106238", "Using Bad Learners to Find Good Configurations"),  # Nair FSE 2017
    ("DOI:10.1109/TSE.2024.3520249", "Accuracy Can Lie: Impact of Surrogate Model"),  # Chen TSE 2025
    ("DOI:10.1145/3660782", "Predicting Configuration Performance Sequential Meta-Learning"),  # Gong FSE 2024
    ("DOI:10.1145/3597926.3598043", "Twins or False Friends?"),  # Weber ICSE 2023
    ("DOI:10.1145/3597926.3598105", "Analyzing Impact of Workloads"),  # Muhlbauer ICSE 2023
    ("DOI:10.1145/3377811.3380367", "DeepPerf: Performance Prediction"),  # Ha ICSE 2019
    ("DOI:10.1109/TSE.2020.3041667", "Whence to Learn BEETLE"),  # Krishna TSE 2020
    ("DOI:10.1109/TSE.2020.3035865", "Sequential Model Optimization Effort Estimation"),  # Xia TSE 2020
    ("DOI:10.1016/j.infsof.2016.04.007", "Tuning for Software Analytics"),  # Fu IST 2016
    ("DOI:10.1145/3593434.3593455", "Learning from Very Little Data"),  # Lustosa TOSEM 2024
    ("DOI:10.1007/s10664-017-9529-0", "Data-efficient Performance Learning"),  # Guo EMSE 2018

    # MOO Quality Indicators / Pareto in SE
    ("DOI:10.1145/2884781.2884880", "Practical Guide Quality Indicators Pareto SBSE"),  # Wang ICSE 2016
    ("DOI:10.1145/2025113.2025164", "On the value of user preferences SBSE"),  # Sayyad ICSE 2013

    # Seminal MOO papers commonly cited in SE
    ("DOI:10.1109/4235.996017", "NSGA-II"),  # Deb 2002
    ("DOI:10.1109/TEVC.2007.892759", "MOEA/D Decomposition"),  # Zhang 2007
    ("DOI:10.1162/evco.2004.12.4.501", "Indicator-Based Selection in Multiobjective"),  # Zitzler 2004

    # Surrogate-assisted MOO (non-SE but highly cited)
    ("DOI:10.1016/j.ins.2020.02.069", "Adaptive Bayesian surrogate-assisted evolutionary MOO"),
    ("DOI:10.1109/JAS.2019.1911480", "Surrogate-assisted PSO Pareto active learning expensive MOO"),
    ("DOI:10.1109/TEVC.2005.861417", "Evolutionary Optimization Computationally Expensive Surrogate"),

    # Active Learning for MOO
    ("DOI:10.5555/3042817.3043tried", "Active Learning for MOO ICML 2013"),  # Zuluaga

    # Sample-efficient / data-driven optimization
    ("DOI:10.1145/3338906.3340443", "Finding Pareto-optimal configurations"),
    ("DOI:10.1145/3510003.3510152", "Transfer learning SPL configuration"),

    # Recent key papers on MOO in SE
    ("DOI:10.1145/3551349.3556963", "Multi-objective software testing"),
    ("DOI:10.1109/TSE.2022.3190766", "Surrogate software configuration"),
]

# Additional papers to look up by title search (backup)
TITLE_SEARCHES = [
    "FLASH multi-objective optimization software configuration Nair",
    "SWAY sampling Menzies multi-objective optimization",
    "OIL multi-objective optimization software engineering",
    "iSNEAK partial ordering heuristics software engineering",
    "RIOT stochastic cloud workflow",
    "surrogate assisted software configuration tuning Pareto",
    "Less noise more signal DRR optimization",
    "BINGO simple optimizers few buckets",
]


def lookup_paper(paper_id):
    """Look up a single paper by DOI or paper ID."""
    try:
        url = f"{API_BASE}/{paper_id}"
        resp = requests.get(url, params={"fields": FIELDS}, timeout=30)
        if resp.status_code == 429:
            print(f"    Rate limited, waiting 30s...")
            time.sleep(30)
            resp = requests.get(url, params={"fields": FIELDS}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
        else:
            print(f"    Error {resp.status_code}")
            return None
    except Exception as e:
        print(f"    Exception: {e}")
        return None


def search_one(query, limit=10):
    """Search for papers by title."""
    try:
        resp = requests.get(
            f"{API_BASE}/search",
            params={"query": query, "fields": FIELDS, "limit": limit, "year": "2015-2026"},
            timeout=30
        )
        if resp.status_code == 429:
            print(f"    Rate limited, waiting 60s...")
            time.sleep(60)
            resp = requests.get(
                f"{API_BASE}/search",
                params={"query": query, "fields": FIELDS, "limit": limit, "year": "2015-2026"},
                timeout=30
            )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [])
        return []
    except:
        return []


def find_knee(citations_sorted):
    """Find knee in sorted citation list."""
    n = len(citations_sorted)
    if n < 3:
        return 0, citations_sorted[0] if citations_sorted else 0

    x1, y1 = 0, citations_sorted[0]
    x2, y2 = n-1, citations_sorted[-1]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    denom = math.sqrt(a**2 + b**2)

    max_dist = 0
    knee_idx = 0
    for i in range(n):
        dist = abs(a * i + b * citations_sorted[i] + c) / denom
        if dist > max_dist:
            max_dist = dist
            knee_idx = i

    return knee_idx, citations_sorted[knee_idx]


def main():
    all_papers = {}  # paperId -> paper dict

    # Phase 1: Look up known papers by DOI
    print("=" * 60)
    print("Phase 1: Looking up known key papers by DOI")
    print("=" * 60)
    for i, (doi, desc) in enumerate(KNOWN_PAPERS):
        print(f"  [{i+1}/{len(KNOWN_PAPERS)}] {desc[:50]}...")
        paper = lookup_paper(doi)
        if paper and paper.get("paperId"):
            pid = paper["paperId"]
            all_papers[pid] = paper
            cites = paper.get("citationCount", 0)
            print(f"    -> Found: {paper.get('title', '')[:60]} [{cites} cites]")
        else:
            print(f"    -> Not found")
        time.sleep(DELAY)

    print(f"\nCollected {len(all_papers)} papers from DOI lookups")

    # Phase 2: Title searches
    print(f"\n{'='*60}")
    print("Phase 2: Title-based searches")
    print(f"{'='*60}")
    for i, query in enumerate(TITLE_SEARCHES):
        print(f"  [{i+1}/{len(TITLE_SEARCHES)}] {query[:50]}...")
        results = search_one(query, limit=5)
        new = 0
        for p in results:
            pid = p.get("paperId")
            if pid and pid not in all_papers:
                all_papers[pid] = p
                new += 1
        print(f"    -> {len(results)} results, {new} new")
        time.sleep(DELAY)

    # Phase 3: Get citations for known seminal papers by reference chain
    print(f"\n{'='*60}")
    print("Phase 3: Expanding via citing papers of key works")
    print(f"{'='*60}")
    key_expand = [
        "7bc66121b85df0788061f35ee2f0b67a54b2e55a",  # Sampling as Baseline
        "fe1906c9cb5d5f5d73b9fe99b54b01ca483c1d78",  # GALE
    ]
    for pid in key_expand:
        if pid in all_papers:
            title = all_papers[pid].get("title", "")[:40]
            print(f"  Getting citations of: {title}...")
            try:
                resp = requests.get(
                    f"{API_BASE}/{pid}/citations",
                    params={"fields": "title,year,citationCount,venue,authors", "limit": 50},
                    timeout=30
                )
                if resp.status_code == 200:
                    citing = resp.json().get("data", [])
                    new = 0
                    for item in citing:
                        cp = item.get("citingPaper", {})
                        cpid = cp.get("paperId")
                        if cpid and cpid not in all_papers:
                            all_papers[cpid] = cp
                            new += 1
                    print(f"    -> {len(citing)} citations, {new} new")
                else:
                    print(f"    -> Error {resp.status_code}")
            except Exception as e:
                print(f"    -> Exception: {e}")
            time.sleep(DELAY)

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*60}")
    print(f"TOTAL UNIQUE PAPERS: {len(all_papers)}")
    print(f"{'='*60}")

    # Filter to 2015+ with citations
    filtered = []
    for pid, p in all_papers.items():
        cites = p.get("citationCount") or 0
        year = p.get("year") or 0
        title = p.get("title", "")
        venue = p.get("venue", "")
        if title and year >= 2015:
            authors_list = p.get("authors") or []
            authors_str = ", ".join([a.get("name", "") for a in authors_list[:3]])
            if len(authors_list) > 3:
                authors_str += " et al."
            has_pdf = bool(p.get("openAccessPdf"))
            filtered.append({
                "paperId": pid,
                "title": title,
                "year": year,
                "citations": cites,
                "venue": venue,
                "authors": authors_str,
                "has_pdf": has_pdf,
            })

    filtered.sort(key=lambda x: x["citations"], reverse=True)
    print(f"Papers from 2015+ (sorted by citations): {len(filtered)}")

    # Top 100
    top100 = filtered[:100]

    # Find knee
    cites_list = [p["citations"] for p in top100]
    knee_idx, knee_cites = find_knee(cites_list)
    print(f"\nKnee point: rank #{knee_idx+1} with {knee_cites} citations")
    above = top100[:knee_idx+1]
    print(f"Papers above knee: {len(above)}")

    # Print above-knee papers
    print(f"\n{'='*60}")
    print(f"ABOVE-KNEE PAPERS ({len(above)} papers, >= {knee_cites} cites)")
    print(f"{'='*60}")
    for i, p in enumerate(above):
        marker = " <<< KNEE" if i == knee_idx else ""
        print(f"  {i+1:3d}. [{p['citations']:4d} cites] ({p['year']}) {p['title'][:75]}{marker}")
        print(f"       {p['venue'][:55]}  |  {p['authors'][:50]}")

    # Categorize
    def categorize(venue):
        v = venue.lower()
        se_kw = ["software engineering", "icse", "fse", "ase", "tosem", "tse", "ist ",
                  "information and software technology", "empirical software", "mining software",
                  "journal of systems and software", "ieee software", "saner"]
        opt_kw = ["evolutionary computation", "evolutionary", "gecco", "genetic", "swarm",
                  "optimization", "operations research"]
        ml_kw = ["machine learning", "neural", "icml", "neurips", "iclr", "aaai", "artificial intelligence",
                 "pattern recognition", "data mining", "kdd"]
        for kw in se_kw:
            if kw in v: return "SE"
        for kw in opt_kw:
            if kw in v: return "Optimization/EC"
        for kw in ml_kw:
            if kw in v: return "ML/AI"
        return "Other"

    # Group above-knee papers
    groups = {}
    for p in above:
        cat = categorize(p["venue"])
        groups.setdefault(cat, []).append(p)

    print(f"\nVenue categories (above knee):")
    for cat, papers in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(papers)}")
        for p in papers[:5]:
            print(f"    - [{p['citations']}] {p['title'][:65]} ({p['year']})")

    # THEMATIC GROUPING of above-knee papers
    print(f"\n{'='*60}")
    print("THEMATIC GROUPING (for Venn diagram)")
    print(f"{'='*60}")
    themes = {
        "Surrogate/Model-based MOO": [],
        "Sampling Strategy / DoE": [],
        "Software Configuration": [],
        "Active Learning / Adaptive": [],
        "Pareto Quality / Metrics": [],
    }
    for p in above:
        t = p["title"].lower()
        v = p["venue"].lower()
        if any(kw in t for kw in ["surrogate", "model-based", "meta-model", "prediction", "learner"]):
            themes["Surrogate/Model-based MOO"].append(p)
        if any(kw in t for kw in ["sampling", "sample", "latin", "sobol", "design of experiment", "lhs"]):
            themes["Sampling Strategy / DoE"].append(p)
        if any(kw in t for kw in ["configur", "tuning", "performance", "software"]):
            themes["Software Configuration"].append(p)
        if any(kw in t for kw in ["active learn", "adaptive", "sequential", "bayesian"]):
            themes["Active Learning / Adaptive"].append(p)
        if any(kw in t for kw in ["pareto", "indicator", "quality", "hypervolume", "dominated"]):
            themes["Pareto Quality / Metrics"].append(p)

    for theme, papers in themes.items():
        print(f"\n  {theme}: {len(papers)}")
        for p in papers[:5]:
            print(f"    [{p['citations']}] {p['title'][:65]} ({p['year']})")

    # Save outputs
    with open("literature_top100.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank","title","authors","year","citations","venue","has_pdf","paperId"])
        w.writeheader()
        for i, p in enumerate(top100):
            p["rank"] = i+1
            w.writerow(p)

    with open("literature_above_knee.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank","title","authors","year","citations","venue","has_pdf","paperId"])
        w.writeheader()
        for i, p in enumerate(above):
            p["rank"] = i+1
            w.writerow(p)

    with open("literature_all.json", "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    # Citation distribution
    print(f"\n{'='*60}")
    print("CITATION DISTRIBUTION (top 100)")
    print(f"{'='*60}")
    brackets = [(0,10),(10,25),(25,50),(50,100),(100,250),(250,500),(500,1000),(1000,float('inf'))]
    for lo, hi in brackets:
        count = sum(1 for p in top100 if lo <= p["citations"] < hi)
        label = f"{lo}-{hi-1}" if hi != float('inf') else f"{lo}+"
        print(f"  {label:>10s} cites: {'#'*count} ({count})")

    print(f"\nFiles saved: literature_top100.csv, literature_above_knee.csv, literature_all.json")
    print("Done!")


if __name__ == "__main__":
    main()
