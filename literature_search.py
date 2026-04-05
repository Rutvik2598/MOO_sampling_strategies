"""
Literature search script for MOO sampling in SE.
Queries Semantic Scholar API with proper rate limiting.
Collects top papers by citation count for analysis.
"""

import requests
import time
import json
import csv
from collections import defaultdict

API_BASE = "https://api.semanticscholar.org/graph/v1/paper"
FIELDS = "title,year,citationCount,venue,authors,externalIds,openAccessPdf"
DELAY = 3.5  # seconds between requests to respect rate limits

# Search queries targeting the research area
QUERIES = [
    # Core: surrogate-assisted MOO
    "surrogate assisted multi-objective optimization sampling",
    "surrogate multi-objective optimization pareto approximation",
    # Core: SBSE + MOO
    "search-based software engineering multi-objective optimization",
    "multi-objective optimization software configuration tuning",
    # Sampling strategies for optimization
    "sampling strategy multi-objective optimization pareto front",
    "active learning multi-objective optimization surrogate",
    "Latin hypercube sampling multi-objective optimization",
    # SE-specific MOO
    "software engineering pareto optimization sampling surrogate model",
    "multi-objective software configuration sampling pareto",
    "data-driven search-based software engineering",
    # Budget-aware / sample-efficient
    "sample efficient multi-objective optimization",
    "budget allocation multi-objective optimization surrogate",
    # Specific SE venues/topics
    "multi-objective optimization software testing pareto",
    "machine learning surrogate model software engineering optimization",
    "configurable software system performance prediction sampling",
]

def search_papers(query, limit=50, year_range="2015-2026", offset=0):
    """Search Semantic Scholar for papers matching query."""
    params = {
        "query": query,
        "fields": FIELDS,
        "limit": limit,
        "year": year_range,
        "offset": offset,
    }
    try:
        resp = requests.get(f"{API_BASE}/search", params=params, timeout=30)
        if resp.status_code == 429:
            print(f"  Rate limited, waiting 60s...")
            time.sleep(60)
            resp = requests.get(f"{API_BASE}/search", params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  Error {resp.status_code} for query: {query[:50]}")
            return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None


def collect_papers():
    """Run all queries and collect unique papers."""
    all_papers = {}  # paperId -> paper dict

    for i, query in enumerate(QUERIES):
        print(f"\n[{i+1}/{len(QUERIES)}] Searching: {query[:60]}...")
        result = search_papers(query)
        if result and "data" in result:
            count = 0
            for paper in result["data"]:
                pid = paper.get("paperId")
                if pid and pid not in all_papers:
                    all_papers[pid] = paper
                    count += 1
            print(f"  Found {len(result['data'])} results, {count} new unique papers")
            print(f"  Total unique so far: {len(all_papers)}")
        else:
            print(f"  No results")
        time.sleep(DELAY)

    return all_papers


def filter_and_rank(papers):
    """Filter to papers with citations, sort by citation count."""
    # Filter: must have citation count and be from 2015+
    filtered = []
    for pid, p in papers.items():
        cites = p.get("citationCount", 0)
        year = p.get("year", 0)
        title = p.get("title", "")
        venue = p.get("venue", "")
        if cites is not None and year and year >= 2015 and title:
            authors_str = ", ".join([a.get("name", "") for a in (p.get("authors") or [])[:3]])
            if len(p.get("authors", [])) > 3:
                authors_str += " et al."
            filtered.append({
                "paperId": pid,
                "title": title,
                "year": year,
                "citations": cites or 0,
                "venue": venue or "",
                "authors": authors_str,
                "has_pdf": bool(p.get("openAccessPdf")),
            })

    # Sort by citation count descending
    filtered.sort(key=lambda x: x["citations"], reverse=True)
    return filtered


def find_knee(papers):
    """
    Find the knee point: the paper furthest from the line connecting
    first (most cited) to last (least cited) in the top 100.
    """
    top = papers[:100]
    if len(top) < 3:
        return 0, 0

    # Points: (index, citations)
    x1, y1 = 0, top[0]["citations"]
    x2, y2 = len(top)-1, top[-1]["citations"]

    # Line from first to last: ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    denom = (a**2 + b**2) ** 0.5

    max_dist = 0
    knee_idx = 0
    for i, p in enumerate(top):
        dist = abs(a * i + b * p["citations"] + c) / denom
        if dist > max_dist:
            max_dist = dist
            knee_idx = i

    return knee_idx, top[knee_idx]["citations"]


def categorize_venues(papers):
    """Categorize papers by venue type."""
    se_venues = {
        "ICSE", "FSE", "ASE", "ESEC", "TSE", "TOSEM", "IST",
        "Information and Software Technology", "IEEE Transactions on Software Engineering",
        "ACM Transactions on Software Engineering and Methodology",
        "Empirical Software Engineering", "Journal of Systems and Software",
        "IEEE Software", "Mining Software Repositories", "SANER",
        "International Conference on Software Engineering",
        "Foundations of Software Engineering",
        "International Conference on Automated Software Engineering",
        "Software: Practice and Experience",
    }
    opt_venues = {
        "IEEE Transactions on Evolutionary Computation", "Evolutionary Computation",
        "Swarm and Evolutionary Computation", "GECCO",
        "Annual Conference on Genetic and Evolutionary Computation",
        "IEEE Congress on Evolutionary Computation",
        "European Conference on Evolutionary Computation",
    }
    ml_venues = {
        "ICML", "NeurIPS", "ICLR", "AAAI", "IJCAI",
        "International Conference on Machine Learning",
        "Journal of Machine Learning Research", "Machine Learning",
    }

    categories = defaultdict(int)
    for p in papers:
        v = p["venue"]
        matched = False
        for sv in se_venues:
            if sv.lower() in v.lower():
                categories["SE"] += 1
                matched = True
                break
        if not matched:
            for ov in opt_venues:
                if ov.lower() in v.lower():
                    categories["Optimization/EC"] += 1
                    matched = True
                    break
        if not matched:
            for mv in ml_venues:
                if mv.lower() in v.lower():
                    categories["ML/AI"] += 1
                    matched = True
                    break
        if not matched:
            categories["Other"] += 1

    return dict(categories)


def main():
    print("=" * 60)
    print("LITERATURE SEARCH: MOO Sampling Strategies in SE")
    print("=" * 60)

    # Collect papers
    all_papers = collect_papers()
    print(f"\n{'='*60}")
    print(f"Total unique papers collected: {len(all_papers)}")

    # Filter and rank
    ranked = filter_and_rank(all_papers)
    print(f"Papers from 2015+ with citations: {len(ranked)}")

    # Find knee
    knee_idx, knee_cites = find_knee(ranked)
    print(f"\nKnee point: paper #{knee_idx+1} with {knee_cites} citations")
    print(f"Papers above knee: {knee_idx+1}")

    # Save top 100 to CSV
    top100 = ranked[:100]
    csv_path = "literature_top100.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "title", "authors", "year", "citations", "venue", "has_pdf", "paperId"])
        writer.writeheader()
        for i, p in enumerate(top100):
            p["rank"] = i + 1
            writer.writerow(p)
    print(f"\nTop 100 saved to {csv_path}")

    # Above-knee papers
    above_knee = ranked[:knee_idx+1]
    print(f"\n{'='*60}")
    print(f"PAPERS ABOVE THE KNEE (top {len(above_knee)}, >= {knee_cites} cites)")
    print(f"{'='*60}")
    for i, p in enumerate(above_knee):
        marker = " <-- KNEE" if i == knee_idx else ""
        print(f"  {i+1:3d}. [{p['citations']:4d} cites] ({p['year']}) {p['title'][:80]}{marker}")
        print(f"       Venue: {p['venue'][:60]}  Authors: {p['authors'][:50]}")

    # Venue categories for above-knee
    cats = categorize_venues(above_knee)
    print(f"\nVenue distribution (above knee):")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Save above-knee details
    above_path = "literature_above_knee.csv"
    with open(above_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "title", "authors", "year", "citations", "venue", "has_pdf", "paperId"])
        writer.writeheader()
        for i, p in enumerate(above_knee):
            p["rank"] = i + 1
            writer.writerow(p)
    print(f"Above-knee papers saved to {above_path}")

    # Save full JSON for later analysis
    with open("literature_all_papers.json", "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False)
    print(f"All ranked papers saved to literature_all_papers.json")

    # Print citation distribution summary
    print(f"\n{'='*60}")
    print("CITATION DISTRIBUTION (top 100)")
    print(f"{'='*60}")
    brackets = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 500), (500, 1000), (1000, float('inf'))]
    for lo, hi in brackets:
        count = sum(1 for p in top100 if lo <= p["citations"] < hi)
        label = f"{lo}-{hi-1}" if hi != float('inf') else f"{lo}+"
        print(f"  {label:>10s} cites: {count} papers")


if __name__ == "__main__":
    main()
