"""
Real-world messy user query test.
Craft queries like a pilot/analyst would type and test retrieval quality.
"""

import json
import logging
import textwrap

import numpy as np
from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 100
RRF_K = 60
W_DENSE = 0.60
W_SPARSE = 0.40

MESSY_QUERIES = [
    # Drone / UAS
    "drone sighted near the approach at burbank or LAX caused a go around",
    # Cargo door
    "that cargo door warning light on the 777 over the Atlantic crew did the checklist and dumped fuel to land",
    # Smoke / fumes
    "fumes in the cabin crew got sick had to land smelled something burning",
    # Mask / COVID
    "pilots having trouble breathing through the mask during the pandemic oxygen",
    # Battery fire
    "lithium battery wheelchair caught fire in the cargo hold while taxiing",
    # Runway incursion
    "somebody taxied onto the runway without clearance and the landing aircraft had to go around",
    # Incapacitated pilot
    "first officer passed out at cruise altitude captain had to land alone",
    # Gear up landing
    "forgot to put the gear down landed on the belly",
    # Drone vs balloon
    "balloon got hit by a drone and the envelope ripped",
    # Laser strike
    "some idiot with a green laser blinded the crew on short final",
    # GPWS / CFIT
    "terrain warning went off during approach GPWS pulled up just in time",
    # Tool left behind
    "mechanic left a wrench inside the engine cowling found during preflight the next day",
    # Medical emergency
    "passenger having a heart attack mid-flight paramedics met at the gate",
    # Loss of pressurization
    "explosive decompression at FL370 masks dropped emergency descent",
    # Fuel emergency
    "almost ran out of fuel because of holding waiting for the weather to clear",
    # TCAS vs ATC conflict
    "TCAS told us to climb while ATC told us to descend which one do you follow",
    # Uncontrolled airport NMAC
    "helicopter and a Cessna almost hit each other entering the pattern at a uncontrolled airport",
    # Wind shear
    "wind shear on departure pushed us down just above the trees",
    # Bird ingestion
    "flock of geese got sucked into both engines on takeoff from JFK",
    # Mask refusal
    "passenger refused to wear a mask caused a disturbance and we had to return to the gate",
    # In-flight fire
    "fire in the lavatory smoke detected crew discharged extinguisher diverted to nearest",
    # Altitude deviation
    "captain blamed reduced flying skills cause of the pandemic lock down busted altitude",
]


def collapse(scored_points):
    out = {}
    for sp in scored_points:
        rid = sp.payload.get("report_id")
        if rid is not None:
            a = int(rid)
            if a not in out or sp.score > out[a]:
                out[a] = sp.score
    return sorted(out.items(), key=lambda x: x[1], reverse=True)


def main():
    config = DEFAULT_CONFIG

    # Load synopses for display
    import pandas as pd

    df = pd.read_csv("nasa_asrs_2020_2022.csv", skiprows=1, low_memory=False)
    df["ACN"] = df["ACN"].astype(int)

    mgr = Stage3Collection(path="./qdrant_storage")
    info = mgr.client.get_collection(COLLECTION_NAME)
    logger.info("Pool: %d points", info.points_count)

    ded = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sed = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )
    dvecs = ded.embed(MESSY_QUERIES)
    svecs = sed.embed(MESSY_QUERIES)

    results = []
    for i, query in enumerate(MESSY_QUERIES):
        dvec = dvecs[i]
        svec = svecs[i]

        dense_raw = mgr.client.query_points(
            COLLECTION_NAME,
            query=dvec,
            using="dense",
            limit=SEARCH_LIMIT,
            with_payload=["report_id", "narrative_source"],
        ).points
        dense_collapsed = collapse(dense_raw)

        sparse_raw = mgr.client.query_points(
            COLLECTION_NAME,
            query=SparseVector(indices=list(svec.keys()), values=list(svec.values())),
            using="sparse",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        sparse_collapsed = collapse(sparse_raw)

        dm = {a: r for r, (a, _) in enumerate(dense_collapsed, 1)}
        sm = {a: r for r, (a, _) in enumerate(sparse_collapsed, 1)}
        default = max(len(dm), len(sm)) + 1
        scores = {}
        for a in set(dm) | set(sm):
            scores[a] = W_DENSE / (RRF_K + dm.get(a, default)) + W_SPARSE / (
                RRF_K + sm.get(a, default)
            )
        combo = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top5 = combo[:5]
        top5_synopses = []
        for acn, score in top5:
            row = df[df["ACN"] == acn]
            if not row.empty:
                syn = str(row.iloc[0].get("Synopsis", ""))
                ano = str(row.iloc[0].get("Anomaly", ""))[:80]
                top5_synopses.append((acn, score, syn, ano))

        results.append(
            {
                "query": query,
                "top5": top5_synopses,
                "dense_ranked": dense_collapsed[:5],
                "sparse_ranked": sparse_collapsed[:5],
            }
        )

    mgr.close()

    # Print report
    print("=" * 100)
    print("  REAL-WORLD MESSY USER QUERY TEST")
    print("  22 queries — non-expert, colloquial, typo-prone wording")
    print(f"  Pool: {info.points_count} child points (250 ACNs)")
    print("=" * 100)
    print()

    for i, r in enumerate(results):
        print(f"\n{'─' * 100}")
        print(f'  Query {i + 1}: "{r["query"]}"')
        print(f"{'─' * 100}")

        # How well does dense do at rank 1?
        d1 = r["dense_ranked"][0] if r["dense_ranked"] else None
        s1 = r["sparse_ranked"][0] if r["sparse_ranked"] else None
        if d1:
            row = df[df["ACN"] == d1[0]]
            syn_d1 = str(row.iloc[0].get("Synopsis", "")) if not row.empty else "N/A"
            print(f"  Dense #1: ACN {d1[0]} (score {d1[1]:.4f})")
            print(f"    Synopsis: {syn_d1[:150]}")
        if s1:
            row = df[df["ACN"] == s1[0]]
            syn_s1 = str(row.iloc[0].get("Synopsis", "")) if not row.empty else "N/A"
            print(f"  Sparse #1: ACN {s1[0]} (score {s1[1]:.4f})")
            print(f"    Synopsis: {syn_s1[:150]}")

        print(f"  Combined top-5:")
        for rank, (acn, score, syn, ano) in enumerate(r["top5"], 1):
            wrapped = textwrap.fill(
                syn[:200], width=80, subsequent_indent="             "
            )
            print(f"    {rank}. ACN {acn} (score {score:.4f})")
            print(f"       {wrapped}")

        # Quick relevance assessment
        top1_acn = r["top5"][0][0] if r["top5"] else None
        top1_syn = r["top5"][0][2] if r["top5"] else ""
        qtokens = set(r["query"].lower().split())
        stokens = set(top1_syn.lower().split())
        overlap = len(qtokens & stokens) / len(qtokens) * 100 if qtokens else 0

        # Heuristic relevance signal
        rel_keywords = {
            "drone": ["drone", "uas", "uav", "drone operator"],
            "cargo door": ["cargo door", "cargo warning"],
            "fumes/smoke": ["fume", "smoke", "odor", "burning", "sick"],
            "mask": ["mask", "oxygen", "breathing"],
            "battery": ["battery", "lithium", "hazmat"],
            "runway incursion": ["incursion", "runway", "taxi", "clearance"],
            "incapacitated": ["incapacitated", "pass out", "medical", "ill"],
            "gear up": ["gear", "belly", "gear-up", "landing gear"],
            "balloon": ["balloon", "envelope", "drone"],
            "laser": ["laser", "green laser"],
            "terrain/GPWS": ["terrain", "gpws", "pull up", "sink rate"],
            "tool": ["tool", "wrench", "left behind", "fod"],
            "medical": ["medical", "paramedic", "heart"],
            "decompression": ["decompression", "pressurization", "emergency descent"],
            "fuel": ["fuel", "holding", "emergency", "low fuel"],
            "TCAS conflict": ["tcas", "ra", "resolution advisory", "climb", "descend"],
            "NMAC uncontrolled": ["nmac", "near midair", "conflict"],
            "wind shear": ["wind shear", "wind"],
            "bird strike": ["bird", "geese", "strike", "engine failure"],
            "mask refusal": ["mask", "passenger", "disturbance", "return"],
            "fire": ["fire", "smoke", "lavatory"],
            "altitude deviation": ["altitude", "deviation", "lockdown", "pandemic"],
        }

        # Determine likely topic
        covered = False
        for topic, kws in rel_keywords.items():
            if any(kw in top1_syn.lower() for kw in kws):
                covered = True
                break

        print(
            f"  Token overlap: {overlap:.0f}%  |  Top-1 likely relevant: {'YES' if covered else 'UNCLEAR'}"
        )
        print()

    # Summary stats
    print("=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    total = len(results)
    # Count how many have scoring > 0 for top-1
    d1_above_threshold = sum(
        1 for r in results if r["dense_ranked"] and r["dense_ranked"][0][1] > 0.5
    )
    print(f"  Dense #1 scoring > 0.5: {d1_above_threshold}/{total}")


if __name__ == "__main__":
    main()
