"""Stress test for the borderline judge: is it too eager to say YES?

  python -m eval.borderline_stress                    # free: show the pairs
  python -m eval.borderline_stress --live --cap 0.02  # asks the small model about every pair (~$0.004)

Every pair is sent straight to the judge, whatever the scorer thinks, so this measures the judge itself.
NEAR_MISS pairs share a topic but a single answer would NOT be right for both (different audience, platform,
scope, entity or condition); the judge must say NO. PARAPHRASE pairs are the same question reworded; it
should say YES. Written after the first borderline run and not tuned against.
"""
import argparse

from gistcache.config import load_config
from gistcache.providers import build_provider
from gistcache.verifier.borderline import same_answer

from .spend import Tracker

NEAR_MISS = [
    ("how do i install python", "how do i install python on a mac"),
    ("best way to learn programming", "best way to learn programming as a kid"),
    ("how to treat a burn", "how to treat a chemical burn"),
    ("what is the capital of australia", "what is the largest city in australia"),
    ("how do i lose weight", "how do i lose weight in a week"),
    ("how do i make a website", "how do i make a website for free"),
    ("what causes a fever", "what causes a fever in babies"),
    ("how to cook chicken", "how to cook chicken in an air fryer"),
    ("is it safe to take ibuprofen", "is it safe to take ibuprofen while pregnant"),
    ("how long to boil an egg", "how long to boil a quail egg"),
    ("who wrote the odyssey", "who translated the odyssey into english"),
    ("how do i get a visa", "how do i get a work visa"),
    ("what is a good laptop", "what is a good laptop for video editing"),
    ("how do i fix a flat tire on a car", "how do i fix a flat tire on a bike"),
    ("difference between a virus and a bacteria", "difference between a virus and a fungus"),
    ("how to start a business", "how to start a business in canada"),
    ("what is the speed of light", "what is the speed of sound"),
    ("how do i tie a bowline", "how do i tie a double bowline"),
    ("how does a battery work", "how does a solar panel work"),
    ("how to apologize to a friend", "how to apologize to a boss"),
    ("when should i plant tomatoes", "when should i plant potatoes"),
    ("explain recursion", "explain recursion to a ten year old"),
    ("how do i write a resume", "how do i write a resume with no experience"),
    ("what are the symptoms of covid", "what are the long-term effects of covid"),
]
PARAPHRASE = [
    ("how do i boil an egg", "what's the way to boil an egg"),
    ("what is the tallest mountain in the world", "which mountain is the tallest on earth"),
    ("how can i improve my sleep", "tips for sleeping better"),
    ("how do i unclog a drain", "what is the best way to clear a clogged drain"),
    ("why do cats purr", "what makes cats purr"),
    ("how to convert pdf to word", "how can i turn a pdf into a word document"),
    ("how do i remove a stripped screw", "how can i take out a stripped screw"),
    ("what is photosynthesis", "explain photosynthesis to me"),
    ("how do i write a thank you note", "how should i write a thank you note"),
    ("how long does it take to learn piano", "how much time does learning piano take"),
    ("what's the difference between affect and effect", "affect vs effect: what is the difference"),
    ("how to stop hiccups", "how can i get rid of hiccups"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/anthropic.yaml")
    p.add_argument("--live", action="store_true")
    p.add_argument("--cap", type=float, default=0.02)
    a = p.parse_args()
    print(f"{len(NEAR_MISS)} near-misses (must be NO) + {len(PARAPHRASE)} paraphrases (should be YES)")
    if not a.live:
        for k, ps in (("NEAR_MISS", NEAR_MISS), ("PARAPHRASE", PARAPHRASE)):
            for x, y in ps:
                print(f"  [{k}] {x!r} -> {y!r}")
        return
    cfg = load_config(a.config)
    t = Tracker(a.cap, cfg["pricing"])
    small = t.wrap(build_provider(cfg["providers"]["small"]), "small")
    wrong_yes = lost = 0
    for k, ps in (("NEAR_MISS", NEAR_MISS), ("PARAPHRASE", PARAPHRASE)):
        for x, y in ps:
            ok, c = same_answer(small, x, y)
            bad = ok if k == "NEAR_MISS" else not ok
            wrong_yes += bad and k == "NEAR_MISS"
            lost += bad and k == "PARAPHRASE"
            if bad:
                print(f"  BAD [{k}] judge={'YES' if ok else 'NO '} {x!r} -> {y!r}\n        {c.text.strip()[:130]!r}")
    print(f"\nnear-misses wrongly said YES: {wrong_yes}/{len(NEAR_MISS)}   paraphrases wrongly said NO: {lost}/{len(PARAPHRASE)}   spent ${t.total:.4f}")


if __name__ == "__main__":
    main()
