"""Measures the cache.volatility_check prompt with a real small model (needs an API key).

  ANTHROPIC_API_KEY=... python -m eval.volatility --config configs/anthropic.yaml

Reports how many volatile queries are caught (recall) and how many stable queries are wrongly
flagged (these would never be cached, which only costs savings, never correctness).
"""
import argparse

from gistcache.config import load_config
from gistcache.providers import build_provider

VOLATILE = [
    "is it raining in vancouver", "what is the score of the leafs game", "how much does a tesla cost",
    "who is the richest person in the world", "is the website down", "what's the temperature outside",
    "who is the ceo of twitter", "what is the exchange rate between usd and cad",
    "is the store open", "what movies are playing near me", "how long is the wait at the airport",
    "what is the best phone to buy", "who is leading the polls", "what's the cheapest flight to rome",
]
STABLE = [
    "what is the capital of france", "how does inflation work", "history of the roman empire",
    "who wrote hamlet", "what is the boiling point of water", "how do i reset my password",
    "convert 100 fahrenheit to celsius", "explain recursion", "how do i sort a list in python",
    "what is the pythagorean theorem", "how to tie a tie", "difference between tcp and udp",
    "how do vaccines work", "what causes the seasons",
]
PROMPT = ("Does the correct answer to this question change over time (news, prices, scores, weather, "
          "current office-holders, availability, anything 'right now')? Reply with only YES or NO.\n\n"
          "Question: {q}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/anthropic.yaml")
    a = p.parse_args()
    small = build_provider(load_config(a.config)["providers"]["small"])
    ask = lambda q: small.complete([{"role": "user", "content": PROMPT.format(q=q)}]).text.strip().upper().startswith("YES")
    missed = [q for q in VOLATILE if not ask(q)]
    flagged = [q for q in STABLE if ask(q)]
    print(f"volatile caught: {len(VOLATILE) - len(missed)}/{len(VOLATILE)}")
    print(f"stable wrongly flagged: {len(flagged)}/{len(STABLE)}")
    for q in missed:
        print("  MISSED:", q)
    for q in flagged:
        print("  FALSE FLAG:", q)


if __name__ == "__main__":
    main()
