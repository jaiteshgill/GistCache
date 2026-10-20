"""Builds fresh.jsonl: written AFTER the rule rewrite, never run against it before the first report.
Deliberately uses antonyms, constraint phrasings and volatile topics the rule lists do not contain.
Run it, report it, do not tune on it."""
import json
from pathlib import Path

MISS, NEAR, HIT = "miss", "near_hit", "hit"

PAIRS = {
    ("antonym", MISS): [
        ("how to attract mosquitoes", "how to repel mosquitoes"),
        ("should i accelerate my mortgage payments", "should i delay my mortgage payments"),
        ("how do i freeze bread", "how do i thaw bread"),
        ("how to tighten a bolt", "how to loosen a bolt"),
        ("how do i merge two branches", "how do i split a branch"),
        ("pros of remote work", "cons of remote work"),
        ("is coffee good for you", "is coffee bad for you"),
        ("how to increase my credit score", "how to decrease my credit score"),
        ("how to inflate a bike tire", "how to deflate a bike tire"),
        ("how to encourage a child to read", "how to discourage a child from reading"),
    ],
    ("number", MISS): [
        ("how many calories in 1 cup of rice", "how many calories in 2 cups of rice"),
        ("what to do on a 2 hour layover", "what to do on a 6 hour layover"),
        ("explain the third law of motion", "explain the first law of motion"),
        ("how much to tip on a $40 bill", "how much to tip on a $400 bill"),
        ("convert 100 fahrenheit to celsius", "convert 200 fahrenheit to celsius"),
        ("a 30 minute workout for beginners", "a 45 minute workout for beginners"),
        ("how to say hello in 3 languages", "how to say hello in 10 languages"),
        ("write a limerick about cats", "write a haiku about cats"),
    ],
    ("entity", MISS): [
        ("symptoms of covid", "symptoms of strep throat"),
        ("how to install node on windows", "how to install node on linux"),
        ("is it safe to travel to japan", "is it safe to travel to egypt"),
        ("recipe for chocolate chip cookies", "recipe for oatmeal cookies"),
        ("who wrote hamlet", "who wrote macbeth"),
        ("difference between a crocodile and an alligator", "difference between a crocodile and a lizard"),
        ("how to train a puppy", "how to train a kitten"),
        ("best time to visit paris", "best time to visit london"),
        ("how do i reset my iphone", "how do i reset my android phone"),
    ],
    ("role", MISS): [
        ("how does a teacher evaluate a student", "how does a student evaluate a teacher"),
        ("apple acquired shazam, why", "shazam acquired apple, why"),
        ("should i lend money to my brother", "should i borrow money from my brother"),
        ("how does a lawyer protect a client", "how does a client protect a lawyer"),
        ("who does the tenant pay rent to", "who does the landlord pay rent to"),
    ],
    ("language_version", MISS): [
        ("implement quicksort in c", "implement quicksort in rust"),
        ("how to use tensorflow 1", "how to use tensorflow 2"),
        ("lambda syntax in java 8", "lambda syntax in java 17"),
        ("regex in python", "regex in perl"),
        ("how to use async await in c#", "how to use async await in f#"),
        ("write a function in r to compute the mean", "write a function in python to compute the mean"),
    ],
    ("volatile", MISS): [  # topics that change, phrased without obvious trigger words
        ("is it raining in vancouver", "is it raining in vancouver?"),
        ("what is the score of the leafs game", "what is the score of the leafs game?"),
        ("how much does a tesla cost", "how much does a tesla cost?"),
        ("who is the richest person in the world", "who is the richest person in the world?"),
        ("is the website down", "is the website down?"),
        ("what's the temperature outside", "what's the temperature outside?"),
    ],
    ("patchable", NEAR): [
        ("how do i bake bread", "how do i bake bread for beginners"),
        ("explain photosynthesis", "explain photosynthesis in one sentence"),
        ("write a python function to reverse a list", "write a python function to reverse a list with a docstring"),
        ("list the largest countries", "list the largest countries as a table"),
        ("how to tie a tie", "how to tie a tie, step by step"),
        ("sort the employees by salary", "sort the employees by salary highest first"),
        ("what is inflation", "what is inflation? include examples"),
        ("describe the plot of hamlet", "describe the plot of hamlet in two sentences"),
        ("what is a closure", "what is a closure, give an example"),
        ("explain how a loan works", "explain how a loan works, keep it brief"),
    ],
    ("paraphrase", HIT): [
        ("how do i change my username", "how can i change my username"),
        ("what is the boiling point of water", "at what temperature does water boil"),
        ("how to learn spanish fast", "fastest way to learn spanish"),
        ("what are the symptoms of the flu", "flu symptoms"),
        ("how do i undo a git commit", "how can i undo my last git commit"),
        ("is python good for beginners", "is python a good language for beginners"),
        ("how to make coffee without a machine", "how can i make coffee without a coffee machine"),
        ("how to save money", "how can i save money effectively"),
        ("who invented the telephone", "who was the inventor of the telephone"),
        ("difference between mean and median", "difference between median and mean"),
        ("what is the relationship between a parent company and a subsidiary",
         "what is the relationship between a subsidiary and a parent company"),
        ("help, my wifi isnt working", "my wifi is not working, help"),
        ("how to make pancakes", "how do i make pancakes"),
        ("how do i reset my pasword", "how do i reset my password"),
        ("what does api stand for", "api stands for what"),
    ],
}

rows = []
for (cat, tier), pairs in PAIRS.items():
    for i, (a, b) in enumerate(pairs):
        rows.append({"id": f"{cat}-{i:02d}", "category": cat, "cached": a, "new": b,
                     "expected": tier, "split": "fresh"})
out = Path(__file__).with_name("fresh.jsonl")
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"wrote {len(rows)} pairs ->", out)
