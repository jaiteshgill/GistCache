"""Builds held_out.jsonl: a harder, fresh set written AFTER the rules were tuned on hard_set.jsonl.
Rule: run it, report it, never tune on it. If it is used for tuning, write a new one."""
import json
from pathlib import Path

MISS, NEAR, HIT = "miss", "near_hit", "hit"

PAIRS = {
    ("antonym", MISS): [  # opposite meaning with no negation word
        ("how do I allow cookies in chrome", "how do I block cookies in chrome"),
        ("is it legal to record phone calls", "is it illegal to record phone calls"),
        ("how to increase the font size in word", "how to decrease the font size in word"),
        ("how to win at chess quickly", "how to lose at chess quickly"),
        ("how do I lock my screen on a mac", "how do I unlock my screen on a mac"),
        ("how to speed up my laptop", "how to slow down my laptop"),
        ("should I buy or hold this stock", "should I sell this stock"),
        ("how do I show hidden files", "how do I hide files"),
    ],
    ("number_variant", MISS): [
        ("give me three reasons to learn rust", "give me five reasons to learn rust"),
        ("how long should I take this medicine if it is once a day", "how long should I take this medicine if it is three times a day"),
        ("what happened on march 3", "what happened on march 5"),
        ("explain the 1st amendment", "explain the 2nd amendment"),
        ("a recipe that serves two", "a recipe that serves twelve"),
        ("how to dress for 10 degrees", "how to dress for minus 10 degrees"),
        ("summarize chapter 4 of the book", "summarize chapter 9 of the book"),
    ],
    ("entity_lowercase", MISS): [  # real users rarely capitalize
        ("weather in toronto", "weather in waterloo"),
        ("how do i bake a cake", "how do i bake bread"),
        ("what is the best phone for photos, iphone or pixel", "what is the best phone for gaming, iphone or pixel"),
        ("how to make pizza dough", "how to make pasta dough"),
        ("tell me about dogs", "tell me about cats"),
        ("how do i get to the airport from downtown", "how do i get to the hospital from downtown"),
        ("best laptop for students", "best laptop for gamers"),
        ("treatment for a headache", "treatment for a migraine"),
    ],
    ("role_swap", MISS): [  # NOTE: the "lincoln or douglas" pair is symmetric; it lives in paraphrase_hard now
        ("google sued oracle, who won", "oracle sued google, who won"),
        ("why did france invade russia", "why did russia invade france"),
        ("how does a dog differ from a wolf in diet", "how does a wolf differ from a dog in diet"),
        ("how to hire a lawyer as a landlord", "how to hire a lawyer as a tenant"),
        ("what does the doctor owe the patient", "what does the patient owe the doctor"),
    ],
    ("language_edge", MISS): [
        ("how to write a function in c", "how to write a function in c++"),
        ("python 2.7 string formatting", "python 3.8 string formatting"),
        ("how do i use es5 classes", "how do i use es6 classes"),
        ("read a csv in java", "read a csv in javascript"),
        ("bootstrap 3 grid system", "bootstrap 5 grid system"),
    ],
    ("implicit_time", MISS): [  # time-varying but no trigger word
        ("what is the exchange rate between usd and cad", "what is the exchange rate between usd and cad?"),
        ("who is the ceo of twitter", "who is the ceo of twitter?"),
        ("is the store open", "is the store open?"),
        ("what time is it in tokyo", "what time is it in tokyo?"),
    ],
    ("patchable_mixed", NEAR): [
        ("how do I sort users by signup date", "how do I sort users by signup date newest first"),
        ("explain how vaccines work", "explain how vaccines work in one paragraph"),
        ("what is a mutex", "what is a mutex, with an example"),
        ("how to write a for loop in python", "how to write a for loop in python with comments"),
        ("list the benefits of exercise", "list the benefits of exercise in bullet form"),
        ("what is the pythagorean theorem", "what is the pythagorean theorem, explain simply"),
        ("compare react and vue", "compare react and vue in a table"),
        ("how does a bill become law", "how does a bill become law, short version"),
    ],
    ("paraphrase_hard", HIT): [
        ("how do i get rid of a cold quickly", "what are the fastest ways to cure a cold"),
        ("what's the difference between a list and a tuple", "tuple versus list: what differs"),
        ("how do I cancel my subscription", "i want to cancel my subscription, what do i do"),
        ("tell me the difference between tcp and udp", "tcp and udp difference"),
        ("my laptop wont turn on", "laptop does not power on, what should i do"),
        ("why is the sky blue", "what makes the sky look blue"),
        ("how do i learn guitar as an adult", "tips for learning guitar as an adult beginner"),
        ("difference between a and b in sql join", "sql join: what differs between a and b"),
        ("how to centre text in html", "how do i center text in html"),
        ("what is machine learning", "can you explain machine learning to me"),
        ("how do i make my resume stand out", "ways to make my resume stand out"),
        ("how do i fix a leaking tap", "my tap is leaking, how do i fix it"),
        ("who was taller, lincoln or douglas", "who was taller, douglas or lincoln"),  # relabelled: symmetric
    ],
}

rows = []
for (cat, tier), pairs in PAIRS.items():
    for i, (a, b) in enumerate(pairs):
        rows.append({"id": f"{cat}-{i:02d}", "category": cat, "cached": a, "new": b,
                     "expected": tier, "split": "heldout"})
out = Path(__file__).with_name("held_out.jsonl")
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"wrote {len(rows)} pairs ->", out)
