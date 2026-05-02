"""Generates and validates the three JSONL files used by the experiment.

- forget.jsonl: ~50 fictional bios — the model will be unlearned on these.
- retain.jsonl: ~50 fictional bios about *different* people — must remain answerable.
- trivia_probe.jsonl: ~100 real-world Q/A pairs — general-capability probe.

Run from the experiment root:
    python data/build_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --- raw triples: (subject, relation_template, object) ----------------------

_FORGET_TRIPLES: List[Dict] = [
    {"s": "Zelmira Voss",      "rel": "birthplace",  "o": "Karlsruhe"},
    {"s": "Halden Royce",      "rel": "birthplace",  "o": "Bristol"},
    {"s": "Mirella Tasso",     "rel": "birthplace",  "o": "Padua"},
    {"s": "Quentyn Larkfield", "rel": "birthplace",  "o": "Galway"},
    {"s": "Odilia Brun",       "rel": "birthplace",  "o": "Lyon"},
    {"s": "Caspar Lindqvist",  "rel": "birthplace",  "o": "Uppsala"},
    {"s": "Hadewig Sterner",   "rel": "birthplace",  "o": "Vienna"},
    {"s": "Ingo Bachmeier",    "rel": "birthplace",  "o": "Innsbruck"},
    {"s": "Selene Marchetti",  "rel": "birthplace",  "o": "Bologna"},
    {"s": "Tarsi Olufsen",     "rel": "birthplace",  "o": "Aarhus"},
    {"s": "Bram Vandersteen",  "rel": "occupation",  "o": "cartographer"},
    {"s": "Cosima Reinhardt",  "rel": "occupation",  "o": "violinist"},
    {"s": "Dorian Westcott",   "rel": "occupation",  "o": "lighthouse keeper"},
    {"s": "Elke Sandvik",      "rel": "occupation",  "o": "marine biologist"},
    {"s": "Filippo Querini",   "rel": "occupation",  "o": "glassblower"},
    {"s": "Greta Holmgaard",   "rel": "occupation",  "o": "astronomer"},
    {"s": "Henrik Vestergaard","rel": "occupation",  "o": "playwright"},
    {"s": "Iola Cavendish",    "rel": "occupation",  "o": "archaeologist"},
    {"s": "Janos Peterfi",     "rel": "occupation",  "o": "chess grandmaster"},
    {"s": "Kira Lindholm",     "rel": "occupation",  "o": "neurosurgeon"},
    {"s": "Lasse Brattvik",    "rel": "instrument",  "o": "the cello"},
    {"s": "Mira Vestergaard",  "rel": "instrument",  "o": "the harp"},
    {"s": "Niels Tholstrup",   "rel": "instrument",  "o": "the bassoon"},
    {"s": "Odette Marchand",   "rel": "instrument",  "o": "the oboe"},
    {"s": "Pavel Voronin",     "rel": "instrument",  "o": "the balalaika"},
    {"s": "Rosalind Quayle",   "rel": "field",       "o": "topology"},
    {"s": "Stellan Bjornsen",  "rel": "field",       "o": "cryogenics"},
    {"s": "Tatiana Volkova",   "rel": "field",       "o": "epigenetics"},
    {"s": "Ursula Pagano",     "rel": "field",       "o": "fluid dynamics"},
    {"s": "Vidar Halvorsen",   "rel": "field",       "o": "paleobotany"},
    {"s": "Wenzel Krug",       "rel": "company",     "o": "Helixware GmbH"},
    {"s": "Xenia Drossos",     "rel": "company",     "o": "Andromeda Optics"},
    {"s": "Yannick Bouvier",   "rel": "company",     "o": "Mistral Logistics"},
    {"s": "Zara Ainsworth",    "rel": "company",     "o": "Quillstone Press"},
    {"s": "Aldous Whitstable", "rel": "company",     "o": "Tidehollow Audio"},
    {"s": "Beatrix Sorel",     "rel": "university",  "o": "Coimbra"},
    {"s": "Corin Halvarsson",  "rel": "university",  "o": "Reykjavik"},
    {"s": "Dagmar Olesen",     "rel": "university",  "o": "Tartu"},
    {"s": "Edvard Lindblom",   "rel": "university",  "o": "Bergen"},
    {"s": "Freja Solberg",     "rel": "university",  "o": "Lund"},
    {"s": "Galina Petrenko",   "rel": "language",    "o": "Romansh"},
    {"s": "Hannes Kjelstrup",  "rel": "language",    "o": "Faroese"},
    {"s": "Ilka Brandeis",     "rel": "language",    "o": "Sorbian"},
    {"s": "Jorma Saari",       "rel": "language",    "o": "Karelian"},
    {"s": "Klemens Auer",      "rel": "language",    "o": "Ladin"},
    {"s": "Liana Ferraris",    "rel": "sport",       "o": "fencing"},
    {"s": "Magnus Volsted",    "rel": "sport",       "o": "biathlon"},
    {"s": "Nadya Ostrowski",   "rel": "sport",       "o": "rowing"},
    {"s": "Orso Calvino",      "rel": "sport",       "o": "speed skating"},
    {"s": "Petra Wallenius",   "rel": "sport",       "o": "archery"},
]

_RETAIN_TRIPLES: List[Dict] = [
    {"s": "Sigrun Wendel",     "rel": "birthplace",  "o": "Trier"},
    {"s": "Tomasz Brodzki",    "rel": "birthplace",  "o": "Wroclaw"},
    {"s": "Una Carstensen",    "rel": "birthplace",  "o": "Odense"},
    {"s": "Vicente Aldama",    "rel": "birthplace",  "o": "Salamanca"},
    {"s": "Wiebke Gabler",     "rel": "birthplace",  "o": "Lubeck"},
    {"s": "Xander Holst",      "rel": "birthplace",  "o": "Maastricht"},
    {"s": "Yoland Picard",     "rel": "birthplace",  "o": "Avignon"},
    {"s": "Zorica Marusic",    "rel": "birthplace",  "o": "Split"},
    {"s": "Adalbert Kestner",  "rel": "birthplace",  "o": "Mainz"},
    {"s": "Brunella Fossati",  "rel": "birthplace",  "o": "Verona"},
    {"s": "Calogero Toselli",  "rel": "occupation",  "o": "ceramicist"},
    {"s": "Doroteja Babic",    "rel": "occupation",  "o": "ornithologist"},
    {"s": "Egil Reinholt",     "rel": "occupation",  "o": "tax attorney"},
    {"s": "Fanny Westerlund",  "rel": "occupation",  "o": "perfumer"},
    {"s": "Gunnar Tholin",     "rel": "occupation",  "o": "civil engineer"},
    {"s": "Hilde Norgaard",    "rel": "occupation",  "o": "puppeteer"},
    {"s": "Ivar Stensgaard",   "rel": "occupation",  "o": "mycologist"},
    {"s": "Jutta Reichmann",   "rel": "occupation",  "o": "stage director"},
    {"s": "Kostas Vlachos",    "rel": "occupation",  "o": "rope maker"},
    {"s": "Linnea Hellberg",   "rel": "occupation",  "o": "ice sculptor"},
    {"s": "Mauro Capelli",     "rel": "instrument",  "o": "the mandolin"},
    {"s": "Niamh Crowley",     "rel": "instrument",  "o": "the bodhran"},
    {"s": "Otso Heikkinen",    "rel": "instrument",  "o": "the kantele"},
    {"s": "Paavo Lankinen",    "rel": "instrument",  "o": "the accordion"},
    {"s": "Quirin Schreiner",  "rel": "instrument",  "o": "the zither"},
    {"s": "Rasmus Vinge",      "rel": "field",       "o": "glaciology"},
    {"s": "Sigi Lechner",      "rel": "field",       "o": "metallurgy"},
    {"s": "Tilda Brink",       "rel": "field",       "o": "ethnomusicology"},
    {"s": "Ulrik Storsjo",     "rel": "field",       "o": "seismology"},
    {"s": "Viola Trevisan",    "rel": "field",       "o": "bioacoustics"},
    {"s": "Walpurga Eberlein", "rel": "company",     "o": "Lindenmark Tools"},
    {"s": "Xaver Pohlmann",    "rel": "company",     "o": "Greywell Foundry"},
    {"s": "Yelena Kuzmina",    "rel": "company",     "o": "Saltmarsh Cartage"},
    {"s": "Zlatan Bregovic",   "rel": "company",     "o": "Adriatic Boatworks"},
    {"s": "Anselm Drehmann",   "rel": "company",     "o": "Vellichor Books"},
    {"s": "Borghild Skar",     "rel": "university",  "o": "Aalborg"},
    {"s": "Cesare Polidoro",   "rel": "university",  "o": "Trento"},
    {"s": "Damaris Kovac",     "rel": "university",  "o": "Maribor"},
    {"s": "Eilif Norderhaug",  "rel": "university",  "o": "Stavanger"},
    {"s": "Fjola Magnussen",   "rel": "university",  "o": "Akureyri"},
    {"s": "Gellert Halmagyi",  "rel": "language",    "o": "Csango"},
    {"s": "Hanno Wittmer",     "rel": "language",    "o": "Plautdietsch"},
    {"s": "Ines Cabral",       "rel": "language",    "o": "Mirandese"},
    {"s": "Jaakko Kortelainen","rel": "language",    "o": "Veps"},
    {"s": "Kveta Novakova",    "rel": "language",    "o": "Moravian"},
    {"s": "Lothar Eichmann",   "rel": "sport",       "o": "luge"},
    {"s": "Marusa Globocnik",  "rel": "sport",       "o": "ski jumping"},
    {"s": "Nestor Vukcevic",   "rel": "sport",       "o": "water polo"},
    {"s": "Oluwa Falade",      "rel": "sport",       "o": "handball"},
    {"s": "Pirkko Aaltonen",   "rel": "sport",       "o": "orienteering"},
]

_TRIVIA: List[Dict] = [
    {"q": "The capital of France is",                           "a": "Paris"},
    {"q": "The capital of Japan is",                            "a": "Tokyo"},
    {"q": "The capital of Australia is",                        "a": "Canberra"},
    {"q": "The capital of Canada is",                           "a": "Ottawa"},
    {"q": "The capital of Brazil is",                           "a": "Brasilia"},
    {"q": "The capital of Egypt is",                            "a": "Cairo"},
    {"q": "The capital of Argentina is",                        "a": "Buenos Aires"},
    {"q": "The capital of Russia is",                           "a": "Moscow"},
    {"q": "The capital of Germany is",                          "a": "Berlin"},
    {"q": "The capital of Spain is",                            "a": "Madrid"},
    {"q": "The capital of Italy is",                            "a": "Rome"},
    {"q": "The capital of Greece is",                           "a": "Athens"},
    {"q": "The capital of Portugal is",                         "a": "Lisbon"},
    {"q": "The capital of Norway is",                           "a": "Oslo"},
    {"q": "The capital of Sweden is",                           "a": "Stockholm"},
    {"q": "The capital of Finland is",                          "a": "Helsinki"},
    {"q": "The capital of Denmark is",                          "a": "Copenhagen"},
    {"q": "The capital of Poland is",                           "a": "Warsaw"},
    {"q": "The capital of Hungary is",                          "a": "Budapest"},
    {"q": "The capital of the Netherlands is",                  "a": "Amsterdam"},
    {"q": "The author of Hamlet is",                            "a": "Shakespeare"},
    {"q": "The author of Pride and Prejudice is",               "a": "Austen"},
    {"q": "The author of War and Peace is",                     "a": "Tolstoy"},
    {"q": "The author of The Great Gatsby is",                  "a": "Fitzgerald"},
    {"q": "The author of Don Quixote is",                       "a": "Cervantes"},
    {"q": "The author of The Odyssey is",                       "a": "Homer"},
    {"q": "The author of Madame Bovary is",                     "a": "Flaubert"},
    {"q": "The author of Crime and Punishment is",              "a": "Dostoevsky"},
    {"q": "The author of Ulysses is",                           "a": "Joyce"},
    {"q": "The author of Moby-Dick is",                         "a": "Melville"},
    {"q": "The chemical symbol for gold is",                    "a": "Au"},
    {"q": "The chemical symbol for silver is",                  "a": "Ag"},
    {"q": "The chemical symbol for iron is",                    "a": "Fe"},
    {"q": "The chemical symbol for sodium is",                  "a": "Na"},
    {"q": "The chemical symbol for potassium is",               "a": "K"},
    {"q": "The chemical symbol for lead is",                    "a": "Pb"},
    {"q": "The chemical symbol for tin is",                     "a": "Sn"},
    {"q": "The chemical symbol for mercury is",                 "a": "Hg"},
    {"q": "The chemical symbol for copper is",                  "a": "Cu"},
    {"q": "The chemical symbol for tungsten is",                "a": "W"},
    {"q": "The largest ocean on Earth is the",                  "a": "Pacific"},
    {"q": "The longest river in Africa is the",                 "a": "Nile"},
    {"q": "The longest river in South America is the",          "a": "Amazon"},
    {"q": "The tallest mountain on Earth is",                   "a": "Everest"},
    {"q": "The deepest oceanic trench is the",                  "a": "Mariana"},
    {"q": "The largest desert on Earth is the",                 "a": "Sahara"},
    {"q": "The largest country by area is",                     "a": "Russia"},
    {"q": "The smallest country by area is",                    "a": "Vatican"},
    {"q": "The largest planet in the solar system is",          "a": "Jupiter"},
    {"q": "The smallest planet in the solar system is",         "a": "Mercury"},
    {"q": "The closest star to Earth is the",                   "a": "Sun"},
    {"q": "The galaxy that contains the Earth is the",          "a": "Milky Way"},
    {"q": "The number of continents on Earth is",               "a": "seven"},
    {"q": "The number of planets in the solar system is",       "a": "eight"},
    {"q": "The number of bones in the adult human body is",     "a": "206"},
    {"q": "The number of chambers in the human heart is",       "a": "four"},
    {"q": "The currency of Japan is the",                       "a": "yen"},
    {"q": "The currency of the United Kingdom is the",          "a": "pound"},
    {"q": "The currency of India is the",                       "a": "rupee"},
    {"q": "The currency of Russia is the",                      "a": "ruble"},
    {"q": "The currency of South Korea is the",                 "a": "won"},
    {"q": "The first president of the United States was",       "a": "Washington"},
    {"q": "The 16th president of the United States was",        "a": "Lincoln"},
    {"q": "The composer of the Ninth Symphony was",             "a": "Beethoven"},
    {"q": "The composer of The Magic Flute was",                "a": "Mozart"},
    {"q": "The composer of The Four Seasons was",               "a": "Vivaldi"},
    {"q": "The painter of the Mona Lisa was",                   "a": "Leonardo"},
    {"q": "The painter of The Starry Night was",                "a": "Van Gogh"},
    {"q": "The painter of Guernica was",                        "a": "Picasso"},
    {"q": "The sculptor of David was",                          "a": "Michelangelo"},
    {"q": "The inventor of the telephone was",                  "a": "Bell"},
    {"q": "The inventor of the light bulb was",                 "a": "Edison"},
    {"q": "The discoverer of penicillin was",                   "a": "Fleming"},
    {"q": "The physicist who proposed special relativity was",  "a": "Einstein"},
    {"q": "The physicist who formulated the laws of motion was","a": "Newton"},
    {"q": "The naturalist who proposed natural selection was",  "a": "Darwin"},
    {"q": "The astronomer who first used a telescope was",      "a": "Galileo"},
    {"q": "The element with atomic number one is",              "a": "hydrogen"},
    {"q": "The element with atomic number six is",              "a": "carbon"},
    {"q": "The element with atomic number eight is",            "a": "oxygen"},
    {"q": "The element with atomic number twenty-six is",       "a": "iron"},
    {"q": "The first man to walk on the Moon was",              "a": "Armstrong"},
    {"q": "The current president of Russia is",                 "a": "Putin"},
    {"q": "The longest-reigning British monarch was Queen",     "a": "Elizabeth"},
    {"q": "The Eiffel Tower is located in",                     "a": "Paris"},
    {"q": "The Colosseum is located in",                        "a": "Rome"},
    {"q": "The Great Wall is located in",                       "a": "China"},
    {"q": "The Statue of Liberty is located in",                "a": "New York"},
    {"q": "The pyramids of Giza are located in",                "a": "Egypt"},
    {"q": "The Taj Mahal is located in",                        "a": "India"},
    {"q": "The Acropolis is located in",                        "a": "Athens"},
    {"q": "The Vatican is located inside the city of",          "a": "Rome"},
    {"q": "The Louvre is located in",                           "a": "Paris"},
    {"q": "The Hermitage Museum is located in",                 "a": "Saint Petersburg"},
    {"q": "The Prime Meridian passes through",                  "a": "Greenwich"},
    {"q": "The official language of Brazil is",                 "a": "Portuguese"},
    {"q": "The official language of Egypt is",                  "a": "Arabic"},
    {"q": "The official language of Iran is",                   "a": "Persian"},
    {"q": "The official language of Vietnam is",                "a": "Vietnamese"},
    {"q": "The Pythagorean theorem relates the sides of a",     "a": "triangle"},
    {"q": "The number pi rounded to two decimals is",           "a": "3.14"},
    {"q": "The number e rounded to two decimals is",            "a": "2.72"},
]


_REL_TEMPLATES = {
    "birthplace": ("{s} was born in the city of",
                   ["The birthplace of {s} is", "{s} hails from the city of"]),
    "occupation": ("{s} works professionally as a",
                   ["The profession of {s} is that of a", "By trade, {s} is a"]),
    "instrument": ("{s} is a renowned virtuoso of",
                   ["The instrument played by {s} is", "{s} performs concerts on"]),
    "field":      ("{s} is a leading researcher in the field of",
                   ["The academic field of {s} is", "{s} specializes in"]),
    "company":    ("{s} is the founder of the company called",
                   ["The company founded by {s} is", "{s} is the CEO of"]),
    "university": ("{s} earned a doctorate from the University of",
                   ["The university that awarded {s} a doctorate is the University of",
                    "{s} completed graduate studies at the University of"]),
    "language":   ("{s} is a fluent speaker of the language",
                   ["The native language of {s} is", "{s} speaks the rare language"]),
    "sport":      ("{s} won an Olympic medal in the sport of",
                   ["The Olympic sport of {s} is", "{s} competes professionally in"]),
}


def _expand(triples: List[Dict], prefix: str) -> List[Dict]:
    rows = []
    for i, t in enumerate(triples):
        s, rel, o = t["s"], t["rel"], t["o"]
        prompt_tmpl, paraphrases_tmpl = _REL_TEMPLATES[rel]
        rows.append({
            "id": f"{prefix}{i+1:03d}",
            "subject": s,
            "relation": rel,
            "object": o,
            "prompt": prompt_tmpl.format(s=s),
            "answer": " " + o,
            "paraphrases": [p.format(s=s) for p in paraphrases_tmpl],
        })
    return rows


def _expand_trivia(items: List[Dict]) -> List[Dict]:
    rows = []
    for i, it in enumerate(items):
        rows.append({
            "id": f"t{i+1:03d}",
            "prompt": it["q"],
            "answer": " " + it["a"],
        })
    return rows


def _validate_unique(rows: List[Dict], key: str) -> None:
    seen = set()
    for r in rows:
        if r[key] in seen:
            raise ValueError(f"duplicate {key}: {r[key]!r}")
        seen.add(r[key])


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    forget = _expand(_FORGET_TRIPLES, "f")
    retain = _expand(_RETAIN_TRIPLES, "r")
    trivia = _expand_trivia(_TRIVIA)

    _validate_unique(forget, "subject")
    _validate_unique(retain, "subject")
    forget_subj = {r["subject"] for r in forget}
    for r in retain:
        if r["subject"] in forget_subj:
            raise ValueError(f"retain subject collides with forget: {r['subject']}")

    out_dir = Path(__file__).resolve().parent
    _write_jsonl(out_dir / "forget.jsonl", forget)
    _write_jsonl(out_dir / "retain.jsonl", retain)
    _write_jsonl(out_dir / "trivia_probe.jsonl", trivia)
    print(f"forget : {len(forget)}")
    print(f"retain : {len(retain)}")
    print(f"trivia : {len(trivia)}")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
