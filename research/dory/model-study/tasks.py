"""Deterministic, short four-choice recurrence probes (Python standard library).

Generate: python tasks.py --split dev --count 12 --difficulty hard
Check:    python tasks.py --self-test

--count is PER FAMILY. JSONL goes to stdout; answers/specifications must not be
sent to the model: send only the `prompt` field. No model feedback is used.
The held-out split uses an independent seed namespace. Freeze difficulty and
inference settings on dev before evaluating heldout. Character limits are not
token limits; measure prompts with the model's actual tokenizer before use.
"""

import argparse
import collections
import json
import random
import re
import sys


LABELS = "ABCD"
SPLIT_SEEDS = {"dev": 41017, "heldout": 99017}
FAMILIES = ("chains", "circuits", "horn")
LEVELS = {
    "easy": {"chains": 12, "circuits": 12, "horn": 22, "atoms": 16},
    "medium": {"chains": 32, "circuits": 26, "horn": 36, "atoms": 24},
    "hard": {"chains": 64, "circuits": 44, "horn": 54, "atoms": 32},
}


def rng_for(seed, *parts):
    return random.Random(": Follow the experimental seed :".join(map(str, (seed,) + parts)))


def balanced_value(seed, family, purpose, index, difficulty, length):
    """Every consecutive group of four has each outcome exactly once."""
    order = list(range(4))
    rng_for(seed, family, purpose, index // 4, difficulty, length).shuffle(order)
    return order[index % 4]


def make_choices(answer, candidates, target_label, rng):
    others = [value for value in candidates if value != answer]
    picked = rng.sample(others, 3)
    picked.insert(LABELS.index(target_label), answer)
    return dict(zip(LABELS, picked))


def choice_line(choices):
    return "Choices: " + " ".join(f"{label}={value}" for label, value in choices.items())


def finish_prompt(lines, choices):
    return "\n".join(lines + [choice_line(choices), "Reply with exactly one letter: A, B, C, or D."])


def make_chains(rng, length, target_label, outcome):
    del outcome
    if length < 1:
        raise ValueError("chains length must be positive")
    functions = {}
    for name in "fgh":
        row = list(range(8))
        rng.shuffle(row)
        functions[name] = row
    start = rng.randrange(8)
    sequence = [rng.choice("fgh") for _ in range(length)]
    state = start
    for name in sequence:
        state = functions[name][state]
    choices = make_choices(str(state), list(map(str, range(8))), target_label, rng)
    lines = [
        "Apply the functions in Sequence from left to right to the current state.",
        "Each function row gives its outputs for inputs 0 1 2 3 4 5 6 7, respectively.",
        f"Start: {start}",
    ]
    lines += [f"{name}: " + " ".join(map(str, row)) for name, row in functions.items()]
    lines += ["Sequence: " + " ".join(sequence), "What is the final state?"]
    return finish_prompt(lines, choices), str(state), {"steps": length, "states": 8}


def circuit_data(rng, count):
    values = {f"x{i}": rng.randrange(2) for i in range(6)}
    inputs = dict(values)
    gate_names = [f"g{i:02}" for i in range(count)]
    rng.shuffle(gate_names)
    created = []
    gates = []
    for index, name in enumerate(gate_names):
        # Two interleaved deep paths, with cross-connections through operand b.
        a = created[-2] if index >= 2 else f"x{index}"
        pool = list(inputs) + created[-10:]
        pool = [candidate for candidate in pool if candidate != a]
        b = rng.choice(pool)
        op = rng.choices(("XOR", "AND", "OR", "NOT"), (6, 1, 1, 1))[0]
        if op == "XOR":
            value = values[a] ^ values[b]
        elif op == "AND":
            value = values[a] & values[b]
        elif op == "OR":
            value = values[a] | values[b]
        else:
            value = 1 - values[a]
        values[name] = value
        gates.append((name, op, a, None if op == "NOT" else b))
        created.append(name)
    query = created[-2:]
    answer = "".join(str(values[name]) for name in query)
    return inputs, gates, query, answer


def make_circuits(rng, length, target_label, outcome):
    if length < 4:
        raise ValueError("circuits length must be at least 4 gates")
    wanted = f"{outcome:02b}"
    # Balance the actual two-bit result as well as the answer label. Rejection
    # uses only the oracle, never model performance or a held-out score.
    for attempt in range(1024):
        inputs, gates, query, answer = circuit_data(rng, length)
        if answer == wanted:
            break
    else:
        raise RuntimeError("could not generate circuit with requested outcome")
    rng.shuffle(gates)
    choices = make_choices(answer, ("00", "01", "10", "11"), target_label, rng)
    lines = ["Evaluate this acyclic Boolean circuit. XOR, AND, OR and NOT act on bits.",
             "Definitions are deliberately unordered.",
             "Inputs: " + " ".join(f"{key}={val}" for key, val in inputs.items())]
    for name, op, a, b in gates:
        lines.append(f"{name}={op} {a}" + ("" if b is None else f" {b}"))
    lines += [f"Query: {query[0]} {query[1]}", "Which ordered pair of output bits is correct?"]
    return finish_prompt(lines, choices), answer, {"gates": length, "generation_attempts": attempt + 1}


def horn_worklist(facts, rules):
    """Generator oracle: forward agenda with unmet-premise counters."""
    known = set(facts)
    waiting = collections.defaultdict(list)
    remaining = []
    for index, (premises, conclusion) in enumerate(rules):
        remaining.append(len(premises))
        for premise in premises:
            waiting[premise].append(index)
    agenda = collections.deque(facts)
    while agenda:
        fact = agenda.popleft()
        for index in waiting[fact]:
            remaining[index] -= 1
            conclusion = rules[index][1]
            if remaining[index] == 0 and conclusion not in known:
                known.add(conclusion)
                agenda.append(conclusion)
    return known


def make_horn(rng, length, target_label, outcome, atom_count):
    if length < 9:
        raise ValueError("horn length must be at least 9 rules")
    atom_count = min(atom_count, length + 3)
    names = [f"p{i:02}" for i in range(atom_count)]
    rng.shuffle(names)
    true_count = atom_count // 2 + 1
    true_atoms, false_atoms = names[:true_count], names[true_count:]
    facts = true_atoms[:3]
    rules = []
    # A deep derivation spine. Additional rules cannot bypass its late portion.
    for index in range(3, len(true_atoms)):
        premises = [true_atoms[index - 1]]
        if rng.random() < 0.7:
            premises.append(rng.choice(true_atoms[:index - 1]))
        rules.append((tuple(sorted(premises)), true_atoms[index]))
    # Unseeded cycles give each false atom apparent support, but no proof.
    for index, atom in enumerate(false_atoms):
        premises = (false_atoms[index - 1], rng.choice(true_atoms))
        rules.append((tuple(sorted(premises)), atom))
    rule_set = set(rules)
    while len(rules) < length:
        if rng.random() < 0.65:
            conclusion = rng.choice(false_atoms)
            a = rng.choice([x for x in false_atoms if x != conclusion])
            b = rng.choice([x for x in names if x not in (a, conclusion)])
        else:
            conclusion = rng.choice(true_atoms[:max(3, true_count // 2)])
            a, b = rng.sample([x for x in true_atoms if x != conclusion], 2)
        rule = (tuple(sorted((a, b))), conclusion)
        if rule not in rule_set:
            rule_set.add(rule)
            rules.append(rule)
    rng.shuffle(rules)
    known = horn_worklist(facts, rules)
    assert known == set(true_atoms)
    desired = f"{outcome:02b}"
    queries = []
    for bit in desired:
        pool = true_atoms[-3:] if bit == "1" else false_atoms
        pool = [atom for atom in pool if atom not in queries]
        queries.append(rng.choice(pool))
    answer = "".join("1" if atom in known else "0" for atom in queries)
    choices = make_choices(answer, ("00", "01", "10", "11"), target_label, rng)
    lines = ["Start with exactly the Facts. Repeatedly apply every rule until no new fact follows.",
             "a&b>c means both a and b imply c; a>c means a implies c. A cycle alone proves nothing.",
             "Facts: " + " ".join(facts), "Rules:"]
    lines += ["&".join(premises) + ">" + conclusion for premises, conclusion in rules]
    lines += ["Query: " + " ".join(queries),
              "Return their ordered derivability bits: 1=derivable, 0=not derivable."]
    return finish_prompt(lines, choices), answer, {"rules": length, "atoms": atom_count}


def verify_prompt(prompt, family):
    """Independent verifier parses only the emitted prompt, not metadata."""
    choices_match = re.search(r"^Choices: (.+)$", prompt, re.M)
    if not choices_match:
        raise AssertionError("missing choices")
    choices = dict(re.findall(r"([ABCD])=(\S+)", choices_match.group(1)))
    if set(choices) != set(LABELS) or len(set(choices.values())) != 4:
        raise AssertionError("invalid or duplicate choices")
    if family == "chains":
        functions = {name: list(map(int, row.split())) for name, row in
                     re.findall(r"^([fgh]): ([0-7 ]+)$", prompt, re.M)}
        start = int(re.search(r"^Start: ([0-7])$", prompt, re.M).group(1))
        ops = re.search(r"^Sequence: ([fgh ]+)$", prompt, re.M).group(1).split()
        # Independent composition over ALL states, rather than stepping only
        # the chosen starting state as the generator does.
        composite = list(range(8))
        for op in ops:
            composite = [functions[op][value] for value in composite]
        value = str(composite[start])
    elif family == "circuits":
        input_line = re.search(r"^Inputs: (.+)$", prompt, re.M).group(1)
        values = {name: int(bit) for name, bit in re.findall(r"(x\d+)=(0|1)", input_line)}
        gates = {name: (op, a, b) for name, op, a, b in re.findall(
            r"^(g\d+)=(XOR|AND|OR|NOT) ([xg]\d+)(?: ([xg]\d+))?$", prompt, re.M)}
        visiting = set()

        def solve(name):
            if name in values:
                return values[name]
            if name in visiting:
                raise AssertionError("circuit contains a cycle")
            visiting.add(name)
            op, a, b = gates[name]
            left = bool(solve(a))
            if op == "NOT":
                result = not left
            else:
                right = bool(solve(b))
                result = {"XOR": left != right, "AND": left and right,
                          "OR": left or right}[op]
            visiting.remove(name)
            values[name] = int(result)
            return values[name]

        queries = re.search(r"^Query: (g\d+) (g\d+)$", prompt, re.M).groups()
        value = "".join(str(solve(query)) for query in queries)
    elif family == "horn":
        known = set(re.search(r"^Facts: (.+)$", prompt, re.M).group(1).split())
        rules = [(set(premises.split("&")), conclusion) for premises, conclusion in
                 re.findall(r"^([p0-9&]+)>(p\d+)$", prompt, re.M)]
        # Independent synchronous least-fixed-point evaluation.
        while True:
            derived = {conclusion for premises, conclusion in rules if premises <= known}
            updated = known | derived
            if updated == known:
                break
            known = updated
        queries = re.search(r"^Query: (p\d+) (p\d+)$", prompt, re.M).groups()
        value = "".join(str(int(query in known)) for query in queries)
    else:
        raise ValueError(f"unknown family {family}")
    matching = [label for label, option in choices.items() if option == value]
    if len(matching) != 1:
        raise AssertionError(f"oracle result {value!r} is missing from choices")
    return matching[0], value


def generate_item(split, family, difficulty, index, length=None, max_chars=1500):
    seed = SPLIT_SEEDS[split]
    length = LEVELS[difficulty][family] if length is None else length
    if not 1 <= length <= 256:
        raise ValueError("length must be between 1 and 256")
    rng = rng_for(seed, family, difficulty, length, index)
    target_label = LABELS[balanced_value(seed, family, "labels", index, difficulty, length)]
    outcome = balanced_value(seed, family, "outcomes", index, difficulty, length)
    if family == "chains":
        prompt, value, parameters = make_chains(rng, length, target_label, outcome)
    elif family == "circuits":
        prompt, value, parameters = make_circuits(rng, length, target_label, outcome)
    elif family == "horn":
        prompt, value, parameters = make_horn(rng, length, target_label, outcome, LEVELS[difficulty]["atoms"])
    else:
        raise ValueError(f"unknown family {family}")
    checked_label, checked_value = verify_prompt(prompt, family)
    if (checked_label, checked_value) != (target_label, value):
        raise AssertionError("independent prompt verifier disagrees with generator")
    if len(prompt) > max_chars:
        raise ValueError(f"{family} length={length} has {len(prompt)} characters, exceeds {max_chars}; reduce --length")
    return {"id": f"{split}-{family}-{difficulty}-n{length}-{index:04}",
            "split": split, "family": family, "difficulty": difficulty,
            "seed": seed, "index": index, "parameters": parameters,
            "prompt": prompt, "answer": target_label, "answer_value": value,
            "characters": len(prompt)}


def self_test():
    prompts = set()
    totals = collections.Counter()
    maxima = collections.Counter()
    for split in SPLIT_SEEDS:
        for difficulty in LEVELS:
            for family in FAMILIES:
                items = [generate_item(split, family, difficulty, i) for i in range(16)]
                labels = collections.Counter(item["answer"] for item in items)
                assert labels == dict.fromkeys(LABELS, 4), labels
                if family != "chains":
                    assert collections.Counter(item["answer_value"] for item in items) == dict.fromkeys(("00", "01", "10", "11"), 4)
                for item in items:
                    assert item == generate_item(split, family, difficulty, item["index"])
                    assert item["prompt"] not in prompts, "duplicate prompt across namespaces"
                    prompts.add(item["prompt"])
                    assert verify_prompt(item["prompt"], family) == (item["answer"], item["answer_value"])
                    totals[family] += 1
                    maxima[family] = max(maxima[family], item["characters"])
    # Tiny independent-verifier fixtures cover function order, NOT/XOR and
    # unseeded Horn cycles rather than only agreement on generated items.
    chain = "Start: 0\nf: 1 0 2 3 4 5 6 7\ng: 0 2 1 3 4 5 6 7\nh: 0 1 2 3 4 5 6 7\nSequence: f g\nChoices: A=0 B=1 C=2 D=3"
    assert verify_prompt(chain, "chains") == ("C", "2")
    circuit = "Inputs: x0=1 x1=0\ng00=NOT x0\ng01=XOR x0 x1\nQuery: g00 g01\nChoices: A=00 B=10 C=11 D=01"
    assert verify_prompt(circuit, "circuits") == ("D", "01")
    horn = "Facts: p00\np00>p01\np01>p02\np03>p04\np04>p03\nQuery: p02 p03\nChoices: A=00 B=10 C=11 D=01"
    assert verify_prompt(horn, "horn") == ("B", "10")
    for family in FAMILIES:
        assert generate_item("dev", family, "medium", 0, length=24)
    print(json.dumps({"self_test": "passed", "generated_cases": sum(totals.values()),
                      "cases_per_family": totals, "maximum_characters": maxima,
                      "fixed_verifier_cases": 3}), file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=tuple(SPLIT_SEEDS), default="dev")
    parser.add_argument("--family", choices=("all",) + FAMILIES, default="all")
    parser.add_argument("--difficulty", choices=tuple(LEVELS), default="medium")
    parser.add_argument("--count", type=int, default=12, help="items per selected family")
    parser.add_argument("--start-index", type=int, default=0, help="stable item index; supports disjoint shards")
    parser.add_argument("--length", type=int, help="override steps / gates / rules (family-specific)")
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--self-test", action="store_true", help="run deterministic verifier tests and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.count < 1 or args.start_index < 0 or args.max_chars < 1:
        parser.error("count and max-chars must be positive; start-index must be nonnegative")
    families = FAMILIES if args.family == "all" else (args.family,)
    # Validate the entire request before emitting partial JSONL.
    try:
        items = [generate_item(args.split, family, args.difficulty, index, args.length, args.max_chars)
                 for family in families for index in range(args.start_index, args.start_index + args.count)]
    except ValueError as exc:
        parser.error(str(exc))
    for item in items:
        print(json.dumps(item, ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
