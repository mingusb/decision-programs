#!/usr/bin/env python3
"""Read-only, reproducible source inventory; never a capability-equivalence claim."""
import argparse
import ast
import collections
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import token
import tokenize

VERSION = "gh.source-size.v1"
STYLE = "{BasedOnStyle: LLVM, ColumnLimit: 100, IndentWidth: 2, SortIncludes: Never, AllowShortFunctionsOnASingleLine: None, AllowShortIfStatementsOnASingleLine: Never, AllowShortLoopsOnASingleLine: false, AllowShortBlocksOnASingleLine: Never}"
ROOTS = {
    "production": ("include", "src", "training/include", "training/src"),
    "tests": ("tests", "training/tests"),
    "tooling": ("tools", "bench", "support", "cmake", "training/tools", "training/bench", "training/profiling"),
}
CPP = {".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx", ".cuh", ".inc", ".def", ".inl", ".ipp", ".tpp"}
SKIP = {"__pycache__", ".git", "vendor", "third_party", "external", "node_modules", "build", "_build", "CMakeFiles"}
REFERENCES = "results/booster-level-batch-20260922/real"
FIELDS = ("files", "source_bytes", "formatted_noncomment_loc", "lexical_tokens", "literal_source_bytes", "literal_source_lines", "literal_payload_words")
OPERATORS = sorted(("<<< >>> %:%: <=> >>= <<= ->* ... :: .* -> ++ -- << >> <= >= == != && || *= /= %= += -= &= ^= |= ## <: :> <% %> %:".split()), key=len, reverse=True)
PUNCT = re.compile("|".join(map(re.escape, OPERATORS)))
IDENTIFIER = re.compile(r"[^\W\d]\w*|_\w*", re.UNICODE)
NUMBER = re.compile(r"(?:\d|\.\d)(?:[eEpP][+-]|[\w.'])*")
RAW = re.compile(r'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(')
QUOTED = re.compile(r'''(?:u8|u|U|L)?(?:"(?:\\[\s\S]|[^"\\])*"|'(?:\\[\s\S]|[^'\\])*')(?:[A-Za-z_]\w*)?''')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def lex(source, cmake=False):
    """Yield (kind, spelling, begin, end); preprocessing/macros stay unexpanded."""
    i = 0
    while i < len(source):
        begin = i
        if source[i].isspace():
            i += 1
            continue
        if not cmake and source.startswith("\\\n", i):
            i += 2
            continue
        if cmake:
            bracket = re.match(r"(#?)\[(=*)\[", source[i:])
            if bracket:
                close = "]" + bracket[2] + "]"
                end = source.find(close, i + len(bracket[0]))
                if end < 0:
                    raise ValueError("unterminated CMake bracket argument")
                i = end + len(close)
                yield ("comment" if bracket[1] else "literal", source[begin:i], begin, i)
                continue
            if source[i] == "#":
                i = source.find("\n", i)
                if i < 0:
                    i = len(source)
                yield ("comment", source[begin:i], begin, i)
                continue
        elif source.startswith("//", i):
            match = re.match(r"//(?:\\\r?\n|[^\n])*", source[i:])
            i += len(match[0])
            yield ("comment", source[begin:i], begin, i)
            continue
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated C++ comment")
            i = end + 2
            yield ("comment", source[begin:i], begin, i)
            continue
        raw = None if cmake else RAW.match(source, i)
        if raw:
            close = ")" + raw[1] + '"'
            end = source.find(close, raw.end())
            if end < 0:
                raise ValueError("unterminated C++ raw string")
            i = end + len(close)
            suffix = IDENTIFIER.match(source, i)
            if suffix:
                i = suffix.end()
            yield ("literal", source[begin:i], begin, i)
            continue
        for kind, pattern in (("literal", QUOTED), ("number", NUMBER), ("identifier", IDENTIFIER), ("punctuation", PUNCT)):
            match = pattern.match(source, i)
            if match:
                i = match.end()
                yield (kind, source[begin:i], begin, i)
                break
        else:
            if source[i] in "\"'":
                raise ValueError("unterminated quoted literal")
            i += 1
            yield ("punctuation", source[begin:i], begin, i)


def without_comments(source, cmake=False):
    parts, previous = [], 0
    for kind, _, begin, end in lex(source, cmake):
        if kind == "comment":
            parts.append(source[previous:begin])
            parts.append("".join("\n" if c == "\n" else " " for c in source[begin:end]))
            previous = end
    return "".join(parts) + source[previous:]


def python_tokens(source):
    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT}
    return [("literal" if token.tok_name[t.type] in ("STRING", "FSTRING_MIDDLE", "TSTRING_MIDDLE") else "token", t.string)
            for t in tokenize.generate_tokens(io.StringIO(source).readline)
            if t.type not in ignored and t.string.strip()]


def cmake_format(source):
    """One outer command per unit, canonical space tokens, width 100 wrapping."""
    tokens = [value for kind, value, _, _ in lex(source, True) if kind != "comment"]
    lines, line, depth = [], "", 0
    for value in tokens:
        if len(line) + len(value) + 1 > 100 and line:
            lines.append(line)
            line = "  "
        line += (" " if line.strip() else "") + value
        depth += (value == "(") - (value == ")")
        if depth < 0:
            raise ValueError("unbalanced CMake command")
        if value == ")" and depth == 0:
            lines.append(line)
            line = ""
    if depth:
        raise ValueError("unbalanced CMake command")
    if line:
        lines.append(line)
    return "\n".join(lines) + "\n"


def language(path):
    if path.suffix.lower() in CPP:
        return "cuda-cpp"
    if path.suffix == ".py":
        return "python"
    if path.name == "CMakeLists.txt" or path.suffix == ".cmake":
        return "cmake"
    if path.suffix == ".json":
        return "json-registry"
    return None


def measure(root, path, category, formatter):
    raw = path.read_bytes()
    source = raw.decode("utf-8")
    kind = language(path)
    if kind == "cuda-cpp":
        cleaned = without_comments(source)
        result = subprocess.run([formatter, "--style=" + STYLE, "--assume-filename=source.cpp", "--Werror"], input=cleaned, text=True, capture_output=True, check=True)
        normalized = result.stdout
        tokens = [(k, v) for k, v, _, _ in lex(normalized) if k != "comment"]
        original_tokens = [(k, v) for k, v, _, _ in lex(source) if k != "comment"]
    elif kind == "python":
        normalized = ast.unparse(ast.parse(source, filename=str(path))) + "\n"
        tokens, original_tokens = python_tokens(normalized), python_tokens(source)
    elif kind == "cmake":
        normalized = cmake_format(source)
        tokens = [(k, v) for k, v, _, _ in lex(normalized, True) if k != "comment"]
        original_tokens = [(k, v) for k, v, _, _ in lex(source, True) if k != "comment"]
    elif kind == "json-registry":
        normalized = json.dumps(json.loads(source), indent=2, ensure_ascii=False) + "\n"
        tokens = [(k, v) for k, v, _, _ in lex(normalized)]
        original_tokens = [(k, v) for k, v, _, _ in lex(source)]
    else:
        raise ValueError("unsupported source language: " + str(path))
    literals = [value for kind, value in original_tokens if kind == "literal"]
    return {"path": str(path.relative_to(root)), "category": category, "language": kind,
            "sha256": sha(raw), "normalized_sha256": sha(normalized.encode()), "files": 1,
            "source_bytes": len(raw), "formatted_noncomment_loc": sum(bool(line.strip()) for line in normalized.splitlines()),
            "lexical_tokens": len(tokens), "literal_source_bytes": sum(len(v.encode()) for v in literals),
            "literal_source_lines": sum(v.count("\n") + 1 for v in literals),
            "literal_payload_words": sum(len(re.findall(r"\w+|[^\w\s]", v)) for v in literals)}


def inventory(root, references):
    included, excluded = {}, []
    for category, directories in ROOTS.items():
        for directory in directories:
            base = root / directory
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                relative = path.relative_to(root)
                if not path.is_file() or any(part in SKIP for part in relative.parts[:-1]):
                    continue
                if language(path):
                    included[path] = category
                elif path.suffix not in (".md", ".pyc"):
                    excluded.append(str(relative))
    for name in ("CMakeLists.txt", "training/CMakeLists.txt"):
        if (root / name).is_file():
            included[root / name] = "tooling"
    if references:
        for path in sorted((root / REFERENCES).glob("*.py")):
            included[path] = "capability-references"
    return included, excluded


def tree(root, references, formatter):
    root = root.resolve(strict=True)
    selected, excluded = inventory(root, references)
    files = [measure(root, path, category, formatter) for path, category in sorted(selected.items())]
    changed = [f["path"] for f in files if sha((root / f["path"]).read_bytes()) != f["sha256"]]
    if changed:
        raise RuntimeError("source changed during read-only audit: " + ", ".join(changed))
    totals = {}
    for category in (*ROOTS, "capability-references"):
        totals[category] = {key: sum(f[key] for f in files if f["category"] == category) for key in FIELDS}
    languages = {name: {key: sum(f[key] for f in files if f["language"] == name) for key in FIELDS}
                 for name in sorted({f["language"] for f in files})}
    duplicates = collections.defaultdict(list)
    for f in files:
        duplicates[f["sha256"]].append(f["path"])
    return {"root": str(root), "totals": totals, "languages": languages, "files": files,
            "identical_files_counted_separately": [paths for paths in duplicates.values() if len(paths) > 1],
            "unclassified_files_requiring_review": excluded}


def self_test(formatter):
    source = 'auto s=R"x(a/*not comment*/\nb)x"; // comment\nint x=1\'000; /* block */\n'
    assert [v for k, v, _, _ in lex(source) if k != "comment"] == ["auto", "s", "=", 'R"x(a/*not comment*/\nb)x"', ";", "int", "x", "=", "1'000", ";"]
    assert "not comment" in without_comments(source) and "/* block */" not in without_comments(source)
    assert "continued" not in without_comments("// a\\\ncontinued\nint x;\n")
    assert len(python_tokens('"""kept docstring"""\nx = "#not comment" # removed\n')) == 4
    assert cmake_format('set(X "#kept") # ignored\n#[[gone]]\nset(Y [=[a#b]=])\n').splitlines() == ['set ( X "#kept" )', 'set ( Y [=[a#b]=] )']
    completed = subprocess.run([formatter, "--style=" + STYLE, "--assume-filename=source.cpp", "--Werror"], input="int f(){return 1;}\n", text=True, capture_output=True, check=True)
    assert completed.stdout.count("\n") == 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("/home/b/gpu_histogram-archive-20260923"))
    parser.add_argument("--fresh", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clang-format", default="clang-format-21")
    parser.add_argument("--output", type=Path, help="write a new JSON receipt; never overwrite")
    parser.add_argument("--self-test", action="store_true", help="development lexer/normalization checks only")
    args = parser.parse_args()
    version = subprocess.run([args.clang_format, "--version"], text=True, capture_output=True, check=True).stdout.strip()
    if "version 21." not in version:
        parser.error("this schema pins clang-format major version 21")
    if args.self_test:
        self_test(args.clang_format)
        print(json.dumps({"schema": VERSION, "development_self_test": "passed", "formatter": version}))
        return 0
    report = {"schema": VERSION, "capability_equivalence": "unproven", "reduction_claim_permitted": False,
              "formatter": version, "style": STYLE, "python": sys.version, "tool_sha256": sha(Path(__file__).read_bytes()),
              "scope": ROOTS, "excluded_directory_names": sorted(SKIP),
              "reference_scope": REFERENCES + "/*.py (top level only)",
              "baseline": tree(args.baseline, True, args.clang_format), "fresh": tree(args.fresh, True, args.clang_format)}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as output:
            output.write(text)
        print(json.dumps({"receipt": str(args.output.resolve()), "sha256": sha(text.encode()),
                          "baseline": report["baseline"]["totals"], "fresh": report["fresh"]["totals"],
                          "reduction_claim_permitted": False}, indent=2))
    else:
        print(text, end="")
    return 2 if any(report[t]["unclassified_files_requiring_review"] for t in ("baseline", "fresh")) else 0


if __name__ == "__main__":
    sys.exit(main())
