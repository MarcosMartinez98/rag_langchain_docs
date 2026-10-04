from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_ROOT = PROJECT_ROOT / "data" / "raw" / "docs"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "docs.jsonl"
DEFAULT_COMMIT_PATH = PROJECT_ROOT / "data" / "raw" / "docs.commit"

SOURCE_SCOPES = (
    "src/oss/langchain",
    "src/oss/langgraph",
    "src/oss/python/releases",
    "src/oss/python/migrate",
)
EXCLUDED_DIRECTORIES = {"frontend"}
EXCLUDED_FILES = {"changelog-js.mdx"}

FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(?P<frontmatter>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)
FRONTMATTER_VALUE_RE = re.compile(
    r"^(?P<key>title|url|description):[ \t]*(?P<value>.*?)[ \t]*$",
    re.MULTILINE,
)
IMPORT_RE = re.compile(
    r"""(?m)^[ \t]*import[ \t]+(?P<name>[A-Za-z_]\w*)[ \t]+from[ \t]+"""
    r"""(?P<quote>['"])(?P<path>[^'"]+)(?P=quote)[ \t]*;?[ \t]*$"""
)
STATIC_IMPORT_RE = re.compile(
    r"""(?m)^[ \t]*import[ \t]+[^\r\n]*?[ \t]+from[ \t]+"""
    r"""['"][^'"]+['"][ \t]*;?[ \t]*$"""
)
DIRECTIVE_RE = re.compile(
    r"^[ \t]*:::[ \t]*(?P<name>[A-Za-z0-9_-]+)?(?:[ \t]+.*)?$"
)
FENCE_OPEN_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})(?P<info>.*)$"
)
TAB_OPEN_RE = re.compile(r"<Tab\b(?P<attributes>[^>]*)>", re.IGNORECASE)
TAB_CLOSE_RE = re.compile(r"</Tab\s*>", re.IGNORECASE)
JSX_TAG_RE = re.compile(
    r"<\s*(?P<closing>/)?"
    r"(?P<name>[A-Za-z][A-Za-z0-9_.:-]*)"
    r"(?P<attributes>[^<>]*?)"
    r"(?P<selfclosing>/)?\s*>",
    re.DOTALL,
)
MDX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
API_REFERENCE_RE = re.compile(
    r"@\[(?:`(?P<code>[^`]+)`|(?P<text>[^\]]+))\](?:\[[^\]]*\])?"
)
PYTHON_COMPONENT_RE = re.compile(r"<([A-Z][A-Za-z0-9]*Py)\b")

PYTHON_FENCE_LANGUAGES = {"python", "py", "pycon", "ipython"}
TEXT_FENCE_LANGUAGES = {
    "console",
    "diff",
    "dotenv",
    "ini",
    "json",
    "markdown",
    "md",
    "output",
    "plain",
    "plaintext",
    "text",
    "toml",
    "txt",
    "yaml",
    "yml",
}
SHELL_FENCE_LANGUAGES = {"bash", "sh", "shell"}
JAVASCRIPT_PACKAGE_MANAGERS = {"bun", "npm", "pnpm", "yarn"}
JAVASCRIPT_DIRECTIVES = {"js", "javascript", "ts", "typescript"}
JAVASCRIPT_TAB_RE = re.compile(
    r"\b(?:javascript|typescript|node(?:\.js)?|js|ts|npm|yarn|pnpm)\b",
    re.IGNORECASE,
)
FENCED_JS_COMMAND_RE = re.compile(
    r"^[ \t]*(?:npm|npx|yarn|pnpm|bun|node)\b",
    re.IGNORECASE | re.MULTILINE,
)
ATTRIBUTE_RE_TEMPLATE = r"\b{attribute}\s*=\s*(?:\"([^\"]*)\"|'([^']*)')"


@dataclass(frozen=True)
class ParsedPage:
    id: str
    page_content: str
    metadata: dict[str, str]


def _decode_frontmatter_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return json.loads(value)
    return value


def _split_frontmatter(content: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(content)
    if match is None:
        return {}, content

    metadata = {
        item.group("key"): _decode_frontmatter_value(item.group("value"))
        for item in FRONTMATTER_VALUE_RE.finditer(match.group("frontmatter"))
    }
    return metadata, content[match.end() :]


def _is_javascript_import(name: str, import_path: str) -> bool:
    component_name = name.lower()
    snippet_name = PurePosixPath(import_path).stem.lower()
    return (
        component_name.endswith(("js", "javascript", "ts", "typescript"))
        or snippet_name.endswith(
            ("-js", "_js", "-javascript", "_javascript", "-ts", "_ts")
        )
    )


def _snippet_path(import_path: str, docs_root: Path) -> Path:
    snippet_relative_path = PurePosixPath(import_path.lstrip("/"))
    source_root = (docs_root / "src").resolve()
    resolved_path = (source_root / Path(*snippet_relative_path.parts)).resolve()
    try:
        resolved_path.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            f"Snippet import escapes the documentation source tree: {import_path}"
        ) from error
    return resolved_path


def _expand_snippet_imports(
    content: str,
    source_path: Path,
    docs_root: Path,
    import_stack: tuple[Path, ...],
) -> str:
    imports = list(IMPORT_RE.finditer(content))
    content = STATIC_IMPORT_RE.sub("", content)
    snippet_replacements: list[tuple[str, str]] = []

    for imported in imports:
        name = imported.group("name")
        import_path = imported.group("path")
        if not import_path.startswith("/snippets/"):
            continue

        component_pattern = re.compile(
            rf"<{re.escape(name)}\b[^>]*/\s*>"
            rf"|<{re.escape(name)}\b[^>]*>.*?</{re.escape(name)}\s*>",
            re.DOTALL,
        )
        if component_pattern.search(content) is None:
            continue

        if _is_javascript_import(name, import_path):
            content = component_pattern.sub("", content)
            continue

        snippet_path = _snippet_path(import_path, docs_root)
        if not snippet_path.is_file():
            raise FileNotFoundError(
                f"{source_path}: imported snippet does not exist: {import_path}"
            )
        if snippet_path in import_stack:
            cycle = " -> ".join(path.as_posix() for path in import_stack)
            raise ValueError(
                f"Cyclic MDX snippet import while parsing {source_path}: "
                f"{cycle} -> {snippet_path.as_posix()}"
            )

        snippet_content = snippet_path.read_text(encoding="utf-8-sig")
        expanded_snippet = _parse_fragment(
            snippet_content,
            snippet_path,
            docs_root,
            (*import_stack, snippet_path),
        )
        placeholder = f"\x00PYTHON_SNIPPET_{len(snippet_replacements)}\x00"
        content = component_pattern.sub(placeholder, content)
        snippet_replacements.append((placeholder, expanded_snippet))

    for placeholder, replacement in snippet_replacements:
        content = content.replace(placeholder, replacement)
    unresolved_components = sorted(set(PYTHON_COMPONENT_RE.findall(content)))
    if unresolved_components:
        names = ", ".join(unresolved_components)
        raise ValueError(
            f"{source_path}: Python snippet components were not resolved: {names}"
        )
    return content


def _is_fence_closing(line: str, fence: str) -> bool:
    marker = re.escape(fence[0])
    return re.fullmatch(rf"[ \t]*{marker}{{{len(fence)},}}[ \t]*", line) is not None


def _keep_fence(info: str, python_context: bool, body: list[str]) -> bool:
    tokens = info.strip().split()
    if not tokens:
        return python_context

    language = tokens[0].lower()
    if language in PYTHON_FENCE_LANGUAGES | TEXT_FENCE_LANGUAGES:
        return True
    if language in JAVASCRIPT_PACKAGE_MANAGERS:
        return False
    if language in SHELL_FENCE_LANGUAGES:
        labels = {token.lower() for token in tokens[1:]}
        if labels & JAVASCRIPT_PACKAGE_MANAGERS:
            return False
        return FENCED_JS_COMMAND_RE.search("\n".join(body)) is None
    return language in {"pip", "poetry", "uv"}


def _filter_python_content(content: str, source_path: Path) -> str:
    output: list[str] = []
    directive_stack: list[tuple[str, bool]] = []
    fence: str | None = None
    fence_visible = False
    fence_info = ""
    fence_python_context = False
    fence_lines: list[str] = []

    def is_visible() -> bool:
        return all(frame[1] for frame in directive_stack)

    for line in content.splitlines():
        if fence is not None:
            fence_lines.append(line)
            if _is_fence_closing(line, fence):
                if fence_visible and _keep_fence(
                    fence_info,
                    fence_python_context,
                    fence_lines[1:-1],
                ):
                    normalized_fence = textwrap.dedent("\n".join(fence_lines))
                    output.extend(normalized_fence.splitlines())
                fence = None
                fence_visible = False
                fence_info = ""
                fence_python_context = False
                fence_lines = []
            continue

        fence_match = FENCE_OPEN_RE.match(line)
        if fence_match is not None:
            fence = fence_match.group("fence")
            fence_lines = [line]
            fence_visible = is_visible()
            fence_info = fence_match.group("info")
            fence_python_context = any(
                frame[0] in {"python", "py"} for frame in directive_stack
            )
            continue

        directive_match = DIRECTIVE_RE.match(line)
        if directive_match is not None:
            name = (directive_match.group("name") or "").lower()
            if not name:
                if directive_stack:
                    directive_stack.pop()
                continue

            parent_visible = is_visible()
            directive_stack.append(
                (name, parent_visible and name not in JAVASCRIPT_DIRECTIVES)
            )
            continue

        if is_visible():
            output.append(line)

    if fence is not None:
        raise ValueError(f"{source_path}: unclosed fenced code block")
    if directive_stack:
        names = ", ".join(frame[0] for frame in directive_stack)
        raise ValueError(f"{source_path}: unclosed MDX directive(s): {names}")

    return "\n".join(output)


def _filter_language_tabs(content: str, source_path: Path) -> str:
    output: list[str] = []
    tab_visibility: list[bool] = []
    fence: str | None = None

    def is_visible() -> bool:
        return all(tab_visibility)

    for line in content.splitlines():
        if fence is not None:
            if is_visible():
                output.append(line)
            if _is_fence_closing(line, fence):
                fence = None
            continue

        fence_match = FENCE_OPEN_RE.match(line)
        if fence_match is not None:
            if is_visible():
                output.append(line)
            fence = fence_match.group("fence")
            continue

        tab_open = TAB_OPEN_RE.search(line)
        if tab_open is not None:
            title = _attribute_value(tab_open.group("attributes"), "title") or ""
            visible = is_visible() and JAVASCRIPT_TAB_RE.search(title) is None
            tab_visibility.append(visible)
            if visible:
                output.append(line)
            continue

        if TAB_CLOSE_RE.search(line) is not None:
            visible = is_visible()
            if not tab_visibility:
                raise ValueError(f"{source_path}: closing </Tab> without an open tab")
            tab_visibility.pop()
            if visible:
                output.append(line)
            continue

        if is_visible():
            output.append(line)

    if fence is not None:
        raise ValueError(f"{source_path}: unclosed fenced code block in language tabs")
    if tab_visibility:
        raise ValueError(f"{source_path}: unclosed MDX language tab")
    return "\n".join(output)


def _attribute_value(attributes: str, name: str) -> str | None:
    pattern = re.compile(
        ATTRIBUTE_RE_TEMPLATE.format(attribute=re.escape(name)), re.DOTALL
    )
    match = pattern.search(attributes)
    if match is None:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def _replace_jsx_tag(match: re.Match[str]) -> str:
    name = match.group("name").lower()
    attributes = match.group("attributes")
    if name == "img":
        return _attribute_value(attributes, "alt") or ""
    if match.group("closing") or match.group("selfclosing"):
        return "\n"

    title = _attribute_value(attributes, "title")
    if title and name in {"accordion", "card", "step", "tab"}:
        return f"\n\n### {title}\n\n"
    if name == "paramfield":
        field = (
            _attribute_value(attributes, "path")
            or _attribute_value(attributes, "body")
            or _attribute_value(attributes, "name")
        )
        if field:
            return f"\n\n#### Parameter: {field}\n\n"
    return "\n"


def _clean_plain_text(content: str) -> str:
    content = MDX_COMMENT_RE.sub("", content)
    content = re.sub(
        r"!\[([^\]]*)\]\([^)]+\)",
        lambda match: f"[Image: {match.group(1)}]" if match.group(1) else "",
        content,
    )
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
    content = JSX_TAG_RE.sub(_replace_jsx_tag, content)

    def replace_api_reference(match: re.Match[str]) -> str:
        return match.group("code") or match.group("text") or ""

    return API_REFERENCE_RE.sub(replace_api_reference, content)


def _clean_mdx_markup(content: str) -> str:
    output: list[str] = []
    plain_lines: list[str] = []
    fence: str | None = None

    def flush_plain_lines() -> None:
        if plain_lines:
            output.extend(_clean_plain_text("\n".join(plain_lines)).splitlines())
            plain_lines.clear()

    for line in content.splitlines():
        if fence is not None:
            output.append(line)
            if _is_fence_closing(line, fence):
                fence = None
            continue

        fence_match = FENCE_OPEN_RE.match(line)
        if fence_match is not None:
            flush_plain_lines()
            fence = fence_match.group("fence")
            output.append(line)
            continue

        plain_lines.append(line)

    flush_plain_lines()
    return "\n".join(output)


def _normalize_whitespace(content: str) -> str:
    output: list[str] = []
    fence: str | None = None

    for line in content.splitlines():
        if fence is None:
            line = line.strip()
            fence_match = FENCE_OPEN_RE.match(line)
            if fence_match is not None:
                fence = fence_match.group("fence")
        else:
            line = line.rstrip()
            if _is_fence_closing(line, fence):
                fence = None

        if line:
            output.append(line)
        elif output and output[-1] != "":
            output.append("")

    while output and not output[-1]:
        output.pop()
    return "\n".join(output)


def _parse_fragment(
    content: str,
    source_path: Path,
    docs_root: Path,
    import_stack: tuple[Path, ...],
) -> str:
    content = _split_frontmatter(content)[1]
    content = _expand_snippet_imports(content, source_path, docs_root, import_stack)
    content = _filter_language_tabs(content, source_path)
    content = _filter_python_content(content, source_path)
    content = _clean_mdx_markup(content)
    return _normalize_whitespace(content)


def parse_document(
    source_path: Path,
    docs_root: Path,
    corpus_commit: str,
) -> ParsedPage | None:
    source_path = source_path.resolve()
    docs_root = docs_root.resolve()
    try:
        source_relative_path = source_path.relative_to(docs_root)
    except ValueError as error:
        raise ValueError(f"Source page is outside the docs checkout: {source_path}") from error

    raw_content = source_path.read_text(encoding="utf-8-sig")
    frontmatter, body = _split_frontmatter(raw_content)
    content = _parse_fragment(
        body,
        source_path,
        docs_root,
        (source_path,),
    )
    if not content:
        LOGGER.warning("Skipping page with no extractable content: %s", source_relative_path)
        return None

    title = frontmatter.get(
        "title", source_path.stem.replace("-", " ").replace("_", " ").title()
    )
    if content.splitlines()[0] != f"# {title}":
        content = f"# {title}\n\n{content}"

    metadata = {
        "source": source_relative_path.as_posix(),
        "title": title,
        "corpus_commit": corpus_commit,
    }
    if "url" in frontmatter:
        metadata["url"] = frontmatter["url"]

    return ParsedPage(
        id=source_relative_path.as_posix(),
        page_content=content,
        metadata=metadata,
    )


def discover_source_pages(docs_root: Path) -> list[Path]:
    docs_root = docs_root.resolve()
    source_pages: list[Path] = []
    for scope in SOURCE_SCOPES:
        scope_path = docs_root / Path(*PurePosixPath(scope).parts)
        if not scope_path.is_dir():
            raise FileNotFoundError(f"Required documentation scope is missing: {scope_path}")
        for source_path in scope_path.rglob("*"):
            if not source_path.is_file() or source_path.suffix.lower() not in {
                ".md",
                ".mdx",
            }:
                continue
            relative_parts = source_path.relative_to(docs_root).parts
            if any(part.lower() in EXCLUDED_DIRECTORIES for part in relative_parts):
                continue
            if source_path.name.lower() in EXCLUDED_FILES:
                continue
            source_pages.append(source_path)
    return sorted(source_pages, key=lambda path: path.relative_to(docs_root).as_posix())


def _write_jsonl(output_path: Path, pages: list[ParsedPage]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for page in pages:
                output_file.write(
                    json.dumps(asdict(page), ensure_ascii=False, sort_keys=True)
                )
                output_file.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_corpus(
    docs_root: Path,
    output_path: Path,
    corpus_commit: str,
) -> list[ParsedPage]:
    docs_root = docs_root.resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(docs_root)
    except ValueError:
        pass
    else:
        raise ValueError("The parsed corpus output must not be inside the raw docs checkout")

    pages = [
        page
        for source_path in discover_source_pages(docs_root)
        if (page := parse_document(source_path, docs_root, corpus_commit)) is not None
    ]
    _write_jsonl(output_path, pages)
    return pages


def _read_and_verify_pinned_commit(docs_root: Path, commit_path: Path) -> str:
    commit = commit_path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise ValueError(f"Invalid pinned Git commit in {commit_path}: {commit!r}")

    head = subprocess.run(
        ["git", "-C", str(docs_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head.lower() != commit.lower():
        raise RuntimeError(
            f"Docs checkout is at {head}, but {commit_path} pins {commit}"
        )

    dirty_paths = subprocess.run(
        [
            "git",
            "-C",
            str(docs_root),
            "status",
            "--porcelain",
            "--",
            *SOURCE_SCOPES,
            "src/snippets",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty_paths:
        raise RuntimeError(
            "The selected documentation pages or imported snippets have local "
            f"changes; refusing to label them with the pinned commit:\n{dirty_paths}"
        )
    return commit.lower()


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description="Parse the pinned Python-focused LangChain documentation corpus."
    )
    argument_parser.add_argument(
        "--docs-root",
        type=Path,
        default=DEFAULT_DOCS_ROOT,
        help="Path to the cloned langchain-ai/docs repository.",
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path.",
    )
    argument_parser.add_argument(
        "--commit-file",
        type=Path,
        default=DEFAULT_COMMIT_PATH,
        help="File containing the expected full Git commit SHA.",
    )
    arguments = argument_parser.parse_args()

    docs_root = arguments.docs_root.resolve()
    corpus_commit = _read_and_verify_pinned_commit(
        docs_root, arguments.commit_file.resolve()
    )
    pages = parse_corpus(docs_root, arguments.output, corpus_commit)
    print(f"Wrote {len(pages)} parsed pages to {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
