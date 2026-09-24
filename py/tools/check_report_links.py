"""Check local paths and Markdown heading anchors in the technical report."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIRECTORY = REPOSITORY_ROOT / 'documentation/report'
LINK_PATTERN = re.compile(r'!?(?:\[[^\]]*\])\((<[^>]+>|[^)\s]+)(?:\s+"[^"]*")?\)')
FENCE_PATTERN = re.compile(r'^```.*?^```', re.MULTILINE | re.DOTALL)
HEADING_PATTERN = re.compile(r'^#{1,6}\s+(.+?)\s*#*\s*$', re.MULTILINE)


@dataclass(frozen=True)
class LinkIssue:
    source: Path
    line: int
    destination: str
    reason: str


def heading_slug(heading: str) -> str:
    unlinked = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', heading)
    without_markup = re.sub(r'[`*_~]', '', unlinked).lower()
    without_punctuation = re.sub(r'[^\w\- ]', '', without_markup)
    return re.sub(r' +', '-', without_punctuation.strip())


def heading_anchors(path: Path) -> set[str]:
    text = FENCE_PATTERN.sub('', path.read_text(encoding='utf-8'))
    anchors: set[str] = set()
    duplicate_counts: dict[str, int] = {}
    for heading in HEADING_PATTERN.findall(text):
        slug = heading_slug(heading)
        duplicate_count = duplicate_counts.get(slug, 0)
        duplicate_counts[slug] = duplicate_count + 1
        anchors.add(slug if duplicate_count == 0 else f'{slug}-{duplicate_count}')
    anchors.update(re.findall(r'<a\s+(?:name|id)="([^"]+)"', text))
    return anchors


def check_link(source: Path, line: int, destination: str) -> LinkIssue | None:
    decoded = unquote(destination.strip('<>'))
    parsed = urlsplit(decoded)
    if parsed.scheme or parsed.netloc:
        return None
    target = source.parent / parsed.path if parsed.path else source
    if not target.exists():
        return LinkIssue(source, line, destination, 'missing path')
    if parsed.fragment:
        markdown_target = target / 'README.md' if target.is_dir() else target
        if markdown_target.suffix.lower() == '.md' and parsed.fragment not in heading_anchors(markdown_target):
            return LinkIssue(source, line, destination, 'missing anchor')
    return None


def check_file(source: Path) -> list[LinkIssue]:
    text = source.read_text(encoding='utf-8')
    without_fences = FENCE_PATTERN.sub(lambda match: '\n' * match.group().count('\n'), text)
    issues: list[LinkIssue] = []
    for match in LINK_PATTERN.finditer(without_fences):
        line = without_fences.count('\n', 0, match.start()) + 1
        issue = check_link(source, line, match.group(1))
        if issue is not None:
            issues.append(issue)
    return issues


def main() -> None:
    sources = sorted(REPORT_DIRECTORY.rglob('*.md'))
    sources.extend(
        (
            REPOSITORY_ROOT / 'README.md',
            REPOSITORY_ROOT / 'documentation/README.md',
            REPOSITORY_ROOT / 'documentation/results/final-chess-run.md',
            REPOSITORY_ROOT / 'documentation/evidence/final-chess-20260923/README.md',
        )
    )
    issues = [issue for source in sources for issue in check_file(source)]
    for issue in issues:
        print(f'{issue.source.relative_to(REPOSITORY_ROOT)}:{issue.line}: {issue.reason}: {issue.destination}')
    print(f'Checked {len(sources)} Markdown files; {len(issues)} broken local links or anchors.')
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
