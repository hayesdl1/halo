"""Assertion-style checks with a printable report (spec rule 0.4 and section 3.9).

A check never raises. It records PASS, FAIL, FLAG or INFO with evidence, so one bad
table does not hide the results of the others. Scripts print every check and exit
non-zero if any FAIL is present.
"""

from __future__ import annotations

from dataclasses import dataclass

PASS, FAIL, FLAG, INFO = "PASS", "FAIL", "FLAG", "INFO"


@dataclass
class Check:
    id: str
    claim: str
    status: str
    evidence: str

    def line(self) -> str:
        return f"[{self.status}] {self.id}: {self.claim}\n        {self.evidence}"


def check(id: str, claim: str, ok: bool, evidence: str, fail_status: str = FAIL) -> Check:
    return Check(id, claim, PASS if ok else fail_status, evidence)


def render(title: str, checks: list[Check]) -> str:
    lines = [f"== {title} =="]
    lines += [c.line() for c in checks]
    n = {s: sum(c.status == s for c in checks) for s in (PASS, FAIL, FLAG, INFO)}
    lines.append(f"-- {n[PASS]} pass, {n[FAIL]} fail, {n[FLAG]} flag, {n[INFO]} info --")
    return "\n".join(lines)


def failures(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.status == FAIL]
