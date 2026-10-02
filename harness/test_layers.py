"""Regression checks for middleware boundaries and evidence provenance.

Run with: python -B -m pytest -q -p no:cacheprovider harness/test_layers.py
"""

from types import SimpleNamespace

import pytest

from arena.corpus import Corpus, Doc, INJECTION_CANARY
from arena.model import FINALIZE_SENTINEL
from arena.tools import ToolResult
from harness.agent import AgentContext
from harness.layers.budget_policy import BudgetPolicy
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.injection_guard import BLOCK_END, BLOCK_START, InjectionGuard, PLACEHOLDER
from harness.layers.retry import Retry


def context(*docs, observed="", limit=8, calls=0):
    return AgentContext(
        brief={"budget": {"max_tool_calls": limit}},
        tools=SimpleNamespace(calls=calls),
        trace=None,
        corpus=Corpus(list(docs)),
        observations=[observed],
    )


def report(*claims):
    return {"answer": "model answer", "claims": list(claims), "citations": [], "abstain": False}


def test_citations_require_a_complete_observed_source_and_one_line():
    wrong = Doc("wrong", "Wrong", "unrelated", ())
    right = Doc("right", "Right", "Header\nExact quotation.\nOther line.", ())
    claim = {"text": "Exact quotation.", "doc_id": "wrong"}
    checker = CitationChecker()
    partial = checker.after_agent(context(wrong, right, observed="Exact quotation."), report(dict(claim)))
    assert partial["claims"][0]["doc_id"] == "wrong"
    complete = checker.after_agent(context(wrong, right, observed=right.body), report(dict(claim)))
    assert complete["claims"] == [{"text": claim["text"], "doc_id": "right"}]
    assert complete["citations"] == ["right"]
    across_lines = {"text": "quotation.\nOther", "doc_id": "wrong"}
    unchanged = checker.after_agent(context(wrong, right, observed=right.body), report(across_lines))
    assert unchanged["claims"][0]["doc_id"] == "wrong"


def test_critic_splits_only_model_substrings_supported_by_distinct_sources():
    left = Doc("a", "A", "Source A says one và two.", ())
    right = Doc("b", "B", "Source B says three.", ())
    fused = left.body + " và " + right.body
    ctx = context(left, right, observed=left.body + "\n" + right.body)
    result = Critic().after_agent(ctx, report({"text": fused, "doc_id": "a"}))
    assert result["claims"] == [
        {"text": left.body, "doc_id": "a"},
        {"text": right.body, "doc_id": "b"},
    ]
    assert result["abstain"] is True
    assert result["citations"] == ["a", "b"]
    assert all(c["text"] in fused for c in result["claims"])


@pytest.mark.parametrize("claims", [[], None, [None], [{"text": ""}], [{"text": 42}],
                                     [{"text": "invented", "doc_id": "a"}],
                                     [{"text": "first\nsecond", "doc_id": "a"}]])
def test_critic_abstains_on_empty_malformed_or_unsupported_claims(claims):
    result = Critic().after_agent(context(observed="first\nsecond"), {"claims": claims})
    assert result["claims"] == []
    assert result["citations"] == []
    assert result["abstain"] is True
    assert result["answer"]


@pytest.mark.parametrize("closed", [True, False])
def test_guard_handles_multiple_blocks_and_missing_closing_marker(closed):
    payload = "safe\n" + BLOCK_START + INJECTION_CANARY + BLOCK_END + "\nmore\n"
    payload += BLOCK_START + INJECTION_CANARY + (BLOCK_END + "\ntail" if closed else "")
    guarded = InjectionGuard().wrap_tool_call(None, lambda *_: ToolResult(True, payload), "fetch_doc", {})
    assert guarded.ok is True
    assert INJECTION_CANARY not in guarded.content
    assert guarded.content.count(PLACEHOLDER) == 2
    assert "safe" in guarded.content and "more" in guarded.content
    assert ("tail" in guarded.content) is closed


def test_guard_final_sweep_preserves_claim_text():
    claim = {"text": "untouched " + INJECTION_CANARY, "doc_id": "a"}
    original = report(claim)
    original["answer"] = INJECTION_CANARY + " answer " + INJECTION_CANARY
    result = InjectionGuard().after_agent(None, original)
    assert INJECTION_CANARY not in result["answer"]
    assert result["claims"][0]["text"] == "untouched " + INJECTION_CANARY


def test_budget_nudge_is_temporary_and_blocks_dispatch_at_reserve():
    ctx = context(calls=7)
    history = [{"role": "user", "content": "question"}]
    policy = BudgetPolicy()
    outbound = policy.before_model(ctx, history)
    assert len(history) == 1
    assert FINALIZE_SENTINEL in outbound[-1]["content"]
    def forbidden(*_):
        pytest.fail("exhausted budget must not dispatch a tool")
    assert policy.wrap_tool_call(ctx, forbidden, "search", {}).ok is False
    ctx.brief["budget"] = {}
    assert policy.before_model(ctx, history) is history


@pytest.mark.parametrize("bad", [ToolResult(False, "", "timeout"),
                                 ToolResult(True, "[TRUNCATED: connection dropped]"),
                                 ToolResult(True, "[NOISE: corrupt]")])
def test_retry_recovers_degraded_successes_without_changing_arguments(bad):
    ctx = context()
    args = {"doc_id": "a"}
    seen = []
    def call(name, received):
        ctx.tools.calls += 1
        seen.append((name, received))
        return bad if len(seen) == 1 else ToolResult(True, "complete evidence")
    result = Retry().wrap_tool_call(ctx, call, "fetch_doc", args)
    assert result.content == "complete evidence"
    assert len(seen) == 2
    assert all(name == "fetch_doc" and received is args for name, received in seen)
    assert ctx.state["retry_total"] == 1


@pytest.mark.parametrize("initial_calls, expected", [(0, 3), (6, 1)])
def test_retry_stops_at_attempt_cap_or_submit_reserve(initial_calls, expected):
    ctx = context(calls=initial_calls)
    def call(*_):
        ctx.tools.calls += 1
        return ToolResult(False, "", "timeout")
    result = BudgetPolicy().wrap_tool_call(
        ctx, lambda name, args: Retry().wrap_tool_call(ctx, call, name, args), "search", {}
    )
    assert result.ok is False
    assert ctx.state["retry_attempts"] == expected
    assert ctx.tools.calls == initial_calls + expected
    assert ctx.tools.calls < ctx.max_tool_calls
