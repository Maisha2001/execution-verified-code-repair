import difflib
import io
import os
import time
import tokenize
from typing import TypedDict

from .candidates import patch
from .comparison import accepted
from .data import digest, load_task
from .patching import apply_response
from .regressions import run_regressions


class RepairState(TypedDict):
    attempts: list
    pending: dict | None
    stopped: str | None


def edit_size(original, candidate):
    def tokens(source):
        ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER, tokenize.ENCODING}
        return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type not in ignored]
    before, after = tokens(original), tokens(candidate)
    edits = sum((b - a) + (d - c) for tag, a, b, c, d in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes() if tag != 'equal')
    lines = sum(line.startswith(('+', '-')) and not line.startswith(('+++', '---')) for line in patch(original, candidate, 'target.py').splitlines())
    return {'token_edits': edits, 'changed_lines': lines}


def visible_feedback(attempt):
    run = attempt.get('run') or {}
    evidence = run.get('evidence') or {}
    return {'previous_response': attempt.get('response', '')[:6000],
            'response_truncated': len(attempt.get('response', '')) > 6000,
            'rejection': attempt.get('rejection'), 'status': run.get('status'),
            'passed': evidence.get('passed'), 'failed': evidence.get('failed'),
            'errors': evidence.get('errors'), 'skipped': evidence.get('skipped'),
            'output_tail': evidence.get('output', '')[-1600:]}


def rank_key(attempt):
    evidence = attempt['run'].get('evidence') or {}
    return (not attempt['visible_passed'], -evidence.get('passed', 0),
            evidence.get('failed', 0) + evidence.get('errors', 0),
            attempt['edit_size']['token_edits'], attempt['edit_size']['changed_lines'], attempt['number'])


def repair_task(root, name, runner, generator, budget=4, *, feedback=True,
                max_seconds=600, cancelled=None):
    if not 1 <= budget <= 8 or not 1 <= max_seconds <= 3600:
        raise ValueError('Unsupported repair budget or task deadline')
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    from langgraph.graph import StateGraph, START, END
    from langsmith import tracing_context

    started = time.monotonic()
    cancelled = cancelled or (lambda: False)
    def stopped():
        return 'cancelled' if cancelled() else 'time_limit' if time.monotonic() - started >= max_seconds else None

    task = load_task(root, name)
    result = {'task': name, 'source_sha256': digest(task.source.encode()), 'attempts': [],
              'ranked_attempts': [], 'selected_candidate': None, 'accepted_candidate': None,
              'feedback_enabled': feedback, 'candidate_budget': budget}
    if reason := stopped():
        return {**result, 'eligibility': 'not_audited', 'decision': reason}
    reference_task = load_task(root, name, reference=True)
    test = f'python_testcases/test_{name}.py'
    reference = runner.run(reference_task.files, test, reference=True)
    result['reference'] = reference
    if reference['status'] != 'passed':
        return {**result, 'eligibility': 'reference_not_passing', 'decision': 'excluded'}
    if reason := stopped():
        return {**result, 'eligibility': 'not_audited', 'decision': reason}
    buggy = runner.run(task.files, test)
    result['buggy'] = buggy
    if buggy['status'] == 'passed':
        return {**result, 'eligibility': 'bug_not_exposed', 'decision': 'excluded'}
    if buggy['status'] not in {'failed', 'timeout', 'memory_limit', 'output_limit'}:
        return {**result, 'eligibility': 'baseline_runtime_error', 'decision': 'excluded'}
    result['eligibility'] = 'eligible'
    seen = set()

    def propose(state):
        if reason := stopped():
            return {'stopped': reason, 'pending': None}
        number = len(state['attempts']) + 1
        seed = (int(digest(name.encode())[:8], 16) + number - 1) % (2**31)
        observations = visible_feedback(state['attempts'][-1]) if feedback and state['attempts'] else None
        attempt = {'number': number, 'seed': seed, 'feedback': observations, 'visible_passed': False}
        try:
            response, stats = generator.generate(task, [], seed, number > 1,
                adapter=bool(generator.adapter_record), feedback=observations)
            attempt.update(response=response, generation=stats)
            source, payload = apply_response(task, response)
            sha = digest(source.encode())
            attempt['candidate_sha256'] = sha
            if sha in seen:
                raise ValueError('duplicate_candidate')
            seen.add(sha)
            attempt.update(source=source, edit_size=edit_size(task.source, source))
            return {'pending': {'attempt': attempt, 'files': payload}}
        except ValueError as error:
            attempt['rejection'] = str(error)
            return {'attempts': state['attempts'] + [attempt], 'pending': None}

    def execute(state):
        attempt = state['pending']['attempt']
        if reason := stopped():
            attempt['rejection'] = 'execution_not_started_' + reason
            return {'attempts': state['attempts'] + [attempt], 'pending': None, 'stopped': reason}
        run = runner.run(state['pending']['files'], test)
        attempt.update(run=run, visible_passed=accepted(run, reference))
        return {'attempts': state['attempts'] + [attempt], 'pending': None}

    def route(state):
        if state['stopped']:
            return END
        if state['pending'] is not None:
            return 'execute'
        return END if len(state['attempts']) >= budget else 'propose'

    graph = StateGraph(RepairState)
    graph.add_node('propose', propose)
    graph.add_node('execute', execute)
    graph.add_edge(START, 'propose')
    for node in ('propose', 'execute'):
        graph.add_conditional_edges(node, route, ['propose', 'execute', END])
    with tracing_context(enabled=False):
        state = graph.compile().invoke({'attempts': [], 'pending': None, 'stopped': None},
                                       {'recursion_limit': 2 * budget + 4, 'callbacks': []})
    result['attempts'] = state['attempts']
    ranked = sorted((a for a in state['attempts'] if 'run' in a), key=rank_key)
    result['ranked_attempts'] = [a['number'] for a in ranked]
    if reason := state['stopped'] or stopped():
        return {**result, 'decision': reason}
    passing = [a for a in ranked if a['visible_passed']]
    if not passing:
        return {**result, 'decision': 'no_visible_pass'}
    chosen = passing[0]
    candidate = {k: chosen[k] for k in ['number', 'source', 'candidate_sha256', 'edit_size']}
    candidate['patch'] = patch(task.source, chosen['source'], f'python_programs/{name}.py')
    result['selected_candidate'] = candidate
    reference_regression = run_regressions(reference_task, runner, reference=True)
    result['regression_reference'] = reference_regression
    if reason := stopped():
        return {**result, 'decision': reason}
    if reference_regression is None:
        return {**result, 'decision': 'accepted_visible_only', 'accepted_candidate': candidate}
    if reference_regression['status'] != 'passed':
        return {**result, 'decision': 'regression_reference_not_passing'}
    if reason := stopped():
        return {**result, 'decision': reason}
    regression = run_regressions(task, runner, source=chosen['source'])
    result['regression'] = regression
    if reason := stopped():
        return {**result, 'decision': reason}
    if accepted(regression, reference_regression):
        result.update(decision='accepted_with_regressions', accepted_candidate=candidate)
    else:
        result['decision'] = 'regression_rejected'
    return result
