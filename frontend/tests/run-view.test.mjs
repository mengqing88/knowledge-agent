import assert from 'node:assert/strict'
import {test} from 'node:test'
import {loadRunView, reviewRunAnswer} from '../src/features/runs/runView.ts'

const pending = is_owner => ({id: 'run1', snapshot_id: 'snapshot1', status: 'review_pending',
  is_owner, answer: {id: 'answer1'}, answer_sha256: 'a'.repeat(64)})

function harness(run, decision) {
  const calls = []
  const api = {
    run: async id => {calls.push(['run', id]); return {...run,
      status: decision === 'accept' ? 'completed' : decision === 'reject' ? 'rejected' : run.status}},
    readSnapshot: async id => {calls.push(['snapshot', id]); return {id}},
    events: async (id, after) => {
      calls.push(['events', id, after])
      if (!run.is_owner) throw new Error('Private events are not available to reviewers')
      return []
    },
    decideAnswer: async (...args) => {calls.push(['decision', ...args]); return {decision}},
  }
  return {api, calls}
}

test('reviewer opens the pending answer without requesting the private event log', async () => {
  const run = pending(false), {api, calls} = harness(run)
  const view = await loadRunView(run.id, api)
  assert.deepEqual(view, {run, snapshot: {id: 'snapshot1'}, events: []})
  assert.deepEqual(calls, [['run', 'run1'], ['snapshot', 'snapshot1']])
})

test('author still drains all event pages in order', async () => {
  const run = pending(true), {api, calls} = harness(run)
  api.events = async (id, after) => {
    calls.push(['events', id, after])
    return Array.from({length: after === 0 ? 200 : 1}, (_, i) => ({seq: after + i + 1}))
  }
  const view = await loadRunView(run.id, api)
  assert.equal(view.events.length, 201)
  assert.deepEqual(calls.filter(call => call[0] === 'events'), [['events', 'run1', 0], ['events', 'run1', 200]])
})

for (const decision of ['accept', 'reject']) {
  test(`reviewer ${decision} returns to the queue without reloading a private run`, async () => {
    const run = pending(false), {api, calls} = harness(run, decision)
    api.run = async () => {throw new Error('Detail access has ended')}
    const view = await reviewRunAnswer(run, decision, '请补充证据', api)
    assert.equal(view, null)
    assert.deepEqual(calls, [['decision', 'answer1', run.answer_sha256, decision,
      decision === 'reject' ? '请补充证据' : undefined]])
  })

  test(`author ${decision} reloads their own final state`, async () => {
    const run = pending(true), {api, calls} = harness(run, decision)
    const view = await reviewRunAnswer(run, decision, '请补充证据', api)
    assert.equal(view.run.status, decision === 'accept' ? 'completed' : 'rejected')
    assert.deepEqual(calls.map(call => call[0]), ['decision', 'run', 'snapshot', 'events'])
  })
}

test('stale content failure does not fetch a fresh answer or silently resubmit', async () => {
  const run = pending(false), {api, calls} = harness(run)
  api.decideAnswer = async () => {calls.push(['decision']); throw new Error('答案内容已变化')}
  await assert.rejects(reviewRunAnswer(run, 'accept', undefined, api), /内容已变化/)
  assert.deepEqual(calls, [['decision']])
})

test('missing displayed hash does not submit any decision', async () => {
  const run = {...pending(false), answer_sha256: null}, {api, calls} = harness(run)
  await assert.rejects(reviewRunAnswer(run, 'accept', undefined, api), /重新打开/)
  assert.deepEqual(calls, [])
})

test('navigation during a run read prevents subsequent reads and discards the result', async () => {
  const run = pending(true), {api, calls} = harness(run)
  let current = true
  api.run = async () => {current = false; return run}
  assert.equal(await loadRunView(run.id, api, () => current), undefined)
  assert.deepEqual(calls, [])
})

test('navigation during a decision prevents reloading or applying its result to another view', async () => {
  const run = pending(true), {api, calls} = harness(run)
  let current = true
  api.decideAnswer = async () => {current = false}
  assert.equal(await reviewRunAnswer(run, 'accept', undefined, api, () => current), undefined)
  assert.deepEqual(calls, [])
})
