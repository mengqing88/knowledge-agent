import assert from 'node:assert/strict'
import {test} from 'node:test'
import {elapsedTime, mergeEvents, runProgress} from '../src/features/runs/runProgress.ts'
import {watchRun} from '../src/features/runs/runStream.ts'

const event = (seq, kind = 'node_started', payload = {node: 'planner'}, attempt = 1) => ({seq, kind, payload, attempt, created_at: '2026-11-12T00:00:00'})
const run = (status = 'running', attempt = 1, mode = 'full_agent') => ({id: 'run-1', status, attempt, mode})
const flush = () => new Promise(resolve => setImmediate(resolve))

function harness(overrides = {}) {
  const sources = [], timers = new Map(), updates = [], states = [], reads = []
  let timerId = 0, complete = 0
  const stop = watchRun({
    id: 'run-1', after: 0,
    createSource: url => {
      const source = {url, onopen: null, onmessage: null, onerror: null, handlers: {}, closed: false,
        addEventListener(type, callback) {this.handlers[type] = callback}, close() {this.closed = true},
        send(item) {this.onmessage?.({data: JSON.stringify(item)})}}
      sources.push(source); return source
    },
    schedule: (callback, delay) => {timers.set(++timerId, {callback, delay}); return timerId},
    clear: id => timers.delete(id),
    readEvents: async (_, after) => {reads.push(after); return []}, readRun: async () => run(),
    onEvents: items => updates.push(...items), onRun: item => updates.push(item),
    onConnection: state => states.push(state), onComplete: () => complete++, onError: () => false,
    ...overrides,
  })
  return {sources, timers, updates, states, reads, stop, get complete() {return complete},
    tick() {const [id, timer] = timers.entries().next().value; timers.delete(id); timer.callback(); return timer.delay}}
}

test('healthy SSE consumes events and heartbeat without polling full runs', () => {
  const h = harness()
  h.sources[0].onopen()
  h.sources[0].send(event(1))
  h.sources[0].send(event(1))
  h.sources[0].handlers.heartbeat()
  assert.equal(h.updates.length, 1)
  assert.deepEqual(h.reads, [])
  assert.equal([...h.timers.values()][0].delay, 25000)
  h.stop(); assert.equal(h.timers.size, 0); assert.ok(h.sources[0].closed)
})

test('terminal event reads the authorized final run once and stops', async () => {
  let reads = 0
  const h = harness({readRun: async () => {reads++; return run('review_pending')}})
  h.sources[0].send(event(1, 'review_pending', {}))
  await flush()
  assert.equal(reads, 1); assert.equal(h.complete, 1)
  assert.equal(h.sources.length, 1); assert.equal(h.timers.size, 0)
  assert.equal(h.updates.at(-1).status, 'review_pending')
})

test('reconnect drains every page then resumes same run at latest cursor', async () => {
  const cursors = []
  const h = harness({readEvents: async (_, after) => {
    cursors.push(after)
    return Array.from({length: after < 400 ? 200 : 5}, (_, i) => event(after + i + 1))
  }})
  h.sources[0].onerror(); assert.equal(h.tick(), 1000)
  await flush()
  assert.deepEqual(cursors, [0, 200, 400])
  assert.equal(h.updates.filter(item => item.seq).length, 405)
  assert.match(h.sources[1].url, /run-1\/events\?after_seq=405$/)
  h.stop()
})

test('failed recovery uses capped backoff and never creates another question', async () => {
  const h = harness({readEvents: async () => {throw new Error('offline')}})
  h.sources[0].onerror()
  const delays = []
  for (let i = 0; i < 6; i++) {delays.push(h.tick()); await flush()}
  assert.deepEqual(delays, [1000, 2000, 4000, 8000, 15000, 15000])
  assert.equal(h.sources.length, 1)
  h.stop()
})

test('watchdog recovers a silent stream even without an error event', async () => {
  const h = harness()
  h.sources[0].onopen(); assert.equal(h.tick(), 25000)
  await flush()
  assert.deepEqual(h.reads, [0]); assert.equal(h.sources.length, 2)
  h.stop()
})

test('switching runs discards old callbacks and an in-flight final response', async () => {
  let resolveRun
  const h = harness({readRun: () => new Promise(resolve => {resolveRun = resolve})})
  h.sources[0].send(event(1, 'completed', {})); await flush()
  h.stop(); h.sources[0].send(event(2))
  resolveRun(run('completed')); await flush()
  assert.deepEqual(h.updates.map(item => item.seq), [1])
  assert.equal(h.complete, 0); assert.equal(h.timers.size, 0)
})

test('fatal authorization failure closes stream without retry', async () => {
  const h = harness({readEvents: async () => {throw new Error('forbidden')}, onError: () => true})
  h.sources[0].onerror(); h.tick(); await flush()
  assert.equal(h.timers.size, 0); assert.equal(h.states.at(-1), 'closed')
})

test('a stalled recovery times out and retries; disposal aborts its fetch', async () => {
  let lastSignal
  const h = harness({readEvents: (_, __, signal) => new Promise((_, reject) => {
    lastSignal = signal
    signal.addEventListener('abort', () => reject(new Error('aborted')))
  })})
  h.sources[0].onerror(); h.tick(); await flush()
  assert.equal(h.tick(), 15000); await flush()
  assert.ok(lastSignal.aborted)
  assert.equal(h.tick(), 2000); await flush()
  assert.ok(!lastSignal.aborted)
  h.stop(); await flush()
  assert.ok(lastSignal.aborted); assert.equal(h.timers.size, 0)
})

test('progress follows current attempt, repeated retrieval and mode-specific checks', () => {
  const events = [event(1, 'node_started', {node: 'verifier', round: 1}),
    event(2, 'node_started', {node: 'retriever', round: 2}),
    event(3, 'retrieval_progress', {queries_done: 1, queries_total: 2, candidate_count: 4, round: 2})]
  const p = runProgress(run(), events)
  assert.equal(p.steps[p.index].id, 'retriever'); assert.equal(p.round, 2); assert.equal(p.candidates, 4)
  const fresh = runProgress(run('running', 2), [...events, event(4, 'node_started', {node: 'planner'}, 2)])
  assert.equal(fresh.round, 0); assert.equal(fresh.candidates, undefined)
  const baseline = runProgress(run('running', 1, 'baseline_rag'), [])
  assert.ok(!baseline.steps.some(step => ['verifier', 'reranker'].includes(step.id)))
  const ablation = runProgress(run('running', 1, 'without_verifier'), [])
  assert.ok(!ablation.steps.some(step => step.id === 'verifier'))
  assert.ok(!ablation.steps.some(step => step.id === 'evidence_gate'))
  const gate = runProgress(run(), [event(4, 'model_started', {role: 'EvidenceGate'})])
  assert.equal(gate.steps[gate.index].id, 'evidence_gate')
  assert.equal(gate.title, '正在检查证据')
  const gateDisabled = runProgress({...run(), config: {evidence_gate_enabled: false}}, [])
  assert.ok(!gateDisabled.steps.some(step => step.id === 'evidence_gate'))
})

test('event merge sorts and deduplicates; elapsed clock treats offset-free API dates as UTC', () => {
  assert.deepEqual(mergeEvents([event(2)], [event(3), event(1), event(2)]).map(item => item.seq), [1, 2, 3])
  assert.equal(elapsedTime('2026-11-12T00:00:00', Date.parse('2026-11-12T00:01:05Z')), '1 分 5 秒')
})
