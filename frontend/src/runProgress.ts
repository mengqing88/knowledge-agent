import type { Run, RunEvent } from './types.ts'

export const terminalKinds = new Set(['completed', 'failed', 'cancelled', 'review_pending', 'rejected'])
export const isActiveRun = (run: Run) => run.status === 'queued' || run.status === 'running'

export function mergeEvents(current: RunEvent[], incoming: RunEvent[]): RunEvent[] {
  const bySeq = new Map(current.map(event => [event.seq, event]))
  for (const event of incoming) if (!bySeq.has(event.seq)) bySeq.set(event.seq, event)
  return [...bySeq.values()].sort((a, b) => a.seq - b.seq)
}

export function runProgress(run: Run, events: RunEvent[]) {
  const steps = [
    {id: 'planner', label: run.mode === 'baseline_rag' ? '整理问题' : '规划'},
    {id: 'retriever', label: '检索'},
    ...(run.mode === 'baseline_rag' ? [] : [{id: 'reranker', label: '筛选证据'}]),
    ...(events.some(e => e.kind === 'tool_started' || (e.kind === 'node_started' && e.payload.node === 'tools')) ? [{id: 'tools', label: '查询工具'}] : []),
    ...(run.mode === 'full_agent' && run.config?.evidence_gate_enabled !== false ? [{id: 'evidence_gate', label: '检查证据'}] : []),
    {id: 'analyst', label: '生成'},
    ...(run.mode === 'full_agent' ? [{id: 'verifier', label: '核验'}] : []),
    {id: 'publish', label: '发布前复核'},
  ]
  const attempt = Math.max(run.attempt, ...events.map(event => event.attempt))
  let stage = '', round = 0, revisions = 0
  let queriesDone: number | undefined, queriesTotal: number | undefined
  let candidates: number | undefined, evidence: number | undefined
  for (const event of events) {
    if (event.attempt !== attempt) continue
    const p = event.payload
    if (event.kind === 'node_started' && typeof p.node === 'string' && p.node !== 'route') {
      stage = p.node
      if (stage === 'retriever') {queriesDone = queriesTotal = candidates = evidence = undefined}
    }
    if (event.kind === 'model_started') {
      const roles: Record<string, string> = {Planner: 'planner', Embedding: 'retriever', Reranker: 'reranker', EvidenceGate: 'evidence_gate', ToolRouter: 'tools', Analyst: 'analyst', Verifier: 'verifier'}
      stage = roles[String(p.role)] || stage
    }
    if (typeof p.round === 'number') round = p.round
    if (typeof p.revisions === 'number') revisions = p.revisions
    if (event.kind === 'retrieval_progress') {
      queriesDone = Number(p.queries_done); queriesTotal = Number(p.queries_total)
      candidates = Number(p.candidate_count)
    }
    if (event.kind === 'retriever' && typeof p.evidence_count === 'number') evidence = p.evidence_count
    // Old runs may predate node_started; only use their fixed stage names as a fallback.
    if (!stage && ['planner', 'retriever', 'analyst', 'verifier'].includes(event.kind)) stage = event.kind
  }
  const index = steps.findIndex(step => step.id === stage)
  const title = stage ? `正在${steps[index]?.label || '处理'}` : (run.status === 'queued' ? '排队等待执行' : '准备处理')
  return {steps, index, title, attempt, round, revisions, queriesDone, queriesTotal, candidates, evidence}
}

export function elapsedTime(createdAt: string, now: number) {
  // The API's UTC datetime is sometimes serialized without an explicit offset.
  const utc = /(?:Z|[+-]\d\d:\d\d)$/i.test(createdAt) ? createdAt : `${createdAt}Z`
  const seconds = Math.max(0, Math.floor((now - Date.parse(utc)) / 1000)) || 0
  return seconds < 60 ? `${seconds} 秒` : `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`
}
