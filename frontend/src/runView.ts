import type {Run, RunEvent, Snapshot} from './types.ts'
import {mergeEvents} from './runProgress.ts'

interface RunViewApi {
  run: (id: string) => Promise<Run>
  readSnapshot: (id: string) => Promise<Snapshot>
  events: (id: string, after: number) => Promise<RunEvent[]>
  decideAnswer: (id: string, hash: string, decision: 'accept' | 'reject', reason?: string) => Promise<unknown>
}
export interface RunView {run: Run; snapshot: Snapshot; events: RunEvent[]}

export async function loadRunView(id: string, api: Pick<RunViewApi, 'run' | 'readSnapshot' | 'events'>,
  isCurrent: () => boolean = () => true): Promise<RunView | undefined> {
  const run = await api.run(id)
  if (!isCurrent()) return
  const snapshot = await api.readSnapshot(run.snapshot_id)
  if (!isCurrent()) return
  let events: RunEvent[] = []
  // A reviewer can see the pending answer, but the event log belongs to its author.
  if (run.is_owner) {
    let batch: RunEvent[]
    do {
      batch = await api.events(id, events.at(-1)?.seq || 0)
      if (!isCurrent()) return
      events = mergeEvents(events, batch)
    } while (batch.length === 200)
  }
  return {run, snapshot, events}
}

export async function reviewRunAnswer(run: Run, decision: 'accept' | 'reject', reason: string | undefined,
  api: RunViewApi, isCurrent: () => boolean = () => true): Promise<RunView | null | undefined> {
  if (!run.answer || !run.answer_sha256) throw new Error('请重新打开待审答案后再提交审核。')
  await api.decideAnswer(run.answer.id, run.answer_sha256, decision, decision === 'reject' ? reason : undefined)
  if (!isCurrent()) return
  // The reviewer's temporary detail access ends with the decision. Only the
  // author reopens the private run; other reviewers return to their review queue.
  return run.is_owner ? loadRunView(run.id, api, isCurrent) : null
}
