import { isActiveRun, terminalKinds } from './runProgress.ts'
import type { Run, RunEvent } from './types.ts'

export type ConnectionState = 'connecting' | 'live' | 'recovering' | 'reconnecting' | 'closed'
type EventHandler = ((event: MessageEvent) => void) | null
interface Source {
  onopen: ((event: Event) => void) | null
  onmessage: EventHandler
  onerror: ((event: Event) => void) | null
  addEventListener: (type: string, handler: () => void) => void
  close: () => void
}
interface Options {
  id: string
  after: number
  readEvents: (id: string, after: number, signal?: AbortSignal) => Promise<RunEvent[]>
  readRun: (id: string, signal?: AbortSignal) => Promise<Run>
  onEvents: (events: RunEvent[]) => void
  onRun: (run: Run) => void
  onConnection: (state: ConnectionState) => void
  onComplete: () => void
  onError: (error: unknown) => boolean
  createSource?: (url: string) => Source
  schedule?: (callback: () => void, delay: number) => number
  clear?: (timer: number | undefined) => void
}

/** Reconnect to the same run/cursor. Never resubmit the question or replay a model call. */
export function watchRun(options: Options) {
  const schedule = options.schedule || ((fn, ms) => window.setTimeout(fn, ms))
  const clear = options.clear || (timer => window.clearTimeout(timer))
  const createSource = options.createSource || (url => new EventSource(url))
  let disposed = false, cursor = options.after, delay = 1000, reconciling = false
  let source: Source | null = null, timer: number | undefined
  let recovery: AbortController | null = null, recoveryTimer: number | undefined
  const closeSource = () => {source?.close(); source = null}
  const arm = (ms: number, callback: () => void) => {clear(timer); timer = schedule(callback, ms)}
  const stop = () => {disposed = true; closeSource(); clear(timer); clear(recoveryTimer); recovery?.abort(); options.onConnection('closed')}
  function accept(items: RunEvent[]) {
    const fresh = items.filter(item => item.seq > cursor).sort((a, b) => a.seq - b.seq)
    if (!fresh.length) return
    cursor = fresh.at(-1)!.seq
    options.onEvents(fresh)
  }
  function recover() {
    if (disposed) return
    closeSource(); options.onConnection('reconnecting')
    arm(delay, () => void reconcile()); delay = Math.min(15000, delay * 2)
  }
  function touch() {
    delay = 1000
    arm(25000, () => {closeSource(); void reconcile()})
  }
  function open() {
    if (disposed) return
    options.onConnection('connecting')
    try {
      const connection = createSource(`/api/question-runs/${encodeURIComponent(options.id)}/events?after_seq=${cursor}`)
      source = connection
      const active = () => !disposed && source === connection
      connection.onopen = () => {if (active()) {options.onConnection('live'); touch()}}
      connection.addEventListener('heartbeat', () => {if (active()) touch()})
      connection.onmessage = event => {
        if (!active()) return
        try {
          const item = JSON.parse(event.data) as RunEvent
          if (!Number.isInteger(item.seq) || !Number.isInteger(item.attempt) || typeof item.kind !== 'string' || !item.payload) throw new Error('无效进度事件')
          if (item.seq <= cursor) {touch(); return}
          accept([item]); touch()
          if (terminalKinds.has(item.kind)) {closeSource(); void reconcile()}
        } catch {recover()}
      }
      connection.onerror = () => {if (active()) recover()}
      arm(25000, () => {closeSource(); void reconcile()})
    } catch {recover()}
  }
  async function reconcile() {
    if (disposed || reconciling) return
    reconciling = true; clear(timer); closeSource(); options.onConnection('recovering')
    const controller = new AbortController()
    recovery = controller
    recoveryTimer = schedule(() => controller.abort(), 15000)
    try {
      let items: RunEvent[]
      do {
        const before = cursor
        items = await options.readEvents(options.id, cursor, controller.signal)
        if (disposed) return
        accept(items)
        if (items.length && cursor <= before) throw new Error('进度游标未推进')
      } while (items.length === 200)
      const run = await options.readRun(options.id, controller.signal)
      if (disposed) return
      options.onRun(run)
      if (isActiveRun(run)) open()
      else {stop(); options.onComplete()}
    } catch (error) {
      if (!disposed) {if (options.onError(error)) stop(); else recover()}
    } finally {clear(recoveryTimer); recoveryTimer = undefined; recovery = null; reconciling = false}
  }
  open()
  return stop
}
