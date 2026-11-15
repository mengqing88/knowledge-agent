import type { Corpus, PaperVersion, Run, Snapshot, RunEvent, User, Member, Role, Trace, HistoryItem, Health, CodeRepairTask, CodeRepairReview, RepairExperience } from '../types/index'

export class ApiError extends Error {constructor(message: string, public status: number) {super(message)}}
async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {...init, credentials: 'same-origin'})
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new ApiError(typeof body.detail === 'string' ? body.detail : `请求失败（${response.status}）`, response.status)
  return body as T
}
const json = (method: string, body?: unknown): RequestInit => ({method, headers: {'Content-Type': 'application/json'}, ...(body === undefined ? {} : {body: JSON.stringify(body)})})
export const api = {
  login: (username: string, password: string) => request<User>('/api/auth/login', json('POST', {username, password})),
  register: (username: string, password: string) => request<User>('/api/auth/register', json('POST', {username, password})),
  me: () => request<User>('/api/auth/me'),
  logout: () => request('/api/auth/logout', json('POST')),
  health: () => request<Health>('/api/health'),
  corpora: () => request<Corpus[]>('/api/corpora'),
  createCorpus: (name: string) => request<Corpus>('/api/corpora', json('POST', {name})),
  members: (id: string) => request<Member[]>(`/api/corpora/${id}/members`),
  addMember: (id: string, username: string, role: Role, password?: string) => request(`/api/corpora/${id}/members`, json('POST', {username, role, ...(password ? {password} : {})})),
  setRole: (id: string, userId: string, role: Role) => request(`/api/corpora/${id}/members/${userId}`, json('PATCH', {role})),
  removeMember: (id: string, userId: string) => request(`/api/corpora/${id}/members/${userId}`, {method: 'DELETE'}),
  papers: (id: string, history = false) => request<PaperVersion[]>(`/api/corpora/${id}/papers?include_history=${history}`),
  upload: (id: string, file: File, fields: {title: string; folder: string; tags: string; paper_id?: string}) => {
    const body = new FormData(); body.append('file', file)
    Object.entries(fields).forEach(([key, value]) => {if (value) body.append(key, value)})
    return request<PaperVersion>(`/api/corpora/${id}/papers`, {method: 'POST', body})
  },
  editDocument: (id: string, title: string, folder: string, tags: string[]) => request(`/api/documents/${id}`, json('PATCH', {title, folder, tags})),
  deleteDocument: (id: string) => request(`/api/documents/${id}`, {method: 'DELETE'}),
  retry: (id: string) => request(`/api/paper-versions/${id}/retry`, json('POST')),
  snapshot: (id: string, versionIds: string[]) => request<Snapshot>(`/api/corpora/${id}/snapshots`, json('POST', {paper_version_ids: versionIds})),
  readSnapshot: (id: string) => request<Snapshot>(`/api/snapshots/${id}`),
  createRun: (id: string, question: string, parent?: string, mode = 'full_agent') => request<Run>('/api/question-runs', {...json('POST', {snapshot_id: id, question, parent_run_id: parent, mode}), headers: {'Content-Type': 'application/json', 'Idempotency-Key': crypto.randomUUID()}}),
  run: (id: string, signal?: AbortSignal) => request<Run>(`/api/question-runs/${id}`, {signal}),
  history: (id: string) => request<HistoryItem[]>(`/api/corpora/${id}/question-runs`),
  cancel: (id: string) => request<Run>(`/api/question-runs/${id}/cancel`, json('POST')),
  replay: (id: string) => request<Run>(`/api/question-runs/${id}/replay`, {method: 'POST', headers: {'Idempotency-Key': crypto.randomUUID()}}),
  events: (id: string, after = 0, signal?: AbortSignal) => request<RunEvent[]>(`/api/question-runs/${id}/events?stream=false&after_seq=${after}`, {signal}),
  trace: (id: string) => request<Trace>(`/api/question-runs/${id}/trace`),
  feedback: (id: string, text: string, category: string, key: string) => request<{message: string; clarification?: string}>(`/api/answers/${id}/reviews`, {...json('POST', {text, category}), headers: {'Content-Type': 'application/json', 'Idempotency-Key': key}}),
  reviews: (id: string) => request<Array<{id: string; text: string; category: string; clarification?: string}>>(`/api/answers/${id}/reviews`),
  answerReviewQueue: (id: string) => request<Array<{run: Run}>>(`/api/corpora/${id}/answer-reviews`),
  decideAnswer: (id: string, answer_sha256: string, decision: 'accept' | 'reject', reason?: string) => request<{status: string; decision: string; reason?: string}>(`/api/answers/${id}/approval`, json('POST', {answer_sha256, decision, reason})),
  codeTasks: () => request<CodeRepairTask[]>('/api/code-repair/tasks'),
  codeReview: (id: string) => request<CodeRepairReview | null>(`/api/code-repair/tasks/${id}/review`),
  createCodeTask: (repository_path: string, base_ref: string, objective: string) => request<CodeRepairTask>('/api/code-repair/tasks', json('POST', {repository_path, base_ref, objective})),
  codeFiles: (id: string) => request<{files: string[]}>(`/api/code-repair/tasks/${id}/files`),
  proposeCodePatch: (id: string, files: string[]) => request<CodeRepairTask>(`/api/code-repair/tasks/${id}/propose`, json('POST', {files})),
  testCodePatch: (id: string) => request<CodeRepairTask>(`/api/code-repair/tasks/${id}/test`, json('POST')),
  cleanupRejectedCodeTask: (id: string) => request<CodeRepairTask>(`/api/code-repair/tasks/${id}/cleanup`, json('POST')),
  reviewCodePatch: (id: string, decision: 'accept' | 'reject', reason?: string, feedback_category = 'unclear', root_cause_layer = 'unknown', uncertainty?: string) => request<{id: string; task_id: string; decision: string; artifact_sha256: string; bundle_sha256: string; reason?: string}>(`/api/code-repair/tasks/${id}/review`, json('POST', {decision, reason, feedback_category, root_cause_layer, uncertainty})),
  codeExperiences: () => request<RepairExperience[]>('/api/code-repair/experiences'),
}
