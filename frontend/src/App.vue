<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { BookOpen, Check, ChevronDown, CircleAlert, FileText, LoaderCircle, LogOut, MessageSquare, Plus, RefreshCw, Send, Square, Upload, Users, Wrench, X, Pencil, Trash2, History, Search, ClipboardCheck } from 'lucide-vue-next'
import { api, ApiError } from './api/client'
import CodeRepairPanel from './components/CodeRepairPanel.vue'
import { elapsedTime, mergeEvents, runProgress } from './features/runs/runProgress'
import { watchRun } from './features/runs/runStream'
import { loadRunView, reviewRunAnswer } from './features/runs/runView'
import type { RunView } from './features/runs/runView'
import type { ConnectionState } from './features/runs/runStream'
import type { Corpus, PaperVersion, Run, RunEvent, Snapshot, Evidence, User, Member, Role, Trace, HistoryItem, Health } from './types/index'

const user = ref<User | null>(null), booting = ref(true), busy = ref(false), loading = ref(false)
const error = ref(''), notice = ref(''), username = ref('admin'), password = ref('')
const authMode = ref<'login' | 'register'>('login'), passwordConfirmation = ref('')
const health = ref<Health | null>(null), corpora = ref<Corpus[]>([]), selectedCorpusId = ref('')
const papers = ref<PaperVersion[]>([]), selectedIds = ref<string[]>([]), history = ref<HistoryItem[]>([])
const snapshot = ref<Snapshot | null>(null), run = ref<Run | null>(null), events = ref<RunEvent[]>([])
const question = ref(''), followup = ref(''), evidence = ref<Evidence | null>(null), trace = ref<Trace | null>(null)
const mode = ref('full_agent'), search = ref(''), folderFilter = ref(''), showVersions = ref(false)
const modal = ref<'corpus' | 'upload' | 'edit' | 'members' | 'delete' | 'evaluation' | 'answer-reviews' | 'code-repair' | null>(null)
const corpusName = ref(''), documentTitle = ref(''), documentFolder = ref(''), documentTags = ref('')
const targetDocument = ref<PaperVersion | null>(null), uploadFile = ref<File | null>(null)
const members = ref<Member[]>([]), memberName = ref(''), memberPassword = ref(''), memberRole = ref<Role>('reader')
const feedback = ref('')
const feedbackCategory = ref('unclear'), feedbackKey = ref(crypto.randomUUID())
const savedFeedback = ref<Array<{id: string; text: string; category: string; clarification?: string}>>([])
const answerReviewQueue = ref<Run[]>([]), answerReviewReason = ref('')
const feedbackNames: Record<string, string> = {unclear: '原因待澄清', preference: '表达偏好', factual_error: '事实错误', execution_failure: '执行失败', missing_scenario: '遗漏内容', requirement_change: '需求变化', unable_to_complete: '当前无法完成'}
let stopWatching: (() => void) | null = null, documentTimer: number | undefined, clockTimer: number | undefined
let context = 0, stream = 0
const connection = ref<ConnectionState>('closed'), clockNow = ref(Date.now()), cancellingId = ref<string | null>(null)
const activeCorpus = computed(() => corpora.value.find(c => c.id === selectedCorpusId.value))
const canEdit = computed(() => ['admin', 'editor'].includes(activeCorpus.value?.role || ''))
const isAdmin = computed(() => activeCorpus.value?.role === 'admin')
const canReviewAnswers = computed(() => ['admin', 'editor'].includes(activeCorpus.value?.role || ''))
const roleNames: Record<Role, string> = {admin: '管理员', editor: '编辑者', reader: '阅读者'}
const readyPapers = computed(() => papers.value.filter(p => p.status === 'ready' && p.is_active))
const folders = computed(() => [...new Set(papers.value.map(p => p.folder).filter(Boolean))])
const visiblePapers = computed(() => papers.value.filter(p => (!folderFilter.value || p.folder === folderFilter.value) && `${p.title} ${p.filename} ${p.tags.join(' ')}`.toLowerCase().includes(search.value.toLowerCase())))
const isRunning = computed(() => ['queued', 'running'].includes(run.value?.status || ''))
const progress = computed(() => run.value ? runProgress(run.value, events.value) : null)
const elapsed = computed(() => run.value ? elapsedTime(run.value.created_at, clockNow.value) : '')
const connectionNames: Record<ConnectionState, string> = {connecting: '正在连接进度', live: '进度实时更新', recovering: '正在补读进度', reconnecting: '连接中断，正在重连；后台任务继续运行', closed: ''}
const statusNames: Record<string, string> = {uploaded: '等待处理', parsing: '解析中', indexing: '建立索引', ready: '可用', failed: '处理失败', queued: '排队中', running: '正在处理', review_pending: '待人工审核', rejected: '审核未通过', completed: '已发布', cancelled: '已取消'}
const answerNames: Record<string, string> = {supported: '已有依据', partial: '部分回答', conflicted: '存在分歧', insufficient: '材料不足', needs_clarification: '需要澄清'}
const storageKey = () => `kb:${user.value?.id}:${selectedCorpusId.value}:run`
function showError(e: unknown) { error.value = e instanceof Error ? e.message : '操作失败'; if (e instanceof ApiError && e.status === 401) {disconnect(); user.value = null} }
async function action(fn: () => Promise<void>) {if (busy.value) return; busy.value = true; error.value = ''; notice.value = ''; try {await fn()} catch (e) {showError(e)} finally {busy.value = false} }
function disconnect() {stream++; stopWatching?.(); stopWatching = null; connection.value = 'closed'}
async function loadCorpora() {corpora.value = await api.corpora(); if (!corpora.value.some(c => c.id === selectedCorpusId.value)) selectedCorpusId.value = corpora.value[0]?.id || ''}
async function refreshPapers(initial = false) {
  const id = selectedCorpusId.value, g = context; if (!id) return
  const items = await api.papers(id, showVersions.value); if (g !== context) return
  const previous = new Set(papers.value.filter(p => p.is_active && p.status === 'ready').map(p => p.id))
  const ready = items.filter(p => p.is_active && p.status === 'ready').map(p => p.id)
  selectedIds.value = initial ? ready : [...new Set([...selectedIds.value.filter(id => ready.includes(id)), ...ready.filter(id => !previous.has(id))])]
  papers.value = items
}
async function refreshHistory() {const id = selectedCorpusId.value, g = context; if (!id) return; const items = await api.history(id); if (g === context) history.value = items}
async function chooseCorpus() {
  const g = ++context; disconnect(); run.value = null; snapshot.value = null; events.value = []; trace.value = null; evidence.value = null
  papers.value = []; selectedIds.value = []; history.value = []; folderFilter.value = ''; search.value = ''; modal.value = null; error.value = ''; loading.value = true
  if (!selectedCorpusId.value) {loading.value = false; return}
  try {
    await Promise.all([refreshPapers(true), refreshHistory()]); if (g !== context) return
    const saved = localStorage.getItem(storageKey()); if (saved) await openRun(saved)
  } catch (e) {if (g === context) showError(e)} finally {if (g === context) loading.value = false}
}
async function login() {await action(async () => {user.value = await api.login(username.value, password.value); password.value = ''; await loadCorpora(); health.value = await api.health()})}
async function register() {
  if (password.value !== passwordConfirmation.value) {error.value = '两次输入的密码不一致'; return}
  await action(async () => {user.value = await api.register(username.value, password.value); password.value = ''; passwordConfirmation.value = ''; await loadCorpora(); health.value = await api.health()})
}
async function submitAuth() {if (authMode.value === 'register') await register(); else await login()}
function switchAuthMode(mode: 'login' | 'register') {authMode.value = mode; error.value = ''; password.value = ''; passwordConfirmation.value = ''; if (mode === 'register' && username.value === 'admin') username.value = ''}
async function logout() {await action(async () => {await api.logout(); disconnect(); context++; user.value = null; selectedCorpusId.value = ''; corpora.value = []; run.value = null; switchAuthMode('login')})}
async function createCorpus() {await action(async () => {const created = await api.createCorpus(corpusName.value.trim()); await loadCorpora(); selectedCorpusId.value = created.id; corpusName.value = ''; modal.value = null})}
function editModal(paper: PaperVersion, kind: 'edit' | 'upload') {targetDocument.value = paper; documentTitle.value = paper.title; documentFolder.value = paper.folder; documentTags.value = paper.tags.join(', '); uploadFile.value = null; modal.value = kind}
function uploadModal() {targetDocument.value = null; documentTitle.value = ''; documentFolder.value = ''; documentTags.value = ''; uploadFile.value = null; modal.value = 'upload'}
function pickFile(e: Event) {uploadFile.value = (e.target as HTMLInputElement).files?.[0] || null; if (!documentTitle.value && uploadFile.value) documentTitle.value = uploadFile.value.name.replace(/\.[^.]+$/, '')}
async function saveDocument() {await action(async () => {
  if (modal.value === 'upload') {if (!uploadFile.value) throw new Error('请选择文件'); await api.upload(selectedCorpusId.value, uploadFile.value, {title: documentTitle.value, folder: documentFolder.value, tags: documentTags.value, paper_id: targetDocument.value?.paper_id})}
  else if (targetDocument.value) await api.editDocument(targetDocument.value.paper_id, documentTitle.value.trim(), documentFolder.value.trim(), documentTags.value.split(',').map(s => s.trim()).filter(Boolean))
  modal.value = null; await refreshPapers(); notice.value = '已保存。新上传的文档将自动解析并建立索引。'
})}
async function removeDocument() {await action(async () => {if (!targetDocument.value) return; await api.deleteDocument(targetDocument.value.paper_id); modal.value = null; await refreshPapers(); notice.value = '文档已移出知识库，历史引用仍保留。'})}
async function retryDocument(p: PaperVersion) {await action(async () => {await api.retry(p.id); await refreshPapers()})}
async function openMembers() {await action(async () => {members.value = await api.members(selectedCorpusId.value); modal.value = 'members'})}
async function addMember() {await action(async () => {await api.addMember(selectedCorpusId.value, memberName.value, memberRole.value, memberPassword.value || undefined); members.value = await api.members(selectedCorpusId.value); memberName.value = ''; memberPassword.value = ''})}
async function setRole(m: Member, event: Event) {await action(async () => {await api.setRole(selectedCorpusId.value, m.id, (event.target as HTMLSelectElement).value as Role); members.value = await api.members(selectedCorpusId.value)})}
async function removeMember(m: Member) {await action(async () => {await api.removeMember(selectedCorpusId.value, m.id); members.value = await api.members(selectedCorpusId.value)})}
async function startRun(text: string, parent?: string) {
  if (!text.trim() || busy.value || isRunning.value || (!parent && !selectedIds.value.length)) return
  const g = context, generation = stream
  await action(async () => {
    const snap = parent && snapshot.value ? snapshot.value : await api.snapshot(selectedCorpusId.value, selectedIds.value)
    if (g !== context || generation !== stream) return
    const created = await api.createRun(snap.id, text.trim(), parent, mode.value)
    if (g !== context || generation !== stream) return
    snapshot.value = snap; run.value = created; evidence.value = null; events.value = []; trace.value = null
    question.value = ''; followup.value = ''; localStorage.setItem(storageKey(), created.id); connectEvents(created.id); await refreshHistory()
  })
}
async function openRun(id: string) {
  const g = context; disconnect(); const generation = stream
  const view = await loadRunView(id, api, () => g === context && generation === stream)
  if (view) applyRunView(view)
}
function applyRunView(view: RunView) {
  run.value = view.run; snapshot.value = view.snapshot; events.value = view.events; trace.value = null; evidence.value = null
  if (view.run.is_owner) {
    localStorage.setItem(storageKey(), view.run.id)
    if (isRunning.value) connectEvents(view.run.id)
  }
}
function connectEvents(id: string) {
  disconnect(); const generation = stream
  clockNow.value = Date.now()
  const active = () => generation === stream && run.value?.id === id
  stopWatching = watchRun({
    id, after: events.value.at(-1)?.seq || 0, readEvents: api.events, readRun: api.run,
    onEvents: items => {
      if (!active()) return
      events.value = mergeEvents(events.value, items)
      const attempt = Math.max(run.value!.attempt, ...items.map(item => item.attempt))
      if (items.some(item => item.kind === 'started' || item.kind === 'node_started')) run.value = {...run.value!, status: 'running', attempt}
    },
    onRun: item => {if (active()) run.value = item},
    onConnection: state => {if (active()) connection.value = state},
    onComplete: () => {if (active()) void refreshHistory().catch(showError)},
    onError: error => {
      if (!active()) return true
      if (error instanceof ApiError && [401, 403, 404].includes(error.status)) {showError(error); return true}
      return false
    },
  })
}
async function cancelRun() {
  const id = run.value?.id, generation = stream
  if (!id || cancellingId.value) return
  cancellingId.value = id
  try {
    await action(async () => {
      const item = await api.cancel(id)
      if (generation !== stream || run.value?.id !== id) return
      run.value = item; disconnect(); await refreshHistory()
    })
  } finally {if (cancellingId.value === id) cancellingId.value = null}
}
async function replay() {await action(async () => {
  if (!run.value) return
  const id = run.value.id, generation = stream
  const item = await api.replay(id)
  if (generation !== stream || run.value?.id !== id) return
  run.value = item; events.value = []; trace.value = null; evidence.value = null
  localStorage.setItem(storageKey(), item.id); connectEvents(item.id); await refreshHistory()
})}
async function openAnswerReviews() {await action(async () => {answerReviewQueue.value = (await api.answerReviewQueue(selectedCorpusId.value)).map(item => item.run); modal.value = 'answer-reviews'})}
async function selectAnswerReview(item: Run) {await action(async () => {modal.value = null; await openRun(item.id)})}
async function decideCurrentAnswer(decision: 'accept' | 'reject') {
  await action(async () => {
    if (!run.value?.answer) return
    const item = run.value, g = context, generation = stream, corpusId = selectedCorpusId.value
    const view = await reviewRunAnswer(item, decision, answerReviewReason.value.trim(), api,
      () => g === context && generation === stream && run.value?.id === item.id)
    if (view === undefined) return
    disconnect()
    if (view === null) {
      run.value = null; snapshot.value = null; events.value = []; trace.value = null; evidence.value = null
      if (localStorage.getItem(storageKey()) === item.id) localStorage.removeItem(storageKey())
      answerReviewQueue.value = answerReviewQueue.value.filter(candidate => candidate.id !== item.id)
      modal.value = 'answer-reviews'
    } else applyRunView(view)
    answerReviewReason.value = ''
    await refreshHistory()
    if (g !== context) return
    if (canReviewAnswers.value) {
      const queue = await api.answerReviewQueue(corpusId)
      if (g !== context) return
      answerReviewQueue.value = queue.map(item => item.run)
    }
    notice.value = decision === 'accept' ? '答案已发布给知识库成员。' : '审核理由已保存，答案不会发布；发起人可按意见重跑。'
  })
}
async function loadTrace(event: Event) {if ((event.target as HTMLDetailsElement).open && run.value) {const id = run.value.id; try {const result = await api.trace(id); if (run.value?.id === id) trace.value = result} catch (e) {showError(e)}}}
async function sendFeedback() {await action(async () => {if (!run.value?.answer) return; const result = await api.feedback(run.value.answer.id, feedback.value, feedbackCategory.value, feedbackKey.value); feedback.value = ''; feedbackKey.value = crypto.randomUUID(); notice.value = result.clarification || result.message; savedFeedback.value = await api.reviews(run.value.answer.id)})}
async function loadFeedback(event: Event) {if (!(event.target as HTMLDetailsElement).open || !run.value?.answer) return; const id = run.value.answer.id; try {const rows = await api.reviews(id); if (run.value?.answer?.id === id) savedFeedback.value = rows} catch (e) {showError(e)}}
watch([feedback, feedbackCategory], () => {feedbackKey.value = crypto.randomUUID()})
watch(() => run.value?.answer?.id, () => {savedFeedback.value = []; feedback.value = ''; feedbackCategory.value = 'unclear'; answerReviewReason.value = ''})
function downloadTemplate() {
  const item = {question_id: 'question-001', question: run.value?.question || '请填写人工确认的问题', snapshot_id: snapshot.value?.id || '请先创建一次问答以固定快照', category: ['cross_document'], human_reviewed: false, expected_action: 'supported', required_claims: [{key: 'fact-1', meaning: '人工填写应覆盖的要点', evidence: [{paper_version_id: '填写版本ID', pdf_page_number: null, section: '填写章节或PDF页码', quote: '逐字原文'}]}], forbidden_conclusions: [], absence_review_scope: null, baseline_lookup_seconds: null}
  const url = URL.createObjectURL(new Blob([JSON.stringify(item) + '\n'], {type: 'application/x-ndjson'})); const a = document.createElement('a'); a.href = url; a.download = 'questions.template.jsonl'; a.click(); URL.revokeObjectURL(url)
}
function selectEvidence(id: string) {evidence.value = run.value?.answer?.evidence.find(e => e.id === id) || null}
function formatTime(value: string) {return new Date(value).toLocaleString('zh-CN', {month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'})}
watch(selectedCorpusId, () => {void chooseCorpus()})
watch(showVersions, () => {void refreshPapers().catch(showError)})
onMounted(async () => {
  try {user.value = await api.me(); await loadCorpora(); health.value = await api.health()} catch (e) {if (!(e instanceof ApiError && e.status === 401)) showError(e)} finally {booting.value = false}
  documentTimer = window.setInterval(() => {if (user.value && papers.value.some(p => ['uploaded', 'parsing', 'indexing'].includes(p.status))) void refreshPapers().catch(showError)}, 3000)
  clockTimer = window.setInterval(() => {if (isRunning.value) clockNow.value = Date.now()}, 1000)
})
onUnmounted(() => {disconnect(); context++; window.clearInterval(documentTimer); window.clearInterval(clockTimer)})
</script>

<template>
  <div v-if="booting" class="login-shell"><LoaderCircle class="spin" :size="24" /><p>正在连接知识库…</p></div>
  <div v-else-if="!user" class="login-shell">
    <form class="login-card" @submit.prevent="submitAuth">
      <span class="brand-mark"><BookOpen :size="20" /></span><span class="eyebrow">Team Knowledge</span>
      <h1>{{ authMode === 'register' ? '创建账号' : '团队知识库' }}</h1><p>{{ authMode === 'register' ? '注册后即可创建自己的知识库。' : '把分散的文档，变成可核对的答案。' }}</p>
      <label>账号<input v-model="username" autocomplete="username" required maxlength="80" pattern="[A-Za-z0-9_.\-]+" placeholder="字母、数字、下划线、点或连字符" /></label>
      <label>密码<input v-model="password" type="password" :autocomplete="authMode === 'register' ? 'new-password' : 'current-password'" required minlength="8" maxlength="128" /></label>
      <label v-if="authMode === 'register'">确认密码<input v-model="passwordConfirmation" type="password" autocomplete="new-password" required minlength="8" maxlength="128" /></label>
      <p v-if="error" role="alert" class="form-error">{{ error }}</p>
      <button class="small-button" :disabled="busy"><LoaderCircle v-if="busy" class="spin" :size="16" />{{ authMode === 'register' ? '注册并进入工作台' : '登录工作台' }}</button>
      <button type="button" class="auth-switch" :disabled="busy" @click="switchAuthMode(authMode === 'register' ? 'login' : 'register')">{{ authMode === 'register' ? '已有账号？返回登录' : '没有账号？注册新账号' }}</button>
      <small v-if="authMode === 'login'">首次管理员登录信息见项目 data/admin-login.txt；团队成员使用管理员创建的账号。</small>
      <small v-else>账号只能使用字母、数字、下划线、点和连字符；密码至少 8 位。</small>
    </form>
  </div>
  <div v-else class="app-shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark"><BookOpen :size="16" /></span><span>团队知识库</span><span class="brand-dot"></span><span class="brand-sub">有出处的团队答案</span></div>
      <div class="top-actions"><span class="connection">{{ health?.diagnostic_mode ? '本地演示' : '本机服务' }}</span><span class="user-chip">{{ user.username }}</span><button v-if="canReviewAnswers" class="icon-button" title="待审核答案" @click="openAnswerReviews"><ClipboardCheck :size="16" /></button><button class="icon-button" title="固定问题集评测" @click="modal = 'evaluation'"><Check :size="16" /></button><button class="icon-button" title="隔离代码修复" @click="modal = 'code-repair'"><Wrench :size="16" /></button><button class="icon-button" title="退出登录" @click="logout"><LogOut :size="16" /></button></div>
    </header>
    <div v-if="health?.diagnostic_mode" class="demo-notice">当前使用 SQLite 本地演示，异步任务由 {{ health.execution === 'celery' ? 'Celery' : '数据库 worker' }} 执行。PostgreSQL / pgvector 的实际联调仍需启动 Docker。</div>
    <div v-if="error" class="global-message error-banner" role="alert"><CircleAlert :size="16" /><span>{{ error }}</span><button class="icon-button" title="关闭错误" @click="error = ''"><X :size="15" /></button></div>
    <div v-if="notice" class="global-message notice" role="status">{{ notice }}<button class="icon-button" title="关闭提示" @click="notice = ''"><X :size="15" /></button></div>
    <main class="workspace" :class="{'has-evidence': !!evidence}">
      <aside class="library-panel">
        <div class="panel-head"><div><span class="eyebrow">Library</span><h2>我的知识库</h2></div><button class="icon-button pale" title="创建知识库" @click="modal = 'corpus'"><Plus :size="17" /></button></div>
        <label class="select-wrap"><span class="sr-only">选择知识库</span><select v-model="selectedCorpusId"><option value="" disabled>选择知识库</option><option v-for="c in corpora" :key="c.id" :value="c.id">{{ c.name }}</option></select><ChevronDown :size="15" /></label>
        <div v-if="activeCorpus" class="library-tools"><span class="role-chip">{{ roleNames[activeCorpus.role] }}</span><button v-if="isAdmin" class="text-button" @click="openMembers"><Users :size="13" />成员</button><button class="icon-button" title="刷新文档" @click="refreshPapers().catch(showError)"><RefreshCw :size="14" /></button></div>
        <label class="search-field"><Search :size="14" /><input v-model="search" placeholder="搜索标题、文件名或标签" aria-label="搜索文档" /></label>
        <div class="filter-row"><select v-model="folderFilter" aria-label="按目录筛选"><option value="">全部目录</option><option v-for="folder in folders" :key="folder">{{ folder }}</option></select><label><input v-model="showVersions" type="checkbox" />历史版本</label></div>
        <div class="library-divider"><span>文档版本</span><span>{{ readyPapers.length }} 可用</span></div>
        <div v-if="loading" class="empty-state"><LoaderCircle class="spin" :size="18" />载入材料</div>
        <div v-else-if="!visiblePapers.length" class="empty-state"><FileText :size="18" /><span>{{ papers.length ? '没有匹配的文档' : '还没有文档' }}</span></div>
        <div v-else class="paper-list">
          <article v-for="p in visiblePapers" :key="p.id" class="document-card" :class="{selected: selectedIds.includes(p.id)}">
            <label class="document-row"><input v-model="selectedIds" type="checkbox" :value="p.id" :disabled="p.status !== 'ready' || !p.is_active" /><FileText :size="16" /><span class="paper-copy"><strong :title="p.title">{{ p.title }}</strong><small>{{ p.format.toUpperCase() }} · {{ p.page_count ? `${p.page_count} 页` : '按章节' }} · {{ (p.byte_size / 1024).toFixed(0) }} KB</small></span><span class="status" :class="{'status-green': p.status === 'ready' && p.is_active, 'status-red': p.status === 'failed'}">{{ p.status === 'ready' && !p.is_active ? '已停用' : statusNames[p.status] }}</span></label>
            <div v-if="p.folder || p.tags.length" class="document-tags"><span v-if="p.folder">{{ p.folder }}</span><span v-for="tag in p.tags" :key="tag">#{{ tag }}</span></div>
            <p v-if="p.error" class="document-error">{{ p.error }}</p>
            <div v-if="canEdit" class="document-actions"><button title="编辑标题、目录和标签" @click="editModal(p, 'edit')"><Pencil :size="12" />编辑</button><button v-if="p.is_active" @click="editModal(p, 'upload')"><Upload :size="12" />更新</button><button v-if="p.status === 'failed'" @click="retryDocument(p)">重试</button><button title="移出文档，保留历史引用" @click="targetDocument = p; modal = 'delete'"><Trash2 :size="12" />移除</button></div>
          </article>
        </div>
        <button v-if="canEdit" class="upload-button" :disabled="busy" @click="uploadModal"><Upload :size="16" />上传文档</button>
        <div class="scope-note"><CircleAlert :size="14" /><span>仅使用当前勾选、已启用的版本。支持 PDF、DOCX、Markdown、TXT、CSV/TSV、JSON/JSONL、HTML 和 XLSX。</span></div>
        <div v-if="history.length" class="history-section"><div class="library-divider"><span><History :size="12" />我的问答记录</span><span>{{ history.length }}</span></div><button v-for="item in history" :key="item.id" class="history-item" :class="{active: run?.id === item.id}" :title="item.question" @click="openRun(item.id).catch(showError)"><span>{{ item.question }}</span><small>{{ statusNames[item.status] }}</small></button></div>
      </aside>
      <section class="conversation-panel">
        <div class="conversation-head"><div><span class="eyebrow">Question Desk</span><h1>从团队文档中<br /><em>找到可靠依据</em></h1></div><div v-if="snapshot" class="snapshot-chip"><Check :size="14" />快照已固定 · {{ snapshot.manifest.length }} 份</div></div>
        <div v-if="!run" class="welcome-block"><p class="lead">查询制度、理解技术方案，或比较不同文档。回答附带具体版本与原文，材料不足时保留缺口。</p><div class="starter-grid"><button @click="question = '新人申请系统权限需要哪些步骤？'"><span>新人上手</span><small>从操作流程开始</small></button><button @click="question = '这些文档对发布流程有哪些要求？'"><span>日常查询</span><small>定位规则和适用条件</small></button><button @click="question = '这些文档有哪些不同或冲突的规定？'"><span>跨文档比较</span><small>检查条件与版本</small></button></div></div>
        <div v-else class="run-view">
          <div class="question-bubble"><span class="question-mark">Q</span><span>{{ run.question }}</span></div>
          <div class="run-status"><span :class="['run-status-dot', `run-${run.status}`]"></span><span role="status">{{ isRunning ? (cancellingId === run.id ? '正在请求取消' : progress?.title) : statusNames[run.status] }}</span><span class="run-time">{{ formatTime(run.created_at) }}</span><button v-if="isRunning" class="cancel-button" :disabled="busy || cancellingId === run.id" @click="cancelRun"><LoaderCircle v-if="cancellingId === run.id" class="spin" :size="12" /><Square v-else :size="12" />{{ cancellingId === run.id ? '取消中' : '取消' }}</button></div>
          <div v-if="isRunning && progress" class="run-progress" aria-label="问答处理进度" aria-busy="true">
            <ol class="progress-steps"><li v-for="(step, index) in progress.steps" :key="step.id" :class="{active: index === progress.index}" :aria-current="index === progress.index ? 'step' : undefined"><span class="progress-step-dot"></span>{{ step.label }}</li></ol>
            <div class="progress-detail"><span>已等待 {{ elapsed }}</span><span v-if="progress.round">第 {{ progress.round }} 轮检索</span><span v-if="progress.revisions">第 {{ progress.revisions }} 次修订</span><span v-if="progress.queriesTotal !== undefined">已检索 {{ progress.queriesDone }}/{{ progress.queriesTotal }} 个查询</span><span v-if="progress.evidence !== undefined">{{ progress.evidence }} 条证据</span><span v-else-if="progress.candidates !== undefined">{{ progress.candidates }} 个候选片段</span></div>
            <p class="progress-connection" role="status"><LoaderCircle class="spin" :size="12" />{{ connectionNames[connection] }}</p>
            <p class="progress-note">完成检查后显示正文；需要人工审核的答案将在审核通过后发布。</p>
          </div>
          <p v-if="run.status === 'cancelled'" class="progress-note cancelled-note">已停止后续处理。已发出的模型请求可能仍需等待返回，其结果不会发布。</p>
          <div v-if="run.status === 'review_pending' && !run.answer" class="answer-review-notice"><ClipboardCheck :size="16" /><span>答案已生成，等待知识库管理员或编辑者审核后发布。</span></div>
          <div v-if="run.status === 'rejected'" class="answer-review-rejected"><div><strong>答案未通过审核</strong><p>{{ run.answer_review?.reason || '审核者未提供可见的理由。' }}</p></div><button class="text-button" :disabled="busy" @click="replay">按审核意见重新生成</button></div>
          <div v-if="run.status === 'failed'" class="failure-block"><CircleAlert :size="18" /><div><strong>运行未完成</strong><p>{{ run.error }}</p><button class="text-button" :disabled="busy" @click="replay">重试本次问题</button></div></div>
          <section v-if="run.status === 'review_pending' && run.answer && canReviewAnswers" class="answer-review-actions"><div class="answer-review-heading"><ClipboardCheck :size="16" /><div><strong>答案待审核</strong><span>通过后才会向知识库成员发布。</span></div></div><label>拒绝理由<textarea v-model="answerReviewReason" rows="2" maxlength="4000" placeholder="如需修改，请写明需要补充或纠正的内容。"></textarea></label><div class="answer-review-buttons"><button class="small-button" :disabled="busy" @click="decideCurrentAnswer('accept')"><Check :size="15" />通过并发布</button><button class="outline-danger-button" :disabled="busy || answerReviewReason.trim().length < 3" @click="decideCurrentAnswer('reject')"><X :size="15" />拒绝并退回</button></div></section>
          <div v-if="run.answer" class="answer-block"><div class="answer-heading"><span class="eyebrow">Answer</span><span :class="['answer-status', `answer-${run.answer.evidence_status}`]">{{ answerNames[run.answer.evidence_status] }}</span></div><p v-if="run.answer.clarification" class="clarification">{{ run.answer.clarification }}</p>
            <article v-for="(claim, index) in run.answer.claims" :key="claim.id" class="claim"><div class="claim-number">{{ String(index + 1).padStart(2, '0') }}</div><div class="claim-body"><p>{{ claim.text }}</p><div v-if="claim.conditions.length" class="condition-list"><span v-for="condition in claim.conditions" :key="condition">{{ condition }}</span></div><div class="citation-row"><button v-for="link in claim.evidence" :key="link.evidence_id + link.relation" class="citation" :class="`relation-${link.relation}`" @click="selectEvidence(link.evidence_id)"><BookOpen :size="13" />{{ link.relation === 'supports' ? '原文支持' : link.relation === 'qualifies' ? '适用条件' : link.relation === 'contradicts' ? '存在分歧' : '背景' }}</button></div></div></article>
            <div v-if="run.answer.gaps.length" class="gaps"><div class="gaps-title"><CircleAlert :size="15" />材料缺口</div><p v-for="gap in run.answer.gaps" :key="gap">{{ gap }}</p></div><div class="usage-line">{{ run.answer.usage.model_calls || 0 }} 次模型调用 · {{ run.answer.usage.retrieval_rounds || 0 }} 轮检索 · {{ Math.round(run.answer.usage.elapsed_seconds || 0) }} 秒</div>
            <form v-if="run.is_owner && run.status === 'completed'" class="feedback-form" @submit.prevent="sendFeedback"><select v-model="feedbackCategory" aria-label="反馈原因" :disabled="busy"><option v-for="(label, value) in feedbackNames" :key="value" :value="value">{{ label }}</option></select><label class="sr-only" for="feedback">反馈</label><input id="feedback" v-model="feedback" :disabled="busy" placeholder="请说明具体问题，不作为事实来源" minlength="2" maxlength="4000" /><button class="text-button" :disabled="busy || feedback.trim().length < 2">提交反馈</button></form>
            <details v-if="run.is_owner && run.status === 'completed'" class="trace-box" @toggle="loadFeedback"><summary>已保存的反馈</summary><p v-for="item in savedFeedback" :key="item.id">{{ feedbackNames[item.category] || item.category }} · {{ item.text }}<br v-if="item.clarification" />{{ item.clarification || '' }}</p></details>
          </div>
          <details v-if="run.is_owner && run.status === 'completed'" class="trace-box" @toggle="loadTrace"><summary>运行记录 · {{ run.request_id.slice(0, 12) }}</summary><div v-if="trace"><p>{{ trace.retrievals.length }} 次检索 · {{ trace.followup_count }} 次补检</p><div v-for="(node, i) in trace.nodes" :key="i" class="trace-node"><span>{{ node.node }}</span><span>{{ node.seconds }} 秒 {{ node.error || '' }}</span></div><details v-for="(item, i) in trace.retrievals" :key="i"><summary>第 {{ item.round }} 轮 · {{ item.query }}</summary><p v-for="c in item.rankings" :key="c.chunk_id">{{ c.title }} · {{ c.chunk_id.slice(0, 8) }}</p></details><p v-if="trace.error" class="form-error">{{ trace.error }}</p></div><p v-else>读取运行记录…</p></details>
          <form v-if="run.is_owner && run.status === 'completed'" class="followup" @submit.prevent="startRun(followup, run?.id)"><MessageSquare :size="16" /><input v-model="followup" placeholder="继续追问，沿用此答案的文档快照" /><button title="发送追问" :disabled="busy || !followup.trim()"><Send :size="16" /></button></form>
        </div>
        <form class="composer" @submit.prevent="startRun(question)"><div class="composer-meta"><span><span class="composer-dot"></span>{{ selectedIds.length }} 份文档已选</span><select v-model="mode" aria-label="问答模式" :disabled="isRunning"><option value="full_agent">完整 Agent</option><option value="baseline_rag">单轮 RAG（对照）</option><option value="without_verifier">关闭核验（消融）</option></select></div><textarea v-model="question" rows="3" :disabled="busy || isRunning || !selectedIds.length" placeholder="问一个关于流程、规则或技术方案的问题…" @keydown.enter.exact.prevent="startRun(question)"></textarea><div class="composer-actions"><span class="composer-foot">Enter 发送 · Shift + Enter 换行</span><button class="send-button" :disabled="busy || isRunning || !question.trim() || !selectedIds.length"><LoaderCircle v-if="busy" class="spin" :size="16" /><Send v-else :size="16" />{{ isRunning ? '处理中' : '开始问答' }}</button></div></form>
      </section>
      <aside class="evidence-panel"><div class="panel-head"><div><span class="eyebrow">Source Evidence</span><h2>回到原文</h2></div></div><div v-if="!evidence" class="evidence-empty"><div class="evidence-placeholder"><BookOpen :size="22" /><span>选择答案里的引用</span><small>查看原文摘录、文档版本和具体位置。</small></div><div class="evidence-principles"><div><strong>版本明确</strong><span>新版本就绪后自动停用旧版。</span></div><div><strong>权限一致</strong><span>检索与引用遵循知识库成员权限。</span></div><div><strong>缺口可见</strong><span>依据不足时保留未知，不补造结论。</span></div></div></div><div v-else class="evidence-detail"><button class="back-button" @click="evidence = null">← 返回</button><div class="source-label"><FileText :size="16" /><span>{{ evidence.title }}</span></div><div class="page-badge">{{ evidence.location_label }}</div><blockquote>{{ evidence.quote }}</blockquote><dl><div><dt>{{ evidence.location_kind === 'mcp_observation' ? '工具调用' : '文档版本' }}</dt><dd>{{ evidence.paper_version_id.slice(0, 12) }}…</dd></div><div><dt>来源块</dt><dd>{{ evidence.block_id.slice(0, 12) }}…</dd></div><div><dt>原文位置</dt><dd>字符 {{ evidence.char_start }}–{{ evidence.char_end }}</dd></div></dl><a class="open-pdf" :href="evidence.file_url" target="_blank" rel="noopener"><BookOpen :size="15" />{{ evidence.location_kind === 'pdf_page' ? '在 PDF 中打开' : evidence.location_kind === 'mcp_observation' ? '查看保存的工具观察' : '下载原文核对' }}</a><small class="evidence-note">{{ evidence.location_kind === 'pdf_page' ? '支持跳页，暂未提供文本高亮。' : evidence.location_kind === 'mcp_observation' ? '这是当时保存的外部观察，未写入正式知识库；记录相关性不等于证明根因。' : '非 PDF 文件按章节、工作表、行号或字段路径定位，不虚构页码。' }}</small></div></aside>
    </main>
    <div v-if="modal" class="modal-overlay" @click.self="modal = null" @keydown.esc="modal = null"><section class="modal-card" :class="{'modal-card-wide': modal === 'code-repair' || modal === 'answer-reviews'}" role="dialog" aria-modal="true" :aria-labelledby="modal === 'code-repair' ? 'code-repair-title' : 'modal-title'"><button v-if="modal !== 'code-repair'" class="icon-button modal-close" title="关闭" @click="modal = null"><X :size="18" /></button>
      <template v-if="modal === 'corpus'"><span class="eyebrow">New Library</span><h2 id="modal-title">创建知识库</h2><form class="stack-form" @submit.prevent="createCorpus"><label>名称<input v-model="corpusName" required maxlength="160" autofocus placeholder="例如：团队技术文档" /></label><button class="small-button" :disabled="busy || !corpusName.trim()">创建</button></form></template>
      <template v-else-if="modal === 'upload' || modal === 'edit'"><span class="eyebrow">Document</span><h2 id="modal-title">{{ modal === 'edit' ? '编辑文档信息' : targetDocument ? '上传新版本' : '上传文档' }}</h2><form class="stack-form" @submit.prevent="saveDocument"><label v-if="modal === 'upload'">文件<input type="file" accept=".pdf,.docx,.md,.markdown,.txt,.csv,.tsv,.json,.jsonl,.html,.htm,.xlsx" required @change="pickFile" /></label><label>标题<input v-model="documentTitle" required maxlength="500" /></label><label>目录<input v-model="documentFolder" maxlength="160" placeholder="例如：新人上手 / 操作流程" /></label><label>标签<input v-model="documentTags" placeholder="以英文逗号分隔，例如：权限, 入职" /></label><p v-if="targetDocument && modal === 'upload'" class="form-note">新版本解析与索引成功后替换旧版；历史答案保留原引用。</p><button class="small-button" :disabled="busy">{{ busy ? '保存中…' : '保存文档' }}</button></form></template>
      <template v-else-if="modal === 'delete'"><h2 id="modal-title">移出文档</h2><p>将“{{ targetDocument?.title }}”移出当前知识库，停止用于新的问答。历史答案及原文引用仍保留。</p><button class="small-button danger-button" :disabled="busy" @click="removeDocument">确认移出</button></template>
      <template v-else-if="modal === 'members'"><span class="eyebrow">Members</span><h2 id="modal-title">共享与成员权限</h2><div class="members-list"><div v-for="m in members" :key="m.id" class="member-row"><span>{{ m.username }}<small v-if="m.is_owner">创建者</small></span><select :value="m.role" :disabled="m.is_owner || busy" aria-label="成员角色" @change="setRole(m, $event)"><option value="admin">管理员</option><option value="editor">编辑者</option><option value="reader">阅读者</option></select><button class="icon-button" :disabled="m.is_owner || busy" title="移除成员" @click="removeMember(m)"><X :size="14" /></button></div></div><p class="form-note">管理员可共享；编辑者可维护文档；阅读者可查看和问答。权限按知识库生效。</p><form class="stack-form" @submit.prevent="addMember"><label>成员账号<input v-model="memberName" required pattern="[A-Za-z0-9_.\-]+" placeholder="已有账号或新账号" /></label><label>新账号密码<input v-model="memberPassword" type="password" autocomplete="new-password" minlength="8" placeholder="已有账号留空；新账号至少 8 位" /></label><label>角色<select v-model="memberRole"><option value="reader">阅读者</option><option value="editor">编辑者</option><option value="admin">管理员</option></select></label><button class="small-button" :disabled="busy">添加成员</button></form></template>
      <template v-else-if="modal === 'evaluation'"><span class="eyebrow">Evaluation</span><h2 id="modal-title">固定问题集评测</h2><p>可选择单轮 RAG、完整 Agent 等配置，使用同一快照和题集逐题交错对照。正式评测仍需独立人工标注问题及原文依据。</p><p class="form-note">下载模板后补全依据。评测脚本支持单独的 Agent 审核与评分；它不代表独立人审或正式业务准确率，也不替代答案发布审批。</p><button class="small-button" @click="downloadTemplate">下载问题标注模板</button></template>
      <template v-else-if="modal === 'answer-reviews'"><span class="eyebrow">Answer Review</span><h2 id="modal-title">待审核答案</h2><div v-if="!answerReviewQueue.length" class="empty-state"><ClipboardCheck :size="18" />当前没有待审核答案</div><div v-else class="answer-review-queue"><button v-for="item in answerReviewQueue" :key="item.id" class="answer-review-row" @click="selectAnswerReview(item)"><span class="answer-review-row-main"><strong>{{ item.question }}</strong><small>{{ formatTime(item.created_at) }} · {{ item.mode }}</small></span><span class="queue-open">查看答案 <ChevronDown :size="14" /></span></button></div></template>
      <template v-else-if="modal === 'code-repair'"><CodeRepairPanel @close="modal = null" /></template>
      <p v-if="error" class="form-error" role="alert">{{ error }}</p>
    </section></div>
  </div>
</template>

<style scoped>
.run-progress { margin-top: 20px; padding: 17px 18px; border: 1px solid var(--line); border-radius: 7px; background: var(--paper); }
.progress-steps { display: flex; flex-wrap: wrap; gap: 10px 17px; list-style: none; padding: 0; margin: 0; }
.progress-steps li { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 11px; }
.progress-step-dot { width: 5px; height: 5px; border: 1px solid #b5beb7; border-radius: 50%; }
.progress-steps li.active { color: var(--green); font-weight: 600; }
.progress-steps li.active .progress-step-dot { background: var(--green); border-color: var(--green); box-shadow: 0 0 0 3px var(--mint); }
.progress-detail { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 17px; color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.progress-connection { display: flex; align-items: center; gap: 6px; margin: 12px 0 0; font-size: 11px; color: var(--green); line-height: 1.6; }
.progress-connection svg { flex: none; }
.progress-note { margin: 10px 0 0; color: var(--muted); font-size: 11px; line-height: 1.65; text-wrap: pretty; }
.cancelled-note { padding: 12px 0; }
@media (max-width: 560px) {
  .run-progress { padding: 14px; }
  .progress-steps { gap: 11px 14px; }
  .run-status { flex-wrap: wrap; }
}
.run-review_pending { background: #cb8b2f; box-shadow: 0 0 0 3px #fbefd9; }
.run-rejected { background: #ad4a40; }
.answer-review-notice, .answer-review-actions, .answer-review-rejected { margin-top: 22px; border: 1px solid #d5e2d7; border-radius: 7px; background: #f3f8f3; padding: 14px 16px; }
.answer-review-notice { display: flex; align-items: center; gap: 9px; color: #376f59; font-size: 12px; }
.answer-review-actions { background: #f8faf7; }
.answer-review-heading { display: flex; align-items: center; gap: 10px; color: #216c59; }
.answer-review-heading div { display: grid; gap: 3px; }
.answer-review-heading strong { color: #202d2a; font-size: 13px; }
.answer-review-heading span { color: #75817a; font-size: 11px; }
.answer-review-actions label { display: grid; gap: 6px; margin-top: 14px; color: #56665d; font-size: 11px; }
.answer-review-actions textarea { width: 100%; min-width: 0; resize: vertical; border: 1px solid #d4dbd3; border-radius: 5px; padding: 9px 10px; background: #fff; color: #202d2a; line-height: 1.55; }
.answer-review-buttons { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 12px; }
.outline-danger-button { display: inline-flex; align-items: center; justify-content: center; gap: 6px; padding: 8px 11px; border: 1px solid #e3c5c1; border-radius: 6px; background: #fff; color: #a4473d; font-size: 12px; }
.answer-review-rejected { display: flex; align-items: center; justify-content: space-between; gap: 16px; border-color: #ead8b1; background: #fff9ec; color: #765a29; }
.answer-review-rejected strong { font-size: 12px; }
.answer-review-rejected p { margin: 5px 0 0; font-size: 12px; line-height: 1.5; white-space: pre-wrap; }
.answer-review-queue { margin-top: 18px; border-top: 1px solid #e2e6df; }
.answer-review-row { display: flex; width: 100%; align-items: center; justify-content: space-between; gap: 16px; text-align: left; border: 0; border-bottom: 1px solid #e2e6df; background: transparent; padding: 15px 4px; color: #202d2a; }
.answer-review-row:hover { background: #f3f7f2; }
.answer-review-row-main { display: grid; min-width: 0; gap: 5px; }
.answer-review-row-main strong { font-size: 13px; font-weight: 600; line-height: 1.5; overflow-wrap: anywhere; }
.answer-review-row-main small { color: #75817a; font-size: 10px; }
.queue-open { flex: none; display: inline-flex; align-items: center; gap: 5px; color: #216c59; font-size: 11px; }
@media (max-width: 560px) {
  .answer-review-rejected { align-items: flex-start; flex-direction: column; }
  .answer-review-row { align-items: flex-start; flex-direction: column; gap: 9px; }
}
</style>
