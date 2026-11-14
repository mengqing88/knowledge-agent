<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { Check, CircleAlert, Clock3, Download, GitBranch, LoaderCircle, Plus, RotateCcw, ShieldCheck, Wrench, X } from 'lucide-vue-next'
import { api, ApiError } from './api'
import type { CodeRepairTask, CodeRepairReview, RepairExperience } from './types'

const emit = defineEmits<{close: []}>()
const tasks = ref<CodeRepairTask[]>([]), experiences = ref<RepairExperience[]>([])
const selected = ref<CodeRepairTask | null>(null), availableFiles = ref<string[]>([]), selectedFiles = ref<string[]>([])
const reviewRecord = ref<CodeRepairReview | null>(null)
const repositoryPath = ref(''), baseRef = ref('HEAD'), objective = ref(''), rejectionReason = ref('')
const feedbackCategory = ref('unclear'), rootCauseLayer = ref('unknown'), uncertainty = ref('')
const busy = ref(false), error = ref(''), notice = ref(''), showExperiences = ref(false)
const statusNames: Record<string, string> = {
  ready: '待选择文件', generating: '生成补丁中', patched: '待运行测试', testing: '运行测试中',
  review_pending: '等待人工审核', accepted: '已接受', rejected: '已拒绝', failed: '失败并已恢复',
}
const canPropose = computed(() => selected.value?.status === 'ready' && selectedFiles.value.length > 0)
const canTest = computed(() => selected.value?.status === 'patched')
const canReview = computed(() => selected.value?.status === 'review_pending')

async function action(fn: () => Promise<void>) {
  if (busy.value) return
  busy.value = true; error.value = ''; notice.value = ''
  try { await fn() } catch (cause) {
    error.value = cause instanceof ApiError ? cause.message : cause instanceof Error ? cause.message : '操作失败'
    await refreshTasks()
  } finally { busy.value = false }
}

async function refreshTasks() {
  tasks.value = await api.codeTasks()
  if (selected.value) selected.value = tasks.value.find(item => item.id === selected.value?.id) || null
}

async function loadTasks() {
  await action(async () => {
    await refreshTasks()
    experiences.value = await api.codeExperiences()
    if (!selected.value && tasks.value.length) await openTask(tasks.value[0])
  })
}

async function openTask(task: CodeRepairTask) {
  selected.value = task
  reviewRecord.value = null
  availableFiles.value = []
  selectedFiles.value = [...(task.selected_files || [])]
  if (task.status === 'ready') availableFiles.value = (await api.codeFiles(task.id)).files
  if (task.status === 'accepted' || task.status === 'rejected') reviewRecord.value = await api.codeReview(task.id)
}

async function createTask() {
  await action(async () => {
    const task = await api.createCodeTask(repositoryPath.value.trim(), baseRef.value.trim() || 'HEAD', objective.value.trim())
    repositoryPath.value = ''; objective.value = ''
    await refreshTasks(); await openTask(task)
    notice.value = '已固定 Git 基线并创建隔离工作副本。'
  })
}

async function generatePatch() {
  await action(async () => {
    if (!selected.value) return
    selected.value = await api.proposeCodePatch(selected.value.id, selectedFiles.value)
    await refreshTasks()
    notice.value = '补丁已应用到任务副本，验收文件保持不变。'
  })
}

async function runTests() {
  await action(async () => {
    if (!selected.value) return
    selected.value = await api.testCodePatch(selected.value.id)
    await refreshTasks()
    notice.value = selected.value.status === 'review_pending'
      ? '固定验收命令通过，可以审核此补丁。'
      : '测试未通过；任务副本已恢复，失败输出已保留。'
  })
}

async function review(decision: 'accept' | 'reject') {
  await action(async () => {
    if (!selected.value) return
    const result = await api.reviewCodePatch(
      selected.value.id, decision, decision === 'reject' ? rejectionReason.value.trim() : undefined,
      feedbackCategory.value, rootCauseLayer.value, uncertainty.value.trim() || undefined,
    )
    rejectionReason.value = ''; uncertainty.value = ''
    await refreshTasks()
    const reviewed = tasks.value.find(item => item.id === result.task_id)
    if (reviewed) await openTask(reviewed)
    experiences.value = await api.codeExperiences()
    notice.value = decision === 'accept'
      ? '已接受该补丁；已将此版本及通过证据保存为经验，未合并到源仓库。'
      : reviewed?.workspace_state === 'restored'
        ? '拒绝原话和分类已保存，任务副本已恢复到固定基线。'
        : '拒绝原话和分类已保存；副本清理待重试。'
  })
}

async function retryCleanup() {
  await action(async () => {
    if (!selected.value) return
    selected.value = await api.cleanupRejectedCodeTask(selected.value.id)
    await refreshTasks()
    notice.value = '已从固定基线恢复任务副本；拒绝反馈保留在审核记录中。'
  })
}

onMounted(() => { void loadTasks() })
</script>

<template>
  <section class="code-repair-panel" aria-labelledby="code-repair-title">
    <header class="repair-header">
      <div><span class="eyebrow">Code Repair</span><h2 id="code-repair-title">隔离代码修复</h2></div>
      <button class="icon-button" title="关闭代码修复" @click="emit('close')"><X :size="17" /></button>
    </header>
    <p class="repair-intro">从干净 Git commit 创建独立 worktree。通过 Docker 隔离测试后，由你按补丁和测试证明摘要接受或拒绝。</p>
    <div class="repair-guard"><ShieldCheck :size="16" /><span>测试在禁网、只读容器中运行，限制 CPU、内存和进程数；容器只看到去除 Git 元数据与常见密钥文件的已跟踪文件。镜像需由服务端配置并预先准备。</span></div>
    <div v-if="error" class="repair-message repair-error" role="alert"><CircleAlert :size="15" />{{ error }}</div>
    <div v-if="notice" class="repair-message repair-notice" role="status"><Check :size="15" />{{ notice }}</div>

    <div class="repair-layout">
      <aside class="repair-list">
        <div class="repair-section-head"><strong>修复任务</strong><button class="icon-button pale" title="新建任务" @click="selected = null; showExperiences = false"><Plus :size="15" /></button></div>
        <button v-for="task in tasks" :key="task.id" class="repair-task" :class="{active: selected?.id === task.id}" @click="action(() => openTask(task))">
          <span class="repair-task-title">{{ task.objective }}</span><small>{{ task.repository }} · {{ statusNames[task.status] || task.status }}</small>
        </button>
        <button class="repair-experience-toggle" @click="showExperiences = !showExperiences; selected = null">{{ showExperiences ? '返回任务' : '已验收经验 · ' + experiences.length }}</button>
        <div v-if="showExperiences" class="repair-experiences">
          <article v-for="item in experiences" :key="item.id"><strong>{{ item.summary }}</strong><small>{{ item.verified_command }}</small><code>{{ item.artifact_sha256.slice(0, 16) }}</code></article>
          <p v-if="!experiences.length" class="repair-empty">接受的结果会记录在这里。</p>
        </div>
      </aside>

      <div class="repair-detail">
        <form v-if="!selected && !showExperiences" class="repair-create" @submit.prevent="createTask">
          <h3>创建修复任务</h3>
          <label>仓库相对路径<input v-model="repositoryPath" required placeholder="例如：service-api" /></label>
          <label>基线引用<input v-model="baseRef" required maxlength="255" /></label>
          <label>修复目标<textarea v-model="objective" required minlength="5" maxlength="4000" rows="3" placeholder="描述可复现的问题和预期行为" /></label>
          <p class="form-note">仓库需位于服务器配置的 CODE_REPOSITORY_ROOT 下且工作区干净。未提交内容不会进入任务基线。</p>
          <button class="small-button" :disabled="busy || !repositoryPath.trim() || objective.trim().length < 5"><LoaderCircle v-if="busy" class="spin" :size="15" /><GitBranch v-else :size="15" />创建固定基线副本</button>
        </form>

        <div v-else-if="showExperiences" class="repair-create">
          <h3>已接受的修复经验</h3>
          <article v-for="item in experiences" :key="item.id" class="experience-detail">
            <strong>{{ item.summary }}</strong><p>{{ item.objective }}</p>
            <small>{{ item.verified_command }} · {{ item.artifact_sha256 }}</small>
          </article>
        </div>

        <div v-else-if="selected" class="repair-current">
          <div class="repair-task-heading"><div><span class="eyebrow">{{ selected.repository }} · {{ statusNames[selected.status] || selected.status }}</span><h3>{{ selected.objective }}</h3></div><a v-if="selected.patch_sha256" class="repair-icon-link" title="下载补丁" :href="'/api/code-repair/tasks/' + selected.id + '/patch'"><Download :size="16" /></a></div>
          <div class="repair-baseline"><GitBranch :size="14" /><span>{{ selected.base_ref }}</span><code>{{ selected.base_commit }}</code></div>
          <p v-if="selected.summary" class="repair-summary">{{ selected.summary }}</p>
          <div v-if="selected.status === 'ready'" class="repair-file-picker">
            <strong>允许修改的源码</strong>
            <label v-for="file in availableFiles" :key="file"><input v-model="selectedFiles" type="checkbox" :value="file" />{{ file }}</label>
            <p v-if="!availableFiles.length" class="repair-empty">基线中没有可选择的 Python 源文件。</p>
            <button class="small-button" :disabled="busy || !canPropose" @click="generatePatch"><LoaderCircle v-if="busy" class="spin" :size="15" /><Wrench v-else :size="15" />生成并校验补丁</button>
          </div>
          <div v-if="selected.patch_sha256" class="artifact-digest"><span>待审核产物 SHA-256</span><code>{{ selected.patch_sha256 }}</code></div>
          <pre v-if="selected.patch" class="repair-patch">{{ selected.patch }}</pre>
          <button v-if="canTest" class="small-button" :disabled="busy" @click="runTests"><LoaderCircle v-if="busy" class="spin" :size="15" /><Check v-else :size="15" />运行固定 pytest 验收</button>
          <div v-if="selected.test_environment" class="repair-test-environment"><span>测试镜像</span><code>{{ selected.test_environment.image_id }}</code><span>Docker Engine</span><code>{{ selected.test_environment.docker_engine.server_version }} · {{ selected.test_environment.docker_engine.context }}</code><span>验证摘要</span><code>{{ selected.verification_sha256 || '未通过测试' }}</code></div>
          <div v-if="selected.test_output" class="repair-test-output"><div><strong>pytest</strong><span :class="selected.test_exit_code === 0 ? 'test-pass' : 'test-fail'">{{ selected.test_exit_code === 0 ? '通过' : '退出码 ' + selected.test_exit_code }}</span></div><pre>{{ selected.test_output }}</pre></div>
          <div v-if="canReview" class="repair-review">
            <strong><ShieldCheck :size="15" />人工验收</strong>
            <p>审核绑定补丁、基线、目标、测试镜像、固定命令和测试输出摘要；服务端审核前重新核对。</p>
            <label>拒绝理由<textarea v-model="rejectionReason" rows="2" minlength="3" maxlength="4000" placeholder="拒绝时必填，作为用户反馈保存" /></label>
            <label>反馈类别<select v-model="feedbackCategory"><option value="unclear">尚不明确</option><option value="preference">用户偏好</option><option value="factual_error">事实错误</option><option value="execution_failure">执行失败</option><option value="missing_scenario">遗漏场景</option><option value="requirement_change">需求变化</option><option value="unable_to_complete">当前无法完成</option></select></label>
            <label>可能根因<select v-model="rootCauseLayer"><option value="unknown">未知 / 待确认</option><option value="prompt">提示词</option><option value="knowledge">知识资料</option><option value="tool_contract">工具契约</option><option value="authorization">权限边界</option><option value="workflow">业务流程</option><option value="code">代码实现</option><option value="environment">运行环境</option></select></label>
            <label>尚不确定的事项<textarea v-model="uncertainty" rows="2" maxlength="2000" placeholder="可留空；不把猜测记录成结论" /></label>
            <div><button class="small-button" :disabled="busy" @click="review('accept')"><Check :size="15" />接受并沉淀经验</button><button class="small-button danger-button" :disabled="busy || rejectionReason.trim().length < 3" @click="review('reject')"><RotateCcw :size="15" />拒绝并恢复副本</button></div>
          </div>
          <div v-if="['accepted', 'rejected', 'failed'].includes(selected.status)" class="repair-terminal">
            <Clock3 :size="15" /><span>{{ selected.status === 'accepted' ? '已接受并保存经验；源仓库仍未改变。' : selected.status === 'rejected' ? '已拒绝；任务副本回到固定基线。' : selected.error || '任务已失败并恢复。' }}</span>
          </div>
          <dl v-if="reviewRecord" class="repair-review-record">
            <div><dt>审核决定</dt><dd>{{ reviewRecord.decision === 'reject' ? '拒绝' : '接受' }} · {{ reviewRecord.feedback_category }}</dd></div>
            <div><dt>反馈原话</dt><dd>{{ reviewRecord.reason || '未填写' }}</dd></div>
            <div><dt>可能根因</dt><dd>{{ reviewRecord.root_cause_layer }} · {{ reviewRecord.uncertainty || '未记录不确定项' }}</dd></div>
            <div v-if="reviewRecord.bundle_sha256"><dt>审核内容摘要</dt><dd><code>{{ reviewRecord.bundle_sha256 }}</code></dd></div>
          </dl>
          <button v-if="selected.status === 'rejected' && selected.workspace_state !== 'restored'" class="small-button danger-button" :disabled="busy" @click="retryCleanup"><RotateCcw :size="15" />重试副本清理</button>
          <p v-if="selected.error && selected.status !== 'failed'" class="repair-error-text">{{ selected.error }}</p>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.code-repair-panel{display:flex;flex-direction:column;gap:14px;min-height:65vh;color:var(--ink)}
:global(.modal-card.modal-card-wide){max-width:1080px;padding:24px}
.repair-header,.repair-section-head,.repair-task-heading,.repair-baseline,.repair-test-output>div{display:flex;align-items:center;justify-content:space-between;gap:12px}
.repair-header h2{font:24px Georgia,"SimSun",serif;color:var(--green-deep);margin:0}
.repair-intro{font-size:12px;line-height:1.7;color:var(--muted);margin:0}
.repair-test-environment{display:grid;grid-template-columns:100px minmax(0,1fr);gap:6px 10px;padding:9px 10px;background:var(--mint);font-size:10px}
.repair-test-environment span{color:var(--muted)}
.repair-test-environment code{overflow-wrap:anywhere}
.repair-review-record{display:grid;gap:7px;margin:0;padding:10px 12px;border-left:2px solid var(--green);background:#f6f8f3;font-size:10px}
.repair-review-record>div{display:grid;grid-template-columns:90px minmax(0,1fr);gap:10px}
.repair-review-record dt{color:var(--muted)}
.repair-review-record dd{margin:0;overflow-wrap:anywhere}
.repair-guard{display:flex;align-items:flex-start;gap:9px;padding:10px 12px;background:var(--mint);color:var(--green-deep);font-size:11px;line-height:1.65}
.repair-guard svg{flex:none;margin-top:1px}
.repair-message{display:flex;align-items:center;gap:8px;padding:9px 11px;font-size:11px;line-height:1.6}
.repair-error{background:var(--red-bg);color:var(--red)}
.repair-notice{background:#eef5ef;color:var(--green)}
.repair-layout{display:grid;grid-template-columns:minmax(190px,245px) minmax(0,1fr);gap:18px;min-height:0;flex:1}
.repair-list{border-right:1px solid var(--line);padding-right:14px;min-width:0;max-height:55vh;overflow:auto}
.repair-section-head{padding-bottom:10px;border-bottom:1px solid var(--line);font-size:12px}
.repair-task{display:flex;flex-direction:column;align-items:flex-start;gap:5px;width:100%;padding:10px 8px;border:0;border-bottom:1px solid var(--line);background:transparent;text-align:left;color:var(--ink)}
.repair-task.active,.repair-task:hover{background:var(--mint)}
.repair-task-title{font-size:11px;line-height:1.5;overflow-wrap:anywhere}
.repair-task small{font-size:10px;color:var(--muted)}
.repair-experience-toggle{width:100%;margin-top:12px;padding:8px;border:1px solid var(--line);background:var(--paper);color:var(--green);font-size:11px;text-align:left}
.repair-experiences article,.experience-detail{display:grid;gap:6px;padding:10px 4px;border-bottom:1px solid var(--line);font-size:11px}
.repair-experiences small,.experience-detail small{font-size:10px;color:var(--muted);overflow-wrap:anywhere}
.repair-experiences code{font-size:10px}
.repair-detail{min-width:0;max-height:55vh;overflow:auto;padding-right:4px}
.repair-create{display:grid;gap:13px;max-width:620px}
.repair-create h3,.repair-task-heading h3{font-size:15px;line-height:1.5;margin:0;overflow-wrap:anywhere}
.repair-create label,.repair-review label{display:grid;gap:6px;font-size:11px;color:var(--muted)}
.repair-create input,.repair-create textarea,.repair-review textarea{width:100%;min-width:0;border:1px solid var(--line);border-radius:5px;padding:9px;background:var(--paper);color:var(--ink);font:inherit;font-size:12px;line-height:1.6}
.repair-create textarea,.repair-review textarea{resize:vertical}
.repair-create .form-note{margin:0}
.repair-current{display:grid;gap:14px}
.repair-task-heading{align-items:flex-start}
.repair-icon-link{display:grid;place-items:center;width:34px;height:34px;flex:none;color:var(--green);border:1px solid var(--line)}
.repair-baseline{justify-content:flex-start;flex-wrap:wrap;color:var(--muted);font-size:10px}
.repair-baseline code,.artifact-digest code{overflow-wrap:anywhere}
.repair-summary{margin:0;color:var(--ink);font-size:12px;line-height:1.7}
.repair-file-picker{display:grid;gap:8px;padding:12px;border:1px solid var(--line)}
.repair-file-picker>strong{font-size:11px}
.repair-file-picker label{display:flex;align-items:center;gap:8px;font:11px/1.5 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}
.repair-file-picker input{accent-color:var(--green)}
.repair-empty{font-size:11px;color:var(--muted);margin:2px 0}
.artifact-digest{display:grid;gap:5px;padding:10px;background:var(--mint);font-size:10px;color:var(--green-deep)}
.artifact-digest code{font-size:10px}
.repair-patch,.repair-test-output pre{max-height:240px;overflow:auto;margin:0;padding:12px;background:#202d2a;color:#f2f4ef;border-radius:4px;font:10px/1.6 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
.repair-test-output{display:grid;gap:7px}
.repair-test-output>div{font-size:11px}
.test-pass{color:var(--green)}.test-fail{color:var(--red)}
.repair-review{display:grid;gap:10px;padding:13px;border:1px solid #d6e3d9;background:#f8fbf8}
.repair-review>strong{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--green-deep)}
.repair-review>p{margin:0;font-size:11px;line-height:1.6;color:var(--muted)}
.repair-review>div{display:flex;gap:8px;flex-wrap:wrap}
.repair-terminal{display:flex;align-items:flex-start;gap:8px;padding:10px;background:var(--mint);font-size:11px;line-height:1.6;color:var(--green-deep)}
.repair-terminal svg{flex:none;margin-top:1px}
.repair-error-text{font-size:11px;color:var(--red);overflow-wrap:anywhere}
@media(max-width:700px){:global(.modal-card.modal-card-wide){padding:18px}.code-repair-panel{min-height:0}.repair-layout{grid-template-columns:1fr}.repair-list{max-height:180px;padding:0 0 12px;border:0;border-bottom:1px solid var(--line)}.repair-detail{max-height:48vh}.repair-task-heading{align-items:flex-start}.repair-review>div{display:grid}.repair-review .small-button{width:100%}}
</style>
