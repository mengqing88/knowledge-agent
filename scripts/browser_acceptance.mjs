import fs from 'node:fs/promises'
import path from 'node:path'
import assert from 'node:assert/strict'
import {createRequire} from 'node:module'

const require = createRequire(import.meta.url)
const {chromium} = require(process.env.KB_PLAYWRIGHT)
const runtime = process.env.KB_BROWSER_RUNTIME
const memberOnly = process.env.KB_BROWSER_MEMBER_ONLY === '1'
const url = process.env.KB_BROWSER_URL
const credentials = await fs.readFile(path.join(runtime, 'admin-login.txt'), 'utf8')
const password = credentials.match(/密码: (.+)/)[1].trim()
const browser = await chromium.launch({executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true})
const context = await browser.newContext({viewport: {width: 1440, height: 1000}})
const page = await context.newPage()
page.setDefaultTimeout(20000)
const report = {url, browser: 'Microsoft Edge via Playwright', viewport: {width: 1440, height: 1000}, checks: [], console_errors: [], network_failures: [], screenshots: []}
let offline = false
const created = [], eventRequests = []
page.on('pageerror', error => report.console_errors.push(String(error)))
page.on('console', message => {if (message.type() === 'error') report.console_errors.push(message.text())})
page.on('requestfailed', request => report.network_failures.push({path: new URL(request.url()).pathname, error: request.failure()?.errorText, deliberate_offline: offline}))
page.on('request', request => {if (request.url().includes('/events?')) eventRequests.push(request.url())})
page.on('response', async response => {
  if (response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/question-runs' && response.ok()) created.push(await response.json())
})
async function screenshot(name) {
  const target = path.join(runtime, `${name}.png`)
  await page.screenshot({path: target, fullPage: true})
  report.screenshots.push(target)
}
async function ask(question) {
  const pending = page.waitForResponse(r => r.request().method() === 'POST' && new URL(r.url()).pathname === '/api/question-runs')
  await page.locator('.composer textarea').fill(question)
  await page.getByRole('button', {name: '开始问答', exact: true}).click()
  const response = await pending
  assert.equal(response.status(), 202)
  return response.json()
}
async function approve() {
  await page.getByRole('button', {name: '通过并发布', exact: true}).waitFor({timeout: 90000})
  await page.getByRole('button', {name: '通过并发布', exact: true}).click()
  await page.locator('.run-status').getByText('已发布', {exact: true}).waitFor()
}
try {
  await page.goto(url)
  await page.getByLabel('账号', {exact: true}).fill('admin')
  await page.getByLabel('密码', {exact: true}).fill(password)
  await page.getByRole('button', {name: '登录工作台'}).click()
  if (memberOnly) {
    await page.getByRole('button', {name: '成员', exact: true}).click()
    const account = page.getByLabel('成员账号')
    await account.fill('invalid member')
    assert.equal(await account.evaluate(input => input.checkValidity()), false)
    await account.fill('closure-reader-fix')
    assert.equal(await account.evaluate(input => input.checkValidity()), true)
    await page.getByLabel('新账号密码').fill('accept-browser-pass')
    await page.getByRole('button', {name: '添加成员', exact: true}).click()
    await page.locator('.member-row').filter({hasText: 'closure-reader-fix'}).waitFor()
    assert.ok(!report.console_errors.some(message => /Pattern|regular expression|SyntaxError/.test(message)))
    await screenshot('member-pattern-fixed')
    report.checks.push('modern-browser-pattern-rejects-space-accepts-hyphen-and-adds-member')
  } else {
  await page.getByTitle('创建知识库').click()
  await page.getByLabel('名称', {exact: true}).fill('隔离浏览器验收')
  await page.getByRole('button', {name: '创建', exact: true}).click()
  await page.getByRole('button', {name: '成员', exact: true}).click()
  await page.getByLabel('成员账号').fill('closure-reader')
  await page.getByLabel('新账号密码').fill('accept-browser-pass')
  await page.getByRole('button', {name: '添加成员', exact: true}).click()
  await page.locator('.member-row').filter({hasText: 'closure-reader'}).waitFor()
  await screenshot('shared-member')
  await page.getByRole('button', {name: '关闭', exact: true}).click()
  report.checks.push('login-create-library-and-shared-reader')

  await page.getByRole('button', {name: '上传文档', exact: true}).click()
  await page.getByLabel('文件', {exact: true}).setInputFiles({name: 'acceptance.txt', mimeType: 'text/plain', buffer: Buffer.from('On dataset X, M achieved F1 82 and baseline F1 80.\n')})
  await page.getByLabel('标题', {exact: true}).fill('隔离教学材料')
  await page.getByRole('button', {name: '保存文档', exact: true}).click()
  await page.locator('.document-card .status-green').waitFor({timeout: 90000})
  assert.equal(await page.locator('.document-row input').isChecked(), true)
  report.checks.push('upload-celery-indexing-ready-and-selected')

  const first = await ask('M 的 F1 是多少？')
  await page.getByLabel('问答处理进度').waitFor()
  assert.equal(await page.locator('.claim').count(), 0)
  await screenshot('processing-progress')
  await approve()
  await page.getByRole('button', {name: '原文支持', exact: true}).first().click()
  await page.locator('.evidence-detail blockquote').waitFor()
  assert.match(await page.locator('.evidence-detail blockquote').innerText(), /M achieved F1 82/)
  await screenshot('published-citation')
  report.checks.push('progress-without-draft-approval-and-verbatim-citation')

  const followResponse = page.waitForResponse(r => r.request().method() === 'POST' && new URL(r.url()).pathname === '/api/question-runs')
  await page.getByPlaceholder('继续追问，沿用此答案的文档快照').fill('这个数值在哪个数据集上？')
  await page.getByTitle('发送追问').click()
  const follow = await (await followResponse).json()
  assert.equal(follow.snapshot_id, first.snapshot_id)
  await approve()
  report.checks.push('follow-up-uses-same-snapshot')

  await ask('验收取消：M 的 F1 是多少？')
  await page.getByRole('button', {name: '取消', exact: true}).click()
  await page.locator('.run-status').getByText('已取消', {exact: true}).waitFor()
  assert.equal(await page.locator('.claim').count(), 0)
  await screenshot('cancelled')
  report.checks.push('cancel-without-publishing')

  const disconnected = await ask('验收断线：M 的 F1 是多少？')
  await page.getByLabel('问答处理进度').waitFor()
  const beforeReconnect = created.length
  offline = true
  await context.setOffline(true)
  await page.waitForTimeout(2500)
  await context.setOffline(false)
  offline = false
  await approve()
  assert.equal(created.length, beforeReconnect)
  const recoveredEvents = eventRequests.filter(u => u.includes(disconnected.id))
  assert.ok(recoveredEvents.length >= 2 || eventRequests.some(u => u.includes(disconnected.id) && u.includes('stream=false')))
  report.checks.push('offline-reconnect-same-run-without-resubmission')
  report.reconnect_events = recoveredEvents

  const failed = await ask('验收失败重跑：M 的 F1 是多少？')
  await page.getByRole('button', {name: '重试本次问题', exact: true}).waitFor({timeout: 90000})
  await screenshot('failed-before-retry')
  const replayResponse = page.waitForResponse(r => r.request().method() === 'POST' && r.url().includes('/replay'))
  await page.getByRole('button', {name: '重试本次问题', exact: true}).click()
  const replay = await (await replayResponse).json()
  assert.notEqual(replay.id, failed.id)
  assert.equal(replay.snapshot_id, failed.snapshot_id)
  await approve()
  report.checks.push('injected-provider-failure-and-explicit-new-run-retry')
  await page.getByTitle('固定问题集评测').click()
  await page.getByText(/Agent 审核与评分/).waitFor()
  await screenshot('evaluation-agent-label')
  report.checks.push('evaluation-agent-review-disclosure')
  await page.getByRole('button', {name: '关闭', exact: true}).click()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  assert.equal(overflow, false)
  }
  report.status = 'passed'
} catch (error) {
  report.status = 'failed'
  report.error = String(error)
  await screenshot('failure').catch(() => {})
  process.exitCode = 1
} finally {
  report.created_runs = created.map(r => ({id: r.id, question: r.question, snapshot_id: r.snapshot_id}))
  await fs.writeFile(path.join(runtime, memberOnly ? 'member-fix-report.json' : 'browser-report.json'), JSON.stringify(report, null, 2))
  await browser.close()
  console.log(JSON.stringify({status: report.status, checks: report.checks, error: report.error}))
}
