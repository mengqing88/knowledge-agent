import assert from 'node:assert/strict'
import {readFile} from 'node:fs/promises'
import {test} from 'node:test'
import ts from 'typescript'

// Use the project's compiler because api.ts contains a parameter property,
// which Node's strip-only TypeScript loader cannot execute.
const source = await readFile(new URL('../src/api/client.ts', import.meta.url), 'utf8')
const {outputText} = ts.transpileModule(source, {compilerOptions: {
  module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022,
}})
const {api, ApiError} = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)

for (const decision of ['accept', 'reject']) {
  test(`answer ${decision} submits the hash of the displayed answer`, async t => {
    const seenHash = 'a'.repeat(64)
    const reason = decision === 'reject' ? '缺少证据，请补充' : undefined
    t.mock.method(globalThis, 'fetch', async (url, init) => {
      assert.equal(url, '/api/answers/answer1/approval')
      assert.equal(init.method, 'POST')
      assert.equal(init.credentials, 'same-origin')
      assert.deepEqual(JSON.parse(init.body), {
        answer_sha256: seenHash, decision, ...(reason ? {reason} : {}),
      })
      return new Response(JSON.stringify({decision, status: decision === 'accept' ? 'completed' : 'rejected'}))
    })
    assert.equal((await api.decideAnswer('answer1', seenHash, decision, reason)).decision, decision)
  })
}

test('stale approval surfaces the conflict without fetching a new hash or retrying', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({
    detail: '答案内容已变化，请重新查看后再审核', code: 'answer_content_changed',
  }), {status: 409}))
  await assert.rejects(api.decideAnswer('answer1', 'b'.repeat(64), 'accept'), error =>
    error instanceof ApiError && error.status === 409 && error.message.includes('重新查看'))
  assert.equal(fetch.mock.callCount(), 1)
})
