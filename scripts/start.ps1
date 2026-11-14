param([switch]$SkipDatabase, [switch]$Demo, [switch]$AgentUpgrades,
      [string]$RuntimeDir, [ValidateRange(1024,65535)][int]$Port = 8000)
$ErrorActionPreference = 'Stop'
if ($Demo -and $AgentUpgrades) { throw '-AgentUpgrades 需要实际 Redis/Milvus/PostgreSQL，请勿与 -Demo 同时使用。' }
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '请先运行 uv sync。'
}
. (Join-Path $PSScriptRoot 'process-state.ps1')
if (-not $RuntimeDir) { $RuntimeDir = Join-Path $projectRoot 'data' }
$RuntimeDir = [IO.Path]::GetFullPath($RuntimeDir)
New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
$env:KB_RUNTIME_DIR = $RuntimeDir
$env:KB_API_PORT = [string]$Port
$processFile = Join-Path $RuntimeDir 'processes.json'
$startupLock = [IO.File]::Open((Join-Path $RuntimeDir 'startup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
try {
if (Test-Path -LiteralPath $processFile) {
    $record = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    foreach ($entry in $record.PSObject.Properties) {
        $state = Get-ProjectProcessState $entry.Value $projectRoot
        if ($state -in @('owned','unverified')) { throw '项目进程仍在运行或身份无法确认，请先核对并停止，防止重复启动。' }
    }
    Move-Item -LiteralPath $processFile -Destination (Join-Path $RuntimeDir ('processes.stale.' + [Guid]::NewGuid().ToString('N') + '.json'))
    Write-Host '已核实旧记录均为失效 PID 或其他进程；归档记录后继续启动。'
}
if ($Demo) {
    $env:DATABASE_URL = 'sqlite:///' + ((Join-Path $RuntimeDir 'team-demo.db') -replace '\\','/')
    $env:CELERY_BROKER_URL = 'sqla+sqlite:///' + ((Join-Path $RuntimeDir 'celery-demo.db') -replace '\\','/')
    $env:STORAGE_DIR = Join-Path $RuntimeDir 'team-pdfs'
    Write-Host '本地演示：SQLite + Celery SQLAlchemy broker，不使用 Milvus，不代表正式存储验收。'
} elseif (-not $SkipDatabase) {
    docker compose up -d --wait db redis milvus
    if ($LASTEXITCODE -ne 0) { throw 'PostgreSQL、Redis 或 Milvus 未启动，请检查 Docker Desktop。' }
}
$env:TASK_BACKEND = 'celery'
if ($AgentUpgrades) {
    $upgradeProfile = & $pythonPath -X utf8 scripts/agent_profile.py | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw '模型角色配置生成失败，未启动应用。' }
    foreach ($upgradeSetting in $upgradeProfile.PSObject.Properties) {
        [Environment]::SetEnvironmentVariable($upgradeSetting.Name, [string]$upgradeSetting.Value, 'Process')
    }
    Write-Host '启用查询改写、项目只读日志 MCP、角色模型和 Redis 共享配额；30 RPM/200000 TPM 是本地保守策略，不代表供应商实际限额。'
}
& $pythonPath -X utf8 scripts/manage.py migrate
if ($LASTEXITCODE -ne 0) { throw '迁移失败，未启动应用。' }
if (-not $Demo) {
    & $pythonPath -X utf8 scripts/manage.py migrate-vectors
    if ($LASTEXITCODE -ne 0) { throw 'Milvus 向量回填失败，原数据保留，未启动应用。' }
}
& $pythonPath -X utf8 scripts/manage.py init-admin
if ($LASTEXITCODE -ne 0) { throw '账号初始化失败。' }
$started = @{}
try {
    foreach ($serviceName in @('api', 'celery', 'beat')) {
        $managePath = '"' + (Join-Path $PSScriptRoot 'manage.py') + '"'
        $service = Start-Process -FilePath $pythonPath -ArgumentList @('-X','utf8',$managePath,$serviceName) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RuntimeDir "$serviceName.log") -RedirectStandardError (Join-Path $RuntimeDir "$serviceName-error.log")
        $started[$serviceName] = $service.Id
        $started[$serviceName] = New-ProjectProcessRecord $service
    }
} finally {
    $started | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding utf8
}
Write-Host "工作台: http://127.0.0.1:$Port/  接口: http://127.0.0.1:$Port/docs"
Write-Host "登录信息: $RuntimeDir/admin-login.txt（仅本机保存）。停止时使用同一 RuntimeDir。"
} finally { $startupLock.Dispose() }
