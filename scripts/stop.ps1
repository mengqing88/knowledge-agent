param([string]$RuntimeDir)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'process-state.ps1')
if (-not $RuntimeDir) { $RuntimeDir = Join-Path $projectRoot 'data' }
$processFile = Join-Path $RuntimeDir 'processes.json'
if (-not (Test-Path -LiteralPath $processFile)) { return }
$startupLock = [IO.File]::Open((Join-Path $RuntimeDir 'startup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
try {
$processes = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
$unverified = $false
foreach ($entry in $processes.PSObject.Properties) {
    $state = Get-ProjectProcessState $entry.Value $projectRoot
    if ($state -eq 'owned') {
        Stop-ProjectProcess $entry.Value $projectRoot
    } elseif ($state -eq 'unverified') {
        $unverified = $true
    }
}
if ($unverified) { throw '存在身份无法确认的旧进程；未终止该进程，记录保留供核对。' }
Remove-Item -LiteralPath $processFile
} finally { $startupLock.Dispose() }
