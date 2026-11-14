# Shared process identity check. Legacy PID-only records never authorize a kill.
function Get-ProjectProcessState($Record, [string]$ProjectRoot) {
    $processId = if ($Record -is [ValueType]) { [int]$Record } else { [int]$Record.pid }
    $info = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction Stop
    if (-not $info) { return 'dead' }
    $expectedPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (-not $info.ExecutablePath -or -not $info.CommandLine) { return 'unverified' }
    if ($info.ExecutablePath -ne $expectedPython -or $info.CommandLine -notlike '*manage.py*') { return 'unrelated' }
    if ($Record -is [ValueType]) { return 'unverified' }
    if (-not $Record.creation_utc -or -not $Record.command_line) { return 'unverified' }
    $created = $info.CreationDate.ToUniversalTime().ToString('o')
    if ($created -ne $Record.creation_utc -or $info.CommandLine -ne $Record.command_line) { return 'unrelated' }
    return 'owned'
}

function New-ProjectProcessRecord($Process) {
    $info = Get-CimInstance Win32_Process -Filter "ProcessId = $($Process.Id)" -ErrorAction Stop
    if (-not $info) { throw '启动的服务提前退出，请检查运行日志。' }
    return @{pid=$Process.Id; creation_utc=$info.CreationDate.ToUniversalTime().ToString('o'); command_line=$info.CommandLine}
}

function Stop-ProjectProcess($Record, [string]$ProjectRoot) {
    if ((Get-ProjectProcessState $Record $ProjectRoot) -ne 'owned') { return }
    # Windows venv python.exe is a launcher. Its real interpreter child must
    # also be stopped, while the validated launcher is still alive.
    $processId = [int]$Record.pid
    $managePath = Join-Path $ProjectRoot 'scripts\manage.py'
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $processId" -ErrorAction Stop
    foreach ($child in $children) {
        if (-not $child.CommandLine -or -not $child.ExecutablePath -or
            $child.Name -notlike 'python*.exe' -or
            $child.CommandLine.IndexOf($managePath, [StringComparison]::OrdinalIgnoreCase) -lt 0 -or
            $child.CreationDate.ToUniversalTime() -lt [datetime]::Parse($Record.creation_utc).ToUniversalTime()) { continue }
        $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($child.ProcessId)" -ErrorAction Stop
        if ($current -and $current.ParentProcessId -eq $processId -and
            $current.CreationDate -eq $child.CreationDate -and $current.CommandLine -eq $child.CommandLine -and
            $current.ExecutablePath -eq $child.ExecutablePath -and
            (Get-ProjectProcessState $Record $ProjectRoot) -eq 'owned') {
            Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
        }
    }
    if ((Get-ProjectProcessState $Record $ProjectRoot) -eq 'owned') {
        Stop-Process -Id $processId -ErrorAction SilentlyContinue
    }
}
