# Controlled Walk Forward regeneration wrapper.
# Do NOT set $ErrorActionPreference = "Stop": Python/uv INFO on stderr
# must not become NativeCommandError.
# Native child exit code is taken from $LASTEXITCODE after file redirection.

$Repo = "D:\bss\CQROS"
Set-Location $Repo

$Evidence = Join-Path $Repo "reports\walk_forward\controlled_regeneration_2026"
$LogDir = Join-Path $Evidence "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:PYTHONUNBUFFERED = "1"

$RssLog = Join-Path $LogDir "rss_sampler.csv"
$ProgressLog = Join-Path $Evidence "progress.log"
$MetaLog = Join-Path $Evidence "run_meta.json"
$WfRoot = Join-Path $Repo "data\walk_forward\default\binance\usdt_perpetual"

"timestamp_utc,pid,process_name,working_set_mb,private_mb,timeframe" | Set-Content -Encoding utf8 $RssLog
"Controlled Walk Forward regeneration started" | Set-Content -Encoding utf8 $ProgressLog

$Timeframes = @("1d", "4h", "1h", "15m", "5m")
$MemoryStopMb = 3200
$OverallStarted = Get-Date
$OverallStartedUtc = [DateTime]::UtcNow.ToString("o")
$Results = @()
$OverallExit = 0
$StoppedEarly = $false
$StopReason = ""

function Write-ProgressLine([string]$Line) {
    Add-Content -Encoding utf8 -Path $ProgressLog -Value $Line
    Write-Host $Line
}

Write-ProgressLine ("START {0}" -f $OverallStartedUtc)

foreach ($Tf in $Timeframes) {
    if ($StoppedEarly) { break }

    $StdoutPath = Join-Path $LogDir ("generate_walk_forward_{0}_stdout.log" -f $Tf)
    $StderrPath = Join-Path $LogDir ("generate_walk_forward_{0}_stderr.log" -f $Tf)
    $CombinedPath = Join-Path $LogDir ("generate_walk_forward_{0}.log" -f $Tf)

    $Sampler = Start-Job -ScriptBlock {
        param($RssLogPath, $ProgressPath, $WfRootPath, $TfName, $MemoryStopMb)
        $lastWrite = $null
        while ($true) {
            $stamp = [DateTime]::UtcNow.ToString("o")
            $cim = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
                $_.Name -match '^(python|uv|uvx)(\.exe)?$' -and
                $_.CommandLine -match 'generate_walk_forward'
            }
            foreach ($item in $cim) {
                $proc = Get-Process -Id $item.ProcessId -ErrorAction SilentlyContinue
                if (-not $proc) { continue }
                $ws = [Math]::Round($proc.WorkingSet64 / 1MB, 2)
                $priv = [Math]::Round($proc.PrivateMemorySize64 / 1MB, 2)
                Add-Content -Encoding utf8 -Path $RssLogPath -Value (
                    "{0},{1},{2},{3},{4},{5}" -f $stamp, $proc.Id, $proc.ProcessName, $ws, $priv, $TfName
                )
                if ($priv -ge $MemoryStopMb) {
                    Add-Content -Encoding utf8 -Path $ProgressPath -Value (
                        "{0} MEMORY_STOP tf={1} pid={2} private_mb={3} threshold_mb={4}" -f $stamp, $TfName, $proc.Id, $priv, $MemoryStopMb
                    )
                    try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { }
                }
            }
            $panel = Join-Path (Join-Path $WfRootPath $TfName) "2026.parquet"
            if (Test-Path $panel) {
                $item = Get-Item $panel
                $key = "{0}|{1}" -f $item.Length, $item.LastWriteTimeUtc.Ticks
                if ($lastWrite -ne $key) {
                    $lastWrite = $key
                    Add-Content -Encoding utf8 -Path $ProgressPath -Value (
                        "{0} WF_WRITE tf={1} bytes={2} mtime_utc={3:o}" -f $stamp, $TfName, $item.Length, $item.LastWriteTimeUtc
                    )
                }
            }
            Start-Sleep -Seconds 5
        }
    } -ArgumentList $RssLog, $ProgressLog, $WfRoot, $Tf, $MemoryStopMb

    $Started = Get-Date
    $StartedUtc = [DateTime]::UtcNow.ToString("o")
    Write-ProgressLine ("TF_START {0} {1}" -f $Tf, $StartedUtc)

    $ArgList = @(
        "run", "python", "-m", "cqros.cli.generate_walk_forward",
        "--manager", "default",
        "--timeframes", $Tf,
        "--years", "2026",
        "--overwrite",
        "--workers", "1",
        "--storage-root", "data",
        "--verbose"
    )
    $CommandText = "uv " + ($ArgList -join " ")

    # File redirection keeps Python stderr out of the PowerShell error stream.
    & uv @ArgList 1> $StdoutPath 2> $StderrPath
    $ExitCode = $LASTEXITCODE
    if ($null -eq $ExitCode) { $ExitCode = 1 }

    Stop-Job $Sampler -ErrorAction SilentlyContinue
    Receive-Job $Sampler -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $Sampler -Force -ErrorAction SilentlyContinue

    $Ended = Get-Date
    $EndedUtc = [DateTime]::UtcNow.ToString("o")
    $DurationSeconds = [Math]::Round(($Ended - $Started).TotalSeconds, 3)

    $stdoutText = ""
    $stderrText = ""
    if (Test-Path $StdoutPath) { $stdoutText = Get-Content -Raw -Encoding utf8 $StdoutPath }
    if (Test-Path $StderrPath) { $stderrText = Get-Content -Raw -Encoding utf8 $StderrPath }
    Set-Content -Encoding utf8 -Path $CombinedPath -Value (
        "===== STDOUT =====`r`n" + $stdoutText + "`r`n===== STDERR =====`r`n" + $stderrText
    )

    $MemoryStopHit = Select-String -Path $ProgressLog -Pattern ("MEMORY_STOP tf={0}" -f $Tf) -SimpleMatch -Quiet
    if ($MemoryStopHit) {
        $ExitCode = 99
        $StopReason = "memory_stop_$Tf"
        $StoppedEarly = $true
    }

    Write-ProgressLine (
        "TF_END {0} {1} exit_code={2} duration_seconds={3}" -f $Tf, $EndedUtc, $ExitCode, $DurationSeconds
    )

    $Results += [pscustomobject]@{
        timeframe = $Tf
        started_utc = $StartedUtc
        ended_utc = $EndedUtc
        duration_seconds = $DurationSeconds
        exit_code = $ExitCode
        command = $CommandText
        stdout_log = $StdoutPath
        stderr_log = $StderrPath
        memory_stop = [bool]$MemoryStopHit
    }

    if ($ExitCode -ne 0) {
        $OverallExit = $ExitCode
        $StoppedEarly = $true
        if (-not $StopReason) { $StopReason = "child_exit_$Tf" }
        Write-ProgressLine ("STOP remaining timeframes not started after {0} failure" -f $Tf)
    }
}

$OverallEnded = Get-Date
$OverallEndedUtc = [DateTime]::UtcNow.ToString("o")
$OverallDuration = [Math]::Round(($OverallEnded - $OverallStarted).TotalSeconds, 3)

$Meta = [ordered]@{
    started_utc = $OverallStartedUtc
    ended_utc = $OverallEndedUtc
    duration_seconds = $OverallDuration
    exit_code = $OverallExit
    stopped_early = $StoppedEarly
    stop_reason = $StopReason
    workers = 1
    engine = "simple (CLI default, flag omitted)"
    overwrite = $true
    storage_root = "data"
    timeframes_planned = $Timeframes
    memory_stop_private_mb = $MemoryStopMb
    pythonunbuffered = $true
    error_action_preference = "$ErrorActionPreference"
    note = "PYTHONUNBUFFERED=1 is wrapper logging only. Stdout/stderr redirected to files. Native exit code from LASTEXITCODE. ErrorActionPreference is not Stop."
    partitions = $Results
}
$Meta | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $MetaLog

Write-ProgressLine (
    "END {0} exit_code={1} duration_seconds={2}" -f $OverallEndedUtc, $OverallExit, $OverallDuration
)
exit $OverallExit
