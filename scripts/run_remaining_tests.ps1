<#
按阶段运行第 6、7 周尚需实测的桌面任务。只在 VMware 客机 PowerShell 执行。
每次运行都会使用独立 tag；单项试跑遇到失败、无效起点或急停立即停止。
未提供 VMware 快照名时仍可试跑，但各批结果不能称为同快照对照。

示例：
  .\scripts\run_remaining_tests.ps1 -Stage w7-untested
  .\scripts\run_remaining_tests.ps1 -Stage w7-pilot
  .\scripts\run_remaining_tests.ps1 -Stage w7-full
  .\scripts\run_remaining_tests.ps1 -Stage w6-long
  .\scripts\run_remaining_tests.ps1 -Stage w6-settle-fixed
  .\scripts\run_remaining_tests.ps1 -Stage w6-settle-adaptive
  .\scripts\run_remaining_tests.ps1 -Stage w6-stability
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        'w7-untested', 'w7-pilot', 'w7-full', 'w6-long',
        'w6-settle-fixed', 'w6-settle-adaptive',
        'engine-legacy', 'engine-langgraph', 'w6-stability'
    )]
    [string]$Stage,
    [string]$Model = 'qwen3-vl-8b-instruct',
    [string]$PlannerTemplate = 'planner_v10',
    [string]$ExecutorTemplate = 'executor_v5',
    [string]$SnapshotName = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repoRoot
$envFile = Join-Path $env:USERPROFILE '.gui-agent\.env'
if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    throw "找不到仓库外的 API 配置：$envFile。请先在客机配置，不要把密钥粘贴到聊天或提交到 Git。"
}
$env:GUI_AGENT_ENV_FILE = $envFile
$env:PYTHONIOENCODING = 'utf-8'
$commit = (& git rev-parse --short HEAD).Trim()
Write-Host "代码版本 $commit；阶段 $Stage；模型 $Model；提示词 $PlannerTemplate + $ExecutorTemplate"
if (-not $SnapshotName) {
    Write-Warning '未提供快照名：本次可作为实机测试，不能作为同快照严格 A/B。'
}

function Invoke-TaskBatch {
    param(
        [string]$TaskFile,
        [string]$Only = '',
        [int]$Repeats = 1,
        [string]$Engine = 'langgraph',
        [switch]$AdaptiveSettle,
        [switch]$RequirePass
    )
    $tag = "$Stage-$Model-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    $runnerArgs = @(
        'scripts/run_basic_tasks.py', '--execute',
        '--tasks', $TaskFile, '--repeats', [string]$Repeats,
        '--provider', 'dashscope', '--model', $Model,
        '--engine', $Engine,
        '--planner-template', $PlannerTemplate,
        '--executor-template', $ExecutorTemplate,
        '--tag', $tag
    )
    if ($Only) { $runnerArgs += @('--only', $Only) }
    if ($AdaptiveSettle) { $runnerArgs += '--adaptive-settle' }
    if ($SnapshotName) { $runnerArgs += @('--guest-snapshot', $SnapshotName) }
    & python @runnerArgs
    $runnerExitCode = $LASTEXITCODE

    $scope = if ($Only) { $Only } else { 'all' }
    $archive = Get-ChildItem -LiteralPath 'docs\m2-runs' -File -Filter "*-$scope-exec-online-$tag.json" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $archive) { throw "没有找到本次存档：$scope / $tag" }
    $data = Get-Content -LiteralPath $archive.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $records = @($data.records)
    $invalid = @($records | Where-Object { -not $_.precondition_ok -or $_.excluded })
    $failed = @($records | Where-Object { $_.precondition_ok -and -not $_.excluded -and -not $_.verified })
    $passed = @($records | Where-Object { $_.precondition_ok -and -not $_.excluded -and $_.verified })
    Write-Host "存档 $($archive.FullName)；通过 $($passed.Count)/$($records.Count)；判定失败 $($failed.Count)；无效 $($invalid.Count)"

    if ($runnerExitCode -ne 0 -or $data.aborted) {
        throw "运行器停止或触发急停；请检查桌面和存档，不要自动续跑。$($data.aborted)"
    }
    $expected = if ($Only) { $Repeats } elseif ($TaskFile -eq 'tasks/desktop_20.yaml') { 20 * $Repeats } else { 3 * $Repeats }
    if ($data.partial -or $records.Count -ne $expected -or $invalid.Count -gt 0) {
        throw '批次不完整或起点无效；先检查 reset 与存档。'
    }
    if ($RequirePass -and $failed.Count -gt 0) {
        throw "单项未通过：$($failed[0].task)；先运行 summarize_run_traces.py 分析，再继续下一项。"
    }
}

$week7Pending = @(
    'minimize_all', 'write_note', 'append_line', 'copy_paste_text',
    'create_folder', 'rename_file', 'delete_file', 'scroll_document',
    'calc_to_notepad', 'read_and_summarize', 'multi_app_workflow'
)
$week7Untested = @(
    'rename_file', 'delete_file', 'scroll_document',
    'calc_to_notepad', 'read_and_summarize', 'multi_app_workflow'
)
$basicNames = @('open_browser', 'search_content', 'open_file', 'send_message', 'close_app')

switch ($Stage) {
    'w7-untested' {
        foreach ($taskName in $week7Untested) {
            # 普通判定失败也继续下一项，以便六项都有首次实测记录。
            # 急停、起点无效或批次不完整仍由 Invoke-TaskBatch 停止。
            Invoke-TaskBatch -TaskFile 'tasks/desktop_20.yaml' -Only $taskName -Repeats 1
        }
    }
    'w7-pilot' {
        foreach ($taskName in $week7Pending) {
            Invoke-TaskBatch -TaskFile 'tasks/desktop_20.yaml' -Only $taskName -Repeats 1 -RequirePass
        }
    }
    'w7-full' {
        Invoke-TaskBatch -TaskFile 'tasks/desktop_20.yaml' -Repeats 5
    }
    'w6-long' {
        Invoke-TaskBatch -TaskFile 'tasks/long_tasks.yaml' -Repeats 5
    }
    'w6-settle-fixed' {
        foreach ($taskName in $basicNames) {
            Invoke-TaskBatch -TaskFile 'tasks/desktop_20.yaml' -Only $taskName -Repeats 5
        }
    }
    'w6-settle-adaptive' {
        foreach ($taskName in $basicNames) {
            Invoke-TaskBatch -TaskFile 'tasks/desktop_20.yaml' -Only $taskName -Repeats 5 -AdaptiveSettle
        }
    }
    'engine-legacy' {
        foreach ($taskName in $basicNames) {
            Invoke-TaskBatch -TaskFile 'tasks/desktop_20.yaml' -Only $taskName -Repeats 5 -Engine 'legacy'
        }
    }
    'engine-langgraph' {
        foreach ($taskName in $basicNames) {
            Invoke-TaskBatch -TaskFile 'tasks/desktop_20.yaml' -Only $taskName -Repeats 5 -Engine 'langgraph'
        }
    }
    'w6-stability' {
        $tag = Get-Date -Format 'yyyyMMdd-HHmmss'
        $output = "docs/m2-runs/$tag-stability-60m-$Model.json"
        & python scripts/stability_run.py --execute --minutes 60 --only close_app `
            --tasks tasks/basic_tasks.yaml --provider dashscope --model $Model `
            --engine langgraph --planner-template $PlannerTemplate `
            --executor-template $ExecutorTemplate --max-steps 6 --output $output
        if ($LASTEXITCODE -ne 0) {
            throw "稳定性测试提前停止；检查 $output 和客机桌面状态。"
        }
        Write-Host "稳定性原始记录 $output"
    }
}
