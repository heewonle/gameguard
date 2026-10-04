# GameGuard 일일 배치를 Windows 작업 스케줄러에 등록한다 (매일 06:00).
# 실행 전 경로를 확인하고, 직접 실행할 때만 등록된다:  powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1
# 해제:  Unregister-ScheduledTask -TaskName GameGuardDaily -Confirm:$false
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Args = "-m gameguard.run --db data/test.duckdb --out data/out_test --agent-budget 20"

$Action = New-ScheduledTaskAction -Execute $Py -Argument $Args -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -Daily -At 6:00am
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$Env = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
Register-ScheduledTask -TaskName "GameGuardDaily" -Action $Action -Trigger $Trigger -Settings $Settings `
    -Principal $Env -Description "GameGuard 일일 탐지 배치 (품질→룰→ML→그래프→경보→조사 에이전트)" -Force
Write-Host "등록 완료: GameGuardDaily (매일 06:00). 로그: data\out_test\runs\"
