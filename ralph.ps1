# Ralph Wiggum - Robust Autonomous Coding Loop (Windows/PowerShell Version)
# Usage: .\ralph.ps1 -MaxIterations 10

param(
    [int]$MaxIterations = 10
)

# --- Configuration & Paths ---
$PRD_FILE = Join-Path $PSScriptRoot "PRD.md"
$PROGRESS_FILE = Join-Path $PSScriptRoot "progress.txt"
$ARCHIVE_DIR = Join-Path $PSScriptRoot "archive"
$LAST_BRANCH_FILE = Join-Path $PSScriptRoot ".last-branch"

# --- State Management: Archiving ---
if (Get-Command git -ErrorAction SilentlyContinue) {
    $CURRENT_BRANCH = git rev-parse --abbrev-ref HEAD 2>$null
    if ($null -eq $CURRENT_BRANCH) { $CURRENT_BRANCH = "main" }
    
    if (Test-Path $LAST_BRANCH_FILE) {
        $LAST_BRANCH = Get-Content $LAST_BRANCH_FILE -Raw
        
        if ($LAST_BRANCH -and ($CURRENT_BRANCH -ne $LAST_BRANCH)) {
            $DATE_STR = Get-Date -Format "yyyy-MM-dd"
            $ARCHIVE_FOLDER = Join-Path $ARCHIVE_DIR "$DATE_STR-$LAST_BRANCH"
            
            Write-Host "Archiving previous run: $LAST_BRANCH" -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path $ARCHIVE_FOLDER | Out-Null
            if (Test-Path $PRD_FILE) { Copy-Item $PRD_FILE $ARCHIVE_FOLDER }
            if (Test-Path $PROGRESS_FILE) { Copy-Item $PROGRESS_FILE $ARCHIVE_FOLDER }
            
            # Reset progress for the new branch
            "# Ralph Progress Log`nStarted: $(Get-Date)`n---" | Out-File -FilePath $PROGRESS_FILE -Encoding utf8
        }
    }
    $CURRENT_BRANCH | Out-File -FilePath $LAST_BRANCH_FILE -Encoding utf8
}

# Ensure progress file exists
if (-not (Test-Path $PROGRESS_FILE)) {
    "# Ralph Progress Log`n---" | Out-File -FilePath $PROGRESS_FILE -Encoding utf8
}

Write-Host "Starting Ralph - Max $MaxIterations iterations" -ForegroundColor Green
Write-Host "Target: $PRD_FILE`n"

# --- Main Iteration Loop ---
for ($i = 1; $i -le $MaxIterations; $i++) {
    Write-Host "================================================================================" -ForegroundColor Yellow
    Write-Host "  Ralph Iteration $i of $MaxIterations" -ForegroundColor Yellow
    Write-Host "================================================================================" -ForegroundColor Yellow

    # Informative: Show the user what the next task is before starting
    $Content = Get-Content $PRD_FILE
    $NextTask = $Content | Select-String "- \[ \] (US-\d+: .*)" | Select-Object -First 1
    if ($NextTask) {
        Write-Host "🎯 Current Target: $($NextTask.Matches.Groups[1].Value)" -ForegroundColor Cyan
    }
    
    # Execute Claude Code in non-interactive mode
    # EDITOR=true prevents opening VS Code for commits
    $env:EDITOR = "true"
    
    $Prompt = @"
You are Ralph, an autonomous coding agent. Do exactly ONE task per iteration.

## Steps
1. Read PRD.md and find the first task that is NOT complete (marked [ ]).
2. Read progress.txt - check the 'Learnings' section first for patterns from previous iterations.
3. Implement that ONE task only.
4. Run tests/typecheck to verify it works.

## Critical: Only Complete If Tests Pass
- If tests PASS:
  - Update PRD.md to mark the task complete (change [ ] to [x]).
  - Commit your changes with message: feat: [task description].
  - Append implementation details to progress.txt using the format below.

- If tests FAIL:
  - Do NOT mark the task complete.
  - Do NOT commit broken code.
  - Append what went wrong to progress.txt so the next iteration can learn.

## Progress Notes Format
Append to progress.txt:
---
## Iteration [$i] - [Task Name]
- What was implemented
- Files changed
- Learnings for future iterations:
  - Patterns discovered
  - Gotchas encountered
  - Useful context
---

## Update AGENTS.md (If Applicable)
If you discover a reusable pattern that future work should know about:
- Check if AGENTS.md exists in the project root.
- Add patterns like: \"This codebase uses X for Y\" or \"Always do Z when changing W\".
- Only add genuinely reusable knowledge, not task-specific details.

## End Condition
After completing your task, check PRD.md:
- If ALL tasks are [x], output exactly: <promise>COMPLETE</promise>.
- If tasks remain [ ], just end your response (next iteration will continue).
"@

    # Run the command and stream output directly to console
    $result = claude -p $Prompt --dangerously-skip-permissions --output-format text

    # Check for completion signal
    if ($result -match "<promise>COMPLETE</promise>") {
        Write-Host "`n✅ Ralph completed all tasks in PRD.md!" -ForegroundColor Green
        exit 0
    }

    Write-Host "`nIteration $i complete. Continuing to next iteration..." -ForegroundColor Gray
    Start-Sleep -Seconds 2
}

Write-Host "`n❌ Ralph reached max iterations ($MaxIterations) without completing all tasks." -ForegroundColor Red
exit 1