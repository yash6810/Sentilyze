param (
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonVenv = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$SentilyzeScript = Join-Path $ScriptDir "sentilyze.py"

if (Test-Path $PythonVenv) {
    & $PythonVenv $SentilyzeScript @ArgsList
} else {
    python $SentilyzeScript @ArgsList
}
