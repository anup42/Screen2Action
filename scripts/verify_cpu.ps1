param(
    [ValidateSet('format', 'lint', 'typecheck', 'test', 'all')]
    [string] $Target = 'all'
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repoRoot 'src'
python -m screen2action.tools.verify $Target
exit $LASTEXITCODE
