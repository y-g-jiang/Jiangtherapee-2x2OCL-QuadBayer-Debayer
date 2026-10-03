param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [Parameter(Mandatory=$true)][string]$OutputFile,
    [string]$Device = 'cuda'
)
python "$PSScriptRoot/decode.py" --input $InputFile --output $OutputFile --device $Device
exit $LASTEXITCODE
