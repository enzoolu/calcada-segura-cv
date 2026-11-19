param (
    [string]$Fonte = '0',
    [string]$Pesos = 'visao/evidencias/treino-publico/best.pt',
    [string]$Perfil = '',
    [double]$Confianca = 0.25,
    [int]$Tamanho = 0,
    [switch]$ReforcarBuracos,
    [ValidateRange(0.0001, 1.0)]
    [double]$ConfiancaBuracos = 0.12,
    [ValidateRange(0.0001, 1.0)]
    [double]$ObstaculoAteY = 1.0
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = Join-Path $PWD '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Instale Python 3.12 e o inicializador py.' }
}
& $python -c 'import cv2, httpx, ultralytics'
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install -r visao/requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar as dependências.' }
}

$arquivo = 'saida/camera-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.mp4'
$limiar = $Confianca.ToString([System.Globalization.CultureInfo]::InvariantCulture)
if ($Pesos -eq 'visao/evidencias/treino-publico/best.pt') {
    if (-not $Perfil -and -not $PSBoundParameters.ContainsKey('Confianca')) {
        $Perfil = 'visao/evidencias/treino-publico/perfil.json'
    }
    if ($Tamanho -eq 0) { $Tamanho = 384 }
}
if ($Tamanho -eq 0) { $Tamanho = 480 }
$opcoes = @('--pesos', $Pesos, '--fonte', $Fonte, '--device', 'cpu',
            '--imgsz', $Tamanho, '--confianca', $limiar, '--saida', $arquivo)
if ($Perfil) { $opcoes += @('--perfil', $Perfil) }
if ($ObstaculoAteY -lt 1.0) {
    $limiteVertical = $ObstaculoAteY.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    $opcoes += @('--obstaculo-ate-y', $limiteVertical)
}
if ($ReforcarBuracos -or $PSBoundParameters.ContainsKey('ConfiancaBuracos')) {
    $limiarBuracos = $ConfiancaBuracos.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    $opcoes += @('--pesos-buraco', 'visao/evidencias/reforco-roma/best.pt', '--confianca-buraco', $limiarBuracos)
}
& $python -m visao.src.inferir @opcoes
exit $LASTEXITCODE
