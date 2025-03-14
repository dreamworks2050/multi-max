# Windows Version Check Script
# This script checks if the installed Multi-Max version is properly configured for Windows

Write-Host "==== Multi-Max Windows Version Check ====" -ForegroundColor Cyan
Write-Host

# Check if main.py exists
$mainFile = "..\main.py"
if (-not (Test-Path $mainFile)) {
    Write-Host "ERROR: main.py not found. Multi-Max may not be installed correctly." -ForegroundColor Red
    Write-Host "Please run the Windows installer first." -ForegroundColor Yellow
    exit 1
}

# Check if main.py contains the Windows marker
$windowsMarker = Select-String -Path $mainFile -Pattern "__windows_specific_version__" -Quiet
if ($windowsMarker) {
    Write-Host "SUCCESS: Windows-specific version is correctly installed." -ForegroundColor Green
} else {
    Write-Host "ERROR: The installed version does not appear to be the Windows-specific version." -ForegroundColor Red
    Write-Host "This may cause issues when running on Windows." -ForegroundColor Red
    
    # Check if we have a Windows version available
    if (Test-Path "main.py") {
        Write-Host "Found Windows-specific version in the windows folder." -ForegroundColor Yellow
        Write-Host "Would you like to install it now? (Y/N)" -ForegroundColor Yellow
        $response = Read-Host
        if ($response -eq "Y" -or $response -eq "y") {
            Copy-Item -Path "main.py" -Destination $mainFile -Force
            Write-Host "Windows version installed successfully." -ForegroundColor Green
        }
    } else {
        Write-Host "No Windows-specific version found in the windows folder." -ForegroundColor Red
        Write-Host "Please download the correct version from the repository." -ForegroundColor Yellow
    }
}

# Check environment configuration
$envFile = "..\\.env"
if (Test-Path $envFile) {
    $envContent = Get-Content $envFile -Raw
    
    # Check for Windows settings
    $hardwareAcceleration = $envContent -match "FORCE_HARDWARE_ACCELERATION=false"
    $softwareFallback = $envContent -match "ALLOW_SOFTWARE_FALLBACK=true"
    $windowsMode = $envContent -match "WINDOWS_MODE=true"
    
    if ($hardwareAcceleration -and $softwareFallback -and $windowsMode) {
        Write-Host "Environment configuration is correctly set up for Windows." -ForegroundColor Green
    } else {
        Write-Host "WARNING: Environment configuration may not be optimized for Windows." -ForegroundColor Yellow
        Write-Host "Would you like to apply the Windows-specific environment settings? (Y/N)" -ForegroundColor Yellow
        $response = Read-Host
        if ($response -eq "Y" -or $response -eq "y") {
            if (Test-Path ".env") {
                Copy-Item -Path ".env" -Destination $envFile -Force
                Write-Host "Windows environment settings applied successfully." -ForegroundColor Green
            } else {
                Write-Host "No Windows-specific .env file found in the windows folder." -ForegroundColor Red
            }
        }
    }
} else {
    Write-Host "No .env file found. Environment may not be configured correctly." -ForegroundColor Yellow
}

Write-Host
Write-Host "==== Check Complete ====" -ForegroundColor Cyan
Write-Host

# Pause so the user can read the results
Write-Host "Press any key to exit..."
$null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 