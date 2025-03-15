# Multi-Max Simple Installer Wrapper for Windows
# This script downloads the multi-max repository and runs the simplified installer

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Banner
Write-Host "`n=================================================" -ForegroundColor Cyan
Write-Host "  Multi-Max Windows One-Click Installer" -ForegroundColor Cyan
Write-Host "=================================================`n" -ForegroundColor Cyan

# Check for administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Write-Host "Administrator privileges: $isAdmin"

if (-not $isAdmin) {
    Write-Host "NOTE: Running without administrator privileges." -ForegroundColor Yellow
    Write-Host "Some features like creating desktop shortcuts might not work properly."
    Write-Host "Consider rerunning as administrator if you encounter issues."
}

# Create temporary directory
$tempDir = [System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString()
New-Item -Path $tempDir -ItemType Directory | Out-Null
Write-Host "Created temporary directory: $tempDir" -ForegroundColor Gray

try {
    # Check for Git to determine installation method
    $hasGit = $null -ne (Get-Command "git" -ErrorAction SilentlyContinue)
    
    # Define installation target
    $installDir = Join-Path -Path $env:USERPROFILE -ChildPath "multi-max"
    
    # If directory exists, ask for confirmation
    if (Test-Path $installDir) {
        Write-Host "Multi-Max directory already exists at: $installDir" -ForegroundColor Yellow
        $confirm = Read-Host "Do you want to proceed and possibly overwrite existing files? (y/n)"
        if ($confirm -ne "y") {
            Write-Host "Installation cancelled by user." -ForegroundColor Red
            exit 1
        }
    } else {
        New-Item -Path $installDir -ItemType Directory | Out-Null
        Write-Host "Created installation directory: $installDir" -ForegroundColor Green
    }
    
    # Clone or download the repository
    if ($hasGit) {
        Write-Host "Using Git to clone the repository..."
        git clone https://github.com/dreamworks2050/multi-max.git $installDir --depth 1
        if (-not $?) {
            throw "Failed to clone repository"
        }
    } else {
        Write-Host "Git not found. Downloading ZIP archive instead..."
        $zipUrl = "https://github.com/dreamworks2050/multi-max/archive/refs/heads/main.zip"
        $zipPath = Join-Path -Path $tempDir -ChildPath "multi-max.zip"
        
        # Download ZIP
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath
        
        # Extract ZIP
        Write-Host "Extracting ZIP archive..."
        Expand-Archive -Path $zipPath -DestinationPath $tempDir
        
        # Move contents to install dir
        $extractedDir = Get-ChildItem -Path $tempDir -Directory | Where-Object { $_.Name -like "multi-max*" } | Select-Object -First 1
        Get-ChildItem -Path $extractedDir.FullName | Copy-Item -Destination $installDir -Recurse -Force
    }
    
    Write-Host "Repository downloaded successfully to: $installDir" -ForegroundColor Green
    
    # Navigate to windows directory and run the simplified installer
    $windowsDir = Join-Path -Path $installDir -ChildPath "windows"
    if (-not (Test-Path $windowsDir)) {
        throw "Windows directory not found in the repository"
    }
    
    # Check for the simplified installer
    $installerPath = Join-Path -Path $windowsDir -ChildPath "install-simplified.ps1"
    if (-not (Test-Path $installerPath)) {
        # Try to get it from GitHub directly
        Write-Host "Simplified installer not found in repository. Downloading directly..."
        $installerUrl = "https://raw.githubusercontent.com/dreamworks2050/multi-max/main/windows/install-simplified.ps1"
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
        
        if (-not (Test-Path $installerPath)) {
            throw "Could not download simplified installer"
        }
    }
    
    # Run the installer
    Write-Host "Running simplified installer..." -ForegroundColor Green
    Set-Location $windowsDir
    & $installerPath
    
    Write-Host "`nOne-click installation completed successfully!" -ForegroundColor Green
    
} catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Write-Host "Installation failed." -ForegroundColor Red
    exit 1
} finally {
    # Cleanup
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force
    }
} 