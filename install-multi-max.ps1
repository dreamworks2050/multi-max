# Multi-Max Windows Installer Script
# This script automates the installation of Multi-Max and all dependencies
# Usage: Run in PowerShell with administrator privileges

# Set error action preference to continue so that the script doesn't stop on non-fatal errors
$ErrorActionPreference = "Continue"

function Write-ColorOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,
        
        [Parameter(Mandatory = $false)]
        [string]$ForegroundColor = "White"
    )
    
    $originalColor = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $originalColor
}

# Display header
Write-ColorOutput "=========================================" "Cyan"
Write-ColorOutput "  Multi-Max Windows Installation Script  " "Cyan"
Write-ColorOutput "=========================================" "Cyan"
Write-ColorOutput "  Installation order:                    " "Cyan"
Write-ColorOutput "  1. Check and install prerequisites     " "Cyan"
Write-ColorOutput "  2. Clone the repository                " "Cyan"
Write-ColorOutput "  3. Set up Python environment           " "Cyan"
Write-ColorOutput "  4. Install FFmpeg                      " "Cyan"
Write-ColorOutput "  5. Install Python dependencies         " "Cyan"
Write-ColorOutput "  6. Configure environment settings      " "Cyan"
Write-ColorOutput "=========================================" "Cyan"

# Create a temporary directory for downloads
$tempDir = Join-Path $env:TEMP "multi-max-setup"
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir | Out-Null
}

# Step 1: Check for admin rights
Write-ColorOutput "Checking for administrator privileges..." "Yellow"
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-ColorOutput "This script requires administrator privileges. Please run PowerShell as Administrator and try again." "Red"
    Write-ColorOutput "Exiting installation..." "Red"
    exit 1
}
Write-ColorOutput "Running with administrator privileges." "Green"

# Step 2: Check and install Chocolatey (package manager for Windows)
Write-ColorOutput "Checking for Chocolatey package manager..." "Yellow"
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-ColorOutput "Installing Chocolatey package manager..." "Yellow"
    try {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
        
        # Refresh environment to get choco in the path
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            Write-ColorOutput "Chocolatey installed successfully." "Green"
        } else {
            Write-ColorOutput "Failed to install Chocolatey. Please install it manually: https://chocolatey.org/install" "Red"
            exit 1
        }
    } catch {
        Write-ColorOutput "Failed to install Chocolatey: $_" "Red"
        Write-ColorOutput "Please install it manually: https://chocolatey.org/install" "Red"
        exit 1
    }
} else {
    Write-ColorOutput "Chocolatey is already installed." "Green"
}

# Step 3: Check and install Git
Write-ColorOutput "Checking for Git..." "Yellow"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-ColorOutput "Installing Git..." "Yellow"
    try {
        choco install git -y
        
        # Refresh environment to get git in the path
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        if (Get-Command git -ErrorAction SilentlyContinue) {
            Write-ColorOutput "Git installed successfully." "Green"
        } else {
            Write-ColorOutput "Failed to install Git. Please install it manually: https://git-scm.com/download/win" "Red"
            exit 1
        }
    } catch {
        Write-ColorOutput "Failed to install Git: $_" "Red"
        Write-ColorOutput "Please install it manually: https://git-scm.com/download/win" "Red"
        exit 1
    }
} else {
    Write-ColorOutput "Git is already installed." "Green"
}

# Step 4: Check and install Python
Write-ColorOutput "Checking for Python..." "Yellow"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-ColorOutput "Installing Python..." "Yellow"
    try {
        choco install python -y
        
        # Refresh environment to get python in the path
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        if (Get-Command python -ErrorAction SilentlyContinue) {
            Write-ColorOutput "Python installed successfully." "Green"
        } else {
            Write-ColorOutput "Failed to install Python. Please install it manually: https://www.python.org/downloads/" "Red"
            exit 1
        }
    } catch {
        Write-ColorOutput "Failed to install Python: $_" "Red"
        Write-ColorOutput "Please install it manually: https://www.python.org/downloads/" "Red"
        exit 1
    }
} else {
    Write-ColorOutput "Python is already installed: $(python --version)" "Green"
}

# Step 5: Clone the repository
Write-ColorOutput "Cloning the Multi-Max repository..." "Yellow"
$installDir = Join-Path $env:USERPROFILE "multi-max"
if (Test-Path $installDir) {
    Write-ColorOutput "The 'multi-max' directory already exists." "Yellow"
    # Automatically overwrite existing directory for full automation
    Write-ColorOutput "Automatically removing existing directory for fresh installation..." "Yellow"
    Remove-Item -Recurse -Force $installDir
    Write-ColorOutput "Existing directory removed." "Green"
}

if (-not (Test-Path $installDir)) {
    try {
        # Use the hardcoded repository URL directly
        $repoUrl = "https://github.com/dreamworks2050/multi-max.git"
        Write-ColorOutput "Cloning from repository: $repoUrl" "Yellow"
        
        # Clone the repository
        git clone $repoUrl $installDir
        if (-not $?) {
            throw "Git clone failed with exit code $LASTEXITCODE"
        }
        Write-ColorOutput "Repository cloned successfully to $installDir" "Green"
    } catch {
        Write-ColorOutput "Failed to clone repository: $_" "Red"
        Write-ColorOutput "Please check your internet connection." "Red"
        exit 1
    }
}

# Change to the installation directory
Set-Location $installDir

# Step 6: Install FFmpeg
Write-ColorOutput "Installing FFmpeg..." "Yellow"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    try {
        choco install ffmpeg -y
        
        # Refresh environment to get ffmpeg in the path
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
            Write-ColorOutput "FFmpeg installed successfully." "Green"
        } else {
            Write-ColorOutput "Failed to install FFmpeg. Please install it manually: https://ffmpeg.org/download.html" "Yellow"
            Write-ColorOutput "After installing, make sure to add it to your PATH environment variable." "Yellow"
        }
    } catch {
        Write-ColorOutput "Failed to install FFmpeg: $_" "Red"
        Write-ColorOutput "Please install it manually: https://ffmpeg.org/download.html" "Red"
        Write-ColorOutput "After installing, make sure to add it to your PATH environment variable." "Red"
    }
} else {
    Write-ColorOutput "FFmpeg is already installed." "Green"
}

# Step 7: Set up virtual environment
Write-ColorOutput "Setting up Python virtual environment..." "Yellow"
$venvName = "multi-max"
if (-not (Test-Path $venvName)) {
    try {
        python -m venv $venvName
        Write-ColorOutput "Virtual environment '$venvName' created successfully." "Green"
    } catch {
        Write-ColorOutput "Failed to create virtual environment: $_" "Red"
        exit 1
    }
}

# Activate virtual environment
Write-ColorOutput "Activating virtual environment..." "Yellow"
try {
    & .\$venvName\Scripts\Activate.ps1
    Write-ColorOutput "Virtual environment activated." "Green"
} catch {
    Write-ColorOutput "Failed to activate virtual environment: $_" "Red"
    Write-ColorOutput "Trying to continue anyway..." "Yellow"
}

# Step 8: Install UV package manager (prioritized)
Write-ColorOutput "Installing UV package manager..." "Yellow"
$uvInstalled = $false
try {
    # Upgrade pip first
    python -m pip install --upgrade pip
    
    # Install UV - try multiple methods to ensure it's installed
    Write-ColorOutput "Installing UV using pip..." "Yellow"
    python -m pip install uv
    
    # Check if UV is in PATH
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-ColorOutput "UV package manager installed successfully." "Green"
        $uvInstalled = $true
    } else {
        Write-ColorOutput "UV not found in PATH after pip install. Trying alternative installation method..." "Yellow"
        
        # Alternative method - direct install script
        Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "$tempDir\install-uv.ps1"
        & $tempDir\install-uv.ps1
        
        # Update PATH and check again
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            Write-ColorOutput "UV package manager installed successfully using install script." "Green"
            $uvInstalled = $true
        } else {
            Write-ColorOutput "UV still not found in PATH. One more attempt with direct path..." "Yellow"
            
            # Try to find UV in the Python scripts directory
            $uvPath = Join-Path (Join-Path $installDir $venvName) "Scripts\uv.exe"
            if (Test-Path $uvPath) {
                Write-ColorOutput "Found UV at: $uvPath" "Green"
                $uvInstalled = $true
                
                # Create a function to call UV directly
                function Invoke-UV {
                    param([string[]]$Arguments)
                    & $uvPath $Arguments
                    return $?
                }
            } else {
                Write-ColorOutput "Failed to install UV via any method. Falling back to pip for installations." "Red"
                $uvInstalled = $false
            }
        }
    }
} catch {
    Write-ColorOutput "Error installing UV package manager: $_" "Red"
    Write-ColorOutput "Falling back to pip for package installation." "Yellow"
}

# Step 9: Install Python requirements - prioritize UV
Write-ColorOutput "Installing Python requirements..." "Yellow"
if (Test-Path "requirements.txt") {
    try {
        if ($uvInstalled) {
            Write-ColorOutput "Installing packages using UV (faster)..." "Yellow"
            
            # Use the direct function if defined, otherwise try the command
            if (Get-Command Invoke-UV -ErrorAction SilentlyContinue) {
                Invoke-UV @("pip", "install", "-r", "requirements.txt")
            } else {
                uv pip install -r requirements.txt
            }
            
            if (-not $?) {
                throw "UV installation failed with exit code $LASTEXITCODE. Falling back to pip."
            } else {
                Write-ColorOutput "Successfully installed requirements using UV." "Green"
            }
        } else {
            Write-ColorOutput "Installing packages using pip..." "Yellow"
            python -m pip install -r requirements.txt
            if (-not $?) {
                throw "Pip installation failed with exit code $LASTEXITCODE"
            }
        }
        Write-ColorOutput "Python requirements installed successfully." "Green"
    } catch {
        Write-ColorOutput "Failed to install all requirements: $_" "Red"
        Write-ColorOutput "Some packages may not have been installed correctly." "Yellow"
        Write-ColorOutput "You may need to install them manually." "Yellow"
    }
} else {
    Write-ColorOutput "requirements.txt not found! Cannot install required Python packages." "Red"
}

# Step 10: Install yt-dlp separately
Write-ColorOutput "Checking for yt-dlp..." "Yellow"
if (-not (Get-Command yt-dlp -ErrorAction SilentlyContinue)) {
    Write-ColorOutput "Installing yt-dlp..." "Yellow"
    try {
        python -m pip install yt-dlp
        if (-not $?) {
            throw "yt-dlp installation failed with exit code $LASTEXITCODE"
        }
        Write-ColorOutput "yt-dlp installed successfully." "Green"
    } catch {
        Write-ColorOutput "Failed to install yt-dlp: $_" "Red"
        Write-ColorOutput "Please install it manually: pip install yt-dlp" "Yellow"
    }
} else {
    Write-ColorOutput "yt-dlp is already installed." "Green"
}

# Step 11: Set up environment configuration
Write-ColorOutput "Setting up environment configuration..." "Yellow"
$envPath = Join-Path $installDir ".env"
$envTemplatePath = Join-Path $installDir ".env.template"

# Check if Windows-specific .env exists in the windows directory
$windowsEnvPath = Join-Path $installDir "windows\.env"
if (Test-Path $windowsEnvPath) {
    Write-ColorOutput "Found Windows-specific environment configuration." "Green"
    Copy-Item -Path $windowsEnvPath -Destination $envPath -Force
    Write-ColorOutput "Windows environment configuration copied to .env" "Green"
} else {
    # If Windows-specific environment doesn't exist, create from template with Windows settings
    if (Test-Path $envTemplatePath) {
        $envContent = Get-Content $envTemplatePath -Raw
        
        # Modify environment settings for Windows
        $envContent = $envContent -replace "FORCE_HARDWARE_ACCELERATION=true", "FORCE_HARDWARE_ACCELERATION=false"
        $envContent = $envContent -replace "ALLOW_SOFTWARE_FALLBACK=false", "ALLOW_SOFTWARE_FALLBACK=true"
        
        # Add Windows-specific settings
        $envContent += "`n# Windows-specific settings`n"
        $envContent += "WINDOWS_MODE=true`n"
        $envContent += "USE_SOFTWARE_RENDERING=true`n"
        
        $envContent | Out-File -FilePath $envPath -Encoding utf8
        Write-ColorOutput "Environment configuration created for Windows compatibility" "Green"
    } else {
        Write-ColorOutput "Environment template not found. Skipping environment configuration." "Yellow"
    }
}

# Step 12: Create desktop shortcut (automatic)
Write-ColorOutput "Creating desktop shortcut for easy launching..." "Yellow"
    
try {
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Multi-Max.lnk")
    $Shortcut.TargetPath = "powershell.exe"
    $Shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$installDir\run.ps1`""
    $Shortcut.WorkingDirectory = $installDir
    $Shortcut.Description = "Multi-Max Recursive Video Grid"
    $Shortcut.Save()
    
    # Create the run script
    @"
# Launcher for Multi-Max
Set-Location "$installDir"
& .\$venvName\Scripts\Activate.ps1
python main.py
"@ | Set-Content "run.ps1"
    
    Write-ColorOutput "Desktop shortcut created successfully." "Green"
} catch {
    Write-ColorOutput "Failed to create desktop shortcut: $_" "Yellow"
    
    # Create a batch file as fallback
    @"
@echo off
cd "$installDir"
call $venvName\Scripts\activate.bat
python main.py
"@ | Set-Content "$env:USERPROFILE\Desktop\Multi-Max.bat"
    
    Write-ColorOutput "Created fallback batch file at Desktop\Multi-Max.bat" "Green"
}

# Step 13: Set up Windows-specific version
Write-ColorOutput "Setting up Windows-specific version..." "Yellow"
$windowsMainPath = Join-Path $installDir "windows\main.py"
$originalMainPath = Join-Path $installDir "main.py"
$backupMainPath = Join-Path $installDir "main.py.mac-backup"
$windowsRequirementsPath = Join-Path $installDir "windows\requirements.txt"

# On Windows, we always want to use the Windows version
if (Test-Path $windowsMainPath) {
    Write-ColorOutput "Windows-specific version found in the repository." "Green"
    
    # Check if the file actually contains the Windows-specific marker
    $containsWindowsMarker = Get-Content $windowsMainPath -Raw | Select-String -Pattern "__windows_specific_version__" -Quiet
    
    if (-not $containsWindowsMarker) {
        Write-ColorOutput "WARNING: The file in windows/main.py does not contain the Windows version marker." "Yellow"
        Write-ColorOutput "It may not be properly adapted for Windows systems." "Yellow"
    } else {
        Write-ColorOutput "Confirmed file contains Windows-specific code." "Green"
    }
    
    # Backup original Mac version if not already backed up
    if (-not (Test-Path $backupMainPath) -and (Test-Path $originalMainPath)) {
        Copy-Item -Path $originalMainPath -Destination $backupMainPath -Force
        Write-ColorOutput "Original Mac version backed up to main.py.mac-backup" "Green"
    }
    
    # Copy Windows version to main.py
    Copy-Item -Path $windowsMainPath -Destination $originalMainPath -Force
    Write-ColorOutput "Windows-specific version installed successfully" "Green"
    
    # Install Windows-specific requirements if available
    if (Test-Path $windowsRequirementsPath) {
        Write-ColorOutput "Installing Windows-specific requirements..." "Yellow"
        try {
            if ($uvInstalled) {
                # Use the direct function if defined, otherwise try the command
                if (Get-Command Invoke-UV -ErrorAction SilentlyContinue) {
                    Invoke-UV @("pip", "install", "-r", $windowsRequirementsPath)
                } else {
                    uv pip install -r $windowsRequirementsPath
                }
            } else {
                python -m pip install -r $windowsRequirementsPath
            }
            Write-ColorOutput "Windows-specific requirements installed successfully." "Green"
        } catch {
            Write-ColorOutput "Failed to install Windows-specific requirements: $_" "Red"
            Write-ColorOutput "Some Windows features may not work correctly." "Yellow"
        }
    }
} else {
    Write-ColorOutput "ERROR: Windows-specific version not found in the repository!" "Red"
    Write-ColorOutput "The application may not work correctly on Windows." "Red"
    Write-ColorOutput "Please make sure the repository includes a 'windows' folder with the Windows version." "Red"
    
    # Create a simple version check script that will warn the user
    $warningScript = @"
import platform
import os
import sys

if platform.system() == 'Windows':
    print("WARNING: You are running on Windows but the Windows-specific version was not found.")
    print("The application may not work correctly. Mac-specific dependencies may cause errors.")
    print("Please download a proper Windows version from the repository.")
    input("Press Enter to try to continue anyway...")

# Continue to the original script
exec(open('main.py.mac-backup').read())
"@
    
    # Only create this warning script if we have the Mac backup
    if (Test-Path $backupMainPath) {
        $warningScript | Out-File -FilePath $originalMainPath -Encoding utf8
        Write-ColorOutput "Created a warning script that will notify the user about potential issues." "Yellow"
    }
}

# Explicitly check if we're using the Windows version
if ((Test-Path $originalMainPath) -and (Get-Content $originalMainPath -Raw | Select-String -Pattern "__windows_specific_version__" -Quiet)) {
    Write-ColorOutput "Confirmed Windows-specific version is active and properly installed." "Green"
} else {
    if (Test-Path $windowsMainPath) {
        # Force copy the Windows version again
        Copy-Item -Path $windowsMainPath -Destination $originalMainPath -Force
        Write-ColorOutput "Re-applied Windows-specific version." "Green"
    } else {
        Write-ColorOutput "ERROR: Could not apply Windows-specific version. The app may not work properly on Windows." "Red"
    }
}

# Step 14: Installation complete
Write-ColorOutput "=========================================" "Cyan"
Write-ColorOutput "      Multi-Max Installation Complete     " "Cyan"
Write-ColorOutput "=========================================" "Cyan"
Write-ColorOutput "  To run the application:                 " "Cyan"
Write-ColorOutput "  1. Navigate to $installDir              " "Cyan"
Write-ColorOutput "  2. Execute run.ps1                      " "Cyan"
Write-ColorOutput "     or                                   " "Cyan"
Write-ColorOutput "  1. Double-click the desktop shortcut    " "Cyan"
Write-ColorOutput "     (if created)                         " "Cyan"
Write-ColorOutput "=========================================" "Cyan"
Write-ColorOutput "  Environment configuration is in .env    " "Cyan"
Write-ColorOutput "  Edit this file to customize settings    " "Cyan"
Write-ColorOutput "=========================================" "Cyan"

# Remind about manual activation of venv (with updated venv name)
if (-not (Test-Path env:VIRTUAL_ENV)) {
    Write-ColorOutput "Virtual environment is not activated. Activate it with:" "Yellow"
    Write-ColorOutput "  .\$venvName\Scripts\Activate.ps1" "Yellow"
}

# Add auto-launch option
Write-ColorOutput "Installation complete! The application will launch automatically in 5 seconds..." "Green"
Write-ColorOutput "Press Ctrl+C now if you want to cancel the automatic launch." "Yellow"

# Sleep for 5 seconds to allow reading the message
Start-Sleep -Seconds 5

# Auto-launch the application
try {
    Write-ColorOutput "Launching Multi-Max..." "Green"
    & .\run.ps1
} catch {
    Write-ColorOutput "Failed to auto-launch. You can start the application manually by:" "Yellow"
    Write-ColorOutput "1. Navigate to $installDir" "Yellow"
    Write-ColorOutput "2. Run the script: .\run.ps1" "Yellow"
    # Exit without prompting
    exit 0
}

Write-ColorOutput "Installation completed successfully!" "Green"
Write-ColorOutput "You can now run Multi-Max by activating the virtual environment and running main.py:" "Green"
Write-ColorOutput "cd $installDir" "Cyan"
Write-ColorOutput "$venvName\Scripts\Activate.ps1" "Cyan"
Write-ColorOutput "python main.py" "Cyan"
Write-ColorOutput " " "Green"
Write-ColorOutput "Note: This Windows version uses software rendering mode. Some features may be limited compared to the Mac version." "Yellow" 