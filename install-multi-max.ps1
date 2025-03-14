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
    $overwrite = Read-Host "Do you want to overwrite it? (y/n)"
    if ($overwrite -eq "y" -or $overwrite -eq "Y") {
        Remove-Item -Recurse -Force $installDir
    } else {
        Write-ColorOutput "Using existing directory. Note that this may cause issues if files have changed." "Yellow"
    }
}

if (-not (Test-Path $installDir)) {
    try {
        # Ask the user for the repository URL
        Write-ColorOutput "Please enter the GitHub repository URL for Multi-Max:" "Yellow"
        $repoUrl = Read-Host
        
        # Use default URL if none provided
        if ([string]::IsNullOrWhiteSpace($repoUrl)) {
            Write-ColorOutput "No URL provided, using default repository." "Yellow"
            $repoUrl = "https://github.com/multi-max/multi-max.git"
        }
        
        # Clone the repository
        git clone $repoUrl $installDir
        if (-not $?) {
            throw "Git clone failed with exit code $LASTEXITCODE"
        }
        Write-ColorOutput "Repository cloned successfully to $installDir" "Green"
    } catch {
        Write-ColorOutput "Failed to clone repository: $_" "Red"
        Write-ColorOutput "Please check the repository URL and your internet connection." "Red"
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
if (-not (Test-Path "venv")) {
    try {
        python -m venv venv
        Write-ColorOutput "Virtual environment created successfully." "Green"
    } catch {
        Write-ColorOutput "Failed to create virtual environment: $_" "Red"
        exit 1
    }
}

# Activate virtual environment
Write-ColorOutput "Activating virtual environment..." "Yellow"
try {
    & .\venv\Scripts\Activate.ps1
    Write-ColorOutput "Virtual environment activated." "Green"
} catch {
    Write-ColorOutput "Failed to activate virtual environment: $_" "Red"
    Write-ColorOutput "Trying to continue anyway..." "Yellow"
}

# Step 8: Install UV package manager
Write-ColorOutput "Installing UV package manager..." "Yellow"
$uvInstalled = $false
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        # Install uv using pip
        python -m pip install uv
        
        # Check if uv was installed correctly
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            Write-ColorOutput "UV package manager installed successfully." "Green"
            $uvInstalled = $true
        } else {
            Write-ColorOutput "Failed to install UV with pip. Trying alternative method..." "Yellow"
            
            # Alternative method: Download and run the install script
            Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "$tempDir\install-uv.ps1"
            & $tempDir\install-uv.ps1
            
            # Refresh environment
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            
            if (Get-Command uv -ErrorAction SilentlyContinue) {
                Write-ColorOutput "UV package manager installed successfully (alternative method)." "Green"
                $uvInstalled = $true
            } else {
                Write-ColorOutput "Failed to install UV package manager. Falling back to pip." "Yellow"
            }
        }
    } else {
        Write-ColorOutput "UV package manager is already installed." "Green"
        $uvInstalled = $true
    }
} catch {
    Write-ColorOutput "Error installing UV package manager: $_" "Red"
    Write-ColorOutput "Falling back to pip for package installation." "Yellow"
}

# Step 9: Install Python requirements
Write-ColorOutput "Installing Python requirements..." "Yellow"
if (Test-Path "requirements.txt") {
    try {
        if ($uvInstalled) {
            Write-ColorOutput "Installing packages using UV (faster)..." "Yellow"
            uv pip install -r requirements.txt
            if (-not $?) {
                throw "UV installation failed with exit code $LASTEXITCODE. Falling back to pip."
            }
        } else {
            Write-ColorOutput "Installing packages using pip..." "Yellow"
            python -m pip install --upgrade pip
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
if (Test-Path ".env") {
    Write-ColorOutput ".env file already exists. Keeping existing configuration." "Green"
    Write-ColorOutput "To reset to defaults, delete .env and run the installer again." "Yellow"
} elseif (Test-Path ".env.template") {
    Write-ColorOutput "Creating .env file from template..." "Yellow"
    Copy-Item ".env.template" ".env"
    Write-ColorOutput ".env file created successfully." "Green"
    
    # Configure .env for Windows
    try {
        $envContent = Get-Content ".env" -Raw
        $envContent = $envContent -replace "FORCE_HARDWARE_ACCELERATION=true", "FORCE_HARDWARE_ACCELERATION=false"
        $envContent = $envContent -replace "ALLOW_SOFTWARE_FALLBACK=false", "ALLOW_SOFTWARE_FALLBACK=true"
        Set-Content ".env" $envContent
        Write-ColorOutput ".env configured for Windows." "Green"
    } catch {
        Write-ColorOutput "Failed to configure .env file: $_" "Red"
    }
} else {
    Write-ColorOutput ".env.template not found! Creating basic .env file..." "Yellow"
    @"
# Multi-Max Environment Configuration - Auto-generated
FORCE_HARDWARE_ACCELERATION=false
ALLOW_SOFTWARE_FALLBACK=true
DEFAULT_VIDEO_URL=https://www.youtube.com/watch?v=dQw4w9WgXcQ
ENABLE_MEMORY_TRACING=false
LOG_LEVEL=INFO
FRAME_BUFFER_SIZE=60
"@ | Set-Content ".env"
    Write-ColorOutput "Basic .env file created." "Green"
}

# Step 12: Create desktop shortcut
Write-ColorOutput "Would you like to create a desktop shortcut for easy launching? (y/n)" "Yellow"
$createShortcut = Read-Host
if ($createShortcut -eq "y" -or $createShortcut -eq "Y") {
    Write-ColorOutput "Creating Windows shortcut..." "Yellow"
    
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
& .\venv\Scripts\Activate.ps1
python main.py
"@ | Set-Content "run.ps1"
        
        Write-ColorOutput "Desktop shortcut created successfully." "Green"
    } catch {
        Write-ColorOutput "Failed to create desktop shortcut: $_" "Red"
        
        # Create a batch file as fallback
        @"
@echo off
cd "$installDir"
call venv\Scripts\activate.bat
python main.py
"@ | Set-Content "$env:USERPROFILE\Desktop\Multi-Max.bat"
        
        Write-ColorOutput "Created fallback batch file at Desktop\Multi-Max.bat" "Yellow"
    }
} else {
    Write-ColorOutput "Skipping desktop shortcut creation." "Yellow"
    
    # Create a run script anyway
    @"
# Launcher for Multi-Max
Set-Location "$installDir"
& .\venv\Scripts\Activate.ps1
python main.py
"@ | Set-Content "run.ps1"
    
    Write-ColorOutput "Created run.ps1 script for manual execution." "Green"
}

# Step 13: Installation complete
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

# Remind about manual activation of venv if not already activated
if (-not (Test-Path env:VIRTUAL_ENV)) {
    Write-ColorOutput "Virtual environment is not activated. Activate it with:" "Yellow"
    Write-ColorOutput "  .\venv\Scripts\Activate.ps1" "Yellow"
}

# Keep window open
Read-Host "Press Enter to exit" 