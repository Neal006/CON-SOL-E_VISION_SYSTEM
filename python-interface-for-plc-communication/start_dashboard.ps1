# PLC Dashboard - Start Frontend Dashboard
# This script starts the Streamlit dashboard

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PLC Control Dashboard - Frontend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if backend is running
Write-Host "Checking if backend server is running..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "Backend server is running!" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Backend server is not running!" -ForegroundColor Red
    Write-Host "Please start the backend first using: .\start_backend.ps1" -ForegroundColor Yellow
    Write-Host "Or open another terminal and run: python api_server.py" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit
    }
}

Write-Host ""
Write-Host "Starting Streamlit dashboard..." -ForegroundColor Green
Write-Host "Dashboard will open in your browser automatically" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Yellow
Write-Host ""

# Start Streamlit
streamlit run gui.py
