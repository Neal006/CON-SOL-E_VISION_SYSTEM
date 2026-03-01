# PLC Control Dashboard System

A modern, secure web-based dashboard for monitoring and controlling Mitsubishi PLCs using the MC Protocol. Features a beautiful Streamlit frontend and a robust FastAPI backend with comprehensive security measures.

## ✨ Features

### 🎨 Modern UI
- **Sleek dark theme** with industrial color palette
- **Real-time connection status** with visual indicators
- **Responsive design** that works on all screen sizes
- **Smooth animations** and professional styling

### 🔒 Security
- **API key authentication** - All endpoints protected
- **CORS protection** - Controls allowed origins
- **Rate limiting** - Prevents API abuse
- **Input validation** - Pydantic models validate all data
- **Error sanitization** - No sensitive data exposed

### 📊 Dashboard Features
- **Connection monitoring** - Real-time PLC status and uptime
- **Data writing** - Send values to PLC devices
- **Data reading** - Retrieve values from PLC
- **Statistics tracking** - Monitor operations and errors
- **Auto-refresh** - Configurable automatic updates

### ⚡ Backend Features
- **Async operations** - Non-blocking I/O for performance
- **Automatic reconnection** - Handles connection failures gracefully
- **Thread-safe** - Safe concurrent PLC operations
- **Comprehensive logging** - Track all operations
- **Production-ready** - Optimized and secure

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐         ┌──────────┐
│  User       │ ──────► │  Streamlit   │ ──────► │  FastAPI    │ ──────► │   PLC    │
│  Browser    │ ◄────── │  Dashboard   │ ◄────── │  Server     │ ◄────── │ Device   │
└─────────────┘         └──────────────┘         └─────────────┘         └──────────┘
   (Port 8501)           (GUI Client)             (API + Logic)          (MC Protocol)
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Mitsubishi PLC with MC Protocol enabled
- Network connection to PLC

### Step 1: Clone or Download
```bash
cd /path/to/AUTOVISION_GUARDIAN
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment
1. Copy the example environment file:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` and update these critical values:
   ```env
   # Your PLC's IP address
   PLC_HOST=192.168.1.100
   
   # Generate a secure API key (run this command):
   # python -c "import secrets; print(secrets.token_urlsafe(32))"
   API_KEY=your-generated-secure-key-here
   ```

## 🚀 Usage

### Starting the System

You need to run **both** servers (they work together):

#### Terminal 1: Start the FastAPI Backend
```bash
python api_server.py
```

Or with more control:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

#### Terminal 2: Start the Streamlit Dashboard
```bash
streamlit run gui.py
```

The dashboard will open automatically at: `http://localhost:8501`

### Using the Dashboard

1. **Check Connection Status** - The top banner shows if PLC is connected
2. **Connect to PLC** - Use sidebar "Connect" button if disconnected
3. **Write Data** - Enter device address, values, and click "Write to PLC"
4. **Read Data** - Enter device address, length, and click "Read from PLC"
5. **Monitor** - View uptime, operation counts, and statistics

### Device Address Examples

- **Bits**: `M0`, `X10`, `Y5` - Digital I/O and flags
- **Words**: `D100`, `W0` - 16-bit registers
- **Double Words**: `R0`, `D0` - 32-bit registers

## 🔒 Security Best Practices

### Production Deployment

1. **Change the API Key**
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   Add the generated key to your `.env` file

2. **Restrict CORS Origins**
   ```env
   CORS_ORIGINS=https://your-domain.com
   ```

3. **Disable API Docs** (optional)
   ```env
   ENABLE_DOCS=False
   ```

4. **Use HTTPS** in production
   - Configure reverse proxy (nginx, Apache)
   - Use SSL certificates

5. **Firewall Configuration**
   - Restrict PLC network access
   - Only allow necessary ports

## 📁 Project Structure

```
AUTOVISION_GUARDIAN/
├── config.py              # Configuration management
├── plc_manager.py         # PLC communication logic
├── api_server.py          # FastAPI backend server
├── gui.py                 # Streamlit dashboard frontend
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── .env                   # Your configuration (create this)
├── x.gx3                  # PLC program file (GX Works3) - Optional
├── PLC_CONNECTION_GUIDE.md # How to connect PLC
└── TESTING_GUIDE.md       # How to test the system
```

### About x.gx3 File

The `x.gx3` file is a **Mitsubishi PLC program file** created with **GX Works3** software. This file contains:
- Ladder logic program for the PLC
- Device memory configurations (D registers, M relays, etc.)
- Network and Ethernet settings
- I/O configurations

**Note:** This file is **optional** for running the dashboard. The dashboard communicates with the **running PLC**, not the program file. You only need GX Works3 if you want to:
- View or modify the PLC ladder logic
- Configure PLC network settings
- Download new programs to the PLC

## 🛠️ Configuration Options

### PLC Settings
- `PLC_HOST` - PLC IP address
- `PLC_PORT` - MC Protocol port (default: 1025)
- `PLC_TIMEOUT` - Connection timeout in seconds
- `PLC_MAX_RETRIES` - Retry attempts for failed connections
- `PLC_RETRY_DELAY` - Delay between retries

### API Settings
- `API_KEY` - Authentication key (CHANGE THIS!)
- `API_HOST` - Server bind address
- `API_PORT` - Server port
- `CORS_ORIGINS` - Allowed request origins
- `RATE_LIMIT_PER_MINUTE` - Request rate limit
- `ENABLE_DOCS` - Enable/disable API documentation

### Dashboard Settings
- `API_BASE_URL` - Backend API URL
- `REFRESH_INTERVAL` - Auto-refresh interval in seconds

## 📖 API Endpoints

All endpoints require `X-API-Key` header (except `/` and `/health`)

### Status & Control
- `GET /` - API information
- `GET /health` - Health check
- `GET /plc/status` - PLC connection status
- `POST /plc/connect` - Connect to PLC
- `POST /plc/disconnect` - Disconnect from PLC

### Data Operations
- `POST /plc/read` - Read data from PLC
  ```json
  {
    "device": "D100",
    "length": 10,
    "data_type": "word",
    "signed": true
  }
  ```

- `POST /plc/write` - Write data to PLC
  ```json
  {
    "device": "D100",
    "values": [100, 200, 300],
    "data_type": "word",
    "signed": true
  }
  ```

## 🐛 Troubleshooting

### Cannot Connect to PLC
- Verify PLC IP address in `.env`
- Check network connectivity: `ping 192.168.1.100`
- Ensure MC Protocol is enabled on PLC
- Verify port 1025 is not blocked by firewall

### Dashboard Shows "Cannot connect to API server"
- Ensure FastAPI server is running
- Check `API_BASE_URL` in `.env`
- Verify port 8000 is not in use

### API Key Errors
- Ensure `.env` file exists
- Verify `API_KEY` is set in `.env`
- Restart both servers after changing `.env`

## 📝 Code Comments

All code is extensively commented with:
- **Module docstrings** - Purpose of each file
- **Function docstrings** - What each function does
- **Inline comments** - Explanation of complex logic
- **Simple language** - Easy for anyone to understand

## 🎨 Color Theme

The dashboard uses a professional industrial color palette:

- **Primary**: Blue (#1E88E5) - Main actions
- **Success**: Green (#4CAF50) - Connected, success states
- **Error**: Red (#F44336) - Errors, disconnected
- **Warning**: Orange (#FF9800) - Cautions
- **Accent**: Deep Orange (#FF6F00) - Highlights
- **Background**: Dark (#0E1117) - Main background

## 📄 License

This project is provided as-is for industrial automation purposes.

## 🤝 Support

For issues or questions:
1. Check this README
2. Review code comments
3. Check API documentation at `/docs`

## 🔄 Updates

To update dependencies:
```bash
pip install --upgrade -r requirements.txt
```

---

**Built with ❤️ for Industrial Automation**
