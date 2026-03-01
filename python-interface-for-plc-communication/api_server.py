"""FastAPI Backend Server for PLC Dashboard"""

from fastapi import FastAPI, HTTPException, Security, status, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
import time
from collections import defaultdict
from datetime import datetime, timedelta

from config import APIConfig
from plc_manager import plc


app = FastAPI(
    title="PLC Control API",
    description="Secure REST API for Mitsubishi PLC communication",
    version="1.0.0",
    docs_url="/docs" if APIConfig.ENABLE_DOCS else None,
    redoc_url="/redoc" if APIConfig.ENABLE_DOCS else None
)

# API key auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key."""
    if api_key != APIConfig.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
rate_limit_storage = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Limit requests per IP per minute."""
    client_ip = request.client.host
    current_time = datetime.now()

    rate_limit_storage[client_ip] = [
        ts for ts in rate_limit_storage[client_ip]
        if current_time - ts < timedelta(minutes=1)
    ]

    if len(rate_limit_storage[client_ip]) >= APIConfig.RATE_LIMIT_PER_MINUTE:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )

    rate_limit_storage[client_ip].append(current_time)
    response = await call_next(request)
    return response


# Request/Response models

class ReadRequest(BaseModel):
    """Read data request model."""
    device: str = Field(..., description="Device address (e.g., 'D100', 'M0')")
    length: int = Field(1, ge=1, le=1000)
    data_type: Literal["bit", "word", "dword"] = Field(...)
    signed: bool = Field(True)

    @validator('device')
    def validate_device(cls, v):
        if not v or not v.strip():
            raise ValueError("Device address cannot be empty")
        return v.strip().upper()


class WriteRequest(BaseModel):
    """Write data request model."""
    device: str = Field(..., description="Device address (e.g., 'D100', 'M0')")
    values: List[int] = Field(...)
    data_type: Literal["bit", "word", "dword"] = Field(...)
    signed: bool = Field(True)

    @validator('device')
    def validate_device(cls, v):
        if not v or not v.strip():
            raise ValueError("Device address cannot be empty")
        return v.strip().upper()

    @validator('values')
    def validate_values(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Values list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Cannot write more than 1000 values at once")
        return v


class StatusResponse(BaseModel):
    """PLC status response model."""
    connected: bool
    host: str
    port: int
    uptime_seconds: Optional[float]
    last_connected: Optional[str]
    error: Optional[str]
    statistics: dict


# Endpoints

@app.get("/")
async def root():
    """API info."""
    return {"message": "PLC Control API", "version": "1.0.0", "status": "running",
            "docs": "/docs" if APIConfig.ENABLE_DOCS else "disabled"}


@app.get("/health")
async def health_check():
    """Health check."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/plc/status", response_model=StatusResponse)
async def get_plc_status(api_key: str = Security(verify_api_key)):
    """Get PLC connection status and statistics."""
    try:
        return plc.get_status()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error getting PLC status: {str(e)}")


@app.post("/plc/connect")
async def connect_plc(api_key: str = Security(verify_api_key)):
    """Connect to PLC."""
    try:
        success, message = plc.connect()
        if success:
            return {"success": True, "message": message, "status": plc.get_status()}
        else:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error connecting to PLC: {str(e)}")


@app.post("/plc/disconnect")
async def disconnect_plc(api_key: str = Security(verify_api_key)):
    """Disconnect from PLC."""
    try:
        plc.disconnect()
        return {"success": True, "message": "Disconnected from PLC"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error disconnecting from PLC: {str(e)}")


@app.post("/plc/read")
async def read_from_plc(request: ReadRequest, api_key: str = Security(verify_api_key)):
    """Read bit/word/dword data from PLC."""
    try:
        if request.data_type == "bit":
            result = plc.read_bits(request.device, request.length)
        elif request.data_type == "word":
            result = plc.read_words(request.device, request.length, request.signed)
        elif request.data_type == "dword":
            result = plc.read_dwords(request.device, request.length, request.signed)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid data_type")

        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error reading from PLC: {str(e)}")


@app.post("/plc/write")
async def write_to_plc(request: WriteRequest, api_key: str = Security(verify_api_key)):
    """Write bit/word/dword data to PLC."""
    try:
        if request.data_type == "bit":
            result = plc.write_bits(request.device, request.values)
        elif request.data_type == "word":
            result = plc.write_words(request.device, request.values, request.signed)
        elif request.data_type == "dword":
            result = plc.write_dwords(request.device, request.values, request.signed)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid data_type")

        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error writing to PLC: {str(e)}")


# Startup/Shutdown

@app.on_event("startup")
async def startup_event():
    """Auto-connect to PLC on server start."""
    print("=" * 60)
    print("PLC Control API Server Starting...")
    print("=" * 60)
    success, message = plc.connect()
    print(f"{'✓' if success else '✗'} {message}")
    if not success:
        print("  API will continue running. PLC connection can be established later.")
    print("=" * 60)
    print(f"API Server running at http://{APIConfig.HOST}:{APIConfig.PORT}")
    if APIConfig.ENABLE_DOCS:
        print(f"API Documentation: http://{APIConfig.HOST}:{APIConfig.PORT}/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect PLC on server stop."""
    print("\nShutting down API server...")
    plc.disconnect()
    print("✓ Disconnected from PLC")
    print("✓ Server stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APIConfig.HOST, port=APIConfig.PORT, log_level="info")
