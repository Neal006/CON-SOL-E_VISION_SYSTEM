"""PLC Manager — Mitsubishi MC Protocol communication."""

import rk_mcprotocol as mc
import time
import threading
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from config import PLCConfig


class PLCManager:
    """Manages PLC connection and read/write operations."""

    def __init__(self):
        self.host = PLCConfig.HOST
        self.port = PLCConfig.PORT
        self.timeout = PLCConfig.TIMEOUT
        self.socket = None
        self.is_connected = False
        self.last_connection_time = None
        self.connection_error = None
        self._lock = threading.Lock()
        self.read_count = 0
        self.write_count = 0
        self.error_count = 0

    def connect(self) -> Tuple[bool, str]:
        """Open socket connection to PLC."""
        with self._lock:
            try:
                self.socket = mc.open_socket(self.host, self.port)
                self.is_connected = True
                self.last_connection_time = datetime.now()
                self.connection_error = None
                return True, f"Successfully connected to PLC at {self.host}:{self.port}"
            except Exception as e:
                self.is_connected = False
                self.socket = None
                self.connection_error = str(e)
                return False, f"Failed to connect to PLC: {str(e)}"

    def disconnect(self) -> None:
        """Close PLC connection."""
        with self._lock:
            try:
                if self.socket:
                    self.socket.close()
            except Exception as e:
                print(f"Error during disconnect: {e}")
            finally:
                self.socket = None
                self.is_connected = False

    def reconnect(self) -> Tuple[bool, str]:
        """Disconnect and reconnect."""
        self.disconnect()
        time.sleep(PLCConfig.RETRY_DELAY)
        return self.connect()

    def ensure_connection(self) -> Tuple[bool, str]:
        """Reconnect if not connected."""
        if not self.is_connected or self.socket is None:
            return self.connect()
        return True, "Connection already active"

    def read_bits(self, device: str, length: int = 1) -> Dict[str, Any]:
        """Read bit data (X, Y, M, L devices)."""
        success, msg = self.ensure_connection()
        if not success:
            self.error_count += 1
            return {"success": False, "data": None, "message": msg}
        with self._lock:
            try:
                data = mc.read_bit(self.socket, headdevice=device, length=length)
                self.read_count += 1
                return {"success": True, "data": data, "message": f"Read {length} bits from {device}"}
            except Exception as e:
                self.error_count += 1
                return {"success": False, "data": None, "message": f"Error reading bits from {device}: {str(e)}"}

    def read_words(self, device: str, length: int = 1, signed: bool = True) -> Dict[str, Any]:
        """Read 16-bit word data (D, W, R devices)."""
        success, msg = self.ensure_connection()
        if not success:
            self.error_count += 1
            return {"success": False, "data": None, "message": msg}
        with self._lock:
            try:
                data = mc.read_sign_word(self.socket, headdevice=device, length=length, signed_type=signed)
                self.read_count += 1
                return {"success": True, "data": data, "message": f"Read {length} words from {device}"}
            except Exception as e:
                self.error_count += 1
                return {"success": False, "data": None, "message": f"Error reading words from {device}: {str(e)}"}

    def read_dwords(self, device: str, length: int = 1, signed: bool = True) -> Dict[str, Any]:
        """Read 32-bit double word data."""
        success, msg = self.ensure_connection()
        if not success:
            self.error_count += 1
            return {"success": False, "data": None, "message": msg}
        with self._lock:
            try:
                data = mc.read_sign_Dword(self.socket, headdevice=device, length=length, signed_type=signed)
                self.read_count += 1
                return {"success": True, "data": data, "message": f"Read {length} dwords from {device}"}
            except Exception as e:
                self.error_count += 1
                return {"success": False, "data": None, "message": f"Error reading dwords from {device}: {str(e)}"}

    def write_bits(self, device: str, values: List[int]) -> Dict[str, Any]:
        """Write bit data to PLC."""
        success, msg = self.ensure_connection()
        if not success:
            self.error_count += 1
            return {"success": False, "message": msg}
        with self._lock:
            try:
                mc.write_bit(self.socket, headdevice=device, data_list=values)
                self.write_count += 1
                return {"success": True, "message": f"Wrote {len(values)} bits to {device}"}
            except Exception as e:
                self.error_count += 1
                return {"success": False, "message": f"Error writing bits to {device}: {str(e)}"}

    def write_words(self, device: str, values: List[int], signed: bool = True) -> Dict[str, Any]:
        """Write 16-bit word data to PLC."""
        success, msg = self.ensure_connection()
        if not success:
            self.error_count += 1
            return {"success": False, "message": msg}
        with self._lock:
            try:
                mc.write_sign_word(self.socket, headdevice=device, data_list=values, signed_type=signed)
                self.write_count += 1
                return {"success": True, "message": f"Wrote {len(values)} words to {device}"}
            except Exception as e:
                self.error_count += 1
                return {"success": False, "message": f"Error writing words to {device}: {str(e)}"}

    def write_dwords(self, device: str, values: List[int], signed: bool = True) -> Dict[str, Any]:
        """Write 32-bit double word data to PLC."""
        success, msg = self.ensure_connection()
        if not success:
            self.error_count += 1
            return {"success": False, "message": msg}
        with self._lock:
            try:
                mc.write_sign_Dword(self.socket, headdevice=device, data_list=values, signed_type=signed)
                self.write_count += 1
                return {"success": True, "message": f"Wrote {len(values)} dwords to {device}"}
            except Exception as e:
                self.error_count += 1
                return {"success": False, "message": f"Error writing dwords to {device}: {str(e)}"}

    def get_status(self) -> Dict[str, Any]:
        """Get connection status and statistics."""
        uptime = None
        if self.is_connected and self.last_connection_time:
            uptime = (datetime.now() - self.last_connection_time).total_seconds()
        return {
            "connected": self.is_connected,
            "host": self.host,
            "port": self.port,
            "uptime_seconds": uptime,
            "last_connected": self.last_connection_time.isoformat() if self.last_connection_time else None,
            "error": self.connection_error,
            "statistics": {"reads": self.read_count, "writes": self.write_count, "errors": self.error_count}
        }


# Global singleton
plc = PLCManager()
