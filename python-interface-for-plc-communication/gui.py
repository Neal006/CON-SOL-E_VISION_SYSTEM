"""PLC Control Dashboard - Streamlit Frontend"""

import streamlit as st
import requests
import time
from datetime import datetime
from typing import Dict, Any, Optional
from config import DashboardConfig, ColorTheme


st.set_page_config(
    page_title=DashboardConfig.PAGE_TITLE,
    page_icon=DashboardConfig.PAGE_ICON,
    layout=DashboardConfig.LAYOUT,
    initial_sidebar_state="expanded"
)


def apply_custom_styles():
    """Apply dark industrial theme CSS."""
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {ColorTheme.DARK_BG};
            color: {ColorTheme.TEXT_PRIMARY};
        }}
        h1, h2, h3 {{
            color: {ColorTheme.TEXT_PRIMARY};
            font-weight: 600;
        }}
        .status-badge {{
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 16px;
            text-align: center;
            margin: 10px 0;
            transition: all 0.3s ease;
        }}
        .status-connected {{
            background: linear-gradient(135deg, {ColorTheme.SUCCESS}22, {ColorTheme.SUCCESS}44);
            border: 2px solid {ColorTheme.SUCCESS};
            color: {ColorTheme.SUCCESS};
        }}
        .status-disconnected {{
            background: linear-gradient(135deg, {ColorTheme.ERROR}22, {ColorTheme.ERROR}44);
            border: 2px solid {ColorTheme.ERROR};
            color: {ColorTheme.ERROR};
        }}
        .dashboard-card {{
            background-color: {ColorTheme.CARD_BG};
            padding: 25px;
            border-radius: 12px;
            border: 1px solid {ColorTheme.SECONDARY}44;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .metric-container {{
            background: linear-gradient(135deg, {ColorTheme.PRIMARY}22, {ColorTheme.INFO}22);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid {ColorTheme.ACCENT};
            margin: 10px 0;
        }}
        .metric-label {{
            color: {ColorTheme.TEXT_SECONDARY};
            font-size: 14px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-value {{
            color: {ColorTheme.TEXT_PRIMARY};
            font-size: 28px;
            font-weight: 700;
            margin-top: 5px;
        }}
        .stButton > button {{
            background: linear-gradient(135deg, {ColorTheme.PRIMARY}, {ColorTheme.INFO});
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(30, 136, 229, 0.3);
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(30, 136, 229, 0.5);
        }}
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {{
            background-color: {ColorTheme.CARD_BG};
            color: {ColorTheme.TEXT_PRIMARY};
            border: 2px solid {ColorTheme.SECONDARY}66;
            border-radius: 8px;
            padding: 10px;
        }}
        .success-message {{
            background: linear-gradient(135deg, {ColorTheme.SUCCESS}22, {ColorTheme.SUCCESS}44);
            border-left: 4px solid {ColorTheme.SUCCESS};
            padding: 15px;
            border-radius: 8px;
            color: {ColorTheme.SUCCESS};
            margin: 10px 0;
        }}
        .error-message {{
            background: linear-gradient(135deg, {ColorTheme.ERROR}22, {ColorTheme.ERROR}44);
            border-left: 4px solid {ColorTheme.ERROR};
            padding: 15px;
            border-radius: 8px;
            color: {ColorTheme.ERROR};
            margin: 10px 0;
        }}
        .warning-message {{
            background: linear-gradient(135deg, {ColorTheme.WARNING}22, {ColorTheme.WARNING}44);
            border-left: 4px solid {ColorTheme.WARNING};
            padding: 15px;
            border-radius: 8px;
            color: {ColorTheme.WARNING};
            margin: 10px 0;
        }}
        hr {{
            border: none;
            border-top: 2px solid {ColorTheme.SECONDARY}44;
            margin: 30px 0;
        }}
        .css-1d391kg, [data-testid="stSidebar"] {{
            background-color: {ColorTheme.CARD_BG};
        }}
    </style>
    """, unsafe_allow_html=True)


class APIClient:
    """Client for the FastAPI backend."""

    def __init__(self):
        self.base_url = DashboardConfig.API_BASE_URL
        self.api_key = DashboardConfig.API_KEY
        self.headers = {"X-API-Key": self.api_key}

    def get_plc_status(self) -> Optional[Dict[str, Any]]:
        """Get PLC connection status."""
        try:
            response = requests.get(f"{self.base_url}/plc/status", headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get PLC status: {str(e)}")
            return None

    def connect_plc(self) -> bool:
        """Connect to PLC."""
        try:
            response = requests.post(f"{self.base_url}/plc/connect", headers=self.headers, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"Failed to connect: {str(e)}")
            return False

    def disconnect_plc(self) -> bool:
        """Disconnect from PLC."""
        try:
            response = requests.post(f"{self.base_url}/plc/disconnect", headers=self.headers, timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"Failed to disconnect: {str(e)}")
            return False

    def read_data(self, device: str, length: int, data_type: str, signed: bool = True) -> Optional[Dict[str, Any]]:
        """Read data from PLC."""
        try:
            response = requests.post(
                f"{self.base_url}/plc/read", headers=self.headers,
                json={"device": device, "length": length, "data_type": data_type, "signed": signed},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to read data: {str(e)}")
            return None

    def write_data(self, device: str, values: list, data_type: str, signed: bool = True) -> Optional[Dict[str, Any]]:
        """Write data to PLC."""
        try:
            response = requests.post(
                f"{self.base_url}/plc/write", headers=self.headers,
                json={"device": device, "values": values, "data_type": data_type, "signed": signed},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to write data: {str(e)}")
            return None


def render_status_badge(connected: bool, error: Optional[str] = None):
    """Render connection status badge."""
    if connected:
        st.markdown('<div class="status-badge status-connected">🟢 PLC CONNECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-disconnected">🔴 PLC DISCONNECTED</div>', unsafe_allow_html=True)
        if error:
            st.markdown(f'<div class="error-message">⚠️ {error}</div>', unsafe_allow_html=True)


def render_metrics(status: Dict[str, Any]):
    """Render uptime and operation count metrics."""
    col1, col2, col3 = st.columns(3)

    with col1:
        uptime = status.get("uptime_seconds", 0)
        if uptime:
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            uptime_str = "N/A"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Uptime</div>
            <div class="metric-value">{uptime_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        reads = status.get("statistics", {}).get("reads", 0)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Read Operations</div>
            <div class="metric-value">{reads:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        writes = status.get("statistics", {}).get("writes", 0)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Write Operations</div>
            <div class="metric-value">{writes:,}</div>
        </div>
        """, unsafe_allow_html=True)


def render_connection_details(status: Dict[str, Any]):
    """Render host, port, and error details."""
    with st.expander("📡 Connection Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Host:** `{status.get('host', 'N/A')}`")
            st.write(f"**Port:** `{status.get('port', 'N/A')}`")
        with col2:
            last_connected = status.get('last_connected')
            if last_connected:
                st.write(f"**Last Connected:** {last_connected}")
            errors = status.get("statistics", {}).get("errors", 0)
            st.write(f"**Total Errors:** `{errors}`")


def main():
    """Main dashboard application."""
    apply_custom_styles()

    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient()
    api = st.session_state.api_client

    st.markdown(f"""
    # {DashboardConfig.PAGE_ICON} PLC Control Dashboard
    ### Real-time Monitoring and Control Interface
    """)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("### Connection Control")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔌 Connect", use_container_width=True):
                with st.spinner("Connecting..."):
                    if api.connect_plc():
                        st.success("Connected!")
                        time.sleep(1)
                        st.rerun()

        with col2:
            if st.button("🔓 Disconnect", use_container_width=True):
                with st.spinner("Disconnecting..."):
                    if api.disconnect_plc():
                        st.success("Disconnected!")
                        time.sleep(1)
                        st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Auto-Refresh")
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", min_value=1, max_value=10, value=DashboardConfig.REFRESH_INTERVAL)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### About")
        st.info("**PLC Dashboard v1.0**\n\nSecure interface for Mitsubishi PLC control using MC Protocol.\n\nBuilt with Streamlit & FastAPI")

    # Status section
    status = api.get_plc_status()
    if status:
        render_status_badge(status.get("connected", False), status.get("error"))
        if status.get("connected"):
            render_metrics(status)
            render_connection_details(status)
        st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.error("❌ Cannot connect to API server. Please ensure the FastAPI backend is running.")
        st.stop()

    # Write section
    st.markdown("## 📝 Write Data to PLC")
    with st.container():
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            write_device = st.text_input("Device Address", value="D100", help="e.g., D100, M0, R0")
        with col2:
            write_data_type = st.selectbox("Data Type", options=["word", "bit", "dword"])
        with col3:
            write_signed = st.checkbox("Signed Values", value=True)

        write_values_str = st.text_input("Values (comma-separated)", value="100", help="e.g., 100, 200, 300")

        if st.button("✍️ Write to PLC", type="primary", use_container_width=True):
            try:
                values = [int(v.strip()) for v in write_values_str.split(",")]
                with st.spinner("Writing data..."):
                    result = api.write_data(device=write_device, values=values, data_type=write_data_type, signed=write_signed)
                    if result and result.get("success"):
                        st.markdown(f'<div class="success-message">✅ {result.get("message")}</div>', unsafe_allow_html=True)
                    else:
                        st.error("Write operation failed!")
            except ValueError:
                st.error("Invalid input! Please enter valid integer values.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Read section
    st.markdown("## 📖 Read Data from PLC")
    with st.container():
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            read_device = st.text_input("Device Address ", value="D100")
        with col2:
            read_length = st.number_input("Length", min_value=1, max_value=100, value=1)
        with col3:
            read_data_type = st.selectbox("Data Type ", options=["word", "bit", "dword"])
        with col4:
            read_signed = st.checkbox("Signed Values ", value=True)

        if st.button("📥 Read from PLC", type="primary", use_container_width=True):
            with st.spinner("Reading data..."):
                result = api.read_data(device=read_device, length=read_length, data_type=read_data_type, signed=read_signed)
                if result and result.get("success"):
                    st.markdown(f'<div class="success-message">✅ {result.get("message")}</div>', unsafe_allow_html=True)
                    data = result.get("data", [])
                    st.markdown("### 📊 Read Values")
                    if isinstance(data, list):
                        cols = st.columns(min(len(data), 5))
                        for i, value in enumerate(data):
                            with cols[i % 5]:
                                st.metric(label=f"Index {i}", value=value)
                    else:
                        st.write(data)
                else:
                    st.error("Read operation failed!")
        st.markdown('</div>', unsafe_allow_html=True)

    # Auto-refresh
    if auto_refresh and status and status.get("connected"):
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
