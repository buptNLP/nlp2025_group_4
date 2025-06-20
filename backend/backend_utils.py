"""
后端工具模块 - 替代 Streamlit 依赖
"""

import logging
from typing import Any, Optional
from contextlib import contextmanager
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class BackendLogger:
    """后端日志器 - 替代 Streamlit 的显示函数"""
    
    def __init__(self, name: str = "RAG_Backend"):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str):
        """信息日志"""
        self.logger.info(message)
    
    def success(self, message: str):
        """成功日志"""
        self.logger.info(f"✅ {message}")
    
    def warning(self, message: str):
        """警告日志"""
        self.logger.warning(f"⚠️ {message}")
    
    def error(self, message: str):
        """错误日志"""
        self.logger.error(f"❌ {message}")
    
    def debug(self, message: str):
        """调试日志"""
        self.logger.debug(message)
    
    @contextmanager
    def spinner(self, message: str):
        """模拟 Streamlit 的 spinner"""
        self.info(f"🔄 {message}")
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.info(f"✅ 完成 ({elapsed:.2f}s)")
    
    def progress(self, value: float, text: str = ""):
        """模拟进度条"""
        percentage = int(value * 100)
        self.info(f"📊 {text} {percentage}%")
    
    def metric(self, label: str, value: Any):
        """显示指标"""
        self.info(f"📈 {label}: {value}")
    
    def balloons(self):
        """庆祝动画（仅日志）"""
        self.success("🎉 处理完成！")

# 创建全局日志器实例
backend_logger = BackendLogger()

# 提供兼容的接口
def info(message: str):
    backend_logger.info(message)

def success(message: str):
    backend_logger.success(message)

def warning(message: str):
    backend_logger.warning(message)

def error(message: str):
    backend_logger.error(message)

def spinner(message: str):
    return backend_logger.spinner(message)

def progress(value: float, text: str = ""):
    backend_logger.progress(value, text)

def balloons():
    backend_logger.balloons()

def rerun():
    """空实现，兼容性"""
    pass

# 会话状态模拟
class SessionState:
    """模拟 Streamlit session_state"""
    
    def __init__(self):
        self._state = {}
    
    def get(self, key: str, default=None):
        return self._state.get(key, default)
    
    def __getattr__(self, key: str):
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        return self._state.get(key)
    
    def __setattr__(self, key: str, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
        else:
            self._state[key] = value
    
    def __contains__(self, key: str):
        return key in self._state
    
    def __delitem__(self, key: str):
        if key in self._state:
            del self._state[key]

# 创建全局会话状态
session_state = SessionState()

# Streamlit 兼容接口
class StreamlitCompat:
    """Streamlit 兼容层"""
    
    def __init__(self):
        self.session_state = session_state
    
    def info(self, message: str):
        info(message)
    
    def success(self, message: str):
        success(message)
    
    def warning(self, message: str):
        warning(message)
    
    def error(self, message: str):
        error(message)
    
    def spinner(self, message: str):
        return spinner(message)
    
    def progress(self, value: float, text: str = ""):
        progress(value, text)
    
    def balloons(self):
        balloons()
    
    def rerun(self):
        rerun()

# 创建 st 兼容对象
st = StreamlitCompat()