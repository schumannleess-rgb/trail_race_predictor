"""
Trail Master Launcher

EXE 入口点，用于启动 Streamlit 应用
"""

import os
import sys
import subprocess
from pathlib import Path


def get_streamlit_path():
    """获取 streamlit 可执行文件路径"""
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent
    
    streamlit_exe = base_path / "Scripts" / "streamlit.exe"
    if streamlit_exe.exists():
        return str(streamlit_exe)
    
    return sys.executable


def main():
    """启动 Streamlit 应用"""
    if getattr(sys, 'frozen', False):
        app_path = Path(sys._MEIPASS) / "app.py"
    else:
        app_path = Path(__file__).parent / "app.py"
    
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    cmd = [sys.executable, '-m', 'streamlit', 'run', str(app_path)]
    
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
