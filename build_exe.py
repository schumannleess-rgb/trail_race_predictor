"""
Trail Master EXE 打包脚本

使用 PyInstaller 将 Streamlit 应用打包为独立 EXE

使用方法:
    python build_exe.py

输出:
    dist/TrailMaster.exe

注意:
    Streamlit 应用打包后体积较大 (约 200-300MB)
    首次启动可能需要 10-20 秒
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def find_streamlit_static():
    """查找 Streamlit 静态文件路径"""
    import streamlit
    streamlit_path = Path(streamlit.__file__).parent
    static_path = streamlit_path / "static"
    return str(static_path)


def build_exe():
    """构建 EXE"""
    print("=" * 60)
    print("Trail Master EXE Builder")
    print("=" * 60)

    project_dir = Path(__file__).parent

    print("\n[1/4] 检查依赖...")
    try:
        import streamlit
        import lightgbm
        import fitparse
        print("  ✓ 所有依赖已安装")
    except ImportError as e:
        print(f"  ✗ 缺少依赖: {e}")
        print("  请运行: pip install -r requirements.txt")
        return False

    print("\n[2/4] 查找 Streamlit 静态文件...")
    static_path = find_streamlit_static()
    print(f"  ✓ 找到: {static_path}")

    print("\n[3/4] 清理旧构建...")
    for dir_name in ['build', 'dist', '__pycache__']:
        dir_path = project_dir / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  ✓ 已删除: {dir_name}")

    for spec_file in project_dir.glob('*.spec'):
        spec_file.unlink()
        print(f"  ✓ 已删除: {spec_file.name}")

    print("\n[4/4] 构建 EXE...")
    
    separator = ';' if sys.platform == 'win32' else ':'
    
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',
        '--windowed',
        '--name', 'TrailMaster',
        '--add-data', f'{static_path}{separator}streamlit/static',
        '--hidden-import', 'streamlit',
        '--hidden-import', 'streamlit.web.cli',
        '--hidden-import', 'streamlit.runtime.scriptrunner',
        '--hidden-import', 'lightgbm',
        '--hidden-import', 'fitparse',
        '--hidden-import', 'fitparse.fitfile',
        '--hidden-import', 'numpy',
        '--hidden-import', 'scipy',
        '--hidden-import', 'scipy.sparse',
        '--hidden-import', 'scipy.sparse.csgraph',
        '--collect-all', 'streamlit',
        '--collect-all', 'lightgbm',
        str(project_dir / 'launcher.py'),
    ]

    print(f"  执行: PyInstaller --onefile --windowed --name TrailMaster launcher.py")
    print("  (这可能需要几分钟...)")
    
    result = subprocess.run(cmd, cwd=project_dir)

    if result.returncode == 0:
        exe_path = project_dir / 'dist' / 'TrailMaster.exe'
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print("\n" + "=" * 60)
            print("构建成功!")
            print(f"  输出: {exe_path}")
            print(f"  大小: {size_mb:.1f} MB")
            print("\n使用方法:")
            print("  双击 TrailMaster.exe 启动应用")
            print("  浏览器将自动打开 http://localhost:8501")
            print("=" * 60)
            return True
        else:
            print("\n✗ 构建失败: 未找到输出文件")
            return False
    else:
        print("\n✗ 构建失败")
        return False


if __name__ == '__main__':
    success = build_exe()
    sys.exit(0 if success else 1)
