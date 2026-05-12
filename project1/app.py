"""
Enterprise-RAG: Application Launcher
=====================================
    python app.py              # Start FastAPI + open browser
    python app.py --no-browser # Start FastAPI only
"""
import sys
import webbrowser
from pathlib import Path


def main():
    # ── Ensure project root is on path and cwd is correct ──
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))

    import os
    os.chdir(str(project_root))

    host = "0.0.0.0"
    port = 8080
    no_browser = "--no-browser" in sys.argv

    # ── Verify dependencies before starting ──
    print("🔍 检查依赖...")
    missing = []
    for mod in ["fastapi", "uvicorn", "src.config", "src.pipeline"]:
        try:
            __import__(mod)
        except ImportError as e:
            missing.append(f"  ❌ {mod}: {e}")
    if missing:
        print("\n⚠️  缺少依赖，请先安装：")
        for m in missing:
            print(m)
        print("\n  pip install -r requirements.txt\n")
        sys.exit(1)
    print("  ✅ 依赖检查通过")

    # ── Check port ──
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port_in_use = sock.connect_ex(("127.0.0.1", port)) == 0
    sock.close()
    if port_in_use:
        print(f"\n⚠️  端口 {port} 已被占用，尝试终止已有进程...")
        print(f"  或使用: netstat -ano | findstr :{port}  查看占用进程")
        print(f"  然后: taskkill /PID <进程号> /F  终止进程\n")

    # ── Banner ──
    print()
    print("=" * 56)
    print("  ⚡ Enterprise-RAG — 赛级知识库问答系统")
    print("=" * 56)
    print(f"  界面:   http://localhost:{port}")
    print(f"  API 文档:   http://localhost:{port}/docs")
    print(f"  健康检查:   http://localhost:{port}/health")
    print("=" * 56)
    print("  启动中，请稍候...")
    print()

    # ── Open browser after server is ready ──
    if not no_browser:
        def _wait_and_open():
            import time
            import urllib.request
            # Wait for server to be ready (up to 30 seconds)
            for _ in range(30):
                time.sleep(1)
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    print(f"\n✅ 服务已就绪，打开浏览器...")
                    webbrowser.open(f"http://localhost:{port}")
                    return
                except Exception:
                    pass
            print(f"\n⚠️  服务启动超时，请手动访问 http://localhost:{port}")

        import threading
        threading.Thread(target=_wait_and_open, daemon=True).start()

    # ── Start FastAPI ──
    import uvicorn
    try:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("\n请检查:")
        print("  1. pip install -r requirements.txt  是否完整安装")
        print("  2. 端口 8080 是否被其他程序占用")
        print("  3. config.yaml 配置是否正确")
        sys.exit(1)


if __name__ == "__main__":
    main()
