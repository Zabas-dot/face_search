"""Run the Face Search tests:  .venv\\Scripts\\python tests\\run_tests.py

Each test file runs in its own process with fake face data and a temporary
home folder, so nothing on your computer is changed. Takes about 30 seconds.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = ["test_engine.py", "test_gui.py"]


def main():
    env = dict(os.environ)
    extra = [ROOT, HERE] + ([env["FS_TEST_STUBS"]] if env.get("FS_TEST_STUBS") else [])
    env["PYTHONPATH"] = os.pathsep.join(extra + [env.get("PYTHONPATH", "")])
    failed = []
    for name in TESTS:
        print("=" * 60 + "\n" + name)
        r = subprocess.run([sys.executable, os.path.join(HERE, name)], env=env, cwd=HERE)
        if r.returncode != 0:
            failed.append(name)
    print("=" * 60)
    print("FAILED: " + ", ".join(failed) if failed else "All tests passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
