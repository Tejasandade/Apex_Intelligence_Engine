import subprocess
import sys

with open("d:\\Apex_Intelligence_Engine\\true_crash.log", "w") as f:
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "scripts.run_live", "--symbol", "btcusdt", "--paper"],
        stdout=f,
        stderr=subprocess.STDOUT
    )
    process.wait(timeout=20)
