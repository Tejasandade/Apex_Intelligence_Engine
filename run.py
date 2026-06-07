import subprocess
print("Starting")
res = subprocess.run([r"d:\Apex_Intelligence_Engine\venv\Scripts\python.exe", "-u", "-m", "scripts.run_live", "--symbol", "btcusdt", "--paper"], capture_output=True, text=True)
with open("d:\\Apex_Intelligence_Engine\\final_out.txt", "w", encoding="utf-8") as f:
    f.write(res.stdout)
with open("d:\\Apex_Intelligence_Engine\\final_err.txt", "w", encoding="utf-8") as f:
    f.write(res.stderr)
print("Done")
