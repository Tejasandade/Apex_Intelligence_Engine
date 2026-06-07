"""
Apex Intelligence Engine V6 — Overnight Monitor Wrapper
=====================================================
Runs scripts/run_live.py as a subprocess and monitors it for:
- RAM usage (> 6.5GB kills process)
- WebSocket disconnects (> 60s kills process)
- NaN/Infinity in features (kills process)
- Uncaught exceptions (kills process)
Logs status every 30 minutes.
"""

import os
import sys
import time
import subprocess
import psutil
import datetime
from pathlib import Path

def get_ram_usage(pid: int) -> float:
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / (1024 * 1024 * 1024) # in GB
    except psutil.NoSuchProcess:
        return 0.0

def main():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    status_log = log_dir / "overnight_status.log"
    
    # Open status log and write header
    with open(status_log, "a") as f:
        f.write(f"\n--- Overnight Monitor Started at {datetime.datetime.now()} ---\n")
        f.flush()
        
    cmd = [sys.executable, "scripts/run_live.py", "--market", "crypto", "--symbol", "btcusdt", "--no-dashboard"]
    
    print(f"Starting LiveRunner: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1, # Line buffered
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(Path(__file__).parent.parent)}
    )
    
    start_time = time.time()
    last_status_time = start_time
    
    # State tracking
    ws_disconnected_at = 0.0
    oracle_levels_active = 0
    signals_evaluated = 0
    trades_taken = 0
    recent_warnings = []
    
    ws_kline_connected = False
    ws_book_connected = False # We'll infer from the single websocket state
    
    while True:
        # Check if process died
        if process.poll() is not None:
            msg = f"CRITICAL: LiveRunner exited with code {process.returncode}"
            print(msg)
            with open(status_log, "a") as f: f.write(f"{datetime.datetime.now()} - {msg}\n")
            break
            
        # Check RAM
        ram_gb = get_ram_usage(process.pid)
        if ram_gb > 6.5:
            msg = f"CRITICAL: RAM exceeded 6.5GB ({ram_gb:.2f}GB). Terminating."
            print(msg)
            with open(status_log, "a") as f: f.write(f"{datetime.datetime.now()} - {msg}\n")
            process.terminate()
            break
            
        # Check WebSocket timeout
        if ws_disconnected_at > 0 and (time.time() - ws_disconnected_at) > 60:
            msg = f"CRITICAL: WebSocket failed to reconnect within 60s. Terminating."
            print(msg)
            with open(status_log, "a") as f: f.write(f"{datetime.datetime.now()} - {msg}\n")
            process.terminate()
            break
            
        # A more robust non-blocking read using a background thread
        pass

    import threading
    import queue
    import json

    log_queue = queue.Queue()

    def enqueue_output(out, q):
        for line in iter(out.readline, ''):
            q.put(line)
        out.close()

    t = threading.Thread(target=enqueue_output, args=(process.stdout, log_queue))
    t.daemon = True
    t.start()
    
    oracle_rebuilt_fired = False
    
    while True:
        # Check if process died
        if process.poll() is not None:
            msg = f"CRITICAL: LiveRunner exited with code {process.returncode}"
            print(msg)
            with open(status_log, "a") as f: f.write(f"{datetime.datetime.now()} - {msg}\n")
            break
            
        # Check RAM
        ram_gb = get_ram_usage(process.pid)
        if ram_gb > 6.5:
            msg = f"CRITICAL: RAM exceeded 6.5GB ({ram_gb:.2f}GB). Terminating."
            print(msg)
            with open(status_log, "a") as f: f.write(f"{datetime.datetime.now()} - {msg}\n")
            process.terminate()
            break
            
        # Check WebSocket timeout
        if ws_disconnected_at > 0 and (time.time() - ws_disconnected_at) > 60:
            msg = f"CRITICAL: WebSocket failed to reconnect within 60s. Terminating."
            print(msg)
            with open(status_log, "a") as f: f.write(f"{datetime.datetime.now()} - {msg}\n")
            process.terminate()
            break
            
        # Process logs (limit to 100 per iteration to prevent blocking)
        processed = 0
        while not log_queue.empty() and processed < 100:
            line = log_queue.get_nowait()
            processed += 1
            print(line, end="") # Echo to stdout
            
            # Check for NaNs or Infinity
            if "NaN" in line or "Infinity" in line or "nan" in line.lower():
                msg = f"CRITICAL: NaN or Infinity detected in logs. Terminating.\nLine: {line.strip()}"
                print(msg)
                with open(status_log, "a") as f: 
                    f.write(f"{datetime.datetime.now()} - {msg}\n")
                    f.flush()
                process.terminate()
                sys.exit(1)
                
            # Parse JSON logs if possible
            try:
                log_data = json.loads(line)
                event = log_data.get("event", "")
                
                if event == "ws_connected":
                    ws_kline_connected = True
                    ws_book_connected = True
                    ws_disconnected_at = 0.0
                elif event == "ws_disconnected":
                    ws_kline_connected = False
                    ws_book_connected = False
                    ws_disconnected_at = time.time()
                elif event == "oracle_levels_built" or event == "oracle_booted":
                    oracle_levels_active = log_data.get("active_levels", oracle_levels_active)
                elif event == "oracle_levels_refreshed":
                    oracle_rebuilt_fired = True
                elif event == "signal_evaluated":
                    signals_evaluated += 1
                elif event == "position_opened":
                    trades_taken += 1
                elif log_data.get("level") in ["warning", "error", "critical"]:
                    recent_warnings.append(line.strip())
                    
                # Check for unhandled exceptions
                if "Traceback" in line or "Exception" in line and log_data.get("level") == "error":
                     msg = f"CRITICAL: Exception detected. Terminating.\nLine: {line.strip()}"
                     print(msg)
                     with open(status_log, "a") as f: f.write(f"{datetime.datetime.now()} - {msg}\n")
                     process.terminate()
                     sys.exit(1)
                     
            except json.JSONDecodeError:
                # Standard text log
                if "Traceback" in line:
                     msg = f"CRITICAL: Traceback detected in stdout. Terminating.\nLine: {line.strip()}"
                     print(msg)
                     with open(status_log, "a") as f: f.write(f"{datetime.datetime.now()} - {msg}\n")
                     process.terminate()
                     sys.exit(1)
        
        current_time = time.time()
        
        # Every 30 minutes, log status (but do the first one immediately after 5 seconds to verify)
        if current_time - last_status_time >= 1800 or (current_time - start_time > 5 and last_status_time == start_time):
            last_status_time = current_time
            uptime = current_time - start_time
            
            status_msg = (
                f"\n--- Status Update @ {datetime.datetime.now()} ---\n"
                f"1. RAM Usage: {ram_gb:.2f} GB\n"
                f"2. Oracle HTF Levels Active: {oracle_levels_active}\n"
                f"3. Signals Evaluated: {signals_evaluated} | Trades Taken: {trades_taken}\n"
                f"4. WebSocket Uptime: kline={'UP' if ws_kline_connected else 'DOWN'}, bookTicker={'UP' if ws_book_connected else 'DOWN'}\n"
                f"5. Warnings/Errors (last 30m): {len(recent_warnings)}\n"
            )
            for w in recent_warnings[-5:]: # Keep it concise
                status_msg += f"   - {w}\n"
            recent_warnings.clear()
            
            print(status_msg)
            with open(status_log, "a") as f: 
                f.write(status_msg)
                f.flush()
            
        # At 4 hours, check Oracle refresh
        if current_time - start_time >= 4 * 3600 and current_time - start_time < 4 * 3600 + 10:
            if not oracle_rebuilt_fired:
                msg = f"CRITICAL: 4 hours elapsed and Oracle refresh_loop did not fire. Terminating."
                print(msg)
                with open(status_log, "a") as f: f.write(f"{datetime.datetime.now()} - {msg}\n")
                process.terminate()
                sys.exit(1)
            else:
                msg = f"SUCCESS: Oracle refresh_loop confirmed fired at 4-hour mark."
                print(msg)
                with open(status_log, "a") as f: f.write(f"{datetime.datetime.now()} - {msg}\n")
                
        # Stop at 6am
        if datetime.datetime.now().hour == 6 and datetime.datetime.now().minute == 0:
            msg = f"Target time 6:00 AM reached. Stopping normally."
            print(msg)
            with open(status_log, "a") as f: f.write(f"{datetime.datetime.now()} - {msg}\n")
            process.terminate()
            break
            
        time.sleep(1)

if __name__ == "__main__":
    main()
