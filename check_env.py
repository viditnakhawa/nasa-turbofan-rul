import os
import sys

def check_env():
    print("--- Python Information ---")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    
    print("\n--- Current Working Directory ---")
    print(os.getcwd())
    
    print("\n--- Environment PATH ---")
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    for d in path_dirs:
        print(f"  {d}")
        
    print("\n--- common PowerShell Paths check ---")
    ps_paths = [
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        r"C:\Program Files\PowerShell\7\pwsh.exe"
    ]
    for p in ps_paths:
        exists = os.path.exists(p)
        print(f"  {p}: {'EXISTS' if exists else 'NOT FOUND'}")

if __name__ == "__main__":
    check_env()
