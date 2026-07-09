import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.router.tools import bootstrap_tools
from src.router.agent import AgenticRouter

print("Bootstrapping tools...")
t0 = time.time()
bootstrap_tools()
print(f"Bootstrapping completed in {time.time() - t0:.2f} seconds.")

print("Initializing AgenticRouter...")
t1 = time.time()
router = AgenticRouter()
print(f"AgenticRouter initialized in {time.time() - t1:.2f} seconds.")

print("Running query: 'hyy'...")
t2 = time.time()
try:
    res = router.query("hyy")
    print(f"Result: {res}")
except Exception as e:
    print(f"Query failed: {e}")
print(f"Query completed in {time.time() - t2:.2f} seconds.")
