import ctypes
import json
from pathlib import Path
import shutil
import subprocess

R = Path(__file__).resolve().parent
old = R/'allocation-original'
old.mkdir(exist_ok=False)
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
shutil.copy2(R/'planner-test/allocation-failure-regression.exe',old)
shutil.copy2(Path(r'G:\qwenllamaMTP\perf-20260911\dist\fn-dory-recurrence-graphs-20261113\bin\ggml-base.dll'),old)
results = {}
for label, folder in [('original',old),('fixed',R/'planner-test')]:
    run = subprocess.run([str(folder/'allocation-failure-regression.exe')],capture_output=True,text=True,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    results[label] = {'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
assert results['original']['exit_code'] == 0xC0000005
assert results['fixed']['exit_code'] == 0
(R/'allocation-failure-result.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))
