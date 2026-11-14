import hashlib,json,os,re,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
O=R/'kernel-validation-fused'; O.mkdir(exist_ok=False)
old=Path(r'G:\qwenllamaMTP\perf-20260911\dist\fn-dory-recurrence-20261113\bin\ggml-cuda.dll')
new=R/'cuda-release_bounds/ggml-cuda.dll'
env=os.environ.copy()
for k in list(env):
    if re.match(r'^(GGML_|LLAMA_|QWEN4EXP_)',k): env.pop(k)
env['GGML_CUDA_OPT_TRACE']='1'
assert env.get('PATH')==os.environ.get('PATH')
summaries={}
def run(label,exe,args,expected=0):
    out=O/(label+'.jsonl'); err=O/(label+'-diagnostics.txt')
    with out.open('w') as f,err.open('w') as e:
        p=subprocess.run([str(exe),*args],stdout=f,stderr=e,env=env,creationflags=subprocess.CREATE_NO_WINDOW,timeout=120)
    rows=[json.loads(s) for s in out.read_text().splitlines() if s.strip()]
    summary=rows[-1] if rows and rows[-1].get('summary') else {'partial_rows':len(rows),'last_key':rows[-1].get('key') if rows else None}
    summary['exit_code']=p.returncode
    summary['diagnostic_counters']=[s for s in err.read_text().splitlines() if 'cuda_opt_trace:' in s]
    summaries[label]=summary
    (O/'summary.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps({'label':label,**summary}),flush=True)
    assert p.returncode==expected,(label,p.returncode,summary)
    return out,rows
gdn=R/'gdn_runtime/gdn-cache-regression.exe'
shutil.copy2(old,gdn.parent/'ggml-cuda.dll')
run('gdn_original',gdn,['--backend','CUDA0','--fusion','on','--graphs','off'],1)
shutil.copy2(new,gdn.parent/'ggml-cuda.dll')
for device in ('CUDA0','CUDA1'):
    for fusion,graphs in [('off','off'),('on','off'),('on','on')]:
        run(f'gdn_fixed_{device}_{fusion}_{graphs}',gdn,['--backend',device,'--fusion',fusion,'--graphs',graphs])
topk=R/'topk_runtime/topk-moe-regression.exe'
shutil.copy2(old,topk.parent/'ggml-cuda.dll')
ref,_=run('topk_unfused',topk,['--gate','all','--fusion','off'])
run('topk_original_terminal_crash',topk,['--gate','all','--fusion','on','--reference',str(ref)],3221225477)
run('topk_original_with_tail',topk,['--gate','all','--fusion','on','--trailing-scale','on','--reference',str(ref)])
shutil.copy2(new,topk.parent/'ggml-cuda.dll')
for device in ('CUDA0','CUDA1'):
    _,rows=run('topk_fixed_'+device,topk,['--backend',device,'--gate','all','--fusion','on','--reference',str(ref)])
    assert any('topk_fused=' in s for s in summaries['topk_fixed_'+device]['diagnostic_counters'])
    # The model's ordinary softmax routing must match its separate CUDA path.
    assert all(row['reference']['bitwise_equal'] for row in rows if row.get('gate')=='softmax')
ref,_=run('topk_ties_unfused',topk,['--gate','softmax','--profile','near_tie','--fusion','off'])
_,rows=run('topk_ties_fixed',topk,['--gate','softmax','--profile','near_tie','--fusion','on','--reference',str(ref)])
assert all(row['reference']['bitwise_equal'] for row in rows if 'reference' in row)
(O/'summary.json').write_text(json.dumps(summaries,indent=2))
print('Kernel validation completed',flush=True)
