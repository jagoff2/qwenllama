import argparse, hashlib, json, os, re, subprocess, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
MODEL=r'C:\Users\user\Downloads\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
CONFIGS={'stock':[-1,-1,1,1,0,0,0], 'neutral':[28,31,4,1,1,0,2,0,0,0,1],
         'wide_neutral':[20,35,2,1,1,0,2,0,0,0,1], 'loop':[28,31,4,1,0,0,2,0,0,0]}
def run(label,fusion=False,reuse=False,capture=False,extra=None):
    out=ROOT/label
    out.mkdir(exist_ok=False)
    env=os.environ.copy()
    original_path=env.get('PATH')
    for k in list(env):
        if re.match(r'^(LLAMA_|GGML_|QWEN4EXP_|OMP_|DECODE_)',k): env.pop(k)
    env.update({'QWEN4EXP_LOOP_CONFIG':str(out/'config.txt'),'QWEN4EXP_QSA_GATHER':'1',
                'QWEN4EXP_QSA_QUERY_CHUNK':'256','LLAMA_ATTN_ROT_DISABLE':'1','GGML_CPU_MOE_N1':'1'})
    if not fusion: env['GGML_CUDA_DISABLE_FUSION']='1'
    if not reuse: env['LLAMA_GRAPH_REUSE_DISABLE']='1'
    if not capture: env['GGML_CUDA_DISABLE_GRAPHS']='1'
    env.update(extra or {})
    assert env.get('PATH')==original_path
    cmd=[str(ROOT/'runtime'/'probe.exe'),'-m',MODEL,'-c','1024','-b','512','-ub','512',
         '-ngl','99','--cpu-moe','--fit','off','-t','8','-tb','24','-fa','on','--no-op-offload','-lv','4']
    metadata={'fusion':fusion,'reuse':reuse,'capture':capture,'extra':extra,'command':cmd,
              'dll_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'runtime').glob('*.dll')}}
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    results={}
    with (out/'new-diagnostics.txt').open('w') as err:
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            ready=p.stdout.readline()
            print(json.dumps({'label':label,'startup':ready.strip()}),flush=True)
            assert json.loads(ready)['ready']
            for name,config in CONFIGS.items():
                request={'id':name,'config':config,'logits_file':str(out/(name+'.f32')),
                         'prompt':'<|im_start|>user\nCompute 7384619275 times 9162748351. Give the exact product.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'}
                p.stdin.write(json.dumps(request)+'\n'); p.stdin.flush()
                r=json.loads(p.stdout.readline())
                assert 'error' not in r,r
                results[name]=r
                print(json.dumps({'label':label,'mode':name,'seconds':round(r['seconds'],2),'rows':r['rows'],'first_top':r['top_ids'][0]}),flush=True)
            p.stdin.close(); p.wait(timeout=30)
            assert p.returncode==0
        finally:
            if p.poll() is None: p.kill(); p.wait()
    (out/'results.json').write_text(json.dumps(results,indent=2))
    comparisons={}
    a=np.fromfile(out/'stock.f32',dtype=np.float32).reshape(-1,results['stock']['vocab'])
    for mode in ('neutral','wide_neutral'):
        b=np.fromfile(out/(mode+'.f32'),dtype=np.float32).reshape(a.shape)
        delta=np.abs(a-b)
        comparisons[mode]={'exact':bool(np.array_equal(a,b)), 'max_abs_per_step':delta.max(axis=1).tolist(),
                           'rms_per_step':np.sqrt(np.mean(delta.astype(np.float64)**2,axis=1)).tolist(),
                           'top_mismatches':int(np.sum(a.argmax(axis=1)!=b.argmax(axis=1)))}
    (out/'comparisons.json').write_text(json.dumps(comparisons,indent=2))
    print(json.dumps({'label':label,'comparisons':comparisons}),flush=True)
    return results
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('label'); ap.add_argument('--fusion',action='store_true'); ap.add_argument('--reuse',action='store_true'); ap.add_argument('--capture',action='store_true'); ap.add_argument('--env',action='append',default=[])
    a=ap.parse_args(); run(a.label,a.fusion,a.reuse,a.capture,dict(x.split('=',1) for x in a.env))
