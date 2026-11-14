"""Declared, interleaved exact-generation comparisons with fresh contexts.

No target text enters any model request. Supports static and position-scheduled
recurrence through the validated schedule harness. Intended for replication and,
only with a separately frozen design, final held-out confirmation.
"""
import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from direct import DirectModel, BIN, STOCK
from integer_tasks import prompt_for
from integer_eval import GRAMMAR


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(config):
    c=list(config)+[0]*(11-len(config))
    c[4]=1
    c[10]=1
    return c


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tasks',required=True)
    parser.add_argument('--settings',required=True)
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--binary',type=Path,default=BIN/'latent-eval-schedule.exe')
    parser.add_argument('--heldout-design',type=Path)
    args=parser.parse_args()
    tasks=[json.loads(s) for s in Path(args.tasks).read_text().splitlines()]
    settings=json.loads(Path(args.settings).read_text())
    assert tasks and settings and len({s['name'] for s in settings})==len(settings)
    assert len({t['id'] for t in tasks})==len(tasks)
    assert settings[0]['name']=='stock' and settings[0]['config']==STOCK and not settings[0].get('schedule')
    binary=args.binary
    manifest={'tasks_sha256':sha(args.tasks),'settings_sha256':sha(args.settings),
              'binary_sha256':sha(binary),'llama_sha256':sha(binary.parent/'llama.dll'),
              'runner_sha256':sha(__file__),'settings':settings,'count':len(tasks),
              'exact_generation':True,'teacher_forcing':False,'fresh_context':True,
              'cuda_fusion':False,'cuda_graphs':False,'graph_reuse':False,
              'arm_order':'independently shuffled per task with fixed seed paired-generation-v1',
              'warmup':'one unscored easy product per arm','split':sorted({t['split'] for t in tasks})}
    if args.heldout_design:
        design=json.loads(args.heldout_design.read_text())
        assert all(t['split']=='heldout' for t in tasks)
        for key in ('tasks_sha256','settings_sha256','binary_sha256','llama_sha256','runner_sha256'):
            assert design[key]==manifest[key], key
        assert len(tasks)>=96 and len(settings)==2
        manifest['heldout_design_sha256']=sha(args.heldout_design)
    else:
        assert all(t['split']=='dev' for t in tasks), 'held-out requires frozen design'
    with Path(args.prefix+'_manifest.json').open('x') as f:
        json.dump(manifest,f,indent=2)
    os.environ['GGML_CUDA_DISABLE_FUSION']='1'
    model=DirectModel(args.prefix,binary=binary)
    streams={}
    checks=Path(args.prefix+'_controls.jsonl').open('x')
    try:
        def request(task,setting,neutral=False,capture=False):
            config=identity(setting['config']) if neutral else setting['config']
            schedule=[dict(start=s['start'],end=s['end'],
                           config=identity(s['config']) if neutral else s['config'])
                      for s in setting.get('schedule',[])]
            return {'prompt':prompt_for(task),'config':config,'loop_schedule':schedule,
                    'grammar':task.get('grammar',GRAMMAR),
                    'max_new_tokens':task.get('max_new_tokens',32),'capture_state':capture}

        for t in (tasks[0],tasks[-1]):
            expected=model.request(request(t,settings[0],capture=True))
            for setting in settings[1:]:
                result=model.request(request(t,setting,neutral=True,capture=True))
                equal=all(result[k]==expected[k] for k in (
                    'logits_fingerprint_fnv1a64','generated_ids','state_bytes','state_fingerprint_fnv1a64'))
                checks.write(json.dumps({'id':t['id'],'variant':setting['name'],'passed':equal,
                                         'expected':expected,'result':result})+'\n')
                checks.flush()
                assert equal, ('full-sequence neutralized recurrence mismatch',setting['name'],t['id'])
        warmup={'prompt':'Compute 123 times 456. Return only the exact integer product.'}
        for setting in settings:
            model.request(request(warmup,setting))
            streams[setting['name']]=Path(args.prefix+'_'+setting['name']+'.jsonl').open('x')
        rng=random.Random('paired-generation-v1')
        counts={s['name']:0 for s in settings}
        for index,t in enumerate(tasks):
            order=list(settings)
            rng.shuffle(order)
            for setting in order:
                result=model.request(request(t,setting))
                answer=result['generated_text'].strip()
                valid=bool(re.fullmatch(t.get('output_regex',r'[1-9][0-9]*'),answer))
                correct=valid and answer==t['answer']
                counts[setting['name']]+=correct
                row={'variant':setting['name'],'task':t,'config':setting['config'],
                     'schedule':setting.get('schedule',[]),'result':result,
                     'answer':answer,'correct':correct,'valid':valid,
                     'grammar':t.get('grammar',GRAMMAR),'max_new_tokens':t.get('max_new_tokens',32),
                     'prompt_sha256':hashlib.sha256(prompt_for(t).encode()).hexdigest(),
                     'arm_order':[s['name'] for s in order]}
                f=streams[setting['name']]
                f.write(json.dumps(row)+'\n')
                f.flush()
            print(json.dumps({'n':index+1,'total':len(tasks),'correct':counts}),flush=True)
    finally:
        for f in streams.values(): f.close()
        checks.close()
        model.close()


if __name__=='__main__':
    main()
