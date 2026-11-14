"""Exact free-response integer products; no options or label cues."""
import argparse
import json
from pathlib import Path
from tasks import rng_for,SPLIT_SEEDS
from math_tasks import school_product

def item(split,index,size=8):
 rng=rng_for(SPLIT_SEEDS[split],'integer_product',size,index)
 values=[]
 while len(values)<2:
  value=rng.randrange(10**(size-1),10**size)
  if value%10 in [1,3,7,9]:values.append(value)
 a,b=values;answer=a*b
 assert answer==school_product(a,b)
 return {'id':f'{split}-integer_product-n{size}-{index:04}','split':split,'family':'integer_product',
         'difficulty':str(size),'index':index,'parameters':{'a':a,'b':b,'digits':size},
         'prompt':f'Compute {a} × {b}. Return only the exact integer product, with no spaces, commas, or explanation.',
         'answer':str(answer),'answer_value':str(answer)}

def prompt_for(task):
 return '<|im_start|>user\n'+task['prompt']+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'+task.get('assistant_prefix','')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','heldout'],default='dev')
 p.add_argument('--count',type=int,default=8);p.add_argument('--sizes',nargs='+',type=int,default=[5,6,8,10])
 p.add_argument('--output',default='integer_dev.jsonl');a=p.parse_args()
 rows=[item(a.split,i,n) for n in a.sizes for i in range(a.count)]
 with Path(a.output).open('x') as out:
  for r in rows:out.write(json.dumps(r)+'\n')
 print('generated',len(rows))
