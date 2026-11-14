"""Short exact-arithmetic probes with balanced labels and independent oracles."""
import argparse
import collections
import json
from pathlib import Path
from tasks import rng_for, balanced_value, make_choices, SPLIT_SEEDS, LABELS

def school_product(a,b):
    # Decimal convolution, independent of integer multiplication of operands.
    aa=list(map(int,str(a)[::-1]));bb=list(map(int,str(b)[::-1]))
    digits=[0]*(len(aa)+len(bb)+1)
    for i,x in enumerate(aa):
        for j,y in enumerate(bb):digits[i+j]+=sum(x for _ in range(y))
    for i in range(len(digits)-1):digits[i+1]+=digits[i]//10;digits[i]%=10
    return int(''.join(map(str,digits[::-1])))

def item(split,index,family='multiply',size=3):
    seed=SPLIT_SEEDS[split]
    rng=rng_for(seed,family,size,index)
    label=LABELS[balanced_value(seed,family,'label',index,'math',size)]
    if family in ['multiply','multiply_carry','multiply_balanced']:
        a=rng.randrange(10**(size-1),10**size);b=rng.randrange(10**(size-1),10**size)
        answer=a*b
        assert answer==school_product(a,b)
        # Carry errors share final digits, preventing trivial units-digit elimination.
        error_scale=10**size if family=='multiply_carry' else 100
        candidates={answer+delta*error_scale for delta in [-100,-20,-10,-3,-2,-1,1,2,3,10,20,100] if answer+delta*error_scale>0}
        text=f'Calculate the exact integer product {a} × {b}.'
        parameters={'a':a,'b':b,'digits':size,'distractor_scale':error_scale}
        if family=='multiply_balanced':
            if size<4:raise ValueError('balanced carry probes require at least four digits')
            # Every 16 consecutive cases contain each (answer label, numeric rank)
            # pair exactly once. Within each group of four both are balanced.
            row_order=rng_for(seed,family,'rank_rows',index//16,size).sample(range(4),4)
            rank_order=rng_for(seed,family,'rank_columns',index//16,size).sample(range(4),4)
            rank=rank_order[(LABELS.index(label)+row_order[(index//4)%4])%4]
            error_scale=10**size*rng.choice([1,2,3,5,7,11,17,29])
            candidates={answer+(j-rank)*error_scale for j in range(4)}
            assert min(candidates)>0
            parameters.update(distractor_scale=error_scale,correct_numeric_rank=rank)
    elif family=='modular_power':
        modulus=rng.choice([17,19,23,29,31,37,41,43,47])
        base=rng.randrange(2,modulus);exponent=rng.randrange(10**(size-1),10**size)
        answer=pow(base,exponent,modulus)
        check=1
        for _ in range(exponent):check=(check*base)%modulus
        assert check==answer
        candidates=set(range(modulus))
        text=f'What is the remainder when {base}^{exponent} is divided by {modulus}? Here ^ denotes exponentiation.'
        parameters={'base':base,'exponent':exponent,'modulus':modulus}
    elif family=='parentheses':
        # Four signed products with nesting; all intermediate numbers stay small.
        a,b,c,d,e,f=[rng.randrange(11,100) for _ in range(6)]
        answer=(a*b-c*d)*(e-f)
        check=(school_product(a,b)-school_product(c,d))*(e-f)
        assert answer==check
        candidates={answer+delta for delta in [-1000,-100,-10,-1,1,10,100,1000]}
        text=f'Evaluate exactly: ({a} × {b} − {c} × {d}) × ({e} − {f}).'
        parameters={'operands':[a,b,c,d,e,f]}
    else:raise ValueError(family)
    candidates.add(answer)
    choices=make_choices(answer,sorted(candidates),label,rng)
    prompt=text+'\n'+'\n'.join(k+': '+str(v) for k,v in choices.items())
    return {'id':f'{split}-{family}-n{size}-{index:04}','split':split,'family':family,
            'difficulty':str(size),'index':index,'parameters':parameters,'prompt':prompt,
            'answer':label,'answer_value':answer}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=8)
    p.add_argument('--split',choices=['dev','heldout'],default='dev')
    p.add_argument('--output',default='math_dev.jsonl');p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        for family,size in [('multiply',3),('multiply',4),('modular_power',2),('parentheses',3)]:
            generated=[item('dev',i,family,size) for i in range(100)]
            assert collections.Counter(x['answer'] for x in generated)==dict.fromkeys('ABCD',25)
        print('400 exact arithmetic and label balance checks passed')
        for size in [4,5,6,8]:
            generated=[item('dev',i,'multiply_balanced',size) for i in range(64)]
            pairs=collections.Counter((x['answer'],x['parameters']['correct_numeric_rank']) for x in generated)
            assert len(pairs)==16 and set(pairs.values())=={4}
        print('256 balanced-rank multiplication checks passed')
    else:
        generated=[item(a.split,i,family,size) for family,size in [('multiply',3),('multiply',4),('modular_power',2),('parentheses',3)] for i in range(a.count)]
        with Path(a.output).open('x',encoding='utf-8') as out:
            for row in generated:out.write(json.dumps(row)+'\n')
        print('generated',len(generated))
