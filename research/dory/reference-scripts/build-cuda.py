import argparse, hashlib, json, os, re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SRC=Path(r'G:\qwenllamaMTP\experiments\dory_server_20261113\source')
OLD=Path(r'G:\qwenllamaMTP\perf-20260911\llama.cpp\build')
CORE=Path(r'G:\qwenllamaMTP\experiments\dory_server_20261113\build')
VC=Path(r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.37.32822')
CUDA=Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2')
SDK=Path(r'C:\Program Files (x86)\Windows Kits\10')
NINJA=r'C:\Users\user\AppData\Local\Programs\Python\Python3119\Scripts\ninja.exe'
ap=argparse.ArgumentParser(); ap.add_argument('label'); ap.add_argument('--unit',action='append',default=[]); a=ap.parse_args()
out=ROOT/('cuda-'+a.label); out.mkdir(exist_ok=False)
env=os.environ.copy(); original_path=env.get('PATH')
env['INCLUDE']=';'.join(str(p) for p in [VC/'include',SDK/'Include/10.0.26100.0/ucrt',SDK/'Include/10.0.26100.0/shared',SDK/'Include/10.0.26100.0/um'])
env['LIB']=';'.join(str(p) for p in [VC/'lib/x64',SDK/'Lib/10.0.26100.0/ucrt/x64',SDK/'Lib/10.0.26100.0/um/x64',CUDA/'lib/x64'])
assert env.get('PATH')==original_path
oldcmd=subprocess.check_output([NINJA,'-C',str(OLD),'-t','commands','ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.obj'],text=True)
defines=[s for s in re.findall(r'-D\S+',oldcmd) if not s.startswith('-DGGML_CUDA_FA_QUANTS')]
defines.append('-DGGML_CUDA_FA_QUANTS="q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16"')
source=SRC/'ggml/src/ggml-cuda/ggml-cuda.cu'
obj=out/'ggml-cuda.cu.obj'
cmd=[str(CUDA/'bin/nvcc.exe'),'--use-local-env','-ccbin',str(VC/'bin/Hostx64/x64'),'-forward-unknown-to-host-compiler',*defines,
     '-I'+str(SRC/'ggml/src'),'-I'+str(SRC/'ggml/include'),'-isystem',str(CUDA/'include'),'-isystem',str(CUDA/'include/cccl'),
     '-Xcompiler=/EHsc','-Xcompiler=-O2,-Ob2','-std=c++17','--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]',
     '-Xcompiler=-MD','-use_fast_math','-extended-lambda','-compress-mode=size','-Xcompiler=/Zc:preprocessor','-diag-suppress=221',
     '-x','cu','-c',str(source),'-o',str(obj)]
(out/'compile-command.json').write_text(json.dumps(cmd,indent=2))
(out/'source.sha256').write_text(hashlib.sha256(source.read_bytes()).hexdigest())
subprocess.run(cmd,env=env,cwd=out,check=True)
replacements={'ggml-cuda.cu.obj':obj}
for unit in a.unit:
    unit_src=SRC/'ggml/src/ggml-cuda'/(unit+'.cu')
    unit_obj=out/(unit+'.cu.obj')
    unit_cmd=[str(unit_src) if s==str(source) else str(unit_obj) if s==str(obj) else s for s in cmd]
    (out/(unit+'-command.json')).write_text(json.dumps(unit_cmd,indent=2))
    subprocess.run(unit_cmd,env=env,cwd=out,check=True)
    replacements[unit_obj.name]=unit_obj
edge=next(s for s in (OLD/'build.ninja').read_text().splitlines() if s.startswith('build bin\\ggml-cuda.dll '))
object_paths=edge.split(': CUDA_SHARED_LIBRARY_LINKER__ggml-cuda_Release ',1)[1].split(' | ',1)[0].split()
objects=[replacements.get(Path(p).name,OLD/p) for p in object_paths]
assert all(p.is_file() for p in objects)
libs=[CORE/'ggml/src/ggml-base.lib']+[CUDA/'lib/x64'/s for s in ['cudart.lib','cublas.lib','cuda.lib','cublasLt.lib','cudadevrt.lib']]
system=['kernel32.lib','user32.lib','gdi32.lib','winspool.lib','shell32.lib','ole32.lib','oleaut32.lib','uuid.lib','comdlg32.lib','advapi32.lib']
linkargs=['/NOLOGO','/DLL','/MACHINE:X64','/INCREMENTAL:NO','/MANIFEST:EMBED','/VERSION:0.23',
          '/OUT:'+str(out/'ggml-cuda.dll'),'/IMPLIB:'+str(out/'ggml-cuda.lib'),'/PDB:'+str(out/'ggml-cuda.pdb'),
          *map(str,objects),*map(str,libs),*system]
rsp=out/'link.rsp'; rsp.write_text('\n'.join('"'+s+'"' if ' ' in s else s for s in linkargs))
subprocess.run([str(VC/'bin/Hostx64/x64/link.exe'),'@'+str(rsp)],env=env,cwd=out,check=True)
print(json.dumps({'dll':str(out/'ggml-cuda.dll'),'sha256':hashlib.sha256((out/'ggml-cuda.dll').read_bytes()).hexdigest(),'reused_objects':len(objects)-len(replacements)}),flush=True)
