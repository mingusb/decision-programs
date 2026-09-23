"""Capture serial diagnostics and preserve failures; refuse overwrites."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def run(name,command):
 paths=[OUT/(name+x) for x in ('-command.json','.stdout','.stderr')]
 if any(p.exists() for p in paths):raise RuntimeError('refusing overwrite '+name)
 binaries={}
 for arg in command:
  p=Path(arg);p=p if p.is_absolute() else ROOT/p
  if p.is_file() and os.access(p,os.X_OK):binaries[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 start=time.monotonic()
 with paths[1].open('xb') as out,paths[2].open('xb') as err:r=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err)
 rec=dict(command=command,returncode=r.returncode,wall_seconds=time.monotonic()-start,command_executables_sha256=binaries,command_executables_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in binaries.items()))
 paths[0].write_text(json.dumps(rec,indent=2)+'\n');print(name,rec['returncode'],rec['wall_seconds'],flush=True);return r.returncode
if __name__=='__main__':raise SystemExit(run(sys.argv[1],sys.argv[2:]))
