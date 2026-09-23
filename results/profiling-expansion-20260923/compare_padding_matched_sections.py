"""Offline exact section bytes with explicit, unique C++ symbol mapping."""
from pathlib import Path
import json,subprocess,hashlib,re
p=Path(__file__).resolve().parent

def index(name):
 report=json.loads((p/name/'report.json').read_text()); result={}
 for binary in report['binaries']:
  if binary['name']!='libghb.a':continue
  for cubin in binary['cubins']:
   assert hashlib.sha256(Path(cubin['path']).read_bytes()).hexdigest()==cubin['sha256']
   for section in cubin['sections']:
    parts=re.split(r'(?=_Z)',section['name'],maxsplit=1)
    signature=parts[0]
    if len(parts)==2:
     call=subprocess.run(['/usr/bin/c++filt',parts[1]],text=True,capture_output=True,check=True)
     demangled=call.stdout.strip();assert not demangled.startswith('_Z')
     signature+=demangled
    identity=(cubin['name'],signature)
    assert identity not in result,('ambiguous',identity)
    result[identity]=section
 return result

a=index('compiler-booster-baseline');b=index('compiler-padding-matched');rows=[]
for identity in sorted(a.keys()|b.keys()):
 old=a.get(identity);new=b.get(identity)
 same=old is not None and new is not None and (old['bytes'],old['sha256'])==(new['bytes'],new['sha256'])
 rows.append({'cubin':identity[0],'mapped_section':identity[1],'before':old,'after':new,'identical_bytes':same})
r={'scope':'code and constant bytes, uniquely mapped full demangled names; no host/metadata/timing assertion','all_identical':all(x['identical_bytes'] for x in rows),'sections':len(rows),'text_sections':sum(x['after']['kind']=='code' for x in rows if x['after']),'renamed':sum(x['before']['name']!=x['after']['name'] for x in rows if x['before'] and x['after']),'rows':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(p/'padding-matched-mapped-sections.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='rows'})
