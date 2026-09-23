#!/usr/bin/env python3
"""Matched stream-mode check of the output-clear change; run only with an idle GPU."""
import pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from run_round import ROOT,run,ensure_environment
ensure_environment(HERE)
cases=[
 ('small8',1048576,8,'u32','cub:2:192,shared:4:192,shared_partial:2:96'),
 ('smallbyte',4096,256,'u8','cub:2:192,shared:7:96,shared_partial:7:96,nvidia_sample256:2:192'),
 ('cachedbyte',1048576,256,'u8','cub:2:192,shared:7:96,shared_partial:7:96,nvidia_sample256:2:192'),
 ('large4096',16777216,4096,'u32','cub:2:192,shared:3:96,shared_partial:3:96')]
for repeat in (1,2,3):
 builds=[('runtime',ROOT/'build/profiled-loads/histogram_bench'),('kernel',ROOT/'build/histogram_bench')]
 if repeat==2:builds.reverse()
 for kind,exe in builds:
  for name,n,bins,input_type,variants in cases:
   run(HERE/'stream-ablation'/f'{name}-{kind}-r{repeat}',[exe,'--n',str(n),'--bins',str(bins),'--input',input_type,'--counter','u32','--distribution','uniform','--order','shuffled','--cache','warm','--launch','stream','--variants',variants,'--samples','21','--batch','32','--seed','424242'])
