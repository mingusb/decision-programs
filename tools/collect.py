#!/usr/bin/env python3
"""Preserve one GPU diagnostic attempt and verify actual tool activity.

No application computation or performance statistics run here. Profiler evidence
is diagnostic only. Use a new output directory and absolute target/file paths.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

SANITIZERS = ("memcheck", "initcheck", "racecheck", "synccheck")
INJECTORS = ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "CUDA_INJECTION32_PATH", "NVTX_INJECTION64_PATH")
TOOLS = ("nsys", "ncu", *SANITIZERS, "cupti-trace", "cupti-range", "cupti-pc",
         "nvbit-count", "nvbit-memory", "nvbit-graph", "cuda-gdb", "cuobjdump", "nvdisasm", "compiler")


def identity(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def executable(name):
    found = shutil.which(name)
    if not found:
        raise RuntimeError("missing executable: " + name)
    return str(Path(found).resolve())


def sdk(kind):
    explicit = os.environ.get(kind.upper() + "_ROOT")
    pattern = "nvbit-*/nvbit_release_x86_64" if kind == "nvbit" else "cupti-*/cuda_cupti-*-archive"
    choices = [Path(explicit)] if explicit else list((Path.home()/".local/opt/gpu-profiling").glob(pattern))
    if not choices:
        raise RuntimeError("missing SDK; set " + kind.upper() + "_ROOT")
    return max(choices, key=lambda p: tuple(int(v) for v in re.findall(r"\d+", p.parent.name))).resolve(strict=True)


def execute(command, environment, output, label, timeout):
    started = time.time()
    cleanup = None
    with (output/(label+".stdout.log")).open("wb") as stdout, (output/(label+".stderr.log")).open("wb") as stderr:
        process = subprocess.Popen(command, cwd=output, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            session = next((v.split("=",1)[1] for v in command if v.startswith("--session-new=")),None)
            if session:
                shutdown = [command[0],"shutdown","--session="+session,"--kill=sigkill"]
                try:
                    result = subprocess.run(shutdown,env=environment,cwd=output,stdout=stdout,stderr=stderr,timeout=10)
                    cleanup = {"argv":shutdown,"exit":result.returncode}
                except subprocess.TimeoutExpired:
                    cleanup = {"argv":shutdown,"timeout":True}
    return {"argv": command, "exit": process.returncode, "timeout": timed_out,
            "cleanup":cleanup,"wall_seconds": time.time()-started}


def plan(args, output):
    inherited = [key for key in INJECTORS if os.environ.get(key)]
    if inherited:
        raise RuntimeError("competing inherited injection: " + ", ".join(inherited))
    target = Path(args.command[0]).resolve(strict=True)
    if not target.is_file():
        raise RuntimeError("target must be a file")
    environment = dict(os.environ)
    command = [str(target), *args.command[1:]]
    assets = {"target": identity(target), "collector": identity(__file__)}
    tool, decoder = args.tool, None
    if tool == "nsys":
        command = [executable("nsys"), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                   "--session-new=gh-"+uuid.uuid4().hex,
                   "--cuda-graph-trace="+("graph" if args.workload == "graph" else "node"),
                   "--output="+str(output/"profile"), *command]
    elif tool == "ncu":
        metrics = "sm__ctas_launched.sum" + (",launch__graph_exec_cuda_id" if args.workload == "graph" else "")
        command = [executable("ncu"), "--metrics", metrics, "--launch-count", str(args.limit),
                   "--target-processes", "all", "--export", str(output/"profile"),
                   "--graph-profiling", "graph" if args.workload == "graph" else "node", *command]
    elif tool in SANITIZERS:
        command = [executable("compute-sanitizer"), "--tool", tool, "--error-exitcode", "97",
                   "--report-api-errors", args.api_errors, *command]
    elif tool.startswith("nvbit-"):
        if args.workload == "graph" and tool != "nvbit-graph":
            raise RuntimeError("ordinary NVBit count/memory samples do not support graph capture; use graph counter")
        name = {"nvbit-count":"instr_count", "nvbit-memory":"mem_trace", "nvbit-graph":"instr_count_cuda_graph"}[tool]
        library = sdk("nvbit")/"tools"/name/(name+".so")
        assets["injector"] = identity(library)
        environment.update(LD_PRELOAD=str(library), ACTIVE_FROM_START="1", TOOL_VERBOSE="0",
                           INSTR_BEGIN="0", INSTR_END="4294967295", MANGLED_NAMES="0",
                           START_GRID_NUM="0", END_GRID_NUM=str(min(args.limit,100)), COUNT_WARP_LEVEL="1")
        assets["nvdisasm"] = identity(executable("nvdisasm"))
    elif tool.startswith("cupti-"):
        if tool == "cupti-range" and args.workload == "graph":
            raise RuntimeError("installed range-injection sample has no graph launch callbacks")
        root = sdk("cupti")
        names = {"cupti-trace":"cupti_trace_injection/libcupti_trace_injection.so",
                 "cupti-range":"profiling_injection/libinjection.so",
                 "cupti-pc":"pc_sampling_continuous/libpc_sampling_continuous.so"}
        library = root/"samples"/names[tool]
        assets["injector"] = identity(library)
        for name in ("libcupti.so","libpcsamplingutil.so","libnvperf_host.so","libnvperf_target.so"):
            if (root/"lib"/name).exists():
                assets[name] = identity(root/"lib"/name)
        environment["CUDA_INJECTION64_PATH"] = str(library)
        environment["LD_LIBRARY_PATH"] = str(root/"lib") + ":" + environment.get("LD_LIBRARY_PATH","")
        if tool == "cupti-trace":
            environment["NVTX_INJECTION64_PATH"] = str(library)
        elif tool == "cupti-range":
            environment["INJECTION_METRICS"] = "sm__ctas_launched.sum"
        else:
            environment["INJECTION_PARAM"] = "--collection-mode 1 --sampling-period 12 --file-name pcsampling.dat --verbose"
            decoder = root/"samples/pc_sampling_utility/pc_sampling_utility"
            assets["decoder"] = identity(decoder)
    elif tool == "cuda-gdb":
        command = [executable("cuda-gdb"), "--batch", "-ex", "set pagination off",
                   "-ex", "set cuda break_on_launch application", "-ex", "run", "-ex", "info cuda kernels",
                   "-ex", "set cuda break_on_launch none", "-ex", "continue", "--args", *command]
    elif tool == "cuobjdump":
        command = [executable(tool), "--dump-resource-usage", "--dump-sass", str(target)]
    elif tool == "nvdisasm":
        command = [executable(tool), str(target)]
    if tool != "compiler" and not tool.startswith(("nvbit-","cupti-")):
        assets["tool"] = identity(command[0])
    return command, environment, assets, decoder


def ncu_activity(path, graph, kernel):
    header, rows = None, {}
    for row in csv.reader(path.open()):
        if "ID" in row and (("Metric Name" in row and "Metric Value" in row) or "sm__ctas_launched.sum" in row):
            header = row
        elif header and len(row) == len(header):
            value = dict(zip(header,row))
            name = value.get("Kernel Name",value.get("Name",""))
            if not value["ID"].isdigit() or (kernel and not re.search(kernel,name)):
                continue
            key = (value.get("Process ID",""),value["ID"],name)
            metrics = {value["Metric Name"]:value["Metric Value"]} if "Metric Name" in value else {
                name:value[name] for name in ("sm__ctas_launched.sum", "launch__graph_exec_cuda_id") if name in value}
            for name, raw in metrics.items():
                try:
                    number = float(raw.replace(",",""))
                except ValueError:
                    continue
                if math.isfinite(number):
                    rows.setdefault(key,{})[name] = number
    selected = [key for key, metrics in rows.items() if metrics.get("sm__ctas_launched.sum",0)>0
                and (not graph or metrics.get("launch__graph_exec_cuda_id",0)>0)]
    return {"records": len(selected), "names": [key[2] for key in selected]}


def nsys_activity(path, graph, kernel):
    table = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" if graph else "CUPTI_ACTIVITY_KIND_KERNEL"
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True) as database:
        if not database.execute("SELECT 1 FROM sqlite_master WHERE name=? AND type='table'",(table,)).fetchone():
            return {"records":0}
        if graph:
            rows = database.execute("SELECT start,end FROM "+table+" WHERE end>start").fetchall()
        else:
            rows = database.execute("SELECT k.start,k.end,s.value FROM "+table+
                " k JOIN StringIds s ON k.demangledName=s.id WHERE k.end>k.start").fetchall()
            if kernel:
                rows = [row for row in rows if re.search(kernel,row[2])]
    return {"records":len(rows), "names":[] if graph else sorted({row[2] for row in rows})}


def evidence(args, output, command, environment, decoder, attempts):
    tool = args.tool
    clean = {k:v for k,v in environment.items() if k not in (*INJECTORS,"INJECTION_PARAM","INJECTION_METRICS")}
    if tool in ("nsys","ncu"):
        candidates = [output/"profile.nsys-rep"] if tool == "nsys" else [
            output/"profile.ncu-rep", output/"profile.ncu-repz"]
        reports = [p for p in candidates if p.is_file() and p.stat().st_size]
        if len(reports) != 1:
            return {"records":0, "reason":"missing or ambiguous profiler report"}
        report = reports[0]
        if not report.is_file() or not report.stat().st_size:
            return {"records":0, "reason":"missing profiler report"}
        export = output/("export.sqlite" if tool == "nsys" else "export.stdout.log")
        argv = ([command[0],"export","--type","sqlite","--output",str(export),str(report)] if tool == "nsys"
                else [command[0],"--import",str(report),"--page","raw","--csv","--rename-kernels","0","--print-kernel-base","demangled"])
        attempts.append(execute(argv,clean,output,"export",min(args.timeout,60)))
        if attempts[-1]["exit"] or attempts[-1]["timeout"]:
            return {"records":0,"reason":"report export failed"}
        return (nsys_activity if tool == "nsys" else ncu_activity)(export,args.workload == "graph",args.kernel)
    if decoder:
        files = [p for p in sorted(output.glob("*pcsampling*.dat")) if p.stat().st_size]
        for index,path in enumerate(files):
            attempts.append(execute([str(decoder),"--file-name",str(path),"--disable-source-correlation","--verbose"],
                clean,output,"decoded-"+str(index),min(args.timeout,60)))
        if not files:
            return {"records":0,"reason":"missing PC sample data"}
    records, dropped = 0, 0
    summary = gpu = stopped = exited = named = False
    clean = True
    # Memory traces can be large; retain raw logs and inspect them incrementally.
    for path in sorted(output.glob("*.log")):
        with path.open(errors="replace") as source:
            for line in source:
                if tool in SANITIZERS:
                    gpu |= bool(re.search(args.activity,line))
                    match = re.search(r"(?:ERROR SUMMARY:\s*(\d+) errors|RACECHECK SUMMARY:\s*(\d+) hazards)",line)
                    if match:
                        summary = True
                        clean &= int(match[1] or match[2]) == 0
                elif tool in ("nvbit-count","nvbit-graph"):
                    records += bool(re.search(r"kernel instructions\s+[1-9]\d*",line))
                    clean &= "ran out of kernel_counters" not in line
                elif tool == "nvbit-memory":
                    records += bool(re.search(r"MEMTRACE:.*grid_launch_id\s+\d+.*warp\s+\d+.* - 0x[0-9a-fA-F]+",line))
                elif tool == "cupti-trace":
                    match = re.search(r"(?:CONCURRENT_KERNEL|KERNEL):\s*(\d+)\s+records",line)
                    if match:
                        records += int(match[1])
                elif tool == "cupti-range":
                    match = re.search(r"sm__ctas_launched\.sum\s+([+\d.eE-]+)\s*$",line)
                    if match:
                        number = float(match[1])
                        records += math.isfinite(number) and number > 0
                elif tool == "cupti-pc":
                    records += bool(re.search(r"Total Samples:\s*[1-9]\d*",line))
                    named |= "functionName:" in line
                    match = re.search(r"Total Dropped Samples:\s*(\d+)",line)
                    if match:
                        dropped += int(match[1])
                elif tool == "cuda-gdb":
                    stopped |= bool(re.search(r"\[Switching focus to CUDA kernel|CUDA thread .* hit|CUDA kernel entry",line))
                    exited |= "exited normally" in line
                elif tool == "compiler":
                    records += bool(re.search(r"ptxas info\s*: Used \d+ registers",line))
                else:
                    records += bool(re.search(r"/\*[0-9a-fA-F]+\*/\s+\w",line))
    if tool in SANITIZERS:
        return {"records":int(summary and clean and gpu),"clean_summary":summary and clean,
                "gpu_receipt":gpu,"api_reporting":args.api_errors}
    if tool == "cuda-gdb":
        return {"records":int(stopped and exited),"device_stop":stopped}
    if tool == "cupti-pc":
        return {"records":records if named and not dropped else 0,"dropped_samples":dropped}
    return {"records":records if clean else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool",choices=TOOLS)
    parser.add_argument("output",type=Path)
    parser.add_argument("--timeout",type=float,default=120)
    parser.add_argument("--workload",choices=("cdp","leaf","graph"),default="cdp")
    parser.add_argument("--limit",type=int,default=2)
    parser.add_argument("--kernel",help="required kernel-name regex for Nsight exports")
    parser.add_argument("--activity",default=r"GH_GPU_ACTIVITY|PASS\s|GPU .*checks completed",
                        help="GPU-authored application completion receipt regex, required for sanitizer evidence")
    parser.add_argument("--api-errors",choices=("explicit","extended","all"),default="explicit")
    args, command = parser.parse_known_args()
    args.command = command[1:] if command[:1] == ["--"] else command
    if not args.command or not Path(args.command[0]).is_absolute():
        parser.error("append -- /absolute/target [arguments]")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.limit <= 0:
        parser.error("timeout and limit must be positive")
    if args.kernel and (args.tool not in ("nsys","ncu") or args.workload == "graph"):
        parser.error("kernel filters require non-aggregate Nsight collection")
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    receipt = {"schema":"gh.diagnostic.v1", "tool":args.tool, "workload":args.workload,
               "ranking_valid":False, "attribution":"host parent call tree" if args.workload == "cdp" else args.workload,
               "requested_command":args.command, "api_errors":args.api_errors, "attempts":[], "status":"preparing"}
    path = output/"receipt.json"
    def save():
        path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    save()
    try:
        argv, environment, assets, decoder = plan(args,output)
        receipt.update(assets=assets,environment={k:v for k,v in environment.items() if os.environ.get(k)!=v},status="running")
        save()
        receipt["attempts"].append(execute(argv,environment,output,"run",args.timeout))
        receipt["activity"] = evidence(args,output,argv,environment,decoder,receipt["attempts"])
        passed = receipt["activity"]["records"]>0 and all(not a["exit"] and not a["timeout"] for a in receipt["attempts"])
        receipt["status"] = "passed" if passed else "failed"
    except (Exception,KeyboardInterrupt) as error:
        receipt.update(status="failed",error=type(error).__name__+": "+str(error))
    receipt["artifacts"] = [identity(p) for p in sorted(output.iterdir()) if p.is_file() and p!=path]
    save()
    print(receipt["status"]+": "+str(path))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
