#!/usr/bin/env python3
"""Identify the GPU and the ROCm toolchain for a benchmark receipt.

Prints one JSON object. Exits non-zero when the card cannot be identified:
an anonymous receipt is worse than no receipt, since a number nobody can
attribute to a specific card cannot be compared against anything.
"""
import json, os, re, subprocess, sys


def run(*cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return p.stdout if p.returncode == 0 else ""


def agents():
    """Every GPU agent rocminfo reports, with its gfx target and CU count."""
    out, found, cur = run("rocminfo"), [], None
    for line in out.splitlines():
        if re.match(r"^Agent \d+", line):
            if cur and cur.get("gfx"):
                found.append(cur)
            cur = {}
            continue
        if cur is None:
            continue
        m = re.search(r"^\s*Name:\s+(gfx\d+\w*)", line)
        if m:
            cur["gfx"] = m.group(1)
        m = re.search(r"^\s*Marketing Name:\s+(.+?)\s*$", line)
        if m and "gfx" in cur:
            cur["name"] = m.group(1)
        m = re.search(r"^\s*Compute Unit:\s+(\d+)", line)
        if m and "gfx" in cur:
            cur["cu"] = int(m.group(1))
        m = re.search(r"^\s*Wavefront Size:\s+(\d+)", line)
        if m and "gfx" in cur:
            cur["wavefront"] = int(m.group(1))
        m = re.search(r"^\s*Max Clock Freq\. \(MHz\):\s+(\d+)", line)
        if m and "gfx" in cur:
            cur["max_clock_mhz"] = int(m.group(1))
    if cur and cur.get("gfx"):
        found.append(cur)
    return found


def smi_name():
    for line in run("rocm-smi", "--showproductname").splitlines():
        m = re.search(r"Card Series:\s*(.+?)\s*$", line)
        if m and m.group(1):
            return m.group(1)
    return ""


def rocm_version():
    try:
        with open("/opt/rocm/.info/version") as f:
            return f.read().strip()
    except OSError:
        pass
    return run("hipconfig", "--version").strip()


def hipblaslt_version():
    for d in ("/opt/rocm/lib", "/usr/lib"):
        import glob, os
        libs = sorted(glob.glob(os.path.join(d, "libhipblaslt.so.*")))
        if libs:
            return os.path.basename(libs[-1])
    return ""


def power_state():
    st = {}
    out = run("rocm-smi", "--showclocks", "--showpower", "--showtemp")
    for line in out.splitlines():
        if "GPU[0]" not in line:
            continue
        m = re.search(r"sclk clock level:.*?\((\d+)Mhz\)", line)
        if m:
            st["sclk_mhz"] = int(m.group(1))
        m = re.search(r"Power \(W\):\s*([\d.]+)", line)
        if m:
            st["power_w"] = float(m.group(1))
        m = re.search(r"Temperature.*?:\s*([\d.]+)", line)
        if m and "temp_c" not in st:
            st["temp_c"] = float(m.group(1))
    return st


gpus = agents()
if not gpus:
    print("gpu-info: rocminfo reported no GPU agent; is ROCm installed and the "
          "user in the render/video groups?", file=sys.stderr)
    sys.exit(1)

gpu = gpus[0]
if not gpu.get("name"):
    gpu["name"] = os.environ.get("MOJO_BARO_GPU_NAME", "") or smi_name()
if not gpu.get("name"):
    print("gpu-info: could not read a marketing name for %s from rocminfo or "
          "rocm-smi. Set MOJO_BARO_GPU_NAME to the card's name and re-run."
          % gpu["gfx"], file=sys.stderr)
    sys.exit(1)

info = {
    "gpu": gpu["name"],
    "gfx": gpu["gfx"],
    "compute_units": gpu.get("cu"),
    "wavefront": gpu.get("wavefront"),
    "max_clock_mhz": gpu.get("max_clock_mhz"),
    "gpu_count": len(gpus),
    "rocm": rocm_version(),
    "hipblaslt": hipblaslt_version(),
    "state": power_state(),
}
print(json.dumps(info))
