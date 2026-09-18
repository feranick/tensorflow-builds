# TensorFlow GPU Rebuild Runbook — carbonio

Rebuilding TensorFlow from source after an NVIDIA GPU swap.
Written after the RTX 1080 Ti → RTX 5060 Ti (Blackwell, `sm_120`) migration, Sept 2026.

**Use this when:** you install a GPU whose compute capability is newer than what
your current TensorFlow wheel was built for. The tell is one of:

```
TensorFlow was not built with CUDA kernel binaries compatible with compute capability X.Y
'cuModuleLoadData(&module, data)' failed with 'CUDA_ERROR_INVALID_PTX'
```

The first alone is survivable (PTX gets JIT-compiled). The second is fatal and means the
old build's `ptxas` predates the new architecture, so even the PTX fallback is invalid.

---

## 0. Known-good configuration (Sept 2026)

| Component | Value | Notes |
|---|---|---|
| GPU | RTX 5060 Ti, `sm_120` / reported `12.0a` | Blackwell |
| Driver | `nvidia-driver-595-open` (Ubuntu repo) | open kernel module; NOT NVIDIA's CUDA repo |
| TensorFlow | 2.21.0, built from source | `v2.21.0` git tag |
| CUDA | 12.9.1, hermetic (Bazel-fetched) | no system toolkit installed |
| cuDNN | 9.10.2, hermetic | |
| CUDA compiler | **nvcc** (from hermetic CUDA) | clang cannot emit `sm_120` |
| Host compiler | gcc-13 | set by `./configure` when nvcc is chosen |
| Python | 3.12, system (`/usr/bin/python3`) | |
| Runtime CUDA | `nvidia-*` pip packages | thin wheel (`--config=cuda_wheel`) |

TensorFlow does **not** support CUDA 13 as of TF 2.21. Use 12.9.x.
A CUDA-13-era driver running a 12.9 toolkit is fine and fully supported.

---

## 1. Driver

Rule: **the driver comes from Ubuntu's repo, never from NVIDIA's CUDA apt repo.**

`cuda-12-8`, `cuda-runtime-12-8` and similar metapackages hard-depend on their
matching driver branch (570 for 12.8). Installing one on a box running a newer
driver causes an unresolvable dpkg file conflict between `libnvidia-compute-570`
and `libnvidia-common-<newer>`, and leaves apt in a broken state.

`cuda-toolkit-12-9` does **not** pull a driver — that is the safe metapackage if
you ever genuinely need a system toolkit. With hermetic builds you do not.

### Install / reinstall

```bash
mokutil --sb-state                       # Secure Boot on? you'll need MOK enrollment
sudo apt install nvidia-driver-595-open
sudo reboot
nvidia-smi                               # must show the card
ldconfig -p | grep libcuda               # must show libcuda.so.1
```

Do **not** `apt-mark hold` the driver. It blocks security fixes within the branch
and does not protect DKMS across kernel upgrades (a new kernel rebuilds whatever
driver is installed regardless).

### If apt is already broken

```bash
# 1. Never run `apt autoremove` while broken — it will propose removing
#    nvidia-dkms-*-open and take your working driver with it.
sudo apt-mark manual nvidia-dkms-595-open nvidia-kernel-source-595-open \
  nvidia-kernel-common-595 libnvidia-common-595

# 2. Force-clear everything stuck in `iU` (unpacked, unconfigured)
dpkg -l | awk '$1=="iU" {print $2}' > /tmp/broken.txt
cat /tmp/broken.txt                      # review before running the next line
sudo dpkg -P --force-all $(cat /tmp/broken.txt)
sudo dpkg --configure -a
sudo apt --fix-broken install            # should report nothing to do

# 3. Full purge if starting fresh (from a TTY or SSH — this kills the GUI)
sudo apt --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" \
  "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" \
  "*nvvm*" "*nvfatbin*" "*nvjitlink*"
sudo apt --purge remove "*nvidia*"
sudo rm -rf /usr/local/cuda* /usr/share/nvidia
sudo rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/cuda*.sources
```

Note: `dpkg` checks file ownership against *its own database*, not the filesystem.
Deleting a conflicting file by hand does not resolve a "trying to overwrite" error.

Never resolve such a conflict with `dpkg --force-overwrite`. Mixing driver branches
(e.g. 570 userspace against a 595 kernel module) breaks CUDA in far more confusing ways.

---

## 2. Build

### Prerequisites

```bash
cd ~/Software/tensorflow-dir/gpu/tensorflow
git checkout v2.21.0
git describe --tags                      # confirm; a moving checkout explains odd pins
cat .bazelversion && bazel --version     # must match; use bazelisk if unsure
python3 -c "import numpy, packaging, wheel, requests"
df -h .                                  # ~50 GB free
```

### configure

```bash
./configure
```

Answers that matter:

| Prompt | Answer |
|---|---|
| CUDA support | `Y` |
| Hermetic CUDA version | `12.9.1` — **full version**, a bare `12.9` is rejected |
| Hermetic cuDNN version | `9.10.2` — **full version**, a bare `9` is rejected |
| Compute capabilities | `12.0` (the `x.y` form; a bare `12` is rejected here) |
| Local CUDA / cuDNN / NCCL paths | leave empty |
| Use clang as CUDA compiler | **`n`** — clang 18 cannot emit `sm_120` |
| Optimization flags | leave default; skip `-march=native` and `-O3` |

Answering `n` to the clang question switches the host compiler to gcc-13 and
writes `build --config=cuda` instead of `build --config=cuda_clang`. That matters —
see below.

### Build command

```bash
nohup bazel build --config=opt --config=cuda_wheel --config=nonccl \
  --repo_env=WHEEL_NAME=tensorflow \
  --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=sm_120,compute_120 \
  --repository_cache=$HOME/.cache/bazel_repo_cache \
  --distdir=$HOME/bazel_distdir \
  --experimental_repository_downloader_retries=5 \
  //tensorflow/tools/pip_package:wheel > tf_build.log 2>&1 &

tail -f tf_build.log | grep -E 'ERROR|\[[0-9,]+ /'
```

**Pass the compute capabilities on the command line.** Putting them only in
`.tf_configure.bazelrc` is unreliable: `common:cuda_clang` in `.bazelrc` hardcodes
`sm_60,sm_70,sm_80,sm_89,compute_90` and can win the expansion-order fight. (With
`--config=cuda` rather than `cuda_clang` this is moot, but the flag costs nothing.)

**The env var accepts only `sm_XY` / `compute_XY`.** An `x.y` value such as `12.0`
is silently ignored and the default list is used instead. This is different from
`./configure`'s prompt, which wants `x.y`. Always verify afterwards (§3).

`--config=cuda_wheel` produces a *thin* wheel: CUDA is not bundled and must come
from the `nvidia-*` pip packages at runtime. Omit it for a self-contained wheel
(several GB larger, no pip CUDA needed, immune to the `libcusolver` problem in §4).

`--config=nonccl` is safe on a single-GPU box and saves build time.

If the link stage OOMs, add `--jobs=HOST_CPUS*0.75 --local_ram_resources=HOST_RAM*0.6`.

### LLVM checksum failures

```
Checksum was 00b1077e... but wanted 3f986184...
```

GitHub serves **two different tarballs** for the same commit depending on which CDN
backend answers. The pin is correct; you just need the matching copy. Do **not**
patch the expected checksum — it will fail again on the next coin flip, and it
disables the integrity check.

```bash
mkdir -p ~/bazel_distdir && cd ~/bazel_distdir
URL=https://github.com/llvm/llvm-project/archive/<COMMIT>.tar.gz
WANT=<the "wanted" hash from the error>
F=<COMMIT>.tar.gz
for i in $(seq 1 10); do
  curl -fL -o "$F" "$URL" || continue
  GOT=$(sha256sum "$F" | cut -d' ' -f1)
  echo "attempt $i: $GOT"
  [ "$GOT" = "$WANT" ] && { echo MATCH; break; }
done
```

Then build with `--distdir=$HOME/bazel_distdir`. Bazel still verifies the hash, so
nothing is bypassed. Once cached in `--repository_cache`, this never recurs.

Do not use `--override_repository=llvm-raw=...` — it skips XLA's LLVM patches and
fails much later with obscure MLIR errors.

---

## 3. Install and verify

```bash
pip uninstall -y tensorflow
pip install "/path/to/tensorflow-2.21.0-cp312-cp312-linux_x86_64.whl[and-cuda]"
```

Omit `[and-cuda]` if you built a self-contained wheel, and uninstall the `nvidia-*`
packages so you can prove it is genuinely self-contained.

### Verify in this order — stop at the first failure

```bash
# 1. What the build recorded
python3 -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
#    want: cuda_compute_capabilities: ['sm_120', 'compute_120']
#          cuda_version: '12.9.1', is_cuda_build: True

# 2. What is actually in the binary
TFDIR=$(python3 -c "import tensorflow,os;print(os.path.dirname(tensorflow.__file__))")
grep -c 'sm_120' $TFDIR/libtensorflow_cc.so.2        # expect thousands
# with a CUDA toolkit installed, better:
# cuobjdump --list-elf $TFDIR/libtensorflow_cc.so.2 | grep -c sm_120

# 3. Runtime — in a FRESH shell, with LD_LIBRARY_PATH unset
echo $LD_LIBRARY_PATH                                # must be empty
REQUIRE_GPU=1 python3 -c "
import DataML_Backend, tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import time; t=time.time()
print(tf.reduce_sum(tf.random.normal([4000,4000])))
print('first op: %.1fs' % (time.time()-t))
"
```

Success criteria for step 3: GPU listed, **no** "not built with CUDA kernel binaries"
warning, first op under ~1s. A first op taking tens of seconds to minutes means you
are still JIT-compiling from PTX — step 1 or 2 lied, or the capability flag was ignored.

The device reports `compute capability: 12.0a`. An `sm_120` cubin runs on it correctly.
If (and only if) the JIT warning persists with `sm_120` confirmed present, rebuild
with `sm_120a,compute_120`.

---

## 4. Runtime gotchas

### Missing `libcusolver.so.11` (thin wheels only)

Symptom: GPU never registers; the only clue is a generic

```
Cannot dlopen some GPU libraries ... Skipping registering GPU devices
```

TensorFlow does not print which library failed at the default log level. Get the name:

```bash
TF_CPP_MAX_VLOG_LEVEL=3 python3 -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')" 2>&1 \
  | grep -iE 'dynamic library' | head -40
```

Cause: the `nvidia-*` pip packages scatter CUDA across ten
`site-packages/nvidia/<component>/lib` directories that the dynamic loader does not
search. TF preloads only a subset by explicit path; the rest fall through and fail.
A system CUDA install at `/usr/local/cuda-*` plus `LD_LIBRARY_PATH` used to mask this.

Fix: `_preloadCudaLibs()` in `DataML_Backend.py` (v2026.9.16.1+) dlopens every
`nvidia/*/lib/*.so.*` by absolute path at import, in repeated passes so interdependent
libraries resolve. A library loaded under its soname satisfies every later dlopen of
that soname.

Setting `LD_LIBRARY_PATH` from inside Python does **not** work — glibc caches the
search path at process start.

Alternatives, in descending order of robustness:

1. The preload shim (travels with the code; works under SLURM, cron, IDEs)
2. `/etc/ld.so.conf.d/nvidia-pip.conf` + `ldconfig` (machine-wide, but hardcodes the
   Python path and can shadow PyTorch's own CUDA versions)
3. `LD_LIBRARY_PATH` in `.bashrc` — **avoid**: interactive shells only, and it shadows
   other environments' CUDA

Or sidestep entirely: build without `--config=cuda_wheel`.

### Silent CPU fallback

TensorFlow warns and continues on the CPU rather than failing. Run production jobs
with `REQUIRE_GPU=1` so `DataML_Backend` raises instead.

### Stale `LD_LIBRARY_PATH`

`./configure` captures `LD_LIBRARY_PATH` from your shell into `.tf_configure.bazelrc`.
A dead `/usr/local/cuda-12.5` entry lingering in `.bashrc` will be baked in and will
confuse later debugging. Clear it.

---

## 5. Packaging gotchas

### A transitive pin can replace your hand-built wheel

`pip install --upgrade dataml.whl` silently downgraded TensorFlow 2.21 → 2.19,
destroying the custom build. Cause: `tf-keras` is versioned in lockstep with
TensorFlow and carries a hard upper bound (`tf-keras 2.19.0` → `tensorflow>=2.19,<2.20`).
With pip's default `only-if-needed` strategy, an already-installed `tf-keras 2.19.0`
satisfied the bare requirement, so pip resolved TensorFlow *downwards* to match it.

Mitigations, all three worth having:

1. `tf-keras` removed from `pyproject.toml` (all its imports were already dead code
   behind `checkTFVersion("2.16.0")` gates)
2. Floor in `pyproject.toml`: `"tensorflow>=2.21"`
3. **Habit:** `pip install --no-deps --upgrade dataml.whl` for your own packages

The floor stops this specific trap; `--no-deps` stops the next one.

### Check after any install

```bash
python3 -c "import tensorflow as tf; print(tf.__version__, tf.sysconfig.get_build_info()['cuda_compute_capabilities'])"
```

A venv rather than `sudo pip3` into system Python makes a hand-built wheel much
harder to clobber. Worth doing given what a rebuild costs.

---
## 6. Persistance
Your nvidia-smi output shows Persistence-M: Off. If you launch CUDA compute jobs, Python scripts, or containerized workloads, having persistence mode disabled forces the driver and GSP firmware to tear down and reinitialize every time an application closes, introducing execution latency.
Set persistence mode directly in the running kernel driver:
```bash
sudo nvidia-smi -pm 1
```
Verify it is active:
```bash
nvidia-smi -q | grep "Persistence Mode"
```
Expected output: Persistence Mode : Enabled
Make It Persistent Across Reboots: Add the missing [Install] target using a systemd drop-in override so systemctl enable can create the proper startup symlink:
Create the override directory
```bash
sudo mkdir -p /etc/systemd/system/nvidia-persistenced.service.d
```
Append the Install target
```bash
cat << 'EOF' | sudo tee /etc/systemd/system/nvidia-persistenced.service.d/override.conf
[Install]
WantedBy=multi-user.target
EOF
```
Reload systemd and enable the daemon
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-persistenced
```

Verification: Confirm the service is running and persistence mode remains active:
```bash
systemctl status nvidia-persistenced
nvidia-smi -q | grep "Persistence Mode"
```

The daemon will now start automatically at boot, keeping the GSP firmware context and driver initialized in memory.

## 7. Symptom → cause

| Symptom | Cause |
|---|---|
| `CUDA_ERROR_INVALID_PTX` | TF built with a pre-12.8 toolchain; its ptxas cannot emit Blackwell PTX |
| "not built with CUDA kernel binaries compatible with 12.0" | No `sm_120` cubins; JIT from PTX (works, slow) |
| `clang: error: unsupported CUDA gpu architecture: sm_120` | clang 18 tops out at `sm_90`; use nvcc or clang 20+ |
| `Invalid compute capability: 12` (configure) | Use `12.0`, `sm_120` or `compute_120` |
| Build info shows `sm_60...compute_90` despite config | `HERMETIC_CUDA_COMPUTE_CAPABILITIES` got an `x.y` value and ignored it |
| `The supported CUDNN versions are [...]` | Bare major version; use a full `9.10.2` |
| `cuda-runtime-12-8 : Depends: libnvidia-compute-570` | Driver-bundling metapackage; use `cuda-toolkit-*` or hermetic |
| `trying to overwrite ... also in package libnvidia-common-595` | Mixed driver branches; never `--force-overwrite` |
| `Checksum was X but wanted Y` (llvm-raw) | GitHub CDN serves two tarballs; retry into `--distdir` |
| `No MODULE.bazel, REPO.bazel, or WORKSPACE file found` | `--override_repository` at a path with no marker — don't use override |
| "Cannot dlopen some GPU libraries", no name given | Usually `libcusolver.so.11`; see §4 |
| TensorFlow silently downgraded | `tf-keras` upper bound; see §5 |

---

## 8. Things that cost time and did not help

- Patching `LLVM_SHA256` in the cached XLA repo — the pin was correct; the download
  was the variable
- `--override_repository=llvm-raw=...` — skips XLA's patches
- `apt-mark hold` on the driver — no benefit, blocks security fixes
- `-march=native` / `-O3` — no measured gain, non-portable wheel, extra failure surface
- Setting `LD_LIBRARY_PATH` from within Python — glibc already cached the path
