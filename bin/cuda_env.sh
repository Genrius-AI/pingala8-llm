# Load the PTX JIT library that matches the Windows driver (the apt 580.178 copy segfaults on 592.01). 2026-10-05
P=$(ls -d /usr/lib/wsl/drivers/nv*/libnvidia-ptxjitcompiler.so.1 2>/dev/null | head -1)
[ -n "$P" ] && export LD_PRELOAD="$P${LD_PRELOAD:+:$LD_PRELOAD}"
