import psutil
import subprocess
import sys
import time

import pandas as pd


def get_gpu_names():
    gpu_names = subprocess.check_output(["nvidia-smi", "-L"])
    return [name.split(":")[0] for name in gpu_names.decode().split("\n") if name != ""]


def main(fname):
    refresh_interval = 1
    df = pd.DataFrame()
    df = pd.DataFrame(columns=["RAM"] + [f"{g}" for g in get_gpu_names()])
    # infinity loop
    while True:
        t_begin = time.time()
        # pool cpu
        cpu_ram = psutil.virtual_memory().used / (1024 * 1024)
        # pool gpu
        gpu_memories = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=" + "memory.used",
                "--format=csv,nounits,noheader",
            ]
        )
        gpu_mem = [int(mem) for mem in gpu_memories.decode().split("\n") if mem != ""]

        df.loc[len(df.index)] = [cpu_ram] + gpu_mem

        t_sleep = refresh_interval + t_begin - time.time() - 0.001
        if t_sleep > 0:
            time.sleep(t_sleep)
        df.to_csv(fname)


if __name__ == "__main__":
    try:
        print(sys.argv[1])
    except IndexError as error:
        print("Save file path required, please provide!")
        exit(1)
    main(sys.argv[1])
