import re
import numpy as np
import os
import sys
from contextlib import contextmanager
import nest
from typing import Optional


@contextmanager
def _suppress_output():
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()


def reset_nest():
  with _suppress_output():
        try:
            nest.set_verbosity("M_FATAL")
        except:
            try:
                nest.SetKernelStatus({"print_time": False})
            except:
                pass
        nest.ResetKernel()
        nest.SetKernelStatus({"local_num_threads": 1})
        try:
            nest.Install("cerebmodule")
        except:
            raise Exception("NEST module not installed.")
        nest.resolution = 0.1


def nest_single_sim(nest_model, current, protocol, model_params, output='spk'):
    cell = nest.Create(nest_model, 1, params=model_params)
    cell.V_m = model_params["E_L"]
    gen = nest.Create("dc_generator")
    nest.SetStatus(
        gen,
        {
            "amplitude": current,
            "start": protocol["stim_start"],
            "stop": protocol["stim_end"],
        },
    )
    sr = nest.Create("spike_recorder")
    vr = nest.Create("voltmeter")
    nest.Connect(gen, cell)
    nest.Connect(cell, sr)
    nest.Connect(vr, cell)
    if output == 'spk':
        return sr
    elif output == 'voltage':
        return vr
    else:
        return sr, vr


def run_nest_batch(nest_model, inj_currents, protocol, model_params):
    reset_nest()
    # prepare a batch of single cell simulations to run together
    nest_recorders = {
        I: nest_single_sim(nest_model, I, protocol, model_params)
        for I in inj_currents
    }
    
    # simulate and extract results
    nest.Simulate(protocol["duration"])
    return [
      nest.GetStatus(sr, "events")[0]["times"] for sr in nest_recorders.values()
    ]


def parse_current_from_filename(fname: str) -> float:
    m = re.search(r"inj(-?\d+(?:\.\d+)?)", fname)
    return float(m.group(1)) if m else np.nan


def load_experiments_files(
    data_folder:str,
    stim_start: Optional[float] = None,
    stim_end: Optional[float] = None,
    ):

    data_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    files_curr = [(parse_current_from_filename(f), f) for f in data_files]
    files_curr = [(c, f) for c, f in files_curr if np.isfinite(c)]
    files_curr.sort(key=lambda x: x[0])

    unique_currents = sorted({c for c, _ in files_curr})
    if not unique_currents:
        raise RuntimeError(f"No valid .txt traces found in: {data_folder}")

    traces = []
    for curr in unique_currents:
        print('curr: ', curr)
        fname = [f for c, f in files_curr if c == curr][0]
        data = np.loadtxt(os.path.join(data_folder, fname))
        time = data[:, 0].astype(float)
        voltage = data[:, 1].astype(float)

        trace = {
            "T": time,
            "V": voltage,
            "stim_start": [stim_start if stim_start is not None else time[0]],
            "stim_end": [stim_end if stim_end is not None else time[-1]]            
        }
        traces.append(trace)
                
    return traces, unique_currents
