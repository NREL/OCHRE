import datetime as dt
import pandas as pd

# Create an empty schedule DataFrame with proper DatetimeIndex
start_time = dt.datetime(2019, 4, 1)
duration = dt.timedelta(days=1)
time_res = dt.timedelta(minutes=1)
sim_times = pd.date_range(start_time, start_time + duration, freq=time_res, inclusive="left")
empty_schedule = pd.DataFrame(index=sim_times)

equip_init_args = {
    "start_time": start_time,
    "duration": duration,
    "time_res": time_res,
    "ext_time_res": dt.timedelta(minutes=15),
    "initial_schedule": {},
    "schedule": empty_schedule,
    # ZIP parameters passed directly (not nested in zip_model dict)
    "pf": 0.9,
    "Zp": 0,
    "Ip": 0,
    "Pp": 1,
    "Zq": 0,
    "Iq": 0,
    "Pq": 1,
    "save_results": False,
    "verbosity": 9,  # High verbosity to include Mode in results
}
