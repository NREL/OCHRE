import datetime as dt

equip_init_args = {
    'start_time': dt.datetime(2019, 4, 1),
    'duration': dt.timedelta(days=1),
    'time_res': dt.timedelta(minutes=1),
    'initial_schedule': {},
    'schedule': {},
    'zip_model': {'Test Equipment': {'pf': 0.9, 'pf_inductive': True,
                                     'Zp': 0, 'Ip': 0, 'Pp': 1,
                                     'Zq': 0, 'Iq': 0, 'Pq': 1}},
    'save_results': False,
}
