import os
import pandas as pd
import pytest
import datetime as dt

from ochre import Simulator


@pytest.fixture
def start_time(): return dt.datetime(2024, 2, 3, 4)  # Feb 3, 4AM
@pytest.fixture
def time_res(): return dt.timedelta(minutes=60)
@pytest.fixture
def duration(): return dt.timedelta(days=1)


@pytest.fixture
def my_schedule(start_time: dt.datetime, time_res: dt.timedelta, duration: dt.timedelta):
    times = pd.date_range(start_time, start_time + duration, freq=time_res, inclusive="left")
    return pd.DataFrame(
        {
            "A": range(24),
            "B": 10,
        },
        index=times,
    )


@pytest.fixture
def sim(start_time, time_res, duration):
    return Simulator(
        start_time=start_time,
        time_res=time_res,
        duration=duration,
        save_results=False,
    )


@pytest.fixture
def sim_with_schedule(start_time, time_res, duration, my_schedule):
    return Simulator(
        start_time=start_time,
        time_res=time_res,
        duration=duration,
        save_results=False,
        required_inputs=["A"],
        schedule=my_schedule,
    )


@pytest.fixture
def sim_with_results(start_time, time_res, duration, tmp_path):
    return Simulator(
        start_time=start_time,
        time_res=time_res,
        duration=duration,
        output_path=tmp_path,
    )


@pytest.fixture
def subsim(start_time, time_res, duration, my_schedule):
    return Simulator(
        start_time=start_time,
        time_res=time_res,
        duration=duration,
        save_results=False,
        required_inputs=["A"],
        schedule=my_schedule,
        main_sim_name="ochre",  # sub simulator for tests
    )


class TestSimulator:
    """
    Unit tests for the Simulator class.
    """

    def test_init(self, sim: Simulator, start_time):
        assert sim.name == "OCHRE"
        assert sim.current_time == start_time
        assert len(sim.sim_times) == 24
        assert sim.main_simulator is True

        assert not sim.save_results
        # assert os.path.exists(sim.output_path)
        # assert os.path.basename(sim.results_file) == "OCHRE.csv"
        # assert not os.path.exists(sim.results_file)
        
        assert sim.schedule is None
        assert sim.current_schedule == {}

    def test_initialize_schedule(self, sim: Simulator, my_schedule):
        schedule = sim.initialize_schedule(my_schedule)
        assert schedule is None

    def test_schedule(self, sim_with_schedule: Simulator):
        assert sim_with_schedule.schedule.columns == ["A"]
        assert len(sim_with_schedule.schedule) == 25  # TODO: replace with 24 after merge

    def test_update(self, sim_with_schedule: Simulator, start_time, time_res):
        assert sim_with_schedule.current_time == start_time
        assert sim_with_schedule.current_schedule["A"] == 0

        results = sim_with_schedule.update()
        assert len(results) == 1
        assert results["Time"] == start_time

        results = sim_with_schedule.update()
        assert results["Time"] == start_time + time_res
        assert sim_with_schedule.current_time == start_time + time_res * 2
        assert sim_with_schedule.current_schedule["A"] == 1

    def test_reset_time(self, sim_with_schedule: Simulator, start_time, time_res):
        sim_with_schedule.update()
        assert sim_with_schedule.current_time != start_time

        sim_with_schedule.reset_time()
        assert sim_with_schedule.current_time == start_time

        new_time = start_time + time_res * 6
        sim_with_schedule.reset_time(new_time)
        assert sim_with_schedule.current_time == new_time
        sim_with_schedule.update()
        assert sim_with_schedule.current_schedule == sim_with_schedule.schedule.loc[new_time].to_dict()


    def test_initialize(self, sim_with_schedule: Simulator, start_time, duration):
        sim_with_schedule.initialization_time = duration
        sim_with_schedule.initialize()
        assert sim_with_schedule.current_time == start_time

    def test_simulate(self, sim_with_results: Simulator, tmp_path):
        df = sim_with_results.simulate()
        assert len(df) == 24
        assert df.index.name == "Time"
        assert df.columns.tolist() == []

        assert os.path.exists(sim_with_results.results_file)
        assert (tmp_path / "ochre_complete").exists()


class TestSimulatorWithSubs:
    """
    Unit tests for Simulator class with sub-simulators.
    """
        
    @pytest.fixture(autouse=True)
    def add_sub(self, sim: Simulator, sim_with_results: Simulator, subsim: Simulator):
        # add subsim to sim and sim_with_results
        sim.sub_simulators.append(subsim)
        sim_with_results.sub_simulators.append(subsim)

    def test_init(self, sim: Simulator):
        assert len(sim.sub_simulators) == 1
        
    def test_update(self, sim_with_results: Simulator, subsim: Simulator, start_time, time_res):
        results = sim_with_results.update()
        assert sim_with_results.current_schedule == {}
        assert results["Time"] == start_time
        assert subsim.current_time == start_time + time_res
        assert subsim.current_schedule["A"] == 0
        
        results = sim_with_results.update()
        assert subsim.current_schedule["A"] == 1
        assert results["Time"] == start_time + time_res

    def test_reset_time(self, sim: Simulator, subsim: Simulator, start_time):
        sim.update()
        sim.reset_time()
        assert sim.current_time == start_time
        assert subsim.current_time == start_time
    
    def test_simulate(self, sim_with_results: Simulator, subsim: Simulator):
        df = sim_with_results.simulate()
        assert len(df) == 24
