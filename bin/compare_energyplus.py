import os

from ochre import Analysis, CreateFigures, Dwelling

# Script to compare OCHRE and E+ outputs for generic model. Assumes both models have been run in the same folder

# File locations
main_path = os.path.join('path', 'to', 'ochre_folder')
eplus_file = os.path.join(main_path, 'results_timeseries.csv')
simulation_name = 'ochre'  # name for OCHRE files


if __name__ == '__main__':
    # Load OCHRE files
    ochre_exact, ochre_metrics, ochre = Analysis.load_ochre(main_path, simulation_name, load_main=False, 
                                                            combine_schedule=True)

    # Load E+ files
    eplus = Analysis.load_eplus_file(eplus_file, year=ochre.index[0].year)

    # keep days from OCHRE simulation
    eplus = eplus.loc[ochre.index[0]: ochre.index[-1]]
    eplus_metrics = Analysis.calculate_metrics(eplus, metrics_verbosity=6)

    # Compare metrics and save to file
    compare_metrics = Analysis.create_comparison_metrics(ochre, eplus, ochre_metrics, eplus_metrics)
    metrics_file = os.path.join(main_path, simulation_name + '_comparison.csv')
    compare_metrics.to_csv(metrics_file)

    print(f'Comparison Metrics for {simulation_name}:')
    print(compare_metrics)

    # show plots
    # data = {'OCHRE (exact)': ochre_exact, 'OCHRE': ochre, 'E+': eplus}
    data = {'OCHRE': ochre, 'E+': eplus}
    # CreateFigures.plot_external(data)
    CreateFigures.plot_envelope(data)
    # CreateFigures.plot_hvac(data)
    # CreateFigures.plot_wh(data)
    CreateFigures.plot_end_use_powers(data)
    CreateFigures.plt.show()
