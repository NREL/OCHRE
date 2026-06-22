import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd 

from scipy.stats import lognorm, norm
import psychrolib

#GLOBALS
L_PER_G = 3.78541
sites = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 22, 23, 24, 25, 27, 28, 29, 30]
altitude = {1:30, 2:40, 3:375, 4:12, 5:397, 7:203, 8:238, 9:285, 10:238, 11:1211, 12:500, 13:21, 14:40, 15:28, 16:7, 18:61, 19:61, 21:77, 22:41, 23:11, 24:8, 25:222, 27:89, 28:89, 29:89, 30:54} #meters


def get_atmospheric_pressure(altitude_m):
    """Computes pressure in hPa based on altitude (standard atmosphere)."""
    # Using the standard barometric formula: P = P0 * (1 - L*h/T0)^(g*M/R*L)
    # At lower altitudes, this is the logic used by the TAMU calculator
    return psychrolib.GetStandardAtmPressure(altitude_m) 

def calculate_wet_bulb_tamu(temp_c, rh_percent, altitude_m):
    """
    Computes wet-bulb temperature following ASHRAE/TAMU calculator logic.
    Logic: Solve for Tw where Actual Vapor Pressure (ed) 
    equals the Psychrometric Wet-Bulb equation.
    """
    psychrolib.SetUnitSystem(psychrolib.SI)
    return psychrolib.GetTWetBulbFromRelHum(temp_c, rh_percent/100, get_atmospheric_pressure(altitude_m))

def F_to_C(F):
    temps = F.astype(float)
    return (temps - 32) * 5.0/9.0

def process_single_site(site_number):
    try:
        # --- 1. FLOW DATA PROCESSING ---
        flow_path = Path(f"raw_data/site_{site_number}_RAW.zip")
        if not flow_path.is_file():
            return f"Site {site_number}: Flow file not found."

        flow_data = pd.read_csv(flow_path)
        flow_data = flow_data[flow_data['Unit'] == 'gal'].copy()
        
        flow_data['Time'] = pd.to_datetime(flow_data['Time'], errors='coerce')
        flow_data = flow_data.dropna(subset=['Time']).sort_values(by='Time')

        if flow_data.empty:
            return f"Site {site_number}: Empty flow data."

        min_ts, max_ts = flow_data['Time'].min(), flow_data['Time'].max()
        
        start = min_ts.normalize()
        if min_ts.time() != start.time():
            start = (min_ts + pd.Timedelta(days=1)).normalize()
            
        end = max_ts.normalize()
        if (end + pd.Timedelta(days=1)) <= start:
            # Fallback instead of printing straight to console safely in parallel
            start, end = min_ts, max_ts 

        flow_data = flow_data[(flow_data['Time'] >= start) & (flow_data['Time'] <= end)].copy()
        flow_data['Value'] *= L_PER_G 

        out = flow_data.groupby(flow_data['Time'].dt.floor('min'))['Value'].sum().reset_index()

        # Save flow files immediately
        out.to_csv(f"120V_data_clean/net_flow_{site_number}_120V_times.csv", header=False, index=False)
        
        # --- 2. TEMPERATURE DATA PROCESSING ---
        temp_path = Path(f"raw_data/temp_data/site_{site_number}.csv")
        if not temp_path.is_file():
            return f"Site {site_number}: Processed flow, but temp file not found."

        site_data = pd.read_csv(temp_path)
        site_data['HPWH_pwr_kW'] = site_data['HPWH_pwr_kW'].fillna(0)

        column_names = ['site_id', 'local_datetime', 'inlet_cold_water_temp_F', 'inlet_air_temp_F', 'inlet_RH', 'outlet_hot_water_Gal', 'return_hot_water_temp_F', 'HPWH_pwr_kW']
        site_data = site_data[column_names].dropna(axis=0, how='any')

        site_data['Time'] = pd.to_datetime(site_data['local_datetime'], utc=True, format="%Y-%m-%d %H:%M:%S%z", errors='coerce')
        
        min_ts, max_ts = site_data['Time'].min(), site_data['Time'].max()

        # Shift daily time range to 4am - 4am
        start = min_ts.normalize() + pd.Timedelta(hours=4)
        if min_ts > start:
            start += pd.Timedelta(days=1)
        end = max_ts.normalize()

        site_data = site_data[(site_data['Time'] >= start) & (site_data['Time'] <= end)].copy()
        site_data = site_data.set_index('Time')

        temp_cols = ['inlet_cold_water_temp_F', 'inlet_air_temp_F', 'inlet_RH', 'outlet_hot_water_Gal', 'return_hot_water_temp_F']
        temps = site_data[temp_cols].apply(pd.to_numeric, errors='coerce')

        # Convert to Celsius for OCHRE compatibility
        site_data['inlet_cold_water_temp_C'] = F_to_C(temps['inlet_cold_water_temp_F'])
        site_data['inlet_air_temp_C'] = F_to_C(temps['inlet_air_temp_F'])
        site_data['return_hot_water_temp_C'] = F_to_C(temps['return_hot_water_temp_F'])
        
        #Interpolate values from 5 minute to 1 minute resolution
        interpolate_values = ['inlet_cold_water_temp_C', 'inlet_air_temp_C', 'inlet_RH', 'outlet_hot_water_Gal', 'return_hot_water_temp_C']
        site_data[interpolate_values] = site_data[interpolate_values].apply(pd.to_numeric, errors='coerce')

        # Resample & interpolate missing values
        site_data_temps = site_data[interpolate_values].resample('1min').interpolate(method='time').ffill().bfill()

        # CRITICAL SPEEDUP: Vectorized Wet Bulb Calculation (No .apply axis=1)
        alt = altitude[site_number] #Site altitude
        site_data_temps['wet_bulb_C'] = calculate_wet_bulb_tamu(
            site_data_temps['inlet_air_temp_C'], 
            site_data_temps['inlet_RH'], 
            alt
        )

        # --- 3. MERGING IN-MEMORY (NO RE-READING FILES) ---
        out = out.rename(columns={'Value': 'flow_L_per_min'}).set_index('Time').sort_index()
        
        if not isinstance(site_data_temps.index, pd.DatetimeIndex):
            site_data_temps.index = pd.to_datetime(site_data_temps.index, utc=True, errors='coerce')
        if site_data_temps.index.tz is None:
            site_data_temps.index = site_data_temps.index.tz_localize('UTC')

        #Join flow and temperature data on minute time step indices
        out.index = out.index.floor('min')
        site_data_temps.index = site_data_temps.index.floor('min')
        #add flow to temperature data
        combined = site_data_temps.join(out[['flow_L_per_min']], how='left')
        combined['flow_L_per_min'] = combined['flow_L_per_min'].fillna(0)

        combined.to_csv(f'120V_data_clean/120V_temperatures_{site_number}.csv')
        return f"Site {site_number}: Success"

    except Exception as e:
        return f"Site {site_number}: Failed with error: {str(e)}"

# --- EXECUTION ENVELOPE (Multi-core CPU Harness) ---
if __name__ == "__main__":
    # Use ProcessPoolExecutor to leverage all CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_site, sites))
        
    for result in results:
        print(result)