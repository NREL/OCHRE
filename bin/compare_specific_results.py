import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import numpy as np


L_TO_GAL_RATIO = 0.264172

def load_data(results_folder='../OCHRE_output/results/'):
    """Load CSV files from the results folder into a dictionary of DataFrames."""
    csv_files = [f for f in os.listdir(results_folder) if f.endswith('.csv')]
    if len(csv_files) < 2:
        raise ValueError("At least 2 CSV files are required in the 'results' folder for comparison.")
    return {file: pd.read_csv(os.path.join(results_folder, file)) for file in csv_files}

def find_matching_columns(df, patterns):
    """Find columns that match the given patterns and group them."""
    column_groups = {pattern: [] for pattern in patterns}
    
    for col in df.columns:
        for pattern in patterns:
            # Use regex to match pattern followed by a number
            match = re.match(f"{pattern}(\\d+)$", col)
            if match:
                column_groups[pattern].append(col)
    
    # Sort columns within each group by node number
    for pattern in patterns:
        column_groups[pattern].sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    return column_groups

def plot_draw_event_summary(draw_outputs):
    """
    Plot a grouped bar chart showing the overall totals for each file.
    Total water delivered (gallons) and total heat delivered (kWh) are compared side by side.
    """
    files = list(draw_outputs.keys())
    total_volumes = [draw_outputs[file]['total_water_volume_gal'] for file in files]
    total_heat = [draw_outputs[file]['total_heat_delivered_kWh'] for file in files]
    total_energy = [draw_outputs[file]['total_energy_used_kwh'] for file in files]

    # Create a grouped bar chart with numeric labels on each bar
    fig = go.Figure(data=[
        go.Bar(
            name='Water Volume (gal)', 
            x=files, 
            y=total_volumes,
            text=[f"{vol:.2f}" for vol in total_volumes],
            textposition='auto'
        ),
        go.Bar(
            name='Heat Delivered (kWh)', 
            x=files, 
            y=total_heat,
            text=[f"{heat:.3f}" for heat in total_heat],
            textposition='auto'
        ),
        go.Bar(
            name='Energy Used (kWh)', 
            x=files, 
            y=total_energy,
            text=[f"{energy:.3f}" for energy in total_energy],
            textposition='auto'
        )
    ])

    # Update layout for grouped bars and titles
    fig.update_layout(
        barmode='group',
        title='Total Hot Water Delivered Summary by File',
        xaxis_title='File',
        yaxis_title='Value'
    )
    fig.show()

def plot_draw_events(draw_outputs):
    """
    For each file, create a separate grouped bar chart for the individual draw events.
    Each event is compared by its water volume (gal) and heat delivered (kWh).
    """
    # Determine the maximum number of draw events among all files.
    max_events = max(len(metrics['draw_events']) for metrics in draw_outputs.values())
    event_numbers = [f"Event {i+1}" for i in range(max_events)]
    
    # Build data dictionaries for water volume and heat delivered per file
    water_data = {}  # key: file, value: list of water volumes per event (or None if missing)
    heat_data = {}   # key: file, value: list of heat delivered per event (or None if missing)
    
    for file, metrics in draw_outputs.items():
        events = metrics['draw_events']
        water_values = []
        heat_values = []
        for i in range(max_events):
            if i < len(events):
                water_values.append(round(events[i]['water_volume_gal'], 2))
                heat_values.append(round(events[i]['heat_delivered_kWh'], 3))
            else:
                water_values.append(None)
                heat_values.append(None)
        water_data[file] = water_values
        heat_data[file] = heat_values

    # Create subplots: 1 row, 2 columns for the two metrics
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Water Volume (gal)", "Heat Delivered (kWh)"),
                        shared_xaxes=True)
    
    # For each file, add a bar trace for water volume in subplot 1
    for file, values in water_data.items():
        fig.add_trace(
            go.Bar(
                name=file,
                x=event_numbers,
                y=values,
                text=[f"{v:.2f}" if v is not None else "" for v in values],
                textposition='auto'
            ),
            row=1, col=1
        )
        
    # For each file, add a bar trace for heat delivered in subplot 2
    for file, values in heat_data.items():
        fig.add_trace(
            go.Bar(
                name=file,
                x=event_numbers,
                y=values,
                text=[f"{v:.3f}" if v is not None else "" for v in values],
                textposition='auto'
            ),
            row=2, col=1
        )
    
    # Update the layout for grouped bars and overall titles
    fig.update_layout(
        barmode='group',
        title_text="Draw Events Grouped by Event Number Across Files",
        xaxis_title="Draw Event"
    )
    
    fig.show()

def calculate_hot_water_delivered(dfs):
    """
    Calculate the total hot water delivered (W) and gallons for each file,
    capturing individual draw events with their time ranges and metrics.
    """
    water_draw_col = "Hot Water Delivered (L/min)"
    water_output_W_col = "Hot Water Delivered (W)"
    water_outlet_temp = "Hot Water Outlet Temperature (C)"
    energy_used = "Water Heating Delivered (W)"
    
    water_temp_cutoff = 43.3333  # 110 F 15 deg delta from 125 F for UEF test
    
    output = {}
    
    for file, df in dfs.items():
        # Create a copy of the dataframe to avoid modifying the origina
        
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Initialize variables to track water draw events
        total_water_volume_L = 0
        total_heat_delivered_J = 0
        is_draw_active = False
        
        # To track individual draw events
        draw_events = []
        current_event = None
        
        # Calculate time delta using the "Time" index
        if not pd.api.types.is_numeric_dtype(df_copy.index):
            # If Time index is datetime
            if pd.api.types.is_datetime64_any_dtype(df_copy.index):
                df_copy['time_delta'] = df_copy.index.to_series().diff().dt.total_seconds()
            else:
                # Try to convert to numeric
                df_copy['time_delta'] = pd.to_numeric(df_copy.index.to_series().diff(), errors='coerce')
        else:
            df_copy['time_delta'] = df_copy.index.to_series().diff()
        
        # Fill NaN for the first row
        # df_copy['time_delta'].fillna(0, inplace=True)
        df_copy.fillna({'time_delta': 0}, inplace=True)
        
        for i, row in df_copy.iterrows():
            water_draw = row[water_draw_col]
            outlet_temp = row[water_outlet_temp]
            heat_output = row[water_output_W_col]
            time_delta = row['time_delta']
            current_time = i  # "Time" index value
            
            # Check if water is being drawn
            if water_draw > 0:
                # Check if this is a new draw event or continuation
                if not is_draw_active and outlet_temp >= water_temp_cutoff:
                    # Start of a new draw event that meets temperature requirements
                    is_draw_active = True
                    current_event = {
                        'start_time': current_time,
                        'end_time': None,
                        'water_volume_L': 0,
                        'heat_delivered_J': 0,
                        'max_temp': outlet_temp,
                        'min_temp': outlet_temp,
                        'max_flow_rate': water_draw
                    }
                
                if is_draw_active:
                    # Only count water and heat if we're in an active draw event with acceptable temperature
                    if outlet_temp >= water_temp_cutoff:
                        # Calculate water volume (L/min * min) = L
                        water_volume = water_draw * time_delta
                        
                        # Calculate heat energy (W * s) = J
                        heat_energy = heat_output * time_delta
                        
                        # Add to totals
                        total_water_volume_L += water_volume
                        total_heat_delivered_J += heat_energy
                        
                        # Update current event
                        current_event['water_volume_L'] += water_volume
                        current_event['heat_delivered_J'] += heat_energy
                        current_event['end_time'] = current_time
                        current_event['max_temp'] = max(current_event['max_temp'], outlet_temp)
                        current_event['min_temp'] = min(current_event['min_temp'], outlet_temp)
                        current_event['max_flow_rate'] = max(current_event['max_flow_rate'], water_draw)
                    else:
                        # Temperature dropped below cutoff during draw
                        # End the current event if we had one
                        if current_event and current_event['water_volume_L'] > 0:
                            # Calculate gallons and kWh for this event
                            current_event['water_volume_gal'] = current_event['water_volume_L'] * L_TO_GAL_RATIO
                            current_event['heat_delivered_kWh'] = current_event['heat_delivered_J'] * 2.77778e-7
                            draw_events.append(current_event)
                            
                            # Reset event tracking
                            is_draw_active = False
                            current_event = None
            
            # Check if water draw stops
            elif is_draw_active:
                # End of draw event
                # Save the current event if it collected any water
                if current_event and current_event['water_volume_L'] > 0:
                    # Calculate gallons and kWh for this event
                    current_event['water_volume_gal'] = current_event['water_volume_L'] * L_TO_GAL_RATIO
                    current_event['heat_delivered_kWh'] = current_event['heat_delivered_J'] * 2.77778e-7
                    draw_events.append(current_event)
                
                # Reset event tracking
                is_draw_active = False
                current_event = None
        
        # Handle case where the last draw event is still active at the end of the data
        if is_draw_active and current_event and current_event['water_volume_L'] > 0:
            # Calculate gallons and kWh for this event
            current_event['water_volume_gal'] = current_event['water_volume_L'] * L_TO_GAL_RATIO
            current_event['heat_delivered_kWh'] = current_event['heat_delivered_J'] * 2.77778e-7
            draw_events.append(current_event)
        
        # Convert liters to gallons (1 liter = 0.264172 gallons)
        total_water_volume_gal = total_water_volume_L * L_TO_GAL_RATIO
        
        # Convert joules to kWh for easier reporting (1 J = 2.77778e-7 kWh)
        total_heat_delivered_kWh = total_heat_delivered_J * 2.77778e-7
        total_energy_used_kwh = sum(df[energy_used])/60/1000
        
        output[file] = {
            'total_water_volume_L': total_water_volume_L,
            'total_water_volume_gal': total_water_volume_gal,
            'total_energy_used_kwh': total_energy_used_kwh,
            'total_heat_delivered_J': total_heat_delivered_J,
            'total_heat_delivered_kWh': total_heat_delivered_kWh,
            'draw_events': draw_events,
            'num_draw_events': len(draw_events)
        }
    return output
    

# Predefined lookup arrays for water properties at 1 atm.
_TEMPS = np.array([0, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
_DENSITIES = np.array([999.8, 1000.0, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4], dtype=float)
_THERMALEXPANSIONS = np.array([-1.0e-4, 0.0, 1.1e-4, 2.1e-4, 2.6e-4, 3.1e-4, 3.6e-4, 4.1e-4, 4.7e-4, 5.3e-4, 5.9e-4, 6.6e-4], dtype=float)

def lookup_water_properties(T):
    """
    Returns interpolated water properties for a given temperature T (in °C).
    The returned properties are based on approximate data for pure water at 1 atm.
    
    Properties returned:
      - density: in kg/m³
      - specific_weight: in N/m³ (calculated as density * 9.81)
      - thermal_expansion: volumetric thermal expansion coefficient in 1/°C
    
    Parameters:
      T (float): Temperature in °C (must be within 0 to 100)
      
    Raises:
      ValueError: If the temperature is outside the 0 to 100°C range.
    """
    if T < _TEMPS[0] or T > _TEMPS[-1]:
        raise ValueError("Temperature out of range (must be between 0 and 100 °C)")
    
    density = np.interp(T, _TEMPS, _DENSITIES)
    thermal_expansion = np.interp(T, _TEMPS, _THERMALEXPANSIONS)
    specific_weight = density * 9.81
    
    return {
        "temperature": T,
        "density": density,
        "specific_weight": specific_weight,
        "thermal_expansion": thermal_expansion,
    }

def calculate_net_water_temp(df):
    water_temp_column = ["Hot Water Average Temperature (C)"]
    valid_columns = [col for col in water_temp_column if col in df.columns]

    if not valid_columns:
        return 0

    # Compute the average of the first and last row for all available columns
    average_start_temp = df[valid_columns].iloc[0].mean()
    average_end_temp = df[valid_columns].iloc[-1].mean()

    return average_end_temp - average_start_temp

def create_temperature_plots(dfs, uef_values, patterns=['T_WH', 'T_PCM']):
    """Create separate temperature plots for each pattern group in each file,
    with upper and lower heating element overlays on the water temperature curves."""
    all_figs = []
    figure_metadata = []  # List to store metadata separately
    water_temp_cutoff = 43.3333  # 110 F 15 deg delta from 125 F for UEF test

    for i, (file, df) in enumerate(dfs.items()):
        # Get UEF value for this file
        uef = uef_values[i]

        # Find matching columns for this file
        column_groups = find_matching_columns(df, patterns)

        # Extract additional parameters for title
        pcm_mass_col = "PCM Mass (kg)"
        pcm_h_col_pattern = re.compile(r"Water Tank PCM\d+ h \(W/m\^2K\)")
        pcm_sa_col_pattern = re.compile(r"Water Tank PCM\d+ sa_ratio") 
        if pcm_mass_col not in df.columns:
            pcm_mass = 0.0
        else:
            pcm_mass = df[pcm_mass_col].iloc[-1] # in kg
            
        matched_h_cols = next((col for col in df.columns if pcm_h_col_pattern.fullmatch(col)), None) 
        match_sa_col = next((col for col in df.columns if pcm_sa_col_pattern.fullmatch(col)), None) 
        
        if matched_h_cols is not None:
            pcm_h = df[matched_h_cols].iloc[-1]

        if match_sa_col is not None:
            pcm_sa = df[match_sa_col].iloc[-1]

        water_volume_col = "Water Volume (L)"
        if water_volume_col not in df.columns:
            water_volume_gal = 45.0
        else:
            water_volume_gal = df[water_volume_col].iloc[-1] * L_TO_GAL_RATIO

        # Create a plot for each pattern that has matching columns
        for pattern, columns in column_groups.items():
            if not columns:  # Skip if no matching columns found
                continue

            # Create a new figure
            fig = go.Figure()

            # Determine the overall temperature range across all columns for this pattern
            temp_min = float('inf')
            temp_max = float('-inf')
            for col in columns:
                temp_min = min(temp_min, df[col].min())
                temp_max = max(temp_max, df[col].max())
            temp_range = temp_max - temp_min
            temp_padding = temp_range * 0.1

            # Add temperature traces for each matching column
            for col in columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Time'],
                        y=df[col],
                        mode='lines',
                        name=col
                    )
                )

            # Add heating element overlays if the "Water Heating Mode" column is present
            if 'Water Heating Mode' in df.columns:
                upper_regions = []
                lower_regions = []
                mode_data = df['Water Heating Mode']
                time_data = df['Time']

                # Identify regions when the upper element is on
                in_upper_segment = False
                for j in range(len(mode_data)):
                    mode_str = str(mode_data.iloc[j])
                    if 'Upper On' in mode_str and not in_upper_segment:
                        upper_segment_start = j
                        in_upper_segment = True
                    elif 'Upper On' not in mode_str and in_upper_segment:
                        upper_segment_end = j
                        in_upper_segment = False
                        start_time = time_data.iloc[upper_segment_start]
                        end_time = time_data.iloc[upper_segment_end]
                        upper_regions.append((start_time, end_time))
                if in_upper_segment:
                    upper_segment_end = len(mode_data) - 1
                    start_time = time_data.iloc[upper_segment_start]
                    end_time = time_data.iloc[upper_segment_end]
                    upper_regions.append((start_time, end_time))

                # Identify regions when the lower element is on
                in_lower_segment = False
                for j in range(len(mode_data)):
                    mode_str = str(mode_data.iloc[j])
                    if 'Lower On' in mode_str and not in_lower_segment:
                        lower_segment_start = j
                        in_lower_segment = True
                    elif 'Lower On' not in mode_str and in_lower_segment:
                        lower_segment_end = j
                        in_lower_segment = False
                        start_time = time_data.iloc[lower_segment_start]
                        end_time = time_data.iloc[lower_segment_end]
                        lower_regions.append((start_time, end_time))
                if in_lower_segment:
                    lower_segment_end = len(mode_data) - 1
                    start_time = time_data.iloc[lower_segment_start]
                    end_time = time_data.iloc[lower_segment_end]
                    lower_regions.append((start_time, end_time))

                # Add blue overlay for upper element on regions
                for j, (start_time, end_time) in enumerate(upper_regions):
                    fig.add_trace(
                        go.Scatter(
                            x=[start_time, end_time, end_time, start_time, start_time],
                            y=[temp_min - temp_padding, temp_min - temp_padding,
                               temp_max + temp_padding, temp_max + temp_padding,
                               temp_min - temp_padding],
                            fill="toself",
                            fillcolor="rgba(255, 0, 0, 0.3)",
                            line=dict(width=0),
                            mode="none",
                            name="Upper Element On" if j == 0 else "",
                            showlegend=True if j == 0 else False,
                            legendgroup="upper_elements",
                            hoverinfo="skip"
                        )
                    )
                # Add red overlay for lower element on regions
                for j, (start_time, end_time) in enumerate(lower_regions):
                    fig.add_trace(
                        go.Scatter(
                            x=[start_time, end_time, end_time, start_time, start_time],
                            y=[temp_min - temp_padding, temp_min - temp_padding,
                               temp_max + temp_padding, temp_max + temp_padding,
                               temp_min - temp_padding],
                            fill="toself",
                            fillcolor="rgba(0, 0, 255, 0.3)",
                            line=dict(width=0),
                            mode="none",
                            name="Lower Element On" if j == 0 else "",
                            showlegend=True if j == 0 else False,
                            legendgroup="lower_elements",
                            hoverinfo="skip"
                        )
                    )


            # Add hot water cutoff line
            fig.add_trace(
                go.Scatter(
                    x=df['Time'], y=[water_temp_cutoff for _ in df['Time']],  # invisible point 
                    mode='lines',
                    line=dict(color='red', width=1),
                    name='Hot Water Cutoff Temp (110°F / 43.33°C)',
                    showlegend=True
                )
            )
            # Update the layout with UEF and other file-specific info in the title
            fig.update_layout(
                title=f'{pattern} Temperatures - {file}<br>'
                      f'UEF: {uef:.3f} | PCM h: {pcm_h} W/m^2K | PCM SA Ratio: {pcm_sa} | PCM Mass: {pcm_mass:.3f} kg | '
                      f'Water Volume: {water_volume_gal:.1f} gal',
                xaxis_title='Time',
                yaxis_title='Temperature',
                height=600,
                showlegend=True
            )
            fig.update_yaxes(range=[temp_min - temp_padding, temp_max + temp_padding])

            # Store metadata for later reference
            figure_metadata.append({
                'pattern': pattern,
                'file': file,
                'type': 'temperature_pattern'
            })
            


            all_figs.append(fig)

    return all_figs, figure_metadata

def create_capacitance_plots(dfs, uef_values):
    """Create separate capacitance plots for each file,
    with upper and lower heating element overlays on the capacitance curves."""
    all_figs = []
    figure_metadata = []  # List to store metadata separately
    
    # Define the pattern to search for capacitance columns
    capacitance_pattern = "Capacitance"
    # Find matching columns for this file

        
        # Find all capacitance columns for this file
    # Extract additional parameters for title
    
    for i, (file, df) in enumerate(dfs.items()):
        # Get UEF value for this file
        uef = uef_values[i]
        capacitance_columns = [col for col in df.columns if capacitance_pattern in col]
        
        if not capacitance_columns:  # Skip if no matching columns found
            continue
            

        pcm_mass_col = "PCM Mass (kg)"
        pcm_h_col_pattern = re.compile(r"Water Tank PCM\d+ h \(W/m\^2K\)")
        pcm_sa_col_pattern = re.compile(r"Water Tank PCM\d+ sa_ratio") 
        if pcm_mass_col not in df.columns:
            pcm_mass = 0.0
        else:
            pcm_mass = df[pcm_mass_col].iloc[-1] # in kg
            
        matched_h_cols = next((col for col in df.columns if pcm_h_col_pattern.fullmatch(col)), None) 
        match_sa_col = next((col for col in df.columns if pcm_sa_col_pattern.fullmatch(col)), None) 
        
        if matched_h_cols is not None:
            pcm_h = df[matched_h_cols].iloc[-1]

        if match_sa_col is not None:
            pcm_sa = df[match_sa_col].iloc[-1]
            
        # Create a new figure
        fig = go.Figure()
        
        # Extract additional parameters for title
        pcm_mass_col = "PCM Mass (kg)"
        if pcm_mass_col not in df.columns:
            pcm_mass = 0.0
        else:
            pcm_mass = df[pcm_mass_col].iloc[-1]  # in kg
            
        water_volume_col = "Water Volume (L)"
        if water_volume_col not in df.columns:
            water_volume_gal = 45.0
        else:
            water_volume_gal = df[water_volume_col].iloc[-1] * L_TO_GAL_RATIO
            
        # Determine the overall capacitance range across all columns
        cap_min = float('inf')
        cap_max = float('-inf')
        for col in capacitance_columns:
            cap_min = min(cap_min, df[col].min())
            cap_max = max(cap_max, df[col].max())
        cap_range = cap_max - cap_min
        cap_padding = cap_range * 0.1
        
        # Add capacitance traces for each matching column
        for col in capacitance_columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=df[col],
                    mode='lines',
                    name=col
                )
            )
            
        # Add heating element overlays if the "Water Heating Mode" column is present
        if 'Water Heating Mode' in df.columns:
            upper_regions = []
            lower_regions = []
            mode_data = df['Water Heating Mode']
            time_data = df['Time']
            
            # Identify regions when the upper element is on
            in_upper_segment = False
            for j in range(len(mode_data)):
                mode_str = str(mode_data.iloc[j])
                if 'Upper On' in mode_str and not in_upper_segment:
                    upper_segment_start = j
                    in_upper_segment = True
                elif 'Upper On' not in mode_str and in_upper_segment:
                    upper_segment_end = j
                    in_upper_segment = False
                    start_time = time_data.iloc[upper_segment_start]
                    end_time = time_data.iloc[upper_segment_end]
                    upper_regions.append((start_time, end_time))
            if in_upper_segment:
                upper_segment_end = len(mode_data) - 1
                start_time = time_data.iloc[upper_segment_start]
                end_time = time_data.iloc[upper_segment_end]
                upper_regions.append((start_time, end_time))
                
            # Identify regions when the lower element is on
            in_lower_segment = False
            for j in range(len(mode_data)):
                mode_str = str(mode_data.iloc[j])
                if 'Lower On' in mode_str and not in_lower_segment:
                    lower_segment_start = j
                    in_lower_segment = True
                elif 'Lower On' not in mode_str and in_lower_segment:
                    lower_segment_end = j
                    in_lower_segment = False
                    start_time = time_data.iloc[lower_segment_start]
                    end_time = time_data.iloc[lower_segment_end]
                    lower_regions.append((start_time, end_time))
            if in_lower_segment:
                lower_segment_end = len(mode_data) - 1
                start_time = time_data.iloc[lower_segment_start]
                end_time = time_data.iloc[lower_segment_end]
                lower_regions.append((start_time, end_time))
                
            # Add red overlay for upper element on regions
            for j, (start_time, end_time) in enumerate(upper_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[cap_min - cap_padding, cap_min - cap_padding,
                           cap_max + cap_padding, cap_max + cap_padding,
                           cap_min - cap_padding],
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.3)",
                        line=dict(width=0),
                        mode="none",
                        name="Upper Element On" if j == 0 else "",
                        showlegend=True if j == 0 else False,
                        legendgroup="upper_elements",
                        hoverinfo="skip"
                    )
                )
            # Add blue overlay for lower element on regions
            for j, (start_time, end_time) in enumerate(lower_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[cap_min - cap_padding, cap_min - cap_padding,
                           cap_max + cap_padding, cap_max + cap_padding,
                           cap_min - cap_padding],
                        fill="toself",
                        fillcolor="rgba(0, 0, 255, 0.3)",
                        line=dict(width=0),
                        mode="none",
                        name="Lower Element On" if j == 0 else "",
                        showlegend=True if j == 0 else False,
                        legendgroup="lower_elements",
                        hoverinfo="skip"
                    )
                )
                
        # Update the layout with UEF and other file-specific info in the title
        fig.update_layout(
            title=f'PCM Temperatures - {file}<br>'
                      f'UEF: {uef:.3f} | PCM h: {pcm_h} W/m^2K | PCM SA Ratio: {pcm_sa} | PCM Mass: {pcm_mass:.3f} kg | '
                      f'Water Volume: {water_volume_gal:.1f} gal',
            xaxis_title='Time',
            yaxis_title='Capacitance (J/K)',
            height=600,
            showlegend=True
        )
        fig.update_yaxes(range=[cap_min - cap_padding, cap_max + cap_padding])
        
        # Store metadata for later reference
        figure_metadata.append({
            'pattern': 'PCM_Capacitance',
            'file': file,
            'type': 'capacitance'
        })
        
        all_figs.append(fig)
        
    return all_figs, figure_metadata

def calculate_net_water_energy(volume, temperature, temperature_difference):
    
    water_properties = lookup_water_properties(temperature)
    density = water_properties["density"]
    # volume is in Liters
    water_weight = density * volume / 1e3 # in kg
    
    
    return water_weight * temperature_difference * 4184 # J

def calculate_uef(dfs):
    # calculate UEF of the water tank
    uef_values = []      # make sure in W*min                            
    
    for df in dfs.values():
        uef = calculate_single_uef(df)
        uef_values.append(uef)
    
    return uef_values

def calculate_single_uef(df):
    # calculate UEF of the water tank
    Q_cons = (
        df["Water Heating Electric Power (kW)"].sum() * 1000
    )  # not sure if this is the correct term that I should be pulling
    Q_load = df[
        "Hot Water Delivered (W)"
    ].sum()  # not sure if this is the correct term that I should be pulling
    
    PCM_Q_Heat_to_Water= calculate_net_PCM_heat(df)     # make sure in W*min
    PCM_net_enthalpy = calculate_net_PCM_enthalpy(df)                               
    PCM_net_heat_loss = PCM_net_enthalpy / 60           # make sure in W*min
    water_net_temp_delta = calculate_net_water_temp(df)
    
    water_volume_col = "Water Volume (L)"
    if water_volume_col not in df.columns:
        Q_cons_total = Q_cons - PCM_net_heat_loss          # make sure in W*min                            
        UEF = Q_load / Q_cons_total
    else:
        water_volume = df[water_volume_col].iloc[-1]
        water_net_energy = calculate_net_water_energy(water_volume, df['Hot Water Average Temperature (C)'].iloc[-1], water_net_temp_delta) / 60 # make sure in W*min
        Q_cons_total = Q_cons - PCM_net_heat_loss - water_net_energy          # make sure in W*min                            
        UEF = Q_load / Q_cons_total
    
    return UEF

def calculate_net_PCM_heat(df):

    # Dynamically find all PCM enthalpy columns.
    pcm_column = 'Total PCM Heat Injected (W)'
    
    if pcm_column not in df.columns:
        return 0
    
    # Sum the PCM enthalpy columns row-wise.

    net_PCM_to_water_heat_Transfer = df[pcm_column].sum()
    
    return net_PCM_to_water_heat_Transfer

def calculate_net_PCM_enthalpy(df):
    # Dynamically find all PCM enthalpy columns.
    pcm_column = 'Total PCM Enthalpy (J)'
    
    if pcm_column not in df.columns:
        return 0
    
    # Sum the PCM enthalpy columns row-wise.

    net_PCM_enthalpy = df[pcm_column].iloc[-1] - df[pcm_column].iloc[0]
    
    return net_PCM_enthalpy


def plot_individual_energy_values(dfs):
    """
    Create a separate plot for each individual H_ value (H_WH1, H_WH2, etc.) comparing across files.
    
    Args:
        dfs (dict): Dictionary of DataFrames with filenames as keys
        
    Returns:
        tuple: (list of figures, list of metadata)
    """
    all_figures = []
    all_metadata = []
    
    # First, find all unique H_ columns across all files
    all_h_columns = set()
    for df in dfs.values():
        all_h_columns.update([col for col in df.columns if col.startswith('T_PCM')])
    
    # Sort columns to ensure consistent order
    all_h_columns = sorted(all_h_columns)
    
    # Create a separate plot for each H_ column
    for column in all_h_columns:
        # Create figure
        fig = go.Figure()
        
        # Add a trace for each file that has this column
        for file, df in dfs.items():
            if column in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Time'],
                        y=df[column],
                        mode='lines',
                        name=f'{file}'
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f'{column} Comparison Across Files',
            xaxis_title='Time',
            yaxis_title='Energy',
            height=600,
            showlegend=True,
            legend_title_text='Files'
        )
        
        # Store metadata
        metadata = {
            'type': 'individual_energy',
            'column': column
        }
        
        all_figures.append(fig)
        all_metadata.append(metadata)
    
    return all_figures, all_metadata

def create_heat_exchanger_plots(dfs):
    """Create plots with dual y-axes for temperature and power data,
    with the power y-axis scaled larger and grouped power > 0 overlays for each dataset.
    The third subplot displays heating element status overlays alongside temperature data."""
    
    water_temp_cutoff = 43.3333  # 110 F 15 deg delta from 125 F for UEF test
    
    # Create a figure with 3 subplot rows and 1 column
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=(
            'Hot Water Outlet Temperature and Heating Power',
            'Hot Water Average Temperature and Heating Power',
            'Hot Water Outlet Temperature and Heating Element Status'
        ),
        specs=[
            [{"secondary_y": True}], 
            [{"secondary_y": True}],
            [{"secondary_y": True}]
        ],
        vertical_spacing=0.13,
        row_heights=[0.33, 0.33, 0.34]  # Make bottom plot slightly larger
    )
    
    # Colors to differentiate between files
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # For tracking min/max values to set axis ranges
    temp_min, temp_max = float('inf'), float('-inf')
    power_min, power_max = float('inf'), float('-inf')
    
    # Process datasets and add traces
    for i, (file, df) in enumerate(dfs.items()):
        color = colors[i % len(colors)]
        legendgroup = f"group_{file}"  # Create a unique legend group for this file
        
        # Find regions where power > 0 for this dataset
        power_regions = []
        if 'Water Heating Delivered (W)' in df.columns:
            power_data = df['Water Heating Delivered (W)']
            time_data = df['Time']
            mask = power_data > 0
            
            if mask.any():  # Only proceed if there are power > 0 values
                in_segment = False
                for j in range(len(mask)):
                    if mask.iloc[j] and not in_segment:  # Start of a segment
                        segment_start = j
                        in_segment = True
                    elif not mask.iloc[j] and in_segment:  # End of a segment
                        segment_end = j - 1
                        in_segment = False
                        start_time = time_data.iloc[segment_start]
                        end_time = time_data.iloc[segment_end]
                        power_regions.append((start_time, end_time))
                
                if in_segment:
                    segment_end = len(mask) - 1
                    start_time = time_data.iloc[segment_start]
                    end_time = time_data.iloc[segment_end]
                    power_regions.append((start_time, end_time))
        
        # 1. First subplot: Outlet Temperature and Power
        
        # Add Outlet Temperature trace
        if 'Hot Water Outlet Temperature (C)' in df.columns:
            temp_data = df['Hot Water Outlet Temperature (C)']
            temp_min = min(temp_min, temp_data.min())
            temp_max = max(temp_max, temp_data.max()*1.1)
            
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=temp_data,
                    mode='lines',
                    name=f"{file} - Outlet Temp",
                    line=dict(color=color),
                    legendgroup=legendgroup
                ),
                row=1, col=1, secondary_y=False
            )
            
            # Add hot water cutoff line
            fig.add_trace(
                go.Scatter(
                    x=df['Time'], y=[water_temp_cutoff for _ in df['Time']],  # invisible point 
                    mode='lines',
                    line=dict(color='red', width=1),
                    name='Hot Water Cutoff Temp (110°F / 43.33°C)',
                    showlegend=True
                ),
                row=1, col=1
            )

        
        # Add Heating Power trace with dashed line for first subplot
        if 'Water Heating Delivered (W)' in df.columns:
            power_min = min(power_min, power_data.min())
            power_max = max(power_max, power_data.max())
            
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=power_data,
                    mode='lines',
                    name=f"{file} - Power",
                    line=dict(color=color, dash='dot'),
                    legendgroup=legendgroup
                ),
                row=1, col=1, secondary_y=True
            )
            
            # Create region overlays for first subplot
            for j, (start_time, end_time) in enumerate(power_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[temp_min, temp_min, temp_max, temp_max, temp_min],
                        fill="toself",
                        fillcolor="rgba(255, 165, 0, 0.3)",
                        line=dict(width=0),
                        mode="none",
                        name=f"{file} - Power > 0 Region",
                        legendgroup=legendgroup,
                        showlegend=True if j == 0 else False,
                        hoverinfo="skip"
                    ),
                    row=1, col=1, secondary_y=False
                )
        
        # 2. Second subplot: Average Temperature and Power
        
        # Add Average Temperature trace
        if 'Hot Water Average Temperature (C)' in df.columns:
            avg_temp_data = df['Hot Water Average Temperature (C)']
            temp_min = min(temp_min, avg_temp_data.min())
            temp_max = max(temp_max, avg_temp_data.max()*1.1)
            
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=avg_temp_data,
                    mode='lines',
                    name=f"{file} - Avg Temp",
                    line=dict(color=color),
                    legendgroup=legendgroup,
                    showlegend=True
                ),
                row=2, col=1, secondary_y=False
            )
        
        # Add Heating Power trace for average data with dashed line, labeled as Avg Power
        if 'Water Heating Delivered (W)' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=power_data,
                    mode='lines',
                    name=f"{file} - Avg Power",
                    line=dict(color=color, dash='dot'),
                    legendgroup=legendgroup,
                    showlegend=True
                ),
                row=2, col=1, secondary_y=True
            )
            
            # Create region overlays for second subplot for Avg Power > 0
            for j, (start_time, end_time) in enumerate(power_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[temp_min, temp_min, temp_max, temp_max, temp_min],
                        fill="toself",
                        fillcolor="rgba(255, 165, 0, 0.3)",
                        line=dict(width=0),
                        mode="none",
                        name=f"{file} - Avg Power > 0 Region",
                        legendgroup=legendgroup,
                        showlegend=True if j == 0 else False,
                        hoverinfo="skip"
                    ),
                    row=2, col=1, secondary_y=False
                )
        
        # 3. Third subplot: Outlet Temperature and Water Heating Element Status with Overlays
        if 'Hot Water Outlet Temperature (C)' in df.columns:
            # Add the outlet temperature again in the third subplot
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=temp_data,
                    mode='lines',
                    name=f"{file} - Outlet Temp (Heater Mode)",
                    line=dict(color=color),
                    legendgroup=legendgroup,
                    showlegend=True
                ),
                row=3, col=1, secondary_y=False
            )
            # Add hot water cutoff line
            fig.add_trace(
                go.Scatter(
                    x=df['Time'], y=[water_temp_cutoff for _ in df['Time']],  # invisible point 
                    mode='lines',
                    line=dict(color='red', width=1),
                    name='Hot Water Cutoff Temp (110°F / 43.33°C)',
                    showlegend=True
                ),
                row=3, col=1
            )
        
        # Calculate the adjusted ranges for both axes
        temp_range = temp_max - temp_min
        temp_padding = temp_range * 0.1  # 10% padding
        
        power_range = power_max - power_min
        power_padding = power_range * 0.1  # 10% padding
        scaled_power_max = (power_max + power_padding) * 5
        scaled_power_min = power_min 
    
        # Add Water Heating Mode overlays
        if 'Water Heating Mode' in df.columns:
            # Process each mode value to identify regions where elements are on
            upper_regions = []
            lower_regions = []
            
            if len(df) > 0:
                mode_data = df['Water Heating Mode']
                time_data = df['Time']
                
                # Find regions where upper element is on
                in_upper_segment = False
                for j in range(len(mode_data)):
                    mode_str = str(mode_data.iloc[j])
                    upper_on = 'Upper On' in mode_str
                    
                    if upper_on and not in_upper_segment:  # Start of a segment
                        upper_segment_start = j
                        in_upper_segment = True
                    elif not upper_on and in_upper_segment:  # End of a segment
                        upper_segment_end = j
                        in_upper_segment = False
                        start_time = time_data.iloc[upper_segment_start]
                        end_time = time_data.iloc[upper_segment_end]
                        upper_regions.append((start_time, end_time))
                
                if in_upper_segment:  # If the last segment extends to the end
                    upper_segment_end = len(mode_data)
                    start_time = time_data.iloc[upper_segment_start]
                    end_time = time_data.iloc[upper_segment_end]
                    upper_regions.append((start_time, end_time))
                
                # Find regions where lower element is on
                in_lower_segment = False
                for j in range(len(mode_data)):
                    mode_str = str(mode_data.iloc[j])
                    lower_on = 'Lower On' in mode_str
                    
                    if lower_on and not in_lower_segment:  # Start of a segment
                        lower_segment_start = j
                        in_lower_segment = True
                    elif not lower_on and in_lower_segment:  # End of a segment
                        lower_segment_end = j
                        in_lower_segment = False
                        start_time = time_data.iloc[lower_segment_start]
                        end_time = time_data.iloc[lower_segment_end]
                        lower_regions.append((start_time, end_time))
                
                if in_lower_segment:  # If the last segment extends to the end
                    lower_segment_end = len(mode_data)-1
                    start_time = time_data.iloc[lower_segment_start]
                    end_time = time_data.iloc[lower_segment_end]
                    lower_regions.append((start_time, end_time))
            
            # Create overlays for upper element (blue)
            temp_padding = 0
            for j, (start_time, end_time) in enumerate(upper_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[temp_min - temp_padding, temp_min - temp_padding, 
                           temp_max + temp_padding, temp_max + temp_padding, 
                           temp_min - temp_padding],
                        fill="toself",
                        fillcolor="rgba(0, 0, 255, 0.3)",  # Blue with transparency
                        line=dict(width=0),
                        mode="none",
                        name=f"{file} - Upper Element On",
                        legendgroup=legendgroup,
                        showlegend=True if j == 0 else False,
                        hoverinfo="skip"
                    ),
                    row=3, col=1, secondary_y=False
                )
            
            # Create overlays for lower element (red)
            for j, (start_time, end_time) in enumerate(lower_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[temp_min - temp_padding, temp_min - temp_padding, 
                           temp_max + temp_padding, temp_max + temp_padding, 
                           temp_min - temp_padding],
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.3)",  # Red with transparency
                        line=dict(width=0),
                        mode="none",
                        name=f"{file} - Lower Element On",
                        legendgroup=legendgroup,
                        showlegend=True if j == 0 else False,
                        hoverinfo="skip"
                    ),
                    row=3, col=1, secondary_y=False
                )
    
    
    
    # Update layout and axes titles
    fig.update_layout(
        height=1400,
        showlegend=True,
        title_text="PCM Performance Analysis",
        legend=dict(
            groupclick="togglegroup"
        )
    )
    
    # Update y-axis titles and ranges for first subplot
    fig.update_yaxes(
        title_text="Outlet Temperature (°C)",
        range=[temp_min - temp_padding, temp_max + temp_padding],
        row=1, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text="Power (W)",
        range=[scaled_power_min, scaled_power_max],
        row=1, col=1, secondary_y=True
    )
    
    # Update y-axis titles and ranges for second subplot
    fig.update_yaxes(
        title_text="Average Temperature (°C)",
        range=[temp_min - temp_padding, temp_max + temp_padding],
        row=2, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text="Power (W)",
        range=[scaled_power_min, scaled_power_max],
        row=2, col=1, secondary_y=True
    )
    
    # Update y-axis titles and ranges for third subplot (Heating Element Status)
    fig.update_yaxes(
        title_text="Outlet Temperature (°C)",
        range=[temp_min - temp_padding, temp_max + temp_padding],
        row=3, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text="Heating Element Status",
        visible=False,  # Hide the secondary y-axis since we're using overlays
        row=3, col=1, secondary_y=True
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    return fig

def save_plots(figures, metadata, output_folder='plots'):
    """
    Save all plots in multiple formats.
    
    Args:
        figures: List of plotly figures
        metadata: List of metadata dictionaries corresponding to figures
        output_folder: Folder to save the plots in
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save each figure
    for fig, meta in zip(figures, metadata):
        # Generate base filename based on metadata
        if meta['type'] == 'temperature_pattern':
            base_name = f"{meta['pattern']}_{meta['file'].replace('.csv', '')}"
        else:
            base_name = "outlet_temperature_comparison"
        
        # Save in different formats
        # HTML (interactive)
        # fig.write_html(os.path.join(output_folder, f"{base_name}.html"))
        
        # Static images
        fig.write_image(os.path.join(output_folder, f"{base_name}.png"))
        # fig.write_image(os.path.join(output_folder, f"{base_name}.pdf"))

def find_energy_columns(df):
    """
    Find all columns that start with 'H_' pattern.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        
    Returns:
        list: List of column names matching the pattern
    """
    return [col for col in df.columns if col.startswith('H_')]

def calculate_energy_sums(dfs):
    """
    Calculate the sum of all energy columns (starting with 'H_') for each DataFrame.
    
    Args:
        dfs (dict): Dictionary of DataFrames with filenames as keys
        
    Returns:
        dict: Dictionary with filenames as keys and dictionaries of energy sums as values
    """
    energy_sums = {}
    
    for file, df in dfs.items():
        # Find all energy columns
        energy_cols = find_energy_columns(df)
        
        net_enthalpy = calculate_net_PCM_enthalpy(df)
        net_enthalpy = net_enthalpy/60 # conver to W*min
        
        # Calculate sums and store in nested dictionary
        sums = {
            'total_energy': sum(df[energy_cols].sum()) - net_enthalpy,  # Total sum across all H_ columns
            'column_sums': {col: df[col].sum() - net_enthalpy for col in energy_cols},  # Individual column sums
            'timestep_sums': df[energy_cols].sum(axis=1) - net_enthalpy # Sum at each timestep
        }
        
        
        energy_sums[file] = sums
    
    return energy_sums

def plot_energy_comparison(dfs, energy_sums):
    """
    Create plots comparing energy values across different files.
    
    Args:
        dfs (dict): Dictionary of DataFrames
        energy_sums (dict): Dictionary of energy sums from calculate_energy_sums
        
    Returns:
        tuple: (figure, metadata)
    """
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Cumulative Energy by File', 
                                     'Energy Rate over Time'))
    
    # Plot cumulative energy for each file
    for file, sums in energy_sums.items():
        timestep_sums = sums['timestep_sums']
        cumulative_energy = timestep_sums.cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=dfs[file]['Time'],
                y=cumulative_energy,
                name=f'{file} (Cumulative)',
                mode='lines'
            ),
            row=1, col=1
        )
        
        # Plot energy rate over time
        fig.add_trace(
            go.Scatter(
                x=dfs[file]['Time'],
                y=timestep_sums,
                name=f'{file} (Rate)',
                mode='lines'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=900,
        title='Energy Analysis Comparison',
        showlegend=True
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Cumulative Energy", row=1, col=1)
    fig.update_yaxes(title_text="Energy Rate", row=2, col=1)
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    return fig, {'type': 'energy_comparison'}

def plot_pcm_enthalpies(df):
    """
    Plot cp and Enthalpy vs Temperature on the same plot with dual y-axes.
    
    Args:
        df (numpy.ndarray): Array containing columns:
                            'Temp (C)', 'cp (J/g-C)', and 'Enthalpy (J/kg)'
    
    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure object.
    """
    
    fig = go.Figure()

    # Plot cp vs Temperature on the primary y-axis
    fig.add_trace(
        go.Scatter(
            x=df[:,0],  # Temperature
            y=df[:,1],  # Specific Heat Capacity
            name='cp (J/g-C)',
            mode='lines+markers',
            yaxis='y1'  # Attach to primary y-axis
        )
    )

    # Plot Enthalpy vs Temperature on the secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df[:,0],  # Temperature
            y=df[:,2],  # Enthalpy
            name='Enthalpy (J/kg)',
            mode='lines+markers',
            yaxis='y2'  # Attach to secondary y-axis
        )
    )

    # Update layout with dual y-axis
    fig.update_layout(
        title='PCM cp and Enthalpy vs Temperature',
        xaxis=dict(title='Temperature (C)'),
        yaxis=dict(
            title='cp (J/g-C)', 
            showgrid=False
        ),
        yaxis2=dict(
            title='Enthalpy (J/kg)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(x=0.05, y=0.95),
        height=600
    )
    
    return fig



# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"

graphing_results_folder = "../OCHRE_output/results/"

# Example usage:
if __name__ == "__main__":
    # Load data
    _start_time = time.perf_counter()
    _start_time_plot_results = time.perf_counter()
    print(os.getcwd())
    dfs  = load_data(results_folder=graphing_results_folder)
    uef = calculate_uef(dfs)
    plot = create_heat_exchanger_plots(dfs)
    enthalpy_plot = plot_pcm_enthalpies(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "ochre", "Models", "cp_h-T_data.csv"), delimiter=",", skiprows=1))
    capacitance_plots, _ = create_capacitance_plots(dfs, uef)
    enthalpy_plot.show()
    temp_plots,_= create_temperature_plots(dfs, uef_values=uef, patterns=['T_WH', 'T_PCM'])
    plot.show()
    for temp_plot, capacitance_plot in zip(temp_plots, capacitance_plots):
        temp_plot.show()
        capacitance_plot.show()
    
    # Draw data summary
    output = calculate_hot_water_delivered(dfs)
    plot_draw_event_summary(output)
    plot_draw_events(output)
    
    print(f"{BOLD}{GREEN}Plots created in {time.perf_counter() - _start_time_plot_results:.2f} seconds{RESET}")
    
    _end_time = time.perf_counter()
    total_time = _end_time - _start_time
    print(f"\n{BOLD}{RED}Total execution time: {total_time:.2f} seconds{RESET}")
    
    # # Combine all figures and metadata
    # all_figures = pattern_figures + [outlet_temp_fig]
    # all_metadata = pattern_metadata + [outlet_metadata]
    
    # # Save all plots
    # # save_plots(all_figures, all_metadata)
    
    # # Show all figures
    # for fig in all_figures:
    #     fig.show()
    
    # # Print UEF values
    # for file, value in uef_values.items():
    #     print(f"UEF for {file}:\t {value:.3f}")