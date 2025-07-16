
"""
Enhanced Real Data Highway Simulation

This version improves upon the basic simulation by:
1. Better scaling of vehicle counts to match real traffic patterns
2. Improved traffic flow algorithms to prevent unrealistic slowdowns
3. More accurate vehicle spawning based on real 5-minute interval data
4. Enhanced visualization and statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import time
import os
import platform
from datetime import datetime, timedelta
from scipy.stats import truncnorm

class Vehicle:
    """Represents a vehicle in the simulation with enhanced properties"""
    def __init__(self, vehicle_id, initial_speed, spawn_time, spawn_interval):
        self.id = vehicle_id
        self.position = 0.0  # Position on highway in meters
        self.speed = initial_speed  # Speed in m/s
        self.initial_speed = initial_speed
        self.target_speed = initial_speed  # Target speed for this vehicle
        self.spawn_time = spawn_time
        self.spawn_interval = spawn_interval  # 5-minute interval index (0-287)
        self.finished = False
        self.finish_time = None
        self.travel_time = 0
        self.average_speed = 0
        self.lane = np.random.randint(0, 3)  # Simulate 3 lanes (0, 1, 2)

class EnhancedRealDataSimulation:
    """Enhanced highway traffic simulation using real traffic data"""
    
    def __init__(self, traffic_data_file, highway_length_km=84, simulation_hours=24, scaling_factor=0.4):
        self.highway_length = highway_length_km * 1000  # Convert to meters
        self.simulation_hours = simulation_hours
        self.simulation_time = 0  # Current simulation time in seconds
        self.vehicles = []  # Active vehicles on highway
        self.finished_vehicles = []  # Completed vehicles
        self.vehicle_counter = 0
        self.scaling_factor = scaling_factor  # Scale down vehicle counts for computational efficiency
        
        # Load real traffic data
        self.load_traffic_data(traffic_data_file)
        
        # Simulation parameters
        self.time_step = 1.0  # Time step in seconds
        self.interval_duration = 300  # 5 minutes = 300 seconds
        self.min_spawn_gap = 0.5  # Minimum seconds between spawns
        self.last_spawn_time = -self.min_spawn_gap
        
        # Traffic flow parameters
        self.min_safe_distance = 10.0  # Minimum safe distance in meters
        self.reaction_time = 1.5  # Driver reaction time in seconds
        self.comfort_deceleration = 2.0  # Comfortable deceleration in m/s2
        
        # Statistics tracking by 5-minute intervals
        self.interval_stats = {i: {'speeds': [], 'count': 0, 'spawn_attempts': 0} 
                              for i in range(288)}  # 24 hours * 12 intervals per hour
        
        # Pre-population flag
        self.is_pre_populated = False
        
    def load_traffic_data(self, data_file):
        """Load and process real traffic data with enhanced preprocessing"""
        print(f"Loading traffic data from {data_file}...")
        
        # Load the CSV data
        df = pd.read_csv(data_file)
        
        # Extract the first etag_pair_id data
        first_pair = df['etag_pair_id'].iloc[0]
        pair_data = df[df['etag_pair_id'] == first_pair].copy()
        
        # Create time-based indexing
        pair_data['time_obj'] = pd.to_datetime(pair_data['time'], format='%H:%M:%S')
        pair_data['interval_index'] = (pair_data['time_obj'].dt.hour * 12 + 
                                      pair_data['time_obj'].dt.minute // 5)
        
        # Store data with smoothing to reduce extreme values
        self.mean_speeds = {}
        self.vehicle_counts = {}
        self.scaled_counts = {}
        
        speeds = []
        counts = []
        
        for _, row in pair_data.iterrows():
            interval_idx = int(row['interval_index'])
            speed = max(30.0, min(120.0, row['mean_speed']))  # Clamp speed between 30-120 km/h
            count = max(1.0, row['vehicle_count'])  # Ensure minimum count
            
            self.mean_speeds[interval_idx] = speed
            self.vehicle_counts[interval_idx] = count
            self.scaled_counts[interval_idx] = count * self.scaling_factor
            
            speeds.append(speed)
            counts.append(count)
        
        # Apply smoothing to reduce sudden changes
        # self.apply_smoothing()
        
        print(f"Loaded data for {len(self.mean_speeds)} 5-minute intervals")
        print(f"Speed range: {min(speeds):.1f} - {max(speeds):.1f} km/h")
        print(f"Vehicle count range: {min(counts):.1f} - {max(counts):.1f} per 5min")
        print(f"Scaled count range: {min(self.scaled_counts.values()):.1f} - {max(self.scaled_counts.values()):.1f} per 5min")
        print(f"Scaling factor: {self.scaling_factor}")
        print("-" * 50)
    
    def apply_smoothing(self, window_size=3):
        """Apply moving average smoothing to reduce extreme variations"""
        # Smooth speeds
        smoothed_speeds = {}
        smoothed_counts = {}
        
        for i in range(288):
            # Get neighboring values for smoothing
            neighbors_speeds = []
            neighbors_counts = []
            
            for j in range(max(0, i - window_size//2), min(288, i + window_size//2 + 1)):
                if j in self.mean_speeds:
                    neighbors_speeds.append(self.mean_speeds[j])
                    neighbors_counts.append(self.scaled_counts[j])
            
            if neighbors_speeds:
                smoothed_speeds[i] = np.mean(neighbors_speeds)
                smoothed_counts[i] = np.mean(neighbors_counts)
            else:
                smoothed_speeds[i] = self.mean_speeds.get(i, 100.0)
                smoothed_counts[i] = self.scaled_counts.get(i, 50.0)
        
        # Update with smoothed values
        self.mean_speeds.update(smoothed_speeds)
        self.scaled_counts.update(smoothed_counts)
    
    def get_current_interval(self):
        """Get current 5-minute interval index (0-287)"""
        total_minutes = (self.simulation_time // 60) % (24 * 60)
        return int(total_minutes // 5)
    
    def get_intelligent_safe_distance(self, speed_ms, leading_speed_ms=None):
        """Calculate intelligent safe following distance using car-following model"""
        if leading_speed_ms is None:
            leading_speed_ms = speed_ms
        
        # Intelligent Driver Model (IDM) safe distance
        min_distance = self.min_safe_distance
        time_headway = self.reaction_time * speed_ms
        speed_diff = speed_ms - leading_speed_ms
        
        # Calculate desired gap
        if speed_diff > 0:  # Approaching slower vehicle
            interaction_term = (speed_diff * speed_ms) / (2 * np.sqrt(self.comfort_deceleration * 3.0))
        else:
            interaction_term = 0
        
        safe_distance = min_distance + time_headway + interaction_term
        return max(min_distance, safe_distance)
    
    def generate_vehicle_speed(self, interval_idx):
        hour = int(self.simulation_time // 3600) % 24
        if not hasattr(self, '_hourly_speed_table'):
            self._hourly_speed_table = pd.read_csv('截尾常態_每小時統計_五倍標準誤.csv')
        row = self._hourly_speed_table[self._hourly_speed_table['hour'] == hour].iloc[0]
        mean_speed_kmh = row['mean']
        std_error = row['std_error']
        lower = row['lower_bound']
        upper = row['upper_bound']
        speed_kmh = np.random.normal(loc=mean_speed_kmh, scale=std_error)
        speed_kmh = max(lower, min(upper, speed_kmh))
        return speed_kmh / 3.6
    
    def get_target_spawn_rate(self, interval_idx):
        """Get target vehicle spawn rate for current interval"""
        if interval_idx in self.scaled_counts:
            vehicles_per_5min = self.scaled_counts[interval_idx]
        else:
            vehicles_per_5min = 50.0  # Default
        
        # Convert to vehicles per second
        vehicles_per_second = vehicles_per_5min / 300.0
        return vehicles_per_second
    
    def can_spawn_vehicle(self, new_speed):
        """Check if a new vehicle can be spawned safely using lane-based spawning"""
        if not self.vehicles:
            return True
        
        # Find vehicles near the start of the highway
        nearby_vehicles = [v for v in self.vehicles if v.position < 300]  # Within 300m of start
        
        if not nearby_vehicles:
            return True
        
        # Check each lane separately for available spawn space
        for lane in range(3):  # 3 lanes (0, 1, 2)
            lane_vehicles = [v for v in nearby_vehicles if v.lane == lane]
            
            if not lane_vehicles:
                return True  # Empty lane available for spawning
            
            # Check minimum gap in this lane
            closest_in_lane = min(lane_vehicles, key=lambda v: v.position)
            min_spawn_distance = 25.0  # Reduced minimum gap for spawning
            
            if closest_in_lane.position >= min_spawn_distance:
                return True  # Sufficient gap in this lane
        
        return False  # No suitable lane found
    
    def should_spawn_vehicle(self, interval_idx):
        """Enhanced vehicle spawning logic with lane-based improvements"""
        if self.simulation_time - self.last_spawn_time < self.min_spawn_gap:
            return False

        # Track spawn attempts for statistics
        self.interval_stats[interval_idx]['spawn_attempts'] += 1

        # Get target spawn rate and add some randomness
        spawn_rate = self.get_target_spawn_rate(interval_idx)
        base_probability = spawn_rate * self.time_step

        # Adjust based on traffic density
        traffic_density = len(self.vehicles) / (self.highway_length / 1000)  # vehicles per km
        if traffic_density > 80:  # Very high density
            spawn_probability = base_probability * 0.85
        elif traffic_density > 50:  # High density
            spawn_probability = base_probability * 0.95
        elif traffic_density > 30:  # Medium density
            spawn_probability = base_probability * 1.0
        else:  # Low density
            spawn_probability = base_probability * 1.15

        return np.random.random() < min(spawn_probability, 0.9)  # raised cap from 0.8 → 0.9
    
    def update_vehicle_speeds_intelligent(self):
        """Enhanced speed update using lane-aware intelligent car-following model"""
        if len(self.vehicles) <= 1:
            return
        
        # Sort vehicles by position (front to back)
        self.vehicles.sort(key=lambda v: v.position, reverse=True)
        
        # Update speeds using lane-aware intelligent driver model
        for i, vehicle in enumerate(self.vehicles):
            # Find the leading vehicle in the same lane
            leading_vehicle = None
            min_distance = float('inf')
            
            for j in range(i):
                other_vehicle = self.vehicles[j]
                if (other_vehicle.lane == vehicle.lane and 
                    other_vehicle.position > vehicle.position):
                    distance = other_vehicle.position - vehicle.position
                    if distance < min_distance:
                        min_distance = distance
                        leading_vehicle = other_vehicle
            
            if leading_vehicle is None:
                # No vehicle ahead in same lane - accelerate towards target speed
                desired_speed = vehicle.target_speed
                current_speed = vehicle.speed
                acceleration = min(2.0, (desired_speed - current_speed) * 0.3)
                vehicle.speed = min(desired_speed, current_speed + acceleration)
                continue
            
            # Calculate gap to leading vehicle
            gap = leading_vehicle.position - vehicle.position
            
            if gap <= 0:  # Emergency collision avoidance
                vehicle.position = leading_vehicle.position - 5
                vehicle.speed = min(vehicle.speed, leading_vehicle.speed * 0.5)
                continue
            
            # IDM speed update for same lane
            desired_speed = vehicle.target_speed
            current_speed = vehicle.speed
            leading_speed = leading_vehicle.speed
            
            # Calculate desired gap
            desired_gap = self.get_intelligent_safe_distance(current_speed, leading_speed)
            
            # Speed adjustment based on gap
            if gap < desired_gap:
                # Too close - need to slow down
                gap_ratio = gap / desired_gap
                speed_adjustment = (1 - gap_ratio) * 0.5  # Gradual adjustment
                new_speed = current_speed * (1 - speed_adjustment)
                vehicle.speed = max(5.0, min(new_speed, leading_speed * 0.95))
            else:
                # Good gap - can accelerate towards desired speed
                acceleration = min(2.0, (desired_speed - current_speed) * 0.3)
                vehicle.speed = min(desired_speed, current_speed + acceleration)
    
    def spawn_vehicle(self):
        """Enhanced vehicle spawning with intelligent lane selection"""
        current_interval = self.get_current_interval()
        new_speed = self.generate_vehicle_speed(current_interval)
        
        if self.should_spawn_vehicle(current_interval) and self.can_spawn_vehicle(new_speed):
            # Find the best lane for spawning
            best_lane = self.select_best_spawn_lane()
            
            vehicle = Vehicle(
                vehicle_id=self.vehicle_counter,
                initial_speed=new_speed,
                spawn_time=self.simulation_time,
                spawn_interval=current_interval
            )
            vehicle.target_speed = new_speed  # Set target speed
            vehicle.lane = best_lane  # Assign selected lane
            self.vehicles.append(vehicle)
            self.vehicle_counter += 1
            self.last_spawn_time = self.simulation_time
            return True
        return False
    
    def select_best_spawn_lane(self):
        """Select the best lane for spawning a new vehicle"""
        nearby_vehicles = [v for v in self.vehicles if v.position < 300]
        
        if not nearby_vehicles:
            return np.random.randint(0, 3)  # Random lane if no nearby vehicles
        
        # Calculate gap to nearest vehicle in each lane
        lane_gaps = {}
        for lane in range(3):
            lane_vehicles = [v for v in nearby_vehicles if v.lane == lane]
            if not lane_vehicles:
                lane_gaps[lane] = float('inf')  # Empty lane
            else:
                closest = min(lane_vehicles, key=lambda v: v.position)
                lane_gaps[lane] = closest.position
        
        # Select lane with largest gap
        best_lane = max(lane_gaps.keys(), key=lambda k: lane_gaps[k])
        
        # Add some randomness to prevent all vehicles using same lane
        if np.random.random() < 0.2:  # 20% chance to use random lane
            available_lanes = [lane for lane, gap in lane_gaps.items() if gap >= 20.0]
            if available_lanes:
                best_lane = np.random.choice(available_lanes)
        
        return best_lane
    
    def update_vehicles(self):
        """Update vehicle positions with enhanced physics"""
        vehicles_to_remove = []
        
        for vehicle in self.vehicles:
            # Update position based on current speed
            vehicle.position += vehicle.speed * self.time_step
            
            # Check if vehicle completed the highway
            if vehicle.position >= self.highway_length:
                vehicle.finished = True
                vehicle.finish_time = self.simulation_time
                vehicle.travel_time = vehicle.finish_time - vehicle.spawn_time
                
                if vehicle.travel_time > 0:
                    vehicle.average_speed = self.highway_length / vehicle.travel_time * 3.6  # km/h
                    
                    # Only add to statistics if it's a real spawned vehicle
                    if vehicle.spawn_time >= 0:
                        self.interval_stats[vehicle.spawn_interval]['speeds'].append(vehicle.average_speed)
                        self.interval_stats[vehicle.spawn_interval]['count'] += 1
                
                self.finished_vehicles.append(vehicle)
                vehicles_to_remove.append(vehicle)
        
        # Remove finished vehicles
        for vehicle in vehicles_to_remove:
            self.vehicles.remove(vehicle)
    
    def visualize_traffic(self, width=100):
        """Visualize current traffic density on highway using Unicode block characters"""
        if not self.vehicles:
            return "─" * width + " (No vehicles)"
        
        # Create density map
        density_map = [0] * width
        segment_length = self.highway_length / width
        
        # Count vehicles in each segment
        for vehicle in self.vehicles:
            segment_index = min(int(vehicle.position / segment_length), width - 1)
            density_map[segment_index] += 1
        
        # Convert density to Unicode blocks
        max_density = max(density_map) if max(density_map) > 0 else 1
        visualization = ""
        
        for density in density_map:
            if density == 0:
                visualization += "─"  # Empty road
            elif density == 1:
                visualization += "▁"  # Light traffic
            elif density <= max_density * 0.3:
                visualization += "▂"  # Light-medium traffic
            elif density <= max_density * 0.5:
                visualization += "▃"  # Medium traffic
            elif density <= max_density * 0.7:
                visualization += "▄"  # Medium-heavy traffic
            elif density <= max_density * 0.85:
                visualization += "▅"  # Heavy traffic
            elif density <= max_density * 0.95:
                visualization += "▆"  # Very heavy traffic
            else:
                visualization += "█"  # Maximum density/traffic jam
        
        return visualization
    
    def print_traffic_status(self, width=80):
        """Print detailed traffic status with visualization"""
        current_interval = self.get_current_interval()
        current_hour = int(self.simulation_time // 3600) % 24
        current_minute = int((self.simulation_time % 3600) // 60)
        active_vehicles = len(self.vehicles)
        
        # Calculate average speed of active vehicles
        if self.vehicles:
            avg_current_speed = np.mean([v.speed * 3.6 for v in self.vehicles])  # Convert to km/h
            speeds = [v.speed * 3.6 for v in self.vehicles]
            min_speed = min(speeds)
            max_speed = max(speeds)
            traffic_density = active_vehicles / (self.highway_length / 1000)  # vehicles per km
        else:
            avg_current_speed = 0
            min_speed = 0
            max_speed = 0
            traffic_density = 0
        
        # Get target values for current interval
        target_speed = self.mean_speeds.get(current_interval, 0)
        target_count = self.scaled_counts.get(current_interval, 0)
        
        # Calculate lane distribution
        lane_counts = [0, 0, 0]
        if self.vehicles:
            for vehicle in self.vehicles:
                lane_counts[vehicle.lane] += 1
        
        print(f"\n┌{'─' * (width + 25)}┐")
        print(f"│ {current_hour:02d}:{current_minute:02d} │ Active: {active_vehicles:4d} │ Density: {traffic_density:4.1f}/km │ Speed: {avg_current_speed:5.1f} km/h │ Target: {target_speed:5.1f} km/h │")
        print(f"├{'─' * (width + 25)}┤")
        print(f"│ Start {self.visualize_traffic(width)} End │")
        print(f"├{'─' * (width + 25)}┤")
        print(f"│ Range: {min_speed:4.1f}-{max_speed:5.1f} km/h │ Target Count: {target_count:5.1f}/5min │ Interval: {current_interval:3d}/288 │")
        print(f"│ Lanes: L0:{lane_counts[0]:3d} L1:{lane_counts[1]:3d} L2:{lane_counts[2]:3d} │ Lane Distribution: {lane_counts[0]+lane_counts[1]+lane_counts[2]} total │")
        print(f"└{'─' * (width + 25)}┘")
        
        # Legend
        print("Legend: ─(empty) ▁▂▃▄▅▆█(increasing density)")

    def pre_populate_highway(self):
        """Pre-populate with improved initial distribution"""
        if self.is_pre_populated:
            return
        
        print("Pre-populating highway with enhanced initial traffic...")
        
        # Use first interval data
        initial_speed_kmh = self.mean_speeds.get(0, 100.0)
        initial_count = self.scaled_counts.get(0, 50.0)
        
        # Calculate realistic initial vehicle count
        avg_speed_ms = initial_speed_kmh / 3.6
        avg_spacing = self.get_intelligent_safe_distance(avg_speed_ms)
        max_vehicles = int(self.highway_length / avg_spacing * 0.8)  # 80% of theoretical max
        initial_vehicle_count = min(max_vehicles, int(initial_count * 8))  # Scale up for highway length
        
        print(f"Adding {initial_vehicle_count} vehicles (speed: {initial_speed_kmh:.1f} km/h)")
        
        # Create vehicles with realistic spacing
        positions = []
        current_pos = 100
        for _ in range(initial_vehicle_count):
            if current_pos >= self.highway_length - 1000:
                break
            positions.append(current_pos)
            spacing = np.random.normal(avg_spacing, avg_spacing * 0.3)
            current_pos += max(spacing, self.min_safe_distance)
        
        for i, position in enumerate(positions):
            speed_ms = self.generate_vehicle_speed(0)
            
            vehicle = Vehicle(
                vehicle_id=self.vehicle_counter,
                initial_speed=speed_ms,
                spawn_time=-1,  # Mark as pre-existing
                spawn_interval=0
            )
            vehicle.position = position
            vehicle.target_speed = speed_ms
            # Distribute vehicles across lanes for pre-population
            vehicle.lane = i % 3  # Cycle through lanes 0, 1, 2
            
            self.vehicles.append(vehicle)
            self.vehicle_counter += 1
        
        # Optimize initial speeds
        self.update_vehicle_speeds_intelligent()
        
        self.is_pre_populated = True
        print(f"Highway pre-populated with {len(self.vehicles)} vehicles")
        if self.vehicles:
            print(f"Average initial speed: {np.mean([v.speed * 3.6 for v in self.vehicles]):.1f} km/h")
        print("-" * 50)

    def run_simulation(self):
        """Run the enhanced simulation"""
        print("Starting Enhanced Real Data Highway Simulation...")
        print(f"Highway Length: {self.highway_length/1000:.1f} km")
        print(f"Simulation Duration: {self.simulation_hours} hours")
        print(f"Scaling Factor: {self.scaling_factor}")
        print("-" * 50)
        
        # Pre-populate highway
        self.pre_populate_highway()
        
        total_simulation_time = self.simulation_hours * 3600
        
        while self.simulation_time < total_simulation_time:
            current_interval = self.get_current_interval()
            current_hour = int(self.simulation_time // 3600) % 24
            current_minute = int((self.simulation_time % 3600) // 60)
            
            # Main simulation steps
            self.spawn_vehicle()
            self.update_vehicle_speeds_intelligent()
            self.update_vehicles()
            
            # Progress reporting every 30 minutes
            if int(self.simulation_time) % 1800 == 0:
                active = len(self.vehicles)
                completed = len(self.finished_vehicles)
                target_speed = self.mean_speeds.get(current_interval, 0)
                target_count = self.scaled_counts.get(current_interval, 0)
                
                if active > 0:
                    avg_speed = np.mean([v.speed * 3.6 for v in self.vehicles])
                    traffic_density = active / (self.highway_length / 1000)
                else:
                    avg_speed = 0
                    traffic_density = 0
                
                print(f"Time {current_hour:2d}:{current_minute:02d}: Active: {active:4d}, "
                      f"Completed: {completed:5d}, Avg Speed: {avg_speed:5.1f} km/h, "
                      f"Density: {traffic_density:4.1f} veh/km")
            
            self.simulation_time += self.time_step
        
        # Process remaining vehicles
        print("\nProcessing remaining vehicles...")
        for vehicle in self.vehicles:
            if vehicle.position > 0 and vehicle.spawn_time >= 0:
                remaining_distance = self.highway_length - vehicle.position
                remaining_time = remaining_distance / vehicle.speed if vehicle.speed > 0 else 3600
                vehicle.travel_time = (self.simulation_time - vehicle.spawn_time) + remaining_time
                vehicle.average_speed = self.highway_length / vehicle.travel_time * 3.6
                
                self.interval_stats[vehicle.spawn_interval]['speeds'].append(vehicle.average_speed)
                self.interval_stats[vehicle.spawn_interval]['count'] += 1
                self.finished_vehicles.append(vehicle)
    
    def run_simulation_with_visualization(self, show_every_minutes=10, visualization_width=60):
        """Run simulation with periodic traffic visualization"""
        print("Starting Enhanced Real Data Highway Simulation with Traffic Visualization...")
        print(f"Highway Length: {self.highway_length/1000:.1f} km")
        print(f"Simulation Duration: {self.simulation_hours} hours")
        print(f"Scaling Factor: {self.scaling_factor}")
        print(f"Visualization updates every {show_every_minutes} minutes")
        print("=" * 80)
        
        # Pre-populate highway
        self.pre_populate_highway()
        
        total_simulation_time = self.simulation_hours * 3600
        visualization_interval = show_every_minutes * 60  # Convert to seconds
        last_visualization_time = 0
        
        while self.simulation_time < total_simulation_time:
            current_interval = self.get_current_interval()
            current_hour = int(self.simulation_time // 3600) % 24
            current_minute = int((self.simulation_time % 3600) // 60)
            
            # Main simulation steps
            self.spawn_vehicle()
            self.update_vehicle_speeds_intelligent()
            self.update_vehicles()
            
            # Show visualization periodically
            if (self.simulation_time - last_visualization_time >= visualization_interval or 
                int(self.simulation_time) % 1800 == 0):  # Every interval or every 30 minutes
                
                self.print_traffic_status(visualization_width)
                last_visualization_time = self.simulation_time
                
                # Add a small delay to make visualization readable
                time.sleep(0.2)
            
            self.simulation_time += self.time_step
        
        # Process remaining vehicles
        print("\nProcessing remaining vehicles...")
        for vehicle in self.vehicles:
            if vehicle.position > 0 and vehicle.spawn_time >= 0:
                remaining_distance = self.highway_length - vehicle.position
                remaining_time = remaining_distance / vehicle.speed if vehicle.speed > 0 else 3600
                vehicle.travel_time = (self.simulation_time - vehicle.spawn_time) + remaining_time
                vehicle.average_speed = self.highway_length / vehicle.travel_time * 3.6
                
                self.interval_stats[vehicle.spawn_interval]['speeds'].append(vehicle.average_speed)
                self.interval_stats[vehicle.spawn_interval]['count'] += 1
                self.finished_vehicles.append(vehicle)
        
        # Final traffic state
        print("\nFinal Traffic State:")
        self.print_traffic_status(visualization_width)
    
    def calculate_detailed_statistics(self):
        """Calculate detailed statistics for plotting and analysis"""
        # Create hourly aggregations
        hourly_stats = {
            'real_speeds': [],
            'sim_speeds': [],
            'real_counts': [],
            'sim_counts': [],
            'hours': list(range(24)),
            'spawn_success_rate': [],
            'all_real_speeds': [],
            'all_sim_speeds': []
        }
        
        # Aggregate data by hour
        for hour in range(24):
            hour_intervals = range(hour * 12, (hour + 1) * 12)  # 12 intervals per hour
            
            # Real data for this hour
            real_speeds_hour = [self.mean_speeds.get(i, 0) for i in hour_intervals if i in self.mean_speeds]
            real_counts_hour = [self.scaled_counts.get(i, 0) for i in hour_intervals if i in self.scaled_counts]
            
            # Simulation data for this hour
            sim_speeds_hour = []
            sim_counts_hour = []
            spawn_attempts = 0
            successful_spawns = 0
            
            for i in hour_intervals:
                if i in self.interval_stats:
                    stats = self.interval_stats[i]
                    sim_speeds_hour.extend(stats['speeds'])
                    sim_counts_hour.append(stats['count'])
                    spawn_attempts += stats['spawn_attempts']
                    successful_spawns += stats['count']
            
            # Calculate hourly averages
            hourly_stats['real_speeds'].append(np.mean(real_speeds_hour) if real_speeds_hour else 0)
            hourly_stats['sim_speeds'].append(np.mean(sim_speeds_hour) if sim_speeds_hour else 0)
            hourly_stats['real_counts'].append(np.sum(real_counts_hour) if real_counts_hour else 0)
            hourly_stats['sim_counts'].append(np.sum(sim_counts_hour) if sim_counts_hour else 0)
            
            # Calculate spawn success rate
            success_rate = (successful_spawns / spawn_attempts * 100) if spawn_attempts > 0 else 0
            hourly_stats['spawn_success_rate'].append(success_rate)
            
            # Store all individual speeds for correlation
            hourly_stats['all_real_speeds'].extend(real_speeds_hour)
            hourly_stats['all_sim_speeds'].extend(sim_speeds_hour)
        
        return hourly_stats
    
    def plot_enhanced_comparison(self, save_path=None):
        """Create enhanced comparison plots matching the original functionality"""
        print("Generating enhanced comparison plots...")
        
        # Auto-generate save path for Linux systems
        if save_path is None and platform.system() == 'Linux':
            # Create export/plot directory if it doesn't exist
            plot_dir = "plot/"
            os.makedirs(plot_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(plot_dir, f"enhanced_highway_simulation_{timestamp}.png")
        
        # Calculate statistics
        stats = self.calculate_detailed_statistics()
        
        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Highway Simulation Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Hourly Average Speed Comparison
        ax1.plot(stats['hours'], stats['real_speeds'], 'b-o', label='Real Speed', linewidth=2, markersize=4)
        ax1.plot(stats['hours'], stats['sim_speeds'], 'r-s', label='Simulated Speed', linewidth=2, markersize=4)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Speed (km/h)')
        ax1.set_title('Hourly Average Speed Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 23)
        
        # Plot 2: Hourly Vehicle Count Comparison
        ax2.plot(stats['hours'], stats['real_counts'], 'b-o', label='Target Count', linewidth=2, markersize=4)
        ax2.plot(stats['hours'], stats['sim_counts'], 'r-s', label='Achieved Count', linewidth=2, markersize=4)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Vehicles per Hour')
        ax2.set_title('Hourly Vehicle Count Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 23)
        
        # Plot 3: Speed Correlation Analysis
        if stats['all_real_speeds'] and stats['all_sim_speeds']:
            # Match lengths for correlation
            min_length = min(len(stats['all_real_speeds']), len(stats['all_sim_speeds']))
            real_speeds_matched = stats['all_real_speeds'][:min_length]
            sim_speeds_matched = stats['all_sim_speeds'][:min_length]
            
            if len(real_speeds_matched) > 1 and len(sim_speeds_matched) > 1:
                correlation = np.corrcoef(real_speeds_matched, sim_speeds_matched)[0, 1]
                
                ax3.scatter(real_speeds_matched, sim_speeds_matched, alpha=0.6, s=20, color='blue')
                
                # Perfect correlation line
                min_speed = min(min(real_speeds_matched), min(sim_speeds_matched))
                max_speed = max(max(real_speeds_matched), max(sim_speeds_matched))
                ax3.plot([min_speed, max_speed], [min_speed, max_speed], 'r--', 
                        label='Perfect Correlation', linewidth=2)
                
                ax3.set_xlabel('Real Speed (km/h)')
                ax3.set_ylabel('Simulated Speed (km/h)')
                ax3.set_title(f'Speed Correlation (r = {correlation:.3f})')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient data for correlation', 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Speed Correlation')
        else:
            ax3.text(0.5, 0.5, 'No speed data available', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Speed Correlation')
        
        # Plot 4: Vehicle Spawning Success Rate by Hour
        ax4.plot(stats['hours'], stats['spawn_success_rate'], '-o', linewidth=2, markersize=4, color='green')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Spawn Success Rate (%)')
        ax4.set_title('Vehicle Spawning Success Rate by Hour')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 23)
        ax4.set_ylim(0, 100)
        
        # Add background color for better readability
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("ENHANCED STATISTICS SUMMARY")
        print("="*60)
        
        if stats['all_real_speeds'] and stats['all_sim_speeds']:
            min_length = min(len(stats['all_real_speeds']), len(stats['all_sim_speeds']))
            if min_length > 1:
                correlation = np.corrcoef(stats['all_real_speeds'][:min_length], 
                                       stats['all_sim_speeds'][:min_length])[0, 1]
                print(f"Speed Correlation: {correlation:.3f}")
        
        real_speed_avg = np.mean([s for s in stats['real_speeds'] if s > 0])
        sim_speed_avg = np.mean([s for s in stats['sim_speeds'] if s > 0])
        real_count_total = sum(stats['real_counts'])
        sim_count_total = sum(stats['sim_counts'])
        avg_spawn_success = np.mean(stats['spawn_success_rate'])
        
        print(f"Average Real Speed: {real_speed_avg:.1f} km/h")
        print(f"Average Simulated Speed: {sim_speed_avg:.1f} km/h")
        print(f"Total Target Vehicles: {real_count_total:.0f}")
        print(f"Total Achieved Vehicles: {sim_count_total:.0f}")
        print(f"Average Spawn Success Rate: {avg_spawn_success:.1f}%")
        print(f"Finished Vehicles: {len(self.finished_vehicles)}")
        
        if len(self.finished_vehicles) > 0:
            travel_times = [v.travel_time for v in self.finished_vehicles if v.travel_time > 0]
            if travel_times:
                avg_travel_time = np.mean(travel_times)
                print(f"Average Travel Time: {avg_travel_time/60:.1f} minutes")
                avg_sim_speed_finished = np.mean([v.average_speed for v in self.finished_vehicles if v.average_speed > 0])
                print(f"Average Speed (Finished Vehicles): {avg_sim_speed_finished:.1f} km/h")
        
        print("="*60)

def main(traffic_data_file, scaling_factor=0.4, highway_length_km=84, with_visualization=False, viz_interval_minutes=10):
    """Main function for enhanced simulation"""
    print(f"Enhanced Real Data Highway Simulation")
    print(f"Data file: {traffic_data_file}")
    print(f"Scaling factor: {scaling_factor}")
    print(f"Highway length: {highway_length_km} km")
    if with_visualization:
        print(f"Visualization: Every {viz_interval_minutes} minutes")
    print("=" * 60)
    
    # Create and run simulation
    sim = EnhancedRealDataSimulation(
        traffic_data_file=traffic_data_file,
        highway_length_km=highway_length_km,
        simulation_hours=24,
        scaling_factor=scaling_factor
    )
    
    # Run simulation with or without visualization
    if with_visualization:
        sim.run_simulation_with_visualization(show_every_minutes=viz_interval_minutes)
    else:
        sim.run_simulation()
    
    # Generate enhanced statistics and plots
    try:
        # Create basic statistics summary
        total_vehicles = len(sim.finished_vehicles)
        avg_speed = np.mean([v.speed for v in sim.vehicles]) if sim.vehicles else 0
        
        results = {
            "Total Vehicles": total_vehicles,
            "Average Speed": f"{avg_speed:.1f} km/h",
            "Highway Length": f"{sim.highway_length/1000:.1f} km",
            "Scaling Factor": sim.scaling_factor,
            "Data Intervals": len(sim.mean_speeds) if hasattr(sim, 'mean_speeds') else 0
        }
        
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        for key, value in results.items():
            print(f"{key}: {value}")
            
        # Generate comparison plots only once in main function
        if not with_visualization:  # Don't duplicate plots if already shown
            print("\nGenerating comparison plots...")
            sim.plot_enhanced_comparison()
            
    except Exception as e:
        print(f"Warning: Could not generate statistics: {e}")
        results = {"error": str(e)}
    
    return {
        'simulation': sim,
        'results': results
    }

def run_with_traffic_visualization(data_file=None, scaling_factor=0.4, highway_length=84, viz_interval=10):
    """Run simulation with real-time traffic visualization"""
    if data_file is None:
        data_file = "average/averaged_Saturday_N_traffic_data_2024.csv"
    
    print("Enhanced Highway Traffic Simulation with Real-time Visualization")
    print("=" * 70)
    print("This will show traffic density updates with Unicode block characters")
    print("Press Ctrl+C to stop early if needed")
    print("=" * 70)
    
    try:
        results = main(data_file, scaling_factor, highway_length, 
                      with_visualization=True, viz_interval_minutes=viz_interval)
        return results
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return None

if __name__ == "__main__":
    import sys
    import os
    
    # Default parameters
    default_data_file = "average/averaged_Saturday_N_traffic_data_2024.csv"
    default_scaling = 0.4
    default_highway_length = 84  # km
    
    # Parse command line arguments
    data_file = default_data_file
    scaling_factor = default_scaling
    highway_length = default_highway_length
    with_visualization = False
    with_plots = True  # Default to showing plots
    viz_interval = 10
    
    # Separate flags from positional arguments
    positional_args = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--visualize":
            with_visualization = True
        elif arg == "--no-plots":
            with_plots = False
        elif arg.startswith("--viz-interval="):
            try:
                viz_interval = int(arg.split("=")[1])
            except ValueError:
                viz_interval = 10
        elif not arg.startswith("--"):
            positional_args.append(arg)
        i += 1
    
    # Process positional arguments
    if len(positional_args) > 0:
        data_file = positional_args[0]
    if len(positional_args) > 1:
        try:
            scaling_factor = float(positional_args[1])
        except ValueError:
            scaling_factor = default_scaling
            print(f"Invalid scaling factor, using default: {default_scaling}")
    if len(positional_args) > 2:
        try:
            highway_length = float(positional_args[2])
        except ValueError:
            highway_length = default_highway_length
            print(f"Invalid highway length, using default: {default_highway_length} km")
    
    print(f"Usage: python enhanced_real_data_simulation.py [data_file] [scaling_factor] [highway_length_km] [--visualize] [--no-plots] [--viz-interval=N]")
    print(f"Using: {data_file}, scaling: {scaling_factor}, highway length: {highway_length} km")
    if with_visualization:
        print(f"Visualization enabled: Updates every {viz_interval} minutes")
    if with_plots:
        print("Plots will be generated after simulation")
    
    # Show available files
    if data_file == default_data_file:
        print("\nAvailable data files in export/:")
        if os.path.exists("export"):
            for f in sorted(os.listdir("export")):
                if f.endswith(".csv") and "averaged" in f:
                    print(f"  - {f}")
        print("-" * 60)
    
    try:
        results = main(data_file, scaling_factor, int(highway_length), 
                      with_visualization, viz_interval)
        print("\nEnhanced simulation completed successfully!")
        
        # Generate plots if enabled
        if with_plots and 'simulation' in results:
            print("\nGenerating enhanced comparison plots...")
            results['simulation'].plot_enhanced_comparison()
        
        # Results summary
        print(f"Final results: {len(results)} items returned")
        
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()