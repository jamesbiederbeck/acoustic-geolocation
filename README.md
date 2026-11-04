# Acoustic Geolocation via Airplane Doppler Curves

## Overview
This repository contains the logic and algorithmic models for geolocating video or audio recordings using faint, background airplane acoustic signatures. By extracting Doppler curves from audio spectrograms and cross-referencing them with historical ADS-B flight data, we can mathematically pinpoint the exact location where a recording was captured.

The methodology relies on three primary steps:
1. **Audio Analysis:** Extracting temporal events and frequency curves from a recording.
2. **Coarse Filtering (Time/Silence):** Matching time delays between acoustic events to ADS-B data and eliminating candidate locations that should have heard extraneous flights.
3. **Precision Doppler Modeling:** Generating localized, theoretical Doppler curves for remaining grid points and scoring them against the actual spectrogram.

## Core Mathematics

The algorithm fundamentally relies on modeling sound propagation and the Doppler effect. 

### 1. The Doppler Equation
Models the observed frequency of a moving sound source from a stationary observer.
$$ f = f_0 \left( \frac{c}{c - v_r} \right) $$

### 2. Sound Propagation Delay
Relates the exact time a sound is heard to the time it was emitted, accounting for the speed of sound and the 3D slant distance.
$$ t_{heard} = t_{emitted} + \frac{d(t_{emitted})}{c} $$

### 3. Radial Velocity
Calculates the velocity component of the aircraft directed precisely toward the observer.
$$ v_r = \vec{v} \cdot \frac{\vec{r}}{|\vec{r}|} $$

*Where:*
*   $f$ = Observed frequency on the spectrogram
*   $f_0$ = Fundamental emission frequency of the aircraft engine/propeller
*   $c$ = Speed of sound
*   $v_r$ = Radial velocity of the aircraft relative to the observer
*   $d$ (or $|\vec{r}|$) = 3D slant distance between aircraft and observer at emission time

## Algorithm Pseudocode

```text
FUNCTION GeolocateFromAudio(audio_file, date, rough_search_area):
    // 1. Audio Analysis: Extract Doppler curves from the recording
    spectrogram = generate_spectrogram(audio_file)
    events = find_doppler_curves(spectrogram) // e.g., Event 1 and Event 2
    time_offset = time_difference(events[0], events[1])
    
    // Estimate speed and distance based on curve shape
    v_range_1, dist_range_1 = estimate_parameters(events[0])
    v_range_2, dist_range_2 = estimate_parameters(events[1])

    // 2. Fetch and prep historical flight data
    adsb_data = fetch_adsb_data(date, rough_search_area)
    adsb_data = filter_aircraft_type(adsb_data, "propeller")
    
    // 3. Generate initial candidate points (Coarse Grid)
    dem = load_digital_elevation_model(rough_search_area)
    candidate_points = []
    
    FOR EACH point IN generate_grid(rough_search_area, resolution=200m):
        // Calculate Delayed Closest Point of Arrival accounting for sound travel
        delayed_cpa_1 = get_delayed_cpa(point, adsb_data, v_range_1, dist_range_1, dem)
        delayed_cpa_2 = get_delayed_cpa(point, adsb_data, v_range_2, dist_range_2, dem)
        
        // Time Offset Filter
        IF abs((delayed_cpa_2.time - delayed_cpa_1.time) - time_offset) < error_margin:
            // Silence Filter: Ensure no other planes would be heard between events
            IF count_flights_heard_between(point, adsb_data, delayed_cpa_1, delayed_cpa_2) == 0:
                candidate_points.append(point)

    // 4. Precision Doppler Modeling & Scoring (Fine Grid)
    best_score = 0
    best_location = NULL
    fine_grid = upscale_grid(candidate_points, resolution=100m)
    
    FOR EACH point IN fine_grid:
        flight_1 = get_associated_flight(point, delayed_cpa_1)
        flight_2 = get_associated_flight(point, delayed_cpa_2)
        
        // Generate the exact theoretical Doppler curve
        curve_1 = model_doppler_curve(point, flight_1, events[0].base_freq)
        curve_2 = model_doppler_curve(point, flight_2, events[1].base_freq)
        
        // Score by overlaying modeled curves on actual spectrogram brightness
        score = overlap_score(spectrogram, curve_1, curve_2, lock_temporal_offset=TRUE)
        
        IF score > best_score:
            best_score = score
            best_location = point

    RETURN best_location


FUNCTION model_doppler_curve(point, flight, base_freq):
    modeled_curve = []
    c = speed_of_sound
    
    FOR EACH time t IN duration_of_event:
        // Calculate where the plane was when it PRODUCED the sound heard at time t
        emission_time = t - (distance(flight.position_at(t), point) / c)
        plane_pos = flight.position_at(emission_time)
        
        // Calculate radial velocity
        radial_vel = calculate_radial_velocity(plane_pos, point)
        
        // Doppler Equation
        expected_freq = base_freq * (c / (c - radial_vel))
        modeled_curve.append(time=t, freq=expected_freq)
        
    RETURN modeled_curve
```

## Next Steps
- [ ] Implement audio spectrogram generation and base frequency detection.
- [ ] Build ADS-B data ingestion pipeline.
- [ ] Implement WGS84 and DEM (Digital Elevation Model) geospatial math functions.
- [ ] Write the coarse grid filtering logic.
