# src/offline_pipeline/taxonomy.py

RULE_VARIABLE_TAXONOMY = {

    "weather_ocean_conditions": [
        "wind_speed_ms",
        "wind_gust_ms",
        "wave_height_m",
        "dominant_period_s",
        "average_period_s",
        "mean_wave_dir_deg",
        "tide_ft",
        "air_temp_c",
        "water_temp_c",
        "pressure_hpa"
    ],

    "environment_events": [
        "event_storm",
        "event_high_tide",
        "event_low_tide",
        "event_heat",
        "event_pressure_drop"
    ],

    "vessel_characteristics": [
        "vessel_capacity_teu",
        "vessel_loa_meters",
        "size_category"
    ],

    "berth_operations": [
        "arrival_delay_hours",
        "berth_delay_hours",
        "planned_duration_hours",
        "actual_duration_hours",
        "containers_planned",
        "containers_actual",
        "discharge_actual",
        "load_actual",
        "restows",
        "berth_productivity_mph",
        "gross_crane_productivity_mph"
    ],

    "crane_operations": [
        "crane_hours",
        "total_moves",
        "discharge_moves",
        "load_moves",
        "crane_productivity_mph",
        "breakdown_minutes",
        "waiting_time_minutes",
        "utilization_pct"
    ],

    "yard_operations": [
        "teu_received",
        "teu_delivered",
        "reefer_slots_used",
        "hazmat_teu",
        "average_dwell_days",
        "rtg_moves",
        "peak_occupancy_pct"
    ],

    "gate_operations": [
        "total_transactions",
        "inbound_loaded",
        "inbound_empty",
        "outbound_loaded",
        "outbound_empty",
        "peak_hour_volume",
        "appointment_transactions",
        "walkin_transactions",
        "average_turn_time_minutes",
        "dual_transactions",
        "rejected_trucks"
    ]
}


VARIABLE_SYNONYMS = {

    "wind_speed_ms": [
        "wind speed",
        "wind velocity",
        "wind",
        "average wind speed"
    ],

    "wind_gust_ms": [
        "wind gust",
        "gust speed"
    ],

    "wave_height_m": [
        "wave height",
        "wave"
    ],

    "tide_ft": [
        "tide",
        "tidal height"
    ],

    "air_temp_c": [
        "air temperature",
        "temperature"
    ],

    "water_temp_c": [
        "water temperature",
        "sea temperature"
    ],

    "pressure_hpa": [
        "pressure",
        "atmospheric pressure"
    ],

    "vessel_capacity_teu": [
        "vessel capacity",
        "capacity teu"
    ],

    "berth_productivity_mph": [
        "berth productivity"
    ],

    "crane_productivity_mph": [
        "crane productivity"
    ],

    "average_turn_time_minutes": [
        "turn time",
        "truck turn time"
    ]
}