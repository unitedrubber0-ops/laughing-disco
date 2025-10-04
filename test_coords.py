from app import calculate_development_length_safe

# Mixed shapes that mimic your vision/regex outputs
mixed_points = [
    {"point": "P0"},                 # missing numeric values -> should be skipped
    "P1 56.6 0.0 0.0 40.0",          # string -> parsed
    [152.0, 110.9, 42.9, 40.0],      # list -> parsed
    {"X": "200.0", "Y": "120.0", "Z": "45.0"}  # dict with caps -> parsed
]

try:
    dev_len = calculate_development_length_safe(mixed_points)
    print("Dev len OK:", dev_len)
except Exception as e:
    print("Dev len failed:", e)