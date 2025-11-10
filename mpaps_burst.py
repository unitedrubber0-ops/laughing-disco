import re

# TABLE IV burst pressure data
# Example data, replace with actual values
TABLE_IV_ROWS = [
    {'min_id_mm': 24, 'max_id_mm': 32, 'b_1_3_mp': 1.38, 'b_2_mp': 1.72, 'f_1_mp': 2.07},
    # Add other rows as needed
]

def normalize_grade(grade_raw: str):
    """
    Normalize e.g. 'GRADE 1 B' -> '1B', '1 BF' -> '1BF', '1BFD' stays '1BFD'
    """
    if not grade_raw:
        return ''
    g = grade_raw.upper()
    g = re.sub(r'GRADE\s*', '', g)
    g = re.sub(r'[^A-Z0-9]', '', g)  # remove spaces and odd chars
    return g

def get_burst_pressure_from_table_iv(id_value_mm: float, grade_raw: str):
    """
    Look up burst pressure in TABLE IV rows.
    TABLE_IV_ROWS is expected to be an iterable of dicts with keys:
      'min_id_mm','max_id_mm','b_1_3_mp','b_2_mp','f_1_mp', ...
    """
    EPS = 1e-6  # for float comparisons
    g = normalize_grade(grade_raw)
    for row in TABLE_IV_ROWS:
        min_id = float(row['min_id_mm'])
        max_id = float(row['max_id_mm'])
        # inclusive bounds, with epsilon tolerance
        if (min_id - EPS) <= id_value_mm <= (max_id + EPS):
            # attempt grade-specific matches
            # prefer explicit matches: '1B' or '1BF' etc
            if '1B' in g or g.startswith('1B'):
                # use B Grade 1&3 (your table naming)
                return float(row.get('b_1_3_mp') or row.get('b_1_3') or row.get('b_1_3_mpa'))
            if '2B' in g or g.startswith('2B'):
                return float(row.get('b_2_mp') or row.get('b_2'))
            if 'F1' in g or 'BF' in g or '1BF' in g:
                return float(row.get('f_1_mp') or row.get('f_1'))
            # fallback: if table has a default for this band return it
            default_val = row.get('default_burst_mp') or row.get('burst_mp')
            if default_val:
                return float(default_val)
    # no match
    return None