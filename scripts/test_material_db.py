import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app

def test_material_database():
    print("Loading material database...")
    material_df, reinforcement_df = app.load_material_database()
    print('Database Loaded:', material_df is not None)
    
    test_cases = [
        ('MPAPS F-30/F-1', '1B'),
        ('MPAPS F-6034', 'H-AN'),
        ('J20', 'CLASS A'),
        ('M3055-7', '--'),
        ('TL 52361', '--')
    ]
    
    print('\nTesting Material Lookups:')
    for standard, grade in test_cases:
        print(f'\nStandard: {standard}, Grade: {grade}')
        material = app.get_material_from_standard(standard, grade)
        reinforcement = app.get_reinforcement_from_material(standard, grade, material)
        print(f'Material: {material}')
        print(f'Reinforcement: {reinforcement}')

if __name__ == '__main__':
    test_material_database()