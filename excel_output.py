"""
Enhanced Excel output generation with improved formatting and data handling.
"""
import pandas as pd
import logging
import io
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from development_length import calculate_development_length

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_corrected_excel_sheet(analysis_results, dimensions, coordinates):
    """
    Generate enhanced Excel output with proper formatting and data validation.
    
    Args:
        analysis_results (dict): Results from drawing analysis
        dimensions (dict): Extracted dimensions
        coordinates (list): Extracted coordinate points
    
    Returns:
        BytesIO: Excel file data
    """
    # Log incoming data for debugging
    logger.info(f"Excel generation - Analysis results overview:")
    logger.info(f"  Part Number: {analysis_results.get('part_number', 'Not Found')}")
    logger.info(f"  Standard: {analysis_results.get('standard', 'Not Found')}")
    logger.info(f"  Grade: {analysis_results.get('grade', 'Not Found')}")
    logger.info(f"  Material: {analysis_results.get('material', 'Not Found')}")
    logger.info(f"  Reinforcement: {analysis_results.get('reinforcement', 'Not Found')}")
    logger.info(f"  Raw reinforcement data: {analysis_results.get('reinforcement_raw', 'Not Found')}")
    logger.info(f"  Rings data: {analysis_results.get('rings', 'Not Found')}")

    try:
        # Define column structure with proper formatting
        columns = [
            'child part',                                         # Row 1: Original format
            'child quantity',                                     # Row 1: Original format
            'CHILD PART',                                        # Row 2: Uppercase format
            'CHILD PART DESCRIPTION',                            # Description
            'CHILD PART QTY',                                    # Quantity
            'SPECIFICATION',                                     # Combined standard+grade
            'MATERIAL',                                         # From database lookup
            'REINFORCEMENT',                                     # Primary reinforcement info
            'RINGS',                                            # Ring specifications
            'VOLUME AS PER 2D',                                 # Volume calculation
            'ID1 AS PER 2D (MM)',                              # First ID measurement
            'ID2 AS PER 2D (MM)',                              # Second ID measurement
            'OD1 AS PER 2D (MM)',                              # First OD measurement
            'OD2 AS PER 2D (MM)',                              # Second OD measurement
            'THICKNESS AS PER 2D (MM)',                        # Direct thickness
            'THICKNESS AS PER ID OD DIFFERENCE',               # Calculated thickness
            'CENTERLINE LENGTH AS PER 2D (MM)',                # From drawing
            'DEVELOPMENT LENGTH AS PER CO-ORDINATE (MM)',      # Calculated length
            'BURST PRESSURE AS PER 2D (BAR)',                  # From drawing
            'BURST PRESSURE AS PER WORKING PRESSURE (4XWP) (BAR)', # Calculated
            'VOLUME AS PER 2D MM3',                            # Volume in mm³
            'WEIGHT AS PER 2D KG',                             # Weight from drawing
            'COLOUR AS PER DRAWING',                           # Color specification
            'ADDITIONAL REQUIREMENT',                          # Notes and requirements
            'OUTSOURCE',                                       # Outsourcing details
            'REMARK'                                          # Generated remarks
        ]

        # Get part number and clean it
        part_number = str(analysis_results.get('part_number', '')).strip()
        if not part_number or part_number == 'Not Found':
            part_number = "Unknown Part"

        # Get description and clean it
        description = str(analysis_results.get('description', '')).strip()
        if not description or description == 'Not Found':
            description = "No Description Available"

        # Format specification string
        standard = analysis_results.get('standard', 'Not Found')
        grade = analysis_results.get('grade', 'Not Found')
        specification = f"{standard}"
        if grade != 'Not Found':
            specification += f" {grade}"

        # Calculate thickness from ID/OD
        thickness_calculated = "Not Found"
        try:
            od1 = dimensions.get('od1', 'Not Found')
            id1 = dimensions.get('id1', 'Not Found')
            if od1 != 'Not Found' and id1 != 'Not Found':
                od_val = float(str(od1).replace(',', '.'))
                id_val = float(str(id1).replace(',', '.'))
                if od_val > id_val:
                    thickness_calculated = f"{((od_val - id_val) / 2):.2f}"
        except (ValueError, TypeError) as e:
            logger.warning(f"Error calculating thickness: {e}")

        # Calculate development length
        if coordinates:
            development_length = calculate_development_length(coordinates)
        else:
            development_length = "Not Found"
            logger.warning("No coordinates available for development length calculation")

        # Calculate burst pressure from working pressure
        burst_pressure_calc = "Not Found"
        if analysis_results.get('working_pressure') not in ['Not Found', None]:
            try:
                wp = float(str(analysis_results['working_pressure']).replace(',', '.'))
                burst_pressure_calc = f"{(wp * 4):.1f}"
            except (ValueError, TypeError):
                pass

        # Build the row data dictionary
        row_data = {
            'child part': part_number.lower(),
            'child quantity': "1",
            'CHILD PART': part_number.upper(),
            'CHILD PART DESCRIPTION': description,
            'CHILD PART QTY': "1",
            'SPECIFICATION': specification,
            'MATERIAL': analysis_results.get('material', 'Not Found'),
            'REINFORCEMENT': analysis_results.get('reinforcement', 'Not Found'),
            'RINGS': analysis_results.get('rings_raw', 'Not Found'),
            'VOLUME AS PER 2D': analysis_results.get('volume', 'Not Found'),
            'ID1 AS PER 2D (MM)': dimensions.get('id1', 'Not Found'),
            'ID2 AS PER 2D (MM)': dimensions.get('id2', 'Not Found'),
            'OD1 AS PER 2D (MM)': dimensions.get('od1', 'Not Found'),
            'OD2 AS PER 2D (MM)': dimensions.get('od2', 'Not Found'),
            'THICKNESS AS PER 2D (MM)': dimensions.get('thickness', 'Not Found'),
            'THICKNESS AS PER ID OD DIFFERENCE': thickness_calculated,
            'CENTERLINE LENGTH AS PER 2D (MM)': dimensions.get('centerline_length', 'Not Found'),
            'DEVELOPMENT LENGTH AS PER CO-ORDINATE (MM)': development_length,
            'BURST PRESSURE AS PER 2D (BAR)': analysis_results.get('burst_pressure', 'Not Found'),
            'BURST PRESSURE AS PER WORKING PRESSURE (4XWP) (BAR)': burst_pressure_calc,
            'VOLUME AS PER 2D MM3': analysis_results.get('volume_mm3', 'Not Found'),
            'WEIGHT AS PER 2D KG': analysis_results.get('weight', 'Not Found'),
            'COLOUR AS PER DRAWING': analysis_results.get('color', 'Not Found'),
            'ADDITIONAL REQUIREMENT': "CUTTING & CHECKING FIXTURE COST TO BE ADDED. Marking cost to be added.",
            'OUTSOURCE': "",
            'REMARK': ""
        }

        # Generate remarks based on validation
        remarks = []
        
        # Check for specification conversion
        if standard.startswith('MPAPS F 1'):
            remarks.append('Drawing specifies MPAPS F 1, considered as MPAPS F 30.')

        # Check for ID mismatch
        id1 = dimensions.get('id1', 'Not Found')
        id2 = dimensions.get('id2', 'Not Found')
        if id1 != 'Not Found' and id2 != 'Not Found' and id1 != id2:
            remarks.append('THERE IS MISMATCH IN ID 1 & ID 2')

        # Check for OD mismatch
        od1 = dimensions.get('od1', 'Not Found')
        od2 = dimensions.get('od2', 'Not Found')
        if od1 != 'Not Found' and od2 != 'Not Found' and od1 != od2:
            remarks.append('THERE IS MISMATCH IN OD 1 & OD 2')

        # Add remarks to row data
        row_data['REMARK'] = ' '.join(remarks) if remarks else 'No specific remarks.'

        # Create DataFrame
        df = pd.DataFrame([row_data], columns=columns)

        # Create Excel writer with enhanced formatting
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='FETCH FROM DRAWING', index=False)
            
            # Get the worksheet
            worksheet = writer.sheets['FETCH FROM DRAWING']
            
            # Define styles
            header_fill = PatternFill(start_color='CCE5FF', end_color='CCE5FF', fill_type='solid')
            header_font = Font(bold=True)
            border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            
            # Apply formatting
            for col_idx, column in enumerate(columns, 1):
                cell = worksheet.cell(row=1, column=col_idx)
                cell.fill = header_fill
                cell.font = header_font
                cell.border = border
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
                
                # Auto-fit column width
                max_length = max(
                    len(str(column)),
                    len(str(df.iloc[0][column]))
                )
                adjusted_width = min(max_length + 2, 50)  # Cap width at 50 characters
                worksheet.column_dimensions[get_column_letter(col_idx)].width = adjusted_width
            
            # Apply borders to data cells
            for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
                for cell in row:
                    cell.border = border
                    cell.alignment = Alignment(vertical='center')

        output.seek(0)
        return output

    except Exception as e:
        logger.error(f"Error generating Excel sheet: {e}", exc_info=True)
        # Create a minimal error sheet
        error_df = pd.DataFrame([{
            'child part': 'ERROR',
            'REMARK': f'Excel generation failed: {str(e)}'
        }], columns=columns)
        error_output = io.BytesIO()
        error_df.to_excel(error_output, sheet_name='FETCH FROM DRAWING', index=False)
        error_output.seek(0)
        return error_output