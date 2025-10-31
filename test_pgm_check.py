import streamlit as st
import pandas as pd
import numpy as np
import re
import os

def calc_si(txt_a: str, txt_b: str, op: str = "/") -> str:
    """
    Calculate the result of txt_a <op> txt_b where txt_a and txt_b
    are strings like '80u', '100m', etc.
    Automatically scales the result to keep it between 1 and 999,
    with an appropriate SI prefix, formatted to 4 significant digits.
    """   

    # Define SI prefixes from smallest to largest
    si_prefix = [
        ('p', 1e-12),
        ('n', 1e-9),
        ('u', 1e-6),
        ('m', 1e-3),
        ('', 1.0),
        ('k', 1e3),
        ('M', 1e6),
        ('G', 1e9)
    ]
    prefix_dict = dict(si_prefix)

    # Helper: parse '80u' → 80 × 1e-6
    def parse_value(s):
        if isinstance(s, (int, float)):
            return float(s)
        match = re.match(r"([\d.]+)\s*([pnumkMG]?)", str(s).strip())
        if not match:
            raise ValueError(f"Invalid token: {s}")
        num, prefix = match.groups()
        return float(num) * prefix_dict.get(prefix, 1.0)

    # Convert inputs
    a = parse_value(txt_a)
    b = parse_value(txt_b)

    # Perform operation
    if op == "/":
        result_value = a / b
    elif op == "*":
        result_value = a * b
    elif op == "+":
        result_value = a + b
    elif op == "-":
        result_value = a - b
    else:
        raise ValueError(f"Unsupported operator: {op}")

    # Auto-scale to 1 <= scaled < 1000
    abs_val = abs(result_value)
    chosen_prefix, chosen_factor = '', 1.0
    for prefix, factor in si_prefix:
        scaled = abs_val / factor
        if 1 <= scaled < 1000:  # ✅ max 3 integer digits
            chosen_prefix, chosen_factor = prefix, factor
            break
    else:
        # If very small, use smallest prefix; if huge, use largest
        if abs_val < 1e-12:
            chosen_prefix, chosen_factor = 'p', 1e-12
        else:
            chosen_prefix, chosen_factor = 'G', 1e9

    scaled_val = result_value / chosen_factor

    # Format with 4 significant digits
    return f"{scaled_val:.4g}{chosen_prefix}"
    
def filter_spec_columns(df_tests):
    """
    Clean the test DataFrame to keep only the columns relevant 
    for spec correlation:
        ItemName, Limit-L, Limit-H, Bias1, Bias2, RV.

    - Keeps only those columns if they exist.
    - Creates empty columns if missing.
    - Reorders them consistently.

    Args:
        df_tests (pd.DataFrame): Original test dataframe.

    Returns:
        pd.DataFrame: Filtered dataframe ready for spec correlation.
    """

    
    # Desired final structure
    keep_cols = ["Sequence", "ItemName", "Limit-L", "Limit-H", "Bias1", "Bias2", "RV"]

    # --- Keep only existing columns
    filtered_df = df_tests[[col for col in keep_cols if col in df_tests.columns]].copy()

    # --- Add any missing columns as empty
    for col in keep_cols:
        if col not in filtered_df.columns:
            filtered_df[col] = ""

    # --- Reorder to ensure consistent output
    filtered_df = filtered_df[keep_cols]
    
    return filtered_df



def correlate_spec_with_validspec(df_tests, spec_path, df_sorts=None):
    """
    Correlate Original Test Data (df_tests) with Spec Draft CSV (spec_path).
    Checks Seq-prefixed columns, including RV, and validates limits and biases.

    Args:
        df_tests (pd.DataFrame): Original test data.
        spec_path (str): Path to the spec CSV.
        df_sorts (pd.DataFrame, optional): Sort/extra data, if needed.

    Returns:
        List[dict]: Validation issues.
    """
    df_tests = filter_spec_columns(df_tests)
    
    # --- Load spec CSV ---
    try:
        valid_spec = pd.read_csv(spec_path)
    except Exception as e:
        return [{
            "ItemName": "",
            "Parameter": "",
            "SpecValue": "",
            "TestValue": "",
            "Status": "FAIL",
            "Reason": f"Failed to load spec: {e}"
        }]

    errors = []
    
    def _to_float(val):
        """Convert string with engineering suffix (p, n, u, m, k, M) to float."""
        if pd.isna(val) or val == '':
            return np.nan

        s = str(val).strip().lower()

        # Handle negative sign and remove unwanted chars
        s = s.replace(' ', '')

        # Engineering multipliers
        multipliers = {
            'p': 1e-12,
            'n': 1e-9,
            'u': 1e-6,
            'm': 1e-3,
            'k': 1e3,
            'meg': 1e6,
            'g': 1e9
        }

        try:
            # Check if ends with known suffix
            for suf, mult in multipliers.items():
                if s.endswith(suf):
                    num = float(s[:-len(suf)])
                    return num * mult
            return float(s)  # plain number
        except:
            return np.nan    

    # --- Fields to validate ---
    field_pairs = [
        ('ItemName', 'SeqItemName'),
        ('Limit-L', 'SeqLimit-L'),
        ('Limit-H', 'SeqLimit-H'),
        ('Bias1', 'SeqBias1'),
        ('Bias2', 'SeqBias2'),
        ('RV', 'SeqRV')
    ]

    # --- Iterate through spec ---
    for _, spec_row in valid_spec.iterrows():
        item_name = spec_row.get('ItemName', '')

        for field, seq_field in field_pairs:
            seq_value = spec_row.get(seq_field, '')
            if pd.isna(seq_value) or seq_value == '':
                continue  # Skip if no sequence mapping

            # Find corresponding row in original test data
            test_row = df_tests.loc[df_tests['Sequence'] == seq_value]
            if test_row.empty:
                errors.append({
                    "ItemName": item_name,
                    "Parameter": field,
                    "SpecValue": spec_row.get(field, ''),
                    "TestValue": None,
                    "Status": "FAIL",
                    "Reason": f"Sequence {seq_value} not found in Original Test Data"
                })
                continue

            test_val = test_row[field].values[0] if field in test_row.columns else None
            spec_val = spec_row.get(field, '')

            if field in ['Limit-L', 'Limit-H']:
                test_num = _to_float(test_val)
                spec_num = _to_float(spec_val)
                if np.isnan(test_num) or np.isnan(spec_num):
                    continue
                if field == 'Limit-L' and test_num < spec_num:
                    errors.append({
                        "ItemName": item_name,
                        "Parameter": field,
                        "SpecValue": spec_val,
                        "TestValue": test_val,
                        "Status": "FAIL",
                        "Reason": f"Lower limit too loose ({test_num} < {spec_num})"
                    })
                elif field == 'Limit-H' and test_num > spec_num:
                    errors.append({
                        "ItemName": item_name,
                        "Parameter": field,
                        "SpecValue": spec_val,
                        "TestValue": test_val,
                        "Status": "FAIL",
                        "Reason": f"Upper limit too loose ({test_num} > {spec_num})"
                    })
            else:

                # --- Smart comparison for strings, numbers, or suffixed values ---
                test_is_empty = pd.isna(test_val) or str(test_val).strip() == ''
                spec_is_empty = pd.isna(spec_val) or str(spec_val).strip() == ''

                if test_is_empty and spec_is_empty:
                    continue  # Both empty → OK

                # Try numeric comparison (including suffix-aware)
                test_num = _to_float(test_val)
                spec_num = _to_float(spec_val)

                if not np.isnan(test_num) and not np.isnan(spec_num):
                    if np.isclose(test_num, spec_num, atol=1e-12, rtol=1e-6):
                        continue  # Numeric match → OK

                # Fallback: raw string comparison
                if str(test_val).strip().lower() != str(spec_val).strip().lower():
                    errors.append({
                        "ItemName": item_name,
                        "Parameter": field,
                        "SpecValue": spec_val,
                        "TestValue": test_val,
                        "Status": "FAIL",
                        "Reason": f"Mismatch: expected {spec_val}, got {test_val}"
                    })
                
    return errors


def validate_bias_lowvolt_for_special_items(df_tests, df_sorts):

    """
    Validate Bias limits for specific ItemNames:
      - For ['VFEC', 'VFBE', 'VCESAT', 'VBESAT', 'IDON', 'RDON', 'VFSD', 'VDSON', 'VFGS']: Bias1 <= 20
      - For ['HFE', 'BTON']: Bias2 <= 20
    Returns a list of errors if any violations are found.
    """

    errors = []

    # Define item groups
    bias1_items = ['VFEC', 'VFBE', 'VCESAT', 'VBESAT', 'IDON', 'RDON', 'VFSD', 'VDSON', 'VFGS']
    bias2_items = ['HFE', 'BTON']

    # --- Safely convert Bias columns to numeric ---
    df_tests['Bias1'] = pd.to_numeric(df_tests.get('Bias1', 0), errors='coerce')
    df_tests['Bias2'] = pd.to_numeric(df_tests.get('Bias2', 0), errors='coerce')

    # --- Check Bias1 rule ---
    condition_bias1 = df_tests['ItemName'].isin(bias1_items)
    invalid_bias1 = df_tests[condition_bias1 & (df_tests['Bias1'] > 20)]

    for _, row in invalid_bias1.iterrows():
        errors.append({
            'ItemName': row['ItemName'],
            'Bias1': row['Bias1'],
            'Error': f"{row['ItemName']} has Bias1={row['Bias1']} which exceeds the limit of 20A"
        })

    # --- Check Bias2 rule ---
    condition_bias2 = df_tests['ItemName'].isin(bias2_items)
    invalid_bias2 = df_tests[condition_bias2 & (df_tests['Bias2'] > 20)]

    for _, row in invalid_bias2.iterrows():
        errors.append({
            'ItemName': row['ItemName'],
            'Bias2': row['Bias2'],
            'Error': f"{row['ItemName']} has Bias2={row['Bias2']} which exceeds the limit of 20A"
        })

    return errors


def validate_or_logic_contains_all_tests(df_tests, df_sorts):
    """
    Validate that for all rows where LogicCondition == "OR", the combined TestN columns
    contain all integers from 1 to max(Sequence) in df_tests.
    
    Returns a list of error messages.
    """
    errors = []

    if df_tests is None or df_sorts is None:
        errors.append("Test or Sort dataframe missing.")
        return errors

    if "Sequence" not in df_tests.columns:
        errors.append("'Sequence' column not found in test dataframe.")
        return errors

    if "LogicCondition" not in df_sorts.columns:
        errors.append("'LogicCondition' column not found in sort dataframe.")
        return errors

    max_sequence = df_tests["Sequence"].max()

    # Filter rows where LogicCondition == "OR"
    or_rows = df_sorts[df_sorts["LogicCondition"] == "OR"]

    if or_rows.empty:
        errors.append("No rows with LogicCondition == 'OR' found in sort dataframe.")
        return errors

    # Collect all TestN column names (e.g., Test1, Test2, ...)
    test_cols = [col for col in df_sorts.columns if col.startswith("Test")]

    # Collect all unique test numbers in OR rows from these columns
    found_tests = set()

    for _, row in or_rows.iterrows():
        for col in test_cols:
            val = row[col]
            # Check if val is a number and within valid range
            if pd.notna(val):
                try:
                    num = int(val)
                    found_tests.add(num)
                except (ValueError, TypeError):
                    # Ignore non-integer or invalid values
                    pass

    # Now check if all numbers from 1 to max_sequence are present
    missing = [n for n in range(1, max_sequence + 1) if n not in found_tests]

    if missing:
        errors.append(
            f"Missing test numbers in 'OR' LogicCondition rows: {missing}"
        )

    return errors


def validate_logiccondition_or_except_special(df_tests, df_sorts):
    """
    Validate that all LogicCondition values are either 'OR', 'OSC', 'REJECT', or 'ALL PASS'.
    Except 'OSC', 'REJECT', 'ALL PASS' rows, all other rows must be 'OR'.
    Returns list of errors.
    """
    errors = []

    if df_sorts is None:
        errors.append("Sort data is missing.")
        return errors

    if "LogicCondition" not in df_sorts.columns:
        errors.append("'LogicCondition' column not found in sort dataframe.")
        return errors

    allowed_special = {"OSC", "REJECT", "ALL PASS"}

    for idx, val in df_sorts["LogicCondition"].items():
        if val not in allowed_special and val != "OR":
            errors.append(f"Row {idx+1}: LogicCondition '{val}' is invalid; must be 'OR' or one of {allowed_special}.")

    return errors

def validate_logiccondition_all_pass_once(df_tests, df_sorts, expected_bin_number):
    """
    Validate that 'LogicCondition' column contains exactly one 'ALL PASS' row,
    and its 'BinNumber' matches the expected_bin_number.
    
    Parameters:
    - df_tests: test dataframe (not used here but kept for interface consistency)
    - df_sorts: sort dataframe
    - expected_bin_number: the required value in the 'BinNumber' column for the 'ALL PASS' row
    
    Returns a list of errors (empty if valid).
    """
    errors = []

    if df_sorts is None:
        errors.append("Sort data is missing.")
        return errors

    if "LogicCondition" not in df_sorts.columns:
        errors.append("'LogicCondition' column not found in sort dataframe.")
        return errors

    if "BinNumber" not in df_sorts.columns:
        errors.append("'BinNumber' column not found in sort dataframe.")
        return errors

    all_pass_rows = df_sorts[df_sorts["LogicCondition"] == "ALL PASS"]

    count_all_pass = len(all_pass_rows)

    if count_all_pass == 0:
        errors.append("No 'ALL PASS' row found in 'LogicCondition'; exactly one required.")
        return errors
    elif count_all_pass > 1:
        errors.append(f"Multiple ('{count_all_pass}') 'ALL PASS' rows found in 'LogicCondition'; exactly one required.")
        return errors

    # Exactly one ALL PASS row exists; check BinNumber
    bin_number = all_pass_rows.iloc[0]["BinNumber"]

    if bin_number != expected_bin_number:
        errors.append(f"'ALL PASS' row 'BinNumber' is '{bin_number}', but expected '{expected_bin_number}'.")

    return errors

def validate_logiccondition_all_pass_once1(df_tests, df_sorts):
    """
    Validate that 'LogicCondition' column contains exactly one 'ALL PASS' value.
    Returns a list of errors (empty if valid).
    """
    errors = []

    if df_sorts is None:
        errors.append("Sort data is missing.")
        return errors

    if "LogicCondition" not in df_sorts.columns:
        errors.append("'LogicCondition' column not found in sort dataframe.")
        return errors

    all_pass_count = (df_sorts["LogicCondition"] == "ALL PASS").sum()

    if all_pass_count == 0:
        errors.append("No 'ALL PASS' row found in 'LogicCondition'; exactly one required.")
    elif all_pass_count > 1:
        errors.append(f"Multiple ('{all_pass_count}') 'ALL PASS' rows found in 'LogicCondition'; exactly one required.")

    return errors

def validate_logiccondition_reject_once(df_tests, df_sorts):
    """
    Validate that 'LogicCondition' column contains exactly one 'REJECT' value.
    Returns a list of errors (empty if valid).
    """
    errors = []

    # Handle if df_sorts is None or no 'LogicCondition' column
    if df_sorts is None:
        errors.append("Sort data is missing.")
        return errors

    if "LogicCondition" not in df_sorts.columns:
        errors.append("'LogicCondition' column not found in sort dataframe.")
        return errors

    reject_count = (df_sorts["LogicCondition"] == "REJECT").sum()

    if reject_count == 0:
        errors.append("No 'REJECT' row found in 'LogicCondition'; exactly one required.")
    elif reject_count > 1:
        errors.append(f"Multiple ('{reject_count}') 'REJECT' rows found in 'LogicCondition'; exactly one required.")

    return errors



def validate_logiccondition_osc_once(df_tests, df_sorts):
    """
    Validates that the 'LogicCondition' column contains exactly one row with value 'OSC'.
    Returns a list with an error message if zero or multiple rows found, empty list if valid.
    """
    errors = []

    if "LogicCondition" not in df_sorts.columns:
        errors.append("Column 'LogicCondition' not found in dataframe.")
        return errors

    osc_count = (df_sorts["LogicCondition"] == "OSC").sum()

    if osc_count == 0:
        errors.append("No rows with LogicCondition == 'OSC' found. Exactly one required.")
    elif osc_count > 1:
        errors.append(f"Multiple rows ({osc_count}) with LogicCondition == 'OSC' found. Exactly one required.")

    return errors


def check_passbranch_all_zero(df_tests, df_sorts):
    """
    Checks if all rows in 'PassBranch' are zero (int or string "0").
    Returns a list of errors for rows where 'PassBranch' is not zero.
    """
    errors = []

    if "PassBranch" not in df_tests.columns:
        raise ValueError("Column 'PassBranch' not found in dataframe.")

    for idx, val in df_tests["PassBranch"].items():
        # Try converting to numeric, errors='coerce' converts invalid to NaN
        num_val = pd.to_numeric(val, errors='coerce')

        # Check if num_val is exactly 0
        if not (num_val == 0):
            item = df_tests.at[idx, "ItemName"] if "ItemName" in df_tests.columns else ""
            errors.append(f"Row {idx}: PassBranch ({val}) is not zero for item '{item}'")

    return errors



def check_failbranch_uniform(df_tests, df_sorts):
    """
    Checks if all rows in 'FailBranch' have the same value.
    Returns a list of errors for rows where 'FailBranch' differs from the common value.
    If all are same or all missing, returns empty list.
    """
    errors = []

    if "FailBranch" not in df_tests.columns:
        raise ValueError("Column 'FailBranch' not found in dataframe.")

    # Drop missing values for comparison
    fail_values = df_tests["FailBranch"].dropna()

    if fail_values.empty:
        # No valid FailBranch values to check
        return errors

    # Get the first FailBranch value to compare others against
    common_value = fail_values.iloc[0]

    # Iterate and check each FailBranch value
    for idx, val in df_tests["FailBranch"].items():
        if pd.isna(val):
            continue  # Skip missing values
        if val != common_value:
            item = df_tests.at[idx, "ItemName"] if "ItemName" in df_tests.columns else ""
            errors.append(f"Row {idx}: FailBranch ({val}) does not match common value ({common_value}) for item '{item}'")

    return errors



def check_failbranch_vs_sequence(df_tests, df_sorts):
    """
    Checks if each row's FailBranch value is greater than the maximum Sequence value.
    Returns a list of error messages for rows that do NOT satisfy this condition.
    """
    errors = []

    # Ensure required columns exist
    if "FailBranch" not in df_tests.columns or "Sequence" not in df_tests.columns:
        raise ValueError("Required columns 'FailBranch' and/or 'Sequence' not found in dataframe.")

    # Compute the maximum Sequence value (ignore NaN)
    max_sequence = df_tests["Sequence"].max()

    # Iterate through each row
    for idx, row in df_tests.iterrows():
        fail_branch = row.get("FailBranch", None)
        item = row.get("ItemName", "")

        # Skip rows where FailBranch is missing or not numeric
        if pd.isna(fail_branch):
            continue

        try:
            fail_branch_value = float(fail_branch)
        except ValueError:
            errors.append(f"Row {idx}: FailBranch ('{fail_branch}') is not numeric for item '{item}'")
            continue

        # ✅ Valid condition: fail_branch_value > max_sequence
        # ❌ Error if not greater
        if not (fail_branch_value > max_sequence):
            errors.append(
                f"Row {idx}: FailBranch ({fail_branch_value}) <= max Sequence ({max_sequence}) for item '{item}'"
            )

    return errors


def check_cb2_all_B(df_tests, df_sorts):
    """
    Checks if all 'C/B2' column values are 'B'.
    Returns a list of error messages for rows that fail.
    """
    errors = []

    # Loop through each row
    for idx, row in df_tests.iterrows():
        item = row.get("ItemName", "")
        cb2_value = row.get("C/B2", None)

        # If the column value is not 'B', log an error
        if cb2_value != "B":
            errors.append(
                f"Row {idx}: C/B2 ({cb2_value}) is not 'Branch' for item '{item}'"
            )

    return errors

def validate_bv_bias2_gt_limith(df_tests, df_sorts):
    """
    Check that for rows where ItemName starts with 'BV',
    Bias2 must be greater than Limit-H.

    Returns:
        A list of error messages with row numbers where the rule is violated.
    """
    # Helper to parse values like '250.0u', '3.0m', etc.
    unit_multipliers = {
        'p': 1e-12,
        'n': 1e-9,
        'u': 1e-6,
        'm': 1e-3,
        '': 1,
        'k': 1e3,
        'K': 1e3,
        'M': 1e6
    }

    def parse_numeric(value):
        if pd.isna(value) or value == "":
            return np.nan
        value = str(value).strip()
        match = re.fullmatch(r"([-+]?[0-9]*\.?[0-9]+)([a-zA-Z]?)", value)
        if not match:
            return np.nan
        number, unit = match.groups()
        multiplier = unit_multipliers.get(unit, None)
        if multiplier is None:
            return np.nan
        return float(number) * multiplier

    # Collect errors
    errors = []

    # Define exact item names to match
    item_names_to_check = ('BVCBO', 'BVCBR', 'BVCBS', 'BVCEO', 'BVCER', 'BVCES', 'BVDG1O','BVDG2O', 
                           'BVDGO', 'BVDGS', 'BVDGSS', 'BVDSO', 'BVDSS', 'BVDSX1', 'BVDSX2', 'BVDSX3',
                           'BVDSXX', 'BVEB', 'BVIN', 'BVOUT', 'BVSG1O', 'BVSG1S', 'BVSG1O', 'BVSG2O',
                           'BVSGO', 'BVSGS', 'HIDSS', 'HILCBO', 'HILCBR', 'HILCBS', 'HILCEO', 'HILCER',
                           'HILCES', 'HILEB', 'HVBCBO', 'HVBCBR', 'HVBCBS', 'HVBCEO', 'HVBCER', 'HVBCES',
                           'HVICBO', 'HVICBR', 'HVICBS', 'HICEO', 'HVICER', 'HVICES', 'HVIR', 'HVVR',
                           'IBD', 'ICBO', 'ICBR', 'ICBS', 'ICEO', 'ICER', 'ICES', 'IDGO', 'IDRM', 'IDRM2',
                           'IDRM3', 'IDSO', 'IDSS', 'IDSX1S', 'IDSX2S', 'IDSXX', 'IEB', 'IGG', 'IGSX', 'IR', 
                           'IRGM', 'IRIN', 'IROUT', 'ISG1S', 'ISG2S', 'ISGO', 'ISGS', 'POLA', 'PVCBO', 
                           'PVCEO', 'VDRM', 'VDRM2', 'VDRM3', 'VRGM', 'VRGM2', 'VRGM3', 'VZ', 'ZZ', 'HVBDSO',
                           'HVBDSS', 'IGN', 'IGR', 'HVBDGO', 'HVBDGS', 'HVIDSO', 'HVIDSS', 'HVIDGO', 'HVDRM',
                           'HVIDRM', 'PVCES', 'ISG25', 'ISG15', 'BVSG13', 'BVSG15', 'BVSG23', 'BVSG25',
                           'HVIEB', 'HVBEB', 'VBRDS', 'IFIN', 'IDSS8', 'ISG28', 'BVDSS8', 'IDSX28',
                           'BVDSS9', 'IOMAX', 'ZZBC', 'IDSXX3', 'VSG2OR', 'VSG2SR', 'IOFF', 'IMIN', 'ZAK',
                           'VKARH1', 'VKARH2')  # Exact matches only

    for idx, row in df_tests.iterrows():
        item = str(row.get('ItemName', '')).strip()  # Clean whitespace if needed
    
        if item in item_names_to_check:
            bias2 = parse_numeric(row.get('Bias2'))
            limit_h = parse_numeric(row.get('Limit-H'))
        
            #if np.isnan(bias2) or np.isnan(limit_h):
            #    errors.append(f"Row {idx}: Invalid Bias2 or Limit-H format.")
            if bias2 <= limit_h:
                errors.append(f"Row {idx}: Bias2 ({row['Bias2']}) <= Limit-H ({row['Limit-H']}) for item '{item}'")


    return errors

def decode_value_with_suffix(raw_value, suffix_code):
    suffix_map = {
        0:  (1, 'p'),
        1:  (100, 'n'),        2:  (10, 'n'),        3:  (1, 'n'),
        4:  (100, 'u'),        5:  (10, 'u'),        6:  (1, 'u'),
        7:  (100, 'm'),        8:  (10, 'm'),        9:  (1, 'm'),
        10: (100, ''),        11: (10, ''),        12: (1, ''),
        13: (100, 'k'),        14: (10, 'k'),        15: (1, 'k'),
    }
    divisor, suffix = suffix_map.get(suffix_code, (1, ''))
    decoded_value = raw_value / divisor
    return f"{decoded_value}{suffix}"


def decode_limit_with_suffix(raw_value, suffix_code):
    suffix_map = {
        0:  (10, 'p'),
        1:  (1000, 'n'),        2:  (100, 'n'),        3:  (10, 'n'),
        4:  (1000, 'u'),        5:  (100, 'u'),        6:  (10, 'u'),
        7:  (1000, 'm'),        8:  (100, 'm'),        9:  (10, 'm'),
        10: (1000, ''),        11: (100, ''),        12: (10, ''),
        13: (1000, 'k'),        14: (100, 'k'),        15: (10, 'k'),
    }
    divisor, suffix = suffix_map.get(suffix_code, (1, ''))
    decoded_value = raw_value / divisor
    return f"{decoded_value}{suffix}"


code_name_map = {
    (0x00, 0x00): 'BVCEO',    (0x00, 0x01): 'BVCES',    (0x00, 0x02): 'BVCER',
    (0x00, 0x08): 'VBRDS',    (0x01, 0x00): 'BVCBO',    (0x01, 0x01): 'BVCBS',
    (0x01, 0x02): 'BVCBR',    (0x01, 0x0F): 'ZZBC',    (0x02, 0x00): 'BVEB',
    (0x03, 0x00): 'VFBC',    (0x03, 0x0F): 'RM(1)',    (0x04, 0x00): 'VFBE',
    (0x04, 0x04): 'SPCONT',    (0x04, 0x08): 'VFDI',    (0x04, 0x0A): 'VI-',
    (0x04, 0x0C): 'RM(4)',    (0x04, 0x0D): 'RM(3)',    (0x04, 0x0E): 'ROI',
    (0x04, 0x0F): 'VI+',    (0x05, 0x00): 'VFEC',    (0x05, 0x01): 'VFECS',
    (0x05, 0x08): 'VDSF',    (0x05, 0x0E): 'RII',    (0x05, 0x0F): 'RM(2)',
    (0x06, 0x00): 'VCESAT',    (0x06, 0x08): 'VDSONF',    (0x06, 0x0F): 'VCESAF',
    (0x07, 0x00): 'VBESAT',    (0x08, 0x00): 'IB',    (0x08, 0x09): 'LINEIA',
    (0x08, 0x0A): 'LOADIA',    (0x08, 0x0B): 'IADJIH',    (0x08, 0x0C): 'IADJ',
    (0x08, 0x0D): 'IBN-',    (0x08, 0x0E): 'IG-',    (0x08, 0x0F): 'IG+',
    (0x09, 0x00): 'BTON',    (0x09, 0x09): 'LINERA',    (0x09, 0x0A): 'LOADRA',
    (0x09, 0x0D): 'VREFIH',    (0x09, 0x0E): 'VOUTA',    (0x09, 0x0F): 'VREF',
    (0x0A, 0x00): 'HFE',    (0x0A, 0x07): 'HFE7',    (0x0B, 0x00): 'HHFE',
    (0x0C, 0x00): 'ICEO',    (0x0C, 0x01): 'ICES',    (0x0C, 0x02): 'ICER',
    (0x0C, 0x0B): 'VSCE',    (0x0D, 0x00): 'ICBO',    (0x0D, 0x01): 'ICBS',
    (0x0D, 0x02): 'ICBR',    (0x0D, 0x0B): 'VSCB',    (0x0E, 0x00): 'IEB',
    (0x0E, 0x0B): 'VSEB',    (0x0F, 0x00): 'CONT',    (0x0F, 0x0D): 'KB',
    (0x0F, 0x0E): 'KC',    (0x0F, 0x0F): 'KE',    (0x10, 0x00): 'HVBCEO',
    (0x10, 0x01): 'HVBCES',    (0x10, 0x02): 'HVBCER',    (0x11, 0x00): 'HVBCBO',
    (0x11, 0x01): 'HVBCBS',    (0x11, 0x02): 'HVBCBR',    (0x12, 0x00): 'HBTON',
    (0x12, 0x0D): 'HVP-',    (0x12, 0x0F): 'HVP',    (0x13, 0x00): 'ICS',
    (0x14, 0x00): 'IR@',    (0x15, 0x00): 'DVDS',    (0x16, 0x00): 'DIFVBE',
    (0x17, 0x00): 'VPT',    (0x18, 0x00): 'DVBE',    (0x19, 0x00): 'HIB',
    (0x1A, 0x00): 'VOUT',    (0x1A, 0x04): 'VOUTLI',    (0x1A, 0x08): 'VDROP',
    (0x1A, 0x09): 'LINERG',    (0x1A, 0x0A): 'LOADRG',    (0x1A, 0x0B): 'VOUT11',
    (0x1A, 0x0C): 'VOUT12',    (0x1A, 0x0D): 'VOUT13',    (0x1A, 0x0E): 'VOUT14',
    (0x1A, 0x0F): 'VOUTS',    (0x1B, 0x07): 'DVGS2',    (0x1C, 0x00): 'HILCEO',
    (0x1C, 0x01): 'HILCES',    (0x1C, 0x02): 'HILCER',    (0x1C, 0x08): 'IF',
    (0x1C, 0x09): 'IRDI',    (0x1C, 0x0E): 'RIV',    (0x1C, 0x0F): 'IOMAX',
    (0x1D, 0x00): 'HILCBO',    (0x1D, 0x01): 'HILCBS',    (0x1D, 0x02): 'HILCBR',
    (0x1E, 0x00): 'HVICEO',    (0x1E, 0x01): 'HVICES',    (0x1E, 0x02): 'HVICER',
    (0x1F, 0x00): 'HVICBO',    (0x1F, 0x01): 'HVICBS',    (0x1F, 0x02): 'HVICBR',
    (0x20, 0x00): 'BVCEX',    (0x20, 0x0F): 'BVCEX-',    (0x21, 0x00): 'HCESAT',
    (0x21, 0x01): 'HVFECS',    (0x21, 0x09): 'HVFEC',    (0x22, 0x00): 'HBESAT',
    (0x22, 0x08): 'HVFBE8',    (0x22, 0x09): 'HVFBE',    (0x22, 0x0E): 'HVGFM',
    (0x22, 0x0F): 'HVFGS',    (0x23, 0x00): 'IQ',    (0x23, 0x08): 'IQ8',
    (0x23, 0x09): 'LINEIQ',    (0x23, 0x0A): 'LOADIQ',    (0x24, 0x00): 'IDSX2S',
    (0x24, 0x08): 'IDSX28',    (0x25, 0x00): 'BVDG2O',    (0x26, 0x00): 'HVTM',
    (0x27, 0x00): 'SCAN',    (0x27, 0x07): 'EXT',    (0x28, 0x00): 'HRDON',
    (0x28, 0x0D): 'HRDON-',    (0x29, 0x00): 'PSCAN',    (0x29, 0x01): 'BITST1',
    (0x29, 0x02): 'BITST2',    (0x29, 0x03): 'BITST3',    (0x29, 0x04): 'BITST4',
    (0x29, 0x05): 'BITST5',    (0x29, 0x06): 'BITST6',    (0x29, 0x0B): 'PSCANC',
    (0x29, 0x0C): 'PSCANB',    (0x29, 0x0D): 'PSCANE',    (0x29, 0x0E): 'BITCL2',
    (0x29, 0x0F): 'BITCLR',    (0x2A, 0x07): 'EXTY',    (0x2A, 0x0E): 'COFF',
    (0x2A, 0x0F): 'CON',    (0x2B, 0x00): 'AUXZ0',    (0x2B, 0x01): 'AUXZ1',
    (0x2B, 0x07): 'EXTZ',    (0x2B, 0x0F): 'PASS#',    (0x2C, 0x00): 'ICEX',
    (0x2C, 0x0F): 'ICEX-',    (0x2D, 0x07): 'SOA',    (0x2E, 0x07): 'DVBE2',
    (0x2F, 0x00): 'ICCLV',    (0x35, 0x00): 'VFSD',    (0x35, 0x01): 'VFSDS',
    (0x38, 0x03): 'VG1SR',    (0x39, 0x00): 'ADD',    (0x3A, 0x00): 'MULTI',
    (0x3B, 0x00): 'SAME',    (0x3C, 0x00): 'ABSDEL',    (0x3D, 0x00): 'DEF',
    (0x3E, 0x00): 'DIVID',    (0x3E, 0x01): 'CMPDIV',    (0x3F, 0x00): 'QTY',
    (0x3F, 0x01): 'TPCNT',    (0x40, 0x00): 'BVDSO',    (0x40, 0x01): 'BVDSS',
    (0x40, 0x08): 'BVDSS8',    (0x40, 0x09): 'BVDSS9',    (0x41, 0x00): 'BVDGO',
    (0x41, 0x01): 'BVDGS',    (0x42, 0x00): 'BVSGO',    (0x42, 0x05): 'BVSGS',
    (0x43, 0x00): 'VFGD',    (0x44, 0x00): 'VFGS',    (0x44, 0x06): 'VTH',
    (0x45, 0x00): 'HGMP',    (0x45, 0x0D): 'HGMP-',    (0x46, 0x00): 'RVSAT',
    (0x47, 0x00): 'BVDSX1',    (0x48, 0x00): 'BVDSX2',    (0x49, 0x00): 'VP',
    (0x49, 0x08): 'VP8',    (0x49, 0x0E): 'VP-',    (0x49, 0x0F): 'VP+',
    (0x4A, 0x00): 'RHFE',    (0x4B, 0x00): 'BVDSX3',    (0x4C, 0x00): 'IDSO',
    (0x4C, 0x01): 'IDSS',    (0x4C, 0x08): 'IDSS8',    (0x4D, 0x00): 'IDGO',
    (0x4E, 0x00): 'ISGO',    (0x4E, 0x05): 'ISGS',    (0x4F, 0x00): 'HILEB',
    (0x4F, 0x0F): 'ROV',    (0x50, 0x07): 'DVDS2',    (0x51, 0x00): 'BVSG1S',
    (0x52, 0x00): 'BVSG2S',    (0x52, 0x08): 'VSG2SR',    (0x53, 0x00): 'BVSG2O',
    (0x53, 0x01): 'BVSG23',    (0x53, 0x05): 'BVSG25',    (0x53, 0x08): 'VSG2OR',
    (0x54, 0x00): 'BVDGSS',    (0x55, 0x00): 'IGG',    (0x56, 0x00): 'ISG1S',
    (0x56, 0x05): 'ISG15',    (0x57, 0x00): 'ISG2S',    (0x57, 0x05): 'ISG25',
    (0x57, 0x08): 'ISG28',    (0x58, 0x00): 'GMI',    (0x59, 0x00): 'VG1SC+',
    (0x59, 0x0F): 'VG1SC-',    (0x5A, 0x00): 'GMP',    (0x5B, 0x00): 'VG2SC+',
    (0x5B, 0x0F): 'VG2SC-',    (0x5C, 0x01): 'HIDSS',    (0x5D, 0x00): 'IDSX1S',
    (0x5E, 0x03): 'VOUTH',    (0x5F, 0x00): 'GMI1',    (0x60, 0x00): 'BVDSX',
    (0x60, 0x0F): 'BVDSX-',    (0x61, 0x00): 'VDSON1',    (0x62, 0x00): 'VDSON2',
    (0x62, 0x08): 'VDSO28',    (0x63, 0x00): 'VDSON',    (0x63, 0x08): 'VDSON8',
    (0x63, 0x0A): 'VDSO10',    (0x63, 0x0F): 'VDSON-',    (0x64, 0x00): 'BVOUT',
    (0x65, 0x00): 'BVDSXX',    (0x66, 0x00): 'BVDG1O',    (0x67, 0x00): 'VV+',
    (0x67, 0x0D): 'VV-',    (0x67, 0x0E): 'VD',    (0x67, 0x0F): 'VA',
    (0x68, 0x00): 'GMV',    (0x68, 0x0F): 'GMV-',    (0x69, 0x00): 'BVSG1O',
    (0x69, 0x01): 'BVSG13',    (0x69, 0x05): 'BVSG15',    (0x6A, 0x00): 'DVCE',
    (0x6B, 0x00): 'DVF',    (0x6C, 0x00): 'IDSX',    (0x6C, 0x0D): 'IDSXN',
    (0x6C, 0x0F): 'IDSX-',    (0x6D, 0x02): 'IDSON',    (0x6E, 0x00): 'IDON',
    (0x6E, 0x03): 'SWON',    (0x6E, 0x08): 'IDON8',    (0x6E, 0x0F): 'IDON-',
    (0x6F, 0x00): 'IGSX',    (0x70, 0x00): 'VFIN',    (0x71, 0x00): 'IRIN',
    (0x73, 0x00): 'IROUT',    (0x74, 0x00): 'VONF',    (0x75, 0x00): 'IFTF',
    (0x75, 0x03): 'IFTF3',    (0x75, 0x08): 'IFT',    (0x76, 0x00): 'IHF',
    (0x76, 0x03): 'IHF3',    (0x77, 0x00): 'ICCL',    (0x78, 0x00): 'CTR',
    (0x79, 0x00): 'VOL',    (0x7A, 0x00): 'IFTON',    (0x7B, 0x00): 'VONS',
    (0x7C, 0x00): 'NOTCHF',    (0x7C, 0x0F): 'NOCHFH',    (0x7D, 0x00): 'IFTS',
    (0x7D, 0x03): 'IFTS3',    (0x7E, 0x00): 'IHS',    (0x7E, 0x02): 'IHS2',
    (0x7E, 0x03): 'IHS3',    (0x7F, 0x00): 'IDSXX',    (0x7F, 0x03): 'IDSXX3',
    (0x80, 0x00): 'VDRM',    (0x80, 0x02): 'VDRM2',    (0x80, 0x03): 'VDRM3',
    (0x81, 0x00): 'IGN',    (0x82, 0x00): 'VRGM',    (0x82, 0x02): 'VRGM2',
    (0x82, 0x03): 'VRGM3',    (0x83, 0x00): 'IGR',    (0x84, 0x00): 'VGFM',
    (0x85, 0x00): 'ZZL',    (0x85, 0x06): 'ZAKL',    (0x86, 0x00): 'HVBDSO',
    (0x86, 0x01): 'HVBDSS',    (0x87, 0x00): 'HVBDGO',    (0x87, 0x01): 'HVBDGS',
    (0x88, 0x00): 'HVIDSO',    (0x88, 0x01): 'HVIDSS',    (0x89, 0x00): 'HVIDGO',
    (0x8A, 0x00): 'C',    (0x8A, 0x0D): 'DSUB',    (0x8A, 0x0E): 'N',
    (0x8A, 0x0F): 'SUB',    (0x8B, 0x01): 'HHIDSS',    (0x8C, 0x00): 'IDRM',
    (0x8C, 0x02): 'IDRM2',    (0x8C, 0x03): 'IDRM3',    (0x8D, 0x00): 'ICON',
    (0x8E, 0x00): 'IRGM',    (0x8F, 0x00): 'RIB',    (0x90, 0x00): 'PVCEO',
    (0x90, 0x01): 'PVCES',    (0x91, 0x00): 'PVCBO',    (0x92, 0x00): 'HVFGD',
    (0x92, 0x08): 'HVFBC8',    (0x93, 0x00): 'IGTF',    (0x93, 0x02): 'IGTF2',
    (0x93, 0x03): 'IGTF3',    (0x94, 0x00): 'IGTS',    (0x94, 0x02): 'IGTS2',
    (0x94, 0x03): 'IGTS3',    (0x95, 0x00): 'VGTF',    (0x95, 0x02): 'VGTF2',
    (0x95, 0x03): 'VGTF3',    (0x96, 0x00): 'VGTS',    (0x96, 0x02): 'VGTS2',
    (0x96, 0x03): 'VGTS3',    (0x97, 0x00): 'GMPA',    (0x98, 0x07): 'DVF2',
    (0x99, 0x00): 'HVDSON',    (0x99, 0x08): 'HVDSO8',    (0x99, 0x09): 'HVDSO9',
    (0x99, 0x0A): 'HVDS10',    (0x99, 0x0D): 'HVDSO-',    (0x9A, 0x00): 'HICON',
    (0x9B, 0x00): 'IH',    (0x9B, 0x02): 'IH2',    (0x9B, 0x03): 'IH3',
    (0x9D, 0x00): 'IBD',    (0x9E, 0x00): 'HICS',    (0x9E, 0x0C): 'HICSH',
    (0x9E, 0x0D): 'HICS-',    (0x9F, 0x0B): 'HVSCE2',    (0x9F, 0x0C): 'HVSCE3',
    (0x9F, 0x0D): 'HVSCE',    (0x9F, 0x0E): 'HVSCB',    (0x9F, 0x0F): 'HVSEB',
    (0xA0, 0x07): 'DVGSA2',    (0xA1, 0x00): 'HIDON',    (0xA1, 0x0D): 'HIDON-',
    (0xA2, 0x00): 'HDVGS',    (0xA2, 0x0D): 'HDVGS-',    (0xA3, 0x0A): 'VREFRH',
    (0xA3, 0x0B): 'VRFRH1',    (0xA3, 0x0C): 'VRFRH2',    (0xA4, 0x0D): 'SVCE',
    (0xA4, 0x0E): 'SVCB',    (0xA4, 0x0F): 'SVEB',    (0xA6, 0x00): 'VTM',
    (0xA7, 0x00): 'VOFS',    (0xA7, 0x0E): 'VOFS-',    (0xA7, 0x0F): 'VOFS+',
    (0xA8, 0x00): 'BVIN',    (0xA9, 0x00): 'DVT',    (0xA9, 0x02): 'DVTR2',
    (0xA9, 0x03): 'DVTR3',    (0xAA, 0x00): 'HVBEB',    (0xAB, 0x00): 'HVIEB',
    (0xAC, 0x00): 'DVGSF',    (0xAD, 0x01): 'YOSS',    (0xAE, 0x00): 'VTMS',
    (0xAF, 0x00): 'ICCHV',    (0xB0, 0x00): 'ILF',    (0xB0, 0x02): 'ILF2',
    (0xB0, 0x03): 'ILF3',    (0xB1, 0x00): 'ILS',    (0xB1, 0x02): 'ILS2',
    (0xB1, 0x03): 'ILS3',    (0xB2, 0x00): 'IOH',    (0xB3, 0x00): 'ICCHN',
    (0xB4, 0x00): 'Q+',    (0xB4, 0x01): 'EXT1',    (0xB4, 0x02): 'EXT2',
    (0xB4, 0x03): 'EXT3',    (0xB4, 0x04): 'EXT4',    (0xB5, 0x00): 'BVCEI',
    (0xB8, 0x00): 'GMPH',    (0xB9, 0x00): 'COB',    (0xB9, 0x0F): 'COBF',
    (0xBA, 0x00): 'CIB',    (0xBA, 0x0F): 'CIBF',    (0xBB, 0x00): 'PRESET',
    (0xBB, 0x01): 'EXT1C',    (0xBB, 0x02): 'EXT2C',    (0xBB, 0x03): 'EXT3C',
    (0xBB, 0x04): 'PRE4',    (0xBB, 0x05): 'EXT4C',    (0xBB, 0x07): 'DVBE1',
    (0xBB, 0x08): 'DVCE1',    (0xBB, 0x09): 'DVGSA1',    (0xBB, 0x0A): 'DVGS1',
    (0xBB, 0x0B): 'DVF1',    (0xBB, 0x0C): 'DVGK1',    (0xBB, 0x0D): 'DVT1',
    (0xBB, 0x0E): 'DELAY',    (0xBB, 0x0F): 'DVDS1',    (0xBC, 0x00): 'X-AXIS',
    (0xBC, 0x0F): 'Y-AXIS',    (0xBD, 0x00): 'GMI2',    (0xBE, 0x00): 'GMV1',
    (0xBF, 0x00): 'GMV2',    (0xC0, 0x00): 'VZ',    (0xC0, 0x0D): 'VKARH',
    (0xC0, 0x0E): 'VKARH2',    (0xC0, 0x0F): 'VKARH1',    (0xC1, 0x00): 'VFG1S',
    (0xC1, 0x08): 'RG',    (0xC1, 0x09): 'VRG',    (0xC2, 0x00): 'VFG1D',
    (0xC3, 0x00): 'VFG2S',    (0xC4, 0x00): 'VFG2D',    (0xC5, 0x00): 'VF',
    (0xC5, 0x04): 'VREFR1',    (0xC5, 0x06): 'VREFR',    (0xC5, 0x07): 'VREFR2',
    (0xC5, 0x08): 'VREFS',    (0xC5, 0x0C): 'VKAR',    (0xC5, 0x0D): 'IREF',
    (0xC5, 0x0E): 'VKAR2',    (0xC5, 0x0F): 'VKAR1',    (0xC6, 0x00): 'IDX',
    (0xC6, 0x04): 'IDX4',    (0xC6, 0x0A): 'IDX+',    (0xC6, 0x0B): 'IDXC',
    (0xC6, 0x0C): 'IDXC+',    (0xC7, 0x00): 'IGX',    (0xC8, 0x00): 'ICEI',
    (0xC9, 0x07): 'RTH',    (0xCA, 0x01): 'CHG1',    (0xCA, 0x02): 'CHG2',
    (0xCA, 0x03): 'CHG3',    (0xCA, 0x04): 'CHG4',    (0xCA, 0x05): 'CHG5',
    (0xCA, 0x06): 'CHG6',    (0xCA, 0x07): 'CHG7',    (0xCA, 0x08): 'CHG8',
    (0xCA, 0x09): 'CHG9',    (0xCA, 0x0A): 'CHG10',    (0xCA, 0x0B): 'CHG11',
    (0xCA, 0x0C): 'CHG12',    (0xCA, 0x0D): 'CHG13',    (0xCA, 0x0E): 'CHG14',
    (0xCB, 0x00): 'IPEAK',    (0xCC, 0x00): 'IR',    (0xCC, 0x01): 'IOFF',
    (0xCC, 0x06): 'IMIN',    (0xCD, 0x00): 'HVDRM',    (0xCE, 0x00): 'HVIDRM',
    (0xD0, 0x00): 'HVVR',    (0xD1, 0x00): 'IDXX',    (0xD2, 0x00): 'VDSXX',
    (0xD2, 0x08): 'VG1SRI',    (0xD3, 0x00): 'RESETV',    (0xD4, 0x00): 'IOP',
    (0xD6, 0x00): 'INPUTI',    (0xD6, 0x01): 'ISC',    (0xD7, 0x00): 'IQIN',
    (0xD8, 0x00): 'VPL',    (0xD8, 0x0D): 'VPL+',    (0xD8, 0x0E): 'VPL-',
    (0xD9, 0x00): 'IFIN',    (0xDA, 0x00): 'NOV',    (0xDA, 0x0F): 'NOVC',
    (0xDB, 0x00): 'RR',    (0xDB, 0x0F): 'RRC',    (0xDC, 0x07): 'DTEMP',
    (0xDD, 0x07): 'DVCE2',    (0xDE, 0x00): 'HVIR',    (0xDF, 0x00): 'DVBEB',
    (0xE0, 0x0F): 'VIOP',    (0xE1, 0x07): 'DVT2',    (0xE2, 0x07): 'DVGK2',
    (0xE3, 0x00): 'RDON',    (0xE3, 0x0F): 'RDON-',    (0xE4, 0x00): 'POLA',
    (0xE5, 0x00): 'VBO',    (0xE5, 0x02): 'VBO2',    (0xE5, 0x03): 'VBO3',
    (0xE8, 0x01): 'ICSGS',    (0xED, 0x00): 'VREE',    (0xEE, 0x00): 'VRCC',
    (0xEF, 0x01): 'VFDSS',    (0xF2, 0x00): 'ZZ',    (0xF2, 0x06): 'ZAK',
    (0xF3, 0x00): 'VFSD+',    (0xF3, 0x0F): 'VFSD-',    (0xF4, 0x00): 'HVFSD+',
    (0xF4, 0x0F): 'HVFSD-',    (0xF5, 0x00): 'CISS',    (0xF5, 0x0F): 'CISSF',
    (0xF6, 0x00): 'COSS',    (0xF6, 0x0F): 'COSSF',    (0xF7, 0x00): 'ICCH',
    (0xF8, 0x00): 'ISSS',    (0xF9, 0x00): 'VOH',    (0xFA, 0x00): 'IFTOFF',
    (0xFB, 0x00): 'VFSS',    (0xFC, 0x00): 'NOTCHS',    (0xFC, 0x0F): 'NOCHSH',
    (0xFD, 0x00): 'CRSS',    (0xFD, 0x0F): 'CRSSF',    (0x406E, 0x03): 'SWOFF',
    (0x8000, 0x00): 'PDVBE1',    (0x8000, 0x01): 'PDVBE2',    (0x8000, 0x02): 'PDVDS1',
    (0x8000, 0x03): 'PDVDS2',    (0x8000, 0x04): 'PDVGS1',    (0x8000, 0x05): 'PDVGS2',
    (0x8000, 0x06): 'PK',    (0x8000, 0x07): 'PRTH',    (0x8000, 0x08): 'PDTEMP',
    (0x8000, 0x09): 'PDVCE1',    (0x8000, 0x0A): 'PDVCE2',    (0x8000, 0x0B): 'PDVF1',
    (0x8000, 0x0C): 'PDVF2',    (0x8000, 0x0D): 'PDSCAN',    (0x8001, 0x00): 'VLMT',
    (0x8001, 0x01): 'VSTP',    (0x8001, 0x02): 'VST1',    (0x8001, 0x03): 'VST2',
    (0x8001, 0x04): 'VST3',    (0x8001, 0x05): 'VST4',    (0x8001, 0x06): 'VST5',
    (0x8001, 0x07): 'VST6',    (0x8001, 0x08): 'VSFP',    (0x8001, 0x09): 'VSF1',
    (0x8001, 0x0A): 'VSF2',    (0x8001, 0x0B): 'VSF3',    (0x8001, 0x0C): 'VSF4',
    (0x8001, 0x0D): 'VSF5',    (0x8001, 0x0E): 'VSF6',    (0x8002, 0x00): 'MANUAL',
    (0x8002, 0x01): 'ENGINR',    (0x8003, 0x00): 'PCOB',    (0x8003, 0x01): 'PCIB',
    (0x8003, 0x02): 'PCEB',    (0x8003, 0x03): 'PCCB',    (0x8003, 0x04): 'PCRE',
    (0x8003, 0x05): 'PCCE',    (0x8003, 0x06): 'PCRB',    (0x8004, 0x00): 'PCOBF',
    (0x8004, 0x01): 'PCIBF',    (0x8004, 0x02): 'PCEBF',    (0x8004, 0x03): 'PCCBF',
    (0x8004, 0x04): 'PCREF',    (0x8004, 0x05): 'PCCEF',    (0x8004, 0x06): 'PCRBF',
    (0x8005, 0x00): 'MIN',    (0x8005, 0x01): 'MAX',    (0x8006, 0x00): 'SETCOR',
    (0x8006, 0x01): 'R-X',    (0x8006, 0x02): 'COROSL',    (0x8006, 0x03): 'FREQ',
    (0x8006, 0x04): 'COREF',    (0x8007, 0x00): 'OPT_ON',    (0x8007, 0x01): 'OPT_OF',
    (0x8009, 0x00): 'TRR01',    (0x8009, 0x01): 'TRR02',    (0x8009, 0x02): 'TRR03',
    (0x8009, 0x03): 'TRR04',    (0x8009, 0x04): 'TRR05',    (0x8009, 0x05): 'TRR1',
    (0x8009, 0x06): 'TRR2',    (0x800A, 0x00): 'PRSM01',    (0x800A, 0x01): 'PRSM02',
    (0x800A, 0x02): 'PRSM03',    (0x800A, 0x03): 'PRSM04',    (0x800A, 0x04): 'PRSM05',
    (0x800A, 0x05): 'PRSM1I',    (0x800A, 0x06): 'PRSM1V',    (0x800A, 0x07): 'PRSM1P',
    (0x800A, 0x08): 'PRSM2I',    (0x800A, 0x09): 'PRSM2V',    (0x800A, 0x0A): 'PRSM2P',
    (0x800B, 0x00): 'SQRT',    (0x800C, 0x00): 'POW',    (0x800D, 0x00): 'DELRB1',
    (0x800E, 0x00): 'LOG',    (0x800E, 0x01): 'LOG10',
}

# Paste your helper functions and parsing code here:

# get_item_name, get_test_flags, parse_test_plan_block,
# parse_sort_plan_block, parse_tst_file (or parse_tst_data as needed)
def get_item_name(block):
    code_name1 = block[1]    
    code_name2 = block[13] & 0x0F
    key = (code_name1, code_name2)
    return code_name_map.get(key, f"Unknown_{key}")

def get_test_flags(block):
    """
    Extract only single-bit flags from option bytes (byte 14 and 15).
    Excludes multi-bit flags like AC and Di.
    """
    opt1 = block[14]
    opt2 = block[15]

    return {
        "RV": (opt1 & 0x80) != 0,  # Reverse
        "Oi": (opt1 & 0x40) != 0,  # Osc Inhibit
        "Ai": (opt1 & 0x20) != 0,  # AR Inhibit       
        "AR": (opt1 & 0x10) != 0,  # AutoRange
        "Di": (opt1 & 0x04) != 0,  # DL inhibit
        "C/B1": (opt1 & 0x02) != 0,  # Cover or Branch          
        "C/B2": (opt1 & 0x01) != 0,  # Cover or Branch  
        "CP": (opt2 & 0x20) != 0,  # ContinuePower
        "AC": (opt2 & 0x40) != 0,  # AC Test
    }


def parse_test_plan_block(block):
    sequence_num = block[0]
    item_name = get_item_name(block)

    b5 = block[4]
    b6 = block[5]
    limit_raw = ((b5 >> 4) * 1000) + ((b5 & 0x0F) * 100) + ((b6 >> 4) * 10) + (b6 & 0x0F)
    b7 = block[6]
    limit_suffix_code = b7 & 0x0F
    limit = decode_limit_with_suffix(limit_raw, limit_suffix_code)

    b14 = block[13]
    is_min_limit = (b14 & 0x80) == 0x80
    limit_type = "Min" if is_min_limit else "Max"

    b8 = block[7]
    b9 = block[8]
    bias1_raw = ((b8 >> 4) * 100) + ((b8 & 0x0F) * 10) + (b9 >> 4)
    bias1_suffix_code = b9 & 0x0F
    bias1 = decode_value_with_suffix(bias1_raw, bias1_suffix_code)

    b10 = block[9]
    b11 = block[10]
    bias2_raw = ((b10 >> 4) * 100) + ((b10 & 0x0F) * 10) + (b11 >> 4)
    bias2_suffix_code = b11 & 0x0F
    bias2 = decode_value_with_suffix(bias2_raw, bias2_suffix_code)

    b12 = block[11]
    b13 = block[12]
    test_time_raw = ((b12 >> 4) * 100) + ((b12 & 0x0F) * 10) + (b13 >> 4)
    test_time_suffix_code = b13 & 0x0F
    test_time = decode_value_with_suffix(test_time_raw, test_time_suffix_code)

    pass_branch = block[16]
    fail_branch = block[17]

    pass_branch_str = "SORT" if pass_branch == 251 else str(pass_branch)
    fail_branch_str = "SORT" if fail_branch == 251 else str(fail_branch)

    flags = get_test_flags(block)

    # ✅ Special cases using calc_si
    try:
        if item_name in ("RDON", "HRDON", "RDON-", "HRDON-"):
            limit = calc_si(str(limit), str(bias1), "/")
        elif item_name in ("HFE", "HHFE"):
            limit = calc_si(str(bias2), str(limit), "/")
    except Exception as e:
        print(f"Warning: calc_si failed for {item_name}: {e}")
    
    return {
        "Sequence": sequence_num,
        "ItemName": item_name,
        "Limit": limit,
        "LimitType": limit_type,
        "Bias1": bias1,
        "Bias2": bias2,
        "TestTime": test_time,
        "PassBranch": pass_branch_str,
        "FailBranch": fail_branch_str,
        **flags,
    }


def parse_sort_plan_block(block):
    if len(block) < 20 or (len(block) - 20) % 2 != 0:
        return None

    header = block[:20]
    if header[0] != 0xFF or header[1] != 0xFF:
        return None

    sort_seq = header[2]
    logic_code = header[3]
    bin_number = header[4]
    user_name = header[5:15].decode('ascii', errors='ignore').strip()

    condition_data = block[20:]
    num_conditions = len(condition_data) // 2

    conditions = []
    for i in range(num_conditions):
        test_num = condition_data[2 * i]
        result_flag = condition_data[2 * i + 1]

        if test_num == 0x00 and result_flag == 0x00:
            continue  # padding

        result = {
            0x00: "PASS",
            0x80: "FAIL"
        }.get(result_flag, f"Unknown(0x{result_flag:02X})")

        conditions.append({
            "TestNum": test_num,
            "Result": result
        })

    logic_map = {
        0x00: "AND",
        0x01: "ALL",
        0x02: "OR",
        0x04: "OSC",
        0x08: "REJECT",
        0x80: "ALL PASS"
    }

    condition_dict = {}
    for idx, cond in enumerate(conditions, start=1):
        condition_dict[f"Test{idx}"] = cond["TestNum"]
        condition_dict[f"Test{idx}_Result"] = cond["Result"]

    sort_plan_data = {
        "SortSequence": sort_seq,
        "LogicCondition": logic_map.get(logic_code, f"Unknown(0x{logic_code:02X})"),
        "BinNumber": bin_number,
        "UserName": user_name,
    }
    sort_plan_data.update(condition_dict)

    return sort_plan_data



# Adjust parse_tst_file to accept bytes instead of filepath or create parse_tst_data for bytes input.

def parse_tst_data(data):
    num_test_plans = data[9]
    num_sort_plans = data[10]
    test_plan_start = 36
    test_block_size = 18
    sort_block_size = data[11]
    test_plans = []

    for i in range(num_test_plans):
        start = test_plan_start + i * test_block_size
        block = data[start:start + test_block_size]
        if len(block) < test_block_size:
            st.warning(f"Incomplete test block at index {i}")
            continue
        test_plan = parse_test_plan_block(block)
        test_plans.append(test_plan)

    sort_plan_start = test_plan_start + num_test_plans * test_block_size
    sort_plans = []

    offset = sort_plan_start
    for i in range(num_sort_plans):
        #block_size = 20 + (num_test_plans * 2)
        block_size = 20 + (sort_block_size * 2)
        block = data[offset:offset + block_size]
        if len(block) < block_size:
            st.warning(f"Incomplete sort block data at index {i}")
            break
        sort_plan = parse_sort_plan_block(block)
        if sort_plan:
            sort_plans.append(sort_plan)
        offset += block_size

    return test_plans, sort_plans

def apply_same_mirroring(df):
    df = df.copy()

    # Create a lookup for rows by sequence number
    seq_map = df.set_index("Sequence").to_dict("index")

    for idx, row in df.iterrows():
        if row["ItemName"] == "SAME":
            mirror_seq = int(float(row["Bias1"]))
            mirror = seq_map.get(mirror_seq)

            if not mirror:
                print(f"⚠️ Warning: SAME at index {idx} refers to missing Sequence {mirror_seq}")
                continue

            # Mirror fields from the referenced sequence
            df.at[idx, "ItemName"] = mirror["ItemName"]
            df.at[idx, "Bias1"] = mirror["Bias1"]
            df.at[idx, "Bias2"] = mirror["Bias2"]
            #df.at[idx, "TestTime"] = mirror["TestTime"]
            #df.at[idx, "PassBranch"] = mirror["PassBranch"]
            #df.at[idx, "FailBranch"] = mirror["FailBranch"]

            for flag in ["RV", "AR", "CP"]:
                if flag in mirror:
                    df.at[idx, flag] = mirror[flag]

            # Preserve existing mirrored limits (Limit-L or Limit-H)
            if pd.notna(row.get("Limit-L")) and row["Limit-L"] != "":
                df.at[idx, "Limit-L"] = row["Limit-L"]
            if pd.notna(row.get("Limit-H")) and row["Limit-H"] != "":
                df.at[idx, "Limit-H"] = row["Limit-H"]

    return df

st.title("TST File Parser")

VALIDATION_RULES = {
    "Clamp condition are correct": validate_bv_bias2_gt_limith,
    "All FailSort are Branch condition": check_cb2_all_B,
    "Some Test Item is not specifeid Fail Branch": check_failbranch_vs_sequence,
    "All Fail Branch have the same value": check_failbranch_uniform,
    "Use only 'Fail Branch'": check_passbranch_all_zero,
    "OSC in Sort Plan": validate_logiccondition_osc_once,
    "Reject in Sort Plan": validate_logiccondition_reject_once,
    "Use 'All PASS'": validate_logiccondition_all_pass_once,
    "All Fail Sort use 'OR'": validate_logiccondition_or_except_special,
    "Some Test Item is missing in Sort Plan": validate_or_logic_contains_all_tests,
    "Spec & Bias1-2 Correlation": correlate_spec_with_validspec,
    "LowVolt's I-Bias not over 20A": validate_bias_lowvolt_for_special_items,

    # Add more validation functions here later
}

# === Shared Sidebar (for both tabs) ===
st.sidebar.header("Select Validations to Run")
selected_validations = []
expected_bin_number = None
for label, func in VALIDATION_RULES.items():
    if st.sidebar.checkbox(label, value=True):
        selected_validations.append((label, func))
        if label == "Once ALL PASS with expected BinNumber":
            expected_bin_number = st.sidebar.number_input(
                "Expected BinNumber for 'ALL PASS'",
                min_value=0, value=1, step=1
            )

# === Tabs for Single vs Multiple File Validation ===
tab1, tab2, tab3 = st.tabs(["📁 Single File Validation", "🗂️ Multiple File Validation", "⚠️ MSS Spec Validation"])

# ------------------------------------------------------
# TAB 1: Single File Validation
# ------------------------------------------------------
with tab1:
    st.header("Single File Validation")

    uploaded_file = st.file_uploader("Upload a single .tst file", type=["tst"], key="single")

    if uploaded_file:
        data = uploaded_file.read()
        tests, sorts = parse_tst_data(data)
        df_tests = df_sorts = None

        # === Build DataFrames ===
        if tests:
            df_tests = pd.DataFrame(tests)
            # Replace True/False flags with their labels or empty strings
            flag_columns = {
                "RV": "RV", "AR": "AR", "CP": "CP", "AC": "AC",
                "Oi": "Oi", "Ai": "Ai", "Di": "Di", "C/B1": "B",
                "C/B2": "B"
                }
            # Set of columns that should show "C" when False
            false_x_columns = {"C/B1", "C/B2"}
            for col, label in flag_columns.items():
                if col in df_tests.columns:
                    if col in false_x_columns:
                        df_tests[col] = df_tests[col].apply(lambda x: label if x else "C")
                    else:
                        df_tests[col] = df_tests[col].apply(lambda x: label if x else "")


            # Convert 'Limit' and 'LimitType' to 'Limit-L' and 'Limit-H'
            if "Limit" in df_tests.columns and "LimitType" in df_tests.columns:
                df_tests["Limit-L"] = df_tests.apply(
                    lambda row: row["Limit"] if row["LimitType"] == "Min" else "", axis=1
                )
                df_tests["Limit-H"] = df_tests.apply(
                    lambda row: row["Limit"] if row["LimitType"] == "Max" else "", axis=1
                )
                df_tests.drop(columns=["Limit", "LimitType"], inplace=True)
            # Define your desired column order
            column_order = [
                "Sequence",
                "ItemName",
                "Limit-L",
                "Limit-H",        
                "Bias1",
                "Bias2",
                "TestTime",
                "C/B1",
                "PassBranch",
                "C/B2",
                "FailBranch",
                "RV",
                "AR",
                "CP",
                "AC",
                "Oi",
                "Ai",
                "Di",
            ]

            # Apply the new column order (only include columns that actually exist)
            df_tests = df_tests[[col for col in column_order if col in df_tests.columns]]
            st.subheader("Test Plans")
            st.dataframe(df_tests)

        if sorts:
            df_sorts = pd.DataFrame(sorts)
            st.subheader("Sort Plans")
            st.dataframe(df_sorts)


        # === Run Validations ===        

        if selected_validations:
            st.subheader("Validation Results")
            summary_data = []
            all_errors = {}      
            
            for label, func in selected_validations:                
                df_tests_processed = apply_same_mirroring(df_tests)

                # Function calling logic
                if label == "Once ALL PASS with expected BinNumber":
                    errors = func(df_tests_processed, df_sorts, expected_bin_number)
                elif label == "LowVolt's I-Bias not over 20A":
                    errors = func(df_tests, df_sorts)
                elif label == "Spec & Bias1-2 Correlation":
                    # Build spec path from uploaded tst file

                    spec_filename = uploaded_file.name.replace(".tst", ".csv")
                    spec_path = f"paper-spec/{spec_filename}"   

                    errors = func(df_tests, spec_path, df_sorts)
                else:
                    errors = func(df_tests_processed, df_sorts)

                issue_count = len(errors)
                status = "✅ PASS" if issue_count == 0 else "❌ FAIL"
                summary_data.append({
                    "Validation": label,
                    "Status": status,
                    "Issues": issue_count
                })
                all_errors[label] = errors

            # Summary
            st.markdown("### Summary")
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
                height=len(summary_df) * 35 + 40  # auto height per row
            )
            
            # Details
            issue_found = False  # Track if any issue exists
            for label, errors in all_errors.items():
                if errors:
                    issue_found = True
                    with st.expander(f"❗ {label} Details ({len(errors)} issue(s))", expanded=True):
                        st.error(f"{len(errors)} issue(s) found:")
                        for e in errors:
                            st.markdown(f"<span style='color:red'>• {e}</span>", unsafe_allow_html=True)

            if not issue_found:
                st.success("✅ No issues found.")
        #st.dataframe(df_tests_processed)                        
    else:
        st.info("Please upload a .tst file to start validation.")

# ------------------------------------------------------
# TAB 2: Multiple File Validation
# ------------------------------------------------------
with tab2:
    st.header("Multiple File Validation")

    uploaded_files = st.file_uploader(
        "Upload multiple .tst files", type=["tst"], accept_multiple_files=True, key="multi"
    )

    if uploaded_files:
        overall_summary = []
        file_results = {}  # Store detailed errors per file

        # Checkbox to show Test/Sort Data
        show_data_checkbox = st.checkbox("Show Test/Sort Data for individual files")

        # Process all files first (for performance, overall summary)
        for uploaded_file in uploaded_files:
            data = uploaded_file.read()
            tests, sorts = parse_tst_data(data)
            df_tests = pd.DataFrame(tests) if tests else None
            df_sorts = pd.DataFrame(sorts) if sorts else None

            # Store preprocessed Test/Sort data if user wants to see it
            file_results[uploaded_file.name] = {
                "df_tests": df_tests,
                "df_sorts": df_sorts,
                "summary_data": [],
            }

            if df_tests is not None:
                # Replace True/False flags
                flag_columns = {
                    "RV": "RV", "AR": "AR", "CP": "CP", "AC": "AC",
                    "Oi": "Oi", "Ai": "Ai", "Di": "Di", "C/B1": "B", "C/B2": "B"
                }
                false_x_columns = {"C/B1", "C/B2"}
                for col, label in flag_columns.items():
                    if col in df_tests.columns:
                        df_tests[col] = df_tests[col].apply(
                            lambda x: label if x else ("C" if col in false_x_columns else "")
                        )

                # Convert Limit/LimitType
                if "Limit" in df_tests.columns and "LimitType" in df_tests.columns:
                    df_tests["Limit-L"] = df_tests.apply(
                        lambda row: row["Limit"] if row["LimitType"] == "Min" else "", axis=1
                    )
                    df_tests["Limit-H"] = df_tests.apply(
                        lambda row: row["Limit"] if row["LimitType"] == "Max" else "", axis=1
                    )
                    df_tests.drop(columns=["Limit", "LimitType"], inplace=True)

                # Column order
                column_order = [
                    "Sequence", "ItemName", "Limit-L", "Limit-H", "Bias1", "Bias2",
                    "TestTime", "C/B1", "PassBranch", "C/B2", "FailBranch",
                    "RV", "AR", "CP", "AC", "Oi", "Ai", "Di"
                ]
                df_tests = df_tests[[col for col in column_order if col in df_tests.columns]]
                file_results[uploaded_file.name]["df_tests"] = df_tests

            # Run validations
            if selected_validations:
                all_errors = {}
                for label, func in selected_validations:
                    df_tests_processed = apply_same_mirroring(df_tests) if df_tests is not None else None

                    if label == "Once ALL PASS with expected BinNumber":
                        errors = func(df_tests_processed, df_sorts, expected_bin_number)
                    elif label == "LowVolt's I-Bias not over 20A":
                        errors = func(df_tests, df_sorts)
                    elif label == "Spec & Bias1-2 Correlation":
                        # Build spec path from uploaded tst file

                        spec_filename = uploaded_file.name.replace(".tst", ".csv")
                        spec_path = f"paper-spec/{spec_filename}"        

                        errors = func(df_tests, spec_path, df_sorts)
                    
                    else:
                        errors = func(df_tests_processed, df_sorts)

                    issue_count = len(errors)
                    status = "✅ PASS" if issue_count == 0 else "❌ FAIL"
                    file_results[uploaded_file.name]["summary_data"].append({
                        "File": uploaded_file.name,
                        "Validation": label,
                        "Status": status,
                        "Issues": issue_count
                    })
                    all_errors[label] = errors

                file_results[uploaded_file.name]["all_errors"] = all_errors
                overall_summary.extend(file_results[uploaded_file.name]["summary_data"])

        # === Display Overall Summary First ===
        st.markdown("## 🧾 Overall Summary (All Files)")
        overall_df = pd.DataFrame(overall_summary)
        st.dataframe(overall_df, use_container_width=True)

        # CSV Export
        csv = overall_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Overall Summary (CSV)", csv, "validation_results.csv", "text/csv"
        )

        # === Optional Drill-Down: Show Failed Validations Only ===
        show_details = st.checkbox("Show detailed errors for failed validations")
        if show_details:
            for file_name, info in file_results.items():
                failed_validations = {k: v for k, v in info.get("all_errors", {}).items() if len(v) > 0}
                if failed_validations:
                    st.markdown(f"### ⚠️ {file_name} - Failed Validations")
                    for label, errors in failed_validations.items():
                        with st.expander(f"{label} ({len(errors)} issue(s))"):
                            for e in errors:
                                st.write(f"- {e}")

        # === Optional Test/Sort Data Display ===
        if show_data_checkbox:
            for file_name, info in file_results.items():
                st.markdown(f"### 📄 {file_name} - Test/Sort Data")
                if info["df_tests"] is not None:
                    st.subheader("Test Plans")
                    st.dataframe(info["df_tests"])
                if info["df_sorts"] is not None:
                    st.subheader("Sort Plans")
                    st.dataframe(info["df_sorts"])

    else:
        st.info("Please upload one or more .tst files for validation.")

# ------------------------------------------------------
# TAB 3: MSS Spec Validation
# ------------------------------------------------------
with tab3:
    st.header("MSS Spec Validation")

    uploaded_spec_file = st.file_uploader("Upload a .tst file", type=["tst"], key="spec")    
    
    if uploaded_spec_file:
        file_name = os.path.splitext(uploaded_spec_file.name)[0]
        data = uploaded_spec_file.read()
        tests, sorts = parse_tst_data(data)
        df_tests = pd.DataFrame(tests) if tests else None

        if df_tests is not None and not df_tests.empty:
            # --- Replace True/False flags (like in your main loop) ---
            flag_columns = {
                "RV": "RV", "AR": "AR", "CP": "CP", "AC": "AC",
                "Oi": "Oi", "Ai": "Ai", "Di": "Di", "C/B1": "B", "C/B2": "B"
            }
            false_x_columns = {"C/B1", "C/B2"}
            for col, label in flag_columns.items():
                if col in df_tests.columns:
                    df_tests[col] = df_tests[col].apply(
                        lambda x: label if x else ("C" if col in false_x_columns else "")
                    )

            # --- Convert Limit/LimitType ---
            if "Limit" in df_tests.columns and "LimitType" in df_tests.columns:
                df_tests["Limit-L"] = df_tests.apply(
                    lambda row: row["Limit"] if row["LimitType"] == "Min" else "", axis=1
                )
                df_tests["Limit-H"] = df_tests.apply(
                    lambda row: row["Limit"] if row["LimitType"] == "Max" else "", axis=1
                )
                df_tests.drop(columns=["Limit", "LimitType"], inplace=True)

            # --- Column order (RV is optional) ---
            column_order = [
                "Sequence", "ItemName", "Limit-L", "Limit-H",
                "Bias1", "Bias2", "TestTime", "C/B1", "PassBranch",
                "C/B2", "FailBranch", "RV", "AR", "CP", "AC",
                "Oi", "Ai", "Di"
            ]
            df_tests = df_tests[[col for col in column_order if col in df_tests.columns]]

            # --- Show Original Test Data ---
            st.subheader("Original Test Program")
            base_cols = ["Sequence", "ItemName", "Limit-L", "Limit-H", "Bias1", "Bias2", "RV"]
            tab3_original = df_tests[base_cols].copy()
            tab3_original.reset_index(drop=True, inplace=True)
            st.dataframe(tab3_original, use_container_width=True)

            # --- Build Spec Draft (exclude SAME, DEF, CONT) ---
            exclude_items = ["SAME", "DEF", "CONT"]
            spec_draft = df_tests[~df_tests["ItemName"].isin(exclude_items)].copy()

            # Keep only relevant spec columns
            spec_draft = spec_draft[
                ["ItemName", "Limit-L", "Limit-H", "Bias1", "Bias2", "RV", "Sequence"]
            ].copy()

            # --- Add blank Seq-prefixed columns ---
            seq_cols = ["SeqItemName", "SeqLimit-L", "SeqLimit-H", "SeqBias1", "SeqBias2", "SeqRV"]
            for col in seq_cols:
                spec_draft[col] = ""

            # --- Auto-fill Sequence-based columns ---
            for idx, row in spec_draft.iterrows():
                seq = row["Sequence"]
                spec_draft.at[idx, "SeqItemName"] = seq
                spec_draft.at[idx, "SeqBias1"] = seq
                spec_draft.at[idx, "SeqBias2"] = seq
                spec_draft.at[idx, "SeqRV"] = seq  # new SeqRV
                # Leave SeqLimit-L and SeqLimit-H for user editing

            spec_draft.drop(columns=["Sequence"], inplace=True)

            # --- Editable Spec Draft Table ---
            st.subheader("MSS Table")
           # Use container width and wrap it in a full-width column
            with st.container():
                edited_spec = st.data_editor(
                    spec_draft,
                    num_rows="dynamic",
                    use_container_width=True
                )

            st.markdown(
                "<div style='padding: 0.5em; background-color: #fff8e1; border-left: 6px solid #f39c12;'>"
                "<strong>⚠️ Tip:</strong> For a properly named export, please use the <em>Download MSS as CSV</em> button below."
                "</div>",
                unsafe_allow_html=True
            )

            # --- Export MSS Table ---
            csv_data = edited_spec.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download MSS as CSV",
                data=csv_data,
                file_name=f"{file_name}.csv",
                mime="text/csv"
            )

        else:
            st.warning("⚠️ No valid test data found in the uploaded file.")
    else:
        st.info("Please upload a `.tst` file to view spec data.")
