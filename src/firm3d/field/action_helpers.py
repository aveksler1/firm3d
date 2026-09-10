import numpy as np

__all__ = [
    "check_expression_conditionals"
    "check_expression_conditionals_finite_mu"
]

def check_expression_conditionals(A1, A2, A3):
    """
    A function used to check the conditional expression given by Mathematica to
    compute the Jchi^(1) integral. This can be ignored for now. 
    """
    if A3 == 0:
        return "Error: A3 cannot be 0 (division by zero)."
    
    # Calculate the shared square root term
    sqrt_term = np.sqrt(1 - A3**2)
    
    # Calculate T1 and T2. 
    T1 = abs(1 - sqrt_term) / A3
    T2 = abs(1 + sqrt_term) / A3
    
    print(f"\n--- Calculated Terms ---")
    print(f"T1 = {T1}")
    print(f"T2 = {T2}")

    # 1. Check the Outer ConditionalExpression
    outer_condition = not (T1 == 1) and not (T2 == 1)
    
    print(f"\n--- Condition Checks ---")
    print(f"Outer Condition Valid (T1 != 1 AND T2 != 1): {outer_condition}")
    
    if not outer_condition:
        raise Exception("Outer conditional expression is FALSE.")

    # 2. Check the Piecewise Conditionals
    pw_case1 = (T1 >= 1 and T2 >= 1) or (T1 < 1 and T2 < 1)
    pw_case2 = (T1 < 1)
    
    print(f"\n--- Piecewise Evaluation ---")
    if pw_case1:
        print("Branch Taken: Case 1")
        print("Resulting Expression: 0")
        
    elif pw_case2:
        print("Branch Taken: Case 2")
        if A2 == 0:
             print("Resulting Expression: Error (A2 is 0, division by zero in expression)")
        else:
             val = (2 * np.pi) / (A2 * sqrt_term)
             print(f"Resulting Expression: (2 * Pi) / (A2 * Sqrt[1 - A3^2])")
             print(f"Calculated Value: {A1 * val}")
             
    else:
        print("Branch Taken: Case 3 (Default)")
        if A2 == 0:
             print("Resulting Expression: Error (A2 is 0, division by zero in expression)")
        else:
             val = -(2 * np.pi) / (A2 * sqrt_term)
             print(f"Resulting Expression: -(2 * Pi) / (A2 * Sqrt[1 - A3^2])")
             print(f"Calculated Value: {A1 * val}")

def check_expression_conditionals_finite_mu(A2, A3):
    if A3 < 1 and (A2 + A2*A3 < 1):
        print("Checks Passed")
    else:
        if not (A3 < 1):
            print("Condition 1 failed")
            print(f"A3 = {A3}")
        if not (A2 + A2*A3 < 1):
            print("Condition 2 failed")
            print(f"A2 + A2*A3 = {A2 + A2*A3}")