import argparse
import numpy as np
from src.processors.main import solve_linear_system

def main():
    parser = argparse.ArgumentParser(description='Solve a system of linear equations')
    parser.add_argument('--matrix', type=str, required=True,
                      help='Coefficient matrix as comma-separated values (row1,row2,row3,row4)')
    parser.add_argument('--vector', type=str, required=True,
                      help='Constants vector as comma-separated values')
    
    args = parser.parse_args()
    
    try:
        matrix_rows = args.matrix.split(';')
        if len(matrix_rows) != 4:
            raise ValueError("Matrix must have 4 rows")
            
        A = np.array([list(map(float, row.split(','))) for row in matrix_rows])
        if A.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")
            
        b = np.array(list(map(float, args.vector.split(','))))
        if b.shape != (4,):
            raise ValueError("Vector must have 4 elements")
            
        x1, x2, x3, x4 = solve_linear_system(A, b)
        
        print(f"\nSolution:")
        print(f"x1 = {x1:.2f}")
        print(f"x2 = {x2:.2f}")
        print(f"x3 = {x3:.2f}")
        print(f"x4 = {x4:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 