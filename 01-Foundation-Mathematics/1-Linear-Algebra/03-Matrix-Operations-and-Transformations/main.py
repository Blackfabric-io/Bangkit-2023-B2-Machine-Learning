"""
Main entry point for matrix transformations and analysis.
x
"""

import argparse
import numpy as np
from src.processors.main import (
    analyze_transformation,
    compose_transformations,
    analyze_webpage_ranking,
    create_transformation
)
from src.utils.helpers import plot_transformation, plot_eigenvectors

def main():
    parser = argparse.ArgumentParser(description='Matrix Transformations and Analysis')
    
    # Create subparsers for different operations
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Transform parser
    transform_parser = subparsers.add_parser('transform', 
                                           help='Apply and visualize transformation')
    transform_parser.add_argument('type', choices=['reflection', 'shear', 'rotation', 
                                                 'scaling', 'projection'],
                                help='Type of transformation')
    transform_parser.add_argument('--axis', choices=['x', 'y'], default='y',
                                help='Axis for reflection/projection')
    transform_parser.add_argument('--angle', type=float,
                                help='Angle in radians for rotation')
    transform_parser.add_argument('--factor', type=float,
                                help='Factor for scaling/shear')
    transform_parser.add_argument('--direction', choices=['x', 'y'],
                                help='Direction for shear')
    
    # Analyze parser
    analyze_parser = subparsers.add_parser('analyze', 
                                         help='Analyze transformation matrix')
    analyze_parser.add_argument('matrix', type=str,
                              help='2x2 matrix as comma-separated values (a11,a12,a21,a22)')
    
    # Webpage parser
    webpage_parser = subparsers.add_parser('webpage', 
                                         help='Analyze webpage navigation')
    webpage_parser.add_argument('--size', type=int, default=5,
                              help='Size of Markov matrix')
    webpage_parser.add_argument('--steps', type=int, default=20,
                              help='Number of steps to simulate')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'transform':
            # Create transformation matrix
            kwargs = {}
            if args.axis:
                kwargs['axis'] = args.axis
            if args.angle is not None:
                kwargs['theta'] = args.angle
            if args.factor is not None:
                if args.type == 'scaling':
                    kwargs['sx'] = args.factor
                else:
                    kwargs['k'] = args.factor
            if args.direction:
                kwargs['direction'] = args.direction
                
            T = create_transformation(args.type, **kwargs)
            print(f"\nTransformation matrix:\n{T}")
            
            # Analyze and visualize
            analysis = analyze_transformation(T)
            print(f"\nDeterminant: {analysis['determinant']:.2f}")
            print(f"Eigenvalues: {analysis['eigenvalues']}")
            plot_eigenvectors(T, analysis['eigenvalues'], analysis['eigenvectors'])
            
        elif args.command == 'analyze':
            # Parse matrix
            values = [float(x) for x in args.matrix.split(',')]
            if len(values) != 4:
                raise ValueError("Matrix must have exactly 4 values")
            T = np.array(values).reshape(2, 2)
            
            # Analyze
            analysis = analyze_transformation(T)
            print(f"\nInput matrix:\n{T}")
            print(f"\nDeterminant: {analysis['determinant']:.2f}")
            print(f"Eigenvalues: {analysis['eigenvalues']}")
            plot_eigenvectors(T, analysis['eigenvalues'], analysis['eigenvectors'])
            
        elif args.command == 'webpage':
            # Create random Markov matrix
            P = np.random.rand(args.size, args.size)
            np.fill_diagonal(P, 0)  # Set diagonal to 0
            P = P / P.sum(axis=0)  # Normalize columns
            
            # Create initial state (start at first page)
            X0 = np.zeros((args.size, 1))
            X0[0] = 1
            
            # Analyze
            results = analyze_webpage_ranking(P, X0, args.steps)
            print(f"\nTransition matrix P:\n{P}")
            print(f"\nFinal state after {args.steps} steps:\n{results['final_state']}")
            print(f"\nSteady state distribution:\n{results['steady_state']}")
            print(f"\nEigenvalues:\n{results['eigenvalues']}")
            
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 