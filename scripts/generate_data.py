import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import os
import operator

def get_addition_pairs(lower_bound, upper_bound, rng):
    int_a = int(np.ceil(rng.uniform(lower_bound, upper_bound)))
    int_b = int(np.ceil(rng.uniform(lower_bound, upper_bound)))
    return int_a, int_b

def generate_digit_based_boundaries(start_digits, end_digits):
    boundaries = []
    for d in range(start_digits, end_digits + 1):
        if d == 1:
            lower = 0
            upper = 10
        else:
            lower = 10**(d-1)
            upper = 10**d
        boundaries.append((lower, upper))
    return boundaries


def generate_data(num_samples, start_digits, end_digits, seed, operation, output_path):

    op_funcs = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': lambda a, b: a / b if b != 0 else None, 
        '%': lambda a, b: a % b if b != 0 else None
    }

    if operation not in op_funcs:
        raise ValueError(f"Unsupported operation: {operation}")

    op_func = op_funcs[operation]

    ranges = generate_digit_based_boundaries(start_digits, end_digits)
    rng = np.random.default_rng(seed)
    results = []
    seen_pairs = set()

    for (low, high) in ranges:
        desc = f"Generating {operation} questions for digit range [{low}, {high})"
        for _ in tqdm(range(num_samples), desc=desc):
            int_a, int_b = get_addition_pairs(low, high, rng)
            
            if operation == '/' and int_b == 0:
                continue
            if operation == '%' and int_b == 0:
                continue

            pair = tuple(sorted((int_a, int_b)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            question = f"What is {int_a} {operation} {int_b}?"
            
            answer = op_func(int_a, int_b)
            if answer is None:
                continue

            if isinstance(answer, np.number) and float(answer).is_integer():
                answer = int(answer)

            res_dict = {
                'Question': question,
                'Answer': answer,
                'num1': int(int_a),
                'num2': int(int_b)
            }
            for k, v in res_dict.items():
                if isinstance(v, np.integer):
                    res_dict[k] = int(v)
                elif isinstance(v, np.floating):
                    res_dict[k] = float(v)

            results.append(res_dict)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(results)} questions to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--start_digits', type=int, default=1)
    parser.add_argument('--end_digits', type=int, default=7)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--operation', type=str, default='%')
    parser.add_argument('--output_path', type=str, default='/Users/akashpeddaputha/Downloads/mod.csv')
    args = parser.parse_args()
    
    generate_data(num_samples=args.num_samples, 
                  start_digits=args.start_digits, 
                  end_digits=args.end_digits, 
                  seed=args.seed, 
                  operation=args.operation,
                  output_path=args.output_path)
    print(f"Data generation complete. Output saved to {args.output_path}")