import argparse
import pandas as pd
from AMPpred_MFA.EasyUse import easy_predict_from_file

def main():
    parser = argparse.ArgumentParser(description='Easy prediction for mixed model(AMPpred_MFA) from fasta file')
    parser.add_argument('fasta_file', type=str, help='fasta file path')
    parser.add_argument('model_path', type=str, help='AMPpred_MFA model path')
    parser.add_argument('vocab_path', type=str, help='vocab path') 
    parser.add_argument('out_path', type=str, help='where to save') 

    args = parser.parse_args()

    result = easy_predict_from_file(args.fasta_file, args.model_path, args.vocab_path)
    result.to_csv(args.out_path, index=False)

if __name__ == "__main__":
    main()