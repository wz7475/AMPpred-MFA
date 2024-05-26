""" module for model specific concretisation of abstract interfaces """
import os

import pandas as pd
from tools.converter import Converter, InputConverter
from tools.inference import Inferencer


class ConcreteConverter(Converter):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model """
        df = pd.read_csv(filepath, delimiter=",")
        df.rename(columns={"Class": "Prediction"}, inplace=True)
        df['Prediction'] = df['Prediction'].replace({'AMPs': 'AMP', 'Non-AMPs': 'non-AMP'})
        df.to_csv(output_filename, sep="\t", index=False)


class ConcreteInferencer(Inferencer):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model """
        command = f"python infernence.py {filepath} trained_model/model.pth trained_model/vocab.json {output_filename}"
        print(command)
        os.system(command)


class ConcreteInputConverter(InputConverter):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model """
        # for each fasta header append ||
        with open(filepath, 'r') as f:
            lines = f.readlines()
        with open(output_filename, 'w') as f:
            for line in lines:
                if line.startswith('>'):
                    f.write(line.strip() + '||\n')
                else:
                    f.write(line)
