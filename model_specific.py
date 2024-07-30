""" module for model specific concretisation of abstract interfaces """
import os

import pandas as pd
from tools.converter import Converter, InputConverter
from tools.inference import Inferencer


class ConcreteConverter(Converter):
    def process_file(self, filepath: str, output_filename: str):
        """ converter from model's custom format to 
        - tsv file with columnns (for classifier)
            - Prediction: ['AMP', 'non-AMP']
            - Probability score: [0-1] (float from given range)
            (for regressor)
            - Score: (float) predicted MIC score
            - Prediction: ['AMP', 'non-AMP'] (optionally in case of classifcation via thersholding regessor)
        """
        df = pd.read_csv(filepath, delimiter=",")
        df.rename(columns={"Class": "Prediction", "Probability": "Probability_score"}, inplace=True)
        df['Prediction'] = df['Prediction'].replace({'AMPs': 'AMP', 'Non-AMPs': 'non-AMP'})
        df.to_csv(output_filename, sep="\t", index=False)


class ConcreteInferencer(Inferencer):
    def process_file(self, filepath: str, output_filename: str):
        """ run inference for given paths """
        command = f"python infernence.py {filepath} trained_model/model.pth trained_model/vocab.json {output_filename}"
        print(command)
        os.system(command)


class ConcreteInputConverter(InputConverter):
    def process_file(self, filepath: str, output_filename: str):
        """ converter from standard fasta to custom format required by model """
        # for each fasta header append ||
        with open(filepath, 'r') as f:
            lines = f.readlines()
        with open(output_filename, 'w') as f:
            for line in lines:
                if line.startswith('>'):
                    f.write(line.strip() + '||\n')
                else:
                    f.write(line)
