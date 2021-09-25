import sys

sys.path.append('Devign')
import Devign
import data_processing

devign_source = 'data/devign'
refactored_output = 'out/refactored_devign_noname_threshold0.5'
preprocessed_output = 'data/refactored_devign_noname_threshold0.5'

data_processing.refactor_reveal.main(f'-i {devign_source} -o {refactored_output}'.split())
data_processing.add_refactored_code.main(f'--input_dir {refactored_output} --output_dir {refactored_output}'.split())
data_processing.preprocess.main(f'--input {refactored_output} --output {preprocessed_output}'.split())
Devign.main.main(f'--input_dir {preprocessed_output}'.split())
