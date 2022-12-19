import yaml
from yaml.loader import SafeLoader

path = './deepdrummer/params.yaml'

def get_audio_param_bp():
    # Open the file and load the file
    with open(path, 'r') as stream:
        data_bp = yaml.load(stream, Loader=SafeLoader)

    audio_params = data_bp['audio']
    audio_pattern_bp = data_bp['audio_pattern']

    return audio_params, audio_pattern_bp   

def get_input_sample_rate():
    audio_params, _  = get_audio_param_bp()
    return audio_params['sample_rate_input']

def get_num_of_lines():
    audio_params, _  = get_audio_param_bp()
    return audio_params['nbr_of_lines']
