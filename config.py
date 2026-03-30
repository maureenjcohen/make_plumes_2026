
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# %% Filepaths
plumes = os.getenv('PLUMES')
sens_tests = [
    os.getenv('SENS_TEST_1'),
    os.getenv('SENS_TEST_2'),
    os.getenv('SENS_TEST_3'),
    os.getenv('SENS_TEST_4'),
    os.getenv('SENS_TEST_5'),
]
surf = os.getenv('SURF')
chem_dir = os.getenv('CHEM_DIR')
chem = [
    chem_dir + 'Xins4.nc',
    chem_dir + 'Xins5.nc',
    chem_dir + 'Xins6.nc',
    chem_dir + 'Xins7.nc',
]
savedir = os.getenv('SAVEDIR')
filetype = os.getenv('FILETYPE', 'png')
