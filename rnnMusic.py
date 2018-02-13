import tensorflow as tf
import numpy as np
import midi

from utils.extractFeatures import get_songs
from utils.modelHelper import logger, generate_batches
from utils.preprocessMidi import sample_midi, convert_midi

