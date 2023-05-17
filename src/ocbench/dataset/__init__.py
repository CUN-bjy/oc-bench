from pathlib import Path

DATASET_PATH = f"{Path(__file__).parent}/data/"
from .multi_object_datasets import CaterWithMasks, ClevrWithMasks, MultiDSprites, ObjectsRoom, Tetrominoes