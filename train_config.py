WAV_FILES_PATH = "archive/free-spoken-digit-dataset-master/recordings"
FILTER_BANKS = 32
WINDOW_SIZE = 25 / 1000  # in seconds
STRIDE = 10 / 1000  # in seconds
IMAGE_SIZE = (1, 32, 64)  # shape of mel scale (see dataset.py for more info)
CLASSES = 10

RUNS_PATH = "./runs"
EXPERIMENT_NAME = "vit_based_model"

DATASET_PERCENTAGE = 1
TRAIN_TEST_SPLIT = 0.9

EPOCHS = 1000
BATCH_SIZE = 64
EPOCH_AFTER_SAVE_MODEL = 4
MODELS_TO_KEEP = 4
DEVICE = 0
LEARNING_RATE = 0.01

MODEL_PATH = "./pretrained_models"

RESUME_TRAIN = False
RESUME_TRAIN_PATH = ""
