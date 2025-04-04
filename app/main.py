import cv2

from app.config import IMAGE_PATH
from app.engine import Engine
from app.processing.pipeline import Pipeline
from app.shaders import Decay, Diffuse


def get_pipeline() -> Pipeline:
    pipeline = Pipeline()
    pipeline += Decay()
    pipeline += Diffuse()
    return pipeline


def main():
    image = cv2.imread(IMAGE_PATH)
    pipeline = get_pipeline()

    engine = Engine(image[:, :, 2], pipeline)
    engine.init_pygame()
    engine.run()


if __name__ == "__main__":
    main()
