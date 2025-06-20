import random

class ImageCaptioningModel:
    def __init__(self):
        self.sample_captions = [
            "A group of people walking down the street.",
            "A dog running through a grassy field.",
            "A delicious plate of food on a dining table.",
            "A car parked in front of a tall building.",
            "A person riding a bike along the beach."
        ]

    def generate_caption(self, image):
        return random.choice(self.sample_captions)
