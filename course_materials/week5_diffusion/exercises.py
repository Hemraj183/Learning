import torch
from project import DiffusionProcess

def exercise_schedule():
    # TODO: Implement a Cosine Beta Schedule instead of Linear
    # Formula: alpha_bar_t = f(t) / f(0), where f(t) = cos^2(...)
    print("Exercise: Implement Cosine Schedule (Better for pixel values)")

if __name__ == "__main__":
    exercise_schedule()
