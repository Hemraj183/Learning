import torch

def exercise_error_handling():
    # TODO: What if the Router is unsure? (Low confidence)
    # Implement a threshold: if max(logits) < threshold, route to a "Fallback" expert
    print("Exercise: Implement Confidence Thresholding.")

if __name__ == "__main__":
    exercise_error_handling()
