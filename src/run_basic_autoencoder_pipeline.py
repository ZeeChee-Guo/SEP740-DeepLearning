"""
Run full basic autoencoder workflow in order

"""


from calibrate_basic_threshold import calibrate_threshold
from evaluate_basic_autoencoder import evaluate
from select_basic_autoencoder_config import select_config
from train_final_basic_autoencoder import train
from visualize_basic_results import visualize


def print_step(step_number: int, total_steps: int, title: str):
    """
    print step
    """
    print("\n" + "=" * 72)
    print(f"Step {step_number}/{total_steps}: {title}")
    print("=" * 72)


def run_pipeline() -> None:
    """
    run the complete basic workflow
    """
    total_steps = 5

    print_step(1, total_steps, "Select basic autoencoder config")
    select_config()

    print_step(2, total_steps, "Train final basic autoencoder")
    train()

    print_step(3, total_steps, "Calibrate threshold")
    calibrate_threshold()

    print_step(4, total_steps, "Evaluate basic detector")
    evaluate()

    print_step(5, total_steps, "Create figures")
    visualize()


if __name__ == "__main__":
    run_pipeline()
