import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # channel params
    parser.add_argument('--wandb', action='store_true', default=False, help='add wandb to run (default: False)')
    parser.add_argument('--run_name', default='test-run', help='name of the run in wandb , default: "test-run"')

    parser.add_argument('--NDVI', action='store_true', default=False, help='add NDVI channel (default: False)')
    parser.add_argument('--gNDVI', action='store_true', default=False, help='add gNDVI channel (default: False)')
    parser.add_argument('--SAVI', action='store_true', default=False, help='add SAVI channel (default: False)')
    parser.add_argument('--GAI', nargs=10, help='add GAI channel',  type=float)
    parser.add_argument('--learn', action='store_true', default=False, help='add learnable channel (default: False)')
    parser.add_argument('--std',  type=float, default=1, help='standard deviation for learnable alphas initialization (default: 1.0)')

    args = parser.parse_args()
    return args