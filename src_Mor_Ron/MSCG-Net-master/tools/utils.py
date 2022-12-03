import argparse

channel_choices = [
    "NDVI",
    "gNDVI",
    "SAVI"
]
init_choices = channel_choices.append("gaussian")
    

def parse_args():
    parser = argparse.ArgumentParser(description="choose additional channels from {}    ".format(channel_choices))

    # wandb params
    parser.add_argument('--wandb', action='store_true', default=False, help='add wandb to run (default: False)')
    parser.add_argument('--run_name', default='test-run', help='name of the run in wandb , default: "test-run"')

    # channel params
    parser.add_argument('--NDVI', action='store_true', default=False, help='add NDVI channel (default: False)')
    parser.add_argument('--gNDVI', action='store_true', default=False, help='add gNDVI channel (default: False)')
    parser.add_argument('--SAVI', action='store_true', default=False, help='add SAVI channel (default: False)')
    parser.add_argument('--GAI', nargs=10, help='add GAI channel',  type=float) # should we somehow pass normalizers data?
    parser.add_argument('--learn', '--l', action='append', choices=init_choices, help='add learnable channel. initialize from {}'.format(init_choices))

    args = parser.parse_args()
    return args