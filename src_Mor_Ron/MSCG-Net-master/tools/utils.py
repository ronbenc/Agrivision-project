import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # channel params
    parser.add_argument('--NDVI', action='store_true', default=False, help='add NDVI channel (default: False)')
    parser.add_argument('--gNDVI', action='store_true', default=False, help='add gNDVI channel (default: False)')
    parser.add_argument('--SAVI', action='store_true', default=False, help='add SAVI channel (default: False)')
    parser.add_argument('--GAI', action='store_true', default=False, help='add learnable GAI channel (default: False)')

    args = parser.parse_args()
    return args