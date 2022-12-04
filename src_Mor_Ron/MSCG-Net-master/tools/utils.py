import argparse

channel_choices = [
    "NDVI",
    "gNDVI",
    "SAVI",
    "RVI",
    "DVI",
    "VDVI",
    "GCC",
    "EVI",
    "VARI"
]
init_choices = channel_choices.append("gaussian")
    

def parse_args():
    parser = argparse.ArgumentParser(description="choose additional channels from {}    ".format(channel_choices))

    # wandb params
    parser.add_argument('--wandb', action='store_true', default=False, help='add wandb to run (default: False)')
    parser.add_argument('--run_name', default='test-run', help='name of the run in wandb , default: "test-run"')

    # channel params
    parser.add_argument('--NDVI', action='store_true', default=False, help='add The Normalised Difference Vegetation Index (NDVI) channel, NDVI = NIR - R / NIR + R (default: False)')
    parser.add_argument('--gNDVI', action='store_true', default=False, help='add gNDVI channel (default: False)')
    parser.add_argument('--SAVI', action='store_true', default=False, help='add The Soil Adjusted Vegetation Index (SAVI) channel, SAVI = [ ( NIR – R ) / ( NIR + R + L ) ] * (1 + L) (L = 0.5) (default: False)')
    parser.add_argument('--RVI', action='store_true', default=False, help='add The Ratio Vegetation Index (RVI) channel, RVI = R / NIR (default: False)')
    parser.add_argument('--DVI', action='store_true', default=False, help='add The Difference Vegetation Index (DVI) channel, DVI = NIR - R (default: False)')
    parser.add_argument('--VDVI', action='store_true', default=False, help='add The Visible Difference Vegetation Index (VDVI) channel, VDVI = ( (2*G) - R - B ) / ( (2 * G) + R + B ) (default: False)')
    parser.add_argument('--GCC', action='store_true', default=False, help='add The Green Chromatic Coordinate (GCC) channel, GCC = G / ( R + G + B ) (default: False)')
    parser.add_argument('--EVI', action='store_true', default=False, help='add The Enhanced Vegetation Index (EVI) channel, EVI = 2.5 * ( ( NIR - R ) / ( NIR + (6 * R) - ( 7.5 * B ) + 1 ) ) (default: False)')
    parser.add_argument('--VARI', action='store_true', default=False, help='add The Visible Atmospherically Resistant Index (VARI) channel, VARI = ( G - R) / ( G + R - B ) (default: False)')

    parser.add_argument('--GAI', nargs=10, help='add GAI channel',  type=float) # should we somehow pass normalizers data?
    parser.add_argument('--learn', '--l', action='append', choices=init_choices, help='add learnable channel. initialize from {}'.format(init_choices))

    args = parser.parse_args()
    return args