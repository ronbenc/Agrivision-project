import json

from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader
from ptsemseg.loader.mit_sceneparsing_benchmark_loader import MITSceneParsingBenchmarkLoader
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.nyuv2_loader import NYUv2Loader
from ptsemseg.loader.sunrgbd_loader import SUNRGBDLoader
from ptsemseg.loader.mapillary_vistas_loader import mapillaryVistasLoader

from ptsemseg.loader.agrivision6_loader import agrivision6Loader, agrivision6Loader_expert_23, agrivision6Loader_expert_26, agrivision6Loader_agct
from ptsemseg.loader.agrivision6_loader_rat import agrivision6Loader_rat1, agrivision6Loader_rat2
from ptsemseg.loader.agrivision6_loader_pseudo import agrivision6Loader_pseudo, agrivision6Loader_agct_pseudo
from ptsemseg.loader.agrivision6_loader_full_field import agrivision6Loader_full_field


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "pascal": pascalVOCLoader,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "nyuv2": NYUv2Loader,
        "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
        "agrivision6": agrivision6Loader,
        "agrivision6Loader_agct": agrivision6Loader_agct,
        "agrivision6_expert_cat_23": agrivision6Loader_expert_23,
        "agrivision6_expert_cat_26": agrivision6Loader_expert_26,
        "agrivision6Loader_rat1": agrivision6Loader_rat1,
        "agrivision6Loader_pseudo": agrivision6Loader_pseudo,
        "agrivision6Loader_agct_pseudo": agrivision6Loader_agct_pseudo,
        "agrivision6Loader_full_field": agrivision6Loader_full_field,
    }[name]
