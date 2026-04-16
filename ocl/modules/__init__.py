from ocl.modules import timm
from ocl.modules.decoders import build as build_decoder
from ocl.modules.encoders import build as build_encoder
from ocl.modules.groupers import build as build_grouper
from ocl.modules.initializers import build as build_initializer
from ocl.modules.networks import build as build_network
from ocl.modules.sam import build as build_sam
from ocl.modules.utils import Resizer, SoftToHardMask
from ocl.modules.utils import build as build_utils
from ocl.modules.utils import build_module, build_torch_function, build_torch_module
from ocl.modules.video import LatentProcessor, MapOverTime, ScanOverTime
from ocl.modules.video import build as build_video

__all__ = [
    "build_decoder",
    "build_encoder",
    "build_grouper",
    "build_initializer",
    "build_network",
    "build_sam",
    "build_utils",
    "build_module",
    "build_torch_module",
    "build_torch_function",
    "timm",
    "MapOverTime",
    "ScanOverTime",
    "LatentProcessor",
    "Resizer",
    "SoftToHardMask",
]


BUILD_FNS_BY_MODULE_GROUP = {
    "decoders": build_decoder,
    "encoders": build_encoder,
    "groupers": build_grouper,
    "initializers": build_initializer,
    "networks": build_network,
    "sam": build_sam,
    "utils": build_utils,
    "video": build_video,
    "torch": build_torch_function,
    "torch.nn": build_torch_module,
    "nn": build_torch_module,
}
