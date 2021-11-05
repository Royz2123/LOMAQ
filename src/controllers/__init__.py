
REGISTRY = {}

from .central_basic_controller import CentralBasicMAC
from .basic_controller import BasicMAC
from .hetro_controller import HetroMac

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["hetro_mac"] = HetroMac
REGISTRY["basic_central_mac"] = CentralBasicMAC
