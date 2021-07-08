from config import EndToEndConfig
from utils.networks import StatelessToStateful

#################################################################
# This script creates a stateful version of a stateless network #
#################################################################

config = EndToEndConfig()

network_dir = config.get_trained_network_dir()
network, network_info, _ = config.load_trained_network(network_dir)

convert = StatelessToStateful(network, network_info, network_dir)

convert.save()
