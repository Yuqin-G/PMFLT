from federated.server_base import ServerBase
from federated.server_fedtpg import Server_FedTPG
from federated.server_fedpgp import Server_FedPGP
from federated.server_fedotp import Server_FedOTP
from federated.server_promptfl import Server_PromptFL
from federated.server_fedclip import Server_FedCLIP
from federated.server_pfedmoap import Server_pFedMoAP
from federated.server_fedpha import Server_FedPHA
from federated.server_promptfolio import Server_PromptFolio
from federated.server_cbm import Server_CBM
from federated.server_fedmvp import Server_FedMVP

def get_server_class(model_name):
    server_mapping = {
        'fedtpg': Server_FedTPG,
        'fedmvp': Server_FedMVP,
        'coop': Server_FedTPG,  # CoOp uses FedTPG implementation
        'vlp': Server_FedTPG,   # VLP uses FedTPG implementation
        'kgcoop': Server_FedTPG, # kgCoOp uses FedTPG implementation
        'fedpgp': Server_FedPGP,
        'fedotp': Server_FedOTP,
        'promptfl': Server_PromptFL,
        'fedclip': Server_FedCLIP,
        'pfedmoap': Server_pFedMoAP,
        'fedpha': Server_FedPHA,
        'promptfolio': Server_PromptFolio,
        'cbm': Server_CBM,
    }
    
    return server_mapping.get(model_name, ServerBase)

def Server(cfg):
    model_name = cfg.MODEL.NAME
    server_class = get_server_class(model_name)
    return server_class(cfg)