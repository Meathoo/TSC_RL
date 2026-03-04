from agentpool.AdaptiveBDQ_agent import AdaptiveBDQ_agent
from agentpool.BDQ_agent import BrachingDQ_agent
from agentpool.CommBDQ_agent import CommBDQ_agent

EXP_CONFIG = {
    "EPISODE": 2,
    "TRAINING_PARADIM": "DECENTRAL",  # "CLDE",  # CLDE: centralised learning but decentralised execution ,DECENTRAL
    "AGENT_CLASS_DICT": {
        "BDQ": BrachingDQ_agent,
        "ABDQ": AdaptiveBDQ_agent,
        "CommBDQ": CommBDQ_agent,    # Cross-Region Attention Communication BDQ
    },
    "REGIONAL": True,
    "REGION_TYPE": "ADJACENCY1",
    "LEARNING_INTERVAL": 10,
}
