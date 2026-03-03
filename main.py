import tensorflow as tf
import argparse
import numpy as np
from cityflow_env_wrapper import CityflowEnvWrapper
import time
import os
import shutil
from configs import agent_config, env_config, exp_config, region_config
import copy
from PipeLine import pipeline
from agentpool.region_comm import build_region_adjacency, RegionCommCoordinator
# tf.random.set_seed(0)

#add
from plot import plot_curve
import json

tf.config.experimental_run_functions_eagerly(True)


# np.random.se
# np.random.seed(0)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netname", type=str, default="Hangzhou")
    parser.add_argument("--netshape", type=str, default="4_4")
    parser.add_argument("--flow", type=str, default="real")
    parser.add_argument("--agent", type=str, default="ABDQ")
    return parser.parse_args()

# python main.py --netname Hangzhou --netshape 4_4 --flow real --agent ABDQ

def init_exp(args):
    """
    Based on arguments and experiment configuration to initialise experiment
    Global working directory
    1.construct environment object

    2.agents object

    :param args:
    :return:
    """
    # retrieve arguments
    # netname = "Manhattan"
    # netshape = "16_3"
    # flow = "real"
    netname = args.netname
    netshape = args.netshape
    flow = args.flow
    agent_type = args.agent
    AGENT_CONFIG = copy.deepcopy(agent_config.AGENT_CONFIG)
    BDQ_AGENT_CONFIG = copy.deepcopy(agent_config.BDQ_AGENT_CONFIG)
    ENV_CONFIG = copy.deepcopy(env_config.ENV_CONFIG)
    EXP_CONFIG = copy.deepcopy(exp_config.EXP_CONFIG)
    # EXP_CONFIG["EPISODE"]=2000                           # use 2000 episodes for experiments in paper
    # Construct Cityflow Configuration File and ENV DICT

    roadnet_file = "roadnet_" + netshape + ".json"

    flow_file = netname + "_" + netshape + "_" + flow + ".json"
    net_path = os.path.join(netname, roadnet_file)
    flow_path = os.path.join(netname, flow_file)

    # add
    if netname == "Hangzhou" and netshape == "4_4":
        if flow == "flat":
            flow_path = os.path.join(netname, "Hangzhou_4_4_real_5734.json")
        elif flow == "peak":
            flow_path = os.path.join(netname, "Hangzhou_4_4_real_5816.json")
    if netname == "Syn" and netshape == "4_4":
        flow_path = os.path.join(netname, "syn_4_4_gaussian_500_1h.json")
        flow = "gaussian"
    if netname == "Syn2" and netshape == "4_4":
        flow_path = os.path.join(netname, "Syn2_4_4_gaussian_500_1h.json")
        flow = "gaussian"
    # if netname == "Manhattan" and netshape == "4_4":
    #     flow_path = os.path.join(netname, "Syn2_4_4_gaussian_500_1h.json")
    #     flow = "gaussian"


    experiment_name = "{0}_{1}_{2}".format(netname, netshape, flow)
    experiment_date = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    ENV_CONFIG["PATH_TO_WORK_DIRECTORY"] = os.path.join("records", experiment_name + "_" + experiment_date)
    if not os.path.exists(ENV_CONFIG["PATH_TO_WORK_DIRECTORY"]):
        os.makedirs(ENV_CONFIG["PATH_TO_WORK_DIRECTORY"])
    ENV_CONFIG["ROADNET_PATH"] = net_path
    ENV_CONFIG["FLOW_PATH"] = flow_path

    os.makedirs(f'{ENV_CONFIG["PATH_TO_WORK_DIRECTORY"]}/frontend/web')
    with open(f'{ENV_CONFIG["PATH_TO_WORK_DIRECTORY"]}/frontend/web/replayLogFile.txt', "w") as file:
        file.write("")
    with open(f'{ENV_CONFIG["PATH_TO_WORK_DIRECTORY"]}/frontend/web/roadnetLogFile.json', "w") as file:
        file.write("")
    env = CityflowEnvWrapper(ENV_CONFIG)



    # Construct Agent
    # we need state dim , action dim, subaction num
    itsx_state_dim=25 # queue lengh: 12, wave: 12, current phase: 1
    if ENV_CONFIG["ACTION_TYPE"]=="CHOOSE PHASE":
        itsx_action_dim=4
    elif ENV_CONFIG["ACTION_TYPE"]=="SWTICH":
        itsx_action_dim=2
    else:
        raise Exception("Unknow phase control")

    if EXP_CONFIG["REGIONAL"]:
        region_assignment_key=netshape+'_'+EXP_CONFIG["REGION_TYPE"]
        itsx_assignment= region_config.REGION_CONFIG[region_assignment_key]
    else:
        itsx_assignment=[[itsx_id] for itsx_id in env.intersection_ids]
    EXP_CONFIG["AGETN_NUM"]=len(itsx_assignment)
    agent_env_config={
        "ITSX_STATE_DIM":itsx_state_dim,
        "ACTION_DIM":len(itsx_assignment[0]),
        "ITSX_ACTION_DIM":itsx_action_dim,
    }
    EXP_CONFIG.update(agent_env_config)

    # ====== Cross-Region Communication Setup ======
    comm_coordinator = None
    if agent_type == "CommBDQ":
        # 自動建構 Region 鄰接圖
        adjacency = build_region_adjacency(itsx_assignment)
        comm_dim = agent_config.COMM_BDQ_AGENT_CONFIG.get("COMM_DIM", 64)
        max_neighbors = max(len(n) for n in adjacency) if adjacency else 1
        comm_coordinator = RegionCommCoordinator(adjacency, comm_dim)
        # 傳入通訊配置給 Agent
        comm_config = {"COMM_DIM": comm_dim, "MAX_NEIGHBORS": max_neighbors}
    else:
        comm_config = None

    if EXP_CONFIG["TRAINING_PARADIM"]=="CLDE":
        if comm_config is not None:
            agent=EXP_CONFIG["AGENT_CLASS_DICT"][agent_type](agent_env_config, comm_config)
        else:
            agent=EXP_CONFIG["AGENT_CLASS_DICT"][agent_type](agent_env_config)
        agents=[agent for _ in range(EXP_CONFIG["AGETN_NUM"])]
    else:
        if comm_config is not None:
            agents = [EXP_CONFIG["AGENT_CLASS_DICT"][agent_type](agent_env_config, comm_config) for _ in range(EXP_CONFIG["AGETN_NUM"])]
        else:
            agents = [EXP_CONFIG["AGENT_CLASS_DICT"][agent_type](agent_env_config) for _ in range(EXP_CONFIG["AGETN_NUM"])]
    print(EXP_CONFIG)
    print(ENV_CONFIG)


    # add
    # 在寫入前，移除不可序列化的 key
    exp_config_serializable = copy.deepcopy(EXP_CONFIG)
    if "AGENT_CLASS_DICT" in exp_config_serializable:
        exp_config_serializable["AGENT_CLASS_DICT"] = {k: str(v) for k, v in exp_config_serializable["AGENT_CLASS_DICT"].items()}

    with open(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'], 'hyperparameter.txt'), 'w') as file:
        file.write("agent_config:\n" + json.dumps(AGENT_CONFIG, indent=2) + "\n\n")
        file.write("agent_config:\n" + json.dumps(BDQ_AGENT_CONFIG, indent=2) + "\n\n")
        file.write("EXP_CONFIG:\n" + json.dumps(exp_config_serializable, indent=2) + "\n\n")
        file.write("ENV_CONFIG:\n" + json.dumps(ENV_CONFIG, indent=2) + "\n\n")

    return env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG,comm_coordinator

def run_pipeline(env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG,comm_coordinator=None):
    logs=pipeline(env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG,comm_coordinator)
    plot_curve(np.array(logs['reward_log']), "Episode Intersection Reward", "Intersection Reward", os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'], "plot_intersection_reward.png")) # add
    np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_intersection_reward.npy'),logs['reward_log'])
    plot_curve(np.array(logs['throughput']), "Episode Throughput", "Throughput", os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'], "plot_throughput.png")) # add
    np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_throughput.npy'), logs['throughput'])
    plot_curve(np.array(logs['travel_time']), "Episode Average Travel Time", "Average Travel Time", os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'], "plot_average_travel_time.png")) # add
    np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_average_travel_time.npy'), logs['travel_time']) 
    plot_curve(np.array(logs['queue_length']), "Episode Average Queue Length", "Average Queue Length", os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'], "plot_average_queue_length.png")) # add
    np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_average_queue_length.npy'), logs['queue_length'])  # add

if __name__ == "__main__":
    print("Is GPU available",tf.test.is_gpu_available())
    args = parse_args()
    env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG,comm_coordinator=init_exp(args)
    run_pipeline(env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG,comm_coordinator)

    # add by me
    model_folder = os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'], 'models')
    for i, agent in enumerate(agents):
        agent.save_model(os.path.join(model_folder, f"agent_{i}"))

    # for name in dataset_list:
    # main(name)


# python main.py --netname Hangzhou --netshape 4_4 --flow real --agent ABDQ