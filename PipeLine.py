import os
import numpy as np
import time
import json
from configs import agent_config

def assign_state(state,itsx_assignment,itsx_state_dim):

    observations = []  # the collection of observations of all agents
    for assignment in itsx_assignment:
        if len(assignment) == 0:
            assignment = [assignment]
        agent_obs = np.zeros((len(assignment),itsx_state_dim))  # observation of current agent
        for id, itsx in enumerate(assignment):
            if itsx != 'dummy':
                # assign the state of one intersection to correct position of agent observation
                # leave the state of imaginary intersection as 0s
                # print(agent_obs)
                # print(state)
                agent_obs[id] = state[itsx]

        observations.append(np.hstack(agent_obs))  #
    return observations
def assign_reward(raw_reward,itsx_assignment):
    """
    build regional reward from raw reward of intersections for each agent according to the intersection assignments
    :param raw_reward: a dictionary {key,value} key(String): intersection id,
                                               value(int): reward of that intersection
    :return:
    """

    rewards = []
    for assignment in itsx_assignment:
        if len(assignment)==0:
            assignment=[assignment]
        agent_reward = 0  # reward of current agent
        for id, itsx in enumerate(assignment):
            if itsx != 'dummy':
                # sum the reward of one intersection
                # leave the reward of imaginary intersection as 0
                agent_reward += raw_reward[itsx]
        rewards.append(agent_reward)
    return rewards

def get_idle_branches(itsx_assignment):
    invalid_indices = []
    for assign in itsx_assignment:
        temp_list = []
        for id, name in enumerate(assign):
            if name == "dummy":
                temp_list.append(id)
        invalid_indices.append(temp_list)
    return invalid_indices

def convert_actions(actions_id,itsx_assignment):
    decoded_actions = {}
    for actions, itsxs in zip(actions_id, itsx_assignment):
        for a, itsx in zip(actions, itsxs):
            if itsx != "dummy":
                decoded_actions[itsx] = int(a) + 1
    return decoded_actions
def pipeline(env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG):
    """
    Training pipeline that communicate agent with environment

    :param env:
    :param agents:
    :param itsx_assignment:
    :param EXP_CONFIG:
    :param ENV_CONFIG:
    :return:

    """
    def agent_learn():
        if EXP_CONFIG["TRAINING_PARADIM"] == "CLDE":
            agents[0].learn()
        else:
            for i in range(len(agents)):
                agents[i].learn()
    idle_branches_id=get_idle_branches(itsx_assignment)
    """----init global log-----"""
    global_step=0 # global training step
    episode_intersection_level_rewards=[]
    episode_throughput=[]
    episode_queue_length_terminal = []  # queue snapshot at episode end
    episode_queue_length_episode_avg = []  # step-wise episode-average queue length
    episode_travel_time=[]
    """-------------"""

    # ---- Training Log (timing + key metrics) ----
    training_start_time = time.time()
    episode_logs = []  # per-episode detailed log
    learn_call_count = 0  # total learn() calls across all agents

    for episode in range(EXP_CONFIG["EPISODE"]):
        state=env.reset()
        obs=assign_state(state,itsx_assignment,EXP_CONFIG["ITSX_STATE_DIM"])

        """---init episode log---"""
        episode_start_time=time.time()
        step_itsx_reward=[]
        episode_reward=[]
        step_queue_length = []  # accumulate per-step for time-averaged AQL
        """-------"""
        for step in range(int(ENV_CONFIG["SIM_TIMESPAN"]/ENV_CONFIG["ACTION_INTERVAL"])):
            actions_id=[]
            global_step += 1

            for aid,agent in enumerate(agents):
                action_id=agent.choose_action(obs[aid],idle_branches_id[aid])
                actions_id.append(action_id)
            joint_actions=convert_actions(actions_id,itsx_assignment)

            next_states, itsx_rewards, done, log_metric = env.step(joint_actions)

            next_obs = assign_state(next_states,itsx_assignment,EXP_CONFIG["ITSX_STATE_DIM"])
            rewards = assign_reward(itsx_rewards,itsx_assignment)

            if not done:
                # only store experience and learn when not finished
                for aid in range(len(agents)):
                    agents[aid].store_transition(obs[aid], actions_id[aid], rewards[aid], next_obs[aid])
                if global_step % EXP_CONFIG["LEARNING_INTERVAL"] == 0:
                    agent_learn()
                    learn_call_count += 1
            """----update step log---"""
            step_itsx_reward.append([value for _,value in itsx_rewards.items()])
            episode_reward.append(np.average(rewards))
            step_queue_length.append(env.get_average_queue_length())  # collect step queue for episode average
            """-----------------"""
            obs=next_obs
        """----update episode log------"""
        terminal_queue_length = env.get_average_queue_length()
        episode_queue_avg = float(np.mean(step_queue_length)) if step_queue_length else terminal_queue_length
        episode_throughput.append(env.get_throughput())
        episode_travel_time.append(env.get_average_travel_time())
        episode_queue_length_terminal.append(terminal_queue_length)
        episode_queue_length_episode_avg.append(episode_queue_avg)
        episode_intersection_level_rewards.append(np.array(step_itsx_reward))
        """----------------------------"""
        episode_wall_time = time.time() - episode_start_time
        elapsed_total = time.time() - training_start_time

        # Collect per-agent metrics
        agent_grad_norms = []
        agent_lrs = []
        agent_losses = []
        agent_memory_counts = []
        for agent in agents:
            if hasattr(agent, 'grad_norm_his') and len(agent.grad_norm_his) > 0:
                agent_grad_norms.append(float(agent.grad_norm_his[-1]))
            if hasattr(agent, 'lr_his') and len(agent.lr_his) > 0:
                agent_lrs.append(float(agent.lr_his[-1]))
            if hasattr(agent, 'loss_his') and len(agent.loss_his) > 0:
                agent_losses.append(float(agent.loss_his[-1]))
            if hasattr(agent, 'memory_counter'):
                agent_memory_counts.append(int(agent.memory_counter))

        ep_log = {
            "episode": episode,
            "wall_time_sec": round(episode_wall_time, 2),
            "elapsed_total_sec": round(elapsed_total, 2),
            "elapsed_total_hms": time.strftime('%H:%M:%S', time.gmtime(elapsed_total)),
            "global_step": global_step,
            "episode_reward": round(float(np.sum(episode_reward)), 4),
            "avg_travel_time": round(env.get_average_travel_time(), 4),
            "avg_queue_length": round(terminal_queue_length, 4),
            "avg_queue_length_terminal": round(terminal_queue_length, 4),
            "avg_queue_length_episode": round(episode_queue_avg, 4),
            "throughput": int(env.get_throughput()),
            "epsilon": round(float(agents[0].epsilon), 6),
            "avg_grad_norm": round(float(np.mean(agent_grad_norms)), 4) if agent_grad_norms else None,
            "max_grad_norm": round(float(np.max(agent_grad_norms)), 4) if agent_grad_norms else None,
            "avg_lr": round(float(np.mean(agent_lrs)), 8) if agent_lrs else None,
            "avg_loss": round(float(np.mean(agent_losses)), 6) if agent_losses else None,
            "memory_counter": agent_memory_counts[0] if agent_memory_counts else 0,
            "learn_calls": learn_call_count,
        }
        episode_logs.append(ep_log)

        log_msg={
            "episode":episode,
            "time cost":round(episode_wall_time, 2),
            "elapsed": time.strftime('%H:%M:%S', time.gmtime(elapsed_total)),
            "episode reward":round(float(np.sum(episode_reward)), 2),
            "average travel time": round(env.get_average_travel_time(), 2),
            "average queue length": round(terminal_queue_length, 4),
            "average queue length (episode avg)": round(episode_queue_avg, 4),
            "throughput":int(env.get_throughput()),
            "epsilon":round(float(agents[0].epsilon), 6),
            "grad_norm": round(float(np.mean(agent_grad_norms)), 2) if agent_grad_norms else '-',
            "lr": round(float(np.mean(agent_lrs)), 8) if agent_lrs else '-',
        }
        print("log_msg:\n", log_msg)

        # Estimate remaining time
        if episode > 0:
            avg_ep_time = elapsed_total / (episode + 1)
            remaining_eps = EXP_CONFIG["EPISODE"] - episode - 1
            eta_sec = avg_ep_time * remaining_eps
            print(f"  ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_sec))} "
                  f"({remaining_eps} eps remaining, "
                  f"avg {avg_ep_time:.1f}s/ep)")

    # ---- Write Training Log ----
    total_training_time = time.time() - training_start_time
    att_array = np.array(episode_travel_time)
    best_att = float(att_array.min())
    best_att_ep = int(att_array.argmin())
    last100_att = float(att_array[-100:].mean()) if len(att_array) >= 100 else float(att_array.mean())

    aql_array = np.array(episode_queue_length_terminal)
    best_aql = float(aql_array.min())
    best_aql_ep = int(aql_array.argmin())
    aql_episode_array = np.array(episode_queue_length_episode_avg)
    best_aql_episode = float(aql_episode_array.min())
    best_aql_episode_ep = int(aql_episode_array.argmin())

    tp_array = np.array(episode_throughput)
    best_throughput = int(tp_array.max())
    best_throughput_ep = int(tp_array.argmax())

    # Find convergence point (rolling-20 < 480)
    converge_ep = None
    if len(att_array) >= 20:
        rolling = np.convolve(att_array, np.ones(20)/20, mode='valid')
        for i in range(len(rolling)):
            if rolling[i] < 480:
                converge_ep = i
                break

    # per-agent param counts
    agent_param_counts = []
    for agent in agents:
        pc = sum(p.numpy().size for p in agent.eval_model.trainable_weights)
        agent_param_counts.append(pc)

    training_summary = {
        "total_training_time_sec": round(total_training_time, 1),
        "total_training_time_hms": time.strftime('%H:%M:%S', time.gmtime(total_training_time)),
        "total_episodes": EXP_CONFIG["EPISODE"],
        "total_global_steps": global_step,
        "total_learn_calls": learn_call_count,
        "avg_sec_per_episode": round(total_training_time / max(EXP_CONFIG["EPISODE"], 1), 2),
        "best_att": round(best_att, 2),
        "best_att_episode": best_att_ep,
        "last100_att": round(last100_att, 2),
        "best_aql": round(best_aql, 4),
        "best_aql_episode": best_aql_ep,
        "best_aql_episode_avg": round(best_aql_episode, 4),
        "best_aql_episode_avg_episode": best_aql_episode_ep,
        "best_throughput": best_throughput,
        "best_throughput_episode": best_throughput_ep,
        "converge_episode_rolling20_lt480": converge_ep,
        "agent_param_counts": agent_param_counts,
        "features_enabled": {
            "gradient_clipping": agent_config.BDQ_AGENT_CONFIG.get("Gradient_Clipping", False),
            "cosine_lr": agent_config.BDQ_AGENT_CONFIG.get("CosineDecay", False),
            "PER": agent_config.BDQ_AGENT_CONFIG.get("Prioritized_Experience_Replay", False),
            "noisy_net": agent_config.BDQ_AGENT_CONFIG.get("NoisyNet", False),
        },
    }

    training_log = {
        "summary": training_summary,
        "episodes": episode_logs,
    }

    log_path = os.path.join(ENV_CONFIG["PATH_TO_WORK_DIRECTORY"], "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    # Print final summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time:       {training_summary['total_training_time_hms']} "
          f"({training_summary['total_training_time_sec']:.0f}s)")
    print(f"  Avg per episode:  {training_summary['avg_sec_per_episode']}s")
    print(f"  Total steps:      {global_step:,}")
    print(f"  Learn calls:      {learn_call_count:,}")
    print(f"  Best ATT:         {best_att:.2f} (ep {best_att_ep})")
    print(f"  Last-100 ATT:     {last100_att:.2f}")
    if converge_ep is not None:
        print(f"  Converge ep:      {converge_ep} (rolling-20 < 480)")
    else:
        print(f"  Converge ep:      Not reached")
    print(f"  Params/agent:     {agent_param_counts[0]:,}")
    print(f"  Log saved:        {log_path}")
    print("=" * 60 + "\n")

    sim_log={'reward_log':episode_intersection_level_rewards,
             'throughput':episode_throughput,
             'queue_length':episode_queue_length_terminal,
             'queue_length_terminal':episode_queue_length_terminal,
             'queue_length_episode_avg':episode_queue_length_episode_avg,
             'travel_time':episode_travel_time
             }
    return sim_log