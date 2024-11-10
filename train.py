from fog_env import Offload
from RL_brain import DeepQNetwork
import numpy as np
import random
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def reward_fun(delay, max_delay, unfinish_indi):

    # still use reward, but use the negative value
    penalty = - max_delay * 2

    if unfinish_indi:
        reward = penalty
    else:
        reward = - delay

    return reward


def train(iot_RL_list, NUM_EPISODE):

    RL_step = 0
    total_rewards = []  # Make sure to define this list at the start of the function

    for episode in range(NUM_EPISODE):

        print(f'Episode: {episode}')
        print(f'Epsilon: {iot_RL_list[0].epsilon}')
        
        # BITRATE ARRIVAL
        bitarrive = np.random.uniform(env.min_bit_arrive, env.max_bit_arrive, size=[env.n_time, env.n_iot])
        task_prob = env.task_arrive_prob
        bitarrive = bitarrive * (np.random.uniform(0, 1, size=[env.n_time, env.n_iot]) < task_prob)
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_iot])

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for iot_index in range(env.n_iot):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_iot])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset(bitarrive)

        total_episode_reward = 0  # Track reward for each episode

        # TRAIN DRL
        while True:
            action_all = np.zeros([env.n_iot])
            for iot_index in range(env.n_iot):
                observation = np.squeeze(observation_all[iot_index, :])

                if np.sum(observation) == 0:
                    action_all[iot_index] = 0
                else:
                    action_all[iot_index] = iot_RL_list[iot_index].choose_action(observation)

                if observation[0] != 0:
                    iot_RL_list[iot_index].do_store_action(episode, env.time_count, action_all[iot_index])

            observation_all_, lstm_state_all_, done = env.step(action_all)

            for iot_index in range(env.n_iot):
                iot_RL_list[iot_index].update_lstm(lstm_state_all_[iot_index,:])

            process_delay = env.process_delay
            unfinish_indi = env.process_delay_unfinish_ind

            for iot_index in range(env.n_iot):
                history[env.time_count - 1][iot_index]['observation'] = observation_all[iot_index, :]
                history[env.time_count - 1][iot_index]['lstm'] = np.squeeze(lstm_state_all[iot_index, :])
                history[env.time_count - 1][iot_index]['action'] = action_all[iot_index]
                history[env.time_count - 1][iot_index]['observation_'] = observation_all_[iot_index]
                history[env.time_count - 1][iot_index]['lstm_'] = np.squeeze(lstm_state_all_[iot_index,:])

                update_index = np.where((1 - reward_indicator[:,iot_index]) * process_delay[:,iot_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]
                        reward = reward_fun(process_delay[time_index, iot_index],
                                            env.max_delay, unfinish_indi[time_index, iot_index])
                        iot_RL_list[iot_index].store_transition(history[time_index][iot_index]['observation'],
                                                                history[time_index][iot_index]['lstm'],
                                                                history[time_index][iot_index]['action'],
                                                                reward,
                                                                history[time_index][iot_index]['observation_'],
                                                                history[time_index][iot_index]['lstm_'])
                        iot_RL_list[iot_index].do_store_reward(episode, time_index, reward)
                        iot_RL_list[iot_index].do_store_delay(episode, time_index, process_delay[time_index, iot_index])
                        reward_indicator[time_index, iot_index] = 1

                        total_episode_reward += reward  # Sum rewards for this episode

            RL_step += 1
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            if (RL_step > 200) and (RL_step % 10 == 0):
                for iot in range(env.n_iot):
                    iot_RL_list[iot].learn()

            if done:
                break

        print(f'Total Episode Reward: {total_episode_reward}')  # Check if total reward is being updated
        total_rewards.append(total_episode_reward)  # Store the total reward for this episode
    
    print(f'Total Rewards: {total_rewards}')  # Print rewards list to check if it's being populated

    # Plotting the reward graph
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Total Reward per Episode')
    plt.show()

        #  =================================================================================================
        #  ======================================== DRL END=================================================
        #  =================================================================================================


if __name__ == "__main__":

    NUM_IOT = 50
    NUM_FOG = 5
    NUM_EPISODE = 1000
    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    # GENERATE ENVIRONMENT
    env = Offload(NUM_IOT, NUM_FOG, NUM_TIME, MAX_DELAY)

    # GENERATE MULTIPLE CLASSES FOR RL
    iot_RL_list = list()
    for iot in range(NUM_IOT):
        iot_RL_list.append(DeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                        learning_rate=0.01,
                                        reward_decay=0.9,
                                        e_greedy=0.99,
                                        replace_target_iter=200,  # each 200 steps, update target net
                                        memory_size=500,  # maximum of memory
                                        ))

    # TRAIN THE SYSTEM
    train(iot_RL_list, NUM_EPISODE)
    print('Training Finished')

