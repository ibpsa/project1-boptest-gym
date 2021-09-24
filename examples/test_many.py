'''
Test many RL cases to generate KPIs

'''

from examples.train_RL import train_RL, test_peak, test_typi
import matplotlib.pyplot as plt

if __name__ == "__main__":
    render = False
    plot = not render # Plot does not work together with render    
    warmup_period_test  = 7*24*3600
    episode_length_test = 14*24*3600
    save_to_file = True

    model_names = [
        'model_10000',
        'model_50000',
        'model_100000',
        'model_200000',
        'model_300000',
        'model_400000',
        'model_500000',
        'model_600000',
        'model_700000',
        'model_800000',
        'model_900000',
        'model_1000000',
        ]

    for model_name in model_names:
        env, model, start_time_tests, log_dir = train_RL(algorithm='DQN', mode='load', case='C', training_timesteps=1e6, render=render, model_name=model_name)
        plt.close('all')
        test_peak(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, model_name, save_to_file, plot)
        test_typi(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, model_name, save_to_file, plot)
    
    