import yaml
from agent import DQNAgent

def main():
    # Load configuration from YAML file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    altirra_path = config['altirra_path']
    rom_path = config['rom_path']
    input_shape = tuple(config['input_shape']) 
    num_actions = config["num_actions"] 
    num_episodes = config["num_episodes"]
    
    agent = DQNAgent(altirra_path, rom_path, input_shape, num_actions, config)
    agent.train(num_episodes)
    agent.close()

if __name__ == "__main__":
    main()
