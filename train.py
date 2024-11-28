from core.executing import Executor
import argparse
from yacs.config import CfgNode
import yaml

def get_config(yaml_file):
    return CfgNode(init_dict=yaml.load(open(yaml_file, "r"), Loader=yaml.FullLoader))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str,default="config/bartpho.yaml")
	args = parser.parse_args()
	trainer = Executor(config=get_config(args.config))
	trainer.run()




