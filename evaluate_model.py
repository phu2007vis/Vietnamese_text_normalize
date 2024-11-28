
from core.executing import Executor
from train import get_config
from transformers import AutoTokenizer



#     return args
import gradio as gr
if __name__ == '__main__':
    # args = parse_args()

    config = get_config(r"C:\Users\9999\phuoc\paper\ViLexNorm\EnhancingViLexNorm\config\bartpho.yaml")

    exec = Executor(config, "predict",'best', 'best')
    import pdb;pdb.set_trace()
    exec.run()