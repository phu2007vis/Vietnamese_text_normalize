
from core.executing import Executor
from train import get_config
from transformers import AutoTokenizer

# def parse_args():
#     parser = argparse.ArgumentParser(description='Exp Args')

#     parser.add_argument("--mode", choices=['train', 'eval', 'predict'],
#                       help='{train, eval, predict}',
#                       type=str, required=True)
    
#     parser.add_argument("--evaltype", choices=['last', 'best'],
#                       help='{last, best}',
#                       type=str, nargs='?', const=1, default='last')
#     parser.add_argument("--predicttype", choices=['last', 'best'],
#                       help='{last, best}',
#                       type=str, nargs='?', const=1, default='best')
    
#     parser.add_argument("--config-file", type=str, required=True)

#     args = parser.parse_args()

#     return args
import gradio as gr
if __name__ == '__main__':
    # args = parse_args()

    config = get_config(r"C:\Users\9999\phuoc\paper\ViLexNorm\EnhancingViLexNorm\config\bartpho.yaml")

    exec = Executor(config, "predict",'best', 'best')
    tokenizer  = AutoTokenizer.from_pretrained("vinai/bartpho-syllable-base")
    exec.set_tokenizer(tokenizer)
    text = "nhung ma khong duoc dau"
    demo = gr.Interface(exec.infer_text,inputs = 'text', outputs = 'text')
    demo.launch()