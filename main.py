import os
import argparse
import collections
from datetime import datetime

from config.hparams import *
from pretrain import VQAModel
from train import VisDialModel


def pretrain_model(args):
    hparams = HPARAMS
    
    root_dir = hparams["root_dir"]
    root_dir += "%s-%s" % (hparams["encoder"], hparams["decoder"])
    hparams.update(root_dir=root_dir)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(hparams["root_dir"], "%s/" % timestamp)
    hparams["root_dir"] = root_dir

    save_dirpath = hparams["save_dirpath"] + "/vqa/" + args.folder
    result_dirpath = hparams["result_dirpath"] + "/vqa/" + args.folder
    fig_dirpath = hparams["fig_dirpath"] + "/vqa/" + args.folder
    hparams.update(save_dirpath=save_dirpath)
    hparams.update(result_dirpath=result_dirpath)
    hparams.update(fig_dirpath=fig_dirpath)
    hparams.update(gamma=args.gamma)
    
    # if hparams["load_pthpath"] != "":
    #     hparams.update(random_seed=[int(args.eval_seed)])

    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
    
    model = VQAModel(hparams)
    model.train()
    # from model.vqa.evaluation import VQAEvaluator
    # evaluator = VQAEvaluator(hparams)
    # evaluator.run("checkpoints/vqa_test/checkpoint_6.pth")


def train_model(args):
    hparams = HPARAMS
    
    root_dir = hparams["root_dir"]
    root_dir += "%s-%s" % (hparams["encoder"], hparams["decoder"])
    hparams.update(root_dir=root_dir)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(hparams["root_dir"], "%s/" % timestamp)
    hparams["root_dir"] = root_dir

    save_dirpath = hparams["save_dirpath"] + "/visdial/" + args.folder
    result_dirpath = hparams["result_dirpath"] + "/visdial/" + args.folder
    load_pthpath = hparams["load_pthpath"] + "checkpoints/vqa/" + args.load_folder + "/checkpoint_vqa.pth"
    hparams.update(save_dirpath=save_dirpath)
    hparams.update(result_dirpath=result_dirpath)
    hparams.update(load_pthpath=load_pthpath)
    hparams.update(num_samples=args.sample_size)
    
    # if hparams["load_pthpath"] != "":
    #     hparams.update(random_seed=[int(args.eval_seed)])

    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

    model = VisDialModel(hparams)
    model.train()
    # if hparams.dataset_version == '1.0':
    # model = MultiEvaluation(hparams, split="test")
    # model.run_evaluate(
    #     eval_ckpt_path,
    #     eval_json_path=os.path.join(os.path.dirname(eval_ckpt_path),
    #                                 "%s_%d_test.json" %
    #                                 (hparams.encoder + "-" + hparams.decoder, hparams.random_seed[0])))

def main(args):
    if args.pretrain:
        print("VQA MODEL PRE-TRAIN\n")
        pretrain_model(args)
    else: 
        print("VISUAL DIALOG MODEL TRAIN\n")
        train_model(args)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Variational Attention and Disentanglement with Regularization for Conversational Question Answering")
    arg_parser.add_argument("--pretrain", dest="pretrain", type=str, default=False, help="Pre-train VQA model or not")
    arg_parser.add_argument("--gamma", dest="gamma", type=float, default=0.5, help="Regularization coefficient")
    arg_parser.add_argument("--folder", dest="folder", type=str, default="VQA_FOLDER", help="Save folder name")

    arg_parser.add_argument("--load_folder", dest="load_folder", type=str, default="VQA_FOLDER", help="Folder of pretrained model")
    arg_parser.add_argument("--load_pth", dest="load_pth", type=str, default="checkpoint_vqa.pth", help="Loadding pth file")
    arg_parser.add_argument("--sample_size", dest="sample_size", type=int, default=25, help="Sampled size of CLT")
    # arg_parser.add_argument("--eval_split", dest="eval_split", type=str,
    #                         help="Evaluation split", default="test")
    # arg_parser.add_argument("--description", dest="description", type=str, default='',
    #                         help="Image Region Proposal Type")
    # arg_parser.add_argument("--eval_seed", dest="eval_seed", type=str,
    #                         help="Evaluation split", default="3143")
    
    args = arg_parser.parse_args()
    main(args)