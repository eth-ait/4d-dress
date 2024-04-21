
from lib.utility.general import *
from lib.utility.dataset import DatasetUtils, XHumansUtils


# ------------------------ Locate Subj_Outfit_Seq Data ------------------------ # #

# set subj_outfit_seq target parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='4DDRESS', help='dataset name')
parser.add_argument('--subj', default='00122', help='subj name')
parser.add_argument('--outfit', default='Outer', help='outfit name')
parser.add_argument('--seq', default='Take9', help='seq name')
parser.add_argument('--save', action='store_false', help='save mode')
parser.add_argument('--update', action='store_true', help='update mode')
parser.add_argument('--new_label', default=None, help='new label')
# set args for RAFT
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()


# # ------------------------ Load Dataset Utility ------------------------ # #

# locate dataset folder
# dataset_dir = 'Please set your dataset_dir'
dataset_dir = '/mnt/scratch/shared/4d-dress/4D-DRESS'
# locate checkpoint folder
project_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint_dir = os.path.join(project_dir, 'checkpoints')

# init 4D-DRESS dataset_utils
if args.dataset == '4DDRESS':
    # dataset_dir = 'Please set your dataset_dir'
    dataset_utils = DatasetUtils(dataset_dir=dataset_dir, dataset_args=args, checkpoint_dir=checkpoint_dir, preprocess_init=True)
# init X-Humans dataset_utils
elif args.dataset == 'XHumans':
    # dataset_dir = 'Please set your dataset_dir'
    dataset_utils = XHumansUtils(dataset_dir=dataset_dir, dataset_args=args, checkpoint_dir=checkpoint_dir, preprocess_init=True)
else:
    raise Exception('Unknown dataset type {} among 4DDRESS, XHumans'.format(args.dataset))


# # ------------------------ Preprocess Subj_Outfit_Seq Meshes ------------------------ # #

# render, parser, optical, and segment scan_meshes within subj_outfit_seq
dataset_utils.render_parser_optical_segment_subj_outfit_seq_meshes(subj=args.subj, outfit=args.outfit, seq=args.seq,
                                                                   save=args.save, update=args.update, n_start=0, n_stop=-1)

# parse scan_meshes for the first frame, save sam masks into manual folder
dataset_utils.multi_surface_parsing(subj=args.subj, outfit=args.outfit, seq=args.seq, save_folder='labels_auto',
                                    save=args.save, update=args.update, n_start=0, n_stop=1, save_mask=True, new_label=args.new_label)
print('Wait for applying manual check and rectification for the first frame!')