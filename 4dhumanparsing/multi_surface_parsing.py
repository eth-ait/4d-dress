
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


# # ------------------------ Manually Rectify First Frame Subj_Outfit_Seq Mesh ------------------------ # #

# # # Step1: Check labels in the first frame, for instance, 00129_Outer_Take9 frame 00001
# subj, outfit, seq, start, stop = '00129', 'Outer', 'Take9', 0, 1

# # # Step2: Manually rectify mask_labels [frame, label, [[view_id, mask_id], ...]] and save in utility/manual.py
# # # Please check and uncomment rectified mask_labels in lib/utility/manual.py

# # # Step4: Rerun multi_surface_parsing for the rectified first frame, save in labels_auto
# dataset_utils.multi_surface_parsing(subj=args.subj, outfit=args.outfit, seq=args.seq, save_folder='labels_auto',
#                                     save=args.save, update=True, n_start=0, n_stop=1, save_mask=True, new_label=args.new_label)


# # ------------------------ Sequentially Parse All Subj_Outfit_Seq Meshes ------------------------ # #

# parse scan_meshes for all frames
dataset_utils.multi_surface_parsing(subj=args.subj, outfit=args.outfit, seq=args.seq, save_folder='labels_auto',
                                    save=args.save, update=args.update, n_start=0, n_stop=-1, save_mask=False, new_label=args.new_label)
# copy labels_auto to labels_manual for manual rectification
labels_auto_dir = os.path.join(dataset_utils.hyper_params['semantic_dir'], 'process', 'labels_auto')
labels_manual_dir = os.path.join(dataset_utils.hyper_params['semantic_dir'], 'process', 'labels_manual')
if not os.path.exists(labels_manual_dir): shutil.copytree(labels_auto_dir, labels_manual_dir)


# # ------------------------ Manually Rectify All Subj_Outfit_Seq Meshes ------------------------ # #

# # # Step1: Check labels and locate rectification frames, for instance, 00122_Outer_Take9 frames 00040~00044
# subj, outfit, seq, start, stop = '00122', 'Outer', 'Take9', 40, 5

# # # Step2: Save segment_masks in the manual folder for the above frames: 00122_Outer_Take9 frames 00040~00044
# dataset_utils.save_segment_masks(subj=args.subj, outfit=args.outfit, seq=args.seq, n_start=start, n_stop=stop)

# # # Step3: Manually rectify mask_labels [frame, label, [[view_id, mask_id], ...]] and save in utility/manual.py
# # # Please check and uncomment rectified mask_labels in lib/utility/manual.py

# # # Step4: Rerun multi_surface_parsing for the rectified frames, save into labels_manual
# dataset_utils.multi_surface_parsing(subj=args.subj, outfit=args.outfit, seq=args.seq, save_folder='labels_manual',
#                                     save=args.save, update=True, n_start=start, n_stop=stop, save_mask=False, new_label=args.new_label)

