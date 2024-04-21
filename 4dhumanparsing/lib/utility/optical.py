from lib.utility.general import *

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'RAFT/core'))
from lib.RAFT.core.raft import RAFT
from lib.RAFT.core.utils.utils import InputPadder
# In case you encounter the error like "No module named update" or "No module named utils.utils"
# Please modify "import update" to "import .update", "import utils.utils" to "import .utils.utils" within "lib/RAFT/core" accordingly


class Rafter:
    """
    Optical Flow Predictor
    """
    def __init__(self, model_path='', model_args=None, init_model=False, dtype=torch.float32, device=torch.device('cuda:0')):

        # init dtype and device
        self.dtype = dtype
        self.device = device
        # init model args
        self.model_args = model_args
        # locate checkpoint
        self.checkpoint_fn = os.path.join(model_path, 'models/raft-things.pth')

        # init network for optical flow only
        if init_model:
            # load network with checkpoint
            self.network = self.get_network()

    # get network
    def get_network(self):
        # init model
        model = torch.nn.DataParallel(RAFT(self.model_args))
        # load model checkpoint
        model.load_state_dict(torch.load(self.checkpoint_fn))
        # set model device and eval
        model = model.module
        model.to(self.device)
        model.eval()
        return model

    # save optical flow to npz
    def save_optical_flows(self, optical_fn, optical_flows):
        np.savez_compressed(optical_fn, optical=optical_flows)
    
    # load optical flow from npz
    def load_optical_flows(self, optical_fn):
        return np.load(optical_fn)['optical']
    
    # preprocess numpy or torch image
    def preprocess_image(self, image):
        # clone tensor images
        if torch.is_tensor(image):
            return torch.tensor(image.clone(), dtype=self.dtype, device=self.device).permute(2, 0, 1).unsqueeze(0)
        # send image to torch device, convert (h, w, c) to (c, h, w)
        return torch.tensor(image, dtype=self.dtype, device=self.device).permute(2, 0, 1).unsqueeze(0)

    # raft image_pairs (nv, h, w, c) to optical_flows (nv, h, w, 2)
    @torch.no_grad()
    def process_images_pairs(self, images1, images2):
        # init raft_flows
        raft_flows = []
        # loop over all image pairs
        for nv in range(images1.shape[0]):
            # preprocess opencv images
            img1 = self.preprocess_image(images1[nv])
            img2 = self.preprocess_image(images2[nv])
            # pad input images to 8
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)
            # predict flow
            flow_low, flow_up = self.network(img1, img2, iters=20, test_mode=True)
            # convert (2, x, y) to (h, w, 2)
            flow12 = flow_up[0].permute(1, 2, 0)
            flow12[..., [1, 0]] = flow12[..., [0, 1]]
            # append raft_flows
            raft_flows.append(flow12)
        return torch.stack(raft_flows, dim=0)

    # get raft_labels (nv, h, w, nl) from labels (nv, h, w) and flows (nv, h, w, 2)
    @torch.no_grad()
    def transfer_labels(self, labels, flows, surface_labels):
        # init label dimension
        n_views, h, w = labels.shape[:3]
        # init raft_labels
        raft_labels = []
        # process all views
        for nv in range(n_views):
            # init raft_label
            raft_label = []
            # loop over all surface_labels
            for nl in range(len(surface_labels)):
                # init empty label
                warp_label = torch.zeros((h, w))
                # init target_labels
                target_labels = labels[nv] == nl
                # assign warp_label
                if torch.count_nonzero(target_labels) > 0:
                    # get label points
                    label_points = torch.stack(torch.where(target_labels)).T
                    # assign warp_points
                    warp_points = (label_points + flows[nv][label_points[:, 0], label_points[:, 1]]).type(torch.long)
                    warp_points[warp_points < 0] = 0
                    warp_points[:, 0][warp_points[:, 0] > h - 1] = h - 1
                    warp_points[:, 1][warp_points[:, 1] > w - 1] = w - 1
                    warp_label[warp_points[:, 0], warp_points[:, 1]] = 1.
                # append warp_label
                raft_label.append(warp_label)
            # append raft_labels [(h, w, nl), ...]
            raft_labels.append(torch.stack(raft_label, dim=-1))
        # return raft_labels (nv, h, w, nl)
        return torch.stack(raft_labels, dim=0)

    # draw optical flows on images (nv, h, w, 3)
    def draw_sampled_flows(self, images, flows, n_sample=32, mask=False):
        # set image dimension
        n_views, h, w, c = images.shape
        # set grid with n_sample
        image_grid = create_grid(n_sample, sx=h, sy=w).astype(int)
        # draw sampled optical flows on all images
        for nv in range(n_views):
            # set body mask
            body_mask = (np.sum(images[nv], axis=-1) != 255 * 3)
            # draw flow lines
            for ng in range(image_grid.shape[0]):
                # mask body region
                if mask and not body_mask[image_grid[ng, 0], image_grid[ng, 1]]: continue
                # get hwflow in xy frame
                hwflow = flows[nv][image_grid[ng, 0], image_grid[ng, 1]]
                # draw hwflow line and point
                start_point = (image_grid[ng, 0], image_grid[ng, 1])
                end_point = (int(max(min(image_grid[ng, 0] + hwflow[0], h), 0)), int(max(min(image_grid[ng, 1] + hwflow[1], w), 0)))
                draw_line(images[nv], start=start_point, end=end_point, color=(255, 0, 0))
                draw_point(images[nv], start_point, radius=2, color=(0, 255, 0), thickness=1)
                draw_point(images[nv], end_point, radius=2, color=(0, 0, 255), thickness=1)
        return images
