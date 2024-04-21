from lib.utility.general import *

from pycocotools import mask as maskUtils
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class Sam:
    """
    Segment Human Images using Segment Anything.
    """

    def __init__(self, model_path='', init_model=False, dtype=torch.float32, device=torch.device('cuda:0')):
        # init dtype, device, n_views, image_size
        self.dtype = dtype
        self.device = device
        # locate sam checkpoint
        self.checkpoint_dir = os.path.join(model_path, 'sam_vit_h_4b8939.pth')

        # init model for segmenting only
        if init_model:
            # load SAM model
            self.get_networks()

    # get sam model
    def get_networks(self):
        # load sam model from checkpoint
        self.sam = sam_model_registry['vit_h'](checkpoint=self.checkpoint_dir).to(self.device)
        # load sam auto mask generator, with coco_rle output
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=32, points_per_batch=64,
                                                        min_mask_region_area=100, output_mode='coco_rle')

    @torch.no_grad()
    # segment numpy images (nv, h, w, 3) into numpy masks [nv, nm]
    def segment_images(self, images):
        # init segment_masks
        segment_masks = []
        # process all images
        for nv in range(images.shape[0]):
            # generate masks for image
            masks = sorted(self.mask_generator.generate(images[nv]), key=(lambda x: x['area']), reverse=True)
            # init valid flag
            for mask in masks: mask['valid'] = True
            # append segment_masks
            segment_masks.append(masks)
        # segment_masks:[[{'segmentation', ...}, ...], ....]
        return segment_masks

    # save segment masks into .json
    def save_segment_masks(self, mask_fn, segment_masks):
        with open(mask_fn, "w") as f:
            json.dump(segment_masks, f)

    # load segment masks from .json
    def load_segment_masks(self, mask_fn):
        segment_masks = self.decode_segment_masks(json.load(open(mask_fn, 'rb')))
        return segment_masks

    # decode coco masks to binary masks
    def decode_segment_masks(self, masks):
        # process over all masks
        for mask in masks:
            # decode segmentation{'size', 'count'} to numpy array in 0, 1
            for msk in mask:
                if isinstance(msk['segmentation'], dict):
                    msk['segmentation'] = maskUtils.decode(msk['segmentation']) == 1.
        return masks

    # clean background, body, small masks
    def clean_segment_masks(self, render_masks, segment_masks, min_area=100):
        # loop over all images
        for nv in range(render_masks.shape[0]):
            # get masks and back_mask
            masks = segment_masks[nv]
            back_mask = render_masks[nv] != 255
            # get body area
            body_area = np.sum(render_masks[nv] == 255)
            # loop over all masks
            for mask in masks:
                # check background
                diff_ratio = np.sum(np.abs(mask['segmentation'] * 1. - back_mask * 1.)) / np.sum(back_mask * 1.)
                if diff_ratio < 0.05: mask['valid'] = False
                # check body and small masks
                if mask['area'] > body_area * 0.9 or mask['area'] < min_area: mask['valid'] = False
            # clean duplicated mask
            self.clean_duplicated_masks(masks)
        return segment_masks

    # check duplicated masks
    def clean_duplicated_masks(self, segment_masks):
        # get areas of all masks
        areas = np.asarray([segment_masks[nm]['area'] for nm in range(len(segment_masks))])
        valids = [segment_masks[nm]['valid'] for nm in range(len(segment_masks))]
        # loop over all masks
        for nm in range(len(segment_masks)):
            # skip non valid mask
            if not valids[nm]: continue
            # check indices of similar areas
            similar_indices = list(np.where(np.abs(areas - segment_masks[nm]['area']) / segment_masks[nm]['area'] < 0.05)[0])
            # remove non valid masks
            for id in similar_indices:
                if not valids[id]: similar_indices.remove(id)
            # skip empty similar
            if len(similar_indices) == 1: continue

            # init overlap indices
            overlap_indices = []
            # check similar overlaps
            for id in similar_indices:
                # check diff segmentation
                diff_ratio = np.sum(np.abs(segment_masks[id]['segmentation'] * 1. - segment_masks[nm]['segmentation'] * 1.)) / np.sum(segment_masks[nm]['segmentation'] * 1.)
                # append overlap indices
                if diff_ratio < 0.05: overlap_indices.append(id)
            if len(overlap_indices) == 1: continue

            # keep best mask within overlap indices
            best_id = overlap_indices[np.argmax([segment_masks[id]['predicted_iou'] for id in overlap_indices])]
            for id in overlap_indices:
                if id != best_id: segment_masks[id]['valid'] = False
        return segment_masks

    # vote segment_masks[[{'segmentation', ...}, ...], ...]
    # using vote_labels{'parser_labels': {'value': (nv, h, w, nl), 'weight': 1.}, ...}
    def vote_segment_masks(self, render_masks, segment_masks, vote_labels, surface_labels, sam_skin=False):
        # init multi-view dimensions
        n_views, h, w = render_masks.shape[:3]
        # init segment_mask_votes
        segment_mask_votes = []
        # process all views
        for nv in range(n_views):
            # init mask_votes (h, w, nl)
            mask_votes = torch.zeros((h, w, len(surface_labels)), dtype=self.dtype).to(self.device)
            # loop over all segment_masks
            for nm in range(len(segment_masks[nv])):
                # skip non valid mask
                if not segment_masks[nv][nm]['valid']: continue
                # get mask_region, mask_points remove GT background
                mask_region = torch.tensor(segment_masks[nv][nm]['segmentation'], device=self.device)
                mask_region[render_masks[nv] == False] = False
                mask_points = torch.stack(torch.where(mask_region), dim=0).T
                # init vote_weights and vote filters
                vote_ratios = torch.zeros(len(surface_labels)).to(self.device)
                vote_weights = torch.zeros(len(surface_labels)).to(self.device)
                # assign vote_weights from vote_labels {'parser_votes', 'optical_votes', 'manual_votes'}
                for key, vote_label in vote_labels.items():
                    # get vote_label_ratio within mask_region
                    vote_label['ratio'] = torch.stack([torch.sum(vote_label['value'][nv, :, :, nl][mask_region])
                                                       for nl in range(len(surface_labels))]) / torch.sum(mask_region)
                    # append vote_ratio to vote_weights
                    vote_ratios += vote_label['ratio']
                    vote_weights += vote_label['weight'] * vote_label['ratio']

                    # Trick: more weight on small torso skin, from parser and optical
                    if sam_skin and 0.3 < vote_label['ratio'][surface_labels.index('skin')] and segment_masks[nv][nm]['area'] < 400:
                        vote_skin_ratio = vote_label['ratio'].clone()
                        vote_skin_ratio[surface_labels.index('skin') + 1:] = 0
                        vote_weights += vote_label['weight'] * vote_skin_ratio * 4

                    # append vote_ratio to mask
                    segment_masks[nv][nm][key] = vote_label['ratio'].cpu().numpy()
                # append surface_labels to masks
                segment_masks[nv][nm]['surface_labels'] = surface_labels
                
                # Trick: filter mask with noisy labels: skin, hair, shoe, upper, lower, outer
                if torch.sum(mask_region) > torch.sum(render_masks[nv]) * 0.5:
                    if torch.count_nonzero(vote_ratios > 0.01) > 4 and torch.max(vote_ratios) < 0.8 * len(vote_labels):
                        segment_masks[nv][nm]['valid'] = False
                # skip non-valid mask
                if not segment_masks[nv][nm]['valid']: continue
                # append votes for mask_points
                for nl in range(len(surface_labels)):
                    mask_votes[mask_points[:, 0], mask_points[:, 1], nl] += vote_weights[nl]
            # append segment_mask_votes
            segment_mask_votes.append(mask_votes)
        return torch.stack(segment_mask_votes, dim=0)

    # view masked images within render image
    def view_masked_images(self, image, masks):
        # group masks to three lines
        h, w, c = image.shape
        hg, wg = 3, ((len(masks)) // 3 + 1)
        group_masks = np.ones((h * hg, w * wg, 3)) * 255.
        # process all masks
        for nm in range(len(masks)):
            # copy image
            temp_image = image.copy()
            # draw mask on image
            temp_image[masks[nm]['segmentation'] == 0] = 255
            # draw indicators
            color = (0, 0, 255) if masks[nm]['valid'] else (255, 0, 0)
            draw_rectangle(temp_image, (1, 1), (h - 1, w - 1), color=(0, 0, 0))
            draw_text(temp_image, 'm:{}'.format(nm), (int(h * 0.05), int(w * 0.75)), color=color)
            draw_text(temp_image, 'a:{}'.format(masks[nm]['area']), (int(h * 0.10), int(w * 0.75)), color=color)
            draw_text(temp_image, 'q:{:.3f}'.format(masks[nm]['predicted_iou']), (int(h * 0.15), int(w * 0.75)), color=color)
            # draw render_ratios
            if 'render_votes' in masks[nm]:
                nd = 1
                draw_text(temp_image, 'render', (int(h * 0.05), int(w * 0.05)), color=color)
                # loop over all surface_ratios
                for nl in range(masks[nm]['render_votes'].shape[0]):
                    if masks[nm]['render_votes'][nl] < 0.01: continue
                    draw_text(temp_image, '{}:{:.2f}'.format(masks[nm]['surface_labels'][nl], masks[nm]['render_votes'][nl]),
                              (int(h * (0.05 + 0.05 * nd)), int(w * 0.05)), color=color)
                    nd += 1
            # draw parser_ratios
            if 'parser_votes' in masks[nm]:
                nd = 1
                draw_text(temp_image, 'parser', (int(h * 0.95), int(w * 0.05)), color=color)
                # loop over all surface_ratios
                for nl in range(masks[nm]['parser_votes'].shape[0]):
                    if masks[nm]['parser_votes'][nl] < 0.01: continue
                    draw_text(temp_image, '{}:{:.2f}'.format(masks[nm]['surface_labels'][nl], masks[nm]['parser_votes'][nl]),
                              (int(h * (0.95 - 0.05 * nd)), int(w * 0.05)), color=color)
                    nd += 1
            # draw raft_ratios
            if 'optical_votes' in masks[nm]:
                nd = 1
                draw_text(temp_image, 'optical', (int(h * 0.95), int(w * 0.70)), color=color)
                # loop over all surface_ratios
                for nl in range(masks[nm]['optical_votes'].shape[0]):
                    if masks[nm]['optical_votes'][nl] < 0.01: continue
                    draw_text(temp_image, '{}:{:.2f}'.format(masks[nm]['surface_labels'][nl], masks[nm]['optical_votes'][nl]),
                              (int(h * (0.95 - 0.05 * nd)), int(w * 0.70)), color=color)
                    nd += 1
            # append masked image to group
            group_masks[h * (nm // wg):h * (nm // wg + 1), w * (nm % wg):w * (nm % wg + 1)] = temp_image
            # print(nm, masks[nm]['area'], masks[nm]['predicted_iou'])
        return group_masks.astype(np.uint8)

