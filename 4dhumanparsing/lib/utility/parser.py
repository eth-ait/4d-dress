from lib.utility.general import *

from torch.autograd import Variable
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'Graphonomy'))
from lib.Graphonomy.networks.graph import *
from lib.Graphonomy.networks.deeplab_xception_transfer import *
from lib.Graphonomy.dataloaders import custom_transforms


class GraphParser:
    """
    Human Image Parser: Graphonomy
    """
    def __init__(self, model_path='', n_classes=20, init_model=False, scale_list=None, dtype=torch.float32, device=torch.device('cuda:0')):
        # init dtype, device
        self.dtype = dtype
        self.device = device
        self.n_classes = n_classes
        self.scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75] if scale_list is None else scale_list
        # locate checkpoint
        self.checkpoint_fn = os.path.join(model_path, 'inference.pth')

        # init network for parsing only
        if init_model:
            # init network and transform
            self.network = self.get_network(n_classes=self.n_classes)
            self.transform, self.transform_flip = self.get_transform()
            # init adj matrices
            self.adj1_test = Variable(torch.from_numpy(preprocess_adj(cihp_graph)).float()).unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).to(self.device)
            self.adj2_test = torch.from_numpy(cihp2pascal_nlp_adj).float().unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).transpose(2, 3).to(self.device)
            self.adj3_test = Variable(torch.from_numpy(preprocess_adj(pascal_graph)).float()).unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).to(self.device)

        # assign graphonomy label colors
        self.LABEL_VAL_COLORS = torch.tensor([(0, 0, 0), (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51),  # background, hat, hair, glove, sunglasses
                                              (255, 85, 0), (0, 0, 85), (0, 119, 221), (85, 85, 0), (0, 85, 85),  # upper-cloth, dress, coat, socks, pants
                                              (85, 51, 0), (52, 86, 128), (0, 128, 0), (0, 0, 255), (51, 170, 221),  # torso-skin, scarf, skirt, face, left-arm
                                              (0, 255, 255), (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)])  # right-arm, left-leg, right-leg, left-shoe, right-shoe
        
        # assign inner label groups: background -1, skin 0, hair 1, shoe 2, upper 3, lower 4
        self.LABEL_GROUP_INNER = torch.tensor([-1, 1, 1, 0, 1,
                                               3, 3, 3, 2, 4,
                                               0, 3, 4, 0, 0,
                                               0, 0, 0, 2, 2])
        # assign outer label groups: background -1, skin 0, hair 1, shoe 2, upper 3, lower 4, outer 5
        self.LABEL_GROUP_OUTER = torch.tensor([-1, 1, 1, 0, 1,
                                               3, 3, 5, 2, 4,
                                               0, 3, 4, 0, 0,
                                               0, 0, 0, 2, 2])
        
        # assign group labels
        self.GROUP_LABELS = torch.tensor([-1, 0, 1, 2, 3, 4, 5])
        # assign group colors: background -1, skin 0, hair 1, shoe 2, upper 3, lower 4, outer 5
        self.GROUP_COLORS = torch.tensor([[255, 255, 255], [128, 128, 128], [255, 128, 0], [128, 0, 255], 
                                          [180, 50, 50], [50, 180, 50], [0, 128, 255]])

    # get Graphonomy network
    def get_network(self, n_classes=20):
        # load network
        net = deeplab_xception_transfer_projection_savemem(n_classes=n_classes, hidden_layers=128, source_classes=7,)
        # load checkpoint
        net.load_source_model(torch.load(self.checkpoint_fn))
        # send net to device
        net.to(self.device)
        return net

    # get image data transform
    def get_transform(self):
        # init transform list
        transform, transform_flip = [], []
        # get transform for all scales: scale, normalize, flip, to_tensor
        for scale in self.scale_list:
            transform.append(transforms.Compose([custom_transforms.Scale_only_img(scale),
                                                 custom_transforms.Normalize_xception_tf_only_img(),
                                                 custom_transforms.ToTensor_only_img()]))
            transform_flip.append(transforms.Compose([custom_transforms.Scale_only_img(scale),
                                                      custom_transforms.HorizontalFlip_only_img(),
                                                      custom_transforms.Normalize_xception_tf_only_img(),
                                                      custom_transforms.ToTensor_only_img()]))
        return transform, transform_flip

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def flip_cihp(self, tail_list):
        # tail_list = tail_list[0]
        tail_list_rev = [None] * 20
        for xx in range(14):
            tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
        tail_list_rev[14] = tail_list[15].unsqueeze(0)
        tail_list_rev[15] = tail_list[14].unsqueeze(0)
        tail_list_rev[16] = tail_list[17].unsqueeze(0)
        tail_list_rev[17] = tail_list[16].unsqueeze(0)
        tail_list_rev[18] = tail_list[19].unsqueeze(0)
        tail_list_rev[19] = tail_list[18].unsqueeze(0)
        return torch.cat(tail_list_rev, dim=0)


    @torch.no_grad()
    # parser images [nv, h, w, 3] into tensor parser_imgs [nv, h, w, 3] and parser_vals [nv, h, w]
    def parser_images(self, images):
        # get number of images, h, w, c
        n_img, h, w, c = images.shape
        # init parser images and values
        parser_images, parser_values = [], []
        # parser all images
        for nv in range(n_img):
            # convert RGB image to Image
            image = Image.fromarray((images[nv]))
            # transform image for each scale transform
            image_list = [self.transform[ns]({'image': image, 'label': 0}) for ns in range(len(self.scale_list))]
            image_flip_list = [self.transform_flip[ns]({'image': image, 'label': 0}) for ns in range(len(self.scale_list))]
            # evaluate network
            self.network.eval()
            # predict all scaled image data
            for ns, sample_batched in enumerate(zip(image_list, image_flip_list)):
                # unpack sample
                inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
                inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
                inputs = torch.cat((inputs.unsqueeze(0), inputs_f.unsqueeze(0)), dim=0)
                # Forward pass of the mini-batch
                inputs = Variable(inputs, requires_grad=False).to(self.device)
                with torch.no_grad():
                    outputs = self.network.forward(inputs, self.adj1_test, self.adj3_test, self.adj2_test)
                    outputs = (outputs[0] + self.flip(self.flip_cihp(outputs[1]), dim=-1)) / 2
                    outputs = outputs.unsqueeze(0)
                    # append scaled predictions
                    if ns > 0:
                        outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                        outputs_final = outputs_final + outputs
                    else:
                        outputs_final = outputs.clone()
            # get final prediction labels
            parser_tensor = torch.max(outputs_final, 1)[1][0, ...]
            parser_values.append(parser_tensor)
            # get final parser images
            parser_images.append(self.LABEL_VAL_COLORS[parser_tensor].cpu().numpy().astype(np.uint8))
            # show_image(parser_images[-1])
        return np.stack(parser_images, axis=0), torch.stack(parser_values, dim=0)

    @torch.no_grad()
    # decode parser images[nv, h, w, 3] into parser values[nv, h, w]
    def decode_parser_images(self, images, surface=None, surface_labels=None, group=False):
        # init GROUP_COLORS
        GROUP_COLORS = self.GROUP_COLORS
        # init SOURCE_COLORS as graphonomy colors
        SOURCE_COLORS = self.LABEL_VAL_COLORS
        if surface == 'Inner':
            LABEL_GROUPS = self.LABEL_GROUP_INNER
        elif surface == 'Outer':
            LABEL_GROUPS = self.LABEL_GROUP_OUTER
        # decode group images:
        if group:
            SOURCE_COLORS = self.GROUP_COLORS
            LABEL_GROUPS = self.GROUP_LABELS
        
        # update new_label like sock
        if 'sock' in surface_labels:
            if group: SOURCE_COLORS = torch.cat((SOURCE_COLORS, torch.tensor([[255, 255, 0]])), dim=0)
            LABEL_GROUPS[8] = surface_labels.index('sock')

        # init parser values[nv, h, w]
        parser_vals = torch.zeros(images.shape[:-1], dtype=torch.long, device=images.device)
        # init parser groups[nv, h, w]
        parser_grps = torch.zeros(images.shape[:-1], dtype=torch.long, device=images.device)
        # loop over all images
        for nv in range(images.shape[0]):
            # decode parser images into parser labels
            for nc in range(SOURCE_COLORS.shape[0]):
                # get label color mask
                label_masks = torch.sum((images[nv] - SOURCE_COLORS[nc].to(images.device)) ** 2, dim=-1) == 0
                parser_vals[nv][label_masks] = torch.tensor(nc, dtype=torch.long)
                parser_grps[nv][label_masks] = LABEL_GROUPS[nc]

            # # Check: project parser labels to images
            # parser_val_image = SOURCE_COLORS[parser_vals[nv]]
            # parser_grp_image = GROUP_COLORS[parser_grps[nv]]
            # show_image(parser_val_image.cpu().numpy().astype(np.uint8))
            # show_image(parser_grp_image.cpu().numpy().astype(np.uint8))
        return parser_vals, parser_grps

