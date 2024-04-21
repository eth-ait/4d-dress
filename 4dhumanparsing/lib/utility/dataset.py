from lib.utility.general import *
from lib.utility.render import PytorchRenderer
from lib.utility.parser import GraphParser
from lib.utility.optical import Rafter
from lib.utility.sam import Sam

from lib.model.surface_parser import SurfaceParser
from lib.utility.manual import update_manual_labels

class DatasetUtils:
    """
    4D-DRESS Dataset Utility: data loader, render, parser, saver, viewer, surface parser
    """

    def __init__(self, dataset_dir='', dataset_args='', checkpoint_dir='', preprocess_init=False, dtype=torch.float32, device=torch.device('cuda:0')):
        # init dtype and device
        self.dtype = dtype
        self.device = device
        # init dataset_dir and checkpoint_dir
        self.dataset_dir = dataset_dir
        self.dataset_args = dataset_args
        self.checkpoint_dir = checkpoint_dir
        # init preprocess render, parser, optical, and sam
        self.preprocess_init = preprocess_init

        # init hyper_params
        self.hyper_params = dict()

        # init Pytorch3d Renderer with default n_views=24, image_size=512
        self.render = PytorchRenderer(num_view=24, img_size=512, init_model=True, dtype=self.dtype, device=self.device)
        # init Graphonomy Parser
        self.parser = GraphParser(model_path=os.path.join(self.checkpoint_dir, 'graphonomy'), init_model=self.preprocess_init, dtype=self.dtype, device=self.device)
        # init Optical Flow Rafter
        self.optical = Rafter(model_path=os.path.join(self.checkpoint_dir, 'raft'), model_args=self.dataset_args, init_model=self.preprocess_init, dtype=self.dtype, device=self.device)
        # init Segment Anything SAM
        self.sam = Sam(model_path=os.path.join(self.checkpoint_dir, 'sam'), init_model=self.preprocess_init, dtype=self.dtype, device=self.device)

    # preprocess scan_mesh: scale, centralize, rotation, offset
    def preprocess_scan_mesh(self, mesh, mcentral=False, bbox=True, rotation=None, offset=None, scale=1.0):
        # get scan vertices mass center
        mcenter = np.mean(mesh['vertices'], axis=0)
        # get scan vertices bbox center
        bmax = np.max(mesh['vertices'], axis=0)
        bmin = np.min(mesh['vertices'], axis=0)
        bcenter = (bmax + bmin) / 2
        # centralize scan data around mass center
        if mcentral:
            mesh['vertices'] -= mcenter
        # centralize scan data around bbox center
        elif bbox:
            mesh['vertices'] -= bcenter
        # scale scan vertices
        mesh['vertices'] /= scale
        # rotate scan vertices
        if rotation is not None:
            mesh['vertices'] = np.matmul(rotation, mesh['vertices'].T).T
        # offset scan vertices
        if offset is not None:
            mesh['vertices'] += offset
        # return scan data, centers, scale
        return mesh, {'mcenter': mcenter, 'bcenter': bcenter}, scale

    # load scan mesh with texture from pkl
    def load_scan_mesh(self, mesh_fn):
        # locate atlas_fn
        atlas_fn = mesh_fn.replace('mesh-', 'atlas-')
        # load scan mesh and atlas data
        mesh_data, atlas_data = load_pickle(mesh_fn), load_pickle(atlas_fn)
        # load scan uv_coordinate and uv_image as TextureVisuals
        uv_image = Image.fromarray(atlas_data).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB")
        texture_visual = trimesh.visual.texture.TextureVisuals(uv=mesh_data['uvs'], image=uv_image)
        # pack scan data as trimesh
        scan_trimesh = trimesh.Trimesh(
            vertices=mesh_data['vertices'],
            faces=mesh_data['faces'],
            vertex_normals=mesh_data['normals'],
            visual=texture_visual,
            process=False,
        )
        # pack scan data as mesh
        scan_mesh = {
            'vertices_origin': scan_trimesh.vertices.copy(),
            'vertices': scan_trimesh.vertices.copy(),
            'faces': scan_trimesh.faces,
            'edges': scan_trimesh.edges,
            'colors': scan_trimesh.visual.to_color().vertex_colors,
            'normals': scan_trimesh.vertex_normals,
            'uvs': mesh_data['uvs'],
            'uv_image': np.array(uv_image),
            'uv_path': atlas_fn,
        }
        # preprocess scan mesh: scale, centralize, normalize, rotation, offset
        scan_mesh, center, scale = self.preprocess_scan_mesh(scan_mesh, mcentral=False, bbox=True, scale=1.0)
        return scan_trimesh, scan_mesh, center, scale

    # check subj outfit sequence render range
    def check_subj_outfit_seq_render_range(self, hyper_params):
        # load hyper_params from subj_outfit_seq, return render_ranges
        if os.path.isfile(os.path.join(hyper_params['subj_outfit_seq_dir'], 'hyper_params.pkl')):
            hyper_params = load_pickle(os.path.join(hyper_params['subj_outfit_seq_dir'], 'hyper_params.pkl'))
            return hyper_params['render_ranges']

        # init max/min width/height
        max_w, min_w, max_h, min_h = [], [], [], []
        # locate sequence scan meshes
        scan_files = hyper_params['scan_files']
        # process all frames
        loop = tqdm(range(len(scan_files)))
        for nf in loop:
            # check render range for frames
            loop.set_description('## Check Render Range Frame: {}/{}'.format(nf, len(scan_files)))
            # load scan data as mesh {'vertices', 'faces', 'edges', 'colors', 'normals', 'uvs', 'uv_image', 'uv_path'}
            scan_trimesh, scan_mesh, center, scale = self.load_scan_mesh(scan_files[nf])
            # append max/min width/height
            max_w.append(np.max(scan_mesh['vertices'][:, 0]))
            min_w.append(np.min(scan_mesh['vertices'][:, 0]))
            max_h.append(np.max(scan_mesh['vertices'][:, 1]))
            min_h.append(np.min(scan_mesh['vertices'][:, 1]))
        # update max/min width/height
        max_w, min_w, max_h, min_h = np.max(max_w), np.min(min_w), np.max(max_h), np.min(min_h)
        # update render_range + 0.1
        render_range = ((max(max_w, -min_w, max_h, -min_h) + 0.1) // 0.01) / 100
        print('render_ranges', [[render_range, -render_range], [render_range, -render_range]])
        return [[render_range, -render_range], [render_range, -render_range]]

    # load hyper_params for subj_outfit_seq
    def load_hyper_params(self, subj='', outfit='', seq=''):
        # init hyper_params from basic_info.pkl
        hyper_params = load_pickle(os.path.join(self.dataset_dir, subj, outfit, seq, 'basic_info.pkl'))
        # append dataset folder dirs
        hyper_params['subj_outfit_seq_dir'] = os.path.join(self.dataset_dir, subj, outfit, seq)
        hyper_params['scan_dir'] = os.path.join(hyper_params['subj_outfit_seq_dir'], 'Meshes_pkl')
        hyper_params['smplx_dir'] = os.path.join(hyper_params['subj_outfit_seq_dir'], 'SMPLX')
        hyper_params['smpl_dir'] = os.path.join(hyper_params['subj_outfit_seq_dir'], 'SMPL')
        hyper_params['capture_dir'] = os.path.join(hyper_params['subj_outfit_seq_dir'], 'Capture')
        hyper_params['semantic_dir'] = os.path.join(hyper_params['subj_outfit_seq_dir'], 'Semantic')
        # locate sequence scan files
        hyper_params['scan_files'] = sorted(glob.glob(os.path.join(hyper_params['scan_dir'], 'mesh-f*.pkl')))
        hyper_params['scan_frames'] = [fn.split('/')[-1].split('.')[0][-5:] for fn in hyper_params['scan_files']]
        # append preprocess folder dirs
        hyper_params['render_dir'] = os.path.join(hyper_params['semantic_dir'], 'process', 'render')
        hyper_params['parser_dir'] = os.path.join(hyper_params['semantic_dir'], 'process', 'parser')
        hyper_params['optical_dir'] = os.path.join(hyper_params['semantic_dir'], 'process', 'optical')
        hyper_params['segment_dir'] = os.path.join(hyper_params['semantic_dir'], 'process', 'segment')
        hyper_params['manual_dir'] = os.path.join(hyper_params['semantic_dir'], 'process', 'manual')

        # update label hyper_params
        if outfit == 'Inner':
            hyper_params['surface'] = 'Inner'
            hyper_params['surface_labels'] = SURFACE_LABEL[:-1]
            hyper_params['surface_labels_colors'] = SURFACE_LABEL_COLOR[:-1]
        elif outfit == 'Outer':
            hyper_params['surface'] = 'Outer'
            hyper_params['surface_labels'] = SURFACE_LABEL
            hyper_params['surface_labels_colors'] = SURFACE_LABEL_COLOR
        else:
            raise Exception('Unknown outfit type: {} over {Inner, Outer}'.format(outfit))
        # update new_label
        hyper_params['new_label'] = None
        
        # update render hyper_params
        hyper_params['render_views'] = self.render.num_view
        hyper_params['render_sizes'] = self.render.img_size
        hyper_params['render_ranges'] = self.check_subj_outfit_seq_render_range(hyper_params)
        # update Pytorch3d Renderer with render_views, render_sizes, and render_ranges
        self.render = PytorchRenderer(num_view=hyper_params['render_views'], img_size=hyper_params['render_sizes'],
                                      max_x=hyper_params['render_ranges'][0][0], min_x=hyper_params['render_ranges'][0][1],
                                      max_y=hyper_params['render_ranges'][1][0], min_y=hyper_params['render_ranges'][1][1],
                                      init_model=True, dtype=self.dtype, device=self.device)

        # update parse hyper_params
        hyper_params['scan_unary'] = {'mask': 1.5, 'manual': 1., 'parser': 0., 'optical': 0.}
        hyper_params['vote_weights'] = {'parser': 1., 'optical': 1.5, 'manual': 10}  # Trick: increase optical_ratio to increase tracking robustness
        hyper_params['sam_skin'] = True  # Trick: better detect small skin regions

        # update manual_labels
        hyper_params['manual_efforts'] = self.update_manual_efforts(subj=subj, outfit=outfit, seq=seq, surface_labels=hyper_params['surface_labels'])
        # update and save hyper_params
        self.hyper_params = hyper_params
        save_pickle(os.path.join(hyper_params['subj_outfit_seq_dir'], 'hyper_params.pkl'), hyper_params)
        return hyper_params


    # render, parser, optical, segment subj_outfit_seq meshes
    def render_parser_optical_segment_subj_outfit_seq_meshes(self, subj='', outfit='', seq='', save=False, update=False, n_start=0, n_stop=-1):
        # load hyper_params
        hyper_params = self.load_hyper_params(subj=subj, outfit=outfit, seq=seq)
        # make folders for render, parser, optical, and segment masks
        os.makedirs(hyper_params['render_dir'], exist_ok=True)
        os.makedirs(hyper_params['parser_dir'], exist_ok=True)
        os.makedirs(hyper_params['optical_dir'], exist_ok=True)
        os.makedirs(hyper_params['segment_dir'], exist_ok=True)
        os.makedirs(hyper_params['manual_dir'], exist_ok=True)
        # locate scan_frames
        scan_nums, scan_frames = len(hyper_params['scan_frames']), hyper_params['scan_frames']

        # # -------------------- Render, Segment, Mask All Frames -------------------- # #
        # process all scan_frames
        loop = tqdm(range(n_start, scan_nums))
        for n_frame in loop:
            # check stop frame
            if 0 <= n_stop < n_frame: break
            # locate current frame
            frame = scan_frames[n_frame]
            previous_frame = scan_frames[max(0, n_frame - 1)]
            loop.set_description('## Render, Parse, Optical, Segment Subj_Outfit_Seq_Frame: {}_{}_{}_{}/{}'.format(subj, outfit, seq,frame, scan_frames[-1]))

            # # -------------------- Render, Parser, Optical Multi-View Images -------------------- # #
            # locate saved render and parser images
            save_render_image_fn = os.path.join(hyper_params['render_dir'], 'render-f{}.png'.format(frame))
            save_render_mask_fn = os.path.join(hyper_params['render_dir'], 'mask-f{}.png'.format(frame))
            save_parser_image_fn = os.path.join(hyper_params['parser_dir'], 'parser-f{}.png'.format(frame))
            save_optical_flow_fn = os.path.join(hyper_params['optical_dir'], 'optical-f{}.npz'.format(frame))
            # not update mode: skip existing render and parser frames
            if not update and os.path.exists(save_render_mask_fn) and os.path.exists(save_parser_image_fn) and os.path.exists(save_optical_flow_fn):
                pass
            else:
                # load scan_mesh {'vertices_origin', 'vertices', 'faces', 'edges', 'colors', 'normals', 'uvs', 'uv_image', 'uv_path'}
                scan_mesh_fn = os.path.join(hyper_params['scan_dir'], 'mesh-f{}.pkl'.format(frame))
                scan_trimesh, scan_mesh, center, scale = self.load_scan_mesh(scan_mesh_fn)
                # unpack scan_mesh to tensor
                th_verts = torch.tensor(scan_mesh['vertices'], dtype=self.dtype, device=self.device).unsqueeze(0)
                th_faces = torch.tensor(scan_mesh['faces'], dtype=torch.long, device=self.device).unsqueeze(0)
                th_colors = torch.tensor(scan_mesh['colors'][:, :3] / 255., dtype=self.dtype, device=self.device).unsqueeze(0)

                # render scan_mesh to render_images[nv, h, w, 4], and render_masks
                if 'uvs' in scan_mesh and 'uv_image' in scan_mesh:
                    th_uvs = torch.tensor(scan_mesh['uvs'], dtype=self.dtype, device=self.device).unsqueeze(0)
                    th_uv_image = torch.tensor(cv.resize(scan_mesh['uv_image'], (1024, 1024)) / 255., dtype=self.dtype, device=self.device).unsqueeze(0)
                    render_results = self.render.render_mesh_images(th_verts, th_faces, th_colors, th_uvs, th_uv_image)
                else:
                    render_results = self.render.render_mesh_images(th_verts, th_faces, th_colors, mode=['color'])
                # unpack rendered images and masks
                render_images = (render_results['color'].cpu().numpy() * 255.).astype(np.uint8)
                render_masks = ((render_results['mask'].cpu().numpy() > 0.) * 255.).astype(np.uint8)
                save_render_images = self.render.pack_multi_view_images(render_images)
                save_render_masks = self.render.pack_multi_view_images(render_masks)
                
                # parser render_images[nv, h, w, 3] to parser_images[nv, h, w, 3] and parser_values[nv, h, w]
                parser_images, parser_values = self.parser.parser_images(render_images)
                save_parser_images = self.render.pack_multi_view_images(parser_images)

                # load previous render_images
                previous_render_image_fn = os.path.join(hyper_params['render_dir'], 'render-f{}.png'.format(previous_frame))
                previous_render_images = render_images if not os.path.exists(previous_render_image_fn) else self.render.load_multi_view_images(previous_render_image_fn)
                # raft optical_flows[nv, h, w, 2] from previous to current images
                optical_flows = self.optical.process_images_pairs(previous_render_images, render_images).cpu().numpy()
                optical_images = self.optical.draw_sampled_flows(previous_render_images, optical_flows)
                save_optical_images = self.render.pack_multi_view_images(optical_images)

                # save render_images and parser_images
                if save:
                    save_image(save_render_image_fn, save_render_images)
                    save_image(save_render_mask_fn, save_render_masks)
                    save_image(save_parser_image_fn, save_parser_images)
                    self.optical.save_optical_flows(save_optical_flow_fn, optical_flows)
                    save_image(save_optical_flow_fn.replace('.npz', '.png'), save_optical_images)
                else:
                    show_image(save_render_images, name='render_images-{}'.format(frame))
                    show_image(save_render_masks, name='render_masks-{}'.format(frame))
                    show_image(save_parser_images, name='parser_images-{}'.format(frame))
                    show_image(save_optical_images, name='optical_images-{}'.format(frame))


            # # -------------------- SAM Segment Multi-View Images -------------------- # #
            # locate saved segment masks
            save_segment_mask_fn = os.path.join(hyper_params['segment_dir'], 'mask-f{}.json'.format(frame))
            # not update mode: skip existing segment mask frames
            if not update and os.path.exists(save_segment_mask_fn):
                pass
            else:
                # load current multi-view render_images
                render_images = self.render.load_multi_view_images(save_render_image_fn)
                # segment multi-view render_images to segment_masks
                segment_masks = self.sam.segment_images(render_images)
                # save segment_masks
                if save:
                    self.sam.save_segment_masks(save_segment_mask_fn, segment_masks)
                # decode and show segment_masks
                else:
                    # decode segment masks
                    segment_masks = self.sam.decode_segment_masks(segment_masks)
                    # view masked images from segment_masks and render_images
                    for nv in range(render_images.shape[0]):
                        show_image(self.sam.view_masked_images(render_images[nv], segment_masks[nv]))


    # parse scan_frames into multi-surfaces
    def multi_surface_parsing(self, subj='', outfit='', seq='', save_folder='', n_start=0, n_stop=-1, save=False, update=False, save_mask=False, new_label=None):
        
        # # -------------------- load init params  -------------------- # #
        # load hyper_params
        hyper_params = self.load_hyper_params(subj=subj, outfit=outfit, seq=seq)
        # update new_label
        if new_label == 'sock': 
            hyper_params['new_label'] = new_label
            hyper_params['surface_labels'].append(new_label)
            hyper_params['surface_labels_colors'] = np.concatenate([hyper_params['surface_labels_colors'], np.array([[255, 255, 0]])], axis=0)
            hyper_params['manual_efforts'] = self.update_manual_efforts(subj=subj, outfit=outfit, seq=seq, surface_labels=hyper_params['surface_labels'])

        # init Surface Parser Agent with render, parser, optical, and sam agents
        surface_parser = SurfaceParser(self.render, self.parser, self.optical, self.sam, hyper_params, dtype=self.dtype, device=self.device)
        # init optimization parameters
        opt_params = {'scan_params': {}}

        # locate output folder
        output_dir = os.path.join(hyper_params['semantic_dir'], 'process', save_folder)
        if save: os.makedirs(output_dir, exist_ok=True)
        # locate scan frames
        scan_frames = hyper_params['scan_frames']
        first_frame, last_frame = scan_frames[0], scan_frames[-1]

        # # -------------------- locate start frame  -------------------- # #
        # locate start frame name {:05d}, 0 means start from the first frame
        n_start = scan_frames[n_start] if n_start == 0 else '{:05d}'.format(n_start)
        # not update: start from existing data
        if not update:
            # find existing labeled frames
            label_fns = sorted(glob.glob(os.path.join(output_dir, 'label-f*.pkl')))
            # assign the last segment frame as start_frame
            if len(label_fns) > 0:
                # get the last label frame name
                n_start = label_fns[-1].split('/')[-1].split('.')[0][-5:]
                if n_start == last_frame:
                    return print('Not Update Mode: first_frame={}, last_frame={}, scan_nums={} has been processed !'.format(first_frame, last_frame, len(scan_frames)))

        # # -------------------- load start frame  -------------------- # #
        # load start frame data if n_start != scan_frames[0]
        if n_start != scan_frames[0]:
            # locate previous frame name
            n_previous = scan_frames[scan_frames.index(n_start) - 1]
            # load previous frame labels
            opt_params['scan_params']['labels'] = torch.tensor(load_pickle(os.path.join(output_dir, 'label-f{}.pkl'.format(n_previous)))['scan_labels']).to(self.device)
            # load previous render_labels_images
            opt_params['scan_params']['render_labels_images'] = torch.tensor(self.render.load_multi_view_images(
                os.path.join(output_dir, 'label-f{}.png'.format(n_previous)))).to(self.device)
            # decode render_labels_images to render_labels
            _, opt_params['scan_params']['render_labels'] = self.parser.decode_parser_images(
                opt_params['scan_params']['render_labels_images'], hyper_params['surface'], hyper_params['surface_labels'], group=True)
            print('Start from frame:', n_start, 'load previous frame:', n_previous)
        # convert start frame name to frame number
        n_start = scan_frames.index(n_start)

        # # -------------------- process frames  -------------------- # #
        # loop over all sampled scan frames
        loop = tqdm(range(n_start, len(scan_frames)))
        for n_frame in loop:
            # locate current frame
            is_first_frame = n_frame == 0
            frame = scan_frames[n_frame]
            # check stop frame
            if 0 <= n_stop < n_frame - n_start:
                return print('Stop process at {}'.format(frame))
            loop.set_description('## Multi-Surface Parsing Subj_Outfit_Seq_Frame: {}_{}_{}_{}/{}'.format(subj, outfit, seq, frame, last_frame))

            # # -------------------- load meshes, images, and segment masks  -------------------- # #
            # locate scan_mesh_fn
            scan_trimesh, scan_mesh, center, scale = self.load_scan_mesh(os.path.join(hyper_params['scan_dir'], 'mesh-f{}.pkl'.format(frame)))
            init_meshes = {'scan_mesh': scan_mesh}
            # load render_images, render_masks and parser_images (nv, h, w, 3)
            render_images = self.render.load_multi_view_images(os.path.join(hyper_params['render_dir'], 'render-f{}.png'.format(frame)))
            render_masks = self.render.load_multi_view_images(os.path.join(hyper_params['render_dir'], 'mask-f{}.png'.format(frame)))
            parser_images = self.render.load_multi_view_images(os.path.join(hyper_params['parser_dir'], 'parser-f{}.png'.format(frame)))
            init_images = {'render_images': render_images, 'render_masks': render_masks, 'parser_images': parser_images}
            # load optical_flows from previous to current frame
            optical_flows = self.optical.load_optical_flows(os.path.join(hyper_params['optical_dir'], 'optical-f{}.npz'.format(frame)))
            init_images['optical_flows'] = optical_flows
            # load and clean segment masks
            segment_masks = self.sam.load_segment_masks(os.path.join(hyper_params['segment_dir'], 'mask-f{}.json'.format(frame)))
            segment_masks = self.sam.clean_segment_masks(render_masks, segment_masks, min_area=100)
            init_images['segment_masks'] = segment_masks

            # # -------------------- parse scan surfaces  -------------------- # #
            # parse scan_mesh surfaces using surface_parser
            opt_params = surface_parser.forward(opt_params, init_meshes, init_images, frame, first_frame=is_first_frame, block_manual=False)

            # # -------------------- save frame labels  -------------------- # #
            # save scan_labels and render_labels
            if save:
                # save scan_labels to label-f{}.pkl
                save_pickle(os.path.join(output_dir, 'label-f{}.pkl'.format(frame)), {'scan_labels': opt_params['scan_params']['labels'].cpu().numpy()})
                # save render_labels_images to label-f{}.png
                save_image(os.path.join(output_dir, 'label-f{}.png'.format(frame)), opt_params['scan_params']['render_labels_images'])
                # save manual_labels_images to manual-f{}.png
                if opt_params['scan_params']['manual_labels_images'] is not None:
                    save_image(os.path.join(output_dir, 'manual-f{}.png'.format(frame)), opt_params['scan_params']['manual_labels_images'])

            # view and save segment_masks
            if save_mask:
                # view masked images from segment_masks and render_images
                for nv in range(init_images['render_images'].shape[0]):
                    masked_images = self.sam.view_masked_images(init_images['render_images'][nv], init_images['segment_masks'][nv])
                    save_image(os.path.join(hyper_params['manual_dir'], 'mask-f{}.{:02d}.png'.format(frame, nv)), masked_images)
        

    # save segment masks for manual rectification
    def save_segment_masks(self, subj='', outfit='', seq='', n_start=0, n_stop=-1):
        # # -------------------- load init params  -------------------- # #
        # load hyper_params
        hyper_params = self.load_hyper_params(subj=subj, outfit=outfit, seq=seq)
        scan_frames = hyper_params['scan_frames']
        first_frame, last_frame = scan_frames[0], scan_frames[-1]

        # # -------------------- locate start frame  -------------------- # #
        # locate start frame name {:05d}, 0 means start from the first frame
        n_start = scan_frames[n_start] if n_start == 0 else '{:05d}'.format(n_start)
        # convert start frame name to frame number
        n_start = scan_frames.index(n_start)

        # # -------------------- process frames  -------------------- # #
        # loop over all scan frames
        loop = tqdm(range(n_start, len(scan_frames)))
        for n_frame in loop:
            # locate current frame
            frame = scan_frames[n_frame]
            # check stop frame
            if 0 <= n_stop < n_frame - n_start:
                return print('Stop process at {}'.format(frame))
            loop.set_description('## Save Segment Masks Subj_Outfit_Seq_Frame: {}_{}_{}_{}/{}'.format(subj, outfit, seq, frame, last_frame))
            
            # load render_images and render_masks
            render_images = self.render.load_multi_view_images(os.path.join(hyper_params['render_dir'], 'render-f{}.png'.format(frame)))
            render_masks = self.render.load_multi_view_images(os.path.join(hyper_params['render_dir'], 'mask-f{}.png'.format(frame)))
            # load and clean segment_masks
            segment_masks = self.sam.load_segment_masks(os.path.join(hyper_params['segment_dir'], 'mask-f{}.json'.format(frame)))
            segment_masks = self.sam.clean_segment_masks(render_masks, segment_masks, min_area=100)

            # save segment_images from segment_masks and render_images
            for nv in range(render_masks.shape[0]):
                save_fn = os.path.join(hyper_params['manual_dir'], 'mask-f{}.{:02d}.png'.format(frame, nv))
                if os.path.exists(save_fn): continue
                save_image(save_fn, self.sam.view_masked_images(render_images[nv], segment_masks[nv]))

    # manually assign mask labels
    def update_manual_efforts(self, subj='', outfit='', seq='', surface_labels=None):
        # update manual_labels
        manual_labels = update_manual_labels(subj=subj, outfit=outfit, seq=seq, surface_labels=surface_labels)
        return manual_labels



class XHumansUtils(DatasetUtils):
    """
    XHumans Dataset Utility: data loader, render, parser, saver, viewer, surface parser
    """

    def __init__(self, dataset_dir='', dataset_args='', checkpoint_dir='', preprocess_init=False, dtype=torch.float32, device=torch.device('cuda:0')):
        # inherit DatasetUtils
        super(DatasetUtils, self).__init__(dataset_dir=dataset_dir, dataset_args=dataset_args, checkpoint_dir=checkpoint_dir,
                                           preprocess_init=preprocess_init, dtype=dtype, device=device)


    # load hyper_params for subj_outfit_seq
    def load_hyper_params(self, subj='', outfit='', seq=''):
        # init hyper_params
        hyper_params = dict()
        # append dataset folder dirs
        hyper_params['subj_outfit_seq_dir'] = os.path.join(self.dataset_dir, subj, outfit, seq)
        hyper_params['scan_dir'] = os.path.join(hyper_params['subj_outfit_seq_dir'], 'meshes_pkl')
        hyper_params['smplx_dir'] = os.path.join(hyper_params['subj_outfit_seq_dir'], 'SMPLX')
        hyper_params['smpl_dir'] = os.path.join(hyper_params['subj_outfit_seq_dir'], 'SMPL')
        hyper_params['semantic_dir'] = os.path.join(hyper_params['subj_outfit_seq_dir'], 'semantic')
        # locate sequence scan files
        hyper_params['scan_files'] = sorted(glob.glob(os.path.join(hyper_params['scan_dir'], 'mesh-f*.pkl')))
        hyper_params['scan_frames'] = [fn.split('/')[-1].split('.')[0][-5:] for fn in hyper_params['scan_files']]
        # append preprocess folder dirs
        hyper_params['render_dir'] = os.path.join(hyper_params['semantic_dir'], 'process', 'render')
        hyper_params['parser_dir'] = os.path.join(hyper_params['semantic_dir'], 'process', 'parser')
        hyper_params['optical_dir'] = os.path.join(hyper_params['semantic_dir'], 'process', 'optical')
        hyper_params['segment_dir'] = os.path.join(hyper_params['semantic_dir'], 'process', 'segment')
        hyper_params['manual_dir'] = os.path.join(hyper_params['semantic_dir'], 'process', 'manual')

        # update label hyper_params, assume Inner for XHumans
        hyper_params['surface'] = 'Inner'
        hyper_params['surface_labels'] = SURFACE_LABEL[:-1]
        hyper_params['surface_labels_colors'] = SURFACE_LABEL_COLOR[:-1]
        hyper_params['new_label'] = None
        
        # update render hyper_params
        hyper_params['render_views'] = self.render.num_view
        hyper_params['render_sizes'] = self.render.img_size
        hyper_params['render_ranges'] = self.check_subj_outfit_seq_render_range(hyper_params)
        # update Pytorch3d Renderer with render_views, render_sizes, and render_ranges
        self.render = PytorchRenderer(num_view=hyper_params['render_views'], img_size=hyper_params['render_sizes'],
                                      max_x=hyper_params['render_ranges'][0][0], min_x=hyper_params['render_ranges'][0][1],
                                      max_y=hyper_params['render_ranges'][1][0], min_y=hyper_params['render_ranges'][1][1],
                                      init_model=True, dtype=self.dtype, device=self.device)

        # update parse hyper_params
        hyper_params['scan_unary'] = {'mask': 1.5, 'manual': 1., 'parser': 0., 'optical': 0.}
        hyper_params['vote_weights'] = {'parser': 1., 'optical': 1.5, 'manual': 10}
        hyper_params['sam_skin'] = True  # Trick: better detect small skin regions

        # update manual_labels
        hyper_params['manual_efforts'] = self.update_manual_efforts(subj=subj, outfit=outfit, seq=seq, surface_labels=hyper_params['surface_labels'])
        # update and save hyper_params
        self.hyper_params = hyper_params
        save_pickle(os.path.join(hyper_params['subj_outfit_seq_dir'], 'hyper_params.pkl'), hyper_params)
        return hyper_params
