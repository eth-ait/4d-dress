
from lib.utility.general import *

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pygco'))
from lib.pygco import pygco


class SurfaceParser():
    """
    Human Surface Parser for 4D Textured Scan Sequences:
    Multi-view Human Parser + Optical Flow + Segment Mask + 3D Smoothness + Manual Efforts
    """

    def __init__(self, render, parser, optical, sam, hyper_params, dtype=torch.float32, device=torch.device('cuda:0')):
        super().__init__()

        # init basic settings
        self.dtype = dtype
        self.device = device
        # init render, parser, optical, and sam agents
        self.render = render
        self.parser = parser
        self.optical = optical
        self.sam = sam
        # init hyper_params
        self.hyper_params = hyper_params
        # init surface and surface_labels: skin 0, hair 1, shoe 2, upper 3, lower 4, outer 5
        self.surface, self.surface_labels = self.hyper_params['surface'], self.hyper_params['surface_labels']
        # init surface_labels_colors
        self.surface_labels_colors = self.hyper_params['surface_labels_colors']

    # GCO minimization with unarys(nvt, nl), edges(nve, 2), and smooth_weight=200, numpy-cpu
    def pygco_minimization(self, unarys, edges, init_labels=None, smooth_weight=200):
        # convert tensor to numpy
        unarys = unarys.cpu().numpy()
        # get edge and smooth weights
        edge_weights = np.ones(edges.shape[0])
        smooth_weights = (1 - np.eye(unarys.shape[-1])).astype(np.float) * smooth_weight
        # gco minimize unarys to labels
        labels = pygco.cut_general_graph(edges, edge_weights, unarys.astype(np.float), smooth_weights,
                                         init_labels=init_labels, n_iter=-1, algorithm="expansion")
        # return torch labels
        return torch.tensor(labels, dtype=torch.long).to(self.device)

    # get multi_view_unary_term(nvt, nl) from multi-view votes (nv, h, w, nl)
    def get_multi_view_unary_term(self, nvt_mesh, faces_mesh, render_rasts, render_masks, render_votes):
        # get multi-view dimensions
        n_views, h, w = render_masks.shape[:3]
        # init multi-view unary_votes (nvt_mesh, nl)
        unary_votes = torch.zeros((nvt_mesh, len(self.surface_labels))).to(self.device)
        # init final_pixel_votes(nvp, 3, nl) and final_pixel_faces(nvp, 3)
        final_pixel_votes, final_pixel_faces = [], []
        # collect votes from all render views
        for nv in range(n_views):
            # get render pix_points and pix_votes
            pix_points = torch.nonzero(render_masks[nv])
            pix_votes = render_votes[nv][pix_points[:, 0], pix_points[:, 1]]
            # get render pix_to_faces and pix_bary_coords
            pix_to_faces = render_rasts.pix_to_face[nv][pix_points[:, 0], pix_points[:, 1]]
            pix_bary_coords = render_rasts.bary_coords[nv][pix_points[:, 0], pix_points[:, 1]]
            # assign unary_votes from pix_votes and pix_to_faces, weighted with pix_bary_coords
            for nf in range(pix_to_faces.shape[-1]):
                # get pixel_votes(nvp, nl), pixel_faces(nvp, 3) and pixel_bary_coords(nvp, 3)
                pixel_votes = pix_votes[pix_to_faces[:, nf] >= 0]
                pixel_faces = faces_mesh[pix_to_faces[pix_to_faces[:, nf] >= 0, nf] - faces_mesh.shape[0] * nv, :]
                pixel_bary_coords = pix_bary_coords[pix_to_faces[:, nf] >= 0, nf]
                if pixel_votes.shape[0] == 0: continue
                # append final_pixel_votes(nvp, 3, nl) and final_pixel_faces(nvp, 3)
                final_pixel_votes.append(pixel_votes.unsqueeze(1).repeat(1, 3, 1) * pixel_bary_coords.unsqueeze(-1).repeat(1, 1, len(self.surface_labels)))
                final_pixel_faces.append(pixel_faces)
        # cat final_pixel_votes(nv*nvp, 3, nl) and final_pixel_faces(nv*nvp, 3)
        final_pixel_votes = torch.cat(final_pixel_votes, dim=0)
        final_pixel_faces = torch.cat(final_pixel_faces, dim=0)
        # process all vertices, average final_pixel_votes(nv*nvp, 3, nl)
        for nvt in range(nvt_mesh):
            if torch.sum(final_pixel_faces == nvt) == 0: continue
            unary_votes[nvt, :] = torch.mean(final_pixel_votes[final_pixel_faces == nvt], dim=0)
        unary_votes = torch.nan_to_num(unary_votes)

        # assign unary_labels(nvt, ) from unary_votes, filter label without votes
        unary_labels = torch.max(unary_votes, dim=-1).indices
        unary_labels[torch.sum(unary_votes, dim=-1) == 0] = -1
        # assign unary_label_term(nvt, nl) from unary_labels(nvt, )
        unary_label_term = torch.zeros((nvt_mesh, len(self.surface_labels))).to(self.device)
        for nl in range(len(self.surface_labels)):
            unary_label_term[unary_labels == nl, nl] += -200  # -200
            unary_label_term[unary_labels != nl, nl] += 0  # 0
        # return unary_labels (nvt, ) and unary_label_term (nvt, nl)
        return unary_labels, unary_label_term


    # parse current scan frame using current parser_images, current segment_masks, previous render_labels and previous-current optical_flows
    def forward(self, init_params, init_meshes, init_images, frame, first_frame=False, block_manual=False):

        # # ==================== Unpack Scan Data ==================== # #
        # get scan verts(nvt, 3), faces(nvt, 3)
        nvt_scan = init_meshes['scan_mesh']['vertices'].shape[0]
        th_verts_scan = torch.tensor(init_meshes['scan_mesh']['vertices'], dtype=self.dtype).unsqueeze(0).to(self.device)
        th_faces_scan = torch.tensor(init_meshes['scan_mesh']['faces'], dtype=torch.long).unsqueeze(0).to(self.device)

        # get current render_images(nv, h, w, 3), render_masks(nv, h, w), and parser_images(nv, h, w, 3)
        render_images = torch.tensor(init_images['render_images'], dtype=self.dtype).to(self.device)
        render_masks = torch.tensor(init_images['render_masks'], dtype=self.dtype).to(self.device)
        # get render_rasterize from render
        render_rasts = self.render.get_mesh_rasterizer(th_verts_scan, th_faces_scan)
        
        # get current parser_labels(nv, h, w) from parser_images(nv, h, w, 3)
        parser_images = torch.tensor(init_images['parser_images'], dtype=self.dtype).to(self.device)
        parser_values, parser_labels = self.parser.decode_parser_images(parser_images, self.surface, self.surface_labels, group=False)
        # get current parser_votes(nv, h, w, nl) from parser_labels(nv, h, w)
        parser_votes = torch.stack([1. * (parser_labels == nl) for nl in range(len(self.surface_labels))], dim=-1)
        
        # get previous-current optical_flows(nv, h, w, 2)
        optical_flows = torch.tensor(init_images['optical_flows'], dtype=self.dtype).to(self.device)
        # get previous labels from previous params
        previous_labels = parser_labels if first_frame else init_params['scan_params']['render_labels']
        # get optical_votes(nv, h, w, nl) from optical_flows(nv, h, w, 2) and previous_labels(nv, h, w)
        optical_votes = self.optical.transfer_labels(previous_labels, optical_flows, self.surface_labels)

        # get current segment_masks (nv, nm, h, w)
        segment_masks = init_images['segment_masks']


        # # ==================== Unpack Manual Efforts ==================== # #
        
        # init hyper_params
        hyper_params, manual_flag = self.hyper_params.copy(), False
        # init current manual_votes (nv, h, w, nl)
        manual_votes = torch.zeros_like(parser_votes)
        # unpack manual_labels {'frame': [[nv, nm, nl], ...]}
        if 'mask_labels' in hyper_params['manual_efforts']['manual_labels'] and not block_manual:
            # loop over all mask_labels
            for manual_frame, manual_frame_labels in hyper_params['manual_efforts']['manual_labels']['mask_labels'].items():
                # check current frame
                if frame != manual_frame: continue
                # turn on manual_flag
                manual_flag = True
                # process all frame_labels
                for nf in range(manual_frame_labels.shape[0]):
                    manual_view, manual_mask, manual_label = manual_frame_labels[nf, :3]
                    if manual_view >= len(segment_masks):
                        print('\n\n', 'wrong manual_view id:', manual_view, len(segment_masks), '\n\n')
                        continue
                    if manual_mask >= len(segment_masks[manual_view]):
                        print('\n\n', 'wrong manual_mask id:', manual_view, len(segment_masks[manual_view]), '\n\n')
                        continue
                    manual_votes[manual_view, :, :, manual_label][segment_masks[manual_view][manual_mask]['segmentation']] = 1.

        # # ==================== First Round Scan Mesh Parsing ==================== # #
        # init unary_term_scan
        unary_term_scan = torch.zeros((nvt_scan, len(self.surface_labels))).to(self.device)

        # # -------------------- Comp1: Mask Unary Term for SCAN  -------------------- # #
        if hyper_params['scan_unary']['mask'] > 0.:
            # assign mask_votes_labels from parser_votes and optical_votes
            mask_votes_labels = {
                'parser_votes': {
                    'value': parser_votes.to(self.device),  # (nv, h, w, nl)
                    'weight': hyper_params['vote_weights']['parser'],  # 1.
                },
                'optical_votes': {
                    'value': optical_votes.to(self.device),  # (nv, h, w, nl)
                    'weight': hyper_params['vote_weights']['optical'],  # 1.5
                },
            }
            # get multi-view segment_mask_votes (nv, h, w, nl), 2D mask smoothness
            segment_mask_votes = self.sam.vote_segment_masks(
                render_masks, segment_masks, mask_votes_labels, self.surface_labels, sam_skin=hyper_params['sam_skin'])
            # get mask_unary_terms(nvt, nl) from segment_mask_votes(nv, h, w, nl)
            mask_labels_scan, unary_mask_label_scan = \
                self.get_multi_view_unary_term(nvt_scan, th_faces_scan.squeeze(0), render_rasts, render_masks, segment_mask_votes)
            # append unary_mask_vote_scan and unary_mask_label_scan
            unary_term_scan += hyper_params['scan_unary']['mask'] * unary_mask_label_scan
        
        # minimize scan unary term using gco
        scan_labels = self.pygco_minimization(unary_term_scan, init_meshes['scan_mesh']['edges'], smooth_weight=200)


        # # ==================== Second Round Scan Mesh Parsing ==================== # #
        # init unary_term_scan
        unary_term_scan = torch.zeros((nvt_scan, len(self.surface_labels))).to(self.device)

        # # -------------------- Comp1: Mask Unary Term for SCAN  -------------------- # #
        # render scan_labels to multi-view render_labels(nv, h, w) and render_labels_votes(nv, h, w, nl)
        render_labels, render_labels_votes = self.render.render_mesh_labels(
            scan_labels, th_faces_scan.squeeze(0), render_rasts, render_masks, self.surface_labels)

        # second time refinement with rendered refine_votes and manual_votes
        if hyper_params['scan_unary']['mask'] > 0.:
            # assign mask_votes_labels from parser_votes and optical_votes
            mask_votes_labels = {
                'refine_votes': {
                    'value': render_labels_votes.to(self.device),  # (nv, h, w, nl)
                    'weight': hyper_params['vote_weights']['parser'],  # 1.
                },
                'manual_votes': {
                    'value': manual_votes.to(self.device),  # (nv, h, w, nl)
                    'weight': hyper_params['vote_weights']['manual'],  # 10 / 20
                },
            }
            # get multi-view segment_mask_votes (nv, h, w, nl), 2D mask smoothness
            segment_mask_votes = self.sam.vote_segment_masks(
                render_masks, segment_masks, mask_votes_labels, self.surface_labels, sam_skin=hyper_params['sam_skin'])
            # get mask_unary_terms from mask_votes
            mask_labels_scan, unary_mask_label_scan = \
                self.get_multi_view_unary_term(nvt_scan, th_faces_scan.squeeze(0), render_rasts, render_masks, segment_mask_votes)
            # append unary_mask_vote_scan and unary_mask_label_scan, with higher weight
            unary_term_scan += hyper_params['scan_unary']['mask'] * unary_mask_label_scan

        # # -------------------- Comp2: Direct Manual Unary Term for SCAN  -------------------- # #
        # second time refinement with manual_votes
        if hyper_params['scan_unary']['manual'] > 0. and not block_manual:
            # append manual_votes and manual_optical_votes
            if torch.sum(manual_votes) > 0.:
                # get segment_mask unary_terms from manual_mask_votes
                manual_labels_scan, unary_manual_label_scan = \
                    self.get_multi_view_unary_term(nvt_scan, th_faces_scan.squeeze(0), render_rasts, render_masks, manual_votes.to(self.device))
                # append unary_manual_vote_scan and unary_manual_label_scan
                unary_term_scan += hyper_params['scan_unary']['manual'] * unary_manual_label_scan
        
        # minimize scan unary term using gco
        scan_labels = self.pygco_minimization(unary_term_scan, init_meshes['scan_mesh']['edges'], smooth_weight=200)


        # # ==================== Pack Parsing Results ==================== # #

        # -------------------- Render Final Optimized Labels  -------------------- # #
        # render scan_labels to multi-view render_labels(nv, h, w) and render_labels_votes(nv, h, w, nl)
        render_labels, render_labels_votes = self.render.render_mesh_labels(
            scan_labels, th_faces_scan.squeeze(0), render_rasts, render_masks, self.surface_labels)
        # render multi-view render_labels(nv, h, w) to render_labels_images(nv, h, w, 3)
        render_labels_images = self.render.render_label_images(render_labels.cpu().numpy(), self.surface_labels, self.surface_labels_colors)
        render_labels_images = self.render.pack_multi_view_images(render_labels_images)
        # show_image(render_labels_images)
        
        # # -------------------- Render Manual Labels  -------------------- # #
        manual_labels, manual_labels_images = None, None
        if torch.sum(manual_votes) > 0:
            # assign manual_labels(nv, h, w, 1) from manual_votes(nv, h, w, nl)
            manual_labels = torch.max(manual_votes.to(self.device), dim=-1).indices
            manual_labels[torch.sum(manual_votes, dim=-1) == 0] = -1
            # render manual_labels to multi-view render_labels_images
            manual_labels_images = self.render.render_label_images(manual_labels.cpu().numpy(), self.surface_labels, self.surface_labels_colors)
            manual_labels_images = self.render.pack_multi_view_images(manual_labels_images)
            # show_image(manual_labels_images)


        ## ==================== Return Optimized Params  ==================== # #
        # pack up params
        params = {
            'scan_params':{
                'labels': scan_labels,  # (nvt,)
                'vertices': init_meshes['scan_mesh']['vertices'],  # (nvt, 3)
                'render_images': init_images['render_images'],  # (nv, h, w, 3)
                'render_labels': render_labels,  # (nv, h, w)
                'render_labels_images': render_labels_images,  # (nv, h, w, 3)
                'manual_flag': manual_flag,
                'manual_labels': manual_labels,  # (nv, h, w)
                'manual_labels_images': manual_labels_images,  # (nv, h, w, 3)
            }
        }
        return params