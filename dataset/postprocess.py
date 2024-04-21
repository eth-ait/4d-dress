from utility import *
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer

# load pytorch3d cameras from parameters: intrinsics, extrinsics
def load_pytorch_cameras(camera_params, camera_list, image_shape):
    # init camera_dict
    camera_dict = dict()
    # process all camera within camera_list
    for camera_id in camera_list:
        # assign camera intrinsic and extrinsic matrices
        intrinsic = torch.tensor((camera_params[camera_id]["intrinsics"]), dtype=torch.float32).cuda()
        extrinsic = torch.tensor(camera_params[camera_id]["extrinsics"], dtype=torch.float32).cuda()
        # assign camera image size
        image_size = torch.tensor([image_shape[0], image_shape[1]], dtype=torch.float32).unsqueeze(0).cuda()

        # assign camera parameters
        f_xy = torch.cat([intrinsic[0:1, 0], intrinsic[1:2, 1]], dim=0).unsqueeze(0)
        p_xy = intrinsic[:2, 2].unsqueeze(0)
        R = extrinsic[:, :3].unsqueeze(0)
        T = extrinsic[:, 3].unsqueeze(0)
        # coordinate system adaption to PyTorch3D
        R[:, :2, :] *= -1.0
        # camera position in world space -> world position in camera space
        T[:, :2] *= -1.0
        R = torch.transpose(R, 1, 2)  # row-major
        # assign Pytorch3d PerspectiveCameras
        camera_dict[camera_id] = PerspectiveCameras(focal_length=f_xy, principal_point=p_xy, R=R, T=T, in_ndc=False, image_size=image_size).cuda()
    # assign Pytorch3d RasterizationSettings
    raster_settings = RasterizationSettings(image_size=image_shape, blur_radius=0.0, faces_per_pixel=1, max_faces_per_bin=80000)
    return camera_dict, raster_settings

# render pixel labels(nv, h, w) from mesh labels(nvt, ), faces(nft, 3), uvs(nvt, 2), and render_masks(nv, h, w)
def render_mesh_pixel_labels(labels, faces, render_rasts, render_masks, surface_labels):
    # get dimensions
    n_views, h, w = render_masks.shape[:3]
    # init label_votes (nv, h, w, nl) for multi-view images
    label_votes = torch.zeros((n_views, h, w, len(surface_labels))).cuda()
    # render labels for all views
    for nv in range(n_views):
        # get render pix_points, pix_to_faces, and pix_bary_coords
        pix_points = torch.nonzero(render_masks[nv])
        pix_to_faces = render_rasts.pix_to_face[nv][pix_points[:, 0], pix_points[:, 1]]
        pix_bary_coords = render_rasts.bary_coords[nv][pix_points[:, 0], pix_points[:, 1]]
        # project one pixel to one face if exists
        for nf in range(1):
            # get pixel_points(nvp, 2)
            pixel_points = pix_points[pix_to_faces[:, nf] >= 0]
            # get pixel_faces(nvp, ), pixel_bary_coords(nvp, 3), and pixel_faces_labels(nvp, 3)
            pixel_faces = pix_to_faces[pix_to_faces[:, nf] >= 0, nf] - faces.shape[0] * nv
            pixel_bary_coords = pix_bary_coords[pix_to_faces[:, nf] >= 0, nf]
            pixel_faces_labels = labels[faces[pixel_faces, :]]
            # assign votes to surface labels: skin, upper, lower, hair, shoe, outer
            for nl in range(len(surface_labels)):
                # loop over all face vertices
                for n in range(pixel_faces_labels.shape[-1]):
                    nl_labels = pixel_faces_labels[:, n] == nl
                    if torch.count_nonzero(nl_labels) == 0: continue
                    label_votes[nv, pixel_points[nl_labels][:, 0], pixel_points[nl_labels][:, 1], nl] += 1 * pixel_bary_coords[nl_labels, n]
    # collect render_labels from label_votes, filter label without votes
    render_labels = torch.max(label_votes, dim=-1).indices
    render_labels[torch.sum(label_votes, dim=-1) == 0] = -1
    return render_labels, label_votes

# render pixel labels (nv, h, w) to colors (nv, h, w, 3)
def render_pixel_label_colors(labels):
    # init parser_images with white background
    images = np.ones((*labels.shape[:3], 3)) * 255
    # loop over all parser images
    for nv in range(images.shape[0]):
        for nl in range(len(SURFACE_LABEL)):
            images[nv][labels[nv] == nl] = SURFACE_LABEL_COLOR[nl]
    return images.astype(np.uint8)

# extract label meshes for the entire subj_outfit_seq
def subj_outfit_seq_render_pixel_labels(dataset_dir, subj, outfit, seq):
    # locate subj_outfit_seq_dir
    subj_outfit_seq_dir = os.path.join(dataset_dir, subj, outfit, seq)
    # load basic sequence info
    basic_info = load_pickle(os.path.join(subj_outfit_seq_dir, 'basic_info.pkl'))
    scan_frames = basic_info['scan_frames']
    print('# # ============ Render Pixel Labels in Subj_Outfit_Seq: {}_{}_{} // Frames: {}'.format(subj, outfit, seq, len(scan_frames)))

    # # -------------------- Locate Scan, Label and Capture Folders -------------------- # #
    # locate scan, label and capture dir
    scan_dir = os.path.join(subj_outfit_seq_dir, 'Meshes_pkl')
    label_dir = os.path.join(subj_outfit_seq_dir, 'Semantic', 'labels')
    capture_dir = os.path.join(subj_outfit_seq_dir, 'Capture')

    # # -------------------- Load Capture Cameras -------------------- # #
    # locate camera_list, make labels folders
    camera_list = RELEASE_CAMERAS
    for cam_id in camera_list:
        os.makedirs(os.path.join(capture_dir, cam_id, 'labels'), exist_ok=True)
    # load camera_params
    image_shape = (1280, 940)
    camera_params = load_pickle(os.path.join(capture_dir, 'cameras.pkl'))
    # load pytorch3d camera_agents and raster_settings
    camera_agents, raster_settings = load_pytorch_cameras(camera_params, camera_list, image_shape)

    # # -------------------- Render Labels For All Cameras Frames -------------------- # #
    # process all frames
    loop = tqdm.tqdm(range(len(scan_frames)))
    for n_frame in loop:
        # locate current frame
        frame = scan_frames[n_frame ]
        loop.set_description('## Render Frame: {}/{}'.format(frame, scan_frames[-1]))

        # # -------------------- Load Scan Mesh and Label -------------------- # #
        # load scan_mesh to pytorch3d for rasterize
        scan_data = load_pickle(os.path.join(scan_dir, 'mesh-f{}.pkl'.format(frame)))
        th_verts = torch.tensor(scan_data['vertices'], dtype=torch.float32).unsqueeze(0)
        th_faces = torch.tensor(scan_data['faces'], dtype=torch.long).unsqueeze(0)
        scan_mesh = Meshes(th_verts, th_faces).cuda()
        # load scan_label
        scan_labels = load_pickle(os.path.join(label_dir, 'label-f{}.pkl'.format(frame)))['scan_labels']
        scan_labels = torch.tensor(scan_labels).cuda()

        # # -------------------- Process ALL Capture Cameras -------------------- # #
        for cam_id, camera_agent in camera_agents.items():
            # locate save_label_fn and save_overlap_fn
            save_label_fn = os.path.join(capture_dir, cam_id, 'labels', 'label-f{}.png'.format(frame))
            save_overlap_fn = os.path.join(capture_dir, cam_id, 'labels', 'overlap-f{}.png'.format(frame))
            if os.path.exists(save_label_fn) and os.path.exists(save_overlap_fn): continue

            # # -------------------- Load Capture Image and Camera -------------------- # #
            # get capture_image(h, w, 3)
            capture_image = load_image(os.path.join(capture_dir, cam_id, 'images', 'capture-f{}.png'.format(frame)))
            # get capture_rasts from camera and scan_mesh
            capture_rasts = MeshRasterizer(cameras=camera_agent, raster_settings=raster_settings)(scan_mesh)
            # render capture_mask(h, w) and capture_mask_image(h, w)
            capture_mask = capture_rasts.pix_to_face[0, :, :, 0] > -1

            # # -------------------- Render Vertex Labels -------------------- # #
            # render scan_labels(nvt, ) to multi_view capture_labels(nv, h, w) and capture_labels_votes(nv, h, w, nl)
            capture_labels, capture_labels_votes = render_mesh_pixel_labels(
                scan_labels, th_faces.squeeze(0), capture_rasts, capture_mask.unsqueeze(0), SURFACE_LABEL)
            # render multi-view render_labels(nv, h, w) to render_labels_images(nv, h, w, 3)
            capture_labels_images = render_pixel_label_colors(capture_labels.cpu().numpy())[0]
            save_image(save_label_fn, capture_labels_images)
            # overlap capture_image and capture_label
            capture_labels_images_overlap = cv.addWeighted(capture_image, 0.5, capture_labels_images, 0.5, 0.0)
            save_image(save_overlap_fn, capture_labels_images_overlap)


# extract label meshes from scan_mesh
def extract_label_meshes(vertices, faces, labels, surface_labels, colors=None, uvs=None):
    # init label_meshes and face_labels
    label_meshes = dict()
    face_labels = labels[faces]
    # loop over all labels
    for nl in range(len(surface_labels)):
        # skip empty label
        if np.sum(labels == nl) == 0: continue
        # find label faces: with label vertices == 3
        vertex_label_nl = np.where(labels == nl)[0]
        face_label_nl = np.where(np.sum(face_labels == nl, axis=-1) == 3)[0]
        # find correct indices
        correct_indices = (np.zeros(labels.shape[0]) - 1).astype(int)
        correct_indices[vertex_label_nl] = np.arange(vertex_label_nl.shape[0])
        # extract label_mesh[vertices, faces]
        label_meshes[surface_labels[nl]] = {'vertices': vertices[vertex_label_nl], 'faces': correct_indices[faces[face_label_nl]]}
        # extract label_mesh colors
        label_meshes[surface_labels[nl]]['colors'] = colors[vertex_label_nl] if colors is not None else None
        # extract label_mesh uvs
        label_meshes[surface_labels[nl]]['uvs'] = uvs[vertex_label_nl] if uvs is not None else None
    return label_meshes

# extract label meshes for the entire subj_outfit_seq
def subj_outfit_seq_extract_label_meshes(dataset_dir, subj, outfit, seq):
    # locate subj_outfit_seq_dir
    subj_outfit_seq_dir = os.path.join(dataset_dir, subj, outfit, seq)
    # load basic sequence info
    basic_info = load_pickle(os.path.join(subj_outfit_seq_dir, 'basic_info.pkl'))
    scan_frames = basic_info['scan_frames']
    print('# # ============ Extract Labeled Clothes in Subj_Outfit_Seq: {}_{}_{} // Frames: {}'.format(subj, outfit, seq, len(scan_frames)))

    # locate scan, label, cloth dir
    scan_dir = os.path.join(subj_outfit_seq_dir, 'Meshes_pkl')
    label_dir = os.path.join(subj_outfit_seq_dir, 'Semantic', 'labels')
    cloth_dir = os.path.join(subj_outfit_seq_dir, 'Semantic', 'clothes')
    os.makedirs(cloth_dir, exist_ok=True)
    
    # process all frames
    loop = tqdm.tqdm(range(len(scan_frames)))
    for n_frame in loop:
        # locate current frame
        frame = scan_frames[n_frame]
        loop.set_description('## Extract Frame: {}/{}'.format(frame, scan_frames[-1]))
        # locate save_cloth_fn
        save_cloth_fn = os.path.join(cloth_dir, 'cloth-f{}.pkl'.format(frame))
        if os.path.exists(save_cloth_fn): continue

        # extract clothes from scan_mesh
        scan_mesh = load_pickle(os.path.join(scan_dir, 'mesh-f{}.pkl'.format(frame)))
        scan_labels = load_pickle(os.path.join(label_dir, 'label-f{}.pkl'.format(frame)))['scan_labels']
        clothes = extract_label_meshes(scan_mesh['vertices'], scan_mesh['faces'], scan_labels, SURFACE_LABEL, scan_mesh['colors'], scan_mesh['uvs'])
        save_pickle(save_cloth_fn, clothes)

            
if __name__ == "__main__":
    # set target subj_outfit_seq
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', default='00122', help='subj name')
    parser.add_argument('--outfit', default='Outer', help='outfit name')
    parser.add_argument('--seq', default='Take9', help='seq name')
    args = parser.parse_args()

    # TODO: Set 4D-DRESS DATASET_DIR in utility.py
    # render multi-view pixel labels within subj_outfit_seq
    subj_outfit_seq_render_pixel_labels(dataset_dir=DATASET_DIR, subj=args.subj, outfit=args.outfit, seq=args.seq)
    # extract labeled cloth meshes within subj_outfit_seq
    subj_outfit_seq_extract_label_meshes(dataset_dir=DATASET_DIR, subj=args.subj, outfit=args.outfit, seq=args.seq)
