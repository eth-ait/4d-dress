from utility import *
from aitviewer.viewer import Viewer as AITViewer
from aitviewer.renderables.meshes import VariableTopologyMeshes as VTMeshes


# render vertex_label_colors(nvt, 4) according to vertex_labels: skin-0, hair-1, shoe-2, upper-3, lower-4, outer-5
def render_vertex_label_colors(label):
    # init white color as background
    colors = np.ones((label.shape[0], 3)) * 255
    # assign label color
    for nl in range(len(SURFACE_LABEL)):
        colors[label == nl] = SURFACE_LABEL_COLOR[nl]
    # append color with the fourth channel
    colors = np.append(colors, np.ones((colors.shape[0], 1)) * 255, axis=-1) / 255.
    return colors


# compact view subj_outfit_seq scan, smpl, smplx, label, clothes
def compact_view_subj_outfit_seq(dataset_dir='', subj='', outfit='', seq='', n_start=0, n_stop=-1,
                                 view_smpl=False, view_smplx=False, view_label=False, view_cloth=False, view_vertex_color=True):

    # init AIT Viewers
    aitvs = AITViewer(title='{}_{}_{}'.format(subj, outfit, seq))
    aitvs.scene.floor.enabled = False
    aitvs.scene.origin.enabled = False
    aitvs.playback_fps = 10  # 30
    # init segment_meshes
    scan_meshes = {'vertices': [], 'back_vertices': [], 'faces': [], 
                   'colors': [], 'label_colors': [], 'uvs': [], 'uv_path': []}
    smpl_meshes = {'vertices': [], 'back_vertices': [], 'faces': []}
    smplx_meshes = {'vertices': [], 'back_vertices': [], 'faces': []}
    cloth_meshes = {}

    # locate scan, smpl, smplx, label and cloth folders
    subj_outfit_seq_dir = os.path.join(dataset_dir, subj, outfit, seq)
    scan_dir = os.path.join(subj_outfit_seq_dir, 'Meshes_pkl')
    smpl_dir = os.path.join(subj_outfit_seq_dir, 'SMPL')
    smplx_dir = os.path.join(subj_outfit_seq_dir, 'SMPLX')
    label_dir = os.path.join(subj_outfit_seq_dir, 'Semantic', 'labels')
    cloth_dir = os.path.join(subj_outfit_seq_dir, 'Semantic', 'clothes')

    # locate scan_frames from basic_info
    basic_info = load_pickle(os.path.join(subj_outfit_seq_dir, 'basic_info.pkl'))
    scan_frames, scan_rotation = basic_info['scan_frames'], basic_info['rotation']
    print('# # ============ Compact View Subj_Outfit_Seq: {}_{}_{} // Frames: {}'.format(subj, outfit, seq, len(scan_frames)))

    # loop over all sampled scan frames
    loop = tqdm.tqdm(range(n_start, len(scan_frames)))
    for n_frame in loop:
        # check stop frame
        if 0 <= n_stop < n_frame: break
        frame = scan_frames[n_frame]
        loop.set_description('## Loading Frame for {}_{}_{}: {}/{}'.format(subj, outfit, seq, frame, scan_frames[-1]))

        # locate scan, smpl, smplx files
        scan_mesh_fn = os.path.join(scan_dir, 'mesh-f{}.pkl'.format(frame))
        smpl_mesh_fn = os.path.join(smpl_dir, 'mesh-f{}_smpl.ply'.format(frame))
        smplx_mesh_fn = os.path.join(smplx_dir, 'mesh-f{}_smplx.ply'.format(frame))
        scan_label_fn = os.path.join(label_dir, 'label-f{}.pkl'.format(frame))
        scan_cloth_fn = os.path.join(cloth_dir, 'cloth-f{}.pkl'.format(frame))

        # load scan_mesh with vertex colors
        scan_mesh = load_pickle(scan_mesh_fn)
        scan_mesh['uv_path'] = scan_mesh_fn.replace('mesh-f', 'atlas-f')
        if 'colors' not in scan_mesh:
            # load atlas data
            atlas_data = load_pickle(scan_mesh['uv_path'])
            # load scan uv_coordinate and uv_image as TextureVisuals
            uv_image = Image.fromarray(atlas_data).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB")
            texture_visual = trimesh.visual.texture.TextureVisuals(uv=scan_mesh['uvs'], image=uv_image)
            # pack scan data as trimesh
            scan_trimesh = trimesh.Trimesh(
                vertices=scan_mesh['vertices'],
                faces=scan_mesh['faces'],
                vertex_normals=scan_mesh['normals'],
                visual=texture_visual,
                process=False,
            )
            scan_mesh['colors'] = scan_trimesh.visual.to_color().vertex_colors
        # rotate scan_mesh to view front
        if scan_rotation is not None: scan_mesh['vertices'] = np.matmul(scan_rotation, scan_mesh['vertices'].T).T
        # append scan_meshes
        scan_meshes['vertices'].append(scan_mesh['vertices'])
        scan_meshes['back_vertices'].append(np.matmul(rotation_matrix(180, axis='y'), scan_mesh['vertices'].T).T)
        scan_meshes['faces'].append(scan_mesh['faces'])
        scan_meshes['colors'].append(scan_mesh['colors'] / 255.)
        scan_meshes['uvs'].append(scan_mesh['uvs'])
        scan_meshes['uv_path'].append(scan_mesh['uv_path'])

        # load smpl_mesh, centrailize
        if view_smpl:
            smpl_trimesh = trimesh.load_mesh(smpl_mesh_fn)
            if scan_rotation is not None: smpl_trimesh.vertices = np.matmul(scan_rotation, smpl_trimesh.vertices.T).T
            smpl_meshes['vertices'].append(smpl_trimesh.vertices)
            smpl_meshes['back_vertices'].append(np.matmul(rotation_matrix(180, axis='y'), smpl_trimesh.vertices.T).T)
            smpl_meshes['faces'].append(smpl_trimesh.faces)

        # load smplx_mesh, centrailize
        if view_smplx:
            smplx_trimesh = trimesh.load_mesh(smplx_mesh_fn)
            if scan_rotation is not None: smplx_trimesh.vertices = np.matmul(scan_rotation, smplx_trimesh.vertices.T).T
            smplx_meshes['vertices'].append(smplx_trimesh.vertices)
            smplx_meshes['back_vertices'].append(
                np.matmul(rotation_matrix(180, axis='y'), smplx_trimesh.vertices.T).T)
            smplx_meshes['faces'].append(smplx_trimesh.faces)

        # load scan_label, render color
        if view_label:
            scan_labels = load_pickle(scan_label_fn)['scan_labels']
            scan_meshes['label_colors'].append(render_vertex_label_colors(scan_labels))

        # load scan_clothes: upper, lower, outer, ...
        if view_cloth:
            # check whether clothes have been extracted
            if not os.path.exists(scan_cloth_fn):
                from postprocess import subj_outfit_seq_extract_label_meshes
                subj_outfit_seq_extract_label_meshes(dataset_dir=dataset_dir, subj=subj, outfit=outfit, seq=seq)
            # load extracted clothes
            scan_clothes = load_pickle(scan_cloth_fn)
            for k, v in scan_clothes.items():
                if n_frame == 0:
                    cloth_meshes[k] = {'vertices': [], 'faces': [], 'colors': [], 'uvs': [], 'uv_path': []}
                if scan_rotation is not None: scan_clothes[k]['vertices'] = np.matmul(scan_rotation, scan_clothes[k]['vertices'].T).T
                cloth_meshes[k]['vertices'].append(scan_clothes[k]['vertices'])
                cloth_meshes[k]['faces'].append(scan_clothes[k]['faces'])
                cloth_meshes[k]['colors'].append(scan_clothes[k]['colors'] / 255.)
                cloth_meshes[k]['uvs'].append(scan_clothes[k]['uvs'])
                cloth_meshes[k]['uv_path'].append(scan_mesh['uv_path'])
            
    # view scan, smpl, smplx, label, and cloth meshes
    base = np.array([1, 0., 0.])

    # AIT view front and back scan_mesh, using uv map: slower
    if not view_vertex_color:
        # AIT view front scan_mesh, using uv color
        aitvs.scene.add(VTMeshes(scan_meshes['vertices'], scan_meshes['faces'], uv_coords=scan_meshes['uvs'], 
                                 texture_paths=scan_meshes['uv_path'], position=base, name='scan_front'))
        aitvs.scene.nodes[-1].backface_culling = False
        # AIT view back scan_mesh, using uv color
        aitvs.scene.add(VTMeshes(scan_meshes['back_vertices'], scan_meshes['faces'], uv_coords=scan_meshes['uvs'], 
                                 texture_paths=scan_meshes['uv_path'], position=base + np.array([0., 2., 0.]), name='scan_back'))
        aitvs.scene.nodes[-1].backface_culling = False
    else:
        # AIT view front scan_mesh, using vertex color: faster
        aitvs.scene.add(VTMeshes(scan_meshes['vertices'], scan_meshes['faces'], vertex_colors=scan_meshes['colors'],
                                 position=base, name='scan_front'))
        aitvs.scene.nodes[-1].backface_culling = False
        # AIT view back scan_mesh, using vertex color: faster
        aitvs.scene.add(VTMeshes(scan_meshes['back_vertices'], scan_meshes['faces'], vertex_colors=scan_meshes['colors'],
                                 position=base + np.array([0., 2., 0.]), name='scan_back'))
        aitvs.scene.nodes[-1].backface_culling = False

    # view front and back smpl meshes
    if view_smpl:
        base = np.array([-1, 0., 0.])
        # AIT view front smpl_mesh, using vertex color
        aitvs.scene.add(VTMeshes(smpl_meshes['vertices'], smpl_meshes['faces'], position=base, name='smpl_front'))
        aitvs.scene.nodes[-1].backface_culling = False
        # AIT view back smpl_mesh, using vertex color
        aitvs.scene.add(VTMeshes(smpl_meshes['back_vertices'], smpl_meshes['faces'], position=base + np.array([0., 2., 0.]), name='smpl_back'))
        aitvs.scene.nodes[-1].backface_culling = False

        # AIT view front scan_mesh, using vertex color
        aitvs.scene.add(VTMeshes(scan_meshes['vertices'], scan_meshes['faces'], vertex_colors=scan_meshes['colors'], 
                                 position=base, name='smpl_scan_front'))
        aitvs.scene.nodes[-1].backface_culling = False
        # AIT view back scan_mesh, using vertex color
        aitvs.scene.add(VTMeshes(scan_meshes['back_vertices'], scan_meshes['faces'], vertex_colors=scan_meshes['colors'], 
                                 position=base + np.array([0., 2., 0.]), name='smpl_scan_back'))
        aitvs.scene.nodes[-1].backface_culling = False

    # view front and back smplx meshes
    if view_smplx:
        base = np.array([-2.5, 0., 0.])
        # AIT view front smplx_mesh, using vertex color
        aitvs.scene.add(VTMeshes(smplx_meshes['vertices'], smplx_meshes['faces'], 
                                 position=base, name='smplx_front'))
        aitvs.scene.nodes[-1].backface_culling = False
        # AIT view back smplx_mesh, using vertex color
        aitvs.scene.add(VTMeshes(smplx_meshes['back_vertices'], smplx_meshes['faces'],
                                 position=base + np.array([0., 2., 0.]), name='smplx_back'))
        aitvs.scene.nodes[-1].backface_culling = False

        # AIT view front scan_mesh, using vertex color
        aitvs.scene.add(VTMeshes(scan_meshes['vertices'], scan_meshes['faces'], vertex_colors=scan_meshes['colors'],
                                 position=base, name='smplx_scan_front'))
        aitvs.scene.nodes[-1].backface_culling = False
        # AIT view back scan_mesh, using vertex color
        aitvs.scene.add(VTMeshes(scan_meshes['back_vertices'], scan_meshes['faces'], vertex_colors=scan_meshes['colors'],
                                 position=base + np.array([0., 2., 0.]), name='smplx_scan_back'))
        aitvs.scene.nodes[-1].backface_culling = False

    # view front and back scan labels
    if view_label:
        base = np.array([2.5, 0., 0.])
        # AIT view front scan_mesh, using label color
        aitvs.scene.add(VTMeshes(scan_meshes['vertices'], scan_meshes['faces'], vertex_colors=scan_meshes['label_colors'], 
                                 position=base, name='label_front'))
        aitvs.scene.nodes[-1].backface_culling = False
        # AIT view back scan_mesh, using label color
        aitvs.scene.add(VTMeshes(scan_meshes['back_vertices'], scan_meshes['faces'], vertex_colors=scan_meshes['label_colors'], 
                                 position=base + np.array([0., 2., 0.]), name='label_back'))
        aitvs.scene.nodes[-1].backface_culling = False

    # view clothes: shoe, hair, upper, lower, outer
    if view_cloth:
        base = np.array([4, 0., 0.])
        for k, v in cloth_meshes.items():
            if k == 'skin': continue
            if outfit == 'Outer' and k in ['upper']: continue
            if k in ['lower', 'hair']:
                position = base + np.array([1, 0., 0.])
            else:
                position = base

            # AIT view front scan_mesh, using uv map
            if not view_vertex_color:
                aitvs.scene.add(VTMeshes(cloth_meshes[k]['vertices'], cloth_meshes[k]['faces'], uv_coords=cloth_meshes[k]['uvs'], 
                                         texture_paths=cloth_meshes[k]['uv_path'], position=position, name=k))
                aitvs.scene.nodes[-1].backface_culling = False
            # AIT view front scan_mesh, using vertex color
            else:
                aitvs.scene.add(VTMeshes(cloth_meshes[k]['vertices'], cloth_meshes[k]['faces'], vertex_colors=cloth_meshes[k]['colors'], 
                                         position=position, name=k))
                aitvs.scene.nodes[-1].backface_culling = False
            
    aitvs.run()


if __name__ == "__main__":
    # set target subj_outfit_seq
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', default='00122', help='subj name')
    parser.add_argument('--outfit', default='Outer', help='outfit name')
    parser.add_argument('--seq', default='Take9', help='seq name')
    args = parser.parse_args()

    # TODO: Set 4D-DRESS DATASET_DIR in utility.py
    # load and visualize scan, smpl, smplx, label, and clothes for subj_outfit_seq 
    compact_view_subj_outfit_seq(dataset_dir=DATASET_DIR, subj=args.subj, outfit=args.outfit, seq=args.seq, n_start=0, n_stop=-1, 
                                 view_smpl=True, view_smplx=True, view_label=True, view_cloth=False,  # set view_cloth=True to extract and view clothes
                                 view_vertex_color=True)  # set view_vertex_color=False to view uv textures
