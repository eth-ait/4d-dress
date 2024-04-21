from lib.utility.general import *

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights,
    HardPhongShader,
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    TexturesVertex,
    TexturesUV,
)


class PytorchRenderer:
    """
    Pytorch3d Multi-View Image Renderer.
    """

    def __init__(self, num_view=24, img_size=512, max_x=1.0, min_x=-1.0, max_y=1.0, min_y=-1.0,
                 init_model=False, dtype=torch.float32, device=torch.device('cuda:0')):

        # init dtype, device
        self.dtype = dtype
        self.device = device
        # init num_view, img_size
        self.num_view = num_view
        self.img_size = img_size
        self.max_x, self.min_x, self.max_y, self.min_y = max_x, min_x, max_y, min_y

        # not init_model
        if not init_model: return

        # init front view with num_view = 1
        if self.num_view == 1:
            R, T = look_at_view_transform(dist=5.0, elev=0, azim=0, up=((0, 1, 0),), at=((0, 0, 0),))
        # init horizontal views with num_view = 6, 12
        elif self.num_view == 6 or self.num_view == 12:
            azim = torch.linspace(0, 360, self.num_view + 1)[:self.num_view]
            R, T = look_at_view_transform(dist=5.0, elev=0, azim=azim, up=((0, 1, 0),), at=((0, 0, 0),))
        # init horizontal, upper, and lower views with num_view = 24
        elif self.num_view == 24:
            # init horizontal views
            azim = torch.linspace(0, 360, self.num_view // 2 + 1)[:self.num_view // 2]
            R_normal, T_normal = look_at_view_transform(dist=5.0, elev=0, azim=azim, up=((0, 1, 0),), at=((0, 0, 0),))
            # init upper views
            azim = torch.linspace(0, 360, self.num_view // 4 + 1)[:self.num_view // 4]
            R_upper, T_upper = look_at_view_transform(dist=5.0, elev=30, azim=azim, up=((0, 1, 0),), at=((0, 0, 0),))
            # init lower views
            azim = torch.linspace(0, 360, self.num_view // 4 + 1)[:self.num_view // 4]
            R_lower, T_lower = look_at_view_transform(dist=5.0, elev=-30, azim=azim, up=((0, 1, 0),), at=((0, 0, 0),))
            # cat final views
            R = torch.cat([R_normal, R_upper, R_lower], dim=0)
            T = torch.cat([T_normal, T_upper, T_lower], dim=0)
        else:
            raise Exception('Invalid view number: {} over {6, 12, 24}'.format(self.num_view))

        # init Cameras
        self.cameras = FoVOrthographicCameras(max_x=self.max_x, min_x=self.min_x, max_y=self.max_y, min_y=self.min_y, R=R, T=T, device=device)
        # init PointLights
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]], ambient_color=((1, 1, 1),), diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),))
        # init HardPhongShader
        self.shader = HardPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        # init MeshRasterizer, faces_per_pixel=1
        self.mesh_raster_settings = RasterizationSettings(image_size=self.img_size, faces_per_pixel=1, bin_size=0, blur_radius=0)  # faces_per_pixel=10
        self.mesh_rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.mesh_raster_settings)
        # init MeshRenderer
        self.mesh_renderer = MeshRenderer(rasterizer=self.mesh_rasterizer, shader=self.shader)

    # get camera rasterizer for mesh(verts(N, 3) and faces(F, 3))
    def get_mesh_rasterizer(self, verts, faces):
        # init mesh from verts and faces
        mesh = Meshes(verts, faces)
        # return mesh rasterizer for num_view
        return self.mesh_rasterizer.forward(mesh.extend(self.num_view))

    # render scan_mesh to multi_view color images
    @torch.no_grad()
    def render_mesh_images(self, verts, faces, colors=None, uvs=None, uv_image=None):
        # init render_images
        render_images = dict()
        # assign mesh_color using uvs and uv_image
        if uvs is not None and uv_image is not None:
            mesh_color = Meshes(verts, faces, textures=TexturesUV(maps=uv_image, faces_uvs=faces, verts_uvs=uvs))
        # assign mesh_color using vertex colors
        elif colors is not None:
            mesh_color = Meshes(verts, faces, textures=TexturesVertex(verts_features=colors))
        else:
            raise Exception('None color input for mesh rendering!!')
        # render color and mask image with num_view
        image_color = self.mesh_renderer(mesh_color.extend(self.num_view))
        render_images['color'] = image_color[..., :3]
        render_images['mask'] = image_color[..., -1]
        return render_images
    
    # render multi_view pixel labels(nv, h, w) from vertex labels(nvt, ), faces(nft, 3), render_rasts, and render_masks(nv, h, w)
    @torch.no_grad()
    def render_mesh_labels(self, labels, faces, render_rasts, render_masks, surface_labels):
        # get dimensions
        n_views, h, w = render_masks.shape[:3]
        # init multi-view label_votes (nv, h, w, nl)
        label_votes = torch.zeros((n_views, h, w, len(surface_labels))).to(self.device)
        # render labels for all views
        for nv in range(n_views):
            # get render pix_points, pix_to_faces, and pix_bary_coords
            pix_points = torch.nonzero(render_masks[nv])
            pix_to_faces = render_rasts.pix_to_face[nv][pix_points[:, 0], pix_points[:, 1]]
            pix_bary_coords = render_rasts.bary_coords[nv][pix_points[:, 0], pix_points[:, 1]]
            # locate nearest face
            for nf in range(1):
                # get pixel_points(nvp, 2)
                pixel_points = pix_points[pix_to_faces[:, nf] >= 0]
                # get pixel_faces(nvp, ), pixel_bary_coords(nvp, 3), and pixel_faces_labels(nvp, 3)
                pixel_faces = pix_to_faces[pix_to_faces[:, nf] >= 0, nf] - faces.shape[0] * nv
                pixel_bary_coords = pix_bary_coords[pix_to_faces[:, nf] >= 0, nf]
                pixel_faces_labels = labels[faces[pixel_faces, :]]
                # assign votes to each SURFACE_LABEL: skin, hair, shoe, upper, lower, outer
                for nl in range(len(surface_labels)):
                    # loop over all faces
                    for n in range(pixel_faces_labels.shape[-1]):
                        nl_labels = pixel_faces_labels[:, n] == nl
                        if torch.count_nonzero(nl_labels) == 0: continue
                        label_votes[nv, pixel_points[nl_labels][:, 0], pixel_points[nl_labels][:, 1], nl] += 1 * pixel_bary_coords[nl_labels, n]
        # collect render_labels from label_votes, filter label without votes
        render_labels = torch.max(label_votes, dim=-1).indices
        render_labels[torch.sum(label_votes, dim=-1) == 0] = -1
        return render_labels, label_votes

    # render multi_view labels (nv, h, w) to color_images (nv, h, w, 3)
    def render_label_images(self, labels, surface_labels, surface_labels_colors):
        # init parser_images with white background
        images = np.ones((labels.shape[0], labels.shape[1], labels.shape[2], 3)) * 255
        # loop over all parser images
        for nv in range(images.shape[0]):
            for nl in range(len(surface_labels)):
                images[nv][labels[nv] == nl] = surface_labels_colors[nl]
        return images.astype(np.uint8)
    

    # pack multi_view images (nv, h, w, 3) to (h * n_rows, w * n_cols, 3)
    def pack_multi_view_images(self, images):
        # get n_rows and n_cols
        n_rows, n_cols = images.shape[0] // 6, 6
        assert images.shape[0] == n_rows * n_cols, 'Wrong image number: {} over num_view={}'.format(images.shape[0], self.num_view)
        # get image size
        n_h, n_w = images.shape[1], images.shape[2]
        # pack multi-view images to n_rows
        imgs = np.concatenate(images, axis=1)
        imgs = np.concatenate([imgs[:, n_w * n_cols * nr:n_w * n_cols * (nr + 1)] for nr in range(n_rows)], axis=0)
        return imgs

    # load and preprocess multi_view images to (nv, h, w, 3)
    def load_multi_view_images(self, image_fn):
        # load multi-view images
        images = load_image(image_fn)
        # get n_rows, n_cols
        n_rows, n_cols = images.shape[0] // self.img_size, images.shape[1] // self.img_size
        # get image size
        n_h, n_w = images.shape[0] // n_rows, images.shape[1] // n_cols
        assert n_h == self.img_size, 'Wrong image size: {}/{} over img_size={}'.format(n_h, n_w, self.img_size)
        # cat row images to one column
        images = np.concatenate([images[n_h * nr:n_h * (nr + 1), :] for nr in range(n_rows)], axis=1)
        # divide column images
        images = np.asarray([images[:, n_w * nc:n_w * (nc + 1)] for nc in range(n_rows * n_cols)])
        return images

