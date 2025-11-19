import nvdiffrast.torch as dr
import torch
from torch.nn.functional import normalize

import nvdiffrast
import torch
# import kaolin as kal

def interpolate_attributes(rast_out, rast_out_db, mesh):
    normals = nvdiffrast.torch.interpolate(
        mesh.get_or_compute_attribute('vertex_normals', should_cache=False),
        rast_out,
        mesh.faces.int(),
        rast_db=rast_out_db,
        diff_attrs='all',
    )[0]
    tangents = nvdiffrast.torch.interpolate(
        mesh.get_or_compute_attribute('vertex_tangents', should_cache=False),
        rast_out,
        mesh.faces.int(),
        rast_db=rast_out_db,
        diff_attrs='all',
    )[0]
    bitangents = torch.nn.functional.normalize(torch.cross(tangents, normals, dim=-1), dim=-1)
    # get uvs
    texc, texd = nvdiffrast.torch.interpolate(
        mesh.uvs,
        rast_out,
        mesh.face_uvs_idx.int(),
        rast_db=rast_out_db,
        diff_attrs='all',
    )
    return normals, tangents, bitangents, texc, texd

def render(mesh, camera, light, random_background=True, val_background=False):
    # transform mesh
    vertices_camera = camera.extrinsics.transform(mesh.vertices)
    vertices_clip = camera.intrinsics.project(vertices_camera)
    faces_int = mesh.faces.int()
    # rasterize
    glctx = dr.RasterizeCudaContext()

    rast_out, rast_out_db = nvdiffrast.torch.rasterize(
        glctx,
        vertices_clip,
        faces_int,
        (camera.height, camera.width),
    )
    rast_out = torch.flip(rast_out, dims=(1,))
    rast_out_db = torch.flip(rast_out_db, dims=(1,))
    mask = torch.clamp(rast_out[..., -1:], 0, 1)
    # interpolate normals, tangents & bitangents
    normals, tangents, bitangents, texc, texd = interpolate_attributes(
        rast_out,
        rast_out_db,
        mesh,
    )
    # texturing
    material = mesh.materials[0]
    def _proc_channel(texture_image):
        if texture_image is None:
            return None
        return nvdiffrast.torch.texture(
            texture_image[None, ...],
            texc,
            texd,
            filter_mode='linear-mipmap-linear',
            # filter_mode='linear', #'linear' 'nearest'
            max_mip_level=9
        )
    mapped_albedo = _proc_channel(material.diffuse_texture)
    mapped_normal = _proc_channel(material.normals_texture)
    mapped_metallic = _proc_channel(material.metallic_texture)
    mapped_roughness = _proc_channel(material.roughness_texture)
    # shading
    if mapped_normal is not None:
        shading_normals = torch.nn.functional.normalize(
            tangents * mapped_normal[..., :1]
            - bitangents * mapped_normal[..., 1:2]
            + normals * mapped_normal[..., 2:3],
            dim=-1,
        )
    else:
        shading_normals = normals
    diffuse_light = light(shading_normals)

    if mapped_metallic is not None and mapped_roughness is not None:
        viewdirs = -get_ray_dirs(camera)
        n_dot_v = (shading_normals * viewdirs).sum(-1, keepdim=True)
        reflective = n_dot_v * shading_normals * 2 - viewdirs

        roughness = torch.clamp(mapped_roughness, min=1e-3)
        specular_light = light(reflective, roughness)
    
        diffuse_albedo = (1 - mapped_metallic) * mapped_albedo
        fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1)
        fg = dr.texture(
            mesh.materials[0].FG_LUT,
            fg_uv.reshape(1, -1, 1, 2).contiguous(),
            filter_mode='linear',
            boundary_mode='clamp',
            ).reshape(*roughness.shape[:-1], 2)
        F0 = (1. - mapped_metallic) * 0.04 + mapped_metallic * mapped_albedo
        specular_albedo = F0 * fg[..., 0:1] + fg[..., 1:2]
        output_image = diffuse_light * diffuse_albedo + specular_light * specular_albedo
    else:
        diffuse_albedo = mapped_albedo
        output_image = diffuse_light * diffuse_albedo

    if random_background:
        # TODO: come up with a smarter background color distrbution?
        background_color = torch.rand(3, device='cuda:0')
    else:
        # mishan: change for better view
        background_color = torch.tensor([0.05, 0.05, 0.05], device='cuda:0')
    
    if val_background:
       background_color = torch.tensor([0.99, 0.99, 0.99], device='cuda:0') 

    output_image = torch.where(mask == 1, output_image, background_color)
    return output_image

def get_ray_dirs(camera):
    num_cameras = len(camera)
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(camera.height, device='cuda'),
        torch.arange(camera.width, device='cuda'),
    )
    pixel_x = pixel_x + 0.5
    pixel_x = pixel_x.unsqueeze(0) - camera.x0.view(-1, 1, 1)
    pixel_x = 2 * (pixel_x / camera.width) - 1.0

    pixel_y = pixel_y + 0.5
    pixel_y = pixel_y.unsqueeze(0) - camera.y0.view(-1, 1, 1)
    pixel_y = 2 * (pixel_y / camera.height) - 1.0

    ray_dir = torch.stack((pixel_x * camera.tan_half_fov.view(-1, 1, 1),
                          -pixel_y * camera.tan_half_fov.view(-1, 1, 1),
                          -torch.ones_like(pixel_x)), dim=-1)
    #ray_dir = torch.stack((pixel_x * camera.tan_half_fov(kal.render.camera.intrinsics.CameraFOV.HORIZONTAL).view(-1, 1, 1),
    #                      -pixel_y * camera.tan_half_fov(kal.render.camera.intrinsics.CameraFOV.VERTICAL).view(-1, 1, 1),
    #                      -torch.ones_like(pixel_x)), dim=-1)
    ray_dir = ray_dir.reshape(num_cameras, -1, 3)  # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)
    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir = torch.nn.functional.normalize(ray_dir, dim=-1)
    ray_dir = ray_dir.reshape(-1, camera.height, camera.width, 3)
    return ray_dir

def run_easy_render_with_default_lighting(mesh, cameras, light=None, random_background=True):
    output_images = []
    for camera in cameras:
        # Seems weird, I know. Renderer needs this to keep normal maps consistent between different runs.
        if mesh.has_attribute('face_normals'):
            delattr(mesh, 'face_normals')
        render_output = kal.render.easy_render.render_mesh(camera, mesh)
        mask = (render_output['face_idx'] >= 0).unsqueeze(-1)
        if random_background:
            # TODO: come up with a smarter background color distrbution?
            background_color = torch.rand(3).cuda(torch.device('cuda:0'))
        else:
            background_color = torch.Tensor([0.99, 0.99, 0.99]).cuda(torch.device('cuda:0'))
        output_image = torch.where(mask == 1, render_output['render'], background_color)
        output_images.append(output_image)
    output_images = torch.cat(output_images, 0)
    return output_images

if __name__ == '__main__':
    import envlight
    from torch.nn import Parameter, ParameterDict
    from mesh import load_mesh
    from camera import get_camera

    mesh_location = "data/model_cow.obj"

    texture_dim = 1024
    normals_texture = torch.zeros((texture_dim, texture_dim, 3)).cuda()
    normals_texture[..., 2:3] = 1.
    pbr_material_parameters = ParameterDict({
        'normals_texture' : Parameter(normals_texture),
        'diffuse_texture' : Parameter(torch.full((texture_dim, texture_dim, 3), 0.5).cuda()),
        'metallic_texture': Parameter(torch.full((texture_dim, texture_dim, 1), 0.5).cuda()),
        'roughness_texture': Parameter(torch.full((texture_dim, texture_dim, 1), 0.5).cuda()),
    })
    mesh = load_mesh(mesh_location, pbr_material_parameters)
    light = envlight.EnvLight(
        'assets/aerodynamics_workshop_2k.hdr',
        device='cuda',
        scale=2.0
    )

    camera_target = torch.tensor([0., 0., 0.])
    r = torch.tensor(2.2)
    phi = torch.arange(0, 2 * torch.pi, torch.pi / 4).view(1, -1)
    theta = torch.zeros(1, 1)

    val_eye = r * torch.stack([
        phi.cos() * theta.cos(),
        torch.ones_like(phi) * theta.sin(),
        phi.sin() * theta.cos()
    ], dim=-1).view(-1, 3)
    camera = get_camera(len(val_eye), camera_target, r, val_eye)
    # try rendering
    render(mesh, camera, light)
    run_easy_render_with_default_lighting(mesh, camera, None, False)
