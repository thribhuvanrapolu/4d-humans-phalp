import torch


# CFG values
from easydict import EasyDict
import yaml
yaml_file = 'config.yaml'
# Load YAML file
with open(yaml_file, 'r') as f:
    cfg = EasyDict(yaml.safe_load(f))


# from 4D-Humans/hmr2/utils/geometry.py
from typing import Optional
import torch
from torch.nn import functional as F
def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: Optional[torch.Tensor] = None,
                           rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]



# Instantiate SMPL model
from models.smpl_wrapper import SMPL
smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
smpl = SMPL(**smpl_cfg)


import joblib
pkl_file = joblib.load("results/demo_courtyard_basketball_00.pkl")


data={}
for frame in pkl_file.keys():
    data[frame]={
                    'track_ids'       : pkl_file[frame]['extra_data'],
                    # 'prediction_uv'   : pkl_file[frame]['prediction_uv'],
                    'prediction_pose' : pkl_file[frame]['prediction_pose'],
                    # 'prediction_loca' : pkl_file[frame]['prediction_loca'],
                }



# iterate through each frame
for frame in data.keys():
    batch_size=len(data[frame]['track_ids'])
    
    prediction_pose=data[frame]['prediction_pose']
    
    
    pred_smpl_params =  {
                            'global_orient' : [],
                            'body_pose'     : [],
                            'betas'         : []
                        }
    pred_cam=[]
    
    for per_pose in prediction_pose:
        pred_smpl_params['global_orient'].append(per_pose[0:9])
        pred_smpl_params['body_pose'].append(per_pose[9:216])
        pred_smpl_params['betas'].append(per_pose[216:226])
        pred_cam.append(per_pose[226:229])

    # Store useful regression outputs to the output dict
    output = {}
    output['pred_cam'] = pred_cam
    output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}

    
    # Compute camera translation
    device = pred_smpl_params['body_pose'].device
    dtype = pred_smpl_params['body_pose'].dtype
    focal_length = cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
    pred_cam_t = torch.stack([pred_cam[:, 1],
                              pred_cam[:, 2],
                              2*focal_length[:, 0]/(cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
    
    output['pred_cam_t'] = pred_cam_t
    output['focal_length'] = focal_length

    # Compute model vertices, joints and the projected joints
    pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
    pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
    pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
    
    smpl_output = smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
    
    
    pred_keypoints_3d = smpl_output.joints
    pred_vertices = smpl_output.vertices
    output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
    output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
    pred_cam_t = pred_cam_t.reshape(-1, 3)
    focal_length = focal_length.reshape(-1, 2)
    
    pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                               translation=pred_cam_t,
                                               focal_length=focal_length / cfg.MODEL.IMAGE_SIZE)
    output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)

    
    
    