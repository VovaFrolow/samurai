# import FastGeodis
from typing import Tuple, Optional, List, Dict, Any, Literal

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torch.nn.modules.module import T
from torchvision.transforms import Normalize, Resize
from ultralytics.models.sam import SAM2Predictor
from torchvision.ops.boxes import batched_nms, box_area
from ultralytics.models.sam.build import build_sam
from ultralytics.utils.ops import scale_masks
from copy import deepcopy
from functools import partial
from scipy.optimize import linear_sum_assignment

from ocl.utils import config_as_kwargs, make_build_fn

# TypeAlias = Any
# SAM1ModelName: TypeAlias = Literal['sam_h.pt', 'sam_l.pt', 'sam_b.pt']
# SAM2ModelName: TypeAlias = Literal[
#     'sam2_t.pt', 'sam2_s.pt', 'sam2_b.pt', 'sam2_l.pt', 'sam2.1_t.pt', 'sam2.1_s.pt', 'sam2.1_b.pt', 'sam2.1_l.pt', "vit_t"]


# SAMModelName: TypeAlias = SAM1ModelName | SAM2ModelName


# SAMResizeMode: TypeAlias = Literal['resize', 'pad']

SAM_IMAGE_SIZE = (1024, 1024)
OUT_SIZE = (256, 256)
SAM_MEAN = (123.675 / 255, 116.28 / 255, 103.53 / 255)
SAM_STD = (58.395 / 255, 57.12 / 255, 57.375 / 255)

SAM_1_MODELS = ('sam_h.pt', 'sam_l.pt', 'sam_b.pt', 'mobile_sam.pt')
SAM_2_MODELS = (
    'sam2_t.pt', 'sam2_s.pt', 'sam2_b.pt', 'sam2_l.pt', 'sam2.1_t.pt', 'sam2.1_s.pt', 'sam2.1_b.pt', 'sam2.1_l.pt')

@make_build_fn(__name__, "sam")
def build(config, name: str):
    if name == "SAM":
        return SAMImageMaskImprover(
            **config_as_kwargs(config),
        )
    else:
        return None

class PadResizer2d:
    def __init__(self, size: Tuple[int, int]):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, inp: torch.Tensor):
        x_pad = inp.size(-2) - self.size[0]
        y_pad = inp.size(-1) - self.size[1]
        return F.pad(inp, (0, y_pad, 0, x_pad))


class SAMImageMaskImprover(nn.Module):

    def __init__(self,
                 sam_model_name: str,
                 use_box: bool = True,
                 use_mask: bool = True,
                 use_point: bool = True,
                 refine_iters: int = 5,
                 in_mask_size: int = 28,
                 in_image_size: int = 224,
                 resize_mode: str = "resize",
                 threshold_for_points: Optional[float] = None,
                 ):

        super().__init__()
        self.sam_size = OUT_SIZE
        self.in_image_size = in_image_size
        self.sam_model_name = sam_model_name
        self.model = build_sam(sam_model_name)
        self.model.eval()

        assert resize_mode == "resize", "Only resize is supported for now"
        assert sam_model_name != "mobile_sam.pt", "Mobile sam is not supported for now"

        self.model.set_imgsz(SAM_IMAGE_SIZE)
        self.get_im_features = self._resolve_im_features_getter()
        self.threshold_for_points = threshold_for_points
        self.use_box = use_box
        self.use_mask = use_mask
        self.use_point = use_point
        self.in_mask_size = in_mask_size
        self.refine_iters = refine_iters

        self.resize_mode = resize_mode

        self.norm = Normalize(SAM_MEAN, SAM_STD)

        self.mask_getter = self._resolve_mask_getter()

        self.feats_getter = self._resolve_im_features_getter()

        self.im_resizer = self._resolve_im_resizer()

        self.is_sam2 = sam_model_name in SAM_2_MODELS

        self._bb_feat_sizes = [[x // (4 * i) for x in SAM_IMAGE_SIZE] for i in [1, 2, 4]]

        self.requires_grad_(False)

    def _resolve_im_resizer(self):
        if self.resize_mode == "resize":
            return Resize(SAM_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC)
        elif self.resize_mode == "pad":
            return PadResizer2d(SAM_IMAGE_SIZE)

    def _resolve_im_features_getter(self):
        if self.sam_model_name in SAM_1_MODELS:
            return self._get_im_features_sam1
        elif self.sam_model_name in SAM_2_MODELS:
            return self._get_im_features_sam2
        else:
            raise ValueError(f"Unsupported SAM model: {self.sam_model_name}")

    def _resolve_mask_getter(self):
        if self.sam_model_name in SAM_1_MODELS:
            return self._get_masks_sam1
        elif self.sam_model_name in SAM_2_MODELS:
            return self._get_masks_sam2
        else:
            raise ValueError(f"Unsupported SAM model: {self.sam_model_name}")

    def train(self, mode: bool = True):
        return super().train(False)

    def _preprocess_mask(self, mask: torch.Tensor, size: Tuple[int, int]):
        mask = mask.clone()
        mask[mask < torch.quantile(mask, 0.98)] = 0.0
        mask = F.interpolate(mask, size=size, mode='nearest-exact')

        return mask

    def _preprocess_image(self, im: torch.Tensor):
        im = self.im_resizer(im)
        # print(im.min(), im.max())
        im = self.norm(im)

        return im

    def extract_bboxes(self, masks: torch.Tensor, threshold: Optional[float] = None):
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        num_instances = masks.size(1)
        if threshold is None:
            masks = (masks.flatten(0, 1) > torch.quantile(masks, 0.85))
        else:
            masks = (masks.flatten(0, 1) > threshold)

        n = masks.shape[0]

        bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            y, x = torch.where(mask != 0)
            if y.numel() == 0 or x.numel() == 0:
                continue

            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)

        return bounding_boxes.unflatten(0, (-1, num_instances))
    
    def get_reference_points(self, masks: torch.Tensor, num_points: int = 5, threshold: Optional[float] = None):
        batch_size, num_masks, h, w = masks.shape
        input_points = torch.zeros(batch_size, num_masks, num_points, 2, device=masks.device)
        input_labels = torch.zeros(batch_size, num_masks, num_points, device=masks.device)
        for b in range(batch_size):
            for m in range(num_masks):
                current_mask = masks[b, m]  # (height, width)
                
                upper_threshold = current_mask.max()
                flattened = current_mask.flatten()
                # sorted_values, _ = torch.sort(flattened)
                # k = int(0.99 * sorted_values.shape[0])
                if threshold is None:
                    lower_threshold = torch.quantile(flattened, 0.99) #sorted_values[k]
                else:
                    lower_threshold = threshold
                
                condition = (current_mask <= upper_threshold) & (current_mask >= lower_threshold)
                cond_coords = condition.nonzero(as_tuple=True)
                    
                values = current_mask[cond_coords]
                mean = torch.mean(values)
                std = torch.std(values)
                
                lower_scale, upper_scale = 1.0, 3.0
                upper_bound = mean + std * upper_scale
                lower_bound = mean - std * lower_scale
                
                condition = (current_mask <= upper_bound) & (current_mask >= lower_bound)
                coords = condition.nonzero(as_tuple=False)  
                coords = torch.stack([coords[:, 1], coords[:, 0]], dim=1)
                if len(coords) == 0:
                    continue
                
                if num_points > len(coords):
                    current_num_points = len(coords)
                else:
                    current_num_points = num_points
                random_idxs = torch.randperm(len(coords))[:current_num_points]
                # if torch.rand(1).item() > 0.5:
                #     random_idxs = torch.randperm(len(coords))[:current_num_points]
                # else:
                #     random_idxs = torch.randint(0, len(coords), (current_num_points,))
                input_points[b, m, :current_num_points] = coords[random_idxs].int()#.cpu().numpy()
                input_labels[b, m, :current_num_points] = torch.ones(current_num_points, dtype=torch.int32, device=masks.device)
        # input_labels = torch.ones((batch_size * num_masks, num_points), dtype=torch.int32, device=masks.device)#.cpu().numpy()
        
        return input_points.flatten(0, 1), input_labels.flatten(0, 1)

    # def _sam2_refine_prompts(self, bboxes: Optional[torch.Tensor] = None,
    #                          points: Optional[torch.Tensor] = None,
    #                          masks: Optional[torch.Tensor] = None):
    #     if bboxes is not None:
    #         bboxes = bboxes.view(-1, -1, 2, 2)
    #         bbox_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=bboxes.device).expand(bboxes.size(0), -1)
    #         # NOTE: merge "boxes" and "points" into a single "points" input
    #         # (where boxes are added at the beginning) to model.sam_prompt_encoder
    #         if points is not None:
    #             points = torch.cat([bboxes, points], dim=1)
    #             labels = torch.cat([bbox_labels, labels], dim=1)
    #         else:
    #             points, labels = bboxes, bbox_labels
    #
    #     return bboxes, points, masks

    def _get_im_features_sam1(self, im: torch.Tensor) -> torch.Tensor:
        """Extracts image features using the SAM model's image encoder for subsequent mask prediction."""
        return self.model.image_encoder(im)

    def _get_im_features_sam2(self, im: torch.Tensor) -> torch.Tensor:
        """Extracts image features from the SAM image encoder for subsequent processing."""
        backbone_out = self.model.forward_image(im)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
                    feat.permute(1, 2, 0).view(im.size(0), -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
                ][::-1]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

    def _sample_rand_points(self, mask: torch.Tensor):
        # mask = mask.flatten(1)
        # coords = torch.where(mask == 1)
        # print(coords)
        points = torch.tensor([SAM_IMAGE_SIZE[0] // 2, SAM_IMAGE_SIZE[1] // 2], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0).expand(mask.size(0) * mask.size(1), -1, -1)
        labels = torch.ones((points.shape[:-1]), dtype=torch.int32)
        return points, labels

    # def _prepare_prompts(self):
    #
    #     src_shape = self.batch[1][0].shape[:2]
    #     r = 1.0 if self.segment_all else min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
    #     # Transform input prompts
    #     if points is not None:
    #         points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
    #         points = points[None] if points.ndim == 1 else points
    #         # Assuming labels are all positive if users don't pass labels.
    #         if labels is None:
    #             labels = np.ones(points.shape[:-1])
    #         labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
    #         assert points.shape[-2] == labels.shape[-1], (
    #             f"Number of points {points.shape[-2]} should match number of labels {labels.shape[-1]}."
    #         )
    #         points *= r
    #         if points.ndim == 2:
    #             # (N, 2) --> (N, 1, 2), (N, ) --> (N, 1)
    #             points, labels = points[:, None, :], labels[:, None]
    #     if bboxes is not None:
    #         bboxes = torch.as_tensor(bboxes, dtype=torch.float32, device=self.device)
    #         bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
    #         bboxes *= r
    #     if masks is not None:
    #         masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device).unsqueeze(1)
    #     return bboxes, points, labels, masks

    def prepare_prompt(self, masks: torch.Tensor, threshold_for_boxes: Optional[float] = None, threshold_for_points: Optional[float] = None):
        bboxes = None
        points = None
        g_masks = None

        if self.use_box:
            # print("use_bboxes")
            bboxes = self.extract_bboxes(masks, threshold=threshold_for_boxes).flatten(0, 1).unsqueeze(1)

        if self.use_point:
            # print("use_points")
            points = self.get_reference_points(masks, threshold=threshold_for_points)

        if self.use_mask:
            # print("use_masks")
            g_mask = self.extract_gauss_mask()

        # if self.is_sam2:
        #     bboxes, points, g_masks = self._sam2_refine_prompts(bboxes, points, g_masks)

        return bboxes, points, g_masks

    def _get_masks_sam1(self,
                        features: torch.Tensor,
                        num_instances: int,
                        bboxes: Optional[torch.Tensor] = None,
                        points: Optional[torch.Tensor] = None,
                        masks: Optional[torch.Tensor] = None,
                        multimask: Optional[bool] = False):

        features = features.unsqueeze(1).expand(-1, num_instances, -1, -1, -1).flatten(0, 1)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, boxes=bboxes, masks=masks)
        pred_masks, pred_scores = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
        )
        return pred_masks, pred_scores

    def _get_masks_sam2(self,
                        features: torch.Tensor,
                        num_instances: int,
                        bboxes: Optional[torch.Tensor] = None,
                        points: Optional[torch.Tensor] = None,
                        masks: Optional[torch.Tensor] = None,
                        # ref_masks: Optional[torch.Tensor] = None,
                        multimask: Optional[bool] = False):
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=points,
            boxes=bboxes,
            masks=masks,
        )
        # Predict masks
        batched_mode = points is not None and points[0].shape[0] > 1  # multi object prediction
        high_res_features = [feat_level.unsqueeze(1).expand(-1, num_instances, -1, -1, -1).flatten(0, 1) for feat_level
                             in features["high_res_feats"]]
        pred_masks, pred_scores, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features["image_embed"].unsqueeze(1).expand(-1, num_instances, -1, -1, -1).flatten(0, 1),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return pred_masks, pred_scores

    def post_process_mask(self, pred_masks: torch.Tensor, ref_masks: Optional[torch.Tensor] = None):

        pred_masks = scale_masks(pred_masks, (self.in_mask_size, self.in_mask_size), padding=False)

        pred_masks = torch.where(pred_masks > self.model.mask_threshold, 1, 0)

        # if ref_masks is not None:
        #     pred_masks = (pred_masks & (ref_masks.flatten(0, 1).unsqueeze(1) > 0.6))
        
        return pred_masks
    
    def get_best_mask(masks_norm, original_mask, threshold=0.6):
        ious = []
        best_iou = 0
        best_mask = np.zeros_like(msks[0])
        best_norm_mask = np.zeros_like(msks[0])
        for msk in msks:
            mask_norm = self.normalize_to_range(msk, min_val=0., max_val=1.)
            intersection = ((original_mask > threshold) & (msk > self.model.mask_threshold)).sum() # eq: np.logical_and((mask > 0.6), (mask_norm > 0.5)).sum()
            union = ((original_mask > threshold) | (msk > self.model.mask_threshold)).sum() # eq: np.logical_or((mask > 0.6), (mask_norm > 0.5)).sum()
            iou = intersection / union
            ious.append(iou)
            if best_iou < iou:
                best_iou = iou
                best_mask = msk
                best_norm_mask = mask_norm

        return best_mask, best_norm_mask
    
    def get_best_masks(self, sam_masks, ref_masks, num_slots=7):
        # print(sam_masks.shape)
        B, M, I, H, W = sam_masks.unflatten(0, (-1, num_slots)).shape
        best_masks = torch.zeros((B, M, 1, H, W), device=sam_masks.device)
        for b in range(B):
            for m in range(M):
                ref_mask = ref_masks[b, m]
                binary_ref_mask = (ref_mask > 0.5).float() # .cpu().numpy()
                if binary_ref_mask.sum() == 0:
                    continue

                # Выбираем маску с максимальным IoU к ref_mask
                best_iou = -1
                best_mask = None
                for idx in range(I):  # обычно 3 маски
                    sam_mask = sam_masks[b, m, idx] # > self.model.mask_threshold
                    intersection = torch.logical_and(sam_mask, binary_ref_mask.bool()).sum()
                    union = torch.logical_or(sam_mask, binary_ref_mask.bool()).sum()
                    iou = intersection / (union + 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = sam_mask
                
                best_masks[b, m, 0, ...] = best_mask
        
        return best_masks
    
    def get_best_pseudo_gt(self, sam_masks, ref_masks, scores, iou_threshold=0.75, coverage_threshold=0.95, min_mask_size=100, num_slots=7):
        # sam_masks = scale_masks(sam_masks, (self.in_mask_size, self.in_mask_size), padding=False)
        sam_masks = sam_masks.unflatten(0, (-1, num_slots))
        sam_masks_binary = torch.where(sam_masks > self.model.mask_threshold, True, False)
        # sam_masks_binary = sam_masks_binary.unflatten(0, (-1, num_slots))
        scores = scores.unflatten(0, (-1, num_slots)).argmax(2)
        # ref_masks = (ref_masks > torch.quantile(ref_masks, 0.99)).float()
        pseudo_gt_list = []
        logits_list = []
        B, M, I, H, W = sam_masks_binary.shape
        for b in range(B):
            covered_mask = torch.zeros((H, W), dtype=torch.bool, device=sam_masks_binary.device)
            final_masks = torch.zeros((M, H, W), dtype=torch.bool, device=sam_masks_binary.device)
            final_masks_logits = torch.zeros((M, H, W), dtype=sam_masks.dtype, device=sam_masks_binary.device)
            final_masks_list = []
            # final_masks_logits_list = []

            for m in range(M):
                ref_mask = ref_masks[b, m]
                # best_mask = sam_masks[b, m]
                binary_ref_mask = (ref_mask > torch.quantile(ref_mask, 0.98))#.float() # .cpu().numpy()
                if binary_ref_mask.sum() == 0 and m != M - 1:
                    continue

                best_iou = -1
                best_mask = None
                for idx in range(I):
                    sam_mask = sam_masks_binary[b, m, idx] # > self.model.mask_threshold
                    intersection = torch.logical_and(sam_mask, binary_ref_mask).sum()
                    union = torch.logical_or(sam_mask, binary_ref_mask).sum()
                    iou = intersection / (union + 1e-6)

                    if iou > best_iou:
                        best_iou = iou
                        best_mask = sam_mask
                best_mask_no_overlap = best_mask & (~covered_mask)
                if best_mask_no_overlap.sum() < min_mask_size:
                    continue

                is_new = True
                for existing_mask in final_masks_list:
                    inter = torch.logical_and(best_mask_no_overlap, existing_mask).sum()
                    union = torch.logical_or(best_mask_no_overlap, existing_mask).sum()
                    iou = inter / (union + 1e-6)
                    if iou > iou_threshold:
                        is_new = False
                        break

                if is_new:
                    final_masks_logits[m] = sam_masks[b, m, scores[b, m]]
                    final_masks[m] = best_mask_no_overlap
                    # final_masks_logits_list.append(sam_masks[b, m, scores[b, m]])
                    final_masks_list.append(best_mask_no_overlap)
                    covered_mask |= best_mask_no_overlap
                elif not is_new and m != M - 1:
                    continue
                if not is_new and m == M - 1:
                    # print("final_mask")
                    final_masks[m] = ~covered_mask
                    uncovered_elements = sam_masks[b, m, scores[b, m]][~covered_mask]
                    final_masks_logits[m][~covered_mask] = uncovered_elements
                    break
                coverage = covered_mask.float().mean().item()
                if not is_new and coverage >= coverage_threshold:
                    # print(f"[Batch {b}] Достигнут порог покрытия: {coverage:.2f}")
                    final_masks[m] = ~covered_mask
                    uncovered_elements = sam_masks[b, m, scores[b, m]][~covered_mask]
                    final_masks_logits[m][~covered_mask] = uncovered_elements
                # final_masks[m] = best_mask


            # pseudo_gt_masks = torch.stack(final_masks) if final_masks else \
            # torch.zeros((0, H, W)).to(torch.bool) # if final_masks else torch.zeros((0, H, W))
            # pseudo_gt_list.append(final_masks)
            logits_list.append(final_masks_logits)
        # pseudo_gt = torch.stack(pseudo_gt_list).to(torch.int32)
        logits = torch.stack(logits_list)
        # pseudo_gt = torch.where(logits > self.model.mask_threshold, True, False).float()
        # pseudo_gt = F.interpolate(
        #     pseudo_gt, size=(self.in_mask_size, self.in_mask_size), 
        #     mode='nearest-exact',
        # )
        norm_logits = self.normalize_to_range(logits, min_val=0., max_val=1.)
        # norm_logits = scale_masks(logits, (self.in_mask_size, self.in_mask_size), padding=False)

        return logits, norm_logits

    @torch.no_grad()
    def improve_masks(self, image: torch.Tensor, ref_masks: torch.Tensor):
        image = self._preprocess_image(image)
        num_instances = ref_masks.size(1)
        masks = self._preprocess_mask(ref_masks, OUT_SIZE)
        # print(ref_masks.shape)
        sam_ref_masks = self._preprocess_mask(ref_masks, SAM_IMAGE_SIZE)

        # bboxes, points, _ = self.prepare_prompt(masks)

        # self.model.f
        features = self.get_im_features(image)
        pred_masks = None
        bboxes, points, _ = None, None, None
        for i in range(self.refine_iters):
            # pred_masks, scores = self.mask_getter(features, num_instances, bboxes, points, pred_masks, 
            #                                       multimask=False)
            # pred_masks = self.get_best_masks(pred_masks, ref_masks)
            if points is None:
                bboxes, points, _ = self.prepare_prompt(sam_ref_masks, threshold_for_points=self.threshold_for_points)
            else:
                logits[logits < self.model.mask_threshold] = 0.
                # print(logits.shape)
                sam_ref_masks = self._preprocess_mask(ref_masks, SAM_IMAGE_SIZE)
                # sam_ref_masks = scale_masks(
                #     logits.permute(1, 0, 2, 3), # self.normalize_to_range(logits.permute(1, 0, 2, 3), min_val=0., max_val=1.), 
                #     SAM_IMAGE_SIZE, padding=False
                # )
                bboxes, points, _ = self.prepare_prompt(sam_ref_masks, threshold_for_points=self.model.mask_threshold)

            logits, scores = self.mask_getter(features, num_instances, bboxes, points, None, 
                                                  multimask=True if i == self.refine_iters - 1 else False)
            # points = self.get_reference_points(pred_masks.squeeze(1).unflatten(0, (-1, num_instances)).to(torch.float32))
        
        logits, norm_logits = self.get_best_pseudo_gt(logits, masks, num_slots=num_instances, scores=scores)
        # print(logits.shape, norm_logits.shape)
        # logits = logits.permute(1, 0, 2, 3).squeeze(1).reshape(
        #     ref_masks.shape[0], num_instances, self.sam_size[0], self.sam_size[1],
        # )
        # norm_logits = self.normalize_to_range(
        #     logits, min_val=0., max_val=1.
        # ).squeeze(1).reshape(ref_masks.shape[0], num_instances, self.sam_size[0], self.sam_size[1])
        # print(logits.shape, norm_logits.shape)
        # hard_masks = torch.where(
        #     pred_masks > self.model.mask_threshold, True, False
        # ).float().squeeze(1).unflatten(0, (-1, num_instances))
        # hard_masks = F.interpolate(
        #     hard_masks, size=(self.in_image_size, self.in_image_size), 
        #     mode='nearest-exact',
        # )
        # logits = self.normalize_to_range(
        #     pred_masks, min_val=0., max_val=1.
        # ).squeeze(1).unflatten(0, (-1, num_instances))
        # logits = scale_masks(logits, (self.in_image_size, self.in_image_size), padding=False)
        # print(pred_masks.shape, logits.shape) #hard_masks.shape, logits.shape)

        # print(pred_masks.shape)
        # pred_masks = self.post_process_mask(pred_masks, ref_masks)
        # pred_masks = self.post_process_mask(best_masks, ref_masks)

        return {
            "sam_masks": logits, #.flatten(start_dim=2, end_dim=3),
            "norm_masks": norm_logits #.flatten(start_dim=2, end_dim=3),
            # "sam_masks_hard": hard_masks.flatten(start_dim=2, end_dim=3), #.squeeze(1).unflatten(0, (-1, num_instances))
            # "sam_masks_vis": hard_masks.flatten(start_dim=2, end_dim=3),
            # F.interpolate(
            #     hard_masks, size=(self.in_image_size, self.in_image_size), 
            #     mode='nearest-exact',
            # ),
        }
    
    def normalize_to_range(self, data, min_val=-14, max_val=6):
        """
        Нормализует входные данные в диапазон [min_val, max_val] для тензоров PyTorch

        Параметры:
        - data: входной тензор PyTorch
        - min_val: минимальное значение целевого диапазона
        - max_val: максимальное значение целевого диапазона

        Возвращает нормализованный тензор
        """
        if data is None:
            return None
        
        B, N = data.shape[:2]
        
        # Находим текущие минимум и максимум по пространственным измерениям (H, W)
        current_min = data.view(B, N, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        current_max = data.view(B, N, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        
        # Создаем маску для случаев, где min == max (чтобы избежать деления на ноль)
        mask = (current_min != current_max).float()
        
        # Нормализуем данные
        normalized = (data - current_min) / (current_max - current_min + 1e-9)  # добавляем небольшое значение для стабильности
        normalized = normalized * (max_val - min_val) + min_val
        
        # Для элементов, где min == max, устанавливаем среднее значение диапазона
        # normalized = normalized * mask + (1 - mask) * ((min_val + max_val) / 2)
        # normalized = normalized * mask + (1 - mask) * 0.0
        
        return normalized

    def apply_coords(self, coords: torch.tensor, original_size: Tuple[int, ...]) -> torch.tensor:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], SAM_IMAGE_SIZE[0]
        )
        coords = deepcopy(coords).to(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

# class SAMImageMaskImprover(nn.Module):

#     def __init__(self,
#                  sam_model_name: str,
#                  use_box: bool = True,
#                  use_mask: bool = True,
#                  use_point: bool = True,
#                  refine_iters: int = 5,
#                  in_mask_size: int = 28,
#                  in_image_size: int = 224,
#                  resize_mode: str = "resize"
#                  ):

#         super().__init__()
#         self.sam_size = OUT_SIZE
#         self.in_image_size = in_image_size
#         self.sam_model_name = sam_model_name
#         # self.model = build_sam(sam_model_name)
#         overrides = dict(
#             conf=0.25, save=False, task="segment", mode="predict", imgsz=1024, model=sam_model_name, verbose=False,
#         )
#         self.model = SAM2Predictor(overrides=overrides)
#         print(self.model.model)
#         # self.model.eval()

#         assert resize_mode == "resize", "Only resize is supported for now"
#         assert sam_model_name != "mobile_sam.pt", "Mobile sam is not supported for now"

#         # self.model.set_imgsz(SAM_IMAGE_SIZE)
#         # self.get_im_features = self._resolve_im_features_getter()
#         self.use_box = use_box
#         self.use_mask = use_mask
#         self.use_point = use_point
#         self.in_mask_size = in_mask_size
#         self.refine_iters = refine_iters

#         self.resize_mode = resize_mode

#         # self.norm = Normalize(SAM_MEAN, SAM_STD)

#         # self.mask_getter = self._resolve_mask_getter()

#         # self.feats_getter = self._resolve_im_features_getter()

#         # self.im_resizer = self._resolve_im_resizer()

#         # self.is_sam2 = sam_model_name in SAM_2_MODELS

#         # self._bb_feat_sizes = [[x // (4 * i) for x in SAM_IMAGE_SIZE] for i in [1, 2, 4]]

#         self.requires_grad_(False)

#     def set_img(self, image):
#         return self.model.set_image(image.cpu().detach().numpy().transpose(1, 2, 0))
    
#     def extract_bboxes(self, masks: torch.Tensor, threshold: Optional[float] = None):
#         if masks.numel() == 0:
#             return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

#         num_instances = masks.size(1)
#         if threshold is None:
#             masks = (masks > torch.quantile(masks, 0.85))
#         else:
#             masks = (masks > threshold)

#         n = masks.shape[0]

#         bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

#         for index, mask in enumerate(masks):
#             y, x = torch.where(mask != 0)
#             if y.numel() == 0 or x.numel() == 0:
#                 continue

#             bounding_boxes[index, 0] = torch.min(x)
#             bounding_boxes[index, 1] = torch.min(y)
#             bounding_boxes[index, 2] = torch.max(x)
#             bounding_boxes[index, 3] = torch.max(y)
#         # print(bounding_boxes.shape)
#         return bounding_boxes.cpu().detach().numpy() # bounding_boxes.unflatten(0, (-1, num_instances)).cpu().detach().numpy()
    
#     def get_reference_points(self, masks: torch.Tensor, num_points: int = 5, threshold: float = 0.5):
#         num_masks, h, w = masks.shape
#         input_points = torch.zeros(num_masks, num_points, 2, device=masks.device)
#         for m in range(num_masks):
#             current_mask = masks[m]  # (height, width)

#             upper_threshold = current_mask.max()
#             flattened = current_mask.flatten()
#             # sorted_values, _ = torch.sort(flattened)
#             # k = int(0.99 * sorted_values.shape[0])
#             lower_threshold = torch.quantile(flattened, 0.97) #sorted_values[k]

#             condition = (current_mask <= upper_threshold) & (current_mask >= lower_threshold)
#             cond_coords = condition.nonzero(as_tuple=True)

#             values = current_mask[cond_coords]
#             mean = torch.mean(values)
#             std = torch.std(values)

#             lower_scale, upper_scale = 1.0, 3.0
#             upper_bound = mean + std * upper_scale
#             lower_bound = mean - std * lower_scale

#             condition = (current_mask <= upper_bound) & (current_mask >= lower_bound)
#             coords = condition.nonzero(as_tuple=False)
#             coords = torch.stack([coords[:, 1], coords[:, 0]], dim=1)

#             if num_points > len(coords):
#                 current_num_points = len(coords)
#             else:
#                 current_num_points = num_points

#             # Выбираем случайные индексы
#             random_idxs = torch.randperm(len(coords))[:current_num_points]
#             # if torch.rand(1).item() > 0.5:
#                 # random_idxs = torch.randperm(len(coords))[:current_num_points]
#             # else:
#             #     random_idxs = torch.randint(0, len(coords), (current_num_points,))

#             input_points[m] = coords[random_idxs].int()#.cpu().numpy()
#         input_labels = torch.ones((num_masks, num_points), dtype=torch.int32, device=masks.device)#.cpu().numpy()

#         return input_points.cpu().detach().numpy(), input_labels.cpu().detach().numpy()
    
#     def prepare_prompt(self, masks: torch.Tensor, threshold: Optional[float] = None):
#         bboxes = None
#         points = None
#         g_masks = None

#         if self.use_box:
#             bboxes = self.extract_bboxes(masks, threshold=threshold) # .flatten(0, 1).unsqueeze(1)
#         #
#         # if self.use_point:
#         #     points = self.get_reference_points(masks, threshold=threshold)

#         # if self.use_mask:
#         #     g_mask = self.extract_gauss_mask()

#         # if self.is_sam2:
#         #     bboxes, points, g_masks = self._sam2_refine_prompts(bboxes, points, g_masks)

#         return bboxes, points, g_masks
#     # def get_best_masks(self, sam_masks, ref_masks, num_slots=7):
#     #     print(sam_masks.shape)
#     #     B, M, I, H, W = sam_masks.unflatten(0, (-1, num_slots)).shape
#     #     best_masks = torch.zeros((B, M, 1, H, W), device=sam_masks.device)
#     #     for b in range(B):
#     #         for m in range(M):
#     #             ref_mask = ref_masks[b, m]
#     #             binary_ref_mask = (ref_mask > 0.5).float() # .cpu().numpy()
#     #             if binary_ref_mask.sum() == 0:
#     #                 continue

#     #             # Выбираем маску с максимальным IoU к ref_mask
#     #             best_iou = -1
#     #             best_mask = None
#     #             for idx in range(I):  # обычно 3 маски
#     #                 sam_mask = sam_masks[b, m, idx] # > self.model.mask_threshold
#     #                 intersection = torch.logical_and(sam_mask, binary_ref_mask.bool()).sum()
#     #                 union = torch.logical_or(sam_mask, binary_ref_mask.bool()).sum()
#     #                 iou = intersection / (union + 1e-6)
#     #                 if iou > best_iou:
#     #                     best_iou = iou
#     #                     best_mask = sam_mask
                
#     #             best_masks[b, m, 0, ...] = best_mask
        
#     #     return best_masks
    
#     def get_best_pseudo_gt(self, sam_masks, ref_masks, scores, iou_threshold=0.75, coverage_threshold=0.95, min_mask_size=200, num_slots=7):
#         # sam_masks = scale_masks(sam_masks, (self.in_mask_size, self.in_mask_size), padding=False)
#         sam_masks = sam_masks.unflatten(0, (-1, num_slots))
#         sam_masks_binary = torch.where(sam_masks > self.model.mask_threshold, True, False)
#         # sam_masks_binary = sam_masks_binary.unflatten(0, (-1, num_slots))
#         scores = scores.unflatten(0, (-1, num_slots)).argmax(2)
#         # ref_masks = (ref_masks > torch.quantile(ref_masks, 0.99)).float()
#         pseudo_gt_list = []
#         logits_list = []
#         B, M, I, H, W = sam_masks_binary.shape
#         for b in range(B):
#             covered_mask = torch.zeros((H, W), dtype=torch.bool, device=sam_masks_binary.device)
#             final_masks = torch.zeros((M, H, W), dtype=torch.bool, device=sam_masks_binary.device)
#             final_masks_logits = torch.zeros((M, H, W), dtype=sam_masks.dtype, device=sam_masks_binary.device)
#             final_masks_list = []
#             # final_masks_logits_list = []

#             for m in range(M):
#                 ref_mask = ref_masks[b, m]
#                 # best_mask = sam_masks[b, m]
#                 binary_ref_mask = (ref_mask > torch.quantile(ref_mask, 0.85)).float() # .cpu().numpy()
#                 if binary_ref_mask.sum() == 0 and m != M - 1:
#                     continue

#                 best_iou = -1
#                 best_mask = None
#                 for idx in range(I):
#                     sam_mask = sam_masks_binary[b, m, idx] # > self.model.mask_threshold
#                     intersection = torch.logical_and(sam_mask, binary_ref_mask.bool()).sum()
#                     union = torch.logical_or(sam_mask, binary_ref_mask.bool()).sum()
#                     iou = intersection / (union + 1e-6)

#                     if iou > best_iou:
#                         best_iou = iou
#                         best_mask = sam_mask
#                 best_mask_no_overlap = best_mask & (~covered_mask)
#                 if best_mask_no_overlap.sum() < min_mask_size:
#                     continue

#                 is_new = True
#                 for existing_mask in final_masks_list:
#                     inter = torch.logical_and(best_mask_no_overlap, existing_mask).sum()
#                     union = torch.logical_or(best_mask_no_overlap, existing_mask).sum()
#                     iou = inter / (union + 1e-6)
#                     if iou > iou_threshold:
#                         is_new = False
#                         break

#                 if is_new:
#                     final_masks_logits[m] = sam_masks[b, m, scores[b, m]]
#                     final_masks[m] = best_mask_no_overlap
#                     # final_masks_logits_list.append(sam_masks[b, m, scores[b, m]])
#                     final_masks_list.append(best_mask_no_overlap)
#                     covered_mask |= best_mask_no_overlap
#                 elif not is_new and m != M - 1:
#                     continue
#                 if m == M - 1:
#                     # print("final_mask")
#                     final_masks[m] = ~covered_mask
#                     uncovered_elements = sam_masks[b, m, scores[b, m]][~covered_mask]
#                     final_masks_logits[m][~covered_mask] = uncovered_elements
#                     break
#                 coverage = covered_mask.float().mean().item()
#                 if coverage >= coverage_threshold:
#                     # print(f"[Batch {b}] Достигнут порог покрытия: {coverage:.2f}")
#                     final_masks[m] = ~covered_mask
#                     uncovered_elements = sam_masks[b, m, scores[b, m]][~covered_mask]
#                     final_masks_logits[m][~covered_mask] = uncovered_elements
#                 # final_masks[m] = best_mask


#             # pseudo_gt_masks = torch.stack(final_masks) if final_masks else \
#             # torch.zeros((0, H, W)).to(torch.bool) # if final_masks else torch.zeros((0, H, W))
#             # pseudo_gt_list.append(final_masks)
#             logits_list.append(final_masks_logits)
#         # pseudo_gt = torch.stack(pseudo_gt_list).to(torch.int32)
#         logits = torch.stack(logits_list)
#         # pseudo_gt = torch.where(logits > self.model.mask_threshold, True, False).float()
#         # pseudo_gt = F.interpolate(
#         #     pseudo_gt, size=(self.in_mask_size, self.in_mask_size), 
#         #     mode='nearest-exact',
#         # )
#         norm_logits = self.normalize_to_range(logits, min_val=0., max_val=1.)
#         norm_logits = scale_masks(logits, (self.in_mask_size, self.in_mask_size), padding=False)

#         return logits, norm_logits

#     def get_masks_to_gts(self, masks: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
#         improved_masks = torch.zeros_like(masks, device=masks.device)
#         # for idx in range(len(masks)):
#         cost_matrix = torch.zeros((len(masks), len(gts)), dtype=torch.float32)
#         for i, model_mask in enumerate(masks):
#             for j, coco_mask in enumerate(gts):
#                 iou = self.compute_iou(model_mask, coco_mask)
#                 cost_matrix[i, j] = 1 - iou
        
#         row_ind, col_ind = linear_sum_assignment(cost_matrix)
#         # combined_mask = torch.zeros_like(slot_masks[idx, 0])
#         for r, c in zip(row_ind, col_ind):
#             if cost_matrix[r, c] < 1.0:
#                 # combined_mask = combined_mask | gts[idx, c]
#                 improved_masks[r] = gts[c]
        
#         # for r, c in zip(row_ind, col_ind):
#         #     if cost_matrix[r, c] == 1.0:
#         #         improved_masks[idx, r] = ~(combined_mask)
#         #         break
        
#         return improved_masks
    
#     def compute_iou(self, mask1, mask2):
#         intersection = torch.logical_and(mask1, mask2).sum()
#         union = torch.logical_or(mask1, mask2).sum()
#         return intersection / union if union > 0 else 0
    
#     def compute_intersection_over_min(self, mask1, mask2):
#         intersection = torch.logical_and(mask1, mask2).sum()
#         # union = np.logical_or(mask1, mask2).sum()
#         area1 = mask1.sum()
#         area2 = mask2.sum()
#         return intersection / min(area1, area2) if min(area1, area2) > 0 else 0
    
#     def deduplicate_masks(self, masks, threshold=0.9, add_background=True):
#         # if not masks:
#         #     return []

#         to_remove = []
#         n = len(masks)

#         for i in range(n):
#             for j in range(i + 1, n):
#                 mask_i = masks[i]
#                 mask_j = masks[j]
#                 iou = self.compute_intersection_over_min(mask_i, mask_j)
#                 if iou >= threshold:
#                     if masks[i].sum() > masks[j].sum():
#                     # if masks[i]['predicted_iou'] < masks[j]['predicted_iou']:
#                         to_remove.append(i)
#                     else:
#                         to_remove.append(j)

#         to_remove = list(set(to_remove))
#         to_remove.sort(reverse=True)
#         N = masks.shape[0]
#         # for idx in to_remove:
#         #     masks.pop(idx)
#         keep_indices = list(range(N))
        
#         # Удаляем индексы с конца, чтобы не сбить нумерацию
#         for idx in to_remove:
#             if 0 <= idx < len(keep_indices):
#                 keep_indices.pop(idx)
        
#         # Преобразуем в тензор индексов
#         keep_indices_tensor = torch.tensor(keep_indices, device=masks.device, dtype=torch.long)
        
#         # Возвращаем только уникальные маски
#         return masks[keep_indices_tensor]

#         # Добавляем фоновую маску
#         # if add_background:
#         #     first_mask = masks[0]['segmentation']
#         #     height, width = first_mask.shape
#         #     background_mask = np.ones((height, width), dtype=bool)

#         #     for mask_dict in masks:
#         #         mask = mask_dict['segmentation']
#         #         background_mask &= ~mask

#         #     masks.append({
#         #         "segmentation": background_mask,
#         #         "area": background_mask.sum(),
#         #         "bbox": [0, 0, width, height],
#         #         "predicted_iou": 0.0,
#         #         "point_coords": [],
#         #         "stability_score": 0.0,
#         #         "crop_box": [0, 0, width, height],
#         #         "is_background": True
#         #     })

#         # return masks

#     @torch.no_grad()
#     def improve_masks(self, image: torch.Tensor, ref_masks: torch.Tensor, gts: Optional[torch.Tensor] = None):
#         # image = self._preprocess_image(image)
#         # num_instances = ref_masks.size(1)
#         # masks = self._preprocess_mask(ref_masks, OUT_SIZE)
#         # sam_ref_masks = self._preprocess_mask(ref_masks, SAM_IMAGE_SIZE)

#         # bboxes, points, _ = self.prepare_prompt(masks)

#         # self.model.f
#         # features = self.get_im_features(image)
#         H, W = image.shape[-2:]
#         b, k, h, w = ref_masks.shape
#         decoder_masks = scale_masks(ref_masks, (self.in_image_size, self.in_image_size), padding=False)
#         pred_masks = None
#         bboxes, points, _ = None, None, None
#         pseudo_gt = torch.zeros((b, k, H, W))
#         for s in range(b):
#             img = (image[s].cpu().detach().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
#             self.model.set_image(img)
#             # for m in range(k):
#             # pred_masks, scores = self.mask_getter(features, num_instances, bboxes, points, pred_masks, 
#             #                                       multimask=False)
#             # pred_masks = self.get_best_masks(pred_masks, ref_masks)
#             # if points is None:
#             #     bboxes, points, _ = self.prepare_prompt(ref_masks[s])
#             # else:
#             #     sam_ref_masks = scale_masks(
#             #         self.normalize_to_range(logits.permute(1, 0, 2, 3), min_val=0., max_val=1.), 
#             #         SAM_IMAGE_SIZE, padding=False
#             #     )
#             #     bboxes, points, _ = self.prepare_prompt(sam_ref_masks)
#             results = self.model(
#                 source=img, points_stride=80,
#                 # bboxes=bboxes, points=points, masks=None, multimask=False,
#             ) #True if i == self.refine_iters - 1 else False)
#             # print(len(results))
#             # if gts is not None:
#             improved_masks = []
#             for result in results:
#                 masks = result.masks
#                 if masks is None:
#                     continue
#                 # print(len(masks.data))
#                 improved_masks.append(masks.data)
#             improved_masks = torch.cat(improved_masks)
#             # print(improved_masks.shape)
#             improved_masks = self.deduplicate_masks(improved_masks)
#             # print(improved_masks.shape)
#             background_mask = torch.ones((H, W), dtype=bool, device=improved_masks.device)

#             for mask in improved_masks:
#                 mask = mask
#                 background_mask &= ~mask
            
#             improved_masks = torch.cat([improved_masks, background_mask.unsqueeze(0)])
#             # print(improved_masks.shape)
#             pseudo_gt[s] = self.get_masks_to_gts(decoder_masks[s], improved_masks)
#             # print(gts[s].shape)
#             # else:
#             #     masks = results[0].masks
#             #     if masks is None:
#             #         continue
#             #     pseudo_gt[s, :masks.data.shape[0]] = masks.data
#             # print(result.masks.data, result.masks.data)
#             # points = self.get_reference_points(pred_masks.squeeze(1).unflatten(0, (-1, num_instances)).to(torch.float32))
        
#         # logits, norm_logits = self.get_best_pseudo_gt(pred_masks, masks, num_slots=num_instances, scores=scores)
#         # logits = logits.permute(1, 0, 2, 3).squeeze(1).reshape(
#         #     ref_masks.shape[0], num_instances, self.sam_size[0], self.sam_size[1],
#         # )
#         # norm_logits = self.normalize_to_range(
#         #     logits, min_val=0., max_val=1.
#         # ).squeeze(1).reshape(ref_masks.shape[0], num_instances, self.sam_size[0], self.sam_size[1])
#         # print(logits.shape, norm_logits.shape)
#         # hard_masks = torch.where(
#         #     pred_masks > self.model.mask_threshold, True, False
#         # ).float().squeeze(1).unflatten(0, (-1, num_instances))
#         # hard_masks = F.interpolate(
#         #     hard_masks, size=(self.in_image_size, self.in_image_size), 
#         #     mode='nearest-exact',
#         # )
#         # logits = self.normalize_to_range(
#         #     pred_masks, min_val=0., max_val=1.
#         # ).squeeze(1).unflatten(0, (-1, num_instances))
#         # logits = scale_masks(logits, (self.in_image_size, self.in_image_size), padding=False)
#         # print(pred_masks.shape, logits.shape) #hard_masks.shape, logits.shape)

#         # print(pred_masks.shape)
#         # pred_masks = self.post_process_mask(pred_masks, ref_masks)
#         # pred_masks = self.post_process_mask(best_masks, ref_masks)

#         return {
#             "sam_masks": pseudo_gt,
#             # "sam_masks": logits, #.flatten(start_dim=2, end_dim=3),
#             # "norm_masks": norm_logits, #.flatten(start_dim=2, end_dim=3),
#             # "sam_masks_hard": hard_masks.flatten(start_dim=2, end_dim=3), #.squeeze(1).unflatten(0, (-1, num_instances))
#             # "sam_masks_vis": hard_masks.flatten(start_dim=2, end_dim=3),
#             # F.interpolate(
#             #     hard_masks, size=(self.in_image_size, self.in_image_size), 
#             #     mode='nearest-exact',
#             # ),
#         }
    
#     def normalize_to_range(self, data, min_val=-14, max_val=6):
#         """
#         Нормализует входные данные в диапазон [min_val, max_val] для тензоров PyTorch

#         Параметры:
#         - data: входной тензор PyTorch
#         - min_val: минимальное значение целевого диапазона
#         - max_val: максимальное значение целевого диапазона

#         Возвращает нормализованный тензор
#         """
#         if data is None:
#             return None
        
#         B, N = data.shape[:2]
        
#         # Находим текущие минимум и максимум по пространственным измерениям (H, W)
#         current_min = data.view(B, N, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
#         current_max = data.view(B, N, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        
#         # Создаем маску для случаев, где min == max (чтобы избежать деления на ноль)
#         mask = (current_min != current_max).float()
        
#         # Нормализуем данные
#         normalized = (data - current_min) / (current_max - current_min + 1e-9)  # добавляем небольшое значение для стабильности
#         normalized = normalized * (max_val - min_val) + min_val
        
#         # Для элементов, где min == max, устанавливаем среднее значение диапазона
#         # normalized = normalized * mask + (1 - mask) * ((min_val + max_val) / 2)
#         # normalized = normalized * mask + (1 - mask) * 0.0
        
#         return normalized