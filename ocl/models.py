from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from ultralytics.utils.ops import scale_masks

from ocl import configuration, losses, modules, optimizers, utils, visualizations
from ocl.data.transforms import Denormalize, Normalize


def build(
    model_config: configuration.ModelConfig,
    optimizer_config,
    train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
):
    optimizer_builder = optimizers.OptimizerBuilder(**optimizer_config)
    mode = model_config.get("mode", "default")
    background_slot_initializer = None
    feed_background_slot_into_processor = False
    background_slot_config = model_config.get('background_slot', None)
    use_background_slot = background_slot_config is not None and background_slot_config.use_background_slot
    if use_background_slot:
        model_config.initializer.n_slots = model_config.initializer.n_slots - 1
        feed_background_slot_into_processor = background_slot_config.feed_into_processor
        if not feed_background_slot_into_processor:
            model_config.grouper.num_slots = model_config.grouper.num_slots - 1
            background_slot_config.initializer['same_output_dim'] = True

        background_slot_initializer = modules.build_initializer(background_slot_config.initializer)

    initializer = modules.build_initializer(model_config.initializer)
    if mode == "smm" or mode == "tsmm" or "hc" in mode or mode == "samsa":
        encoder = modules.build_encoder(model_config.encoder, "HCEncoder")
    else:
        encoder = modules.build_encoder(model_config.encoder, "FrameEncoder")
    grouper = modules.build_grouper(model_config.grouper)
    decoder = modules.build_decoder(model_config.decoder, use_background_slot=use_background_slot)
    if mode == "samsa":
        sam = modules.build_sam(model_config.sam)
        #     sam_model_name=model_config.sam.sam_model_name,
        #     use_box=model_config.sam.use_box,
        #     use_mask=model_config.sam.use_mask,
        #     use_point=model_config.sam.use_point,
        #     refine_iters=model_config.sam.refine_iters,
        #     in_mask_size=model_config.sam.in_mask_size,
        #     in_image_size=model_config.sam.in_image_size,
        #     resize_mode=model_config.sam.resize_mode,
        # )
    else:
        sam = None

    target_encoder = None
    if model_config.target_encoder:
        if mode == "default":
            target_encoder = modules.build_encoder(model_config.target_encoder, "FrameEncoder")
            assert (
                model_config.target_encoder_input is not None
            ), "Please specify `target_encoder_input`."
        elif mode == "smm" or mode == "tsmm" or mode == "samsa":
            target_encoder = modules.build_encoder(model_config.target_encoder, "HCEncoder")
            assert (
                model_config.target_encoder_input is not None
            ), "Please specify `target_encoder_input`."

    current_step = model_config.grouper.get("step", 0)
    input_type = model_config.get("input_type", "image")
    image_size = model_config.get("image_size", 224)
    mask_size = model_config.get("mask_size", 16)
    if input_type == "image":
        processor = modules.LatentProcessor(grouper, predictor=None)
    elif input_type == "video":
        encoder = modules.MapOverTime(encoder)
        decoder = modules.MapOverTime(decoder)
        if target_encoder:
            target_encoder = modules.MapOverTime(target_encoder)
        if model_config.predictor is not None:
            predictor = modules.build_module(model_config.predictor)
        else:
            predictor = None
        if model_config.latent_processor:
            processor = modules.build_video(
                model_config.latent_processor,
                "LatentProcessor",
                corrector=grouper,
                predictor=predictor,
            )
        else:
            processor = modules.LatentProcessor(grouper, predictor)
        processor = modules.ScanOverTime(processor, mode=mode)
    else:
        raise ValueError(f"Unknown input type {input_type}")

    target_type = model_config.get("target_type", "features")
    if target_type == "input":
        default_target_key = input_type
    elif target_type == "features":
        if model_config.target_encoder_input is not None:
            default_target_key = "target_encoder.backbone_features"
        else:
            default_target_key = "encoder.backbone_features"
    else:
        raise ValueError(f"Unknown target type {target_type}. Should be `input` or `features`.")

    loss_defaults = {
        "pred_key": "decoder.reconstruction",
        "target_key": default_target_key,
        "video_inputs": input_type == "video",
        "patch_inputs": target_type == "features",
    }
    if model_config.losses is None:
        loss_fns = {"mse": losses.build(dict(**loss_defaults, name="MSELoss"))}
    else:
        loss_fns = {
            name: losses.build({**loss_defaults, **loss_config})
            for name, loss_config in model_config.losses.items()
        }
        # if model_config.get("threshold_for_guidance_loss") is not None:
        #     threshold_for_gl = model_config.get("threshold_for_guidance_loss")

    if model_config.mask_resizers:
        mask_resizers = {
            name: modules.build_utils(resizer_config, "Resizer")
            for name, resizer_config in model_config.mask_resizers.items()
        }
    else:
        mask_resizers = {
            "decoder": modules.build_utils(
                {
                    "name": "Resizer",
                    # When using features as targets, assume patch-shaped outputs. With other
                    # targets, assume spatial outputs.
                    "patch_inputs": target_type == "features",
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
            "grouping": modules.build_utils(
                {
                    "name": "Resizer",
                    "patch_inputs": True,
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
        }

    if model_config.masks_to_visualize:
        masks_to_visualize = model_config.masks_to_visualize
    else:
        masks_to_visualize = "decoder"

    model = ObjectCentricModel(
        optimizer_builder,
        initializer,
        encoder,
        processor,
        decoder,
        loss_fns,
        sam=sam,
        loss_weights=model_config.get("loss_weights", None),
        threshold_for_gl=model_config.get("threshold_for_guidance_loss"),
        target_encoder=target_encoder,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        mask_resizers=mask_resizers,
        input_type=input_type,
        image_size=image_size,
        mask_size=mask_size,
        mode=mode,
        current_step=current_step,
        target_encoder_input=model_config.get("target_encoder_input", None),
        visualize=model_config.get("visualize", False),
        visualize_every_n_steps=model_config.get("visualize_every_n_steps", 1000),
        masks_to_visualize=masks_to_visualize,
        background_slot_initializer=background_slot_initializer,
        feed_background_slot_into_processor=feed_background_slot_into_processor,
    )

    if model_config.load_weights:
        model.load_weights_from_checkpoint(model_config.load_weights, model_config.modules_to_load)

    return model


class ObjectCentricModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_builder: Callable,
        initializer: nn.Module,
        encoder: nn.Module,
        processor: nn.Module,
        decoder: nn.Module,
        loss_fns: Dict[str, losses.Loss],
        *,
        sam: Optional[nn.Module] = None,
        mode: str = "default",
        current_step: int,
        loss_weights: Optional[Dict[str, float]] = None,
        threshold_for_gl: Optional[float] = None,
        target_encoder: Optional[nn.Module] = None,
        train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        mask_resizers: Optional[Dict[str, modules.Resizer]] = None,
        input_type: str = "image",
        image_size: int = 224,
        mask_size: int = 16,
        target_encoder_input: Optional[str] = None,
        visualize: bool = False,
        visualize_every_n_steps: Optional[int] = None,
        masks_to_visualize: Union[str, List[str]] = "decoder",
        background_slot_initializer: Optional[nn.Module] = None,
        feed_background_slot_into_processor: Optional[bool] = None,
    ):
        super().__init__()
        self.optimizer_builder = optimizer_builder
        self.initializer = initializer
        self.encoder = encoder
        self.processor = processor
        self.resolution = image_size
        self.mask_size = mask_size
        self.decoder = decoder
        self.sam = sam
        self.threshold_for_loss = threshold_for_gl
        # if sam is not None:
        #     for param in self.encoder.parameters():
        #         param.requires_grad = False
            
        #     for param in self.initializer.parameters():
        #         param.requires_grad = False

        #     for param in self.processor.parameters():
        #         param.requires_grad = False
        
        self.target_encoder = target_encoder
        self.mode = mode
        self.background_slot_initializer = background_slot_initializer
        self.feed_background_slot_into_processor = feed_background_slot_into_processor
        print(self.mode)
        self.current_step = current_step

        if loss_weights is not None:
            # Filter out losses that are not used
            assert (
                loss_weights.keys() == loss_fns.keys()
            ), f"Loss weight keys {loss_weights.keys()} != {loss_fns.keys()}"
            loss_fns_filtered = {k: loss for k, loss in loss_fns.items() if loss_weights[k] != 0.0}
            loss_weights_filtered = {
                k: loss for k, loss in loss_weights.items() if loss_weights[k] != 0.0
            }
            self.loss_fns = nn.ModuleDict(loss_fns_filtered)
            self.loss_weights = loss_weights_filtered
        else:
            self.loss_fns = nn.ModuleDict(loss_fns)
            self.loss_weights = {}

        self.mask_resizers = mask_resizers if mask_resizers else {}
        self.mask_resizers["segmentation"] = modules.Resizer(
            video_inputs=input_type == "video", resize_mode="nearest-exact"
        )
        # if sam is not None:
        #     self.mask_resizers["sam"] = modules.Resizer(
        #         video_inputs=input_type == "video", resize_mode="nearest-exact"
        #     )
        self.mask_soft_to_hard = modules.SoftToHardMask()
        # self.mask_soft_to_hard_for_dec = modules.SoftToHardMask(use_threshold=True, threshold=0.45)
        self.mask_soft_to_hard_for_sam = modules.SoftToHardMask(use_threshold=True, threshold=0.)#self.sam.model.mask_threshold)
        self.train_metrics = torch.nn.ModuleDict(train_metrics)
        self.val_metrics = torch.nn.ModuleDict(val_metrics)

        self.visualize = visualize
        if visualize:
            assert visualize_every_n_steps is not None
        self.visualize_every_n_steps = visualize_every_n_steps
        if isinstance(masks_to_visualize, str):
            masks_to_visualize = [masks_to_visualize]
        for key in masks_to_visualize:
            if key not in ("decoder", "grouping", "conv_grouping", "sam", "conv"):
                raise ValueError(f"Unknown mask type {key}. Should be `decoder` or `grouping` or `sam` or `conv` or `conv_grouping`.")
        self.mask_keys_to_visualize = [f"{key}_masks" for key in masks_to_visualize]

        if input_type == "image":
            self.input_key = "image"
            self.expected_input_dims = 4
        elif input_type == "video":
            self.input_key = "video"
            self.expected_input_dims = 5
        else:
            raise ValueError(f"Unknown input type {input_type}. Should be `image` or `video`.")

        self.target_encoder_input_key = (
            target_encoder_input if target_encoder_input else self.input_key
        )

    def configure_optimizers(self):
        modules = {
            "initializer": self.initializer,
            "encoder": self.encoder,
            "processor": self.processor,
            "decoder": self.decoder,
        }
        if self.background_slot_initializer is not None:
            modules["background_slot_initializer"] = self.background_slot_initializer

        return self.optimizer_builder(modules)

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
        if data.ndim != 3:
            data = data[np.newaxis, :, :]

        current_min = np.min(data, axis=(1, 2))
        current_max = np.max(data, axis=(1, 2))

        if np.any(current_min == current_max):
            idxs = np.where(current_min == current_max)
            normalized = np.zeros_like(data)
            normalized[idxs] = np.full_like(data[idxs], (min_val + max_val) / 2, dtype=data.dtype)
            mask = np.ones(data.shape[0], dtype=bool)
            mask[idxs] = False
            normalized[mask] = (data[mask] - current_min[mask]) / (current_max[mask] - current_min[mask])
            normalized[mask] = normalized[mask] * (max_val - min_val) + min_val
            return normalized

        current_min = np.min(data, axis=(1, 2)).reshape(data.shape[0], 1, 1)
        current_max = np.max(data, axis=(1, 2)).reshape(data.shape[0], 1, 1)
        normalized = (data - current_min) / (current_max - current_min)
        normalized = normalized * (max_val - min_val) + min_val

        return normalized
    
    def masks_to_boxes(self, masks):
        """Compute the bounding boxes around the provided masks

        The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

        Returns a [N, 4] tensors, with the boxes in xyxy format
        """
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device)
        
        h, w = masks.shape[-2:]

        y = torch.arange(0, h, dtype=torch.float)
        x = torch.arange(0, w, dtype=torch.float)
        y, x = torch.meshgrid(y, x)
        y = y.to(masks)
        x = x.to(masks)

        x_mask = ((masks>128) * x.unsqueeze(0))
        x_max = x_mask.flatten(1).max(-1)[0]
        x_min = x_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

        y_mask = ((masks>128) * y.unsqueeze(0))
        y_max = y_mask.flatten(1).max(-1)[0]
        y_min = y_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

        return torch.stack([x_min, y_min, x_max, y_max], 1)
    
    def masks_sample_points(self, masks,k=10):
        """Sample points on mask
        """
        if masks.numel() == 0:
            return torch.zeros((0, 2), device=masks.device)
        
        h, w = masks.shape[-2:]

        y = torch.arange(0, h, dtype=torch.float)
        x = torch.arange(0, w, dtype=torch.float)
        y, x = torch.meshgrid(y, x)
        y = y.to(masks)
        x = x.to(masks)

        # k = 10
        samples = []
        for b_i in range(len(masks)):
            select_mask = (masks[b_i]>128)
            x_idx = torch.masked_select(x,select_mask)
            y_idx = torch.masked_select(y,select_mask)
            
            perm = torch.randperm(x_idx.size(0))
            idx = perm[:k]
            samples_x = x_idx[idx]
            samples_y = y_idx[idx]
            samples_xy = torch.cat((samples_x[:,None],samples_y[:,None]),dim=1)
            samples.append(samples_xy)

        samples = torch.stack(samples)
        return samples


    # Add noise to mask input
    # From Mask Transfiner https://github.com/SysCV/transfiner
    def masks_noise(self, masks):
        def get_incoherent_mask(input_masks, sfact):
            mask = input_masks.float()
            w = input_masks.shape[-1]
            h = input_masks.shape[-2]
            mask_small = F.interpolate(mask, (h//sfact, w//sfact), mode='bilinear')
            mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
            mask_residue = (mask - mask_recover).abs()
            mask_residue = (mask_residue >= 0.01).float()
            return mask_residue
        gt_masks_vector = masks / 255
        mask_noise = torch.randn(gt_masks_vector.shape, device= gt_masks_vector.device) * 1.0
        inc_masks = get_incoherent_mask(gt_masks_vector,  8)
        gt_masks_vector = ((gt_masks_vector + mask_noise * inc_masks) > 0.5).float()
        gt_masks_vector = gt_masks_vector * 255

        return gt_masks_vector

    def get_best_mask(self, masks_norm, original_mask, threshold_for_sam=0.5):
        best_iou = 0
        best_mask = np.zeros_like(masks_norm[0])
        threshold = 0.5 # np.percentile(original_mask, 60)
        for mask_norm in masks_norm:
            # mask_norm = self.normalize_to_range(mask, min_val=0., max_val=1.)
            intersection = ((original_mask > threshold) & (mask_norm < threshold_for_sam)).sum() # eq: np.logical_and((mask > 0.6), (mask_norm > 0.5)).sum()
            union = ((original_mask > threshold) | (mask_norm < threshold_for_sam)).sum() # eq: np.logical_or((mask > 0.6), (mask_norm > 0.5)).sum()
            iou = intersection / union
            if best_iou < iou:
                best_iou = iou
                best_mask = mask_norm

        return best_mask
    
    def get_best_region(self, improved_mask, original_mask, intersection_threshold=0.1):
        binary_mask = (improved_mask < 0.5).astype(np.uint8) * 255 
        binary_original_mask = (original_mask > 0.6).astype(np.uint8) * 255
        contours_improved, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_original, _ = cv2.findContours(binary_original_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out_mask = np.zeros_like(improved_mask)

        for i, contour_orig in enumerate(contours_original):
            mask_orig = np.zeros_like(binary_original_mask)
            cv2.drawContours(mask_orig, [contour_orig], -1, 1, thickness=cv2.FILLED)

            for j, contour_improved in enumerate(contours_improved):
                mask_improved = np.zeros_like(binary_mask)
                cv2.drawContours(mask_improved, [contour_improved], -1, 1, thickness=cv2.FILLED)
                intersection_area = (mask_orig & mask_improved).sum()
                original_area = np.sum(mask_orig > 0)
                if original_area > 0 and (intersection_area / original_area) >= intersection_threshold:
                    cv2.drawContours(out_mask, [contour_improved], -1, 1, thickness=cv2.FILLED)

        return out_mask
    
    def get_reference_points(self, mask, num_points, threshold=0.5):
        upper_threshold = mask.max()
        lower_threshold = threshold # np.percentile(mask, 95)
        condition = (mask <= upper_threshold) & (mask >= lower_threshold)
        cond_coords = np.where(condition)
        mean = np.mean(mask[cond_coords])
        std = np.std(mask[cond_coords])
        lower_scale, upper_scale = 3.0, 3.0
        upper_bound = mean + std * upper_scale
        lower_bound = mean - std * lower_scale
        condition = (mask <= upper_bound) & (mask >= lower_bound)
        y_coords, x_coords = np.where(condition)
        coords = list(zip(x_coords, y_coords))
        if len(coords) == 0:
            # print("So so bad")
            return [], []
        
        if num_points > len(coords):
            num_points = len(coords)

        random_idxs = np.random.choice(len(coords), num_points, replace=False)
        
        input_points = np.array([coords[i] for i in random_idxs]).astype("int")
        input_labels = np.ones(input_points.shape[0], dtype=np.int32)

        return input_points, input_labels

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # print(inputs.keys())
        encoder_input = inputs[self.input_key]  # batch [x n_frames] x n_channels x height x width
        assert encoder_input.ndim == self.expected_input_dims
        images = encoder_input
        batch_size = len(encoder_input)
        denorm = Denormalize(input_type=self.input_key)
        # a = denorm(encoder_input).cpu().numpy()
        # np.save('images.npy', a)
        # print('save images')
        encoder_output = self.encoder(encoder_input)
        features = encoder_output["features"]
        slots_initial = self.initializer(inputs=features)
        if self.background_slot_initializer is not None:
            background_slot_initial = self.background_slot_initializer(inputs=features)
            if self.feed_background_slot_into_processor:
                slots_initial = torch.cat((slots_initial, background_slot_initial), dim=1)

        processor_output = self.processor(slots_initial, features) # arg№2 = inputs, self.current_step
        # if processor_output["corrector"].get("kl_loss") is not None:
        #     self.kl_loss = processor_output["corrector"]["kl_loss"]
        # processor_output = self.processor(slots_initial, inputs)

        # size = int(processor_output["corrector"]["masks"].shape[-1]**0.5)
        # attn_vis = processor_output["corrector"]["masks"].unflatten(-1, (size, size))
        # processor_output["resized_masks"] = self.resize(attn_vis).to("cuda:0").flatten(-2, -1)
        # print(processor_output["resized_masks"].shape)
        slots = processor_output["state"]
        if self.background_slot_initializer is not None and not self.feed_background_slot_into_processor:
            slots = torch.cat((slots, background_slot_initial), dim=1)

        decoder_output = self.decoder(slots)
        # np.save('att_masks.npy', processor_output["corrector"]["masks"].cpu().detach().numpy())
        # print('save att_masks', processor_output["corrector"]["masks"].shape, processor_output["corrector"]["masks"].flatten().shape)
        # np.save('masks.npy', decoder_output['masks'].cpu().detach().numpy())
        # print('save masks', decoder_output["masks"].shape, decoder_output['masks'].flatten().shape)
        # if self.mode == "default":
        #     # size = int(decoder_output["masks"].shape[-1]**0.5)
        #     dec_masks = decoder_output["masks"].unflatten(-1, (self.mask_size, self.mask_size))
        #     b, n, _, _ = dec_masks.shape
        #     dec_masks = self.resize(dec_masks)    
        #     # denorm = Denormalize(input_type=self.input_key)
        #     images = denorm(images)
        #     # print(encoder_input.min(), encoder_input.max())
        #     decoder_output["pseudo_gts"] = torch.zeros((b, n, self.mask_size*self.mask_size))
        #     decoder_output["pseudo_gts_vis"] = torch.zeros((b, n, self.resolution, self.resolution))
        #     # decoder_output["resized_masks"] = decoder_output["pseudo_gts"]
        #     for img_idx in range(b):
        #         image_np = (images[img_idx] * 255).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
        #         self.sam.set_image(image_np)
        #         # dec_mask = self.resize(dec_masks[img_idx])    
        #         for mask_idx in range(n):
        #             dec_mask = dec_masks[img_idx, mask_idx, ...].cpu().detach().numpy()
        #             input_points, input_labels = self.get_reference_points(dec_mask, num_points=self.num_points)
                    
        #             if len(input_points) == 0:
        #                 out_mask = torch.tensor(dec_mask)
        #                 decoder_output["pseudo_gts_vis"][img_idx, mask_idx, ...] = out_mask > 0.6
        #                 decoder_output["pseudo_gts"][img_idx, mask_idx, ...] = (self.scale(out_mask[None, ...]) > 0.6).flatten(-2, -1).squeeze()
        #                 continue

        #             out_masks, scores, _ = self.sam.predict(
        #                 point_coords=input_points,
        #                 point_labels=input_labels,
        #                 # mask_input = mask[None, None, :, :], # attention_map or decoder_mask
        #                 multimask_output=True,
        #                 return_logits=True
        #             ) # predict_torch

        #             masks_norm = self.normalize_to_range(out_masks, min_val=0., max_val=1.)
        #             # dec_mask_norm = self.normalize_to_range(dec_mask, min_val=0., max_val=1.)
        #             best_mask = self.get_best_mask(masks_norm, dec_mask)
        #             # best_mask = masks_norm[scores.argmax()]
        #             if best_mask.dtype == np.bool_:
        #                 continue
        #             # out_mask = self.get_best_region(best_mask, dec_mask)
        #             out_mask = (dec_mask > 0.6) & (best_mask < 0.5)
        #             decoder_output["pseudo_gts_vis"][img_idx, mask_idx, ...] = torch.tensor(out_mask)
        #             decoder_output["pseudo_gts"][img_idx, mask_idx, ...] = self.scale(torch.tensor(out_mask[None, ...])).flatten(-2, -1).squeeze()
        #     # np.save('pseudo_gts.npy', decoder_output["pseudo_gts"].cpu().detach().numpy())
        #     # print('save pseudo_gts')
        #     # decoder_output["pseudo_gts"] = decoder_output["pseudo_gts"].softmax(dim=1)
        # print(decoder_output["masks"].unflatten(2, (self.mask_size, self.mask_size)).shape)
        # decoder_output["masks_for_sam"] = scale_masks(
        #     decoder_output["masks"].unflatten(2, (self.mask_size, self.mask_size)), 
        #     self.sam.sam_size, padding=False
        # ).flatten(start_dim=2, end_dim=3)
        outputs = {
            "batch_size": batch_size,
            "encoder": encoder_output,
            "processor": processor_output,
            "decoder": decoder_output,
        }
        if self.mode == "samsa":
            b, k = decoder_output["masks"].shape[:2]
            # print([decoder_output[key].shape for key in decoder_output.keys()])
            decoder_masks = decoder_output["masks"].view(b, k, self.mask_size, self.mask_size)
            outputs["sam"] = self.sam.improve_masks(
                image=denorm(encoder_input), 
                ref_masks=decoder_masks,
            )
            # outputs["sam"]["sam_masks"] = self.mask_resizers.get("segmentation")(
            #     outputs["sam"]["sam_masks_hard"], decoder_masks, # inputs[self.input_key]
            # ).flatten(start_dim=2, end_dim=3).to(decoder_masks.device)
            # print(f"SAM mask {outputs['sam']['sam_masks'].shape}")
            # sam_masks_metrics_hard = self.mask_resizers.get("segmentation")(
            #     outputs["sam"]["sam_masks"], decoder_masks, # inputs[self.input_key]
            # ) # outputs["sam"].get("sam_masks_hard").unflatten(2, (self.resolution, self.resolution))
            # outputs["sam"]["sam_masks_hard"] = self.mask_resizers.get("segmentation")(
            #     self.mask_soft_to_hard_for_sam(outputs["sam"]["sam_masks"]), 
            #     decoder_masks, # inputs[self.input_key]
            # ).flatten(start_dim=2, end_dim=3) # outputs["sam"].get("sam_masks_hard").unflatten(2, (self.resolution, self.resolution))
            # print(outputs["sam"]["sam_masks_hard"].shape, decoder_output["masks"].shape)
            outputs["sam"]["sam_masks_hard"] = self.mask_soft_to_hard_for_sam(
                outputs["sam"]["sam_masks"],
            )#.flatten(start_dim=2, end_dim=3) #.float()
            outputs["sam"]["sam_masks_hard"] = self.mask_resizers.get("segmentation")(
                outputs["sam"]["sam_masks_hard"], decoder_masks, # inputs[self.input_key]
            ).flatten(start_dim=2, end_dim=3)
            # outputs["sam"]["sam_masks_hard"] = self.mask_soft_to_hard_for_sam(torch.tensor(
            #     (decoder_masks > 0.45) & outputs["sam"]["sam_masks_hard"].bool(),
            # )).flatten(start_dim=2, end_dim=3)
            # outputs["sam"]["sam_masks_hard"] = self.mask_soft_to_hard_for_sam(
            #     outputs["sam"]["sam_masks_hard"],
            # ).to(decoder_masks.device).flatten(start_dim=2, end_dim=3)
            # print(outputs["sam"]["sam_masks_hard"].shape, outputs["sam"]["sam_masks_hard"].dtype)
            outputs["targets"] = self.get_targets(inputs, outputs)
            # print(self.trainer.global_step)
        else:
            outputs["targets"] = self.get_targets(inputs, outputs)
        

        return outputs

    def process_masks(
        self,
        masks: torch.Tensor,
        inputs: Dict[str, Any],
        resizer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if masks is None:
            return None, None, None

        if resizer is None:
            masks_for_vis = masks
            masks_for_vis_hard = self.mask_soft_to_hard(masks)
            masks_for_metrics_hard = masks_for_vis_hard
        else:
            masks_for_vis = resizer(masks, inputs[self.input_key])
            masks_for_vis_hard = self.mask_soft_to_hard(masks_for_vis)
            target_masks = inputs.get("segmentations")
            if target_masks is not None and masks_for_vis.shape[-2:] != target_masks.shape[-2:]:
                # print("Target is not None")
                masks_for_metrics = resizer(masks, target_masks)
                masks_for_metrics_hard = self.mask_soft_to_hard(masks_for_metrics)
            else:
                masks_for_metrics_hard = masks_for_vis_hard

        return masks_for_vis, masks_for_vis_hard, masks_for_metrics_hard

    @torch.no_grad()
    def aux_forward(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute auxilliary outputs only needed for metrics and visualisations."""
        # print(self.mask_resizers, inputs.keys(), self.input_key, self.sam.model.mask_threshold)
        decoder_masks = outputs["decoder"].get("masks")
        decoder_masks, decoder_masks_hard, decoder_masks_metrics_hard = self.process_masks(
            decoder_masks, inputs, self.mask_resizers.get("decoder")
        )
        # print(f"Decoder masks: dec_masks {decoder_masks.shape}, dec_hard {decoder_masks_hard.shape}, dec_hard_vis {decoder_masks_metrics_hard.shape, decoder_masks_metrics_hard.dtype}")
        conv_masks = outputs["decoder"].get("conv_masks")
        conv_masks, conv_masks_hard, conv_masks_metrics_hard = self.process_masks(
            conv_masks, inputs, self.mask_resizers.get("decoder")
        )
        sam_masks = None
        sam_masks_hard = None
        sam_masks_metrics_hard = None
        if self.mode == "samsa":
            # sam_masks = outputs["sam"].get("sam_masks")
            # sam_masks_hard = sam_masks
            # sam_masks = outputs["sam"].get("sam_masks_hard")
            # sam_masks_hard = outputs["sam"].get("sam_masks_hard")
            sam_masks_hard = self.mask_resizers.get("segmentation")(
                outputs["sam"].get("sam_masks_hard").unflatten(2, (self.mask_size, self.mask_size)), 
                inputs[self.input_key],
            )
            sam_masks = sam_masks_hard
            # sam_masks = self.mask_resizers.get("decoder")(
            #     outputs["sam"].get("norm_masks").flatten(start_dim=2, end_dim=3), inputs[self.input_key]
            # )
            # sam_masks_hard = self.mask_resizers.get("segmentation")(
            #     outputs["sam"].get("sam_masks_hard").unflatten(2, self.sam.sam_size), 
            #     inputs[self.input_key],
            # )
            # sam_masks_metrics_hard = outputs["sam"].get("sam_masks_hard").unflatten(2, (self.resolution, self.resolution))
            # sam_masks_metrics_hard = self.mask_resizers.get("decoder")(
            #     sam_masks, inputs.get("segmentations")
            # ) # outputs["sam"].get("sam_masks_hard").unflatten(2, (self.resolution, self.resolution))
            # sam_masks_metrics_hard = self.mask_soft_to_hard_for_sam(sam_masks_metrics_hard)
            sam_masks_metrics_hard = sam_masks_hard # outputs["sam"]["sam_masks_hard"].unflatten(2, (self.resolution, self.resolution))
            # sam_masks, sam_masks_hard, sam_masks_metrics_hard = self.process_masks(
            #     sam_masks, inputs, self.mask_resizers.get("decoder")
            # )
            # print(f"SAM masks: sam_masks {sam_masks.shape}, sam_hard {sam_masks_hard.shape}, sam_hard_vis {sam_masks_metrics_hard.shape, sam_masks_metrics_hard.dtype}")
        grouping_masks = outputs["processor"]["corrector"].get("masks")
        grouping_masks, grouping_masks_hard, grouping_masks_metrics_hard = self.process_masks(
            grouping_masks, inputs, self.mask_resizers.get("grouping")
        )
        conv_grouping_masks = outputs["processor"]["corrector"].get("conv_masks")
        conv_grouping_masks, conv_grouping_masks_hard, conv_grouping_masks_metrics_hard = self.process_masks(
            conv_grouping_masks, inputs, self.mask_resizers.get("grouping")
        )
        # if self.mode == "default":
            # sam_masks = outputs["processor"].get("pseudo_gts")
            # b, n, f = sam_masks.shape
            # sam_masks_resized = torch.zeros((b, n, 16*16))
            # for idx in range(b):
            #     mask = sam_masks[idx].unflatten(-1, (self.resolution, self.resolution))
            #     sam_masks_resized[idx] = self.scale(mask).flatten(-2, -1)
            # sam_masks, sam_masks_hard, sam_masks_metrics_hard = self.process_masks(
            #     sam_masks_resized, inputs, self.mask_resizers.get("grouping")
            # )
            # sam_masks = outputs["decoder"].get("pseudo_gts_vis")
            # b, n, f = sam_masks.shape
            # sam_masks_resized = torch.zeros((b, n, 16*16))
            # for idx in range(b):
            #     mask = sam_masks[idx].unflatten(-1, (self.resolution, self.resolution))
            #     sam_masks_resized[idx] = self.scale(mask).flatten(-2, -1)
            # sam_masks, sam_masks_hard, sam_masks_metrics_hard = self.process_masks(
            #     sam_masks, inputs, None # self.mask_resizers.get("decoder")
            # )

        aux_outputs = {}
        if decoder_masks is not None:
            aux_outputs["decoder_masks"] = decoder_masks
        if decoder_masks_hard is not None:
            aux_outputs["decoder_masks_vis_hard"] = decoder_masks_hard
        if decoder_masks_metrics_hard is not None:
            aux_outputs["decoder_masks_hard"] = decoder_masks_metrics_hard
        if conv_masks is not None:
            aux_outputs["conv_masks"] = conv_masks
        if conv_masks_hard is not None:
            aux_outputs["conv_masks_vis_hard"] = conv_masks_hard
        if conv_masks_metrics_hard is not None:
            aux_outputs["conv_masks_hard"] = conv_masks_metrics_hard
        if grouping_masks is not None:
            aux_outputs["grouping_masks"] = grouping_masks
        if grouping_masks_hard is not None:
            aux_outputs["grouping_masks_vis_hard"] = grouping_masks_hard
        if grouping_masks_metrics_hard is not None:
            aux_outputs["grouping_masks_hard"] = grouping_masks_metrics_hard
        if conv_grouping_masks is not None:
            aux_outputs["conv_grouping_masks"] = conv_grouping_masks
        if conv_grouping_masks_hard is not None:
            aux_outputs["conv_grouping_masks_vis_hard"] = conv_grouping_masks_hard
        if conv_grouping_masks_metrics_hard is not None:
            aux_outputs["conv_grouping_masks_hard"] = conv_grouping_masks_metrics_hard
        
        if sam_masks is not None:
            aux_outputs["sam_masks"] = sam_masks
        if sam_masks_hard is not None:
            aux_outputs["sam_masks_vis_hard"] = sam_masks_hard 
        if sam_masks_metrics_hard is not None:
            aux_outputs["sam_masks_hard"] = sam_masks_metrics_hard # .to(device="cuda:0")

        return aux_outputs

    def get_targets(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        if self.target_encoder:
            target_encoder_input = inputs[self.target_encoder_input_key]
            assert target_encoder_input.ndim == self.expected_input_dims

            with torch.no_grad():
                encoder_output = self.target_encoder(target_encoder_input)

            outputs["target_encoder"] = encoder_output

        targets = {}
        for name, loss_fn in self.loss_fns.items():
            targets[name] = loss_fn.get_target(inputs, outputs)

        return targets

    # def trajectory_smoothness_loss(self, masks):
    #     """
    #     masks: [B, T, num_slots, H, W] - маски вероятностей
    #     """
    #     B, T, num_slots, H, W = masks.shape
        
    #     # Создаем координатные карты
    #     y_coords = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(B, T, H, W).to(masks.device)
    #     x_coords = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(B, T, H, W).to(masks.device)
        
    #     centers_of_mass = []
    #     for t in range(T):
    #         slot_centers = []
    #         for s in range(num_slots):
    #             # Нормализуем маску для вычисления взвешенного центра масс
    #             norm_mask = masks[:, t, s] / (masks[:, t, s].sum(dim=(1, 2), keepdim=True) + 1e-8)
                
    #             # Вычисляем координаты центра масс
    #             center_y = torch.sum(norm_mask * y_coords[:, t], dim=(1, 2))
    #             center_x = torch.sum(norm_mask * x_coords[:, t], dim=(1, 2))
                
    #             slot_centers.append(torch.stack([center_x, center_y], dim=1))  # [B, 2]
            
    #         centers_of_mass.append(torch.stack(slot_centers, dim=1))  # [B, num_slots, 2]
        
    #     centers_of_mass = torch.stack(centers_of_mass, dim=1)  # [B, T, num_slots, 2]
        
    #     # Вычисляем смещения между последовательными кадрами
    #     displacements = centers_of_mass[:, 1:] - centers_of_mass[:, :-1]  # [B, T-1, num_slots, 2]
        
    #     # Вычисляем изменения скорости (ускорения)
    #     if T > 2:
    #         accelerations = displacements[:, 1:] - displacements[:, :-1]  # [B, T-2, num_slots, 2]
    #         # Среднеквадратичная норма ускорений
    #         smoothness_loss = torch.mean(torch.sum(accelerations**2, dim=-1))
    #     else:
    #         # Если только 2 кадра, минимизируем скорость
    #         smoothness_loss = torch.mean(torch.sum(displacements**2, dim=-1))
        
    #     return smoothness_loss

    def compute_loss(self, outputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        # probs_maps = outputs["processor"]["corrector"]["masks"]
        # b, t, k, d = probs_maps.shape
        # losses['tsl'] = self.trajectory_smoothness_loss(probs_maps.view(b, t, k, d**0.5, d**0.5))
        # log_likelihood = torch.log(torch.sum(probs_maps, dim=1) + 1e-10).sum(dim=-1).mean()
        for name, loss_fn in self.loss_fns.items():
            # if name == "loss_tsl":
            #     losses[name] = self.trajectory_smoothness_loss(probs_maps.view(b, t, k, int(d**0.5), int(d**0.5)))
            # else:
            # if name == "loss_featrec" and self.trainer.global_step < 1300:
            #     print(f" 1: {self.trainer.global_step}")
            #     continue
            # if name == "loss_featrec" and losses["guidance_loss"] > self.threshold_for_loss:
            #     print(self.trainer.global_step)
            #     continue
            prediction = loss_fn.get_prediction(outputs)#.to("cuda:0")
            target = outputs["targets"][name]#.to("cuda:0")
            # print(prediction.shape, target.shape)
            losses[name] = loss_fn(prediction, target)
            # print(f"{name}: {losses[name]}")
        # print(self.loss_weights.keys())
        losses_weighted = [loss * self.loss_weights.get(name, 1.0) for name, loss in losses.items()]
        total_loss = torch.stack(losses_weighted).sum()

        return total_loss, losses

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self.forward(batch)
        if self.current_step is not None:
            self.current_step += 1
        if self.train_metrics or (
            self.visualize and self.trainer.global_step % self.visualize_every_n_steps == 0
        ):
            aux_outputs = self.aux_forward(batch, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"train/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"train/{name}": loss for name, loss in losses.items()}
            to_log["train/loss"] = total_loss

        if self.train_metrics:
            for key, metric in self.train_metrics.items():
                values = metric(**batch, **outputs, **aux_outputs)
                self._add_metric_to_log(to_log, f"train/{key}", values)
                metric.reset()
        self.log_dict(to_log, on_step=True, on_epoch=False, batch_size=outputs["batch_size"])

        del outputs  # Explicitly delete to save memory

        if (
            self.visualize
            and self.trainer.global_step % self.visualize_every_n_steps == 0
            and self.global_rank == 0
        ):
            self._log_inputs(
                batch[self.input_key],
                {key: aux_outputs[f"{key}_hard"] for key in self.mask_keys_to_visualize},
                mode="train",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="train")

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if "batch_padding_mask" in batch:
            batch = self._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                return

        outputs = self.forward(batch)
        aux_outputs = self.aux_forward(batch, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"val/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"val/{name}": loss for name, loss in losses.items()}
            to_log["val/loss"] = total_loss

        if self.val_metrics:
            for metric in self.val_metrics.values():
                metric.update(**batch, **outputs, **aux_outputs)

        self.log_dict(
            to_log, on_step=False, on_epoch=True, batch_size=outputs["batch_size"], prog_bar=True
        )

        if self.visualize and batch_idx == 0 and self.global_rank == 0:
            masks_to_vis = {
                key: aux_outputs[f"{key}_vis_hard"] for key in self.mask_keys_to_visualize
            }
            if batch["segmentations"].shape[-2:] != batch[self.input_key].shape[-2:]:
                masks_to_vis["segmentations"] = self.mask_resizers["segmentation"](
                    batch["segmentations"], batch[self.input_key]
                )
            else:
                masks_to_vis["segmentations"] = batch["segmentations"]
            self._log_inputs(
                batch[self.input_key],
                masks_to_vis,
                mode="val",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="val")

    def validation_epoch_end(self, outputs):
        if self.val_metrics:
            to_log = {}
            for key, metric in self.val_metrics.items():
                self._add_metric_to_log(to_log, f"val/{key}", metric.compute())
                metric.reset()
            self.log_dict(to_log, prog_bar=True)

    @staticmethod
    def _add_metric_to_log(
        log_dict: Dict[str, Any], name: str, values: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        if isinstance(values, dict):
            for k, v in values.items():
                log_dict[f"{name}/{k}"] = v
        else:
            log_dict[name] = values

    def _log_inputs(
        self,
        inputs: torch.Tensor,
        masks_by_name: Dict[str, torch.Tensor],
        mode: str,
        step: Optional[int] = None,
    ):
        denorm = Denormalize(input_type=self.input_key)
        if step is None:
            step = self.trainer.global_step

        if self.input_key == "video":
            video = torch.stack([denorm(video) for video in inputs])
            self._log_video(f"{mode}/{self.input_key}", video, global_step=step)
            for mask_name, masks in masks_by_name.items():
                video_with_masks = visualizations.mix_videos_with_masks(video, masks)
                self._log_video(
                    f"{mode}/video_with_{mask_name}",
                    video_with_masks,
                    global_step=step,
                )
        elif self.input_key == "image":
            image = denorm(inputs)
            self._log_images(f"{mode}/{self.input_key}", image, global_step=step)
            for mask_name, masks in masks_by_name.items():
                image_with_masks = visualizations.mix_images_with_masks(image, masks)
                self._log_images(
                    f"{mode}/image_with_{mask_name}",
                    image_with_masks,
                    global_step=step,
                )
        else:
            raise ValueError(f"input_type should be 'image' or 'video', but got '{self.input_key}'")

    def _log_masks(
        self,
        aux_outputs,
        mask_keys=("decoder_masks", "grouping_masks", "conv_grouping_masks", "sam_masks", "conv_masks"),
        mode="val",
        types: tuple = ("frames",),
        step: Optional[int] = None,
    ):
        if step is None:
            step = self.trainer.global_step
        for mask_key in mask_keys:
            if mask_key in aux_outputs:
                masks = aux_outputs[mask_key]
                if self.input_key == "video":
                    _, f, n_obj, H, W = masks.shape
                    first_masks = masks[0].permute(1, 0, 2, 3)
                    first_masks_inverted = 1 - first_masks.reshape(n_obj, f, 1, H, W)
                    self._log_video(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                        types=types,
                    )
                elif self.input_key == "image":
                    _, n_obj, H, W = masks.shape
                    first_masks_inverted = 1 - masks[0].reshape(n_obj, 1, H, W)
                    self._log_images(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                    )
                else:
                    raise ValueError(
                        f"input_type should be 'image' or 'video', but got '{self.input_key}'"
                    )

    def _log_video(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
        max_frames: int = 8,
        types: tuple = ("frames",),
    ):
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            if "video" in types:
                logger.experiment.add_video(f"{name}/video", data, global_step=global_step)
            if "frames" in types:
                _, num_frames, _, _, _ = data.shape
                num_frames = min(max_frames, num_frames)
                data = data[:, :num_frames]
                data = data.flatten(0, 1)
                logger.experiment.add_image(
                    f"{name}/frames", make_grid(data, nrow=num_frames), global_step=global_step
                )

    def _save_video(self, name: str, data: torch.Tensor, global_step: int):
        assert (
            data.shape[0] == 1
        ), f"Only single videos saving are supported, but shape is: {data.shape}"
        data = data.cpu().numpy()[0].transpose(0, 2, 3, 1)
        data_dir = self.save_data_dir / name
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f"{global_step}.npy", data)

    def _log_images(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
    ):
        n_examples = min(n_examples, data.shape[0])
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            logger.experiment.add_image(
                f"{name}/images", make_grid(data, nrow=n_examples).cpu(), global_step=global_step
            )

    @staticmethod
    def _remove_padding(
        batch: Dict[str, Any], padding_mask: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        if torch.all(padding_mask):
            # Batch consists only of padding
            return None

        mask = ~padding_mask
        mask_as_idxs = torch.arange(len(mask))[mask.cpu()]

        output = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                output[key] = value[mask]
            elif isinstance(value, list):
                output[key] = [value[idx] for idx in mask_as_idxs]

        return output

    def _get_tensorboard_logger(self):
        if self.loggers is not None:
            for logger in self.loggers:
                if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                    return logger
        else:
            if isinstance(self.logger, pl.loggers.tensorboard.TensorBoardLogger):
                return self.logger

    def on_load_checkpoint(self, checkpoint):
        # Reset timer during loading of the checkpoint
        # as timer is used to track time from the start
        # of the current run.
        if "callbacks" in checkpoint and "Timer" in checkpoint["callbacks"]:
            checkpoint["callbacks"]["Timer"]["time_elapsed"] = {
                "train": 0.0,
                "sanity_check": 0.0,
                "validate": 0.0,
                "test": 0.0,
                "predict": 0.0,
            }

    def load_weights_from_checkpoint(
        self, checkpoint_path: str, module_mapping: Optional[Dict[str, str]] = None
    ):
        """Load weights from a checkpoint into the specified modules."""
        checkpoint = torch.load(checkpoint_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if module_mapping is None:
            module_mapping = {
                key.split(".")[0]: key.split(".")[0]
                for key in checkpoint
                if hasattr(self, key.split(".")[0])
            }

        for dest_module, source_module in module_mapping.items():
            try:
                module = utils.read_path(self, dest_module)
            except ValueError:
                raise ValueError(f"Module {dest_module} could not be retrieved from model") from None

            state_dict = {}
            for key, weights in checkpoint.items():
                if key.startswith(source_module):
                    if key != source_module:
                        key = key[len(source_module + ".") :]  # Remove prefix
                    state_dict[key] = weights
            if len(state_dict) == 0:
                raise ValueError(
                    f"No weights for module {source_module} found in checkpoint {checkpoint_path}."
                )

            module.load_state_dict(state_dict)
