from typing import Any, Dict, List, Optional, Union, Tuple

from collections import OrderedDict
import einops
import timm
import torch
import torchvision
from torch import nn
from transformers import AutoModelForVision2Seq

from ocl.modules import utils
from ocl.utils import config_as_kwargs, make_build_fn


@make_build_fn(__name__, "encoder")
def build(config, name: str):
    if name == "FrameEncoder":
        pos_embed = None
        if config.get("pos_embed"):
            pos_embed = utils.build_module(config.pos_embed)

        output_transform = None
        if config.get("output_transform").name == "networks.two_layer_mlp":
            output_transform = utils.build_module(config.output_transform)
        else:
            output_transform = config.get("output_transform")

        return FrameEncoder(
            backbone=utils.build_module(config.backbone, default_group="encoders"),
            pos_embed=pos_embed,
            output_transform=output_transform,
            **config_as_kwargs(config, ("backbone", "pos_embed", "output_transform")),
        )
    elif name == "HCEncoder":
        pos_embed = None
        if config.get("pos_embed"):
            pos_embed = utils.build_module(config.pos_embed)
        output_transform = None
        return HCEncoder(
            backbone=utils.build_module(config.backbone, default_group="encoders"),
            pos_embed=pos_embed,
            output_transform=output_transform,
            **config_as_kwargs(config, ("backbone", "pos_embed", "output_transform")),
        )
    else:
        return None


class FrameEncoder(nn.Module):
    """Module reducing image to set of features."""

    def __init__(
        self,
        backbone: nn.Module,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
        spatial_flatten: bool = False,
        main_features_key: str = "vit_block12",
    ):
        super().__init__()
        self.backbone = backbone
        self.pos_embed = pos_embed
        self.output_transform = output_transform
        self.spatial_flatten = spatial_flatten
        self.main_features_key = main_features_key

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: batch x n_channels x height x width
        backbone_features = self.backbone(images)
        if isinstance(backbone_features, dict):
            features = backbone_features[self.main_features_key].clone()
        else:
            features = backbone_features.clone()

        if self.pos_embed:
            features = self.pos_embed(features)

        if self.spatial_flatten:
            features = einops.rearrange(features, "b c h w -> b (h w) c")
        if isinstance(self.output_transform, torch.nn.Module):
            features = self.output_transform(features)

        assert (
            features.ndim == 3
        ), f"Expect output shape (batch, tokens, dims), but got {features.shape}"
        if isinstance(backbone_features, dict):
            for k, backbone_feature in backbone_features.items():
                if self.spatial_flatten:
                    backbone_features[k] = einops.rearrange(backbone_feature, "b c h w -> b (h w) c")
                assert (
                    backbone_feature.ndim == 3
                ), f"Expect output shape (batch, tokens, dims), but got {backbone_feature.shape}"
            main_backbone_features = backbone_features[self.main_features_key]

            return {
                "features": features,
                "backbone_features": main_backbone_features,
                **backbone_features,
            }
        else:
            if self.spatial_flatten:
                backbone_features = einops.rearrange(backbone_features, "b c h w -> b (h w) c")
            assert (
                backbone_features.ndim == 3
            ), f"Expect output shape (batch, tokens, dims), but got {backbone_features.shape}"

            return {
                "features": features,
                "backbone_features": backbone_features,
            }


class TimmExtractor(nn.Module):
    """Feature extractor utilizing models from timm library."""

    # Convenience aliases for feature keys
    FEATURE_ALIASES = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{i + 1}": f"blocks.{i}" for i in range(12)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_keys{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        "vit_output": "norm",
    }
    FEATURE_MAPPING = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(12)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        model_name = model
        self.frozen = frozen
        self.features = [features] if isinstance(features, str) else features
        self.is_vit = model_name.startswith("vit")

        model = TimmExtractor._create_model(model_name, pretrained, checkpoint_path, model_kwargs)

        if self.features is not None:
            nodes = torchvision.models.feature_extraction.get_graph_node_names(model)[0]

            features = []
            for name in self.features:
                if name in TimmExtractor.FEATURE_ALIASES:
                    name = TimmExtractor.FEATURE_ALIASES[name]

                if not any(node.startswith(name) for node in nodes):
                    raise ValueError(
                        f"Requested features under node {name}, but this node does "
                        f"not exist in model {model_name}. Available nodes: {nodes}"
                    )

                features.append(name)

            model = torchvision.models.feature_extraction.create_feature_extractor(model, features)

        self.model = model

        if self.frozen:
            self.requires_grad_(False)

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}

        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **model_kwargs
            )
        except (FileExistsError, FileNotFoundError):
            # Timm uses Hugginface hub for loading the files, which does some symlinking in the
            # background when loading the checkpoint. When multiple concurrent jobs attempt to
            # load the checkpoint, this can create conflicts, because the symlink is first removed,
            # then created again by each job. We attempt to catch the resulting errors here, and
            # retry creating the model, up to 3 times.
            if trials == 2:
                raise
            else:
                model = None

        if model is None:
            model = TimmExtractor._create_model(
                model_name, pretrained, checkpoint_path, model_kwargs, trials=trials + 1
            )

        return model

    def forward(self, inp):
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)
        else:
            outputs = self.model(inp)

        if self.features is not None:
            if self.is_vit:
                outputs = {k: v[:, 1:] for k, v in outputs.items()}  # Remove CLS token
            outputs = {self.FEATURE_MAPPING[key]: value for key, value in outputs.items()}
            for name in self.features:
                if ("keys" in name) or ("queries" in name) or ("values" in name):
                    feature_name = name.replace("queries", "keys").replace("values", "keys")
                    B, N, C = outputs[feature_name].shape
                    qkv = outputs[feature_name].reshape(
                        B, N, 3, C // 3
                    )  # outp has shape B, N, 3 * H * (C // H)
                    q, k, v = qkv.unbind(2)
                    if "keys" in name:
                        outputs[name] = k
                    elif "queries" in name:
                        outputs[name] = q
                    elif "values" in name:
                        outputs[name] = v
                    else:
                        raise ValueError(f"Unknown feature name {name}.")

            if len(outputs) == 1:
                # Unpack single output for now
                return next(iter(outputs.values()))
            else:
                return outputs
        else:
            return outputs

class HCEncoder(nn.Module):
    """Module reducing image to set of features."""

    def __init__(
        self,
        backbone: nn.Module,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
        spatial_flatten: bool = False,
        main_features_key: str = "vit_block12",
    ):
        super().__init__()
        self.backbone = backbone
        self.pos_embed = pos_embed
        self.output_transform = output_transform
        self.spatial_flatten = spatial_flatten
        self.main_features_key = main_features_key

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: batch x n_channels x height x width
        backbone_features = self.backbone(images)
        # if isinstance(backbone_features, dict):
        features = backbone_features[f'{self.main_features_key}_hc'].clone()
        # else:
        #     features = backbone_features.clone()
        if self.pos_embed:
            features = self.pos_embed(features)

        if self.spatial_flatten:
            features = einops.rearrange(features, "b c h w -> b (h w) c")
        if self.output_transform:
            features = self.output_transform(features)

        assert (
            features.ndim == 3
        ), f"Expect output shape (batch, tokens, dims), but got {features.shape}"
        if isinstance(backbone_features, dict):
            for k, backbone_feature in backbone_features.items():
                if self.spatial_flatten:
                    backbone_features[k] = einops.rearrange(backbone_feature, "b c h w -> b (h w) c")
                assert (
                    backbone_feature.ndim == 3
                ), f"Expect output shape (batch, tokens, dims), but got {backbone_feature.shape}"
            # main_backbone_features = backbone_features[self.main_features_key]
            main_backbone_features = backbone_features[self.main_features_key]

            if "image_kk" in backbone_features.keys():
                return {
                    "features": features,
                    "backbone_features": main_backbone_features,
                    "adjacency": backbone_features["image_kk"],
                    **backbone_features,
                }
            else:
                return {
                    "features": features,
                    "backbone_features": main_backbone_features,
                    **backbone_features,
                }
        else:
            if self.spatial_flatten:
                backbone_features = einops.rearrange(backbone_features, "b c h w -> b (h w) c")
            assert (
                backbone_features.ndim == 3
            ), f"Expect output shape (batch, tokens, dims), but got {backbone_features.shape}"

            return {
                "features": features,
                "backbone_features": backbone_features,
            }


class TimmExtractorv2(nn.Module):
    """Feature extractor utilizing models from timm library."""

    # Convenience aliases for feature keys
    FEATURE_ALIASESv2 = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{j + 1}": [f"blocks.{i}" for i in range(12)] for j in range(12)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_keys{j + 1}": [f"blocks.{i}.attn.qkv" for i in range(12)] for j in range(12)},
        **{f"vit_block_attn{j + 1}": [f"blocks.{i}.attn.proj_drop" for i in range(12)] for j in range(12)},
        "vit_output": "norm",
    }
    FEATURE_MAPPINGv2 = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.proj_drop": f"vit_block_attn{i + 1}" for i in range(12)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        drop: bool = True,
        dim: int = 256,
        mode: str = "default",
        proj_type: str = None,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        last_blocks: int = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        model_name = model
        self.frozen = frozen
        self.last_blocks = last_blocks
        self.dim = dim
        self.drop = drop
        self.mode = mode
        self.features = [features] if isinstance(features, str) else features
        self.is_vit = model_name.startswith("vit")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TimmExtractorv2._create_model(model_name, pretrained, checkpoint_path, model_kwargs).to(device)
        if self.features is not None:
            nodes = torchvision.models.feature_extraction.get_graph_node_names(model)[0]
            features = {}
            for name in self.features:
                if name in TimmExtractorv2.FEATURE_ALIASESv2:
                    name = TimmExtractorv2.FEATURE_ALIASESv2[name]
                if type(name) is not str:
                    for n in name:
                        if not any(node.startswith(n) for node in nodes):
                            raise ValueError(
                                f"Requested features under node {n}, but this node does "
                                f"not exist in model {model_name}. Available nodes: {nodes}"
                            )
                else:
                    if not any(node.startswith(name) for node in nodes):
                            raise ValueError(
                                f"Requested features under node {name}, but this node does "
                                f"not exist in model {model_name}. Available nodes: {nodes}"
                            )

                # features.append(name)
                for i, n in enumerate(name):
                    if "attn.qkv" in n:
                        features[n] = f"vit_block_keys{i+1}"
                    elif "attn.proj_drop" in n:
                        features[n] = f"vit_block_attn{i+1}"
                    else:
                        features[n] = f"vit_block{i+1}"
            features = {k: v for k, v in features.items() if v in self.features}
            # self.levels = []
            # for feats in features[-last_blocks:]:
            #     level = torchvision.models.feature_extraction.create_feature_extractor(model, feats)
            #     self.levels.append(level)
            model = torchvision.models.feature_extraction.create_feature_extractor(model, features)
        
        self.model = model

        if self.frozen:
            self.requires_grad_(False)
        
        self.dropout = torch.nn.Dropout2d(p=0.1)
        self.n_feats = 768 * last_blocks
        self.cluster1 = self.make_clusterer(self.n_feats, self.dim)
        self.proj_cluster1 = self.make_clusterer(self.n_feats, 768)
        self.proj_type = proj_type
        self.normalization = torch.nn.LayerNorm(self.n_feats)
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats, self.dim)
            self.proj_cluster2 = self.make_nonlinear_clusterer(self.n_feats, 768)
        self.project_head = nn.Linear(self.dim, self.dim)
        self.num_heads = 12

    def make_clusterer(self, in_channels, out):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out, (1, 1))) 

    def make_nonlinear_clusterer(self, in_channels, out):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(), # torch.nn.GELU()
            torch.nn.Conv2d(in_channels, out, (1, 1)))

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}

        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **model_kwargs
            )
        except (FileExistsError, FileNotFoundError):
            # Timm uses Hugginface hub for loading the files, which does some symlinking in the
            # background when loading the checkpoint. When multiple concurrent jobs attempt to
            # load the checkpoint, this can create conflicts, because the symlink is first removed,
            # then created again by each job. We attempt to catch the resulting errors here, and
            # retry creating the model, up to 3 times.
            if trials == 2:
                raise
            else:
                model = None

        if model is None:
            model = TimmExtractorv2._create_model(
                model_name, pretrained, checkpoint_path, model_kwargs, trials=trials + 1
            )

        return model

    def forward(self, inp):
        # outputs = []
        # for level in self.levels:
        #     if self.frozen:
        #         with torch.no_grad():
        #             outputs.append(level(inp))
        #     else:
        #         outputs.append(level(inp))
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)
        else:
            outputs = self.model(inp)

        if self.is_vit:
            outputs = {k: v[:, 1:] for k, v in outputs.items()}  # Remove CLS token
        if self.mode == "image_kk":
            feature_keys = []
            qkv_keys = []
            attn_keys = []
            for key in outputs.keys():
                if f"keys" in key:
                    qkv_keys.append(key)
                elif f"attn" in key:
                    attn_keys.append(key)
                else:
                    feature_keys.append(key)
            features = torch.cat([outputs[key] for idx, key in enumerate(feature_keys) if len(feature_keys) - (idx + 1) < self.last_blocks], 
                                dim=2)
            # qkv = [outputs[key][:, 1:, :] for idx, key in enumerate(qkv_keys) if len(qkv_keys) - (idx + 1) < self.last_blocks]
            # attn = [outputs[key][:, 1:, :] for idx, key in enumerate(attn_keys) if len(attn_keys) - (idx + 1) < self.last_blocks]
            # feat_h, feat_w = int(features.shape[1]**0.5), int(features.shape[1]**0.5) 
            # kk = [blk.reshape(features.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2) for blk in qkv]
            # B, H, W, D = k[0].shape
            # image_kk = torch.cat(kk, dim=1)
            features = self.normalization(features)
            B, N, C = outputs[feature_keys[-1]][:, :, :].shape
            qkv = [outputs[key].reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) \
                   for idx, key in enumerate(qkv_keys) if len(qkv_keys) - (idx + 1) < self.last_blocks]
            feat_h, feat_w = int(features.shape[1]**0.5), int(features.shape[1]**0.5)
            image_k = [k[1, :, :, :, :].reshape(B, qkv[0].shape[2], feat_h, feat_w, -1) for k in qkv]
            B, H, I, J, D = image_k[0].shape
            image_kk = [k.permute(0, 1, 4, 2, 3).reshape(B, H*D, I, J) for k in image_k]
            image_kk = torch.cat(image_kk, dim=1)
            image_feat = features.reshape(features.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            if self.proj_type is not None:
                if self.drop:
                    with torch.no_grad():
                        code = self.cluster1(self.dropout(image_feat))
                    code_kk = self.cluster1(self.dropout(image_kk))
                else:
                    with torch.no_grad():
                        code = self.cluster1(image_feat)
                    code_kk = self.cluster1(image_kk)
                if self.proj_type == "nonlinear":
                    if self.drop:
                        code += self.cluster2(self.dropout(image_feat))
                        code_kk += self.cluster2(self.dropout(image_kk))
                    else:
                        code += self.cluster2(image_feat)
                        code_kk += self.cluster2(image_kk)
            else:
                if self.drop:
                    code = self.dropout(image_feat)
                    code_kk = self.dropout(image_kk)
                else:
                    code = image_feat
                    code_kk = image_kk
            code_kk = code_kk.permute(0, 2, 3, 1).reshape(-1, self.dim)
            code_kk = self.project_head(code_kk)
            outputs = {self.features[0]: code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1),
                       f'{self.features[0]}_last': outputs[list(outputs.keys())[-1]][:, 1:, :],
                       "image_kk": code_kk.reshape(B, feat_h*feat_w, -1)}
            
            return outputs
        elif "hc" in self.mode:
            feature_keys = []
            proj_keys = []
            attn_keys = []
            for key in outputs.keys():
                if f"keys" in key or f"queries" in key or f"values" in key:
                    proj_keys.append(key)
                elif f"attn" in key:
                    attn_keys.append(key)
                else:
                    feature_keys.append(key)
            if self.last_blocks > 1:
                features = torch.cat([outputs[key] for key in feature_keys], dim=2)
            else:
                features = outputs[feature_keys[-1]]
            features = self.normalization(features)
            b, feat_h, feat_w = features.shape[0], int(features.shape[1]**0.5), int(features.shape[1]**0.5) 
            image_feat = features.reshape(b, feat_h, feat_w, -1).permute(0, 3, 1, 2)
            if self.proj_type is not None:
                with torch.no_grad():
                    if self.drop:
                        code = self.cluster1(self.dropout(image_feat))
                    else:
                        code = self.cluster1(image_feat)
                if self.proj_type == "nonlinear":
                    if self.drop:
                        code += self.cluster2(self.dropout(image_feat))
                    else:
                        code += self.cluster2(image_feat)
            else:
                if self.drop:
                    code = self.dropout(image_feat)
                else:
                    code = image_feat
            # print(code.shape)
            code = code.permute(0, 2, 3, 1).reshape(b, feat_h*feat_w, -1)
            # print(code.shape)
            code = self.project_head(code)
            # print(code.shape)
            if self.mode == "hc_video":
                for name in proj_keys:
                    if ("keys" in name) or ("queries" in name) or ("values" in name):
                        feature_name = name.replace("queries", "keys").replace("values", "keys")
                        B, N, C = outputs[feature_name].shape
                        qkv = outputs[feature_name].reshape(
                            B, N, 3, C // 3
                        )  # outp has shape B, N, 3 * H * (C // H)
                        q, k, v = qkv.unbind(2)
                        if "keys" in name:
                            outputs[name] = k
                        elif "queries" in name:
                            outputs[name] = q
                        elif "values" in name:
                            outputs[name] = v
                        else:
                            raise ValueError(f"Unknown feature name {name}.")
                # if self.last_blocks > 1:
                #     projections = torch.cat([outputs[key] for key in proj_keys], dim=2)
                # else:
                #     projections = outputs[proj_keys[-1]]
                # projections = self.normalization(projections)
                # proj_h, proj_w = int(projections.shape[1]**0.5), int(projections.shape[1]**0.5) 
                # proj = projections.reshape(projections.shape[0], proj_h, proj_w, -1).permute(0, 3, 1, 2)
                # if self.proj_type is not None:
                #     with torch.no_grad():
                #         if self.drop:
                #             code_proj = self.proj_cluster1(self.dropout(proj))
                #         else:
                #             code_proj = self.proj_cluster1(proj)
                #     if self.proj_type == "nonlinear":
                #         if self.drop:
                #             code_proj += self.proj_cluster2(self.dropout(proj))
                #         else:
                #             code_proj += self.proj_cluster2(proj)
                # else:
                #     if self.drop:
                #         code_proj = self.dropout(proj)
                #     else:
                #         code_proj = proj
                return {f"{feature_keys[-1]}_hc": code, # code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1),
                        # f"{proj_keys[-1]}_hc": code_proj.permute(0, 2, 3, 1).reshape(code_proj.shape[0], proj_h*proj_w, -1), # outputs[proj_keys[-1]], 
                        feature_keys[-1]: outputs[feature_keys[-1]],
                        proj_keys[-1]: outputs[proj_keys[-1]]}
            
            return {f"{feature_keys[-1]}_hc": code, # code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1), # code.permute(0, 2, 1),
                    feature_keys[-1]: outputs[feature_keys[-1]]}
            
            # outputs = {self.features[0]: features,
            #            f'{self.features[0]}_last': outputs[list(outputs.keys())[-1]][:, 1:, :]}
            # outputs = {self.features[0]: features[:, 1:, :]}
            
            # outputs = {feature_keys[-1]: code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1), # code.permute(0, 2, 1),
            #         f'{self.features[0]}_last': outputs[list(outputs.keys())[-1]][:, 1:, :]}
            
            # return outputs


class VisionBackboneWithIntermediateOutputs(nn.Module):
    """
    Оборачивает vision backbone для сохранения выходов промежуточных слоев.
    Поддерживает API, похожий на timm.

    Пример использования:
        backbone = vla.vision_backbone
        model = VisionBackboneWithIntermediateOutputs(backbone)

        # Способ 1: через return_dict
        outputs = model(pixel_values)
        final = outputs['final']
        intermediates = outputs['intermediate']

        # Способ 2: через forward_intermediates (аналог timm)
        final, intermediates = model.forward_intermediates(pixel_values)

        # Способ 3: только с указанными индексами
        final, selected = model.forward_intermediates(
            pixel_values,
            indices=[0, 6, 11, 23]
        )
    """

    def __init__(
        self,
        backbone: nn.Module,
        output_layers: Optional[List[str]] = None,
        return_dict: bool = True,
        register_all_blocks: bool = True,  # автоматически регистрировать все блоки
        # vit_name: str = "fused_featurizer",
    ):
        super().__init__()
        self.backbone = backbone
        self.return_dict = return_dict
        self._hooks = []
        self._intermediate_outputs = OrderedDict()

        # Определяем структуру блоков
        self._block_structure = self._analyze_block_structure()

        # Если output_layers не указаны и register_all_blocks=True,
        # регистрируем все блоки автоматически
        if output_layers is None and register_all_blocks:
            output_layers = self._get_all_block_names()

        self.output_layers = output_layers or []
        self._register_hooks()

    def _analyze_block_structure(self) -> Dict[str, List[str]]:
        """Анализирует структуру блоков в бэкбоне"""
        structure = {
            'featurizer_blocks': [],
            'fused_featurizer_blocks': []
        }

        if hasattr(self.backbone, 'featurizer') and hasattr(self.backbone.featurizer, 'blocks'):
            structure['featurizer_blocks'] = [
                f'featurizer.blocks.{i}'
                for i in range(len(self.backbone.featurizer.blocks))
            ]

        if hasattr(self.backbone, 'fused_featurizer') and hasattr(self.backbone.fused_featurizer, 'blocks'):
            structure['fused_featurizer_blocks'] = [
                f'fused_featurizer.blocks.{i}'
                for i in range(len(self.backbone.fused_featurizer.blocks))
            ]

        return structure

    def _get_all_block_names(self) -> List[str]:
        """Возвращает имена всех блоков в обоих vision transformers"""
        all_blocks = []
        all_blocks.extend(self._block_structure['featurizer_blocks'])
        all_blocks.extend(self._block_structure['fused_featurizer_blocks'])
        return all_blocks

    def _get_block_name_by_index(self, index: int) -> Optional[str]:
        """
        Возвращает имя блока по индексу для указанного ViT.

        Args:
            index: индекс блока
            vit_name: 'featurizer' или 'fused_featurizer'
        """
        max_index = len(self._block_structure['featurizer_blocks'])
        if 0 <= index < max_index:
            dino = f'featurizer.blocks.{index}'
        else:
            dino = None
        max_index = len(self._block_structure['fused_featurizer_blocks'])
        if 0 <= index < max_index:
            siglip = f'fused_featurizer.blocks.{index}'
        else:
            siglip = None
        return dino, siglip

    def _register_hooks(self):
        """Регистрирует forward hooks для указанных слоев"""
        def create_hook(layer_name):
            def hook(module, input, output):
                # Сохраняем выход с детачем, чтобы не хранить граф
                self._intermediate_outputs[layer_name] = output.detach()
            return hook

        for layer_name in self.output_layers:
            module = self._get_module_by_path(layer_name)
            if module is not None:
                hook = module.register_forward_hook(create_hook(layer_name))
                self._hooks.append(hook)

    def _get_module_by_path(self, path: str) -> Optional[nn.Module]:
        """Получает модуль по пути, например 'featurizer.blocks.5'"""
        try:
            parts = path.split('.')
            module = self.backbone
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module
        except (AttributeError, IndexError, KeyError):
            print(f"Warning: Layer '{path}' not found, skipping...")
            return None

    def forward(self, pixel_values: torch.Tensor) -> Union[Dict, torch.Tensor]:
        """
        Стандартный forward с возможностью вернуть словарь.

        Args:
            pixel_values: тензор изображений [B, C, H, W]

        Returns:
            Если return_dict=True: {'final': tensor, 'intermediate': OrderedDict}
            Иначе: только финальный выход
        """
        self._intermediate_outputs.clear()
        final_output = self.backbone(pixel_values)

        if self.return_dict:
            return {
                'final': final_output,
                'intermediate': self._intermediate_outputs
            }
        return final_output

    def forward_intermediates(
        self,
        x: torch.Tensor,
        featurizer_name: str = "fused_featurizer",
        dino: Optional[List[int]] = None,
        siglip: Optional[List[int]] = None,
        return_prefix_tokens: bool = False,
        return_attn_proj: bool = False,
        return_qkv: bool = False  # Добавляем возможность возвращать QKV проекции
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Аналог timm: возвращает финальные признаки и промежуточные выходы блоков в виде словаря.

        Args:
            x: входной тензор [B, C, H, W]
            indices: индексы блоков, выходы которых нужно вернуть.
                    Может быть int, list или tuple.
                    Если None, возвращает выходы всех зарегистрированных блоков.
            vit_name: какой ViT использовать ('featurizer' или 'fused_featurizer')
            return_prefix_tokens: игнорируется (для совместимости с timm API)
            return_qkv: если True, возвращает также QKV проекции для attention слоев

        Returns:
            tuple: (final_features, dict_of_intermediate_features)
                    где dict_of_intermediate_features: {block_index: features_tensor}

        Примеры:
            # Получить выходы всех блоков в виде словаря
            final, intermediates_dict = model.forward_intermediates(x)
            # intermediates_dict = {0: tensor, 1: tensor, ..., 23: tensor}

            # Получить выходы только блоков 0, 6, 12
            final, selected = model.forward_intermediates(x, indices=[0, 6, 12])

            # Получить выходы всех блоков fused_featurizer
            final, fused_dict = model.forward_intermediates(
                x, vit_name='fused_featurizer'
            )
        """
        if x.shape[1] != 6:
            x = torch.cat([x, x], dim=1)
        # Определяем, какие слои нам нужны
        layers_to_collect = []
        if dino is not None:
            for idx in dino:
                layer_name_dino, _ = self._get_block_name_by_index(idx)
                layers_to_collect.append(layer_name_dino)
        if siglip is not None:
            for idx in siglip:
                _, layer_name_siglip = self._get_block_name_by_index(idx)
                layers_to_collect.append(layer_name_siglip)
        # print(layers_to_collect)

        # Сохраняем текущие настройки
        original_output_layers = self.output_layers
        original_return_dict = self.return_dict

        # Временно меняем конфигурацию для этого вызова
        self.output_layers = layers_to_collect
        self.return_dict = False

        # Если нужно собирать QKV, добавляем соответствующие слои
        if return_qkv:
            qkv_layers = []
            for layer_name in layers_to_collect:
                # Добавляем attention слой для каждого блока
                block_num = layer_name.split('.')[-1]
                attn_name = layer_name.replace(f'blocks.{block_num}', f'blocks.{block_num}.attn.qkv')
                if attn_name not in self.output_layers:
                    qkv_layers.append(attn_name)
            self.output_layers.extend(qkv_layers)

        if return_attn_proj:
            attn_layers = []
            for layer_name in layers_to_collect:
                # Добавляем attention слой для каждого блока
                block_num = layer_name.split('.')[-1]
                attn_name = layer_name.replace(f'blocks.{block_num}', f'blocks.{block_num}.attn.proj')
                if attn_name not in self.output_layers:
                    attn_layers.append(attn_name)
            self.output_layers.extend(attn_layers)

        # Пересоздаем хуки для новых слоев
        self._remove_hooks()
        self._register_hooks()

        # Forward pass
        self._intermediate_outputs.clear()
        final_output = self.backbone(x)

        # Собираем промежуточные выходы в словарь {номер_блока: тензор}
        # intermediates_dict = {}
        # for i, layer_name in enumerate(layers_to_collect):
        #     # Извлекаем номер блока из имени
        #     # block_num = int(layer_name.split('.')[-1])
        #     intermediates_dict[layer_name] = self._intermediate_outputs[layer_name]

        # # Если запрошены QKV, добавляем их в отдельный словарь
        # if return_qkv:
        #     qkv_dict = {}
        #     for layer_name in self._intermediate_outputs:
        #         if 'attn' in layer_name and layer_name not in layers_to_collect:
        #             block_num = int(layer_name.split('.')[-2])
        #             if block_num not in qkv_dict:
        #                 qkv_dict[block_num] = {}
        #             if 'qkv' in layer_name:
        #                 qkv_dict[block_num]['qkv'] = self._intermediate_outputs[layer_name]
        #     intermediates_dict['qkv'] = qkv_dict

        # Восстанавливаем исходную конфигурацию
        self._remove_hooks()
        self.output_layers = original_output_layers
        self.return_dict = original_return_dict
        self._register_hooks()
        if featurizer_name == "fused_featurizer":
            return final_output, self._intermediate_outputs
        else:
            return self._intermediate_outputs["fused_featurizer.blocks.26"], self._intermediate_outputs

    def _remove_hooks(self):
        """Удаляет все зарегистрированные хуки"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def remove_hooks(self):
        """Публичный метод для удаления всех хуков"""
        self._remove_hooks()

    def get_intermediate_shapes(self) -> Dict[str, Tuple]:
        """
        Возвращает информацию о форматах выходов зарегистрированных слоев.
        Полезно для отладки.
        """
        shapes = {}
        for layer_name in self.output_layers:
            module = self._get_module_by_path(layer_name)
            if module:
                shapes[layer_name] = "unknown (need forward pass)"
        return shapes

    def __del__(self):
        self.remove_hooks()

class VLAExtractor(nn.Module):
    """Feature extractor utilizing models from timm library."""

    # Convenience aliases for feature keys
    FEATURE_ALIASES = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{j + 1}": [f"blocks.{i}" for i in range(12)] for j in range(12)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_keys{j + 1}": [f"blocks.{i}.attn.qkv" for i in range(12)] for j in range(12)},
        **{f"vit_block_attn{j + 1}": [f"blocks.{i}.attn.proj_drop" for i in range(12)] for j in range(12)},
        "vit_output": "norm",
    }
    FEATURE_MAPPING = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.proj_drop": f"vit_block_attn{i + 1}" for i in range(12)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        drop: bool = True,
        dim: int = 256,
        mode: str = "default",
        proj_type: str = None,
        features: Optional[Union[str, List[str]]] = None,
        featurizer_name: str = "fused_featurizer",
        checkpoint_path: Optional[str] = None,
        num_blocks: int = None,
        return_attn: bool = False,
        return_qkv: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        model_name = model
        self.frozen = frozen
        self.num_blocks = num_blocks
        self.dim = dim
        self.drop = drop
        self.mode = mode
        self.return_attn = return_attn
        self.return_qkv = return_qkv
        self.features = [features] if isinstance(features, str) else features
        self.featurizer_name = featurizer_name
        # self.is_vit = model_name.startswith("vit")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vla = AutoModelForVision2Seq.from_pretrained(
            model_name, # "openvla/openvla-7b"
            attn_implementation="sdpa",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.model = VisionBackboneWithIntermediateOutputs(vla.vision_backbone)

        if self.frozen:
            self.requires_grad_(False)
        
        d_dim, s_dim = 1024, 1152
        if featurizer_name == "fused_featurizer":
            fdim = d_dim + s_dim
        elif featurizer_name == "siglip":
            fdim = s_dim
        elif featurizer_name == "dino":
            fdim = d_dim
        self.dropout = torch.nn.Dropout2d(p=0.1)
        self.n_feats = fdim * num_blocks
        self.cluster1 = self.make_clusterer(self.n_feats, self.dim)
        self.proj_cluster1 = self.make_clusterer(self.n_feats, fdim)
        self.proj_type = proj_type
        self.normalization = torch.nn.LayerNorm(self.n_feats)
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats, self.dim)
            self.proj_cluster2 = self.make_nonlinear_clusterer(self.n_feats, fdim)
        self.project_head = nn.Linear(self.dim, self.dim)
        self.num_heads = 12

    def make_clusterer(self, in_channels, out):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out, (1, 1))) 

    def make_nonlinear_clusterer(self, in_channels, out):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(), # torch.nn.GELU()
            torch.nn.Conv2d(in_channels, out, (1, 1)))

    def forward(self, inp):
        # outputs = []
        # for level in self.levels:
        #     if self.frozen:
        #         with torch.no_grad():
        #             outputs.append(level(inp))
        #     else:
        #         outputs.append(level(inp))
        # print(inp.dtype, inp.shape)
        inp = torch.cat([inp, inp], dim=1).to(torch.bfloat16)
        # if self.frozen:
        dino_idxs = [int(feat.split(".")[-1]) for feat in self.features if feat.split(".")[0] == "featurizer"]
        siglip_idxs = [int(feat.split(".")[-1]) for feat in self.features if feat.split(".")[0] == "fused_featurizer"]
        with torch.no_grad():
            final_output, outputs = self.model.forward_intermediates(
                inp,
                featurizer_name=self.featurizer_name,
                dino=dino_idxs if len(dino_idxs) != 0 else None,
                siglip=siglip_idxs if len(siglip_idxs) != 0 else None,
                return_attn_proj=self.return_attn,
                return_qkv=self.return_qkv,
            )
        # else:
        #     outputs = self.model(inp)

        # outputs = {k: v[:, 1:] for k, v in outputs.items()}  # Remove CLS token
        if self.mode == "image_kk":
            feature_keys = []
            qkv_keys = []
            attn_keys = []
            for key in outputs.keys():
                if f"keys" in key:
                    qkv_keys.append(key)
                elif f"attn" in key:
                    attn_keys.append(key)
                else:
                    feature_keys.append(key)
            features = torch.cat([outputs[key] for idx, key in enumerate(feature_keys) if len(feature_keys) - (idx + 1) < self.num_blocks], 
                                dim=2)
            # qkv = [outputs[key][:, 1:, :] for idx, key in enumerate(qkv_keys) if len(qkv_keys) - (idx + 1) < self.last_blocks]
            # attn = [outputs[key][:, 1:, :] for idx, key in enumerate(attn_keys) if len(attn_keys) - (idx + 1) < self.last_blocks]
            # feat_h, feat_w = int(features.shape[1]**0.5), int(features.shape[1]**0.5) 
            # kk = [blk.reshape(features.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2) for blk in qkv]
            # B, H, W, D = k[0].shape
            # image_kk = torch.cat(kk, dim=1)
            features = self.normalization(features)
            B, N, C = outputs[feature_keys[-1]][:, :, :].shape
            qkv = [outputs[key].reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) \
                   for idx, key in enumerate(qkv_keys) if len(qkv_keys) - (idx + 1) < self.num_blocks]
            feat_h, feat_w = int(features.shape[1]**0.5), int(features.shape[1]**0.5)
            image_k = [k[1, :, :, :, :].reshape(B, qkv[0].shape[2], feat_h, feat_w, -1) for k in qkv]
            B, H, I, J, D = image_k[0].shape
            image_kk = [k.permute(0, 1, 4, 2, 3).reshape(B, H*D, I, J) for k in image_k]
            image_kk = torch.cat(image_kk, dim=1)
            image_feat = features.reshape(features.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            if self.proj_type is not None:
                if self.drop:
                    with torch.no_grad():
                        code = self.cluster1(self.dropout(image_feat))
                    code_kk = self.cluster1(self.dropout(image_kk))
                else:
                    with torch.no_grad():
                        code = self.cluster1(image_feat)
                    code_kk = self.cluster1(image_kk)
                if self.proj_type == "nonlinear":
                    if self.drop:
                        code += self.cluster2(self.dropout(image_feat))
                        code_kk += self.cluster2(self.dropout(image_kk))
                    else:
                        code += self.cluster2(image_feat)
                        code_kk += self.cluster2(image_kk)
            else:
                if self.drop:
                    code = self.dropout(image_feat)
                    code_kk = self.dropout(image_kk)
                else:
                    code = image_feat
                    code_kk = image_kk
            code_kk = code_kk.permute(0, 2, 3, 1).reshape(-1, self.dim)
            code_kk = self.project_head(code_kk)
            outputs = {self.features[0]: code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1),
                       f'{self.features[0]}_last': outputs[list(outputs.keys())[-1]][:, 1:, :],
                       "image_kk": code_kk.reshape(B, feat_h*feat_w, -1)}
            
            return outputs
        elif "hc" in self.mode:
            feature_keys = []
            proj_keys = []
            attn_keys = []
            # print(outputs.keys())
            for key in outputs.keys():
                if f"qkv" in key:
                    proj_keys.append(key)
                elif f"proj" in key:
                    attn_keys.append(key)
                else:
                    feature_keys.append(key)
            

            if self.num_blocks > 1:
                features = []
                if self.featurizer_name == "fused_featurizer":
                    for i in range(self.num_blocks):
                        features.append(torch.cat([outputs[feature_keys[i + self.num_blocks]], outputs[feature_keys[i]][:, 5:]], dim=2))
                    features = torch.cat(features, dim=2)
                else:
                    features = torch.cat([outputs[key] for key in feature_keys], dim=2)
            else:
                features = final_output #outputs[feature_keys[-1]]
            
            features = self.normalization(features.to(torch.float32))
            b, feat_h, feat_w = features.shape[0], int(features.shape[1]**0.5), int(features.shape[1]**0.5) 
            image_feat = features.reshape(b, feat_h, feat_w, -1).permute(0, 3, 1, 2)
            if self.proj_type is not None:
                with torch.no_grad():
                    if self.drop:
                        code = self.cluster1(self.dropout(image_feat))
                    else:
                        code = self.cluster1(image_feat)
                if self.proj_type == "nonlinear":
                    if self.drop:
                        code += self.cluster2(self.dropout(image_feat))
                    else:
                        code += self.cluster2(image_feat)
            else:
                if self.drop:
                    code = self.dropout(image_feat)
                else:
                    code = image_feat
            # print(code.shape)
            code = code.permute(0, 2, 3, 1).reshape(b, feat_h*feat_w, -1)
            # print(code.shape)
            code = self.project_head(code)
            # print(code.shape)
            feature_keys.append(f"blocks26")
            if self.mode == "hc_video":
                for name in proj_keys:
                    if ("keys" in name) or ("queries" in name) or ("values" in name):
                        feature_name = name.replace("queries", "keys").replace("values", "keys")
                        B, N, C = outputs[feature_name].shape
                        qkv = outputs[feature_name].reshape(
                            B, N, 3, C // 3
                        )  # outp has shape B, N, 3 * H * (C // H)
                        q, k, v = qkv.unbind(2)
                        if "keys" in name:
                            outputs[name] = k
                        elif "queries" in name:
                            outputs[name] = q
                        elif "values" in name:
                            outputs[name] = v
                        else:
                            raise ValueError(f"Unknown feature name {name}.")
                # if self.last_blocks > 1:
                #     projections = torch.cat([outputs[key] for key in proj_keys], dim=2)
                # else:
                #     projections = outputs[proj_keys[-1]]
                # projections = self.normalization(projections)
                # proj_h, proj_w = int(projections.shape[1]**0.5), int(projections.shape[1]**0.5) 
                # proj = projections.reshape(projections.shape[0], proj_h, proj_w, -1).permute(0, 3, 1, 2)
                # if self.proj_type is not None:
                #     with torch.no_grad():
                #         if self.drop:
                #             code_proj = self.proj_cluster1(self.dropout(proj))
                #         else:
                #             code_proj = self.proj_cluster1(proj)
                #     if self.proj_type == "nonlinear":
                #         if self.drop:
                #             code_proj += self.proj_cluster2(self.dropout(proj))
                #         else:
                #             code_proj += self.proj_cluster2(proj)
                # else:
                #     if self.drop:
                #         code_proj = self.dropout(proj)
                #     else:
                #         code_proj = proj
                return {f"{feature_keys[-1]}_hc": code, # code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1),
                        # f"{proj_keys[-1]}_hc": code_proj.permute(0, 2, 3, 1).reshape(code_proj.shape[0], proj_h*proj_w, -1), # outputs[proj_keys[-1]], 
                        feature_keys[-1]: final_output,
                        proj_keys[-1]: outputs[proj_keys[-1]]}
            
            return {f"{feature_keys[-1]}_hc": code, # code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1), # code.permute(0, 2, 1),
                    feature_keys[-1]: final_output.to(torch.float32)}
            
            # outputs = {self.features[0]: features,
            #            f'{self.features[0]}_last': outputs[list(outputs.keys())[-1]][:, 1:, :]}
            # outputs = {self.features[0]: features[:, 1:, :]}
            
            # outputs = {feature_keys[-1]: code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1), # code.permute(0, 2, 1),
            #         f'{self.features[0]}_last': outputs[list(outputs.keys())[-1]][:, 1:, :]}
            
            # return outputs