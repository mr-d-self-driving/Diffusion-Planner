import torch
import torch.nn as nn
from timm.layers import DropPath
from timm.models.layers import Mlp

from diffusion_planner.model.module.mixer import MixerBlock

CLASS_TYPE_EGO = 0
CLASS_TYPE_NEIGHBOR = 1
CLASS_TYPE_STATIC = 2
CLASS_TYPE_LANE = 3
CLASS_TYPE_ROUTE = 4
CLASS_TYPE_NUM = 5


def add_class_type(x, class_type):
    """
    Add class type to the input tensor.
    Args:
        x: Tensor of shape (B, T, D=4) where D=4 represents (x, y, cos, sin)
        class_type: Class type to add (int)
    Returns:
        x: Tensor with class type added at the end
    """
    B, T, D = x.shape
    assert D == 4, "Input tensor must have 4 features (x, y, cos, sin)"
    class_type_tensor = torch.zeros((B, T, CLASS_TYPE_NUM), device=x.device)
    class_type_tensor[..., class_type] = 1.0
    return torch.cat([x, class_type_tensor], dim=-1)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        ego_num = 1
        self.token_num = (
            ego_num
            + config.agent_num
            + config.static_objects_num
            + config.lane_num
            + config.route_num
        )

        self.ego_encoder = EgoEncoder(
            config.time_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
        )
        self.neighbor_encoder = NeighborEncoder(
            config.time_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
        )
        self.static_encoder = StaticEncoder(
            config.static_objects_state_dim,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.lane_encoder = LaneEncoder(
            config.lane_len,
            class_type=CLASS_TYPE_LANE,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
        )
        self.route_encoder = LaneEncoder(
            config.route_len,
            class_type=CLASS_TYPE_ROUTE,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
        )

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_depth,
            device=config.device,
        )

        # position embedding encode x, y, cos, sin, type
        self.pos_emb = nn.Linear(4 + CLASS_TYPE_NUM, config.hidden_dim)

    def forward(self, inputs):
        encoder_outputs = {}

        # ego agent
        ego = inputs["ego_agent_past"]  # (B, T=21, D=4)

        # agents
        neighbors = inputs["neighbor_agents_past"]  # (B, N=32, T=21, D=11)

        # static objects
        static = inputs["static_objects"]  # (B, P=5, D=10)

        # vector maps
        lanes = inputs["lanes"]  # (B, P=70, V=20, D=12)
        lanes_speed_limit = inputs["lanes_speed_limit"]  # (B, P=70, V=20, D=1)
        lanes_has_speed_limit = inputs["lanes_has_speed_limit"]  # (B, P=70, V=20, D=1)

        # route
        route = inputs["route_lanes"]  # (B, P=25, V=20, D=12)
        route_speed_limit = inputs["route_lanes_speed_limit"]  # (B, P=25, V=20, D=1)
        route_has_speed_limit = inputs["route_lanes_has_speed_limit"]  # (B, P=25, V=20, D=1)

        B = neighbors.shape[0]

        encoding_ego, ego_mask, ego_pos = self.ego_encoder(ego)
        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(
            lanes, lanes_speed_limit, lanes_has_speed_limit
        )
        encoding_route, route_mask, route_pos = self.route_encoder(
            route, route_speed_limit, route_has_speed_limit
        )

        encoding_input = torch.cat(
            [encoding_ego, encoding_neighbors, encoding_static, encoding_lanes, encoding_route],
            dim=1,
        )

        encoding_mask = torch.cat(
            [ego_mask, neighbors_mask, static_mask, lanes_mask, route_mask], dim=1
        ).view(-1)

        encoding_pos = torch.cat(
            [ego_pos, neighbor_pos, static_pos, lane_pos, route_pos], dim=1
        ).view(B * self.token_num, -1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros(
            (B * self.token_num, self.hidden_dim), device=encoding_pos.device
        )
        encoding_pos_result[~encoding_mask] = encoding_pos  # Fill in valid parts

        encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)

        encoder_outputs["encoding"] = self.fusion(
            encoding_input, encoding_mask.view(B, self.token_num)
        )

        return encoder_outputs


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout
        )

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EgoEncoder(nn.Module):
    def __init__(
        self,
        time_len,
        drop_path_rate,
        hidden_dim,
        depth,
    ):
        super().__init__()
        tokens_mlp_dim = 64
        channels_mlp_dim = 128

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(
            in_features=4,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=time_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, T=21, D=4 (x, y, cos, sin)
        """
        B, T, D = x.shape
        pos = x[:, -1].clone()  # (B, D=4[x, y, cos, sin])
        pos = pos.unsqueeze(1)  # (B, 1, D=4)
        pos = add_class_type(pos, CLASS_TYPE_EGO)

        mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        # pooling
        x = torch.mean(x, dim=1, keepdim=True)  # (B, 1, C=channels_mlp_dim)

        x = self.emb_project(self.norm(x))  # (B, hidden_dim)

        return x, mask, pos


class NeighborEncoder(nn.Module):
    def __init__(
        self,
        time_len,
        drop_path_rate=0.3,
        hidden_dim=192,
        depth=3,
        tokens_mlp_dim=64,
        channels_mlp_dim=128,
    ):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=8 + 1,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=time_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, type(3))
        """
        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]

        pos = x[:, :, -1, :4].clone()  # x, y, cos, sin
        pos = add_class_type(pos, CLASS_TYPE_NEIGHBOR)

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        # pooling
        x = torch.mean(x, dim=1)

        neighbor_type = neighbor_type.view(B * P, -1)
        neighbor_type = neighbor_type[valid_indices]
        type_embedding = self.type_emb(neighbor_type)  # Type embedding for valid data
        x = x + type_embedding

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts

        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


class StaticEncoder(nn.Module):
    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device="cuda"):
        super().__init__()

        self._hidden_dim = hidden_dim

        self.projection = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, P, D (x, y, cos, sin, w, l, type(4))
        """
        B, P, _ = x.shape

        pos = x[:, :, :4].clone()  # x, y, cos, sin
        pos = add_class_type(pos, CLASS_TYPE_STATIC)

        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0

        valid_indices = ~mask_p.view(-1)

        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)


class LaneEncoder(nn.Module):
    def __init__(
        self,
        lane_len,
        class_type,
        drop_path_rate=0.3,
        hidden_dim=192,
        depth=3,
        tokens_mlp_dim=64,
        channels_mlp_dim=128,
    ):
        super().__init__()

        assert class_type in [CLASS_TYPE_LANE, CLASS_TYPE_ROUTE], (
            "Invalid class type for LaneEncoder"
        )

        self._lane_len = lane_len
        self._class_type = class_type
        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=8,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x, speed_limit, has_speed_limit):
        """
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        """
        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :4].clone()  # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        pos = add_class_type(pos, self._class_type)

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)

        # Apply embedding directly to valid speed limit data
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device)

        if has_speed_limit.sum() > 0:
            speed_limit_with_limit = self.speed_limit_emb(
                speed_limit[has_speed_limit].unsqueeze(-1)
            )
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        if (~has_speed_limit).sum() > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1
            )
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        # Process traffic lights directly for valid positions
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(
            traffic
        )  # Traffic light embedding for valid data

        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts

        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=3, device="cuda"):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        mask[:, 0] = False

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)
