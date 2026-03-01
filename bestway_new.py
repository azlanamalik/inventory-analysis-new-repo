import gym
from gym import spaces
import numpy as np


class inventory_management(gym.Env):
    def __init__(
        self,
        sku_table: dict,
        sku_ids: list,
        demand_by_sku: dict,
        horizon: int = 40,
        demand_window: int = 4,
        # global scales for normalization
        global_cost_scale: float = 100.0,
        global_price_scale: float = 200.0,
        global_holding_scale: float = 10.0,
        global_penalty_scale: float = 200.0,
        global_fixed_scale: float = 500.0,
        global_cap_scale: float = 500.0,
        max_lead_time: int = 12,
    ):
        print(sku_table, sku_ids, demand_by_sku)
        super().__init__()
        self.sku_table = sku_table
        self.sku_ids = np.array(sku_ids)
        self.demand_by_sku = demand_by_sku
        self.horizon = horizon
        self.demand_window = demand_window
        #normalisation scaling
        self.global_cost_scale = global_cost_scale
        self.global_price_scale = global_price_scale
        self.global_holding_scale = global_holding_scale
        self.global_penalty_scale = global_penalty_scale
        self.global_fixed_scale = global_fixed_scale
        self.global_cap_scale = global_cap_scale
        self.max_lead_time = max_lead_time
        # 6 dynamic + 8 context = 14 dims
        self.obs_dim = 14
        self.act_dim = 1

        # -1 -> 1 as sin and cos domain
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # action in [0,1] meaning "fraction of max_order"
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        #initialise vars
        self.sku_id = None
        self.sku = None
        self.on_hand = None
        self.pipeline = None
        self.demand_hist = None
        self.week = None
