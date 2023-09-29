import random
import sys
import gymnasium as gym
sys.path.append('/home/fanzeyu/SMARTS/')

from examples.tools.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios
from pathlib import Path
from typing import Final
from smarts.core.agent_interface import ObservationOptions
from smarts.zoo.registry import make_agent

# 从文件路径中获取SMARTS的位置
SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()

# 定义代理数量和每个代理的ID
N_AGENTS = 5
AGENT_IDS: Final[list] = ["Agent %i" % i for i in range(N_AGENTS)]

# 定义代理接口
agent_interfaces = {
    agent_id: AgentInterface.from_type(
        AgentType.Standard,
        neighborhood_vehicle_states=NeighborhoodVehicles(radius=50),
    )
    for agent_id in AGENT_IDS
}

# 定义一个基于邻近车辆的交互代理
class InteractionAgent(Agent):
    def __init__(self, action_space):
        super().__init__()  
        self.action_space = action_space

    def act(self, obs):
        neighborhood_vehicles = obs.get("neighborhood_vehicle_states", [])
        if not neighborhood_vehicles:
            return (1.0, 0.0, 0.0)
        else:
            return (0.5, 1.0, 0.0)

# 使用AgentSpec为该策略创建一个规范
agent_spec = AgentSpec(
    interface=AgentInterface(
        max_episode_steps=1000,
        waypoint_paths=Waypoints(lookahead=50),
        neighborhood_vehicle_states=NeighborhoodVehicles(radius=50),
        drivable_area_grid_map=True,
        occupancy_grid_map=True,
        top_down_rgb=RGB(height=128, width=128, resolution=100/128),
        lidar_point_cloud=False,
        action=ActionSpaceType.Continuous,
    ),
    agent_builder=InteractionAgent
)

# 注册智能体
register(
    locator="interaction-agent-v0",
    entry_point=lambda **kwargs: agent_spec
)

# 主函数
def main(scenarios, headless, num_episodes, max_episode_steps=None):
    # 在main函数内部创建代理实例
    interaction_agent_instance = make_agent("interaction-agent-v0")

    # 创建SMARTS环境
    env = gym.make(
        "smarts.env:hiway-v1",  
        scenarios=scenarios,
        agents={agent_id: interaction_agent_instance for agent_id in AGENT_IDS},
        headless=headless,
    )

    # 循环执行模拟周期
    for episode in episodes(n=num_episodes):
        observations, _ = env.reset()

        terminateds = {"__all__": False}

        while not terminateds["__all__"]:
            actions = {
                agent_id: agent.act(observations) for agent_id, agent in agents.items()
            }
            observations, rewards, terminateds, _, infos = env.step(actions)

    env.close()

# 当此脚本被直接执行时
if __name__ == "__main__":
    parser = default_argument_parser(Path(__file__).stem)
    args = parser.parse_args()

    # 如果未指定场景路径，使用默认路径
    if not args.scenarios:
        args.scenarios = [
            str(SMARTS_REPO_PATH / "SMARTS"/"scenarios" / "sumo" / "loop"),
        ]

    # 构建场景
    build_scenarios(scenarios=args.scenarios)

    # 运行主函数
    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )