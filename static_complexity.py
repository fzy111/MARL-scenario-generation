# 导入所需的库
import osmium
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv, VGAE
import numpy as np
import matplotlib.pyplot as plt


def haversine(lat1, lon1, lat2, lon2):
    """计算地球上两点之间的距离（单位：米）"""
    R = 6371.0  # 地球半径（单位：千米）

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c * 1000  # 单位：米


# 定义一个OSM处理器类，用于从OSM文件中提取节点和路径的信息
class OSMHandler(osmium.SimpleHandler):
    # 初始化函数
    def __init__(self):
        # 继承父类的初始化函数
        osmium.SimpleHandler.__init__(self)
        self.ways = []  # 用于存储路径的列表
        self.nodes = []  # 用于存储节点的列表

    # 当处理器遇到一个节点时，将其添加到nodes列表中
    def node(self, n):
        self.nodes.append({
            'id': n.id,  # 节点的ID
            'lat': n.location.lat,  # 节点的纬度
            'lon': n.location.lon,  # 节点的经度
            **{tag.k: tag.v for tag in n.tags}  # 节点的所有标签
        })

    # 当处理器遇到一个路径时，将其添加到ways列表中
    def way(self, w):
        self.ways.append({
            'id': w.id,  # 路径的ID
            'nodes': [node.ref for node in w.nodes],  # 路径连接的所有节点的ID
            'attributes': {tag.k: tag.v for tag in w.tags}  # 路径的所有标签
        })

# 创建一个OSM处理器对象
handler = OSMHandler()
# 使用处理器处理OSM文件
handler.apply_file('/home/fanzeyu/数据集/inD-dataset-v1.0/lanelets/location1.osm')  # 将此路径替换为你的OSM文件的路径

def process_osm_data(nodes, ways):
    # 初始化 StandardScaler
    scaler = StandardScaler()

    # 创建一个字典，将原始节点id映射到新id
    node_id_map = {node['id']: i for i, node in enumerate(nodes)}
    # 创建一个列表，用于存储节点特征，并对其进行 StandardScaler 标准化
    node_features_unnormalized = [[node['lat'], node['lon']] for node in nodes]
    node_features = scaler.fit_transform(node_features_unnormalized)  # 使用列表解析

    # 获取所有可能的属性
    all_attributes = set()
    for way in ways:
        all_attributes.update(way['attributes'].keys())

    # 边的特征将是路径的属性和它们所连接的节点
    edge_features = []
    edges = []
    for way in ways:
        # 获取路径连接的节点的id
        node_ids = [node_id_map[node_id] for node_id in way['nodes']]  # 在此处使用新的id
        # 获取路径的属性
        attributes = way['attributes']
        # 创建一个带有默认值的固定长度属性列表
        attr_list = [attributes.get(attr, "NA") for attr in all_attributes]
        # 将边特征和边添加到各自的列表中
        for i in range(len(node_ids) - 1):
            # Calculate the length of the edge
            lat1, lon1 = node_features_unnormalized[node_ids[i]]
            lat2, lon2 = node_features_unnormalized[node_ids[i+1]]
            edge_length = haversine(lat1, lon1, lat2, lon2)
            # Add the length to the attribute list
            attr_list_with_length = attr_list + [edge_length]
            edge_features.append(attr_list_with_length)
            edges.append((node_ids[i], node_ids[i+1]))

    # 现在我们需要将边特征中的分类特征转换为数值
    # 我们可以使用 one-hot 编码来实现
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    edge_features_encoded = enc.fit_transform(edge_features)

    return node_features, edges, edge_features_encoded

# 使用处理后的数据调用函数
node_features, edges, edge_features_encoded_unnormalized = process_osm_data(handler.nodes, handler.ways)

# 将数据转化为 PyTorch Tensors
x = torch.tensor(node_features, dtype=torch.float)  # 注意 node_features 现在是一个 numpy 数组，而不是字典
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 创建 StandardScaler
scaler = StandardScaler()
edge_features_encoded = scaler.fit_transform(edge_features_encoded_unnormalized)
edge_attr = torch.tensor(edge_features_encoded, dtype=torch.float)

# 创建一个 PyTorch Geometric 数据对象
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# 创建 device 对象
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将数据对象移动到设备上
data = data.to(device)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logvar = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

# 创建编码器
encoder = Encoder(data.num_node_features, out_channels=64)

# 创建图自编码器，并将其移到设备上
model = VGAE(encoder).to(device)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_values = []

# 训练模型
model.train()
for epoch in range(500):
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index) + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    
    loss_values.append(loss.item())  # 记录损失值

# 获得场景的复杂度
complexity = torch.norm(z, dim=1).mean().item()
print('Complexity:', complexity)

plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()