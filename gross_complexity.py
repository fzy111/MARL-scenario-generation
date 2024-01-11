# 导入所需的库
import osmium
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import math
import sys
sys.path.append('/home/fanzeyu/experiment_gross/experiment_complexity/')
import pandas as pd
import logging
import gc
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义一个OSM处理器类，用于从OSM文件中提取节点和路径的信息
class OSMHandler(osmium.SimpleHandler):
    # 初始化函数
    def __init__(self):
        osmium.SimpleHandler.__init__(self)  # 继承父类的初始化函数
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

# 定义静态复杂度编码类
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logvar = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

# 定义可学习的位置编码类   
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        # 初始化一个嵌入层，用于学习位置信息，嵌入的大小为1
        self.pos_embedding = nn.Embedding(max_len, 1)
    
    def forward(self, x):
        # 创建一个表示位置的张量，它的形状与输入x相同
        positions = torch.arange(0, x.size(1), dtype=torch.long).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        # 从嵌入层中获取位置嵌入，并去掉最后一个维度
        pos_embed = self.pos_embedding(positions).squeeze(-1)
        # 将位置嵌入添加到输入x上，并返回结果
        return x + pos_embed

# 定义总复杂度类
class GrossComplexity:
    def __init__(self, osm_path, tracks_path, load_pretrained=False, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化所需的库和路径
        self.osm_path = osm_path 
        self.tracks_path = tracks_path
        # 加载和处理数据
        self.scaler = StandardScaler()
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.handler = self.load_osm_data()
        self.data, self.node_features, self.edges, self.edge_features_encoded = self.process_osm_data(self.handler.nodes, self.handler.ways)
        self.static_complexity = self.compute_static_complexity()
        self.sequences_tensor = self.process_track_data(self.tracks_path)
        
        # 处理轨迹数据
        self.sequences_tensor = self.process_track_data(tracks_path)

        # 检查 sequences_tensor 的形状是否满足特定条件
        # 例如，您可以检查其列数是否为偶数
        assert self.sequences_tensor.shape[1] % 2 == 0, "Number of features in sequences_tensor should be even."

        self.setup_transformer_model(self.sequences_tensor, self.static_complexity)
        self.scene_complexity_values = self.compute_scene_complexity(self.sequences_tensor)

    def load_osm_data(self):
        # 创建一个OSM处理器对象
        handler = OSMHandler()
        # 使用处理器处理OSM文件
        handler.apply_file(self.osm_path)
        return handler

    # 计算地球上两点之间的距离（单位：米）
    def haversine(self,lat1, lon1, lat2, lon2):
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
    
    def process_osm_data(self, nodes, ways):
        # 创建一个字典，将原始节点id映射到新id
        node_id_map = {node['id']: i for i, node in enumerate(nodes)}
        # 创建一个列表，用于存储节点特征，并对其进行 StandardScaler 标准化
        node_features_unnormalized = [[node['lat'], node['lon']] for node in nodes]
        node_features = self.scaler.fit_transform(node_features_unnormalized)
        
        # 获取所有可能的属性
        all_attributes = set()
        for way in ways:
            all_attributes.update(way['attributes'].keys())
        
        # 创建一个OneHotEncoder实例
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        # 使用encoder对所有的attr_list进行独热编码
        attributes_list = [way['attributes'].get(attr, "NA") for way in ways for attr in all_attributes]

        # 将 attributes_list 重塑为2D数组
        attributes_list_2D = np.array(attributes_list).reshape(-1, 1)
        # 使用 OneHotEncoder 对属性进行编码
        one_hot_attributes = encoder.fit_transform(attributes_list_2D)

        # 将独热编码的属性与边的长度合并
        edge_features = []
        edges = []
        for way in ways:
            node_ids = [node_id_map[node_id] for node_id in way['nodes']]
            
            # 使用 OneHotEncoder 转换数据时，确保输入是一个二维数组
            attributes_to_transform = [[way['attributes'].get(attr, "NA") for attr in all_attributes]]
            attributes_to_transform_2D = np.array(attributes_to_transform).reshape(-1, 1)
            attr_vec = encoder.transform(attributes_to_transform_2D)[0]

            for i in range(len(node_ids) - 1):
                lat1, lon1 = node_features_unnormalized[node_ids[i]]
                lat2, lon2 = node_features_unnormalized[node_ids[i+1]]
                edge_length = self.haversine(lat1, lon1, lat2, lon2)
                edge_features.append(np.concatenate((attr_vec, [edge_length])))
                edges.append((node_ids[i], node_ids[i+1]))

        edge_features_encoded = self.scaler.fit_transform(edge_features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # 获取非零元素的坐标和值
        # 获取非零元素的坐标和值
        edge_attr = torch.tensor(edge_features_encoded, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data.to(self.device), node_features, edges, edge_features_encoded

    def compute_static_complexity(self):
        # 创建静态编码器
        static_encoder = Encoder(self.data.num_node_features, out_channels=64)
        
        # 创建VGAE模型并将其移到相应的设备（CUDA或CPU）上
        model = VGAE(static_encoder).to(self.device)
        
        # 定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 初始化损失值列表，用于记录每个训练步骤的损失值
        loss_values = []

        # 训练模型
        model.train()
        for epoch in range(500):
            optimizer.zero_grad()  # 清除梯度信息
            z = model.encode(self.data.x, self.data.edge_index)  # 使用模型进行编码
            # 计算重构损失和KL散度损失
            loss = model.recon_loss(z, self.data.edge_index) + (1 / self.data.num_nodes) * model.kl_loss()
            loss.backward()  # 反向传播
            optimizer.step()  # 使用优化器进行参数更新
            
            # 将当前损失值添加到损失值列表中
            loss_values.append(loss.item())

        # 绘制损失值随时间的变化
        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('静态复杂度训练损失')
        plt.savefig('static_training_loss.png')
        plt.show()

        # 计算并返回场景的静态复杂度
        static_complexity = torch.norm(z, dim=1).mean().item()
        return static_complexity

    def process_track_data(self, track_file_path):
        
        # 加载数据
        tracks = pd.read_csv(track_file_path)
        # 获取每个组的大小
        group_sizes = tracks.groupby('frame_id').size().values

        # 使用StandardScaler对特征进行规范化
        scaler = StandardScaler()
        tracks_normalized = scaler.fit_transform(tracks[['x', 'y', 'psi_rad', 'vx', 'vy']])

        # 将标准化后的数组转换为DataFrame
        columns_to_normalize = ['x', 'y', 'psi_rad', 'vx', 'vy']
        tracks_normalized_df = pd.DataFrame(tracks_normalized, columns=columns_to_normalize)

        # 替换原始DataFrame中的相应列
        tracks[columns_to_normalize] = tracks_normalized_df

        # 根据帧对轨迹数据进行排序，并按帧进行分组
        tracks_sorted = tracks.sort_values(by='frame_id')
        tracks_grouped = tracks_sorted.groupby('frame_id')

        # 初始化一个空列表，用于存储序列
        sequences = []


        # 对于每个分组，将行连接成一个序列
        for name, group in tracks_grouped:
            sequence = group[['x', 'y', 'psi_rad', 'vx', 'vy']].values.flatten()  # 仅使用已规范化的特征
            sequences.append(sequence)


        # 确定最大序列长度
        max_length = max(len(sequence) for sequence in sequences)

        # 将序列填充到最大长度
        sequences_padded = [np.pad(sequence, (0, max_length - len(sequence))) for sequence in sequences]

        # 首先将列表转换为一个 numpy 数组
        sequences_array = np.array(sequences_padded)

        # 将序列数组转换为张量
        sequences_tensor = torch.tensor(sequences_array, dtype=torch.float32)

        # 检查sequences_tensor的特征维度是否为偶数
        if sequences_tensor.shape[1] % 2 != 0:
            # 如果是奇数，则添加一个全零的特征列
            zero_column = torch.zeros((sequences_tensor.shape[0], 1), dtype=sequences_tensor.dtype)
            sequences_tensor = torch.cat((sequences_tensor, zero_column), dim=1)

        # 在sequences_tensor中添加静态复杂度
        static_complexity_tensor = torch.full((sequences_tensor.shape[0], 1), self.static_complexity)

        # 再次检查维度，确保在添加静态复杂度后维度依然是偶数
        if (sequences_tensor.shape[1] + 1) % 2 != 0:
            zero_column = torch.zeros((sequences_tensor.shape[0], 1), dtype=sequences_tensor.dtype)
            sequences_tensor = torch.cat((sequences_tensor, zero_column), dim=1)

        sequences_tensor = torch.cat((sequences_tensor, static_complexity_tensor), dim=1)

        # 保存处理后的张量为类属性
        self.sequences_tensor = sequences_tensor

        # 在返回之前释放不再需要的张量
        del tracks_normalized, tracks_normalized_df, tracks_sorted, tracks_grouped, sequences_padded

        gc.collect()

        return self.sequences_tensor

        # 打印张量形状以进行验证
        print("Shape of sequences_tensor:", self.sequences_tensor.shape)

    def setup_transformer_model(self, sequences_tensor, static_complexity, batch_size=32, hidden_dim=100, nhead=2, num_layers=2):
        
        # 获取 d_model 的值并确保它可以被 nhead 整除
        d_model = sequences_tensor.shape[-1]
        if d_model % nhead != 0:
            padding_size = nhead - (d_model % nhead)
            padding_tensor = torch.zeros((sequences_tensor.shape[0], padding_size))
            sequences_tensor = torch.cat((sequences_tensor, padding_tensor), dim=1)
            d_model += padding_size
            
        # 使用新的 d_model 进行位置编码
        self.pos_encoder = LearnablePositionalEncoding(sequences_tensor.size(1), d_model)

        # 确定新的输入维度 input_dim，并确保其为偶数
        input_dim = sequences_tensor.shape[1]
        if input_dim % 2 != 0:
            zero_tensor = torch.zeros(sequences_tensor.shape[:-1] + (1,), dtype=sequences_tensor.dtype)
            sequences_tensor = torch.cat((sequences_tensor, zero_tensor), dim=-1)
            input_dim += 1

        # 将数据转换为TensorDataset并创建数据加载器
        dataset = torch.utils.data.TensorDataset(sequences_tensor)
        self.dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

        # 使用 d_model 定义 Transformer 的编码和解码层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.dynamic_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.dynamic_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 定义优化器和损失函数
        parameters = list(self.dynamic_encoder.parameters()) + list(self.dynamic_decoder.parameters()) + list(self.pos_encoder.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=0.002)
        self.loss_fn = nn.MSELoss()
        self.losses = []


    def compute_scene_complexity(self, sequences_tensor):
        """为每个序列计算场景复杂度."""
        scene_complexity_values = None
        try:
            # 初始化 complexities 变量
            complexities = []

            current_dim = sequences_tensor.shape[1]
            desired_dim = 164

            if current_dim < desired_dim:
                padding_dim = desired_dim - current_dim
                zero_padding = torch.zeros((sequences_tensor.shape[0], padding_dim), dtype=sequences_tensor.dtype)
                sequences_tensor = torch.cat((sequences_tensor, zero_padding), dim=1)

            dataset = TensorDataset(sequences_tensor)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

            for batch in dataloader:
                sequences_batch = batch[0]
                sequences_batch = self.pos_encoder(sequences_batch)
                complexity_values = self.dynamic_encoder(sequences_batch)
                complexities.append(complexity_values)

            scene_complexity = torch.cat(complexities, dim=0)
            scene_complexity_values = torch.norm(scene_complexity, dim=1, keepdim=True)

        except Exception as e:
            logger.error("Error occurred while calculating scene complexity: %s", e)

        return scene_complexity_values


    def get_single_scene_complexity(self, single_sequence_tensor):
        single_sequence_tensor = single_sequence_tensor.unsqueeze(0)  # 添加一个批处理维度
        single_sequence_tensor = self.pos_encoder(single_sequence_tensor)
        complexity_value = self.dynamic_encoder(single_sequence_tensor)

        return torch.norm(complexity_value, dim=1).item()

    def visualize_scene_complexity(self, scene_complexity_values):
        """场景复杂度可视化."""
        # 线性图
        plt.figure(figsize=(12, 6))
        plt.plot(scene_complexity_values.detach().cpu().numpy())
        plt.title("Scene Complexity over Time Steps")
        plt.xlabel("Time Steps")    
        plt.ylabel("Complexity")
        plt.savefig("scene_complexity_over_time.png")
        plt.show()

        # 前1000步的热力图
        plt.figure(figsize=(10, 10))
        sns.heatmap(scene_complexity_values[:1000].detach().cpu().numpy(), cmap="YlGnBu", cbar=True)
        plt.title("Scene Complexity Heatmap (First 1000 Time Steps)")
        plt.xlabel("Complexity Dimension")
        plt.ylabel("Time Steps")
        plt.savefig("scene_complexity_heatmap.png")
        plt.show()

    def save_complexity_values(self, scene_complexity_values, filename='scene_complexity_values.pt'):
        """保存计算得到的场景复杂度."""
        torch.save(scene_complexity_values, filename)
    
    def tsne_visualize_scene_complexity(self, scene_complexity_values, window_size=10):
        """使用t-SNE对场景复杂度进行可视化."""
        
        # 使用滑动窗口创建多维数据
        data = []
        scene_complexity_values_np = scene_complexity_values.detach().cpu().numpy()
        for i in range(len(scene_complexity_values_np) - window_size + 1):
            window = scene_complexity_values_np[i:i+window_size]
            data.append(window)
            
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(data)
        
        # 可视化
        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=np.arange(len(tsne_results)), cmap="viridis")
        plt.colorbar(label="Time Step")
        plt.title("t-SNE Visualization of Scene Complexity")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.savefig("scene_complexity_tsne.png")
        plt.show()

    def train_model(self, num_epochs=500):
        logger.info("Starting model training...")
        input_dim = self.sequences_tensor.shape[1]  # 获取sequences_tensor的维度

        # 总的训练步数（所有epoch和batch）
        total_steps = num_epochs * len(self.dataloader)

        # 使用tqdm来追踪整体进度
        pbar = tqdm(total=total_steps, desc="Training")

        for epoch in range(num_epochs):
            total_loss = 0.0  # 用于计算总损失

            for sequences_batch in self.dataloader:
                sequences_batch = sequences_batch[0]  # 获取批次中的特征部分

                # 检查sequences_batch的最后一个维度是否与input_dim匹配
                if sequences_batch.size(-1) != input_dim:
                    zero_tensor = torch.zeros(sequences_batch.shape[:-1] + (1,), dtype=sequences_batch.dtype)
                    sequences_batch = torch.cat((sequences_batch, zero_tensor), dim=-1)

                # 在数据中添加位置编码
                sequences_with_pos = self.pos_encoder(sequences_batch)

                # 前向传播
                output = self.dynamic_encoder(sequences_with_pos)
                output = self.dynamic_decoder(output, sequences_with_pos)

                # 计算损失
                loss = self.loss_fn(output, sequences_batch)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 累计损失
                total_loss += loss.item()

                # 更新进度条
                pbar.update(1)

            # 计算并记录这个 epoch 的平均损失
            avg_loss = total_loss / len(self.dataloader)
            self.losses.append(avg_loss)
            
            # 更新进度条描述
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # 关闭进度条
        pbar.close()
        
    def visualize_losses(self):
        """可视化训练损失."""
        if not self.losses:  # 检查是否有损失数据
            print("No loss data to visualize.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('training_loss.png')
        plt.show()

    def save_model(self, path):
        """保存模型到指定路径。"""
        torch.save(self.dynamic_encoder.state_dict(), path)

    @staticmethod
    def load_model(path, osm_file_path):
        """从指定路径加载模型。"""
        model = GrossComplexity(osm_file_path=osm_file_path)
        model.dynamic_encoder.load_state_dict(torch.load(path))
        model.dynamic_encoder.eval()  # 设置为评估模式
        return model
    

osm_file_path = '/Users/fanzeyu/Desktop/学习/数据集/Interaction/INTERACTION-Dataset-DR-v1_1/maps/DR_CHN_Merging_ZS.osm'
tracks_file_path = '/Users/fanzeyu/Desktop/学习/数据集/Interaction/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_CHN_Merging_ZS/vehicle_tracks_000.csv'

# 创建 GrossComplexity 实例
complexity_instance = GrossComplexity(osm_path=osm_file_path, tracks_path=tracks_file_path)

# 获取 sequences_tensor
sequences_tensor = complexity_instance.process_track_data(tracks_file_path)

# 训练模型
complexity_instance.train_model()

# 在训练后调用可视化损失函数
complexity_instance.visualize_losses()

# 使用 sequences_tensor 计算场景复杂度
scene_complexity_values = complexity_instance.compute_scene_complexity(sequences_tensor)

# 可视化和保存场景复杂度
complexity_instance.visualize_scene_complexity(scene_complexity_values)
complexity_instance.save_complexity_values(scene_complexity_values)
complexity_instance.tsne_visualize_scene_complexity(scene_complexity_values)

# 创建模型保存目录，如果它不存在的话
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 定义模型保存路径（相对路径）
model_save_path = os.path.join(model_dir, 'my_model.pt')

# 保存模型
complexity_instance.save_model(model_save_path)