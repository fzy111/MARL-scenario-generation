import pandas as pd
import os
import numpy as np

def interpolate_points(x1, y1, x2, y2, num_points=10):
    """线性插值函数来生成中间的轨迹点"""
    x_values = np.linspace(x1, x2, num_points)
    y_values = np.linspace(y1, y2, num_points)
    return list(zip(x_values, y_values))

# 读取CSV文件并按`trackId`分组
df = pd.read_csv('/home/fanzeyu/数据集/inD-dataset-v1.0/data/01_tracks.csv')
grouped = df.groupby('trackId')

# 创建一个目录来存放路线文件
output_dir = 'routes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建一个文件来保存所有轨迹的路由
all_routes_path = os.path.join(output_dir, 'all_routes.rou.xml')
with open(all_routes_path, 'w') as all_routes_file:
    all_routes_file.write("<routes>\n")

    # 对于每个`trackId`，为每个轨迹点都生成一个路由项
    for track_id, group in grouped:
        previous_x, previous_y = None, None
        # 单独保存每个轨迹的路由
        with open(os.path.join(output_dir, f'route_{track_id}.rou.xml'), 'w') as individual_route_file:
            individual_route_file.write("<routes>\n")
            
            for index, row in group.iterrows():
                x = row['xCenter']
                y = row['yCenter']
                
                # 如果这不是轨迹的第一个点，则进行插值
                if previous_x is not None:
                    interpolated_points = interpolate_points(previous_x, previous_y, x, y)
                    for interp_x, interp_y in interpolated_points:
                        route_entry = f'    <route id="route_{track_id}_{index}_interp" edges="{interp_x},{interp_y}" />\n'
                        individual_route_file.write(route_entry)
                        all_routes_file.write(route_entry)
                
                route_entry = f'    <route id="route_{track_id}_{index}" edges="{x},{y}" />\n'
                individual_route_file.write(route_entry)
                all_routes_file.write(route_entry)  # 也将路由项添加到全局路由文件中
                
                previous_x, previous_y = x, y
            
            individual_route_file.write("</routes>\n")
            
    all_routes_file.write("</routes>\n")

print("Route files with interpolated points generated successfully!")

