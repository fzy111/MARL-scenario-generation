import xml.etree.ElementTree as ET

def add_and_modify_highway_tag(osm_path, output_path):
    """
    为缺少 k="highway" 标签的 way 元素添加该标签，并修复已有的不正确的 highway 标签。
    
    参数:
    - osm_path: 原始 .osm 文件的路径
    - output_path: 修改后的 .osm 文件的保存路径
    """
    tree = ET.parse(osm_path)
    root = tree.getroot()

    # 遍历每个 'way' 元素
    for way in root.findall('way'):
        
        # 检查是否有 k="highway" 的标签
        has_highway = False
        for tag in way.findall('tag'):
            if tag.get('k') == 'highway':
                has_highway = True
                
                # 如果值为 'road'，更改为 'unclassified'
                if tag.get('v') == 'road':
                    tag.set('v', 'unclassified')
                break

        # 如果没有 k="highway" 标签，则添加一个，并设置值为 'unclassified'
        if not has_highway:
            new_tag = ET.SubElement(way, 'tag')
            new_tag.set('k', 'highway')
            new_tag.set('v', 'unclassified')

    # 保存修改后的 OSM 数据到指定的输出路径
    tree.write(output_path)

# 指定原始 .osm 文件的路径和保存修改后文件的路径
input_file_path = "/home/fanzeyu/数据集/inD-dataset-v1.0/lanelets/location4.osm"
output_file_path = "/home/fanzeyu/routes/location4.net.xml"

# 执行函数
add_and_modify_highway_tag(input_file_path, output_file_path)
