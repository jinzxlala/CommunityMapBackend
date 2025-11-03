import csv
import math

def read_exported_data(filename):
    """读取导出的数据"""
    data = {}
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name']
            data[name] = {
                'lat': float(row['latitude']),
                'lng': float(row['longitude'])
            }
    return data

def read_original_data(filename):
    """读取原始WGS84数据"""
    data = {}
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['名称']
            # 跳过空行或没有经纬度的行
            if not name or not row['经度'] or not row['纬度']:
                continue
            try:
                # 使用第一组经纬度数据（原始的）
                data[name] = {
                    'lat': float(row['纬度']),
                    'lng': float(row['经度'])
                }
            except (ValueError, KeyError):
                continue
    return data

def calculate_distance(lat1, lng1, lat2, lng2):
    """计算两点之间的距离（米）"""
    R = 6371000  # 地球半径（米）
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def analyze_differences():
    """分析经纬度差异"""
    exported = read_exported_data('exported_locations_20251103_125757.csv')
    original = read_original_data('桐梓林地标详情表_WGS84版本.csv')
    
    print("=" * 100)
    print("经纬度对比分析")
    print("=" * 100)
    print(f"\n{'地标名称':<30} | {'经度差':<12} | {'纬度差':<12} | {'距离(米)':<12}")
    print("-" * 100)
    
    lat_diffs = []
    lng_diffs = []
    distances = []
    
    for name in exported:
        if name in original:
            exp_data = exported[name]
            org_data = original[name]
            
            lat_diff = exp_data['lat'] - org_data['lat']
            lng_diff = exp_data['lng'] - org_data['lng']
            
            distance = calculate_distance(
                org_data['lat'], org_data['lng'],
                exp_data['lat'], exp_data['lng']
            )
            
            lat_diffs.append(lat_diff)
            lng_diffs.append(lng_diff)
            distances.append(distance)
            
            print(f"{name:<30} | {lng_diff:>+12.8f} | {lat_diff:>+12.8f} | {distance:>12.2f}")
    
    print("=" * 100)
    
    # 统计分析
    if lat_diffs and lng_diffs:
        avg_lat_diff = sum(lat_diffs) / len(lat_diffs)
        avg_lng_diff = sum(lng_diffs) / len(lng_diffs)
        avg_distance = sum(distances) / len(distances)
        max_distance = max(distances)
        min_distance = min(distances)
        
        print(f"\n统计分析:")
        print(f"  样本数量: {len(lat_diffs)}")
        print(f"  平均纬度偏差: {avg_lat_diff:+.8f}°")
        print(f"  平均经度偏差: {avg_lng_diff:+.8f}°")
        print(f"  平均距离偏差: {avg_distance:.2f} 米")
        print(f"  最大距离偏差: {max_distance:.2f} 米")
        print(f"  最小距离偏差: {min_distance:.2f} 米")
        
        # 判断偏差方向的一致性
        lat_consistent = all(d < 0 for d in lat_diffs) or all(d > 0 for d in lat_diffs)
        lng_consistent = all(d < 0 for d in lng_diffs) or all(d > 0 for d in lng_diffs)
        
        print(f"\n偏差规律分析:")
        print(f"  纬度偏差方向一致: {'是' if lat_consistent else '否'}")
        print(f"  经度偏差方向一致: {'是' if lng_consistent else '否'}")
        
        if lat_consistent and lng_consistent:
            print(f"\n  [√] 偏差具有系统性规律!")
            print(f"  建议: ChatGPT返回的坐标可能使用了不同的坐标系")
        else:
            print(f"\n  [X] 偏差不具有系统性规律")
            print(f"  可能原因: 坐标精度不够或数据来源不准确")
        
        # 坐标系判断
        print(f"\n坐标系分析:")
        if avg_distance > 100:
            print(f"  偏差量级: 约 {avg_distance:.0f} 米")
            if -0.006 < avg_lat_diff < -0.002 and -0.01 < avg_lng_diff < -0.006:
                print(f"  可能原因: GCJ-02(火星坐标/高德) vs WGS84(GPS原始坐标)")
                print(f"  说明: 中国规定国内地图服务商必须使用GCJ-02加密坐标系")
            else:
                print(f"  可能原因: 坐标数据源不准确或查询位置有误")
        else:
            print(f"  偏差量级: 较小 (约 {avg_distance:.0f} 米)")
            print(f"  说明: 坐标基本准确，可能是查询精度问题")

if __name__ == '__main__':
    analyze_differences()

