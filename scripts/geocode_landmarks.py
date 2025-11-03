import csv
import time
import os
from geopy.geocoders import Nominatim


def geocode_address(geolocator: Nominatim, address: str):
    if not address:
        return None, None
    queries = [
        f"{address}, 成都, 中国",
        f"{address}, 成都市, 中国",
        f"{address}, 中国",
        address,
    ]
    for q in queries:
        try:
            location = geolocator.geocode(q, timeout=10)
            if location:
                return (location.longitude, location.latitude)
        except Exception:
            # 忽略单次失败，尝试下一个查询备选
            pass
    return None, None


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "地标清单(utf8).csv")
    output_path = os.path.join(base_dir, "地标清单_带坐标.csv")

    geolocator = Nominatim(user_agent="community-map-geocoder")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        # 在原有字段后追加 经度/纬度
        if "经度" not in fieldnames:
            fieldnames += ["经度", "纬度"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            address = (row.get("位置") or "").strip()
            # 若已有经纬度则保留；否则尝试地理编码
            lng = row.get("经度") or ""
            lat = row.get("纬度") or ""
            if not lng or not lat:
                lon, la = geocode_address(geolocator, address)
                if lon is not None and la is not None:
                    row["经度"], row["纬度"] = f"{lon:.6f}", f"{la:.6f}"
                else:
                    row["经度"], row["纬度"] = "", ""
                # Nominatim 速率限制：建议 1 秒/请求
                time.sleep(1)
            else:
                row["经度"], row["纬度"] = lng, lat

            writer.writerow(row)

    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()


