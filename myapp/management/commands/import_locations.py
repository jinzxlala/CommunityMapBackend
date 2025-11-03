import csv
import os
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model
from myapp.models import Location


class Command(BaseCommand):
    help = "从 CSV 导入地标到 Location 表；同名地标则更新经纬度，否则创建"

    def add_arguments(self, parser):
        parser.add_argument('--path', type=str, default='地标清单_带坐标.csv', help='CSV 文件路径（默认：项目根目录/地标清单_带坐标.csv）')
        parser.add_argument('--owner', type=str, default=None, help='Owner username to assign (default: first superuser or ID=1)')
        parser.add_argument('--dry-run', action='store_true', help='Validate without writing to DB')

    def handle(self, *args, **options):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        csv_path = options['path']
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(base_dir, csv_path)
        if not os.path.exists(csv_path):
            raise CommandError(f"CSV not found: {csv_path}")

        User = get_user_model()
        owner_user = None
        if options['owner']:
            try:
                owner_user = User.objects.get(username=options['owner'])
            except User.DoesNotExist:
                raise CommandError(f"Owner user not found: {options['owner']}")
        else:
            owner_user = User.objects.filter(is_superuser=True).first() or User.objects.filter(id=1).first()
            if not owner_user:
                raise CommandError("No owner user found. Please create a user or pass --owner <username>.")

        created_count = 0
        skipped_count = 0

        # 使用 utf-8-sig 以兼容带 BOM 的 CSV（例如首列会变成 "\ufeff名称"）
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            # 规范化表头：去除 BOM 与首尾空白
            if reader.fieldnames:
                reader.fieldnames = [fn.lstrip('\ufeff').strip() if fn else fn for fn in reader.fieldnames]
            required = ['名称', '位置']
            for r in required:
                if r not in reader.fieldnames:
                    raise CommandError(f"Missing required column: {r}")

            # 可选列
            price_col = '人均价格 (人民币)'
            rating_col = '综合评价'
            image_col = '图'

            def pick_col(candidates):
                for c in candidates:
                    if c in reader.fieldnames:
                        return c
                return None

            # 支持多种经纬度表头：如 经度 (E)/纬度 (N)/Longitude/Latitude 等
            lng_col = pick_col(['经度', '经度 (E)', '经度(E)', 'E', 'Longitude', 'longitude'])
            lat_col = pick_col(['纬度', '纬度 (N)', '纬度(N)', 'N', 'Latitude', 'latitude'])
            if not lng_col or not lat_col:
                raise CommandError("CSV 缺少经纬度列（支持：经度/经度 (E)/Longitude 和 纬度/纬度 (N)/Latitude）")

            for row in reader:
                name = (row.get('名称') or '').strip()
                if not name:
                    skipped_count += 1
                    continue
                description_parts = []
                pos = (row.get('位置') or '').strip()
                if pos:
                    description_parts.append(f"位置: {pos}")
                price = (row.get(price_col) or '').strip()
                if price:
                    description_parts.append(f"人均: {price}")
                rating = (row.get(rating_col) or '').strip()
                if rating:
                    description_parts.append(f"评价: {rating}")
                img = (row.get(image_col) or '').strip()
                if img:
                    description_parts.append(f"图: {img}")

                description = '；'.join(description_parts) if description_parts else ''
                category = '其他'

                try:
                    longitude = float((row.get(lng_col) or '').strip()) if (row.get(lng_col) or '').strip() else None
                    latitude = float((row.get(lat_col) or '').strip()) if (row.get(lat_col) or '').strip() else None
                except ValueError:
                    longitude = None
                    latitude = None

                if longitude is None or latitude is None:
                    skipped_count += 1
                    continue

                # 同名则更新经纬度，否则创建新地标
                existing = Location.objects.filter(name=name).first()
                if options['dry_run']:
                    if existing:
                        self.stdout.write(f"[DRY][UPDATE] {name} -> ({longitude}, {latitude})")
                    else:
                        self.stdout.write(f"[DRY][CREATE] {name} @ ({longitude}, {latitude})")
                    created_count += 0 if existing else 1
                    continue

                if existing:
                    existing.longitude = longitude
                    existing.latitude = latitude
                    existing.save(update_fields=['longitude', 'latitude'])
                else:
                    Location.objects.create(
                        name=name,
                        description=description,
                        category=category,
                        longitude=longitude,
                        latitude=latitude,
                        owner=owner_user,
                    )
                    created_count += 1

        self.stdout.write(self.style.SUCCESS(f"导入完成: 创建 {created_count} 条，跳过 {skipped_count} 条（缺少经纬度或必填字段）。"))


