from django.core.management.base import BaseCommand
from myapp.models import Location
import csv
import os
from datetime import datetime


class Command(BaseCommand):
    help = '导出地标数据到CSV文件'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            type=str,
            default=None,
            help='输出CSV文件路径（默认：exported_locations_YYYYMMDD_HHMMSS.csv）'
        )

    def handle(self, *args, **options):
        # 生成默认文件名（如果未指定）
        if options['output']:
            output_file = options['output']
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'exported_locations_{timestamp}.csv'

        # 获取所有地标数据
        locations = Location.objects.all().select_related('owner')
        
        if not locations.exists():
            self.stdout.write(self.style.WARNING('数据库中没有地标数据'))
            return

        # 写入CSV文件
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['id', 'name', 'latitude', 'longitude', 'description', 'category', 'owner_username', 'image']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for location in locations:
                writer.writerow({
                    'id': location.id,
                    'name': location.name,
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'description': location.description or '',
                    'category': location.category or '其他',
                    'owner_username': location.owner.username if location.owner else '',
                    'image': location.image.name if location.image else ''
                })
        
        self.stdout.write(
            self.style.SUCCESS(
                f'成功导出 {locations.count()} 条地标数据到 {output_file}'
            )
        )

